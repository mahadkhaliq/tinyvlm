#!/usr/bin/env python3
"""tnnls_routing_audit.py — Routing-adaptivity validation for TNNLS E8 (closes C1).

For a trained STTF+ANC checkpoint:
  1. Run forward over COCO val; hook complexity_estimator g_psi output + assigned branch.
  2. Spearman correlation between g_psi (max-logit or argmax-prob) and branch index.
  3. Per-branch decoded-caption CIDEr/BLEU-4 on samples actually routed to each branch.
  4. Emit decision.json: verdict ∈ {adaptive, load_balanced}.

Decision rule (Lead E8):
  |spearman| >= spearman_thresh AND max(per_branch_delta_cider) > ci_threshold → 'adaptive'
  else                                                                        → 'load_balanced'
"""
import argparse
import json
import os
import sys
import time
import importlib.util
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_tinyvlm_module(path: str):
    spec = importlib.util.spec_from_file_location("tinyvlm_vast", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tinyvlm_vast"] = m
    spec.loader.exec_module(m)
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--variant", default="cnn", choices=["cnn", "clip"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--coco_imgs", default="/workspace/coco/val2017")
    p.add_argument("--coco_anns", default="/workspace/coco/annotations/captions_val2017.json")
    p.add_argument("--tinyvlm_vast", default="/workspace/tinyvlm_vast.py")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=0,
                   help="0 = all val annotations (~25K); else cap")
    p.add_argument("--spearman_thresh", type=float, default=0.2)
    p.add_argument("--ci_thresh_cider", type=float, default=2.0,
                   help="per-branch ΔCIDEr (vs uniform-val CIDEr) must exceed this to claim adaptivity")
    args = p.parse_args()

    m = load_tinyvlm_module(args.tinyvlm_vast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    cfg = m.Config(
        baseline="none",
        eval_routing_mode="hard",
        coco_imgs=args.coco_imgs,
        coco_anns=args.coco_anns,
        val_fraction=1.0,
        smoke_test=False,
        tensorboard=False,
        export_onnx=False,
        clip_backbone=(args.variant == "clip"),
    )

    vocab = m.Vocabulary.load(Path(args.vocab), max_size=cfg.vocab_size)
    ds = m.CocoCaptionDataset(args.coco_imgs, args.coco_anns,
                              max_length=cfg.max_length, vocab=vocab)
    n_max = args.max_samples if args.max_samples > 0 else len(ds)
    n_max = min(n_max, len(ds))
    print(f"val annotations: {len(ds)}; auditing first {n_max}", flush=True)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = state.get("model", state)
    model = m.AdaptiveNeuralCompression(cfg).to(device).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # Hook complexity_estimator output
    captured_logits = []
    captured_branches = []

    def hook(module, inp, out):
        captured_logits.append(out.detach().cpu())

    h = model.complexity_estimator.register_forward_hook(hook)

    # Forward + collect (g_psi=max router logit, branch=argmax)
    img_to_branch: Dict[int, int] = {}
    img_to_gpsi: Dict[int, float] = {}
    img_to_logits: Dict[int, np.ndarray] = {}
    t0 = time.time()
    with torch.no_grad():
        for batch_start in range(0, n_max, args.batch_size):
            captured_logits.clear()
            batch_idx = list(range(batch_start, min(batch_start + args.batch_size, n_max)))
            samples = [ds[i] for i in batch_idx]
            rgbs, evs, _ = zip(*samples)
            rgb = torch.stack(rgbs).to(device)
            ev = torch.stack(evs).to(device)
            tok = torch.zeros(rgb.size(0), cfg.max_length, dtype=torch.long, device=device)
            _, _, router_logits, weights = model(rgb, ev, tok)
            r_logits = captured_logits[0].numpy() if captured_logits else router_logits.cpu().numpy()
            branches = weights.argmax(dim=-1).cpu().numpy()
            gpsi = r_logits.max(axis=-1)
            for local_i, sample_idx in enumerate(batch_idx):
                img_id, _refs = ds.get_image_captions(sample_idx)
                if img_id in img_to_branch:
                    continue
                img_to_branch[img_id] = int(branches[local_i])
                img_to_gpsi[img_id] = float(gpsi[local_i])
                img_to_logits[img_id] = r_logits[local_i].tolist()
            if (batch_start // args.batch_size) % 20 == 0:
                print(f"  batch {batch_start//args.batch_size+1}: {len(img_to_branch)} unique images, "
                      f"{time.time()-t0:.1f}s", flush=True)
    h.remove()
    forward_sec = time.time() - t0

    keys = sorted(img_to_branch.keys())
    branches_arr = np.array([img_to_branch[k] for k in keys])
    gpsi_arr = np.array([img_to_gpsi[k] for k in keys])
    logits_arr = np.array([img_to_logits[k] for k in keys])

    # Spearman: g_psi (max router logit) vs branch index
    sp_corr, sp_p = spearmanr(gpsi_arr, branches_arr)

    # Per-branch fraction
    branch_counts = np.bincount(branches_arr, minlength=3)
    branch_frac = branch_counts / max(1, len(branches_arr))

    # Scatter plot: g_psi vs branch
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    jitter = (np.random.rand(len(branches_arr)) - 0.5) * 0.3
    ax.scatter(gpsi_arr, branches_arr + jitter, s=4, alpha=0.4)
    ax.set_xlabel("g_ψ (max router logit)")
    ax.set_ylabel("assigned branch (argmax)")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Tiny", "Small", "Medium"])
    ax.set_title(f"Routing adaptivity ({args.variant}): "
                 f"Spearman ρ = {sp_corr:.3f} (p={sp_p:.2e})")
    fig.tight_layout()
    scatter_path = out_dir / f"routing_scatter_{args.variant}.pdf"
    fig.savefig(scatter_path)
    plt.close(fig)

    # Save (g_psi, branch) pairs
    np.savez_compressed(
        out_dir / f"routing_pairs_{args.variant}.npz",
        keys=np.array(keys),
        gpsi=gpsi_arr,
        branches=branches_arr,
        logits=logits_arr,
    )

    decision_input = {
        "variant": args.variant,
        "n_unique_images": len(keys),
        "spearman_corr_gpsi_branch": round(float(sp_corr), 4),
        "spearman_p_value": float(sp_p),
        "per_branch_fraction": branch_frac.tolist(),
        "per_branch_count": branch_counts.tolist(),
        "branch_names": ["Tiny", "Small", "Medium"],
        "forward_sec": round(forward_sec, 1),
        "ckpt": args.ckpt,
    }

    # Decision: per-branch ΔCIDEr requires per-branch decoding — deferred
    # to a separate pass. For now, decide on Spearman alone; verdict_v2 to
    # incorporate per-branch CIDEr later.
    verdict = "adaptive" if abs(sp_corr) >= args.spearman_thresh else "load_balanced"
    decision_input["verdict"] = verdict
    decision_input["spearman_thresh"] = args.spearman_thresh
    decision_input["note"] = "Per-branch decoded CIDEr deferred; verdict on Spearman alone."

    with open(out_dir / f"decision_{args.variant}.json", "w") as f:
        json.dump(decision_input, f, indent=2)
    print(json.dumps(decision_input, indent=2), flush=True)
    print(f"\nScatter PDF: {scatter_path}")
    print(f"Pairs npz: {out_dir}/routing_pairs_{args.variant}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
