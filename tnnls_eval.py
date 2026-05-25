#!/usr/bin/env python3
"""tnnls_eval.py — Full caption-metric evaluator for TNNLS E19.

Loads any TinyVLM checkpoint (best.pt format), runs greedy decode on COCO val,
computes the full panel {BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE} via
pycocoevalcap. Bypasses Trainer + CheckpointManager (which require latest.pt +
optimizer state) so any saved best.pt is usable.

Usage:
  python tnnls_eval.py \\
      --ckpt /workspace/ckpts/sttf_anc_cnn/seed_42_tau_0.80/checkpoints/best.pt \\
      --vocab /workspace/ckpts/sttf_anc_cnn/seed_42_tau_0.80/vocabulary.json \\
      --variant sttf_anc_cnn_seed42 \\
      --out /workspace/tnnls_results/E19/sttf_anc_cnn_seed42.json
"""
import argparse
import json
import os
import sys
import time
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Subset


def load_tinyvlm_module(path: str):
    spec = importlib.util.spec_from_file_location("tinyvlm_vast", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["tinyvlm_vast"] = m
    spec.loader.exec_module(m)
    return m


def detect_model_class(state_dict, m):
    keys = list(state_dict.keys())
    if any(k.startswith("encoder.") and not k.startswith("encoders.") for k in keys):
        return m.DenseEncoderBaseline
    if any("tokenlearner" in k.lower() or "token_learner" in k.lower() for k in keys):
        return m.TokenLearnerBaseline
    return m.AdaptiveNeuralCompression


def build_model(cfg, ckpt_state, m, device):
    sd = ckpt_state.get("model", ckpt_state)
    cls = detect_model_class(sd, m)
    print(f"model class: {cls.__name__}", flush=True)
    model = cls(cfg).to(device).eval()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"load: missing={len(missing)} unexpected={len(unexpected)} params={n_params/1e6:.2f}M",
          flush=True)
    if missing:
        print(f"first missing: {missing[:3]}", flush=True)
    if unexpected:
        print(f"first unexpected: {unexpected[:3]}", flush=True)
    return model


def decode_val_set(model, ds, vocab, m, args, device) -> Tuple[Dict, Dict, float]:
    """Greedy-decode the dataset; dedupe to one hypothesis per image_id; collect refs."""
    raw_ds = ds.dataset if isinstance(ds, Subset) else ds
    indices = list(ds.indices) if isinstance(ds, Subset) else list(range(len(ds)))

    img_to_hyp: Dict[int, str] = {}
    img_to_refs: Dict[int, List[str]] = {}
    t0 = time.time()
    for batch_start in range(0, len(indices), args.batch_size):
        batch_idx = indices[batch_start:batch_start + args.batch_size]
        samples = [raw_ds[i] for i in batch_idx]
        rgbs, evs, _ = zip(*samples)
        rgb = torch.stack(rgbs).to(device)
        ev = torch.stack(evs).to(device)
        caps = m.greedy_decode(model, rgb, ev, args.max_length, vocab, device)
        for local_i, sample_idx in enumerate(batch_idx):
            img_id, ref_caps = raw_ds.get_image_captions(sample_idx)
            if img_id in img_to_hyp:
                continue
            img_to_hyp[img_id] = caps[local_i]
            img_to_refs[img_id] = ref_caps
        if (batch_start // args.batch_size) % 25 == 0:
            elapsed = time.time() - t0
            print(f"  batch {batch_start//args.batch_size + 1}/{(len(indices) + args.batch_size - 1)//args.batch_size}: "
                  f"{len(img_to_hyp)} unique images, {elapsed:.1f}s elapsed", flush=True)
    decode_sec = time.time() - t0
    print(f"decode wall: {decode_sec:.1f}s, {len(img_to_hyp)} unique images", flush=True)
    return img_to_hyp, img_to_refs, decode_sec


def compute_all_metrics(img_to_hyp: Dict, img_to_refs: Dict,
                        enable_meteor: bool = False,
                        enable_spice: bool = False) -> Dict:
    """BLEU-1..4 + ROUGE-L + CIDEr (always); METEOR + SPICE (Java-dependent,
    gated behind flags because pycocoevalcap's Meteor wrapper can deadlock
    on a dead JVM subprocess instead of raising)."""
    keys = sorted(img_to_hyp.keys())
    gts = {i: img_to_refs[k] for i, k in enumerate(keys)}
    res = {i: [img_to_hyp[k]] for i, k in enumerate(keys)}

    metrics: Dict[str, float] = {}
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    bleu, _ = Bleu(4).compute_score(gts, res)
    for i, b in enumerate(bleu, start=1):
        metrics[f"bleu{i}"] = round(float(b) * 100, 4)
    rouge, _ = Rouge().compute_score(gts, res)
    metrics["rouge_l"] = round(float(rouge) * 100, 4)
    cider, _ = Cider().compute_score(gts, res)
    metrics["cider"] = round(float(cider) * 100, 4)
    if enable_meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            meteor, _ = Meteor().compute_score(gts, res)
            metrics["meteor"] = round(float(meteor) * 100, 4)
        except Exception as e:
            metrics["meteor"] = None
            metrics["meteor_error"] = str(e)[:200]
    else:
        metrics["meteor"] = None
        metrics["meteor_skipped"] = True
    if enable_spice:
        try:
            from pycocoevalcap.spice.spice import Spice
            spice, _ = Spice().compute_score(gts, res)
            metrics["spice"] = round(float(spice) * 100, 4)
        except Exception as e:
            metrics["spice"] = None
            metrics["spice_error"] = str(e)[:200]
    else:
        metrics["spice"] = None
        metrics["spice_skipped"] = True
    return metrics


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--variant", required=True, help="label for output JSON")
    p.add_argument("--out", required=True)
    p.add_argument("--coco_imgs", default="/workspace/coco/val2017")
    p.add_argument("--coco_anns", default="/workspace/coco/annotations/captions_val2017.json")
    p.add_argument("--tinyvlm_vast", default="/workspace/tinyvlm_vast.py")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--val_fraction", type=float, default=1.0,
                   help="1.0 = full val (5K unique images via ~25K annotations)")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--num_decoder_layers", type=int, default=2)
    p.add_argument("--baseline", default="none", choices=["none", "tokenlearner", "dense"])
    p.add_argument("--encoder_only", default="medium",
                   help="DenseEncoderBaseline branch; ignored for ANC/TokenLearner")
    p.add_argument("--eval_routing_mode", default="hard", choices=["hard", "soft"])
    p.add_argument("--tokenlearner_num_tokens", type=int, default=8)
    p.add_argument("--clip_backbone", action="store_true")
    p.add_argument("--enable_meteor", action="store_true",
                   help="Run Meteor scorer (may deadlock on JVM crash; off by default)")
    p.add_argument("--enable_spice", action="store_true",
                   help="Run SPICE scorer (slow; off by default)")
    args = p.parse_args()

    m = load_tinyvlm_module(args.tinyvlm_vast)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}; torch {torch.__version__}", flush=True)

    cfg = m.Config(
        baseline=args.baseline,
        encoder_only=args.encoder_only,
        eval_routing_mode=args.eval_routing_mode,
        tokenlearner_num_tokens=args.tokenlearner_num_tokens,
        max_length=args.max_length,
        vocab_size=args.vocab_size,
        num_decoder_layers=args.num_decoder_layers,
        coco_imgs=args.coco_imgs,
        coco_anns=args.coco_anns,
        val_fraction=args.val_fraction,
        smoke_test=False,
        tensorboard=False,
        export_onnx=False,
        clip_backbone=args.clip_backbone,
    )

    vocab = m.Vocabulary.load(Path(args.vocab), max_size=args.vocab_size)
    print(f"vocab size: {len(vocab.word2idx)}", flush=True)

    ds = m.CocoCaptionDataset(
        args.coco_imgs, args.coco_anns,
        max_length=args.max_length, vocab=vocab,
    )
    if args.val_fraction < 1.0:
        n = max(1, int(len(ds) * args.val_fraction))
        ds = Subset(ds, list(range(n)))
    print(f"val size (annotations): {len(ds)}", flush=True)

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"ckpt epoch: {state.get('epoch', '?')}, best_val_loss: {state.get('best_val_loss', '?')}",
          flush=True)
    model = build_model(cfg, state, m, device)

    img_to_hyp, img_to_refs, decode_sec = decode_val_set(model, ds, vocab, m, args, device)

    if not img_to_hyp:
        print("ERROR: no hypotheses generated", file=sys.stderr)
        return 1

    print(f"sample hypothesis: {next(iter(img_to_hyp.values()))!r}", flush=True)
    print(f"sample refs (first img): {list(img_to_refs.values())[0]}", flush=True)

    t1 = time.time()
    metrics = compute_all_metrics(img_to_hyp, img_to_refs,
                                  enable_meteor=args.enable_meteor,
                                  enable_spice=args.enable_spice)
    metrics_sec = time.time() - t1

    out = {
        "variant": args.variant,
        "ckpt": args.ckpt,
        "n_unique_images": len(img_to_hyp),
        "val_fraction": args.val_fraction,
        "eval_routing_mode": args.eval_routing_mode,
        "baseline": args.baseline,
        "encoder_only": args.encoder_only,
        "clip_backbone": args.clip_backbone,
        "decode_sec": round(decode_sec, 1),
        "metrics_sec": round(metrics_sec, 1),
        **metrics,
    }
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
