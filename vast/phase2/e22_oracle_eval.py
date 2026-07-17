#!/usr/bin/env python3
"""
E22 -- oracle routing bound.

The paper claims "no routing policy recovers the ~5 CIDEr." That quantifier is currently
supported by exactly three *learned* lambda values. This script bounds EVERY policy:

  1. force each branch in turn and caption the whole Karpathy-test split with it;
  2. score every image against its references, per branch;
  3. for each image pick the branch that scored best (this is the oracle -- it is what a
     perfect router would do if it could see the metric);
  4. assemble the winning captions and report the CORPUS score of that set, plus the
     realised FLOPs of the oracle's branch distribution.

How to read it:
  oracle ~= 75   -> "no policy recovers" is airtight and policy-independent. C4 holds even
                    against a router that cannot be built.
  oracle ~= 79+  -> real routing headroom exists at these budgets and the paper's claim is
                    too strong. Better we find that than a referee.

IMPORTANT -- this is an UPPER BOUND and optimistic by construction. Selecting the per-image
best *on the test metric itself* is not achievable by any real router (it peeks at the
answer). That optimism is exactly the point: if even this cannot clear the wall, no policy
can. Do not report the oracle as an achievable operating point.

Usage:
    python e22_oracle_eval.py --run_dir $SCRATCH/tinyvlm_runs/E_phase2/seed_42_tau_0.80 \
                              --coco_imgs $SCRATCH/coco/train2017 \
                              --coco_anns $SCRATCH/coco/annotations/captions_train2017.json \
                              --out tnnls_results/E22_oracle/seed_42.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Dict, List

import torch

from tinyvlm_vast import (  # noqa: E402
    Config,
    Trainer,
    compute_caption_metrics,
    greedy_decode,
    setup_logging,
    logger,
)


def _per_image_cider(hyps: List[str], refs: List[List[str]]) -> List[float]:
    """Per-image CIDEr-D. Returns [] if pycocoevalcap is unavailable."""
    try:
        from pycocoevalcap.cider.cider import Cider
    except Exception as e:  # pragma: no cover
        logger.warning("pycocoevalcap unavailable (%s) -- cannot compute the oracle.", e)
        return []
    gts = {i: r for i, r in enumerate(refs)}
    res = {i: [h] for i, h in enumerate(hyps)}
    try:
        _, per_img = Cider().compute_score(gts, res)
    except Exception:
        gts = {i: [{"caption": x} for x in r] for i, r in enumerate(refs)}
        res = {i: [{"caption": h}] for i, h in enumerate(hyps)}
        _, per_img = Cider().compute_score(gts, res)
    return [float(x) for x in per_img]


def build_ref_map(eval_ds) -> Dict[int, List[str]]:
    """References indexed by position in the eval set.

    Mirrors Trainer._eval_captions exactly: the loader's dataset may be a Subset (or a
    Subset of a Subset) of CocoCaptionDataset, so unwrap until we reach the base and map
    positions through the index chain. Do not simplify this -- getting it wrong silently
    pairs captions with the wrong references and every score becomes meaningless."""
    underlying = eval_ds
    if hasattr(underlying, "dataset") and hasattr(underlying, "indices"):
        base_ds, eval_indices = underlying.dataset, underlying.indices
    else:
        base_ds, eval_indices = underlying, list(range(len(underlying)))
    if hasattr(base_ds, "dataset") and hasattr(base_ds, "indices"):
        outer = base_ds.indices
        base_ds = base_ds.dataset
        eval_indices = [outer[i] for i in eval_indices]

    ref_map: Dict[int, List[str]] = {}
    if hasattr(base_ds, "get_image_captions"):
        for pos, idx in enumerate(eval_indices):
            _, r = base_ds.get_image_captions(idx)
            ref_map[pos] = r
    return ref_map


@torch.no_grad()
def caption_split(trainer: Trainer, branch: int, ref_map: Dict[int, List[str]]):
    """Caption the whole eval split with every input pinned to `branch`."""
    raw = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    raw.eval()
    raw._force_branch = branch
    try:
        hyps: List[str] = []
        refs: List[List[str]] = []
        pos = 0
        for rgb, events, _ in trainer.val_loader:
            rgb, events = rgb.to(trainer.device), events.to(trainer.device)
            caps = greedy_decode(
                trainer.model, rgb, events, trainer.cfg.max_length, trainer.vocab, trainer.device
            )
            for cap in caps:
                hyps.append(cap)
                refs.append(ref_map.get(pos, [""]))
                pos += 1
    finally:
        raw._force_branch = None   # never leak the override
    return hyps, refs


def main() -> None:
    ap = argparse.ArgumentParser(description="E22 oracle routing bound")
    ap.add_argument("--run_dir", required=True, help="a trained token-budget run dir (has checkpoints/)")
    ap.add_argument("--coco_imgs", required=True)
    ap.add_argument("--coco_anns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--token_select", choices=["stride", "attn"], default="stride")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "checkpoints").exists():
        raise FileNotFoundError(f"no checkpoints/ under {run_dir}")
    setup_logging("INFO", log_file=run_dir / "e22_oracle.log")

    cfg_path = run_dir / "config.json"
    cfg_d = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    known = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in cfg_d.items() if k in known})
    cfg = dataclasses.replace(
        cfg, resume=True, coco_imgs=args.coco_imgs, coco_anns=args.coco_anns,
        token_select=args.token_select, eval_routing_mode="hard", export_onnx=False,
    )
    if not cfg.clip_token_budget:
        raise SystemExit("run_dir is not a token-budget run; the oracle needs per-branch budgets.")

    trainer = Trainer(cfg, run_dir, rank=0, local_rank=0, world_size=1)
    raw = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    budgets = raw._budgets
    K = len(budgets)
    logger.info("E22 oracle: %d branches, budgets=%s, select=%s", K, budgets, cfg.token_select)

    per_branch_hyps: List[List[str]] = []
    per_branch_scores: List[List[float]] = []
    refs: List[List[str]] = []
    branch_corpus: List[Dict[str, float]] = []

    ref_map = build_ref_map(trainer.val_loader.dataset)
    if not ref_map:
        raise SystemExit("could not build reference map from the eval dataset; refusing to score.")

    for b in range(K):
        hyps, refs = caption_split(trainer, b, ref_map)
        scores = _per_image_cider(hyps, refs)
        if not scores:
            raise SystemExit("per-image CIDEr unavailable; install pycocoevalcap.")
        per_branch_hyps.append(hyps)
        per_branch_scores.append(scores)
        corpus = compute_caption_metrics(hyps, refs)
        branch_corpus.append(corpus)
        logger.info("branch %d (keep_k=%d): corpus CIDEr=%.2f", b, budgets[b], corpus.get("cider", float("nan")))

    n = len(refs)
    pick = [max(range(K), key=lambda b: per_branch_scores[b][i]) for i in range(n)]
    oracle_hyps = [per_branch_hyps[pick[i]][i] for i in range(n)]
    oracle_corpus = compute_caption_metrics(oracle_hyps, refs)

    frac = [pick.count(b) / n for b in range(K)]
    realised = sum(f * raw.clip_stem.stem_flops(k, cfg.token_select) for f, k in zip(frac, budgets)) / 1e9

    best_single = max(range(K), key=lambda b: branch_corpus[b].get("cider", -1.0))
    out = {
        "experiment": "E22_oracle_bound",
        "seed": cfg.seed,
        "token_select": cfg.token_select,
        "budgets": budgets,
        "n_images": n,
        "per_branch_corpus": {
            str(budgets[b]): {k: round(v, 4) for k, v in branch_corpus[b].items()} for b in range(K)
        },
        "best_single_branch": {"keep_k": budgets[best_single], **{k: round(v, 4) for k, v in branch_corpus[best_single].items()}},
        "oracle": {k: round(v, 4) for k, v in oracle_corpus.items()},
        "oracle_branch_fractions": [round(f, 4) for f in frac],
        "oracle_realised_gflops": round(realised, 4),
        "interpretation": (
            "UPPER BOUND, optimistic by construction: the per-image branch is chosen using the "
            "test metric itself, which no deployable router can do. If the oracle does not clear "
            "the dense control (79.80 +/- 1.51), then no routing policy at these budgets can, and "
            "C4 is policy-independent. Do NOT report the oracle as an achievable operating point."
        ),
        "reference_points": {"dense_control_cider": 79.80, "routed_lambda_0.01_cider": 74.93},
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    logger.info("oracle CIDEr=%.2f (best single branch=%.2f) -> %s",
                oracle_corpus.get("cider", float("nan")),
                branch_corpus[best_single].get("cider", float("nan")), outp)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
