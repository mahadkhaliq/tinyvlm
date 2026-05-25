#!/usr/bin/env python3
"""paired_stats.py — Paired statistical comparison of two TinyVLM runs.

Inputs: two multi_seed_summary.json files (or per-seed summary.json files).
Computes paired t-test, Cohen's d, and 95% bootstrap CI on the per-seed
differences for each metric. Both inputs must be aligned on the same seeds.

Usage:
    python paired_stats.py --a runs/sttf_anc/multi_seed_summary.json \
                           --b runs/baselines/tokenlearner_3seed/multi_seed_summary.json \
                           --metric final_cider \
                           [--out tables/paired_stats.json]
    python paired_stats.py --a-glob 'runs/sttf_anc/seed_*/summary.json' \
                           --b-glob 'runs/baselines/tokenlearner_3seed/seed_*/summary.json'
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


METRICS = ("final_cider", "final_bleu4", "final_val_accuracy", "best_cider")


def load_summary(path: str) -> Dict[int, Dict]:
    """Return {seed: summary_dict}. Accepts either multi_seed_summary.json
    (per_seed list) or a single summary.json (one seed)."""
    data = json.loads(Path(path).read_text())
    if "per_seed" in data:
        return {int(r["seed"]): r for r in data["per_seed"]}
    if "seed" in data:
        return {int(data["seed"]): data}
    raise ValueError(f"{path} has neither 'per_seed' nor 'seed'")


def load_glob(pattern: str) -> Dict[int, Dict]:
    out: Dict[int, Dict] = {}
    for p in glob.glob(pattern):
        d = json.loads(Path(p).read_text())
        seed = int(d.get("seed"))
        out[seed] = d
    if not out:
        raise FileNotFoundError(f"No summary files matched glob: {pattern}")
    return out


def paired_diffs(a: Dict[int, Dict], b: Dict[int, Dict], metric: str) -> Tuple[List[int], List[float]]:
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        raise ValueError(f"Need ≥2 matched seeds; got {seeds}")
    diffs = []
    for s in seeds:
        va = a[s].get(metric)
        vb = b[s].get(metric)
        if va is None or vb is None:
            raise ValueError(f"Seed {s} missing metric '{metric}'")
        diffs.append(float(va) - float(vb))
    return seeds, diffs


def paired_ttest(diffs: List[float]) -> Tuple[float, int, float]:
    arr = np.asarray(diffs)
    n = len(arr)
    if HAS_SCIPY:
        t, p = stats.ttest_1samp(arr, 0.0)
        return float(t), n - 1, float(p)
    # Manual t-test fallback
    mean = arr.mean()
    sd = arr.std(ddof=1) if n > 1 else 0.0
    if sd == 0:
        return float("inf"), n - 1, 0.0
    t = mean / (sd / math.sqrt(n))
    # Two-sided p (Student's t CDF) — without scipy, approximate via Welch's
    # df via large-sample normal — n is small so this is approximate.
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, n - 1, p


def cohens_d_paired(diffs: List[float]) -> float:
    arr = np.asarray(diffs)
    if arr.std(ddof=1) == 0:
        return float("inf") if arr.mean() != 0 else 0.0
    return float(arr.mean() / arr.std(ddof=1))


def bootstrap_ci(diffs: List[float], n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(diffs)
    n = len(arr)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare(a: Dict[int, Dict], b: Dict[int, Dict],
            metrics=METRICS) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for m in metrics:
        try:
            seeds, diffs = paired_diffs(a, b, m)
        except (ValueError, KeyError) as e:
            out[m] = {"error": str(e)}
            continue
        t, df, p = paired_ttest(diffs)
        d = cohens_d_paired(diffs)
        lo, hi = bootstrap_ci(diffs)
        a_vals = [a[s][m] for s in seeds]
        b_vals = [b[s][m] for s in seeds]
        out[m] = {
            "seeds": seeds,
            "a_mean": float(np.mean(a_vals)),
            "a_std": float(np.std(a_vals, ddof=1)) if len(a_vals) > 1 else 0.0,
            "b_mean": float(np.mean(b_vals)),
            "b_std": float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else 0.0,
            "diff_mean": float(np.mean(diffs)),
            "diff_std": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
            "t": t,
            "df": df,
            "p": p,
            "cohens_d": d,
            "ci95_lo": lo,
            "ci95_hi": hi,
            "n_seeds": len(seeds),
        }
    return out


def fmt_table(name_a: str, name_b: str, results: Dict[str, Dict]) -> str:
    lines = [f"# Paired comparison: {name_a} vs {name_b}", ""]
    lines.append(f"{'metric':22s} {'a_mean±std':18s} {'b_mean±std':18s} {'Δ':>8s} {'p':>8s} {'d':>6s} {'95% CI':>20s}")
    lines.append("-" * 110)
    for m, r in results.items():
        if "error" in r:
            lines.append(f"{m:22s}  {r['error']}")
            continue
        a_str = f"{r['a_mean']:.3f}±{r['a_std']:.3f}"
        b_str = f"{r['b_mean']:.3f}±{r['b_std']:.3f}"
        ci_str = f"[{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}]"
        lines.append(
            f"{m:22s} {a_str:18s} {b_str:18s} {r['diff_mean']:+8.3f} {r['p']:8.4f} {r['cohens_d']:+6.2f} {ci_str:>20s}"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g_a = ap.add_mutually_exclusive_group(required=True)
    g_a.add_argument("--a", type=str, help="multi_seed_summary.json or summary.json (group A)")
    g_a.add_argument("--a-glob", type=str, help="glob for per-seed summary.json files (group A)")
    g_b = ap.add_mutually_exclusive_group(required=True)
    g_b.add_argument("--b", type=str, help="multi_seed_summary.json or summary.json (group B)")
    g_b.add_argument("--b-glob", type=str, help="glob for per-seed summary.json files (group B)")
    ap.add_argument("--metric", type=str, default=None,
                    help="Restrict to one metric; default = all standard metrics")
    ap.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    args = ap.parse_args()

    a = load_summary(args.a) if args.a else load_glob(args.a_glob)
    b = load_summary(args.b) if args.b else load_glob(args.b_glob)
    metrics = (args.metric,) if args.metric else METRICS
    results = compare(a, b, metrics)

    name_a = args.a or args.a_glob
    name_b = args.b or args.b_glob
    print(fmt_table(name_a, name_b, results))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "a_source": name_a, "b_source": name_b, "results": results,
        }, indent=2))
        print(f"\nWritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
