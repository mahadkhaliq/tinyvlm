#!/usr/bin/env python3
"""Aggregate the 3 Phase-2 token-budget seeds and evaluate the pre-registered A1-A4
criteria (TNNLS_STATISTICAL_PLAN_PHASE2.md). Prints aggregate.json to stdout and writes
a NOTES.md verdict next to the run dir. Reads each seed's summary.json (Karpathy-test),
which must contain: cider, bleu4, inference_branch_fraction (list of 3), and either
realised_encoder_gflops or enough to compute it.

Usage: python phase2_aggregate.py <run_dir>   (run_dir has seed_42/ seed_43/ seed_44/)
"""
import json, sys, glob, os, statistics as st

# per-branch stem FLOPs on the 17.6 G convention (from clip_token_budget.stem_flops)
BRANCH_GFLOPS = [8.75, 12.99, 17.60]
DENSE_CIDER = 79.80          # E_karpathy_clean matched dense CLIP control
A1_MIN_CIDER = 78.3
A2_MAX_GFLOPS = 14.5
A3_MAX_BRANCH_FRAC = 0.80
A4_MIN_ABS_RHO = 0.30


def load_seeds(run_dir):
    """Prefer eval_only_summary.json (has the Karpathy-test metrics + routing audit),
    searched recursively; fall back to summary.json."""
    seeds = {}
    patterns = [
        os.path.join(run_dir, "**", "eval_only_summary.json"),
        os.path.join(run_dir, "**", "summary.json"),
    ]
    for pat in patterns:
        for sj in sorted(glob.glob(pat, recursive=True)):
            d = json.load(open(sj))
            key = f"seed_{d.get('seed', len(seeds))}"
            seeds.setdefault(key, d)   # first hit (eval_only) wins per seed
    return seeds


def realised_gflops(frac):
    return sum(f * g for f, g in zip(frac, BRANCH_GFLOPS))


def main(run_dir):
    seeds = load_seeds(run_dir)
    if not seeds:
        print(json.dumps({"error": "no seed summaries found", "run_dir": run_dir}))
        return
    ciders, bleus, realised, fracs, rhos = [], [], [], [], []
    for name, d in seeds.items():
        c = d.get("cider") or d.get("CIDEr") or d.get("final_cider") or d.get("test_cider")
        if c is not None and c < 5:   # pycocoevalcap sometimes reports 0-1 scale
            c *= 100
        if c is not None:
            ciders.append(c)
        b = d.get("bleu4") or d.get("final_bleu4")
        if b is not None:
            bleus.append(b)
        frac = d.get("inference_branch_fraction")
        if frac:
            fracs.append(frac)
            realised.append(d.get("realised_encoder_gflops") or realised_gflops(frac))
        if d.get("complexity_branch_spearman") is not None:
            rhos.append(d["complexity_branch_spearman"])

    def agg(xs):
        if not xs:
            return None
        m = st.mean(xs)
        s = st.pstdev(xs) if len(xs) < 2 else st.stdev(xs)
        return {"mean": round(m, 4), "std": round(s, 4), "values": xs}

    cider = agg(ciders)
    realised_agg = agg(realised) if realised else None
    max_branch = max((max(f) for f in fracs), default=None)

    a1 = cider is not None and cider["mean"] >= A1_MIN_CIDER
    a2 = realised_agg is not None and realised_agg["mean"] <= A2_MAX_GFLOPS
    a3 = max_branch is not None and max_branch <= A3_MAX_BRANCH_FRAC
    a4 = bool(rhos) and st.mean([abs(r) for r in rhos]) >= A4_MIN_ABS_RHO and \
        (all(r >= 0 for r in rhos) or all(r <= 0 for r in rhos))
    track1 = bool(a1 and a2 and a3)

    out = {
        "experiment": "E_phase2_tokenbudget",
        "n_seeds": len(seeds),
        "cider": cider,
        "bleu4": agg(bleus),
        "dense_control_cider": DENSE_CIDER,
        "realised_encoder_gflops": realised_agg,
        "max_branch_fraction": max_branch,
        "inference_branch_fractions": fracs,
        "complexity_branch_spearman": rhos or None,
        "criteria": {
            "A1_quality_parity": {"pass": a1, "threshold": f">= {A1_MIN_CIDER} CIDEr",
                                  "value": cider["mean"] if cider else None},
            "A2_real_flops_saving": {"pass": a2, "threshold": f"<= {A2_MAX_GFLOPS} G realised",
                                     "value": realised_agg["mean"] if realised_agg else None},
            "A3_non_collapse": {"pass": a3, "threshold": f"max branch <= {A3_MAX_BRANCH_FRAC}",
                                "value": max_branch},
            "A4_routing_informative": {"pass": a4, "threshold": f"|rho| >= {A4_MIN_ABS_RHO}, consistent sign",
                                       "value": rhos or None},
        },
        "VERDICT": "TRACK_1 (positive: parity + real FLOPs saving + non-collapse)" if track1
                   else "TRACK_2 (negative/anatomy: one of A1/A2/A3 failed)",
        "track1_gate": "A1 and A2 and A3",
    }
    print(json.dumps(out, indent=2))

    notes = os.path.join(run_dir, "NOTES.md")
    with open(notes, "w") as f:
        f.write(f"# Phase-2 token-budget result\n\n**VERDICT: {out['VERDICT']}**\n\n")
        f.write(f"- CIDEr {cider['mean'] if cider else '?'} vs dense {DENSE_CIDER} "
                f"(A1 parity >= {A1_MIN_CIDER}): {'PASS' if a1 else 'FAIL'}\n")
        f.write(f"- Realised FLOPs {realised_agg['mean'] if realised_agg else '?'} G "
                f"(A2 <= {A2_MAX_GFLOPS}): {'PASS' if a2 else 'FAIL'}\n")
        f.write(f"- Max branch fraction {max_branch} (A3 <= {A3_MAX_BRANCH_FRAC}): {'PASS' if a3 else 'FAIL'}\n")
        f.write(f"- Routing |rho| (A4 >= {A4_MIN_ABS_RHO}): {'PASS' if a4 else 'FAIL'}\n\n")
        f.write("Track 1 requires A1 and A2 and A3. See TNNLS_STATISTICAL_PLAN_PHASE2.md.\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
