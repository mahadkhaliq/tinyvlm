#!/bin/bash
# Aggregate STTF+ANC vs TokenLearner matched-seed paired stats.
# Phase 11 — runs after Phase 5 baselines complete.

set -e
cd "$(dirname "$0")"

BASE=/workspace/runs

OUT_DIR=tables
mkdir -p "$OUT_DIR"

echo "=== Building multi-seed summaries ==="
python3 - << 'PY'
import json, glob
from pathlib import Path

def aggregate(name, dirs, out):
    per = []
    for d in dirs:
        sf = Path(d) / "summary.json"
        if not sf.exists():
            print(f"  MISSING: {sf}")
            continue
        per.append(json.loads(sf.read_text()))
    if not per:
        print(f"NO DATA for {name}")
        return
    summary = {
        "name": name,
        "seeds": [r.get("seed") for r in per],
        "per_seed": per,
        "n_seeds": len(per),
    }
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"WROTE: {out}  ({len(per)} seeds)")

import os
BASE = "/workspace/runs"

aggregate(
    "STTF+ANC (CNN)",
    [f"{BASE}/tinyvlm_cider_seeds/seed_{s}_tau_0.80" for s in (42, 43, 44)],
    f"{BASE}/tinyvlm_cider_seeds/multi_seed_summary.json",
)

aggregate(
    "TokenLearner (matched protocol)",
    [f"{BASE}/baselines/tokenlearner_3seed/seed_{s}_tau_0.80" for s in (42, 43, 44)],
    f"{BASE}/baselines/tokenlearner_matched_summary.json",
)

aggregate(
    "STTF+ANC (CLIP, matched protocol)",
    [f"{BASE}/clip_3seed/seed_{s}_tau_0.80" for s in (42, 43, 44)],
    f"{BASE}/clip_3seed/multi_seed_summary.json",
)
PY

echo
echo "=== Paired stats: STTF+ANC vs TokenLearner ==="
python3 paired_stats.py \
    --a "$BASE/tinyvlm_cider_seeds/multi_seed_summary.json" \
    --b "$BASE/baselines/tokenlearner_matched_summary.json" \
    --out "$OUT_DIR/paired_sttfanc_vs_tokenlearner.json" 2>&1 | tee "$OUT_DIR/paired_sttfanc_vs_tokenlearner.txt"
