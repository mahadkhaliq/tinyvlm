# INSTRUCTION_QUAL.md — Claude Code Automation Prompt for TinyVLM Qualcomm AI Hub Benchmarks

> Paste the block below into a Claude Code session opened in `~/Documents/projects/tinyvlm`. The prompt is **phased, resumable, disk-aware, and S3-backed**: it can be invoked multiple times across sessions, and on each invocation it picks up from the next pending phase, checks AI Hub queue + local disk headroom before executing, downloads results and necessary artifacts after each phase, **pushes each phase's artifacts to S3 via the user-configured rclone helper before any local cleanup**, and skips any phase whose estimated footprint would exceed available disk (logging the skip as Deferred and moving to the next phase).

---

Based on `@INSTRUCTION.md` (format), `@tinyvlm_4.tex` (paper requiring on-device numbers), and `@tinyvlm_consolidated3.md` (S2/S6 deferred items), execute the **Qualcomm AI Hub on-device benchmarking cycle** for TinyVLM in **phased, resumable** mode.

## Goal

Produce verifiable Snapdragon on-device latency + peak memory measurements for the CNN-backbone STTF + ANC model and three reviewer-relevant baselines (Dense MediumEnc., TokenLearner, ANC-only) so Table 5 of `tinyvlm_4.tex` can be backed by raw artefacts (per-run latency CSV, profile JSON, AI Hub job IDs). The CLIP backbone variant is profiled if time permits; otherwise it remains TBD.

This instruction does **not** train, fine-tune, or evaluate captioning quality — those phases are owned by `INSTRUCTION.md` / `instruction_max.md` / `instruction_mahad.md`. This instruction only consumes existing checkpoints and produces ONNX exports + AI Hub profile reports.

## Invocation model

This prompt is **idempotent and re-entrant**. On every invocation:

1. Read `REVISION_STATE_QUAL.md` (schema below); if missing, create it with all phases marked `PENDING`.
2. For each phase Q1 → Q7, skip any phase marked `COMPLETE` or `DEFERRED`. Execute the first phase marked `PENDING`.
3. After executing a phase to its success gate (or failing), **push the phase's artifacts to S3** via the `s3push` helper (see "S3 archival" below), then update `REVISION_STATE_QUAL.md` and `FINDINGS_QUAL.md` before exiting or moving on.
4. Never leave a phase partially applied without updating `REVISION_STATE_QUAL.md` to `IN_PROGRESS` with a recovery note.
5. Before every phase, run the AI Hub Quota Guard (below). If the AI Hub free-tier monthly quota would be exceeded, mark the phase `DEFERRED_QUOTA` and proceed.
6. S3 is the **durable source of truth** for phase artefacts. If a local download fails but the S3 push succeeded, the phase is still `COMPLETE` (with a recovery note); the user can `s3pull` later.

## Environment

Project root: `~/Documents/projects/tinyvlm` (this directory). Source: `tinyvlm_vast.py`. Existing checkpoints local: `tinyvlm_cider_seeds.nosync/seed_{42,43,44}_tau_0.80/checkpoints/best.pt` (CNN), Hellbender `/tmp/tinyvlm_runs/clip_seed{42,43,44}/seed_*_tau_0.80/checkpoints/best.pt` (CLIP). TokenLearner baseline checkpoint: `tokenlearner_baseline.nosync/seed_42_tau_0.80/checkpoints/best.pt`.

**No GPU instance required.** All work runs on the local Mac (ONNX export, AI Hub job submission, result download). AI Hub itself runs the inference on real Snapdragon silicon in Qualcomm's cloud.

Create a git branch `qual-ai-hub-bench` before any code change. Never commit `*.pt`, `*.onnx`, `qai_hub_results/`, or `*.nosync/` content.

## AI Hub credentials

Sign up at <https://aihub.qualcomm.com/> using `murshed@gmail.com`. Free tier provides limited monthly device-hours sufficient for ≤ 6 profile jobs (each ~30 s of device time + queue wait). Generate API token under **Settings → API Tokens** and export:

```bash
export QAI_HUB_API_TOKEN=<token>
qai-hub configure --api_token "$QAI_HUB_API_TOKEN"
```

Verify with:

```bash
qai-hub list-devices | head -20
```

Expected devices for this cycle:
- `Samsung Galaxy S23 (Family)` — Snapdragon 8 Gen 2; closest published surrogate for Snapdragon 888 mobile-class measurements.
- `Samsung Galaxy S24 (Family)` — Snapdragon 8 Gen 3; latest mobile NPU.
- `Snapdragon X Elite CRD` — laptop-class NPU (replaces 888-class column if X2 Elite unavailable).

Jetson Nano is **not** available on AI Hub (NVIDIA hardware). Either (a) leave Jetson row of Table 5 as "deferred to camera-ready" with the same reframe used in `tinyvlm_4.1.tex`, or (b) measure separately on a Lambda-Cloud Jetson Orin Nano image in a future invocation.

## S3 archival via rclone

The user's existing TinyVLM rclone pipeline applies. Required environment variable:

```bash
export TINYVLM_RCLONE_DEST="s3test:vastai-research/ws-34754072/tinyvlm-finalpush"
```

The three reusable helpers (`s3push`, `s3verify`, `s3pull`) from `INSTRUCTION.md` are reused as-is — they are already installed on the local Mac via `~/.bashrc`.

**Pre-flight check at the start of each session:**

```bash
test -n "$TINYVLM_RCLONE_DEST" || { echo "TINYVLM_RCLONE_DEST unset"; exit 1; }
rclone lsd "${TINYVLM_RCLONE_DEST%/*}" > /dev/null \
    || { echo "rclone cannot reach remote"; exit 1; }
```

**S3 layout under `${TINYVLM_RCLONE_DEST}`:**

```
${TINYVLM_RCLONE_DEST}/qual/
├── phase_q1_export/<YYYYMMDD>/    # ONNX files per (backbone, variant, branch)
├── phase_q2_validate/<YYYYMMDD>/  # local-CPU sanity-check outputs
├── phase_q3_s23/<YYYYMMDD>/       # AI Hub Snapdragon 8 Gen 2 profiles
├── phase_q4_s24/<YYYYMMDD>/       # AI Hub Snapdragon 8 Gen 3 profiles
├── phase_q5_xelite/<YYYYMMDD>/    # AI Hub Snapdragon X Elite profiles
├── phase_q6_aggregate/<YYYYMMDD>/ # combined Table 5 JSON + paper patch
├── phase_q7_paper/<YYYYMMDD>/     # tinyvlm_4.2.tex + compiled PDF
└── state/                         # rolling REVISION_STATE_QUAL.md, FINDINGS_QUAL.md
    └── final/                     # canonical end-of-cycle snapshot
```

**Push-verify-cleanup ordering** is identical to `INSTRUCTION.md`: package → s3push → s3verify → local archive → mark COMPLETE.

## State tracking — `REVISION_STATE_QUAL.md`

Maintain at project root:

```yaml
---
ai_hub_quota_estimate_jobs: 12        # free-tier ceiling per month, conservative
ai_hub_jobs_spent: 0
total_uploads: 0
max_uploads: 8
disk_safety_margin_gb: 2
local_results_root: ./qai_hub_results/
rclone_dest: ${TINYVLM_RCLONE_DEST}
s3_verified_phases: []
started: 2026-05-05T00:00:00Z
---

# Phase status

- [ ] phase_q1_onnx_export        PENDING   est_jobs=0 est_local_gb=0.5
- [ ] phase_q2_local_validate     PENDING   est_jobs=0 est_local_gb=0.1
- [ ] phase_q3_snapdragon_8gen2   PENDING   est_jobs=4 est_local_gb=0.1
- [ ] phase_q4_snapdragon_8gen3   PENDING   est_jobs=4 est_local_gb=0.1
- [ ] phase_q5_x_elite            PENDING   est_jobs=2 est_local_gb=0.1
- [ ] phase_q6_aggregate          PENDING   est_jobs=0 est_local_gb=0.1
- [ ] phase_q7_paper_patch        PENDING   est_jobs=0 est_local_gb=0.1

# Deferred / skipped
(append entries here)

# S3 archive index
(append one line per completed phase)
```

After each phase, update the corresponding bullet with both local path and S3 URI.

## AI Hub Quota Guard (run before every phase)

```bash
JOBS_REMAINING=$((AI_HUB_QUOTA_ESTIMATE_JOBS - AI_HUB_JOBS_SPENT))
if [ "$JOBS_REMAINING" -lt "$EST_JOBS" ]; then
    log_defer phase_q${N} DEFERRED_QUOTA "remaining=${JOBS_REMAINING} need=${EST_JOBS}"
    continue
fi
```

If a phase is deferred via DEFERRED_QUOTA, the next session may resume after the AI Hub monthly window resets, or the user may upgrade to a paid tier and re-run without the guard.

## Helper script

A single Python driver `qai_hub_bench.py` (already at project root) exposes four subcommands: `export-onnx`, `submit`, `poll`, `aggregate`. Each phase below names the exact invocations.

---

## Execute phases Q1 through Q7 in strict order

Before each phase: run AI Hub Quota Guard, run Disk Guard, verify preconditions on prior phase outputs.

---

### Phase Q1 — ONNX export

**Preconditions.** Local checkpoints staged at:
- `tinyvlm_cider_seeds.nosync/seed_42_tau_0.80/checkpoints/best.pt` (CNN STTF+ANC, seed 42)
- `tokenlearner_baseline.nosync/seed_42_tau_0.80/checkpoints/best.pt` (TokenLearner baseline)
- `tinyvlm_balanced.nosync/seed_42_tau_0.80/checkpoints/best.pt` (Dense MediumEnc. baseline; ANC disabled)

**Local disk estimate.** +0.5 GB (4 model variants × 3 branches × ~40 MB each).
**AI Hub jobs.** 0.

**Work.** Export four model variants to ONNX (opset 17), one branch per file for ANC variants:

```bash
mkdir -p models/onnx
for SEED in 42; do
    # CNN STTF+ANC, all 3 branches
    for B in 0 1 2; do
        python qai_hub_bench.py export-onnx \
            --backbone cnn --variant sttf_anc \
            --checkpoint tinyvlm_cider_seeds.nosync/seed_${SEED}_tau_0.80/checkpoints/best.pt \
            --output models/onnx/cnn_sttf_anc_b${B}.onnx --branch ${B}
    done
    # Dense MediumEnc. (single path, no router)
    python qai_hub_bench.py export-onnx \
        --backbone cnn --variant dense \
        --checkpoint tinyvlm_balanced.nosync/seed_${SEED}_tau_0.80/checkpoints/best.pt \
        --output models/onnx/cnn_dense.onnx --branch 2
    # TokenLearner baseline
    python qai_hub_bench.py export-onnx \
        --backbone cnn --variant dense \
        --checkpoint tokenlearner_baseline.nosync/seed_${SEED}_tau_0.80/checkpoints/best.pt \
        --output models/onnx/cnn_tokenlearner.onnx --branch 2
done
```

**Success gate.** 5 ONNX files (`cnn_sttf_anc_b{0,1,2}.onnx`, `cnn_dense.onnx`, `cnn_tokenlearner.onnx`) all parse with `onnx.checker.check_model`. Total size ≤ 250 MB.

**Failure mode (Gumbel-softmax router).** If `torch.onnx.export` fails on the Gumbel softmax + hard argmax, the export script forces a single branch via `model.force_branch(b)` and exports the resulting non-conditional graph. If `force_branch` does not exist on the loaded model, add it (set `model._forced_branch = b` and short-circuit the router in `forward()`); re-export.

**Artifacts manifest:**
- `phase_q1_results.tar.gz` containing: `models/onnx/*.onnx`, `qai_hub_results/q1_export.log`, ONNX checker output.

**S3 push:** `s3push phase_q1_results.tar.gz qual/phase_q1_export/$(date +%Y%m%d)/`.

---

### Phase Q2 — Local CPU sanity check

**Preconditions.** Phase Q1 COMPLETE.
**Local disk estimate.** +0.1 GB (per-model output tensors for diff).
**AI Hub jobs.** 0.

**Work.** Run each ONNX file through `onnxruntime` on the local Mac with a fixed dummy input (zeros, shape `(1, 3, 224, 224)`); compare to PyTorch eager-mode output. Diff tolerance: `atol=1e-3` (FP32). This catches export bugs before submitting AI Hub jobs (which cost quota).

```bash
python -c "
import onnxruntime as ort, torch, numpy as np, glob
from tinyvlm_vast import build_model_from_checkpoint, Config
for path in sorted(glob.glob('models/onnx/*.onnx')):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    x = np.zeros((1,3,224,224), dtype=np.float32)
    out = sess.run(None, {'image': x})[0]
    print(f'{path}: shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}')
"
```

**Success gate.** All 5 ONNX files run without runtime errors; output shape matches expected `(1, vocab_size, max_len)` or branch-equivalent; output stats finite (no NaN/Inf).

**Artifacts manifest:**
- `phase_q2_results.tar.gz` containing: `qai_hub_results/q2_validate.log`, output-stats CSV per file.

---

### Phase Q3 — AI Hub profile on Snapdragon 8 Gen 2 (Galaxy S23)

**Preconditions.** Phase Q2 COMPLETE.
**Local disk estimate.** +0.1 GB (per-job profile JSON ~5 MB).
**AI Hub jobs.** 4 (3 ANC branches + dense). TokenLearner deferred to Q4 to keep S23 quota under cap.

**Work.** Submit jobs sequentially (batched submit is OK if SDK supports; sequential is safe), poll until done.

```bash
DEV="Samsung Galaxy S23 (Family)"
for B in 0 1 2; do
    python qai_hub_bench.py submit \
        --onnx models/onnx/cnn_sttf_anc_b${B}.onnx \
        --device "$DEV" --label cnn_sttf_anc_b${B}_s23
done
python qai_hub_bench.py submit \
    --onnx models/onnx/cnn_dense.onnx \
    --device "$DEV" --label cnn_dense_s23

# Poll every 60 s until all 4 are COMPLETE
while true; do
    DONE=0
    for L in cnn_sttf_anc_b{0,1,2}_s23 cnn_dense_s23; do
        STATUS=$(python qai_hub_bench.py poll --label "$L" 2>&1 | tail -1)
        echo "$STATUS"
        echo "$STATUS" | grep -q SUCCESS && DONE=$((DONE+1))
    done
    [ $DONE -eq 4 ] && break
    sleep 60
done
```

**Success gate.** All 4 jobs return `SUCCESS` from AI Hub; per-job profile JSON downloaded to `qai_hub_results/`; latency in milliseconds populated in `jobs.json` for each label.

**Artifacts manifest:**
- `phase_q3_results.tar.gz` containing: `qai_hub_results/{cnn_sttf_anc_b*_s23,cnn_dense_s23}_profile.json`, `qai_hub_results/jobs.json`, AI Hub job IDs.

**S3 push:** `s3push phase_q3_results.tar.gz qual/phase_q3_s23/$(date +%Y%m%d)/`.

---

### Phase Q4 — AI Hub profile on Snapdragon 8 Gen 3 (Galaxy S24)

**Preconditions.** Phase Q3 COMPLETE (so quota usage is known).
**AI Hub jobs.** 4 (same 3 ANC branches + dense; TokenLearner only if quota remaining ≥ 5).

Same workflow as Q3, replace `DEV="Samsung Galaxy S24 (Family)"` and label suffix `_s24`. Plus optional TokenLearner submission if quota allows (prefer S24 over S23 for the TokenLearner row since S24 is the higher-traffic reviewer-relevant chip).

**Success gate.** Same as Q3.

---

### Phase Q5 — AI Hub profile on Snapdragon X Elite

**Preconditions.** Phase Q4 COMPLETE.
**AI Hub jobs.** 2 (just `cnn_sttf_anc_b1` as the routing-mean stand-in plus `cnn_dense` for the side-by-side; full 3-branch sweep deferred to keep quota).

```bash
DEV="Snapdragon X Elite CRD"
python qai_hub_bench.py submit --onnx models/onnx/cnn_sttf_anc_b1.onnx --device "$DEV" --label cnn_sttf_anc_b1_xelite
python qai_hub_bench.py submit --onnx models/onnx/cnn_dense.onnx       --device "$DEV" --label cnn_dense_xelite
```

Poll loop identical to Q3.

**Success gate.** 2 jobs return SUCCESS; latency populated.

---

### Phase Q6 — Aggregate into Table 5 numbers

**Preconditions.** Phases Q3–Q5 COMPLETE.
**AI Hub jobs.** 0.

**Work.** Combine per-branch latencies into the routing-distribution-weighted mean for STTF+ANC. Routing distribution from N-Caltech eval is `(0.31, 0.34, 0.35)`; from CNN-COCO eval is similar (record exact numbers from `tinyvlm_cider_seeds.nosync/seed_42_tau_0.80/metrics.jsonl` final epoch).

```bash
python qai_hub_bench.py aggregate --output qai_hub_results/table5_measured.json
python -c "
import json
runs = json.load(open('qai_hub_results/table5_measured.json'))
ROUTING = {'b0':0.31, 'b1':0.34, 'b2':0.35}
for dev, d in runs.items():
    sttf_anc = sum(d.get(f'cnn_sttf_anc_{b}_s23',0)*ROUTING[b] for b in ROUTING)
    print(f'{dev}: STTF+ANC weighted = {sttf_anc:.2f} ms; dense = {d.get(\"cnn_dense_s23\",0):.2f} ms')
" > qai_hub_results/table5_summary.txt
cat qai_hub_results/table5_summary.txt
```

Peak memory + standard deviation are read directly from each profile JSON's `execution_summary`.

**Success gate.** `table5_measured.json` contains keyed entries for each (device, variant) pair; routing-weighted STTF+ANC latency computed and within an order of magnitude of FLOPs-implied estimate (rough sanity).

**Artifacts manifest:**
- `phase_q6_results.tar.gz` containing: `qai_hub_results/table5_measured.json`, `qai_hub_results/table5_summary.txt`, weighting calculation log.

---

### Phase Q7 — Paper patch (`tinyvlm_4.2.tex`)

**Preconditions.** Phase Q6 COMPLETE.
**AI Hub jobs.** 0.

**Work.** Copy `tinyvlm_4.tex` to `tinyvlm_4.2.tex`. Replace Table 5 numbers with measured values from `qai_hub_results/table5_measured.json`. Update §"On-Device Edge Evaluation" prose to:
- Cite **Snapdragon 8 Gen 2** (Galaxy S23) and **Snapdragon 8 Gen 3** (Galaxy S24) — drop "Snapdragon 888" since AI Hub doesn't carry it. Strict honesty: write the actual chips profiled.
- If X Elite jobs succeeded, add a third row.
- Replace "Battery Manager API" / "ina3221" power-monitor sentence with the Qualcomm AI Hub protocol: "models exported to ONNX (opset 17) and profiled on AI Hub via Hexagon NPU execution provider; latency averaged over the AI Hub default 100-iteration profile after warm-up."
- Drop or rephrase any Jetson row: either remove it (and the Jetson sentence) or keep with "TBD — measured separately on Jetson Orin Nano in supplementary material."
- Add AI Hub job IDs in a footnote for reproducibility (e.g., `\footnote{AI Hub job IDs: \texttt{j-...,j-...}}`).

Compile:

```bash
pdflatex -interaction=nonstopmode tinyvlm_4.2.tex
pdflatex -interaction=nonstopmode tinyvlm_4.2.tex
pdftotext tinyvlm_4.2.pdf - | awk '/\f/{n++} /^References$/{print "Refs page",n+1; exit}'
```

If References lands on page > 9, run the same trim cuts as `tinyvlm_4.1.tex` (drop ANC fig pair, compress N-Caltech, etc.) until References on page ≤ 9.

**Success gate.** `tinyvlm_4.2.pdf` compiles clean; main content ≤ 9 pages; Table 5 cells reference measured numbers tied to AI Hub job IDs in the footnote; no `Snapdragon 888` text remains unless AI Hub somehow added it.

**Artifacts manifest:**
- `phase_q7_results.tar.gz` containing: `tinyvlm_4.2.tex`, `tinyvlm_4.2.pdf`, `qai_hub_results/`, footnote text snippet, FINDINGS_QUAL.md updates.

**S3 push:** `s3push phase_q7_results.tar.gz qual/phase_q7_paper/$(date +%Y%m%d)/`. Final state snapshot: `s3push tinyvlm_4.2.pdf qual/state/final/tinyvlm_4.2.pdf`.

---

## Final actions (after Phase Q7 success gate or abort)

1. Commit all changes to branch `qual-ai-hub-bench`; push to origin. One clean commit per phase.
2. Append a `## Full-cycle summary` section to `FINDINGS_QUAL.md` with: phases COMPLETE, phases DEFERRED (with reasons), AI Hub jobs consumed, AI Hub job IDs (so reviewers can be pointed at them if requested), table of measured per-device latencies, peak memory, and the **S3 archive index**.
3. **Final S3 state sync:** `s3push REVISION_STATE_QUAL.md state/REVISION_STATE_QUAL.md`, `s3push FINDINGS_QUAL.md state/FINDINGS_QUAL.md`, `s3push tinyvlm_4.2.{tex,pdf} qual/state/final/`.
4. Print a terminal summary: AI Hub job IDs, per-device latencies, S3 archive index, location of `tinyvlm_4.2.pdf`.

## Deliverables

**On local machine, at end of cycle:**
- `qai_hub_results/jobs.json`, `qai_hub_results/table5_measured.json`, per-job profile JSONs.
- `models/onnx/*.onnx` (5 files).
- `tinyvlm_4.2.tex` and `tinyvlm_4.2.pdf` at project root.
- `REVISION_STATE_QUAL.md` and `FINDINGS_QUAL.md` at project root.
- Branch `qual-ai-hub-bench` pushed to origin.

**In S3:**
- `${TINYVLM_RCLONE_DEST}/qual/phase_q{1..7}_*/<date>/` per-phase archives.
- `${TINYVLM_RCLONE_DEST}/qual/state/final/{tinyvlm_4.2.pdf, REVISION_STATE_QUAL.md, FINDINGS_QUAL.md}`.

## Operating rules

- Read-only inputs (never edit): `INSTRUCTION.md`, `tinyvlm_consolidated*.md`, `tinyvlm_neurips_review*.md`, `tinyvlm_4.tex`, `tinyvlm_4.1.tex`. The 4.2 patch happens on a fresh copy.
- Run AI Hub Quota Guard before every phase. Run Disk Guard before every phase. Run Precondition Check before every phase.
- **Verify the rclone pipeline before the first phase** each session: `$TINYVLM_RCLONE_DEST` set, `rclone lsd "${TINYVLM_RCLONE_DEST%/*}"` succeeds.
- **Push-verify-cleanup ordering is strict.** Never invert.
- **Honest framing.** Paper text must name the actual chips profiled (S23 = SD 8 Gen 2; S24 = SD 8 Gen 3; X Elite). Do not relabel measurements as "Snapdragon 888" if 888 was not the device.
- If AI Hub queue times exceed 30 min on any single job, log it in `FINDINGS_QUAL.md` and back off polling to every 5 min to reduce API call rate.
- If `torch.onnx.export` fails on the conditional ANC routing, fall back to per-branch single-path export and weight-aggregate at Phase Q6. Never fabricate numbers.
- Reviewer-blocking issue is exactly one: "edge paper without measured edge runtime." This instruction's whole purpose is to close that issue. Do not declare the cycle complete until `tinyvlm_4.2.pdf` has Table 5 cells tied to AI Hub job IDs.
