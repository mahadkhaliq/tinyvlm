# INSTRUCTION_TNNLS_QUAL.md — Qualcomm AI Hub benchmarking for TNNLS revision

> Paste this block into a Claude Code session opened in `~/Documents/projects/tinyvlm`.
> Phased, resumable, and idempotent: each invocation picks up from the first `PENDING` phase,
> checks quota and disk before executing, and updates `REVISION_STATE_TNNLS_QUAL.md` after each
> phase before exiting.

---

Based on `@INSTRUCTION_QUAL.md` (prior cycle structure), `@FINDINGS_QUAL.md` (prior measurements),
`@tinyvlm_tnnls_v1.tex` (TNNLS revision paper), and `@TNNLS_STATE.json` (experiment registry),
execute the **TNNLS Qualcomm AI Hub benchmarking cycle** in phased, resumable mode.

## Context and goal

The prior Qual cycle (COMPLETE, see `REVISION_STATE_QUAL.md`) produced:
- CNN STTF+ANC and Dense MediumEnc. latency on SD 888 + X Elite for the captioning model
  (encoder + captioning decoder; 5 ONNX files in `models/onnx/`, AI Hub job IDs recorded)
- Those numbers are already in `tinyvlm_tnnls_v1.tex` `tab:ondevice`

**Three remaining gaps this cycle must close:**

| Gap | Closes | Status |
|-----|--------|--------|
| N-Caltech101 latency in `tab:ncaltech` is *estimated* (FLOPs-ratio scaling), not directly measured | E10 / C3 | **Actionable now** |
| CLIP-redesign (E4) on-device profile: real per-branch FLOPs gap after E4 CLIP-ANC redesign | E4 / F3 | **Blocked — no E4 ckpt yet** |
| Routing-weight update: re-aggregate `tab:ondevice` latency once 5-seed E11 routing histogram is available | E11 / C4 | **Blocked — E11 held** |

**Primary objective:** close E10 — export encoder-only classification ONNX for ANC+Dense,
profile on SD 888 and at least one newer device using AI Hub token 2 (~9 free jobs),
replace the FLOPs-scaled estimates in `tab:ncaltech` with direct measurements.

## Invocation model

Idempotent and re-entrant. On every invocation:

1. Read `REVISION_STATE_TNNLS_QUAL.md`; if missing, create it with all phases `PENDING`.
2. Skip phases marked `COMPLETE` or `DEFERRED`. Execute the first `PENDING` phase.
3. After each phase: update `REVISION_STATE_TNNLS_QUAL.md` and `FINDINGS_TNNLS_QUAL.md`
   before exiting or continuing.
4. Never leave a phase partially applied without marking it `IN_PROGRESS` with a recovery note.
5. Run the AI Hub Quota Guard before every phase that submits jobs. If quota would be exceeded,
   mark the phase `DEFERRED_QUOTA` and continue to the next.

## Environment

Project root: `~/Documents/projects/tinyvlm`.

**Existing artefacts (do not re-export or re-profile):**
- `models/onnx/cnn_sttf_anc_b{0,1,2}.onnx` — encoder + captioning decoder, all three ANC branches
- `models/onnx/cnn_dense.onnx` — Dense MediumEnc., encoder + captioning decoder
- `models/onnx/cnn_tokenlearner.onnx` — TokenLearner, encoder + captioning decoder
- `qai_hub_results/jobs.json` — prior-cycle job IDs (SD 888 and X Elite captioning profiles)

**Checkpoints available now:**
- `tinyvlm_cider_seeds.nosync/seed_42_tau_0.80/checkpoints/best.pt` (CNN STTF+ANC, COCO)
- `tokenlearner_baseline.nosync/seed_42_tau_0.80/checkpoints/best.pt`

**Checkpoints not yet available (blocked phases will defer automatically):**
- E4 CLIP-redesign checkpoint — produced by E4 training (held until E1 week-1 diagnosis)
- E11 5-seed CNN checkpoints — produced by E11 (held)

Branch: create `tnnls/qual-bench` before any code change. Never commit `*.pt`, `*.onnx`,
`qai_hub_results/`, or `*.nosync/` content.

## AI Hub credentials — token 2

The prior cycle used the original AI Hub token (revoke after submission if not already done).
This cycle uses **token 2** configured in `murshed@gmail.com` account: ~9 free jobs available
(per `TNNLS_STATE.json` E10 note). Generate/confirm at
`https://aihub.qualcomm.com/` → Settings → API Tokens, then:

```bash
export QAI_HUB_API_TOKEN=<token-2>
qai-hub configure --api_token "$QAI_HUB_API_TOKEN"
qai-hub list-devices | head -20
```

Preferred devices for this cycle:
- `Samsung Galaxy S21 (Family)` — SD 888; matches prior cycle for direct comparison
- `Samsung Galaxy S23 (Family)` — SD 8 Gen 2; was skipped (quota) in prior cycle
- `Snapdragon X Elite CRD` — already profiled; re-use prior numbers, no new jobs needed

## S3 archival

Use the `s3push` helper if rclone is available:

```bash
export TINYVLM_RCLONE_DEST="s3test:vastai-research/ws-34754072/tinyvlm-finalpush"
test -n "$TINYVLM_RCLONE_DEST" || { echo "TINYVLM_RCLONE_DEST unset — local-only mode"; }
rclone lsd "${TINYVLM_RCLONE_DEST%/*}" > /dev/null 2>&1 \
    && echo "rclone reachable" || echo "rclone unavailable — skip S3, local-only"
```

If rclone is unavailable (as in the prior cycle), skip S3 and record `s3_verified_phases: []`
in the state file. Artefacts are retained locally.

S3 layout under `${TINYVLM_RCLONE_DEST}/tnnls_qual/`:
```
phase_tq1_ncaltech_onnx/<YYYYMMDD>/
phase_tq2_ncaltech_profile/<YYYYMMDD>/
phase_tq3_clip_redesign/<YYYYMMDD>/   ← blocked until E4 ckpt
phase_tq4_aggregate/<YYYYMMDD>/
phase_tq5_routing_update/<YYYYMMDD>/  ← blocked until E11
state/
```

## State tracking — `REVISION_STATE_TNNLS_QUAL.md`

Create at project root if missing:

```yaml
---
ai_hub_quota_estimate_jobs: 9
ai_hub_jobs_spent: 0
token: token-2
disk_safety_margin_gb: 2
local_results_root: ./qai_hub_results/
rclone_dest: check-at-runtime
s3_verified_phases: []
started: <today>
---

# Phase status

- [ ] phase_tq1_ncaltech_onnx     PENDING   est_jobs=0  est_local_gb=0.2
- [ ] phase_tq2_ncaltech_profile  PENDING   est_jobs=6  est_local_gb=0.1
- [ ] phase_tq3_clip_redesign     PENDING   est_jobs=3  est_local_gb=0.2  note=BLOCKED_E4_CKPT
- [ ] phase_tq4_aggregate         PENDING   est_jobs=0  est_local_gb=0.1
- [ ] phase_tq5_routing_update    PENDING   est_jobs=0  est_local_gb=0.1  note=BLOCKED_E11

# Deferred / skipped
(append entries here)

# S3 archive index
(append one line per completed phase)
```

## AI Hub Quota Guard

Before any phase that submits jobs:

```bash
JOBS_REMAINING=$((AI_HUB_QUOTA_ESTIMATE_JOBS - AI_HUB_JOBS_SPENT))
if [ "$JOBS_REMAINING" -lt "$EST_JOBS" ]; then
    log_defer phase_tq${N} DEFERRED_QUOTA "remaining=${JOBS_REMAINING} need=${EST_JOBS}"
    continue
fi
```

---

## Phase TQ1 — N-Caltech101 encoder-only ONNX export

**What and why.** The prior cycle exported encoder + captioning *decoder* for captioning
profiling. N-Caltech101 is a *classification* task: the model is
`encoder-branch → projection → linear(384 → 101)` — no decoder. Exporting this smaller graph
gives accurate N-Caltech101 inference latency. The linear head weights do not affect latency
measurements; random initialisation is fine.

**Preconditions.**
- `tinyvlm_cider_seeds.nosync/seed_42_tau_0.80/checkpoints/best.pt` exists.
- Branch `tnnls/qual-bench` created.

**AI Hub jobs.** 0.

**Code change required.** Add `_ANCEncoderClassifierWrapper` to `qai_hub_bench.py` and add a
`"ncaltech_anc"` variant to `cmd_export_onnx`. Insert after the `_ANCBranchWrapper` class:

```python
class _ANCEncoderClassifierWrapper(nn.Module):
    """ANC single-branch encoder + linear classifier head for recognition profiling.

    No captioning decoder — gives accurate N-Caltech101 / DVS128 latency.
    Weights of the linear head don't affect latency; random init is fine.
    """

    def __init__(self, anc_model, branch: int, num_classes: int = 101):
        super().__init__()
        self.encoder = anc_model.encoders[branch]
        self.projection = anc_model.projections[branch]
        self.classifier = nn.Linear(384, num_classes)

    def forward(self, rgb: torch.Tensor, events: torch.Tensor):
        feat = self.encoder(rgb, events)
        encoded = self.projection(feat)
        return self.classifier(encoded)
```

In `_build_model`, add a branch after the `tokenlearner` block:

```python
if variant == "ncaltech_anc":
    model = AdaptiveNeuralCompression(cfg)
    model.load_state_dict(state, strict=False)
    model.eval()
    return _ANCEncoderClassifierWrapper(model, branch, num_classes=101), cfg
```

In `cmd_export_onnx`, add input/output handling for `"ncaltech_anc"` (two inputs: `rgb`,
`events`; one output: `class_logits`):

```python
if args.variant == "ncaltech_anc":
    dummy_rgb = torch.randn(1, 3, 224, 224)
    dummy_events = torch.zeros(1, 2, 56, 56)
    torch.onnx.export(
        wrapper,
        (dummy_rgb, dummy_events),
        str(out_path),
        opset_version=17,
        input_names=["rgb", "events"],
        output_names=["class_logits"],
        dynamo=False,
    )
```

**Export commands.** After adding the new variant:

```bash
CNN_CKPT="tinyvlm_cider_seeds.nosync/seed_42_tau_0.80/checkpoints/best.pt"

# ANC three branches
for B in 0 1 2; do
    python qai_hub_bench.py export-onnx \
        --variant ncaltech_anc \
        --checkpoint "$CNN_CKPT" \
        --output models/onnx/ncaltech_sttf_anc_b${B}.onnx \
        --branch ${B}
done

# Dense MediumEnc. classification baseline (branch 2 = Medium, no router)
python qai_hub_bench.py export-onnx \
    --variant ncaltech_anc \
    --checkpoint "$CNN_CKPT" \
    --output models/onnx/ncaltech_dense.onnx \
    --branch 2
```

**Local CPU sanity check.** Verify all four ONNX files run without error:

```bash
python -c "
import onnxruntime as ort, numpy as np, glob
for path in sorted(glob.glob('models/onnx/ncaltech_*.onnx')):
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    rgb = np.zeros((1,3,224,224), dtype=np.float32)
    ev  = np.zeros((1,2,56,56),   dtype=np.float32)
    out = sess.run(None, {'rgb': rgb, 'events': ev})[0]
    assert out.shape == (1, 101), f'unexpected shape {out.shape} for {path}'
    assert not (out != out).any(), f'NaN in {path}'
    print(f'{path}: OK  shape={out.shape}  mean={out.mean():.3f}')
"
```

**Success gate.** 4 ONNX files (`ncaltech_sttf_anc_b{0,1,2}.onnx`, `ncaltech_dense.onnx`)
pass ONNX checker + onnxruntime CPU; total size ≤ 120 MB (no decoder, much smaller than
captioning variants).

**Failure mode.** If `AdaptiveNeuralCompression` does not have `.encoders` / `.projections`
attributes (e.g., loaded from an older checkpoint), call `model.encoders = model.encoder`
or adapt to match the attribute names in the loaded state dict.

**Artifacts:** `models/onnx/ncaltech_*.onnx`, `qai_hub_results/tq1_export.log`.

---

## Phase TQ2 — N-Caltech101 AI Hub profiling (E10)

**Preconditions.** Phase TQ1 COMPLETE.
**AI Hub jobs.** Up to 6 (3 branches + dense × SD 888; optionally + SD 8 Gen 2 if quota allows).
**Quota guard.** EST_JOBS=6; if JOBS_REMAINING < 6, profile SD 888 only (3 jobs: b1 + dense +
one more branch) and mark S23 DEFERRED_QUOTA.

**SD 888 profiling (Samsung Galaxy S21 — matches prior cycle for direct comparison):**

```bash
DEV_888="Samsung Galaxy S21 (Family)"
for B in 0 1 2; do
    python qai_hub_bench.py submit \
        --onnx models/onnx/ncaltech_sttf_anc_b${B}.onnx \
        --device "$DEV_888" \
        --label ncaltech_sttf_anc_b${B}_s21
done
python qai_hub_bench.py submit \
    --onnx models/onnx/ncaltech_dense.onnx \
    --device "$DEV_888" \
    --label ncaltech_dense_s21

# Poll until all 4 complete
while true; do
    DONE=0
    for L in ncaltech_sttf_anc_b0_s21 ncaltech_sttf_anc_b1_s21 \
              ncaltech_sttf_anc_b2_s21 ncaltech_dense_s21; do
        STATUS=$(python qai_hub_bench.py poll --label "$L" 2>&1 | tail -1)
        echo "$STATUS"
        echo "$STATUS" | grep -q SUCCESS && DONE=$((DONE+1))
    done
    [ $DONE -eq 4 ] && break
    echo "waiting... $DONE/4 done"
    sleep 60
done
```

**Optional SD 8 Gen 2 profiling (if quota JOBS_REMAINING ≥ 6 after SD 888):**

```bash
DEV_S23="Samsung Galaxy S23 (Family)"
for B in 0 1 2; do
    python qai_hub_bench.py submit \
        --onnx models/onnx/ncaltech_sttf_anc_b${B}.onnx \
        --device "$DEV_S23" \
        --label ncaltech_sttf_anc_b${B}_s23
done
python qai_hub_bench.py submit \
    --onnx models/onnx/ncaltech_dense.onnx \
    --device "$DEV_S23" \
    --label ncaltech_dense_s23
```

**Success gate.** All submitted jobs return `SUCCESS`; per-job profile JSON in `qai_hub_results/`;
`latency_ms` populated in `jobs.json` for each label.

**Routing-weighted aggregation.** Use N-Caltech101 routing utilisation from `TNNLS_STATE.json`
or COCO proxy `(f0=0.31, f1=0.34, f2=0.35)` until E11 provides real N-Caltech101 weights:

```python
import json
jobs = json.load(open("qai_hub_results/jobs.json"))
ROUTING = {"b0": 0.31, "b1": 0.34, "b2": 0.35}

for suffix in ["s21", "s23"]:
    keys = [f"ncaltech_sttf_anc_b{b}_{suffix}" for b in range(3)]
    if not all(k in jobs and jobs[k].get("latency_ms") for k in keys):
        print(f"  {suffix}: not all jobs complete, skipping")
        continue
    weighted = sum(jobs[f"ncaltech_sttf_anc_b{b}_{suffix}"]["latency_ms"] * ROUTING[f"b{b}"]
                   for b in range(3))
    dense_ms = jobs.get(f"ncaltech_dense_{suffix}", {}).get("latency_ms", "n/a")
    print(f"  {suffix}  STTF+ANC={weighted:.2f} ms  Dense={dense_ms}")
```

**Note on routing weights.** The aggregated latency will need to be *re-computed* in Phase TQ5
once E11 supplies real N-Caltech101 routing histograms. Record the proxy value now; update later.

**Artifacts:** `qai_hub_results/ncaltech_*_profile.json`, updated `qai_hub_results/jobs.json`.

---

## Phase TQ3 — CLIP-redesign ONNX export + AI Hub profiling (E4)

**Status: BLOCKED.** This phase cannot proceed until E4 (CLIP-ANC redesign with real per-branch
FLOPs gap) produces trained checkpoints. At the start of each session, check:

```bash
ls tinyvlm_clip_redesign.nosync/seed_42*/checkpoints/best.pt 2>/dev/null \
    && echo "E4 ckpt found — proceed with TQ3" \
    || echo "E4 ckpt not found — deferring TQ3"
```

If no checkpoint is found, mark the phase `DEFERRED_E4_CKPT` and proceed to TQ4.

**When E4 checkpoint is available.** Export three CLIP-redesign branches
(variant `clip_anc`, adapting `_CLIPBranchWrapper` if needed to match E4 architecture changes)
and profile on SD 888:

```bash
E4_CKPT="tinyvlm_clip_redesign.nosync/seed_42_tau_0.80/checkpoints/best.pt"
DEV_888="Samsung Galaxy S21 (Family)"

for B in 0 1 2; do
    python qai_hub_bench.py export-onnx \
        --variant clip_anc \
        --checkpoint "$E4_CKPT" \
        --output models/onnx/clip_redesign_b${B}.onnx \
        --branch ${B}
    python qai_hub_bench.py submit \
        --onnx models/onnx/clip_redesign_b${B}.onnx \
        --device "$DEV_888" \
        --label clip_redesign_b${B}_s21
done
```

**Goal.** Verify that E4's redesigned CLIP branches (which should have real FLOPs differences,
unlike the current CLIP-ANC where all branches run the 17.6 GFLOPs ViT and have ~98 ms latency)
now show measurable per-branch latency differences. If `b0` is still within 2% of `b2`, note
that the FLOPs redesign has not yet produced on-device savings.

**Artifacts:** `models/onnx/clip_redesign_b{0,1,2}.onnx`,
`qai_hub_results/clip_redesign_*_profile.json`.

---

## Phase TQ4 — Aggregate and patch `tinyvlm_tnnls_v1.tex`

**Preconditions.** Phase TQ2 COMPLETE (SD 888 N-Caltech101 numbers available).
**AI Hub jobs.** 0.

**Aggregate N-Caltech101 numbers:**

```bash
python qai_hub_bench.py aggregate --output qai_hub_results/tnnls_table_ncaltech.json
```

Manually verify the JSON contains `ncaltech_sttf_anc_weighted_ms` and `ncaltech_dense_ms`
for each device profiled.

**Paper patch.** Copy `tinyvlm_tnnls_v1.tex` to `tinyvlm_tnnls_v2.tex`. Update `tab:ncaltech`:

1. Replace the FLOPs-scaled latency estimates in the STTF+ANC and Dense rows with the directly
   measured values from `tnnls_table_ncaltech.json`.
2. Update the table caption: remove "estimated by scaling the Snapdragon 888 captioning
   measurements … by the per-method FLOPs ratio" and replace with "measured via Qualcomm
   AI Hub on Snapdragon 888 (Samsung Galaxy S21); ONNX classification-head variant (no
   captioning decoder)."
3. Add AI Hub job IDs in a footnote: `\footnote{AI Hub job IDs for N-Caltech101 encoder
   profiling: \texttt{<ids>}.}`
4. If S23 jobs completed: add an S23 column or note in the caption.
5. In §4.4 prose: update the latency claim to cite the measured value.

**Do not change** `tab:ondevice` (captioning latency) — those numbers are already measured
and correct in the prior cycle.

**Compile and verify:**

```bash
pdflatex -interaction=nonstopmode tinyvlm_tnnls_v2.tex
pdflatex -interaction=nonstopmode tinyvlm_tnnls_v2.tex
pdftotext tinyvlm_tnnls_v2.pdf - | grep -c "estimated by scaling"
# must return 0 — no FLOPs-scaling language remaining for N-Caltech101
```

**Success gate.** `tinyvlm_tnnls_v2.pdf` compiles clean; `tab:ncaltech` latency rows contain
measured values with AI Hub job IDs cited; no "estimated by scaling" text remains for
N-Caltech101. Closes E10 / C3.

**Artifacts:** `tinyvlm_tnnls_v2.tex`, `tinyvlm_tnnls_v2.pdf`,
`qai_hub_results/tnnls_table_ncaltech.json`.

---

## Phase TQ5 — Routing-weight update for `tab:ondevice` (E11)

**Status: BLOCKED.** Requires E11 5-seed sweep results, which are HELD until E1 week-1
diagnosis completes.

**No new AI Hub jobs needed.** The existing SD 888 captioning per-branch latencies
(Tiny=11.08 ms, Small=20.11 ms, Medium=40.11 ms) are already measured. Once E11 provides
real 5-seed routing histograms `(f0, f1, f2)`, simply re-aggregate:

```python
f0, f1, f2 = <E11 values>  # replace COCO proxy (0.31, 0.34, 0.35)
weighted = f0 * 11.078 + f1 * 20.110 + f2 * 40.111
print(f"updated routing-weighted latency: {weighted:.2f} ms")
```

If the new weights differ from 0.31/0.34/0.35 by more than 5% relative, update the
`tab:ondevice` caption and prose in `tinyvlm_tnnls_v2.tex`.

At the start of each session, check whether E11 is complete:

```bash
python -c "
import json
state = json.load(open('TNNLS_STATE.json'))
print(state['experiments']['E11']['status'])
"
# 'held' → skip; 'complete' → proceed
```

---

## Final actions (after TQ4 success gate, or after all non-blocked phases complete)

1. Update `TNNLS_STATE.json`: set `E10.status = "complete"`, add `artifacts` entry pointing
   to `qai_hub_results/tnnls_table_ncaltech.json` and the AI Hub job IDs.
2. Commit all changes to `tnnls/qual-bench`; push to origin. One commit per phase.
3. Append `## TNNLS Qual cycle summary` to `FINDINGS_TNNLS_QUAL.md` with: phases complete,
   phases deferred (with reasons), jobs consumed, job IDs, measured latency table,
   and notes on what changes once TQ3/TQ5 unblock.
4. If rclone is available: push phase archives to S3 under `tnnls_qual/`.
5. Print terminal summary: E10 status, AI Hub job IDs, N-Caltech101 measured latency,
   remaining quota on token 2, and path to `tinyvlm_tnnls_v2.pdf`.

## Deliverables (after TQ2+TQ4 complete)

- `models/onnx/ncaltech_*.onnx` — classification-head ONNX variants (no decoder)
- `qai_hub_results/ncaltech_*_profile.json` — raw AI Hub profile JSONs
- `qai_hub_results/tnnls_table_ncaltech.json` — aggregated latency table
- `tinyvlm_tnnls_v2.tex` and `tinyvlm_tnnls_v2.pdf` — paper with measured N-Caltech101 latency
- `REVISION_STATE_TNNLS_QUAL.md` and `FINDINGS_TNNLS_QUAL.md` — cycle state + findings
- `TNNLS_STATE.json` — E10 marked complete
- Branch `tnnls/qual-bench` pushed to origin

## Operating rules

- Read-only inputs (never edit directly): `INSTRUCTION_QUAL.md`, `FINDINGS_QUAL.md`,
  `tinyvlm_tnnls_v1.tex`. The patch targets a fresh copy `tinyvlm_tnnls_v2.tex`.
- Use AI Hub token 2. Do not exceed the ~9-job estimate without checking with the supervisor.
- **Honest framing.** Name the actual chip profiled. Do not extrapolate SD 888 measurements
  to S23/S24 unless those jobs actually ran.
- If any phase takes > 30 min of queue wait on AI Hub, back off polling to 5-min intervals
  and log queue time in `FINDINGS_TNNLS_QUAL.md`.
- If `torch.onnx.export` fails on the new `_ANCEncoderClassifierWrapper`, check attribute
  names against the loaded checkpoint's state dict keys and adapt the wrapper. Do not
  fabricate numbers.
- Blocked phases (TQ3, TQ5) should be checked at the start of each session and promoted
  from `DEFERRED` to `PENDING` once their preconditions are met.
- The reviewer-blocking issue for E10 is exactly one sentence: "N-Caltech101 latency is
  estimated, not measured." This instruction's primary purpose is to change that sentence
  to cite real AI Hub job IDs.
