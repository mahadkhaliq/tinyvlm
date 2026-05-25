# TNNLS_AUTOMATION.md — Autonomous Lead-side Driver for TinyVLM TNNLS Revision

> Drop-in agent spec for executing the **Lead-side vast.ai workload** from `../TNNLS_LEAD_INSTRUCTIONS.md` (top-level) on vast instance **37721982** in a phased, resumable, disk-aware, S3-backed pipeline. Modeled on the SpikeHippo `INSTRUCTION.md` template.
>
> **Scope:** This file automates only the six Lead E-tasks that require GPU time on vast: **E18, E19, E8, E9, E7, E15** (in that order — cheapest first). Mac-side work (E3, E6, E10 AI Hub, E13/E14 theory, E20 code-release, E21 framing rewrite, E22 integration) stays human-driven per Lead instructions. PhD-owned E-tasks (E1, E2, E4, E5, E11, E12, E16, E17) run on the PhD cluster, not here.
>
> **Repository note:** This file lives in the **nested** `tinyvlm/tinyvlm/` git tree (canonical source of `tinyvlm_vast.py`, 2372 lines). The top-level `tinyvlm/` workspace is the parent dir but is **not a git checkout** — it hosts planning docs (`TNNLS_LEAD_INSTRUCTIONS.md`, `TNNLS_PHD_INSTRUCTIONS.md`, `TNNLS_REVISION_PLAN.md`, `TinyVLM_Expert_Review.md`) and a separate, stale copy of `tinyvlm_vast.py`. Treat the nested copy as authoritative for all code/automation.

---

## Invocation model

This spec is **idempotent and re-entrant**. On every invocation:

1. Read `TNNLS_AUTO_STATE.md`; if missing, create with all phases `PENDING`.
2. Run **Pre-flight** (rclone reachable, vastai CLI usable, instance reachable, env vars present). Abort on any failure.
3. For each phase in declared order: skip if `COMPLETE`/`DEFERRED_*`; execute the first `PENDING`.
4. Per phase, in strict order: **Budget Guard → Disk Guard → Precondition Check → run → package → s3push → s3verify → scp pull → local verify → remote cleanup → mark COMPLETE**.
5. After every phase (success or deferral) push rolling state (`TNNLS_AUTO_STATE.md`, `FINDINGS_TNNLS.md`) to S3 so a crash mid-cycle is recoverable.
6. **Never** leave a phase partially applied without marking `IN_PROGRESS` with a recovery note.
7. May run one phase per session or as many as fit budget. S3 is the durable source of truth; local is a convenience copy.

---

## Environment

```bash
# Local Mac (caller side)
export TINYVLM_INSTANCE_ID=37721982
export TINYVLM_RCLONE_DEST="s3research:vastai-research/tinyvlm/tnnls"
export TINYVLM_LOCAL_ROOT="/Users/mahbub/Documents/projects/tinyvlm/tinyvlm"   # nested git tree
export TINYVLM_RESULTS_ROOT="${TINYVLM_LOCAL_ROOT}/tnnls_results"
export TINYVLM_GPU_BUDGET_HR=100        # total across all phases
export TINYVLM_MAX_UPLOADS=10           # ceiling on code re-uploads

# Vast instance (resolved at runtime via `vastai ssh-url`)
# Working dir on instance: /workspace
# Storage: 100 GB /workspace, 16 GB / (system, never put data here)
# GPU: 1× RTX 5090 (32 GB VRAM), driver 590.48.01, CUDA 13.1
# Python: 3.12.3 (vastai/pytorch:2.10.0-cu130 mini image — torch NOT preinstalled)
```

Active branch: `tnnls/plan-code-support` (current; this is where commit `6ebd6a1 — Add TNNLS baseline and routing eval controls` already landed). Push to `origin` only — `thenexuslab` mirror is **not** configured in this checkout despite memory note (verify before adding).

---

## S3 archival via rclone

Past `REVISION_STATE_QUAL.md` note "rclone unavailable on local Mac" is **stale**. Verified working: `rclone v1.74.0`, remote `s3research:vastai-research/` reachable, write+list under `tinyvlm/tnnls/` confirmed.

**Required env var:** `TINYVLM_RCLONE_DEST="s3research:vastai-research/tinyvlm/tnnls"`

**Three shell helpers** (install in `~/.bashrc` locally and on instance):

```bash
s3push() {
    local src="$1" suffix="$2"
    rclone copy --progress --transfers 8 --checkers 16 \
        "$src" "${TINYVLM_RCLONE_DEST}/${suffix}" \
        && echo "PUSHED: ${TINYVLM_RCLONE_DEST}/${suffix}"
}

s3verify() {
    local suffix="$1"
    rclone size "${TINYVLM_RCLONE_DEST}/${suffix}" 2>/dev/null \
        | grep -qE 'Total size: [1-9]'
}

s3pull() {
    local suffix="$1" dest="$2"
    mkdir -p "$dest"
    rclone copy --progress "${TINYVLM_RCLONE_DEST}/${suffix}" "$dest"
}
```

**S3 layout under `${TINYVLM_RCLONE_DEST}`:**

```
${TINYVLM_RCLONE_DEST}/
├── phase_E18_wallclock_cache/<YYYYMMDD>/
├── phase_E19_full_metrics/<YYYYMMDD>/
├── phase_E8_routing_adaptivity/<YYYYMMDD>/
├── phase_E9_soft_vs_hard/<YYYYMMDD>/
├── phase_E7_dense_smallenc/<YYYYMMDD>/
├── phase_E15_lambda_pareto/<YYYYMMDD>/
├── checkpoints/                # durable copies of kept ckpts
│   ├── phase_E7_dense_small/   # 3 seeds × best.pt
│   └── phase_E15_lambda_best/  # 5 settings × best.pt
├── state/                      # rolling state snapshots
│   ├── TNNLS_AUTO_STATE.md
│   ├── FINDINGS_TNNLS.md
│   └── final/                  # end-of-cycle snapshot
└── logs/                       # consolidated per-phase logs
```

**One-time instance setup** (executes during Phase 0; idempotent):

```bash
SSH_URL=$(vastai ssh-url ${TINYVLM_INSTANCE_ID} | sed 's|ssh://||')
SSH_HOST=$(echo "$SSH_URL" | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | cut -d: -f3)

scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    ~/.config/rclone/rclone.conf root@${SSH_HOST}:/root/.config/rclone/rclone.conf

ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no root@${SSH_HOST} \
    "mkdir -p /root/.config/rclone && chmod 600 /root/.config/rclone/rclone.conf && \
     cat >> /root/.bashrc << 'BASHRC_END'
export TINYVLM_RCLONE_DEST='${TINYVLM_RCLONE_DEST}'
s3push() { rclone copy --progress --transfers 8 --checkers 16 \"\$1\" \"\${TINYVLM_RCLONE_DEST}/\$2\"; }
s3verify() { rclone size \"\${TINYVLM_RCLONE_DEST}/\$1\" 2>/dev/null | grep -qE 'Total size: [1-9]'; }
s3pull() { mkdir -p \"\$2\"; rclone copy --progress \"\${TINYVLM_RCLONE_DEST}/\$1\" \"\$2\"; }
BASHRC_END"

# Sanity check on instance
ssh -p "$SSH_PORT" root@${SSH_HOST} "source /root/.bashrc && rclone lsd s3research:vastai-research/tinyvlm/" \
    > /dev/null || { echo "remote rclone broken"; exit 1; }
```

**Pre-flight check (start of every session):**

```bash
test -n "$TINYVLM_RCLONE_DEST" || { echo "TINYVLM_RCLONE_DEST unset"; exit 1; }
rclone lsd "${TINYVLM_RCLONE_DEST%/*}" > /dev/null \
    || { echo "rclone cannot reach remote"; exit 1; }
vastai show instance "$TINYVLM_INSTANCE_ID" | grep -q running \
    || { echo "instance not running — start: vastai start instance $TINYVLM_INSTANCE_ID"; exit 1; }
SSH_URL=$(vastai ssh-url ${TINYVLM_INSTANCE_ID} | sed 's|ssh://||')
SSH_HOST=$(echo "$SSH_URL" | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | cut -d: -f3)
ssh -p "$SSH_PORT" -o ConnectTimeout=10 root@${SSH_HOST} \
    "source /root/.bashrc && rclone lsd s3research:vastai-research/tinyvlm/" > /dev/null \
    || { echo "remote rclone broken — re-run one-time instance setup"; exit 1; }
```

**Push-verify-cleanup invariant:**

```
run_phase     → produces artifacts on remote /workspace/tnnls_results/E<id>/
package_phase → tars into /workspace/artifacts/phase_E<id>_results.tar.gz
s3push        → rclone copy FROM remote to S3 (no local bandwidth)
s3verify      → confirm uploaded object size > 0
local pull    → scp tar from remote to ./tnnls_results/E<id>/<date>/
local verify  → tar -tzf on downloaded archive
remote clean  → ONLY after s3verify passes (local failure recoverable via s3pull)
mark COMPLETE → in TNNLS_AUTO_STATE.md, record s3_uri + local_path
```

If S3 push itself fails: mark phase `IN_PROGRESS` with recovery note, log error verbatim in `FINDINGS_TNNLS.md`, exit. Do **not** clean up remote until S3 push succeeds on retry.

---

## State tracking — `TNNLS_AUTO_STATE.md`

YAML front-matter + markdown phase board at nested repo root (`tinyvlm/tinyvlm/`):

```yaml
---
instance_id: 37721982
rclone_dest: s3research:vastai-research/tinyvlm/tnnls
gpu_hours_budget: 100
gpu_hours_spent: 0.0
total_uploads: 0
max_uploads: 10
disk_safety_margin_gb: 10
local_downloads_root: ./tnnls_results/
started: 2026-05-25T00:00:00Z
schema_version: 1
---
```

Phase list (see `TNNLS_AUTO_STATE.md` for current state). Per-phase bullet on completion:

```
- [x] phase_E<id>_...  COMPLETE  gh_spent=X.X  remote_peak_gb=Y.Y  local_gb=Z.Z  \
      s3_uri=${TINYVLM_RCLONE_DEST}/phase_E<id>_.../YYYYMMDD/  finished=<iso8601>
```

On deferral:

```
- [~] phase_E<id>_...  DEFERRED_{DISK,BUDGET,PRECONDITION,GATE_FAIL}  reason="..."  \
      partial_s3_uri=<uri or "none">
```

---

## Disk Guard (run before every phase)

```bash
SSH_URL=$(vastai ssh-url ${TINYVLM_INSTANCE_ID} | sed 's|ssh://||')
SSH_HOST=$(echo "$SSH_URL" | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | cut -d: -f3)

REMOTE_AVAIL_GB=$(ssh -p $SSH_PORT root@$SSH_HOST \
    "df -BG /workspace | awk 'NR==2 {gsub(\"G\",\"\",\$4); print \$4}'")
LOCAL_AVAIL_GB=$(df -BG "$TINYVLM_LOCAL_ROOT" | awk 'NR==2 {gsub("G","",$4); print $4}')

# Read EST_REMOTE_GB, EST_LOCAL_GB from TNNLS_AUTO_STATE.md for current phase
if [ "$REMOTE_AVAIL_GB" -lt $((EST_REMOTE_GB + 10)) ]; then
    ssh -p $SSH_PORT root@$SSH_HOST "bash /workspace/scripts/cleanup_intermediate.sh"
    REMOTE_AVAIL_GB=$(ssh -p $SSH_PORT root@$SSH_HOST \
        "df -BG /workspace | awk 'NR==2 {gsub(\"G\",\"\",\$4); print \$4}'")
    if [ "$REMOTE_AVAIL_GB" -lt $((EST_REMOTE_GB + 10)) ]; then
        log_defer "phase_E${id}" DEFERRED_DISK \
            "remote_avail=${REMOTE_AVAIL_GB}G need=${EST_REMOTE_GB}G+10G margin"
        continue
    fi
fi
if [ "$LOCAL_AVAIL_GB" -lt $((EST_LOCAL_GB + 5)) ]; then
    log_defer "phase_E${id}" DEFERRED_DISK \
        "local_avail=${LOCAL_AVAIL_GB}G need=${EST_LOCAL_GB}G+5G margin"
    continue
fi
```

`scripts/cleanup_intermediate.sh` on instance:

```bash
#!/bin/bash
find /workspace/runs -name 'epoch_*.pt' -delete 2>/dev/null
rm -rf /root/.cache/pip 2>/dev/null
find /workspace -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null
ls -t /workspace/artifacts/phase_*_results.tar.gz 2>/dev/null | tail -n +2 | xargs rm -f
df -h /workspace
```

**Remote disk model:** 100 GB `/workspace`. Baseline after env + COCO ≈ 35 GB. Peak across all six phases if nothing cleaned ≈ 45 GB. Headroom comfortable; cleanup posture is **conservative** — keep intermediate checkpoints unless guard triggers.

---

## Budget Guard (run before every phase)

```bash
GH_REMAINING=$(python3 -c "import yaml; \
    d=yaml.safe_load(open('TNNLS_AUTO_STATE.md').read().split('---')[1]); \
    print(d['gpu_hours_budget'] - d['gpu_hours_spent'])")
if (( $(echo "$GH_REMAINING < $PHASE_BUDGET" | bc -l) )); then
    log_defer "phase_E${id}" DEFERRED_BUDGET \
        "remaining=${GH_REMAINING}h need=${PHASE_BUDGET}h"
    continue
fi
```

Total budget: **100 GPU-hr**. Track `gpu_hours_spent` after every phase via tmux session wall-clock. Skip priority on tight budgets:

1. Drop E15 λ₂ extremes (keep {0.05, 0.1, 0.5}, drop {0.01, 1.0}) → saves ~20 GPU-hr
2. Drop one E7 seed (keep 42, 43; drop 44) → saves ~8 GPU-hr
3. Defer E15 entirely if still tight — E7+E8+E9+E18+E19 already address fatal/critical issues.

---

## Upload workflow (Phase 0 + per-phase code changes)

```bash
SSH_URL=$(vastai ssh-url ${TINYVLM_INSTANCE_ID} | sed 's|ssh://||')
SSH_HOST=$(echo "$SSH_URL" | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | cut -d: -f3)

# Tar from nested git tree
tar --exclude='*.pt' --exclude='checkpoints_*' --exclude='runs_*' \
    --exclude='*.nosync' --exclude='__pycache__' --exclude='.git' \
    --exclude='*.tar.gz' --exclude='.qai_hub' --exclude='qai_hub_results' \
    --exclude='models/onnx' --exclude='jetson_results' --exclude='jetson_nano' \
    --exclude='jetson_nano_share.zip' --exclude='.DS_Store' \
    -czf /tmp/tinyvlm_src_$(date +%Y%m%d_%H%M).tar.gz \
    -C "$TINYVLM_LOCAL_ROOT" \
    tinyvlm_vast.py CLAUDE.md TNNLS_AUTOMATION.md requirements.txt

# Verify no secrets leaked
tar -tzf /tmp/tinyvlm_src_*.tar.gz | grep -iE 'qai_hub|\.env|client\.ini|token|secret' \
    && { echo "SECRET DETECTED — aborting"; exit 1; }

scp -P "$SSH_PORT" /tmp/tinyvlm_src_*.tar.gz root@${SSH_HOST}:/workspace/
ssh -p "$SSH_PORT" root@${SSH_HOST} \
    "cd /workspace && tar -xzf tinyvlm_src_*.tar.gz && rm tinyvlm_src_*.tar.gz"

# Increment total_uploads in TNNLS_AUTO_STATE.md
```

**Checkpoint upload** (one-time, Phase 0): Best STTF+ANC checkpoint from `../tinyvlm_full.nosync/.../best.pt` (top-level) is needed for E8, E9, E18, E19. ≈250 MB.

```bash
BEST_CKPT=$(ls -t /Users/mahbub/Documents/projects/tinyvlm/tinyvlm_full.nosync/*/checkpoints/best.pt 2>/dev/null | head -1)
test -f "$BEST_CKPT" || { echo "no best.pt found"; exit 1; }
ssh -p "$SSH_PORT" root@${SSH_HOST} "mkdir -p /workspace/ckpts/sttf_anc_cnn_full"
scp -P "$SSH_PORT" "$BEST_CKPT" \
    root@${SSH_HOST}:/workspace/ckpts/sttf_anc_cnn_full/best.pt
```

---

## Archive workflow (after every phase, before any remote cleanup)

```bash
DATE=$(date +%Y%m%d)
PHASE_NAME="phase_E${ID}_${LABEL}"
LOCAL_DIR="${TINYVLM_RESULTS_ROOT}/E${ID}/${DATE}"
S3_SUFFIX="${PHASE_NAME}/${DATE}"
mkdir -p "$LOCAL_DIR"

ssh -p $SSH_PORT root@$SSH_HOST "bash /workspace/scripts/package_phase_E${ID}.sh"

ssh -p $SSH_PORT root@$SSH_HOST "\
    source /root/.bashrc && cd /workspace/artifacts && \
    s3push phase_E${ID}_results.tar.gz ${S3_SUFFIX}/"

ssh -p $SSH_PORT root@$SSH_HOST \
    "source /root/.bashrc && s3verify ${S3_SUFFIX}/phase_E${ID}_results.tar.gz" \
    || { echo "S3 verify FAIL"; mark_phase_in_progress ${ID} "s3 verify failed"; exit 1; }

scp -P $SSH_PORT root@$SSH_HOST:/workspace/artifacts/phase_E${ID}_results.tar.gz \
    "$LOCAL_DIR/"
if [ $? -ne 0 ]; then
    echo "Local download failed; S3 copy exists — phase still COMPLETE"
    RECOVERY_NOTE="local_download_deferred; recover via: s3pull ${S3_SUFFIX}/ ${LOCAL_DIR}/"
fi

if [ -f "$LOCAL_DIR/phase_E${ID}_results.tar.gz" ]; then
    (cd "$LOCAL_DIR" && tar -tzf phase_E${ID}_results.tar.gz > /dev/null) \
        && (cd "$LOCAL_DIR" && tar -xzf phase_E${ID}_results.tar.gz) \
        || RECOVERY_NOTE="local_archive_corrupt; retry via s3pull"
fi

s3push TNNLS_AUTO_STATE.md state/TNNLS_AUTO_STATE.md
s3push FINDINGS_TNNLS.md   state/FINDINGS_TNNLS.md

# Per-phase remote cleanup runs here (see each phase below)
# Update TNNLS_AUTO_STATE.md (s3_uri, local_path, recovery_note, finished timestamp)
```

---

## Remote execution workflow

Launch each job inside a named tmux session:

```bash
ssh -p $SSH_PORT root@$SSH_HOST \
    "tmux new -d -s tnnls_E${ID} 'cd /workspace && \
     python tinyvlm_vast.py <args> > /workspace/logs/E${ID}.log 2>&1; \
     touch /workspace/logs/E${ID}.done'"
```

Polling cadence: `max(5 min, expected_duration / 20)`. Per-poll: tail last 40 lines of log, check for `.done` sentinel, parse epoch count. Tee summary into `FINDINGS_TNNLS.md` after each phase completes.

---

## Feedback loop

If a phase's success gate fails, may modify code and re-upload up to **3 times per phase**, total **10 uploads** across the cycle. After 3 iterations, mark `DEFERRED_GATE_FAIL`, append diagnosis to `FINDINGS_TNNLS.md`, continue to next phase.

---

## Code prerequisites (verified 2026-05-25)

All required flags **already landed** in nested `tinyvlm_vast.py` commit `6ebd6a1 (May 24 23:22 PDT)`:

| Flag | Line | Use | Default |
|---|---|---|---|
| `--encoder_only {tiny,small,medium}` | 2264 | E7 (with `--baseline dense`) | `"medium"` |
| `--eval_routing_mode {hard,soft}` | 2271 | E9 | `"hard"` |
| `--baseline {none,tokenlearner,dense}` | 2261 | E7 selects `dense` | `"none"` |
| `--lambda_flops` | 2238 | E15 sweep | `0.1` |
| `--seeds` | 2279 | E7, multi-seed | `None` |
| `--eval_only` | 2251 | E8/E9/E19 inference-only | `False` |
| `--clip_backbone` | 2218 | E8/E9 CLIP arm | `False` |

**Hard-routing implementation** (lines 791–812): `use_soft_routing = self.training or self.cfg.eval_routing_mode == "soft"`. At inference with `eval_routing_mode="hard"`, only the argmax branch executes — true FLOPs savings, samples grouped by `branch_idx`.

**DenseEncoderBaseline class** (lines 827+): separate from `AdaptiveNeuralCompression`; selected via `--baseline dense --encoder_only <branch>`. Has own `_ENCODER_FACTORY` dispatch dict.

No further code patches needed before running phases.

---

## Execute phases in declared order

Before every phase: Disk Guard → Budget Guard → Precondition Check. Any guard failure → defer + continue.

---

### Phase 0 — Instance setup (one-time)

**Preconditions:** vast instance running. `tinyvlm_full.nosync/.../best.pt` findable on Mac.
**GPU budget:** 0 hr.
**Disk:** +35 GB on instance.

```bash
SSH_URL=$(vastai ssh-url $TINYVLM_INSTANCE_ID | sed 's|ssh://||')
SSH_HOST=$(echo "$SSH_URL" | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | cut -d: -f3)
ssh -p $SSH_PORT root@$SSH_HOST "echo OK" || exit 1

# Install deps (mini image, no torch preinstalled)
ssh -p $SSH_PORT root@$SSH_HOST "\
    pip install --quiet torch==2.10.0 torchvision transformers tensorboard \
        pycocotools pillow tqdm bert-score onnx nltk scipy matplotlib \
        open_clip_torch pycocoevalcap pyyaml && \
    apt-get update -qq && apt-get install -y -qq default-jre-headless && \
    python -m nltk.downloader -q punkt punkt_tab"

# Configure rclone on instance (see S3 archival §one-time-instance-setup above)

# Upload code (tar workflow above)

# Upload reference checkpoint
BEST_CKPT=$(ls -t /Users/mahbub/Documents/projects/tinyvlm/tinyvlm_full.nosync/*/checkpoints/best.pt | head -1)
ssh -p $SSH_PORT root@$SSH_HOST "mkdir -p /workspace/ckpts/sttf_anc_cnn_full"
scp -P $SSH_PORT "$BEST_CKPT" root@$SSH_HOST:/workspace/ckpts/sttf_anc_cnn_full/best.pt

# Download COCO 2017
ssh -p $SSH_PORT root@$SSH_HOST "\
    mkdir -p /workspace/coco && cd /workspace/coco && \
    [ ! -d train2017 ] && wget -q http://images.cocodataset.org/zips/train2017.zip && unzip -q train2017.zip; \
    [ ! -d val2017 ] && wget -q http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip; \
    [ ! -d annotations ] && wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip; \
    rm -f *.zip"

# Verify import + ckpt load
ssh -p $SSH_PORT root@$SSH_HOST "cd /workspace && python -c \"
import torch
ckpt = torch.load('/workspace/ckpts/sttf_anc_cnn_full/best.pt', map_location='cpu', weights_only=False)
print('CKPT keys:', list(ckpt.keys())[:5])
import sys, importlib.util as u
spec = u.spec_from_file_location('tv', 'tinyvlm_vast.py')
m = u.module_from_spec(spec); sys.modules['tv'] = m; spec.loader.exec_module(m)
assert hasattr(m, 'Config') and hasattr(m, 'DenseEncoderBaseline')
print('OK')
\""

# Create dirs
ssh -p $SSH_PORT root@$SSH_HOST \
    "mkdir -p /workspace/artifacts /workspace/logs /workspace/scripts /workspace/tnnls_results"
```

**Success gate:** rclone reachable from instance, COCO present, ckpt loads, code imports clean, all required flags pre-verified.

**Artifacts:** none (setup-only). Mark COMPLETE without S3 push.

---

### Phase E18 — Wall-clock + cache memory

**Why:** Closes M3.
**Preconditions:** Phase 0 COMPLETE.
**Remote disk:** +0.1 GB.
**GPU budget:** 0.5 hr.

**Work:**

1. Scrape GPU-hours from existing logs (already on Mac — `tinyvlm_full.nosync/*/train.log` etc) — Mac-side, no GPU needed.
2. Instrument inference cache mem on instance:

```bash
ssh -p $SSH_PORT root@$SSH_HOST "tmux new -d -s tnnls_E18 \
  'cd /workspace && python -c \"
import torch, json, sys, importlib.util as u
spec = u.spec_from_file_location(\\\"tv\\\", \\\"tinyvlm_vast.py\\\")
m = u.module_from_spec(spec); sys.modules[\\\"tv\\\"] = m; spec.loader.exec_module(m)
cfg = m.Config(sttf_tau=0.80, eval_routing_mode=\\\"hard\\\")
model = m.AdaptiveNeuralCompression(cfg).cuda().eval()
ckpt = torch.load(\\\"/workspace/ckpts/sttf_anc_cnn_full/best.pt\\\", weights_only=False)
model.load_state_dict(ckpt[\\\"model\\\"] if \\\"model\\\" in ckpt else ckpt, strict=False)
torch.cuda.reset_peak_memory_stats()
rgb = torch.randn(1, 3, 224, 224).cuda()
ev  = torch.zeros(1, 2, 224, 224).cuda()
tok = torch.zeros(1, 64, dtype=torch.long).cuda()
with torch.no_grad():
    for _ in range(100):
        _ = model(rgb, ev, tok)
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(json.dumps({\\\"peak_mem_MB\\\": peak_mb, \\\"frames\\\": 100, \\\"tau\\\": 0.80}))
\" > /workspace/tnnls_results/E18/cache_mem.json 2>&1; \
   touch /workspace/logs/E18.done'"
```

3. ONNX sizes: `ls -lh /Users/mahbub/Documents/projects/tinyvlm/models/onnx/` (Mac-side).

**Success gate:** `cache_mem.json` has `peak_mem_MB > 0`; three GPU-hour numbers extracted.

**Package** (`scripts/package_phase_E18.sh`):

```bash
#!/bin/bash
mkdir -p /workspace/artifacts/E18_extracted
cp /workspace/tnnls_results/E18/cache_mem.json /workspace/artifacts/E18_extracted/
cp /workspace/logs/E18.log /workspace/artifacts/E18_extracted/ 2>/dev/null || true
cd /workspace/artifacts && tar -czf phase_E18_results.tar.gz E18_extracted/
```

**Remote cleanup:** none.

---

### Phase E19 — BLEU-4 / METEOR / SPICE everywhere

**Why:** Closes M5.
**Preconditions:** Phase 0 COMPLETE. Reference checkpoints uploaded.
**Remote disk:** +0.5 GB.
**GPU budget:** 5 hr (~1 hr per checkpoint × 5).

**Note:** Only STTF+ANC CNN best.pt uploaded in Phase 0. Other 4 (Dense Medium, Dense Small, TokenLearner, STTF+ANC CLIP) need their own ckpt. Dense Small does not exist yet (produced by E7); skip its row, add post-hoc after E7 completes. Upload available ckpts:

```bash
for tag in dense_medium tokenlearner sttf_anc_clip; do
    CKPT=$(ls -t /Users/mahbub/Documents/projects/tinyvlm/*${tag}*.nosync/*/checkpoints/best.pt 2>/dev/null | head -1)
    [ -f "$CKPT" ] && {
        ssh -p $SSH_PORT root@$SSH_HOST "mkdir -p /workspace/ckpts/${tag}"
        scp -P $SSH_PORT "$CKPT" root@$SSH_HOST:/workspace/ckpts/${tag}/best.pt
    }
done
```

**Write `tnnls_eval.py`** (commit to nested tree): loads ckpt, runs greedy beam=5 caption generation on COCO val, then `pycocoevalcap` for BLEU-1..4, METEOR, ROUGE-L, CIDEr, SPICE. SPICE requires Java (installed in Phase 0).

**Run per checkpoint:**

```bash
for tag in dense_medium tokenlearner sttf_anc_cnn_full sttf_anc_clip; do
    ssh -p $SSH_PORT root@$SSH_HOST "tmux new -d -s tnnls_E19_${tag} \
      'cd /workspace && python tnnls_eval.py \
         --ckpt /workspace/ckpts/${tag}/best.pt \
         --variant ${tag} \
         --out /workspace/tnnls_results/E19/${tag}_metrics.json \
         > /workspace/logs/E19_${tag}.log 2>&1; \
       touch /workspace/logs/E19_${tag}.done'"
done
```

**Success gate:** Per-ckpt JSON has BLEU-4, METEOR, CIDEr, SPICE all > 0. Aggregate `E19_summary.csv` has one row per ckpt.

**Package:** `tnnls_results/E19/*.json`, `E19_summary.csv`, logs, `tnnls_eval.py`.

**Cleanup:** keep ckpts for E8/E9.

---

### Phase E8 — Routing-adaptivity validation

**Why:** Closes C1.
**Preconditions:** Phase 0 COMPLETE. STTF+ANC CNN ckpt uploaded.
**Remote disk:** +0.3 GB.
**GPU budget:** 3 hr.

**Write `tnnls_routing_audit.py`** (commit to nested tree): load ckpt, hook `complexity_estimator` g_ψ output + `branch_idx = weights.argmax(dim=-1)` per COCO val sample. Save (g_ψ, branch) npz.

Per Lead E8 decision rule, emit `tnnls_results/E8/decision.json`:

```json
{"spearman_cnn": 0.XX, "spearman_clip": 0.YY,
 "per_branch_delta_cider_cnn": [...], "per_branch_delta_cider_clip": [...],
 "ci_threshold": ZZ, "verdict": "adaptive|load_balanced"}
```

**Decision rule:** if `|spearman| >= 0.2 AND max(per_branch_delta_cider) > ci_threshold` → `verdict=adaptive`, keep figure as Figure 3. Else → `verdict=load_balanced`, replace prose globally (Lead does this in `tinyvlm_tnnls_v1.tex`).

**Success gate:** `decision.json` exists with numeric `spearman_*` and `verdict ∈ {adaptive, load_balanced}`. Scatter PDF produced.

**Package:** audit script, npz pairs, scatter PDF, per-branch metrics, decision.json.

**Cleanup:** delete npz after download.

---

### Phase E9 — Soft vs hard routing gap

**Why:** Closes C2.
**Preconditions:** Phase 0 COMPLETE. CNN+CLIP STTF+ANC ckpts uploaded. `--eval_routing_mode` already in code.
**Remote disk:** +0.1 GB.
**GPU budget:** 2 hr.

```bash
for backbone in cnn clip; do
    CKPT_PATH=/workspace/ckpts/sttf_anc_${backbone}_full/best.pt
    [ "$backbone" = "clip" ] && BACKBONE_FLAG="--clip_backbone" || BACKBONE_FLAG=""
    for mode in soft hard; do
        ssh -p $SSH_PORT root@$SSH_HOST "tmux new -d -s tnnls_E9_${backbone}_${mode} \
          'cd /workspace && python tinyvlm_vast.py \
             --eval_only \
             --resume ${CKPT_PATH} \
             --eval_routing_mode ${mode} \
             ${BACKBONE_FLAG} \
             --coco_imgs /workspace/coco/val2017 \
             --coco_anns /workspace/coco/annotations/captions_val2017.json \
             --output_dir /workspace/tnnls_results/E9/${backbone}_${mode} \
             > /workspace/logs/E9_${backbone}_${mode}.log 2>&1; \
           touch /workspace/logs/E9_${backbone}_${mode}.done'"
    done
done
```

Aggregate: `E9_summary.json` with `{cnn: {soft:X, hard:Y, delta_cider:X-Y}, clip:{...}}`.

**Success gate:** Two ΔCIDEr numbers (one per backbone), each with one-sentence interpretation.

**Cleanup:** none.

---

### Phase E7 — FLOPs-matched Dense SmallEncoder (3 seeds)

**Why:** Closes C6. Uses `DenseEncoderBaseline` (separate class from ANC, line 827+).
**Preconditions:** Phase 0 COMPLETE. `--baseline dense` + `--encoder_only small` both available (verified).
**Remote disk:** +12 GB peak, +0.5 GB after cleanup.
**GPU budget:** 30 hr (RTX 5090 ≈ 10 hr/run × 3 seeds sequential).

```bash
for seed in 42 43 44; do
    ssh -p $SSH_PORT root@$SSH_HOST "tmux new -d -s tnnls_E7_seed${seed} \
      'cd /workspace && python tinyvlm_vast.py \
         --coco_imgs /workspace/coco/train2017 \
         --coco_anns /workspace/coco/annotations/captions_train2017.json \
         --baseline dense --encoder_only small --seeds ${seed} \
         --epochs 10 --batch_size 16 \
         --output_dir /workspace/runs/E7_dense_small_seed${seed} \
         > /workspace/logs/E7_seed${seed}.log 2>&1; \
       touch /workspace/logs/E7_seed${seed}.done'"
done
```

Aggregate across 3 seeds → `aggregate.json` with CIDEr/BLEU-4/METEOR/SPICE mean ± 95% CI + Cohen's d vs STTF+ANC (from existing logs).

**Success gate:** 3 `summary.json`, `aggregate.json`, FLOPs estimate ∈ [7, 12] GFLOPs.

**Package:** per-seed `summary.json` + `metrics.jsonl`, `aggregate.json`, logs. **Best.pt excluded from tar**; pushed separately to `${TINYVLM_RCLONE_DEST}/checkpoints/phase_E7_dense_small/seed_<S>/best.pt`.

**Cleanup:** after S3 push, on each seed run delete `epoch_*.pt`, keep `best.pt + summary.json`. Push best.pt to S3 checkpoints prefix.

---

### Phase E15 — λ₂ Pareto frontier (5 settings)

**Why:** Closes T4.
**Preconditions:** Phase 0 COMPLETE. (`--lambda_flops` exists at line 2238.)
**Remote disk:** +20 GB peak, +0.8 GB after cleanup.
**GPU budget:** 50 hr (5 runs × 10 hr).

```bash
for lam in 0.01 0.05 0.1 0.5 1.0; do
    ssh -p $SSH_PORT root@$SSH_HOST "tmux new -d -s tnnls_E15_lam${lam} \
      'cd /workspace && python tinyvlm_vast.py \
         --coco_imgs /workspace/coco/train2017 \
         --coco_anns /workspace/coco/annotations/captions_train2017.json \
         --seeds 42 --epochs 10 --batch_size 16 \
         --lambda_flops ${lam} \
         --output_dir /workspace/runs/E15_lambda_${lam} \
         > /workspace/logs/E15_lam${lam}.log 2>&1; \
       touch /workspace/logs/E15_lam${lam}.done'"
done
```

Plus scrape per-epoch E[FLOPs] trajectory from existing `tinyvlm_full.nosync/`, `tinyvlm_balanced.nosync/` logs (Mac-side, no GPU).

Aggregate: `pareto.json` + `fig:lambda_pareto.pdf`.

**Success gate:** 5 points, Pareto frontier renders.

**Package:** 5 × summary.json, pareto.json, pareto plot, λ₂-trajectory plot, logs. Best-λ₂ ckpt to `checkpoints/phase_E15_lambda_best/`.

**Cleanup:** keep best-λ₂ ckpt, delete other 4. Drop all `epoch_*.pt`.

**Budget-tight contingency:** if `gpu_hours_remaining < 30` at phase start, drop λ₂ ∈ {0.01, 1.0} — run only {0.05, 0.1, 0.5}, save 20 GPU-hr. Mark `partial_run: dropped_extremes` in NOTES.

---

## Final actions (after E15 success gate or budget exhaustion)

```bash
cd "$TINYVLM_LOCAL_ROOT"
git add tnnls_results/ TNNLS_AUTO_STATE.md FINDINGS_TNNLS.md tnnls_eval.py tnnls_routing_audit.py
git commit -m "tnnls: lead-side vast cycle complete (E18, E19, E8, E9, E7, E15)"
git push origin tnnls/plan-code-support
# (thenexuslab mirror not configured; add via `git remote add` only if access verified)

# Append Full-cycle summary to FINDINGS_TNNLS.md:
# - phases COMPLETE / DEFERRED with reasons
# - gpu_hours_spent / peak_remote_gb
# - submission readiness against TNNLS issue inventory (F/C/T/M codes)
# - S3 archive index (per-phase URIs)

s3push TNNLS_AUTO_STATE.md state/final/TNNLS_AUTO_STATE.md
s3push FINDINGS_TNNLS.md   state/final/FINDINGS_TNNLS.md

vastai stop instance $TINYVLM_INSTANCE_ID
```

---

## Operating rules

- **Read-only inputs** (top-level workspace): `../TNNLS_LEAD_INSTRUCTIONS.md`, `../TNNLS_PHD_INSTRUCTIONS.md`, `../TNNLS_REVISION_PLAN.md`, `../TinyVLM_Expert_Review.md`, `../TinyVLM_Review_Journal_Recommendations.md`, `../tinyvlm_4.3.tex`, `../CLAUDE.md`. Never edit during automation.
- **Editable** (nested tree): `TNNLS_AUTO_STATE.md`, `FINDINGS_TNNLS.md`, `tnnls_results/E*/`, `tnnls_eval.py`, `tnnls_routing_audit.py`.
- **Never edit:** `tinyvlm_tnnls_v1.tex` (paper-side integration is Lead-manual per file ownership table).
- **Disk Guard / Budget Guard / Precondition Check run before every phase.**
- **Push-verify-clean ordering strict:** s3push → s3verify → scp local → local verify → remote cleanup. Never invert.
- **Secret hygiene:** every upload tar runs `grep -iE 'qai_hub|\.env|client\.ini|token|secret'` and aborts on hit. `.qai_hub/`, `qai_hub_results/`, `~/.zshrc` never uploaded.
- **Tier-1 priority:** if budget forces partial completion, E18+E19+E8+E9 close M5/C1/C2/M3 (paper-critical). E7 closes C6 (also critical). E15 closes T4 (theory — defer first if budget tight).
- **Go/no-go for paper integration is Lead-manual.** Automation produces `aggregate.json` per E-task; Lead reads, decides, edits manuscript.
- **AI Hub token rotation** is user's manual task post-submission (per `security_tokens.md` memory). Not in this pipeline.
- **Top-level stale copies:** A divergent `tinyvlm_vast.py` (1953 lines) and helper docs exist in the parent `tinyvlm/` workspace. Treat as scratch only; nested copy is canonical.

---

## Phase dependency graph

```
Phase 0 (setup) ─┬─> E18 (cheap, validates instance)
                 ├─> E19 (existing ckpts only)
                 ├─> E8  (existing ckpt)
                 ├─> E9  (existing ckpt; uses --eval_routing_mode)
                 ├─> E7  (new training; uses --baseline dense --encoder_only small)
                 └─> E15 (new training; uses --lambda_flops sweep)

E7 unblocks → E19 Dense SmallEnc row (post-hoc append, no separate phase).
```

Single GPU → sequential. Execute lowest-budget first to amortize setup risk.

---

## Deliverables

**On Mac (nested tree), end of cycle:**

- `tnnls_results/E{18,19,8,9,7,15}/<date>/` — per-phase archives.
- `TNNLS_AUTO_STATE.md` — all phases COMPLETE/DEFERRED_* + S3 archive index.
- `FINDINGS_TNNLS.md` — per-phase outcomes + headline numbers + cycle summary.
- Branch `tnnls/plan-code-support` pushed to `origin` with per-phase commits.

**In S3, durable:**

- `${TINYVLM_RCLONE_DEST}/phase_E*/<date>/` — all per-phase tars + extracted.
- `${TINYVLM_RCLONE_DEST}/checkpoints/phase_{E7,E15}_*/` — kept best.pt files.
- `${TINYVLM_RCLONE_DEST}/state/` — rolling + final snapshots.
- `${TINYVLM_RCLONE_DEST}/logs/` — consolidated logs.
