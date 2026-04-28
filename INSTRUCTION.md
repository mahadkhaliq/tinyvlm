# INSTRUCTION.md — Claude Code Automation Prompt for TinyVLM NeurIPS 2026 Revision

> Paste the block below into a Claude Code session opened in the TinyVLM project directory with `claude --dangerously-skip-permissions`. The prompt is **phased, resumable, disk-aware, and S3-backed**: it can be invoked multiple times across sessions. On each invocation it picks up from the next pending phase, checks remote and local disk headroom before executing, downloads results after each phase, pushes each phase's artifacts to S3 via rclone before any remote cleanup, and skips any phase whose estimated footprint exceeds available disk (logging the skip as DEFERRED and moving on).

---

Based on `IMPLEMENTATION-PLAN.md`, `tinyvlm_vast.py`, and `tinyvlm2.2.tex`, execute the NeurIPS 2026 major-revision cycle for TinyVLM in **phased, resumable** mode.

## Invocation model

This prompt is **idempotent and re-entrant**. On every invocation:

1. Read `REVISION_STATE.md` (schema below); if missing, create it with all phases marked `PENDING`.
2. For each phase 1 → 4, skip any phase marked `COMPLETE` or `DEFERRED`. Execute the first phase marked `PENDING`.
3. After a phase reaches its success gate (or fails), **push artifacts to S3** via the user-configured rclone helper, download to local, then update `REVISION_STATE.md` and `FINDINGS.md` before exiting or moving to the next phase.
4. You may run one phase per session or as many as fit the session budget — but **never leave a phase partially applied without updating `REVISION_STATE.md` to `IN_PROGRESS` with a recovery note**.
5. Before every phase, run the Disk Guard. If estimated remote or local disk need exceeds available headroom, mark the phase `DEFERRED_DISK` and proceed to the next. If a later phase cannot run without a deferred earlier phase's outputs, cascade the deferral with a note.
6. S3 is treated as the **durable source of truth**. If a local download fails but the S3 push succeeded, the phase is still marked `COMPLETE` (with a recovery note).

## Environment

Project root is the current directory. Source code is `tinyvlm_vast.py`. Paper source is `tinyvlm2.2.tex`. Figures are in `Figure/`. Vast.ai instance **id `$TINYVLM_INSTANCE_ID`** (set this env var before running; see pre-flight checks). Start it if not running via `vastai start instance $TINYVLM_INSTANCE_ID`; stop (do not destroy) after the last completed phase via `vastai stop instance $TINYVLM_INSTANCE_ID`.

Remote workspace: `/workspace/tinyvlm/`. All training outputs go under `/workspace/tinyvlm/runs/`.

Create a git branch `neurips2026-revision` before any code change and push to origin after each phase. Never commit `*.pt`, `runs/`, `checkpoints/`, or `*.tar.gz` files; extend `.gitignore` if needed.

## S3 archival via rclone

**Required environment variables (set before invoking):**

```bash
export TINYVLM_INSTANCE_ID="<your-vast-ai-instance-id>"
export TINYVLM_RCLONE_DEST="s3:<bucket>/<path>"   # e.g. "s3:my-lab/tinyvlm/neurips2026"
```

**Three reusable shell helpers** (install in `~/.bashrc` locally and on the instance):

```bash
s3push() {
    local src="$1" suffix="$2"
    rclone copy --progress --transfers 8 --checkers 16 \
        "$src" "${TINYVLM_RCLONE_DEST}/${suffix}" \
        && echo "PUSHED: ${TINYVLM_RCLONE_DEST}/${suffix}"
}

s3verify() {
    rclone size "${TINYVLM_RCLONE_DEST}/$1" 2>/dev/null \
        | grep -qE 'Total size: [1-9]'
}

s3pull() {
    local suffix="$1" dest="$2"
    mkdir -p "$dest"
    rclone copy --progress "${TINYVLM_RCLONE_DEST}/${suffix}" "$dest"
}
```

**One-time instance setup** (do this once per new vast.ai instance):

```bash
# Copy rclone config to instance
vastai scp ~/.config/rclone/rclone.conf ${TINYVLM_INSTANCE_ID}:~/.config/rclone/rclone.conf
vastai ssh ${TINYVLM_INSTANCE_ID} -- "chmod 600 ~/.config/rclone/rclone.conf"

# Install helpers and env var on instance
vastai ssh ${TINYVLM_INSTANCE_ID} -- "cat >> ~/.bashrc << 'BASHRC_END'
export TINYVLM_RCLONE_DEST='${TINYVLM_RCLONE_DEST}'
s3push() { local src=\"\$1\" suffix=\"\$2\"; rclone copy --progress --transfers 8 --checkers 16 \"\$src\" \"\${TINYVLM_RCLONE_DEST}/\${suffix}\"; }
s3verify() { rclone size \"\${TINYVLM_RCLONE_DEST}/\$1\" 2>/dev/null | grep -qE 'Total size: [1-9]'; }
s3pull() { mkdir -p \"\$2\"; rclone copy --progress \"\${TINYVLM_RCLONE_DEST}/\$1\" \"\$2\"; }
BASHRC_END"

# Install Python dependencies on instance
vastai ssh ${TINYVLM_INSTANCE_ID} -- "pip install torch torchvision tqdm pillow pycocotools \
    pycocoevalcap nltk tensorboard open_clip_torch scipy matplotlib"

# Download COCO 2017 on instance (if not present)
vastai ssh ${TINYVLM_INSTANCE_ID} -- "
    mkdir -p /workspace/coco/annotations /workspace/coco/train2017
    [ -f /workspace/coco/annotations/captions_train2017.json ] || (
        cd /workspace/coco && \
        wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
        unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip
    )
    [ \"\$(ls /workspace/coco/train2017 | wc -l)\" -gt 100 ] || (
        cd /workspace/coco && \
        wget -q http://images.cocodataset.org/zips/train2017.zip && \
        unzip -q train2017.zip && rm train2017.zip
    )
"
```

**Pre-flight check at the start of every session:**

```bash
# Verify env vars set
test -n "$TINYVLM_INSTANCE_ID" || { echo "TINYVLM_INSTANCE_ID unset"; exit 1; }
test -n "$TINYVLM_RCLONE_DEST"  || { echo "TINYVLM_RCLONE_DEST unset";  exit 1; }

# Verify local rclone
rclone lsd "${TINYVLM_RCLONE_DEST%/*}" > /dev/null || { echo "local rclone broken"; exit 1; }

# Verify remote rclone
vastai ssh ${TINYVLM_INSTANCE_ID} -- \
    "source ~/.bashrc && rclone lsd \"\${TINYVLM_RCLONE_DEST%/*}\"" \
    > /dev/null || { echo "remote rclone broken — run one-time setup"; exit 1; }

# Verify COCO present on instance
vastai ssh ${TINYVLM_INSTANCE_ID} -- \
    "[ -f /workspace/coco/annotations/captions_train2017.json ] \
     && [ \"\$(ls /workspace/coco/train2017 | wc -l)\" -gt 100 ]" \
    || { echo "COCO not ready on instance"; exit 1; }
```

**S3 layout:**

```
${TINYVLM_RCLONE_DEST}/
├── phase_1_hard_routing/<YYYYMMDD>/
│   ├── phase_1_results.tar.gz
│   └── extracted/
├── phase_2_figures/<YYYYMMDD>/…
├── phase_3a_vocabulary/<YYYYMMDD>/…
├── phase_3b_clip_backbone/<YYYYMMDD>/…
├── phase_4_paper/<YYYYMMDD>/…
├── checkpoints/                   # durable best.pt files
├── state/                         # rolling REVISION_STATE.md, FINDINGS.md
│   └── final/
└── logs/                          # consolidated training logs
```

**Push-verify-cleanup ordering (strict invariant):**

```
run_phase       → artifacts on remote instance
package_phase   → tar into /workspace/tinyvlm/artifacts/phase_N_results.tar.gz
s3push remote   → rclone copy from instance to S3
s3verify        → rclone size confirms non-zero
local download  → vastai scp to results/revision_2026/phase_N_<date>/
local verify    → tar -tzf on downloaded archive
remote cleanup  → ONLY after s3verify passes
mark COMPLETE   → in REVISION_STATE.md with s3_uri and local_path
```

**Rolling state backup after every phase:**

```bash
s3push REVISION_STATE.md state/REVISION_STATE.md
s3push FINDINGS.md       state/FINDINGS.md
```

## State tracking — `REVISION_STATE.md`

Maintain at project root with this schema:

```
---
instance_id: <TINYVLM_INSTANCE_ID>
gpu_hours_budget: 12
gpu_hours_spent: 0.0
total_uploads: 0
max_uploads: 10
disk_safety_margin_gb: 3
local_downloads_root: ./results/revision_2026/
started: <iso8601>
rclone_dest: <TINYVLM_RCLONE_DEST>
s3_verified_phases: []
---

# Phase status

- [ ] phase_1_hard_routing       PENDING  budget_gh=0.5  est_remote_gb=0.2  est_local_gb=0.1
- [ ] phase_2_figures            PENDING  budget_gh=3.0  est_remote_gb=2.0  est_local_gb=0.5
- [ ] phase_3a_vocabulary        PENDING  budget_gh=2.0  est_remote_gb=1.0  est_local_gb=0.3
- [ ] phase_3b_clip_backbone     PENDING  budget_gh=5.0  est_remote_gb=3.0  est_local_gb=0.5
- [ ] phase_4_paper              PENDING  budget_gh=0.5  est_remote_gb=0.2  est_local_gb=0.1

# Deferred / skipped

# S3 archive index
```

Update bullets after each phase:

```
- [x] phase_N_...  COMPLETE  gh_spent=X.X  s3_uri=<uri>  local=<path>  finished=<iso8601>
- [~] phase_N_...  DEFERRED_DISK  reason="..."
```

## Disk Guard (run before every phase)

```bash
REMOTE_AVAIL_GB=$(vastai ssh ${TINYVLM_INSTANCE_ID} -- \
    "df -BG / | awk 'NR==2 {gsub(\"G\",\"\",\$4); print \$4}'")
LOCAL_AVAIL_GB=$(df -BG . | awk 'NR==2 {gsub("G","",$4); print $4}')
# Read est_remote_gb and est_local_gb from REVISION_STATE.md for current phase
if [ "$REMOTE_AVAIL_GB" -lt $((EST_REMOTE_GB + DISK_SAFETY_MARGIN)) ]; then
    vastai ssh ${TINYVLM_INSTANCE_ID} -- \
        "find /workspace/tinyvlm/runs -name 'epoch_*.pt' -delete; \
         find /workspace/tinyvlm -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
    REMOTE_AVAIL_GB=$(...)  # recheck
    [ "$REMOTE_AVAIL_GB" -lt $((EST_REMOTE_GB + DISK_SAFETY_MARGIN)) ] && \
        { log_defer DEFERRED_DISK; continue; }
fi
[ "$LOCAL_AVAIL_GB" -lt $((EST_LOCAL_GB + 2)) ] && \
    { log_defer DEFERRED_DISK; continue; }
```

## Upload workflow (per phase that changes code)

```bash
tar --exclude='*.pt' --exclude='runs/' --exclude='__pycache__' \
    --exclude='.git' --exclude='*.tar.gz' \
    -czf tinyvlm_src_$(date +%Y%m%d_%H%M).tar.gz \
    tinyvlm_vast.py tinyvlm2.2.tex Figure/ IMPLEMENTATION-PLAN.md

vastai scp tinyvlm_src_*.tar.gz ${TINYVLM_INSTANCE_ID}:/workspace/tinyvlm/
vastai ssh ${TINYVLM_INSTANCE_ID} -- \
    "cd /workspace/tinyvlm && tar -xzf tinyvlm_src_*.tar.gz && rm tinyvlm_src_*.tar.gz"
```

Increment `total_uploads` in `REVISION_STATE.md` each time. Do not re-upload unless code changed.

## Remote execution workflow

Launch each training job inside a named `tmux` session:

```bash
vastai ssh ${TINYVLM_INSTANCE_ID} -- \
    "tmux new -d -s phase${N}_run \
     'cd /workspace/tinyvlm && python tinyvlm_vast.py \
        --coco_imgs /workspace/coco/train2017 \
        --coco_anns /workspace/coco/annotations/captions_train2017.json \
        <phase-specific flags> \
        2>&1 | tee logs/phase${N}_run.log'"
```

Poll for completion every `max(5 min, expected_duration / 20)`:

```bash
vastai ssh ${TINYVLM_INSTANCE_ID} -- \
    "tail -40 /workspace/tinyvlm/logs/phase${N}_run.log"
```

Tee a summary into `FINDINGS.md` under a new dated subsection at the end of every experiment.

## Feedback loop

If a phase's success gate is missed, modify code and re-upload up to **3 times per phase**. Total re-uploads across the full run must not exceed **10** (tracked in `REVISION_STATE.md::total_uploads`). If a phase still fails after 3 iterations, mark `DEFERRED_BUDGET` and move on.

Total GPU budget: **12 GPU-hours**. If `remaining < phase_budget` before a phase, mark it `DEFERRED_BUDGET`.

---

## Execute phases in strict order

Before executing any phase: run Disk Guard, run Budget Guard, verify preconditions.

---

### Phase 1 — Hard-routing inference fix (critical; blocks all FLOPs claims)

**Preconditions.** None.
**Remote disk estimate.** +0.2 GB (smoke-test outputs only).
**GPU budget.** 0.5 h.

**Problem.** `AdaptiveNeuralCompression.forward` runs all three encoders on every input regardless of routing weights (soft weighted sum). Actual FLOPs = 2+8+20 = 30 GFLOPs. Paper claims ∼10.3 GFLOPs, which is only valid under hard routing.

**Work.**

1. Edit `AdaptiveNeuralCompression.forward` in `tinyvlm_vast.py` to split on `self.training`:
   - **Training path (unchanged):** all three encoders run; soft Gumbel weighted sum; `comp_cost` is soft expected FLOPs. No gradient issues.
   - **Eval path (new):** group samples by `weights.argmax(dim=-1)`, run only the selected encoder per group. Use vectorised batching by branch index to avoid a per-sample Python loop. Example:
     ```python
     branch_idx = weights.argmax(dim=-1)                      # [B]
     encoded    = torch.zeros(B, self.cfg.hidden_dim, device=rgb.device)
     comp_cost  = torch.zeros(B, device=rgb.device)
     for k, enc, proj in zip(range(len(self.encoders)), self.encoders, self.projections):
         mask = (branch_idx == k)
         if not mask.any():
             continue
         feat = enc(rgb[mask], events[mask])                  # only selected samples
         encoded[mask] = proj(feat)
         comp_cost[mask] = enc.flops
     ```
   - Return signature unchanged: `(token_logits, comp_cost, router_logits, weights)`.

2. Update `_export_onnx` to add a comment that ONNX export uses eval-mode hard routing.

3. Run smoke test to verify:
   ```bash
   python tinyvlm_vast.py --smoke_test --epochs 2
   ```
   Check that `comp_cost.mean()` at eval time is approximately `Σ_k f_k · FLOPs(E_k)` (not 30 GFLOPs).

4. Run with `--eval_cider_freq 1 --epochs 2 --smoke_test` and verify `_validate` still works end-to-end.

**Success gate.**
- `comp_cost.mean()` at eval ≈ hard-routing average FLOPs (verify from `UtilizationTracker` fractions).
- Smoke test completes without error.
- `comp_cost` during training is still soft expected FLOPs (training path unchanged).
- Git diff confirms only `AdaptiveNeuralCompression.forward` and `_export_onnx` changed.

**Artifacts manifest:**
- `phase_1_results.tar.gz` containing: git diff `tinyvlm_vast.py.diff`, smoke-test log, `metrics.jsonl` from the 2-epoch smoke test confirming `comp_cost` values.

**Remote cleanup.** Delete smoke-test run directory after download.

---

### Phase 2 — Replace figures from verified canonical run data

**Status: COMPLETE — no GPU run required.**

**Preconditions.** None (data already available locally).
**Remote disk estimate.** 0 GB.
**GPU budget.** 0 h.

**Finding.** The canonical 3-seed COCO run is intact in `results-tinyvlm_prev/tinyvlm_cider_seeds.nosync/`. The paper's claimed numbers were cross-verified against the actual `metrics.jsonl`:

| Paper claim | Actual value | Status |
|---|---|---|
| CIDEr 36.70 ± 0.37 | 36.703 ± 0.370 | ✅ exact |
| Val acc 45.89 % ± 0.05 % | 45.888 % ± 0.050 % | ✅ exact |
| Val acc > train acc (epoch 9) | 0.4591 > 0.4429 | ✅ confirmed |
| Val loss not diverging | 4.55 → 4.20, monotone | ✅ confirmed |
| All 5 τ-sweep CIDEr values | match to 2 d.p. | ✅ exact |

The vocabulary was correctly used in all seeds (vocabulary.json present). The old `Figure/*.PNG` files were from old unregularised dev runs.

**Work completed.**
The four replacement figures were generated directly from seed 42's `metrics.jsonl` using the `tinyvlm` conda environment:
- `Figure/f_STTF_Acc.PNG` — accuracy 0–9 epochs, val > train ✅
- `Figure/f_STTF_Loss.PNG` — loss 0–9 epochs, val < train, no divergence ✅
- `Figure/ANC_Acc.PNG` — same data, consistent y-axis ✅
- `Figure/ANC_Loss.PNG` — same data, consistent y-axis ✅

All four use identical axis limits (accuracy: 0.36–0.47; loss: 4.15–5.00) and x-axis ends at epoch 9.

**Artifacts manifest.**
- Updated `Figure/*.PNG` (4 files) — already written to local repo.
- Source data: `results-tinyvlm_prev/tinyvlm_cider_seeds.nosync/seed_{42,43,44}_tau_0.80/` (metrics.jsonl, summary.json, vocabulary.json).

**Remote cleanup.** N/A.

---

### Phase 3A — Vocabulary verification

**Status: COMPLETE — vocabulary confirmed in use by canonical run.**

The `vocabulary.json` file is present in every seed directory of the `tinyvlm_cider_seeds` run, confirming the invertible `Vocabulary` class was used for the published CIDEr=36.70 result. No code change or re-run is needed.

The code guard (asserting `vocab is not None` for COCO path) is a good defensive addition but not blocking. It can be added as a minor hardening step in Phase 4 (paper update) alongside the paper edits, requiring no GPU time.

---

### Phase 3B — CLIP ViT-B/32 backbone integration (required for NeurIPS-competitive CIDEr)

**Preconditions.** Phase 3A COMPLETE (vocabulary must be correct before CLIP CIDEr is meaningful).
**Remote disk estimate.** +3 GB (CLIP model weights ~600 MB + training run).
**GPU budget.** 5 h.

**Problem.** CIDEr 36.70 is far below competitive COCO captioning results (competitive range: 80–140+ CIDEr). Root cause: from-scratch CNN encoder with no pretrained visual features. Fix: use frozen CLIP ViT-B/32 as the visual feature extractor; keep ANC routing over projection heads; keep the transformer decoder.

**Work.**

1. Install `open_clip_torch` on instance:
   ```bash
   vastai ssh ${TINYVLM_INSTANCE_ID} -- "pip install open_clip_torch"
   ```

2. Add to `Config`:
   ```python
   clip_backbone: bool = False
   clip_model: str = "ViT-B-32"
   clip_pretrained: str = "openai"
   ```

3. Add `CLIPStemEncoder` class:
   ```python
   class CLIPStemEncoder(nn.Module):
       """Frozen CLIP ViT-B/32 visual stem; outputs [B, 512] CLS token."""
       def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
           super().__init__()
           import open_clip
           model, _, self.preprocess = open_clip.create_model_and_transforms(
               model_name, pretrained=pretrained
           )
           self.visual = model.visual
           self.visual.requires_grad_(False)
           self.out_dim = 512  # ViT-B/32 CLS dim

       @torch.no_grad()
       def forward(self, rgb: torch.Tensor) -> torch.Tensor:
           return self.visual(rgb)  # [B, 512]
   ```

4. Modify `AdaptiveNeuralCompression.__init__` to optionally insert `CLIPStemEncoder` as a pre-encoder:
   - If `cfg.clip_backbone`: construct `self.clip_stem = CLIPStemEncoder(...)`. The three ANC branches become linear projectors from 512-dim CLIP features to their respective target dims (128, 256, 384), followed by a 2-layer MLP each. This preserves the "heterogeneous capacity" routing property while using pretrained features.
   - The `complexity_estimator` takes the 512-dim CLIP CLS token instead of event-camera features (since COCO has no events, events are zero-padded anyway).

5. Add `--clip_backbone` flag to `build_parser` and `config_from_args`.

6. Update `CocoCaptionDataset.__getitem__` to pass raw PIL images for CLIP preprocessing when `clip_backbone=True`. The dataset must apply the CLIP `preprocess` transform instead of the standard ImageNet normalization.

7. Train with CLIP backbone:
   ```bash
   vastai ssh ${TINYVLM_INSTANCE_ID} -- "tmux new -d -s phase3b \
     'cd /workspace/tinyvlm && python tinyvlm_vast.py \
       --coco_imgs /workspace/coco/train2017 \
       --coco_anns /workspace/coco/annotations/captions_train2017.json \
       --epochs 15 --batch_size 32 --seed 42 \
       --clip_backbone \
       --lr 3e-4 --weight_decay 0.01 \
       --eval_cider_freq 2 \
       --output_dir /workspace/tinyvlm/runs/phase3b_clip_seed42 \
       2>&1 | tee /workspace/tinyvlm/logs/phase3b_clip.log'"
   ```

8. If CIDEr > 80 after seed 42, run seeds 43 and 44 for 3-seed reporting.

9. Record results in `FINDINGS.md`. If CIDEr does not reach 80 after 15 epochs, try:
   - Reducing batch size and increasing epochs to 25
   - Unfreezing the top 4 layers of CLIP visual encoder with LR 1e-5
   - This counts as one re-upload iteration; maximum 3 per phase.

**Success gate.**
- Seed 42 CIDEr ≥ 80 after ≤ 20 epochs (verified from `metrics.jsonl` caption records).
- Routing balance maintained (all three branches used; `UtilizationTracker` fractions each ≥ 15 %).
- CIDEr reported as mean ± std over 3 seeds.
- Hard-routing FLOPs at eval ≈ `Σ_k f_k · FLOPs_k` (from Phase 1 fix, still in effect).

**Artifacts manifest:**
- `phase_3b_results.tar.gz` containing: code diff, `metrics.jsonl` for all seeds, `summary.json`, 20 sample decoded captions, figure exports (train/val curves), training log.
- **S3 checkpoint backup:** push `runs/phase3b_clip_seed42/checkpoints/best.pt` to `checkpoints/phase_3b_clip_best/` via `s3push`. This is the primary deliverable checkpoint.

**Remote cleanup.** Delete non-best-seed CLIP run directories.

---

### Phase 4 — Paper update (`tinyvlm2.2.tex`)

**Preconditions.** Phases 1, 2, 3A COMPLETE at minimum; Phase 3B strongly preferred.
**Remote disk estimate.** +0.2 GB (paper build on local machine; no remote needed).
**GPU budget.** 0.5 h (CPU/text work).

**Work.**

All edits are to `tinyvlm2.2.tex` in the local repository.

1. **Section 3.2 (ANC routing):** Replace "In practice, we use top-k routing to activate only a subset of branches, reducing computation" with: "During training, all branches process each input via soft Gumbel-Softmax routing to enable end-to-end differentiation. At inference, we switch to hard argmax routing — only the branch selected by the router executes — so the realised per-sample FLOPs equal the branch's actual cost rather than an expectation over all branches."

2. **Table 3 (main results — FLOPs column):** The formula `0.31×2 + 0.34×8 + 0.35×20 ≈ 10.3 GFLOPs` is now correct *only after the Phase 1 fix*. Verify the routing fractions from Phase 2 `metrics.jsonl` and update the fractions in the caption if they differ. If Phase 3B CLIP results are available, add a second STTF+ANC row for the CLIP-backbone variant.

3. **Section 4.1 (Training Details):** If Phase 3B is complete, add a paragraph: "For the CLIP-backbone variant, we replace the from-scratch CNN encoder with a frozen CLIP ViT-B/32 visual stem (pretrained on LAION-400M via OpenCLIP). The ANC routing branches become lightweight projection heads from the 512-dim CLIP CLS token. Only the projection heads and decoder are trained; CLIP weights are held fixed throughout. This variant is denoted STTF+ANC+CLIP in Tables 3 and 5."

4. **Figures 2–5:** Replace `\includegraphics` paths to point to the new `Figure/*.PNG` files from Phase 2. Update captions to state x-axis range (epochs 0–9) and remove any claim that contradicts the actual figure content.

5. **Abstract and Section 5 (Results):** Update CIDEr and val-accuracy numbers to match Phase 2 actuals (from `FINDINGS.md`). If Phase 3B CIDEr is available, add it as the headline result; demote the scratch-trained result to an ablation.

6. **Section 5 (Results text):** Remove the claim "Validation accuracy consistently *exceeds* training accuracy" if this is only true for the regularised run and not visible in the old figures. Rewrite to say: "With dropout ($p=0.2$) and label smoothing ($\epsilon=0.1$) applied only during training, the regularised 10-epoch run shows validation accuracy marginally exceeding training accuracy at epoch 9 ($X\%$ vs $Y\%$), with train/val gap below Z percentage points throughout."

7. **Build check:**
   ```bash
   pdflatex tinyvlm2.2.tex && bibtex tinyvlm2.2 && \
   pdflatex tinyvlm2.2.tex && pdflatex tinyvlm2.2.tex
   ```
   Must complete with no errors. Warnings are acceptable; undefined references are not.

**Success gate.**
- `tinyvlm2.2.tex` compiles to PDF without errors.
- FLOPs formula and values are consistent between Section 3.2, Table 3, and the Phase 1 code fix.
- All four figures have x-axis ending at epoch 9; captions match the figure content.
- Every numerical claim in the abstract, introduction, and results is cross-referenceable to a line in `FINDINGS.md` from Phases 2–3.
- No figure or number contradicts the text description of it.

**Artifacts manifest:**
- `phase_4_results.tar.gz` containing: `tinyvlm2.2.tex`, `tinyvlm2.2.pdf`, `Figure/` directory with all 4 updated PNGs.
- **S3 final snapshot:** push `tinyvlm2.2.pdf` and `tinyvlm2.2.tex` to `${TINYVLM_RCLONE_DEST}/state/final/`.

**Remote cleanup.** Nothing (all work is local).

---

## Final actions (after Phase 4 success gate or abort)

1. Commit all changes to branch `neurips2026-revision`; push to origin. One clean commit per phase.
2. Append `## Full-cycle summary` to `FINDINGS.md`: phases COMPLETE, phases DEFERRED (with reasons), total `gpu_hours_spent`, total uploads, actual CIDEr results, and S3 archive index.
3. **Final S3 state sync:**
   ```bash
   s3push REVISION_STATE.md state/REVISION_STATE.md
   s3push FINDINGS.md       state/FINDINGS.md
   ```
4. Execute `vastai stop instance ${TINYVLM_INSTANCE_ID}` — do NOT destroy.
5. Print terminal summary: all local artifact paths and S3 archive index.

## Deliverables

**Local machine, end of cycle:**
- `results/revision_2026/phase_{1..4}_<date>/` — per-phase artifact archives, extracted.
- `tinyvlm2.2.tex` at project root, compiling cleanly to PDF.
- `Figure/f_STTF_Acc.PNG`, `Figure/f_STTF_Loss.PNG`, `Figure/ANC_Acc.PNG`, `Figure/ANC_Loss.PNG` — regenerated from 10-epoch COCO run.
- `FINDINGS.md` with per-phase numerical results.
- `REVISION_STATE.md` in final form.
- Branch `neurips2026-revision` pushed to origin.

**S3 (durable):**
- Per-phase archives at `${TINYVLM_RCLONE_DEST}/phase_{1..4}_.../<date>/`.
- Best checkpoint at `${TINYVLM_RCLONE_DEST}/checkpoints/phase_{2,3b}_best/`.
- Final snapshot at `${TINYVLM_RCLONE_DEST}/state/final/`.

## Operating rules

- **Phase 1 is a hard prerequisite for all efficiency numbers.** Do not report FLOPs from any run unless Phase 1 code is in effect on the instance.
- Read-only inputs (never edit): `IMPLEMENTATION-PLAN.md` primary sections, the original `tinyvlm2.2.tex` (keep a backup before any edit as `tinyvlm2.1.tex`).
- Run Disk Guard before every phase. Run Budget Guard before every phase.
- **Push-verify-download-clean ordering is strict.** Never clean remote before S3 verify passes.
- If the S3 push fails, do not clean the remote. Mark the phase `IN_PROGRESS`, log the error in `FINDINGS.md`, and exit for user diagnosis.
- Phases 1 and 2 are non-negotiable for submission readiness. If either is DEFERRED, explicitly flag in `FINDINGS.md` that the submission is not NeurIPS-ready.
- If re-running Phase 2 produces CIDEr or accuracy numbers that materially differ from the paper's claims, record the actual numbers. Do not modify `metrics.jsonl` or figure data to match the paper. Update the paper to match reality.
- Do not commit `*.pt` checkpoint files, `runs/` directories, or raw COCO images to git.
