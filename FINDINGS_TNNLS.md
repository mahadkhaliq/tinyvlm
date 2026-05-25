# FINDINGS_TNNLS.md — Per-phase outcomes for TNNLS Lead-side automation

Tracks results, headline numbers, and decision rationale for each phase executed via `TNNLS_AUTOMATION.md` on vast instance 37721982.

---

## 2026-05-25 — Cycle initialized

- `TNNLS_AUTOMATION.md` authored in nested tree (INSTRUCTION.md-style scaffolding).
- `TNNLS_AUTO_STATE.md` initialized; 7 phases pending (Phase 0 setup + E18, E19, E8, E9, E7, E15).
- S3 verified: `rclone v1.74.0` reachable, prefix `s3research:vastai-research/tinyvlm/tnnls/` writable.
- Instance 37721982 probed: RTX 5090 32 GB, `/workspace` 100 GB, mini image (no torch preinstalled).
- Code prerequisites verified — all required flags landed in nested `tinyvlm_vast.py` commit `6ebd6a1` (May 24): `--encoder_only`, `--eval_routing_mode`, `--baseline dense`, `DenseEncoderBaseline` class, hard-routing branch grouping. No further patches needed.

## Per-phase outcomes

### Phase 0 — Instance setup — IN PROGRESS (2026-05-25)

| Step | Status | Notes |
|---|---|---|
| 0.1 rclone + helpers on instance | ✓ DONE | `s3push`/`s3verify`/`s3pull` in `/root/.bashrc`, S3 prefix reachable from instance |
| 0.2 deps install | ✓ DONE | torch 2.12.0.dev20260401+cu129 (Blackwell-ready, cc=12.0 verified via matmul); transformers 5.9.0, nltk 3.9.4 + punkt, onnx 1.21.0, pycocoevalcap, pycocotools, open_clip_torch, bert_score, scipy, matplotlib, pyyaml, openjdk 21 |
| 0.3 COCO 2017 download | ⏳ IN PROGRESS | annotations (796 MB) + val2017 (788 MB) done; train2017.zip ~17 GB / 18 GB downloaded; unzip pending |
| 0.4 upload code | ✓ DONE | nested `tinyvlm_vast.py` (95 KB, 2372 lines) + CLAUDE.md + TNNLS_AUTOMATION.md on `/workspace/` |
| 0.5 upload reference ckpt | ✓ DONE | `tinyvlm_full.nosync/seed_42_tau_0.80/checkpoints/best.pt` (244 MB) → `/workspace/ckpts/sttf_anc_cnn_full/best.pt`, md5 verified |
| 0.6 import + ckpt smoke | ✓ DONE | Config defaults (encoder_only="medium", eval_routing_mode="hard"), DenseEncoderBaseline+ANC present, ckpt loads 82 tensors, 0 missing/unexpected on AdaptiveNeuralCompression |

Also uploaded supplementary: `tokenlearner_baseline.nosync/seed_42_tau_0.80/checkpoints/best.pt` (217 MB) → `/workspace/ckpts/tokenlearner/best.pt`. Md5 not verified.

**Note:** No `STTF+ANC CLIP` ckpt found locally; **E9 CLIP arm and E19 CLIP row will defer** until ckpt produced. No standalone `Dense Medium` ckpt — using `--encoder_only medium` override on STTF+ANC ckpt as proxy (imperfect, flag in NOTES).

### Phase E18 — Wall-clock + cache memory — COMPLETE (2026-05-25)

**Synthetic 100-frame forward-pass benchmark** on STTF+ANC CNN ckpt, RTX 5090, hard routing:

```json
{
  "peak_mem_MB": 389.63,
  "frames": 100,
  "tau": 0.80,
  "batch": 1,
  "H": 224, "W": 224,
  "wall_sec": 0.351,
  "ms_per_frame": 3.514,
  "device": "NVIDIA GeForce RTX 5090",
  "torch_version": "2.12.0.dev20260401+cu129",
  "eval_routing_mode": "hard"
}
```

**GPU-hours per training run** (from local `metrics.jsonl`, scraped):

| Config | Epochs | World size | Wall hr | **GPU-hr** | Min/epoch |
|---|---|---|---|---|---|
| STTF+ANC CNN | 10 | 2× | 1.63 | **3.27** | 9.80 |
| TokenLearner | 11 | 4× | 0.72 | **2.90** | 3.95 |
| Balanced (sanity) | 10 | 2× | 1.64 | **3.29** | 9.87 |

**ONNX export sizes** (`models/onnx/`, 1.45 GB total):

| Variant | MB |
|---|---|
| `cnn_sttf_anc_b0` (Tiny branch)   | 38.0 |
| `cnn_sttf_anc_b1` (Small branch)  | 45.0 |
| `cnn_sttf_anc_b2` (Medium branch) | 65.1 |
| `cnn_dense` | 65.1 |
| `cnn_tokenlearner` | 64.5 |
| `clip_anc_b0/b1/b2` | 389.1 / 389.5 / 390.0 |

**Paper §4.1 candidate text:**
> Training each STTF+ANC CNN run takes ≈3.3 GPU-hr on 2× RTX 5090 (10 epochs on MS-COCO Karpathy-train). Peak GPU memory during inference is ≈390 MB at τ=0.80 with batch=1, 224×224 on RTX 5090 over a 100-frame synthetic window. CNN ONNX export sizes are 38–65 MB per branch (1.45 GB total across CNN+CLIP variants).

**Closes:** M3 (wall-clock + cache memory in §4.1).

**Artifacts:** `/workspace/tnnls_results/E18/cache_mem.json`, `/workspace/logs/E18.log`.

### Phase E19 — Full caption-metric panel — COMPLETE for CNN (2026-05-25)

Bypassed broken `run_eval_only` path (`_maybe_resume` requires `latest.pt` + optimizer state; only `best.pt` available). Wrote `tnnls_eval.py` that loads ckpt directly via `torch.load + load_state_dict(strict=False)`. Greedy decode on COCO val (full 25K annotations → 5K unique images), pycocoevalcap BLEU-1..4 + ROUGE-L + CIDEr. METEOR + SPICE gated (JVM deadlock on first Meteor invocation).

**Results (eval_routing_mode=hard, val_fraction=1.0, RTX 5090):**

| Method | n_seeds | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |
|---|---|---|---|---|---|---|---|
| **STTF+ANC CNN** (mean ± 95% CI) | 3 | 51.22 ± 1.38 | 33.09 ± 0.95 | 20.94 ± 0.64 | **13.71 ± 0.40** | 35.82 ± 0.37 | **36.02 ± 1.08** |
| TokenLearner | 1 | 40.32 | 19.57 | 9.66 | 5.55 | 26.79 | 5.10 |

**Headline:** STTF+ANC CNN beats TokenLearner by **+30.9 CIDEr** and **+8.2 BLEU-4** at matched data + decoder. 3-seed CI very tight (CIDEr ±1.08 across seeds 42/43/44).

**Decode wall-clock**: 80–158 sec per ckpt on RTX 5090 (5K unique images).
**Metrics compute**: 2 sec per ckpt (Bleu+Rouge+Cider; Meteor+Spice skipped).

**Deferred:**
- METEOR + SPICE: pycocoevalcap Meteor wrapper deadlocks on dead JVM subprocess. Need timeout wrapper or separate eval. Tracked.
- Dense Medium row: no standalone ckpt; will use `--encoder_only medium` override on ANC ckpt as proxy or run fresh training.
- STTF+ANC CLIP row: no CLIP ckpt locally. Defer until CLIP ckpt produced.
- Dense Small row: produced by E7 (separate phase).

**Closes:** M5 partial (CIDEr + BLEU-4 + ROUGE-L confirmed; METEOR + SPICE in follow-up).

**Artifacts:** `tnnls_results/E19/{sttf_anc_cnn_seed{42,43,44}.json, tokenlearner_seed42.json, aggregate.json}`. S3: `s3://…/phase_E19_full_metrics/20260525/`.

### Phase E8 — Routing-adaptivity audit — COMPLETE (2026-05-25)

Hooked `complexity_estimator` over full COCO val (5000 unique images). `routing_pairs_cnn.npz` saved.

**Result:**
- Per-branch fraction at inference (hard argmax): **[0.00, 0.00, 1.00]** — all 5000 images routed to Medium.
- `complexity_estimator` output: **std=0.0 per branch** (constant logits regardless of input).
- Softmax mean (eval): [0.31, 0.34, 0.35] — matches training utilization but argmax always = Medium.
- Spearman ρ(g_ψ, branch) = NaN (no variance in branches array).

**Root cause** (per `CLAUDE.md`): `CocoCaptionDataset` simulates events as zero tensor. complexity_estimator input is constant → output constant → routing collapses at inference. Training-time soft routing produced apparent 31/34/35 utilization via Gumbel noise.

**Verdict:** `routing_collapse_at_inference`. **Paper action required:** replace "adaptive routing" claim with "load-balanced soft training + hard-argmax inference selects highest-capacity branch". On real event data (DVS128/N-Caltech101) re-audit may show different result.

**Closes:** C1 (with strong negative result; per Lead E8 decision rule, this maps to "report as negative result in §5").

**Artifacts:** `tnnls_results/E8/{decision_cnn.json, routing_pairs_cnn.npz, routing_scatter_cnn.pdf}`. S3: `s3://…/phase_E8_routing_adaptivity/20260525/`.

### Phase E9 — Soft vs hard routing gap — COMPLETE (CNN seed_42, 2026-05-25)

Two `tnnls_eval.py` runs on same STTF+ANC seed_42 ckpt, hard vs soft routing.

| Metric | hard | soft | Δ (soft − hard) |
|---|---|---|---|
| BLEU-1 | 51.32 | 50.32 | -1.00 |
| BLEU-2 | 33.15 | 32.07 | -1.08 |
| BLEU-3 | 20.95 | 20.11 | -0.84 |
| BLEU-4 | 13.72 | 13.16 | -0.57 |
| ROUGE-L | 35.80 | 35.10 | -0.70 |
| **CIDEr** | **36.16** | **33.61** | **-2.55** |

**Interpretation:** hard mode collapses to Medium-only (E8 finding); soft mode averages all 3 branches (weights 0.31/0.34/0.35). Hard > soft because Medium-alone outperforms weighted mix of Tiny+Small+Medium. ANC at inference recovers full Medium-encoder quality while running 1 branch (≈3× FLOPs reduction vs running all 3).

**Closes:** C2 (with reverse-direction finding: hard routing improves CIDEr vs soft training behavior).

**Deferred:** CLIP arm (no CLIP ckpt locally).

**Artifacts:** `tnnls_results/E9/{sttf_anc_cnn_seed42_soft.json, aggregate.json}`. S3: `s3://…/phase_E9_soft_vs_hard/20260525/`.

### Phase E7 — FLOPs-matched Dense SmallEncoder — COMPLETE (3 seeds, 2026-05-25)

`DenseEncoderBaseline` (single SmallEncoder, no routing) trained from scratch × 3 seeds on COCO train2017, 10 epochs each.

**Per-seed results:**

| Seed | val_loss | val_acc | CIDEr | BLEU-4 |
|---|---|---|---|---|
| 42 | 3.9203 | 45.94% | 40.03 | 14.10 |
| 43 | 3.9177 | 46.00% | **45.93** | 16.15 |
| 44 | 3.9204 | 45.96% | 40.35 | 14.55 |
| **mean ± 95%CI** | 3.92±0.004 | 45.97%±0.08 | **42.10 ± 8.24** | 14.93 ± 2.63 |

**Comparison vs STTF+ANC CNN** (3-seed mean = 36.02 ± 1.08):

| | Dense Small | STTF+ANC | Δ | Cohen's d | Welch t (df≈2) | p (two-sided) |
|---|---|---|---|---|---|---|
| CIDEr | 42.10 | 36.02 | **+6.08** | **2.57** | 3.15 | 0.084 |

**Headline:** Dense SmallEncoder (~8 GFLOPs, no routing) **outperforms STTF+ANC CNN by +6.08 CIDEr** at lower FLOPs. Effect size very large (d=2.57), p=0.08 (marginal, driven by seed_43 outlier 45.93 vs 40.0 for others).

**Paper implications (fatal C6 finding):**
- ANC's value proposition on COCO captioning is undermined: simpler dense baseline at smaller compute matches/beats it.
- ANC's defensible niche: real event-camera data (DVS128/N-Caltech101) where complexity_estimator can be input-adaptive — pending PhD-track E1/E2 results.
- On COCO with synthetic zero events, "adaptive compute" cannot be justified. Recommend repositioning ANC as "load-balanced soft-routing during training + capacity selector at inference" rather than input-adaptive computation.

**Closes:** C6.

**Artifacts:** `tnnls_results/E7/{seed_{42,43,44}/{summary,metrics,config}.json, aggregate.json}`. S3: `s3://…/phase_E7_dense_smallenc/20260525/` + `s3://…/checkpoints/phase_E7_dense_small/seed_{42,43,44}/best.pt`.

### Phase E15 — λ₂ Pareto frontier — COMPLETE (2026-05-25)

`lambda_flops ∈ {0.01, 0.05, 0.1, 0.5, 1.0}`, seed 42, 10 epochs each, STTF+ANC CNN. ~58 min per setting on RTX 5090.

| λ_flops | CIDEr | BLEU-4 | val_loss | encoder_gflops | dominant branch (eval) |
|---|---|---|---|---|---|
| 0.01 | **38.79** | 13.78 | 3.884 | **8.0** | **Small** (100%) |
| 0.05 | 27.64 | 10.83 | 3.956 | 2.0 | Tiny (100%) |
| 0.10 (default) | 26.72 | 10.28 | 3.957 | 2.0 | Tiny (100%) |
| 0.50 | 27.19 | 10.55 | 3.957 | 2.0 | Tiny (100%) |
| 1.00 | 27.11 | 10.35 | 3.954 | 2.0 | Tiny (100%) |

**Headline finding:** Router collapses to **single branch at eval** regardless of λ_flops. Branch identity is hyperparameter-driven, not input-driven:
- λ=0.01: collapses to **Small** branch (8 GFLOPs), best quality CIDEr=38.79.
- λ≥0.05: collapses to **Tiny** branch (2 GFLOPs), CIDEr clusters near ~27.
- No true Pareto in the middle: bimodal collapse, not smooth frontier.

**Triangulation with E7 + E8 + E9:**
- E8 (cider_seeds ckpt, λ_flops=0.01, λ_balance=0.5, 4-GPU DDP): collapse to **Medium** (20 GFLOPs).
- E15 (seed_42 ckpt, λ_flops=0.01, λ_balance=0.01 default, 1-GPU): collapse to **Small** (8 GFLOPs).
- E15 (seed_42, λ_flops=0.1, default cfg): collapse to **Tiny** (2 GFLOPs).
- Dense Small E7 (no router, no λ_balance): CIDEr=42.10 ± 8.24 at 8 GFLOPs — beats every λ in the sweep at any FLOPs setting.

**Paper implications (T4 closure):**
- The "FLOPs target = 5 G but achieves ~10 G" question is moot — the model can be tuned to any of (2, 8, 20) GFLOPs by changing (λ_flops, λ_balance) but always collapses to single branch.
- Default λ_flops=0.1 is dominated; λ=0.01 gives best CIDEr but still loses to Dense Small.
- The routing scheme functions as a hyperparameter-driven **branch selector**, not adaptive compute.

**Closes:** T4 (with negative result — no graded Pareto, bimodal collapse).

**Artifacts:** `tnnls_results/E15/aggregate.json + pareto.{pdf,png} + per-λ summary.json + metrics.jsonl`. S3: `s3://…/phase_E15_lambda_pareto/20260525/` + `s3://…/checkpoints/phase_E15_lambda_best/best.pt` (λ=0.01).

---

## Full-cycle summary (2026-05-25, complete)

### Phases executed

| Phase | Status | GPU-hr | Key finding |
|---|---|---|---|
| 0 setup | ✅ | 0 | torch 2.12 cu129 Blackwell + COCO 19 GB + 4 ckpts uploaded |
| E18 | ✅ | 0.01 | Peak 390 MB inference mem, 3.5 ms/frame RTX 5090 |
| E19 | ✅ | 0.12 | STTF+ANC 3-seed CIDEr **36.02 ± 1.08**; TokenLearner 5.10 |
| E8  | ✅ | 0.02 | **Routing collapse 100% Medium** at inference (zero-event artifact) |
| E9  | ✅ | 0.05 | ΔCIDEr(soft−hard) = **−2.55** (hard wins via Medium-alone) |
| E7  | ✅ | 2.90 | **Dense Small CIDEr 42.10 ± 8.24** > STTF+ANC by **+6.08** (d=2.57, p=0.084) |
| E15 | ✅ | 4.85 | λ-driven bimodal collapse: λ=0.01→Small(8G,38.8) vs λ≥0.05→Tiny(2G,27) |
| **TOTAL** | | **~7.95 GPU-hr** | |

### Cost & infra

- Instance 37721982: RTX 5090 (32 GB VRAM), 100 GB /workspace, $0.80/hr.
- Total: ~7.95 GPU-hr × $0.80 = **~$6.40** + ~$1 storage = ~$7.50.
- S3: all phase tars + plots + best checkpoints durably stored.

### S3 archive index

```
s3research:vastai-research/tinyvlm/tnnls/
├── phase_E18_wallclock_cache/20260525/
├── phase_E19_full_metrics/20260525/
├── phase_E8_routing_adaptivity/20260525/
├── phase_E9_soft_vs_hard/20260525/
├── phase_E7_dense_smallenc/20260525/
├── phase_E15_lambda_pareto/20260525/
├── checkpoints/phase_E7_dense_small/seed_{42,43,44}/best.pt
├── checkpoints/phase_E15_lambda_best/best.pt
└── state/{TNNLS_AUTO_STATE,FINDINGS_TNNLS,TNNLS_AUTOMATION}.md
```

### Critical findings for paper (TNNLS revision)

1. **Routing collapse is reproducible across hyperparameter choices** (E8 + E15). At inference, hard argmax always selects a single branch. Which branch is hyperparameter-driven (λ_flops, λ_balance), not input-driven (events are zero on COCO).
2. **Hard routing > soft routing on CIDEr** by 2.55 (E9). Soft averages all 3 branches; hard picks one. Recovers Medium-alone quality at ~3× FLOPs reduction vs running 3 branches.
3. **Dense Small (no routing) dominates STTF+ANC on COCO** by +6.08 CIDEr at lower FLOPs (E7). Effect very large (Cohen's d=2.57). C6 fully exposed.
4. **The "adaptive compute" framing cannot be sustained on COCO**. Defensible niche for ANC: real event-camera data (DVS128/N-Caltech101) — pending PhD-track E1/E2/E4 results.
5. **STTF+ANC vs TokenLearner gap is enormous** (+30.9 CIDEr at matched data + decoder). TokenLearner is genuinely worse at this task. This is the only headline-positive ANC story still standing on COCO.

### Recommended paper repositioning

- **Drop**: "adaptive routing per sample"; "FLOPs target = 5G" Pareto promise.
- **Keep**: STTF (token reuse for video); ANC as "load-balanced soft training + hard inference branch selector" framing; on-device edge measurements (AI Hub).
- **Add (honest reporting)**:
  - E8 negative result on COCO routing collapse → §5 "Routing analysis" subsection.
  - E7 Dense Small comparison row to Table 1 with the +6.08 finding noted.
  - E15 Pareto + λ collapse note in §3.5 or appendix.
- **Defer**: ANC vindication on real event data — needs PhD-track DVS128/N-Caltech101 retrains (E1, E2, E4 per PhD instructions).

### Deferred items (not blocking this cycle)

- METEOR + SPICE for E19 (pycocoevalcap Meteor JVM deadlock; needs timeout wrapper).
- CLIP variants of E8, E9, E19 (no local CLIP ckpt; needs separate training or recover from vast.ai earlier).
- Dense Medium row in E19 (use `--baseline dense --encoder_only medium` retrain).
- E15 multi-seed (currently single seed; needs 2 more seeds for ±CI).
- 3-seed Dense Small variance (seed_43 outlier 45.9 vs 40.0 — investigate or run more seeds).
