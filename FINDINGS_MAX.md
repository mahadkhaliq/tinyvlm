# FINDINGS_MAX.md — Mahbub lane (TinyVLM final push)

## 2026-05-01 — Kickoff

### Pre-flight discoveries
- Mahad already finished CLIP backbone (per `REVISION_STATE.md` phase_3b_clip_backbone): CIDEr=89.72, BLEU4=27.77, ValAcc=52.47% after 50 epochs on Hellbender A100. Paper `tinyvlm2.3.tex` reports CIDEr 85.58 / BLEU4 27.15 (older 15-epoch number). Need Mahad to push latest 50-epoch numbers via S3 OR confirm which to use.
- Mahad's hard-routing fix landed (phase_1_hard_routing). `_validate` reports `encoder_gflops` ≈ 17.6 single-branch at eval (per REVISION_STATE.md note).
- Vast instance 35151914 running; 4× RTX 5090, PyTorch 2.11.0+cu130, rclone configured.
- COCO 2017 NOT on vast — kicked off background download to `/workspace/data/coco`.
- TokenLearner baseline checkpoint already at `/workspace/runs/tokenlearner_baseline` (217 MB) — likely seed 42 only. Phase 5 needs seeds 43 + 44 added.

### Pivot — Issue 3 effectively closed
TokenLearner CIDEr 45.09 vs Mahad's CLIP CIDEr 85.58 (or 89.72) = paper now leads not trails. Phase 3 Fix-A multi-token bridge becomes an **optional Pareto extension** rather than gap-closure. Phase 3 Fix-D (combined model) deferred — only retain as insurance if Phase 5 retrains reveal regression.

### Phase 2 reframe — DONE
Edits to `tinyvlm/tinyvlm2.3.tex`:
- L115 contributions item 1: scoped 84% claim to "event-based inputs (DVS128 Gesture, N-Caltech101)"; explicitly noted COCO single-frame limitation; added MSR-VTT/EgoSchema in-progress mention pointing to supplementary.
- L486 limitations: expanded to explain temporal-cache pathway requires temporally adjacent frames; MSR-VTT + EgoSchema named as in-progress; Kinetics/SSv2 retained as future work.

### Open questions for Mahad (sync needed)
- Confirm canonical CLIP CIDEr to cite in Table 3: 85.58 (15 epoch) or 89.72 (50 epoch)?
- Is Mahad doing 3-seed CLIP retrain, or single-seed? Phase 5 paired stats need seeds 42/43/44 matched across all rows.
- Will Mahad push his updated `sec_method_clip.tex` / `sec_results_clip.tex` partials to S3?

## 2026-05-01T11:50Z — Phase 5 (matched-seed baselines + paired stats) COMPLETE

### Results
TokenLearner with current code/protocol (batch 8, λ_balance 0.01, λ_flops 0.1):
- seed 42 (retrained): CIDEr 50.85, BLEU4 17.52, ValAcc 47.10
- seed 43: CIDEr 56.27, BLEU4 18.77, ValAcc 47.22
- seed 44: CIDEr 50.04, BLEU4 16.70, ValAcc 47.13
- mean ± std: CIDEr **52.39 ± 3.39**, BLEU4 17.66 ± 1.04, ValAcc 47.15 ± 0.06

Original paper TokenLearner = 45.09 single-seed, batch 16, λ=0. Different protocol — superseded.

### Paired t-test (STTF+ANC vs TokenLearner, seeds 42/43/44)
| Metric | STTF+ANC | TokenLearner | Δ | p | Cohen's d | 95% CI |
|---|---|---|---|---|---|---|
| CIDEr | 36.70 ± 0.37 | 52.39 ± 3.39 | -15.69 | 0.018 | -4.27 | [-19.85, -12.92] |
| BLEU4 | 14.20 ± 0.14 | 17.66 ± 1.04 | -3.46 | 0.037 | -2.94 | [-4.69, -2.35] |
| ValAcc | 0.459 | 0.471 | -0.013 | 0.003 | -11.18 | [-0.014, -0.012] |

CIDEr gap WIDENED from paper's 8.4 → 15.7 with corrected baseline. Story now:
- Scratch CNN POC trails TokenLearner by 15.7 CIDEr (real, paired-significant cost of routing under under-parameterised encoder)
- CLIP backbone (Mahad: 89.72) reverses gap by +33.2 CIDEr above TokenLearner

### Paper updates
- Table 3 row TokenLearner: 45.09 → 52.39 ± 3.39 (3 seeds), single ★ significance
- Table 3 caption: matched-seed framing, paired-t footnote
- Table 3 trade-off line: -8.4 → -15.7 CIDEr; new line "+33.2 vs CLIP backbone"
- §5 stats prose: rewrite from one-sample t-test to paired t-test
- §6.1 conclusion: paired stats; CLIP backbone framed as gap-reversal

### Open
- Mahad's CLIP CIDEr 89.72 (per his REVISION_STATE.md phase_3b) not yet reflected in committed paper (still 85.58 from earlier 15-ep run). His push pending.
- GPT-2 decoder commit 2705f5f exists but no numbers committed yet.

## 2026-05-02T05:08Z — Phase CLIP (matched 3-seed CLIP+STTF+ANC) COMPLETE

### Results
CLIP+STTF+ANC, 3 seeds, 15 epochs, 4×5090 DDP, batch 4 per rank:
- seed 42: CIDEr 82.35, BLEU4 26.10, ValAcc 50.80
- seed 43: CIDEr 83.65, BLEU4 25.95, ValAcc 50.79
- seed 44: CIDEr 82.07, BLEU4 26.30, ValAcc 50.70
- mean ± std: CIDEr **82.69 ± 0.84**, BLEU4 26.12 ± 0.18, ValAcc 50.76% ± 0.06%

Tight std (0.84) — indicates well-converged, low-variance training. Compare:
- TokenLearner: CIDEr std 3.39 (4× wider)
- CNN-POC STTF+ANC: CIDEr std 0.37

### Paired t-test (CLIP+STTF+ANC vs TokenLearner, matched 3 seeds)
| Metric | CLIP+STTF+ANC | TokenLearner | Δ | p | Cohen's d | 95% CI |
|---|---|---|---|---|---|---|
| CIDEr | 82.69 ± 0.84 | 52.39 ± 3.39 | **+30.30** | 0.002 | **+11.90** | [+27.4, +32.0] |
| BLEU4 | 26.12 ± 0.18 | 17.66 ± 1.04 | +8.46 | 0.007 | +6.97 | [+7.2, +9.6] |
| ValAcc | 0.508 | 0.471 | +0.036 | 0.0001 | +47.44 | [+0.036, +0.037] |

Massive effect sizes. Headline secured.

### Paper updates (tinyvlm2.4.tex)
- Abstract: 89.72 → 82.69 ± 0.84 (matched 3 seeds), 89.72 retained as 50-ep peak ref
- Table 3: bold row "STTF+ANC (Ours, CLIP, 3 seeds)" with mean ± std and paired-test markers
- §5 Headline subsection: full paired stats; "$+30.30$ above TokenLearner"
- §6.1 conclusion + §6.4 limitations: same reframe

### Cost
~12.5 GPU-h on vast 4×5090 ($31). Total Mahbub-lane spend so far: ~21.5 GPU-h ($54).
