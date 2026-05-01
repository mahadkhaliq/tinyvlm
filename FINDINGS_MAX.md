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
