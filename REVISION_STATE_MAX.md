---
owner: mahbub
instance_id: 35151914
ssh_addr: ssh -p 38027 root@209.50.14.22
gpu_hours_budget: 100
gpu_hours_spent: 0.0
hard_stop_gpu_hours: 150
hard_stop_dollars: 300
total_uploads: 0
max_uploads: 12
disk_safety_margin_gb: 5
local_downloads_root: ./results/finalpush_max/
rclone_dest: s3test:vastai-research/ws-34754072/tinyvlm-finalpush
s3_verified_phases: []
started: 2026-05-01T08:00:00Z
target_eod: 2026-05-03T23:59:00Z
hard_deadline: 2026-05-06T18:00:00Z
---

# Phase status

- [x] phase_2_reframe                COMPLETE     gh_spent=0  note="paper §1 contributions + §6.4 limitations edited; STTF 84% scoped to event streams; MSR-VTT/EgoSchema added as in-progress supplementary"
- [ ] phase_3_cider_gap_fixA         PENDING      budget_gh=36  est_remote_gb=4  est_local_gb=1  note="re-evaluate scope: Mahad CLIP CIDEr=85.58 already closes TokenLearner gap (45.09); may downgrade to single Pareto-extension run"
- [ ] phase_3_cider_gap_fixD         DEFERRED_PRECONDITION  reason="Mahad CLIP exceeds 43.5 success-gate target; Fix-D unnecessary unless Phase 5 reveals gap regression"
- [ ] phase_5_baselines              PENDING      budget_gh=36  est_remote_gb=4  est_local_gb=1  note="awaiting COCO download bg; will retrain TokenLearner+dense × seeds 43,44 (seed 42 already in /workspace/runs/tokenlearner_baseline)"
- [ ] phase_6_token_reduction        PENDING      budget_gh=8   est_remote_gb=1  est_local_gb=0.3  note="eval-only re-runs on existing /workspace/runs/tinyvlm_tau"
- [ ] phase_paper_max                PENDING      budget_gh=0   est_remote_gb=0  est_local_gb=0

# Deferred / skipped

- 2026-05-01: phase_3_cider_gap_fixD DEFERRED_PRECONDITION — Mahad's CLIP backbone (CIDEr 85.58 in tinyvlm2.3.tex L91, 89.72 per his REVISION_STATE.md) already exceeds Fix-D combined target (≥47). Fix-D held as insurance only.

# S3 archive index

(append one line per completed phase with canonical URI)

# Environment notes

- Vast SSH: `ssh -p 38027 root@209.50.14.22`
- Vast image: `vastai/pytorch:cuda-13.0.2-auto`, PyTorch 2.11.0+cu130, 4× RTX 5090 (32 GB each)
- Vast disk: 27 GB free on /workspace (99 GB total). Spikehippo using 9.3 GB — candidate for eviction if disk tight.
- Existing vast runs in /workspace/runs/: tinyvlm_cider_seeds (732M), tinyvlm_seed43 (244M), tinyvlm_seed44 (244M), tinyvlm_tau (1.2G), tokenlearner_baseline (217M).
- COCO 2017 NOT present at /workspace/data/coco. Background download started 2026-05-01 ~early — track via `/workspace/data/coco/download.log`.
- Bucket: `s3test:vastai-research/ws-34754072/tinyvlm-finalpush`. rclone v1.73.4 installed.
- Git: `tinyvlm/` subdir on branch `neurips2026-finalpush-max`, remote `origin = https://github.com/mahadkhaliq/tinyvlm.git`.
