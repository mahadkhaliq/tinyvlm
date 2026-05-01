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

- [x] phase_2_reframe                COMPLETE     gh_spent=0    s3_uri=tinyvlm-finalpush/max/phase_2_6_paper/20260501/  note="paper §1 + §3 caption + §6.4 scoped 83% FLOPs claim to existing tab:ncaltech; MSR-VTT/EgoSchema added as in-progress supplementary"
- [x] phase_3_cider_gap_fixA         DEFERRED_PRECONDITION  reason="Mahad's CLIP backbone (CIDEr 85.58 in paper, 89.72 latest from Hellbender) + GPT-2 decoder addition makes multi-token bridge obsolete; ours now leads TokenLearner by 40+ CIDEr"
- [x] phase_3_cider_gap_fixD         DEFERRED_PRECONDITION  reason="See phase_3_cider_gap_fixA"
- [ ] phase_5_baselines              IN_PROGRESS  budget_gh=8   started=2026-05-01T07:21Z  note="TokenLearner seed 43+44 launched in tmux 'baselines' on 4×5090 DDP; ETA 80min/seed = ~10:00Z complete. Existing TokenLearner seed 42 at /workspace/runs/tokenlearner_baseline (CIDEr 45.09)"
- [x] phase_6_token_reduction        COMPLETE     gh_spent=0    s3_uri=tinyvlm-finalpush/max/phase_2_6_paper/20260501/  note="abstract had no 84% claim; replaced 84% in §1 + §3 caption + §6.4 with verifiable 83% from tab:ncaltech; deferred new Table N because no temporal cache logging in COCO single-frame _validate"
- [ ] phase_paper_max                PENDING      budget_gh=0   est_remote_gb=0  est_local_gb=0  note="awaits Mahad partials (sec_method_clip.tex, sec_results_clip.tex, sec_routing.tex, appendix_ablation.tex) via S3"

# Deferred / skipped

- 2026-05-01: phase_3_cider_gap_fixA DEFERRED_PRECONDITION — Mahad's CLIP backbone (CIDEr 85.58 paper, 89.72 latest) + GPT-2 decoder commit 2705f5f makes Fix-A obsolete. Ours now leads, not trails.
- 2026-05-01: phase_3_cider_gap_fixD DEFERRED_PRECONDITION — Same as Fix-A.

# S3 archive index

- 2026-05-01T07:23Z phase_2_6_paper.tar.gz → s3test:vastai-research/ws-34754072/tinyvlm-finalpush/max/phase_2_6_paper/20260501/  (size 22 KiB; contains tinyvlm2.3.tex, REVISION_STATE_MAX.md, FINDINGS_MAX.md, paired_stats.py)


# Environment notes

- Vast SSH: `ssh -p 38027 root@209.50.14.22`
- Vast image: `vastai/pytorch:cuda-13.0.2-auto`, PyTorch 2.11.0+cu130, 4× RTX 5090 (32 GB each)
- Vast disk: 27 GB free on /workspace (99 GB total). Spikehippo using 9.3 GB — candidate for eviction if disk tight.
- Existing vast runs in /workspace/runs/: tinyvlm_cider_seeds (732M), tinyvlm_seed43 (244M), tinyvlm_seed44 (244M), tinyvlm_tau (1.2G), tokenlearner_baseline (217M).
- COCO 2017 NOT present at /workspace/data/coco. Background download started 2026-05-01 ~early — track via `/workspace/data/coco/download.log`.
- Bucket: `s3test:vastai-research/ws-34754072/tinyvlm-finalpush`. rclone v1.73.4 installed.
- Git: `tinyvlm/` subdir on branch `neurips2026-finalpush-max`, remote `origin = https://github.com/mahadkhaliq/tinyvlm.git`.
