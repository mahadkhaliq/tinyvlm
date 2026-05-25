---
instance_id: 37721982
rclone_dest: s3research:vastai-research/tinyvlm/tnnls
gpu_hours_budget: 100
gpu_hours_spent: 0.0
total_uploads: 0
max_uploads: 10
disk_safety_margin_gb: 10
local_downloads_root: ./tnnls_results/
local_root: /Users/mahbub/Documents/projects/tinyvlm/tinyvlm
git_branch: tnnls/plan-code-support
started: 2026-05-25T00:00:00Z
schema_version: 1
---

# Phase status

- [ ] phase_0_setup                  PENDING   budget_gh=0    est_remote_gb=35   est_local_gb=0
- [ ] phase_E18_wallclock_cache      PENDING   budget_gh=0.5  est_remote_gb=0.1  est_local_gb=0.05
- [ ] phase_E19_full_metrics         PENDING   budget_gh=5    est_remote_gb=0.5  est_local_gb=0.3
- [ ] phase_E8_routing_adaptivity    PENDING   budget_gh=3    est_remote_gb=0.3  est_local_gb=0.2
- [ ] phase_E9_soft_vs_hard          PENDING   budget_gh=2    est_remote_gb=0.1  est_local_gb=0.05
- [ ] phase_E7_dense_smallenc        PENDING   budget_gh=30   est_remote_gb=12   est_local_gb=2
- [ ] phase_E15_lambda_pareto        PENDING   budget_gh=50   est_remote_gb=20   est_local_gb=3

# Deferred / skipped

(append DEFERRED_DISK, DEFERRED_BUDGET, DEFERRED_PRECONDITION, DEFERRED_GATE_FAIL entries here with timestamps and reasons)

# S3 archive index

(append one s3_uri line per completed phase — restores full cycle from S3 alone)

# Code prerequisites verified (2026-05-25, commit 6ebd6a1)

- [x] `tinyvlm_vast.py::Config.encoder_only` (default `"medium"`) — line 188
- [x] `tinyvlm_vast.py::Config.eval_routing_mode` (default `"hard"`) — line 193
- [x] `--encoder_only {tiny,small,medium}` argparse — line 2264
- [x] `--eval_routing_mode {hard,soft}` argparse — line 2271
- [x] `--baseline {none,tokenlearner,dense}` — line 2261
- [x] `DenseEncoderBaseline` class — line 827 (`_ENCODER_FACTORY` dispatch)
- [x] Hard-routing branch grouping in `AdaptiveNeuralCompression.forward` — lines 791–812
- [x] `--eval_only`, `--clip_backbone`, `--lambda_flops`, `--seeds` — all present
- [x] No further code patches required before phase execution.

# Instance snapshot (2026-05-25, vastai show)

- ID 37721982 — running
- GPU: 1× RTX 5090 (32 GB VRAM)
- vCPUs 16.0, RAM 61.9 GB
- Disk: 100 GB `/workspace` (data), 16 GB `/` overlay (system, never put data)
- Image: `vastai/pytorch:2.10.0-cu130-cuda-13.1-mini-py312-2026-03-19` (torch NOT preinstalled)
- Driver 590.48.01, CUDA 13.1, Python 3.12.3
- SSH: resolve via `vastai ssh-url 37721982`
- Rate: $0.80/hr (~$65 total at 80 GPU-hr execution)
