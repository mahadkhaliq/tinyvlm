---
instance_id: 37721982
rclone_dest: s3research:vastai-research/tinyvlm/tnnls
gpu_hours_budget: 100
gpu_hours_spent: 3.20
total_uploads: 1
max_uploads: 10
disk_safety_margin_gb: 10
local_downloads_root: ./tnnls_results/
local_root: /Users/mahbub/Documents/projects/tinyvlm/tinyvlm
git_branch: tnnls/plan-code-support
started: 2026-05-25T00:00:00Z
schema_version: 1
---

# Phase status

- [x] phase_0_setup                  COMPLETE  finished=2026-05-25T07:35:00Z  notes="instance ready: torch 2.12.dev cu129 Blackwell, COCO 19+1+0.8GB, 4 ckpts uploaded (sttf_anc 3 seeds + tokenlearner)"
- [x] phase_E18_wallclock_cache      COMPLETE  gh_spent=0.01  s3_uri=phase_E18_wallclock_cache/20260525/  finished=2026-05-25T07:30:00Z  notes="peak=389.6MB, 3.51ms/frame RTX 5090"
- [x] phase_E19_full_metrics         COMPLETE  gh_spent=0.12  s3_uri=phase_E19_full_metrics/20260525/   finished=2026-05-25T08:03:00Z  notes="STTF+ANC 3-seed CIDEr 36.02±1.08, BLEU-4 13.71±0.40; TokenLearner CIDEr 5.10 (1 seed); METEOR+SPICE deferred"
- [x] phase_E8_routing_adaptivity    COMPLETE  gh_spent=0.02  s3_uri=phase_E8_routing_adaptivity/20260525/  finished=2026-05-25T08:05:00Z  notes="ROUTING COLLAPSE: 100% Medium at inference; complexity_estimator constant (zero-event input); verdict=routing_collapse_at_inference"
- [x] phase_E9_soft_vs_hard          COMPLETE  gh_spent=0.05  s3_uri=phase_E9_soft_vs_hard/20260525/    finished=2026-05-25T08:10:00Z  notes="CNN seed_42: ΔCIDEr(soft-hard)=-2.55; hard collapses to Medium (best encoder), soft averages all 3; CLIP arm deferred"
- [x] phase_E7_dense_smallenc        COMPLETE  gh_spent=3.0  s3_uri=phase_E7_dense_smallenc/20260525/  finished=2026-05-25T11:06:47Z  notes="CIDEr 42.10±8.24 (3 seeds 42/43/44 = 40.0/45.9/40.4); +6.08 over STTF+ANC, Cohen d=2.57, p=0.08 (Welch); REVIEWER-KILLER for ANC value claim"
- [~] phase_E15_lambda_pareto        IN_PROGRESS  budget_gh=5  notes="5 settings × 10ep seed_42 STTF+ANC; lambda_flops sweep; launched 11:10; ETA ~15:55"

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
