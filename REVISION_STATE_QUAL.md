---
ai_hub_quota_estimate_jobs: 12
ai_hub_jobs_spent: 11
total_uploads: 8
note_token_rotation: rotated mid-cycle. Both should be revoked at aihub.qualcomm.com Settings -> API Tokens after submission.
max_uploads: 8
disk_safety_margin_gb: 2
local_results_root: ./qai_hub_results/
rclone_dest: SKIPPED_DEADLINE
s3_verified_phases: []
started: 2026-05-06T00:00:00Z
note: rclone unavailable on local Mac; S3 archival skipped. Local-only artefacts due to deadline.
---

# Phase status

- [x] phase_q1_onnx_export        COMPLETE  jobs_used=0  artefact=models/onnx/*.onnx (5 files, 265 MB)
- [x] phase_q2_local_validate     COMPLETE  jobs_used=0  all 5 ONNX run finite on onnxruntime CPU
- [x] phase_q3_snapdragon_888     COMPLETE  jobs_used=4  device="Samsung Galaxy S21 (Family)"  ids=jgle66mep, j56qee4vg, jp3qvv0x5, jpe4eem75
- [x] phase_q4_x_elite            COMPLETE  jobs_used=4  device="Snapdragon X Elite CRD"  ids=jgd7eerlg, jgzvoodzp, jg99jj9lg, j5wm226zg  (full 3-branch sweep + dense)
- [x] phase_q5_aggregate          COMPLETE  jobs_used=0  qai_hub_results/table5_measured.json  SD888_STTF+ANC=24.31ms_routing_weighted
- [x] phase_q6_paper_patch        COMPLETE  jobs_used=0  tinyvlm_4.2.{tex,pdf}  20p, refs=p10, main<=9p

# Deferred / skipped
- phase_s23_8gen2: SKIPPED_QUOTA
- phase_s24_8gen3: SKIPPED_QUOTA
- phase_clip_xelite: SKIPPED_QUOTA (CLIP profiling done on SD 888 only; X Elite CLIP deferred)
- jetson_nano: NOT_AVAILABLE on AI Hub (NVIDIA hardware); paper row dropped

# Phase clip
- [x] phase_clip_anc_export       COMPLETE  models/onnx/clip_anc_b{0,1,2}.onnx  (~389 MB each, fastpath off for MHA export)
- [x] phase_clip_anc_sd888        COMPLETE  jobs_used=3  ids=j5mvqq0d5, jgnrllzk5, jpr188l0g  weighted=98.00ms; iso b2=99.38ms; ratio=1.014x

# Devices verified available (qai-hub list-devices smoke)
- Samsung Galaxy S21 (Family) — qualcomm-snapdragon-888, sm8350
- Snapdragon X Elite CRD — qualcomm-snapdragon-x-elite, sc8380xp
- Snapdragon X2 Elite CRD — qualcomm-snapdragon-x2-elite, sc8480xp (bonus, unused)

# S3 archive index
(skipped — local-only)
