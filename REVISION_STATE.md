---
instance_id: UNSET
gpu_hours_budget: 12
gpu_hours_spent: 0.0
total_uploads: 0
max_uploads: 10
disk_safety_margin_gb: 3
local_downloads_root: ./results/revision_2026/
started: 2026-04-26
rclone_dest: UNSET
s3_verified_phases: []
---

# Phase status

- [x] phase_1_hard_routing       COMPLETE  gh_spent=0.0  note="hard argmax routing at eval implemented; _validate now reports encoder_gflops + val_branch_*_hard_fraction; smoke test confirmed 17.60 GFLOPs single-branch at eval"
- [x] phase_2_figures            COMPLETE  gh_spent=0.0  note="figures regenerated from verified metrics.jsonl (results-tinyvlm_prev); no re-run needed"
- [x] phase_3a_vocabulary        COMPLETE  gh_spent=0.0  note="vocabulary.json confirmed present in canonical cider_seeds run; no code change needed"
- [x] phase_3b_clip_backbone     COMPLETE  gh_spent=4.5  note="CIDEr=85.58, BLEU4=27.15, ValAcc=51.52% after 15 epochs (batch=64, lr=3e-4, RTX 3070); hard routing confirmed 17.60 GFLOPs; results in /tmp/tinyvlm_clip_seed42/"
- [x] phase_4_paper              COMPLETE  gh_spent=0.0  note="all edits done: routing clarification, CLIP section, Table 3 filled (CIDEr=85.58, BLEU4=27.15, Acc=51.52%), abstract updated, limitations updated"

# Training command for phase_3b (run once COCO images extracted)

```bash
unzip -q /home/qubit/data/coco/train2017.zip -d /home/qubit/data/coco/
/home/qubit/anaconda3/envs/tinyvlm/bin/python3 /home/qubit/Downloads/tinyvlm/tinyvlm_vast.py \
    --coco_imgs /home/qubit/data/coco/train2017 \
    --coco_anns /home/qubit/data/coco/annotations/captions_train2017.json \
    --epochs 15 --batch_size 4 \
    --clip_backbone \
    --lr 3e-4 \
    --target_budget 18e9 \
    --eval_cider_freq 2 \
    --output_dir /tmp/tinyvlm_clip_seed42 \
    --no_resume
```

# Deferred / skipped

(none)

# S3 archive index

(phases 1–3a completed locally; no remote artifacts to archive)
