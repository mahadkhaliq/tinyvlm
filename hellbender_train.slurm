#!/bin/bash
#SBATCH --job-name=tinyvlm_clip
#SBATCH --partition=gpu          # adjust to Hellbender's GPU partition name
#SBATCH --gres=gpu:1             # 1 GPU; change to gpu:4 + torchrun for multi-GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load cuda/12.4
source activate tinyvlm

COCO_DIR=${COCO_DIR:-$SCRATCH/coco}

python tinyvlm_vast.py \
    --coco_imgs "$COCO_DIR/train2017" \
    --coco_anns "$COCO_DIR/annotations/captions_train2017.json" \
    --clip_backbone \
    --lr 3e-4 \
    --target_budget 18e9 \
    --epochs 15 \
    --batch_size 64 \
    --num_workers 8 \
    --eval_cider_freq 2 \
    --output_dir "$SCRATCH/tinyvlm_runs/clip_seed42"
