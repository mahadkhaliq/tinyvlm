#!/bin/bash
# hellbender_eval.slurm — S1: eval-only pass on an existing checkpoint
# Computes CIDEr-D, BLEU-4, METEOR on the full Karpathy val split.
#
# Usage:
#   CHECKPOINT_DIR=$SCRATCH/tinyvlm_runs/clip_50ep sbatch hellbender_eval.slurm
#   CHECKPOINT_DIR=$SCRATCH/tinyvlm_runs/clip_seed42 sbatch hellbender_eval.slurm
#
# Results written to: $CHECKPOINT_DIR/seed_42_tau_0.80/eval_only_results.json
#SBATCH --job-name=tinyvlm_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

for ver in cuda/12.4 cuda/12.2 cuda/12.1 cuda/12.0 cuda/11.8 cuda; do
    module load $ver 2>/dev/null && break
done
source activate tinyvlm 2>/dev/null || conda activate tinyvlm

COCO_DIR=${COCO_DIR:-$SCRATCH/coco}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$SCRATCH/tinyvlm_runs/clip_50ep}

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "ERROR: CHECKPOINT_DIR not found: $CHECKPOINT_DIR"
    exit 1
fi

echo "=== TinyVLM eval-only: $CHECKPOINT_DIR ==="
echo "COCO: $COCO_DIR"
echo "Time: $(date)"

python tinyvlm_vast.py \
    --coco_imgs   "$COCO_DIR/train2017" \
    --coco_anns   "$COCO_DIR/annotations/captions_train2017.json" \
    --clip_backbone \
    --lr          3e-4 \
    --target_budget 18e9 \
    --epochs      1 \
    --batch_size  64 \
    --num_workers 8 \
    --eval_cider_freq 1 \
    --eval_only \
    --output_dir  "$CHECKPOINT_DIR" \
    --no_onnx \
    --no_tensorboard

echo "=== Done: $(date) ==="
echo "Results: $CHECKPOINT_DIR/seed_42_tau_0.80/eval_only_results.json"
