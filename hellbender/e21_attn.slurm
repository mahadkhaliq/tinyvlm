#!/bin/bash
# E21 — attention-ranked token selection (the falsification test for C4).
# 3 seeds as a job array. Submit from the repo root:  sbatch hellbender/e21_attn.slurm
#
# Compare against, from the paper (all in CORRECTED FLOPs units):
#   dense control        79.80 +/- 1.51  @ 4.29 G
#   routed, stride rule  74.93 +/- 0.25  @ 3.17 G   (lambda_balance=0.01)
# If attn also lands ~75  -> C4 is structural, the paper's claim hardens.
# If attn lands ~78-80    -> our diagnosis is wrong and we need to know.
#
#SBATCH --job-name=tinyvlm_e21
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=42,43,44          # SLURM_ARRAY_TASK_ID *is* the seed
#SBATCH --output=tinyvlm_e21_seed%a_%A.out
#SBATCH --error=tinyvlm_e21_seed%a_%A.err

set -uo pipefail
module load cuda/12.4 || module load cuda
source activate tinyvlm || conda activate tinyvlm

SEED=$SLURM_ARRAY_TASK_ID
COCO_DIR=${COCO_DIR:-$SCRATCH/coco}
OUT=$SCRATCH/tinyvlm_runs/E21_attn

cd "$SLURM_SUBMIT_DIR/vast/phase2" || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[E21] seed=$SEED  node=$(hostname)  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "[E21] token_select=attn  budgets=24,36,49  (attn costs 2.31/3.26/4.29 G -- NOT stride's 2.13/3.17/4.29)"

# --no_onnx is REQUIRED for attn: the top-k gather is data-dependent, so export
# cannot succeed. It is caught and non-fatal, but the traceback looks like a crash.
# Auto-resumes from checkpoints/latest.pt if this job was preempted or requeued.
python tinyvlm_vast.py \
    --coco_imgs "$COCO_DIR/train2017" \
    --coco_anns "$COCO_DIR/annotations/captions_train2017.json" \
    --clip_backbone \
    --clip_token_budget \
    --token_select attn \
    --token_budgets 24,36,49 \
    --lambda_balance 0.01 \
    --lr 3e-4 \
    --target_budget 18e9 \
    --epochs 15 \
    --batch_size 64 \
    --num_workers 8 \
    --seeds "$SEED" \
    --eval_routing_mode hard \
    --no_onnx \
    --output_dir "$OUT"

echo "[E21] seed=$SEED done rc=$?"
echo "[E21] results: $OUT  -> aggregate with vast/phase2/phase2_aggregate.py once all 3 seeds land"
