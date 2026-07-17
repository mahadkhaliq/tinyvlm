#!/bin/bash

# E20 — single-branch budget sweep, router REMOVED.
# Separates token-dropping cost from the fixed cost of the routed 3-branch config.
# 3 budgets x 3 seeds = 9 runs as a job array. Submit:  sbatch hellbender/e20_single_branch.slurm
#
# Read (pre-register this BEFORE looking):
#   keep_k=49 (drops nothing, no router) ~= 79.8  -> architecture sound; then keep_k=36 vs the routed
#                                                    lambda=0.01 run (74.93, also ~36 tokens) isolates
#                                                    token-dropping from routing.
#   keep_k=49 ~= 75                      -> the routing apparatus itself costs ~5 CIDEr, paid before a
#                                           single token is dropped. A MORE novel finding.
# You win under both outcomes.
#
#SBATCH --job-name=tinyvlm_e20
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-8               # 9 combos: index = budget_idx*3 + seed_idx
#SBATCH --output=tinyvlm_e20_%a_%A.out
#SBATCH --error=tinyvlm_e20_%a_%A.err

set -uo pipefail
module load cuda/12.4 || module load cuda
source activate tinyvlm || conda activate tinyvlm

BUDGETS=(24 36 49)
SEEDS=(42 43 44)
I=$SLURM_ARRAY_TASK_ID
K=${BUDGETS[$((I / 3))]}
SEED=${SEEDS[$((I % 3))]}

COCO_DIR=${COCO_DIR:-$SCRATCH/coco}
OUT=$SCRATCH/tinyvlm_runs/E20_single_branch/keep_${K}

cd "$SLURM_SUBMIT_DIR/vast/phase2" || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[E20] keep_k=$K seed=$SEED  node=$(hostname)"

# Router bypassed: one branch, one head, straight through. lambda_balance/entropy are set to 0
# because both terms are inert with a single expert (balance = 1.0 exactly, entropy = 0).
python tinyvlm_vast.py \
    --coco_imgs "$COCO_DIR/train2017" \
    --coco_anns "$COCO_DIR/annotations/captions_train2017.json" \
    --clip_backbone \
    --clip_token_budget \
    --single_branch_keep_k "$K" \
    --token_select stride \
    --lambda_balance 0 \
    --lambda_entropy 0 \
    --no_onnx \
    --lr 3e-4 \
    --target_budget 18e9 \
    --epochs 15 \
    --batch_size 64 \
    --num_workers 8 \
    --seeds "$SEED" \
    --eval_routing_mode hard \
    --output_dir "$OUT"

echo "[E20] keep_k=$K seed=$SEED done rc=$?"
