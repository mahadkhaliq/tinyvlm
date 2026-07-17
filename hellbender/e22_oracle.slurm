#!/bin/bash
# E22 — oracle routing bound. Eval only, NO training.
# Submit:  sbatch hellbender/e22_oracle.slurm
#
# Bounds what ANY routing policy could achieve at these budgets: caption the whole
# Karpathy-test split once per branch, then per image take the branch that scored best.
#
#   oracle ~= 75   -> "no routing policy recovers the ~5 CIDEr" is airtight and
#                     policy-independent. C4 holds even against a router that cannot be built.
#   oracle ~= 79+  -> real headroom exists at these budgets and our claim is too strong.
#
# This is an UPPER BOUND and optimistic by construction (it picks the per-image winner using
# the test metric itself, which no deployable router can do). That optimism is the point:
# if even the oracle cannot clear the wall, nothing can. Do NOT report it as an achievable
# operating point.
#
#SBATCH --job-name=tinyvlm_e22
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=42,43,44          # one oracle per trained seed
#SBATCH --output=tinyvlm_e22_seed%a_%A.out
#SBATCH --error=tinyvlm_e22_seed%a_%A.err

set -uo pipefail
module load cuda/12.4 || module load cuda
source activate tinyvlm || conda activate tinyvlm

SEED=$SLURM_ARRAY_TASK_ID
COCO_DIR=${COCO_DIR:-$SCRATCH/coco}

# Point this at the trained token-budget runs you want bounded. Default assumes the
# pre-registered lambda=0.01 run; override RUN_ROOT to bound a different sweep point.
RUN_ROOT=${RUN_ROOT:-$SCRATCH/tinyvlm_runs/E_phase2}
RUN_DIR="$RUN_ROOT/seed_${SEED}_tau_0.80"

cd "$SLURM_SUBMIT_DIR/vast/phase2" || exit 1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [ ! -d "$RUN_DIR/checkpoints" ]; then
  echo "[E22] no checkpoints at $RUN_DIR -- set RUN_ROOT to a trained token-budget run dir." >&2
  exit 1
fi

echo "[E22] seed=$SEED  run_dir=$RUN_DIR"

python e22_oracle_eval.py \
    --run_dir "$RUN_DIR" \
    --coco_imgs "$COCO_DIR/train2017" \
    --coco_anns "$COCO_DIR/annotations/captions_train2017.json" \
    --token_select stride \
    --out "$SLURM_SUBMIT_DIR/tnnls_results/E22_oracle/seed_${SEED}.json"

echo "[E22] seed=$SEED done rc=$?"
