#!/bin/bash
#SBATCH --job-name=e1_sanity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=e1_sanity_%j.out
#SBATCH --error=e1_sanity_%j.err

# E1 Week-1 input-sanity check for DVS128 Gesture.
#
# Stage 1 (always): verify event volumes have non-zero per-sample std.
# Stage 2 (optional): verify routing histogram is not 100%-one-branch.
#
# Before submitting:
#   1. Transfer the CNN checkpoint from your local machine:
#      scp tinyvlm_cider_seeds.nosync/seed_42_tau_0.80/checkpoints/best.pt \
#          <user>@hellbender.rnet.missouri.edu:$SCRATCH/tinyvlm_ckpts/cnn_seed42.pt
#   2. Copy this repo to Hellbender (or git clone/pull):
#      git clone <repo-url> ~/tinyvlm  (or git pull if already present)
#   3. Submit: sbatch hellbender_e1_sanity.slurm
#
# DVS128 data (~1.3 GB) is downloaded automatically by tonic on first run.

set -e

module load cuda/12.4 || module load cuda
source activate tinyvlm 2>/dev/null || conda activate tinyvlm

# paths
REPO_DIR=${REPO_DIR:-$HOME/tinyvlm}
DATA_DIR=${DATA_DIR:-$SCRATCH/dvs128}
CKPT=${CKPT:-$SCRATCH/tinyvlm_ckpts/cnn_seed42.pt}
OUT_DIR=${OUT_DIR:-$SCRATCH/tnnls_results/E1_sanity}

cd "$REPO_DIR"

# install tonic if not present
python -c "import tonic" 2>/dev/null || pip install tonic -q

mkdir -p "$DATA_DIR" "$OUT_DIR"

echo "=== E1 sanity check starting ==="
echo "  data_dir  : $DATA_DIR"
echo "  checkpoint: $CKPT"
echo "  output_dir: $OUT_DIR"
echo "  date      : $(date)"

# Stage 1 + 2 if checkpoint exists; Stage 1 only otherwise
if [ -f "$CKPT" ]; then
    python "$REPO_DIR/e1_sanity_check.py" \
        --data_dir "$DATA_DIR" \
        --checkpoint "$CKPT" \
        --n_samples 200 \
        --output_dir "$OUT_DIR"
else
    echo "WARNING: checkpoint not found at $CKPT — running Stage 1 (data check) only"
    python "$REPO_DIR/e1_sanity_check.py" \
        --data_dir "$DATA_DIR" \
        --n_samples 200 \
        --output_dir "$OUT_DIR"
fi

echo ""
echo "=== Results saved to $OUT_DIR ==="
ls -lh "$OUT_DIR/"

# print summary inline so it appears in the .out log
python -c "
import json, pathlib
s = json.loads(pathlib.Path('$OUT_DIR/summary.json').read_text())
print('GO/NO-GO :', s['go_nogo'])
print('Stage 1  :', s['stage1']['verdict'],
      ' mean_std=', round(s['stage1']['per_sample_std_mean'],4))
if s['stage2']:
    print('Stage 2  :', s['stage2']['verdict'],
          ' fracs=', s['stage2']['branch_fractions'])
"
