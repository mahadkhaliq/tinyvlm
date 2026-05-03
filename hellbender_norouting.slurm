#!/bin/bash
# hellbender_norouting.slurm — S5: CLIP + no ANC routing (iso-backbone ablation)
# Trains frozen CLIP ViT-B/32 + single 384-d MLP head + same decoder, no routing.
# This is the key ablation to isolate what ANC contributes on top of CLIP.
#
# Usage (single seed):
#   sbatch hellbender_norouting.slurm
#
# Usage (3 seeds in parallel — submit 3 times with different SEED):
#   SEED=42 sbatch hellbender_norouting.slurm
#   SEED=43 sbatch hellbender_norouting.slurm
#   SEED=44 sbatch hellbender_norouting.slurm
#
# Expected runtime: ~90 min on A100 (same as CLIP-3seed 15ep runs)
# Expected CIDEr: compare to 82.69 ± 0.84 (CLIP+ANC) to isolate routing gain
#SBATCH --job-name=tinyvlm_norouting
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

for ver in cuda/12.4 cuda/12.2 cuda/12.1 cuda/12.0 cuda/11.8 cuda; do
    module load $ver 2>/dev/null && break
done
source activate tinyvlm 2>/dev/null || conda activate tinyvlm

COCO_DIR=${COCO_DIR:-$SCRATCH/coco}
SEED=${SEED:-42}
OUT=$SCRATCH/tinyvlm_runs/clip_norouting_seed${SEED}

echo "=== TinyVLM CLIP no-routing ablation: seed=${SEED} ==="
echo "Output: $OUT"
echo "Time: $(date)"

python tinyvlm_vast.py \
    --coco_imgs   "$COCO_DIR/train2017" \
    --coco_anns   "$COCO_DIR/annotations/captions_train2017.json" \
    --clip_backbone \
    --no_anc \
    --seed        $SEED \
    --lr          3e-4 \
    --target_budget 18e9 \
    --epochs      15 \
    --batch_size  64 \
    --num_workers 8 \
    --eval_cider_freq 2 \
    --output_dir  "$OUT" \
    --no_resume \
    --no_onnx \
    --no_tensorboard

echo "=== Done: seed=${SEED} $(date) ==="
echo "CIDEr result in: $OUT/seed_${SEED}_tau_0.80/metrics.jsonl"
