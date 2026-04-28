#!/bin/bash
# hellbender_setup.sh — one-time environment + data setup on Mizzou Hellbender
# Run once per session from the repo root: bash hellbender_setup.sh

set -e

# ── 1. Load modules (adjust versions to what Hellbender has) ─────────────────
module load cuda/12.4 || module load cuda
module load python/3.11 || module load python

# ── 2. Create / activate conda env ───────────────────────────────────────────
ENV_NAME=tinyvlm
if ! conda env list | grep -q "^${ENV_NAME}"; then
    conda create -y -n $ENV_NAME python=3.11
fi
source activate $ENV_NAME || conda activate $ENV_NAME

# ── 3. Install PyTorch (pick the right CUDA wheel for Hellbender's GPU) ──────
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 -q

# ── 4. Install remaining requirements ────────────────────────────────────────
pip install -r requirements.txt -q

# ── 5. Download COCO 2017 (images + annotations) ─────────────────────────────
COCO_DIR=${COCO_DIR:-$SCRATCH/coco}
mkdir -p "$COCO_DIR"

if [ ! -f "$COCO_DIR/train2017.zip" ]; then
    echo "Downloading COCO train2017 images (~18 GB)..."
    wget -q --show-progress -O "$COCO_DIR/train2017.zip" \
        http://images.cocodataset.org/zips/train2017.zip
fi

if [ ! -d "$COCO_DIR/train2017" ] || [ -z "$(ls -A $COCO_DIR/train2017)" ]; then
    echo "Extracting train2017..."
    unzip -q "$COCO_DIR/train2017.zip" -d "$COCO_DIR"
fi

if [ ! -f "$COCO_DIR/annotations/captions_train2017.json" ]; then
    echo "Downloading COCO annotations (~241 MB)..."
    wget -q --show-progress -O "$COCO_DIR/annotations_trainval2017.zip" \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q "$COCO_DIR/annotations_trainval2017.zip" -d "$COCO_DIR"
fi

echo ""
echo "Setup complete."
echo "  Images : $COCO_DIR/train2017/  ($(ls $COCO_DIR/train2017 | wc -l) files)"
echo "  Anns   : $COCO_DIR/annotations/captions_train2017.json"
echo ""
echo "Run training with:"
echo "  python tinyvlm_vast.py \\"
echo "    --coco_imgs $COCO_DIR/train2017 \\"
echo "    --coco_anns $COCO_DIR/annotations/captions_train2017.json \\"
echo "    --clip_backbone --lr 3e-4 --target_budget 18e9 \\"
echo "    --epochs 15 --batch_size 64 --num_workers 8 \\"
echo "    --eval_cider_freq 2 --output_dir \$SCRATCH/tinyvlm_runs"
