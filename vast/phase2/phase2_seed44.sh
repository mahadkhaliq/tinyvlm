#!/usr/bin/env bash
# phase2_seed44.sh — round out lam=0.5 to n=3 (seed 44 was lost to a watchdog).
# SELF-DESTRUCTS on completion (and via a 3.5h hard watchdog) using the instance's
# OWN scoped API key, so billing stops even if the control plane is idle. Env in:
#   SELF_ID  = this instance's contract id
#   SELF_KEY = this instance's scoped api key (can only manage this instance)
set -uo pipefail
WORK=/workspace; COCO=$WORK/coco; KJ=$COCO/dataset_coco.json
export RCLONE_CONFIG=$WORK/rclone.conf PYTHONPATH=$WORK DEBIAN_FRONTEND=noninteractive
exec > >(tee -a "$WORK/seed44.log") 2>&1
DEST=s3:tinyvlm-neurips2026/finalpush/phase2_lambda_sweep
status(){ printf '%s\n' "$(date -u) $2" | rclone rcat "$DEST/STATUS_$1" 2>/dev/null || true; }
selfdestruct(){ pip install -q vastai >/dev/null 2>&1 || true; vastai destroy instance "${SELF_ID}" --api-key "${SELF_KEY}" 2>/dev/null || vastai destroy instance "${SELF_ID}" 2>/dev/null || true; }
trap 'selfdestruct' EXIT
# hard 3.5h watchdog -> self-destruct regardless
setsid bash -c "sleep 12600; pip install -q vastai >/dev/null 2>&1; vastai destroy instance ${SELF_ID} --api-key ${SELF_KEY} 2>/dev/null || vastai destroy instance ${SELF_ID} 2>/dev/null" </dev/null >/dev/null 2>&1 &

echo "[s44] START $(date -u) self_id=${SELF_ID}"
apt-get install -y -qq unzip rclone >/dev/null 2>&1 || true
pip install -q pycocotools pillow tqdm tensorboard scipy matplotlib onnx open_clip_torch pycocoevalcap nltk 2>&1 | tail -1
python -m nltk.downloader punkt punkt_tab >/dev/null 2>&1 || true
command -v rclone >/dev/null 2>&1 || (curl -s https://rclone.org/install.sh | bash >/dev/null 2>&1)
command -v unzip  >/dev/null 2>&1 || { status FAILED_S44 "no unzip"; exit 1; }

python - <<'PY'
import torch,sys,importlib.util
assert torch.cuda.is_available()
s=importlib.util.spec_from_file_location("tv","/workspace/tinyvlm_vast.py");m=importlib.util.module_from_spec(s);sys.modules["tv"]=m;s.loader.exec_module(m)
assert hasattr(m.Config(),"clip_token_budget"); import clip_token_budget
print("[preflight] OK")
PY
[ $? -ne 0 ] && { status FAILED_S44 preflight; exit 1; }

# data
mkdir -p "$COCO"; cd "$COCO"; dl(){ curl -fsSL "$1" -o "$2"; }
[ -d train2017 ]   || { dl http://images.cocodataset.org/zips/train2017.zip t.zip && unzip -q t.zip && rm -f t.zip; }
[ -d val2017 ]     || { dl http://images.cocodataset.org/zips/val2017.zip v.zip && unzip -q v.zip && rm -f v.zip; }
[ -d annotations ] || { dl http://images.cocodataset.org/annotations/annotations_trainval2017.zip a.zip && unzip -q a.zip && rm -f a.zip; }
[ -f "$KJ" ]       || { dl https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip c.zip && unzip -q -o c.zip dataset_coco.json && rm -f c.zip; }
{ [ -d train2017 ] && [ -f annotations/captions_train2017.json ] && [ -f "$KJ" ]; } || { status FAILED_S44 "data missing"; exit 1; }
status S44_DATA_READY "staged"

# single run: lam=0.5 seed 44
cd "$WORK"; OUT="$WORK/runs/lam_0.5/seed_44"
echo "[s44] TRAIN $(date -u)"
python tinyvlm_vast.py \
  --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
  --karpathy_json "$KJ" --karpathy_eval_split val \
  --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
  --lr 3e-4 --target_budget 14e9 --lambda_balance 0.5 --epochs 15 --batch_size 64 \
  --num_workers 8 --eval_cider_freq 2 --early_stop_patience 3 --seed 44 \
  --no_onnx --no_tensorboard --no_resume --output_dir "$OUT"
echo "[s44] TEST-EVAL $(date -u)"
python tinyvlm_vast.py --eval_only --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
  --seed 44 --karpathy_json "$KJ" --karpathy_eval_split test \
  --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
  --output_dir "$OUT"
rclone copy "$OUT" "$DEST/lam_0.5/seed_44" --include "**/*.json" --include "*.log" 2>/dev/null || true
rclone copy "$WORK/seed44.log" "$DEST/seed44.log" 2>/dev/null || true
status S44_COMPLETE "lam0.5 seed44 done"
echo "[s44] COMPLETE $(date -u) — self-destructing"
# EXIT trap self-destructs
