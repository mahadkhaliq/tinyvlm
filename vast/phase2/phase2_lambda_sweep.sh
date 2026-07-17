#!/usr/bin/env bash
# phase2_lambda_sweep.sh — critic's #1 fix: sweep the token-budget experiment over
# lambda_balance to test whether the inference-time collapse is a lambda artifact.
# lambda=0.01 is already done (E_phase2_tokenbudget); this runs lambda in {0.1, 0.5}
# x seeds {42,43,44}, aggregates A1-A4 per lambda, pushes to S3. Same patched code.
set -uo pipefail
WORK=/workspace; COCO=$WORK/coco; KJ=$COCO/dataset_coco.json
export RCLONE_CONFIG=$WORK/rclone.conf PYTHONPATH=$WORK
export DEBIAN_FRONTEND=noninteractive
exec > >(tee -a "$WORK/sweep.log") 2>&1
DEST=s3:tinyvlm-neurips2026/finalpush/phase2_lambda_sweep
status(){ printf '%s\n' "$(date -u) $2" | rclone rcat "$DEST/STATUS_$1" 2>/dev/null || true; }
echo "[sweep] START $(date -u)"

# ---- deps: MUST install unzip + rclone (image lacks both) ----
apt-get install -y -qq unzip rclone >/dev/null 2>&1 || true
pip install -q pycocotools pillow tqdm tensorboard scipy matplotlib onnx open_clip_torch pycocoevalcap nltk 2>&1 | tail -2
python -m nltk.downloader punkt punkt_tab >/dev/null 2>&1 || true
command -v rclone >/dev/null 2>&1 || (curl -s https://rclone.org/install.sh | bash >/dev/null 2>&1)
command -v unzip  >/dev/null 2>&1 || { status FAILED "no unzip"; echo "[sweep] FATAL no unzip"; exit 1; }

# ---- preflight + smoke gate (fail cheap) ----
python - <<'PY'
import torch,sys,importlib.util
assert torch.cuda.is_available(),"no CUDA"
s=importlib.util.spec_from_file_location("tv","/workspace/tinyvlm_vast.py"); m=importlib.util.module_from_spec(s); sys.modules["tv"]=m; s.loader.exec_module(m)
assert hasattr(m.Config(),"clip_token_budget"),"tinyvlm_vast.py not patched"
import clip_token_budget
print("[preflight] OK", torch.cuda.get_device_name(0))
PY
[ $? -ne 0 ] && { status FAILED preflight; exit 1; }
python tinyvlm_vast.py --smoke_test --smoke_test_size 64 --clip_backbone --clip_token_budget --epochs 1 --no_onnx --output_dir "$WORK/smoke" 2>&1 | tail -4
[ ${PIPESTATUS[0]} -ne 0 ] && { status FAILED smoke; echo "[sweep] SMOKE FAILED"; exit 1; }
status STARTED "preflight+smoke ok"

# ---- data (download + extract; guards skip if present) ----
mkdir -p "$COCO"; cd "$COCO"; dl(){ curl -fsSL "$1" -o "$2"; }
[ -d train2017 ]   || { dl http://images.cocodataset.org/zips/train2017.zip t.zip && unzip -q t.zip && rm -f t.zip; }
[ -d val2017 ]     || { dl http://images.cocodataset.org/zips/val2017.zip v.zip && unzip -q v.zip && rm -f v.zip; }
[ -d annotations ] || { dl http://images.cocodataset.org/annotations/annotations_trainval2017.zip a.zip && unzip -q a.zip && rm -f a.zip; }
[ -f "$KJ" ]       || { dl https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip c.zip && unzip -q -o c.zip dataset_coco.json && rm -f c.zip; }
{ [ -d train2017 ] && [ -f annotations/captions_train2017.json ] && [ -f "$KJ" ]; } || { status FAILED "data missing"; exit 1; }
status DATA_READY "coco+karpathy staged"

# ---- sweep: lambda in {0.1, 0.5} x seeds ----
cd "$WORK"
for LAM in 0.1 0.5; do
  TAG="lam_${LAM}"
  for S in 42 43 44; do
    echo "[sweep] ===== $TAG seed $S TRAIN $(date -u) ====="
    python tinyvlm_vast.py \
      --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
      --karpathy_json "$KJ" --karpathy_eval_split val \
      --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
      --lr 3e-4 --target_budget 14e9 --lambda_balance "$LAM" --epochs 15 --batch_size 64 \
      --num_workers 8 --eval_cider_freq 2 --early_stop_patience 3 --seed "$S" \
      --no_onnx --no_tensorboard --no_resume \
      --output_dir "$WORK/runs/$TAG/seed_$S"
    echo "[sweep] ===== $TAG seed $S TEST-EVAL $(date -u) ====="
    python tinyvlm_vast.py --eval_only --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
      --seed "$S" --karpathy_json "$KJ" --karpathy_eval_split test \
      --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
      --output_dir "$WORK/runs/$TAG/seed_$S"
    rclone copy "$WORK/runs/$TAG/seed_$S" "$DEST/$TAG/seed_$S" --include "**/*.json" --include "*.log" 2>/dev/null || true
    status "PROGRESS_${TAG}_${S}" "done"
  done
  python phase2_aggregate.py "$WORK/runs/$TAG" > "$WORK/runs/$TAG/aggregate.json" 2>/dev/null || true
  rclone copy "$WORK/runs/$TAG" "$DEST/$TAG" --include "aggregate.json" --include "NOTES.md" 2>/dev/null || true
  status "AGG_${TAG}" "aggregated"
done

rclone copy "$WORK/sweep.log" "$DEST/sweep.log" 2>/dev/null || true
status SWEEP_COMPLETE "lambda 0.1 + 0.5 x 3 seeds done"
echo "[sweep] SWEEP_COMPLETE $(date -u)"
