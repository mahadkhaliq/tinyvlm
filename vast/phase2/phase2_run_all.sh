#!/usr/bin/env bash
# phase2_run_all.sh — on-instance token-budget CLIP routing experiment (vast.ai).
# Pivotal Phase-2 experiment (OPUS48_PLAN_A). Uploads: a tinyvlm_vast.py ALREADY
# patched per vast/phase2/integrate.md + clip_token_budget.py + phase2_aggregate.py.
#
# Flow: deps -> SMOKE GATE (fail cheap if patch broken) -> COCO+Karpathy -> 3 ANC
# token-budget seeds (train Karpathy train/val, eval Karpathy TEST) -> aggregate
# A1-A4 -> push to S3. Dense control is NOT rerun (reuse E_karpathy_clean 79.80).
set -uo pipefail
WORK=/workspace
exec > >(tee -a "$WORK/phase2.log") 2>&1
echo "[phase2] START $(date -u)"

COCO="$WORK/coco"; KJ="$COCO/dataset_coco.json"
export RCLONE_CONFIG="$WORK/rclone.conf"
export PYTHONPATH="$WORK:${PYTHONPATH:-}"          # so clip_token_budget.py imports
DEST="s3:tinyvlm-neurips2026/finalpush/phase2_tokenbudget"
status() { printf '%s\n' "$(date -u) $2" | rclone rcat "$DEST/STATUS_$1" 2>/dev/null || true; }

# ---- cost safeguard: hard 7h watchdog -> power off (stops GPU billing even if
#      training hangs and the control plane stops polling). Auto-poweroff also runs
#      at normal completion (end of script). Instance still needs `vastai destroy`
#      from the control plane to release it, but a powered-off instance bills ~$0 GPU.
( sleep 25200; echo "[phase2] WATCHDOG 7h hit — powering off"; \
  printf '%s\n' "$(date -u) watchdog-timeout" | rclone rcat "$DEST/STATUS_TIMEOUT" 2>/dev/null || true; \
  sudo poweroff || poweroff || shutdown -h now ) &
WATCHDOG=$!

# ---- deps (do NOT reinstall torch/torchvision/numpy on the pytorch image) ----
# NOTE: the pytorch devel image lacks `unzip` and `rclone` — install both or COCO
# never extracts (silent) and results never reach S3.
export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq unzip rclone >/dev/null 2>&1 || true
pip install -q pycocotools pillow tqdm tensorboard scipy matplotlib onnx open_clip_torch pycocoevalcap nltk 2>&1 | tail -2
python -m nltk.downloader punkt punkt_tab >/dev/null 2>&1 || true
command -v rclone >/dev/null 2>&1 || (curl -s https://rclone.org/install.sh | bash >/dev/null 2>&1)
command -v unzip  >/dev/null 2>&1 || { echo "[phase2] FATAL: no unzip"; status FAILED "no unzip"; exit 1; }

# ---- preflight: CUDA + the patch is present ----
python - <<'PY'
import torch, sys, importlib.util
assert torch.cuda.is_available(), "no CUDA"
s=importlib.util.spec_from_file_location("tv","/workspace/tinyvlm_vast.py")
m=importlib.util.module_from_spec(s); sys.modules["tv"]=m; s.loader.exec_module(m)
c=m.Config()
assert hasattr(c,"clip_token_budget") and hasattr(c,"token_budgets"), \
    "tinyvlm_vast.py NOT patched with Phase-2 flags (apply integrate.md before upload)"
import clip_token_budget  # must be importable
print("[preflight] OK gpu=%s budgets=%s" % (torch.cuda.get_device_name(0), c.token_budgets))
PY
[ $? -ne 0 ] && { status FAILED "preflight"; echo "[phase2] PREFLIGHT FAILED"; exit 1; }

# ---- SMOKE GATE: run the patched path on synthetic data (fail cheap before COCO/GPU-h) ----
echo "[phase2] smoke gate..."
python tinyvlm_vast.py --smoke_test --smoke_test_size 64 \
   --clip_backbone --clip_token_budget --epochs 1 --no_onnx \
   --output_dir "$WORK/smoke" 2>&1 | tail -8
if [ ${PIPESTATUS[0]} -ne 0 ]; then status FAILED "smoke gate"; echo "[phase2] SMOKE FAILED — patch broken, aborting before spend"; exit 1; fi
status STARTED "preflight + smoke ok"

# ---- data ----
mkdir -p "$COCO"; cd "$COCO"
dl(){ curl -fsSL "$1" -o "$2"; }
[ -d train2017 ]   || { dl http://images.cocodataset.org/zips/train2017.zip t.zip && unzip -q t.zip && rm t.zip; }
[ -d val2017 ]     || { dl http://images.cocodataset.org/zips/val2017.zip v.zip && unzip -q v.zip && rm v.zip; }
[ -d annotations ] || { dl http://images.cocodataset.org/annotations/annotations_trainval2017.zip a.zip && unzip -q a.zip && rm a.zip; }
[ -f "$KJ" ]       || { dl https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip c.zip && unzip -q -o c.zip dataset_coco.json && rm c.zip; }
status DATA_READY "coco+karpathy staged"

# ---- 3 token-budget ANC seeds (dense control already exists: E_karpathy_clean 79.80) ----
cd "$WORK"
NAME=phase2_tokenbudget_anc
for S in 42 43 44; do
  echo "[phase2] ===== $NAME seed $S TRAIN $(date -u) ====="
  python tinyvlm_vast.py \
    --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
    --karpathy_json "$KJ" --karpathy_eval_split val \
    --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
    --lr 3e-4 --target_budget 14e9 --lambda_balance 0.01 --epochs 15 --batch_size 64 \
    --num_workers 8 --eval_cider_freq 2 --early_stop_patience 3 --seed "$S" \
    --no_onnx --no_tensorboard \
    --output_dir "$WORK/runs/$NAME/seed_$S"
  echo "[phase2] ===== $NAME seed $S TEST-EVAL $(date -u) ====="
  python tinyvlm_vast.py --eval_only --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
    --seed "$S" --karpathy_json "$KJ" --karpathy_eval_split test \
    --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
    --output_dir "$WORK/runs/$NAME/seed_$S"
  rclone copy "$WORK/runs/$NAME/seed_$S" "$DEST/$NAME/seed_$S" \
    --include "*.json" --include "**/*.json" --include "**/best.pt" --include "*.log" 2>/dev/null || true
  status "PROGRESS_${S}" "done"
done

# ---- aggregate A1-A4 and push ----
python phase2_aggregate.py "$WORK/runs/$NAME" > "$WORK/runs/$NAME/aggregate.json" 2>"$WORK/runs/$NAME/aggregate.err" || true
rclone copy "$WORK/runs/$NAME" "$DEST/$NAME" --include "aggregate.json" --include "NOTES.md" 2>/dev/null || true
rclone copy "$WORK/phase2.log" "$DEST/phase2.log" 2>/dev/null || true
status COMPLETE "3 token-budget seeds + test eval + aggregate done"
echo "[phase2] COMPLETE $(date -u)"

# ---- auto power-off on completion so GPU billing stops without waiting for a poll ----
kill "$WATCHDOG" 2>/dev/null || true
echo "[phase2] powering off to stop billing (control plane still runs 'vastai destroy')"
sleep 10
sudo poweroff || poweroff || shutdown -h now
