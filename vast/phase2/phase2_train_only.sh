#!/usr/bin/env bash
# phase2_train_only.sh — lean re-run: data already downloaded+extracted, deps+rclone
# installed. Runs the 3 token-budget ANC seeds + Karpathy-test eval + aggregate + push.
# Writes STATUS_COMPLETE2 (the scoped S3 key cannot delete the earlier stale
# STATUS_COMPLETE from the unzip-less crash cascade). No in-container watchdog here —
# the setsid 18h backstop is already running, and the control plane polls + destroys.
set -uo pipefail
WORK=/workspace; COCO=$WORK/coco; KJ=$COCO/dataset_coco.json
export RCLONE_CONFIG=$WORK/rclone.conf PYTHONPATH=$WORK
exec > >(tee -a "$WORK/phase2_train.log") 2>&1
DEST=s3:tinyvlm-neurips2026/finalpush/phase2_tokenbudget
status(){ printf '%s\n' "$(date -u) $2" | rclone rcat "$DEST/STATUS_$1" 2>/dev/null || true; }
cd "$WORK"

# verify data really extracted (the whole reason for this re-run)
if ! { [ -d "$COCO/train2017" ] && [ -f "$COCO/annotations/captions_train2017.json" ] && [ -f "$KJ" ]; }; then
  status FAILED "data still missing at re-run"; echo "[rerun] DATA MISSING — abort"; exit 1
fi
echo "[rerun] data OK: train=$(ls $COCO/train2017 | wc -l) imgs; annotations + Karpathy json present"
rm -rf "$WORK/runs"    # clear the crashed empty seed dirs
status RERUN_STARTED "data extracted; training 3 token-budget seeds"

NAME=phase2_tokenbudget_anc
for S in 42 43 44; do
  echo "[rerun] ===== $NAME seed $S TRAIN $(date -u) ====="
  python tinyvlm_vast.py \
    --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
    --karpathy_json "$KJ" --karpathy_eval_split val \
    --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
    --lr 3e-4 --target_budget 14e9 --lambda_balance 0.01 --epochs 15 --batch_size 64 \
    --num_workers 8 --eval_cider_freq 2 --early_stop_patience 3 --seed "$S" \
    --no_onnx --no_tensorboard --no_resume \
    --output_dir "$WORK/runs/$NAME/seed_$S"
  echo "[rerun] ===== $NAME seed $S TEST-EVAL $(date -u) ====="
  python tinyvlm_vast.py --eval_only --clip_backbone --clip_token_budget --token_budgets 24,36,49 \
    --seed "$S" --karpathy_json "$KJ" --karpathy_eval_split test \
    --coco_imgs "$COCO/train2017" --coco_anns "$COCO/annotations/captions_train2017.json" \
    --output_dir "$WORK/runs/$NAME/seed_$S"
  rclone copy "$WORK/runs/$NAME/seed_$S" "$DEST/$NAME/seed_$S" \
    --include "*.json" --include "**/*.json" --include "**/best.pt" --include "*.log" 2>/dev/null || true
  status "PROGRESS_${S}" "done"
done

python phase2_aggregate.py "$WORK/runs/$NAME" > "$WORK/runs/$NAME/aggregate.json" 2>"$WORK/runs/$NAME/aggregate.err" || true
rclone copy "$WORK/runs/$NAME" "$DEST/$NAME" --include "aggregate.json" --include "NOTES.md" 2>/dev/null || true
rclone copy "$WORK/phase2_train.log" "$DEST/phase2_train.log" 2>/dev/null || true
status COMPLETE2 "3 token-budget seeds + test eval + aggregate done (re-run)"
echo "[rerun] COMPLETE2 $(date -u)"
