#!/usr/bin/env bash
# Run ON THE VM inside tmux:   tmux new -s train   then   bash ~/02_train.sh [which]
#   which = mnist | mini | all (default: all = MNIST FM U-Net, then MiniImageNet FM U-Net)
# Safe to re-run: each job resumes from its latest epoch checkpoint if one exists,
# and is skipped if model_final.pth.tar already exists.
set -euo pipefail
source "$HOME/config.sh"
WHICH="${1:-all}"
cd "$HOME/$REPO_DIR/code"

run() {
  local name="$1" ckpt_freq="$2"
  local exp="../logs/${name}_${EXP_SUFFIX}"
  if [[ -f "$exp/model_final.pth.tar" ]]; then
    echo "== $name already finished ($exp/model_final.pth.tar), skipping"; return
  fi
  local resume=()
  local last
  last=$(ls "$exp"/epoch_*.pth.tar 2>/dev/null | sort | tail -1 || true)
  if [[ -n "$last" ]]; then
    echo "== resuming $name from $last"; resume=(--resume "$last")
  fi
  mkdir -p "$exp"
  echo "== training $name  ($(date))"
  local start=$SECONDS
  PYTHONUNBUFFERED=1 python3 train.py "configs/${name}.yaml" --output "$EXP_SUFFIX" -c "$ckpt_freq" "${resume[@]}" \
    2>&1 | tee -a "$exp/train_log.txt"
  echo "== $name done in $(( (SECONDS - start) / 60 )) min (this session)  ($(date))" | tee -a "$exp/train_log.txt"
  nvidia-smi --query-gpu=name --format=csv,noheader >> "$exp/train_log.txt"
}

case "$WHICH" in
  mnist) run mnist_fm_unet 5 ;;
  mini)  run miniimagenet_fm_unet 2 ;;   # ckpt every 2 epochs: less lost on preemption
  all)   run mnist_fm_unet 5; run miniimagenet_fm_unet 2 ;;
  *) echo "usage: $0 [mnist|mini|all]"; exit 1 ;;
esac
