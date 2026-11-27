#!/usr/bin/env bash
# Run ON THE VM after training. Computes FID for one checkpoint: the no-guidance
# baseline plus each CFG scale (bonus). Output: <exp>/eval_results/*.txt
#
#   bash ~/03_eval.sh unet                       # your U-Net run
#   bash ~/03_eval.sh vit /path/to/vit/model_final.pth.tar   # Shubhangi's ViT checkpoint
#   SCALES="nocfg" bash ~/03_eval.sh unet        # baseline only
set -euo pipefail
source "$HOME/config.sh"
ARCH="${1:?usage: $0 unet|vit [checkpoint]}"
cd "$HOME/$REPO_DIR/code"

CKPT="${2:-../logs/miniimagenet_fm_${ARCH}_${EXP_SUFFIX}/model_final.pth.tar}"
[[ -f "$CKPT" ]] || { echo "checkpoint not found: $CKPT"; exit 1; }
OUT="$(dirname "$CKPT")/eval_results"
mkdir -p "$OUT"
SCALES="${SCALES:-nocfg cfg1.5 cfg2.5 cfg4.0}"

for s in $SCALES; do
  cfgfile="configs/eval/miniimagenet_fm_${ARCH}_${s}.yaml"
  echo "== FID $ARCH $s  ($(date))"
  start=$SECONDS
  PYTHONUNBUFFERED=1 python3 eval.py "$cfgfile" "$CKPT" 2>&1 | tee "$OUT/fid_${s}.txt"
  echo "eval wall time: $(( SECONDS - start )) s" | tee -a "$OUT/fid_${s}.txt"
done

echo; echo "== Summary ($ARCH)"
grep -H "FID:" "$OUT"/fid_*.txt
