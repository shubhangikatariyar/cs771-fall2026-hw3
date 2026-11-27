#!/usr/bin/env bash
# Run on your Mac. Copies the submission-relevant files of a finished run from
# the VM into results/ (config, train log, TensorBoard events, sample grids,
# model_final.pth.tar, FID logs). No intermediate epoch checkpoints, no eval images.
#   bash gcp/04_fetch_results.sh mnist_fm_unet
#   bash gcp/04_fetch_results.sh miniimagenet_fm_unet
set -euo pipefail
source "$(dirname "$0")/config.sh"
NAME="${1:?usage: $0 <config name, e.g. mnist_fm_unet>}"
REMOTE="~/$REPO_DIR/logs/${NAME}_${EXP_SUFFIX}"
DEST="results/${NAME}_${EXP_SUFFIX}"
mkdir -p "$DEST"

# Pack only what we want on the VM side, then copy one tarball.
gcloud compute ssh --zone="$ZONE" "$VM_NAME" --command "cd $REMOTE && \
  tar czf /tmp/${NAME}.tgz --ignore-failed-read \
    config.txt train_log.txt logs sample-*.png model_final.pth.tar eval_results 2>/dev/null; ls -la /tmp/${NAME}.tgz"
gcloud compute scp --zone="$ZONE" "$VM_NAME:/tmp/${NAME}.tgz" "/tmp/${NAME}.tgz"
tar xzf "/tmp/${NAME}.tgz" -C "$DEST"
rm -f "/tmp/${NAME}.tgz"

echo "Fetched into $DEST:"; ls -la "$DEST"
