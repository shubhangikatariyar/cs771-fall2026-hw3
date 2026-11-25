#!/bin/bash
# CHTC job: train MiniImageNet Latent FM with UNet backbone (80 epochs)
CONFIG=miniimagenet_fm_unet
mkdir -p logs

finish() { tar czf "results_${CONFIG}.tar.gz" logs job_output.txt 2>/dev/null || tar czf "results_${CONFIG}.tar.gz" job_output.txt; }
trap finish EXIT

{
  set -euxo pipefail
  hostname; nvidia-smi || true; python3 --version
  tar xzf hw3_code.tar.gz
  pip install --quiet -r code/requirements.txt
  cd code
  python train.py "./configs/${CONFIG}.yaml"
  cd ..
} > job_output.txt 2>&1
