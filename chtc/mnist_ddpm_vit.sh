#!/bin/bash
# CHTC job: train MNIST DDPM with vit backbone (runs inside a PyTorch docker image)
CONFIG=mnist_ddpm_vit
mkdir -p logs

# Always return a tarball (with the job log), even if something below fails
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
