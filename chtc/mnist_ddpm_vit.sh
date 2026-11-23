#!/bin/bash
# CHTC job: train MNIST DDPM with vit backbone
set -euo pipefail
CONFIG=mnist_ddpm_vit

tar xzf hw3_code.tar.gz
python3 -m venv venv
source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r code/requirements.txt

cd code
python train.py "./configs/${CONFIG}.yaml"
cd ..

tar czf "results_${CONFIG}.tar.gz" logs
