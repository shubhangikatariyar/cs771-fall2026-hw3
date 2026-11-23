#!/bin/bash
# Incrementally sync what CHTC jobs need (code/, pretrained/, chtc/) to the submit node
# using rsync (only changed files are sent), then rebuild hw3_code.tar.gz remotely.
#
# Usage:  CHTC_USER=netid ./transfer_to_chtc.sh
# Optional: CHTC_HOST (default ap2002.chtc.wisc.edu), CHTC_DIR (default ~/cs771-hw3)
set -euo pipefail

CHTC_USER="${CHTC_USER:?set CHTC_USER to your CHTC username}"
CHTC_HOST="${CHTC_HOST:-ap2002.chtc.wisc.edu}"
CHTC_DIR="${CHTC_DIR:-cs771-hw3}"   # relative to remote home

cd "$(dirname "$0")"

ssh "${CHTC_USER}@${CHTC_HOST}" "mkdir -p ${CHTC_DIR}/chtc_logs"

rsync -avz --partial --progress \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
    --exclude 'data/' --exclude 'logs/' --exclude 'models/' \
    --exclude '*.pth.tar' \
    code pretrained chtc \
    "${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/"

# Rebuild the bundle the .sub files transfer to the execute node
ssh "${CHTC_USER}@${CHTC_HOST}" \
    "cd ${CHTC_DIR} && tar czf hw3_code.tar.gz code pretrained && chmod +x chtc/*.sh && ls -lh hw3_code.tar.gz"

echo "Done. On the submit node: cd ${CHTC_DIR} && condor_submit chtc/mnist_ddpm_unet.sub"
