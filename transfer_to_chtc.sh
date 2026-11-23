#!/bin/bash
# Incrementally sync what CHTC jobs need (code/, pretrained/, chtc/) to the submit node
# using rsync (only changed files are sent), then rebuild hw3_code.tar.gz remotely.
#
# Usage:  CHTC_USER=netid ./transfer_to_chtc.sh
# Optional: CHTC_HOST (default ap2002.chtc.wisc.edu), CHTC_DIR (default ~/cs771-hw3)
set -euo pipefail

CHTC_USER="asehgal9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="~/cs771-hw3"   # relative to remote home

cd "$(dirname "$0")"

# Share one authenticated SSH connection across all ssh/rsync calls, so the
# password / 2FA is only asked once.
CTL="$(mktemp -d)/ctl"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CTL}" -o ControlPersist=120)
trap 'ssh "${SSH_OPTS[@]}" -O exit "${CHTC_USER}@${CHTC_HOST}" 2>/dev/null || true' EXIT

ssh "${SSH_OPTS[@]}" "${CHTC_USER}@${CHTC_HOST}" "mkdir -p ${CHTC_DIR}/chtc_logs"

rsync -avz -e "ssh ${SSH_OPTS[*]}" --partial --progress \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
    --exclude 'data/' --exclude 'logs/' --exclude 'models/' \
    --exclude '*.pth.tar' \
    code pretrained chtc \
    "${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/"

# Rebuild the bundle the .sub files transfer to the execute node
ssh "${SSH_OPTS[@]}" "${CHTC_USER}@${CHTC_HOST}" \
    "cd ${CHTC_DIR} && tar czf hw3_code.tar.gz code pretrained && chmod +x chtc/*.sh && ls -lh hw3_code.tar.gz"

echo "Done. On the submit node: cd ${CHTC_DIR} && condor_submit chtc/mnist_ddpm_unet.sub"
