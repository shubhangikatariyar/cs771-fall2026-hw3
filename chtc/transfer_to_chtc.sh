#!/bin/bash
# Sync code, pretrained weights, and chtc scripts to the CHTC submit node
# Usage:
#   CHTC_USER=your_netid ./chtc/transfer_to_chtc.sh
# Optional overrides:
#   CHTC_HOST="ap2001.chtc.wisc.edu" (or ap2002.chtc.wisc.edu)
#   CHTC_DIR="~/cs771-hw3"

set -euo pipefail

CHTC_USER="${CHTC_USER:-}"
if [ -z "$CHTC_USER" ]; then
    echo "ERROR: Please provide your UW NetID via CHTC_USER, e.g.:"
    echo "  CHTC_USER=skatariyar ./chtc/transfer_to_chtc.sh"
    exit 1
fi

CHTC_HOST="${CHTC_HOST:-ap2001.chtc.wisc.edu}"
CHTC_DIR="${CHTC_DIR:-~/cs771-hw3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "Connecting to ${CHTC_USER}@${CHTC_HOST}..."
CTL="$(mktemp -d)/ctl"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CTL}" -o ControlPersist=120)
trap 'ssh "${SSH_OPTS[@]}" -O exit "${CHTC_USER}@${CHTC_HOST}" 2>/dev/null || true' EXIT

# Ensure remote directories exist
ssh "${SSH_OPTS[@]}" "${CHTC_USER}@${CHTC_HOST}" "mkdir -p ${CHTC_DIR}/chtc_logs ${CHTC_DIR}/results"

echo "Syncing code/, pretrained/, and chtc/..."
rsync -avz -e "ssh ${SSH_OPTS[*]}" --partial --progress \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
    --exclude 'data/' --exclude 'logs/' --exclude 'models/' \
    --exclude '*.pth.tar' \
    code pretrained chtc \
    "${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/"

echo "Rebuilding hw3_code.tar.gz bundle on the submit node..."
ssh "${SSH_OPTS[@]}" "${CHTC_USER}@${CHTC_HOST}" \
    "cd ${CHTC_DIR} && tar czf hw3_code.tar.gz code pretrained && chmod +x chtc/*.sh && ls -lh hw3_code.tar.gz"

echo "=================================================================="
echo "Transfer complete! Next steps:"
echo "1. SSH into CHTC:"
echo "   ssh ${CHTC_USER}@${CHTC_HOST}"
echo "2. Submit jobs:"
echo "   cd ${CHTC_DIR}"
echo "   condor_submit chtc/mnist_fm_vit.sub"
echo "   condor_submit chtc/miniimagenet_fm_vit.sub"
echo "=================================================================="
