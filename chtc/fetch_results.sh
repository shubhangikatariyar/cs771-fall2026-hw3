#!/bin/bash
# Fetch finished CHTC job results and stage final models/grids into results/.
# Usage:
#   CHTC_USER=your_netid ./chtc/fetch_results.sh
# Optional overrides:
#   CHTC_HOST="ap2001.chtc.wisc.edu"
#   CHTC_DIR="~/cs771-hw3"

set -euo pipefail

CHTC_USER="${CHTC_USER:-}"
if [ -z "$CHTC_USER" ]; then
    echo "ERROR: Please provide your UW NetID via CHTC_USER, e.g.:"
    echo "  CHTC_USER=skatariyar ./chtc/fetch_results.sh"
    exit 1
fi

CHTC_HOST="${CHTC_HOST:-ap2001.chtc.wisc.edu}"
CHTC_DIR="${CHTC_DIR:-~/cs771-hw3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

CTL="$(mktemp -d)/ctl"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CTL}" -o ControlPersist=120)
trap 'ssh "${SSH_OPTS[@]}" -O exit "${CHTC_USER}@${CHTC_HOST}" 2>/dev/null || true' EXIT

FETCH_DIR=chtc_fetched
mkdir -p "$FETCH_DIR" logs results

echo "Checking for results_*.tar.gz on ${CHTC_HOST}..."
rsync -avz -e "ssh ${SSH_OPTS[*]}" --partial --progress \
    --include 'results_*.tar.gz' --exclude '*' \
    "${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/" "$FETCH_DIR/"

shopt -s nullglob
tarballs=("$FETCH_DIR"/results_*.tar.gz)
if [ ${#tarballs[@]} -eq 0 ]; then
    echo "No results_*.tar.gz found on the submit node yet (jobs may still be running in condor_q)."
    exit 0
fi

echo "Extracting tarballs..."
for tb in "${tarballs[@]}"; do
    tar xzf "$tb" -C . --exclude job_output.txt
done

echo "Staging final checkpoints and grids into results/..."
for exp in logs/*/; do
    [ -d "$exp" ] || continue
    name="$(basename "$exp")"
    final="$(find "$exp" -name 'model_final.pth.tar' | head -n 1)"
    [ -z "$final" ] && final="$(find "$exp" -name 'epoch_*.pth.tar' | sort | tail -n 1)"
    excludes=()
    while IFS= read -r ck; do
        [ "$ck" != "$final" ] && excludes+=(--exclude "$(basename "$ck")")
    done < <(find "$exp" -name 'epoch_*.pth.tar' -o -name 'model_final.pth.tar')
    rsync -a "${excludes[@]}" "$exp" "results/$name/"
    echo "Saved: results/$name  (final checkpoint: $(basename "${final:-none}"))"
done

echo "Done! Final results staged in results/."
