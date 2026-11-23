#!/bin/bash
# Fetch finished CHTC job outputs and stage the submission payload in results/.
#
# 1. rsync results_*.tar.gz from the submit node (single SSH login)
# 2. extract each into ./logs/ (full logs, gitignored)
# 3. copy to results/<experiment>/ only what the submission needs: config,
#    TensorBoard events, sample grids, and the FINAL epoch checkpoint
#    (intermediate epoch_*.pth.tar are skipped)
#
# Usage:  CHTC_USER=netid ./fetch_results.sh
# Optional: CHTC_HOST (default ap2002.chtc.wisc.edu), CHTC_DIR (default cs771-hw3)
set -euo pipefail

CHTC_USER="asehgal9"
CHTC_HOST="ap2001.chtc.wisc.edu"
CHTC_DIR="~/cs771-hw3"

cd "$(dirname "$0")"

CTL="$(mktemp -d)/ctl"
SSH_OPTS=(-o ControlMaster=auto -o "ControlPath=${CTL}" -o ControlPersist=120)
trap 'ssh "${SSH_OPTS[@]}" -O exit "${CHTC_USER}@${CHTC_HOST}" 2>/dev/null || true' EXIT

FETCH_DIR=chtc_fetched
mkdir -p "$FETCH_DIR" logs results

rsync -avz -e "ssh ${SSH_OPTS[*]}" --partial --progress \
    --include 'results_*.tar.gz' --exclude '*' \
    "${CHTC_USER}@${CHTC_HOST}:${CHTC_DIR}/" "$FETCH_DIR/"

shopt -s nullglob
tarballs=("$FETCH_DIR"/results_*.tar.gz)
if [ ${#tarballs[@]} -eq 0 ]; then
    echo "No results_*.tar.gz on the submit node yet (jobs still running?)"; exit 0
fi

for tb in "${tarballs[@]}"; do
    tar xzf "$tb" -C .   # tarball contains logs/<experiment>/...
done

for exp in logs/*/; do
    name="$(basename "$exp")"
    # newest epoch checkpoint = final model
    final="$(find "$exp" -name 'epoch_*.pth.tar' | sort | tail -n 1)"
    excludes=()
    while IFS= read -r ck; do
        [ "$ck" != "$final" ] && excludes+=(--exclude "$(basename "$ck")")
    done < <(find "$exp" -name 'epoch_*.pth.tar')
    rsync -a "${excludes[@]}" "$exp" "results/$name/"
    echo "results/$name  (final checkpoint: $(basename "${final:-none}"))"
done
