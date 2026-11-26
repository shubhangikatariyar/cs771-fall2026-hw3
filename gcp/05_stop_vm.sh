#!/usr/bin/env bash
# Run on your Mac.
#   bash gcp/05_stop_vm.sh          # stop: GPU billing ends, disk (and logs/) kept
#   bash gcp/05_stop_vm.sh delete   # delete VM and disk: only after fetching results
set -euo pipefail
source "$(dirname "$0")/config.sh"
if [[ "${1:-stop}" == "delete" ]]; then
  gcloud compute instances delete "$VM_NAME" --zone="$ZONE"
else
  gcloud compute instances stop "$VM_NAME" --zone="$ZONE"
  echo "Stopped. Restart later with: gcloud compute instances start $VM_NAME --zone=$ZONE"
fi
