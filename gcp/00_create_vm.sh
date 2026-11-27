#!/usr/bin/env bash
# Run on your Mac. Creates the GPU VM (this starts billing).
set -euo pipefail
source "$(dirname "$0")/config.sh"

gcloud config set project "$PROJECT"

args=(
  --zone="$ZONE"
  --machine-type="$MACHINE"
  --image-project="$IMAGE_PROJECT"
  --image-family="$IMAGE_FAMILY"
  --boot-disk-size="${BOOT_DISK_GB}GB"
  --boot-disk-type=pd-balanced
  --maintenance-policy=TERMINATE
  --metadata=install-nvidia-driver=True
  --scopes=default
)
[[ -n "$ACCELERATOR" ]] && args+=(--accelerator="$ACCELERATOR")
[[ "$SPOT" == "true" ]] && args+=(--provisioning-model=SPOT --instance-termination-action=STOP)

gcloud compute instances create "$VM_NAME" "${args[@]}"

echo
echo "VM created. Wait ~2 min for the NVIDIA driver install, then:"
echo "  gcloud compute scp --zone=$ZONE gcp/01_setup_vm.sh gcp/02_train.sh gcp/03_eval.sh gcp/config.sh $VM_NAME:~"
echo "  gcloud compute ssh --zone=$ZONE $VM_NAME"
