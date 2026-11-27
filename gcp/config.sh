# Shared settings for the GCP scripts. Edit these, then `source gcp/config.sh`
# is done automatically by the other scripts.

PROJECT="cs771-513220"
ZONE="us-central1-a"            # pick a zone that has your GPU type and quota
VM_NAME="akshat-hw3"

# GPU choice:
#   L4 (recommended): MACHINE="g2-standard-8", ACCELERATOR="" (L4 is built into g2)
#   T4:               MACHINE="n1-standard-8", ACCELERATOR="type=nvidia-tesla-t4,count=1"
MACHINE="g2-standard-8"
ACCELERATOR=""

# "true" = Spot VM (much cheaper, can be preempted; training resumes from the
# last epoch checkpoint). "false" = on-demand.
SPOT="true"

BOOT_DISK_GB=150

# Deep Learning VM image with PyTorch + CUDA. Family names change over time;
# list current ones with:
#   gcloud compute images list --project deeplearning-platform-release \
#     --no-standard-images --filter="family~pytorch" --format="value(family)" | sort -u
IMAGE_PROJECT="deeplearning-platform-release"
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"

REPO_URL="https://github.com/shubhangikatariyar/cs771-fall2026-hw3.git"
REPO_DIR="cs771-fall2026-hw3"   # on the VM, relative to $HOME
EXP_SUFFIX="gcp"                # train.py --output; fixed so --resume finds the same folder
