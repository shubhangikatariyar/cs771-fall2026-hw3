#!/usr/bin/env bash
# Run ON THE VM (after ssh). Clones the repo, installs deps, checks the GPU.
set -euo pipefail
source "$HOME/config.sh"

nvidia-smi

cd "$HOME"
if [[ ! -d "$REPO_DIR" ]]; then
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only

# The DLVM image already has torch with CUDA; install the rest without
# letting pip replace torch.
python3 -m pip install --upgrade pip
grep -v -i '^torch' requirements.txt > /tmp/reqs_no_torch.txt
python3 -m pip install -r /tmp/reqs_no_torch.txt

python3 - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available - check the NVIDIA driver"
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
EOF

# python3-dev: torch.compile (used by TAESD encode/decode) needs Python.h
sudo apt-get install -y tmux python3-dev >/dev/null 2>&1 || true
echo "Setup OK. Next: tmux new -s train, then bash ~/02_train.sh"
