#!/usr/bin/env bash
# CS 771 Fall 2026 — Homework 3: End-to-End Reproduction Script
# Usage:
#   ./reproduce.sh [target]
# Targets:
#   mnist_ddpm_unet | mnist_ddpm_vit | mnist_fm_unet | mnist_fm_vit |
#   miniimagenet_fm_unet | miniimagenet_fm_vit | all

set -euo pipefail

TARGET="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${SCRIPT_DIR}/code"

echo "=== CS 771 Fall 2026 HW3 Reproduction ==="
echo "Target: ${TARGET}"
echo "Working directory: ${CODE_DIR}"

run_exp() {
    local cfg="$1"
    local desc="$2"
    echo "--------------------------------------------------------"
    echo "Running: ${desc}"
    echo "Config: configs/${cfg}"
    echo "--------------------------------------------------------"
    python "${CODE_DIR}/train.py" "${CODE_DIR}/configs/${cfg}"
}

cd "${CODE_DIR}"

case "${TARGET}" in
    mnist_ddpm_unet)
        run_exp "mnist_ddpm_unet.yaml" "MNIST DDPM with U-Net (30 ep, 500 steps)"
        ;;
    mnist_ddpm_vit)
        run_exp "mnist_ddpm_vit.yaml" "MNIST DDPM with ViT (30 ep, 500 steps)"
        ;;
    mnist_fm_unet)
        run_exp "mnist_fm_unet.yaml" "MNIST Flow Matching with U-Net (30 ep, 10 steps)"
        ;;
    mnist_fm_vit)
        run_exp "mnist_fm_vit.yaml" "MNIST Flow Matching with ViT (30 ep, 10 steps)"
        ;;
    miniimagenet_fm_unet)
        run_exp "miniimagenet_fm_unet.yaml" "MiniImageNet Latent FM with U-Net (80 ep, 10 steps)"
        ;;
    miniimagenet_fm_vit)
        run_exp "miniimagenet_fm_vit.yaml" "MiniImageNet Latent FM with ViT (80 ep, 10 steps)"
        ;;
    all)
        echo "Running all 6 configurations sequentially..."
        run_exp "mnist_ddpm_unet.yaml" "MNIST DDPM U-Net"
        run_exp "mnist_ddpm_vit.yaml" "MNIST DDPM ViT"
        run_exp "mnist_fm_unet.yaml" "MNIST FM U-Net"
        run_exp "mnist_fm_vit.yaml" "MNIST FM ViT"
        run_exp "miniimagenet_fm_unet.yaml" "MiniImageNet FM U-Net"
        run_exp "miniimagenet_fm_vit.yaml" "MiniImageNet FM ViT"
        ;;
    *)
        echo "Unknown target: ${TARGET}"
        echo "Available targets: mnist_ddpm_unet, mnist_ddpm_vit, mnist_fm_unet, mnist_fm_vit, miniimagenet_fm_unet, miniimagenet_fm_vit, all"
        exit 1
        ;;
esac

echo "========================================================"
echo "Finished experiment: ${TARGET}"
echo "Logs and sample grids saved to ../logs/"
echo "========================================================"
