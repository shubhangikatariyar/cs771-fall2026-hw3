# Assignment 3 — Team Plan

**Course:** BMI/CS 771 Fall 2026, Homework 3 — Image Generation with Diffusion and Flow Matching Models (PyTorch)  
**Team size:** 3  
**Total:** 12 pts + 4 bonus

Starter scaffolding (data pipelines, training loops, configs, eval scripts, autoencoder weights) is provided. The work is to fill in missing model code in four files, train, record results, and write the PDF report.

- `code/libs/unet.py` — U-Net decoder (skip concatenation, residual blocks, cross-attention, upsampling)
- `code/libs/vit.py` — Vision Transformer forward pass (LightningDiT, patchification, adaLN-Zero modulation)
- `code/libs/ddpm.py` — Denoising Diffusion Probabilistic Models (loss, reverse denoising, sampling)
- `code/libs/fm.py` — Flow Matching with Rectified Flow (loss, Euler ODE integration)
- `code/libs/{ddpm,fm}.py` — Latent modeling (TAESD autoencoder encoding/decoding hooks)

---

## What needs to be implemented

| # | File / location | Pts | Task |
|---|-----------------|:---:|------|
| **4.1a** | `libs/unet.py`<br>`UNet.forward` | 1 | Complete U-Net decoder: reverse pass matching encoder outputs, residual blocks (time $t$), cross-attention (class $c$), and upsamplers with skip concatenation |
| **4.1b** | `libs/vit.py`<br>`LightningDiT.forward` | 1 | Complete DiT forward pass: 2D positional embeddings, stacked DiT blocks with adaLN-Zero time/class modulation, adaptive layer norm, and linear unpatchify |
| **4.2a** | `libs/ddpm.py`<br>`compute_loss` | 1 | DDPM simplified MSE loss $\|\epsilon - \epsilon_\theta(x_t, c, t)\|_2^2$ (Algorithm 1, line 5) with discrete $t \sim \mathcal{U}\{0, T-1\}$ and $\epsilon \sim \mathcal{N}(0, I)$ |
| **4.2b** | `libs/ddpm.py`<br>`p_sample` & `generate` | 1 | Reverse denoising step (Equation 11 posterior mean $\mu_\theta$ + variance $\sigma_t z$) and full reverse sampling loop (Algorithm 2) from $t = T-1$ to $0$ |
| **4.2c** | MNIST DDPM Experiments | 1 | Train DDPM on MNIST with both U-Net and ViT (30 epochs, 500 steps); record curves and sample grids |
| **4.3a** | `libs/fm.py`<br>`compute_loss` | 1 | Rectified Flow regression loss $\|v_\theta(x_t, c, t) - (x_{\text{data}} - z)\|_2^2$ along straight path $x_t = (1-t)z + t x_{\text{data}}$ for $t \sim \mathcal{U}[0, 1]$ |
| **4.3b** | `libs/fm.py`<br>`generate` | 1 | Forward Euler ODE integration $y(t + \Delta t) \approx y(t) + \Delta t \cdot v_\theta(y(t), c, t)$ from $t = 0$ (noise) to $t = 1$ (image) |
| **4.3c** | MNIST FM Experiments | 1 | Train FM on MNIST with both U-Net and ViT (30 epochs, 10 steps); compare sampling speed and quality vs DDPM |
| **4.4a** | `libs/{fm,ddpm}.py`<br>`use_vae` hooks | 1 | Latent space modeling: encode inputs with `self.vae.encode(x_start)` in `compute_loss`; decode latents with `self.vae.decode(latents)` in `generate` |
| **4.4b** | MiniImageNet Experiments | 3 | Train Latent FM on MiniImageNet with U-Net and ViT (80 epochs, 10 steps); evaluate EMA checkpoints with `eval.py` to record FID |
| **Bonus** | Classifier-Free Guidance (CFG) | +2 | Implement CFG extrapolation $\tilde{v} = v_{\text{uncond}} + s(v_{\text{cond}} - v_{\text{uncond}})$ and evaluate on MiniImageNet |
| **Bonus** | Representation Learning | +2 | Align generative model features with pretrained vision encoder (REPA / REG) |

---

## Role split

- **Ashvin — U-Net & DDPM:** U-Net decoder (`libs/unet.py`) + DDPM loss/sampling (`libs/ddpm.py`) + MNIST DDPM runs + DDPM writeup.
- **Shubhangi — ViT & Flow Matching:** LightningDiT forward pass (`libs/vit.py`) + Flow Matching loss/Euler ODE (`libs/fm.py`) + MNIST FM (ViT) & MiniImageNet FM (ViT) runs + ViT/FM writeup.
- **Akshat — Latent Modeling, Infra & Bonus:** Latent TAESD hooks in `libs/{fm,ddpm}.py` + MNIST FM (UNet) & MiniImageNet FM (UNet) runs + FID evaluation + Classifier-Free Guidance bonus (+2 pts).

Tracks are independent code-wise. Backbones (`unet.py`, `vit.py`) and generative samplers (`ddpm.py`, `fm.py`) can be coded and unit-tested in parallel without blocking each other. Owners can swap tracks if preferred — update this section if you do.

---

## Task dependencies

**All three implementation tracks are independent — nobody is blocked waiting on another person's code.** Everyone can start and finish their implementation in parallel from day one.

| Track | File edited | Depends on another member? |
|-------|-------------|----------------------------|
| Ashvin (U-Net & DDPM) | `code/libs/unet.py`<br>`code/libs/ddpm.py` | No |
| Shubhangi (ViT & FM) | `code/libs/vit.py`<br>`code/libs/fm.py` | No |
| Akshat (Latent & CFG) | `code/libs/ddpm.py`<br>`code/libs/fm.py`<br>`code/libs/guidance.py` | No (hooks into existing methods) |

- DDPM and Flow Matching wrap backbones via `build_arch` in `libs/backbones.py` — neither Ashvin nor Shubhangi is blocked by the other's backbone.
- U-Net and ViT both satisfy the common signature $f_\theta(x, c, t)$, so any generative model can run with either backbone as soon as both are in place.

**Couplings that need coordination (not task dependencies):**
1. **Ashvin and Akshat** edit `libs/ddpm.py`; **Shubhangi and Akshat** edit `libs/fm.py`. Akshat's changes are strictly inside the `if self.use_vae:` blocks of `compute_loss` and `generate`. Use separate feature branches (`feat/unet`, `feat/vit`, `feat/ddpm`, `feat/fm`, `feat/latent`) and merge via PRs.
2. **GPU sharing** and the **MiniImageNet 80-epoch training** are the main scheduling constraints — MiniImageNet is the longest compute job, so launch it as early as possible.

---

## DAY 1 — Setup + core implementation

**Morning (all together)**
1. Set up GPU instances; install dependencies (`pip install -r requirements.txt`). Record any extra packages for writeup.
2. Verify pretrained autoencoder checkpoints in `pretrained/` (`taesd_encoder.pth`, `taesd_decoder.pth`).
3. Verify automatic data downloading for MNIST.
4. Read helper code for your track (`libs/blocks.py`, `libs/config.py`, `libs/models.py`).

**Afternoon/evening (parallel)**
- **Ashvin:** Implement `UNet.forward` decoder in `libs/unet.py`; verify on dummy input `(B, C, H, W)`.
- **Ashvin:** Implement `compute_loss`, `p_sample`, and `generate` in `libs/ddpm.py`.
- **Shubhangi:** Implement `LightningDiT.forward` in `libs/vit.py`; verify on dummy input `(B, C, H, W)`.
- **Shubhangi:** Implement `compute_loss` and `generate` (Euler ODE) in `libs/fm.py`.
- **Akshat:** Add TAESD encoding/decoding hooks for `use_vae` in `libs/ddpm.py` and `libs/fm.py`.
- **Akshat:** Verify dummy forward pass for latent input `(B, 4, 8, 8)`.

**End-of-Day-1 checkpoint — launch MNIST runs:**
- **Ashvin:** Launch `mnist_ddpm_unet.yaml` (30 epochs, 500 steps).
- **Shubhangi:** Launch `mnist_fm_vit.yaml` (30 epochs, 10 steps).
- **Akshat:** Launch `mnist_fm_unet.yaml` (30 epochs, 10 steps).
- Run inside `tmux`; monitor GPU with `watch -n 0.1 nvidia-smi`.

---

## DAY 2 — MNIST verification + launch overnight MiniImageNet

**Morning (parallel)**
- All: Check MNIST training curves in TensorBoard (`tensorboard --logdir=../logs`) and visual grids (`sample-029.png`).
- **Ashvin:** Launch `mnist_ddpm_vit.yaml` (30 epochs, 500 steps) to complete MNIST DDPM comparisons.
- **Akshat & Shubhangi:** Verify MiniImageNet dataset caching via `datasets.py` (`timm/mini-imagenet`).
- Smoke-test 1 epoch of MiniImageNet Latent FM to ensure VAE encoding/decoding is working properly.

**Afternoon → overnight (CRITICAL COMPUTE WINDOW)**
- **Akshat:** Launch full MiniImageNet Latent FM with U-Net (`configs/miniimagenet_fm_unet.yaml`, 80 epochs).
- **Shubhangi:** Launch full MiniImageNet Latent FM with ViT (`configs/miniimagenet_fm_vit.yaml`, 80 epochs).
- Keep jobs running overnight under `tmux`.

**End-of-Day-2 checkpoint:** all code implemented; all MNIST runs complete; both 80-epoch MiniImageNet runs in progress. Start drafting writeup sections.

---

## DAY 3 — Results, FID evaluation, writeup, submit

**Morning**
- All: Verify completion of overnight MiniImageNet runs; pull final checkpoints (`model_final.pth.tar`) and sample grids.
- **Akshat:** Run `eval.py` on MiniImageNet checkpoints to compute official FID scores:
  ```bash
  python eval.py configs/miniimagenet_fm_unet.yaml ../logs/<unet_exp>/epoch_079.pth.tar
  python eval.py configs/miniimagenet_fm_vit.yaml ../logs/<vit_exp>/epoch_079.pth.tar
  ```
- **Akshat (Bonus):** Test Classifier-Free Guidance (`test_cfg.cfg=True`, `cfg_scale=2.5`), produce side-by-side grids with/without guidance, and evaluate FID improvement (+2 pts).

**Afternoon — writeup (each owns assigned sections; one assembles)**
- **Ashvin:** Write U-Net architecture, DDPM derivation, and MNIST DDPM vs FM comparison section.
- **Shubhangi:** Write ViT/DiT architecture, Flow Matching derivation, and MiniImageNet ViT analysis section.
- **Akshat:** Write Latent modeling analysis, MiniImageNet U-Net results, FID comparison table, and CFG bonus section.
- Clearly identify each member's contributions in the report.

**Late afternoon — package & submit**
- Copy required configs, training logs, and final `model_final.pth.tar` checkpoints into `results/` (NOT `data/`, `logs/`, `models/`, or intermediate epoch checkpoints).
- Verify relative paths everywhere.
- Package the submission:
  ```bash
  python zip_submission.py
  ```
- All three members submit the same PDF and zip archive.

---

## Risk notes

- **MiniImageNet 80 epochs is the longest compute job** — protect the overnight window between Day 2 and Day 3.
- Do MNIST first as a fast correctness gate (~1 hour per run) before committing compute to MiniImageNet.
- Flow Matching with 10 Euler steps is significantly faster to sample than DDPM with 500 steps; use this speedup during debugging.
- Shut down idle GPU instances when not training.

---

## Git workflow (branch protection is ON)

`main` is protected — **no direct pushes**. Every change goes through a PR:

1. Branch off `main`: `git checkout -b feat/<thing>` (e.g. `feat/unet`, `feat/vit`, `feat/ddpm`, `feat/fm`, `feat/latent`, `feat/cfg`).
2. Commit and push the branch: `git push -u origin feat/<thing>`.
3. Open a PR on GitHub.
4. **Get one of the other members to approve.** You cannot approve your own PR.
5. Merge after approval.

Do **not** commit `data/`, `logs/`, `models/`, or loose intermediate `epoch_*.pth`/`*.pth.tar`.
**Exception:** everything under `results/` *is* tracked — that's the submission payload (configs, logs, and final checkpoints).
