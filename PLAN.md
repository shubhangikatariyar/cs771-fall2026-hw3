# Assignment 3 — Team Execution Plan & Work Division

**Course:** BMI / CS 771 Fall 2026: Homework Assignment 3 — Image Generation with Diffusion and Flow Matching Models  
**Team:** Ashvin, Akshat, Shubhangi (GitHub: `shubhangikatariyar`)  
**Due Date:** Nov 25, 2026 (midnight) | **Extended Update Window:** Nov 30, 2026 (midnight)  
**Total Points:** 12 Points Core + up to 4 Bonus Points  
**Repository:** [https://github.com/shubhangikatariyar/cs771-fall2026-hw3](https://github.com/shubhangikatariyar/cs771-fall2026-hw3)

---

## 1. Executive Summary & Grading Breakdown

This assignment investigates two deep generative modeling paradigms: **Denoising Diffusion Probabilistic Models (DDPM)** and **Flow Matching (FM) with Rectified Flow**. We implement their training objectives, sampling algorithms, and neural backbones (both a convolutional **U-Net** and a Transformer-based **Vision Transformer / LightningDiT**), experimenting on **MNIST** (in pixel space) and **MiniImageNet** (in the latent space of a pretrained **TAESD** autoencoder).

### Score Distribution

| Section | Topic | Points | Target Files / Scope |
|---|---|:---:|---|
| **4.1** | **Prediction Networks** | **2 pts** | `libs/unet.py` (1 pt) & `libs/vit.py` (1 pt) |
| **4.2** | **DDPM** | **3 pts** | `libs/ddpm.py` (2 pts: `p_sample`, `generate`, `compute_loss`) + MNIST Experiments (1 pt) |
| **4.3** | **Flow Matching (Rectified Flow)** | **3 pts** | `libs/fm.py` (2 pts: `compute_loss`, `generate`) + MNIST Experiments (1 pt) |
| **4.4** | **Latent Modeling & MiniImageNet** | **4 pts** | Latent space TAESD integration (1 pt) + MiniImageNet FM experiments & FID (3 pts) |
| **4.5** | **Bonus** | **up to +4 pts** | Classifier-Free Guidance (CFG, +2 pts) / Representation Learning (REPA/REG, +2 pts) |
| **5.0** | **Writeup** | **Graded** | Joint PDF report with figures, FID, sample grids, and individual contributions |

---

## 2. Team Work Division & Track Ownership

Taking inspiration from our successful division in HW2 (where Shubhangi owned the ViT/Transformer track, Ashvin owned the CNN/ResNet track, and Akshat owned detection/loss/repo infra), we divide HW3 into three complementary, parallel, and well-balanced tracks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRACK ASSIGNMENT OVERVIEW                             │
├──────────────────────┬──────────────────────────────┬───────────────────────┤
│ SHUBHANGI            │ ASHVIN                       │ AKSHAT                │
│ (ViT & Flow Matching)│ (U-Net & DDPM)               │ (Latent, Infra & CFG) │
├──────────────────────┼──────────────────────────────┼───────────────────────┤
│ • libs/vit.py        │ • libs/unet.py               │ • libs/{fm,ddpm}.py   │
│   LightningDiT       │   UNet decoder               │   use_vae TAESD hooks │
│ • libs/fm.py         │ • libs/ddpm.py               │ • Repo & Git setup    │
│   compute_loss       │   compute_loss               │   PR & Branch protect │
│   generate (Euler)   │   p_sample & generate        │ • libs/guidance.py    │
│ • MNIST FM (ViT)     │ • MNIST DDPM (UNet & ViT)    │   CFG Bonus (+2 pts)  │
│ • MiniImageNet FM    │ • DDPM vs FM comparison      │ • MNIST FM (UNet)     │
│   (ViT run, 80 ep)   │ • Variance schedule writeup  │ • MiniImageNet FM     │
│ • ViT & FM Writeup   │ • UNet & DDPM Writeup        │   (UNet run, 80 ep)   │
│                      │                              │ • FID eval & Report   │
└──────────────────────┴──────────────────────────────┴───────────────────────┘
```

### Detailed Responsibilities

#### 1. Shubhangi — ViT Backbone & Flow Matching Lead
- **Code:**
  - `code/libs/vit.py`: Implement the missing forward pass of `LightningDiT`. Process patch tokens with learnable 2D positional embeddings, apply stacked DiT blocks with **adaLN-Zero** conditioning on diffusion time $t$ and class label $y$, and finish with adaptive layer norm, linear projection, and reshape to spatial feature map.
  - `code/libs/fm.py`:
    - `compute_loss`: Implement the Rectified Flow regression loss: sample $t \sim \mathcal{U}[0, 1]$, noise $z \sim \mathcal{N}(0, I)$, define straight path $x_t = (1-t)z + t x_{\text{data}}$, and calculate $\mathcal{L}_{FM} = \|f_\theta(x_t, c, t) - (x_{\text{data}} - z)\|_2^2$.
    - `generate`: Implement forward Euler ODE integration: $y(t + \Delta t) \approx y(t) + \Delta t f_\theta(y(t), c, t)$ stepping from $t = 0$ (noise) to $t = 1$ (image) with $\Delta t = 1.0 / \text{timesteps}$.
- **Experiments:**
  - Train FM on MNIST with ViT (`configs/mnist_fm_vit.yaml`, 30 epochs, 10 steps).
  - Train Latent FM on MiniImageNet with ViT (`configs/miniimagenet_fm_vit.yaml`, 80 epochs, 10 steps).
- **Writeup:**
  - ViT/LightningDiT architecture, patchification, and adaLN-Zero conditioning mechanisms.
  - Flow Matching formulation, straight velocity interpolation, and Euler ODE sampling principles.
  - MiniImageNet ViT training curves, sample grids, and generation quality analysis.

#### 2. Ashvin — U-Net Backbone & DDPM Lead
- **Code:**
  - `code/libs/unet.py`: Implement the decoder of `UNet.forward`. Unpack cached encoder outputs in reverse order, pass through symmetric stages composed of residual blocks (conditioned on time $t$), cross-attention transformer blocks (conditioned on class embedding $c$), and upsampling modules.
  - `code/libs/ddpm.py`:
    - `compute_loss`: Implement DDPM training objective (Algorithm 1, line 5): sample random discrete timestep $t \sim \mathcal{U}\{0, T-1\}$, sample Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$, compute $x_t$ via `q_sample`, and compute MSE loss $\|\epsilon - \epsilon_\theta(x_t, c, t)\|_2^2$.
    - `p_sample`: Implement Equation 11 single reverse denoising step: calculate predicted posterior mean $\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, c, t)\right)$, and add posterior variance $\sigma_t z$ for $t > 0$.
    - `generate`: Implement Algorithm 2 reverse sampling loop from $t = T-1$ down to $0$.
- **Experiments:**
  - Train DDPM on MNIST with U-Net (`configs/mnist_ddpm_unet.yaml`, 30 epochs, 500 steps).
  - Train DDPM on MNIST with ViT (`configs/mnist_ddpm_vit.yaml`, 30 epochs, 500 steps).
- **Writeup:**
  - U-Net decoder architecture, skip connection concatenation, and time/class conditioning injection.
  - DDPM theoretical foundations: forward Markov chain, reparameterization trick, reverse posterior derivation.
  - Comparative analysis on MNIST: DDPM vs FM (500 steps vs 10 steps sampling speed, FID/visual quality comparison).

#### 3. Akshat — Latent Modeling, Infra, MiniImageNet U-Net & Bonus Lead
- **Code & Infra:**
  - Repository management under `shubhangikatariyar/cs771-fall2026-hw3`, branch rules, environment check.
  - Latent modeling integration (`use_vae` support in `code/libs/ddpm.py` and `code/libs/fm.py`):
    - In `compute_loss`: if `self.use_vae` is True, encode input RGB images into latent space via `self.vae.encode(x_start)` before applying noise/flow.
    - In `generate`: if `self.use_vae` is True, decode sampled latent tensors back to image space via `self.vae.decode(latents)` before pixel postprocessing/clamping.
  - **Bonus (+2 pts)**: Classifier-Free Guidance (CFG):
    - Implement CFG extrapolation $\tilde{v} = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$ in `fm.py` and `ddpm.py`.
    - Experiment with guidance scales $s \in [1.0, 2.5, 4.0]$ and visualize class adherence vs mode coverage.
- **Experiments:**
  - Train FM on MNIST with U-Net (`configs/mnist_fm_unet.yaml`, 30 epochs, 10 steps).
  - Train Latent FM on MiniImageNet with U-Net (`configs/miniimagenet_fm_unet.yaml`, 80 epochs, 10 steps).
  - Run evaluation (`eval.py`) across all checkpoints to obtain official FID scores.
- **Writeup:**
  - Latent space vs pixel space modeling: compression ratio (8x downsampling, 4 latent channels), computational efficiency, and boundary artifacts.
  - MiniImageNet U-Net results, training dynamics, sample grids, and FID benchmark against ViT.
  - Classifier-Free Guidance (CFG) analysis: qualitative samples across scale sweeps and FID tradeoffs.
  - Final report compilation and Canvas zip packaging verification.

---

## 3. Task Dependencies & Parallel Workflow

**All three implementation tracks are independent and can proceed simultaneously from Day 1:**

```
                  ┌────────────────────────────────────────┐
                  │          DAY 1: IMPLEMENTATION         │
                  └────────────────────────────────────────┘
                        /             |             \
                       /              |              \
             [Shubhangi]          [Ashvin]          [Akshat]
            • libs/vit.py       • libs/unet.py    • TAESD latent hooks
            • libs/fm.py        • libs/ddpm.py    • libs/guidance.py (CFG)
                  \                   |                   /
                   \                  |                  /
                  ┌────────────────────────────────────────┐
                  │       DAY 2: MNIST RUNS & VERIFY       │
                  └────────────────────────────────────────┘
                        /             |             \
             [Shubhangi]          [Ashvin]          [Akshat]
             MNIST FM (ViT)      MNIST DDPM        MNIST FM (UNet)
                                 (UNet & ViT)
                  \                   |                   /
                   \                  |                  /
                  ┌────────────────────────────────────────┐
                  │   DAYS 2-3: MINIIMAGENET (80 EPOCHS)   │
                  └────────────────────────────────────────┘
                        /                           \
             [Shubhangi]                             [Akshat]
             MiniImageNet FM (ViT)                   MiniImageNet FM (UNet)
                  \                                 /
                   \                               /
                  ┌────────────────────────────────────────┐
                  │   DAY 4: EVALUATION, WRITEUP & SUBMIT  │
                  └────────────────────────────────────────┘
             • eval.py FID computation across all checkpoints
             • Generate comparison tables & figure grids
             • Assemble writeup/report.pdf & run zip_submission.py
```

### Critical Path & Risk Management
1. **Longest Compute Jobs:** MiniImageNet training runs for 80 epochs. These must be launched by Day 2 evening at the latest to allow ample time for evaluation and FID computation.
2. **GPU Allocation:** Each team member should run jobs in `tmux` on a GPU instance (T4 or better). If GPUs are constrained, prioritize MNIST first (takes ~1-1.5 hrs), followed immediately by MiniImageNet.
3. **No Cross-Blocking:** Because `unet.py`, `vit.py`, `ddpm.py`, and `fm.py` are modular, bugs in DDPM do not block FM testing, and ViT does not block U-Net.

---

## 4. Day-by-Day Timeline

### Day 1: Setup, Model Implementation & Unit Verification
- [ ] **All:** Clone repo, set up python environment (`pip install -r requirements.txt`), verify GPU availability.
- [ ] **Akshat:** Ensure GitHub repo settings, branch rules, and initial commit are clean.
- [ ] **Shubhangi:** Implement `LightningDiT.forward` in `libs/vit.py`. Run dummy forward pass shape check.
- [ ] **Shubhangi:** Implement `compute_loss` and `generate` (Euler ODE) in `libs/fm.py`.
- [ ] **Ashvin:** Implement `UNet.forward` decoder in `libs/unet.py`. Run dummy forward pass shape check.
- [ ] **Ashvin:** Implement `compute_loss`, `p_sample`, and `generate` in `libs/ddpm.py`.
- [ ] **Akshat:** Implement `use_vae` encoding/decoding hooks in `libs/fm.py` and `libs/ddpm.py`.
- [ ] **Checkpoint:** PRs created (`feat/vit`, `feat/unet`, `feat/ddpm`, `feat/fm`, `feat/latent`) and peer-reviewed.

### Day 2: MNIST Experiments & Launch Overnight MiniImageNet
- [ ] **Ashvin:** Train `mnist_ddpm_unet.yaml` (30 epochs, 500 steps).
- [ ] **Ashvin:** Train `mnist_ddpm_vit.yaml` (30 epochs, 500 steps).
- [ ] **Shubhangi:** Train `mnist_fm_vit.yaml` (30 epochs, 10 steps).
- [ ] **Akshat:** Train `mnist_fm_unet.yaml` (30 epochs, 10 steps).
- [ ] **All:** Verify MNIST sample grids in `logs/` (digits should be crisp and class-consistent).
- [ ] **Overnight Launches (CRITICAL):**
  - **Akshat:** Launch `miniimagenet_fm_unet.yaml` (80 epochs, 10 steps).
  - **Shubhangi:** Launch `miniimagenet_fm_vit.yaml` (80 epochs, 10 steps).

### Day 3: MiniImageNet Completion, FID Evaluation & CFG Bonus
- [ ] **Akshat & Shubhangi:** Monitor MiniImageNet training completion and pull `model_final.pth.tar`.
- [ ] **Akshat:** Run `eval.py` to generate evaluation samples and compute FID scores on both MiniImageNet models.
- [ ] **Akshat:** Test CFG bonus (`test_cfg.cfg=True`, `cfg_scale=2.5`), produce side-by-side comparison grids with and without guidance.
- [ ] **Ashvin & Shubhangi:** Draft sections for report: DDPM vs FM comparison, U-Net vs ViT architectures, loss curves.

### Day 4: Final Writeup Assembly & Submission
- [ ] **All:** Assemble `writeup/report.pdf` covering all required questions and figures.
- [ ] **All:** Check contributions section to ensure all team members' efforts are clearly stated.
- [ ] **Akshat:** Verify `results/` folder contains required logs, configs, and final checkpoints (`model_final.pth.tar`). Ensure NO intermediate epoch checkpoints or datasets are tracked.
- [ ] **Akshat:** Run `python zip_submission.py` to create `submission.zip`. Inspect archive contents.
- [ ] **All:** Submit identical `submission.zip` and PDF report to Canvas before deadline.

---

## 5. Git Workflow & Collaboration Rules

- **Branch Protection:** `main` is protected. No direct commits to `main`.
- **Branch Naming:**
  - `feat/vit-backbone` (Shubhangi)
  - `feat/fm-rectified` (Shubhangi)
  - `feat/unet-backbone` (Ashvin)
  - `feat/ddpm-diffusion` (Ashvin)
  - `feat/latent-vae` (Akshat)
  - `feat/cfg-bonus` (Akshat)
  - `docs/report` (All)
- **Code Review:** Every PR requires at least **one peer review and approval** prior to merging.
- **Clean Commits:** Do NOT commit `data/`, `logs/`, or intermediate checkpoints `epoch_*.pth.tar`. Only final model weights in `results/` should be tracked.
