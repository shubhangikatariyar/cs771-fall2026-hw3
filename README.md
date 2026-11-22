# CS 771 Fall 2026 — Homework 3: Image Generation with Diffusion and Flow Matching Models

PyTorch implementation for **BMI/CS 771 HW3**. This project explores two prominent families of deep generative models: **Denoising Diffusion Probabilistic Models (DDPMs)** and **Flow Matching (FM) with Rectified Flow**. We implement training objectives and sampling algorithms for both models, coupled with two prediction backbones: convolutional **U-Net** and Transformer-based **Vision Transformer (LightningDiT)**, for class-conditional image synthesis on **MNIST** (pixel space) and **MiniImageNet** (latent space via pretrained TAESD autoencoder).

- **Points:** 12 core points + up to 4 bonus points
- **Team:** Ashvin, Akshat, Shubhangi (GitHub: `shubhangikatariyar`)
- **Plan & Task Assignment:** see [PLAN.md](PLAN.md)
- **Report & Writeup:** see [writeup/report.md](writeup/report.md)

---

## 1. What Needs to Be Done

All core implementation is strictly confined to **four files** in `code/libs/`:

### Summary of Implementation Tasks

| # | File / Class / Function | Pts | Task Description |
|---|---|:---:|---|
| **4.1a** | `code/libs/unet.py`<br>`UNet.forward` | **1** | **U-Net Decoder:** Implement reverse pass matching encoder feature maps; apply residual blocks (conditioned on time $t$), cross-attention transformer blocks (conditioned on class embedding $c$), and upsamplers with skip concatenation. |
| **4.1b** | `code/libs/vit.py`<br>`LightningDiT.forward` | **1** | **ViT Forward Pass:** Implement DiT transformer backbone; add learnable 2D positional embeddings to patch tokens, pass through stacked DiT blocks with **adaLN-Zero** conditioning on time and class, apply final adaptive norm and linear unpatchify. |
| **4.2a** | `code/libs/ddpm.py`<br>`compute_loss` | **1** | **DDPM Training Loss:** Sample timestep $t \sim \mathcal{U}\{0, T-1\}$ and Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$; diffuse input via `q_sample(x_start, t, noise)`; compute simplified MSE loss $\|\epsilon - \epsilon_\theta(x_t, c, t)\|_2^2$ (Algorithm 1, line 5). |
| **4.2b** | `code/libs/ddpm.py`<br>`p_sample` & `generate` | **1** | **DDPM Sampling:** In `p_sample`, compute posterior mean $\mu_\theta(x_t, t)$ using predicted noise (Equation 11) and add variance $\sigma_t z$ for $t > 0$. In `generate`, run the full reverse denoising chain from $t = T-1$ down to $0$ (Algorithm 2). |
| **4.2c** | MNIST Experiments | **1** | Train DDPM on MNIST with both U-Net and ViT backbones (30 epochs, 500 steps). Record loss curves, sample grids, and compare visual fidelity and generation speed. |
| **4.3a** | `code/libs/fm.py`<br>`compute_loss` | **1** | **Rectified Flow Loss:** Sample continuous time $t \sim \mathcal{U}[0, 1]$ and noise $z \sim \mathcal{N}(0, I)$; construct linear interpolation path $x_t = (1-t)z + t x_{\text{data}}$; compute velocity MSE loss $\|f_\theta(x_t, c, t) - (x_{\text{data}} - z)\|_2^2$. |
| **4.3b** | `code/libs/fm.py`<br>`generate` | **1** | **Flow Matching Sampling:** Implement forward Euler ODE solver: start from Gaussian noise $y(0) \sim \mathcal{N}(0, I)$ and step $y(t + \Delta t) \approx y(t) + \Delta t f_\theta(y(t), c, t)$ from $t = 0$ to $t = 1$ with step size $\Delta t = 1.0 / \text{timesteps}$. |
| **4.3c** | MNIST Experiments | **1** | Train FM on MNIST with both U-Net and ViT backbones (30 epochs, 10 steps). Compare sampling efficiency (10 steps vs 500 steps) and generation quality with DDPM. |
| **4.4a** | `code/libs/{fm,ddpm}.py`<br>`use_vae` Latent Hooks | **1** | **Latent Modeling:** In `compute_loss`, when `use_vae=True`, encode input images to latent space via `self.vae.encode(x_start)`. In `generate`, decode generated latents to pixel space via `self.vae.decode(latents)` before clipping/postprocessing. |
| **4.4b** | MiniImageNet Experiments | **3** | Train Latent FM on MiniImageNet (80 epochs, 10 steps) with both U-Net and ViT. Evaluate final EMA checkpoints using `eval.py` to record official FID scores. |
| **4.5** | **Bonus (+4 max)** | **+4** | **Classifier-Free Guidance (CFG, +2 pts):** Implement extrapolation $\tilde{v} = v_{\text{uncond}} + s(v_{\text{cond}} - v_{\text{uncond}})$ and evaluate on MiniImageNet.<br>**Representation Learning (REPA/REG, +2 pts):** Pretrained vision encoder feature alignment. |

---

## 2. Repository Structure

```
.
├── code/
│   ├── configs/                 # YAML configuration files
│   │   ├── mnist_ddpm_unet.yaml       # MNIST DDPM with U-Net (30 ep, 500 steps)
│   │   ├── mnist_ddpm_vit.yaml        # MNIST DDPM with ViT (30 ep, 500 steps)
│   │   ├── mnist_fm_unet.yaml         # MNIST FM with U-Net (30 ep, 10 steps)
│   │   ├── mnist_fm_vit.yaml          # MNIST FM with ViT (30 ep, 10 steps)
│   │   ├── miniimagenet_fm_unet.yaml  # MiniImageNet Latent FM with U-Net (80 ep)
│   │   └── miniimagenet_fm_vit.yaml   # MiniImageNet Latent FM with ViT (80 ep)
│   ├── libs/
│   │   ├── unet.py              # U-Net backbone (EDIT: decoder in UNet.forward)
│   │   ├── vit.py               # Vision Transformer (EDIT: LightningDiT.forward)
│   │   ├── ddpm.py              # DDPM model (EDIT: compute_loss, p_sample, generate, VAE hooks)
│   │   ├── fm.py                # Flow Matching (EDIT: compute_loss, generate, VAE hooks)
│   │   ├── blocks.py            # ResNet, attention, and DiT building blocks
│   │   ├── backbones.py         # Backbone factory (build_arch)
│   │   ├── models.py            # Generative model factory (build_model)
│   │   ├── tiny_autoencoder.py  # TAESD autoencoder for latent modeling
│   │   ├── datasets.py          # MNIST and MiniImageNet loaders
│   │   ├── guidance.py          # Classifier-Free Guidance utilities
│   │   ├── fid_score.py         # Inception-v3 FID evaluation
│   │   ├── inception.py         # Inception network wrapper
│   │   ├── config.py            # Architecture and training defaults
│   │   └── utils.py             # Logging, EMA, meters, and visualization helpers
│   ├── train.py                 # Main training script
│   └── eval.py                  # Evaluation script (samples generation & FID)
├── data/                        # Datasets directory (auto-downloaded, NOT committed)
├── logs/                        # Training checkpoints and tensorboard logs (gitignored)
├── pretrained/                  # TAESD checkpoints (taesd_encoder.pth, taesd_decoder.pth)
├── results/                     # Tracked final submission deliverables (configs, logs, model_final)
├── writeup/                     # Report LaTeX/Markdown sources and final report.pdf
├── PLAN.md                      # Team work division, schedule, and dependencies
├── README.md                    # Project documentation (this file)
├── requirements.txt             # Environment dependencies
├── reproduce.sh                 # End-to-end reproduction script
└── zip_submission.py            # Submission packaging utility
```

---

## 3. Environment Setup

Install dependencies listed in `requirements.txt`:

```bash
# Clone the repository
git clone https://github.com/shubhangikatariyar/cs771-fall2026-hw3.git
cd cs771-fall2026-hw3

# Create and activate virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Device Selection
By default, configurations run on CUDA: `devices: ["cuda:0"]`.
- For CPU debugging: override `devices: ["cpu"]`.
- For Apple Silicon Mac: override `devices: ["mps"]`.

---

## 4. Running Experiments

All commands below are executed from within the `code/` directory:

```bash
cd code
```

### 4.1 MNIST Experiments (Pixel Space)

#### Denoising Diffusion Probabilistic Models (DDPM)
```bash
# Train DDPM with U-Net (30 epochs, 500 diffusion steps)
python train.py ./configs/mnist_ddpm_unet.yaml

# Train DDPM with ViT (30 epochs, 500 diffusion steps)
python train.py ./configs/mnist_ddpm_vit.yaml
```

#### Flow Matching (Rectified Flow)
```bash
# Train FM with U-Net (30 epochs, 10 Euler integration steps)
python train.py ./configs/mnist_fm_unet.yaml

# Train FM with ViT (30 epochs, 10 Euler integration steps)
python train.py ./configs/mnist_fm_vit.yaml
```

### 4.2 MiniImageNet Experiments (Latent Space)

Latent Flow Matching models compress $3 \times 64 \times 64$ images into $4 \times 8 \times 8$ latent representations using the pretrained TAESD autoencoder:

```bash
# Train Latent FM with U-Net (80 epochs, 10 integration steps)
python train.py ./configs/miniimagenet_fm_unet.yaml

# Train Latent FM with ViT (80 epochs, 10 integration steps)
python train.py ./configs/miniimagenet_fm_vit.yaml
```

### 4.3 Evaluation & FID Score Calculation

Evaluation automatically loads the Exponential Moving Average (EMA) checkpoint, generates conditional samples, saves sample grids, and calculates the Fréchet Inception Distance (FID):

```bash
# Evaluate a checkpoint (e.g. final model checkpoint or experiment directory)
python eval.py ./configs/miniimagenet_fm_unet.yaml ../logs/<exp_dir>/epoch_079.pth.tar

# Evaluate ViT checkpoint
python eval.py ./configs/miniimagenet_fm_vit.yaml ../logs/<exp_dir>/epoch_079.pth.tar
```

### 4.4 Resuming Training
To resume training from an intermediate checkpoint:
```bash
python train.py ./configs/miniimagenet_fm_vit.yaml --resume ../logs/<exp_dir>/epoch_040.pth.tar
```

### 4.5 Monitoring with TensorBoard
```bash
# From repo root or code/
tensorboard --logdir=../logs
```
Training generates sample grids named `sample-<epoch>.png` in the experiment directory every 5 epochs (and at the final epoch).

---

## 5. Team Responsibilities & Track Ownership

| Team Member | Core Focus & Code Responsibilities | Assigned Experiments | Writeup Ownership |
|---|---|---|---|
| **Shubhangi** | • `libs/vit.py` (`LightningDiT.forward`)<br>• `libs/fm.py` (`compute_loss`, `generate`) | • MNIST FM (ViT)<br>• MiniImageNet FM (ViT) | • ViT / DiT Architecture & adaLN-Zero<br>• Flow Matching formulation & Euler ODE<br>• MiniImageNet ViT analysis |
| **Ashvin** | • `libs/unet.py` (`UNet.forward` decoder)<br>• `libs/ddpm.py` (`compute_loss`, `p_sample`, `generate`) | • MNIST DDPM (UNet)<br>• MNIST DDPM (ViT) | • U-Net Architecture & cross-attention<br>• DDPM derivation & sampling algorithms<br>• DDPM vs FM comparison |
| **Akshat** | • Latent VAE hooks in `libs/{fm,ddpm}.py`<br>• Repo management, scripts & environment<br>• Bonus: Classifier-Free Guidance (CFG) | • MNIST FM (UNet)<br>• MiniImageNet FM (UNet)<br>• FID evaluation (`eval.py`) | • Latent space vs pixel space analysis<br>• MiniImageNet U-Net results & FID<br>• CFG Bonus evaluation & report assembly |

---

## 6. Git Workflow

`main` is protected — **no direct commits to `main`**.

1. Create a feature branch:
   ```bash
   git checkout -b feat/<feature-name>
   # Examples: feat/vit-backbone, feat/unet-backbone, feat/ddpm, feat/fm, feat/latent-vae
   ```
2. Commit and push your changes:
   ```bash
   git add .
   git commit -m "feat(module): descriptive message"
   git push -u origin feat/<feature-name>
   ```
3. Open a Pull Request on GitHub and assign teammates as reviewers.
4. Require at least **one peer review and approval** before merging into `main`.
5. Merge via squash-merge or standard merge.

---

## 7. Submission Checklist

Before submitting on Canvas:
1. Ensure all code uses **relative paths** only.
2. Copy required configuration files, training logs, and final checkpoints (`model_final.pth.tar`) into `results/`.
3. Confirm that **NO** datasets (`data/`), raw logs (`logs/`), or intermediate epoch checkpoints are tracked in the submission.
4. Place the final report PDF into `writeup/report.pdf`.
5. Package the submission archive:
   ```bash
   python zip_submission.py
   ```
6. Verify the contents of `submission.zip`:
   - `code/`
   - `writeup/`
   - `results/`
7. **Every team member must submit the same `submission.zip` and `report.pdf` on Canvas.**
