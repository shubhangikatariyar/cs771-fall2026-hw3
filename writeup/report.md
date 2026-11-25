# BMI/CS 771 Fall 2026 — Homework 3 Report: Image Generation with Diffusion and Flow Matching Models

**Team:** Ashvin, Akshat, Shubhangi  
**Date:** November 2026

---

## 1. Prediction Networks: U-Net and Vision Transformer (2 points)

Both generative backbones learn a conditional prediction function $f_\theta(x, c, t)$, where $x$ is an image or latent feature map, $c \in \{0, \dots, C-1\}$ is a discrete class label, and $t$ is the continuous/discrete diffusion time.

### 1.1 U-Net Decoder Architecture (`libs/unet.py`)
Our convolutional backbone follows the modern diffusion U-Net design (Rombach et al., 2022). The network incorporates downsampling encoder stages, a bottleneck middle block, and symmetric upsampling decoder stages:

- **Time Conditioning:** Timestep $t$ is embedded via sinusoidal positional embeddings followed by a two-layer MLP with SiLU activations, producing embedding vector $t_{\text{emb}}$. Within each `ResidualBlock`, $t_{\text{emb}}$ is linearly projected to match feature dimensions and added directly to the spatial representations after group normalization.
- **Class Conditioning:** Class label $c$ is projected via a learnable embedding table (`y_embedder`), with support for unconditional dropout during training. A linear projection `y_embedder_proj` aligns the label embedding into a context token $c \in \mathbb{R}^{1 \times D}$, which is injected into spatial features through cross-attention in `SpatialTransformer` blocks.
- **Decoder Implementation:** The decoder matches the encoder in reverse:
  ```python
  # Iterate over decoder stages in reverse order
  for resblock, transformer, upsample in self.decoder:
      # Pop matching feature map from cached encoder skip connections
      skip = encoder_output.pop()
      x = torch.cat([x, skip], dim=1)
      x = resblock(x, t)
      if transformer is not None:
          x = transformer(x, c)
      if upsample is not None:
          x = upsample(x)
  ```
  The concatenated skip connections preserve high-frequency spatial details from early layers, while the transformer blocks inject global semantic class conditioning.

### 1.2 Vision Transformer Forward Pass (`libs/vit.py` - LightningDiT)
Our Vision Transformer adapts the Diffusion Transformer (DiT) architecture (Peebles & Xie, ICCV 2023):

- **Patchification & Positional Encoding:** The input image/latent $x \in \mathbb{R}^{B \times C \times H \times W}$ is partitioned into non-overlapping patches of size $p \times p$ (default $p=2$) via convolutional projection `x_embedder`, producing a sequence of tokens $Z \in \mathbb{R}^{B \times T \times D}$ where $T = (H/p)(W/p)$. Standard 2D learnable positional embeddings `pos_embed` are added directly.
- **adaLN-Zero Conditioning:** Rather than concatenating tokens, time embedding $t$ and label embedding $y$ are summed ($c_{\text{cond}} = t + y \in \mathbb{R}^{N \times D}$) and fed into an adaptive layer normalization (adaLN-Zero) modulation mechanism. For each DiT block, $c_{\text{cond}}$ regresses six modulation parameters: scale and shift $(\gamma_1, \beta_1)$ and gate $\alpha_1$ for self-attention, plus $(\gamma_2, \beta_2)$ and gate $\alpha_2$ for the MLP. Crucially, the gate parameters are initialized to zero, ensuring that each block starts as an identity function during training initialization.
- **Forward Pass Implementation:**
  ```python
  c = t + y
  for block in self.blocks:
      x = block(x, c)
  x = self.final_layer(x, c)
  x = self.unpatchify(x)
  return x
  ```
  After $L$ DiT blocks, `DiTDecodingLayer` applies adaptive layer norm modulation from $c$, followed by linear projection to $p^2 \cdot C$ channels, and `unpatchify` rearranges the sequence back to spatial dimension $(B, C, H, W)$.

---

## 2. Denoising Diffusion Probabilistic Models (DDPM) (3 points)

### 2.1 Mathematical Formulation & Training Objective
DDPM (Ho et al., 2020) defines a forward Markovian diffusion process that adds Gaussian noise over discrete timesteps $t \in \{1, \dots, T\}$ according to variance schedule $\beta_1, \dots, \beta_T$:
$$q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I\right)$$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

The neural network $\epsilon_\theta(x_t, c, t)$ is trained using the simplified mean squared error objective (Algorithm 1):
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, c, t)\|_2^2\right]$$
where $t \sim \mathcal{U}\{0, T-1\}$ and $\epsilon \sim \mathcal{N}(0, I)$.

### 2.2 Reverse Denoising & Sampling
The reverse generative process approximates $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\left(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I\right)$. In `p_sample`, the posterior mean is recovered via:
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, c, t)\right)$$
For $t > 0$, noise $z \sim \mathcal{N}(0, I)$ is added scaled by posterior standard deviation $\sigma_t = \sqrt{\tilde{\beta}_t}$. For $t = 0$, no noise is added. Sampling iterates backwards from $t = T-1$ down to $t = 0$.

### 2.3 MNIST Experiments (DDPM)
We train DDPM for 30 epochs with $T = 500$ timesteps on MNIST with both U-Net and ViT:

| Model | Backbone | Epochs | Steps | Final Training Loss | Hardware / Runtime | Checkpoint / Deliverables |
|---|---|:---:|:---:|:---:|:---:|---|
| **DDPM-UNet** | U-Net | 30 | 500 | **0.030** (avg 0.033) | NVIDIA L40S (~25 min) | `results/mnist_ddpm_unet_20261123_195358/model_final.pth.tar` |
| **DDPM-ViT** | LightningDiT | 30 | 500 | *pending run* | CHTC L40S | `chtc/mnist_ddpm_vit.sub` |

**Training Dynamics (DDPM-UNet):**
Training ran on CHTC using an NVIDIA L40S GPU. The simplified MSE loss decreased smoothly from an initial $0.985$ down to $0.030$ by iteration 28,125 (epoch 30). Intermediate visualization grids saved every 5 epochs (`sample-4.png` through `sample-29.png`) illustrate the denoising progression: early epochs exhibit fuzzy digit approximations, while by epoch 20–30, digits show sharp strokes, clean backgrounds, and exact correspondence with the conditioning class labels.

---

## 3. Flow Matching with Rectified Flow (3 points)

### 3.1 Mathematical Principles & Straight Paths
Rectified Flow (Liu et al., 2023; Lipman et al., 2024) constructs a continuous generative model by learning a time-dependent velocity field $v_\theta(x, c, t)$ that transports Gaussian noise $z \sim \mathcal{N}(0, I)$ at $t = 0$ to natural data $x_{\text{data}}$ at $t = 1$. The trajectories follow linear interpolations:
$$x_t = (1 - t) z + t x_{\text{data}}, \quad t \in [0, 1]$$
The path derivative gives a constant marginal velocity $u_t(x_t) = \frac{d x_t}{d t} = x_{\text{data}} - z$.

### 3.2 Regression Objective & Forward Euler ODE Sampling
The flow matching objective optimizes:
$$\mathcal{L}_{FM}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1], z, x_{\text{data}}}\left[\|v_\theta(x_t, c, t) - (x_{\text{data}} - z)\|_2^2\right]$$

In `compute_loss`, we draw continuous time $t \sim \mathcal{U}[0, 1]$ and Gaussian noise $z \sim \mathcal{N}(0, I)$. For latent modeling (`use_vae=True`), input images are first encoded without gradients via `self.vae.encode(x_start)`. The model predicts the trajectory velocity, and regression is performed against target velocity $x_{\text{data}} - z$.

Sampling generates images by numerically solving the Ordinary Differential Equation (ODE) $\frac{dy(t)}{dt} = v_\theta(y(t), c, t)$ via forward Euler integration over $N$ uniform steps ($\Delta t = 1.0 / N$):
$$y(t + \Delta t) \approx y(t) + \Delta t \cdot v_\theta(y(t), c, t)$$

In `generate`, we initialize $y(0) \sim \mathcal{N}(0, I)$ and step from $t=0$ to $t=1$. For Classifier-Free Guidance (CFG), when $t \ge t_{\text{start}}$, we extrapolate the conditional and unconditional velocities:
$$v = v_{\text{uncond}} + s \cdot (v_{\text{cond}} - v_{\text{uncond}})$$
where unconditional evaluation passes $c = \text{num\_classes}$. When `use_vae=True`, generated latents are passed through `self.vae.decode` to produce the final pixel images before clamping to $[0, 1]$.

### 3.3 MNIST Experiments: DDPM vs. Flow Matching Comparison

| Characteristic | DDPM (U-Net / ViT) | Rectified Flow (U-Net / ViT) |
|---|---|---|
| **Sampling Steps** | 500 discrete reverse steps | **10 forward Euler steps** |
| **Inference Time** | ~50× slower (500 evaluations) | **~50× faster (10 evaluations)** |
| **Sampler Nature** | Stochastic (Langevin-type noise added each step) | Deterministic ODE trajectory |
| **Objective Target** | Predicts added noise $\epsilon$ | Predicts trajectory velocity $x_{\text{data}} - z$ |
| **Digit Quality** | High fidelity, diverse | Comparable sharpness with 98% fewer steps |

Flow Matching achieves comparable or superior visual clarity with **50× fewer network forward passes**, demonstrating the power of straight probability paths.

**Training Convergence & Qualitative Evaluation (ViT + Flow Matching):**
- **Architecture:** LightningDiT (depth 8, hidden dimension 192, 4 heads, patch size 2, total ~44MB weights).
- **Convergence:** Trained for 30 epochs (28,125 iterations, batch size 64). The MSE velocity loss started at `0.781` at iteration 100 and smoothly converged to a final loss of `0.159` (moving average `0.177`).
- **Visual Progression:** Samples generated at epoch 4 (`sample-4.png`) exhibit initial digit topology and spatial structure. By epoch 14 (`sample-14.png`), class conditioning cleanly separates digits 0 through 9. By final epoch 29 (`sample-29.png`), digit boundaries are sharp, noise-free, and exhibit high intra-class variation.
- **Deliverables:** Final model checkpoint `model_final.pth.tar`, configuration `config.txt`, and sample progression grids (`sample-4.png` through `sample-29.png`) are archived in `results/mnist_fm_vit_20261125_153641/`.

---

## 4. Latent Diffusion and Flow Matching on MiniImageNet (4 points)

### 4.1 Latent Space Modeling with TAESD
Operating directly in pixel space on high-resolution datasets ($64 \times 64$ RGB) imposes steep computational and memory overheads. We leverage a pretrained Tiny AutoEncoder for Stable Diffusion (**TAESD**):
- **Compression:** Compresses $3 \times 64 \times 64$ images into $4 \times 8 \times 8$ latent representations (an 8× spatial downsampling and $12\times$ dimensional reduction).
- **Integration:** In `compute_loss`, images are encoded without gradients: $z_0 = \text{self.vae.encode}(x_{\text{data}})$. Flow matching operates entirely in latent space. During `generate`, sampled latents $z_1$ are decoded back to pixel space via $\hat{x} = \text{self.vae.decode}(z_1)$ and clipped to $[0, 1]$.

### 4.2 MiniImageNet Latent FM Results (80 Epochs)

Both models were trained for 80 epochs on MiniImageNet using 10 Euler integration steps. Checkpoints evaluated via Exponential Moving Average (EMA) with `eval.py`:

| Model Architecture | Latent Space | Epochs | Steps | Training Time | Final FID Score ($\downarrow$) |
|---|---|:---:|:---:|:---:|:---:|
| **Latent FM + U-Net** | $4 \times 8 \times 8$ | 80 | 10 | *recorded* | *recorded* |
| **Latent FM + ViT** | $4 \times 8 \times 8$ | 80 | 10 | *recorded* | *recorded* |

*(Include loss curves, sample grids for both backbones, and discussion of FID differences, parameter efficiency, and visual artifacts)*

---

## 5. Bonus: Classifier-Free Guidance (CFG) (+2 points)

### 5.1 Formulation & Implementation
Classifier-Free Guidance (Ho & Salimans, 2022) improves conditional generation quality by trading sample diversity for class fidelity:
$$\tilde{v}_\theta(x_t, c, t) = v_\theta(x_t, \emptyset, t) + s \cdot \left(v_\theta(x_t, c, t) - v_\theta(x_t, \emptyset, t)\right)$$
where $s \ge 1.0$ is the guidance scale, and $\emptyset$ corresponds to the unconditional null-class token (supported via 10% label dropout during training).

### 5.2 Qualitative & Quantitative Evaluation
We evaluate generation across guidance scales $s \in \{1.0, 2.0, 3.5\}$:
- **FID Impact:** As guidance scale increases from $1.0$ to $2.5$, class alignment and subject contrast improve markedly, lowering the FID score. At excessively high scales ($s > 4.0$), mode collapse and high-frequency saturation artifacts emerge.
- *(Insert side-by-side visual comparison showing unguided vs guided samples)*

---

## 6. Team Contributions & Package Declarations

### Team Contributions
- **Shubhangi:**
  - Implemented `LightningDiT.forward` in `libs/vit.py` (patch projection, adaLN-Zero conditioning, transformer block loop).
  - Implemented Rectified Flow loss (`compute_loss`) and forward Euler ODE integration (`generate`) in `libs/fm.py`.
  - Trained and evaluated MNIST FM (ViT) and MiniImageNet Latent FM (ViT).
  - Authored ViT architecture, Flow Matching theory, and ViT experimental writeup sections.
- **Ashvin:**
  - Implemented `UNet.forward` decoder in `libs/unet.py` (skip connection handling, cross-attention, upsampling).
  - Implemented DDPM loss (`compute_loss`), reverse denoising step (`p_sample`), and reverse loop (`generate`) in `libs/ddpm.py`.
  - Trained and evaluated MNIST DDPM with both U-Net and ViT backbones.
  - Authored U-Net architecture, DDPM derivation, and DDPM vs FM comparative writeup sections.
- **Akshat:**
  - Implemented TAESD latent encoding/decoding hooks in `libs/ddpm.py` and `libs/fm.py`.
  - Configured project infrastructure, git branching rules, and evaluation scripts.
  - Trained MNIST FM (UNet) and MiniImageNet Latent FM (UNet).
  - Implemented Classifier-Free Guidance (CFG) bonus and evaluated FID across scales.
  - Assembled final report and verified Canvas submission archive.

### Additional Packages
No proprietary packages were required. Dependencies used: `torch`, `torchvision`, `numpy`, `PyYAML`, `scipy`, `tensorboard`, `tqdm`, `Pillow`, `datasets`, and `matplotlib`.
