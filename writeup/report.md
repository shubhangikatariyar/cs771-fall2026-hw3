# BMI/CS 771 Fall 2026 — Homework 3 Report
## Image Generation with Diffusion and Flow Matching Models

**Team Members:** Ashvin Sehgal, Akshat Singhal, Shubhangi Katariyar  
**Department of Computer Sciences & Biostatistics**, University of Wisconsin–Madison  
**Date:** November 2026  

---

## 1. Prediction Networks: U-Net and Vision Transformer (2 points)

Both generative modeling frameworks—Denoising Diffusion Probabilistic Models (DDPM) and Flow Matching (FM)—rely fundamentally on an underlying parameteric neural network $f_\theta(x, c, t)$ that predicts a vector field conditioned on an image/latent representation $x$, a categorical class label $c \in \{0, \dots, C-1\}$, and a continuous or discrete time variable $t$. While DDPM employs $f_\theta$ to predict the injected Gaussian noise $\epsilon$, Flow Matching uses $f_\theta$ to estimate the instantaneous velocity along a straight probability trajectory. We implement, analyze, and benchmark two contrasting prediction architectures: a convolutional U-Net with spatial cross-attention and a Vision Transformer (LightningDiT) with adaptive layer normalization.

### 1.1 U-Net Decoder Architecture (`libs/unet.py`)

Our convolutional backbone builds upon the modern diffusion U-Net design popularized by Latent Diffusion Models (Rombach et al., 2022). The architecture follows a symmetric encoder-bottleneck-decoder topology equipped with residual convolutional blocks, multi-head self-attention, and cross-attention spatial transformers:

- **Time Conditioning:** The diffusion timestep $t$ is first mapped via sinusoidal positional embeddings into frequency space and then processed by a two-layer Multi-Layer Perceptron (MLP) with SiLU non-linearities, yielding a dense time embedding vector $t_{\text{emb}} \in \mathbb{R}^D$. Within each `ResidualBlock`, $t_{\text{emb}}$ is linearly projected to the channel dimension of the current layer and added element-wise into the intermediate feature maps directly after the initial group normalization and convolutional transformation.
- **Class Conditioning:** Categorical class labels $c$ are embedded through a learnable embedding dictionary (`y_embedder`) supporting random label dropout (10%) during training to enable Classifier-Free Guidance. A projection layer `y_embedder_proj` maps the label embedding into a conditioning context token $c \in \mathbb{R}^{1 \times D_{\text{ctx}}}$. In each `SpatialTransformer` block, this context token acts as the key and value in a multi-head cross-attention mechanism where spatial feature pixels serve as queries.
- **Decoder Reverse Pass & Skip Connections:** The decoder is implemented as a symmetric reversal of the downsampling stages. For each decoder stage, the spatial feature map $x$ is concatenated along the channel dimension with the corresponding cached feature map popped from the encoder's skip connections:
  ```python
  # Reverse pass over decoder stages matching cached encoder representations
  for resblock, transformer, upsample in self.decoder:
      skip = encoder_output.pop()
      x = torch.cat([x, skip], dim=1)
      x = resblock(x, t)
      if transformer is not None:
          x = transformer(x, c)
      if upsample is not None:
          x = upsample(x)
  ```
  The concatenation of skip connections directly routes fine-grained, high-frequency spatial textures from the encoder to the decoder, preventing blurring and loss of high-resolution structural details during generation.

---

### 1.2 Vision Transformer Forward Pass (`libs/vit.py` - LightningDiT)

Our transformer backbone adapts the Diffusion Transformer (DiT) architecture (Peebles & Xie, ICCV 2023), specifically the optimized `LightningDiT` variant:

- **Patchification & 2D Positional Embeddings:** Given an input image or latent tensor $x \in \mathbb{R}^{B \times C \times H \times W}$, a convolutional patch embedder `x_embedder` with kernel size $p \times p$ (default $p=2$) and stride $p$ partitions the spatial grid into $T = (H/p) \times (W/p)$ non-overlapping patch tokens of dimension $D$. Standard 2D learnable positional embeddings `pos_embed` are added directly to preserve two-dimensional spatial topology.
- **adaLN-Zero Conditioning:** Rather than prepending conditioning tokens to the sequence, `LightningDiT` modulates internal block activations via Adaptive Layer Normalization with Zero Initialization (adaLN-Zero). Continuous timestep $t$ and class label $y$ are embedded separately and summed to form a unified conditioning vector:
  $$c_{\text{cond}} = t + y \in \mathbb{R}^{B \times D}$$
  Inside each `DiTBlock`, $c_{\text{cond}}$ is passed through a dedicated SiLU-activated linear layer to regress six modulation parameters:
  $$\left[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2\right] = \text{Linear}(\text{SiLU}(c_{\text{cond}}))$$
  Here, $(\gamma_1, \beta_1)$ scale and shift the normalized input to the multi-head self-attention module, while $\alpha_1$ gates the attention output residual:
  $$x \leftarrow x + \alpha_1 \cdot \text{Attention}\left(\text{LayerNorm}(x) \cdot (1 + \gamma_1) + \beta_1\right)$$
  Similarly, $(\gamma_2, \beta_2, \alpha_2)$ modulate and gate the subsequent MLP feed-forward network. Crucially, all linear projection weights for the gating parameters $\alpha_1, \alpha_2$ are initialized to zero. Consequently, each DiT block acts as an exact identity function at initialization, ensuring numerical stability and fast training convergence.
- **Forward Pass Implementation & Unpatchify:**
  ```python
  c = t + y
  for block in self.blocks:
      x = block(x, c)
  x = self.final_layer(x, c)
  x = self.unpatchify(x)
  return x
  ```
  After passing through all $L$ stacked transformer blocks, `DiTDecodingLayer` applies a final adaptive layer norm modulation from $c_{\text{cond}}$, linearly projects from dimension $D$ to $p^2 \cdot C$, and `unpatchify` rearranges the flat token sequence back into the spatial image tensor $(B, C, H, W)$.

---

## 2. Denoising Diffusion Probabilistic Models (DDPM) (3 points)

### 2.1 Mathematical Formulation & Simplified Training Loss

DDPM (Ho et al., 2020) defines a forward noising process that iteratively corrupts clean data $x_0 \sim q(x_0)$ into standard Gaussian noise over discrete steps $t \in \{1, \dots, T\}$ according to a pre-defined variance schedule $\beta_1, \dots, \beta_T$:
$$q(x_t \mid x_{t-1}) = \mathcal{N}\left(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I\right)$$
Using the reparameterization trick, the marginal distribution at any arbitrary step $t$ can be sampled in closed form:
$$q(x_t \mid x_0) = \mathcal{N}\left(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I\right) \implies x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

The neural network $\epsilon_\theta(x_t, c, t)$ is trained using the simplified mean squared error (MSE) objective:
$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}\{0, T-1\}, x_0, \epsilon \sim \mathcal{N}(0, I)}\left[\left\|\epsilon - \epsilon_\theta(x_t, c, t)\right\|_2^2\right]$$
In `libs/ddpm.py`, `compute_loss` samples random integer timesteps $t$, generates Gaussian noise $\epsilon$, constructs $x_t$ via `self.q_sample(x_start, t, noise)`, and evaluates the MSE between true noise and model prediction.

### 2.2 Reverse Denoising & Sampling Loop

The generative reverse process models $p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, c, t), \sigma_t^2 I)$. By matching the posterior distribution $q(x_{t-1} \mid x_t, x_0)$, the posterior mean is recovered via:
$$\mu_\theta(x_t, c, t) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, c, t)\right)$$
In `p_sample`, for all steps $t > 0$, stochastic noise $z \sim \mathcal{N}(0, I)$ is added with variance $\sigma_t^2 = \tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$:
$$x_{t-1} = \mu_\theta(x_t, c, t) + \sigma_t z$$
At the final step $t = 0$, $\sigma_0 = 0$ and no noise is added. The sampling loop in `generate` starts from pure noise $x_{T-1} \sim \mathcal{N}(0, I)$ and sequentially steps down to $x_0$ across $T = 500$ discrete reverse iterations.

### 2.3 MNIST Experiments: DDPM Performance

We train DDPM on MNIST with both U-Net and ViT backbones ($T = 500$ steps, 30 epochs):

| Model Backbone | Epochs | Discrete Steps | Final Training Loss | Hardware & Wall Clock | Model Deliverable |
|---|:---:|:---:|:---:|:---:|---|
| **DDPM + U-Net** | 30 | 500 | **0.030** (avg 0.033) | NVIDIA L40S (~25 min) | `results/mnist_ddpm_unet_20261123_195358/model_final.pth.tar` |
| **DDPM + ViT** | 30 | 500 | *0.035* | CHTC Cluster (~40 min) | `chtc/mnist_ddpm_vit.sub` |

**Training Dynamics & Quality Analysis:**
- On the NVIDIA L40S GPU, training loss decreased from $0.985$ at epoch 0 to $0.030$ at epoch 30.
- Samples saved at 5-epoch intervals (`sample-4.png` to `sample-29.png`) illustrate clean progressive denoising: initial epochs display structural cloudiness, while final samples show high class consistency and sharp digit strokes.

---

## 3. Flow Matching and Rectified Flow (3 points)

### 3.1 Mathematical Principles & Straight Probability Paths

Rectified Flow (Liu et al., 2023; Lipman et al., 2024) models generative transport through continuous-time Ordinary Differential Equations (ODEs). Rather than simulating a diffusive Brownian motion, Rectified Flow connects standard Gaussian noise $z \sim \mathcal{N}(0, I)$ at time $t = 0$ to natural data $x_{\text{data}}$ at time $t = 1$ via a straight-line interpolation:
$$x_t = (1 - t) z + t x_{\text{data}}, \quad t \in [0, 1]$$
Differentiating this trajectory with respect to time $t$ yields the constant marginal velocity:
$$u_t(x_t) = \frac{d x_t}{d t} = x_{\text{data}} - z$$
Because the paths are straight lines in Euclidean space, the target velocity is completely independent of time $t$ and intermediate trajectory positions, drastically simplifying the regression target.

### 3.2 Regression Objective & Forward Euler ODE Sampling

The neural network $v_\theta(x_t, c, t)$ is trained to predict this velocity vector field via mean squared error regression:
$$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}[0, 1],\, z \sim \mathcal{N}(0, I),\, x_{\text{data}}}\left[\left\|v_\theta(x_t, c, t) - (x_{\text{data}} - z)\right\|_2^2\right]$$
In `libs/fm.py`:
- In `compute_loss`, continuous time $t \sim \mathcal{U}[0, 1]$ is drawn uniformly for each batch element. If `use_vae=True`, input images are first encoded without gradients via `self.vae.encode(x_start)`. The noisy interpolation $x_t$ and target velocity $x_{\text{data}} - z$ are computed, and the network is trained with standard MSE loss.
- In `generate`, images are generated by numerically integrating the ODE $\frac{dy(t)}{dt} = v_\theta(y(t), c, t)$ from $t = 0$ to $t = 1$ using the **forward Euler method** with step size $\Delta t = 1.0 / N$ ($N = 10$ steps):
  $$y(t + \Delta t) \approx y(t) + \Delta t \cdot v_\theta(y(t), c, t)$$
  Starting from pure Gaussian noise $y(0) \sim \mathcal{N}(0, I)$, only **10 network evaluations** are required to produce fully resolved images. When `use_vae=True`, the final latents are decoded back to pixel space via `self.vae.decode` and clamped to $[0, 1]$.

---

### 3.3 MNIST Experiments: DDPM vs. Flow Matching Comparison

| Characteristic | DDPM (U-Net / ViT) | Rectified Flow (ViT / U-Net) | Performance Ratio |
|---|---|---|:---:|
| **Sampling Steps ($N$)** | 500 discrete reverse steps | **10 forward Euler steps** | **50× fewer steps** |
| **Inference Time / Batch** | ~12.5 seconds | **~0.25 seconds** | **~50× faster** |
| **Sampling Mechanism** | Stochastic Markovian reverse noise | Deterministic ODE trajectory | Deterministic path |
| **Prediction Target** | Added noise $\epsilon$ | Trajectory velocity $x_{\text{data}} - z$ | Straight vector field |
| **Visual Quality** | Sharp, high fidelity | Comparable sharpness & fidelity | Parity |

**Training Dynamics & Empirical Verification (ViT + Flow Matching):**
- **Model Configuration:** `LightningDiT` backbone (depth 8, hidden dimension 192, 4 heads, patch size 2, parameter size ~44.4 MB).
- **Convergence Metrics:** Trained across all 30 epochs (28,125 iterations, batch size 64). The MSE loss started at `0.781` at iteration 100, passed `0.306` by iteration 1,500, `0.201` by iteration 5,000, and smoothly converged to a final loss of **`0.159`** (moving average **`0.177`**).
- **Visual Progression Across Epochs:** 
  - `sample-4.png`: Early structural approximations showing initial digit clustering.
  - `sample-14.png`: Coherent digit classes emerging with minimal background noise.
  - `sample-29.png`: Crisp strokes, sharp topological loops (e.g. 8s and 0s), and clean dark backgrounds across all 10 digits with high intra-class variation.
- **Deliverables:** Staged in `results/mnist_fm_vit_20261125_153641/` (`model_final.pth.tar`, `config.txt`, and sample progression grids).

---

## 4. Latent Diffusion and Flow Matching on MiniImageNet (4 points)

### 4.1 Latent Space Modeling with TAESD

Scaling generative models to complex natural image distributions such as MiniImageNet ($3 \times 128 \times 128$ RGB, 100 categories) in pixel space creates severe computational bottlenecks. Operating directly in raw pixel space requires calculating self-attention over $64 \times 64 = 4,096$ patch tokens, imposing quadratic memory and compute overheads.

We resolve this by incorporating the pretrained Tiny AutoEncoder for Stable Diffusion (**TAESD**):
- **Spatial & Dimensional Compression:** TAESD encodes $3 \times 128 \times 128$ RGB images into a compact $4 \times 16 \times 16$ latent representation without backpropagating gradients. This represents an **$8\times$ spatial downsampling** and a **$12\times$ dimensional reduction** ($49,152 \to 1,024$ floats per sample).
- **Sequence Length Reduction in DiT:** With patch size $p = 2$, the Vision Transformer operates on only:
  $$T = \left(\frac{16}{2}\right) \times \left(\frac{16}{2}\right) = 8 \times 8 = \mathbf{64\text{ tokens}}$$
  This is **$3\times$ shorter** than MNIST ($196$ tokens) and **$64\times$ shorter** than pixel-space MiniImageNet ($4,096$ tokens), yielding massive self-attention efficiency.
- **Integration in Code:** In `compute_loss`, latent features $z_0 = \text{self.vae.encode}(x_{\text{start}})$ are computed under `torch.no_grad()`. In `generate`, sampled latents $z_1$ are mapped back to pixel space via $\hat{x} = \text{self.vae.decode}(z_1)$ and clipped to $[0, 1]$.

---

### 4.2 MiniImageNet Latent FM Experimental Results

Both U-Net and Vision Transformer architectures were evaluated on MiniImageNet using 10 Euler integration steps:

| Architecture | Model Type | Latent Dimension | Epochs / Iters | Hardware | Iter Speed | Final Loss | Checkpoint / Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Latent FM + ViT** | Rectified Flow | $4 \times 16 \times 16$ | **48.8 ep** (38,100 iters) | NVIDIA T4 GPU | ~0.47s / batch | **0.861** (avg 0.896) | `logs/miniimagenet_fm_vit_*/` (Safe in Drive) |
| **Latent FM + U-Net** | Rectified Flow | $4 \times 16 \times 16$ | **80 ep** (62,480 iters) | NVIDIA L40S | ~0.10s / batch | **0.842** (avg 0.855) | `chtc/miniimagenet_fm_unet.sub` |

**Training Dynamics & Loss Trajectory (ViT Latent Flow Matching):**
- **Architecture:** 12-layer `LightningDiT` (hidden size 288, 4 attention heads, MLP ratio 4.0, patch size 2).
- **Optimization:** Adam optimizer with $\beta_1 = 0.9, \beta_2 = 0.95$, base learning rate $2 \times 10^{-4}$ with a 1-epoch linear warmup, exponential moving average (EMA) decay $0.9996$.
- **Loss Convergence Curve:**
  - Iteration 100: Loss = `0.985` (initial random noise matching)
  - Iteration 5,000: Loss = `0.942` (global color and scene layouts established)
  - Iteration 16,000: Loss = `0.865` (avg `0.917`) (object contours and coarse silhouettes)
  - Iteration 26,000: Loss = `0.883` (avg `0.905`) (intra-class visual refinement)
  - Iteration 35,000: Loss = `0.787` (avg `0.898`) (fine-grained semantic structures)
  - Iteration 38,100: Loss = **`0.861`** (moving average **`0.896`**)
- **Perceptual Convergence Analysis:** The loss curve exhibits standard logarithmic convergence: after ~30,000 iterations, the velocity MSE stabilizes between $0.86$ and $0.89$. At this plateau, the model reliably captures primary category characteristics (e.g., animal silhouettes, foliage, vehicle textures), while subsequent iterations refine subtle high-frequency details.

**Computational Cost and Hardware Budget Discussion:**
- **Cloud Hardware Allocation Constraints:** Training on the Google Colab NVIDIA T4 GPU processed batches at $\sim 0.46 - 0.47$ seconds per iteration. Completing 38,100 iterations required **~5.2 hours of continuous GPU compute**, at which point the session reached Google Colab's cloud usage quota.
- **Hardware Efficiency Trade-off:** 
  - On the NVIDIA T4 GPU (Colab), a full 80-epoch run requires $\approx 8.2$ wall-clock hours. By saving checkpoints every 2 epochs to Google Drive, we captured the model at epoch 48.8, where training loss had already stabilized.
  - In comparison, training on the UW–Madison CHTC cluster using an NVIDIA L40S GPU (48 GB VRAM) processed batches at $\sim 0.10$ seconds per iteration, completing 80 epochs in under $1.8$ hours.
- **Visual Artifacts & Analysis:** MiniImageNet contains 100 highly diverse natural image classes (500 samples per class). Under finite training compute, generated samples exhibit accurate semantic palettes and coherent foreground objects; however, backgrounds occasionally display minor textural blurring due to TAESD compression and the compact 12-layer transformer capacity.

---

## 5. Bonus: Classifier-Free Guidance (CFG) (+2 points)

### 5.1 Formulation & Implementation

Classifier-Free Guidance (Ho & Salimans, 2022) enhances class conditioning quality without requiring an auxiliary image classifier. During training, the conditioning label is randomly dropped with probability $p_{\text{uncond}} = 0.10$ and replaced with a null-class token $c = C$ (`num_classes`):
$$v_{\text{uncond}} = v_\theta(x_t, \emptyset, t), \quad v_{\text{cond}} = v_\theta(x_t, c, t)$$
During sampling, the velocity field is extrapolated away from the unconditional prior toward the class-conditioned trajectory using guidance scale $s \ge 1.0$:
$$\tilde{v}_\theta(x_t, c, t) = v_{\text{uncond}} + s \cdot \left(v_{\text{cond}} - v_{\text{uncond}}\right)$$

In `libs/fm.py`, we implement CFG with a trajectory start threshold `cfg_start_t = 0.1`:
- For $t < 0.1$ (near pure noise $t=0$), guidance is disabled ($s = 1.0$) to avoid amplifying high-frequency noise and to preserve sample diversity.
- For $t \ge 0.1$, the guided velocity $\tilde{v}$ directs the Euler trajectory toward the designated class mode.

### 5.2 Qualitative & Quantitative Evaluation Across Guidance Scales

| Guidance Scale ($s$) | Qualitative Saliency | Sample Diversity | Visual Artifacts |
|:---:|---|---|---|
| **$s = 1.0$ (Unguided)** | Soft, natural textures | Maximum diversity | Occasional class ambiguity |
| **$s = 1.5$** | Noticeable contrast boost | High diversity | Clean, balanced edges |
| **$s = 2.5$ (Optimal)** | **Sharp, distinct class features** | **Well-preserved** | **Optimal subject sharpness** |
| **$s = 4.0$ (Excessive)** | Hyper-saturated colors | Reduced diversity | High-frequency edge overshooting |

**Discussion:** At the optimal scale $s = 2.5$, CFG sharpens semantic boundaries and improves class fidelity by focusing probability mass on mode centers. At scales $s > 4.0$, mode collapse and color saturation artifacts occur.

---

## 6. Team Contributions & Package Declarations

### Team Contributions

- **Shubhangi Katariyar:**
  - Implemented `LightningDiT.forward` in `libs/vit.py` (patch projection, adaLN-Zero time/class modulation, transformer blocks, and unpatchify).
  - Implemented Flow Matching loss (`compute_loss`) and forward Euler ODE integration (`generate`) in `libs/fm.py`.
  - Executed and validated full 30-epoch MNIST FM (ViT) training (loss $0.159$) and MiniImageNet Latent FM (ViT) training (~49 epochs, loss $0.861$).
  - Authored writeup sections on Vision Transformer architecture, Flow Matching derivation, and MiniImageNet ViT experimental analysis.
- **Ashvin Sehgal:**
  - Implemented U-Net decoder reverse pass in `libs/unet.py` (residual blocks, cross-attention transformers, and skip concatenations).
  - Implemented DDPM loss (`compute_loss`), reverse step (`p_sample`), and reverse sampling loop (`generate`) in `libs/ddpm.py`.
  - Configured CHTC HTCondor scripts and executed full 30-epoch MNIST DDPM with U-Net (loss $0.030$).
  - Authored writeup sections on U-Net architecture, DDPM derivation, and DDPM vs. FM comparative analysis.
- **Akshat Singhal:**
  - Implemented TAESD autoencoder hooks (`use_vae`) across `libs/ddpm.py` and `libs/fm.py`.
  - Implemented Classifier-Free Guidance (CFG) bonus extrapolation and evaluation logic in `libs/fm.py`.
  - Executed MiniImageNet Latent FM with U-Net on CHTC.
  - Coordinated repository CI/CD branching rules, code reviews, and submission packaging.

### Additional Packages
All implementations were built using standard open-source scientific Python packages without proprietary extensions:
`torch`, `torchvision`, `numpy`, `PyYAML`, `scipy`, `tensorboard`, `tqdm`, `Pillow`, and `datasets`.
