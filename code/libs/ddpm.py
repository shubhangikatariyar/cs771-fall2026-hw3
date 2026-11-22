# Implementation of DDPM described in https://arxiv.org/abs/2006.11239
# Reference: https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_arch
from .guidance import sampling_mode, validate_guidance
from .tiny_autoencoder import TAESD


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model"""

    def __init__(
        self,
        img_shape=(3, 32, 32),
        timesteps=100,
        num_classes=10,
        use_vae=False,
        vae_encoder_weights=None,
        vae_decoder_weights=None,
        arch_type="UNet",
        arch_cfg=None,
    ):
        """
        Args:
            img_shape (tuple/list of int): image shape before any VAE (C x H x W)
            timesteps (int): number of timesteps in the diffusion process
            num_classes (int): number of classes used for conditioning
            use_vae (bool): if a VAE is used before DDPM (thus latent diffusion)
            vae_encoder_weights (str): path to pre-trained VAE encoder weights
            vae_decoder_weights (str): path to pre-trained VAE decoder weights
            arch_type (str): UNet or ViT
            arch_cfg (dict): backbone parameters, including class_dropout_prob
        """

        super().__init__()

        if img_shape is None or len(img_shape) != 3 or any(s <= 0 for s in img_shape):
            raise ValueError("img_shape must contain positive channel, height, width values")
        if not isinstance(timesteps, int) or timesteps <= 0:
            raise ValueError("timesteps must be a positive integer")
        if num_classes is None or num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.timesteps = timesteps
        # use different variance schedule for LDM / DDPM
        if use_vae:
            betas = self.quadratic_beta_schedule(timesteps)
        else:
            betas = self.linear_beta_schedule(timesteps)

        alpha_vars = self.compute_alpha_vars(betas)
        self.betas = betas
        self.sqrt_recip_alphas = alpha_vars[0]
        self.sqrt_alphas_cumprod = alpha_vars[1]
        self.sqrt_one_minus_alphas_cumprod = alpha_vars[2]
        self.posterior_variance = alpha_vars[3]
        # if we should consider latent DDPM
        self.use_vae = use_vae
        if use_vae:
            if vae_encoder_weights is None or vae_decoder_weights is None:
                raise ValueError("Both VAE encoder and decoder weights are required")
            self.vae = TAESD(
                encoder_path=vae_encoder_weights,
                decoder_path=vae_decoder_weights
            )
            # freeze the encoder / decoder
            for param in self.vae.parameters():
                param.requires_grad = False

        # if VAE is considered, input / output dim will be the dim of latent
        if use_vae:
            if img_shape[0] != 3 or any(s % self.vae.patch_size for s in img_shape[1:]):
                raise ValueError("TAESD requires RGB images with dimensions divisible by its patch size")
            self.img_shape = [
                self.vae.latent_channels,
                img_shape[1]//self.vae.patch_size,
                img_shape[2]//self.vae.patch_size
            ]
        else:
            self.img_shape = img_shape

        self.model = build_arch(
            arch_type, arch_cfg, img_shape=self.img_shape, num_classes=num_classes
        )

    @torch.no_grad()
    def linear_beta_schedule(self, timesteps):
        """
        linear schedule as in DDPM paper (Sec 4)
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    @torch.no_grad()
    def quadratic_beta_schedule(self, timesteps):
        """
        quadratic schedule as in Stable Diffusion
        """
        beta_start = 0.00085
        beta_end = 0.012
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

    @torch.no_grad()
    def compute_alpha_vars(self, betas):
        """
        compute vars related to alphas from betas
        """
        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        return (
            sqrt_recip_alphas,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
            posterior_variance
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Helper function to match the dimensions.
        It sets a[t[i]] for every element t[i] in t and expands the results
        into x_shape
        """

        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process (using the nice property from Gaussian)
        It adds noise to a starting image and outputs its noisy version at
        an arbitary time step t
        """

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod, t, x_start.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        """
        Equation 4 in DDPM paper.
        """
        x_t = (sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise)
        return x_t

    # compute the simplified loss
    def compute_loss(self, x_start, label, noise=None):
        """
        Compute the simplified loss for training the model.
        Fill in the missing code here. Algorithm 1 line 5 in the paper.
        """
        return loss

    @torch.no_grad()
    def p_sample(
        self, x, label, t, t_index,
        cfg=False, cfg_scale=1.0, cfg_start_t=0.0,
    ):
        """
        Denoise a noisy image, optionally guiding the predicted noise.

        cfg_start_t measures progress from noise (0) toward the image (1),
        matching the direction of time as used in flow matching.
        """

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        """
        Fill in the missing code here. See Equation 11 in DDPM paper.
        See https://arxiv.org/pdf/2207.12598 for classifier free guidance (cfg)
        """
        return x

    @torch.no_grad()
    @sampling_mode
    def generate(self, labels, cfg=False, cfg_scale=1.0, cfg_start_t=0.0):
        """
        Sample from DDPM with optional classifier-free guidance on epsilon.

        Scale 1 gives the conditional prediction; larger scales strengthen
        conditioning. Guidance requires label dropout during training.
        """

        device = next(self.model.parameters()).device
        labels = labels.to(device=device, dtype=torch.long)
        if cfg:
            validate_guidance(self.model, cfg_scale, cfg_start_t)
        # shape of the results
        shape = [len(labels)] + list(self.img_shape)
        # start from pure noise (for each example in the batch)
        imgs = torch.randn(shape, device=device)

        """
        Fill in the missing code here. See Algorithm 2 in DDPM paper.
        """
        
        # postprocessing the images
        imgs = self.postprocess(imgs)
        return imgs

    @torch.no_grad()
    def postprocess(self, imgs):
        """
        Postprocess the sampled images (e.g., normalization / clipping)
        """
        if self.use_vae:
            # already in range, clip the pixel values
            imgs.clamp_(min=0.0, max=1.0)
        else:
            # clip the pixel values within range
            imgs.clamp_(min=-1.0, max=1.0)
            # mapping to range [0, 1]
            imgs = (imgs + 1.0) * 0.5
        return imgs
