import math

from typing import Union
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .utils import default

################################################################################
# Normalization
################################################################################

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for
                numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator
                for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


################################################################################
# Attention
################################################################################

class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=4,
        qkv_bias=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x (tensor): Input feature map of size B x H x W x C
        """
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    """Multi-head Cross Self-Attention."""

    def __init__(
        self,
        dim,
        context_dim,
        num_heads=4,
        qkv_bias=False,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(context_dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        """
        Args:
            x (tensor): Input feature map of size B x H x W x C (used for q)
            context (tensor): Input context feature of size B x T x C (used for k, v)
        """
        B, H, W, _ = x.shape
        B, T, _ = context.shape

        # q with shape (B, nHead, H * W, C)
        q = self.q(x).reshape(B, H * W, self.num_heads, -1).permute(0, 2, 1, 3)
        # kv with shape (2, B, nHead, T, C)
        kv = self.kv(context).reshape(B, T, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W / T, C)
        k, v = kv.reshape(2, B * self.num_heads, T, -1).unbind(0)
        q = q.reshape(B * self.num_heads, H*W, -1)

        # attention
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        x = self.proj(x)
        return x


class DiTAttention(nn.Module):
    """Multi-head self-attention with optional QK normalization and 2D RoPE."""
    def __init__(
        self,
        dim: int,
        grid_size: Union[int, list],
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        use_rmsnorm: bool = False,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop_ratio = attn_drop
        self.proj_drop_ratio = proj_drop
        self.use_rope = use_rope

        if use_rmsnorm:
            norm_layer = RMSNorm

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop > 0 \
            else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 \
            else nn.Identity()

        if self.use_rope:
            self.rope = ViTRotaryEmbedding(dim, num_heads, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): Input feature map of size B x N x C
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B, N, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        # self-attention using fused kernel
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_ratio if self.training else 0.,
        )

        # Combine attention heads and project back to the feature dimension.
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


################################################################################
# FFNs / MLPs
################################################################################

class GEGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2 * out_dim)

    def forward(self, x):
        x = self.proj(x)
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class MLP_SD(nn.Module):
    """MLP with GEGLU. This is the FFN used by stable diffusion."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp = nn.Sequential(
            GEGLU(in_features, hidden_features),
            nn.Dropout(0.0),
            nn.Linear(hidden_features, out_features)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


class SwiGLUFFN(nn.Module):
    """MLP with SwiGLU. This is the FFN used by DiT."""
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


################################################################################
# Positional Embeddings
################################################################################

class SinusoidalPE(nn.Module):
    """
    sinusoidal position encoding for time steps as in the Transformer paper
    """
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period=max_period

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(
                start=0, end=half, dtype=torch.float32) / half
        )
        self.freqs = nn.Parameter(freqs, requires_grad=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class ViTRotaryEmbedding(nn.Module):
    """
    2D Rotary positional embedding w. learnable params for vision transformers
    Ref: https://arxiv.org/abs/2403.13298
    Note: the implmentation assumes fixed size
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: Union[int, list],
        theta: float = 10.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.theta = theta
        if num_heads <= 0 or dim % num_heads or (dim // num_heads) % 4:
            raise ValueError("2D RoPE requires a head dimension divisible by four")

        if isinstance(grid_size, int):
            max_x = max_y = grid_size
        else:
            assert len(grid_size) == 2
            # Tokens are flattened in row-major order: width varies fastest.
            max_y, max_x = grid_size[0], grid_size[1]

        self.num_pts = max_x * max_y

        freqs = self.init_2d_freqs(dim//num_heads, num_heads, theta).view(2, -1)
        t_x, t_y = self.init_t_xy(max_x, max_y)

        self.freqs = nn.Parameter(freqs, requires_grad=True)
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)

    @staticmethod
    def init_2d_freqs(dim: int, num_heads: int, theta: float) -> torch.Tensor:
        freqs_x = []
        freqs_y = []
        mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
        for i in range(num_heads):
            angles = torch.rand(1) * 2 * torch.pi
            fx = torch.cat(
                [mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)],
                dim=-1
            )
            fy = torch.cat(
                [mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)],
                dim=-1
            )
            freqs_x.append(fx)
            freqs_y.append(fy)
        freqs_x = torch.stack(freqs_x, dim=0)
        freqs_y = torch.stack(freqs_y, dim=0)
        freqs = torch.stack([freqs_x, freqs_y], dim=0)
        return freqs

    @staticmethod
    def init_t_xy(end_x: int, end_y: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(end_x * end_y, dtype=torch.float32)
        t_x = (t % end_x).to(t)
        t_y = torch.div(t, end_x, rounding_mode='floor').to(t)
        return t_x, t_y

    @staticmethod
    def reshape_for_broadcast(
        freqs: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        ndim = x.ndim
        assert 0 <= 1 < ndim
        if freqs.shape == (x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
        elif freqs.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
            shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        else:
            raise ValueError("Rope dimension mismatch.")
        return freqs.view(*shape)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.view(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim = -1)
        x = torch.stack((-x2, x1), dim = -1)
        x = x.view(*x.shape[:-2], -1)
        return x

    def compute_mixed_cis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute freqs for x and y axis
        freqs_x = self.freqs[0].view(self.num_heads, -1).unsqueeze(1) \
            * self.freqs_t_x.view(1, self.num_pts, 1)
        freqs_y = self.freqs[1].view(self.num_heads, -1).unsqueeze(1) \
            * self.freqs_t_y.view(1, self.num_pts, 1)

        # separate out the real / complex components
        # (not the most efficient way, but fine for now)
        freqs_cos = torch.cos(freqs_x + freqs_y).repeat_interleave(2, dim=-1)
        freqs_sin = torch.sin(freqs_x + freqs_y).repeat_interleave(2, dim=-1)
        return freqs_cos, freqs_sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freqs_cos, freqs_sin = self.compute_mixed_cis()
        freqs_cos = self.reshape_for_broadcast(freqs_cos, x)
        freqs_sin = self.reshape_for_broadcast(freqs_sin, x)
        return  x * freqs_cos + self.rotate_half(x) * freqs_sin


################################################################################
# Key building blocks
################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Same as DiT.
    """
    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256
    ) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.time_embed = SinusoidalPE(frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.time_embed(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout
    needed for classifier-free guidance. Same as DiT.
    """
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        dropout_prob: float = 0.0
    ):
        super().__init__()
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError("dropout_prob must be between 0.0 and 1.0")
        use_cfg_embedding = dropout_prob > 0
        # # classes + 1, with the last slot reserved for cfg (if needed)
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels for classifier-free guidance.
        """
        if force_drop_ids is None:
            # randomly select labels / conditions to drop
            drop_ids = torch.rand(
                labels.shape[0], device=labels.device
            ) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        # dropped labels -> set to self.num_classes
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer w. cross/self-attention used by stable diffusion.
    This block includes one self-attention, one cross-attention, and one MLP.
    """
    def __init__(
        self,
        dim,
        context_dim,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        groups=4
    ):
        """
        Args:
            dim (int): Number of input channels.
            context_dim (int): Number of input context channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()

        self.group_norm = nn.GroupNorm(groups, dim)
        self.proj_in = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim,
            context_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.mlp = MLP_SD(
            in_features=dim, hidden_features=int(dim * mlp_ratio)
        )
        self.norm3 = norm_layer(dim)

    def forward(self, x, context):
        """
        Args:
            x (tensor): Input feature map of size B x C x H x W (used for q)
            context (tensor): Input context feature of size B x T x C (used for k, v)
        """
        shortcut = x
        x = self.proj_in(self.group_norm(x))
        x = x.permute(0, 2, 3, 1)  # B C H W -> B H W C
        x = self.self_attn(self.norm1(x)) + x
        x = self.cross_attn(self.norm2(x), context) + x
        x = self.mlp(self.norm3(x)) + x
        x = x.permute(0, 3, 1, 2) # B H W C -> B C H W
        return self.proj_out(x) + shortcut


class ResBlock(nn.Module):
    """A slightly altered residual block with time modulation"""
    def __init__(self, in_channel, time_emb_dim, out_channel=None, groups=4):
        """
        Args:
            in_channel (int): Number of input channels.
            time_emb_dim (int): Number of input time embedding dimensions.
            out_channel (int): Number of output channels
            (default: same as in_channel)
            groups (int): number of groups used in groupnorm (default: 4)
        """

        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.norm1 = nn.GroupNorm(groups, in_channel)
        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(
            in_features=time_emb_dim, out_features=out_channel, bias=True
        )
        self.norm2 = nn.GroupNorm(groups, out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, stride=1, padding=1
        )
        self.act = nn.SiLU()
        if in_channel == out_channel:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(
                in_channel, out_channel, kernel_size=1, stride=1
            )

    def forward(self, x, t_emb, cond=None):
        """
        Args:
            x (tensor): Input feature map of size B x C x H x W
            t_emb (tensor): Input time embedding of size B x T x C
        """

        # pre-norm + conv1
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # Time modulation (with shift only, similar to stable diffusion)
        if t_emb is not None:
            t_hidden = self.time_emb_proj(self.act(t_emb))
            h = h + t_hidden[:, :, None, None]
        # norm
        h = self.norm2(h)
        h = self.act(h)
        # conv2
        h = self.conv2(h)
        # Skip connection
        return h + self.conv_shortcut(x)


class DiTBlock(nn.Module):
    """
    DiT Block. New features includes:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - AdaLN residual gates
    - No shift AdaLN.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        grid_size,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_rmsnorm=False,
        use_rope=False,
        wo_shift=False,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = DiTAttention(
            hidden_size,
            grid_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_rope=use_rope,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # SwiGLU with recomended hidden dim
        self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c):
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(4, dim=1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DiTDecodingLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        use_rmsnorm: bool = False
    ) -> None:
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6
            )
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        # pixel shuffle for upsampling
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


################################################################################
# Misc helper functions
################################################################################

class Upsample(nn.Module):
    """Upsampling using nearest neighbor + conv2D"""

    def __init__(self, dim, dim_out=None, scale_factor=2):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.conv = nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling using strided convolution"""

    def __init__(self, dim, dim_out=None, scale_factor=2):
        super().__init__()
        self.scale_factor = int(scale_factor)
        self.conv = nn.Conv2d(
            dim, default(dim_out, dim), kernel_size=3, stride=self.scale_factor, padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        return x


def modulate(x, shift, scale):
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
