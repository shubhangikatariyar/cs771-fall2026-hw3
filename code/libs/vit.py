"""
Code is modified from Lightning DiTk,
(https://github.com/hustvl/LightningDiT/tree/main/models)
which was in turn built on original DiT & SiT.
(https://github.com/facebookresearch/DiT; https://github.com/willisma/SiT)
"""
import math

import torch
import torch.nn as nn

from .blocks import DiTBlock, DiTDecodingLayer, TimestepEmbedder, LabelEmbedder


class LightningDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=(32, 32),
        patch_size=2,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_qknorm=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
    ):
        super().__init__()

        # set up parameters
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.num_heads = num_heads
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.depth = depth
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob

        # compute patch stats
        if len(input_size) != 2 or any(size <= 0 for size in input_size):
            raise ValueError("input_size must contain a positive height and width")
        if patch_size <= 0 or any(size % patch_size for size in input_size):
            raise ValueError("Input height and width must be divisible by patch_size")
        if num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if hidden_size <= 0 or hidden_size % 4:
            raise ValueError("2D sinusoidal position embeddings require hidden_size divisible by four")
        if use_rope and (hidden_size // num_heads) % 4:
            raise ValueError("2D RoPE requires a head dimension divisible by four")
        self.input_size = tuple(input_size)
        self.patch_size = patch_size
        self.grid_size = tuple(size // patch_size for size in input_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # patch embedding using conv2D
        self.x_embedder = nn.Conv2d(
            in_channels, hidden_size, patch_size, stride=patch_size, bias=True
        )
        # Fixed position embeddings match the row-major order of flattened patches.
        self.register_buffer(
            "pos_embed", self.build_2d_pos_embed(hidden_size, self.grid_size)
        )

        # time embedding
        self.t_embedder = TimestepEmbedder(hidden_size)
        # label / condition embedding
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob
        )
        # Optional rotary embeddings supplement the token-level position embeddings.
        grid_size = self.grid_size

        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size,
                num_heads,
                grid_size,
                mlp_ratio=mlp_ratio,
                use_qknorm=use_qknorm,
                use_rmsnorm=use_rmsnorm,
                use_rope=use_rope,
                wo_shift=wo_shift,
                ) for _ in range(depth)
        ])
        self.final_layer = DiTDecodingLayer(
            hidden_size, patch_size, self.out_channels, use_rmsnorm=use_rmsnorm
        )

        # initialize all parameters
        self.initialize_weights()

    @staticmethod
    def build_2d_pos_embed(hidden_size, grid_size, max_period=10000):
        """Return (1, H*W, D) embeddings: column features followed by row features."""
        if hidden_size <= 0 or hidden_size % 4:
            raise ValueError("2D position embeddings require hidden_size divisible by four")
        height, width = grid_size
        quarter_dim = hidden_size // 4
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(quarter_dim, dtype=torch.float32)
            / quarter_dim
        )
        rows, columns = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        column_angles = columns.reshape(-1, 1) * freqs
        row_angles = rows.reshape(-1, 1) * freqs
        return torch.cat(
            [column_angles.sin(), column_angles.cos(), row_angles.sin(), row_angles.cos()],
            dim=-1,
        ).unsqueeze(0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        # non-standard, but OK for last layer
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h, w = self.grid_size
        if x.shape[1] != h * w:
            raise ValueError("Token count does not match the configured patch grid")

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, y=None, t=None):
        """
        Forward pass of LightningDiT.
        Args:
            x: (N, C, H, W) tensor of spatial inputs (images or latents)
            t: (N,) tensor of diffusion timesteps
            y: (N,) tensor of class labels

        Returns:
            x: (N, C, H, W) tensor of spatial outputs
        """

        if tuple(x.shape[1:]) != (self.in_channels, *self.input_size):
            raise ValueError(
                f"Expected input shape (N, {self.in_channels}, "
                f"{self.input_size[0]}, {self.input_size[1]}), got {tuple(x.shape)}"
            )
        # (N, C, H, W)
        x = self.x_embedder(x)
        # (N, C, H, W) -> (N, T, D)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed.to(dtype=x.dtype)
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        
        """
        Fill in the missing code here. See details in DiT paper.
        """

        return x
