"""Backbone construction, independent of the diffusion and flow wrappers."""

from copy import deepcopy

from .config import ARCH_DEFAULTS
from .unet import UNet
from .vit import LightningDiT


def build_arch(arch_type="UNet", arch_cfg=None, *, img_shape=None, num_classes=None):
    """Build a backbone using the effective image/latent shape and class count.

    Metadata supplied by a generation model takes precedence over arch_cfg.
    Neither the caller's configuration nor the architecture defaults are mutated.
    """
    if arch_type not in ARCH_DEFAULTS:
        raise ValueError(f"Unsupported model architecture: {arch_type!r}")
    cfg = deepcopy(ARCH_DEFAULTS[arch_type])
    cfg.update(deepcopy(arch_cfg or {}))
    if num_classes is not None:
        cfg["num_classes"] = num_classes
    if img_shape is not None:
        cfg["in_channels"] = img_shape[0]
        if arch_type == "UNet":
            cfg["out_channels"] = img_shape[0]
        else:
            cfg["input_size"] = tuple(img_shape[1:])
    if arch_type == "UNet":
        return UNet(**cfg)
    return LightningDiT(**cfg)
