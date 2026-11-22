"""Shared sampling mode and classifier-free guidance validation."""

from functools import wraps
import math


def sampling_mode(generate):
    """Disable training-time label dropout while sampling, then restore modes."""
    @wraps(generate)
    def wrapped(self, *args, **kwargs):
        modes = [(module, module.training) for module in self.model.modules()]
        self.model.eval()
        try:
            return generate(self, *args, **kwargs)
        finally:
            for module, training in modes:
                module.training = training
    return wrapped


def validate_guidance(model, cfg_scale, cfg_start_t):
    if cfg_scale is None or not math.isfinite(cfg_scale) or cfg_scale < 1.0:
        raise ValueError("cfg_scale must be finite and at least 1.0")
    if not math.isfinite(cfg_start_t) or not 0.0 <= cfg_start_t <= 1.0:
        raise ValueError("cfg_start_t must be between 0.0 and 1.0")
    if model.class_dropout_prob <= 0:
        raise ValueError(
            "Classifier-free guidance requires a backbone trained with "
            "class_dropout_prob > 0"
        )
