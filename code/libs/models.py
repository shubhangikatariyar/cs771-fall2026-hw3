from .ddpm import DDPM
from .fm import FM

def build_model(model_type, model_cfg):
    """a help function to create DDPM / FM models"""
    if model_type == "DDPM":
        model = DDPM(**model_cfg)
        return model
    elif model_type == "FM":
        model = FM(**model_cfg)
        return model
    else:
        raise ValueError(f"Unsupported model type: {model_type!r}")
