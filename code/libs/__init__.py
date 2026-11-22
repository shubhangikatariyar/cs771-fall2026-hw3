from .config import load_config
from .datasets import build_dataset, build_dataloader
from .models import build_model
from .utils import save_checkpoint, ModelEMA, AverageMeter


def compute_fid_score(*args, **kwargs):
    # SciPy/Inception are only needed for FID, not for model construction/training.
    from .fid_score import compute_fid_score as compute
    return compute(*args, **kwargs)

__all__ = [
    "load_config",
    "build_dataset",
    "build_dataloader",
    "build_model",
    "save_checkpoint",
    "ModelEMA",
    "AverageMeter",
    "compute_fid_score"
]
