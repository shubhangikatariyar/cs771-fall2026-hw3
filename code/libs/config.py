from copy import deepcopy
import yaml


ARCH_DEFAULTS = {
    "UNet": {
        "dim": 64,
        "context_dim": 64,
        "dim_mults": (1, 2, 4),
        "attn_levels": (0, 1),
        # used for CFG, keep it as is to reuse the trained model
        "class_dropout_prob": 0.1,
    },
    "ViT": {
        "patch_size": 2,
        "hidden_size": 192,
        "depth": 8,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        # used for CFG, keep it as is to reuse the trained model
        "class_dropout_prob": 0.1,
    },
}

# DEFAULT defines the default params used for training / inference
# the parameters here will be overwritten if a yaml config is specified
DEFAULTS = {
    # default: single gpu
    "devices": ["cuda:0"],
    "dtype": "fp32",
    # DDPM or FM
    "model_type": "DDPM",
    # output folder that stores all log files and checkpoints
    "output_folder": None,
    "dataset": {
        "name": "MNIST",
        # training split
        "train_split": "train",
        "test_split": "test",
        "data_folder": None,
    },
    "loader": {
        # batch size
        "batch_size": 64,
        # number of workers for data fetching
        "num_workers": 4,
    },

    # generation model; backbone settings live in arch_cfg
    "model": {
        # shape of input image (C x H W)
        "img_shape": None,
        # number of timesteps in the diffusion process
        "timesteps": 1000,
        # number of classes used for conditioning
        "num_classes": None,
        "arch_type": "UNet",
        "arch_cfg": {},
        # if to use VAE for latent diffusion model
        "use_vae": False,
        "vae_encoder_weights": None,
        "vae_decoder_weights": None,
    },
    # training config
    "train_cfg": {
        # learning rate
        "lr": 1e-4,
        # number of epochs in training
        "epochs": 30,
        # #samples (per category) to visualize when evaluating the model
        "num_eval_viz_samples": 10,
        # Maximum images generated per call during training visualization
        "viz_batch_size": 20,
        # ema decay
        "ema_decay": 0.9996,
        # warm up epochs
        "warmup_epochs": 1,
    },
    # test config
    "test_cfg": {
        # optional FID reference folder; None computes FID from dataloader
        "test_data_folder": None,
        # Optional .npz override; otherwise use data_folder/fid_stats/fid-<hash>.npz
        "fid_stats_path": None,
        # #samples (per category) to draw for evaluating the model
        "num_eval_samples": 100,
        # Maximum images generated per call, independent of the FID loader
        "sample_batch_size": 50,
        "cfg": True,
        "cfg_scale": 2.5,
        # fraction of the trajectory from noise to image before guidance starts
        "cfg_start_t": 0.1,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict) and isinstance(dst[k], dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = deepcopy(v)

def load_default_config():
    return _complete_config(deepcopy(DEFAULTS))

def _complete_config(config):
    model = config["model"]
    arch_type = model["arch_type"]
    if arch_type not in ARCH_DEFAULTS:
        raise ValueError(f"Unsupported model architecture: {arch_type!r}")
    _merge(ARCH_DEFAULTS[arch_type], model["arch_cfg"])
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.safe_load(fd) or {}
    _merge(defaults, config)
    return _complete_config(config)
