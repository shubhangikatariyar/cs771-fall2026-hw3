# python imports
import argparse
import os
import time
import glob
import tempfile
from pathlib import Path
from pprint import pprint

# torch imports
import torch
import torch.backends.cudnn as cudnn

# for visualization
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm

# our code
from libs import (
    build_dataset,
    build_dataloader,
    load_config,
    build_model,
    compute_fid_score
)


################################################################################
def main(args):
    """Generate samples and optionally compute FID against reference images."""

    """1. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    sample_batch_size = cfg["test_cfg"]["sample_batch_size"]
    if type(sample_batch_size) is not int or sample_batch_size <= 0:
        raise ValueError("test_cfg.sample_batch_size must be a positive integer")

    fid_data_folder = cfg["test_cfg"]["test_data_folder"]
    if fid_data_folder is not None and not os.path.isdir(fid_data_folder):
        raise ValueError(f"FID reference directory does not exist: {fid_data_folder}")

    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch >= 0:
            ckpt_file = os.path.join(
                args.ckpt, "epoch_{:03d}.pth.tar".format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, "*.pth.tar")))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    """2. create model"""
    # dataset and update model parameters
    test_dataset, num_classes, img_shape = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["test_split"],
        cfg["dataset"]["data_folder"],
        use_vae=cfg["model"]["use_vae"],
    )
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["img_shape"] = img_shape
    pprint(cfg)
    if fid_data_folder is None:
        # data loaders
        test_loader = build_dataloader(test_dataset, **cfg["loader"], shuffle=False)

    # model
    model = build_model(
        cfg['model_type'], cfg['model']
    ).to(torch.device(cfg["devices"][0]))
    # set model to evaluation
    model.eval()
    # enable cudnn benchmark
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    """3. load ckpt and setup output folder"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt
    checkpoint = torch.load(
        ckpt_file,
        map_location=cfg["devices"][0],
        weights_only=True
    )
    print("Loading model ...")
    # load the EMA version of the model
    model.load_state_dict(checkpoint["state_dict_ema"])
    del checkpoint

    # Keep each evaluation's images separate so FID cannot include old samples.
    output_root = os.path.join(
        os.path.dirname(ckpt_file),
        os.path.basename(ckpt_file).removesuffix(".pth.tar")
    )
    os.makedirs(output_root, exist_ok=True)
    output_folder = tempfile.mkdtemp(prefix="eval_", dir=output_root)
    print(f"Saving generated images to {output_folder}")
    for cat_idx in range(num_classes):
        os.makedirs(
            os.path.join(output_folder, 'cat-{:d}'.format(cat_idx)),
            exist_ok=True
        )

    """4. Generating Images for Evaluation"""
    print("\nGenerating images ...")

    # draw samples from the model
    device = torch.device(cfg["devices"][0])
    sample_labels = torch.arange(0, num_classes, dtype=torch.long).to(device)

    for sample_idx in tqdm(range(cfg["test_cfg"]["num_eval_samples"])):
        # Samples are drawn from EMA version of the model.
        for labels_chunk in sample_labels.split(sample_batch_size):
            imgs = model.generate(
                labels_chunk,
                cfg=cfg["test_cfg"]["cfg"],
                cfg_scale=cfg["test_cfg"]["cfg_scale"],
                cfg_start_t=cfg["test_cfg"]["cfg_start_t"],
            ).cpu()
            # Use original class IDs; chunk-local indices would overwrite files.
            for cat_idx, img in zip(labels_chunk.tolist(), imgs):
                output_file = os.path.join(
                    output_folder,
                    'cat-{:d}'.format(cat_idx),
                    '{:d}.png'.format(sample_idx)
                )
                save_image(img, output_file)

    print("\nEvaluating FID scores ...")
    fid_score = compute_fid_score(
        test_loader if fid_data_folder is None else fid_data_folder,
        output_folder,
        stats_path=cfg["test_cfg"]["fid_stats_path"],
        stats_dir=Path(cfg["dataset"]["data_folder"]) / "fid_stats",
        target_size=min(img_shape[1:]),
        batch_size=cfg["loader"]["batch_size"],
        num_workers=cfg["loader"]["num_workers"],
    )
    print("FID: ", fid_score)
    return output_folder

################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Evaluate DDPM/FM for Image Generation"
    )
    parser.add_argument("config", type=str, metavar="DIR", help="path to a config file")
    parser.add_argument("ckpt", type=str, metavar="DIR", help="path to a checkpoint")
    parser.add_argument("-e", "--epoch", type=int, default=-1, help="checkpoint epoch")
    args = parser.parse_args()
    main(args)
