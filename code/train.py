# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.backends.cudnn as cudnn

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

# for visualization
import torchvision
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# our code
from libs import (
    load_config,
    build_dataset,
    build_dataloader,
    build_model,
    save_checkpoint,
    ModelEMA,
    AverageMeter
)


################################################################################
def main(args):
    """main function that handles training"""

    """1. Load config / Setup folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    viz_batch_size = cfg["train_cfg"]["viz_batch_size"]
    if type(viz_batch_size) is not int or viz_batch_size <= 0:
        raise ValueError("train_cfg.viz_batch_size must be a positive integer")

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg["output_folder"]):
        os.mkdir(cfg["output_folder"])
    cfg_filename = os.path.basename(args.config).replace(".yaml", "")
    suffix = args.output or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_folder = os.path.join(
        cfg['output_folder'], os.path.splitext(os.path.basename(args.config))[0] + '_' + suffix
    )
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, "logs"))

    """2. create dataset / dataloader"""
    # dataset and update model parameters
    train_dataset, num_classes, img_shape = build_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["train_split"],
        cfg["dataset"]["data_folder"],
        use_vae=cfg["model"]["use_vae"],
    )
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["img_shape"] = img_shape
    pprint(cfg)
    # data loaders
    train_loader = build_dataloader(train_dataset, **cfg["loader"])

    """3. create model and optimizer"""
    # model
    model = build_model(
        cfg['model_type'], cfg['model']
    ).to(torch.device(cfg["devices"][0]))
    # set model to training
    model.train()
    # optimizer Adam (without weight decay)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train_cfg"]["lr"],
        betas = (0.9, 0.95)
    )
    # enable cudnn benchmark
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # enable model EMA (appendix B of DDPM paper)
    model_ema = ModelEMA(model, decay=cfg["train_cfg"]["ema_decay"])

    # Create the scaler before resuming so its loss-scale history can be restored.
    scaler = GradScaler() if cfg["dtype"] == "fp16" else None
    if scaler is not None:
        print("Using mixed precision training")

    """4. Resume from model / Misc"""
    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(
                args.resume,
                map_location=cfg["devices"][0],
                weights_only=True
            )
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint["optimizer"])
            # Older checkpoints and disabled scalers have no state to restore.
            if scaler is not None and checkpoint.get("scaler"):
                scaler.load_state_dict(checkpoint["scaler"])
            print(
                "=> loaded checkpoint '{:s}' (epoch {:d})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, "config.txt"), "w") as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training loop"""
    print("\nStart training ...")

    # start training
    max_epochs = cfg["train_cfg"]["epochs"]
    device = torch.device(cfg["devices"][0])

    start = time.time()
    batch_time = AverageMeter()
    loss_tracker = AverageMeter()
    num_iters_per_epoch = len(train_loader)

    # As DDPM training samples the time step for each sample, epoch only controls
    # the total number of iterations (and not the actual epochs)
    for epoch in range(args.start_epoch, max_epochs):
        # reset the iters (so that resuming is possible)
        total_iters = num_iters_per_epoch * epoch
        warmup_epochs = cfg["train_cfg"]["warmup_epochs"]

        # adjust the learning rate
        lr = cfg["train_cfg"]["lr"]
        if epoch < cfg["train_cfg"]["warmup_epochs"]:
            lr *= (epoch + 1) / warmup_epochs

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # the main training loop
        for iter_idx, batch in enumerate(train_loader):
            # fetching imgs and their labels
            img, label = batch
            img = img.to(device)
            label = label.to(device)

            # zero out the optmizer
            optimizer.zero_grad()

            # if mixed precision training
            if scaler != None:
                # compute loss
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = model.compute_loss(img, label)
                # update the optimizer / scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # compute loss
                loss = model.compute_loss(img, label)

                # backward loss and update the parameters
                loss.backward()
                optimizer.step()

            # update the EMA version of the model
            model_ema.update(model)
            total_iters += 1

            # print the loss values
            if (total_iters % args.print_freq) == 0:
                # measure elapsed time (sync all kernels)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                batch_time.update((time.time() - start) / args.print_freq)
                start = time.time()
                # record the loss value
                loss_tracker.update(loss.item())
                # log to tensorboard
                tb_writer.add_scalar(
                    "training/loss", loss_tracker.val, total_iters
                )
                # print to terminal
                block1 = "Iter {:08d}".format(total_iters)
                block2 = "Time {:.2f} ({:.2f})".format(
                    batch_time.val, batch_time.avg
                )
                block3 = "LR {:0.5f}".format(lr)
                block4 = "Loss {:.3f} ({:.3f})".format(
                    loss_tracker.val, loss_tracker.avg
                )
                print("\t".join([block1, block2, block3, block4]))

        # save ckpt / draw samples once in a while
        if ((epoch + 1) == max_epochs) or (
            (args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0)
        ):
            save_states = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "state_dict_ema": model_ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                save_states["scaler"] = scaler.state_dict()
            is_final = (epoch + 1) == max_epochs
            save_checkpoint(
                save_states,
                is_final,
                file_folder=ckpt_folder,
                file_name="epoch_{:03d}.pth.tar".format(epoch),
            )

            # To save some memory, release previous training step gradients before sampling.
            optimizer.zero_grad(set_to_none=True)

            # Generate every class, with a bounded batch size for any dataset.
            sample_labels = torch.arange(0, num_classes, dtype=torch.long).to(device)
            all_imgs = []

            for _ in range(cfg["train_cfg"]["num_eval_viz_samples"]):
                # Samples are drawn from EMA version of the model.
                # This will initially produce samples lagged behind current epoch,
                # yet eventually stablize samples during late stage of training
                for labels_chunk in sample_labels.split(viz_batch_size):
                    all_imgs.append(model_ema.module.generate(labels_chunk).cpu())

            # Assemble and save the grid on CPU to avoid retaining GPU images.
            all_imgs = torch.cat(all_imgs, dim=0)
            save_image(
                all_imgs,
                os.path.join(ckpt_folder, 'sample-{:d}.png'.format(epoch)),
                nrow=cfg["model"]["num_classes"]
            )

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == "__main__":
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description="Train DDPM/FM for Image Generation"
    )
    parser.add_argument("config", metavar="DIR", help="path to a config file")
    parser.add_argument(
        "-p",
        "--print-freq",
        default=100,
        type=int,
        help="print frequency (default: 100 iterations)",
    )
    parser.add_argument(
        "-c",
        "--ckpt-freq",
        default=5,
        type=int,
        help="checkpoint frequency (default: every 5 epochs)",
    )
    parser.add_argument(
        "--output", default="", type=str, help="name of exp folder (default: none)"
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to a checkpoint (default: none)",
    )
    args = parser.parse_args()
    main(args)
