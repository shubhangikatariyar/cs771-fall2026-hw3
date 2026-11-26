"""Figures for the U-Net flow matching runs (MNIST and MiniImageNet), the U-Net
CFG grids, and a sampling-time comparison. Run from the repo root:

    TORCH_COMPILE_DISABLE=1 python writeup/make_unet_figures.py

Needs the final checkpoints in results/ (kept locally, not in git).
"""
import glob
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.utils import save_image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "writeup", "figures")
RES = os.path.join(ROOT, "results")
pattern = re.compile(r"Iter (\d+).*Loss ([\d.]+) \(([\d.]+)\)")


def read_log(path):
    with open(path) as f:
        rows = [m.groups() for m in map(pattern.search, f) if m]
    return [int(r[0]) for r in rows], [float(r[1]) for r in rows]


def read_tb(folder):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(folder)
    acc.Reload()
    s = acc.Scalars("training/loss")
    return [e.step for e in s], [e.value for e in s]


def smooth(v, w):
    return [sum(v[i - w + 1:i + 1]) / w for i in range(w - 1, len(v))]


def loss_plot(series, iters_per_epoch, title, out, w=20):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    for name, (steps, values), color in series:
        ep = [s / iters_per_epoch for s in steps]
        ax.plot(ep, values, color=color, alpha=0.25, lw=1)
        ax.plot(ep[w - 1:], smooth(values, w), color=color, lw=2,
                label=f"{name} (mean of last {w} logged values)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow matching loss")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out))
    plt.close(fig)


def crop_first_columns(src, out, cols=10):
    """MiniImageNet grids have 100 columns (one per class); keep the first `cols`."""
    img = Image.open(src)
    tile = 130  # 128 px + 2 px padding
    img.crop((0, 0, cols * tile + 2, img.height)).save(os.path.join(FIG, out))


# ---------------------------------------------------------------- loss curves
unet_mnist = read_log(os.path.join(RES, "mnist_fm_unet_gcp", "train_log.txt"))
vit_mnist = read_tb(glob.glob(os.path.join(RES, "mnist_fm_vit_*", "logs"))[0])
loss_plot([("U-Net", unet_mnist, "#2a78d6"), ("ViT", vit_mnist, "#eb6834")],
          938, "Flow matching on MNIST", "fm_mnist_loss.png", w=5)

unet_mini = read_log(os.path.join(RES, "miniimagenet_fm_unet_gcp", "train_log.txt"))
vit_mini = read_log(glob.glob(os.path.join(RES, "miniimagenet_fm_vit_*", "job_output.txt"))[0])
loss_plot([("U-Net", unet_mini, "#2a78d6"), ("ViT", vit_mini, "#eb6834")],
          781, "Latent flow matching on MiniImageNet", "mini_fm_loss.png")

# ---------------------------------------------------------------- sample grids
Image.open(os.path.join(RES, "mnist_fm_unet_gcp", "sample-29.png")).save(
    os.path.join(FIG, "fm_unet_samples.png"))
crop_first_columns(os.path.join(RES, "miniimagenet_fm_unet_gcp", "sample-79.png"), "mini_unet_samples.png")
crop_first_columns(os.path.join(RES, "miniimagenet_fm_unet_gcp", "sample-3.png"), "mini_unet_samples_epoch4.png")

if os.environ.get("PLOTS_ONLY") == "1":
    sys.exit(0)

# ---------------------------------------------------------------- models
sys.path.insert(0, os.path.join(ROOT, "code"))
os.chdir(os.path.join(ROOT, "code"))  # configs use paths relative to code/
from libs import build_model, load_config  # noqa: E402

device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def load(cfg_name, ckpt, num_classes, img_shape):
    cfg = load_config(os.path.join("configs", cfg_name))
    cfg["model"]["num_classes"] = num_classes
    cfg["model"]["img_shape"] = img_shape
    model = build_model(cfg["model_type"], cfg["model"]).to(device).eval()
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict_ema"])
    return model


def sync():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def time_generate(model, labels, repeats=3, **kw):
    model.generate(labels[:2], **kw)  # warm-up on a tiny batch
    sync()
    start = time.perf_counter()
    for _ in range(repeats):
        model.generate(labels, **kw)
    sync()
    return (time.perf_counter() - start) / repeats


mini = load("miniimagenet_fm_unet.yaml",
            os.path.join(RES, "miniimagenet_fm_unet_gcp", "model_final.pth.tar"), 100, (3, 128, 128))

# CFG grids: classes 0-9 in columns, 5 rows, same starting noise for every scale.
labels = torch.arange(10, device=device).repeat(5)
with torch.no_grad():
    for scale in ((1.0, 2.5, 4.0) if os.environ.get("TIMING_ONLY") != "1" else ()):
        torch.manual_seed(0)
        imgs = mini.generate(labels, cfg=scale != 1.0, cfg_scale=scale, cfg_start_t=0.1)
        save_image(imgs.cpu(), os.path.join(FIG, f"cfg_unet_s{scale}.png"), nrow=10)

# Sampling time for one batch of 100 labels (one per class).
lines = [f"device: {device}"]
mnist_labels = torch.arange(10, device=device).repeat(10)
for name, cfg_name, run in [("DDPM U-Net", "mnist_ddpm_unet.yaml", "mnist_ddpm_unet_*"),
                            ("FM U-Net", "mnist_fm_unet.yaml", "mnist_fm_unet_gcp"),
                            ("DDPM ViT", "mnist_ddpm_vit.yaml", "mnist_ddpm_vit_*"),
                            ("FM ViT", "mnist_fm_vit.yaml", "mnist_fm_vit_*")]:
    ckpt = glob.glob(os.path.join(RES, run, "model_final.pth.tar"))[0]
    model = load(cfg_name, ckpt, 10, (1, 28, 28))
    sec = time_generate(model, mnist_labels, repeats=1 if "DDPM" in name else 5)
    lines.append(f"MNIST {name}: {sec:.2f} s per 100 images")
    print(lines[-1], flush=True)
    del model
mini_labels = torch.arange(100, device=device)
print("MNIST done; timing MiniImageNet", flush=True)
lines.append(f"MiniImageNet FM U-Net, no CFG: {time_generate(mini, mini_labels):.2f} s per 100 images")
lines.append("MiniImageNet FM U-Net, CFG 2.5: "
             f"{time_generate(mini, mini_labels, cfg=True, cfg_scale=2.5, cfg_start_t=0.1):.2f} s per 100 images")
mini_vit = load("miniimagenet_fm_vit.yaml",
                os.path.join(RES, glob.glob(os.path.join(RES, "miniimagenet_fm_vit_*"))[0], "model_final.pth.tar"),
                100, (3, 128, 128)) if glob.glob(os.path.join(RES, "miniimagenet_fm_vit_*", "model_final.pth.tar")) else None
if mini_vit is not None:
    lines.append(f"MiniImageNet FM ViT, no CFG: {time_generate(mini_vit, mini_labels):.2f} s per 100 images")
out = os.path.join(RES, "sampling_time.txt")
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
