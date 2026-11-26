"""Plot MiniImageNet latent FM (ViT) training loss. Run from the repo root."""
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ITERS_PER_EPOCH = 781  # 50,000 images / batch size 64
pattern = re.compile(r"Iter (\d+).*Loss ([\d.]+) \(([\d.]+)\)")

path = glob.glob("results/miniimagenet_fm_vit_*/job_output.txt")[0]
with open(path) as f:
    rows = [m.groups() for m in map(pattern.search, f) if m]
epochs = [int(r[0]) / ITERS_PER_EPOCH for r in rows]
values = [float(r[1]) for r in rows]
window = 20
smooth = [sum(values[i - window + 1:i + 1]) / window for i in range(window - 1, len(values))]

fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
ax.plot(epochs, values, color="#2a78d6", alpha=0.3, lw=1, label="logged every 100 iterations")
ax.plot(epochs[window - 1:], smooth, color="#2a78d6", lw=2, label="mean of last 20 logged values")
ax.set_xlabel("Epoch")
ax.set_ylabel("Flow matching loss")
ax.set_title("Latent FM with ViT on MiniImageNet")
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("writeup/figures/mini_vit_loss.png")
