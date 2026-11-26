"""Plot MNIST DDPM training loss from the job logs. Run from the repo root."""
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ITERS_PER_EPOCH = 938  # 60,000 images / batch size 64
pattern = re.compile(r"Iter (\d+).*Loss ([\d.]+) \(([\d.]+)\)")

fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
for key, name, color in [("unet", "U-Net", "#2a78d6"), ("vit", "ViT", "#eb6834")]:
    path = glob.glob(f"results/mnist_ddpm_{key}_*/job_output.txt")[0]
    with open(path) as f:
        rows = [m.groups() for m in map(pattern.search, f) if m]
    epochs = [int(r[0]) / ITERS_PER_EPOCH for r in rows]
    ax.plot(epochs, [float(r[2]) for r in rows], color=color, lw=2, label=name)
    ax.annotate(f"{name} {float(rows[-1][2]):.3f}", (epochs[-1], float(rows[-1][2])),
                xytext=(6, 5 if key == "vit" else -5), textcoords="offset points", va="center", fontsize=9)

ax.set_yscale("log")
ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.4], ["0.03", "0.05", "0.1", "0.2", "0.4"])
ax.minorticks_off()
ax.set_xlim(0, 34.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training loss (running average, log scale)")
ax.set_title("DDPM on MNIST")
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("writeup/figures/ddpm_loss.png")
