"""Plot convergence curves from sweep21 data."""
import json, glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load all convergence files
curves = {}
for f in sorted(glob.glob("convergence_conv_*.json")):
    with open(f) as fh:
        data = json.load(fh)
    label = data["desc"].replace("conv: ", "")
    curves[label] = data

# Color scheme
colors = {
    "full_best_5min": "#2196F3",       # blue
    "original_baseline_5min": "#F44336", # red
    "only_coupledWD_5min": "#4CAF50",    # green
    "only_lowerWD_5min": "#FF9800",      # orange
    "only_mom97_5min": "#9C27B0",        # purple
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for label, data in curves.items():
    c = data["curve"]
    steps = [p["step"] for p in c]
    color = colors.get(label, "gray")

    axes[0].plot(steps, [p["val_loss"] for p in c], label=label, color=color, linewidth=1.5)
    axes[1].plot(steps, [p["text_bpb"] for p in c], label=label, color=color, linewidth=1.5)
    axes[2].plot(steps, [p["audio_loss"] for p in c], label=label, color=color, linewidth=1.5)

axes[0].set_ylabel("val_loss")
axes[1].set_ylabel("text_bpb")
axes[2].set_ylabel("audio_loss")

for ax in axes:
    ax.set_xlabel("step")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

plt.suptitle("Omni Model Convergence: Optimizer Component Ablation", fontsize=14)
plt.tight_layout()
plt.savefig("convergence_curves.png", dpi=150, bbox_inches="tight")
print("Saved convergence_curves.png")

# Also make a zoomed-in version focusing on the last 50% of training
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for label, data in curves.items():
    c = data["curve"]
    mid = len(c) // 2
    steps = [p["step"] for p in c[mid:]]
    color = colors.get(label, "gray")
    axes2[0].plot(steps, [p["val_loss"] for p in c[mid:]], label=label, color=color, linewidth=1.5)
    axes2[1].plot(steps, [p["text_bpb"] for p in c[mid:]], label=label, color=color, linewidth=1.5)
    axes2[2].plot(steps, [p["audio_loss"] for p in c[mid:]], label=label, color=color, linewidth=1.5)

axes2[0].set_ylabel("val_loss")
axes2[1].set_ylabel("text_bpb")
axes2[2].set_ylabel("audio_loss")
for ax in axes2:
    ax.set_xlabel("step")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
plt.suptitle("Convergence (Last 50% of Training) — Zoom", fontsize=14)
plt.tight_layout()
plt.savefig("convergence_curves_zoom.png", dpi=150, bbox_inches="tight")
print("Saved convergence_curves_zoom.png")
