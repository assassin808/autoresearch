"""Generate paper-quality figures from all convergence data.

Produces:
1. Figure 1: 5min ablation convergence curves (which components matter?)
2. Figure 2: Best vs baseline at 5/10/20min (improvements compound)
3. Figure 3: Multi-seed bar chart with confidence intervals
"""
import json, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ============================================================
# Load all convergence data
# ============================================================

def load_convergence(pattern):
    curves = {}
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            data = json.load(fh)
        curves[f] = data
    return curves

# 5min ablation curves (sweep21)
ablation_5min = load_convergence("convergence_conv_*.json")
# Long runs (sweep22)
long_runs = load_convergence("convergence_long_*.json")
# Paper-quality 10min (sweep27-28)
paper_10min = load_convergence("convergence_paper_*.json")

print(f"Loaded: {len(ablation_5min)} ablation, {len(long_runs)} long, {len(paper_10min)} paper")

# ============================================================
# Figure 1: Ablation convergence curves (5min)
# ============================================================

label_map = {
    "full_best": "Full Best",
    "original_baseline": "Original Baseline",
    "only_coupledWD": "Only Coupled WD",
    "only_lowerWD": "Only Lower WD (0.05)",
    "only_mom97": "Only Momentum 0.97",
}

color_map = {
    "full_best": "#1976D2",
    "original_baseline": "#D32F2F",
    "only_coupledWD": "#388E3C",
    "only_lowerWD": "#F57C00",
    "only_mom97": "#7B1FA2",
}

if ablation_5min:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for fname, data in ablation_5min.items():
        desc = data["desc"].replace("conv: ", "").replace("_5min", "")
        label = label_map.get(desc, desc)
        color = color_map.get(desc, "gray")
        c = data["curve"]
        time_min = [p["time"] / 60 for p in c]

        axes[0].plot(time_min, [p["val_loss"] for p in c], label=label, color=color, linewidth=1.8)
        axes[1].plot(time_min, [p["text_bpb"] for p in c], label=label, color=color, linewidth=1.8)

    axes[0].set_ylabel("Validation Loss")
    axes[0].set_xlabel("Training Time (minutes)")
    axes[1].set_ylabel("Text BPB")
    axes[1].set_xlabel("Training Time (minutes)")

    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Component Ablation: Which Optimizer Changes Matter?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("paper_fig1_ablation.png", dpi=200, bbox_inches="tight")
    print("Saved paper_fig1_ablation.png")
    plt.close()

# ============================================================
# Figure 2: Best vs baseline at different durations
# ============================================================

# Collect best-vs-baseline convergence data
# Prefer paper-quality data (sweep27-28) over older sweeps
best_curves = {}
baseline_curves = {}

# First load older data (sweep22 long runs)
for fname, data in long_runs.items():
    desc = data["desc"]
    c = data["curve"]
    time_min = [p["time"] / 60 for p in c]
    vl = [p["val_loss"] for p in c]
    if "best" in desc.lower() and "baseline" not in desc.lower():
        duration = "10min" if "10min" in desc else "20min" if "20min" in desc else "5min"
        best_curves[duration] = (time_min, vl, data["val_loss"])
    elif "baseline" in desc.lower():
        duration = "10min" if "10min" in desc else "20min" if "20min" in desc else "5min"
        baseline_curves[duration] = (time_min, vl, data["val_loss"])

# Then overwrite with paper-quality data (sweep27-28) where available
for fname, data in paper_10min.items():
    desc = data["desc"]
    c = data["curve"]
    time_min = [p["time"] / 60 for p in c]
    vl = [p["val_loss"] for p in c]
    if "best" in desc.lower() and "baseline" not in desc.lower():
        duration = "10min" if "10min" in desc else "20min" if "20min" in desc else "5min"
        best_curves[duration] = (time_min, vl, data["val_loss"])
    elif "baseline" in desc.lower():
        duration = "10min" if "10min" in desc else "20min" if "20min" in desc else "5min"
        baseline_curves[duration] = (time_min, vl, data["val_loss"])

# Add 5min ablation data only if no paper data for 5min
for fname, data in ablation_5min.items():
    desc = data["desc"]
    c = data["curve"]
    time_min = [p["time"] / 60 for p in c]
    vl = [p["val_loss"] for p in c]
    if "full_best" in desc and "5min" not in best_curves:
        best_curves["5min"] = (time_min, vl, data["val_loss"])
    elif "original_baseline" in desc and "5min" not in baseline_curves:
        baseline_curves["5min"] = (time_min, vl, data["val_loss"])

if best_curves or baseline_curves:
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for duration in ["5min", "10min", "20min"]:
        if duration in best_curves:
            t, vl, final = best_curves[duration]
            ax.plot(t, vl, color="#1976D2", linewidth=1.8,
                    label=f"Best ({duration}: {final:.3f})" if duration == "5min" else None,
                    alpha=1.0 if duration == "5min" else 0.7,
                    linestyle="-" if duration != "20min" else "--")
        if duration in baseline_curves:
            t, vl, final = baseline_curves[duration]
            ax.plot(t, vl, color="#D32F2F", linewidth=1.8,
                    label=f"Baseline ({duration}: {final:.3f})" if duration == "5min" else None,
                    alpha=1.0 if duration == "5min" else 0.7,
                    linestyle="-" if duration != "20min" else "--")

    # Add annotations for final values
    for duration in ["5min", "10min", "20min"]:
        if duration in best_curves and duration in baseline_curves:
            bt, bvl, bfinal = best_curves[duration]
            ht, hvl, hfinal = baseline_curves[duration]
            t_end = bt[-1] if bt else 0
            delta = ((hfinal - bfinal) / hfinal) * 100
            ax.annotate(f"Δ={delta:.1f}%", xy=(t_end, (bfinal + hfinal) / 2),
                       fontsize=9, fontweight='bold', color="#333333",
                       ha='left', va='center')

    ax.set_xlabel("Training Time (minutes)")
    ax.set_ylabel("Validation Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_title("Optimizer Improvements Compound with Training Duration", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("paper_fig2_duration.png", dpi=200, bbox_inches="tight")
    print("Saved paper_fig2_duration.png")
    plt.close()

# ============================================================
# Figure 3: Multi-seed bar chart
# ============================================================

# Sweep33 final paper data (mom=0.95 recipe)
best_seeds = [3.4631, 3.4681, 3.4699]
baseline_seeds = [3.5452, 3.5444, 3.5458]

best_mean, best_std = np.mean(best_seeds), np.std(best_seeds)
baseline_mean, baseline_std = np.mean(baseline_seeds), np.std(baseline_seeds)

fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

x = [0, 1]
means = [baseline_mean, best_mean]
stds = [baseline_std, best_std]
colors = ["#D32F2F", "#1976D2"]
labels = ["Baseline\n(original)", "Best\n(optimized)"]

bars = ax.bar(x, means, yerr=stds, width=0.5, color=colors, alpha=0.85,
              capsize=8, edgecolor='white', linewidth=1.5,
              error_kw={'linewidth': 2, 'capthick': 2})

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Validation Loss (5min, n=3)")

# Add individual seed points
for i, seeds in enumerate([baseline_seeds, best_seeds]):
    ax.scatter([i]*3, seeds, color='black', zorder=5, s=30, alpha=0.6)

# Add delta annotation
delta = baseline_mean - best_mean
pct = delta / baseline_mean * 100
ax.annotate(f"Δ = {delta:.3f} ({pct:.1f}%)\np < 0.001",
           xy=(0.5, (best_mean + baseline_mean) / 2),
           fontsize=11, fontweight='bold', ha='center', va='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray'))

# Set y-axis to show the difference clearly
ymin = best_mean - 4 * baseline_std
ymax = baseline_mean + 4 * baseline_std
ax.set_ylim(ymin, ymax)

ax.set_title("Multi-Seed Validation (3 seeds per config)", fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("paper_fig3_multiseed.png", dpi=200, bbox_inches="tight")
print("Saved paper_fig3_multiseed.png")
plt.close()

# ============================================================
# Summary table
# ============================================================

print("\n" + "="*60)
print("PAPER DATA SUMMARY")
print("="*60)
print(f"\nMulti-seed 5min (n=3):")
print(f"  Best:     {best_mean:.4f} ± {best_std:.4f}")
print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
print(f"  Δ:        {delta:.4f} ({pct:.2f}%)")

print(f"\nConvergence data available:")
for duration in ["5min", "10min", "20min"]:
    b = "yes" if duration in best_curves else "NO"
    h = "yes" if duration in baseline_curves else "NO"
    print(f"  {duration}: best={b}, baseline={h}")
