"""Sweep 20: Multi-seed runs for statistical significance.

The ablation in sweep19 showed noisy results (run-to-run variance ~0.02).
For NeurIPS, we need proper error bars.

Strategy: Run 3 seeds each for:
1. Full best config (reference)
2. Original baseline (all improvements removed)
3. -coupledWD only
4. -lowerWD only (WD=0.1)
5. Audio ratio 0.1, 0.3 (fix from sweep19)

This gives us 3 measurements per config for mean ± std.
"""
import subprocess, re, os

PYTHON = ".venv/bin/python"

SEEDS = [42, 137, 2024]

def run_exp(desc, train_mods=None, prepare_mods=None):
    with open("train.py") as f:
        original_train = f.read()
    with open("prepare.py") as f:
        original_prepare = f.read()
    modified_train = original_train
    modified_prepare = original_prepare
    if train_mods:
        for old, new in train_mods:
            if modified_train.count(old) == 0:
                print(f"  WARNING: train.py not found: {old[:80]}...")
                with open("train.py", "w") as f:
                    f.write(original_train)
                with open("prepare.py", "w") as f:
                    f.write(original_prepare)
                return None
            modified_train = modified_train.replace(old, new, 1)
    if prepare_mods:
        for old, new in prepare_mods:
            if modified_prepare.count(old) == 0:
                print(f"  WARNING: prepare.py not found: {old[:80]}...")
                with open("train.py", "w") as f:
                    f.write(original_train)
                with open("prepare.py", "w") as f:
                    f.write(original_prepare)
                return None
            modified_prepare = modified_prepare.replace(old, new, 1)
    with open("train.py", "w") as f:
        f.write(modified_train)
    if prepare_mods:
        with open("prepare.py", "w") as f:
            f.write(modified_prepare)
    os.system(f'git add train.py prepare.py && git commit -m "exp: {desc}" --allow-empty 2>/dev/null')
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"\n{'='*60}\nEXP {commit}: {desc}\n{'='*60}")
    try:
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original_train)
        with open("prepare.py", "w") as f:
            f.write(original_prepare)
        return None
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)
    if not vl or result.returncode != 0:
        print(f"  CRASHED")
        print(output[-500:] if output else "no output")
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original_train)
        with open("prepare.py", "w") as f:
            f.write(original_prepare)
        return None
    val_loss, text_bpb, audio_loss = float(vl.group(1)), float(tb.group(1)), float(al.group(1))
    print(f"  val_loss={val_loss:.6f} text_bpb={text_bpb:.6f} audio_loss={audio_loss:.6f}")
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{text_bpb:.6f}\t{audio_loss:.6f}\t16.3\tkeep\t{desc}\n")
    with open("train.py", "w") as f:
        f.write(original_train)
    with open("prepare.py", "w") as f:
        f.write(original_prepare)
    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}


SEED_OLD = "torch.manual_seed(42)\ntorch.cuda.manual_seed(42)"
def seed_mod(s):
    return (SEED_OLD, f"torch.manual_seed({s})\ntorch.cuda.manual_seed({s})")

# Configs to test with multiple seeds
configs = {
    "full_best": [],
    "original_baseline": [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ],
    "-coupledWD": [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
    ],
    "-lowerWD": [
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
    ],
}

experiments = []

# Multi-seed runs for each config
for config_name, train_mods in configs.items():
    for seed in SEEDS:
        mods = list(train_mods) + [seed_mod(seed)]
        experiments.append((f"ms_{config_name}_s{seed}", mods, None))

# Audio ratio experiments (fix: modify prepare.py, not train.py)
for ratio in [0.1, 0.3]:
    for seed in SEEDS[:1]:  # just 1 seed for these
        experiments.append((
            f"audio_ratio_{ratio}_s{seed}",
            [seed_mod(seed)],
            [("AUDIO_MIX_RATIO = 0.2", f"AUDIO_MIX_RATIO = {ratio}")]
        ))

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

# Aggregate by config
print("\n" + "="*60)
print("SWEEP 20 MULTI-SEED SUMMARY")
print("="*60)

from collections import defaultdict
import statistics

config_results = defaultdict(list)
for desc, r in results:
    if r:
        # Extract config name (strip seed suffix)
        config = desc.rsplit("_s", 1)[0]
        config_results[config].append(r)

print(f"\n  {'config':<25} | {'val_loss':>12} | {'text_bpb':>12} | {'audio':>12} | n")
print(f"  {'-'*25} | {'-'*12} | {'-'*12} | {'-'*12} | -")
for config in configs:
    key = f"ms_{config}"
    runs = config_results.get(key, [])
    if len(runs) >= 2:
        vl = [r['val_loss'] for r in runs]
        tb = [r['text_bpb'] for r in runs]
        al = [r['audio_loss'] for r in runs]
        print(f"  {config:<25} | {statistics.mean(vl):.4f}±{statistics.stdev(vl):.4f} | "
              f"{statistics.mean(tb):.4f}±{statistics.stdev(tb):.4f} | "
              f"{statistics.mean(al):.4f}±{statistics.stdev(al):.4f} | {len(runs)}")
    elif len(runs) == 1:
        r = runs[0]
        print(f"  {config:<25} | {r['val_loss']:.4f}         | {r['text_bpb']:.4f}         | "
              f"{r['audio_loss']:.4f}         | 1")
    else:
        print(f"  {config:<25} | FAIL")

# Audio ratio results
for ratio in [0.1, 0.3]:
    key = f"audio_ratio_{ratio}"
    runs = config_results.get(key, [])
    if runs:
        r = runs[0]
        print(f"  {'audio_ratio='+str(ratio):<25} | {r['val_loss']:.4f}         | {r['text_bpb']:.4f}         | "
              f"{r['audio_loss']:.4f}         | 1")

print("\nAll individual runs:")
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
