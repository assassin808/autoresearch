"""Sweep 33: Multi-seed validation of final recipe (with mom=0.95).

3 seeds × best (mom=0.95) at 5min and 10min.
This is the definitive paper data.
"""
import subprocess, re, os

PYTHON = ".venv/bin/python"

SEED_OLD = "torch.manual_seed(42)\ntorch.cuda.manual_seed(42)"


def run_exp(desc, train_mods=None, prepare_mods=None, timeout=600):
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
                with open("train.py", "w") as f: f.write(original_train)
                with open("prepare.py", "w") as f: f.write(original_prepare)
                return None
            modified_train = modified_train.replace(old, new, 1)
    if prepare_mods:
        for old, new in prepare_mods:
            if modified_prepare.count(old) == 0:
                print(f"  WARNING: prepare.py not found: {old[:80]}...")
                with open("train.py", "w") as f: f.write(original_train)
                with open("prepare.py", "w") as f: f.write(original_prepare)
                return None
            modified_prepare = modified_prepare.replace(old, new, 1)
    with open("train.py", "w") as f: f.write(modified_train)
    if prepare_mods:
        with open("prepare.py", "w") as f: f.write(modified_prepare)
    os.system(f'git add train.py prepare.py && git commit -m "exp: {desc}" --allow-empty 2>/dev/null')
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"\n{'='*60}\nEXP {commit}: {desc}\n{'='*60}")
    try:
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        with open("results.tsv", "a") as f: f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f: f.write(original_train)
        with open("prepare.py", "w") as f: f.write(original_prepare)
        return None
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)
    if not vl or result.returncode != 0:
        print(f"  CRASHED")
        print(output[-500:] if output else "no output")
        with open("results.tsv", "a") as f: f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f: f.write(original_train)
        with open("prepare.py", "w") as f: f.write(original_prepare)
        return None
    val_loss, text_bpb, audio_loss = float(vl.group(1)), float(tb.group(1)), float(al.group(1))
    print(f"  val_loss={val_loss:.6f} text_bpb={text_bpb:.6f} audio_loss={audio_loss:.6f}")
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{text_bpb:.6f}\t{audio_loss:.6f}\t16.3\tkeep\t{desc}\n")
    with open("train.py", "w") as f: f.write(original_train)
    with open("prepare.py", "w") as f: f.write(original_prepare)
    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}


import re as re2
with open("prepare.py") as f:
    m = re2.search(r'TIME_BUDGET\s*=\s*(\d+)', f.read())
    time_budget = int(m.group(1)) if m else 300

BASELINE_MODS = [
    ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
     "return WEIGHT_DECAY  # constant WD"),
    ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
    # Note: train.py now has momentum=0.95, baseline also uses 0.95
    ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ("MATRIX_LR = 0.03", "MATRIX_LR = 0.06"),
    ("TOTAL_BATCH_SIZE = 2**15", "TOTAL_BATCH_SIZE = 2**17"),
]

PREPARE_10MIN = [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")]

SEEDS = [42, 137, 2024]

experiments = []

# Multi-seed best at 5min
for seed in SEEDS:
    experiments.append((
        f"final: best_5min_s{seed}",
        [(SEED_OLD, f"torch.manual_seed({seed})\ntorch.cuda.manual_seed({seed})")],
        None, 600))

# Multi-seed baseline at 5min
for seed in SEEDS:
    experiments.append((
        f"final: baseline_5min_s{seed}",
        BASELINE_MODS + [(SEED_OLD, f"torch.manual_seed({seed})\ntorch.cuda.manual_seed({seed})")],
        None, 600))

# Multi-seed best at 10min
for seed in SEEDS:
    experiments.append((
        f"final: best_10min_s{seed}",
        [(SEED_OLD, f"torch.manual_seed({seed})\ntorch.cuda.manual_seed({seed})")],
        PREPARE_10MIN, 1800))

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods, to = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=to)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 33: FINAL MULTI-SEED VALIDATION")
print("="*60)

from collections import defaultdict
import statistics

config_results = defaultdict(list)
for desc, r in results:
    if r:
        if "best_5min" in desc:
            config_results["best_5min"].append(r)
        elif "baseline_5min" in desc:
            config_results["baseline_5min"].append(r)
        elif "best_10min" in desc:
            config_results["best_10min"].append(r)

print(f"\n  {'config':<15} | {'val_loss':>12} | {'text_bpb':>12} | {'audio':>12}")
print(f"  {'-'*15} | {'-'*12} | {'-'*12} | {'-'*12}")
for config in ["best_5min", "baseline_5min", "best_10min"]:
    runs = config_results[config]
    if len(runs) >= 2:
        vl = [r['val_loss'] for r in runs]
        tb = [r['text_bpb'] for r in runs]
        al = [r['audio_loss'] for r in runs]
        print(f"  {config:<15} | {statistics.mean(vl):.4f}±{statistics.stdev(vl):.4f} | "
              f"{statistics.mean(tb):.4f}±{statistics.stdev(tb):.4f} | "
              f"{statistics.mean(al):.4f}±{statistics.stdev(al):.4f}")

# Significance test
if len(config_results["best_5min"]) >= 3 and len(config_results["baseline_5min"]) >= 3:
    best_vl = [r['val_loss'] for r in config_results["best_5min"]]
    base_vl = [r['val_loss'] for r in config_results["baseline_5min"]]
    delta = statistics.mean(base_vl) - statistics.mean(best_vl)
    pct = delta / statistics.mean(base_vl) * 100
    print(f"\n  5min Δ = {delta:.4f} ({pct:.2f}%)")
