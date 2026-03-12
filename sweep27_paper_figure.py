"""Sweep 27: Clean paper-quality comparison data.

Run best vs baseline at 5min, 10min, 20min with convergence logging.
This gives the cleanest possible data for the key paper figure:
"Optimizer improvements compound with training duration."

Also: multi-seed 5min runs of the LATEST best recipe to establish
proper confidence intervals.
"""
import subprocess, re, os, json

PYTHON = ".venv/bin/python"

CONVERGENCE_LOGGING_CODE = '''
EVAL_INTERVAL_STEPS = 100
convergence_log = []
'''

LOGGING_INJECTION = '''
    if step > 10 and step % EVAL_INTERVAL_STEPS == 0:
        model.eval()
        with torch.no_grad():
            with autocast_ctx:
                _vl, _tb, _al = evaluate_val_loss(model, tokenizer, DEVICE_BATCH_SIZE)
            convergence_log.append({"step": step, "time": total_training_time, "progress": progress,
                "val_loss": _vl, "text_bpb": _tb, "audio_loss": _al, "train_loss": debiased_smooth_loss})
        model.train()

'''

FINAL_LOG_INJECTION = '''
import json
with open("convergence_log.json", "w") as f:
    json.dump(convergence_log, f)
'''

SEED_OLD = "torch.manual_seed(42)\ntorch.cuda.manual_seed(42)"

# Best recipe mods (from current train.py which has LR=0.03, batch=64K, WD=0.05, etc.)
BEST_BASE = []  # current train.py IS the best

# Original baseline mods
BASELINE_MODS = [
    ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
     "return WEIGHT_DECAY  # constant WD"),
    ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
    ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
    ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
    ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
    ("MATRIX_LR = 0.03", "MATRIX_LR = 0.06"),
    ("TOTAL_BATCH_SIZE = 2**15", "TOTAL_BATCH_SIZE = 2**17"),
]


def run_exp(desc, train_mods=None, prepare_mods=None, timeout=600, log_convergence=False):
    with open("train.py") as f:
        original_train = f.read()
    with open("prepare.py") as f:
        original_prepare = f.read()
    modified_train = original_train

    if log_convergence:
        modified_train = modified_train.replace(
            "train_loader = make_dataloader(",
            CONVERGENCE_LOGGING_CODE + "\ntrain_loader = make_dataloader(", 1)
        modified_train = modified_train.replace(
            "    step += 1\n", LOGGING_INJECTION + "    step += 1\n", 1)
        modified_train = modified_train.replace(
            "# Final summary", FINAL_LOG_INJECTION + "\n# Final summary", 1)

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
    if log_convergence and os.path.exists("convergence_log.json"):
        with open("convergence_log.json") as f: log = json.load(f)
        safe_desc = desc.replace(' ', '_').replace(':', '_')
        log_path = f"convergence_{safe_desc}.json"
        with open(log_path, "w") as f:
            json.dump({"desc": desc, "commit": commit, "val_loss": val_loss,
                       "text_bpb": text_bpb, "audio_loss": audio_loss, "curve": log}, f)
        print(f"  Saved {len(log)} convergence points")
        os.remove("convergence_log.json")
    with open("train.py", "w") as f: f.write(original_train)
    with open("prepare.py", "w") as f: f.write(original_prepare)
    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}


import re as re2
with open("prepare.py") as f:
    m = re2.search(r'TIME_BUDGET\s*=\s*(\d+)', f.read())
    time_budget = int(m.group(1)) if m else 300

SEEDS = [42, 137, 2024]

experiments = []

# Multi-seed 5min for latest best
for seed in SEEDS:
    experiments.append((
        f"paper: best_5min_s{seed}",
        [(SEED_OLD, f"torch.manual_seed({seed})\ntorch.cuda.manual_seed({seed})")],
        None, 600, False))

# Multi-seed 5min for baseline
for seed in SEEDS:
    experiments.append((
        f"paper: baseline_5min_s{seed}",
        BASELINE_MODS + [(SEED_OLD, f"torch.manual_seed({seed})\ntorch.cuda.manual_seed({seed})")],
        None, 600, False))

# 10min best with convergence
experiments.append((
    "paper: best_10min_conv",
    [],
    [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")],
    1200, True))

# 10min baseline with convergence
experiments.append((
    "paper: baseline_10min_conv",
    BASELINE_MODS,
    [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")],
    1200, True))

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods, to, log_conv = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=to, log_convergence=log_conv)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 27 PAPER FIGURE SUMMARY")
print("="*60)

# Multi-seed stats
from collections import defaultdict
import statistics

config_results = defaultdict(list)
for desc, r in results:
    if r and "5min" in desc:
        config = "best" if "best" in desc else "baseline"
        config_results[config].append(r)

print("\nMulti-seed 5-min comparison (n=3):")
print(f"  {'config':<12} | {'val_loss':>12} | {'text_bpb':>12} | {'audio':>12}")
print(f"  {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}")
for config in ["best", "baseline"]:
    runs = config_results[config]
    if len(runs) >= 2:
        vl = [r['val_loss'] for r in runs]
        tb = [r['text_bpb'] for r in runs]
        al = [r['audio_loss'] for r in runs]
        print(f"  {config:<12} | {statistics.mean(vl):.4f}±{statistics.stdev(vl):.4f} | "
              f"{statistics.mean(tb):.4f}±{statistics.stdev(tb):.4f} | "
              f"{statistics.mean(al):.4f}±{statistics.stdev(al):.4f}")

# 10min comparison
print("\n10-min comparison:")
for desc, r in results:
    if r and "10min" in desc:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")

# All individual runs
print("\nAll runs:")
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
