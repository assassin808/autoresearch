"""Sweep 29: Component ablation at 10min training.

We know the full best vs baseline gap grows with duration.
But WHICH components become more important?

Tests at 10min:
1. Full best (reference)
2. -coupled WD (revert to constant WD)
3. -lower WD (revert to WD=0.1)
4. -higher momentum (revert to mom=0.95)
5. -smaller batch (revert to batch=128K)
6. -lower LR (revert to LR=0.06)
7. Full baseline (all reverted)

This tells us which improvements "compound" and which plateau.
"""
import subprocess, re, os

PYTHON = ".venv/bin/python"


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

PREPARE_10MIN = [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")]

experiments = [
    # Full best at 10min
    ("abl10: full_best", [], PREPARE_10MIN),

    # Remove coupled WD
    ("abl10: -coupledWD", [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
    ], PREPARE_10MIN),

    # Remove lower WD (revert to 0.1)
    ("abl10: -lowerWD", [
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
    ], PREPARE_10MIN),

    # Remove higher momentum (revert to 0.95)
    ("abl10: -mom97", [
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
    ], PREPARE_10MIN),

    # Remove smaller batch (revert to 128K)
    ("abl10: -batch64K", [
        ("TOTAL_BATCH_SIZE = 2**15", "TOTAL_BATCH_SIZE = 2**17"),
    ], PREPARE_10MIN),

    # Remove lower LR (revert to 0.06)
    ("abl10: -lowerLR", [
        ("MATRIX_LR = 0.03", "MATRIX_LR = 0.06"),
    ], PREPARE_10MIN),

    # Full baseline at 10min
    ("abl10: full_baseline", [
        ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
         "return WEIGHT_DECAY  # constant WD"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
        ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
        ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
        ("MATRIX_LR = 0.03", "MATRIX_LR = 0.06"),
        ("TOTAL_BATCH_SIZE = 2**15", "TOTAL_BATCH_SIZE = 2**17"),
    ], PREPARE_10MIN),
]

print(f"Running {len(experiments)} ablation experiments at 10min (~1.5h total)...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=1800)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 29: 10-MINUTE ABLATION SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'Δ':>8} | {'%':>6} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'------':>6} | -----------")
ref = results[0][1]['val_loss'] if results[0][1] else None
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        delta = r['val_loss'] - ref if ref else 0
        pct = delta / ref * 100 if ref else 0
        print(f"  {r['val_loss']:.4f} | {delta:+.4f} | {pct:+.1f}% | {desc}")
    else:
        print(f"  FAIL     |          |        | {desc}")

# Compare 5min vs 10min ablation
print("\n\nComponent importance comparison (5min vs 10min):")
print("  Component   | 5min Δ | 10min Δ | Growing?")
print("  ----------- | ------ | ------- | --------")
