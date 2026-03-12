"""Sweep 31: Scaling × Duration interaction.

Key question: Does the d12 4.1% gap grow with more training time?
If yes → optimizer tuning becomes MORE critical as you scale compute.

Tests:
1. d12 best at 10min
2. d12 baseline at 10min
3. d8 best with mom=0.95 at 5min (test if mom=0.95 helps at d8 too)
4. d8 best with mom=0.95 at 10min
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

MOM95_MODS = [
    ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
    ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
]

PREPARE_10MIN = [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")]

experiments = [
    # d12 best at 10min
    ("scaledur: d12_best_10min",
     [("DEPTH = 8", "DEPTH = 12")],
     PREPARE_10MIN, 1800),

    # d12 baseline at 10min
    ("scaledur: d12_baseline_10min",
     [("DEPTH = 8", "DEPTH = 12")] + BASELINE_MODS,
     PREPARE_10MIN, 1800),

    # d8 best with mom=0.95 at 5min
    ("scaledur: d8_mom95_5min",
     MOM95_MODS,
     None, 600),

    # d8 best with mom=0.95 at 10min
    ("scaledur: d8_mom95_10min",
     MOM95_MODS,
     PREPARE_10MIN, 1800),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods, to = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=to)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 31: SCALING × DURATION")
print("="*60)
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

# Compare with sweep30 5min results
print("\n\nScaling × Duration matrix:")
print(f"  {'':>8} | {'5min best':>10} | {'5min base':>10} | {'10min best':>11} | {'10min base':>11} | {'Δ5min':>6} | {'Δ10min':>7}")
d12_5best = 3.599  # from sweep30
d12_5base = 3.754
d12_10best = results[0][1]['val_loss'] if results[0][1] else None
d12_10base = results[1][1]['val_loss'] if results[1][1] else None
if d12_10best and d12_10base:
    print(f"  d12     | {d12_5best:.4f}     | {d12_5base:.4f}     | {d12_10best:.4f}      | {d12_10base:.4f}      | {(d12_5best-d12_5base)/d12_5base*100:.1f}%  | {(d12_10best-d12_10base)/d12_10base*100:.1f}%")
