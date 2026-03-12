"""Sweep 32: Final recipe optimization.

Based on sweep29 (momentum reversal at 10min) and sweep31 (confirmation):
- mom=0.95 is better at 10min, equal at 5min
- Test mom=0.96 as compromise
- Test updated best recipe at multiple durations

Also: what if we use WARMDOWN_RATIO=0.75 (current default) with mom=0.95?
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

MOM95_MODS = [
    ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
    ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
]

MOM96_MODS = [
    ("momentum=0.97, ns_steps=10", "momentum=0.96, ns_steps=10"),
    ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.96"),
]

PREPARE_10MIN = [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")]

experiments = [
    # Reference: current best at 5min
    ("recipe: ref_5min", [], None, 600),

    # mom=0.96 at 5min
    ("recipe: mom96_5min", MOM96_MODS, None, 600),

    # mom=0.95 at 5min (already confirmed 3.478)
    ("recipe: mom95_5min", MOM95_MODS, None, 600),

    # mom=0.96 at 10min
    ("recipe: mom96_10min", MOM96_MODS, PREPARE_10MIN, 1800),

    # mom=0.95 + WD=0.03 at 5min
    ("recipe: mom95_wd03_5min",
     MOM95_MODS + [("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03")],
     None, 600),

    # mom=0.95 + WD=0.03 at 10min
    ("recipe: mom95_wd03_10min",
     MOM95_MODS + [("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03")],
     PREPARE_10MIN, 1800),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods, to = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=to)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 32: FINAL RECIPE")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
