"""Sweep 25: Final push on new best (LR0.04 + batch64K).

New best: 3.478 with LR=0.04, batch=64K, WD=0.05, mom=0.97, warmdown=0.75, coupledWD.

Tests:
1. Reference (new best baseline)
2. WD=0.03 on new base
3. WD=0.07 on new base (slightly higher)
4. Warmdown=0.8 on new base
5. Warmdown=0.7 on new base
6. Mom=0.98 on new base
7. Mom=0.96 on new base
8. LR=0.03 on new base
9. LR=0.05 on new base
10. batch32K (even more steps, even noisier)
11. LR0.03+batch32K
12. 10min training with new best
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
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=timeout)
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


# Find TIME_BUDGET from prepare.py
import re as re2
with open("prepare.py") as f:
    m = re2.search(r'TIME_BUDGET\s*=\s*(\d+)', f.read())
    time_budget = int(m.group(1)) if m else 300

experiments = [
    # Reference
    ("final: reference", [], None),

    # WD sweep
    ("final: WD0.03", [("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03")], None),
    ("final: WD0.07", [("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.07")], None),

    # Warmdown sweep
    ("final: warmdown0.8", [("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.8")], None),
    ("final: warmdown0.7", [("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7")], None),

    # Momentum sweep
    ("final: mom0.98", [
        ("momentum=0.97, ns_steps=10", "momentum=0.98, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.98"),
    ], None),
    ("final: mom0.96", [
        ("momentum=0.97, ns_steps=10", "momentum=0.96, ns_steps=10"),
        ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.96"),
    ], None),

    # LR fine-tune around 0.04
    ("final: LR0.03", [("MATRIX_LR = 0.04", "MATRIX_LR = 0.03")], None),
    ("final: LR0.05", [("MATRIX_LR = 0.04", "MATRIX_LR = 0.05")], None),

    # Even smaller batch
    ("final: batch32K", [("TOTAL_BATCH_SIZE = 2**15", "TOTAL_BATCH_SIZE = 2**14")], None),

    # LR0.03 + batch32K
    ("final: LR0.03+batch32K", [
        ("MATRIX_LR = 0.04", "MATRIX_LR = 0.03"),
        ("TOTAL_BATCH_SIZE = 2**15", "TOTAL_BATCH_SIZE = 2**14"),
    ], None),
]

# Add 10min experiment
experiments.append(("final: 10min", [],
    [(f"TIME_BUDGET = {time_budget}", f"TIME_BUDGET = {time_budget * 2}")]))

print(f"Running {len(experiments)} experiments...")
results = []
timeouts = [600] * 11 + [1200]
for i, item in enumerate(experiments):
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=timeouts[i])
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 25 FINAL PUSH SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
