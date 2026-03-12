"""Sweep 24: LR refinement around 0.04-0.06 range.

sweep23 showed LR0.04 (3.503) beats LR0.06 (3.518). Test finer grid.
Also combine best LR with batch64K and other winners.
"""
import subprocess, re, os

PYTHON = ".venv/bin/python"

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


experiments = [
    # Fine LR grid
    ("lr: 0.03", [("MATRIX_LR = 0.06", "MATRIX_LR = 0.03")], None),
    ("lr: 0.035", [("MATRIX_LR = 0.06", "MATRIX_LR = 0.035")], None),
    ("lr: 0.045", [("MATRIX_LR = 0.06", "MATRIX_LR = 0.045")], None),
    ("lr: 0.05", [("MATRIX_LR = 0.06", "MATRIX_LR = 0.05")], None),
    ("lr: 0.055", [("MATRIX_LR = 0.06", "MATRIX_LR = 0.055")], None),

    # Combine best LR with batch64K
    ("lr0.04+batch64K", [
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.04"),
        ("TOTAL_BATCH_SIZE = 2**16", "TOTAL_BATCH_SIZE = 2**15"),
    ], None),

    # batch64K + LR0.05
    ("lr0.05+batch64K", [
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.05"),
        ("TOTAL_BATCH_SIZE = 2**16", "TOTAL_BATCH_SIZE = 2**15"),
    ], None),

    # batch64K + LR0.035
    ("lr0.035+batch64K", [
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.035"),
        ("TOTAL_BATCH_SIZE = 2**16", "TOTAL_BATCH_SIZE = 2**15"),
    ], None),

    # Best combo: LR0.04 + WD0.03
    ("lr0.04+WD0.03", [
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.04"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03"),
    ], None),

    # Best combo: batch64K + WD0.03
    ("batch64K+WD0.03", [
        ("TOTAL_BATCH_SIZE = 2**16", "TOTAL_BATCH_SIZE = 2**15"),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03"),
    ], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 24 LR REFINEMENT SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
