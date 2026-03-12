"""Sweep 23: Learning rate schedule exploration.

Now that we've established the best optimizer settings, explore the LR schedule
more aggressively. The warmdown ratio was the least impactful component — maybe
we need a fundamentally different schedule shape.

Tests:
1. Cosine schedule (replace linear warmdown with cosine decay)
2. WD=0.03 (push lower WD further)
3. WD=0.02 (even lower)
4. LR=0.08 with best settings (higher LR + better optimizer)
5. LR=0.04 (lower LR for reference)
6. Warmup=0.05 (brief warmup phase)
7. batch=64K with best (more steps)
8. Cosine + WD=0.03 (combine best schedule + WD)
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


# Cosine schedule replacement
# Replace the linear warmdown with cosine decay
COSINE_SCHEDULE = """def get_lr_multiplier(progress):
    import math
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    # Cosine decay from 1.0 to FINAL_LR_FRAC
    decay_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
    return FINAL_LR_FRAC + 0.5 * (1.0 - FINAL_LR_FRAC) * (1.0 + math.cos(math.pi * decay_progress))"""

LINEAR_SCHEDULE = """def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC"""


experiments = [
    # 1. Cosine schedule (no flat region, smooth decay throughout)
    ("sched: cosine", [(LINEAR_SCHEDULE, COSINE_SCHEDULE)], None),

    # 2. WD=0.03 (even lower)
    ("sched: WD0.03", [("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03")], None),

    # 3. WD=0.02
    ("sched: WD0.02", [("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.02")], None),

    # 4. Higher LR
    ("sched: LR0.08", [("MATRIX_LR = 0.06", "MATRIX_LR = 0.08")], None),

    # 5. Lower LR
    ("sched: LR0.04", [("MATRIX_LR = 0.06", "MATRIX_LR = 0.04")], None),

    # 6. Brief warmup
    ("sched: warmup0.05", [("WARMUP_RATIO = 0.0", "WARMUP_RATIO = 0.05")], None),

    # 7. Batch 64K (more steps)
    ("sched: batch64K", [("TOTAL_BATCH_SIZE = 2**16", "TOTAL_BATCH_SIZE = 2**15")], None),

    # 8. Cosine + WD0.03
    ("sched: cosine+WD0.03", [
        (LINEAR_SCHEDULE, COSINE_SCHEDULE),
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03"),
    ], None),

    # 9. Cosine + warmup
    ("sched: cosine+warmup0.05", [
        (LINEAR_SCHEDULE, COSINE_SCHEDULE),
        ("WARMUP_RATIO = 0.0", "WARMUP_RATIO = 0.05"),
    ], None),

    # 10. WD=0.03 + warmup
    ("sched: WD0.03+warmup0.05", [
        ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.03"),
        ("WARMUP_RATIO = 0.0", "WARMUP_RATIO = 0.05"),
    ], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 23 SCHEDULE SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
