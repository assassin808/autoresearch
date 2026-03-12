"""Sweep 30: Scaling study — do optimizer findings transfer across model sizes?

Test best vs baseline at depth 4, 8, 12, 16.
Current default depth=8 (88M params, dim=512).

Depth 4:  dim=256, ~22M params
Depth 8:  dim=512, ~88M params (default)
Depth 12: dim=768, ~198M params
Depth 16: dim=1024, ~352M params

Key question: Does the best/baseline gap grow or shrink with model size?
If it grows → optimizer findings are MORE important for larger models (great paper story).
If it shrinks → optimizer sensitivity decreases with scale (also interesting).
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

experiments = []
for depth in [4, 8, 12, 16]:
    # Best config at this depth
    depth_mod = [("DEPTH = 8", f"DEPTH = {depth}")] if depth != 8 else []
    experiments.append((f"scale: best_d{depth}", depth_mod, None))
    # Baseline at this depth
    experiments.append((f"scale: baseline_d{depth}", depth_mod + BASELINE_MODS, None))

print(f"Running {len(experiments)} scaling experiments...")
results = []
# Depth 16 might need more time or OOM — use longer timeout
for item in experiments:
    desc, train_mods, prepare_mods = item
    to = 900 if "d16" in desc else 600
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=to)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 30: SCALING STUDY")
print("="*60)
print(f"  {'depth':>5} | {'best':>8} | {'baseline':>8} | {'Δ':>8} | {'%':>6}")
print(f"  {'-----':>5} | {'--------':>8} | {'--------':>8} | {'--------':>8} | {'------':>6}")

for depth in [4, 8, 12, 16]:
    best_r = None
    base_r = None
    for desc, r in results:
        if f"d{depth}" in desc:
            if "best" in desc: best_r = r
            if "baseline" in desc: base_r = r
    if best_r and base_r:
        delta = best_r['val_loss'] - base_r['val_loss']
        pct = delta / base_r['val_loss'] * 100
        print(f"  {depth:>5} | {best_r['val_loss']:.4f} | {base_r['val_loss']:.4f} | {delta:+.4f} | {pct:+.1f}%")
    else:
        best_s = f"{best_r['val_loss']:.4f}" if best_r else "FAIL"
        base_s = f"{base_r['val_loss']:.4f}" if base_r else "FAIL"
        print(f"  {depth:>5} | {best_s:>8} | {base_s:>8} |          |")
