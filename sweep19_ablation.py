"""Sweep 19: Ablation study for the best optimizer recipe.

Best: cWD + WD=0.05 + mom=0.97 + warmdown=0.75 → 3.494

Systematic ablation: remove each component one at a time to measure individual
contribution. This is critical for the paper.

Also includes scaling experiments at different depths.
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


# Note: train.py now has the BEST config as default:
# WD=0.05, coupledWD, mom=0.97, warmdown=0.75

experiments = [
    # ===== ABLATIONS: remove one component at a time =====

    # Full best (reference)
    ("ablation: full_best (reference)", [], None),

    # Remove coupled WD (revert to constant WD)
    ("ablation: -coupledWD",
     [("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
       "return WEIGHT_DECAY  # constant WD")], None),

    # Remove lower WD (revert to WD=0.1)
    ("ablation: -lowerWD (WD=0.1)",
     [("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1")], None),

    # Remove higher momentum (revert to mom=0.95)
    ("ablation: -mom97 (mom=0.95)",
     [("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
      ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95")], None),

    # Remove longer warmdown (revert to 0.7)
    ("ablation: -warmdown0.75 (wd=0.7)",
     [("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7")], None),

    # Remove ALL improvements (original baseline)
    ("ablation: original_baseline",
     [("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
       "return WEIGHT_DECAY  # constant WD"),
      ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
      ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
      ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
      ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
     ], None),

    # ===== SCALING: different model depths =====

    # Depth 4 (smaller model)
    ("scale: depth4_best", [("DEPTH = 8", "DEPTH = 4")], None),

    # Depth 4 original (for comparison)
    ("scale: depth4_original",
     [("DEPTH = 8", "DEPTH = 4"),
      ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
       "return WEIGHT_DECAY  # constant WD"),
      ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
      ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
      ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
      ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
     ], None),

    # Depth 6 best
    ("scale: depth6_best", [("DEPTH = 8", "DEPTH = 6")], None),

    # Depth 6 original
    ("scale: depth6_original",
     [("DEPTH = 8", "DEPTH = 6"),
      ("return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD (sweep16: decays with LR)",
       "return WEIGHT_DECAY  # constant WD"),
      ("WEIGHT_DECAY = 0.05", "WEIGHT_DECAY = 0.1"),
      ("momentum=0.97, ns_steps=10", "momentum=0.95, ns_steps=10"),
      ("return (1 - frac) * 0.85 + frac * 0.97", "return (1 - frac) * 0.85 + frac * 0.95"),
      ("WARMDOWN_RATIO = 0.75", "WARMDOWN_RATIO = 0.7"),
     ], None),

    # ===== AUDIO RATIO on best baseline =====
    # Audio ratio sweep to see if best optimizer changes optimal ratio
    ("ratio: audio0.1_best",
     [("AUDIO_MIX_RATIO = 0.2", "AUDIO_MIX_RATIO = 0.1")], [
      ("AUDIO_MIX_RATIO = 0.2", "AUDIO_MIX_RATIO = 0.1"),
     ]),

    ("ratio: audio0.3_best",
     [("AUDIO_MIX_RATIO = 0.2", "AUDIO_MIX_RATIO = 0.3")], [
      ("AUDIO_MIX_RATIO = 0.2", "AUDIO_MIX_RATIO = 0.3"),
     ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 19 ABLATION STUDY SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
