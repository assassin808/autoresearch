"""Sweep 12: Final combination — ALL winning changes together.

Winning changes:
- batch128K (was 262K)
- NS10 (was 5)
- window=LLLL (was SSSL)
- MuonWD=0.15 (was 0.2) — already in train.py
- audio_ratio=0.2 (was 0.3) — good text/audio balance
- embedWD=0.1 at ratio=0.2 helps
- MuonWD=0.1 at ratio=0.2 helps

Now test the mega-combinations.
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


# Note: train.py already has batch128K, NS10, LLLL, MuonWD=0.15

experiments = [
    # ===== Baseline with current config (LLLL already in train.py) =====
    ("baseline: b128K+NS10+LLLL+WD0.15", [
    ], None),

    # ===== Best balanced: ratio=0.2 + MuonWD=0.1 =====
    ("ratio0.2+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== ratio=0.2 + WD=0.1 + embedWD=0.1 =====
    ("ratio0.2+WD0.1+embedWD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("EMBED_WD_AUDIO = 0.0", "EMBED_WD_AUDIO = 0.1"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== ratio=0.15 with tuning =====
    ("ratio0.15+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.15"),
    ]),
    ("ratio0.15+WD0.15", [
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.15"),
    ]),

    # ===== ratio=0.2 + coupled WD + WD0.1 =====
    ("ratio0.2+WD0.1+coupledWD", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== ratio=0.2 + freq-proportional WD =====
    ("ratio0.2+freqWD", [
        ("print(f\"Per-row WD: text={EMBED_WD_TEXT}, audio={EMBED_WD_AUDIO}\")",
         """import math
FREQ_WD_C = 0.5
text_est_freq = 8260.0
audio_est_freq = 2360.0
embed_wd_per_row[:AUDIO_START_ID] = FREQ_WD_C / math.sqrt(text_est_freq)
embed_wd_per_row[AUDIO_START_ID:] = FREQ_WD_C / math.sqrt(audio_est_freq)
print(f"H1: Freq-proportional WD: text={embed_wd_per_row[0].item():.4f}, audio={embed_wd_per_row[AUDIO_START_ID].item():.4f}")"""),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== Aggressive: ratio=0.1 =====
    ("ratio0.1+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.1"),
    ]),

    # ===== MuonWD grid at ratio=0.2 =====
    ("ratio0.2+WD0.05", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.05"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),
    ("ratio0.2+WD0.0", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.0"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 12 SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

print(f"\nBest balanced: val_loss=3.542 (ratio0.2+MuonWD=0.1)")
print(f"Best text-focused: val_loss=3.223 (ratio0.05)")
