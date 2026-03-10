"""Run all hypothesis experiments sequentially."""
import subprocess, re, sys, os

PYTHON = ".venv/bin/python"

def run_exp(desc, modifications):
    """Modify train.py, run training, parse results, restore."""
    # Read original
    with open("train.py") as f:
        original = f.read()

    # Apply modifications
    modified = original
    for old, new in modifications:
        modified = modified.replace(old, new, 1)

    with open("train.py", "w") as f:
        f.write(modified)

    # Commit
    os.system(f'git add train.py && git commit -m "exp: {desc}" --allow-empty 2>/dev/null')
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()

    print(f"\n{'='*60}")
    print(f"EXP {commit}: {desc}")
    print(f"{'='*60}")

    # Run training
    try:
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0.000000\t0.000000\t0.000000\t16.3\tcrash\t{desc} — timeout\n")
        with open("train.py", "w") as f:
            f.write(original)
        return None

    # Parse results
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)

    if not vl or result.returncode != 0:
        print(f"  CRASHED (exit={result.returncode})")
        print(output[-500:] if output else "no output")
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0.000000\t0.000000\t0.000000\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original)
        return None

    val_loss = float(vl.group(1))
    text_bpb = float(tb.group(1))
    audio_loss = float(al.group(1))

    status = "keep" if audio_loss > 0 else "crash"
    print(f"  val_loss={val_loss:.6f} text_bpb={text_bpb:.6f} audio_loss={audio_loss:.6f}")

    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{text_bpb:.6f}\t{audio_loss:.6f}\t16.3\t{status}\t{desc}\n")

    # Restore original
    with open("train.py", "w") as f:
        f.write(original)

    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = [
    # EXP 1: Baseline with 115h audio
    ("baseline-115h: per-row WD text=0.0 audio=2.5", []),

    # EXP 2-5: Re-tune audio WD with more data
    ("115h audio, per-row WD audio=1.5", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 1.5"),
    ]),
    ("115h audio, per-row WD audio=1.0", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 1.0"),
    ]),
    ("115h audio, per-row WD audio=0.5", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.5"),
    ]),
    ("115h audio, per-row WD audio=0.2", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.2"),
    ]),

    # EXP 6: H2 — Progressive audio upweighting
    ("H2: progressive audio upweight max=1.5", [
        ("AUDIO_UPWEIGHT_SCHEDULE = False", "AUDIO_UPWEIGHT_SCHEDULE = True"),
    ]),
    ("H2: progressive audio upweight max=1.2", [
        ("AUDIO_UPWEIGHT_SCHEDULE = False", "AUDIO_UPWEIGHT_SCHEDULE = True"),
        ("AUDIO_UPWEIGHT_MAX = 1.5", "AUDIO_UPWEIGHT_MAX = 1.2"),
    ]),

    # EXP 8: H3 — Gradient norm balancing
    ("H3: gradient norm balancing", [
        ("GRAD_NORM_BALANCE = False", "GRAD_NORM_BALANCE = True"),
    ]),

    # EXP 9-10: H4 — Per-row adaptive LR
    ("H4: per-row LR alpha=0.3", [
        ("PERROW_LR = False", "PERROW_LR = True"),
    ]),
    ("H4: per-row LR alpha=0.5", [
        ("PERROW_LR = False", "PERROW_LR = True"),
        ("PERROW_LR_ALPHA = 0.3", "PERROW_LR_ALPHA = 0.5"),
    ]),

    # EXP 11-12: H5 — AdaDecay WD
    ("H5: AdaDecay WD beta=0.99", [
        ("ADADECAY_WD = False", "ADADECAY_WD = True"),
    ]),
    ("H5: AdaDecay WD beta=0.95", [
        ("ADADECAY_WD = False", "ADADECAY_WD = True"),
        ("ADADECAY_BETA = 0.99", "ADADECAY_BETA = 0.95"),
    ]),

    # EXP 13-14: H6 — MILES rebalancing
    ("H6: MILES rebalance alpha=0.5", [
        ("MODALITY_REBALANCE = False", "MODALITY_REBALANCE = True"),
    ]),
    ("H6: MILES rebalance alpha=0.8", [
        ("MODALITY_REBALANCE = False", "MODALITY_REBALANCE = True"),
        ("MODALITY_REBALANCE_ALPHA = 0.5", "MODALITY_REBALANCE_ALPHA = 0.8"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

# Print summary
print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL   |        |             | {desc}")
