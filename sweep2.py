"""Second sweep: finer WD grid + fixed H5 + combinations."""
import subprocess, re, os

PYTHON = ".venv/bin/python"

def run_exp(desc, modifications):
    with open("train.py") as f:
        original = f.read()
    modified = original
    for old, new in modifications:
        modified = modified.replace(old, new, 1)
    with open("train.py", "w") as f:
        f.write(modified)
    os.system(f'git add train.py && git commit -m "exp: {desc}" --allow-empty 2>/dev/null')
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    print(f"\n{'='*60}\nEXP {commit}: {desc}\n{'='*60}")
    try:
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT")
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc} — timeout\n")
        with open("train.py", "w") as f:
            f.write(original)
        return None
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)
    if not vl or result.returncode != 0:
        print(f"  CRASHED (exit={result.returncode})")
        print(output[-500:] if output else "no output")
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original)
        return None
    val_loss, text_bpb, audio_loss = float(vl.group(1)), float(tb.group(1)), float(al.group(1))
    status = "keep" if audio_loss > 0 else "crash"
    print(f"  val_loss={val_loss:.6f} text_bpb={text_bpb:.6f} audio_loss={audio_loss:.6f}")
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{text_bpb:.6f}\t{audio_loss:.6f}\t16.3\t{status}\t{desc}\n")
    with open("train.py", "w") as f:
        f.write(original)
    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}

experiments = [
    # Finer WD grid around optimal
    ("115h audio, per-row WD audio=0.1", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.1"),
    ]),
    ("115h audio, per-row WD audio=0.0 (no audio WD)", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
    ]),
    ("115h audio, per-row WD audio=0.3", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.3"),
    ]),

    # Fixed H5: AdaDecay WD (with the 'step' bug fix)
    ("H5-fixed: AdaDecay WD beta=0.99", [
        ("ADADECAY_WD = False", "ADADECAY_WD = True"),
    ]),

    # Best WD (0.2) + H5
    ("WD=0.2 + H5 AdaDecay", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.2"),
        ("ADADECAY_WD = False", "ADADECAY_WD = True"),
    ]),

    # Best WD (0.2) + H6 (MILES alpha=0.8 was marginally neutral)
    ("WD=0.2 + H6 MILES alpha=0.8", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.2"),
        ("MODALITY_REBALANCE = False", "MODALITY_REBALANCE = True"),
        ("MODALITY_REBALANCE_ALPHA = 0.5", "MODALITY_REBALANCE_ALPHA = 0.8"),
    ]),

    # Try dropout=0.1 with WD=0.2
    ("WD=0.2 + dropout=0.1", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.2"),
        ("DROPOUT = 0.05", "DROPOUT = 0.1"),
    ]),

    # Try lower Muon WD with lower embed WD
    ("WD=0.2 + Muon WD=0.2", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.2"),
        ("WEIGHT_DECAY = 0.5", "WEIGHT_DECAY = 0.2"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 2 SUMMARY")
print("="*60)
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL   |        |             | {desc}")
