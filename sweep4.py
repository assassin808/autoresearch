"""Sweep 4: Fine-tuning around the best config (WD=0.0, Muon WD=0.2)."""
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
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original)
        return None
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)
    if not vl or result.returncode != 0:
        print(f"  CRASHED")
        print(output[-300:] if output else "no output")
        with open("results.tsv", "a") as f:
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f:
            f.write(original)
        return None
    val_loss, text_bpb, audio_loss = float(vl.group(1)), float(tb.group(1)), float(al.group(1))
    print(f"  val_loss={val_loss:.6f} text_bpb={text_bpb:.6f} audio_loss={audio_loss:.6f}")
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{text_bpb:.6f}\t{audio_loss:.6f}\t16.3\tkeep\t{desc}\n")
    with open("train.py", "w") as f:
        f.write(original)
    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}

# Starting from best: WD=0.0, Muon WD=0.2
experiments = [
    # Muon WD fine-tuning
    ("best + Muon WD=0.15", [
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.15"),
    ]),
    ("best + Muon WD=0.1", [
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.1"),
    ]),

    # Dropout tuning
    ("best + dropout=0.02", [
        ("DROPOUT = 0.05", "DROPOUT = 0.02"),
    ]),
    ("best + dropout=0.0", [
        ("DROPOUT = 0.05", "DROPOUT = 0.0"),
    ]),

    # Adam betas tuning
    ("best + Adam betas (0.85, 0.95)", [
        ("ADAM_BETAS = (0.8, 0.95)", "ADAM_BETAS = (0.85, 0.95)"),
    ]),
    ("best + Adam betas (0.9, 0.95)", [
        ("ADAM_BETAS = (0.8, 0.95)", "ADAM_BETAS = (0.9, 0.95)"),
    ]),

    # Warmdown tuning (with more data, maybe different schedule?)
    ("best + warmdown=0.6", [
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.6"),
    ]),
    ("best + warmdown=0.8", [
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.8"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 4 SUMMARY")
print("="*60)
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL   |        |             | {desc}")
