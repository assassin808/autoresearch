"""Sweep 5: Architecture & schedule experiments inspired by literature."""
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
        print(output[-300:] if output else "")
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

# Best config baseline: embed WD=0.0, Muon WD=0.2, dropout=0.0
experiments = [
    # Try higher embed LR (with no WD, embeddings may benefit from higher LR)
    ("best-0 + embed_lr=0.8", [
        ("EMBEDDING_LR = 0.6", "EMBEDDING_LR = 0.8"),
    ]),
    ("best-0 + embed_lr=0.4", [
        ("EMBEDDING_LR = 0.6", "EMBEDDING_LR = 0.4"),
    ]),

    # Try Muon LR=0.07 (between 0.06 and 0.08)
    ("best-0 + Muon LR=0.07", [
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.07"),
    ]),

    # Try batch size change (with 115h audio, maybe different batch optimal)
    # Batch 512K = 16 grad accum steps
    ("best-0 + batch 512K", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**19"),
    ]),

    # Try warmup (now that we have less regularization, may need warmup for stability)
    ("best-0 + warmup=0.02", [
        ("WARMUP_RATIO = 0.0", "WARMUP_RATIO = 0.02"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 5 SUMMARY")
print("="*60)
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL   |        |             | {desc}")
