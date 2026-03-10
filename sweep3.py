"""Final sweep: WD=0.0 combinations + Muon WD tuning."""
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
            f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc} — timeout\n")
        with open("train.py", "w") as f:
            f.write(original)
        return None
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)
    if not vl or result.returncode != 0:
        print(f"  CRASHED")
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
    # WD=0.0 + Muon WD=0.2 (text_bpb improved with Muon WD=0.2)
    ("WD=0.0 + Muon WD=0.2", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
        ("WEIGHT_DECAY = 0.5", "WEIGHT_DECAY = 0.2"),
    ]),

    # WD=0.0 + Muon WD=0.3
    ("WD=0.0 + Muon WD=0.3", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
        ("WEIGHT_DECAY = 0.5", "WEIGHT_DECAY = 0.3"),
    ]),

    # WD=0.0 + no dropout (dropout hurts with more data)
    ("WD=0.0 + dropout=0.0", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
        ("DROPOUT = 0.05", "DROPOUT = 0.0"),
    ]),

    # WD=0.0 + Muon WD=0.2 + no dropout — full de-regularization
    ("WD=0.0 + Muon WD=0.2 + dropout=0.0", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
        ("WEIGHT_DECAY = 0.5", "WEIGHT_DECAY = 0.2"),
        ("DROPOUT = 0.05", "DROPOUT = 0.0"),
    ]),

    # WD=0.0 + higher Muon LR (less regularization = can push LR)
    ("WD=0.0 + Muon LR=0.08", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.08"),
    ]),

    # WD=0.0 + lm_head WD=0.0 (remove ALL WD from vocab params)
    # Note: current setup has lm_head WD via per-row, setting audio=0 also removes lm_head audio WD
    # But Muon WD still applies to transformer matrices
    ("WD=0.0 + Muon WD=0.2 + Muon LR=0.08", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
        ("WEIGHT_DECAY = 0.5", "WEIGHT_DECAY = 0.2"),
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.08"),
    ]),

    # Uniform WD=0.0 everywhere (no per-row, no Muon WD)
    ("all WD=0.0 (no regularization)", [
        ("EMBED_WD_AUDIO = 2.5", "EMBED_WD_AUDIO = 0.0"),
        ("WEIGHT_DECAY = 0.5", "WEIGHT_DECAY = 0.0"),
        ("DROPOUT = 0.05", "DROPOUT = 0.0"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 3 SUMMARY")
print("="*60)
for desc, r in results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL   |        |             | {desc}")
