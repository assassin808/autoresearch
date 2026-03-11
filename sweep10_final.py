"""Sweep 10: Final optimization - combine ALL winning changes.

Winning changes identified:
- batch 128K (from sweep6)
- NS steps=10 (from sweep9: 3.724)
- embedLR=0.6 (from sweep9: 3.727)
- H1 freq WD C=0.5 (from sweep7: 3.783 at batch 262K)
- x0_lambda_init=0.05 (from sweep7: 3.782 at batch 262K)
- MuonWD=0.1-0.2 range

Now combine the top improvements.
"""
import subprocess, re, os

PYTHON = ".venv/bin/python"

def run_exp(desc, modifications):
    with open("train.py") as f:
        original = f.read()
    modified = original
    for old, new in modifications:
        count = modified.count(old)
        if count == 0:
            print(f"  WARNING: pattern not found: {old[:80]}...")
            with open("train.py", "w") as f:
                f.write(original)
            return None
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
        print(output[-500:] if output else "no output")
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


# Freq-proportional WD modifications (reusable)
FREQ_WD_MOD = ("print(f\"Per-row WD: text={EMBED_WD_TEXT}, audio={EMBED_WD_AUDIO}\")",
     """# H1: Frequency-proportional WD = C / sqrt(token_freq)
import math
FREQ_WD_C = 0.5
text_est_freq = 8260.0
audio_est_freq = 2360.0
embed_wd_per_row[:AUDIO_START_ID] = FREQ_WD_C / math.sqrt(text_est_freq)
embed_wd_per_row[AUDIO_START_ID:] = FREQ_WD_C / math.sqrt(audio_est_freq)
print(f"H1: Freq-proportional WD: text={embed_wd_per_row[0].item():.4f}, audio={embed_wd_per_row[AUDIO_START_ID].item():.4f}")""")


experiments = [
    # ===== Combine batch128K + NS10 + embedLR=0.6 =====
    ("batch128K+NS10+embedLR0.6", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.6"),
    ]),

    # ===== Combine batch128K + NS10 + x0=0.05 =====
    ("batch128K+NS10+x0=0.05", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.05)"),
    ]),

    # ===== Combine batch128K + NS10 + embedLR=0.6 + x0=0.05 =====
    ("batch128K+NS10+eLR0.6+x0=0.05", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.6"),
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.05)"),
    ]),

    # ===== Combine batch128K + NS10 + freq WD =====
    ("batch128K+NS10+freqWD", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        FREQ_WD_MOD,
    ]),

    # ===== MEGA COMBO: batch128K+NS10+embedLR0.6+x0=0.05+freqWD =====
    ("MEGA: b128K+NS10+eLR0.6+x0+fWD", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.6"),
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.05)"),
        FREQ_WD_MOD,
    ]),

    # ===== batch128K + NS10 + MuonWD=0.1 =====
    ("batch128K+NS10+MuonWD0.1", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.1"),
    ]),

    # ===== batch128K + NS10 + MuonWD=0.15 =====
    ("batch128K+NS10+MuonWD0.15", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.15"),
    ]),

    # ===== FULL COMBO: b128K+NS10+eLR0.6+x0+MuonWD0.15 =====
    ("FULL: b128K+NS10+eLR0.6+x0+WD0.15", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.6"),
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.05)"),
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.15"),
    ]),

    # ===== FULL COMBO + coupled WD =====
    ("FULL+coupledWD", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.6"),
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.05)"),
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.15"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # WD decays with LR"),
    ]),

    # ===== Try NS steps=15, 20 (at batch 128K to push further) =====
    ("batch128K+NS15", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=15,"),
    ]),
    ("batch128K+NS20", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=20,"),
    ]),

    # ===== Smaller Muon LR at batch 128K (more steps = smaller optimal LR?) =====
    ("batch128K+NS10+MuonLR0.04", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.04"),
    ]),
    ("batch128K+NS10+MuonLR0.05", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.05"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 10 SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

print(f"\nBest previous: val_loss=3.724 (batch128K + NS10)")
