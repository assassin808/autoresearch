"""Sweep 9: Combine winning changes + explore batch size scaling.

Key findings from sweep6:
- batch 128K: 3.732 (HUGE win — more steps matters)
- H9 coupled warmdown: 3.792 (WD*lrm helps)
- NS steps=10: 3.794 (better orthogonalization)
- Muon LR=0.07: 3.796 (slightly higher LR works)

Now combine these and explore further.
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


experiments = [
    # ===== Batch size exploration =====
    # 128K was much better than 256K. Try even smaller.
    ("batch 64K", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**16"),
    ]),
    ("batch 96K (3*32K)", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 3 * 2**15"),  # 98304
    ]),

    # ===== Combine batch 128K + coupled warmdown =====
    ("batch128K + coupled WD", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # WD decays with LR"),
    ]),

    # ===== Combine batch 128K + NS steps 10 =====
    ("batch128K + NS10", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("ns_steps=5,", "ns_steps=10,"),
    ]),

    # ===== Combine batch 128K + Muon LR 0.07 =====
    ("batch128K + MuonLR0.07", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.07"),
    ]),

    # ===== Triple combo: batch128K + coupled WD + NS10 =====
    ("batch128K + coupledWD + NS10", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # WD decays with LR"),
        ("ns_steps=5,", "ns_steps=10,"),
    ]),

    # ===== Quad combo: batch128K + coupled WD + NS10 + MuonLR0.07 =====
    ("batch128K + coupledWD + NS10 + LR0.07", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # WD decays with LR"),
        ("ns_steps=5,", "ns_steps=10,"),
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.07"),
    ]),

    # ===== batch128K + Muon WD sweep (since batch size changes optimal WD) =====
    ("batch128K + MuonWD=0.1", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.1"),
    ]),
    ("batch128K + MuonWD=0.3", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.3"),
    ]),
    ("batch128K + MuonWD=0.15", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("WEIGHT_DECAY = 0.2", "WEIGHT_DECAY = 0.15"),
    ]),

    # ===== batch128K + embed WD sweep (more steps may change embed WD needs) =====
    ("batch128K + embedWD=0.1", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("EMBED_WD_AUDIO = 0.0", "EMBED_WD_AUDIO = 0.1"),
    ]),
    ("batch128K + embedWD=0.5", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("EMBED_WD_AUDIO = 0.0", "EMBED_WD_AUDIO = 0.5"),
    ]),

    # ===== batch128K + warmdown sweep =====
    ("batch128K + warmdown=0.5", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.5"),
    ]),
    ("batch128K + warmdown=0.8", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.8"),
    ]),

    # ===== H10: Weight tying (proper implementation) =====
    # Share wte and lm_head weights by using wte.weight in lm_head forward
    ("batch128K + weight-tied (via init)", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        # Tie weights in __init__ before compile
        ("        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)",
         "        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)\n        self.lm_head.weight = self.transformer.wte.weight  # H10: weight tying"),
        # Fix param counting — lm_head params are now the same as wte
        ("        lm_head_params = list(self.lm_head.parameters())",
         "        lm_head_params = []  # H10: lm_head shares wte weights"),
    ]),

    # ===== batch128K + embed LR sweep =====
    ("batch128K + embedLR=0.6", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.6"),
    ]),
    ("batch128K + embedLR=1.0", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 1.0"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 9 SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

print(f"\nBest previous: val_loss=3.732 (batch 128K)")
