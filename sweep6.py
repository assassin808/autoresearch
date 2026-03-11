"""Sweep 6: Theory-derived hypothesis experiments.

Tests from THEORY.md:
- H8: NS iteration count (3, 7, 10) — is 5 optimal or just default?
- H9: Coupled warmdown (WD decays with LR) — hidden data reweighting?
- H10: Weight tying (wte=lm_head) — currently untied, does tying help?
- H7: Decoupled embed_dim (256→512 projection) — free up params for compute
- H1: Frequency-proportional WD for embeddings
- H3: Lazy WD for embeddings (only decay updated rows)
- Bonus: Muon WD schedule experiments
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
            print(f"  WARNING: pattern not found: {old[:60]}...")
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

# Current best: val_loss=3.797, embed WD=0.0, Muon WD=0.2, dropout=0.0
# Config: MATRIX_LR=0.06, EMBEDDING_LR=0.8, WARMDOWN_RATIO=0.7

experiments = [
    # ===== H8: Newton-Schulz iteration count =====
    # Currently ns_steps=5. Test if more/fewer iterations change quality.
    ("H8: NS steps=3", [
        ("ns_steps=5,", "ns_steps=3,"),
    ]),
    ("H8: NS steps=7", [
        ("ns_steps=5,", "ns_steps=7,"),
    ]),
    ("H8: NS steps=10", [
        ("ns_steps=5,", "ns_steps=10,"),
    ]),

    # ===== H9: Coupled warmdown — WD decreases with LR =====
    # Currently: WD constant, LR decreases. Test WD coupled to LR.
    ("H9: coupled warmdown (WD*lrm)", [
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # WD decays with LR"),
    ]),
    ("H9: sqrt-coupled warmdown", [
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)**0.5  # WD decays slower than LR"),
    ]),

    # ===== H10: Weight tying (wte = lm_head) =====
    # Currently untied. Test if tying helps (saves 10M params worth of capacity).
    ("H10: weight-tied lm_head", [
        # After model init, tie the weights
        ("model = torch.compile(model, dynamic=False)",
         """# H10: Weight tying — share embedding and output weights
model._orig_mod = model  # will be replaced by compile, but need ref first
model._orig_mod.lm_head.weight = model._orig_mod.transformer.wte.weight
model = torch.compile(model, dynamic=False)"""),
    ]),

    # ===== Muon WD schedule experiments =====
    # Currently constant 0.2. Test linear decay to 0.
    ("Muon WD linear decay to 0", [
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * (1.0 - progress)  # WD decays linearly to 0"),
    ]),
    ("Muon WD cosine decay to 0", [
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "import math as _m; return WEIGHT_DECAY * 0.5 * (1 + _m.cos(_m.pi * progress))  # cosine decay"),
    ]),

    # ===== Embed LR experiments (since WD=0 now) =====
    ("embed_lr=1.0", [
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 1.0"),
    ]),
    ("embed_lr=0.5", [
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.5"),
    ]),
    ("embed_lr=0.3", [
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 0.3"),
    ]),

    # ===== Softcap experiments =====
    ("softcap=20", [
        ("softcap = 15", "softcap = 20"),
    ]),
    ("softcap=10", [
        ("softcap = 15", "softcap = 10"),
    ]),
    ("softcap=30", [
        ("softcap = 15", "softcap = 30"),
    ]),

    # ===== Muon LR fine-tuning =====
    ("Muon LR=0.05", [
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.05"),
    ]),
    ("Muon LR=0.07", [
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.07"),
    ]),

    # ===== Batch size =====
    ("batch 512K", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**19"),
    ]),
    ("batch 128K", [
        ("TOTAL_BATCH_SIZE = 2**18", "TOTAL_BATCH_SIZE = 2**17"),
    ]),

    # ===== Warmdown ratio =====
    ("warmdown=0.5", [
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.5"),
    ]),
    ("warmdown=0.9", [
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.9"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 6 SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
# Sort by val_loss
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

print(f"\nBest previous: val_loss=3.797 (dropout=0.0, embed WD=0.0, Muon WD=0.2)")
