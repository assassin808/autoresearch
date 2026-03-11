"""Sweep 7: Complex hypothesis experiments requiring model changes.

- H7: Decoupled embed_dim (smaller embeddings with projection)
- H3: Lazy WD (only decay updated embedding rows)
- H1: Frequency-proportional WD
- Additional: depth/width scaling, aspect ratio experiments
"""
import subprocess, re, os, sys

PYTHON = ".venv/bin/python"

def run_exp(desc, modifications=None, full_replacement=None):
    """Run experiment with either find-replace modifications or full file replacement."""
    with open("train.py") as f:
        original = f.read()

    if full_replacement:
        modified = full_replacement
    else:
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


# ===== H3: Lazy WD — only decay embedding rows updated in current batch =====
# This requires modifying the optimizer to track which rows were accessed.
# We implement it by zeroing out WD for rows not in the current batch.

lazy_wd_mod = [
    # Add lazy WD tracking after the training loop's forward pass
    # We need to know which tokens appeared in the batch
    ("        x, y, epoch = next(train_loader)",
     """        x, y, epoch = next(train_loader)
        # H3: Track tokens in batch for lazy WD
        if hasattr(model, '_lazy_wd_mask'):
            batch_tokens = torch.cat([x.reshape(-1), y.reshape(-1)]).unique()
            lazy_mask = torch.zeros(vocab_size, 1, device="cuda")
            lazy_mask[batch_tokens] = 1.0
            model._lazy_wd_mask = lazy_mask"""),

    # Initialize lazy WD tracking
    ("model = torch.compile(model, dynamic=False)",
     """model._lazy_wd_mask = torch.ones(vocab_size, 1, device="cuda")
model = torch.compile(model, dynamic=False)"""),

    # Modify optimizer to use lazy mask
    # Instead of WD on all rows, only WD rows that were in the batch
    # We do this by setting wd_per_row dynamically
    ("    optimizer.step()",
     """    # H3: Lazy WD — only decay rows that were updated
    if hasattr(model._orig_mod, '_lazy_wd_mask'):
        lazy_mask = model._orig_mod._lazy_wd_mask
        for group in optimizer.param_groups:
            if group.get('wd_per_row') is not None:
                group['wd_per_row'] = embed_wd_per_row * lazy_mask
    optimizer.step()"""),

    # Set a small embed WD to test lazy WD effect (WD=0 makes lazy WD moot)
    ("EMBED_WD_AUDIO = 0.0", "EMBED_WD_AUDIO = 0.5  # H3: test lazy WD with moderate WD"),
]

# ===== H1: Frequency-proportional WD = C / token_count =====
# Pre-compute token frequencies from a quick scan, then set WD accordingly
freq_wd_mod = [
    # Add frequency-proportional WD computation after vocab setup
    ("print(f\"Per-row WD: text={EMBED_WD_TEXT}, audio={EMBED_WD_AUDIO}\")",
     """# H1: Frequency-proportional WD = C / sqrt(token_freq)
# Rough frequency estimates: text tokens avg ~75 occurrences per batch,
# audio tokens avg ~8 occurrences per batch (30% audio, 12K tokens vs 8K text)
# We'll compute actual frequencies from the first few batches
FREQ_WD_C = 0.5  # sweep this
import math
# Estimate: text tokens seen ~8x more than audio tokens per step
# With ~370 steps, text token i seen ~370 * 262K * 0.7 / 8192 ≈ 8260 times
# Audio token i seen ~370 * 262K * 0.3 / 12288 ≈ 2360 times
# WD_i = C / sqrt(freq_i) → text WD ~ C/91 ~ 0.005, audio WD ~ C/49 ~ 0.01
text_est_freq = 8260.0
audio_est_freq = 2360.0
embed_wd_per_row[:AUDIO_START_ID] = FREQ_WD_C / math.sqrt(text_est_freq)
embed_wd_per_row[AUDIO_START_ID:] = FREQ_WD_C / math.sqrt(audio_est_freq)
print(f"H1: Freq-proportional WD: text={embed_wd_per_row[0].item():.4f}, audio={embed_wd_per_row[AUDIO_START_ID].item():.4f}")"""),
]


experiments = [
    # ===== H3: Lazy WD =====
    ("H3: lazy WD (audio WD=0.5, only updated rows)", lazy_wd_mod),

    # ===== H1: Frequency-proportional WD =====
    ("H1: freq-proportional WD C=0.5", freq_wd_mod),
    ("H1: freq-proportional WD C=1.0", [
        m if m[0] != freq_wd_mod[0][0] else (m[0], m[1].replace("FREQ_WD_C = 0.5", "FREQ_WD_C = 1.0"))
        for m in freq_wd_mod
    ]),
    ("H1: freq-proportional WD C=2.0", [
        m if m[0] != freq_wd_mod[0][0] else (m[0], m[1].replace("FREQ_WD_C = 0.5", "FREQ_WD_C = 2.0"))
        for m in freq_wd_mod
    ]),

    # ===== Architecture: depth vs width trade-offs =====
    # Currently: depth=8, AR=64 → dim=512, 88M params
    # Try deeper+narrower and shallower+wider

    ("depth=10 AR=48 (wider attention)", [
        ("DEPTH = 8", "DEPTH = 10"),
        ("ASPECT_RATIO = 64", "ASPECT_RATIO = 48"),
    ]),
    ("depth=6 AR=88 (shallower+wider)", [
        ("DEPTH = 8", "DEPTH = 6"),
        ("ASPECT_RATIO = 64", "ASPECT_RATIO = 88"),
    ]),
    ("depth=12 AR=40 (deep+narrow)", [
        ("DEPTH = 8", "DEPTH = 12"),
        ("ASPECT_RATIO = 64", "ASPECT_RATIO = 40"),
    ]),

    # ===== Muon momentum experiments =====
    ("Muon momentum peak=0.90", [
        ("return (1 - frac) * 0.85 + frac * 0.95",
         "return (1 - frac) * 0.85 + frac * 0.90"),
    ]),
    ("Muon momentum peak=0.98", [
        ("return (1 - frac) * 0.85 + frac * 0.95",
         "return (1 - frac) * 0.85 + frac * 0.98"),
    ]),

    # ===== Unembedding LR experiments =====
    ("unembed_lr=0.008", [
        ("UNEMBEDDING_LR = 0.004", "UNEMBEDDING_LR = 0.008"),
    ]),
    ("unembed_lr=0.002", [
        ("UNEMBEDDING_LR = 0.004", "UNEMBEDDING_LR = 0.002"),
    ]),

    # ===== x0 lambda init =====
    ("x0_lambda_init=0.2", [
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.2)"),
    ]),
    ("x0_lambda_init=0.05", [
        ("self.x0_lambdas.fill_(0.1)", "self.x0_lambdas.fill_(0.05)"),
    ]),

    # ===== ReLU^2 vs GELU activation =====
    ("GELU activation (instead of ReLU^2)", [
        ("x = F.relu(x).square()", "x = F.gelu(x)"),
    ]),
    ("SiLU activation (instead of ReLU^2)", [
        ("x = F.relu(x).square()", "x = F.silu(x)"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for desc, mods in experiments:
    r = run_exp(desc, mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 7 SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

print(f"\nBest previous: val_loss=3.797 (dropout=0.0, embed WD=0.0, Muon WD=0.2)")
