"""Sweep 8: Follow-up experiments based on sweep6+7 results.

Focus areas decided after earlier sweeps complete:
- Architecture (H7): Decoupled embedding dimension
- Combined best settings
- Fine-grained grid around any improvements found
"""
import subprocess, re, os

PYTHON = ".venv/bin/python"

def run_exp(desc, train_mods=None, prepare_mods=None):
    """Run experiment modifying train.py and optionally prepare.py."""
    with open("train.py") as f:
        original_train = f.read()
    with open("prepare.py") as f:
        original_prepare = f.read()

    modified_train = original_train
    modified_prepare = original_prepare

    if train_mods:
        for old, new in train_mods:
            count = modified_train.count(old)
            if count == 0:
                print(f"  WARNING: train.py pattern not found: {old[:80]}...")
                return None
            modified_train = modified_train.replace(old, new, 1)

    if prepare_mods:
        for old, new in prepare_mods:
            count = modified_prepare.count(old)
            if count == 0:
                print(f"  WARNING: prepare.py pattern not found: {old[:80]}...")
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


# ===== H7: Decoupled embed_dim =====
# Replace Embedding(vocab, 512) with Embedding(vocab, 256) + Linear(256, 512)
# This halves embedding params (31M instead of 62M), freeing param budget

# For H7, we need substantial code changes, so prepare them as text replacements
h7_embed_changes = [
    # Add EMBED_DIM config
    ("WINDOW_PATTERN = \"SSSL\"",
     "WINDOW_PATTERN = \"SSSL\"\nEMBED_DIM = 256  # H7: decoupled embedding dimension"),

    # Modify GPTConfig to include embed_dim
    ("    n_embd: int = 768\n    window_pattern",
     "    n_embd: int = 768\n    n_embed: int = 256  # H7: embedding dimension (can differ from model dim)\n    window_pattern"),

    # Modify GPT.__init__ to use smaller embeddings + projection
    ('        self.transformer = nn.ModuleDict({\n            "wte": nn.Embedding(config.vocab_size, config.n_embd),',
     '        self.transformer = nn.ModuleDict({\n            "wte": nn.Embedding(config.vocab_size, config.n_embed),\n            "embed_proj": nn.Linear(config.n_embed, config.n_embd, bias=False),'),

    # Modify value embeddings to use smaller dim too
    ("        self.value_embeds = nn.ModuleDict({\n            str(i): nn.Embedding(config.vocab_size, kv_dim)",
     "        self.value_embeds = nn.ModuleDict({\n            str(i): nn.Embedding(config.vocab_size, min(kv_dim, config.n_embed))"),

    # Add value embed projection if needed
    # Actually, VE is kv_dim which might be smaller than n_embed, so let's leave VE alone for now
    # and just handle the main embedding

    # Modify forward to add projection
    ("        x = self.transformer.wte(idx)\n        x = norm(x)",
     "        x = self.transformer.wte(idx)\n        x = self.transformer.embed_proj(x)\n        x = norm(x)"),

    # Modify lm_head to project back
    ("        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)",
     "        self.lm_head_proj = nn.Linear(config.n_embd, config.n_embed, bias=False)\n        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)"),

    # Modify forward for lm_head projection
    ("        logits = self.lm_head(x)",
     "        logits = self.lm_head(self.lm_head_proj(x))"),

    # Modify build_model_config to pass embed_dim
    ("        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,\n        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,",
     "        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,\n        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,\n        n_embed=EMBED_DIM,"),

    # Init the projection layers
    ("        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)",
     "        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)\n        if hasattr(self.transformer, 'embed_proj'):\n            torch.nn.init.normal_(self.transformer.embed_proj.weight, mean=0.0, std=self.config.n_embed**-0.5)\n        if hasattr(self, 'lm_head_proj'):\n            torch.nn.init.normal_(self.lm_head_proj.weight, mean=0.0, std=self.config.n_embd**-0.5)"),

    # Cast embeddings
    ("        self.transformer.wte.to(dtype=torch.bfloat16)",
     "        self.transformer.wte.to(dtype=torch.bfloat16)\n        if hasattr(self.transformer, 'embed_proj'):\n            self.transformer.embed_proj.to(dtype=torch.bfloat16)"),
]

# Audio mix ratio experiments (modify prepare.py)
audio_ratio_experiments = [
    ("audio_ratio=0.4", None, [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.4"),
    ]),
    ("audio_ratio=0.2", None, [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),
    ("audio_ratio=0.5", None, [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.5"),
    ]),
]

experiments = [
    # Audio mix ratio (modifies prepare.py)
    *audio_ratio_experiments,

    # H7: Decoupled embed_dim = 256
    # ("H7: embed_dim=256 (decoupled)", h7_embed_changes),  # complex, test last

    # Deeper model with same param budget
    ("depth=10 AR=52", [
        ("DEPTH = 8", "DEPTH = 10"),
        ("ASPECT_RATIO = 64", "ASPECT_RATIO = 52"),
    ], None),
    ("depth=12 AR=44", [
        ("DEPTH = 8", "DEPTH = 12"),
        ("ASPECT_RATIO = 64", "ASPECT_RATIO = 44"),
    ], None),
    ("depth=16 AR=32", [
        ("DEPTH = 8", "DEPTH = 16"),
        ("ASPECT_RATIO = 64", "ASPECT_RATIO = 32"),
    ], None),

    # Wider model
    ("depth=6 AR=80", [
        ("DEPTH = 8", "DEPTH = 6"),
        ("ASPECT_RATIO = 64", "ASPECT_RATIO = 80"),
    ], None),

    # Window pattern experiments
    ("window=SSLL", [
        ('WINDOW_PATTERN = "SSSL"', 'WINDOW_PATTERN = "SSLL"'),
    ], None),
    ("window=SLSL", [
        ('WINDOW_PATTERN = "SSSL"', 'WINDOW_PATTERN = "SLSL"'),
    ], None),
    ("window=LLLL (all full attention)", [
        ('WINDOW_PATTERN = "SSSL"', 'WINDOW_PATTERN = "LLLL"'),
    ], None),

    # Higher Muon WD (since WD=0.2 is current best, try 0.25, 0.3)
    ("Muon WD=0.25", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.25"),
    ], None),
    ("Muon WD=0.3", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.3"),
    ], None),

    # Scalar LR experiments
    ("scalar_lr=1.0", [
        ("SCALAR_LR = 0.5", "SCALAR_LR = 1.0"),
    ], None),
    ("scalar_lr=0.2", [
        ("SCALAR_LR = 0.5", "SCALAR_LR = 0.2"),
    ], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    if len(item) == 2:
        desc, train_mods = item
        r = run_exp(desc, train_mods=train_mods)
    else:
        desc, train_mods, prepare_mods = item
        r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 8 SUMMARY")
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
