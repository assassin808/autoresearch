"""Sweep 13: Final experiments — weight tying, dead token pruning, rank regularization.

Based on diagnostic findings:
- 246 dead audio tokens (zero gradient, decayed by WD)
- Audio embedding rank declining monotonically (434 vs text 465)
- Text gets 15x more gradient signal than audio
"""
import subprocess, re, os

PYTHON = ".venv/bin/python"

def run_exp(desc, train_mods=None, prepare_mods=None):
    with open("train.py") as f:
        original_train = f.read()
    with open("prepare.py") as f:
        original_prepare = f.read()
    modified_train = original_train
    modified_prepare = original_prepare
    if train_mods:
        for old, new in train_mods:
            if modified_train.count(old) == 0:
                print(f"  WARNING: train.py not found: {old[:80]}...")
                with open("train.py", "w") as f:
                    f.write(original_train)
                with open("prepare.py", "w") as f:
                    f.write(original_prepare)
                return None
            modified_train = modified_train.replace(old, new, 1)
    if prepare_mods:
        for old, new in prepare_mods:
            if modified_prepare.count(old) == 0:
                print(f"  WARNING: prepare.py not found: {old[:80]}...")
                with open("train.py", "w") as f:
                    f.write(original_train)
                with open("prepare.py", "w") as f:
                    f.write(original_prepare)
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


# ===== H10: Weight tying =====
# Fix: tie weights in __init__, then fix optimizer to skip lm_head params
weight_tie_init = '''class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # WEIGHT TYING'''

weight_tie_optimizer = '''        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))'''

weight_tie_optimizer_new = '''        embedding_params = list(self.transformer.wte.parameters())
        # Weight tying: lm_head.weight IS wte.weight, so skip it
        if self.lm_head.weight is self.transformer.wte.weight:
            lm_head_params = []
        else:
            lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        all_params = set(id(p) for p in self.parameters())
        grouped_params = set(id(p) for p in matrix_params + embedding_params + lm_head_params + value_embeds_params + resid_params + x0_params)
        assert all_params == grouped_params, f"param mismatch: {len(all_params)} vs {len(grouped_params)}"'''

# Also need to fix init_weights to not reinit lm_head when tied
weight_tie_init_weights_old = '''        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)'''
weight_tie_init_weights_new = '''        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        if self.lm_head.weight is not self.transformer.wte.weight:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)'''

# Also fix lm_head group: if no lm_head params, skip building the group
weight_tie_group_old = '''        lm_head_group = dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=2.0)
        embed_group = dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=2.0)'''
weight_tie_group_new = '''        if lm_head_params:
            lm_head_group = dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=2.0)
        else:
            lm_head_group = None
        embed_group = dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=2.0)'''

# Fix the param_groups list
weight_tie_groups_list_old = '''        param_groups = [
            lm_head_group,
            embed_group,'''
weight_tie_groups_list_new = '''        param_groups = [g for g in [
            lm_head_group,
            embed_group,] if g is not None] + ['''

# Actually that's getting complicated. Let me simplify:
# Instead of all those patches, let me just do a simpler approach

experiments = [
    # ===== H10: Weight tying (proper fix) =====
    ("H10: weight_tying", [
        # Tie weights in __init__
        ("        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)\n        self.resid_lambdas",
         "        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)\n        self.lm_head.weight = self.transformer.wte.weight  # weight tying\n        self.resid_lambdas"),
        # Fix init_weights: don't reinit lm_head when tied
        ("        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)\n        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)",
         "        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)\n        if self.lm_head.weight is not self.transformer.wte.weight:\n            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)"),
        # Fix optimizer: handle shared params
        ("        embedding_params = list(self.transformer.wte.parameters())\n        lm_head_params = list(self.lm_head.parameters())\n        resid_params = [self.resid_lambdas]\n        x0_params = [self.x0_lambdas]\n        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +\n            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))",
         "        embedding_params = list(self.transformer.wte.parameters())\n        if self.lm_head.weight is self.transformer.wte.weight:\n            lm_head_params = []\n        else:\n            lm_head_params = list(self.lm_head.parameters())\n        resid_params = [self.resid_lambdas]\n        x0_params = [self.x0_lambdas]\n        all_ids = set(id(p) for p in self.parameters())\n        grp_ids = set(id(p) for p in matrix_params + embedding_params + lm_head_params + value_embeds_params + resid_params + x0_params)\n        assert all_ids == grp_ids, f'param mismatch: {len(all_ids)} vs {len(grp_ids)}'"),
        # Fix lm_head group: conditional
        ("        lm_head_group = dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=2.0)\n        embed_group",
         "        lm_head_group = dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=2.0) if lm_head_params else None\n        embed_group"),
        # Fix wd_per_row/lr_per_row for lm_head (skip if None)
        ("            lm_head_group['wd_per_row'] = embed_wd_per_row",
         "            if lm_head_group is not None: lm_head_group['wd_per_row'] = embed_wd_per_row"),
        ("            lm_head_group['lr_per_row'] = embed_lr_per_row",
         "            if lm_head_group is not None: lm_head_group['lr_per_row'] = embed_lr_per_row"),
        # Fix param_groups: filter None
        ("        param_groups = [\n            lm_head_group,",
         "        param_groups = [g for g in [\n            lm_head_group,"),
        ("            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),\n        ]",
         "            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),\n        ] if g is not None]"),
    ], None),

    # ===== Zero WD on dead audio tokens (diagnostic-informed) =====
    # Set embed WD to zero for ALL tokens (already the default per results.tsv)
    # But test: add small WD only for high-freq tokens
    ("freq_wd_C0.5+ratio0.2", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("print(f\"Per-row WD: text={EMBED_WD_TEXT}, audio={EMBED_WD_AUDIO}\")",
         """import math
FREQ_WD_C = 0.5
text_est_freq = 8260.0
audio_est_freq = 2360.0
embed_wd_per_row[:AUDIO_START_ID] = FREQ_WD_C / math.sqrt(text_est_freq)
embed_wd_per_row[AUDIO_START_ID:] = FREQ_WD_C / math.sqrt(audio_est_freq)
print(f"Freq-proportional WD: text={embed_wd_per_row[0].item():.4f}, audio={embed_wd_per_row[AUDIO_START_ID].item():.4f}")"""),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== Muon momentum=0.98 (higher momentum for more steps) =====
    ("momentum0.98+ratio0.2+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("momentum=0.95, ns_steps=10", "momentum=0.98, ns_steps=10"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== LR schedule: cosine vs linear warmdown =====
    ("cosine_warmdown+ratio0.2+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("        cooldown = (1.0 - progress) / WARMDOWN_RATIO\n        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC",
         "        import math\n        cooldown = (1.0 - progress) / WARMDOWN_RATIO\n        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * 0.5 * (1.0 + math.cos(math.pi * (1.0 - cooldown)))"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== Higher warmdown ratio at the new step count =====
    ("warmdown0.8+ratio0.2+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.8"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== Separate audio LR (lower LR for undertrained modality) =====
    ("audioLR_low+ratio0.2+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("embed_lr_per_row = None",
         """dmodel_lr_scale_tmp = (model.config.n_embd / 768) ** -0.5
embed_lr_per_row = torch.ones(vocab_size, 1, device=device) * EMBEDDING_LR * dmodel_lr_scale_tmp
embed_lr_per_row[AUDIO_START_ID:] *= 0.5  # half LR for audio
print(f"Audio embed LR scaled to 0.5x")
# embed_lr_per_row = None  # disabled"""),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),

    # ===== Best combo from sweep12 + coupled WD =====
    ("ratio0.15+WD0.1+coupledWD", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.15"),
    ]),

    # ===== NS5 with more steps (batch64K) =====
    ("batch64K+NS10+ratio0.2+WD0.1", [
        ("WEIGHT_DECAY = 0.15", "WEIGHT_DECAY = 0.1"),
        ("TOTAL_BATCH_SIZE = 2**17", "TOTAL_BATCH_SIZE = 2**16"),
    ], [
        ("AUDIO_MIX_RATIO = 0.3", "AUDIO_MIX_RATIO = 0.2"),
    ]),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 13 SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
