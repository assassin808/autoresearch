"""Sweep 15: Optimizer-focused omni convergence improvements.

Key insight from sweep14: Cross-modal training (TTS/ASR pairs) consistently
improves audio_loss (5.54→5.33) but hurts text (1.09→1.19 bpb).

Question: Can optimizer modifications minimize the text-quality cost of
cross-modal training while preserving or improving the audio benefit?

Tests:
1. High text ratio omni (t0.7+tts0.15+asr0.15) — minimize text degradation
2. Omni + coupled WD + high text ratio — combine best findings
3. Omni + gradient norm balance (H3) — equalize modality gradients
4. Omni + modality rebalance (H6) — MILES-style dynamic rebalancing
5. Omni + higher matrix LR (0.08) — compensate for multi-task dilution
6. Omni + lower WD (0.05) — less regularization = more room for cross-modal learning
7. Omni + momentum 0.98 + coupled WD — combine best optimizer tricks
8. Omni with cross-modal loss weight 0.5 — downweight paired data loss
9. Omni + deeper model (DEPTH=10) — more capacity for cross-modal learning
10. Omni + text0.65+audio0.05+tts0.15+asr0.15 — light audio, heavy cross-modal
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


# The key change: switch from make_dataloader to make_omni_dataloader
DATALOADER_OLD = 'train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")'

def omni_dataloader(text_r, audio_r, tts_r, asr_r, cross_modal_weight=1.0):
    """Generate the replacement dataloader code for omni training."""
    return (f'train_loader = make_omni_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", '
            f'text_ratio={text_r}, audio_ratio={audio_r}, tts_ratio={tts_r}, asr_ratio={asr_r}, '
            f'cross_modal_weight={cross_modal_weight})')

IMPORT_OLD = 'from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOTAL_VOCAB_SIZE, AUDIO_START_ID, Tokenizer, make_dataloader, evaluate_val_loss'
IMPORT_NEW = 'from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOTAL_VOCAB_SIZE, AUDIO_START_ID, Tokenizer, make_dataloader, make_omni_dataloader, evaluate_val_loss'

# The omni dataloader now returns (inputs, targets, epoch, row_weights)
# We need to modify the training loop to unpack row_weights and use them for loss weighting
UNPACK_OLD = '        x, y, epoch = next(train_loader)'
UNPACK_NEW_WEIGHTED = '        x, y, epoch, row_weights = next(train_loader)'

# For weighted loss: compute per-row loss and weight it
LOSS_OLD = """            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps"""

LOSS_WEIGHTED = """            per_sample_loss = model(x, y, reduction='none').view(x.shape[0], -1).mean(dim=1)
            loss = (per_sample_loss * row_weights).mean()
        train_loss = loss.detach()
        loss = loss / grad_accum_steps"""

# For normal (unweighted) omni training, we just ignore row_weights
UNPACK_IGNORE_WEIGHTS = '        x, y, epoch, _rw = next(train_loader)'

# Also need to fix the prefetch line
PREFETCH_OLD = 'x, y, epoch = next(train_loader)  # prefetch first batch'
PREFETCH_IGNORE = 'x, y, epoch, _rw = next(train_loader)  # prefetch first batch'
PREFETCH_WEIGHTED = 'x, y, epoch, row_weights = next(train_loader)  # prefetch first batch'

experiments = [
    # ===== 1. High text ratio omni =====
    ("omni_high_text: t0.7+tts0.15+asr0.15", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.7, 0.0, 0.15, 0.15)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
    ], None),

    # ===== 2. Omni + coupled WD + high text =====
    ("omni_ht+coupledWD: t0.7+tts0.15+asr0.15", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.7, 0.0, 0.15, 0.15)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD"),
    ], None),

    # ===== 3. Omni + grad norm balance (H3) =====
    ("omni+gradbal: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("GRAD_NORM_BALANCE = False", "GRAD_NORM_BALANCE = True"),
    ], None),

    # ===== 4. Omni + modality rebalance (H6) =====
    ("omni+rebal: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("MODALITY_REBALANCE = False", "MODALITY_REBALANCE = True"),
    ], None),

    # ===== 5. Omni + higher matrix LR =====
    ("omni+matLR0.08: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("MATRIX_LR = 0.06", "MATRIX_LR = 0.08"),
    ], None),

    # ===== 6. Omni + lower WD =====
    ("omni+WD0.05: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("WEIGHT_DECAY = 0.1", "WEIGHT_DECAY = 0.05"),
    ], None),

    # ===== 7. Omni + mom0.98 + coupled WD =====
    ("omni+mom98+cWD: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("momentum=0.95, ns_steps=10", "momentum=0.98, ns_steps=10"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD"),
    ], None),

    # ===== 8. Omni + cross-modal loss weight 0.5 =====
    ("omni+cmw0.5: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2, cross_modal_weight=0.5)),
        (PREFETCH_OLD, PREFETCH_WEIGHTED),
        (UNPACK_OLD, UNPACK_NEW_WEIGHTED),
        (LOSS_OLD, LOSS_WEIGHTED),
    ], None),

    # ===== 9. Omni + deeper model (DEPTH=10) =====
    ("omni+depth10: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("DEPTH = 8", "DEPTH = 10"),
    ], None),

    # ===== 10. Omni + light audio, heavy cross-modal =====
    ("omni_light_audio: t0.65+a0.05+tts0.15+asr0.15", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.65, 0.05, 0.15, 0.15)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
    ], None),

    # ===== 11. Best combo: high text + coupledWD + mom0.98 =====
    ("omni_best: t0.7+tts0.15+asr0.15+cWD+m98", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.7, 0.0, 0.15, 0.15)),
        (PREFETCH_OLD, PREFETCH_IGNORE),
        (UNPACK_OLD, UNPACK_IGNORE_WEIGHTS),
        ("momentum=0.95, ns_steps=10", "momentum=0.98, ns_steps=10"),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD"),
    ], None),

    # ===== 12. Omni + cross-modal loss weight 0.3 (stronger downweight) =====
    ("omni+cmw0.3: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2, cross_modal_weight=0.3)),
        (PREFETCH_OLD, PREFETCH_WEIGHTED),
        (UNPACK_OLD, UNPACK_NEW_WEIGHTED),
        (LOSS_OLD, LOSS_WEIGHTED),
    ], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 15 OMNI OPTIMIZER SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
