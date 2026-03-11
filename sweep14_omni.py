"""Sweep 14: Omni-model convergence experiments.

Key question: How does cross-modal (ASR/TTS) training affect convergence
compared to separate text/audio training?

Tests:
1. Baseline: text-only + audio-only (separate sequences, current approach)
2. Omni: text + audio + TTS pairs + ASR pairs
3. TTS-only cross-modal: text + TTS pairs (text→audio transition)
4. ASR-only cross-modal: text + ASR pairs (audio→text transition)
5. Balanced omni: equal mix of all four modes
6. Omni with coupled WD (our best optimizer finding)
7. Omni with momentum=0.98 (better with more steps)

Measures: val_loss, text_bpb, audio_loss (overall + per-mode)
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
# We need to modify train.py's dataloader call

# Current train.py uses:
#   train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
# We need to change it to use make_omni_dataloader with different ratios

DATALOADER_OLD = 'train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")'

def omni_dataloader(text_r, audio_r, tts_r, asr_r):
    """Generate the replacement dataloader code for omni training."""
    return (f'train_loader = make_omni_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", '
            f'text_ratio={text_r}, audio_ratio={audio_r}, tts_ratio={tts_r}, asr_ratio={asr_r})')

IMPORT_OLD = 'from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOTAL_VOCAB_SIZE, AUDIO_START_ID, Tokenizer, make_dataloader, evaluate_val_loss'
IMPORT_NEW = 'from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOTAL_VOCAB_SIZE, AUDIO_START_ID, Tokenizer, make_dataloader, make_omni_dataloader, evaluate_val_loss'

experiments = [
    # ===== Baseline (current best: separate text + audio) =====
    ("baseline: text0.8+audio0.2", [
    ], [
        ("AUDIO_MIX_RATIO = 0.2", "AUDIO_MIX_RATIO = 0.2"),  # no-op, just for baseline
    ]),

    # ===== Omni: add cross-modal TTS/ASR pairs =====
    ("omni: t0.4+a0.2+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.4, 0.2, 0.2, 0.2)),
    ], None),

    ("omni: t0.5+a0.1+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.2, 0.2)),
    ], None),

    # ===== TTS-focused cross-modal =====
    ("tts_cross: t0.5+a0.1+tts0.3+asr0.1", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.3, 0.1)),
    ], None),

    # ===== ASR-focused cross-modal =====
    ("asr_cross: t0.5+a0.1+tts0.1+asr0.3", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.5, 0.1, 0.1, 0.3)),
    ], None),

    # ===== Heavy cross-modal (60% paired) =====
    ("heavy_cross: t0.2+a0.1+tts0.35+asr0.35", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.2, 0.1, 0.35, 0.35)),
    ], None),

    # ===== Omni + coupled WD =====
    ("omni+coupledWD: t0.4+a0.2+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.4, 0.2, 0.2, 0.2)),
        ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
         "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD"),
    ], None),

    # ===== Omni + momentum=0.98 =====
    ("omni+mom0.98: t0.4+a0.2+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.4, 0.2, 0.2, 0.2)),
        ("momentum=0.95, ns_steps=10", "momentum=0.98, ns_steps=10"),
    ], None),

    # ===== Cross-modal only (no pure text/audio) =====
    ("pure_cross: tts0.5+asr0.5", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.0, 0.0, 0.5, 0.5)),
    ], None),

    # ===== Text-heavy omni =====
    ("text_heavy_omni: t0.6+tts0.2+asr0.2", [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(0.6, 0.0, 0.2, 0.2)),
    ], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 14 OMNI SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
