"""Sweep 17: Cross-modal training on top of the new best baseline.

Key finding from sweep16: baseline+cWD+WD05 = 3.497 (best ever).
The optimizer improvements alone provide more benefit than cross-modal data.

Question: Does cross-modal training provide ANY additional benefit on top of
the optimized baseline? Or does the improved optimizer already capture what
cross-modal data provided?

Also tests: WD sweep around 0.05, momentum sweep, and WD03 as even lower WD.

Tests:
1. New baseline: cWD+WD05 (reference)
2. cWD+WD05+t0.7+tts0.15+asr0.15 (omni on new baseline)
3. cWD+WD05+t0.65+tts0.175+asr0.175
4. cWD+WD05+t0.6+tts0.2+asr0.2
5. cWD+WD05+mom0.98 (baseline, no omni)
6. cWD+WD05+mom0.98+t0.7 (omni + all tricks)
7. cWD+WD03 (even lower WD baseline)
8. cWD+WD08 (interpolated WD)
9. cWD+WD05+mom0.97 (intermediate momentum)
10. cWD+WD05+warmdown0.8 (longer warmdown)
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


DATALOADER_OLD = 'train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")'

def omni_dataloader(text_r, audio_r, tts_r, asr_r):
    return (f'train_loader = make_omni_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", '
            f'text_ratio={text_r}, audio_ratio={audio_r}, tts_ratio={tts_r}, asr_ratio={asr_r})')

IMPORT_OLD = 'from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOTAL_VOCAB_SIZE, AUDIO_START_ID, Tokenizer, make_dataloader, evaluate_val_loss'
IMPORT_NEW = 'from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOTAL_VOCAB_SIZE, AUDIO_START_ID, Tokenizer, make_dataloader, make_omni_dataloader, evaluate_val_loss'

PREFETCH_OLD = 'x, y, epoch = next(train_loader)  # prefetch first batch'
PREFETCH_NEW = 'x, y, epoch, _rw = next(train_loader)  # prefetch first batch'
UNPACK_OLD = '        x, y, epoch = next(train_loader)'
UNPACK_NEW = '        x, y, epoch, _rw = next(train_loader)'

COUPLED_WD_OLD = "return WEIGHT_DECAY  # constant WD (was decaying to 0)"
COUPLED_WD_NEW = "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD"
MOM98_OLD = "momentum=0.95, ns_steps=10"
MOM98_NEW = "momentum=0.98, ns_steps=10"
WD005_OLD = "WEIGHT_DECAY = 0.1"

def omni_base(text_r, audio_r, tts_r, asr_r):
    return [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(text_r, audio_r, tts_r, asr_r)),
        (PREFETCH_OLD, PREFETCH_NEW),
        (UNPACK_OLD, UNPACK_NEW),
    ]

experiments = [
    # ===== 1. New baseline reference =====
    ("newbase: cWD+WD05", [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
    ], None),

    # ===== 2-4. Cross-modal on new baseline =====
    ("newbase+omni_t0.7",
     omni_base(0.7, 0.0, 0.15, 0.15) + [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
    ], None),

    ("newbase+omni_t0.65",
     omni_base(0.65, 0.0, 0.175, 0.175) + [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
    ], None),

    ("newbase+omni_t0.6",
     omni_base(0.6, 0.0, 0.2, 0.2) + [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
    ], None),

    # ===== 5-6. Mom 0.98 variants =====
    ("newbase+mom98",  [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
        (MOM98_OLD, MOM98_NEW),
    ], None),

    ("newbase+mom98+omni_t0.7",
     omni_base(0.7, 0.0, 0.15, 0.15) + [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
        (MOM98_OLD, MOM98_NEW),
    ], None),

    # ===== 7-8. WD alternatives =====
    ("newbase_WD03: cWD+WD03", [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.03"),
    ], None),

    ("newbase_WD08: cWD+WD08", [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.08"),
    ], None),

    # ===== 9. Intermediate momentum =====
    ("newbase+mom97", [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
        (MOM98_OLD, "momentum=0.97, ns_steps=10"),
    ], None),

    # ===== 10. Longer warmdown =====
    ("newbase+warmdown0.8", [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, "WEIGHT_DECAY = 0.05"),
        ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.8"),
    ], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 17 NEW BASELINE SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
