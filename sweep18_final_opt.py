"""Sweep 18: Final optimizer refinement.

Best ever: cWD + WD=0.05 + mom=0.97 → 3.496 / 1.081 / 5.507

Tests:
1. mom0.97 + warmdown0.8 (combine top 2 findings)
2. mom0.97 + WD03 (combine mom+lower WD)
3. mom0.97 + warmdown0.8 + WD03
4. mom0.96 (explore momentum further)
5. mom0.97 + matLR0.07 (slightly higher matrix LR)
6. mom0.97 + matLR0.05 (slightly lower matrix LR)
7. mom0.97 + warmdown0.75
8. mom0.97 + warmdown0.85
9. mom0.97 + ns_steps=12 (more NS iterations)
10. mom0.97 + embLR1.0 (higher embedding LR)
11. mom0.97 + omni_t0.65 (try cross-modal with best optimizer)
12. mom0.97 + omni_t0.65 + warmdown0.8
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

# Base mods for new best optimizer
CWD = ("return WEIGHT_DECAY  # constant WD (was decaying to 0)",
       "return WEIGHT_DECAY * get_lr_multiplier(progress)  # coupled WD")
WD05 = ("WEIGHT_DECAY = 0.1", "WEIGHT_DECAY = 0.05")
WD03 = ("WEIGHT_DECAY = 0.1", "WEIGHT_DECAY = 0.03")
MOM97 = ("momentum=0.95, ns_steps=10", "momentum=0.97, ns_steps=10")
MOM96 = ("momentum=0.95, ns_steps=10", "momentum=0.96, ns_steps=10")
WD08 = ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.8")
WD075 = ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.75")
WD085 = ("WARMDOWN_RATIO = 0.7", "WARMDOWN_RATIO = 0.85")

def omni_base(text_r, audio_r, tts_r, asr_r):
    return [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(text_r, audio_r, tts_r, asr_r)),
        (PREFETCH_OLD, PREFETCH_NEW),
        (UNPACK_OLD, UNPACK_NEW),
    ]

experiments = [
    # 1. mom0.97 + warmdown0.8
    ("mom97+warmdown0.8", [CWD, WD05, MOM97, WD08], None),

    # 2. mom0.97 + WD03
    ("mom97+WD03", [CWD, WD03, MOM97], None),

    # 3. mom0.97 + warmdown0.8 + WD03
    ("mom97+warmdown0.8+WD03", [CWD, WD03, MOM97, WD08], None),

    # 4. mom0.96
    ("mom96", [CWD, WD05, MOM96], None),

    # 5. mom0.97 + matLR0.07
    ("mom97+matLR0.07", [CWD, WD05, MOM97, ("MATRIX_LR = 0.06", "MATRIX_LR = 0.07")], None),

    # 6. mom0.97 + matLR0.05
    ("mom97+matLR0.05", [CWD, WD05, MOM97, ("MATRIX_LR = 0.06", "MATRIX_LR = 0.05")], None),

    # 7. mom0.97 + warmdown0.75
    ("mom97+warmdown0.75", [CWD, WD05, MOM97, WD075], None),

    # 8. mom0.97 + warmdown0.85
    ("mom97+warmdown0.85", [CWD, WD05, MOM97, WD085], None),

    # 9. mom0.97 + ns_steps=12
    ("mom97+ns12", [CWD, WD05,
        ("momentum=0.95, ns_steps=10", "momentum=0.97, ns_steps=12"),
    ], None),

    # 10. mom0.97 + embLR1.0
    ("mom97+embLR1.0", [CWD, WD05, MOM97,
        ("EMBEDDING_LR = 0.8", "EMBEDDING_LR = 1.0"),
    ], None),

    # 11. mom0.97 + omni t0.65
    ("mom97+omni_t0.65",
     omni_base(0.65, 0.0, 0.175, 0.175) + [CWD, WD05, MOM97], None),

    # 12. mom0.97 + omni t0.65 + warmdown0.8
    ("mom97+omni_t0.65+wd0.8",
     omni_base(0.65, 0.0, 0.175, 0.175) + [CWD, WD05, MOM97, WD08], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 18 FINAL OPTIMIZER SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
