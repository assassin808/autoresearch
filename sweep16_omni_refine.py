"""Sweep 16: Refinement of omni-model convergence findings.

Key findings from sweep14+15:
- Cross-modal (TTS/ASR) consistently helps audio but hurts text
- Coupled WD nearly eliminates text penalty at t0.7
- Lower WD (0.05) improves both text and audio
- mom0.98+coupledWD gives best audio (5.393)

This sweep refines the Pareto frontier between text and audio quality.

Tests:
1-4. Ratio sweep: t0.6 to t0.75 with coupledWD
5. WD0.05 + coupledWD + t0.7
6. WD0.05 + coupledWD + t0.6 (more cross-modal)
7. mom98 + coupledWD + WD0.05 + t0.7 (all best tricks, high text)
8. mom98 + coupledWD + WD0.05 + t0.6 (all best tricks, balanced)
9. mom98 + coupledWD + WD0.05 + t0.5+a0.1+tts0.2+asr0.2 (max audio)
10. Baseline + coupledWD (no cross-modal, just optimizer improvement)
11. Baseline + WD0.05 + coupledWD (best optimizer, no cross-modal)
12. ASR-heavy with best opts: t0.6+a0.05+tts0.1+asr0.25+cWD+WD0.05
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
WD005_NEW = "WEIGHT_DECAY = 0.05"

# Common omni mods (import, prefetch, unpack)
def omni_base(text_r, audio_r, tts_r, asr_r):
    return [
        (IMPORT_OLD, IMPORT_NEW),
        (DATALOADER_OLD, omni_dataloader(text_r, audio_r, tts_r, asr_r)),
        (PREFETCH_OLD, PREFETCH_NEW),
        (UNPACK_OLD, UNPACK_NEW),
    ]

experiments = [
    # ===== 1-4. Ratio sweep with coupled WD =====
    ("cWD+t0.60: t0.6+tts0.2+asr0.2",
     omni_base(0.6, 0.0, 0.2, 0.2) + [(COUPLED_WD_OLD, COUPLED_WD_NEW)], None),

    ("cWD+t0.65: t0.65+tts0.175+asr0.175",
     omni_base(0.65, 0.0, 0.175, 0.175) + [(COUPLED_WD_OLD, COUPLED_WD_NEW)], None),

    ("cWD+t0.70: t0.7+tts0.15+asr0.15",
     omni_base(0.7, 0.0, 0.15, 0.15) + [(COUPLED_WD_OLD, COUPLED_WD_NEW)], None),

    ("cWD+t0.75: t0.75+tts0.125+asr0.125",
     omni_base(0.75, 0.0, 0.125, 0.125) + [(COUPLED_WD_OLD, COUPLED_WD_NEW)], None),

    # ===== 5-6. WD0.05 + coupledWD =====
    ("cWD+WD05+t0.7: t0.7+tts0.15+asr0.15",
     omni_base(0.7, 0.0, 0.15, 0.15) + [(COUPLED_WD_OLD, COUPLED_WD_NEW), (WD005_OLD, WD005_NEW)], None),

    ("cWD+WD05+t0.6: t0.6+tts0.2+asr0.2",
     omni_base(0.6, 0.0, 0.2, 0.2) + [(COUPLED_WD_OLD, COUPLED_WD_NEW), (WD005_OLD, WD005_NEW)], None),

    # ===== 7-9. All best tricks (mom98+cWD+WD0.05) at different ratios =====
    ("allbest+t0.7: t0.7+tts0.15+asr0.15",
     omni_base(0.7, 0.0, 0.15, 0.15) + [(COUPLED_WD_OLD, COUPLED_WD_NEW), (WD005_OLD, WD005_NEW), (MOM98_OLD, MOM98_NEW)], None),

    ("allbest+t0.6: t0.6+tts0.2+asr0.2",
     omni_base(0.6, 0.0, 0.2, 0.2) + [(COUPLED_WD_OLD, COUPLED_WD_NEW), (WD005_OLD, WD005_NEW), (MOM98_OLD, MOM98_NEW)], None),

    ("allbest+t0.5: t0.5+a0.1+tts0.2+asr0.2",
     omni_base(0.5, 0.1, 0.2, 0.2) + [(COUPLED_WD_OLD, COUPLED_WD_NEW), (WD005_OLD, WD005_NEW), (MOM98_OLD, MOM98_NEW)], None),

    # ===== 10-11. Baselines with optimizer improvements (no cross-modal) =====
    ("baseline+cWD", [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
    ], None),

    ("baseline+cWD+WD05", [
        (COUPLED_WD_OLD, COUPLED_WD_NEW),
        (WD005_OLD, WD005_NEW),
    ], None),

    # ===== 12. ASR-heavy with best optimizer =====
    ("asr_heavy+cWD+WD05: t0.6+a0.05+tts0.1+asr0.25",
     omni_base(0.6, 0.05, 0.1, 0.25) + [(COUPLED_WD_OLD, COUPLED_WD_NEW), (WD005_OLD, WD005_NEW)], None),
]

print(f"Running {len(experiments)} experiments...")
results = []
for item in experiments:
    desc, train_mods, prepare_mods = item
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods)
    results.append((desc, r))

print("\n" + "="*60)
print("SWEEP 16 OMNI REFINEMENT SUMMARY")
print("="*60)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")
