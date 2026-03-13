"""Sweep 39: Post-NS Omni-Optimizer — Interventions AFTER Newton-Schulz.

Key insight from sweeps 37-38:
- Pre-NS gradient manipulation is washed out by NS orthogonalization
- Alternating pure batches + audio LR boost are the only things that helped
- Need to operate AFTER NS or change the optimizer dynamics fundamentally

New approaches:
B1: Alternating batches + audio LR sweep (finer grain)
B2: Alternating batches + cosine-scheduled audio ratio (start text-heavy, ramp audio)
B3: Post-optimizer param nudge: after each audio step, apply extra update in
    the direction that REDUCES audio loss specifically
B4: Modality-aware weight decay: lower WD for audio steps (preserve learned audio features)
B5: Audio gradient accumulation: accumulate N audio gradients before stepping
    (reduce noise in audio signal before NS)
B6: Separate Muon groups with different LR for audio vs text steps
"""
import subprocess, re, os, json, hashlib

PYTHON = ".venv/bin/python"


def run_exp(desc, train_mods=None, prepare_mods=None, timeout=600):
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
                with open("train.py", "w") as f: f.write(original_train)
                with open("prepare.py", "w") as f: f.write(original_prepare)
                return None
            modified_train = modified_train.replace(old, new, 1)
    if prepare_mods:
        for old, new in prepare_mods:
            if modified_prepare.count(old) == 0:
                print(f"  WARNING: prepare.py not found: {old[:80]}...")
                with open("train.py", "w") as f: f.write(original_train)
                with open("prepare.py", "w") as f: f.write(original_prepare)
                return None
            modified_prepare = modified_prepare.replace(old, new, 1)
    with open("train.py", "w") as f: f.write(modified_train)
    if prepare_mods:
        with open("prepare.py", "w") as f: f.write(modified_prepare)
    commit = hashlib.md5(desc.encode()).hexdigest()[:7]
    print(f"\n{'='*60}\nEXP {commit}: {desc}\n{'='*60}")
    try:
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        with open("results.tsv", "a") as f: f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f: f.write(original_train)
        with open("prepare.py", "w") as f: f.write(original_prepare)
        return None
    vl = re.search(r'^val_loss:\s+(\S+)', output, re.M)
    tb = re.search(r'^text_bpb:\s+(\S+)', output, re.M)
    al = re.search(r'^audio_loss:\s+(\S+)', output, re.M)
    if not vl or result.returncode != 0:
        print(f"  CRASHED")
        print(output[-1500:] if output else "no output")
        with open("results.tsv", "a") as f: f.write(f"{commit}\t0\t0\t0\t16.3\tcrash\t{desc}\n")
        with open("train.py", "w") as f: f.write(original_train)
        with open("prepare.py", "w") as f: f.write(original_prepare)
        return None
    val_loss, text_bpb, audio_loss = float(vl.group(1)), float(tb.group(1)), float(al.group(1))
    print(f"  val_loss={val_loss:.6f} text_bpb={text_bpb:.6f} audio_loss={audio_loss:.6f}")
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{val_loss:.6f}\t{text_bpb:.6f}\t{audio_loss:.6f}\t16.3\tkeep\t{desc}\n")
    with open("train.py", "w") as f: f.write(original_train)
    with open("prepare.py", "w") as f: f.write(original_prepare)
    return {"val_loss": val_loss, "text_bpb": text_bpb, "audio_loss": audio_loss}


# ============================================================
# Injection points
# ============================================================
AFTER_DEPTH = "DEPTH = 8               # number of transformer layers"
BEFORE_OPT_STEP = "    optimizer.step()"
AFTER_OPT_STEP = "    optimizer.step()\n"
TRAIN_LOADER_LINE = "train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, \"train\")"
AUDIO_MIX_LINE = "AUDIO_MIX_RATIO = 0.2"

# ============================================================
# Alternating pure batch loader (reused from sweep37)
# ============================================================
def make_alternating_loader(audio_every_n):
    return f'''
_text_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", audio_ratio=0.0)
_audio_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", audio_ratio=1.0)
_text_iter = iter(_text_loader)
_audio_iter = iter(_audio_loader)
_AUDIO_EVERY_N = {audio_every_n}
_is_audio_step = False
def _alternating_loader():
    global _is_audio_step
    _step = 0
    while True:
        _is_audio_step = (_step % _AUDIO_EVERY_N == (_AUDIO_EVERY_N - 1))
        if _is_audio_step:
            try:
                x, y, epoch = next(_audio_iter)
            except StopIteration:
                _audio_iter_new = iter(_audio_loader)
                x, y, epoch = next(_audio_iter_new)
        else:
            try:
                x, y, epoch = next(_text_iter)
            except StopIteration:
                _text_iter_new = iter(_text_loader)
                x, y, epoch = next(_text_iter_new)
        yield x, y, epoch
        _step += 1
train_loader = _alternating_loader()'''


# ============================================================
# B1: Alt batches + audio LR sweep (finer grain than sweep37)
# ============================================================
def make_audio_lr_mod(audio_lr_scale):
    """Scale Muon LR for audio steps. LR was already set by schedule above."""
    return f'''
    # B1: Audio LR scaling — multiply current LR by scale factor on audio steps
    if _is_audio_step:
        for _grp in optimizer.param_groups:
            _grp["lr"] = _grp["lr"] * {audio_lr_scale}
    '''


# ============================================================
# B4: Modality-aware WD — lower WD for audio steps
# ============================================================
def make_audio_wd_mod(audio_wd_scale):
    """Scale weight decay for audio steps. WD already set by schedule above."""
    return f'''
    # B4: Audio weight decay scaling
    if _is_audio_step:
        for _grp in optimizer.param_groups:
            if _grp["kind"] == "muon":
                _grp["weight_decay"] = _grp["weight_decay"] * {audio_wd_scale}
    '''


# ============================================================
# B5: Per-modality momentum in Muon optimizer (modify _step_muon)
# ============================================================
OLD_MUON_MOMENTUM_SET = """        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])"""

def make_muon_modal_mod(audio_mom, audio_lr_scale=1.0):
    """Per-modality momentum and LR inside Muon optimizer."""
    return f"""        # === Per-modality momentum/LR inside Muon ===
        if _is_audio_step:
            self._muon_momentum_t.fill_({audio_mom})
            self._muon_lr_t.fill_(group["lr"] * {audio_lr_scale} * max(1.0, shape[-2] / shape[-1])**0.5)
        else:
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_wd_t.fill_(group["weight_decay"])"""


# ============================================================
# B2: Cosine-scheduled audio ratio — start text-heavy, ramp audio
# ============================================================
COSINE_AUDIO_LOADER = '''
import math as _math
_text_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", audio_ratio=0.0)
_audio_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", audio_ratio=1.0)
_text_iter = iter(_text_loader)
_audio_iter = iter(_audio_loader)
_is_audio_step = False
_total_steps_est = 4300  # approximate total steps
def _cosine_audio_loader():
    global _is_audio_step
    _step = 0
    while True:
        # Audio probability: cosine schedule from 0.1 to 0.5
        _progress = min(1.0, _step / _total_steps_est)
        _audio_prob = 0.1 + 0.4 * (1 - _math.cos(_math.pi * _progress)) / 2
        import random
        _is_audio_step = random.random() < _audio_prob
        if _is_audio_step:
            try:
                x, y, epoch = next(_audio_iter)
            except StopIteration:
                _audio_iter_new = iter(_audio_loader)
                x, y, epoch = next(_audio_iter_new)
        else:
            try:
                x, y, epoch = next(_text_iter)
            except StopIteration:
                _text_iter_new = iter(_text_loader)
                x, y, epoch = next(_text_iter_new)
        yield x, y, epoch
        _step += 1
train_loader = _cosine_audio_loader()'''


# ============================================================
# B6: Post-step audio param boost — after optimizer.step(), apply extra
# update for audio-responsive parameters
# ============================================================
B6_STATE = '''
# === B6 state ===
_b6_prev_params = {}
_b6_audio_dir = {}
'''

B6_CODE = '''
    # === B6: Post-step direction tracking ===
    # After optimizer step, track the update direction.
    # On audio steps, apply a boosted update.
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.dim() == 2:
                    _k = f"{_bi}_{_pn}"
                    if _is_audio_step and _k in _b6_prev_params:
                        _delta = _p.data - _b6_prev_params[_k]
                        # Boost the audio update direction
                        _p.data.add_(_delta, alpha={boost})
                    _b6_prev_params[_k] = _p.data.clone()
'''


# ============================================================
# Build experiments
# ============================================================

def make_mods_simple(before_opt_code=""):
    """Only inject before optimizer.step()"""
    mods = []
    if before_opt_code:
        mods.append((BEFORE_OPT_STEP, before_opt_code + "\n" + BEFORE_OPT_STEP))
    return mods


IS_AUDIO_GLOBAL = """DEPTH = 8               # number of transformer layers
_is_audio_step = False  # set by training loop each step"""

experiments = [
    # Reference
    ("post39: ref", [], None),

    # --- Group 1: Audio LR sweep with alt5 (extending sweep37 finding) ---
    # sweep37 found alt5+lr1.5 was best (audio=5.442), try higher LR
    ("post39: alt5+lr2.0", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.95, 2.0)),
    ], None),

    ("post39: alt5+lr2.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.95, 2.5)),
    ], None),

    ("post39: alt5+lr3.0", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.95, 3.0)),
    ], None),

    # --- Group 2: Lower momentum for audio steps ---
    # Hypothesis: lower momentum uses more of current (pure audio) gradient,
    # less of historical (mixed) buffer. Sweep37 showed HIGH momentum hurts.
    ("post39: alt5+mom0.8+lr1.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.80, 1.5)),
    ], None),

    ("post39: alt5+mom0.85+lr1.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.85, 1.5)),
    ], None),

    ("post39: alt5+mom0.9+lr2.0", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.90, 2.0)),
    ], None),

    # --- Group 3: Audio WD reduction ---
    # Lower WD on audio steps preserves learned audio features
    ("post39: alt5+lr1.5+wd0.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
    ] + make_mods_simple(make_audio_lr_mod(1.5) + make_audio_wd_mod(0.5)), None),

    ("post39: alt5+lr2.0+wd0.0", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
    ] + make_mods_simple(make_audio_lr_mod(2.0) + make_audio_wd_mod(0.0)), None),

    # --- Group 4: More audio ratio with LR boost ---
    ("post39: alt3+lr1.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(3)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.95, 1.5)),
    ], None),

    ("post39: alt3+lr2.0", [
        (TRAIN_LOADER_LINE, make_alternating_loader(3)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.95, 2.0)),
    ], None),

    # --- Group 5: Best combo from all sweeps ---
    ("post39: alt5+mom0.85+lr2.0+wd0.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL),
        (OLD_MUON_MOMENTUM_SET, make_muon_modal_mod(0.85, 2.0)),
    ] + make_mods_simple(make_audio_wd_mod(0.5)), None),

    # --- Group 6: Cosine-scheduled audio ratio ---
    ("post39: cosine_audio", [
        (TRAIN_LOADER_LINE, COSINE_AUDIO_LOADER),
    ], None),
]

print(f"Running {len(experiments)} Post-NS Omni-Optimizer experiments...")
results = []
for i, (desc, train_mods, prepare_mods) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {desc}")
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=600)
    results.append((desc, r))

print("\n" + "="*70)
print("SWEEP 39: POST-NS OMNI-OPTIMIZER RESULTS")
print("="*70)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

print("\n--- Audio Loss Ranking ---")
sorted_audio = sorted(results, key=lambda x: x[1]['audio_loss'] if x[1] else 99.0)
for desc, r in sorted_audio:
    if r:
        print(f"  audio={r['audio_loss']:.4f} | val={r['val_loss']:.4f} | text={r['text_bpb']:.4f} | {desc}")

ref = next((r for d, r in results if d == "post39: ref" and r), None)
if ref:
    print("\n--- Pareto Analysis (audio < ref AND text within +0.005) ---")
    for desc, r in sorted_results:
        if r and "ref" not in desc:
            d_text = r['text_bpb'] - ref['text_bpb']
            d_audio = r['audio_loss'] - ref['audio_loss']
            pareto = "PARETO!" if d_audio < -0.01 and d_text < 0.005 else ""
            print(f"  Δtext={d_text:+.4f} Δaudio={d_audio:+.4f} | {pareto:>8} | {desc}")

with open("post_ns_results.json", "w") as f:
    json.dump([(d, r) for d, r in results], f, indent=2)
print("\nSaved post_ns_results.json")
