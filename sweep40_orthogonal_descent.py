"""Sweep 40: Orthogonal Descent — Project audio gradients to be orthogonal to text.

Key insight from sweeps 37-39:
- ALL approaches show strict audio-text tradeoff
- Pre-NS gradient manipulation is washed out
- The problem is CAPACITY COMPETITION, not signal quality

New principle: Instead of boosting audio or dampening text,
PROJECT audio gradients to be ORTHOGONAL to the text gradient direction.
This lets audio make progress WITHOUT interfering with text.

Math: If text direction is t̂ (from text momentum/gradient history):
  g_audio_parallel = (g_audio · t̂) t̂      # fights text
  g_audio_perp = g_audio - g_audio_parallel  # orthogonal to text

Use only g_audio_perp for audio steps. Audio makes progress in
directions that don't interfere with text.

Also try:
C1: Orthogonal projection of audio gradients
C2: Layer-wise freezing (bottom for text, top for audio)
C3: Gradient projection INSIDE Muon (modify stacked_grads before NS)
C4: Separate NS with projection — run NS separately per modality
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
TRAIN_LOADER_LINE = 'train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")'

# ============================================================
# Alternating pure batch loader
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
# C1: Orthogonal Projection — project audio gradients to be orthogonal
# to text gradient direction. Operates on transformer matrices.
# ============================================================
C1_STATE = '''
# === C1: Orthogonal projection state ===
_c1_text_dir = {}  # EMA of text gradient direction per param
_c1_beta = 0.9
'''

C1_CODE = '''
    # === C1: Orthogonal Projection ===
    # Track text gradient direction. On audio steps, project audio gradient
    # to be orthogonal to text direction. This prevents competition.
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.grad is not None and _p.dim() == 2:
                    _k = f"{{_bi}}_{{_pn}}"
                    _g = _p.grad.float()
                    _g_flat = _g.reshape(-1)
                    if not _is_audio_step:
                        # Text step: update text direction EMA
                        if _k not in _c1_text_dir:
                            _c1_text_dir[_k] = _g_flat.clone()
                        else:
                            _c1_text_dir[_k].lerp_(_g_flat, 1 - _c1_beta)
                    else:
                        # Audio step: project gradient to be orthogonal to text direction
                        if _k in _c1_text_dir and step > 20:
                            _t = _c1_text_dir[_k]
                            _t_norm = _t / (_t.norm() + 1e-8)
                            # Remove component parallel to text direction
                            _parallel = (_g_flat * _t_norm).sum() * _t_norm
                            _g_perp = _g_flat - {remove_frac} * _parallel
                            _p.grad.copy_(_g_perp.reshape(_p.grad.shape).to(_p.grad.dtype))
'''


# ============================================================
# C2: Layer-wise freezing — freeze bottom layers for audio, top for text
# This creates implicit specialization in the shared backbone.
# ============================================================
C2_CODE_BOTTOM = '''
    # === C2: Layer-wise freezing (freeze bottom for audio) ===
    # Audio: freeze layers 0-{freeze_n}, update layers {freeze_n}+1 to end
    # Text: update all layers normally
    with torch.no_grad():
        if _is_audio_step:
            for _bi, _block in enumerate(model._orig_mod.transformer.h):
                if _bi < {freeze_n}:
                    for _p in _block.parameters():
                        if _p.grad is not None:
                            _p.grad.zero_()
'''

C2_CODE_TOP = '''
    # === C2: Layer-wise freezing (freeze top for audio) ===
    # Audio: freeze layers {freeze_start}+, update layers 0 to {freeze_start}
    # Text: update all layers normally
    with torch.no_grad():
        if _is_audio_step:
            for _bi, _block in enumerate(model._orig_mod.transformer.h):
                if _bi >= {freeze_start}:
                    for _p in _block.parameters():
                        if _p.grad is not None:
                            _p.grad.zero_()
'''

# ============================================================
# C3: Gradient scaling based on per-layer modality sensitivity
# Track how much each layer's gradient changes between text/audio steps.
# Scale layers that are more "audio-sensitive" with higher LR on audio steps.
# ============================================================
C3_STATE = '''
# === C3: Per-layer sensitivity state ===
_c3_text_grad_norm = {}
_c3_audio_grad_norm = {}
_c3_sensitivity = {}
_c3_beta = 0.95
'''

C3_CODE = '''
    # === C3: Per-layer modality sensitivity scaling ===
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.grad is not None and _p.dim() == 2:
                    _k = f"{{_bi}}_{{_pn}}"
                    _gnorm = _p.grad.float().norm().item()
                    if not _is_audio_step:
                        if _k not in _c3_text_grad_norm:
                            _c3_text_grad_norm[_k] = _gnorm
                        else:
                            _c3_text_grad_norm[_k] = _c3_beta * _c3_text_grad_norm[_k] + (1 - _c3_beta) * _gnorm
                    else:
                        if _k not in _c3_audio_grad_norm:
                            _c3_audio_grad_norm[_k] = _gnorm
                        else:
                            _c3_audio_grad_norm[_k] = _c3_beta * _c3_audio_grad_norm[_k] + (1 - _c3_beta) * _gnorm
                    # On audio steps, boost layers where audio gradient is relatively stronger
                    if _is_audio_step and _k in _c3_text_grad_norm and _k in _c3_audio_grad_norm and step > 50:
                        _ratio = _c3_audio_grad_norm[_k] / (_c3_text_grad_norm[_k] + 1e-8)
                        # Layers with high audio/text ratio are "audio-sensitive" — boost them
                        _scale = min({max_boost}, max(0.5, _ratio * {boost_factor}))
                        _p.grad.mul_(_scale)
'''


# ============================================================
# Build experiments
# ============================================================

IS_AUDIO_GLOBAL = """DEPTH = 8               # number of transformer layers
_is_audio_step = False  # set by training loop each step"""

OLD_MUON_MOMENTUM_SET = """        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])"""

def make_muon_audio_lr(audio_lr_scale):
    return f"""        # Audio LR inside Muon
        if _is_audio_step:
            self._muon_lr_t.fill_(group["lr"] * {audio_lr_scale} * max(1.0, shape[-2] / shape[-1])**0.5)
        else:
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_wd_t.fill_(group["weight_decay"])"""


def make_mods(state_code="", before_opt_code=""):
    mods = []
    if state_code:
        mods.append((AFTER_DEPTH, AFTER_DEPTH + state_code))
    if before_opt_code:
        mods.append((BEFORE_OPT_STEP, before_opt_code + "\n" + BEFORE_OPT_STEP))
    return mods


experiments = [
    # Reference
    ("orth40: ref", [], None),

    # --- C1: Orthogonal Projection (core idea) ---
    # Remove 50% of the text-parallel component from audio gradients
    ("orth40: proj0.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, AFTER_DEPTH + C1_STATE),
    ] + [(BEFORE_OPT_STEP, C1_CODE.format(remove_frac=0.5) + "\n" + BEFORE_OPT_STEP)], None),

    # Remove 100% — full orthogonal projection
    ("orth40: proj1.0", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, AFTER_DEPTH + C1_STATE),
    ] + [(BEFORE_OPT_STEP, C1_CODE.format(remove_frac=1.0) + "\n" + BEFORE_OPT_STEP)], None),

    # Remove 75%
    ("orth40: proj0.75", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, AFTER_DEPTH + C1_STATE),
    ] + [(BEFORE_OPT_STEP, C1_CODE.format(remove_frac=0.75) + "\n" + BEFORE_OPT_STEP)], None),

    # C1 + audio LR boost (combine orthogonal projection with best sweep37 trick)
    ("orth40: proj0.5+lr1.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL + C1_STATE),
        (OLD_MUON_MOMENTUM_SET, make_muon_audio_lr(1.5)),
    ] + [(BEFORE_OPT_STEP, C1_CODE.format(remove_frac=0.5) + "\n" + BEFORE_OPT_STEP)], None),

    ("orth40: proj1.0+lr1.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL + C1_STATE),
        (OLD_MUON_MOMENTUM_SET, make_muon_audio_lr(1.5)),
    ] + [(BEFORE_OPT_STEP, C1_CODE.format(remove_frac=1.0) + "\n" + BEFORE_OPT_STEP)], None),

    # --- C2: Layer freezing ---
    # Freeze bottom 2 layers on audio steps (let text own bottom layers)
    ("orth40: freeze_bot2", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
    ] + [(BEFORE_OPT_STEP, C2_CODE_BOTTOM.format(freeze_n=2) + "\n" + BEFORE_OPT_STEP)], None),

    # Freeze bottom 4 layers on audio steps
    ("orth40: freeze_bot4", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
    ] + [(BEFORE_OPT_STEP, C2_CODE_BOTTOM.format(freeze_n=4) + "\n" + BEFORE_OPT_STEP)], None),

    # Freeze top 2 layers on audio steps (let text own top layers)
    ("orth40: freeze_top2", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
    ] + [(BEFORE_OPT_STEP, C2_CODE_TOP.format(freeze_start=6) + "\n" + BEFORE_OPT_STEP)], None),

    # --- C3: Sensitivity-aware scaling ---
    ("orth40: sens1.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, AFTER_DEPTH + C3_STATE),
    ] + [(BEFORE_OPT_STEP, C3_CODE.format(max_boost=2.0, boost_factor=1.5) + "\n" + BEFORE_OPT_STEP)], None),

    # --- Best combinations ---
    # Orthogonal projection + layer freeze + LR boost
    ("orth40: proj0.5+freeze_bot2+lr1.5", [
        (TRAIN_LOADER_LINE, make_alternating_loader(5)),
        (AFTER_DEPTH, IS_AUDIO_GLOBAL + C1_STATE),
        (OLD_MUON_MOMENTUM_SET, make_muon_audio_lr(1.5)),
    ] + [(BEFORE_OPT_STEP, C1_CODE.format(remove_frac=0.5) + C2_CODE_BOTTOM.format(freeze_n=2) + "\n" + BEFORE_OPT_STEP)], None),
]

print(f"Running {len(experiments)} Orthogonal Descent experiments...")
results = []
for i, (desc, train_mods, prepare_mods) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {desc}")
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=600)
    results.append((desc, r))

print("\n" + "="*70)
print("SWEEP 40: ORTHOGONAL DESCENT RESULTS")
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

ref = next((r for d, r in results if d == "orth40: ref" and r), None)
if ref:
    print("\n--- Pareto Analysis (audio < ref AND text < ref + 0.005) ---")
    for desc, r in sorted_results:
        if r and "ref" not in desc:
            d_text = r['text_bpb'] - ref['text_bpb']
            d_audio = r['audio_loss'] - ref['audio_loss']
            pareto = "PARETO!" if d_audio < -0.01 and d_text < 0.005 else ""
            print(f"  Δtext={d_text:+.4f} Δaudio={d_audio:+.4f} | {pareto:>8} | {desc}")

with open("orthogonal_descent_results.json", "w") as f:
    json.dump([(d, r) for d, r in results], f, indent=2)
print("\nSaved orthogonal_descent_results.json")
