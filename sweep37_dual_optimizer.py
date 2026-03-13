"""Sweep 37: Per-Modality Optimizer — Separate optimization for text and audio.

Core hypothesis: The shared Muon momentum buffer is dominated by text gradients
(80% of batch, higher info density). Audio's gradient signal gets "drowned out"
in the shared momentum. By maintaining separate momentum buffers and applying
NS independently per modality, each modality gets its own optimization trajectory.

Three approaches:
A. Alternating pure batches with separate Muon momentum buffers
B. Split momentum in Muon: track text-step and audio-step contributions separately
C. Dual NS: apply Newton-Schulz independently to text and audio gradient components

Implementation: Modify the training loop to alternate between pure text and pure
audio batches, using audio_ratio to control the proportion. Each batch type uses
its own momentum accumulation path.
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
            count = modified_train.count(old)
            if count == 0:
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
# Approach A: Alternating pure batches
# ============================================================
# Instead of 80% text + 20% audio per batch, alternate:
#   4 pure text steps → 1 pure audio step (same 80/20 ratio)
# This gives clean gradient decomposition per modality.
# We use audio_ratio=0.0 for text steps and audio_ratio=1.0 for audio steps.

# The key change: import make_dataloader twice with different ratios,
# alternate between them in the training loop.

# Injection: replace the single dataloader with two alternating ones
OLD_DATALOADER = "train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, \"train\")"

NEW_ALTERNATING_LOADER = """# === DUAL OPTIMIZER: Alternating pure text/audio batches ===
_text_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", audio_ratio=0.0)
_audio_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train", audio_ratio=1.0)
_text_iter = iter(_text_loader)
_audio_iter = iter(_audio_loader)
_AUDIO_EVERY_N = {audio_every_n}  # 1 audio step every N steps
_is_audio_step = False

def _alternating_loader():
    _step = 0
    while True:
        if _step % _AUDIO_EVERY_N == (_AUDIO_EVERY_N - 1):
            x, y, epoch = next(_audio_iter)
            yield x, y, epoch
        else:
            x, y, epoch = next(_text_iter)
            yield x, y, epoch
        _step += 1

train_loader = _alternating_loader()"""

# Track which modality the current step is for
OLD_TRAINING_STEP = "    for micro_step in range(grad_accum_steps):"
NEW_TRAINING_STEP = """    _is_audio_step = (step % _AUDIO_EVERY_N == (_AUDIO_EVERY_N - 1))
    for micro_step in range(grad_accum_steps):"""

# ============================================================
# Approach B: Per-modality momentum in Muon
# ============================================================
# Maintain two momentum buffers. On text steps, only update text momentum.
# On audio steps, only update audio momentum. Final gradient = blend.
# This requires modifying the Muon step function.

# Replace the Muon step to use dual momentum
OLD_MUON_STEP_CALL = """        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)"""

# New Muon step with dual momentum
NEW_DUAL_MUON_STEP = """        # === DUAL OPTIMIZER: Per-modality momentum ===
        if "momentum_buffer_audio" not in state:
            state["momentum_buffer_audio"] = torch.zeros_like(state["momentum_buffer"])
            state["second_momentum_buffer_audio"] = torch.zeros_like(state["second_momentum_buffer"])
        if _is_audio_step:
            # Audio step: update audio momentum, apply NS to audio gradient
            muon_step_fused(stacked_grads, stacked_params,
                            state["momentum_buffer_audio"], state["second_momentum_buffer_audio"],
                            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                            self._muon_beta2_t, group["ns_steps"], red_dim)
        else:
            # Text step: update text momentum, apply NS to text gradient
            muon_step_fused(stacked_grads, stacked_params,
                            state["momentum_buffer"], state["second_momentum_buffer"],
                            self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                            self._muon_beta2_t, group["ns_steps"], red_dim)"""

# ============================================================
# Approach C: Different LR/momentum per modality step
# ============================================================
# On audio steps, use higher momentum (accumulate sparse signal)
# and potentially different LR.

OLD_MOMENTUM_SET = """        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])"""

NEW_MODAL_MOMENTUM = """        # === DUAL OPTIMIZER: Per-modality momentum/LR ===
        if _is_audio_step:
            self._muon_momentum_t.fill_({audio_mom})
            self._muon_lr_t.fill_(group["lr"] * {audio_lr_scale} * max(1.0, shape[-2] / shape[-1])**0.5)
        else:
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_wd_t.fill_(group["weight_decay"])"""

# Need to make _is_audio_step accessible inside optimizer
OLD_IS_AUDIO_GLOBAL = "DEPTH = 8               # number of transformer layers"
NEW_IS_AUDIO_GLOBAL = """DEPTH = 8               # number of transformer layers
_is_audio_step = False  # set by training loop each step"""


# ============================================================
# Build experiments
# ============================================================

experiments = [
    # Reference: standard mixed batches (no changes)
    ("dual37: ref", [], None),

    # --- Approach A: Alternating batches only (no momentum changes) ---
    # Same 80/20 ratio but pure batches instead of mixed
    ("dual37: alt_5", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=5)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
    ], None),

    ("dual37: alt_3", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=3)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
    ], None),

    # --- Approach B: Alternating + dual momentum buffers ---
    # Each modality has its own NS orthogonalization path
    ("dual37: alt5+dualmom", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=5)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MUON_STEP_CALL, NEW_DUAL_MUON_STEP),
    ], None),

    ("dual37: alt3+dualmom", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=3)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MUON_STEP_CALL, NEW_DUAL_MUON_STEP),
    ], None),

    # --- Approach C: Different momentum/LR per modality ---
    # Audio: higher momentum (0.98) to accumulate sparse signal
    ("dual37: alt5+audiomom0.98", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=5)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MOMENTUM_SET, NEW_MODAL_MOMENTUM.format(audio_mom=0.98, audio_lr_scale=1.0)),
    ], None),

    # Audio: higher LR (1.5x) to compensate for fewer steps
    ("dual37: alt5+audiolr1.5", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=5)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MOMENTUM_SET, NEW_MODAL_MOMENTUM.format(audio_mom=0.95, audio_lr_scale=1.5)),
    ], None),

    # Audio: both higher momentum AND higher LR
    ("dual37: alt5+mom98+lr1.5", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=5)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MOMENTUM_SET, NEW_MODAL_MOMENTUM.format(audio_mom=0.98, audio_lr_scale=1.5)),
    ], None),

    # --- Approach B+C: Dual momentum + per-modality settings ---
    ("dual37: dualmom+mom98+lr1.5", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=5)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MOMENTUM_SET, NEW_MODAL_MOMENTUM.format(audio_mom=0.98, audio_lr_scale=1.5)),
        (OLD_MUON_STEP_CALL, NEW_DUAL_MUON_STEP),
    ], None),

    # --- More audio steps (50/50 instead of 80/20) ---
    ("dual37: alt2+dualmom", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=2)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MUON_STEP_CALL, NEW_DUAL_MUON_STEP),
    ], None),

    # Audio: higher momentum 0.99 (very aggressive accumulation)
    ("dual37: alt5+audiomom0.99", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=5)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MOMENTUM_SET, NEW_MODAL_MOMENTUM.format(audio_mom=0.99, audio_lr_scale=1.0)),
    ], None),

    # --- Higher audio ratio baselines (mixed batch, standard optimizer) ---
    # Compare: does higher ratio help audio when using standard shared optimizer?
    ("dual37: mixed_ratio0.3", [],
        [("AUDIO_MIX_RATIO = 0.2", "AUDIO_MIX_RATIO = 0.3")]),

    ("dual37: mixed_ratio0.5", [],
        [("AUDIO_MIX_RATIO = 0.2", "AUDIO_MIX_RATIO = 0.5")]),

    # --- Dual optimizer + high audio ratio ---
    # KEY TEST: with per-modality momentum, can we increase audio share
    # WITHOUT hurting text? If yes, this proves the representation
    # competition hypothesis.
    ("dual37: alt2+dualmom+mom98+lr1.5", [
        (OLD_IS_AUDIO_GLOBAL, NEW_IS_AUDIO_GLOBAL),
        (OLD_DATALOADER, NEW_ALTERNATING_LOADER.format(audio_every_n=2)),
        (OLD_TRAINING_STEP, NEW_TRAINING_STEP),
        (OLD_MOMENTUM_SET, NEW_MODAL_MOMENTUM.format(audio_mom=0.98, audio_lr_scale=1.5)),
        (OLD_MUON_STEP_CALL, NEW_DUAL_MUON_STEP),
    ], None),
]

print(f"Running {len(experiments)} Dual-Optimizer experiments...")
results = []
for i, (desc, train_mods, prepare_mods) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {desc}")
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=600)
    results.append((desc, r))

print("\n" + "="*70)
print("SWEEP 37: DUAL OPTIMIZER RESULTS")
print("="*70)
print(f"  {'val_loss':>8} | {'text_bpb':>8} | {'audio':>8} | description")
print(f"  {'--------':>8} | {'--------':>8} | {'--------':>8} | -----------")
sorted_results = sorted(results, key=lambda x: x[1]['val_loss'] if x[1] else 99.0)
for desc, r in sorted_results:
    if r:
        print(f"  {r['val_loss']:.4f} | {r['text_bpb']:.4f} | {r['audio_loss']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

# Audio-focused ranking
print("\n--- Audio Loss Ranking ---")
sorted_audio = sorted(results, key=lambda x: x[1]['audio_loss'] if x[1] else 99.0)
for desc, r in sorted_audio:
    if r:
        print(f"  audio={r['audio_loss']:.4f} | val={r['val_loss']:.4f} | text={r['text_bpb']:.4f} | {desc}")

# Pareto analysis
ref = next((r for d, r in results if d == "dual37: ref" and r), None)
if ref:
    print("\n--- Pareto Analysis (vs reference) ---")
    for desc, r in sorted_results:
        if r and desc != "dual37: ref":
            d_text = r['text_bpb'] - ref['text_bpb']
            d_audio = r['audio_loss'] - ref['audio_loss']
            pareto = "PARETO!" if d_audio < -0.01 and d_text < 0.005 else ""
            print(f"  Δtext={d_text:+.4f} Δaudio={d_audio:+.4f} | {pareto:>8} | {desc}")

with open("dual_optimizer_results.json", "w") as f:
    json.dump([(d, r) for d, r in results], f, indent=2)
print("\nSaved dual_optimizer_results.json")
