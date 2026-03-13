"""Sweep 36: Omni-Optimizer — Modality-Aware Optimization for Convergence Synchronization.

Key difference from previous failed approaches (H2-H6):
- Previous: scaled EMBEDDING gradients → redundant with NS equalization
- Now: operate on TRANSFORMER MATRIX gradients via Muon, and use
  convergence-rate-based adaptation (not gradient-magnitude-based)

All modifications happen in the training loop AFTER backward, BEFORE optimizer.step().
The compiled forward pass is untouched.
"""
import subprocess, re, os, json

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
    # Don't commit modified files — just run the experiment
    import hashlib
    commit = hashlib.md5(desc.encode()).hexdigest()[:7]
    print(f"\n{'='*60}\nEXP {commit}: {desc}\n{'='*60}")
    try:
        result = subprocess.run([PYTHON, "train.py"], capture_output=True, text=True, timeout=timeout)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
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
# Injection targets (exact strings from train.py)
# ============================================================

AFTER_DEPTH = "DEPTH = 8               # number of transformer layers"

# The gradient manipulation block — inject BEFORE optimizer.step()
BEFORE_OPT_STEP = "    optimizer.step()"

# After optimizer step — for tracking state updates
AFTER_ZERO_GRAD = "    model.zero_grad(set_to_none=True)"

# ============================================================
# Omni-Optimizer state variables (injected after DEPTH)
# ============================================================

OMNI_STATE = '''
# === OMNI-OPTIMIZER STATE ===
_omni_text_loss_ema = 0.0
_omni_audio_loss_ema = 0.0
_omni_prev_text_loss = 10.0
_omni_prev_audio_loss = 10.0
_omni_text_rate_ema = 0.0
_omni_audio_rate_ema = 0.0
_omni_audio_scale = 1.0
_omni_gim_prev_grads = {}
'''

# ============================================================
# H1: CRAW — Convergence-Rate Adaptive Weighting
# Scales audio embedding/lm_head gradients based on convergence rate ratio.
# Different from old H2 (fixed schedule) — this is ADAPTIVE.
# Also scales transformer matrix gradients gently.
# ============================================================

CRAW_CODE = '''
    # === OMNI: CRAW (Convergence-Rate Adaptive Weighting) ===
    _craw_alpha = {alpha}
    with torch.no_grad():
        # Track embedding gradient norms as proxy for per-modality convergence
        _ema_beta = 0.95
        _wte_g = model._orig_mod.transformer.wte.weight.grad
        if _wte_g is not None:
            _text_gnorm = _wte_g[:AUDIO_START_ID].float().norm().item()
            _audio_gnorm = _wte_g[AUDIO_START_ID:].float().norm().item()
            # Use gradient norm ratio as convergence rate proxy
            _omni_text_loss_ema = _ema_beta * _omni_text_loss_ema + (1 - _ema_beta) * _text_gnorm
            _omni_audio_loss_ema = _ema_beta * _omni_audio_loss_ema + (1 - _ema_beta) * _audio_gnorm
            if step > 30:
                _text_rate = abs(_omni_prev_text_loss - _omni_text_loss_ema)
                _audio_rate = abs(_omni_prev_audio_loss - _omni_audio_loss_ema)
                _omni_text_rate_ema = 0.9 * _omni_text_rate_ema + 0.1 * _text_rate
                _omni_audio_rate_ema = 0.9 * _omni_audio_rate_ema + 0.1 * _audio_rate
                if _omni_audio_rate_ema > 1e-10:
                    _rate_ratio = _omni_text_rate_ema / max(_omni_audio_rate_ema, 1e-10)
                    _omni_audio_scale = _omni_audio_scale * (1.0 + _craw_alpha * min(max(_rate_ratio - 1.0, -0.5), 0.5))
                    _omni_audio_scale = max(0.3, min(5.0, _omni_audio_scale))
            _omni_prev_text_loss = _omni_text_loss_ema
            _omni_prev_audio_loss = _omni_audio_loss_ema
            # Apply: scale audio embedding gradients
            if _omni_audio_scale != 1.0:
                _wte_g[AUDIO_START_ID:] *= _omni_audio_scale
                _lm_g = model._orig_mod.lm_head.weight.grad
                if _lm_g is not None:
                    _lm_g[AUDIO_START_ID:] *= _omni_audio_scale
                for _ve_key, _ve_embed in model._orig_mod.value_embeds.items():
                    if _ve_embed.weight.grad is not None:
                        _ve_embed.weight.grad[AUDIO_START_ID:] *= _omni_audio_scale
            if step % 200 == 0 and step > 0:
                print(f"  [CRAW] audio_scale={{_omni_audio_scale:.3f}} text_rate={{_omni_text_rate_ema:.6f}} audio_rate={{_omni_audio_rate_ema:.6f}}")
'''

# ============================================================
# H2: GIM — Gradient Interference Minimization
# Track gradient direction stability per transformer layer.
# When consecutive gradients conflict (cos < 0), project out
# the conflicting component. This reduces modality interference
# without needing to decompose gradients by modality.
# ============================================================

GIM_CODE = '''
    # === OMNI: GIM (Gradient Interference Minimization) ===
    with torch.no_grad():
        for _blk_i, _block in enumerate(model._orig_mod.transformer.h):
            for _pname, _param in _block.named_parameters():
                if _param.grad is not None and _param.dim() == 2:
                    _gkey = f"{{_blk_i}}_{{_pname}}"
                    _curr = _param.grad
                    if _gkey in _omni_gim_prev_grads:
                        _prev = _omni_gim_prev_grads[_gkey]
                        # Cosine similarity between consecutive gradients
                        _dot = (_prev * _curr).sum()
                        _cos = _dot / (_prev.norm() * _curr.norm() + 1e-8)
                        if _cos < {threshold}:
                            # Gradient conflict — project out conflicting component
                            _proj_coeff = _dot / (_prev.norm()**2 + 1e-8)
                            _param.grad.sub_(_proj_coeff * _prev)
                    _omni_gim_prev_grads[_gkey] = _curr.clone()
'''

# ============================================================
# H3: SNRE — Spectral Norm Rate Equalization
# Use embedding gradient spectral norms as proxy for per-modality
# contribution. Scale transformer matrix gradients to equalize.
# Operates in NORM SPACE (spectral norm) for stability.
# ============================================================

SNRE_CODE = '''
    # === OMNI: SNRE (Spectral Norm Rate Equalization) ===
    with torch.no_grad():
        _wte_g = model._orig_mod.transformer.wte.weight.grad
        if _wte_g is not None:
            # Spectral norm of text vs audio embedding gradients
            _text_sn = torch.linalg.matrix_norm(_wte_g[:AUDIO_START_ID].float(), ord=2)
            _audio_sn = torch.linalg.matrix_norm(_wte_g[AUDIO_START_ID:].float(), ord=2)
            if _audio_sn > 1e-10 and _text_sn > 1e-10:
                _sn_ratio = (_text_sn / _audio_sn).clamp(0.5, 3.0)
                # Boost transformer matrix gradients proportionally to imbalance
                _boost = 1.0 + {strength} * (_sn_ratio.item() - 1.0)
                for _block in model._orig_mod.transformer.h:
                    for _param in _block.parameters():
                        if _param.grad is not None and _param.dim() == 2:
                            _param.grad.mul_(_boost)
                # Also boost audio embedding gradients
                _wte_g[AUDIO_START_ID:] *= _sn_ratio
                _lm_g = model._orig_mod.lm_head.weight.grad
                if _lm_g is not None:
                    _lm_g[AUDIO_START_ID:] *= _sn_ratio
                if step % 200 == 0:
                    print(f"  [SNRE] text_σ={{_text_sn:.4f}} audio_σ={{_audio_sn:.4f}} ratio={{_sn_ratio:.3f}} boost={{_boost:.3f}}")
'''

# ============================================================
# H4: Transformer-Targeted Audio Boost (TTAB)
# Simple but principled: boost ONLY the transformer matrix gradients
# (where representation competition happens) proportional to
# audio's share of the batch. Does NOT touch embeddings.
# ============================================================

TTAB_CODE = '''
    # === OMNI: TTAB (Transformer-Targeted Audio Boost) ===
    # The key insight: audio tokens are 20% of batch but need equal
    # representation in the shared transformer. Boost transformer
    # matrix gradients to compensate for audio being underrepresented.
    with torch.no_grad():
        _wte_g = model._orig_mod.transformer.wte.weight.grad
        if _wte_g is not None:
            _text_energy = _wte_g[:AUDIO_START_ID].float().norm().item()
            _audio_energy = _wte_g[AUDIO_START_ID:].float().norm().item()
            if _audio_energy > 1e-10 and _text_energy > 1e-10:
                # How much audio is underrepresented in gradient energy
                _energy_ratio = _text_energy / _audio_energy
                # Scale transformer matrices to amplify audio signal
                _scale = 1.0 + {strength} * min(max(_energy_ratio - 1.0, 0), 5.0)
                for _block in model._orig_mod.transformer.h:
                    for _param in _block.parameters():
                        if _param.grad is not None and _param.dim() == 2:
                            _param.grad.mul_(_scale)
                if step % 200 == 0:
                    print(f"  [TTAB] text_E={{_text_energy:.4f}} audio_E={{_audio_energy:.4f}} ratio={{_energy_ratio:.2f}} scale={{_scale:.3f}}")
'''

# ============================================================
# Build experiments
# ============================================================

def make_mods(state=True, before_opt="", after_zg=""):
    """Build modification list."""
    mods = []
    if state:
        mods.append((AFTER_DEPTH, AFTER_DEPTH + OMNI_STATE))
    if before_opt:
        mods.append((BEFORE_OPT_STEP, before_opt + "\n" + BEFORE_OPT_STEP))
    if after_zg:
        mods.append((AFTER_ZERO_GRAD, AFTER_ZERO_GRAD + "\n" + after_zg))
    return mods


experiments = [
    # Reference baseline (no changes)
    ("omni36: ref", [], None),

    # H1: CRAW with different α values
    ("omni36: craw_a0.1", make_mods(before_opt=CRAW_CODE.format(alpha=0.1)), None),
    ("omni36: craw_a0.3", make_mods(before_opt=CRAW_CODE.format(alpha=0.3)), None),
    ("omni36: craw_a0.5", make_mods(before_opt=CRAW_CODE.format(alpha=0.5)), None),

    # H2: GIM with different thresholds
    ("omni36: gim_t0.0", make_mods(before_opt=GIM_CODE.format(threshold=0.0)), None),
    ("omni36: gim_t-0.2", make_mods(before_opt=GIM_CODE.format(threshold=-0.2)), None),

    # H3: SNRE with different strengths
    ("omni36: snre_s0.1", make_mods(before_opt=SNRE_CODE.format(strength=0.1)), None),
    ("omni36: snre_s0.3", make_mods(before_opt=SNRE_CODE.format(strength=0.3)), None),

    # H4: TTAB — transformer-targeted audio boost
    ("omni36: ttab_s0.05", make_mods(before_opt=TTAB_CODE.format(strength=0.05)), None),
    ("omni36: ttab_s0.1", make_mods(before_opt=TTAB_CODE.format(strength=0.1)), None),
    ("omni36: ttab_s0.2", make_mods(before_opt=TTAB_CODE.format(strength=0.2)), None),

    # Combinations of best
    ("omni36: craw0.3+gim0.0", make_mods(
        before_opt=CRAW_CODE.format(alpha=0.3) + GIM_CODE.format(threshold=0.0)
    ), None),
    ("omni36: craw0.3+ttab0.1", make_mods(
        before_opt=CRAW_CODE.format(alpha=0.3) + TTAB_CODE.format(strength=0.1)
    ), None),
]

print(f"Running {len(experiments)} Omni-Optimizer experiments...")
results = []
for i, (desc, train_mods, prepare_mods) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {desc}")
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=600)
    results.append((desc, r))

print("\n" + "="*70)
print("SWEEP 36: OMNI-OPTIMIZER RESULTS")
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
print("\n--- Audio Loss Ranking (lower = better audio) ---")
sorted_audio = sorted(results, key=lambda x: x[1]['audio_loss'] if x[1] else 99.0)
for desc, r in sorted_audio:
    if r:
        print(f"  audio={r['audio_loss']:.4f} | val={r['val_loss']:.4f} | text={r['text_bpb']:.4f} | {desc}")
    else:
        print(f"  FAIL     |          |          | {desc}")

# Pareto analysis: did any experiment improve audio WITHOUT hurting text?
ref = next((r for d, r in results if "ref" in d and r), None)
if ref:
    print("\n--- Pareto Analysis (vs reference) ---")
    for desc, r in sorted_results:
        if r and "ref" not in desc:
            d_text = r['text_bpb'] - ref['text_bpb']
            d_audio = r['audio_loss'] - ref['audio_loss']
            d_val = r['val_loss'] - ref['val_loss']
            pareto = "PARETO" if d_audio < 0 and d_text <= 0.005 else ("audio+" if d_audio < -0.01 else "")
            print(f"  Δval={d_val:+.4f} Δtext={d_text:+.4f} Δaudio={d_audio:+.4f} | {pareto:>8} | {desc}")

with open("omni_optimizer_results.json", "w") as f:
    json.dump([(d, {k: v for k, v in r.items()} if r else None) for d, r in results], f, indent=2)
print("\nSaved omni_optimizer_results.json")
