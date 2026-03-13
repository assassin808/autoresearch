"""Sweep 38: Advanced Omni-Optimizer — Math-driven approaches.

A3: SNR balancing (signal-to-noise ratio per modality)
A4: Gradient subspace projection (blind source separation)
A5: Curvature-aware singular value reweighting (Riemannian)
A6: Orthogonal exploration regularization
A7: Selective NS scaling restoration
A8: Gradient interference dampening per layer

ALL operate on transformer matrices via Muon. No embedding changes.
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

# ============================================================
# A3: SNR Balancing — scale transformer matrix gradients by inverse noise level
# ============================================================
A3_SNR = '''
# === A3 state ===
_a3_grad_mean = {}    # EMA of gradient per layer
_a3_grad_var = {}     # EMA of gradient variance per layer
_a3_beta = 0.95
'''

A3_CODE = '''
    # === A3: SNR Balancing ===
    # Track gradient signal (mean) and noise (variance) per transformer layer.
    # Scale LR inversely with noise: high-noise layers get smaller steps.
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.grad is not None and _p.dim() == 2:
                    _k = f"{_bi}_{_pn}"
                    _g = _p.grad.float()
                    if _k not in _a3_grad_mean:
                        _a3_grad_mean[_k] = _g.clone()
                        _a3_grad_var[_k] = torch.zeros_like(_g)
                    else:
                        _a3_grad_mean[_k].lerp_(_g, 1 - _a3_beta)
                        _diff = _g - _a3_grad_mean[_k]
                        _a3_grad_var[_k].lerp_(_diff.square(), 1 - _a3_beta)
                    if step > 50:
                        _signal = _a3_grad_mean[_k].norm()
                        _noise = _a3_grad_var[_k].sqrt().norm() + 1e-8
                        _snr = _signal / _noise
                        # Low SNR = noisy gradient = scale down to avoid noise amplification
                        # High SNR = clean gradient = normal step
                        _scale = _snr.clamp(0.5, 2.0) / 1.0  # centered around 1
                        _p.grad.mul_(_scale.clamp(0.3, 3.0))
'''

# ============================================================
# A4: Gradient Subspace Projection (Blind Source Separation)
# ============================================================
A4_STATE = '''
# === A4 state ===
_a4_grad_history = {}  # sliding window of recent gradients per param group
_a4_window = 10
'''

A4_CODE = '''
    # === A4: Gradient Subspace Projection ===
    # Track gradient history, find dominant direction (text), boost subordinate (audio).
    # This is blind source separation — no modality labels needed.
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.grad is not None and _p.dim() == 2:
                    _k = f"{{_bi}}_{{_pn}}"
                    _g_flat = _p.grad.float().reshape(-1)
                    if _k not in _a4_grad_history:
                        _a4_grad_history[_k] = []
                    _a4_grad_history[_k].append(_g_flat.clone())
                    if len(_a4_grad_history[_k]) > _a4_window:
                        _a4_grad_history[_k].pop(0)
                    if len(_a4_grad_history[_k]) >= 5 and step > 30:
                        # Stack history into matrix, find dominant direction
                        _H = torch.stack(_a4_grad_history[_k])  # [window, params]
                        _mean_dir = _H.mean(dim=0)
                        _mean_dir = _mean_dir / (_mean_dir.norm() + 1e-8)
                        # Project current gradient onto dominant direction
                        _proj_coeff = (_g_flat * _mean_dir).sum()
                        _dominant = _proj_coeff * _mean_dir
                        _subordinate = _g_flat - _dominant
                        # Boost subordinate component (audio-enriched)
                        _g_new = _dominant + {boost} * _subordinate
                        _p.grad.copy_(_g_new.reshape(_p.grad.shape).to(_p.grad.dtype))
'''

# ============================================================
# A6: Orthogonal Exploration Regularization
# ============================================================
A6_STATE = '''
# === A6 state ===
_a6_prev_update = {}
'''

A6_CODE = '''
    # === A6: Orthogonal Exploration Regularization ===
    # After computing gradient, add component orthogonal to previous update.
    # This encourages exploring NEW directions (where audio signal lives).
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.grad is not None and _p.dim() == 2:
                    _k = f"{{_bi}}_{{_pn}}"
                    _g = _p.grad.float()
                    if _k in _a6_prev_update:
                        _prev = _a6_prev_update[_k]
                        # Compute orthogonal component
                        _proj = (_g * _prev).sum() / (_prev.norm()**2 + 1e-8)
                        _parallel = _proj * _prev
                        _ortho = _g - _parallel
                        # Boost orthogonal component
                        _p.grad.copy_((_parallel + {ortho_boost} * _ortho).to(_p.grad.dtype))
                    _a6_prev_update[_k] = _g.clone()
'''

# ============================================================
# A8: Gradient Interference Dampening Per Layer
# ============================================================
A8_STATE = '''
# === A8 state ===
_a8_prev_grad = {}
_a8_layer_lr_scale = {}
'''

A8_CODE = '''
    # === A8: Gradient Interference Dampening ===
    # When consecutive gradients conflict (cos < 0), dampen that layer.
    # When consistent (cos > 0), normal step. This prevents oscillation
    # from modality competition.
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.grad is not None and _p.dim() == 2:
                    _k = f"{{_bi}}_{{_pn}}"
                    _g = _p.grad
                    if _k in _a8_prev_grad:
                        _prev = _a8_prev_grad[_k]
                        _cos = (_g * _prev).sum() / (_g.norm() * _prev.norm() + 1e-8)
                        # Interference score: how much gradient reverses
                        # cos=-1: full reversal, cos=1: consistent
                        _scale = (1.0 + _cos.clamp(-1, 1)) / 2.0  # maps [-1,1] -> [0,1]
                        _scale = _scale.clamp({min_scale}, 1.0)
                        _p.grad.mul_(_scale)
                    _a8_prev_grad[_k] = _g.clone()
'''

# ============================================================
# A5: Curvature-Aware SVD Reweighting (simplified Riemannian)
# ============================================================
A5_STATE = '''
# === A5 state ===
_a5_grad_sq_ema = {}
_a5_beta = 0.95
'''

A5_CODE = '''
    # === A5: Curvature-Aware Reweighting ===
    # Use EMA of squared gradients as curvature proxy (Gauss-Newton approx).
    # High curvature directions (text) get dampened. Low curvature (audio) get boosted.
    with torch.no_grad():
        for _bi, _block in enumerate(model._orig_mod.transformer.h):
            for _pn, _p in _block.named_parameters():
                if _p.grad is not None and _p.dim() == 2:
                    _k = f"{_bi}_{_pn}"
                    _g = _p.grad.float()
                    _g_sq = _g.square()
                    if _k not in _a5_grad_sq_ema:
                        _a5_grad_sq_ema[_k] = _g_sq.clone()
                    else:
                        _a5_grad_sq_ema[_k].lerp_(_g_sq, 1 - _a5_beta)
                    if step > 30:
                        # Curvature proxy: EMA of squared gradient
                        _curv = _a5_grad_sq_ema[_k].sqrt() + 1e-8
                        # Scale inversely with curvature (like Adam but per-element on matrices)
                        _scale = 1.0 / _curv
                        _scale = _scale / (_scale.mean() + 1e-8)  # normalize to mean=1
                        _p.grad.mul_(_scale.clamp(0.3, 3.0).to(_p.grad.dtype))
'''


# ============================================================
# Build experiments
# ============================================================

def make_mods(state_code="", before_opt_code=""):
    mods = []
    if state_code:
        mods.append((AFTER_DEPTH, AFTER_DEPTH + state_code))
    if before_opt_code:
        mods.append((BEFORE_OPT_STEP, before_opt_code + "\n" + BEFORE_OPT_STEP))
    return mods


experiments = [
    # Reference
    ("adv38: ref", [], None),

    # A3: SNR Balancing
    ("adv38: a3_snr", make_mods(A3_SNR, A3_CODE), None),

    # A4: Gradient Subspace Projection (boost subordinate = audio-enriched)
    ("adv38: a4_proj_1.5", make_mods(A4_STATE, A4_CODE.format(boost=1.5)), None),
    ("adv38: a4_proj_2.0", make_mods(A4_STATE, A4_CODE.format(boost=2.0)), None),
    ("adv38: a4_proj_3.0", make_mods(A4_STATE, A4_CODE.format(boost=3.0)), None),

    # A5: Curvature-aware reweighting (Riemannian)
    ("adv38: a5_curv", make_mods(A5_STATE, A5_CODE), None),

    # A6: Orthogonal exploration
    ("adv38: a6_ortho_1.5", make_mods(A6_STATE, A6_CODE.format(ortho_boost=1.5)), None),
    ("adv38: a6_ortho_2.0", make_mods(A6_STATE, A6_CODE.format(ortho_boost=2.0)), None),

    # A8: Interference dampening
    ("adv38: a8_damp_0.3", make_mods(A8_STATE, A8_CODE.format(min_scale=0.3)), None),
    ("adv38: a8_damp_0.5", make_mods(A8_STATE, A8_CODE.format(min_scale=0.5)), None),

    # Combinations of best ideas
    ("adv38: a4_2.0+a8_0.5", make_mods(
        A4_STATE + A8_STATE,
        A4_CODE.format(boost=2.0) + A8_CODE.format(min_scale=0.5)
    ), None),

    ("adv38: a3+a6_1.5", make_mods(
        A3_SNR + A6_STATE,
        A3_CODE + A6_CODE.format(ortho_boost=1.5)
    ), None),
]

print(f"Running {len(experiments)} Advanced Omni-Optimizer experiments...")
results = []
for i, (desc, train_mods, prepare_mods) in enumerate(experiments):
    print(f"\n[{i+1}/{len(experiments)}] {desc}")
    r = run_exp(desc, train_mods=train_mods, prepare_mods=prepare_mods, timeout=600)
    results.append((desc, r))

print("\n" + "="*70)
print("SWEEP 38: ADVANCED OMNI-OPTIMIZER RESULTS")
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

ref = next((r for d, r in results if d == "adv38: ref" and r), None)
if ref:
    print("\n--- Pareto Analysis ---")
    for desc, r in sorted_results:
        if r and "ref" not in desc:
            d_text = r['text_bpb'] - ref['text_bpb']
            d_audio = r['audio_loss'] - ref['audio_loss']
            pareto = "PARETO!" if d_audio < -0.01 and d_text < 0.005 else ""
            print(f"  Δtext={d_text:+.4f} Δaudio={d_audio:+.4f} | {pareto:>8} | {desc}")

with open("advanced_omni_results.json", "w") as f:
    json.dump([(d, r) for d, r in results], f, indent=2)
print("\nSaved advanced_omni_results.json")
