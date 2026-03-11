"""Diagnostic instrumentation run.

Single training run with extra logging to measure:
- Per-token frequencies and gradient norms (H1, H4)
- Singular value spectrum before/after NS (H2, H8)
- Embedding effective rank per modality (H6)
- WD/LR ratio during warmdown (H9)

Saves diagnostics to diagnostic_results/ directory.
"""
import os, subprocess, sys

os.makedirs("diagnostic_results", exist_ok=True)

with open("train.py") as f:
    original = f.read()

# Patch 1: Add diagnostic imports and state after model compilation
patch1_target = 'model = torch.compile(model, dynamic=False)'
patch1_replace = '''model = torch.compile(model, dynamic=False)

# ===== DIAGNOSTIC INSTRUMENTATION =====
import json as _json
diag_token_freq = torch.zeros(TOTAL_VOCAB_SIZE, dtype=torch.long, device="cuda")
diag_grad_norms = []
diag_ns_spectra = []
diag_embed_ranks = []
DIAG_LOG_INTERVAL = 50
'''

# Patch 2: Count token frequencies inside training loop (specific indented line)
patch2_target = '        x, y, epoch = next(train_loader)\n\n    # H2: Progressive audio upweighting'
patch2_replace = '''        x, y, epoch = next(train_loader)
        # DIAGNOSTIC: count token frequencies
        diag_token_freq.scatter_add_(0, x.reshape(-1), torch.ones(x.numel(), dtype=torch.long, device="cuda"))

    # H2: Progressive audio upweighting'''

# Patch 3: Add gradient/embedding diagnostics + NS spectrum before optimizer.step()
patch3_target = '    optimizer.step()\n    model.zero_grad(set_to_none=True)'
patch3_replace = '''    # DIAGNOSTIC: per-modality gradient norms
    if step % DIAG_LOG_INTERVAL == 0:
        with torch.no_grad():
            wte_grad = model._orig_mod.transformer.wte.weight.grad
            if wte_grad is not None:
                text_grad = wte_grad[:AUDIO_START_ID].float()
                audio_grad = wte_grad[AUDIO_START_ID:].float()
                text_row_norms = text_grad.norm(dim=1)
                audio_row_norms = audio_grad.norm(dim=1)
                diag_grad_norms.append({
                    "step": step, "progress": progress,
                    "text_grad_norm": text_grad.norm().item(),
                    "audio_grad_norm": audio_grad.norm().item(),
                    "text_grad_mean_row": text_row_norms.mean().item(),
                    "audio_grad_mean_row": audio_row_norms.mean().item(),
                    "text_grad_max_row": text_row_norms.max().item(),
                    "audio_grad_max_row": audio_row_norms.max().item(),
                    "lrm": lrm, "muon_wd": muon_weight_decay,
                })
            # Embedding effective rank (H6)
            wte = model._orig_mod.transformer.wte.weight
            text_emb = wte[:AUDIO_START_ID].float()
            audio_emb = wte[AUDIO_START_ID:].float()
            def _eff_rank(mat):
                import math as _m
                s = torch.linalg.svdvals(mat)
                s = s / s.sum()
                s = s[s > 1e-10]
                return _m.exp(-(s * s.log()).sum().item())
            diag_embed_ranks.append({
                "step": step, "text_eff_rank": _eff_rank(text_emb),
                "audio_eff_rank": _eff_rank(audio_emb),
                "text_norm": text_emb.norm().item(), "audio_norm": audio_emb.norm().item(),
            })
    # DIAGNOSTIC: NS spectrum analysis (every 100 steps)
    if step % 100 == 0:
        with torch.no_grad():
            q_weight = model._orig_mod.transformer.h[0].attn.c_q.weight
            if q_weight.grad is not None:
                g = q_weight.grad.float()
                sv_before = torch.linalg.svdvals(g).cpu().tolist()[:20]
                X = g / (g.norm() * 1.02 + 1e-6)
                for a, b, c in [(8.156554524902461, -22.48329292557795, 15.878769915207462),
                                (4.042929935166739, -2.808917465908714, 0.5000178451051316),
                                (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
                                (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
                                (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)]:
                    A = X.mT @ X
                    B = b * A + c * (A @ A)
                    X = a * X + X @ B
                sv_after = torch.linalg.svdvals(X.float()).cpu().tolist()[:20]
                diag_ns_spectra.append({
                    "step": step, "layer": 0,
                    "sv_before": sv_before, "sv_after": sv_after,
                    "sv_ratio": max(sv_before) / (min(sv_before[:10]) + 1e-10),
                })
    optimizer.step()
    model.zero_grad(set_to_none=True)'''

# Patch 4: Save diagnostics at the end
patch4_target = 'print("---")'
patch4_replace = '''# ===== SAVE DIAGNOSTICS =====
freq_data = {
    "text_freq": diag_token_freq[:AUDIO_START_ID].cpu().tolist(),
    "audio_freq": diag_token_freq[AUDIO_START_ID:].cpu().tolist(),
    "text_total": diag_token_freq[:AUDIO_START_ID].sum().item(),
    "audio_total": diag_token_freq[AUDIO_START_ID:].sum().item(),
}
with open("diagnostic_results/token_frequencies.json", "w") as f:
    _json.dump(freq_data, f)
with open("diagnostic_results/gradient_norms.json", "w") as f:
    _json.dump(diag_grad_norms, f, indent=2)
with open("diagnostic_results/ns_spectra.json", "w") as f:
    _json.dump(diag_ns_spectra, f, indent=2)
with open("diagnostic_results/embed_ranks.json", "w") as f:
    _json.dump(diag_embed_ranks, f, indent=2)
print("Diagnostics saved to diagnostic_results/")
print("---")'''

# Apply patches
modified = original
patches = [
    (patch1_target, patch1_replace, "compile+diag_init"),
    (patch2_target, patch2_replace, "token_freq"),
    (patch3_target, patch3_replace, "grad+embed+ns"),
    (patch4_target, patch4_replace, "save_diag"),
]

for target, replace, name in patches:
    if target not in modified:
        print(f"PATCH FAILED: {name} — target string not found!")
        sys.exit(1)
    modified = modified.replace(target, replace, 1)
    print(f"Patch applied: {name}")

with open("train_diagnostic.py", "w") as f:
    f.write(modified)

print("Running diagnostic training run...")
result = subprocess.run([".venv/bin/python", "train_diagnostic.py"], timeout=600)
if result.returncode != 0:
    print("Diagnostic run failed!")
    sys.exit(1)
print("Diagnostic run complete!")
