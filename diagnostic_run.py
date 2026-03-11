"""Diagnostic instrumentation run.

Single training run with extra logging to measure:
- Per-token frequencies and gradient norms (H1, H4)
- Singular value spectrum before/after NS (H2, H8)
- Embedding effective rank per modality (H6)
- WD/LR ratio during warmdown (H9)

Saves diagnostics to diagnostic_results/ directory.
"""
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import time
import json
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOTAL_VOCAB_SIZE, AUDIO_START_ID, Tokenizer, make_dataloader, evaluate_val_loss

# Import the model and optimizer from train.py (we'll copy the essential bits)
# Actually, let's just patch train.py to add diagnostics

import subprocess, re, sys

# We'll modify train.py to add diagnostic logging and run it
with open("train.py") as f:
    original = f.read()

# Create diagnostics directory
os.makedirs("diagnostic_results", exist_ok=True)

# Add diagnostic code: log every 50 steps
diagnostic_code = '''
# ===== DIAGNOSTIC INSTRUMENTATION =====
import json, math
os.makedirs("diagnostic_results", exist_ok=True)
diag_token_freq = torch.zeros(vocab_size, dtype=torch.long, device="cuda")
diag_grad_norms = []  # per-step: {step, text_grad_norm, audio_grad_norm, ...}
diag_ns_spectra = []  # sampled NS before/after spectra
diag_embed_ranks = []  # effective rank of embedding submatrices
DIAG_LOG_INTERVAL = 50
'''

# Insert after model compilation
modified = original.replace(
    'model = torch.compile(model, dynamic=False)',
    'model = torch.compile(model, dynamic=False)\n' + diagnostic_code
)

# Add token frequency counting in the training loop
token_count_code = '''
        # DIAGNOSTIC: count token frequencies
        diag_token_freq.scatter_add_(0, x.reshape(-1), torch.ones(x.numel(), dtype=torch.long, device="cuda"))
'''

modified = modified.replace(
    '        x, y, epoch = next(train_loader)',
    '        x, y, epoch = next(train_loader)\n' + token_count_code
)

# Add gradient diagnostics after loss.backward() but before optimizer.step()
grad_diag_code = '''
    # DIAGNOSTIC: per-modality gradient norms
    if step % DIAG_LOG_INTERVAL == 0:
        with torch.no_grad():
            wte_grad = model._orig_mod.transformer.wte.weight.grad
            if wte_grad is not None:
                text_grad = wte_grad[:AUDIO_START_ID].float()
                audio_grad = wte_grad[AUDIO_START_ID:].float()
                # Per-row gradient norms
                text_row_norms = text_grad.norm(dim=1)
                audio_row_norms = audio_grad.norm(dim=1)
                diag_grad_norms.append({
                    "step": step,
                    "progress": progress,
                    "text_grad_norm": text_grad.norm().item(),
                    "audio_grad_norm": audio_grad.norm().item(),
                    "text_grad_mean_row": text_row_norms.mean().item(),
                    "audio_grad_mean_row": audio_row_norms.mean().item(),
                    "text_grad_median_row": text_row_norms.median().item(),
                    "audio_grad_median_row": audio_row_norms.median().item(),
                    "text_grad_max_row": text_row_norms.max().item(),
                    "audio_grad_max_row": audio_row_norms.max().item(),
                    "lrm": lrm,
                    "muon_wd": muon_weight_decay,
                })

            # Embedding effective rank (H6)
            wte = model._orig_mod.transformer.wte.weight
            text_emb = wte[:AUDIO_START_ID].float()
            audio_emb = wte[AUDIO_START_ID:].float()

            # Effective rank = exp(entropy of normalized singular values)
            def effective_rank(mat):
                s = torch.linalg.svdvals(mat)
                s = s / s.sum()
                s = s[s > 1e-10]  # avoid log(0)
                entropy = -(s * s.log()).sum().item()
                return math.exp(entropy)

            text_rank = effective_rank(text_emb)
            audio_rank = effective_rank(audio_emb)
            diag_embed_ranks.append({
                "step": step,
                "text_eff_rank": text_rank,
                "audio_eff_rank": audio_rank,
                "text_norm": text_emb.norm().item(),
                "audio_norm": audio_emb.norm().item(),
            })
'''

# Insert the gradient diagnostics before the optimizer.step() call
modified = modified.replace(
    '    optimizer.step()',
    grad_diag_code + '\n    optimizer.step()'
)

# Add NS spectrum logging inside the Muon step (sample occasionally)
# We'll add it by modifying the muon_step_fused function — actually that's compiled,
# so let's just measure the raw gradient singular values outside the fused function
ns_diag_code = '''
    # DIAGNOSTIC: NS spectrum analysis (every 100 steps)
    if step % 100 == 0:
        with torch.no_grad():
            # Get gradient of first transformer layer's Q projection
            q_weight = model._orig_mod.transformer.h[0].attn.c_q.weight
            if q_weight.grad is not None:
                g = q_weight.grad.float()
                sv_before = torch.linalg.svdvals(g).cpu().tolist()[:20]  # top 20 SVs

                # Simulate NS: G @ (G^T G)^{-1/2} approximation
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
                    "step": step,
                    "layer": 0,
                    "sv_before": sv_before,
                    "sv_after": sv_after,
                    "sv_ratio": max(sv_before) / (min(sv_before[:10]) + 1e-10),  # condition number proxy
                })
'''

modified = modified.replace(
    '    optimizer.step()',
    ns_diag_code + '\n    optimizer.step()'
)

# Add final diagnostic save
save_diag_code = '''
# ===== SAVE DIAGNOSTICS =====
# Per-token frequencies
freq_data = {
    "text_freq": diag_token_freq[:AUDIO_START_ID].cpu().tolist(),
    "audio_freq": diag_token_freq[AUDIO_START_ID:].cpu().tolist(),
    "text_total": diag_token_freq[:AUDIO_START_ID].sum().item(),
    "audio_total": diag_token_freq[AUDIO_START_ID:].sum().item(),
}
with open("diagnostic_results/token_frequencies.json", "w") as f:
    json.dump(freq_data, f)

# Gradient norms over training
with open("diagnostic_results/gradient_norms.json", "w") as f:
    json.dump(diag_grad_norms, f, indent=2)

# NS spectra
with open("diagnostic_results/ns_spectra.json", "w") as f:
    json.dump(diag_ns_spectra, f, indent=2)

# Embedding effective ranks
with open("diagnostic_results/embed_ranks.json", "w") as f:
    json.dump(diag_embed_ranks, f, indent=2)

print("Diagnostics saved to diagnostic_results/")
'''

modified = modified.replace(
    'print("---")',
    save_diag_code + '\nprint("---")'
)

# Write modified version and run
with open("train_diagnostic.py", "w") as f:
    f.write(modified)

print("Running diagnostic training run...")
result = subprocess.run([".venv/bin/python", "train_diagnostic.py"], timeout=600)
if result.returncode != 0:
    print("Diagnostic run failed!")
    sys.exit(1)
print("Diagnostic run complete!")
