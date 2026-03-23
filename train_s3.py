#!/usr/bin/env python3
"""
Authentic S3 training — matches mini-omni's exact setup.

Starts from their published checkpoint (post-S1+S2+S3).
Uses S3 data with correct SNAC delay pattern.
All weights unfrozen. LR: 2e-6 → 2e-5 cosine, 1500 warmup.

Includes D1-D7 diagnostic signals from train_observe.py.
"""

import os
import sys
import json
import time
import math
import copy
import random
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.environ.get("MINI_OMNI_REF", "/workspace/mini-omni-ref"))
from litgpt.config import Config
from litgpt.model import GPT

# ============================================================================
# Constants
# ============================================================================
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Audio vocab constants
TEXT_VOCAB_SIZE = 152000
AUDIO_VOCAB_SIZE = 4160

# Mini-omni S3 hyperparameters (exact replication from paper)
S3_CONFIG = {
    # LR: paper says "fine-tuning LR: 4e-6 to 5e-5"
    # Also says "S2-3 joint: 2e-5 → 2e-6" — we use the fine-tuning range
    "lr_max": 2e-5,
    "lr_min": 2e-6,
    "batch_size": 2,           # per-GPU (they used 192 on 8×A100)
    "grad_accum": 16,          # effective batch = 32
    "max_steps": 5000,
    "warmup_steps": 1500,      # paper: 1500 warmup steps
    "weight_decay": 0.01,      # standard AdamW
    "adam_betas": (0.9, 0.95),
    "adam_eps": 1e-8,
    "max_grad_norm": 1.0,
    "audio_weight": 1.0,       # λ for audio loss
    "cb_weights": [1,1,1,1,1,1,1],  # per-codebook weights (Exp A: Moshi-style)
    "diag_every": 100,         # D1-D3,D5-D7
    "basin_every": 1000,       # D4 (expensive)
    "val_every": 200,
    "save_every": 2500,       # each ckpt ~2GB, save sparingly
}


# ============================================================================
# Dataset
# ============================================================================
class OmniS3Dataset(Dataset):
    """Load preprocessed S3 data (from prepare_s3.py, with delay pattern)."""

    def __init__(self, data_path, max_len=2048, s2_mode=False, whisper_dir=None):
        self.data = torch.load(data_path, weights_only=False)
        self.max_len = max_len
        if s2_mode:
            orig_len = len(self.data)
            self.data = [d for d in self.data if d['task'] in ('T1T2', 'A1T2')]
            print(f"[s2_mode] Filtered {orig_len} → {len(self.data)} (T1T2+A1T2 only)")
        print(f"Loaded {len(self.data)} sequences from {data_path}")
        from collections import Counter
        tasks = Counter(d['task'] for d in self.data)
        print(f"  Tasks: {dict(tasks)}")

        # Load whisper features if available
        self.whisper_features = {}  # (shard_idx, row_idx) -> features tensor
        if whisper_dir and os.path.isdir(whisper_dir):
            import glob as _glob
            shard_files = sorted(_glob.glob(os.path.join(whisper_dir, "shard_*.pt")))
            n_loaded = 0
            for sf_path in shard_files:
                shard_idx = int(os.path.basename(sf_path).split("_")[1].split(".")[0])
                shard_data = torch.load(sf_path, weights_only=False)
                for row_idx, entry in enumerate(shard_data):
                    if entry is not None:
                        self.whisper_features[(shard_idx, row_idx)] = entry['features']
                        n_loaded += 1
                del shard_data
            print(f"  Loaded {n_loaded} whisper features from {whisper_dir}")
        else:
            print(f"  No whisper features loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        whisper_ref = d.get('whisper_ref', None)
        whisper_feat = None
        whisper_len = d.get('whisper_len', 0)
        if whisper_ref is not None:
            whisper_feat = self.whisper_features.get(tuple(whisper_ref), None)
        return d['streams'], d['loss_mask_text'], d['loss_mask_audio'], d['task'], whisper_feat, whisper_len


def collate_fn(batch):
    """Pad sequences to same length within batch. Handles optional whisper features."""
    streams_list, lm_text_list, lm_audio_list, tasks, whisper_feats, whisper_len_list = zip(*batch)

    max_len = max(s.shape[1] for s in streams_list)
    B = len(batch)

    streams = torch.zeros(B, 8, max_len, dtype=torch.long)
    lm_text = torch.zeros(B, max_len, dtype=torch.bool)
    lm_audio = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (s, lt, la) in enumerate(zip(streams_list, lm_text_list, lm_audio_list)):
        L = s.shape[1]
        streams[i, :, :L] = s
        lm_text[i, :L] = lt
        lm_audio[i, :L] = la

    # Pad whisper features: None means no audio input for this sample
    has_any_whisper = any(wf is not None for wf in whisper_feats)
    if has_any_whisper:
        # Determine feature dim from first non-None feature
        feat_dim = next(wf.shape[-1] for wf in whisper_feats if wf is not None)
        max_wlen = max(wf.shape[0] for wf in whisper_feats if wf is not None)
        audio_features = torch.zeros(B, max_wlen, feat_dim, dtype=torch.float16)
        # Use actual whisper lengths from tokenization (NOT tensor shape which is always 1500)
        whisper_lens = torch.zeros(B, dtype=torch.long)
        for i, (wf, wl) in enumerate(zip(whisper_feats, whisper_len_list)):
            if wf is not None:
                wlen = wf.shape[0]
                audio_features[i, :wlen] = wf
                whisper_lens[i] = wl  # actual audio length, not padded tensor length
    else:
        audio_features = None
        whisper_lens = None

    return streams, lm_text, lm_audio, tasks, audio_features, whisper_lens


# ============================================================================
# Loss computation — matches mini-omni exactly
# ============================================================================
def compute_losses(model, streams, loss_mask_text, loss_mask_audio,
                    audio_features=None, whisper_lens=None, tasks=None):
    """Compute separate text CE and per-codebook audio CE.

    Mini-omni approach (from GitHub issues #101, #135):
    - Text tasks: compute text CE, audio targets masked with -100
    - Audio tasks: compute audio CE (7 heads), text targets masked with -100
    - Loss = text_loss + Σ audio_cb_losses
    """
    input_ids = [streams[:, i, :] for i in range(8)]

    # Build task list for concat_whisper_feat (needed to skip T1/T1A2 samples)
    task_list = list(tasks) if tasks is not None else None

    with torch.amp.autocast('cuda', dtype=DTYPE):
        xa, xt = model(audio_features=audio_features, input_ids=input_ids,
                       whisper_lens=whisper_lens, task=task_list)

    # Text loss — CE on text head, masked to answer positions
    text_targets = input_ids[7][:, 1:]   # (B, T-1) shifted
    text_logits = xt[:, :-1]             # (B, T-1, 152000)
    text_mask = loss_mask_text[:, 1:]    # (B, T-1)

    if text_mask.any():
        text_loss = F.cross_entropy(
            text_logits[text_mask].view(-1, text_logits.shape[-1]),
            text_targets[text_mask].view(-1),
            reduction='mean'
        )
    else:
        text_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

    # Per-codebook audio losses — CE on each of 7 audio heads
    audio_losses = []
    audio_mask = loss_mask_audio[:, 1:]  # (B, T-1)

    for i in range(7):
        audio_targets_raw = input_ids[i][:, 1:]  # (B, T-1) layer-shifted
        audio_logits = xa[i][:, :-1]              # (B, T-1, 4160)

        if audio_mask.any():
            # Un-layershift targets: raw_id - 152000 - i*4160
            targets_unshifted = (audio_targets_raw[audio_mask] - TEXT_VOCAB_SIZE - i * AUDIO_VOCAB_SIZE).clamp(0, AUDIO_VOCAB_SIZE - 1)
            cb_loss = F.cross_entropy(
                audio_logits[audio_mask].view(-1, audio_logits.shape[-1]),
                targets_unshifted.view(-1),
                reduction='mean'
            )
        else:
            cb_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        audio_losses.append(cb_loss)

    return text_loss, audio_losses


def forward_text_loss(model, streams, loss_mask_text, audio_features=None,
                      whisper_lens=None, tasks=None):
    """Text-only loss for basin probing."""
    input_ids = [streams[:, i, :] for i in range(8)]
    task_list = list(tasks) if tasks is not None else None
    with torch.amp.autocast('cuda', dtype=DTYPE):
        xa, xt = model(audio_features=audio_features, input_ids=input_ids,
                       whisper_lens=whisper_lens, task=task_list)
    targets = input_ids[7][:, 1:]
    logits = xt[:, :-1]
    mask = loss_mask_text[:, 1:]
    if not mask.any():
        return torch.tensor(0.0, device=DEVICE)
    return F.cross_entropy(
        logits[mask].view(-1, logits.shape[-1]),
        targets[mask].view(-1),
        reduction='mean'
    )


# ============================================================================
# LR schedule — cosine with linear warmup (exact mini-omni)
# ============================================================================
def get_cosine_lr(step, max_steps, lr_max, lr_min, warmup_steps):
    if step < warmup_steps:
        return lr_min + (lr_max - lr_min) * step / warmup_steps
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Diagnostics (D1-D7, carried from train_observe.py)
# ============================================================================
class DiagnosticTracker:
    def __init__(self, model):
        self.theta_0 = {name: p.data.clone().float().cpu()
                        for name, p in model.named_parameters() if p.requires_grad}
        self.history = []

    def compute_gradient_diagnostics(self, model, val_loader, step):
        """D1: ρ(t), D2: cos φ(t), D7: per-codebook losses."""
        model.eval()

        # Find a batch with both text and audio masks active
        diag_streams = diag_lt = diag_la = diag_af = diag_wl = diag_tasks = None
        for s, lt, la, tasks_d, af, wl in val_loader:
            if lt.any() and la.any():
                max_t = min(s.shape[2], 512)
                diag_streams = s[:, :, :max_t].to(DEVICE)
                diag_lt = lt[:, :max_t].to(DEVICE)
                diag_la = la[:, :max_t].to(DEVICE)
                diag_af = af.to(DEVICE) if af is not None else None
                diag_wl = wl.to(DEVICE) if wl is not None else None
                diag_tasks = tasks_d
                break
        if diag_streams is None:
            return {"rho": 0, "cos_phi": 0, "layer_cos": {},
                    "text_grad_norm": 0, "audio_grad_norm": 0, "cb_losses": {}}

        # Pass 1: text loss backward
        model.zero_grad()
        text_loss, audio_losses = compute_losses(model, diag_streams, diag_lt, diag_la,
                                                  diag_af, diag_wl, diag_tasks)
        if text_loss.item() > 0:
            text_loss.backward()
        text_grads = {}
        text_grad_norm_sq = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                text_grads[name] = p.grad.clone()
                text_grad_norm_sq += p.grad.norm().item()**2
        text_grad_norm = text_grad_norm_sq**0.5

        # Pass 2: audio loss backward
        model.zero_grad()
        _, audio_losses2 = compute_losses(model, diag_streams, diag_lt, diag_la,
                                           diag_af, diag_wl, diag_tasks)
        audio_loss_total = sum(audio_losses2)
        if audio_loss_total.item() > 0:
            audio_loss_total.backward()
        audio_grads = {}
        audio_grad_norm_sq = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                audio_grads[name] = p.grad.clone()
                audio_grad_norm_sq += p.grad.norm().item()**2
        audio_grad_norm = audio_grad_norm_sq**0.5

        # D1: gradient norm ratio
        rho = audio_grad_norm / (text_grad_norm + 1e-10)

        # D2: global cosine similarity
        dot_prod = 0.0
        for name in text_grads:
            if name in audio_grads:
                dot_prod += (text_grads[name] * audio_grads[name]).sum().item()
        cos_phi = dot_prod / (text_grad_norm * audio_grad_norm + 1e-10)

        # D2 per-layer
        layer_cos = {}
        for layer_idx in range(24):
            prefix = f"transformer.h.{layer_idx}."
            t_norm2 = a_norm2 = dot = 0.0
            for name in text_grads:
                if name.startswith(prefix) and name in audio_grads:
                    dot += (text_grads[name] * audio_grads[name]).sum().item()
                    t_norm2 += text_grads[name].norm().item()**2
                    a_norm2 += audio_grads[name].norm().item()**2
            if t_norm2 > 0 and a_norm2 > 0:
                layer_cos[layer_idx] = dot / (t_norm2**0.5 * a_norm2**0.5 + 1e-10)

        # D11: per-module gradient norms
        module_grad_norms = {"text": {}, "audio": {}}
        for modality, grads in [("text", text_grads), ("audio", audio_grads)]:
            emb_norm2 = adapter_norm2 = lm_head_norm2 = 0.0
            layer_norms2 = defaultdict(float)
            for name, g in grads.items():
                gnorm2 = g.norm().item() ** 2
                if "wte" in name:
                    emb_norm2 += gnorm2
                elif "whisper_adapter" in name:
                    adapter_norm2 += gnorm2
                elif "lm_head" in name:
                    lm_head_norm2 += gnorm2
                else:
                    for li in range(24):
                        if f"transformer.h.{li}." in name:
                            layer_norms2[f"layer{li}"] += gnorm2
                            break
            module_grad_norms[modality]["emb"] = emb_norm2 ** 0.5
            module_grad_norms[modality]["adapter"] = adapter_norm2 ** 0.5
            module_grad_norms[modality]["lm_head"] = lm_head_norm2 ** 0.5
            module_grad_norms[modality].update({k: v ** 0.5 for k, v in layer_norms2.items()})

        # D7: per-codebook losses
        cb_losses = {f"cb{i}": al.item() for i, al in enumerate(audio_losses2)}

        del text_grads, audio_grads, diag_streams, diag_lt, diag_la
        model.zero_grad()
        torch.cuda.empty_cache()

        return {
            "rho": rho,
            "cos_phi": cos_phi,
            "layer_cos": layer_cos,
            "text_grad_norm": text_grad_norm,
            "audio_grad_norm": audio_grad_norm,
            "cb_losses": cb_losses,
            "module_grad_norms": module_grad_norms,
        }

    def compute_displacement(self, model):
        """D5: parameter displacement from checkpoint init."""
        total_disp = emb_disp = backbone_disp = lm_head_disp = adapter_disp = 0.0
        text_emb_disp = audio_emb_disp = 0.0

        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.theta_0:
                continue
            d = (p.data.float().cpu() - self.theta_0[name]).norm().item()
            total_disp += d**2
            if "wte" in name:
                emb_disp += d**2
                # Split text vs audio embedding displacement
                text_emb_disp += (p[:TEXT_VOCAB_SIZE].float().cpu() - self.theta_0[name][:TEXT_VOCAB_SIZE]).norm().item()**2
                audio_emb_disp += (p[TEXT_VOCAB_SIZE:].float().cpu() - self.theta_0[name][TEXT_VOCAB_SIZE:]).norm().item()**2
            elif "lm_head" in name:
                lm_head_disp += d**2
            elif "whisper_adapter" in name:
                adapter_disp += d**2
            else:
                backbone_disp += d**2

        return {
            "total": total_disp**0.5,
            "embedding": emb_disp**0.5,
            "text_embedding": text_emb_disp**0.5,
            "audio_embedding": audio_emb_disp**0.5,
            "backbone": backbone_disp**0.5,
            "lm_head": lm_head_disp**0.5,
            "adapter": adapter_disp**0.5,
        }

    def compute_embedding_rank(self, model):
        """D6: effective rank of text/audio embeddings + key backbone weight matrices."""
        wte = model.transformer.wte.weight.data.float()
        text_emb = wte[:TEXT_VOCAB_SIZE]
        audio_emb = wte[TEXT_VOCAB_SIZE:]

        def effective_rank(W, max_rows=8192):
            if W.shape[0] > max_rows:
                idx = torch.randperm(W.shape[0])[:max_rows]
                W = W[idx]
            try:
                s = torch.linalg.svdvals(W.cpu())
                s = s / s.sum()
                s = s[s > 1e-10]
                entropy = -(s * s.log()).sum().item()
                return math.exp(entropy)
            except:
                return 0.0

        result = {
            "text_rank": effective_rank(text_emb),
            "audio_rank": effective_rank(audio_emb),
        }

        # Backbone weight ranks at layers 0, 6, 12, 18, 23
        for li in [0, 6, 12, 18, 23]:
            block = model.transformer.h[li]
            try:
                attn_w = block.attn.attn.weight.data.float().cpu()
                result[f"layer{li}_attn_rank"] = effective_rank(attn_w)
            except:
                pass
            try:
                mlp_w = block.mlp.fc_1.weight.data.float().cpu()
                result[f"layer{li}_mlp_rank"] = effective_rank(mlp_w)
            except:
                pass

        return result

    def compute_topk_and_histogram(self, model, val_loader, max_batches=20):
        """D8: top-k accuracy and loss histogram for text and audio.

        Measures:
        - Top-1/5/10 accuracy for text and each audio codebook
        - Normalized loss = actual_loss / H_random (where H_random = ln(vocab_size))
        - Per-sample loss histogram (binned)
        """
        import math as _math
        model.eval()
        H_random_text = _math.log(TEXT_VOCAB_SIZE)   # ln(152000) = 11.93
        H_random_audio = _math.log(AUDIO_VOCAB_SIZE)  # ln(4160) = 8.33

        # Accumulators
        text_correct = {1: 0, 5: 0, 10: 0}
        text_total = 0
        text_losses_all = []

        cb_correct = {i: {1: 0, 5: 0, 10: 0} for i in range(7)}
        cb_total = {i: 0 for i in range(7)}
        cb_losses_all = {i: [] for i in range(7)}

        n_batches = 0
        with torch.no_grad():
            for streams, lm_text, lm_audio, tasks, af, wl in val_loader:
                if n_batches >= max_batches:
                    break
                n_batches += 1
                streams = streams.to(DEVICE)
                lm_text = lm_text.to(DEVICE)
                lm_audio = lm_audio.to(DEVICE)
                af_dev = af.to(DEVICE) if af is not None else None
                wl_dev = wl.to(DEVICE) if wl is not None else None

                input_ids = [streams[:, i, :] for i in range(8)]
                task_list = list(tasks) if tasks is not None else None
                with torch.amp.autocast('cuda', dtype=DTYPE):
                    xa, xt = model(audio_features=af_dev, input_ids=input_ids,
                                   whisper_lens=wl_dev, task=task_list)

                # Text top-k
                text_targets = input_ids[7][:, 1:]
                text_logits = xt[:, :-1]
                text_mask = lm_text[:, 1:]
                if text_mask.any():
                    tgt = text_targets[text_mask].view(-1)
                    logit = text_logits[text_mask].view(-1, text_logits.shape[-1]).float()
                    text_total += tgt.numel()
                    for k in [1, 5, 10]:
                        topk_ids = logit.topk(k, dim=-1).indices
                        text_correct[k] += (topk_ids == tgt.unsqueeze(-1)).any(-1).sum().item()
                    # Per-token losses
                    per_token_loss = F.cross_entropy(logit, tgt, reduction='none')
                    text_losses_all.extend(per_token_loss.cpu().tolist())

                # Audio top-k per codebook
                audio_mask = lm_audio[:, 1:]
                if audio_mask.any():
                    for i in range(7):
                        audio_targets_raw = input_ids[i][:, 1:]
                        audio_logits = xa[i][:, :-1]
                        tgt = (audio_targets_raw[audio_mask] - TEXT_VOCAB_SIZE - i * AUDIO_VOCAB_SIZE).clamp(0, AUDIO_VOCAB_SIZE - 1).view(-1)
                        logit = audio_logits[audio_mask].view(-1, audio_logits.shape[-1]).float()
                        cb_total[i] += tgt.numel()
                        for k in [1, 5, 10]:
                            topk_ids = logit.topk(k, dim=-1).indices
                            cb_correct[i][k] += (topk_ids == tgt.unsqueeze(-1)).any(-1).sum().item()
                        per_token_loss = F.cross_entropy(logit, tgt, reduction='none')
                        cb_losses_all[i].extend(per_token_loss.cpu().tolist())

        # Compute accuracies
        text_acc = {f"top{k}": text_correct[k] / max(text_total, 1) for k in [1, 5, 10]}
        cb_acc = {}
        for i in range(7):
            for k in [1, 5, 10]:
                cb_acc[f"cb{i}_top{k}"] = cb_correct[i][k] / max(cb_total[i], 1)

        # Normalized losses (actual / H_random)
        text_mean_loss = sum(text_losses_all) / max(len(text_losses_all), 1)
        cb_mean_losses = {}
        for i in range(7):
            cb_mean_losses[f"cb{i}"] = sum(cb_losses_all[i]) / max(len(cb_losses_all[i]), 1)

        # Loss histogram (10 bins from 0 to H_random)
        import numpy as np
        text_hist = {}
        if text_losses_all:
            counts, edges = np.histogram(text_losses_all, bins=10, range=(0, H_random_text))
            text_hist = {f"bin{j}": int(counts[j]) for j in range(10)}
        cb_hist = {}
        for i in range(7):
            if cb_losses_all[i]:
                counts, edges = np.histogram(cb_losses_all[i], bins=10, range=(0, H_random_audio))
                cb_hist[f"cb{i}"] = {f"bin{j}": int(counts[j]) for j in range(10)}

        model.train()
        return {
            "text_topk": text_acc,
            "text_mean_loss": text_mean_loss,
            "text_normalized_loss": text_mean_loss / H_random_text,
            "audio_topk": cb_acc,
            "audio_mean_losses": cb_mean_losses,
            "audio_normalized_losses": {k: v / H_random_audio for k, v in cb_mean_losses.items()},
            "text_loss_hist": text_hist,
            "audio_loss_hist": cb_hist,
            "n_text_tokens": text_total,
            "n_audio_tokens": cb_total[0],
        }

    def compute_linear_probe(self, model, val_loader, probe_layers=(6, 12, 18),
                              n_train_batches=15, n_val_batches=5, probe_steps=200, lr=1e-3):
        """D9: linear probe on backbone hidden states to predict audio tokens.

        Tests whether backbone intermediate layers contain useful audio info
        even when output loss has plateaued.
        """
        model.eval()

        # Hook to capture hidden states at specific layers
        hidden_captures = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden_captures[layer_idx] = output.detach()
            return hook_fn

        for li in probe_layers:
            h = model.transformer.h[li].register_forward_hook(make_hook(li))
            hooks.append(h)

        # Collect data: run forward, grab hidden states + targets
        train_data = {li: {"hidden": [], "targets": []} for li in probe_layers}
        val_data = {li: {"hidden": [], "targets": []} for li in probe_layers}

        batch_count = 0
        with torch.no_grad():
            for streams, lm_text, lm_audio, tasks, af, wl in val_loader:
                batch_count += 1
                if batch_count > n_train_batches + n_val_batches:
                    break
                streams = streams.to(DEVICE)
                lm_audio = lm_audio.to(DEVICE)
                af_dev = af.to(DEVICE) if af is not None else None
                wl_dev = wl.to(DEVICE) if wl is not None else None

                input_ids = [streams[:, i, :] for i in range(8)]
                task_list = list(tasks) if tasks is not None else None
                with torch.amp.autocast('cuda', dtype=DTYPE):
                    model(audio_features=af_dev, input_ids=input_ids,
                          whisper_lens=wl_dev, task=task_list)

                audio_mask = lm_audio[:, 1:]
                if not audio_mask.any():
                    continue

                # Target: cb0 token (semantic codebook)
                cb0_targets = (input_ids[0][:, 1:][audio_mask] - TEXT_VOCAB_SIZE).clamp(0, AUDIO_VOCAB_SIZE - 1)

                dest = train_data if batch_count <= n_train_batches else val_data
                for li in probe_layers:
                    h = hidden_captures[li][:, :-1][audio_mask].float()  # (N, 896)
                    dest[li]["hidden"].append(h.cpu())
                    dest[li]["targets"].append(cb0_targets.cpu())

        # Remove hooks
        for h in hooks:
            h.remove()

        # Train a linear probe per layer
        results = {}
        for li in probe_layers:
            if not train_data[li]["hidden"] or not val_data[li]["hidden"]:
                results[f"layer{li}"] = {"train_acc": 0, "val_acc": 0, "val_loss": 0}
                continue

            X_train = torch.cat(train_data[li]["hidden"])
            y_train = torch.cat(train_data[li]["targets"])
            X_val = torch.cat(val_data[li]["hidden"])
            y_val = torch.cat(val_data[li]["targets"])

            # Simple linear probe
            probe = torch.nn.Linear(X_train.shape[1], AUDIO_VOCAB_SIZE).to(DEVICE)
            opt = torch.optim.Adam(probe.parameters(), lr=lr)

            # Train
            probe.train()
            n = X_train.shape[0]
            for s in range(probe_steps):
                idx = torch.randint(0, n, (min(256, n),))
                logits = probe(X_train[idx].to(DEVICE))
                loss = F.cross_entropy(logits, y_train[idx].to(DEVICE))
                opt.zero_grad()
                loss.backward()
                opt.step()

            # Eval
            probe.eval()
            with torch.no_grad():
                # Train accuracy
                train_logits = probe(X_train[:2048].to(DEVICE))
                train_acc = (train_logits.argmax(-1) == y_train[:2048].to(DEVICE)).float().mean().item()
                # Val accuracy + loss
                val_logits = probe(X_val.to(DEVICE))
                val_acc = (val_logits.argmax(-1) == y_val.to(DEVICE)).float().mean().item()
                val_loss = F.cross_entropy(val_logits, y_val.to(DEVICE)).item()

            results[f"layer{li}"] = {
                "train_acc": round(train_acc, 4),
                "val_acc": round(val_acc, 4),
                "val_loss": round(val_loss, 4),
                "n_train": n,
                "n_val": X_val.shape[0],
            }
            del probe, X_train, y_train, X_val, y_val

        model.train()
        torch.cuda.empty_cache()
        return results

    def compute_cka(self, model, val_loader, probe_layers=(6, 12, 18)):
        """D12: CKA (Centered Kernel Alignment) between current and step-0 hidden states."""
        model.eval()

        # Hook to capture hidden states
        hidden_captures = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                hidden_captures[layer_idx] = output.detach()
            return hook_fn

        for li in probe_layers:
            h = model.transformer.h[li].register_forward_hook(make_hook(li))
            hooks.append(h)

        # Get one val batch
        batch = None
        for streams, lm_text, lm_audio, tasks, af, wl in val_loader:
            batch = (streams, lm_text, lm_audio, tasks, af, wl)
            break

        if batch is None:
            for h in hooks:
                h.remove()
            model.train()
            return {}

        streams, lm_text, lm_audio, tasks, af, wl = batch
        streams = streams.to(DEVICE)
        af_dev = af.to(DEVICE) if af is not None else None
        wl_dev = wl.to(DEVICE) if wl is not None else None

        with torch.no_grad():
            input_ids = [streams[:, i, :] for i in range(8)]
            task_list = list(tasks) if tasks is not None else None
            with torch.amp.autocast('cuda', dtype=DTYPE):
                model(audio_features=af_dev, input_ids=input_ids,
                      whisper_lens=wl_dev, task=task_list)

        current_hidden = {li: hidden_captures[li].float().cpu() for li in probe_layers}

        # Save step-0 hidden states on first call
        if not hasattr(self, '_cka_hidden_0'):
            self._cka_hidden_0 = {li: h.clone() for li, h in current_hidden.items()}
            for h in hooks:
                h.remove()
            model.train()
            return {f"layer{li}": 1.0 for li in probe_layers}  # CKA with self = 1.0

        # Linear CKA: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
        results = {}
        for li in probe_layers:
            X = self._cka_hidden_0[li].reshape(-1, self._cka_hidden_0[li].shape[-1])  # (N, D)
            Y = current_hidden[li].reshape(-1, current_hidden[li].shape[-1])
            # Center
            X = X - X.mean(0)
            Y = Y - Y.mean(0)
            YtX = Y.T @ X
            XtX = X.T @ X
            YtY = Y.T @ Y
            cka = (YtX.norm() ** 2) / (XtX.norm() * YtY.norm() + 1e-10)
            results[f"layer{li}"] = round(cka.item(), 6)

        for h in hooks:
            h.remove()
        model.train()
        return results

    def compute_embedding_collapse(self, model, n_sample=1000):
        """D13: mean pairwise cosine similarity of audio token embeddings.
        Values near 1.0 indicate embedding collapse.
        """
        wte = model.transformer.wte.weight.data
        audio_emb = wte[TEXT_VOCAB_SIZE:]  # all audio embeddings

        n_audio = audio_emb.shape[0]
        if n_audio < 2:
            return 0.0

        # Sample n_sample random audio embeddings
        n_sample = min(n_sample, n_audio)
        idx = torch.randperm(n_audio)[:n_sample]
        sampled = audio_emb[idx].float()  # (n_sample, D)

        # Normalize
        sampled = F.normalize(sampled, dim=1)

        # Pairwise cosine similarity = sampled @ sampled.T
        cos_sim = sampled @ sampled.T  # (n_sample, n_sample)

        # Mean of upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones(n_sample, n_sample, dtype=torch.bool), diagonal=1)
        mean_cos = cos_sim[mask].mean().item()

        return round(mean_cos, 6)

    def probe_basin_width(self, model, val_loader, n_directions=10,
                           epsilons=[0.01, 0.05, 0.1, 0.5, 1.0]):
        """D4: basin width probing (expensive)."""
        model.eval()

        # Baseline text loss
        baseline_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for streams, lm_text, lm_audio, tasks, af, wl in val_loader:
                if n_batches >= 5:
                    break
                streams = streams.to(DEVICE)
                lm_text = lm_text.to(DEVICE)
                af_dev = af.to(DEVICE) if af is not None else None
                wl_dev = wl.to(DEVICE) if wl is not None else None
                loss = forward_text_loss(model, streams, lm_text, af_dev, wl_dev, tasks)
                baseline_loss += loss.item()
                n_batches += 1
        baseline_loss /= max(n_batches, 1)

        original_params = {name: p.data.clone() for name, p in model.named_parameters()
                          if p.requires_grad}

        results = {}
        for eps in epsilons:
            losses = []
            for _ in range(n_directions):
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.add_(torch.randn_like(p) * eps)

                with torch.no_grad():
                    perturbed_loss = 0.0
                    n_b = 0
                    for streams, lm_text, lm_audio, tasks, af, wl in val_loader:
                        if n_b >= 5:
                            break
                        streams = streams.to(DEVICE)
                        lm_text = lm_text.to(DEVICE)
                        af_dev = af.to(DEVICE) if af is not None else None
                        wl_dev = wl.to(DEVICE) if wl is not None else None
                        loss = forward_text_loss(model, streams, lm_text, af_dev, wl_dev, tasks)
                        perturbed_loss += loss.item()
                        n_b += 1
                    perturbed_loss /= max(n_b, 1)
                losses.append(perturbed_loss)

                for name, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(original_params[name])

            results[eps] = {
                "mean_loss": sum(losses) / len(losses),
                "max_loss": max(losses),
                "degradation": (sum(losses) / len(losses)) - baseline_loss,
            }

        model.train()
        return {"baseline_loss": baseline_loss, "perturbations": results}

    # ------------------------------------------------------------------
    # D14 + D15: GSNR diagnostics and text subspace projection
    # ------------------------------------------------------------------
    @staticmethod
    def _is_gsnr_param(name):
        """Parameters for D14 GSNR tracking (wte + layer 0 + layer 23)."""
        return ("wte" in name or "lm_head" in name or
                "transformer.h.0." in name or "transformer.h.23." in name)

    @staticmethod
    def _is_subspace_param(name):
        """Parameters for D15 subspace analysis (layer 0 + layer 23 only)."""
        return ("transformer.h.0." in name or "transformer.h.23." in name)

    def _build_param_index(self, model, filter_fn):
        """Build ordered list of (name, numel, offset) for param subset."""
        index = []
        offset = 0
        for name, p in model.named_parameters():
            if p.requires_grad and filter_fn(name):
                index.append((name, p.numel(), offset))
                offset += p.numel()
        return index, offset  # index, total_dim

    def _extract_flat_grad(self, model, param_index, total_d, device='cuda', dtype=torch.bfloat16):
        """Extract gradients for indexed params into a flat tensor."""
        flat = torch.zeros(total_d, dtype=dtype, device=device)
        param_dict = {n: p for n, p in model.named_parameters()}
        for name, numel, offset in param_index:
            p = param_dict[name]
            if p.grad is not None:
                flat[offset:offset+numel] = p.grad.view(-1).to(dtype=dtype, device=device)
        return flat

    def _get_diag_batch(self, val_loader):
        """Get a single val batch with both text and audio masks active."""
        for s, lt, la, tasks_d, af, wl in val_loader:
            if lt.any() and la.any():
                max_t = min(s.shape[2], 512)
                return (s[:, :, :max_t].to(DEVICE), lt[:, :max_t].to(DEVICE),
                        la[:, :max_t].to(DEVICE), tasks_d,
                        af.to(DEVICE) if af is not None else None,
                        wl.to(DEVICE) if wl is not None else None)
        return None

    def compute_gsnr_and_subspace(self, model, val_loader, K=16, k_svd=16):
        """D14 + D15 combined: GSNR estimation and text subspace projection.

        D14: Accumulate K batches of text/audio gradients, measure cos at K=1,2,4,8,16.
             If cos increases with K → acoustic noise masks semantic signal.
        D15: SVD on text gradients, project audio gradient onto text subspace.
             If energy_ratio >> k/d → audio has hidden semantic component.
        """
        model.eval()

        # Build param indices
        gsnr_index, gsnr_dim = self._build_param_index(model, self._is_gsnr_param)
        sub_index, sub_dim = self._build_param_index(model, self._is_subspace_param)

        if gsnr_dim == 0 or sub_dim == 0:
            model.train()
            return {"error": "no matching parameters"}

        # D14: running sums on GPU (bfloat16)
        sum_text = torch.zeros(gsnr_dim, dtype=torch.bfloat16, device=DEVICE)
        sum_audio = torch.zeros(gsnr_dim, dtype=torch.bfloat16, device=DEVICE)
        # For GSNR variance: sum of squared per-element values (CPU float32)
        ssq_audio = torch.zeros(gsnr_dim, dtype=torch.float32)

        # D15: text gradient matrix on CPU (float32), audio accumulated on CPU
        K_sub = min(K, k_svd)  # use same K batches for both
        G_T = torch.zeros(K_sub, sub_dim, dtype=torch.float32)
        sum_audio_sub = torch.zeros(sub_dim, dtype=torch.float32)

        cos_at_k = {}
        checkpoints = {1, 2, 4, 8, 16}
        k = 0

        for s, lt, la, tasks_d, af, wl in val_loader:
            if k >= K:
                break
            if not (lt.any() and la.any()):
                continue

            max_t = min(s.shape[2], 512)
            s_dev = s[:, :, :max_t].to(DEVICE)
            lt_dev = lt[:, :max_t].to(DEVICE)
            la_dev = la[:, :max_t].to(DEVICE)
            af_dev = af.to(DEVICE) if af is not None else None
            wl_dev = wl.to(DEVICE) if wl is not None else None

            # --- Text backward ---
            model.zero_grad()
            text_loss, _ = compute_losses(model, s_dev, lt_dev, la_dev,
                                          af_dev, wl_dev, tasks_d)
            if text_loss.item() > 0:
                text_loss.backward()

            g_t_gsnr = self._extract_flat_grad(model, gsnr_index, gsnr_dim)
            sum_text += g_t_gsnr
            if k < K_sub:
                G_T[k] = self._extract_flat_grad(model, sub_index, sub_dim,
                                                  device='cpu', dtype=torch.float32)

            # --- Audio backward ---
            model.zero_grad()
            _, audio_losses = compute_losses(model, s_dev, lt_dev, la_dev,
                                              af_dev, wl_dev, tasks_d)
            audio_total = sum(audio_losses)
            if audio_total.item() > 0:
                audio_total.backward()

            g_a_gsnr = self._extract_flat_grad(model, gsnr_index, gsnr_dim)
            sum_audio += g_a_gsnr
            ssq_audio += g_a_gsnr.float().cpu() ** 2
            if k < K_sub:
                sum_audio_sub += self._extract_flat_grad(model, sub_index, sub_dim,
                                                          device='cpu', dtype=torch.float32)

            k += 1

            # D14: cos at checkpoint K values
            if k in checkpoints:
                st = sum_text.float()
                sa = sum_audio.float()
                cos_val = F.cosine_similarity(st.unsqueeze(0), sa.unsqueeze(0)).item()
                cos_at_k[k] = round(cos_val, 6)

            del s_dev, lt_dev, la_dev, af_dev, wl_dev, g_t_gsnr, g_a_gsnr

        # --- D14: GSNR estimates ---
        gsnr_audio = gsnr_text = 0.0
        if k > 1:
            mu_audio = sum_audio.float().cpu() / k
            var_audio = (ssq_audio / k) - mu_audio ** 2
            var_audio = var_audio.clamp(min=0)
            gsnr_audio = (mu_audio.norm() ** 2 / (var_audio.sum() + 1e-10)).item()

        # --- D15: SVD + projection ---
        d15_result = {}
        k_used = min(k, K_sub)
        if k_used >= 2:
            G_T = G_T[:k_used]
            try:
                U, S, Vt = torch.linalg.svd(G_T, full_matrices=False)
                g_A = sum_audio_sub / k_used
                mu_T = G_T.mean(dim=0)
                g_A_norm_sq = g_A.norm() ** 2

                energy_by_k = {}
                for top_k in [1, 2, 4, 8, min(16, k_used)]:
                    if top_k > k_used:
                        continue
                    V_k = Vt[:top_k]  # (top_k, d)
                    coeffs = V_k @ g_A  # (top_k,)
                    g_A_proj = V_k.T @ coeffs  # (d,)
                    ratio = (g_A_proj.norm() ** 2 / (g_A_norm_sq + 1e-10)).item()
                    energy_by_k[top_k] = round(ratio, 6)

                # Best projection: cos with mean text
                best_k = min(16, k_used)
                V_best = Vt[:best_k]
                g_A_proj_best = V_best.T @ (V_best @ g_A)
                cos_proj = F.cosine_similarity(
                    g_A_proj_best.unsqueeze(0), mu_T.unsqueeze(0)).item()
                cos_raw = F.cosine_similarity(
                    g_A.unsqueeze(0), mu_T.unsqueeze(0)).item()

                d15_result = {
                    "energy_ratio": energy_by_k.get(best_k, 0),
                    "random_baseline": round(best_k / sub_dim, 8),
                    "cos_proj_text": round(cos_proj, 6),
                    "cos_raw_text": round(cos_raw, 6),
                    "top_singular_values": [round(v, 4) for v in S[:best_k].tolist()],
                    "energy_ratio_by_k": {str(kk): v for kk, v in energy_by_k.items()},
                    "param_dim": sub_dim,
                }
            except Exception as e:
                d15_result = {"error": str(e)}

        # Cleanup
        del sum_text, sum_audio, ssq_audio, G_T, sum_audio_sub
        model.zero_grad()
        torch.cuda.empty_cache()
        model.train()

        return {
            "gsnr": {
                "cos_accumulated": {str(kk): v for kk, v in cos_at_k.items()},
                "gsnr_audio": round(gsnr_audio, 6),
                "n_batches": k,
                "gsnr_param_dim": gsnr_dim,
            },
            "text_subspace": d15_result,
        }


# ============================================================================
# Model loading
# ============================================================================
def load_mini_omni_checkpoint(ckpt_dir=os.environ.get("MINI_OMNI_CKPT", "/workspace/mini-omni-ckpt")):
    """Load mini-omni checkpoint with their exact config."""
    config = Config.from_file(f"{ckpt_dir}/model_config.yaml")
    print(f"Config: post_adapter={config.post_adapter}, "
          f"tie_weights={config.tie_word_embeddings}, "
          f"vocab={config.padded_vocab_size}")

    model = GPT(config)

    ckpt = torch.load(f"{ckpt_dir}/lit_model.pth", map_location='cpu', weights_only=True)
    result = model.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")
    del ckpt
    gc.collect()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,} "
          f"(tie_weights → lm_head shares wte)")

    return model


def load_from_checkpoint(path, ckpt_dir=os.environ.get("MINI_OMNI_CKPT", "/workspace/mini-omni-ckpt")):
    """Load model from a previously saved checkpoint .pt file."""
    config = Config.from_file(f"{ckpt_dir}/model_config.yaml")
    model = GPT(config)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    result = model.load_state_dict(ckpt, strict=True)
    print(f"Loaded from {path}: missing={len(result.missing_keys)}, "
          f"unexpected={len(result.unexpected_keys)}")
    del ckpt
    gc.collect()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
    return model


def build_post_s1_checkpoint(ckpt_dir=os.environ.get("MINI_OMNI_CKPT", "/workspace/mini-omni-ckpt")):
    """Build a post-S1 checkpoint: Qwen2-0.5B + trained whisper_adapter + fresh audio embeddings.

    Simulates the state after S1 (adapter training) but before S2 (text adaptation).
    Uses tie_word_embeddings=True (matching published config).
    """
    from transformers import AutoModelForCausalLM

    # Load model with published config (tie_word_embeddings=True)
    config = Config.from_file(f"{ckpt_dir}/model_config.yaml")
    model = GPT(config)

    # Step 1: Load published checkpoint to extract whisper_adapter weights
    ckpt = torch.load(f"{ckpt_dir}/lit_model.pth", map_location='cpu', weights_only=True)
    adapter_weights = {k: v for k, v in ckpt.items() if "whisper_adapter" in k}
    print(f"Extracted {len(adapter_weights)} whisper_adapter weight tensors from published ckpt")
    del ckpt
    gc.collect()

    # Step 2: Load Qwen2-0.5B weights into the model (QKV interleaving etc.)
    print("Loading Qwen2-0.5B pretrained weights...")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=DTYPE)
    qwen_sd = qwen.state_dict()
    litgpt_sd = model.state_dict()

    # QKV interleaving (same logic as train_observe.py)
    for layer_idx in range(24):
        qp = f"model.layers.{layer_idx}.self_attn"
        lp = f"transformer.h.{layer_idx}.attn"

        q_w = qwen_sd[f"{qp}.q_proj.weight"]
        k_w = qwen_sd[f"{qp}.k_proj.weight"]
        v_w = qwen_sd[f"{qp}.v_proj.weight"]
        q_b = qwen_sd[f"{qp}.q_proj.bias"]
        k_b = qwen_sd[f"{qp}.k_proj.bias"]
        v_b = qwen_sd[f"{qp}.v_proj.bias"]

        n_kv_groups, q_per_kv, head_size = 2, 7, 64
        q_w = q_w.view(n_kv_groups, q_per_kv, head_size, -1)
        k_w = k_w.view(n_kv_groups, 1, head_size, -1)
        v_w = v_w.view(n_kv_groups, 1, head_size, -1)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=1).reshape(-1, q_w.shape[-1])

        q_b = q_b.view(n_kv_groups, q_per_kv, head_size)
        k_b = k_b.view(n_kv_groups, 1, head_size)
        v_b = v_b.view(n_kv_groups, 1, head_size)
        qkv_b = torch.cat([q_b, k_b, v_b], dim=1).reshape(-1)

        litgpt_sd[f"{lp}.attn.weight"] = qkv_w
        litgpt_sd[f"{lp}.attn.bias"] = qkv_b
        litgpt_sd[f"{lp}.proj.weight"] = qwen_sd[f"{qp}.o_proj.weight"]

        lp_block = f"transformer.h.{layer_idx}"
        litgpt_sd[f"{lp_block}.mlp.fc_1.weight"] = qwen_sd[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        litgpt_sd[f"{lp_block}.mlp.fc_2.weight"] = qwen_sd[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        litgpt_sd[f"{lp_block}.mlp.proj.weight"] = qwen_sd[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
        litgpt_sd[f"{lp_block}.norm_1.weight"] = qwen_sd[f"model.layers.{layer_idx}.input_layernorm.weight"]
        litgpt_sd[f"{lp_block}.norm_2.weight"] = qwen_sd[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]

    litgpt_sd["transformer.ln_f.weight"] = qwen_sd["model.norm.weight"]

    # Step 3: Embeddings — text from Qwen2, audio random
    qwen_emb = qwen_sd["model.embed_tokens.weight"]  # (151936, 896)
    full_emb = litgpt_sd["transformer.wte.weight"]    # (181120, 896)
    full_emb[:qwen_emb.shape[0]] = qwen_emb
    nn.init.normal_(full_emb[TEXT_VOCAB_SIZE:], std=0.02)
    litgpt_sd["transformer.wte.weight"] = full_emb
    # lm_head shares wte (tie_word_embeddings=True), no separate init needed

    del qwen, qwen_sd
    gc.collect()

    # Step 4: Copy whisper_adapter weights from published ckpt
    for k, v in adapter_weights.items():
        litgpt_sd[k] = v
    print(f"Copied whisper_adapter weights from published checkpoint")

    model.load_state_dict(litgpt_sd, strict=False)
    print(f"Built post-S1 checkpoint: Qwen2 LLM + trained adapters + fresh audio embeddings")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")

    return model


# ============================================================================
# Training loop
# ============================================================================
def train(config=None, output_dir="results/s3_adam"):
    if config is None:
        config = S3_CONFIG

    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---- Model ----
    init_checkpoint = config.get("init_checkpoint", None)
    checkpoint_mode = config.get("checkpoint_mode", "published")
    if init_checkpoint:
        print(f"Loading from saved checkpoint: {init_checkpoint}")
        model = load_from_checkpoint(init_checkpoint)
    elif checkpoint_mode == "post_s1":
        print("Building post-S1 checkpoint (Qwen2 + adapters, no S2)...")
        model = build_post_s1_checkpoint()
    else:
        print("Loading mini-omni checkpoint...")
        model = load_mini_omni_checkpoint()

    # S3: ALL weights unfrozen (including whisper adapter) by default
    for p in model.parameters():
        p.requires_grad = True

    # S2 mode: freeze whisper_adapter
    s2_mode = config.get("s2_mode", False)
    if s2_mode:
        n_frozen = 0
        for name, p in model.named_parameters():
            if "whisper_adapter" in name:
                p.requires_grad = False
                n_frozen += 1
        print(f"[s2_mode] Froze {n_frozen} whisper_adapter params")

    # Audio embedding initialization from text embeddings
    audio_emb_init = config.get("audio_emb_init", None)
    if audio_emb_init and checkpoint_mode == "post_s1":
        wte = model.transformer.wte.weight.data
        text_emb = wte[:TEXT_VOCAB_SIZE].float()  # (152000, 896)
        n_audio = wte.shape[0] - TEXT_VOCAB_SIZE   # 29120
        if audio_emb_init == "sample":
            # Random samples from text embeddings
            idx = torch.randperm(TEXT_VOCAB_SIZE)[:n_audio]
            wte[TEXT_VOCAB_SIZE:] = text_emb[idx].to(wte.dtype)
            print(f"[audio_emb_init=sample] Initialized {n_audio} audio embeddings from random text embeddings")
        elif audio_emb_init == "kmeans":
            # K-means clustering of text embeddings (approximate with mini-batch)
            print(f"[audio_emb_init=kmeans] Running k-means on text embeddings...")
            from torch import cdist
            # Subsample text embeddings for speed
            n_sample = min(50000, TEXT_VOCAB_SIZE)
            sample_idx = torch.randperm(TEXT_VOCAB_SIZE)[:n_sample]
            X = text_emb[sample_idx]  # (50000, 896)
            # Initialize centroids randomly from X
            centroids = X[torch.randperm(n_sample)[:n_audio]].clone()  # (29120, 896)
            for km_iter in range(20):
                dists = cdist(X, centroids)  # (50000, 29120) — large but float32 on CPU
                assignments = dists.argmin(dim=1)
                new_centroids = torch.zeros_like(centroids)
                counts = torch.zeros(n_audio)
                for i in range(n_sample):
                    new_centroids[assignments[i]] += X[i]
                    counts[assignments[i]] += 1
                mask = counts > 0
                new_centroids[mask] /= counts[mask].unsqueeze(1)
                # Re-init empty clusters
                empty = ~mask
                if empty.any():
                    new_centroids[empty] = X[torch.randperm(n_sample)[:empty.sum()]]
                centroids = new_centroids
            wte[TEXT_VOCAB_SIZE:] = centroids.to(wte.dtype)
            print(f"[audio_emb_init=kmeans] Initialized {n_audio} audio embeddings from {n_sample} text embedding k-means")

    # Freeze backbone: only train embeddings + adapter + heads
    if config.get("freeze_backbone", False):
        n_frozen = 0
        for name, p in model.named_parameters():
            if "transformer.h." in name or "transformer.ln_f" in name:
                p.requires_grad = False
                n_frozen += 1
        print(f"[freeze_backbone] Froze {n_frozen} backbone params (transformer.h + ln_f)")

    # Gradient checkpointing for VRAM
    import types
    for block in model.transformer.h:
        block._orig_forward = block.forward
        block.forward = types.MethodType(
            lambda self, *a, **kw: torch.utils.checkpoint.checkpoint(
                self._orig_forward, *a, use_reentrant=False, **kw),
            block)

    model = model.to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable:,}")

    # ---- Optimizer (AdamW, exact mini-omni) ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr_max"],
        betas=tuple(config["adam_betas"]),
        eps=config["adam_eps"],
        weight_decay=config["weight_decay"],
    )

    # ---- GradNorm setup ----
    method = config.get("method", "baseline")
    if method == "gradnorm":
        w_text = nn.Parameter(torch.ones(1, device=DEVICE))
        w_audio_gn = nn.Parameter(torch.ones(1, device=DEVICE))
        gradnorm_opt = torch.optim.Adam([w_text, w_audio_gn], lr=0.025)
        gradnorm_L_text_0 = None
        gradnorm_L_audio_0 = None
        print(f"  [gradnorm] alpha={config.get('gradnorm_alpha', 1.5)}, weight_lr=0.025")

    # ---- Adjust batch for memory-hungry methods ----
    if method in ("grad_proj", "m_sam"):
        # 2 forward passes → need smaller batch
        config["batch_size"] = 1
        config["grad_accum"] = 32
        print(f"  [{method}] batch_size=1, grad_accum=32 (2x forward passes)")

    # ---- Data ----
    data_dir = os.environ.get("S3_DATA_DIR", "/root/.cache/autoresearch/s3_data")
    whisper_dir = os.path.join(data_dir, "whisper_features")
    if not os.path.isdir(whisper_dir):
        whisper_dir = None

    curriculum_step = config.get("curriculum", 0)
    if curriculum_step > 0:
        # Phase 1: A1T2 only (easiest audio task)
        train_dataset_curriculum = OmniS3Dataset(f"{data_dir}/train.pt",
                                                  whisper_dir=whisper_dir)
        # Filter to A1T2 only
        orig_len = len(train_dataset_curriculum.data)
        train_dataset_curriculum.data = [d for d in train_dataset_curriculum.data if d['task'] == 'A1T2']
        print(f"[curriculum] Phase 1 (steps 1-{curriculum_step}): A1T2 only, "
              f"{orig_len} → {len(train_dataset_curriculum.data)} samples")
        train_loader_curriculum = DataLoader(
            train_dataset_curriculum, batch_size=config["batch_size"],
            shuffle=True, collate_fn=collate_fn, num_workers=2,
            pin_memory=True, drop_last=True,
        )
        # Phase 2: all tasks (loaded later to save memory during phase 1)
        print(f"[curriculum] Phase 2 (steps {curriculum_step+1}+): all 4 tasks")

    train_dataset = OmniS3Dataset(f"{data_dir}/train.pt", s2_mode=s2_mode,
                                   whisper_dir=whisper_dir)
    val_dataset = OmniS3Dataset(f"{data_dir}/val.pt",
                                 whisper_dir=whisper_dir)  # val always full (all 4 task types)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=2,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # ---- Diagnostics ----
    tracker = DiagnosticTracker(model)
    diagnostics_log = []

    # ---- Training ----
    print(f"\nS3 Training: {config['max_steps']} steps, "
          f"LR {config['lr_min']:.0e}→{config['lr_max']:.0e}, "
          f"warmup {config['warmup_steps']}, "
          f"eff_batch={config['batch_size']*config['grad_accum']}")
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    # Curriculum: start with A1T2-only loader, switch to full loader at curriculum_step
    if curriculum_step > 0:
        active_loader = train_loader_curriculum
        print(f"[curriculum] Starting with A1T2-only loader")
    else:
        active_loader = train_loader
    train_iter = iter(active_loader)
    curriculum_switched = False

    step = 0
    accum_text_loss = 0.0
    accum_audio_loss = 0.0
    accum_cb_losses = [0.0] * 7
    accum_steps = 0
    t0 = time.time()

    while step < config["max_steps"]:
        # Curriculum: switch from A1T2-only to full dataset
        if curriculum_step > 0 and step >= curriculum_step and not curriculum_switched:
            active_loader = train_loader
            train_iter = iter(active_loader)
            curriculum_switched = True
            print(f"\n[curriculum] Step {step}: switching to all 4 tasks")

        # Get batch
        try:
            streams, lm_text, lm_audio, tasks, audio_features, whisper_lens = next(train_iter)
        except StopIteration:
            train_iter = iter(active_loader)
            streams, lm_text, lm_audio, tasks, audio_features, whisper_lens = next(train_iter)

        streams = streams.to(DEVICE)
        lm_text = lm_text.to(DEVICE)
        lm_audio = lm_audio.to(DEVICE)
        if audio_features is not None:
            audio_features = audio_features.to(DEVICE)
            whisper_lens = whisper_lens.to(DEVICE)

        # ---- Method: gradient projection ----
        method = config.get("method", "baseline")
        cb_w = config.get("cb_weights", [1]*7)

        if method == "grad_proj":
            # Two separate forward+backward passes (no retain_graph, saves memory)
            audio_weight = config.get("audio_weight", 1.0)

            # Pass 1: text loss forward+backward
            model.zero_grad()
            text_loss_1, _ = compute_losses(model, streams, lm_text, lm_audio,
                                            audio_features, whisper_lens, tasks)
            (text_loss_1 / config["grad_accum"]).backward()
            text_grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    text_grads[name] = p.grad.clone()

            # Pass 2: audio loss forward+backward
            model.zero_grad()
            text_loss, audio_losses = compute_losses(model, streams, lm_text, lm_audio,
                                                     audio_features, whisper_lens, tasks)
            audio_loss = sum(cb_w[i] * audio_losses[i] for i in range(7))
            scaled_audio = (audio_weight * audio_loss) / config["grad_accum"]
            scaled_audio.backward()

            # Project: remove destructive component of audio grad
            dot_global = 0.0
            t_norm2 = 0.0
            a_norm2 = 0.0
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and name in text_grads:
                    dot_global += (p.grad * text_grads[name]).sum().item()
                    t_norm2 += text_grads[name].norm().item()**2
                    a_norm2 += p.grad.norm().item()**2

            cos_phi_step = dot_global / (max(t_norm2, 1e-10)**0.5 * max(a_norm2, 1e-10)**0.5)

            if cos_phi_step < 0 and t_norm2 > 0:
                proj_scale = dot_global / t_norm2
                for name, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None and name in text_grads:
                        p.grad.sub_(proj_scale * text_grads[name])

            # Combine: text_grad + projected_audio_grad
            for name, p in model.named_parameters():
                if p.requires_grad and name in text_grads:
                    if p.grad is not None:
                        p.grad.add_(text_grads[name])
                    else:
                        p.grad = text_grads[name].clone()

            del text_grads

            accum_text_loss += text_loss.item()
            accum_audio_loss += audio_loss.item()
            for i in range(7):
                accum_cb_losses[i] += audio_losses[i].item()
            accum_steps += 1

        elif method == "m_sam":
            # M-SAM: modality-aware SAM (perturb along text gradient direction)
            audio_weight = config.get("audio_weight", 1.0)
            sam_rho = config.get("sam_rho", 0.05)

            # Step 1: normal forward-backward to get text gradient direction
            model.zero_grad()
            text_loss, audio_losses = compute_losses(model, streams, lm_text, lm_audio,
                                                     audio_features, whisper_lens, tasks)
            audio_loss = sum(cb_w[i] * audio_losses[i] for i in range(7))
            total_loss = text_loss + audio_weight * audio_loss
            (total_loss / config["grad_accum"]).backward()

            # Step 2: compute perturbation from text gradient (use combined grad as proxy)
            text_grad_norm_sq = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    text_grad_norm_sq += p.grad.norm().item()**2
            text_grad_norm = text_grad_norm_sq**0.5

            # Save params, apply perturbation
            with torch.no_grad():
                old_params = {n: p.data.clone() for n, p in model.named_parameters()
                              if p.requires_grad}
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        p.data.add_(sam_rho * p.grad / (text_grad_norm + 1e-12))

            # Step 3: forward-backward at perturbed point
            model.zero_grad()
            text_loss2, audio_losses2 = compute_losses(model, streams, lm_text, lm_audio,
                                                       audio_features, whisper_lens, tasks)
            audio_loss2 = sum(cb_w[i] * audio_losses2[i] for i in range(7))
            total_loss2 = text_loss2 + audio_weight * audio_loss2
            (total_loss2 / config["grad_accum"]).backward()

            # Step 4: restore params (use perturbed gradient for update)
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        p.data.copy_(old_params[n])
            del old_params

            accum_text_loss += text_loss.item()
            accum_audio_loss += audio_loss.item()
            for i in range(7):
                accum_cb_losses[i] += audio_losses[i].item()
            accum_steps += 1

        else:
            # Standard forward-backward (baseline, adaptive_lambda, gradnorm, entropy_scaled)
            text_loss, audio_losses = compute_losses(model, streams, lm_text, lm_audio,
                                                     audio_features, whisper_lens, tasks)

            # Entropy-scaled: normalize ALL losses by their random baselines
            # so text and audio are on the same [0,1] scale
            if method == "entropy_scaled":
                cb_entropy = config.get("cb_entropy", [1.0]*7)
                H_text_random = math.log(TEXT_VOCAB_SIZE)  # ln(152000) = 11.93
                # Normalized losses: actual / H_random → [0, 1]
                text_loss_norm = text_loss / H_text_random
                audio_loss_norm = sum(cb_w[i] * audio_losses[i] / cb_entropy[i] for i in range(7)) / 7.0
                # Now both are ~[0,1], combine with audio_weight
                audio_weight = config.get("audio_weight", 1.0)
                total_loss = text_loss_norm + audio_weight * audio_loss_norm
                # For logging, use raw audio sum
                audio_loss = sum(cb_w[i] * audio_losses[i] for i in range(7))
            else:
                audio_loss = sum(cb_w[i] * audio_losses[i] for i in range(7))

                # Task weighting
                if method == "gradnorm":
                    audio_weight = w_audio_gn.item()
                    total_loss = w_text.item() * text_loss + audio_weight * audio_loss
                elif method == "adaptive_lambda":
                    audio_weight = config.get("_current_lambda", config.get("audio_weight", 1.0))
                    total_loss = text_loss + audio_weight * audio_loss
                else:
                    audio_weight = config.get("audio_weight", 1.0)
                    total_loss = text_loss + audio_weight * audio_loss

            # Backward with gradient accumulation
            scaled_loss = total_loss / config["grad_accum"]
            scaler.scale(scaled_loss).backward()

            accum_text_loss += text_loss.item()
            accum_audio_loss += audio_loss.item()
            for i in range(7):
                accum_cb_losses[i] += audio_losses[i].item()
            accum_steps += 1

        if accum_steps >= config["grad_accum"]:
            # Gradient clipping
            if method not in ("grad_proj", "m_sam"):
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["max_grad_norm"]
            )

            # LR schedule (use global step for correct schedule across chained runs)
            global_step = config.get("start_step", 0) + step
            lr_total = config.get("lr_total_steps", config["max_steps"] + config.get("start_step", 0))
            lr = get_cosine_lr(global_step, lr_total,
                              config["lr_max"], config["lr_min"],
                              config["warmup_steps"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            if method in ("grad_proj", "m_sam"):
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad()

            step += 1
            avg_text = accum_text_loss / accum_steps
            avg_audio = accum_audio_loss / accum_steps
            avg_cbs = [c / accum_steps for c in accum_cb_losses]
            accum_text_loss = 0.0
            accum_audio_loss = 0.0
            accum_cb_losses = [0.0] * 7
            accum_steps = 0

            # GradNorm weight update
            if method == "gradnorm":
                alpha_gn = config.get("gradnorm_alpha", 1.5)

                # Record initial losses at step 1
                if step == 1:
                    gradnorm_L_text_0 = avg_text
                    gradnorm_L_audio_0 = avg_audio
                    print(f"  [gradnorm] L_text_0={gradnorm_L_text_0:.4f}, L_audio_0={gradnorm_L_audio_0:.4f}")

                if gradnorm_L_text_0 is not None and gradnorm_L_text_0 > 0 and gradnorm_L_audio_0 > 0:
                    ln_f_w = model.transformer.ln_f.weight

                    # Text gradient norm w.r.t. ln_f.weight
                    model.zero_grad()
                    tl_gn, _ = compute_losses(model, streams, lm_text, lm_audio,
                                              audio_features, whisper_lens, tasks)
                    tl_gn.backward()
                    g_text_norm = ln_f_w.grad.norm().item() if ln_f_w.grad is not None else 1e-10

                    # Audio gradient norm w.r.t. ln_f.weight
                    model.zero_grad()
                    _, al_gn = compute_losses(model, streams, lm_text, lm_audio,
                                              audio_features, whisper_lens, tasks)
                    al_gn_total = sum(al_gn)
                    al_gn_total.backward()
                    g_audio_norm = ln_f_w.grad.norm().item() if ln_f_w.grad is not None else 1e-10
                    model.zero_grad()

                    # Differentiable GradNorm loss
                    G_text = w_text * g_text_norm
                    G_audio = w_audio_gn * g_audio_norm
                    G_bar = (G_text + G_audio) / 2

                    r_text = avg_text / gradnorm_L_text_0
                    r_audio = avg_audio / gradnorm_L_audio_0
                    r_bar = (r_text + r_audio) / 2 + 1e-10
                    r_tilde_text = r_text / r_bar
                    r_tilde_audio = r_audio / r_bar

                    target_text = G_bar.detach() * (r_tilde_text ** alpha_gn)
                    target_audio = G_bar.detach() * (r_tilde_audio ** alpha_gn)

                    L_grad = torch.abs(G_text - target_text) + torch.abs(G_audio - target_audio)

                    gradnorm_opt.zero_grad()
                    L_grad.backward()
                    gradnorm_opt.step()

                    # Clamp to positive, then renormalize to sum to 2
                    with torch.no_grad():
                        w_text.clamp_(min=0.01)
                        w_audio_gn.clamp_(min=0.01)
                        wsum = w_text.item() + w_audio_gn.item()
                        w_text.mul_(2.0 / wsum)
                        w_audio_gn.mul_(2.0 / wsum)

                    model.train()
                    optimizer.zero_grad()

            # Console logging
            if step % 10 == 0:
                elapsed = time.time() - t0
                cb_str = " ".join(f"{c:.2f}" for c in avg_cbs)
                gn_suffix = ""
                if method == "gradnorm":
                    gn_suffix = f" w=[{w_text.item():.3f},{w_audio_gn.item():.3f}]"
                print(f"  step {step}/{config['max_steps']} | "
                      f"text={avg_text:.4f} audio={avg_audio:.4f} | "
                      f"CB=[{cb_str}] | "
                      f"lr={lr:.2e} gnorm={grad_norm:.2f}{gn_suffix} | {elapsed:.0f}s")

            # ---- Diagnostics ----
            if step % config["diag_every"] == 0:
                print(f"  [diag] step {step}...")
                diag = {
                    "step": step,
                    "time": time.time() - t0,
                    "text_loss": avg_text,
                    "audio_loss": avg_audio,
                    "cb_losses_train": {f"cb{i}": avg_cbs[i] for i in range(7)},
                    "cb_weighted_train": {f"cb{i}": cb_w[i] * avg_cbs[i] for i in range(7)},
                    "lr": lr,
                    "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                }

                # D5: displacement
                diag["displacement"] = tracker.compute_displacement(model)

                # D6: embedding rank
                diag["embedding_rank"] = tracker.compute_embedding_rank(model)

                # D13: audio embedding cosine collapse
                diag["audio_emb_cos_collapse"] = tracker.compute_embedding_collapse(model)

                # GradNorm weights
                if method == "gradnorm":
                    diag["gradnorm_w_text"] = w_text.item()
                    diag["gradnorm_w_audio"] = w_audio_gn.item()

                # D1, D2, D7: gradient-based
                try:
                    grad_diag = tracker.compute_gradient_diagnostics(
                        model, val_loader, step)
                    diag.update(grad_diag)
                except Exception as e:
                    print(f"  [diag] Gradient diagnostics failed: {e}")

                # D8: top-k accuracy, normalized loss, loss histogram
                try:
                    topk_diag = tracker.compute_topk_and_histogram(
                        model, val_loader, max_batches=20)
                    diag.update(topk_diag)
                    print(f"  [diag] text top1={topk_diag['text_topk']['top1']:.3f} "
                          f"top10={topk_diag['text_topk']['top10']:.3f} "
                          f"norm_loss={topk_diag['text_normalized_loss']:.3f} | "
                          f"cb0 top1={topk_diag['audio_topk'].get('cb0_top1', 0):.3f} "
                          f"top10={topk_diag['audio_topk'].get('cb0_top10', 0):.3f} "
                          f"norm_loss={topk_diag['audio_normalized_losses'].get('cb0', 0):.3f}")
                except Exception as e:
                    print(f"  [diag] Top-k diagnostics failed: {e}")

                # D9: linear probe (every 500 steps — expensive)
                if step % 500 == 0:
                    try:
                        probe_diag = tracker.compute_linear_probe(
                            model, val_loader, probe_layers=(6, 12, 18))
                        diag["linear_probe"] = probe_diag
                        for li in (6, 12, 18):
                            key = f"layer{li}"
                            if key in probe_diag:
                                p = probe_diag[key]
                                print(f"  [probe] layer {li}: "
                                      f"train_acc={p['train_acc']:.3f} "
                                      f"val_acc={p['val_acc']:.3f} "
                                      f"val_loss={p['val_loss']:.3f}")
                    except Exception as e:
                        print(f"  [diag] Linear probe failed: {e}")

                    # D12: CKA (every 500 steps)
                    try:
                        cka_diag = tracker.compute_cka(model, val_loader)
                        diag["cka"] = cka_diag
                        cka_str = " ".join(f"L{k.replace('layer','')}={v:.3f}" for k, v in cka_diag.items())
                        print(f"  [cka] {cka_str}")
                    except Exception as e:
                        print(f"  [diag] CKA failed: {e}")

                    # D14+D15: GSNR and text subspace projection (every 500 steps)
                    try:
                        gsnr_sub = tracker.compute_gsnr_and_subspace(
                            model, val_loader, K=16, k_svd=16)
                        diag.update(gsnr_sub)
                        gd = gsnr_sub.get("gsnr", {})
                        cos_acc = gd.get("cos_accumulated", {})
                        sd = gsnr_sub.get("text_subspace", {})
                        print(f"  [gsnr] cos@1={cos_acc.get('1',0):.4f} "
                              f"cos@4={cos_acc.get('4',0):.4f} "
                              f"cos@16={cos_acc.get('16',0):.4f} "
                              f"GSNR_A={gd.get('gsnr_audio',0):.4f}")
                        print(f"  [subspace] energy={sd.get('energy_ratio',0):.6f} "
                              f"(rand={sd.get('random_baseline',0):.2e}) "
                              f"cos_proj={sd.get('cos_proj_text',0):.4f} "
                              f"cos_raw={sd.get('cos_raw_text',0):.4f}")
                    except Exception as e:
                        print(f"  [diag] GSNR/subspace failed: {e}")

                model.train()
                optimizer.zero_grad()

                # Adaptive λ update based on ρ
                if method == "adaptive_lambda" and "rho" in diag:
                    rho_val = diag["rho"]
                    alpha = config.get("adaptive_alpha", 0.1)
                    current_lambda = config.get("_current_lambda", config.get("audio_weight", 1.0))
                    # Push λ up when ρ<1 (audio too weak), down when ρ>1
                    if abs(rho_val - 1.0) < 1.0:
                        delta = alpha * (1.0 - rho_val)
                        current_lambda = max(0.1, min(10.0, current_lambda + delta))
                        config["_current_lambda"] = current_lambda
                    diag["adaptive_lambda"] = current_lambda
                    print(f"    adaptive λ={current_lambda:.3f} (ρ={rho_val:.4f})")

                diagnostics_log.append(diag)
                with open(f"{output_dir}/diagnostics.json", "w") as f:
                    json.dump(diagnostics_log, f, indent=2, default=str)

            # D4: basin probing (expensive)
            if step % config["basin_every"] == 0:
                print(f"  [basin] step {step}...")
                basin = tracker.probe_basin_width(model, val_loader)
                if diagnostics_log:
                    diagnostics_log[-1]["basin"] = basin
                with open(f"{output_dir}/diagnostics.json", "w") as f:
                    json.dump(diagnostics_log, f, indent=2, default=str)

            # ---- Validation ----
            if step % config["val_every"] == 0:
                model.eval()
                val_text = val_audio = 0.0
                val_cbs = [0.0] * 7
                val_n = 0
                # D10: per-task val loss
                task_text_losses = defaultdict(float)
                task_audio_losses = defaultdict(float)
                task_counts = defaultdict(int)
                with torch.no_grad():
                    for sv, ltv, lav, tv, afv, wlv in val_loader:
                        if val_n >= 50:
                            break
                        sv = sv.to(DEVICE)
                        ltv = ltv.to(DEVICE)
                        lav = lav.to(DEVICE)
                        afv_dev = afv.to(DEVICE) if afv is not None else None
                        wlv_dev = wlv.to(DEVICE) if wlv is not None else None
                        tl, als = compute_losses(model, sv, ltv, lav, afv_dev, wlv_dev, tv)
                        val_text += tl.item()
                        val_audio += sum(a.item() for a in als)
                        for i in range(7):
                            val_cbs[i] += als[i].item()
                        val_n += 1
                        # D10: accumulate per-task losses
                        for task_name in tv:
                            task_text_losses[task_name] += tl.item()
                            task_audio_losses[task_name] += sum(a.item() for a in als)
                            task_counts[task_name] += 1
                val_text /= max(val_n, 1)
                val_audio /= max(val_n, 1)
                val_cbs = [c / max(val_n, 1) for c in val_cbs]
                cb_str = " ".join(f"{c:.2f}" for c in val_cbs)
                # D10: per-task averages
                val_task_losses = {}
                for tn in task_counts:
                    n = task_counts[tn]
                    val_task_losses[tn] = {
                        "text": task_text_losses[tn] / n,
                        "audio": task_audio_losses[tn] / n,
                    }
                task_str = " ".join(f"{tn}={val_task_losses[tn]['text']:.3f}/{val_task_losses[tn]['audio']:.3f}"
                                    for tn in sorted(val_task_losses))
                print(f"  [val] step {step}: text={val_text:.4f} "
                      f"audio={val_audio:.4f} CB=[{cb_str}]")
                print(f"  [val] per-task (text/audio): {task_str}")
                # Save to latest diag entry if one exists for this step
                if diagnostics_log and diagnostics_log[-1].get("step") == step:
                    diagnostics_log[-1]["val_task_losses"] = val_task_losses
                    diagnostics_log[-1]["val_text"] = val_text
                    diagnostics_log[-1]["val_audio"] = val_audio
                    diagnostics_log[-1]["val_cbs"] = {f"cb{i}": val_cbs[i] for i in range(7)}
                    with open(f"{output_dir}/diagnostics.json", "w") as f:
                        json.dump(diagnostics_log, f, indent=2, default=str)
                model.train()

            # ---- Checkpointing ----
            if step % config["save_every"] == 0:
                ckpt_path = f"{output_dir}/step_{step}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [save] {ckpt_path}")

    # ---- Final ----
    print(f"\nTraining complete ({time.time()-t0:.0f}s). Saving...")
    with open(f"{output_dir}/diagnostics.json", "w") as f:
        json.dump(diagnostics_log, f, indent=2, default=str)
    if config.get("save_every", 2500) < 999999:
        torch.save(model.state_dict(), f"{output_dir}/model_final.pt")

    # Final validation
    model.eval()
    val_text = val_audio = 0.0
    val_n = 0
    with torch.no_grad():
        for sv, ltv, lav, tv, afv, wlv in val_loader:
            if val_n >= 100:
                break
            sv = sv.to(DEVICE)
            ltv = ltv.to(DEVICE)
            lav = lav.to(DEVICE)
            afv_dev = afv.to(DEVICE) if afv is not None else None
            wlv_dev = wlv.to(DEVICE) if wlv is not None else None
            tl, als = compute_losses(model, sv, ltv, lav, afv_dev, wlv_dev, tv)
            val_text += tl.item()
            val_audio += sum(a.item() for a in als)
            val_n += 1
    val_text /= max(val_n, 1)
    val_audio /= max(val_n, 1)
    print(f"  Final val: text={val_text:.4f} audio={val_audio:.4f}")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mini-omni S3 training")
    parser.add_argument("--output_dir", default="results/s3_adam")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--start_step", type=int, default=0,
                        help="Global step offset for LR schedule (for chained runs)")
    parser.add_argument("--lr_total_steps", type=int, default=None,
                        help="Total planned steps for LR cosine schedule (default=start_step+max_steps)")
    parser.add_argument("--audio_weight", type=float, default=1.0)
    parser.add_argument("--lr_max", type=float, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--method", default="baseline",
                        choices=["baseline", "grad_proj", "adaptive_lambda", "m_sam",
                                 "gradnorm", "entropy_scaled"],
                        help="Training method variant")
    parser.add_argument("--adaptive_alpha", type=float, default=0.1,
                        help="Adaptive λ step size")
    parser.add_argument("--gradnorm_alpha", type=float, default=1.5,
                        help="GradNorm asymmetry parameter (higher = more aggressive rebalancing)")
    parser.add_argument("--cb_entropy", type=str, default=None,
                        help="Per-codebook entropy values, comma-separated 7 floats (for entropy_scaled)")
    parser.add_argument("--grad_accum", type=int, default=None,
                        help="Gradient accumulation steps (default 16, eff_batch=batch_size*grad_accum)")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Save checkpoint every N steps (default 2500)")
    parser.add_argument("--no_save_model", action="store_true",
                        help="Skip saving model checkpoints (save disk)")
    # Exp A: codebook weighting
    parser.add_argument("--cb_weights", type=str, default=None,
                        help="Per-codebook weights, comma-separated (e.g. '100,10,1,1,10,1,1')")
    # Exp B: M-SAM
    parser.add_argument("--sam_rho", type=float, default=0.05,
                        help="SAM perturbation radius for m_sam method")
    # Exp C: checkpoint mode
    parser.add_argument("--checkpoint_mode", default="published",
                        choices=["published", "post_s1"],
                        help="Checkpoint init: published (post-S3) or post_s1 (Qwen2+adapters)")
    parser.add_argument("--init_checkpoint", default=None,
                        help="Path to saved .pt checkpoint (overrides checkpoint_mode)")
    # Exp D: S2 mode
    parser.add_argument("--s2_mode", action="store_true",
                        help="Authentic S2: text-only training, freeze adapters, full val diagnostics")
    # Batch 4: new experiment strategies
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Freeze transformer layers, only train embeddings + adapter + heads")
    parser.add_argument("--audio_emb_init", default=None,
                        choices=["kmeans", "sample"],
                        help="Initialize audio embeddings from text embeddings (kmeans clusters or random samples)")
    parser.add_argument("--curriculum", type=int, default=0,
                        help="Curriculum: train only A1T2 for first N steps, then all 4 tasks")
    args = parser.parse_args()

    config = dict(S3_CONFIG)
    config["max_steps"] = args.max_steps
    config["audio_weight"] = args.audio_weight
    config["start_step"] = args.start_step
    if args.lr_total_steps is not None:
        config["lr_total_steps"] = args.lr_total_steps
    config["method"] = args.method
    config["adaptive_alpha"] = args.adaptive_alpha
    config["gradnorm_alpha"] = args.gradnorm_alpha
    config["sam_rho"] = args.sam_rho
    if args.cb_entropy is not None:
        config["cb_entropy"] = [float(x) for x in args.cb_entropy.split(",")]
        assert len(config["cb_entropy"]) == 7, f"cb_entropy must have 7 values, got {len(config['cb_entropy'])}"
    config["checkpoint_mode"] = args.checkpoint_mode
    config["init_checkpoint"] = args.init_checkpoint
    config["s2_mode"] = args.s2_mode
    config["freeze_backbone"] = args.freeze_backbone
    config["audio_emb_init"] = args.audio_emb_init
    config["curriculum"] = args.curriculum
    if args.cb_weights is not None:
        config["cb_weights"] = [float(x) for x in args.cb_weights.split(",")]
        assert len(config["cb_weights"]) == 7, f"cb_weights must have 7 values, got {len(config['cb_weights'])}"
    if args.grad_accum is not None:
        config["grad_accum"] = args.grad_accum
    if args.save_every is not None:
        config["save_every"] = args.save_every
    if args.no_save_model:
        config["save_every"] = 999999  # effectively never
    if args.lr_max is not None:
        config["lr_max"] = args.lr_max
    if args.lr_min is not None:
        config["lr_min"] = args.lr_min
    if args.warmup_steps is not None:
        config["warmup_steps"] = args.warmup_steps
    if args.weight_decay is not None:
        config["weight_decay"] = args.weight_decay

    # S2 mode defaults
    if args.s2_mode:
        config["audio_weight"] = 0.0  # no audio loss in training
        if args.lr_max is None:
            config["lr_max"] = 2e-4
        if args.lr_min is None:
            config["lr_min"] = 2e-6

    train(config=config, output_dir=args.output_dir)
