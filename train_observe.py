#!/usr/bin/env python3
"""
Stage 1: Observation Training — Replicate mini-omni S2 with full diagnostics.

Measures 7 diagnostic signals throughout training:
  D1. Gradient norm ratio ρ(t) = λ‖∇L_audio‖ / ‖∇L_text‖
  D2. Gradient interference cos φ(t) per layer
  D3. Per-modality loss velocity
  D4. Basin width probing (periodic)
  D5. Parameter displacement from init
  D6. Embedding effective rank
  D7. Per-codebook loss & gradient norms
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
from dataclasses import dataclass, field

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

# Mini-omni S2 hyperparameters (exact replication)
S2_CONFIG = {
    "lr_max": 4e-4,
    "lr_min": 4e-6,
    "batch_size": 2,          # per-GPU (reduced for 32GB VRAM, they used 192 on 8xA100)
    "grad_accum": 16,         # effective batch = 32
    "max_steps": 3000,        # ~70min on RTX 5090
    "warmup_steps": 200,
    "weight_decay": 0.01,     # standard AdamW
    "adam_betas": (0.9, 0.95),
    "adam_eps": 1e-8,
    "max_grad_norm": 1.0,
    "diag_every": 100,        # measure D1-D3,D5-D7 every N steps
    "basin_every": 1000,      # measure D4 (expensive) every N steps
    "val_every": 200,
}

# Qwen2-0.5B config for mini-omni
MINI_OMNI_CONFIG = Config(
    name="mini-omni",
    block_size=2048,
    vocab_size=151936,
    padded_vocab_size=181120,  # 152000 + 7*4160
    n_layer=24,
    n_head=14,
    n_query_groups=2,
    n_embd=896,
    intermediate_size=4864,
    head_size=64,
    rotary_percentage=1.0,
    parallel_residual=False,
    bias=False,
    lm_head_bias=False,
    norm_class_name="RMSNorm",
    norm_eps=1e-6,
    mlp_class_name="LLaMAMLP",
    rope_base=1000000,
    add_qkv_bias=True,
    text_vocab_size=152000,
    cat_audio_vocab_size=29120,
    audio_vocab_size=4160,
    whisper_adapter_dim=768,
    post_adapter=False,
    asr_adapter="llamamlp",
    tie_word_embeddings=False,
)


# ============================================================================
# Dataset
# ============================================================================
class OmniS2Dataset(Dataset):
    """Load preprocessed S2 data (from prepare_s2.py)."""

    def __init__(self, data_path, max_len=2048):
        self.data = torch.load(data_path, weights_only=False)
        self.max_len = max_len
        print(f"Loaded {len(self.data)} sequences from {data_path}")
        # Count task distribution
        from collections import Counter
        tasks = Counter(d['task'] for d in self.data)
        print(f"  Tasks: {dict(tasks)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d['streams'], d['loss_mask_text'], d['loss_mask_audio'], d['task']


def collate_fn(batch):
    """Pad sequences to same length within batch."""
    streams_list, lm_text_list, lm_audio_list, tasks = zip(*batch)

    max_len = max(s.shape[1] for s in streams_list)
    B = len(batch)

    # Pad all to max_len
    streams = torch.zeros(B, 8, max_len, dtype=torch.long)
    lm_text = torch.zeros(B, max_len, dtype=torch.bool)
    lm_audio = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (s, lt, la) in enumerate(zip(streams_list, lm_text_list, lm_audio_list)):
        L = s.shape[1]
        streams[i, :, :L] = s
        lm_text[i, :L] = lt
        lm_audio[i, :L] = la

    return streams, lm_text, lm_audio, tasks


# ============================================================================
# Model initialization
# ============================================================================
def load_qwen2_weights(model):
    """Initialize transformer from Qwen2-0.5B pretrained weights."""
    from transformers import AutoModelForCausalLM

    print("Loading Qwen2-0.5B pretrained weights...")
    qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B", torch_dtype=DTYPE)
    qwen_sd = qwen.state_dict()

    # Map Qwen2 → litgpt
    mapping = {}
    for layer_idx in range(24):
        qp = f"model.layers.{layer_idx}"
        lp = f"transformer.h.{layer_idx}"

        mapping[f"{qp}.self_attn.q_proj.weight"] = f"{lp}.attn.attn.weight"  # partial
        mapping[f"{qp}.self_attn.q_proj.bias"] = f"{lp}.attn.attn.bias"      # partial
        mapping[f"{qp}.mlp.gate_proj.weight"] = f"{lp}.mlp.fc_1.weight"
        mapping[f"{qp}.mlp.up_proj.weight"] = f"{lp}.mlp.fc_2.weight"
        mapping[f"{qp}.mlp.down_proj.weight"] = f"{lp}.mlp.proj.weight"
        mapping[f"{qp}.input_layernorm.weight"] = f"{lp}.norm_1.weight"
        mapping[f"{qp}.post_attention_layernorm.weight"] = f"{lp}.norm_2.weight"

    litgpt_sd = model.state_dict()

    # Handle QKV concatenation (Qwen2 has separate Q,K,V; litgpt has fused attn)
    for layer_idx in range(24):
        qp = f"model.layers.{layer_idx}.self_attn"
        lp = f"transformer.h.{layer_idx}.attn"

        q_w = qwen_sd[f"{qp}.q_proj.weight"]
        k_w = qwen_sd[f"{qp}.k_proj.weight"]
        v_w = qwen_sd[f"{qp}.v_proj.weight"]

        q_b = qwen_sd[f"{qp}.q_proj.bias"]
        k_b = qwen_sd[f"{qp}.k_proj.bias"]
        v_b = qwen_sd[f"{qp}.v_proj.bias"]

        # Qwen2-0.5B: 14 heads, 2 KV groups, head_size=64
        # Q: (896, 896), K: (128, 896), V: (128, 896)
        # litgpt fused: (896+128+128, 896) = (1152, 896)
        # But litgpt expects interleaved: [q_per_kv, k, v] × n_query_groups
        n_heads = 14
        n_kv_groups = 2
        q_per_kv = n_heads // n_kv_groups  # 7
        head_size = 64

        # Reshape Q into groups
        q_w = q_w.view(n_kv_groups, q_per_kv, head_size, -1)
        k_w = k_w.view(n_kv_groups, 1, head_size, -1)
        v_w = v_w.view(n_kv_groups, 1, head_size, -1)
        qkv_w = torch.cat([q_w, k_w, v_w], dim=1)  # (n_kv, q_per_kv+2, hs, dim)
        qkv_w = qkv_w.reshape(-1, q_w.shape[-1])    # ((q_per_kv+2)*n_kv*hs, dim)

        q_b = q_b.view(n_kv_groups, q_per_kv, head_size)
        k_b = k_b.view(n_kv_groups, 1, head_size)
        v_b = v_b.view(n_kv_groups, 1, head_size)
        qkv_b = torch.cat([q_b, k_b, v_b], dim=1).reshape(-1)

        litgpt_sd[f"{lp}.attn.weight"] = qkv_w
        litgpt_sd[f"{lp}.attn.bias"] = qkv_b

        # Output projection
        litgpt_sd[f"{lp}.proj.weight"] = qwen_sd[f"{qp}.o_proj.weight"]

        # MLP
        lp_block = f"transformer.h.{layer_idx}"
        litgpt_sd[f"{lp_block}.mlp.fc_1.weight"] = qwen_sd[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
        litgpt_sd[f"{lp_block}.mlp.fc_2.weight"] = qwen_sd[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
        litgpt_sd[f"{lp_block}.mlp.proj.weight"] = qwen_sd[f"model.layers.{layer_idx}.mlp.down_proj.weight"]

        # Norms
        litgpt_sd[f"{lp_block}.norm_1.weight"] = qwen_sd[f"model.layers.{layer_idx}.input_layernorm.weight"]
        litgpt_sd[f"{lp_block}.norm_2.weight"] = qwen_sd[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]

    # Final norm
    litgpt_sd["transformer.ln_f.weight"] = qwen_sd["model.norm.weight"]

    # Embeddings: copy text portion, randomly init audio portion
    qwen_emb = qwen_sd["model.embed_tokens.weight"]  # (151936, 896)
    full_emb = litgpt_sd["transformer.wte.weight"]    # (181120, 896)
    full_emb[:qwen_emb.shape[0]] = qwen_emb
    # Audio portion (152000:181120) stays randomly initialized
    nn.init.normal_(full_emb[152000:], std=0.02)
    litgpt_sd["transformer.wte.weight"] = full_emb

    # LM head: copy text portion
    qwen_lm = qwen_sd["lm_head.weight"]  # (151936, 896)
    full_lm = litgpt_sd["lm_head.weight"]  # (181120, 896)
    full_lm[:qwen_lm.shape[0]] = qwen_lm
    nn.init.normal_(full_lm[152000:], std=0.02)
    litgpt_sd["lm_head.weight"] = full_lm

    # Load
    model.load_state_dict(litgpt_sd, strict=False)
    print(f"Loaded Qwen2-0.5B weights (text portion), randomly initialized audio tokens")

    del qwen, qwen_sd
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Diagnostics
# ============================================================================
class DiagnosticTracker:
    """Track all 7 diagnostic signals."""

    def __init__(self, model):
        # Save initial parameters for D5 (displacement)
        self.theta_0 = {name: p.data.clone().float().cpu()
                        for name, p in model.named_parameters() if p.requires_grad}
        self.history = []

    def compute_gradient_diagnostics(self, model, val_loader, step):
        """Compute D1, D2 from separate text/audio backward passes.

        Memory-efficient: uses batch_size=1, short sequences, no retain_graph.
        Runs two separate forward-backward passes (text-only, audio-only).
        """
        model.eval()  # ensure no dropout etc

        # Find a val batch that has both text and audio loss mask active
        # (e.g., T1T2 + T1A2 pair in same batch)
        diag_streams = diag_lt = diag_la = None
        for s, lt, la, tasks_d in val_loader:
            if lt.any() and la.any():
                # Truncate seq length but keep all samples
                max_t = min(s.shape[2], 512)
                diag_streams = s[:, :, :max_t].to(DEVICE)
                diag_lt = lt[:, :max_t].to(DEVICE)
                diag_la = la[:, :max_t].to(DEVICE)
                break
        if diag_streams is None:
            return {"rho": 0, "cos_phi": 0, "layer_cos": {}, "text_grad_norm": 0, "audio_grad_norm": 0, "cb_losses": {}}

        # Pass 1: text loss backward
        model.zero_grad()
        text_loss, audio_losses = compute_losses(model, diag_streams, diag_lt, diag_la)
        if text_loss.item() > 0:
            text_loss.backward()
        text_grads = {}
        text_grad_norm_sq = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                text_grads[name] = p.grad.clone()
                text_grad_norm_sq += p.grad.norm().item()**2
        text_grad_norm = text_grad_norm_sq**0.5

        # Pass 2: audio loss backward (same batch, different loss)
        model.zero_grad()
        _, audio_losses2 = compute_losses(model, diag_streams, diag_lt, diag_la)
        audio_loss_total = sum(audio_losses2) / max(len(audio_losses2), 1)
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

        # D2: global gradient interference
        dot_prod = 0.0
        for name in text_grads:
            if name in audio_grads:
                dot_prod += (text_grads[name] * audio_grads[name]).sum().item()
        cos_phi = dot_prod / (text_grad_norm * audio_grad_norm + 1e-10)

        # D2 per-layer
        layer_cos = {}
        for layer_idx in range(24):
            prefix = f"transformer.h.{layer_idx}."
            t_norm2 = 0.0
            a_norm2 = 0.0
            dot = 0.0
            for name in text_grads:
                if name.startswith(prefix) and name in audio_grads:
                    dot += (text_grads[name] * audio_grads[name]).sum().item()
                    t_norm2 += text_grads[name].norm().item()**2
                    a_norm2 += audio_grads[name].norm().item()**2
            if t_norm2 > 0 and a_norm2 > 0:
                layer_cos[layer_idx] = dot / (t_norm2**0.5 * a_norm2**0.5 + 1e-10)

        # D7: per-codebook losses
        cb_losses = {f"cb{i}": al.item() for i, al in enumerate(audio_losses2)}

        # Clean up
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
        }

    def compute_displacement(self, model):
        """D5: parameter displacement from initialization."""
        total_disp = 0.0
        emb_disp = 0.0
        backbone_disp = 0.0
        lm_head_disp = 0.0

        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.theta_0:
                continue
            d = (p.data.float().cpu() - self.theta_0[name]).norm().item()
            total_disp += d**2
            if "wte" in name:
                emb_disp += d**2
            elif "lm_head" in name:
                lm_head_disp += d**2
            else:
                backbone_disp += d**2

        return {
            "total": total_disp**0.5,
            "embedding": emb_disp**0.5,
            "backbone": backbone_disp**0.5,
            "lm_head": lm_head_disp**0.5,
        }

    def compute_embedding_rank(self, model):
        """D6: effective rank of text vs audio embedding subspaces."""
        wte = model.transformer.wte.weight.data.float()
        text_emb = wte[:152000]
        audio_emb = wte[152000:]

        def effective_rank(W, max_rows=8192):
            # Subsample if too large
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

        return {
            "text_rank": effective_rank(text_emb),
            "audio_rank": effective_rank(audio_emb),
        }

    def probe_basin_width(self, model, val_loader, n_directions=10, epsilons=[0.01, 0.05, 0.1, 0.5, 1.0]):
        """D4: Basin width probing. Expensive — run sparingly."""
        model.eval()

        # Compute baseline text loss
        baseline_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for streams, lm_text, lm_audio, tasks in val_loader:
                if n_batches >= 5:
                    break
                streams = streams.to(DEVICE)
                lm_text = lm_text.to(DEVICE)
                loss = forward_text_loss(model, streams, lm_text)
                baseline_loss += loss.item()
                n_batches += 1
        baseline_loss /= max(n_batches, 1)

        # Probe along random directions
        results = {}
        original_params = {name: p.data.clone() for name, p in model.named_parameters()
                          if p.requires_grad}

        for eps in epsilons:
            losses = []
            for _ in range(n_directions):
                # Random perturbation
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        noise = torch.randn_like(p) * eps
                        p.data.add_(noise)

                # Measure text loss
                with torch.no_grad():
                    perturbed_loss = 0.0
                    n_b = 0
                    for streams, lm_text, lm_audio, tasks in val_loader:
                        if n_b >= 5:
                            break
                        streams = streams.to(DEVICE)
                        lm_text = lm_text.to(DEVICE)
                        loss = forward_text_loss(model, streams, lm_text)
                        perturbed_loss += loss.item()
                        n_b += 1
                    perturbed_loss /= max(n_b, 1)
                losses.append(perturbed_loss)

                # Restore
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


def forward_text_loss(model, streams, loss_mask_text):
    """Compute text-only loss for basin probing."""
    # Split streams into 8 input tensors
    input_ids = [streams[:, i, :] for i in range(8)]

    with torch.amp.autocast('cuda', dtype=DTYPE):
        xa, xt = model(audio_features=None, input_ids=input_ids)

    # Text loss on masked positions
    targets = input_ids[7][:, 1:]  # shifted text targets
    logits = xt[:, :-1]
    mask = loss_mask_text[:, 1:]

    if not mask.any():
        return torch.tensor(0.0, device=DEVICE)

    loss = F.cross_entropy(
        logits[mask].view(-1, logits.shape[-1]),
        targets[mask].view(-1),
        reduction='mean'
    )
    return loss


# ============================================================================
# Training loop
# ============================================================================
def compute_losses(model, streams, loss_mask_text, loss_mask_audio):
    """Compute text and per-codebook audio losses."""
    input_ids = [streams[:, i, :] for i in range(8)]

    with torch.amp.autocast('cuda', dtype=DTYPE):
        xa, xt = model(audio_features=None, input_ids=input_ids)

    # Text loss
    text_targets = input_ids[7][:, 1:]  # (B, T-1)
    text_logits = xt[:, :-1]            # (B, T-1, text_vocab)
    text_mask = loss_mask_text[:, 1:]    # (B, T-1)

    if text_mask.any():
        text_loss = F.cross_entropy(
            text_logits[text_mask].view(-1, text_logits.shape[-1]),
            text_targets[text_mask].view(-1),
            reduction='mean'
        )
    else:
        text_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)

    # Per-codebook audio losses
    audio_losses = []
    audio_mask = loss_mask_audio[:, 1:]

    for i in range(7):
        audio_targets = input_ids[i][:, 1:]  # (B, T-1)
        audio_logits = xa[i][:, :-1]         # (B, T-1, audio_vocab)

        if audio_mask.any():
            cb_loss = F.cross_entropy(
                audio_logits[audio_mask].view(-1, audio_logits.shape[-1]),
                # Need to un-layershift the targets for CE
                (audio_targets[audio_mask] - 152000 - i * 4160).clamp(0, 4159).view(-1),
                reduction='mean'
            )
        else:
            cb_loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
        audio_losses.append(cb_loss)

    return text_loss, audio_losses


def get_cosine_lr(step, max_steps, lr_max, lr_min, warmup_steps):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return lr_min + (lr_max - lr_min) * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def train(config=None, output_dir="results/obs_1"):
    """Main training loop with diagnostics."""
    if config is None:
        config = S2_CONFIG

    os.makedirs(output_dir, exist_ok=True)

    # Model
    print("Building model...")
    model = GPT(MINI_OMNI_CONFIG)
    load_qwen2_weights(model)

    # Freeze whisper adapter (S2: only train LLM)
    for name, p in model.named_parameters():
        if "whisper_adapter" in name:
            p.requires_grad = False

    # Gradient checkpointing to save VRAM
    model.gradient_checkpointing = True
    for block in model.transformer.h:
        block._orig_forward = block.forward
        def _ckpt_forward(self_block, *args, **kwargs):
            return torch.utils.checkpoint.checkpoint(
                self_block._orig_forward, *args, use_reentrant=False, **kwargs)
        import types
        block.forward = types.MethodType(
            lambda self, *a, **kw: torch.utils.checkpoint.checkpoint(
                self._orig_forward, *a, use_reentrant=False, **kw),
            block)

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    # Optimizer (exact mini-omni replication: AdamW)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr_max"],
        betas=config["adam_betas"],
        eps=config["adam_eps"],
        weight_decay=config["weight_decay"],
    )

    # Data
    data_dir = os.environ.get("S2_DATA_DIR", "/root/.cache/autoresearch/s2_data")
    train_dataset = OmniS2Dataset(f"{data_dir}/train.pt")
    val_dataset = OmniS2Dataset(f"{data_dir}/val.pt")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=2,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # Diagnostics
    tracker = DiagnosticTracker(model)
    diagnostics_log = []

    # Training
    print(f"\nStarting training: {config['max_steps']} steps")
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    train_iter = iter(train_loader)

    step = 0
    accum_text_loss = 0.0
    accum_audio_loss = 0.0
    accum_steps = 0
    t0 = time.time()

    while step < config["max_steps"]:
        # Get batch
        try:
            streams, lm_text, lm_audio, tasks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            streams, lm_text, lm_audio, tasks = next(train_iter)

        streams = streams.to(DEVICE)
        lm_text = lm_text.to(DEVICE)
        lm_audio = lm_audio.to(DEVICE)

        # Forward
        text_loss, audio_losses = compute_losses(model, streams, lm_text, lm_audio)
        audio_loss = sum(audio_losses) / max(len(audio_losses), 1)
        audio_weight = config.get("audio_weight", 1.0)
        total_loss = text_loss + audio_weight * audio_loss

        # Backward with gradient accumulation
        scaled_loss = total_loss / config["grad_accum"]
        scaler.scale(scaled_loss).backward()

        accum_text_loss += text_loss.item()
        accum_audio_loss += audio_loss.item()
        accum_steps += 1

        if accum_steps >= config["grad_accum"]:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["max_grad_norm"]
            )

            # Update LR
            lr = get_cosine_lr(step, config["max_steps"],
                              config["lr_max"], config["lr_min"], config["warmup_steps"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            step += 1
            avg_text = accum_text_loss / accum_steps
            avg_audio = accum_audio_loss / accum_steps
            accum_text_loss = 0.0
            accum_audio_loss = 0.0
            accum_steps = 0

            # Logging
            if step % 10 == 0:
                elapsed = time.time() - t0
                print(f"  step {step}/{config['max_steps']} | "
                      f"text={avg_text:.4f} audio={avg_audio:.4f} | "
                      f"lr={lr:.2e} | {elapsed:.0f}s")

            # Diagnostics
            if step % config["diag_every"] == 0:
                print(f"  [diag] Computing diagnostics at step {step}...")
                diag = {"step": step, "time": time.time() - t0}
                diag["text_loss"] = avg_text
                diag["audio_loss"] = avg_audio

                # D5: displacement
                diag["displacement"] = tracker.compute_displacement(model)

                # D6: embedding rank
                diag["embedding_rank"] = tracker.compute_embedding_rank(model)

                # D1, D2, D7: gradient-based (memory-efficient)
                try:
                    grad_diag = tracker.compute_gradient_diagnostics(
                        model, val_loader, step)
                    diag.update(grad_diag)
                except Exception as e:
                    print(f"  [diag] Gradient diagnostics failed: {e}")
                model.train()
                optimizer.zero_grad()  # Clear diagnostic gradients

                diagnostics_log.append(diag)

                # Save periodically
                with open(f"{output_dir}/diagnostics.json", "w") as f:
                    json.dump(diagnostics_log, f, indent=2, default=str)

            # Basin probing (expensive)
            if step % config["basin_every"] == 0:
                print(f"  [basin] Probing basin width at step {step}...")
                basin = tracker.probe_basin_width(model, val_loader)
                diagnostics_log[-1]["basin"] = basin
                with open(f"{output_dir}/diagnostics.json", "w") as f:
                    json.dump(diagnostics_log, f, indent=2, default=str)

            # Validation
            if step % config["val_every"] == 0:
                model.eval()
                val_text = 0.0
                val_audio = 0.0
                val_n = 0
                with torch.no_grad():
                    for streams_v, lm_text_v, lm_audio_v, _ in val_loader:
                        if val_n >= 20:
                            break
                        streams_v = streams_v.to(DEVICE)
                        lm_text_v = lm_text_v.to(DEVICE)
                        lm_audio_v = lm_audio_v.to(DEVICE)
                        tl, als = compute_losses(model, streams_v, lm_text_v, lm_audio_v)
                        val_text += tl.item()
                        val_audio += sum(a.item() for a in als) / len(als)
                        val_n += 1
                val_text /= max(val_n, 1)
                val_audio /= max(val_n, 1)
                print(f"  [val] step {step}: text={val_text:.4f} audio={val_audio:.4f}")
                model.train()

    # Final save
    print("\nTraining complete. Saving final diagnostics and checkpoint...")
    with open(f"{output_dir}/diagnostics.json", "w") as f:
        json.dump(diagnostics_log, f, indent=2, default=str)

    torch.save(model.state_dict(), f"{output_dir}/model_final.pt")
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/obs_1")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--audio_weight", type=float, default=1.0,
                        help="λ: weight on audio loss (0=text only)")
    args = parser.parse_args()

    config = dict(S2_CONFIG)
    config["max_steps"] = args.max_steps
    config["audio_weight"] = args.audio_weight

    train(config=config, output_dir=args.output_dir)
