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

sys.path.insert(0, "/workspace/mini-omni-ref")
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

    def __init__(self, data_path, max_len=2048, s2_mode=False):
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
# Loss computation — matches mini-omni exactly
# ============================================================================
def compute_losses(model, streams, loss_mask_text, loss_mask_audio):
    """Compute separate text CE and per-codebook audio CE.

    Mini-omni approach (from GitHub issues #101, #135):
    - Text tasks: compute text CE, audio targets masked with -100
    - Audio tasks: compute audio CE (7 heads), text targets masked with -100
    - Loss = text_loss + Σ audio_cb_losses
    """
    input_ids = [streams[:, i, :] for i in range(8)]

    with torch.amp.autocast('cuda', dtype=DTYPE):
        xa, xt = model(audio_features=None, input_ids=input_ids)

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


def forward_text_loss(model, streams, loss_mask_text):
    """Text-only loss for basin probing."""
    input_ids = [streams[:, i, :] for i in range(8)]
    with torch.amp.autocast('cuda', dtype=DTYPE):
        xa, xt = model(audio_features=None, input_ids=input_ids)
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
        diag_streams = diag_lt = diag_la = None
        for s, lt, la, tasks_d in val_loader:
            if lt.any() and la.any():
                max_t = min(s.shape[2], 512)
                diag_streams = s[:, :, :max_t].to(DEVICE)
                diag_lt = lt[:, :max_t].to(DEVICE)
                diag_la = la[:, :max_t].to(DEVICE)
                break
        if diag_streams is None:
            return {"rho": 0, "cos_phi": 0, "layer_cos": {},
                    "text_grad_norm": 0, "audio_grad_norm": 0, "cb_losses": {}}

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

        # Pass 2: audio loss backward
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
        """D6: effective rank of text vs audio embeddings."""
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

        return {
            "text_rank": effective_rank(text_emb),
            "audio_rank": effective_rank(audio_emb),
        }

    def probe_basin_width(self, model, val_loader, n_directions=10,
                           epsilons=[0.01, 0.05, 0.1, 0.5, 1.0]):
        """D4: basin width probing (expensive)."""
        model.eval()

        # Baseline text loss
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


# ============================================================================
# Model loading
# ============================================================================
def load_mini_omni_checkpoint(ckpt_dir="/workspace/mini-omni-ckpt"):
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


def build_post_s1_checkpoint(ckpt_dir="/workspace/mini-omni-ckpt"):
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
    checkpoint_mode = config.get("checkpoint_mode", "published")
    if checkpoint_mode == "post_s1":
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

    # ---- Adjust batch for memory-hungry methods ----
    method = config.get("method", "baseline")
    if method in ("grad_proj", "m_sam"):
        # 2 forward passes → need smaller batch
        config["batch_size"] = 1
        config["grad_accum"] = 32
        print(f"  [{method}] batch_size=1, grad_accum=32 (2x forward passes)")

    # ---- Data ----
    data_dir = "/root/.cache/autoresearch/s3_data"
    train_dataset = OmniS3Dataset(f"{data_dir}/train.pt", s2_mode=s2_mode)
    val_dataset = OmniS3Dataset(f"{data_dir}/val.pt")  # val always full (all 4 task types)

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
    train_iter = iter(train_loader)

    step = 0
    accum_text_loss = 0.0
    accum_audio_loss = 0.0
    accum_cb_losses = [0.0] * 7
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

        # ---- Method: gradient projection ----
        method = config.get("method", "baseline")
        cb_w = config.get("cb_weights", [1]*7)
        cb_w_sum = sum(cb_w)

        if method == "grad_proj":
            # Two separate forward+backward passes (no retain_graph, saves memory)
            audio_weight = config.get("audio_weight", 1.0)

            # Pass 1: text loss forward+backward
            model.zero_grad()
            text_loss_1, _ = compute_losses(model, streams, lm_text, lm_audio)
            (text_loss_1 / config["grad_accum"]).backward()
            text_grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    text_grads[name] = p.grad.clone()

            # Pass 2: audio loss forward+backward
            model.zero_grad()
            text_loss, audio_losses = compute_losses(model, streams, lm_text, lm_audio)
            audio_loss = sum(cb_w[i] * audio_losses[i] for i in range(7)) / cb_w_sum
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
            text_loss, audio_losses = compute_losses(model, streams, lm_text, lm_audio)
            audio_loss = sum(cb_w[i] * audio_losses[i] for i in range(7)) / cb_w_sum
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
            text_loss2, audio_losses2 = compute_losses(model, streams, lm_text, lm_audio)
            audio_loss2 = sum(cb_w[i] * audio_losses2[i] for i in range(7)) / cb_w_sum
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
            # Standard forward-backward (baseline, adaptive_lambda)
            text_loss, audio_losses = compute_losses(model, streams, lm_text, lm_audio)
            audio_loss = sum(cb_w[i] * audio_losses[i] for i in range(7)) / cb_w_sum

            # Adaptive λ: adjust based on ρ
            audio_weight = config.get("audio_weight", 1.0)
            if method == "adaptive_lambda":
                audio_weight = config.get("_current_lambda", config.get("audio_weight", 1.0))

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

            # LR schedule
            lr = get_cosine_lr(step, config["max_steps"],
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

            # Console logging
            if step % 10 == 0:
                elapsed = time.time() - t0
                cb_str = " ".join(f"{c:.2f}" for c in avg_cbs)
                print(f"  step {step}/{config['max_steps']} | "
                      f"text={avg_text:.4f} audio={avg_audio:.4f} | "
                      f"CB=[{cb_str}] | "
                      f"lr={lr:.2e} gnorm={grad_norm:.2f} | {elapsed:.0f}s")

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

                # D1, D2, D7: gradient-based
                try:
                    grad_diag = tracker.compute_gradient_diagnostics(
                        model, val_loader, step)
                    diag.update(grad_diag)
                except Exception as e:
                    print(f"  [diag] Gradient diagnostics failed: {e}")
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
                with torch.no_grad():
                    for sv, ltv, lav, _ in val_loader:
                        if val_n >= 50:
                            break
                        sv = sv.to(DEVICE)
                        ltv = ltv.to(DEVICE)
                        lav = lav.to(DEVICE)
                        tl, als = compute_losses(model, sv, ltv, lav)
                        val_text += tl.item()
                        val_audio += sum(a.item() for a in als) / 7
                        for i in range(7):
                            val_cbs[i] += als[i].item()
                        val_n += 1
                val_text /= max(val_n, 1)
                val_audio /= max(val_n, 1)
                val_cbs = [c / max(val_n, 1) for c in val_cbs]
                cb_str = " ".join(f"{c:.2f}" for c in val_cbs)
                print(f"  [val] step {step}: text={val_text:.4f} "
                      f"audio={val_audio:.4f} CB=[{cb_str}]")
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
        for sv, ltv, lav, _ in val_loader:
            if val_n >= 100:
                break
            sv = sv.to(DEVICE)
            ltv = ltv.to(DEVICE)
            lav = lav.to(DEVICE)
            tl, als = compute_losses(model, sv, ltv, lav)
            val_text += tl.item()
            val_audio += sum(a.item() for a in als) / 7
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
    parser.add_argument("--audio_weight", type=float, default=1.0)
    parser.add_argument("--lr_max", type=float, default=None)
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--method", default="baseline",
                        choices=["baseline", "grad_proj", "adaptive_lambda", "m_sam"],
                        help="Training method variant")
    parser.add_argument("--adaptive_alpha", type=float, default=0.1,
                        help="Adaptive λ step size")
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
    # Exp D: S2 mode
    parser.add_argument("--s2_mode", action="store_true",
                        help="Authentic S2: text-only training, freeze adapters, full val diagnostics")
    args = parser.parse_args()

    config = dict(S3_CONFIG)
    config["max_steps"] = args.max_steps
    config["audio_weight"] = args.audio_weight
    config["method"] = args.method
    config["adaptive_alpha"] = args.adaptive_alpha
    config["sam_rho"] = args.sam_rho
    config["checkpoint_mode"] = args.checkpoint_mode
    config["s2_mode"] = args.s2_mode
    if args.cb_weights is not None:
        config["cb_weights"] = [float(x) for x in args.cb_weights.split(",")]
        assert len(config["cb_weights"]) == 7, f"cb_weights must have 7 values, got {len(config['cb_weights'])}"
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
