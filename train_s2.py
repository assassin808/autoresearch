#!/usr/bin/env python3
"""
S2 Training: Mini-omni architecture with Qwen2-0.5B pretrained LLM backbone.

Stage 2 of mini-omni's training pipeline:
- Freeze: whisper_adapter (trained in S1, loaded from mini-omni checkpoint)
- Train: LLM backbone (initialized from Qwen2-0.5B pretrained weights)
- Embedding/lm_head: text portion from Qwen2-0.5B, audio portion randomly initialized

Architecture: 8 parallel streams (1 text + 7 SNAC audio codebooks)
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add mini-omni-ref to path for litgpt imports
sys.path.insert(0, "/workspace/mini-omni-ref")
from litgpt.config import Config
from litgpt.model import GPT


# ============================================================================
# Weight conversion: Qwen2-0.5B (HuggingFace) → litgpt format
# ============================================================================

def convert_qwen2_to_litgpt(hf_path: str, config: Config) -> dict:
    """Convert Qwen2-0.5B HuggingFace weights to litgpt format."""
    from safetensors import safe_open
    import glob

    # Load all HF weights
    hf_weights = {}
    for f in sorted(glob.glob(f"{hf_path}/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                hf_weights[k] = sf.get_tensor(k)

    litgpt_weights = {}
    n_head = config.n_head          # 14
    n_groups = config.n_query_groups  # 2
    head_size = config.head_size     # 64
    q_per_kv = n_head // n_groups    # 7

    # Per-layer conversion
    for i in range(config.n_layer):
        prefix_hf = f"model.layers.{i}"
        prefix_lg = f"transformer.h.{i}"

        # QKV: interleave into grouped format
        # litgpt expects: [group0_q(7*64), group0_k(64), group0_v(64), group1_q(7*64), ...]
        q_w = hf_weights[f"{prefix_hf}.self_attn.q_proj.weight"]  # [896, 896]
        k_w = hf_weights[f"{prefix_hf}.self_attn.k_proj.weight"]  # [128, 896]
        v_w = hf_weights[f"{prefix_hf}.self_attn.v_proj.weight"]  # [128, 896]

        q_b = hf_weights[f"{prefix_hf}.self_attn.q_proj.bias"]    # [896]
        k_b = hf_weights[f"{prefix_hf}.self_attn.k_proj.bias"]    # [128]
        v_b = hf_weights[f"{prefix_hf}.self_attn.v_proj.bias"]    # [128]

        # Reshape into groups
        q_w = q_w.view(n_groups, q_per_kv, head_size, -1)  # [2, 7, 64, 896]
        k_w = k_w.view(n_groups, 1, head_size, -1)          # [2, 1, 64, 896]
        v_w = v_w.view(n_groups, 1, head_size, -1)          # [2, 1, 64, 896]
        qkv_w = torch.cat([q_w, k_w, v_w], dim=1)           # [2, 9, 64, 896]
        qkv_w = qkv_w.reshape(-1, q_w.shape[-1])            # [1152, 896]

        q_b = q_b.view(n_groups, q_per_kv, head_size)
        k_b = k_b.view(n_groups, 1, head_size)
        v_b = v_b.view(n_groups, 1, head_size)
        qkv_b = torch.cat([q_b, k_b, v_b], dim=1).reshape(-1)  # [1152]

        litgpt_weights[f"{prefix_lg}.attn.attn.weight"] = qkv_w
        litgpt_weights[f"{prefix_lg}.attn.attn.bias"] = qkv_b

        # Output projection
        litgpt_weights[f"{prefix_lg}.attn.proj.weight"] = hf_weights[f"{prefix_hf}.self_attn.o_proj.weight"]

        # MLP
        litgpt_weights[f"{prefix_lg}.mlp.fc_1.weight"] = hf_weights[f"{prefix_hf}.mlp.gate_proj.weight"]
        litgpt_weights[f"{prefix_lg}.mlp.fc_2.weight"] = hf_weights[f"{prefix_hf}.mlp.up_proj.weight"]
        litgpt_weights[f"{prefix_lg}.mlp.proj.weight"] = hf_weights[f"{prefix_hf}.mlp.down_proj.weight"]

        # Norms
        litgpt_weights[f"{prefix_lg}.norm_1.weight"] = hf_weights[f"{prefix_hf}.input_layernorm.weight"]
        litgpt_weights[f"{prefix_lg}.norm_2.weight"] = hf_weights[f"{prefix_hf}.post_attention_layernorm.weight"]

    # Embedding: Qwen2 has 151936 tokens, mini-omni needs 181120
    # text_vocab_size=152000 (pad 64), then audio=29120
    hf_emb = hf_weights["model.embed_tokens.weight"]  # [151936, 896]
    full_emb = torch.zeros(config.padded_vocab_size, config.n_embd)
    full_emb[:hf_emb.shape[0]] = hf_emb
    # Random init for padding (151936→152000) and audio vocab (152000→181120)
    nn.init.normal_(full_emb[hf_emb.shape[0]:], mean=0.0, std=0.02)
    litgpt_weights["transformer.wte.weight"] = full_emb

    # Final layer norm
    litgpt_weights["transformer.ln_f.weight"] = hf_weights["model.norm.weight"]

    # lm_head is tied to wte in mini-omni (tie_word_embeddings=True)
    # So we don't need a separate lm_head.weight

    return litgpt_weights


def build_s2_checkpoint(qwen2_path: str, mini_omni_ckpt_path: str, config: Config) -> dict:
    """
    Build S2 checkpoint:
    - LLM backbone from Qwen2-0.5B pretrained
    - Whisper adapter from mini-omni S1-trained checkpoint
    """
    # Convert Qwen2 weights
    print("Converting Qwen2-0.5B weights to litgpt format...")
    litgpt_weights = convert_qwen2_to_litgpt(qwen2_path, config)

    # Load mini-omni checkpoint for adapter weights
    print("Loading mini-omni adapter weights...")
    mini_omni_ckpt = torch.load(mini_omni_ckpt_path, map_location="cpu", weights_only=True)

    # Extract adapter weights
    adapter_keys = [k for k in mini_omni_ckpt if k.startswith("whisper_adapter.")]
    print(f"  Found {len(adapter_keys)} adapter weight tensors")
    for k in adapter_keys:
        litgpt_weights[k] = mini_omni_ckpt[k]

    return litgpt_weights


# ============================================================================
# Dataset: Convert paired audio-text data to mini-omni's 8-stream format
# ============================================================================

def layershift(input_id, layer, stride=4160, shift=152000):
    """Map audio token to its layer-specific range in the vocabulary."""
    return input_id + shift + layer * stride


# SNAC codebook offsets used in our data encoding
SNAC_L0_OFFSET = 8192
SNAC_L1_OFFSET = 12288
SNAC_L2_OFFSET = 16384
SNAC_END_OF_AUDIO = 4097  # mini-omni's end-of-audio token


class OmniDataset(Dataset):
    """
    Dataset for mini-omni S2 training.

    Each sample produces 8 parallel token streams:
    - Stream 0 (x7 in mini-omni): text tokens
    - Streams 1-7 (x0-x6): 7 SNAC audio codebook layers

    SNAC structure per audio frame:
    - L0: 1 token  → stream 1
    - L1: 2 tokens → streams 2, 5
    - L2: 4 tokens → streams 3, 4, 6, 7

    Task types following mini-omni:
    - A1T2: audio input → text output (ASR-like)
    - A1A2: audio input → audio output (speech continuation)
    - AT: audio+text parallel (the main training task)
    """

    def __init__(self, data_path: str, max_seq_len: int = 512, task_weights: dict = None):
        print(f"Loading data from {data_path}...")
        self.data = torch.load(data_path, map_location="cpu", weights_only=False)
        self.max_seq_len = max_seq_len
        self.task_weights = task_weights or {"AT": 1.0}
        print(f"  Loaded {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def _snac_to_7streams(self, codebooks: dict) -> list:
        """
        Convert 3-level SNAC codebooks to 7 interleaved streams.

        Input codebooks (with offsets from our encoding):
          L0: N tokens (offset 8192)
          L1: 2N tokens (offset 12288)
          L2: 4N tokens (offset 16384)

        Output: 7 lists of N tokens each (raw 0-4095 range)
        """
        def to_list(x):
            return x.tolist() if hasattr(x, 'tolist') else list(x)
        l0 = [t - SNAC_L0_OFFSET for t in to_list(codebooks["L0"])]
        l1 = [t - SNAC_L1_OFFSET for t in to_list(codebooks["L1"])]
        l2 = [t - SNAC_L2_OFFSET for t in to_list(codebooks["L2"])]

        n_frames = len(l0)
        assert len(l1) == 2 * n_frames, f"L1 mismatch: {len(l1)} vs {2*n_frames}"
        assert len(l2) == 4 * n_frames, f"L2 mismatch: {len(l2)} vs {4*n_frames}"

        # 7 streams per frame: [L0, L1_0, L2_0, L2_1, L1_1, L2_2, L2_3]
        streams = [[] for _ in range(7)]
        for i in range(n_frames):
            streams[0].append(l0[i])
            streams[1].append(l1[2*i])
            streams[2].append(l2[4*i])
            streams[3].append(l2[4*i + 1])
            streams[4].append(l1[2*i + 1])
            streams[5].append(l2[4*i + 2])
            streams[6].append(l2[4*i + 3])

        return streams

    def __getitem__(self, idx):
        sample = self.data[idx]
        text_tokens = sample["text_tokens"]
        if isinstance(text_tokens, torch.Tensor):
            text_tokens = text_tokens.tolist()

        audio_streams = self._snac_to_7streams(sample["audio_codebooks"])
        n_audio_frames = len(audio_streams[0])
        n_text = len(text_tokens)

        # Build 8-stream input for AT (audio+text parallel) task
        # Format: [padding/BOS] [audio frames] [end_of_audio] [padding]
        # Text stream runs in parallel with audio, then continues after

        # Determine sequence length
        seq_len = min(max(n_audio_frames, n_text) + 2, self.max_seq_len)  # +2 for BOS/EOS margin

        # Initialize 8 streams with padding token (0)
        input_ids = torch.zeros(8, seq_len, dtype=torch.long)
        labels = torch.full((8, seq_len), -100, dtype=torch.long)  # -100 = ignore in loss

        # Stream 7 = text stream (index 7 in input, but labeled as x7 in mini-omni)
        # Streams 0-6 = audio streams

        # Fill audio streams (with layershift applied)
        audio_len = min(n_audio_frames, seq_len - 1)
        for layer in range(7):
            for t in range(audio_len):
                tok = audio_streams[layer][t]
                input_ids[layer, t] = layershift(tok, layer)
            # End of audio marker
            if audio_len < seq_len:
                input_ids[layer, audio_len] = layershift(SNAC_END_OF_AUDIO, layer)
            # Labels for audio: predict next token (shifted by 1)
            for t in range(audio_len):
                if t + 1 < audio_len:
                    labels[layer, t] = audio_streams[layer][t + 1]
                elif t + 1 == audio_len:
                    labels[layer, t] = SNAC_END_OF_AUDIO

        # Fill text stream
        text_len = min(n_text, seq_len - 1)
        for t in range(text_len):
            input_ids[7, t] = text_tokens[t]
        # Text labels: predict next token
        for t in range(text_len - 1):
            labels[7, t] = text_tokens[t + 1]

        return input_ids, labels, audio_len, text_len


def collate_fn(batch):
    """Collate with padding to max length in batch."""
    input_ids_list, labels_list, audio_lens, text_lens = zip(*batch)

    max_len = max(ids.shape[1] for ids in input_ids_list)
    batch_size = len(batch)

    input_ids = torch.zeros(batch_size, 8, max_len, dtype=torch.long)
    labels = torch.full((batch_size, 8, max_len), -100, dtype=torch.long)

    for i, (ids, lbl, _, _) in enumerate(batch):
        seq_len = ids.shape[1]
        input_ids[i, :, :seq_len] = ids
        labels[i, :, :seq_len] = lbl

    return input_ids, labels, list(audio_lens), list(text_lens)


# ============================================================================
# Training Loop
# ============================================================================

@dataclass
class TrainConfig:
    # Data
    data_path: str = "/root/.cache/autoresearch/audio/paired_train.pt"
    max_seq_len: int = 512

    # Model
    qwen2_path: str = ""  # Will be set dynamically
    mini_omni_ckpt: str = "/workspace/mini-omni-ref/checkpoint/lit_model.pth"
    mini_omni_config: str = "/workspace/mini-omni-ref/checkpoint/model_config.yaml"

    # Training
    batch_size: int = 4
    grad_accum_steps: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    max_steps: int = 5000

    # Loss weights
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    save_dir: str = "/workspace/autoresearch/checkpoints/s2"

    # Device
    device: str = "cuda"
    dtype: str = "bfloat16"


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine schedule with warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(cfg: TrainConfig):
    device = torch.device(cfg.device)
    dtype = getattr(torch, cfg.dtype)

    # Load config
    config = Config.from_file(cfg.mini_omni_config)
    print(f"Model config: n_layer={config.n_layer}, n_embd={config.n_embd}, "
          f"n_head={config.n_head}, n_query_groups={config.n_query_groups}")
    print(f"Vocab: text={config.text_vocab_size}, audio={config.audio_vocab_size}×7, "
          f"total={config.padded_vocab_size}")

    # Build checkpoint
    s2_weights = build_s2_checkpoint(cfg.qwen2_path, cfg.mini_omni_ckpt, config)

    # Create model
    print("Creating model...")
    model = GPT(config)

    # Load weights (strict=False because lm_head is tied)
    missing, unexpected = model.load_state_dict(s2_weights, strict=False)
    print(f"  Missing keys: {missing}")
    print(f"  Unexpected keys: {unexpected}")

    # Freeze whisper adapter (S2: only train LLM backbone)
    adapter_params = 0
    for name, param in model.named_parameters():
        if "whisper_adapter" in name:
            param.requires_grad = False
            adapter_params += param.numel()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,}")
    print(f"  Adapter params (frozen): {adapter_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    model = model.to(device=device, dtype=dtype)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    # Dataset
    dataset = OmniDataset(cfg.data_path, max_seq_len=cfg.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
        drop_last=True,
    )

    # ================================================================
    # Dynamics tracking setup
    # ================================================================

    # Group parameters for per-component tracking
    param_groups_named = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "transformer.wte" in name or "lm_head" in name:
            group = "embedding"
        elif "transformer.ln_f" in name:
            group = "final_norm"
        elif ".attn." in name:
            group = "attention"
        elif ".mlp." in name:
            group = "mlp"
        elif ".norm_" in name:
            group = "layer_norm"
        else:
            group = "other"
        if group not in param_groups_named:
            param_groups_named[group] = []
        param_groups_named[group].append((name, p))

    # Save initial param norms for tracking drift
    init_param_norms = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            init_param_norms[name] = p.detach().float().norm().item()

    # Dynamics log — will be saved as JSON
    dynamics_log = []

    def compute_dynamics(model, step_num, text_loss_val, audio_loss_val, audio_layer_losses, lr_val):
        """Compute comprehensive training dynamics at this step."""
        record = {
            "step": step_num,
            "lr": lr_val,
            "text_loss": text_loss_val,
            "audio_loss": audio_loss_val,
            "total_loss": text_loss_val + audio_loss_val,
        }

        # Per-audio-layer losses
        for i, al in enumerate(audio_layer_losses):
            record[f"audio_loss_L{i}"] = al

        # Gradient norms per component
        for group_name, params in param_groups_named.items():
            grad_norms = []
            param_norms = []
            update_ratios = []  # |grad| / |param| — how much the update changes the param
            for pname, p in params:
                if p.grad is not None:
                    gn = p.grad.detach().float().norm().item()
                    pn = p.detach().float().norm().item()
                    grad_norms.append(gn)
                    param_norms.append(pn)
                    if pn > 0:
                        update_ratios.append(gn / pn)
            if grad_norms:
                record[f"grad_norm_{group_name}"] = sum(g**2 for g in grad_norms) ** 0.5
                record[f"param_norm_{group_name}"] = sum(p**2 for p in param_norms) ** 0.5
                record[f"update_ratio_{group_name}"] = sum(update_ratios) / len(update_ratios)

        # Overall gradient norm
        all_grads = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                all_grads.append(p.grad.detach().float().norm().item() ** 2)
        record["grad_norm_total"] = sum(all_grads) ** 0.5

        # Per-layer gradient norms (transformer blocks)
        for layer_idx in range(model.config.n_layer):
            layer_grad_sq = 0.0
            prefix = f"transformer.h.{layer_idx}."
            for pname, p in model.named_parameters():
                if pname.startswith(prefix) and p.grad is not None:
                    layer_grad_sq += p.grad.detach().float().norm().item() ** 2
            record[f"grad_norm_layer{layer_idx}"] = layer_grad_sq ** 0.5

        # Embedding drift from init (how far has wte moved?)
        wte = model.transformer.wte.weight
        wte_norm = wte.detach().float().norm().item()
        record["wte_norm"] = wte_norm
        # Text vs audio embedding norms
        text_emb_norm = wte[:model.config.text_vocab_size].detach().float().norm().item()
        audio_emb_norm = wte[model.config.text_vocab_size:].detach().float().norm().item()
        record["text_emb_norm"] = text_emb_norm
        record["audio_emb_norm"] = audio_emb_norm

        # Text/audio gradient on embeddings
        if wte.grad is not None:
            text_emb_grad = wte.grad[:model.config.text_vocab_size].detach().float().norm().item()
            audio_emb_grad = wte.grad[model.config.text_vocab_size:].detach().float().norm().item()
            record["text_emb_grad_norm"] = text_emb_grad
            record["audio_emb_grad_norm"] = audio_emb_grad

        # Token prediction accuracy (top-1)
        # Stored from the forward pass via closure
        record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

        return record

    # ================================================================
    # Training loop
    # ================================================================
    os.makedirs(cfg.save_dir, exist_ok=True)
    model.train()

    step = 0
    total_loss_acc = 0.0
    text_loss_acc = 0.0
    audio_loss_acc = 0.0
    audio_layer_loss_acc = [0.0] * 7
    text_correct_acc = 0
    text_total_acc = 0
    audio_correct_acc = 0
    audio_total_acc = 0
    data_iter = iter(dataloader)
    epoch = 0

    t0 = time.time()
    dynamics_path = os.path.join(cfg.save_dir, "dynamics.json")

    print(f"\nDynamics will be saved to: {dynamics_path}")
    print(f"Tracking: losses, grad norms (total + per-component + per-layer), "
          f"param norms, update ratios, embedding drift, accuracy\n")

    while step < cfg.max_steps:
        # Get batch
        try:
            input_ids, labels, audio_lens, text_lens = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(dataloader)
            input_ids, labels, audio_lens, text_lens = next(data_iter)

        input_ids = input_ids.to(device)  # [B, 8, T]
        labels = labels.to(device)        # [B, 8, T]

        B, _, T = input_ids.shape

        # Prepare 8-stream input: list of 8 tensors, each [B, T]
        streams = [input_ids[:, i, :] for i in range(8)]

        # Forward pass
        with torch.autocast(device_type="cuda", dtype=dtype):
            xa_list, xt = model(
                audio_features=None,
                input_ids=streams,
                task=["AT"] * B,
            )

            # Text loss
            text_labels = labels[:, 7, :]  # [B, T]
            text_loss = F.cross_entropy(
                xt.reshape(-1, xt.size(-1)),
                text_labels.reshape(-1),
                ignore_index=-100,
            )

            # Audio loss per layer
            audio_layer_losses_val = []
            audio_loss = 0.0
            for layer in range(7):
                audio_labels = labels[:, layer, :]
                layer_logits = xa_list[layer]
                ll = F.cross_entropy(
                    layer_logits.reshape(-1, layer_logits.size(-1)),
                    audio_labels.reshape(-1),
                    ignore_index=-100,
                )
                audio_loss += ll
                audio_layer_losses_val.append(ll.item())
            audio_loss = audio_loss / 7.0

            loss = cfg.text_loss_weight * text_loss + cfg.audio_loss_weight * audio_loss
            loss = loss / cfg.grad_accum_steps

        # Track accuracy
        with torch.no_grad():
            # Text accuracy
            text_mask = text_labels.reshape(-1) != -100
            if text_mask.any():
                text_preds = xt.reshape(-1, xt.size(-1))[text_mask].argmax(dim=-1)
                text_targets = text_labels.reshape(-1)[text_mask]
                text_correct_acc += (text_preds == text_targets).sum().item()
                text_total_acc += text_mask.sum().item()
            # Audio accuracy (average over layers)
            for layer in range(7):
                audio_labels_l = labels[:, layer, :].reshape(-1)
                audio_mask = audio_labels_l != -100
                if audio_mask.any():
                    audio_preds = xa_list[layer].reshape(-1, xa_list[layer].size(-1))[audio_mask].argmax(dim=-1)
                    audio_targets = audio_labels_l[audio_mask]
                    audio_correct_acc += (audio_preds == audio_targets).sum().item()
                    audio_total_acc += audio_mask.sum().item()

        loss.backward()

        total_loss_acc += loss.item() * cfg.grad_accum_steps
        text_loss_acc += text_loss.item()
        audio_loss_acc += audio_loss.item()
        for i in range(7):
            audio_layer_loss_acc[i] += audio_layer_losses_val[i]

        # Gradient accumulation step
        if (step + 1) % cfg.grad_accum_steps == 0:
            opt_step = (step + 1) // cfg.grad_accum_steps

            # Compute dynamics BEFORE clipping (to see raw gradient magnitudes)
            if step % cfg.log_interval == 0 or step < 50:
                lr_now = get_lr(opt_step, cfg)
                record = compute_dynamics(
                    model, step,
                    text_loss_acc / max(1, min(cfg.log_interval, step + 1)),
                    audio_loss_acc / max(1, min(cfg.log_interval, step + 1)),
                    [al / max(1, min(cfg.log_interval, step + 1)) for al in audio_layer_loss_acc],
                    lr_now,
                )
                # Add accuracy
                record["text_accuracy"] = text_correct_acc / max(1, text_total_acc)
                record["audio_accuracy"] = audio_correct_acc / max(1, audio_total_acc)
                record["epoch"] = epoch
                record["elapsed_s"] = time.time() - t0
                dynamics_log.append(record)

            # Gradient clipping — record pre-clip norm
            pre_clip_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
            if dynamics_log and dynamics_log[-1]["step"] == step:
                dynamics_log[-1]["pre_clip_grad_norm"] = pre_clip_norm
                dynamics_log[-1]["grad_clipped"] = pre_clip_norm > 1.0

            # LR schedule
            lr = get_lr(opt_step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            optimizer.zero_grad()

        step += 1

        # Logging
        if step % cfg.log_interval == 0:
            avg_loss = total_loss_acc / cfg.log_interval
            avg_text = text_loss_acc / cfg.log_interval
            avg_audio = audio_loss_acc / cfg.log_interval
            text_acc = text_correct_acc / max(1, text_total_acc)
            audio_acc = audio_correct_acc / max(1, audio_total_acc)
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed

            lr = get_lr(step // cfg.grad_accum_steps, cfg)
            print(f"step {step}/{cfg.max_steps} | loss={avg_loss:.4f} | "
                  f"text={avg_text:.4f} | audio={avg_audio:.4f} | "
                  f"t_acc={text_acc:.3f} | a_acc={audio_acc:.4f} | "
                  f"lr={lr:.2e} | ep={epoch} | {steps_per_sec:.1f} st/s")

            total_loss_acc = 0.0
            text_loss_acc = 0.0
            audio_loss_acc = 0.0
            audio_layer_loss_acc = [0.0] * 7
            text_correct_acc = 0
            text_total_acc = 0
            audio_correct_acc = 0
            audio_total_acc = 0

            # Save dynamics periodically
            with open(dynamics_path, "w") as f:
                json.dump(dynamics_log, f)

        # Save checkpoint
        if step % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"s2_step{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Final save
    ckpt_path = os.path.join(cfg.save_dir, "s2_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
    }, ckpt_path)

    # Save final dynamics
    with open(dynamics_path, "w") as f:
        json.dump(dynamics_log, f)
    print(f"\nTraining complete. Final checkpoint: {ckpt_path}")
    print(f"Dynamics log: {dynamics_path} ({len(dynamics_log)} records)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen2_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--text_loss_weight", type=float, default=1.0)
    parser.add_argument("--audio_loss_weight", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="/workspace/autoresearch/checkpoints/s2")
    args = parser.parse_args()

    # Find Qwen2-0.5B path
    if not args.qwen2_path:
        cache_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B"
        snapshots = list(Path(cache_dir).glob("snapshots/*"))
        if snapshots:
            args.qwen2_path = str(snapshots[0])
        else:
            raise FileNotFoundError("Qwen2-0.5B not found. Run: huggingface_hub.snapshot_download('Qwen/Qwen2-0.5B')")

    cfg = TrainConfig(
        qwen2_path=args.qwen2_path,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        max_seq_len=args.max_seq_len,
        text_loss_weight=args.text_loss_weight,
        audio_loss_weight=args.audio_loss_weight,
        save_dir=args.save_dir,
    )

    print("=" * 60)
    print("S2 Training: Qwen2-0.5B backbone + mini-omni adapter")
    print("=" * 60)
    print(f"  Data: {cfg.data_path}")
    print(f"  Qwen2 weights: {cfg.qwen2_path}")
    print(f"  Adapter from: {cfg.mini_omni_ckpt}")
    print(f"  Batch size: {cfg.batch_size} × {cfg.grad_accum_steps} accum = {cfg.batch_size * cfg.grad_accum_steps} effective")
    print(f"  LR: {cfg.lr}, WD: {cfg.weight_decay}")
    print(f"  Steps: {cfg.max_steps}")
    print(f"  Loss weights: text={cfg.text_loss_weight}, audio={cfg.audio_loss_weight}")
    print()

    train(cfg)
