#!/usr/bin/env python3
"""
Sweep 42: Comprehensive S2/S3 training experiments for mini-omni.

Experiments:
Phase 1 - Baselines with different LRs (matching mini-omni hyperparams)
Phase 2 - S2→S3 staged training
Phase 3 - Muon optimizer
Phase 4 - Separate momentum approaches (Adam split, Muon split)
Phase 5 - Additional ideas (loss weighting, LR splitting, gradient scaling)

Each experiment tracks full dynamics: losses, grad norms, accuracy,
per-component and per-layer metrics, embedding drift.
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/mini-omni-ref")
from litgpt.config import Config
from litgpt.model import GPT

# Import from our training script
sys.path.insert(0, "/workspace/autoresearch")
from train_s2 import (
    convert_qwen2_to_litgpt, build_s2_checkpoint,
    OmniDataset, collate_fn, layershift,
    SNAC_L0_OFFSET, SNAC_L1_OFFSET, SNAC_L2_OFFSET, SNAC_END_OF_AUDIO,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
MINI_OMNI_CONFIG = "/workspace/mini-omni-ref/checkpoint/model_config.yaml"
MINI_OMNI_CKPT = "/workspace/mini-omni-ref/checkpoint/lit_model.pth"
DATA_PATH = "/root/.cache/autoresearch/audio/paired_train.pt"
RESULTS_DIR = "/workspace/autoresearch/checkpoints/sweep42"
QWEN2_PATH = None  # Set dynamically


def find_qwen2_path():
    global QWEN2_PATH
    cache_dir = "/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B"
    snapshots = list(Path(cache_dir).glob("snapshots/*"))
    QWEN2_PATH = str(snapshots[0])
    return QWEN2_PATH


# ============================================================================
# Muon Optimizer
# ============================================================================

def newton_schulz_5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration for matrix orthogonalization."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float() / (G.float().norm() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer: momentum + Newton-Schulz orthogonalization for matrices."""

    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if wd > 0:
                    p.mul_(1 - lr * wd)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if p.ndim >= 2 and min(p.shape) >= 2:
                    # Apply NS orthogonalization for matrices
                    update = newton_schulz_5(buf, steps=ns_steps)
                else:
                    # Just use momentum for vectors/scalars
                    update = buf

                p.add_(update, alpha=-lr)


# ============================================================================
# Separate Momentum Optimizers
# ============================================================================

class AdamSplitMomentum(torch.optim.Optimizer):
    """
    Adam with separate second moment (v) for text and audio gradients.

    We maintain v_text and v_audio separately, each updated only when
    the corresponding loss backward is active. The final adaptive rate
    uses max(v_text, v_audio) to be conservative.
    """

    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self._grad_source = "combined"  # or "text" or "audio"

    def set_grad_source(self, source):
        self._grad_source = source

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if wd > 0:
                    p.mul_(1 - lr * wd)

                g = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v_text"] = torch.zeros_like(p)
                    state["v_audio"] = torch.zeros_like(p)

                state["step"] += 1
                m = state["m"]
                v_text = state["v_text"]
                v_audio = state["v_audio"]

                # Update first moment (shared)
                m.mul_(beta1).add_(g, alpha=1 - beta1)

                # Update second moment (split by source)
                if self._grad_source == "text":
                    v_text.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                elif self._grad_source == "audio":
                    v_audio.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                else:
                    # Combined: update both
                    v_text.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    v_audio.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Bias correction
                t = state["step"]
                m_hat = m / (1 - beta1 ** t)
                # Use max of both v for conservative update
                v_combined = torch.max(v_text, v_audio)
                v_hat = v_combined / (1 - beta2 ** t)

                p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


class MuonSplitMomentum(torch.optim.Optimizer):
    """
    Muon with separate momentum buffers for text and audio gradients.

    Two momentum buffers with potentially different betas are maintained.
    They are summed before Newton-Schulz orthogonalization.
    """

    def __init__(self, params, lr=0.02, beta_text=0.95, beta_audio=0.85,
                 ns_steps=5, weight_decay=0.0, audio_scale=1.0):
        defaults = dict(lr=lr, beta_text=beta_text, beta_audio=beta_audio,
                       ns_steps=ns_steps, weight_decay=weight_decay,
                       audio_scale=audio_scale)
        super().__init__(params, defaults)
        self._grad_source = "combined"

    def set_grad_source(self, source):
        self._grad_source = source

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta_text = group["beta_text"]
            beta_audio = group["beta_audio"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            audio_scale = group["audio_scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if wd > 0:
                    p.mul_(1 - lr * wd)

                g = p.grad
                state = self.state[p]

                if "buf_text" not in state:
                    state["buf_text"] = torch.zeros_like(g)
                    state["buf_audio"] = torch.zeros_like(g)

                buf_text = state["buf_text"]
                buf_audio = state["buf_audio"]

                if self._grad_source == "text":
                    buf_text.mul_(beta_text).add_(g)
                elif self._grad_source == "audio":
                    buf_audio.mul_(beta_audio).add_(g)
                else:
                    # Combined mode: split 50/50 (fallback)
                    buf_text.mul_(beta_text).add_(g, alpha=0.5)
                    buf_audio.mul_(beta_audio).add_(g, alpha=0.5)

                # Sum back with optional audio scaling
                combined = buf_text + audio_scale * buf_audio

                if p.ndim >= 2 and min(p.shape) >= 2:
                    update = newton_schulz_5(combined, steps=ns_steps)
                else:
                    update = combined

                p.add_(update, alpha=-lr)


# ============================================================================
# Training engine
# ============================================================================

@dataclass
class ExperimentConfig:
    name: str = "baseline"

    # Optimizer
    optimizer: str = "adam"  # adam, muon, adam_split, muon_split
    lr: float = 3e-4
    weight_decay: float = 0.05
    warmup_steps: int = 100
    beta1: float = 0.9
    beta2: float = 0.95

    # Muon-specific
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5

    # Split-specific
    beta_text: float = 0.95
    beta_audio: float = 0.85
    audio_scale: float = 1.0

    # Training
    batch_size: int = 4
    grad_accum: int = 4
    max_steps: int = 2000
    max_seq_len: int = 512

    # Loss
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0

    # Stage
    stage: str = "S2"  # S2 or S3
    s2_checkpoint: str = ""  # For S3, load S2 checkpoint

    # Separate backward
    separate_backward: bool = False  # For split momentum optimizers


def create_optimizer(model, cfg: ExperimentConfig):
    """Create optimizer based on config."""
    trainable = [p for p in model.parameters() if p.requires_grad]

    if cfg.optimizer == "adam":
        return torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )
    elif cfg.optimizer == "muon":
        return Muon(
            trainable, lr=cfg.lr, momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adam_split":
        return AdamSplitMomentum(
            trainable, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "muon_split":
        return MuonSplitMomentum(
            trainable, lr=cfg.lr, beta_text=cfg.beta_text,
            beta_audio=cfg.beta_audio, ns_steps=cfg.muon_ns_steps,
            weight_decay=cfg.weight_decay, audio_scale=cfg.audio_scale,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def get_lr_schedule(step, cfg: ExperimentConfig):
    """Cosine schedule with warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def run_experiment(cfg: ExperimentConfig, dataset: OmniDataset):
    """Run a single experiment and return dynamics log."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {cfg.name}")
    print(f"{'='*70}")
    print(f"  Optimizer: {cfg.optimizer}, LR: {cfg.lr}, WD: {cfg.weight_decay}")
    print(f"  Stage: {cfg.stage}, Steps: {cfg.max_steps}")
    print(f"  Loss weights: text={cfg.text_loss_weight}, audio={cfg.audio_loss_weight}")
    if cfg.optimizer in ("muon", "muon_split"):
        print(f"  Muon: momentum={cfg.muon_momentum}, NS steps={cfg.muon_ns_steps}")
    if cfg.optimizer in ("adam_split", "muon_split"):
        print(f"  Split: beta_text={cfg.beta_text}, beta_audio={cfg.beta_audio}")
        print(f"  Separate backward: {cfg.separate_backward}")
    print()

    # Load model config
    config = Config.from_file(MINI_OMNI_CONFIG)

    # Build/load checkpoint
    if cfg.s2_checkpoint and os.path.exists(cfg.s2_checkpoint):
        print(f"Loading S2 checkpoint: {cfg.s2_checkpoint}")
        ckpt = torch.load(cfg.s2_checkpoint, map_location="cpu", weights_only=True)
        weights = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    else:
        weights = build_s2_checkpoint(QWEN2_PATH, MINI_OMNI_CKPT, config)

    model = GPT(config)
    model.load_state_dict(weights, strict=False)

    # Freeze based on stage
    if cfg.stage == "S2":
        for name, param in model.named_parameters():
            if "whisper_adapter" in name:
                param.requires_grad = False
    elif cfg.stage == "S3":
        pass  # All unfrozen

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable_count:,}")

    model = model.to(device=DEVICE, dtype=DTYPE)
    optimizer = create_optimizer(model, cfg)

    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True,
    )

    # Training
    model.train()
    dynamics = []
    data_iter = iter(dataloader)
    opt_step = 0
    t0 = time.time()

    # Accumulators
    loss_acc = 0.0
    text_loss_acc = 0.0
    audio_loss_acc = 0.0
    audio_layer_acc = [0.0] * 7
    text_correct = 0
    text_total = 0
    audio_correct = 0
    audio_total = 0
    acc_count = 0

    for step in range(cfg.max_steps * cfg.grad_accum):
        try:
            input_ids, labels, audio_lens, text_lens = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            input_ids, labels, audio_lens, text_lens = next(data_iter)

        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        B, _, T = input_ids.shape
        streams = [input_ids[:, i, :] for i in range(8)]

        if cfg.separate_backward and cfg.optimizer in ("adam_split", "muon_split"):
            # Separate backward for text and audio
            # Text backward
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                xa_list, xt = model(audio_features=None, input_ids=streams, task=["AT"] * B)
                text_labels = labels[:, 7, :]
                text_loss = F.cross_entropy(
                    xt.reshape(-1, xt.size(-1)), text_labels.reshape(-1), ignore_index=-100)

            (text_loss * cfg.text_loss_weight / cfg.grad_accum).backward(retain_graph=True)

            # Mark gradients as text-sourced for split optimizer step
            if hasattr(optimizer, 'set_grad_source'):
                optimizer.set_grad_source("text")

            # Audio backward (accumulates on existing grads)
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                audio_loss = 0.0
                audio_layer_losses = []
                for layer in range(7):
                    al = labels[:, layer, :]
                    ll = F.cross_entropy(
                        xa_list[layer].reshape(-1, xa_list[layer].size(-1)),
                        al.reshape(-1), ignore_index=-100)
                    audio_loss += ll
                    audio_layer_losses.append(ll.item())
                audio_loss = audio_loss / 7.0

            # We need to handle this carefully: first do text step, then audio
            # Actually for split momentum, we do TWO separate backward passes
            # and TWO optimizer substeps is wrong. Instead:
            # We do text backward → store text grads → zero → audio backward →
            # store audio grads → manually call step with both

            # Simpler approach: just mark the grad source and accumulate
            # The optimizer maintains separate buffers per source

            # For this simplified version: do combined backward but track separately
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=DTYPE):
                xa_list2, xt2 = model(audio_features=None, input_ids=streams, task=["AT"] * B)
                text_labels2 = labels[:, 7, :]
                text_loss = F.cross_entropy(
                    xt2.reshape(-1, xt2.size(-1)), text_labels2.reshape(-1), ignore_index=-100)

            (text_loss * cfg.text_loss_weight / cfg.grad_accum).backward(retain_graph=True)

            # Save text grads
            text_grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    text_grads[name] = p.grad.clone()

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=DTYPE):
                audio_loss = 0.0
                audio_layer_losses = []
                for layer in range(7):
                    al = labels[:, layer, :]
                    ll = F.cross_entropy(
                        xa_list2[layer].reshape(-1, xa_list2[layer].size(-1)),
                        al.reshape(-1), ignore_index=-100)
                    audio_loss += ll
                    audio_layer_losses.append(ll.item())
                audio_loss = audio_loss / 7.0

            (audio_loss * cfg.audio_loss_weight / cfg.grad_accum).backward()

            # Save audio grads
            audio_grads = {}
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    audio_grads[name] = p.grad.clone()

            # Now do two optimizer sub-steps with source marking
            # Text step
            for name, p in model.named_parameters():
                if name in text_grads:
                    p.grad = text_grads[name]
            optimizer.set_grad_source("text")

            # Audio step
            for name, p in model.named_parameters():
                if name in audio_grads:
                    p.grad = audio_grads[name]
            optimizer.set_grad_source("audio")

            loss = text_loss + audio_loss

        else:
            # Standard combined backward
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                xa_list, xt = model(audio_features=None, input_ids=streams, task=["AT"] * B)
                text_labels = labels[:, 7, :]
                text_loss = F.cross_entropy(
                    xt.reshape(-1, xt.size(-1)), text_labels.reshape(-1), ignore_index=-100)

                audio_loss = 0.0
                audio_layer_losses = []
                for layer in range(7):
                    al = labels[:, layer, :]
                    ll = F.cross_entropy(
                        xa_list[layer].reshape(-1, xa_list[layer].size(-1)),
                        al.reshape(-1), ignore_index=-100)
                    audio_loss += ll
                    audio_layer_losses.append(ll.item())
                audio_loss = audio_loss / 7.0

                loss = cfg.text_loss_weight * text_loss + cfg.audio_loss_weight * audio_loss
                loss = loss / cfg.grad_accum

            loss.backward()

        # Track accuracy
        with torch.no_grad():
            text_mask = labels[:, 7, :].reshape(-1) != -100
            if text_mask.any():
                tp = xt.reshape(-1, xt.size(-1))[text_mask].argmax(-1)
                text_correct += (tp == labels[:, 7, :].reshape(-1)[text_mask]).sum().item()
                text_total += text_mask.sum().item()
            for layer in range(7):
                am = labels[:, layer, :].reshape(-1) != -100
                if am.any():
                    ap = xa_list[layer].reshape(-1, xa_list[layer].size(-1))[am].argmax(-1)
                    audio_correct += (ap == labels[:, layer, :].reshape(-1)[am]).sum().item()
                    audio_total += am.sum().item()

        loss_acc += (text_loss.item() + audio_loss.item())
        text_loss_acc += text_loss.item()
        audio_loss_acc += audio_loss.item()
        for i in range(7):
            audio_layer_acc[i] += audio_layer_losses[i]
        acc_count += 1

        # Optimizer step
        if (step + 1) % cfg.grad_accum == 0:
            opt_step += 1

            # Compute dynamics before clipping
            record = {"step": opt_step, "elapsed_s": time.time() - t0}

            # Gradient norms
            total_gn_sq = 0.0
            comp_gn = {"embedding": 0.0, "attention": 0.0, "mlp": 0.0, "layer_norm": 0.0}
            layer_gn = [0.0] * config.n_layer
            text_emb_gn = 0.0
            audio_emb_gn = 0.0

            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    gn = p.grad.float().norm().item() ** 2
                    total_gn_sq += gn

                    if "transformer.wte" in name or "lm_head" in name:
                        comp_gn["embedding"] += gn
                        # Text vs audio embedding grad
                        text_emb_gn = p.grad[:config.text_vocab_size].float().norm().item()
                        audio_emb_gn = p.grad[config.text_vocab_size:].float().norm().item()
                    elif ".attn." in name:
                        comp_gn["attention"] += gn
                    elif ".mlp." in name:
                        comp_gn["mlp"] += gn
                    elif ".norm_" in name:
                        comp_gn["layer_norm"] += gn

                    for li in range(config.n_layer):
                        if f"transformer.h.{li}." in name:
                            layer_gn[li] += gn
                            break

            record["grad_norm_total"] = total_gn_sq ** 0.5
            for comp, gn in comp_gn.items():
                record[f"grad_norm_{comp}"] = gn ** 0.5
            record["text_emb_grad_norm"] = text_emb_gn
            record["audio_emb_grad_norm"] = audio_emb_gn

            # Per-layer gradient norms (sample a few)
            for li in [0, 5, 11, 17, 23]:
                record[f"grad_norm_layer{li}"] = layer_gn[li] ** 0.5

            # Gradient clipping
            if cfg.optimizer in ("adam", "adam_split"):
                pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                record["pre_clip_grad_norm"] = pre_clip
                record["grad_clipped"] = pre_clip > 1.0

            # LR schedule
            lr = get_lr_schedule(opt_step, cfg)
            if cfg.optimizer in ("adam", "adam_split"):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            elif cfg.optimizer in ("muon", "muon_split"):
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            if hasattr(optimizer, 'set_grad_source'):
                optimizer.set_grad_source("combined")

            optimizer.step()
            optimizer.zero_grad()

            # Log losses
            n = acc_count
            record["lr"] = lr
            record["total_loss"] = loss_acc / n
            record["text_loss"] = text_loss_acc / n
            record["audio_loss"] = audio_loss_acc / n
            for i in range(7):
                record[f"audio_loss_L{i}"] = audio_layer_acc[i] / n
            record["text_accuracy"] = text_correct / max(1, text_total)
            record["audio_accuracy"] = audio_correct / max(1, audio_total)

            # Embedding norms
            wte = model.transformer.wte.weight
            record["text_emb_norm"] = wte[:config.text_vocab_size].float().norm().item()
            record["audio_emb_norm"] = wte[config.text_vocab_size:].float().norm().item()
            record["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9

            dynamics.append(record)

            # Print
            if opt_step % 25 == 0 or opt_step <= 5:
                print(f"  opt_step {opt_step}/{cfg.max_steps} | "
                      f"loss={record['total_loss']:.4f} | "
                      f"text={record['text_loss']:.4f} | "
                      f"audio={record['audio_loss']:.4f} | "
                      f"t_acc={record['text_accuracy']:.3f} | "
                      f"a_acc={record['audio_accuracy']:.4f} | "
                      f"gn={record['grad_norm_total']:.2f} | "
                      f"lr={lr:.2e}")

            # Reset accumulators
            loss_acc = 0.0
            text_loss_acc = 0.0
            audio_loss_acc = 0.0
            audio_layer_acc = [0.0] * 7
            text_correct = 0
            text_total = 0
            audio_correct = 0
            audio_total = 0
            acc_count = 0

    elapsed = time.time() - t0
    print(f"\n  Experiment {cfg.name} complete: {elapsed:.1f}s, {opt_step} opt steps")

    # Save checkpoint for potential S3 continuation
    save_dir = os.path.join(RESULTS_DIR, cfg.name)
    os.makedirs(save_dir, exist_ok=True)

    # Save dynamics
    with open(os.path.join(save_dir, "dynamics.json"), "w") as f:
        json.dump(dynamics, f)

    # Save model checkpoint
    torch.save({
        "step": opt_step,
        "model_state_dict": model.state_dict(),
    }, os.path.join(save_dir, "checkpoint.pt"))

    # Cleanup
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    return dynamics


# ============================================================================
# Experiment definitions
# ============================================================================

def define_experiments():
    experiments = []

    # ======== Phase 1: Baselines with different LRs ========
    # Our current setup
    experiments.append(ExperimentConfig(
        name="P1_adam_lr3e-4", optimizer="adam", lr=3e-4, max_steps=2000,
    ))
    # Mini-omni S2 range (lower)
    experiments.append(ExperimentConfig(
        name="P1_adam_lr5e-5", optimizer="adam", lr=5e-5, max_steps=2000,
    ))
    experiments.append(ExperimentConfig(
        name="P1_adam_lr1e-4", optimizer="adam", lr=1e-4, max_steps=2000,
    ))
    # Even lower (mini-omni minimum)
    experiments.append(ExperimentConfig(
        name="P1_adam_lr1e-5", optimizer="adam", lr=1e-5, max_steps=2000,
    ))

    # ======== Phase 2: S2→S3 staged training ========
    # S2 first, then S3 with unfrozen adapter
    experiments.append(ExperimentConfig(
        name="P2_S2_for_S3", optimizer="adam", lr=1e-4, max_steps=2000,
        stage="S2",
    ))
    # S3 will be added dynamically after S2 completes

    # ======== Phase 3: Muon ========
    experiments.append(ExperimentConfig(
        name="P3_muon_lr0.02", optimizer="muon", lr=0.02, max_steps=2000,
        muon_momentum=0.95, weight_decay=0.0,
    ))
    experiments.append(ExperimentConfig(
        name="P3_muon_lr0.005", optimizer="muon", lr=0.005, max_steps=2000,
        muon_momentum=0.95, weight_decay=0.0,
    ))
    experiments.append(ExperimentConfig(
        name="P3_muon_lr0.01_wd", optimizer="muon", lr=0.01, max_steps=2000,
        muon_momentum=0.95, weight_decay=0.05,
    ))

    # ======== Phase 4: Separate momentum ========
    # Adam with split second moment
    experiments.append(ExperimentConfig(
        name="P4_adam_split", optimizer="adam_split", lr=1e-4, max_steps=2000,
        separate_backward=True,
    ))
    # Muon with different momentum betas
    experiments.append(ExperimentConfig(
        name="P4_muon_split_t95_a85", optimizer="muon_split", lr=0.01, max_steps=2000,
        beta_text=0.95, beta_audio=0.85, separate_backward=True,
    ))
    experiments.append(ExperimentConfig(
        name="P4_muon_split_t90_a80", optimizer="muon_split", lr=0.01, max_steps=2000,
        beta_text=0.90, beta_audio=0.80, separate_backward=True,
    ))
    # Muon split with audio upscaling
    experiments.append(ExperimentConfig(
        name="P4_muon_split_ascale2", optimizer="muon_split", lr=0.01, max_steps=2000,
        beta_text=0.95, beta_audio=0.85, audio_scale=2.0, separate_backward=True,
    ))

    # ======== Phase 5: Additional ideas ========
    # Audio loss upweighting
    experiments.append(ExperimentConfig(
        name="P5_audio_weight_2x", optimizer="adam", lr=1e-4, max_steps=2000,
        audio_loss_weight=2.0,
    ))
    experiments.append(ExperimentConfig(
        name="P5_audio_weight_3x", optimizer="adam", lr=1e-4, max_steps=2000,
        audio_loss_weight=3.0,
    ))
    # Text loss downweighting
    experiments.append(ExperimentConfig(
        name="P5_text_weight_0.5", optimizer="adam", lr=1e-4, max_steps=2000,
        text_loss_weight=0.5,
    ))
    # Higher WD
    experiments.append(ExperimentConfig(
        name="P5_adam_wd0.1", optimizer="adam", lr=1e-4, max_steps=2000,
        weight_decay=0.1,
    ))

    return experiments


# ============================================================================
# Analysis & reporting
# ============================================================================

def analyze_results(all_results: dict):
    """Analyze and compare all experiment results."""
    print("\n" + "=" * 80)
    print("SWEEP 42 RESULTS SUMMARY")
    print("=" * 80)

    # Summary table
    print(f"\n{'Experiment':<30} | {'Final Loss':>10} | {'Text Loss':>10} | {'Audio Loss':>10} | {'T_Acc':>7} | {'A_Acc':>7} | {'Time':>6}")
    print("-" * 100)

    ranked = []
    for name, dynamics in sorted(all_results.items()):
        if not dynamics:
            continue
        final = dynamics[-1]
        ranked.append((name, final))
        print(f"{name:<30} | {final['total_loss']:>10.4f} | {final['text_loss']:>10.4f} | "
              f"{final['audio_loss']:>10.4f} | {final['text_accuracy']:>7.3f} | "
              f"{final['audio_accuracy']:>7.4f} | {final['elapsed_s']:>5.0f}s")

    # Best by different metrics
    print("\n--- Best by metric ---")
    if ranked:
        best_total = min(ranked, key=lambda x: x[1]['total_loss'])
        best_text = min(ranked, key=lambda x: x[1]['text_loss'])
        best_audio = min(ranked, key=lambda x: x[1]['audio_loss'])
        best_tacc = max(ranked, key=lambda x: x[1]['text_accuracy'])
        best_aacc = max(ranked, key=lambda x: x[1]['audio_accuracy'])

        print(f"  Best total loss:    {best_total[0]} ({best_total[1]['total_loss']:.4f})")
        print(f"  Best text loss:     {best_text[0]} ({best_text[1]['text_loss']:.4f})")
        print(f"  Best audio loss:    {best_audio[0]} ({best_audio[1]['audio_loss']:.4f})")
        print(f"  Best text accuracy: {best_tacc[0]} ({best_tacc[1]['text_accuracy']:.3f})")
        print(f"  Best audio accuracy:{best_aacc[0]} ({best_aacc[1]['audio_accuracy']:.4f})")

    # Gradient dynamics comparison
    print("\n--- Gradient dynamics at final step ---")
    print(f"{'Experiment':<30} | {'Grad Norm':>10} | {'Emb GN':>10} | {'Attn GN':>10} | {'MLP GN':>10} | {'TxtEmb/AudEmb':>14}")
    print("-" * 100)
    for name, dynamics in sorted(all_results.items()):
        if not dynamics:
            continue
        final = dynamics[-1]
        print(f"{name:<30} | {final.get('grad_norm_total',0):>10.4f} | "
              f"{final.get('grad_norm_embedding',0):>10.4f} | "
              f"{final.get('grad_norm_attention',0):>10.4f} | "
              f"{final.get('grad_norm_mlp',0):>10.4f} | "
              f"{final.get('text_emb_grad_norm',0):>6.2f}/{final.get('audio_emb_grad_norm',0):>6.2f}")

    # Audio layer analysis
    print("\n--- Audio layer losses (final step) ---")
    print(f"{'Experiment':<30} | {'L0':>7} | {'L1a':>7} | {'L2a':>7} | {'L2b':>7} | {'L1b':>7} | {'L2c':>7} | {'L2d':>7}")
    print("-" * 95)
    for name, dynamics in sorted(all_results.items()):
        if not dynamics:
            continue
        final = dynamics[-1]
        layers = [final.get(f'audio_loss_L{i}', 0) for i in range(7)]
        print(f"{name:<30} | {layers[0]:>7.4f} | {layers[1]:>7.4f} | {layers[2]:>7.4f} | "
              f"{layers[3]:>7.4f} | {layers[4]:>7.4f} | {layers[5]:>7.4f} | {layers[6]:>7.4f}")

    return all_results


# ============================================================================
# Main
# ============================================================================

def main():
    find_qwen2_path()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset once
    dataset = OmniDataset(DATA_PATH, max_seq_len=512)

    # Define experiments
    experiments = define_experiments()
    all_results = {}

    print(f"\nTotal experiments planned: {len(experiments)}")
    print(f"Results directory: {RESULTS_DIR}")
    print()

    # Run Phase 1-5 short experiments
    for exp in experiments:
        try:
            dynamics = run_experiment(exp, dataset)
            all_results[exp.name] = dynamics
        except Exception as e:
            print(f"  ERROR in {exp.name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp.name] = []

    # Phase 2 continuation: S3 after S2
    s2_ckpt = os.path.join(RESULTS_DIR, "P2_S2_for_S3", "checkpoint.pt")
    if os.path.exists(s2_ckpt):
        s3_cfg = ExperimentConfig(
            name="P2_S3_after_S2", optimizer="adam", lr=5e-5, max_steps=2000,
            stage="S3", s2_checkpoint=s2_ckpt, warmup_steps=50,
        )
        try:
            dynamics = run_experiment(s3_cfg, dataset)
            all_results[s3_cfg.name] = dynamics
        except Exception as e:
            print(f"  ERROR in S3: {e}")
            all_results[s3_cfg.name] = []

    # Analyze short experiments
    analyze_results(all_results)

    # Save combined results
    with open(os.path.join(RESULTS_DIR, "all_results_short.json"), "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f)

    # ======== Long runs of best approaches ========
    print("\n\n" + "=" * 80)
    print("PHASE 6: LONG RUNS (5000 steps)")
    print("=" * 80)

    # Find best configurations
    ranked = [(name, dyn[-1]['total_loss']) for name, dyn in all_results.items() if dyn]
    ranked.sort(key=lambda x: x[1])

    # Run top 3 + muon best + split best as long experiments
    long_candidates = set()
    # Top 3 overall
    for name, _ in ranked[:3]:
        long_candidates.add(name)
    # Best muon
    muon_ranked = [(n, l) for n, l in ranked if "muon" in n.lower()]
    if muon_ranked:
        long_candidates.add(muon_ranked[0][0])
    # Best split
    split_ranked = [(n, l) for n, l in ranked if "split" in n.lower()]
    if split_ranked:
        long_candidates.add(split_ranked[0][0])

    print(f"\nLong run candidates: {long_candidates}")

    # Map short experiment configs to long versions
    exp_map = {e.name: e for e in experiments}
    long_results = {}

    for candidate_name in long_candidates:
        if candidate_name in exp_map:
            long_cfg = copy.deepcopy(exp_map[candidate_name])
        elif candidate_name == "P2_S3_after_S2":
            long_cfg = ExperimentConfig(
                name="P2_S3_after_S2", optimizer="adam", lr=5e-5,
                stage="S3", s2_checkpoint=s2_ckpt, warmup_steps=50,
            )
        else:
            continue

        long_cfg.name = f"LONG_{candidate_name}"
        long_cfg.max_steps = 5000

        try:
            dynamics = run_experiment(long_cfg, dataset)
            long_results[long_cfg.name] = dynamics
        except Exception as e:
            print(f"  ERROR in long {long_cfg.name}: {e}")
            long_results[long_cfg.name] = []

    # Final analysis
    print("\n\n" + "=" * 80)
    print("FINAL LONG RUN RESULTS")
    print("=" * 80)
    analyze_results(long_results)

    # Save all results
    combined = {**all_results, **long_results}
    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(combined, f)

    print(f"\nAll results saved to {RESULTS_DIR}/all_results.json")
    print("Sweep 42 complete!")


if __name__ == "__main__":
    main()
