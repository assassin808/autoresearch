#!/usr/bin/env python3
"""
Sweep 42 continuation: Fix split momentum, run missing experiments + long runs.
No checkpoint saving (dynamics only) to avoid disk issues.
"""

import os, sys, json, time, math, copy, gc
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/mini-omni-ref")
from litgpt.config import Config
from litgpt.model import GPT

sys.path.insert(0, "/workspace/autoresearch")
from train_s2 import (
    convert_qwen2_to_litgpt, build_s2_checkpoint,
    OmniDataset, collate_fn,
)
from sweep42_s2_experiments import (
    Muon, newton_schulz_5,
    ExperimentConfig, get_lr_schedule,
    DEVICE, DTYPE, MINI_OMNI_CONFIG, MINI_OMNI_CKPT, DATA_PATH, RESULTS_DIR,
)

QWEN2_PATH = str(list(Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B").glob("snapshots/*"))[0])


# ============================================================================
# Fixed split momentum: proper two-pass gradient accumulation
# ============================================================================

class AdamSplitV2(torch.optim.Optimizer):
    """
    Adam with separate second-moment tracking for text vs audio.

    Usage:
      1. zero_grad()
      2. text_loss.backward() → accumulate_grads("text")
      3. zero model grads
      4. audio_loss.backward() → accumulate_grads("audio")
      5. step() — uses combined m, but separate v_text/v_audio
    """
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def accumulate_grads(self, source):
        """Store current gradients into source-specific buffer."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "grad_text" not in state:
                    state["grad_text"] = torch.zeros_like(p)
                    state["grad_audio"] = torch.zeros_like(p)
                if source == "text":
                    state["grad_text"].copy_(p.grad)
                else:
                    state["grad_audio"].copy_(p.grad)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                state = self.state[p]
                if "grad_text" not in state:
                    continue

                g_text = state["grad_text"]
                g_audio = state["grad_audio"]
                g_combined = g_text + g_audio

                if wd > 0:
                    p.mul_(1 - lr * wd)

                if "step" not in state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v_text"] = torch.zeros_like(p)
                    state["v_audio"] = torch.zeros_like(p)

                state["step"] += 1
                m = state["m"]
                v_text = state["v_text"]
                v_audio = state["v_audio"]

                # Shared first moment on combined gradient
                m.mul_(beta1).add_(g_combined, alpha=1 - beta1)

                # Separate second moments
                v_text.mul_(beta2).addcmul_(g_text, g_text, value=1 - beta2)
                v_audio.mul_(beta2).addcmul_(g_audio, g_audio, value=1 - beta2)

                # Bias correction
                t = state["step"]
                m_hat = m / (1 - beta1 ** t)
                # Use max(v_text, v_audio) — conservative denominator
                v_max = torch.max(v_text, v_audio)
                v_hat = v_max / (1 - beta2 ** t)

                p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)

                # Reset buffers
                state["grad_text"].zero_()
                state["grad_audio"].zero_()


class MuonSplitV2(torch.optim.Optimizer):
    """
    Muon with separate momentum for text vs audio, summed before NS.

    Same accumulate_grads pattern as AdamSplitV2.
    """
    def __init__(self, params, lr=0.02, beta_text=0.95, beta_audio=0.85,
                 ns_steps=5, weight_decay=0.0, audio_scale=1.0):
        defaults = dict(lr=lr, beta_text=beta_text, beta_audio=beta_audio,
                       ns_steps=ns_steps, weight_decay=weight_decay,
                       audio_scale=audio_scale)
        super().__init__(params, defaults)

    def accumulate_grads(self, source):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "grad_text" not in state:
                    state["grad_text"] = torch.zeros_like(p)
                    state["grad_audio"] = torch.zeros_like(p)
                if source == "text":
                    state["grad_text"].copy_(p.grad)
                else:
                    state["grad_audio"].copy_(p.grad)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            bt = group["beta_text"]
            ba = group["beta_audio"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]
            ascale = group["audio_scale"]

            for p in group["params"]:
                state = self.state[p]
                if "grad_text" not in state:
                    continue

                if wd > 0:
                    p.mul_(1 - lr * wd)

                if "buf_text" not in state:
                    state["buf_text"] = torch.zeros_like(p)
                    state["buf_audio"] = torch.zeros_like(p)

                # Update separate momentum buffers
                state["buf_text"].mul_(bt).add_(state["grad_text"])
                state["buf_audio"].mul_(ba).add_(state["grad_audio"])

                # Sum before NS
                combined = state["buf_text"] + ascale * state["buf_audio"]

                if p.ndim >= 2 and min(p.shape) >= 2:
                    update = newton_schulz_5(combined, steps=ns_steps)
                else:
                    update = combined

                p.add_(update, alpha=-lr)

                state["grad_text"].zero_()
                state["grad_audio"].zero_()


# ============================================================================
# Training engine with proper split backward
# ============================================================================

def run_experiment(cfg, dataset):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {cfg.name}")
    print(f"{'='*70}")
    print(f"  Opt: {cfg.optimizer}, LR: {cfg.lr}, WD: {cfg.weight_decay}, Steps: {cfg.max_steps}")

    config = Config.from_file(MINI_OMNI_CONFIG)

    if cfg.s2_checkpoint and os.path.exists(cfg.s2_checkpoint):
        ckpt = torch.load(cfg.s2_checkpoint, map_location="cpu", weights_only=True)
        weights = ckpt.get("model_state_dict", ckpt)
    else:
        weights = build_s2_checkpoint(QWEN2_PATH, MINI_OMNI_CKPT, config)

    model = GPT(config)
    model.load_state_dict(weights, strict=False)

    if cfg.stage == "S2":
        for n, p in model.named_parameters():
            if "whisper_adapter" in n:
                p.requires_grad = False

    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model = model.to(device=DEVICE, dtype=DTYPE)

    trainable = [p for p in model.parameters() if p.requires_grad]

    if cfg.optimizer == "adam":
        optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2))
    elif cfg.optimizer == "muon":
        optimizer = Muon(trainable, lr=cfg.lr, momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam_split_v2":
        optimizer = AdamSplitV2(trainable, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "muon_split_v2":
        optimizer = MuonSplitV2(trainable, lr=cfg.lr, beta_text=cfg.beta_text, beta_audio=cfg.beta_audio,
                                ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay, audio_scale=cfg.audio_scale)
    else:
        raise ValueError(f"Unknown: {cfg.optimizer}")

    is_split = cfg.optimizer.endswith("_v2")

    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True)

    model.train()
    dynamics = []
    data_iter = iter(dl)
    opt_step = 0
    t0 = time.time()

    # Accumulators
    loss_acc = text_acc_num = text_acc_den = audio_acc_num = audio_acc_den = 0
    text_loss_acc = audio_loss_acc = 0.0
    audio_layer_acc = [0.0] * 7
    n_acc = 0

    for step in range(cfg.max_steps * cfg.grad_accum):
        try:
            input_ids, labels, _, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            input_ids, labels, _, _ = next(data_iter)

        input_ids = input_ids.to(DEVICE)
        labels = labels.to(DEVICE)
        B = input_ids.shape[0]
        streams = [input_ids[:, i, :] for i in range(8)]

        if is_split:
            # Two-pass: text then audio
            # Pass 1: text
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                xa_list, xt = model(audio_features=None, input_ids=streams, task=["AT"]*B)
                text_loss = F.cross_entropy(xt.reshape(-1, xt.size(-1)), labels[:,7,:].reshape(-1), ignore_index=-100)
            (text_loss * cfg.text_loss_weight / cfg.grad_accum).backward()
            optimizer.accumulate_grads("text")

            # Pass 2: audio
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                xa_list2, xt2 = model(audio_features=None, input_ids=streams, task=["AT"]*B)
                audio_loss = 0.0
                audio_layer_losses = []
                for layer in range(7):
                    ll = F.cross_entropy(xa_list2[layer].reshape(-1, xa_list2[layer].size(-1)),
                                        labels[:,layer,:].reshape(-1), ignore_index=-100)
                    audio_loss += ll
                    audio_layer_losses.append(ll.item())
                audio_loss = audio_loss / 7.0
            (audio_loss * cfg.audio_loss_weight / cfg.grad_accum).backward()
            optimizer.accumulate_grads("audio")

            # Use xt2/xa_list2 for accuracy (last forward pass)
            xa_list_for_acc = xa_list2
            xt_for_acc = xt2
        else:
            # Standard single-pass
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                xa_list, xt = model(audio_features=None, input_ids=streams, task=["AT"]*B)
                text_loss = F.cross_entropy(xt.reshape(-1, xt.size(-1)), labels[:,7,:].reshape(-1), ignore_index=-100)
                audio_loss = 0.0
                audio_layer_losses = []
                for layer in range(7):
                    ll = F.cross_entropy(xa_list[layer].reshape(-1, xa_list[layer].size(-1)),
                                        labels[:,layer,:].reshape(-1), ignore_index=-100)
                    audio_loss += ll
                    audio_layer_losses.append(ll.item())
                audio_loss = audio_loss / 7.0
                loss = (cfg.text_loss_weight * text_loss + cfg.audio_loss_weight * audio_loss) / cfg.grad_accum
            loss.backward()
            xa_list_for_acc = xa_list
            xt_for_acc = xt

        # Track accuracy
        with torch.no_grad():
            tm = labels[:,7,:].reshape(-1) != -100
            if tm.any():
                text_acc_num += (xt_for_acc.reshape(-1, xt_for_acc.size(-1))[tm].argmax(-1) == labels[:,7,:].reshape(-1)[tm]).sum().item()
                text_acc_den += tm.sum().item()
            for layer in range(7):
                am = labels[:,layer,:].reshape(-1) != -100
                if am.any():
                    audio_acc_num += (xa_list_for_acc[layer].reshape(-1, xa_list_for_acc[layer].size(-1))[am].argmax(-1) == labels[:,layer,:].reshape(-1)[am]).sum().item()
                    audio_acc_den += am.sum().item()

        text_loss_acc += text_loss.item()
        audio_loss_acc += audio_loss.item()
        for i in range(7):
            audio_layer_acc[i] += audio_layer_losses[i]
        n_acc += 1

        # Optimizer step
        if (step + 1) % cfg.grad_accum == 0:
            opt_step += 1

            # Grad norms before clipping
            total_gn = 0.0
            emb_gn = attn_gn = mlp_gn = 0.0
            text_emb_gn = audio_emb_gn = 0.0

            if not is_split:
                for n, p in model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        gn2 = p.grad.float().norm().item() ** 2
                        total_gn += gn2
                        if "wte" in n:
                            emb_gn += gn2
                            text_emb_gn = p.grad[:config.text_vocab_size].float().norm().item()
                            audio_emb_gn = p.grad[config.text_vocab_size:].float().norm().item()
                        elif ".attn." in n: attn_gn += gn2
                        elif ".mlp." in n: mlp_gn += gn2

            # Clip + LR
            if cfg.optimizer in ("adam", "adam_split_v2"):
                if not is_split:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            lr = get_lr_schedule(opt_step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.step()
            if not is_split:
                optimizer.zero_grad()

            # Record
            record = {
                "step": opt_step,
                "lr": lr,
                "total_loss": (text_loss_acc + audio_loss_acc) / n_acc,
                "text_loss": text_loss_acc / n_acc,
                "audio_loss": audio_loss_acc / n_acc,
                "text_accuracy": text_acc_num / max(1, text_acc_den),
                "audio_accuracy": audio_acc_num / max(1, audio_acc_den),
                "grad_norm_total": total_gn ** 0.5,
                "grad_norm_embedding": emb_gn ** 0.5,
                "grad_norm_attention": attn_gn ** 0.5,
                "grad_norm_mlp": mlp_gn ** 0.5,
                "text_emb_grad_norm": text_emb_gn,
                "audio_emb_grad_norm": audio_emb_gn,
                "text_emb_norm": model.transformer.wte.weight[:config.text_vocab_size].float().norm().item(),
                "audio_emb_norm": model.transformer.wte.weight[config.text_vocab_size:].float().norm().item(),
                "elapsed_s": time.time() - t0,
            }
            for i in range(7):
                record[f"audio_loss_L{i}"] = audio_layer_acc[i] / n_acc
            dynamics.append(record)

            if opt_step % 50 == 0 or opt_step <= 5:
                print(f"  step {opt_step}/{cfg.max_steps} | loss={record['total_loss']:.4f} | "
                      f"text={record['text_loss']:.4f} | audio={record['audio_loss']:.4f} | "
                      f"t_acc={record['text_accuracy']:.3f} | a_acc={record['audio_accuracy']:.4f} | lr={lr:.2e}")

            # Reset
            text_loss_acc = audio_loss_acc = 0.0
            audio_layer_acc = [0.0] * 7
            text_acc_num = text_acc_den = audio_acc_num = audio_acc_den = n_acc = 0

    elapsed = time.time() - t0
    print(f"  Done: {elapsed:.1f}s")

    save_dir = os.path.join(RESULTS_DIR, cfg.name)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "dynamics.json"), "w") as f:
        json.dump(dynamics, f)

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return dynamics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = OmniDataset(DATA_PATH, max_seq_len=512)
    all_results = {}

    # ========== Missing short experiments ==========
    short_exps = [
        # Missing Phase 5
        ExperimentConfig(name="P5_text_weight_0.5", optimizer="adam", lr=1e-4, max_steps=2000,
                        text_loss_weight=0.5),
        ExperimentConfig(name="P5_adam_wd0.1", optimizer="adam", lr=1e-4, max_steps=2000,
                        weight_decay=0.1),
        # Fixed split experiments
        ExperimentConfig(name="P4v2_adam_split", optimizer="adam_split_v2", lr=1e-4, max_steps=2000),
        ExperimentConfig(name="P4v2_muon_split_t95_a85", optimizer="muon_split_v2", lr=0.01,
                        max_steps=2000, beta_text=0.95, beta_audio=0.85),
        ExperimentConfig(name="P4v2_muon_split_t90_a80", optimizer="muon_split_v2", lr=0.01,
                        max_steps=2000, beta_text=0.90, beta_audio=0.80),
        ExperimentConfig(name="P4v2_muon_split_ascale2", optimizer="muon_split_v2", lr=0.01,
                        max_steps=2000, beta_text=0.95, beta_audio=0.85, audio_scale=2.0),
        # Additional ideas
        # Separate LR: higher audio embedding LR via audio loss upweight + lower text weight
        ExperimentConfig(name="P5_balanced_t0.3_a2", optimizer="adam", lr=1e-4, max_steps=2000,
                        text_loss_weight=0.3, audio_loss_weight=2.0),
        # Muon with WD matching our best from previous sweeps
        ExperimentConfig(name="P3_muon_lr0.02_wd0.05", optimizer="muon", lr=0.02, max_steps=2000,
                        muon_momentum=0.95, weight_decay=0.05),
    ]

    for exp in short_exps:
        try:
            dynamics = run_experiment(exp, dataset)
            all_results[exp.name] = dynamics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[exp.name] = []

    # ========== Long runs (5000 steps) of top configs ==========
    print("\n" + "="*70)
    print("LONG RUNS (5000 steps)")
    print("="*70)

    long_exps = [
        # Top 3 from short runs
        ExperimentConfig(name="LONG_muon_lr0.02", optimizer="muon", lr=0.02, max_steps=5000,
                        muon_momentum=0.95, weight_decay=0.0),
        ExperimentConfig(name="LONG_adam_lr3e-4", optimizer="adam", lr=3e-4, max_steps=5000),
        ExperimentConfig(name="LONG_muon_lr0.01_wd", optimizer="muon", lr=0.01, max_steps=5000,
                        muon_momentum=0.95, weight_decay=0.05),
        ExperimentConfig(name="LONG_adam_lr1e-4", optimizer="adam", lr=1e-4, max_steps=5000),
    ]

    for exp in long_exps:
        try:
            dynamics = run_experiment(exp, dataset)
            all_results[exp.name] = dynamics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[exp.name] = []

    # ========== Analysis ==========
    print("\n" + "="*70)
    print("ALL RESULTS SUMMARY")
    print("="*70)

    # Load previously completed results too
    for name in os.listdir(RESULTS_DIR):
        if name not in all_results:
            dp = os.path.join(RESULTS_DIR, name, "dynamics.json")
            if os.path.exists(dp):
                try:
                    with open(dp) as f:
                        all_results[name] = json.load(f)
                except:
                    pass

    print(f"\n{'Experiment':<32} | {'Total':>7} | {'Text':>7} | {'Audio':>7} | {'TAcc':>5} | {'AAcc':>6} | {'Steps':>5}")
    print("-"*90)
    for name in sorted(all_results.keys()):
        d = all_results[name]
        if not d: continue
        r = d[-1]
        print(f"{name:<32} | {r.get('total_loss',0):>7.3f} | {r.get('text_loss',0):>7.4f} | "
              f"{r.get('audio_loss',0):>7.4f} | {r.get('text_accuracy',0):>5.3f} | "
              f"{r.get('audio_accuracy',0):>6.4f} | {r.get('step',0):>5}")

    with open(os.path.join(RESULTS_DIR, "all_results_v2.json"), "w") as f:
        json.dump({k: v for k, v in all_results.items() if v}, f)

    print(f"\nSaved to {RESULTS_DIR}/all_results_v2.json")


if __name__ == "__main__":
    main()
