#!/usr/bin/env python3
"""
Sweep 43: MuonHybrid + MuonSeparateNS optimizers.

MuonHybrid: Muon+NS for transformer blocks, Adam for embedding/1D params.
MuonSeparateNS: Per-modality NS (apply NS to text/audio buffers separately, then sum).
"""

import os, sys, json, time, math, copy, gc
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

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
    DEVICE, DTYPE, MINI_OMNI_CONFIG, MINI_OMNI_CKPT, DATA_PATH,
)

QWEN2_PATH = str(list(Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B").glob("snapshots/*"))[0])
RESULTS_DIR = "/workspace/autoresearch/checkpoints/sweep43"


# ============================================================================
# MuonHybrid: Muon backbone + Adam embedding
# ============================================================================

class MuonHybrid(torch.optim.Optimizer):
    """
    Hybrid optimizer: Muon+NS for 2D transformer weights, Adam for embedding/1D.

    Param groups:
      - "muon": 2D attention/MLP weights → momentum + NS
      - "adam": wte embedding + 1D params (norms) → Adam with m/v moments
    """

    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, muon_momentum=0.95,
                 ns_steps=5, adam_betas=(0.9, 0.95), adam_eps=1e-8,
                 weight_decay=0.0):
        # Separate params by type, dedup by id for tied weights
        seen_ids = set()
        muon_params = []
        adam_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if id(p) in seen_ids:
                continue
            seen_ids.add(id(p))

            if "wte" in name or "lm_head" in name:
                adam_params.append(p)
            elif p.ndim >= 2 and min(p.shape) >= 2:
                muon_params.append(p)
            else:
                # 1D params: norms, biases
                adam_params.append(p)

        param_groups = [
            {"params": muon_params, "group_type": "muon", "base_lr": muon_lr,
             "lr": muon_lr, "momentum": muon_momentum, "ns_steps": ns_steps,
             "weight_decay": weight_decay},
            {"params": adam_params, "group_type": "adam", "base_lr": adam_lr,
             "lr": adam_lr, "betas": adam_betas, "eps": adam_eps,
             "weight_decay": weight_decay},
        ]

        defaults = dict(lr=muon_lr, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            if group["group_type"] == "muon":
                momentum = group["momentum"]
                ns_steps = group["ns_steps"]

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
                        update = newton_schulz_5(buf, steps=ns_steps)
                    else:
                        update = buf

                    p.add_(update, alpha=-lr)

            elif group["group_type"] == "adam":
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad

                    if wd > 0:
                        p.mul_(1 - lr * wd)

                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)

                    state["step"] += 1
                    m, v = state["m"], state["v"]
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    t = state["step"]
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)


# ============================================================================
# MuonSeparateNS: per-modality Newton-Schulz
# ============================================================================

class MuonSeparateNS(torch.optim.Optimizer):
    """
    Muon with separate momentum for text/audio, NS applied to each BEFORE summing.

    Key difference vs MuonSplitV2:
      MuonSplitV2:   update = NS(buf_text + audio_scale * buf_audio)
      MuonSeparateNS: update = NS(buf_text) + audio_scale * NS(buf_audio)
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

                if p.ndim >= 2 and min(p.shape) >= 2:
                    # Apply NS separately, then sum
                    ns_text = newton_schulz_5(state["buf_text"], steps=ns_steps)
                    ns_audio = newton_schulz_5(state["buf_audio"], steps=ns_steps)
                    update = ns_text + ascale * ns_audio
                else:
                    update = state["buf_text"] + ascale * state["buf_audio"]

                p.add_(update, alpha=-lr)

                state["grad_text"].zero_()
                state["grad_audio"].zero_()


# ============================================================================
# Extended config
# ============================================================================

@dataclass
class Sweep43Config(ExperimentConfig):
    muon_hybrid_adam_lr: float = 3e-4


# ============================================================================
# Training engine
# ============================================================================

def run_experiment(cfg: Sweep43Config, dataset):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {cfg.name}")
    print(f"{'='*70}")
    print(f"  Opt: {cfg.optimizer}, LR: {cfg.lr}, WD: {cfg.weight_decay}, Steps: {cfg.max_steps}")
    if cfg.optimizer == "muon_hybrid":
        print(f"  Muon LR: {cfg.lr}, Adam LR: {cfg.muon_hybrid_adam_lr}")
    if cfg.optimizer == "muon_separate_ns":
        print(f"  beta_text: {cfg.beta_text}, beta_audio: {cfg.beta_audio}, audio_scale: {cfg.audio_scale}")

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

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_count:,}")
    model = model.to(device=DEVICE, dtype=DTYPE)

    # Create optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]

    if cfg.optimizer == "muon_hybrid":
        optimizer = MuonHybrid(
            model, muon_lr=cfg.lr, adam_lr=cfg.muon_hybrid_adam_lr,
            muon_momentum=cfg.muon_momentum, ns_steps=cfg.muon_ns_steps,
            weight_decay=cfg.weight_decay,
        )
        # Log param group sizes
        for pg in optimizer.param_groups:
            n_params = sum(p.numel() for p in pg["params"])
            print(f"  {pg['group_type']} group: {len(pg['params'])} tensors, {n_params:,} params, base_lr={pg['base_lr']:.2e}")
    elif cfg.optimizer == "muon_separate_ns":
        optimizer = MuonSeparateNS(
            trainable, lr=cfg.lr, beta_text=cfg.beta_text, beta_audio=cfg.beta_audio,
            ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay,
            audio_scale=cfg.audio_scale,
        )
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.AdamW(
            trainable, lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2),
        )
    elif cfg.optimizer == "muon":
        optimizer = Muon(
            trainable, lr=cfg.lr, momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown: {cfg.optimizer}")

    is_split = cfg.optimizer == "muon_separate_ns"

    dl = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                    collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=True)

    model.train()
    dynamics = []
    data_iter = iter(dl)
    opt_step = 0
    t0 = time.time()

    # Accumulators
    text_loss_acc = audio_loss_acc = 0.0
    audio_layer_acc = [0.0] * 7
    text_acc_num = text_acc_den = audio_acc_num = audio_acc_den = n_acc = 0

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
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                xa_list, xt = model(audio_features=None, input_ids=streams, task=["AT"]*B)
                text_loss = F.cross_entropy(xt.reshape(-1, xt.size(-1)), labels[:,7,:].reshape(-1), ignore_index=-100)
            (text_loss * cfg.text_loss_weight / cfg.grad_accum).backward()
            optimizer.accumulate_grads("text")

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
        audio_loss_acc += audio_loss.item() if isinstance(audio_loss, float) else audio_loss.item()
        for i in range(7):
            audio_layer_acc[i] += audio_layer_losses[i]
        n_acc += 1

        # Optimizer step
        if (step + 1) % cfg.grad_accum == 0:
            opt_step += 1

            # Grad norms before clipping
            total_gn = emb_gn = attn_gn = mlp_gn = 0.0
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

            # Grad clipping only for Adam-based groups
            if cfg.optimizer == "muon_hybrid":
                # Clip only adam group params
                adam_params = []
                for pg in optimizer.param_groups:
                    if pg["group_type"] == "adam":
                        adam_params.extend(pg["params"])
                if adam_params:
                    torch.nn.utils.clip_grad_norm_(adam_params, 1.0)
            elif cfg.optimizer in ("adam",):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # LR schedule — ratio-based for hybrid, direct for others
            if cfg.optimizer == "muon_hybrid":
                lr_scale = get_lr_schedule(opt_step, cfg) / cfg.lr if cfg.lr > 0 else 0
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["base_lr"] * lr_scale
                display_lr = cfg.lr * lr_scale
            else:
                lr = get_lr_schedule(opt_step, cfg)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                display_lr = lr

            optimizer.step()
            if not is_split:
                optimizer.zero_grad()

            # Record
            record = {
                "step": opt_step,
                "lr": display_lr,
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
            # Per-group LRs for hybrid
            if cfg.optimizer == "muon_hybrid":
                for pg in optimizer.param_groups:
                    record[f"lr_{pg['group_type']}"] = pg["lr"]
            for i in range(7):
                record[f"audio_loss_L{i}"] = audio_layer_acc[i] / n_acc
            dynamics.append(record)

            if opt_step % 50 == 0 or opt_step <= 5:
                extra = ""
                if cfg.optimizer == "muon_hybrid":
                    extra = f" | muon_lr={optimizer.param_groups[0]['lr']:.2e} adam_lr={optimizer.param_groups[1]['lr']:.2e}"
                print(f"  step {opt_step}/{cfg.max_steps} | loss={record['total_loss']:.4f} | "
                      f"text={record['text_loss']:.4f} | audio={record['audio_loss']:.4f} | "
                      f"t_acc={record['text_accuracy']:.3f} | a_acc={record['audio_accuracy']:.4f} | lr={display_lr:.2e}{extra}")

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


# ============================================================================
# Experiments
# ============================================================================

def define_experiments():
    experiments = []

    # Phase 1: MuonHybrid (2000 steps)
    experiments.append(Sweep43Config(
        name="P1_hybrid_m0.02_a3e-4", optimizer="muon_hybrid",
        lr=0.02, muon_hybrid_adam_lr=3e-4, weight_decay=0.0, max_steps=2000,
        muon_momentum=0.95, muon_ns_steps=5,
    ))
    experiments.append(Sweep43Config(
        name="P1_hybrid_m0.01_a1e-4", optimizer="muon_hybrid",
        lr=0.01, muon_hybrid_adam_lr=1e-4, weight_decay=0.0, max_steps=2000,
        muon_momentum=0.95, muon_ns_steps=5,
    ))
    experiments.append(Sweep43Config(
        name="P1_hybrid_m0.02_a1e-3", optimizer="muon_hybrid",
        lr=0.02, muon_hybrid_adam_lr=1e-3, weight_decay=0.0, max_steps=2000,
        muon_momentum=0.95, muon_ns_steps=5,
    ))

    # Phase 2: MuonSeparateNS (2000 steps)
    experiments.append(Sweep43Config(
        name="P2_sepNS_lr0.02_t95_a85", optimizer="muon_separate_ns",
        lr=0.02, beta_text=0.95, beta_audio=0.85, max_steps=2000,
        muon_ns_steps=5, weight_decay=0.0, audio_scale=1.0,
    ))
    experiments.append(Sweep43Config(
        name="P2_sepNS_lr0.01_t95_a85", optimizer="muon_separate_ns",
        lr=0.01, beta_text=0.95, beta_audio=0.85, max_steps=2000,
        muon_ns_steps=5, weight_decay=0.0, audio_scale=1.0,
    ))
    experiments.append(Sweep43Config(
        name="P2_sepNS_lr0.02_t90_a80", optimizer="muon_separate_ns",
        lr=0.02, beta_text=0.90, beta_audio=0.80, max_steps=2000,
        muon_ns_steps=5, weight_decay=0.0, audio_scale=1.0,
    ))

    return experiments


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = OmniDataset(DATA_PATH, max_seq_len=512)
    all_results = {}

    experiments = define_experiments()
    print(f"Sweep 43: {len(experiments)} experiments")
    print(f"Results: {RESULTS_DIR}\n")

    # Phase 1 + 2: short runs
    for exp in experiments:
        # Skip if already done
        dyn_path = os.path.join(RESULTS_DIR, exp.name, "dynamics.json")
        if os.path.exists(dyn_path):
            print(f"Skipping {exp.name} (already done)")
            with open(dyn_path) as f:
                all_results[exp.name] = json.load(f)
            continue
        try:
            dynamics = run_experiment(exp, dataset)
            all_results[exp.name] = dynamics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[exp.name] = []

    # Phase 3: Long runs of best from each approach
    print("\n" + "="*70)
    print("PHASE 3: LONG RUNS (5000 steps)")
    print("="*70)

    # Find best hybrid
    hybrid_results = [(n, d[-1]["total_loss"]) for n, d in all_results.items() if d and n.startswith("P1_")]
    hybrid_results.sort(key=lambda x: x[1])

    # Find best separateNS
    sepns_results = [(n, d[-1]["total_loss"]) for n, d in all_results.items() if d and n.startswith("P2_")]
    sepns_results.sort(key=lambda x: x[1])

    exp_map = {e.name: e for e in experiments}
    long_exps = []

    if hybrid_results:
        best_name = hybrid_results[0][0]
        long_cfg = copy.deepcopy(exp_map[best_name])
        long_cfg.name = f"LONG_{best_name}"
        long_cfg.max_steps = 5000
        long_exps.append(long_cfg)
        print(f"  Best hybrid: {best_name} (loss={hybrid_results[0][1]:.4f})")

    if sepns_results:
        best_name = sepns_results[0][0]
        long_cfg = copy.deepcopy(exp_map[best_name])
        long_cfg.name = f"LONG_{best_name}"
        long_cfg.max_steps = 5000
        long_exps.append(long_cfg)
        print(f"  Best separateNS: {best_name} (loss={sepns_results[0][1]:.4f})")

    for exp in long_exps:
        dyn_path = os.path.join(RESULTS_DIR, exp.name, "dynamics.json")
        if os.path.exists(dyn_path):
            print(f"Skipping {exp.name} (already done)")
            with open(dyn_path) as f:
                all_results[exp.name] = json.load(f)
            continue
        try:
            dynamics = run_experiment(exp, dataset)
            all_results[exp.name] = dynamics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results[exp.name] = []

    # Summary
    print("\n" + "="*70)
    print("SWEEP 43 RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Experiment':<34} | {'Total':>7} | {'Text':>7} | {'Audio':>7} | {'TAcc':>5} | {'AAcc':>6} | {'Steps':>5}")
    print("-"*90)
    for name in sorted(all_results.keys()):
        d = all_results[name]
        if not d: continue
        r = d[-1]
        print(f"{name:<34} | {r.get('total_loss',0):>7.3f} | {r.get('text_loss',0):>7.4f} | "
              f"{r.get('audio_loss',0):>7.4f} | {r.get('text_accuracy',0):>5.3f} | "
              f"{r.get('audio_accuracy',0):>6.4f} | {r.get('step',0):>5}")

    with open(os.path.join(RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump({k: v for k, v in all_results.items() if v}, f)

    print(f"\nSaved to {RESULTS_DIR}/all_results.json")
    print("Sweep 43 complete!")


if __name__ == "__main__":
    main()
