#!/usr/bin/env python3
"""
Fair evaluation of S3 models. Fixes the averaging bug in train_s3.py
by tracking text_n and audio_n separately.

Usage:
  python eval_s3.py --ckpt results/s3_adam/model_final.pt
  python eval_s3.py --ckpt /workspace/mini-omni-ckpt/lit_model.pth  # pretrained baseline
"""

import os
import sys
import json
import gc
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/workspace/mini-omni-ref")
from litgpt.config import Config
from litgpt.model import GPT

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
TEXT_VOCAB_SIZE = 152000
AUDIO_VOCAB_SIZE = 4160


def load_model(ckpt_path):
    config = Config.from_file("/workspace/mini-omni-ckpt/model_config.yaml")
    model = GPT(config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    del ckpt; gc.collect()
    return model.to(DEVICE).to(DTYPE).eval()


def evaluate(model, val_data, max_samples=500):
    """Evaluate with CORRECT averaging: track text_n and audio_n separately."""
    text_losses = []
    audio_losses = []  # per-sample mean of 7 CBs
    cb_losses = [[] for _ in range(7)]
    task_losses = defaultdict(lambda: {"text": [], "audio": []})

    with torch.no_grad():
        for idx, d in enumerate(val_data):
            if idx >= max_samples:
                break

            streams = d['streams'].unsqueeze(0).to(DEVICE)
            lm_text = d['loss_mask_text']
            lm_audio = d['loss_mask_audio']
            task = d['task']

            input_ids = [streams[:, i, :] for i in range(8)]

            with torch.amp.autocast('cuda', dtype=DTYPE):
                xa, xt = model(audio_features=None, input_ids=input_ids)

            # Text loss (only if this sample has text targets)
            text_targets = input_ids[7][:, 1:]
            text_logits = xt[:, :-1]
            text_mask = lm_text[1:]

            if text_mask.any():
                tl = F.cross_entropy(
                    text_logits[0][text_mask].float(),
                    text_targets[0][text_mask.to(DEVICE)],
                    reduction='mean'
                ).item()
                text_losses.append(tl)
                task_losses[task]["text"].append(tl)

            # Audio loss (only if this sample has audio targets)
            audio_mask = lm_audio[1:]
            if audio_mask.any():
                sample_cbs = []
                for i in range(7):
                    at = input_ids[i][:, 1:]
                    al = xa[i][:, :-1]
                    shifted = (at[0][audio_mask.to(DEVICE)] - TEXT_VOCAB_SIZE - i * AUDIO_VOCAB_SIZE).clamp(0, AUDIO_VOCAB_SIZE - 1)
                    cl = F.cross_entropy(al[0][audio_mask.to(DEVICE)].float(), shifted, reduction='mean').item()
                    cb_losses[i].append(cl)
                    sample_cbs.append(cl)
                audio_losses.append(sum(sample_cbs) / 7)
                task_losses[task]["audio"].append(sum(sample_cbs) / 7)

    results = {
        "text_loss": sum(text_losses) / len(text_losses) if text_losses else 0,
        "audio_loss": sum(audio_losses) / len(audio_losses) if audio_losses else 0,
        "text_n": len(text_losses),
        "audio_n": len(audio_losses),
        "cb_losses": {f"cb{i}": sum(cb_losses[i]) / len(cb_losses[i]) if cb_losses[i] else 0
                      for i in range(7)},
        "per_task": {task: {
            "text": sum(v["text"]) / len(v["text"]) if v["text"] else 0,
            "audio": sum(v["audio"]) / len(v["audio"]) if v["audio"] else 0,
            "text_n": len(v["text"]),
            "audio_n": len(v["audio"]),
        } for task, v in task_losses.items()},
    }
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--label", default=None, help="Label for this eval")
    args = parser.parse_args()

    print(f"Loading model from {args.ckpt}...")
    model = load_model(args.ckpt)

    print(f"Loading val data...")
    val_data = torch.load("/root/.cache/autoresearch/s3_data/val.pt", weights_only=False)

    print(f"Evaluating on {min(args.max_samples, len(val_data))} samples...")
    results = evaluate(model, val_data, args.max_samples)

    label = args.label or os.path.basename(os.path.dirname(args.ckpt))
    print(f"\n=== {label} ===")
    print(f"  Text loss:  {results['text_loss']:.4f} (n={results['text_n']})")
    print(f"  Audio loss: {results['audio_loss']:.4f} (n={results['audio_n']})")
    cb_str = ' '.join(f"CB{i}={results['cb_losses'][f'cb{i}']:.2f}" for i in range(7))
    print(f"  Per-CB: {cb_str}")
    print(f"\n  Per task:")
    for task, v in sorted(results['per_task'].items()):
        print(f"    {task}: text={v['text']:.4f} (n={v['text_n']}), audio={v['audio']:.4f} (n={v['audio_n']})")

    # Save results
    out_path = os.path.join(os.path.dirname(args.ckpt), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({"label": label, **results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
