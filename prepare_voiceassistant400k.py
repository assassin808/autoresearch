#!/usr/bin/env python3
"""
Download and prepare VoiceAssistant-400K dataset for S2 training.
Converts HuggingFace dataset to the same .pt format as our existing paired data.

The dataset has:
  - question: text question
  - answer: text answer
  - answer_snac: SNAC tokens in format "# L0 L1a L2a L2b L1b L2c L2d # ..."

For S2 training we need:
  - text_tokens: tokenized answer text
  - audio_codebooks: {"L0": [...], "L1": [...], "L2": [...]}
"""

import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, "/workspace/mini-omni-ref")

# SNAC offsets matching our existing encoding
SNAC_L0_OFFSET = 8192
SNAC_L1_OFFSET = 12288
SNAC_L2_OFFSET = 16384


def parse_answer_snac(snac_str: str):
    """
    Parse answer_snac string into 3-level codebooks.

    Format: "# L0 L1a L2a L2b L1b L2c L2d # L0 L1a L2a L2b L1b L2c L2d # ..."
    Each frame has 7 tokens: [L0, L1_0, L2_0, L2_1, L1_1, L2_2, L2_3]

    Returns dict with L0 (N), L1 (2N), L2 (4N) lists with offsets applied.
    """
    # Split by '#' to get frames
    frames = [f.strip() for f in snac_str.split('#') if f.strip()]

    l0, l1, l2 = [], [], []

    for frame in frames:
        tokens = [int(t) for t in frame.split()]
        if len(tokens) != 7:
            # Skip malformed frames
            continue

        # Frame layout: [L0, L1_0, L2_0, L2_1, L1_1, L2_2, L2_3]
        l0.append(tokens[0] + SNAC_L0_OFFSET)
        l1.append(tokens[1] + SNAC_L1_OFFSET)
        l1.append(tokens[4] + SNAC_L1_OFFSET)
        l2.append(tokens[2] + SNAC_L2_OFFSET)
        l2.append(tokens[3] + SNAC_L2_OFFSET)
        l2.append(tokens[5] + SNAC_L2_OFFSET)
        l2.append(tokens[6] + SNAC_L2_OFFSET)

    return {"L0": l0, "L1": l1, "L2": l2}


def main():
    from datasets import load_dataset, Audio
    from transformers import AutoTokenizer

    out_dir = "/root/.cache/autoresearch/audio"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "voiceassistant400k_train.pt")
    val_path = os.path.join(out_dir, "voiceassistant400k_val.pt")

    if os.path.exists(train_path):
        print(f"Already exists: {train_path}")
        data = torch.load(train_path, map_location="cpu", weights_only=False)
        print(f"  {len(data)} samples")
        return

    # Load tokenizer (same as Qwen2-0.5B used by mini-omni)
    print("Loading tokenizer...")
    qwen2_path = str(list(Path("/root/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B").glob("snapshots/*"))[0])
    tokenizer = AutoTokenizer.from_pretrained(qwen2_path)

    # Load dataset in streaming mode to avoid downloading huge audio files
    print("Loading VoiceAssistant-400K dataset (streaming)...")
    ds = load_dataset('gpt-omni/VoiceAssistant-400K', split='train', streaming=True,
                      columns=['question', 'answer', 'answer_snac'])

    # Process all samples
    train_data = []
    val_data = []
    skipped = 0
    total = 0

    for i, ex in enumerate(ds):
        total = i + 1
        if i % 10000 == 0:
            print(f"  Processing {i}...")

        answer_snac = ex.get('answer_snac', '')
        answer_text = ex.get('answer', '')

        if not answer_snac or not answer_text:
            skipped += 1
            continue

        # Parse SNAC tokens
        codebooks = parse_answer_snac(answer_snac)

        if len(codebooks["L0"]) < 2:
            skipped += 1
            continue

        # Verify 1:2:4 ratio
        n = len(codebooks["L0"])
        if len(codebooks["L1"]) != 2 * n or len(codebooks["L2"]) != 4 * n:
            skipped += 1
            continue

        # Tokenize answer text
        text_tokens = tokenizer.encode(answer_text, add_special_tokens=False)

        sample = {
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "audio_codebooks": {
                "L0": torch.tensor(codebooks["L0"], dtype=torch.long),
                "L1": torch.tensor(codebooks["L1"], dtype=torch.long),
                "L2": torch.tensor(codebooks["L2"], dtype=torch.long),
            },
            "transcript": answer_text,
        }

        # 99% train, 1% val (deterministic split)
        if i % 100 == 0:
            val_data.append(sample)
        else:
            train_data.append(sample)

    print(f"\n  Train: {len(train_data)} samples")
    print(f"  Val: {len(val_data)} samples")
    print(f"  Skipped: {skipped}")

    # Quick stats
    audio_lens = [len(s["audio_codebooks"]["L0"]) for s in train_data[:1000]]
    text_lens = [len(s["text_tokens"]) for s in train_data[:1000]]
    print(f"  Audio frames (first 1000): mean={sum(audio_lens)/len(audio_lens):.0f}, "
          f"min={min(audio_lens)}, max={max(audio_lens)}")
    print(f"  Text tokens (first 1000): mean={sum(text_lens)/len(text_lens):.0f}, "
          f"min={min(text_lens)}, max={max(text_lens)}")

    print(f"\nSaving train to {train_path}...")
    torch.save(train_data, train_path)
    print(f"Saving val to {val_path}...")
    torch.save(val_data, val_path)
    print("Done!")


if __name__ == "__main__":
    main()
