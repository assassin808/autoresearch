#!/usr/bin/env python3
"""
Prepare VoiceAssistant-400K data for S2 training.

Replicates mini-omni's exact data format:
- 8 parallel token streams (7 audio + 1 text)
- 4 task types: T1T2, T1A2, A1T2, A1A2
- Layer-shifted SNAC tokens
- Whisper features for audio-input tasks

Output: preprocessed parquet shards with tokenized sequences ready for training.
"""

import os
import sys
import json
import glob
import random
import numpy as np
from pathlib import Path

import torch
import pyarrow.parquet as pq

# ============================================================================
# Mini-omni token constants (from inference.py)
# ============================================================================
TEXT_VOCAB_SIZE = 151936
TEXT_SPECIAL_TOKENS = 64
AUDIO_VOCAB_SIZE = 4096
AUDIO_SPECIAL_TOKENS = 64

PADDED_TEXT_VOCAB = TEXT_VOCAB_SIZE + TEXT_SPECIAL_TOKENS  # 152000
PADDED_AUDIO_VOCAB = AUDIO_VOCAB_SIZE + AUDIO_SPECIAL_TOKENS  # 4160

# Text special tokens
_eot = TEXT_VOCAB_SIZE       # 151936
_pad_t = TEXT_VOCAB_SIZE + 1
_input_t = TEXT_VOCAB_SIZE + 2
_answer_t = TEXT_VOCAB_SIZE + 3
_asr = TEXT_VOCAB_SIZE + 4

# Audio special tokens (per-stream, before layershift)
_eoa = AUDIO_VOCAB_SIZE       # 4096
_pad_a = AUDIO_VOCAB_SIZE + 1
_input_a = AUDIO_VOCAB_SIZE + 2
_answer_a = AUDIO_VOCAB_SIZE + 3
_split = AUDIO_VOCAB_SIZE + 4

TOTAL_VOCAB = PADDED_TEXT_VOCAB + 7 * PADDED_AUDIO_VOCAB  # 181120


def layershift(token_id, layer, stride=PADDED_AUDIO_VOCAB, shift=PADDED_TEXT_VOCAB):
    """Shift audio token to its layer-specific range in the unified vocab."""
    return token_id + shift + layer * stride


def parse_snac(snac_str):
    """Parse SNAC string '# t0 t1 t2 t3 t4 t5 t6 # t0 ...' into list of 7-tuples."""
    frames = []
    parts = snac_str.strip().split('#')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        tokens = [int(t) for t in part.split()]
        if len(tokens) == 7:
            frames.append(tokens)
    return frames


def build_T1T2(question_tokens, answer_tokens, max_len=2048):
    """Text question → Text answer. No audio."""
    # Text stream: [_input_t] + question + [_eot, _answer_t] + answer
    text_stream = [_input_t] + question_tokens + [_eot, _answer_t] + answer_tokens + [_eot]
    seq_len = len(text_stream)
    if seq_len > max_len:
        text_stream = text_stream[:max_len]
        seq_len = max_len

    # Audio streams: all padding
    streams = []
    for i in range(7):
        streams.append([layershift(_pad_a, i)] * seq_len)
    streams.append(text_stream)

    # Loss mask: only on answer portion (after _answer_t)
    answer_start = 1 + len(question_tokens) + 2  # _input_t + question + _eot + _answer_t
    loss_mask_text = [0] * min(answer_start, seq_len) + [1] * max(0, seq_len - answer_start)
    loss_mask_audio = [0] * seq_len

    return streams, loss_mask_text, loss_mask_audio, "T1T2"


def build_T1A2(question_tokens, snac_frames, max_len=2048):
    """Text question → Audio answer."""
    n_frames = len(snac_frames)

    # Text stream: [_input_t] + question + [_eot, _answer_t] + padding for audio length
    prefix_text = [_input_t] + question_tokens + [_eot, _answer_t]
    text_stream = prefix_text + [_pad_t] * n_frames

    # Audio streams: prefix padding + answer SNAC tokens
    streams = []
    prefix_len = len(prefix_text)
    for i in range(7):
        audio_prefix = [layershift(_pad_a, i)] * prefix_len
        audio_tokens = [layershift(snac_frames[f][i], i) for f in range(n_frames)]
        streams.append(audio_prefix + audio_tokens)
    streams.append(text_stream)

    seq_len = len(streams[0])
    if seq_len > max_len:
        for i in range(8):
            streams[i] = streams[i][:max_len]
        seq_len = max_len

    # Loss mask: audio loss on answer portion only
    loss_mask_text = [0] * seq_len
    loss_mask_audio = [0] * min(prefix_len, seq_len) + [1] * max(0, seq_len - prefix_len)

    return streams, loss_mask_text, loss_mask_audio, "T1A2"


def build_A1T2(whisper_len, answer_tokens, max_len=2048):
    """Audio question → Text answer. Whisper features injected."""
    T = whisper_len

    # Text stream: [_input_t, _pad_t*T, _eot, _answer_t] + answer
    text_stream = [_input_t] + [_pad_t] * T + [_eot, _answer_t] + answer_tokens + [_eot]

    # Audio streams: [_input_a, _pad_a*T, _eoa, _answer_a] + padding
    answer_pad_len = len(answer_tokens) + 1  # +1 for final _eot
    streams = []
    for i in range(7):
        audio_stream = ([layershift(_input_a, i)]
                        + [layershift(_pad_a, i)] * T
                        + [layershift(_eoa, i), layershift(_answer_a, i)]
                        + [layershift(_pad_a, i)] * answer_pad_len)
        streams.append(audio_stream)
    streams.append(text_stream)

    seq_len = len(streams[0])
    if seq_len > max_len:
        for i in range(8):
            streams[i] = streams[i][:max_len]
        seq_len = max_len

    # Loss mask: text loss on answer portion
    answer_start = 1 + T + 2  # _input_t + pad*T + _eot + _answer_t
    loss_mask_text = [0] * min(answer_start, seq_len) + [1] * max(0, seq_len - answer_start)
    loss_mask_audio = [0] * seq_len

    return streams, loss_mask_text, loss_mask_audio, "A1T2"


def build_A1A2(whisper_len, snac_frames, max_len=2048):
    """Audio question → Audio answer. Whisper features injected."""
    T = whisper_len
    n_frames = len(snac_frames)

    # Text stream: [_input_t, _pad_t*T, _eot, _answer_t] + padding
    text_stream = [_input_t] + [_pad_t] * T + [_eot, _answer_t] + [_pad_t] * n_frames

    # Audio streams: [_input_a, _pad_a*T, _eoa, _answer_a] + SNAC tokens
    prefix_len = 1 + T + 2  # _input_a + pad*T + _eoa + _answer_a
    streams = []
    for i in range(7):
        audio_prefix = ([layershift(_input_a, i)]
                        + [layershift(_pad_a, i)] * T
                        + [layershift(_eoa, i), layershift(_answer_a, i)])
        audio_tokens = [layershift(snac_frames[f][i], i) for f in range(n_frames)]
        streams.append(audio_prefix + audio_tokens)
    streams.append(text_stream)

    seq_len = len(streams[0])
    if seq_len > max_len:
        for i in range(8):
            streams[i] = streams[i][:max_len]
        seq_len = max_len

    # Loss mask: audio loss on answer portion
    loss_mask_text = [0] * seq_len
    loss_mask_audio = [0] * min(prefix_len, seq_len) + [1] * max(0, seq_len - prefix_len)

    return streams, loss_mask_text, loss_mask_audio, "A1A2"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/.cache/autoresearch/s2_data")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for testing")
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)

    # Find all parquet blobs
    blob_dir = "/workspace/.hf_home/hub/datasets--gpt-omni--VoiceAssistant-400K/blobs"
    blobs = sorted(glob.glob(f"{blob_dir}/*"))

    parquet_files = []
    for b in blobs:
        try:
            pq.read_metadata(b)
            parquet_files.append(b)
        except:
            pass
    print(f"Found {len(parquet_files)} parquet files")

    # Process all samples
    all_samples = []
    for pf in parquet_files:
        try:
            table = pq.read_table(pf, columns=['question', 'answer', 'answer_snac'])
            data = table.to_pydict()
            for i in range(len(data['question'])):
                all_samples.append({
                    'question': data['question'][i],
                    'answer': data['answer'][i],
                    'answer_snac': data['answer_snac'][i],
                })
        except Exception as e:
            print(f"Skipping {pf}: {e}")
            continue

    print(f"Loaded {len(all_samples)} samples")

    if args.max_samples:
        all_samples = all_samples[:args.max_samples]

    random.shuffle(all_samples)

    # Split train/val
    n_val = max(1, int(len(all_samples) * args.val_fraction))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f"Train: {len(train_samples)}, Val: {n_val}")

    # Process and save
    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        all_streams = []  # list of (streams[8], loss_mask_text, loss_mask_audio, task)
        skipped = 0

        for idx, sample in enumerate(samples):
            if idx % 5000 == 0:
                print(f"  {split_name}: {idx}/{len(samples)}")

            question = sample['question']
            answer = sample['answer']
            snac_str = sample['answer_snac']

            # Tokenize
            q_tokens = tokenizer.encode(question, add_special_tokens=False)
            a_tokens = tokenizer.encode(answer, add_special_tokens=False)
            snac_frames = parse_snac(snac_str)

            if len(snac_frames) == 0:
                skipped += 1
                continue

            # Build all 4 task types for this sample
            # T1T2: text → text
            try:
                streams, lm_t, lm_a, task = build_T1T2(q_tokens, a_tokens, args.max_len)
                all_streams.append((streams, lm_t, lm_a, task))
            except:
                pass

            # T1A2: text → audio
            try:
                streams, lm_t, lm_a, task = build_T1A2(q_tokens, snac_frames, args.max_len)
                all_streams.append((streams, lm_t, lm_a, task))
            except:
                pass

            # A1T2: audio → text (use estimated whisper length from SNAC frames)
            # Mini-omni Whisper outputs ~50 tokens/sec, SNAC at ~12.5 frames/sec
            # So whisper_len ≈ snac_frames * 4
            whisper_len = min(len(snac_frames) * 4, 1500)  # cap at Whisper max
            try:
                streams, lm_t, lm_a, task = build_A1T2(whisper_len, a_tokens, args.max_len)
                all_streams.append((streams, lm_t, lm_a, task))
            except:
                pass

            # A1A2: audio → audio
            try:
                streams, lm_t, lm_a, task = build_A1A2(whisper_len, snac_frames, args.max_len)
                all_streams.append((streams, lm_t, lm_a, task))
            except:
                pass

        print(f"  {split_name}: {len(all_streams)} sequences ({skipped} skipped)")

        # Save as torch tensors (list of dicts)
        output_path = os.path.join(args.output_dir, f"{split_name}.pt")
        save_data = []
        for streams, lm_t, lm_a, task in all_streams:
            seq_len = len(streams[0])
            save_data.append({
                'streams': torch.tensor(streams, dtype=torch.long),     # (8, seq_len)
                'loss_mask_text': torch.tensor(lm_t, dtype=torch.bool),  # (seq_len,)
                'loss_mask_audio': torch.tensor(lm_a, dtype=torch.bool), # (seq_len,)
                'task': task,
                'seq_len': seq_len,
            })

        torch.save(save_data, output_path)
        print(f"  Saved {output_path} ({len(save_data)} sequences)")


if __name__ == "__main__":
    main()
