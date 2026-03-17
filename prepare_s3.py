#!/usr/bin/env python3
"""
Prepare VoiceAssistant-400K data for S3 training with CORRECT delay pattern.

Fixes from prepare_s2.py:
1. SNAC delay pattern: stream i delayed by i positions
2. Separate text/audio loss masks (not both active in same sample)
3. Text stream gets _pad_t during audio answer (not text tokens)
4. Proper sequence lengths accounting for delay flush
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

# Audio special tokens (per-stream, before layershift)
_eoa = AUDIO_VOCAB_SIZE       # 4096
_pad_a = AUDIO_VOCAB_SIZE + 1
_input_a = AUDIO_VOCAB_SIZE + 2
_answer_a = AUDIO_VOCAB_SIZE + 3


def layershift(token_id, layer, stride=PADDED_AUDIO_VOCAB, shift=PADDED_TEXT_VOCAB):
    return token_id + shift + layer * stride


def parse_snac(snac_str):
    """Parse SNAC string '# t0 t1 t2 t3 t4 t5 t6 # ...' into list of 7-tuples."""
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


def build_delayed_audio_streams(snac_frames, n_streams=7):
    """Build 7 audio streams with delay pattern.

    Stream i is delayed by i positions. Total length = n_frames + 6.

    Example with 3 frames:
    pos 0: [a0, pad, pad, pad, pad, pad, pad]
    pos 1: [a1, b0,  pad, pad, pad, pad, pad]
    pos 2: [a2, b1,  c0,  pad, pad, pad, pad]
    pos 3: [pad,b2,  c1,  d0,  pad, pad, pad]
    ...
    """
    n_frames = len(snac_frames)
    total_len = n_frames + n_streams - 1  # flush the pipeline

    streams = []
    for i in range(n_streams):
        stream = []
        for pos in range(total_len):
            frame_idx = pos - i  # delay by i
            if 0 <= frame_idx < n_frames:
                stream.append(snac_frames[frame_idx][i])
            else:
                stream.append(_pad_a)
        streams.append(stream)

    return streams, total_len


def build_T1T2(question_tokens, answer_tokens, max_len=2048):
    """Text question → Text answer. No audio output."""
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

    # Loss: text only, on answer portion
    answer_start = 1 + len(question_tokens) + 2
    loss_mask_text = [0] * min(answer_start, seq_len) + [1] * max(0, seq_len - answer_start)
    loss_mask_audio = [0] * seq_len

    return streams, loss_mask_text, loss_mask_audio, "T1T2"


def build_T1A2(question_tokens, snac_frames, max_len=2048):
    """Text question → Audio answer. With delay pattern."""
    # Build delayed audio streams
    delayed_streams, audio_len = build_delayed_audio_streams(snac_frames)

    # Prefix: [_input_t] + question + [_eot, _answer_t/_answer_a]
    prefix_len = 1 + len(question_tokens) + 2  # _input_t + q + _eot + _answer

    # Text stream: prefix + padding during audio
    text_stream = [_input_t] + question_tokens + [_eot, _answer_t] + [_pad_t] * audio_len

    # Audio streams: prefix padding + delayed audio tokens
    streams = []
    for i in range(7):
        audio_prefix = [layershift(_pad_a, i)] * (prefix_len - 1) + [layershift(_answer_a, i)]
        audio_tokens = [layershift(t, i) for t in delayed_streams[i]]
        streams.append(audio_prefix + audio_tokens)
    streams.append(text_stream)

    seq_len = len(streams[0])
    if seq_len > max_len:
        for i in range(8):
            streams[i] = streams[i][:max_len]
        seq_len = max_len

    # Loss: audio only, on delayed portion (after _answer_a)
    loss_mask_text = [0] * seq_len  # no text loss for audio-output tasks
    loss_mask_audio = [0] * min(prefix_len, seq_len) + [1] * max(0, seq_len - prefix_len)

    return streams, loss_mask_text, loss_mask_audio, "T1A2"


def build_A1T2(whisper_len, answer_tokens, max_len=2048):
    """Audio question → Text answer. Whisper features injected at runtime."""
    T = whisper_len

    text_stream = [_input_t] + [_pad_t] * T + [_eot, _answer_t] + answer_tokens + [_eot]

    answer_pad_len = len(answer_tokens) + 1
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

    # Loss: text only, on answer portion
    answer_start = 1 + T + 2
    loss_mask_text = [0] * min(answer_start, seq_len) + [1] * max(0, seq_len - answer_start)
    loss_mask_audio = [0] * seq_len

    return streams, loss_mask_text, loss_mask_audio, "A1T2"


def build_A1A2(whisper_len, snac_frames, max_len=2048):
    """Audio question → Audio answer. Whisper features + delay pattern."""
    T = whisper_len
    delayed_streams, audio_len = build_delayed_audio_streams(snac_frames)

    # Audio input prefix: [_input_a, _pad_a*T, _eoa, _answer_a]
    prefix_len = 1 + T + 2  # _input_a + pad*T + _eoa + _answer_a

    text_stream = [_input_t] + [_pad_t] * T + [_eot, _answer_t] + [_pad_t] * audio_len

    streams = []
    for i in range(7):
        audio_prefix = ([layershift(_input_a, i)]
                        + [layershift(_pad_a, i)] * T
                        + [layershift(_eoa, i), layershift(_answer_a, i)])
        audio_tokens = [layershift(t, i) for t in delayed_streams[i]]
        streams.append(audio_prefix + audio_tokens)
    streams.append(text_stream)

    seq_len = len(streams[0])
    if seq_len > max_len:
        for i in range(8):
            streams[i] = streams[i][:max_len]
        seq_len = max_len

    # Loss: audio only
    loss_mask_text = [0] * seq_len
    loss_mask_audio = [0] * min(prefix_len, seq_len) + [1] * max(0, seq_len - prefix_len)

    return streams, loss_mask_text, loss_mask_audio, "A1A2"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/root/.cache/autoresearch/s3_data")
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

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
            continue

    print(f"Loaded {len(all_samples)} samples")

    if args.max_samples:
        all_samples = all_samples[:args.max_samples]

    random.shuffle(all_samples)

    n_val = max(1, int(len(all_samples) * args.val_fraction))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f"Train: {len(train_samples)}, Val: {n_val}")

    for split_name, samples in [("train", train_samples), ("val", val_samples)]:
        all_streams = []
        skipped = 0

        for idx, sample in enumerate(samples):
            if idx % 5000 == 0:
                print(f"  {split_name}: {idx}/{len(samples)}")

            question = sample['question']
            answer = sample['answer']
            snac_str = sample['answer_snac']

            q_tokens = tokenizer.encode(question, add_special_tokens=False)
            a_tokens = tokenizer.encode(answer, add_special_tokens=False)
            snac_frames = parse_snac(snac_str)

            if len(snac_frames) == 0:
                skipped += 1
                continue

            # T1T2: text → text (for maintaining text quality)
            try:
                s, lt, la, task = build_T1T2(q_tokens, a_tokens, args.max_len)
                all_streams.append((s, lt, la, task))
            except:
                pass

            # T1A2: text → audio (main S3 task)
            try:
                s, lt, la, task = build_T1A2(q_tokens, snac_frames, args.max_len)
                all_streams.append((s, lt, la, task))
            except:
                pass

            # A1T2: audio → text (ASR-like, with estimated whisper length)
            whisper_len = min(len(snac_frames) * 4, 1500)
            try:
                s, lt, la, task = build_A1T2(whisper_len, a_tokens, args.max_len)
                all_streams.append((s, lt, la, task))
            except:
                pass

            # A1A2: audio → audio (full omni)
            try:
                s, lt, la, task = build_A1A2(whisper_len, snac_frames, args.max_len)
                all_streams.append((s, lt, la, task))
            except:
                pass

        print(f"  {split_name}: {len(all_streams)} sequences ({skipped} skipped)")

        output_path = os.path.join(args.output_dir, f"{split_name}.pt")
        save_data = []
        for streams, lm_t, lm_a, task in all_streams:
            seq_len = len(streams[0])
            save_data.append({
                'streams': torch.tensor(streams, dtype=torch.long),
                'loss_mask_text': torch.tensor(lm_t, dtype=torch.bool),
                'loss_mask_audio': torch.tensor(lm_a, dtype=torch.bool),
                'task': task,
                'seq_len': seq_len,
            })

        torch.save(save_data, output_path)
        print(f"  Saved {output_path} ({len(save_data)} sequences)")


if __name__ == "__main__":
    main()
