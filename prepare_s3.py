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


def find_parquet_files():
    """Find all parquet files in the VA400K blobs directory."""
    blob_dir = os.environ.get("VA400K_BLOBS", "/workspace/.hf_home/hub/datasets--gpt-omni--VoiceAssistant-400K/blobs")
    blobs = sorted(glob.glob(f"{blob_dir}/*"))
    parquet_files = []
    for b in blobs:
        try:
            pq.read_metadata(b)
            parquet_files.append(b)
        except:
            pass
    return parquet_files


def extract_whisper_features(args):
    """Extract Whisper encoder features from question_audio in VA400K.

    Reads parquet files, decodes audio bytes, runs through Whisper encoder,
    saves features as float16 shards to $S3_DATA_DIR/whisper_features/.
    """
    import io
    import soundfile as sf
    import whisper

    output_dir = os.path.join(args.output_dir, "whisper_features")
    os.makedirs(output_dir, exist_ok=True)

    # Load Whisper model
    model_dir = os.environ.get("WHISPER_MODEL_DIR", None)
    print(f"Loading Whisper 'small' model (download_root={model_dir})...")
    model = whisper.load_model("small", download_root=model_dir)
    model.eval()
    device = next(model.parameters()).device
    print(f"Whisper model on {device}")

    parquet_files = find_parquet_files()
    print(f"Found {len(parquet_files)} parquet files")

    total_extracted = 0
    total_skipped = 0

    for pf_idx, pf in enumerate(parquet_files):
        shard_path = os.path.join(output_dir, f"shard_{pf_idx:04d}.pt")

        # Resume support: skip existing shards
        if os.path.exists(shard_path):
            existing = torch.load(shard_path, weights_only=False)
            total_extracted += len(existing)
            print(f"  Shard {pf_idx}: skip (exists, {len(existing)} samples)")
            del existing
            continue

        try:
            table = pq.read_table(pf, columns=['question_audio'])
        except Exception:
            continue
        data = table.to_pydict()
        del table

        shard_features = []
        for i in range(len(data['question_audio'])):
            audio_entry = data['question_audio'][i]
            if audio_entry is None:
                total_skipped += 1
                shard_features.append(None)
                continue

            try:
                # HF audio format: dict with 'bytes' and 'path'
                if isinstance(audio_entry, dict):
                    audio_bytes = audio_entry.get('bytes', None)
                    if audio_bytes is None:
                        total_skipped += 1
                        shard_features.append(None)
                        continue
                else:
                    audio_bytes = audio_entry

                # Decode audio
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)  # mono

                # Resample to 16kHz if needed
                if sr != 16000:
                    import torchaudio.functional as AF
                    audio_tensor = torch.from_numpy(audio_data).float()
                    audio_data = AF.resample(audio_tensor, sr, 16000).numpy()

                # Whisper preprocessing + encode
                audio_data = audio_data.astype(np.float32)
                audio_padded = whisper.pad_or_trim(audio_data)
                mel = whisper.log_mel_spectrogram(audio_padded).to(device)
                with torch.no_grad():
                    features = model.encoder(mel.unsqueeze(0))  # (1, T, 768)
                features = features.squeeze(0)  # (T, 768)

                # Compute actual length (before padding) in Whisper frames
                # Whisper: 30s max, 1500 frames. Audio at 16kHz.
                n_audio_samples = len(audio_data) if isinstance(audio_data, np.ndarray) else audio_data.shape[0]
                # Each Whisper frame = 20ms = 320 samples at 16kHz
                actual_frames = min(n_audio_samples // 320, 1500)

                shard_features.append({
                    'features': features.cpu().half(),  # (T, 768) float16
                    'leng': actual_frames,
                })
                total_extracted += 1
            except Exception as e:
                total_skipped += 1
                shard_features.append(None)
                if total_skipped <= 5:
                    print(f"  Warning: failed to process sample {i} in shard {pf_idx}: {e}")

        torch.save(shard_features, shard_path)
        del data, shard_features

        if (pf_idx + 1) % 10 == 0 or pf_idx == len(parquet_files) - 1:
            print(f"  Shard {pf_idx+1}/{len(parquet_files)}: "
                  f"{total_extracted} extracted, {total_skipped} skipped")

    print(f"\nWhisper extraction complete: {total_extracted} features, "
          f"{total_skipped} skipped")
    print(f"Saved to {output_dir}/")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=os.environ.get("S3_DATA_DIR", "/root/.cache/autoresearch/s3_data"))
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract_whisper", action="store_true",
                        help="Extract Whisper encoder features (GPU required, ~2h)")
    args = parser.parse_args()

    if args.extract_whisper:
        extract_whisper_features(args)
        return

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tok_path = os.environ.get("QWEN2_TOKENIZER", "Qwen/Qwen2-0.5B")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    parquet_files = find_parquet_files()
    print(f"Found {len(parquet_files)} parquet files")

    # Load whisper feature shards (if available)
    whisper_dir = os.path.join(args.output_dir, "whisper_features")
    whisper_shards = {}
    has_whisper = os.path.isdir(whisper_dir)
    if has_whisper:
        shard_files = sorted(glob.glob(os.path.join(whisper_dir, "shard_*.pt")))
        for sf_path in shard_files:
            # Extract shard index from filename: shard_0042.pt -> 42
            shard_idx = int(os.path.basename(sf_path).split("_")[1].split(".")[0])
            whisper_shards[shard_idx] = torch.load(sf_path, weights_only=False)
        print(f"Loaded {len(whisper_shards)} whisper feature shards")
    else:
        print("No whisper features found — A1T2/A1A2 will use estimated lengths")

    # Stream processing: read parquet files one at a time, tokenize immediately,
    # only keep compact tokenized results in memory (not raw text/audio strings)
    all_streams = []
    skipped = 0
    total_raw = 0

    for pf_idx, pf in enumerate(parquet_files):
        try:
            table = pq.read_table(pf, columns=['question', 'answer', 'answer_snac'])
        except Exception as e:
            continue
        data = table.to_pydict()
        del table  # free parquet memory immediately

        # Get whisper features for this shard (if available)
        shard_feats = whisper_shards.get(pf_idx, None)

        for i in range(len(data['question'])):
            total_raw += 1
            question = data['question'][i]
            answer = data['answer'][i]
            snac_str = data['answer_snac'][i]

            q_tokens = tokenizer.encode(question, add_special_tokens=False)
            a_tokens = tokenizer.encode(answer, add_special_tokens=False)
            snac_frames = parse_snac(snac_str)

            if len(snac_frames) == 0:
                skipped += 1
                continue

            # Determine whisper length for A1 tasks
            whisper_feat = None
            if shard_feats is not None and i < len(shard_feats) and shard_feats[i] is not None:
                whisper_feat = shard_feats[i]
                whisper_len = whisper_feat['leng']
            else:
                whisper_len = min(len(snac_frames) * 4, 1500)  # fallback estimate

            # whisper_ref: (shard_idx, row_idx) for loading features at training time
            whisper_ref = (pf_idx, i) if whisper_feat is not None else None

            for builder, builder_args, task_name in [
                (build_T1T2, (q_tokens, a_tokens, args.max_len), "T1T2"),
                (build_T1A2, (q_tokens, snac_frames, args.max_len), "T1A2"),
                (build_A1T2, (whisper_len, a_tokens, args.max_len), "A1T2"),
                (build_A1A2, (whisper_len, snac_frames, args.max_len), "A1A2"),
            ]:
                try:
                    s, lt, la, task = builder(*builder_args)
                    sample = {
                        'streams': torch.tensor(s, dtype=torch.long),
                        'loss_mask_text': torch.tensor(lt, dtype=torch.bool),
                        'loss_mask_audio': torch.tensor(la, dtype=torch.bool),
                        'task': task,
                        'seq_len': len(s[0]),
                    }
                    # Store whisper reference and length for A1 tasks
                    if task_name in ('A1T2', 'A1A2'):
                        sample['whisper_ref'] = whisper_ref
                        sample['whisper_len'] = whisper_len
                    else:
                        sample['whisper_ref'] = None
                        sample['whisper_len'] = 0
                    all_streams.append(sample)
                except:
                    pass

        del data
        if (pf_idx + 1) % 20 == 0:
            print(f"  Processed {pf_idx+1}/{len(parquet_files)} files, "
                  f"{total_raw} samples, {len(all_streams)} sequences")

    print(f"Total: {total_raw} samples → {len(all_streams)} sequences ({skipped} skipped)")

    if args.max_samples:
        all_streams = all_streams[:args.max_samples * 4]

    random.shuffle(all_streams)

    n_val = max(1, int(len(all_streams) * args.val_fraction))
    val_data = all_streams[:n_val]
    train_data = all_streams[n_val:]
    print(f"Train: {len(train_data)}, Val: {n_val}")

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        output_path = os.path.join(args.output_dir, f"{split_name}.pt")
        torch.save(split_data, output_path)
        print(f"  Saved {output_path} ({len(split_data)} sequences)")


if __name__ == "__main__":
    main()
