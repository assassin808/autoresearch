#!/usr/bin/python3
"""
Analyze per-codebook token entropy in S3 training data.

For each of the 7 SNAC audio codebooks, computes:
- Empirical entropy H_cb
- Normalized entropy (1.0 = uniform)
- Top-10 most frequent tokens
- Suggested loss scale = H_cb (for entropy-weighted CB loss)

Usage:
    /usr/bin/python3 analyze_codebook_entropy.py
    /usr/bin/python3 analyze_codebook_entropy.py --data_path /path/to/train.pt
"""

import os
import math
import torch
import numpy as np

# Constants from prepare_s3.py
PADDED_TEXT_VOCAB = 152000
PADDED_AUDIO_VOCAB = 4160
N_CODEBOOKS = 7


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Codebook entropy analysis")
    default_dir = os.environ.get("S3_DATA_DIR", "/root/.cache/autoresearch/s3_data")
    parser.add_argument("--data_path", default=os.path.join(default_dir, "train.pt"),
                        help="Path to train.pt")
    args = parser.parse_args()

    print(f"Loading {args.data_path} ...")
    data = torch.load(args.data_path, weights_only=False)
    print(f"Loaded {len(data)} sequences\n")

    # Count tokens per codebook (only where loss_mask_audio is True)
    counts = [np.zeros(PADDED_AUDIO_VOCAB, dtype=np.int64) for _ in range(N_CODEBOOKS)]
    total_tokens = [0] * N_CODEBOOKS
    audio_samples = 0

    for sample in data:
        mask = sample['loss_mask_audio']  # (seq_len,) bool
        if not mask.any():
            continue
        audio_samples += 1
        streams = sample['streams']  # (8, seq_len) — streams 0-6 are audio, 7 is text

        for i in range(N_CODEBOOKS):
            audio_stream = streams[i]  # layershifted tokens
            masked_tokens = audio_stream[mask]  # only answer positions

            # Un-layershift: raw_token - 152000 - i*4160
            raw = masked_tokens.long() - PADDED_TEXT_VOCAB - i * PADDED_AUDIO_VOCAB
            # Clamp to valid range [0, 4159]
            raw = raw.clamp(0, PADDED_AUDIO_VOCAB - 1)

            for t in raw.numpy():
                counts[i][t] += 1
            total_tokens[i] += len(raw)

    print(f"Audio-output samples analyzed: {audio_samples}")
    print(f"Total audio tokens per codebook: {total_tokens[0]:,}\n")

    H_max = math.log(PADDED_AUDIO_VOCAB)  # log(4160)

    # Header
    print(f"{'CB':>3}  {'H_cb':>8}  {'H_max':>8}  {'H_norm':>7}  {'Uniq':>6}  {'Suggested':>10}")
    print(f"{'':>3}  {'(nats)':>8}  {'(nats)':>8}  {'':>7}  {'tokens':>6}  {'loss_scale':>10}")
    print("-" * 60)

    entropies = []
    for i in range(N_CODEBOOKS):
        c = counts[i]
        total = total_tokens[i]
        if total == 0:
            print(f"  {i}  -- no tokens --")
            entropies.append(0.0)
            continue

        p = c / total
        # Filter out zero entries for entropy calc
        p_nonzero = p[p > 0]
        H = -np.sum(p_nonzero * np.log(p_nonzero))
        H_norm = H / H_max
        n_unique = int(np.sum(c > 0))
        entropies.append(H)

        print(f"  {i}  {H:8.4f}  {H_max:8.4f}  {H_norm:7.4f}  {n_unique:6d}  {H:10.4f}")

    # Top-10 per codebook
    print("\n" + "=" * 60)
    print("Top-10 most frequent tokens per codebook")
    print("=" * 60)
    for i in range(N_CODEBOOKS):
        c = counts[i]
        total = total_tokens[i]
        if total == 0:
            continue
        top_idx = np.argsort(c)[::-1][:10]
        print(f"\n  Codebook {i}:")
        print(f"    {'Rank':>4}  {'Token':>6}  {'Count':>10}  {'Freq%':>7}")
        for rank, idx in enumerate(top_idx):
            freq = 100.0 * c[idx] / total
            print(f"    {rank+1:4d}  {idx:6d}  {c[idx]:10,}  {freq:7.2f}%")

    # Summary: suggested cb_weights for entropy-scaled loss
    print("\n" + "=" * 60)
    print("Suggested cb_weights (entropy-scaled, for --cb_weights flag)")
    print("=" * 60)
    if any(e > 0 for e in entropies):
        # Normalize so max weight = 1.0, then scale
        e_max = max(entropies)
        weights = [e / e_max if e > 0 else 0.0 for e in entropies]
        weight_strs = [f"{w:.3f}" for w in weights]
        print(f"  Raw H:     {','.join(f'{e:.3f}' for e in entropies)}")
        print(f"  Normalized: {','.join(weight_strs)}")
        # Also as integer-ish weights (multiply by 100, round)
        int_weights = [f"{int(round(w * 100))}" for w in weights]
        print(f"  Integer:   {','.join(int_weights)}")


if __name__ == "__main__":
    main()
