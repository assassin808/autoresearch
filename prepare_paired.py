"""Prepare paired text+audio data from LibriSpeech for omni-model training.

Downloads LibriSpeech, extracts transcripts, SNAC-encodes audio, and creates:
1. ASR pairs: [AUDIO_START] audio_tokens [AUDIO_END] [BOS] text_tokens
2. TTS pairs: [BOS] text_tokens [AUDIO_START] audio_tokens [AUDIO_END]
3. Audio-only: [AUDIO_START] audio_tokens [AUDIO_END] (existing)
4. Text-only: [BOS] text_tokens (existing)

Also creates parallel codebook format for Mini-Omni-style training.
"""
import os
import sys
import torch
import torchaudio
import soundfile as sf
import shutil
from pathlib import Path

AUDIO_DIR = os.path.expanduser("~/.cache/autoresearch/audio")
LS_DIR = os.path.join(AUDIO_DIR, "LibriSpeech")

# Token IDs (must match prepare.py)
TEXT_VOCAB_SIZE = 8192
AUDIO_START_ID = TEXT_VOCAB_SIZE       # 8192
AUDIO_END_ID = TEXT_VOCAB_SIZE + 1     # 8193
SNAC_OFFSET = TEXT_VOCAB_SIZE + 2      # 8194
SNAC_CODEBOOK_SIZE = 4096


def flatten_snac(codes):
    """Flatten SNAC 3-level codes into interleaved token sequence."""
    c0 = codes[0][0].tolist()
    c1 = codes[1][0].tolist()
    c2 = codes[2][0].tolist()
    tokens = []
    for i in range(len(c0)):
        tokens.append(SNAC_OFFSET + 0 * SNAC_CODEBOOK_SIZE + c0[i])
        tokens.append(SNAC_OFFSET + 1 * SNAC_CODEBOOK_SIZE + c1[2*i])
        tokens.append(SNAC_OFFSET + 1 * SNAC_CODEBOOK_SIZE + c1[2*i + 1])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i + 1])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i + 2])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i + 3])
    return tokens


def get_parallel_codebooks(codes):
    """Return separate codebook token lists for parallel-head prediction.

    Returns dict with keys 'L0', 'L1', 'L2' containing token ID lists.
    L0 has N frames, L1 has 2N frames, L2 has 4N frames.
    """
    c0 = [SNAC_OFFSET + 0 * SNAC_CODEBOOK_SIZE + c for c in codes[0][0].tolist()]
    c1 = [SNAC_OFFSET + 1 * SNAC_CODEBOOK_SIZE + c for c in codes[1][0].tolist()]
    c2 = [SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c for c in codes[2][0].tolist()]
    return {'L0': c0, 'L1': c1, 'L2': c2}


def parse_transcripts(split_dir):
    """Parse LibriSpeech .trans.txt files into {utterance_id: text} dict."""
    transcripts = {}
    for trans_file in Path(split_dir).rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text
    return transcripts


def download_extract(url, split_dir):
    """Download and extract a LibriSpeech split."""
    if os.path.exists(split_dir):
        print(f"  Already extracted: {split_dir}")
        return
    tar_path = os.path.join(AUDIO_DIR, os.path.basename(url))
    if not os.path.exists(tar_path):
        print(f"  Downloading {url}...")
        import urllib.request
        urllib.request.urlretrieve(url, tar_path)
    print(f"  Extracting...")
    import tarfile
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(AUDIO_DIR)
    os.remove(tar_path)


def encode_paired(split_name, split_dir, snac_model, tokenizer):
    """Encode a LibriSpeech split into paired text+audio sequences.

    Returns list of dicts with keys:
        'audio_flat': flat SNAC tokens [AUDIO_START, ..., AUDIO_END]
        'audio_codebooks': {'L0': [...], 'L1': [...], 'L2': [...]}
        'text_tokens': BPE-encoded text (without BOS)
        'transcript': raw text
        'duration': audio duration in seconds
    """
    transcripts = parse_transcripts(split_dir)
    flac_files = sorted(Path(split_dir).rglob("*.flac"))

    print(f"Encoding {split_name}: {len(flac_files)} files, {len(transcripts)} transcripts...")

    pairs = []
    total_seconds = 0
    errors = 0

    for i, fpath in enumerate(flac_files):
        utt_id = fpath.stem  # e.g., "1272-128104-0000"

        if utt_id not in transcripts:
            errors += 1
            continue

        try:
            wav, sr = sf.read(str(fpath))
            wav = torch.tensor(wav, dtype=torch.float32)
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)
            duration = len(wav) / 24000
            if duration < 1.0 or duration > 25.0:
                continue

            with torch.no_grad():
                codes = snac_model.encode(wav.unsqueeze(0).unsqueeze(0).cuda())

            flat_tokens = flatten_snac(codes)
            codebooks = get_parallel_codebooks(codes)

            # Encode transcript to BPE tokens
            text = transcripts[utt_id]
            text_tokens = tokenizer.encode(text)

            pairs.append({
                'audio_flat': [AUDIO_START_ID] + flat_tokens + [AUDIO_END_ID],
                'audio_codebooks': codebooks,
                'text_tokens': text_tokens,
                'transcript': text,
                'duration': duration,
            })
            total_seconds += duration

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Warning: {fpath}: {e}")

        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(flac_files)} ({total_seconds/3600:.1f}h, {len(pairs)} pairs)")

    print(f"  Done: {len(pairs)} pairs, {total_seconds/3600:.1f}h, {errors} errors")
    return pairs


def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    paired_train_path = os.path.join(AUDIO_DIR, "paired_train.pt")
    paired_val_path = os.path.join(AUDIO_DIR, "paired_val.pt")

    if os.path.exists(paired_train_path) and os.path.exists(paired_val_path):
        train = torch.load(paired_train_path)
        val = torch.load(paired_val_path)
        print(f"Already done: {len(train)} train, {len(val)} val paired docs")
        return

    # Load SNAC model
    import snac
    snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()

    # Load tokenizer
    sys.path.insert(0, "/workspace/autoresearch")
    from prepare import Tokenizer
    tokenizer = Tokenizer.from_directory()

    base = "https://www.openslr.org/resources/12"

    # Training: train-clean-100 (paired)
    train_pairs = []
    for name, url in [
        ("train-clean-100", f"{base}/train-clean-100.tar.gz"),
    ]:
        d = os.path.join(LS_DIR, name)
        download_extract(url, d)
        pairs = encode_paired(name, d, snac_model, tokenizer)
        train_pairs.extend(pairs)

    # Validation: dev-clean (paired)
    d = os.path.join(LS_DIR, "dev-clean")
    download_extract(f"{base}/dev-clean.tar.gz", d)
    val_pairs = encode_paired("dev-clean", d, snac_model, tokenizer)

    torch.save(train_pairs, paired_train_path)
    torch.save(val_pairs, paired_val_path)

    # Stats
    train_audio_h = sum(p['duration'] for p in train_pairs) / 3600
    val_audio_h = sum(p['duration'] for p in val_pairs) / 3600
    train_text_tok = sum(len(p['text_tokens']) for p in train_pairs)
    train_audio_tok = sum(len(p['audio_flat']) for p in train_pairs)

    print(f"\nPaired data saved:")
    print(f"  Train: {len(train_pairs)} pairs, {train_audio_h:.1f}h audio, "
          f"{train_text_tok:,} text tokens, {train_audio_tok:,} audio tokens")
    print(f"  Val: {len(val_pairs)} pairs, {val_audio_h:.1f}h audio")
    print(f"  Avg text/audio ratio: {train_text_tok/train_audio_tok:.3f}")

    # Cleanup
    del snac_model
    torch.cuda.empty_cache()
    if os.path.exists(LS_DIR):
        shutil.rmtree(LS_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
