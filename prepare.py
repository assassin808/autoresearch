"""
Omni-autoresearch data preparation.
Downloads text data shards, trains BPE tokenizer, downloads audio data,
and SNAC-encodes audio into discrete tokens.

Usage:
    python prepare.py                  # full prep (download + tokenizer + audio)
    python prepare.py --num-shards 8   # download only 8 text shards

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
import random
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed for experiment comparability)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 10 * 524288  # number of tokens for val eval (reduced for mixed eval speed)

# ---------------------------------------------------------------------------
# Text vocabulary
# ---------------------------------------------------------------------------

TEXT_VOCAB_SIZE = 8192
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Audio vocabulary (SNAC 24kHz: 3 codebook levels, 4096 entries each)
# ---------------------------------------------------------------------------

SNAC_CODEBOOK_SIZE = 4096
SNAC_NUM_LEVELS = 3
AUDIO_START_ID = TEXT_VOCAB_SIZE       # 8192 — marks beginning of audio sequence
AUDIO_END_ID = TEXT_VOCAB_SIZE + 1     # 8193 — marks end of audio sequence
SNAC_OFFSET = TEXT_VOCAB_SIZE + 2      # 8194 — first SNAC token
# Level i, code c  ->  token_id = SNAC_OFFSET + i * SNAC_CODEBOOK_SIZE + c
TOTAL_VOCAB_SIZE = SNAC_OFFSET + SNAC_NUM_LEVELS * SNAC_CODEBOOK_SIZE  # 20482

# Audio data settings
AUDIO_TRAIN_HOURS = 115   # LibriSpeech train-clean-100 + dev-other + test-clean + test-other
AUDIO_VAL_HOURS = 5       # LibriSpeech dev-clean
AUDIO_MIX_RATIO = 0.5     # fraction of batch rows that are audio (rest are text)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
AUDIO_DIR = os.path.join(CACHE_DIR, "audio")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

# BPE split pattern (GPT-4 style)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# ---------------------------------------------------------------------------
# Text data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards (1 train + 1 val). Download more data first.")
        sys.exit(1)

    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = TEXT_VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    # Build token_bytes lookup for BPB evaluation (text tokens only)
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    # Extend for audio tokens (AUDIO_START, AUDIO_END, SNAC codes) — all get 0 bytes
    num_audio_tokens = TOTAL_VOCAB_SIZE - TEXT_VOCAB_SIZE
    token_bytes_list.extend([0] * num_audio_tokens)
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes ({len(token_bytes_list)} entries) to {token_bytes_path}")

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: \u4f60\u597d"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (text_vocab={enc.n_vocab}, total_vocab={TOTAL_VOCAB_SIZE})")


# ---------------------------------------------------------------------------
# Audio data preparation (SNAC encoding)
# ---------------------------------------------------------------------------

def snac_flatten_codes(codes):
    """
    Flatten SNAC hierarchical codes into an interleaved token sequence.
    codes: list of 3 tensors [L0(T), L1(2T), L2(4T)]
    Returns: list of token IDs with SNAC offsets applied.

    Interleaving pattern per super-frame (7 tokens):
    [L0_i, L1_2i, L1_2i+1, L2_4i, L2_4i+1, L2_4i+2, L2_4i+3]
    """
    c0, c1, c2 = [c.squeeze(0).tolist() for c in codes]
    n_frames = len(c0)
    tokens = []
    for i in range(n_frames):
        tokens.append(SNAC_OFFSET + 0 * SNAC_CODEBOOK_SIZE + c0[i])
        tokens.append(SNAC_OFFSET + 1 * SNAC_CODEBOOK_SIZE + c1[2*i])
        tokens.append(SNAC_OFFSET + 1 * SNAC_CODEBOOK_SIZE + c1[2*i + 1])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i + 1])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i + 2])
        tokens.append(SNAC_OFFSET + 2 * SNAC_CODEBOOK_SIZE + c2[4*i + 3])
    return tokens


def prepare_audio_data():
    """Download multiple LibriSpeech splits and SNAC-encode to token sequences.
    Processes one split at a time, removing raw files after encoding to save disk.
    Training: train-clean-100 + dev-other + test-clean + test-other (~115h)
    Validation: dev-clean (~5h)
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    train_path = os.path.join(AUDIO_DIR, "train_tokens.pt")
    val_path = os.path.join(AUDIO_DIR, "val_tokens.pt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train_docs = torch.load(train_path)
        val_docs = torch.load(val_path)
        train_tok = sum(len(d) for d in train_docs)
        val_tok = sum(len(d) for d in val_docs)
        print(f"Audio: already encoded — train: {len(train_docs)} docs ({train_tok:,} tok), "
              f"val: {len(val_docs)} docs ({val_tok:,} tok)")
        return

    import torchaudio
    import soundfile as sf
    import shutil
    from pathlib import Path

    # Load SNAC model
    import snac
    snac_model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    target_sr = 24000

    def download_and_extract(url, split_dir):
        """Download and extract a LibriSpeech split."""
        tar_path = os.path.join(AUDIO_DIR, os.path.basename(url))
        if os.path.exists(split_dir):
            print(f"  Already extracted: {split_dir}")
            return
        if not os.path.exists(tar_path):
            print(f"  Downloading {url}...")
            import urllib.request
            urllib.request.urlretrieve(url, tar_path)
        print(f"  Extracting {tar_path}...")
        import tarfile
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(AUDIO_DIR)
        os.remove(tar_path)

    def find_flac_files(root_dir):
        return sorted(Path(root_dir).rglob("*.flac"))

    def encode_split(split_name, flac_files):
        print(f"Audio: encoding {split_name} ({len(flac_files)} files)...")
        docs = []
        total_seconds = 0
        errors = 0
        for i, fpath in enumerate(flac_files):
            try:
                waveform, sr = sf.read(str(fpath))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  Warning: skipping {fpath}: {e}")
                continue
            waveform = torch.tensor(waveform, dtype=torch.float32)
            if sr != target_sr:
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            duration = len(waveform) / target_sr
            if duration < 1.0 or duration > 25.0:
                continue
            with torch.no_grad():
                waveform_gpu = waveform.unsqueeze(0).unsqueeze(0).cuda()
                codes = snac_model.encode(waveform_gpu)
            flat_tokens = snac_flatten_codes(codes)
            doc = [AUDIO_START_ID] + flat_tokens + [AUDIO_END_ID]
            docs.append(doc)
            total_seconds += duration
            if (i + 1) % 2000 == 0:
                print(f"  Encoded {i+1}/{len(flac_files)} clips ({total_seconds/3600:.1f}h)")
        if errors:
            print(f"  Skipped {errors} files due to read errors")
        print(f"Audio {split_name}: {len(docs)} docs, {total_seconds/3600:.1f}h, "
              f"avg {sum(len(d) for d in docs)/max(len(docs),1):.0f} tokens/doc")
        return docs

    ls_base = "https://www.openslr.org/resources/12"
    ls_dir = os.path.join(AUDIO_DIR, "LibriSpeech")

    # Process training splits sequentially, cleaning up raw files after each.
    # train-clean-100 (~100h) + dev-other (~5h) + test-clean (~5h) + test-other (~5h) ≈ 115h
    train_splits = [
        ("train-clean-100", f"{ls_base}/train-clean-100.tar.gz"),
        ("dev-other",       f"{ls_base}/dev-other.tar.gz"),
        ("test-clean",      f"{ls_base}/test-clean.tar.gz"),
        ("test-other",      f"{ls_base}/test-other.tar.gz"),
    ]

    all_train_docs = []
    for split_name, url in train_splits:
        split_dir = os.path.join(ls_dir, split_name)
        print(f"\n--- Processing {split_name} ---")
        download_and_extract(url, split_dir)
        flac_files = find_flac_files(split_dir)
        if flac_files:
            docs = encode_split(split_name, flac_files)
            all_train_docs.extend(docs)
        # Remove raw FLAC files to save disk
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    # Validation: dev-clean
    val_split_dir = os.path.join(ls_dir, "dev-clean")
    print(f"\n--- Processing dev-clean (validation) ---")
    download_and_extract(f"{ls_base}/dev-clean.tar.gz", val_split_dir)
    val_flacs = find_flac_files(val_split_dir)
    val_docs = encode_split("val", val_flacs)
    if os.path.exists(val_split_dir):
        shutil.rmtree(val_split_dir)

    # Cleanup empty dirs
    if os.path.exists(ls_dir):
        shutil.rmtree(ls_dir, ignore_errors=True)

    torch.save(all_train_docs, train_path)
    torch.save(val_docs, val_path)
    train_tok = sum(len(d) for d in all_train_docs)
    val_tok = sum(len(d) for d in val_docs)
    print(f"Audio: saved {len(all_train_docs)} train docs ({train_tok:,} tok)")
    print(f"Audio: saved {len(val_docs)} val docs ({val_tok:,} tok)")

    # Cleanup GPU memory
    del snac_model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper for text. Audio uses raw token IDs."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return TOTAL_VOCAB_SIZE  # return full omni vocab size

    def get_text_vocab_size(self):
        return self.enc.n_vocab  # original text-only vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


# ---------------------------------------------------------------------------
# Text document iterator (unchanged)
# ---------------------------------------------------------------------------

def _text_document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over text document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Audio document iterator
# ---------------------------------------------------------------------------

def _audio_document_batches(split, batch_size=64):
    """Infinite iterator over batches of pre-encoded audio token sequences."""
    if split == "train":
        path = os.path.join(AUDIO_DIR, "train_tokens.pt")
    else:
        path = os.path.join(AUDIO_DIR, "val_tokens.pt")

    docs = torch.load(path)
    assert len(docs) > 0, f"No audio docs found in {path}. Run prepare.py first."
    epoch = 1
    while True:
        random.shuffle(docs)
        for i in range(0, len(docs), batch_size):
            yield docs[i:i+batch_size], epoch
        epoch += 1


# ---------------------------------------------------------------------------
# Mixed dataloader (text + audio)
# ---------------------------------------------------------------------------

def make_dataloader(tokenizer, B, T, split, buffer_size=1000, audio_ratio=AUDIO_MIX_RATIO):
    """
    Mixed text+audio dataloader with best-fit packing.
    audio_ratio: fraction of rows filled with audio documents.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    text_batches = _text_document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    text_buffer = []
    audio_buffer = []
    epoch = 1

    # Check if audio data exists
    audio_path = os.path.join(AUDIO_DIR, "train_tokens.pt" if split == "train" else "val_tokens.pt")
    has_audio = os.path.exists(audio_path)
    if has_audio:
        audio_batches = _audio_document_batches(split)
    else:
        audio_ratio = 0.0  # fallback to text-only if no audio

    num_audio_rows = max(1, int(B * audio_ratio)) if has_audio else 0
    num_text_rows = B - num_audio_rows

    def refill_text_buffer():
        nonlocal epoch
        doc_batch, epoch = next(text_batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        text_buffer.extend(token_lists)

    def refill_audio_buffer():
        if not has_audio:
            return
        doc_batch, _ = next(audio_batches)
        audio_buffer.extend(doc_batch)

    def pack_row(row_idx, doc_buffer, refill_fn):
        """Pack documents into a single row using best-fit."""
        pos = 0
        while pos < row_capacity:
            while len(doc_buffer) < buffer_size:
                refill_fn()

            remaining = row_capacity - pos

            # Find largest doc that fits entirely
            best_idx = -1
            best_len = 0
            for i, doc in enumerate(doc_buffer):
                doc_len = len(doc)
                if doc_len <= remaining and doc_len > best_len:
                    best_idx = i
                    best_len = doc_len

            if best_idx >= 0:
                doc = doc_buffer.pop(best_idx)
                row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                pos += len(doc)
            else:
                # No doc fits — crop shortest to fill remaining
                shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                doc = doc_buffer.pop(shortest_idx)
                row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                pos += remaining

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            if row_idx < num_text_rows:
                pack_row(row_idx, text_buffer, refill_text_buffer)
            else:
                pack_row(row_idx, audio_buffer, refill_audio_buffer)

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_val_loss(model, tokenizer, batch_size):
    """
    Combined validation loss on mixed text+audio data.
    Returns average cross-entropy in nats (lower is better).
    Also returns text_bpb and audio_loss separately for analysis.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)

    total_loss = 0.0
    total_tokens = 0
    text_nats = 0.0
    text_bytes = 0
    audio_nats = 0.0
    audio_tokens = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)

        # Text tokens: have byte count > 0
        nbytes = token_bytes[y_flat]
        text_mask = nbytes > 0
        text_nats += (loss_flat * text_mask).sum().item()
        text_bytes += nbytes.sum().item()

        # Audio tokens: in range [AUDIO_START_ID, TOTAL_VOCAB_SIZE)
        audio_mask = y_flat >= AUDIO_START_ID
        audio_nats += (loss_flat * audio_mask).sum().item()
        audio_tokens += audio_mask.sum().item()

        # Total
        total_loss += loss_flat.sum().item()
        total_tokens += y_flat.numel()

    val_loss = total_loss / total_tokens
    text_bpb = text_nats / (math.log(2) * text_bytes) if text_bytes > 0 else 0.0
    audio_loss = audio_nats / audio_tokens if audio_tokens > 0 else 0.0

    return val_loss, text_bpb, audio_loss


# Keep backward-compatible evaluate_bpb (text-only BPB)
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Text-only BPB evaluation (backward compatible)."""
    val_loss, text_bpb, audio_loss = evaluate_val_loss(model, tokenizer, batch_size)
    return text_bpb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for omni-autoresearch")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of text training shards (-1 = all)")
    parser.add_argument("--download-workers", type=int, default=8, help="Parallel download workers")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio data preparation")
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Vocab layout: text={TEXT_VOCAB_SIZE}, audio_special=2, "
          f"SNAC={SNAC_NUM_LEVELS}x{SNAC_CODEBOOK_SIZE}, total={TOTAL_VOCAB_SIZE}")
    print()

    # Step 1: Download text data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer (includes extended token_bytes for audio)
    train_tokenizer()
    print()

    # Step 3: Prepare audio data
    if not args.skip_audio:
        prepare_audio_data()
        print()

    print("Done! Ready to train omni model.")
