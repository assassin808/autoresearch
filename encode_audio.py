"""Standalone SNAC audio encoder — produces train_tokens.pt and val_tokens.pt."""
import os, sys, torch, torchaudio, soundfile as sf, shutil
from pathlib import Path

AUDIO_DIR = os.path.expanduser("~/.cache/autoresearch/audio")
LS_DIR = os.path.join(AUDIO_DIR, "LibriSpeech")
TEXT_VOCAB_SIZE = 8192
AUDIO_START_ID = TEXT_VOCAB_SIZE
AUDIO_END_ID = TEXT_VOCAB_SIZE + 1
SNAC_OFFSET = TEXT_VOCAB_SIZE + 2
SNAC_CODEBOOK_SIZE = 4096

def flatten(codes):
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

def download_extract(url, split_dir):
    if os.path.exists(split_dir):
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

def encode(name, flacs, snac_model):
    print(f"Encoding {name} ({len(flacs)} files)...")
    docs, secs, errs = [], 0.0, 0
    for i, fp in enumerate(flacs):
        try:
            wav, sr = sf.read(str(fp))
            wav = torch.tensor(wav, dtype=torch.float32)
            if sr != 24000:
                wav = torchaudio.functional.resample(wav, sr, 24000)
            dur = len(wav) / 24000
            if dur < 1.0 or dur > 25.0:
                continue
            with torch.no_grad():
                codes = snac_model.encode(wav.unsqueeze(0).unsqueeze(0).cuda())
            doc = [AUDIO_START_ID] + flatten(codes) + [AUDIO_END_ID]
            docs.append(doc)
            secs += dur
        except Exception:
            errs += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(flacs)} ({secs/3600:.1f}h, {errs} errors)")
    print(f"  Done: {len(docs)} docs, {secs/3600:.1f}h")
    return docs

def main():
    train_path = os.path.join(AUDIO_DIR, "train_tokens.pt")
    val_path = os.path.join(AUDIO_DIR, "val_tokens.pt")
    if os.path.exists(train_path) and os.path.exists(val_path):
        t = torch.load(train_path)
        v = torch.load(val_path)
        print(f"Already done: {len(t)} train, {len(v)} val docs")
        return

    import snac
    model = snac.SNAC.from_pretrained("hubertsiuzdak/snac_24khz").cuda().eval()
    base = "https://www.openslr.org/resources/12"

    # Training: train-clean-100
    train_docs = []
    for name, url in [("train-clean-100", f"{base}/train-clean-100.tar.gz")]:
        d = os.path.join(LS_DIR, name)
        download_extract(url, d)
        flacs = sorted(Path(d).rglob("*.flac"))
        train_docs.extend(encode(name, flacs, model))

    # Validation: dev-clean
    d = os.path.join(LS_DIR, "dev-clean")
    download_extract(f"{base}/dev-clean.tar.gz", d)
    val_docs = encode("dev-clean", sorted(Path(d).rglob("*.flac")), model)

    torch.save(train_docs, train_path)
    torch.save(val_docs, val_path)
    print(f"Saved: {len(train_docs)} train ({sum(len(d) for d in train_docs):,} tok)")
    print(f"Saved: {len(val_docs)} val ({sum(len(d) for d in val_docs):,} tok)")

if __name__ == "__main__":
    main()
