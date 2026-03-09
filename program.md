# omni-autoresearch

This is an experiment to have the LLM do its own research on omni-model pretraining.
We train a small GPT on interleaved text + audio (SNAC-encoded) tokens and study
how optimizer choices affect multi-modal pretraining quality.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — constants, data prep, tokenizer, audio SNAC encoding, mixed dataloader, evaluation.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards, tokenizer, AND audio token files (`audio/train_tokens.pt`, `audio/val_tokens.pt`). If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Architecture

This is a small **omni model** (text + audio):

- **Text data**: BPE-tokenized text from climbmix shards (8192 text tokens)
- **Audio data**: LibriSpeech speech, SNAC-encoded into discrete tokens (3 codebook levels x 4096 = 12288 audio tokens)
- **Unified vocab**: 20482 tokens total (text BPE + audio delimiters + SNAC codes)
- **Model**: Single GPT transformer trained on interleaved text and audio sequences
- **Audio format**: `<audio_start> [SNAC interleaved tokens] <audio_end>`
- **Mix ratio**: ~30% audio, ~70% text (configurable in prepare.py)

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). Launch: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only after initial setup.
- Install new packages or add dependencies.
- Modify the evaluation harness.

**Primary metric: val_loss** (average cross-entropy in nats on mixed text+audio validation data). Lower is better.
**Secondary metrics**: text_bpb (text bits-per-byte), audio_loss (audio cross-entropy in nats).

**Research focus: OPTIMIZER TWEAKS.** While architecture changes are fair game, the primary research question is how different optimizer configurations affect omni-model pretraining:
- Muon vs AdamW balance for different parameter groups
- Separate LR schedules for text embeddings vs audio embeddings vs shared transformer
- Audio-specific vs text-specific weight decay
- Warmup/cooldown schedule variations
- Momentum and beta configurations
- Loss weighting between modalities (if added)

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_loss gains.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Always establish the baseline first by running train.py as-is.

## Output format

The script prints:

```
---
val_loss:         2.345678
text_bpb:         0.997900
audio_loss:       3.456789
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     12345.6
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Extract key metrics: `grep "^val_loss:\|^text_bpb:\|^audio_loss:\|^peak_vram_mb:" run.log`

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_loss	text_bpb	audio_loss	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_loss (e.g. 2.345678) — use 0.000000 for crashes
3. text_bpb (e.g. 0.997900) — use 0.000000 for crashes
4. audio_loss (e.g. 3.456789) — use 0.000000 for crashes
5. peak memory in GB (.1f) — use 0.0 for crashes
6. status: `keep`, `discard`, or `crash`
7. short description

## The experiment loop

LOOP FOREVER:

1. Look at the git state
2. Tune `train.py` with an experimental idea
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Read results: `grep "^val_loss:\|^text_bpb:\|^audio_loss:\|^peak_vram_mb:" run.log`
6. If grep is empty, run crashed. `tail -n 50 run.log` for stack trace.
7. Record in results.tsv (do NOT commit results.tsv)
8. If val_loss improved (lower), keep the commit
9. If val_loss is equal or worse, git reset back

**Timeout**: Kill runs exceeding 10 minutes.
**Crashes**: Fix typos/imports, skip fundamentally broken ideas.
**NEVER STOP**: Continue indefinitely until manually interrupted.
