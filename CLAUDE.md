# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Autonomous AI research framework (forked from karpathy/autoresearch) repurposed for **omni-modal pretraining** on `gpt-omni/mini-omni`. The current branch (`omni-basin-shaping`) investigates the audio plateau problem in joint text+audio S3 training.

**Model**: Qwen2-0.5B + Whisper encoder + SNAC 24kHz codec (7 codebook streams). ~500M params, single RTX 5090 (32GB).

## Commands

```bash
# Install dependencies (uv package manager)
uv sync

# Original LLM-only pipeline (prepare.py + train.py)
uv run prepare.py              # One-time data prep (~2 min)
uv run train.py                # 5-min training run

# Omni S3 pipeline (uses /usr/bin/python3, NOT uv python)
/usr/bin/python3 prepare_s3.py # Tokenize VoiceAssistant-400K → /root/.cache/autoresearch/s3_data/
/usr/bin/python3 train_s3.py --output_dir results/s3_foo --max_steps 3000 --no_save_model
/usr/bin/python3 train_s3.py --method m_sam --sam_rho 0.05 --cb_weights "100,10,1,1,10,1,1" --output_dir results/s3_bar

# Analysis
/usr/bin/python3 analyze_diagnostics.py results/s3_adam/diagnostics.json results/s3_lambda3/diagnostics.json
/usr/bin/python3 eval_s3.py

# Batch experiment runners
bash run_new_experiments.sh     # Batch 2: 4 experiments (~12.5h)
bash run_s3_experiments.sh      # S3 experiment suite
```

**Critical**: Use `/usr/bin/python3` for S3 scripts (has torch+CUDA). The uv-managed python lacks torch.

## Architecture

### Two Pipelines

1. **LLM-only** (`prepare.py` → `train.py`): Original autoresearch loop. Agent edits `train.py`, runs 5-min experiments, keeps/discards based on val_bpb. See `program.md` for the autonomous loop protocol.

2. **Omni S3** (`prepare_s3.py` → `train_s3.py`): Joint text+audio training with diagnostics D1-D7 (gradient norms, interference angles, basin width, embedding rank, per-codebook losses). Results go to `results/` subdirs as `diagnostics.json` + `config.json` + `train.log`.

### Training Stages (mini-omni)
- **S1**: Whisper adapter only (frozen LLM)
- **S2**: LLM backbone (frozen adapter) — `train_observe.py`
- **S3**: Everything unfrozen, 4 task types (T1T2, T1A2, A1T2, A1A2) — `train_s3.py`

### Key External Dependencies (via env vars, see `narval_env.sh`)
- `$MINI_OMNI_REF` — model architecture code (litgpt). Default: `/workspace/mini-omni-ref/`
- `$MINI_OMNI_CKPT` — published checkpoint (2.6GB). Default: `/workspace/mini-omni-ckpt/`
- `$HF_HOME` or `$VA400K_BLOBS` — HuggingFace cache (datasets)
- `$S3_DATA_DIR` — preprocessed S3 data (12GB). Default: `/root/.cache/autoresearch/s3_data/`

**Narval (Compute Canada)**: `source narval_env.sh` sets all paths to `/scratch/yang0531/...`. GPU jobs via `sbatch job_train_s3.sh`.

### SNAC Audio Format
7 codebook streams per timestep (4160 vocab each), unified embedding table of 181120 tokens. Stream i is delayed by i positions (delay pattern). Total sequence length = n_frames + 6.

## Key Files

- `train_s3.py` — Main S3 training with all CLI options and D1-D7 diagnostics
- `prepare_s3.py` — S3 data tokenization with SNAC delay pattern
- `train_observe.py` — S2 observation training
- `program.md` — Autonomous research loop instructions (human edits this, agent follows it)
- `SETUP_AND_PIPELINE.md` — Full environment rebuild guide for new servers

## Known Issues
- M-SAM diagnostics show zeros when val loader uses batch_size=1
- Embedding displacement (D5) reads zero when `tie_word_embeddings=True` (parameter name mismatch)
- FlashAttention 3 unsupported on Blackwell (RTX 5090) — uses SDPA+FlexAttention instead
