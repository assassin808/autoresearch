# Setup & Pipeline Guide (for new server / new Claude Code instance)

**Last updated**: 2026-03-19
**Branch**: `omni-basin-shaping`

---

## 1. Project Overview

We study **omni-model pretraining quality** on `gpt-omni/mini-omni` (Qwen2-0.5B + Whisper + SNAC codec, real-time speech-to-speech). The focus is on understanding the **audio plateau** problem: audio val loss plateaus at ~1.89 during S3 (joint text+audio) training and won't go lower regardless of optimizer changes.

### Model Architecture
- **Base LLM**: Qwen2-0.5B (24 layers, 896 dim, 14 heads, 2 KV groups)
- **Audio codec**: SNAC 24kHz (7 codebook streams per timestep, 4160 vocab per stream)
- **Audio adapter**: Whisper encoder → LLaMA MLP adapter → LLM
- **Vocab**: 152000 text + 7×4160 audio = 181120 total (unified embedding table)
- **Config**: `tie_word_embeddings=True` (wte and lm_head share weights)

### Training Stages (mini-omni original)
- **S1**: Train whisper_adapter only (freeze LLM). Data: LibriSpeech audio.
- **S2**: Train LLM backbone (freeze adapter). Data: Open-Orca (text QA) + Moss (audio QA with text output). LR: 2e-4.
- **S3**: Train everything (all unfrozen). Data: VoiceAssistant-400K (4 task types). LR: 2e-5.

### Our Experiments
- **Batch 1** (S3 optimizer): baseline Adam, lambda=3, grad_proj → all same audio plateau
- **Batch 2** (gradient dynamics): CB weighting, M-SAM, skip-S2, text-only S2 → see `EXPERIMENTS_S3_BATCH2.md`

---

## 2. Environment Setup

### Hardware
- GPU: NVIDIA RTX 5090, 32GB VRAM
- **Important**: FA3 (FlashAttention 3) is unsupported on Blackwell architecture. Model uses SDPA + FlexAttention.

### Software
```
Python 3.12.12
PyTorch 2.10.0+cu128 (CUDA 12.8)
transformers 5.3.0
pyarrow 23.0.1
```

**Python path**: `/usr/bin/python3` (NOT the uv-managed python3, which lacks torch)

### Key Directories

| Path | What | Size | Persistent? |
|------|------|------|-------------|
| `/workspace/autoresearch/` | Our code (git repo) | ~1MB | Yes (git) |
| `/workspace/mini-omni-ref/` | mini-omni source (model code) | ~5MB | Needs git clone |
| `/workspace/mini-omni-ckpt/` | Published mini-omni checkpoint | 2.6GB | Needs download |
| `/workspace/.hf_home/` | HuggingFace cache (datasets, models) | 41GB | Needs re-download |
| `/root/.cache/autoresearch/s3_data/` | Preprocessed S3 training data | 12GB | Needs re-generate |
| `/root/.cache/huggingface/token` | HF auth token | tiny | Needs re-set |

---

## 3. Full Rebuild Steps (new server)

### Step 1: Clone repos
```bash
cd /workspace
git clone https://github.com/assassin808/autoresearch.git
cd autoresearch
git checkout omni-basin-shaping

# Model code dependency (litgpt model architecture)
git clone https://github.com/gpt-omni/mini-omni.git /workspace/mini-omni-ref
```

### Step 2: Set HuggingFace token
```bash
mkdir -p /root/.cache/huggingface
echo "hf_YOUR_TOKEN_HERE" > /root/.cache/huggingface/token
# Or: huggingface-cli login
```

The token is needed for downloading VoiceAssistant-400K (gated dataset).

### Step 3: Download mini-omni checkpoint (~2.6GB, ~5min)
```bash
export HF_HOME=/workspace/.hf_home
/usr/bin/python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('gpt-omni/mini-omni', local_dir='/workspace/mini-omni-ckpt')
"
```

The checkpoint contains:
- `lit_model.pth` — model weights (all 3 stages trained)
- `model_config.yaml` — litgpt config (tie_word_embeddings=True, post_adapter=False)

### Step 4: Download VoiceAssistant-400K dataset (~40GB, ~30min)
```bash
export HF_HOME=/workspace/.hf_home
/usr/bin/python3 -c "
from datasets import load_dataset
ds = load_dataset('gpt-omni/VoiceAssistant-400K', cache_dir='/workspace/.hf_home')
print(f'Loaded {len(ds[\"train\"])} samples')
"
```

This downloads parquet files to `/workspace/.hf_home/hub/datasets--gpt-omni--VoiceAssistant-400K/blobs/`.

The dataset has ~470K QA pairs, each with:
- `question`: text question
- `answer`: text answer
- `answer_snac`: SNAC-encoded audio answer (format: `# t0 t1 t2 t3 t4 t5 t6 # t0 t1 ...`)

### Step 5: Download Qwen2-0.5B (needed for tokenizer + Exp C)
```bash
export HF_HOME=/workspace/.hf_home
/usr/bin/python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('Qwen/Qwen2-0.5B', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2-0.5B')
print('Downloaded Qwen2-0.5B')
"
```

### Step 6: Prepare S3 training data (~12GB output, ~20min)
```bash
cd /workspace/autoresearch
/usr/bin/python3 prepare_s3.py --output_dir /root/.cache/autoresearch/s3_data
```

This reads VoiceAssistant-400K parquet blobs, tokenizes with Qwen2 tokenizer, and builds 4 task types per sample:

| Task | Input | Output | Loss on |
|------|-------|--------|---------|
| T1T2 | text question | text answer | text tokens |
| T1A2 | text question | audio answer (SNAC delayed) | audio tokens |
| A1T2 | audio question (whisper placeholder) | text answer | text tokens |
| A1A2 | audio question (whisper placeholder) | audio answer (SNAC delayed) | audio tokens |

Output format: `train.pt` and `val.pt` — list of dicts, each with:
- `streams`: `(8, seq_len)` int64 tensor — 7 audio streams + 1 text stream
- `loss_mask_text`: `(seq_len,)` bool — which positions have text loss
- `loss_mask_audio`: `(seq_len,)` bool — which positions have audio loss
- `task`: string (T1T2/T1A2/A1T2/A1A2)

**Data split**: 98% train, 2% val (seed=42).
**Expected counts**: ~460K train sequences (~115K QA pairs × 4 tasks), ~9.4K val sequences.

### Step 7: Verify everything works
```bash
cd /workspace/autoresearch
/usr/bin/python3 -c "
import torch, sys
sys.path.insert(0, '/workspace/mini-omni-ref')
from litgpt.config import Config
from litgpt.model import GPT

# Load model
config = Config.from_file('/workspace/mini-omni-ckpt/model_config.yaml')
model = GPT(config)
ckpt = torch.load('/workspace/mini-omni-ckpt/lit_model.pth', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt, strict=True)
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')

# Load data
data = torch.load('/root/.cache/autoresearch/s3_data/val.pt', weights_only=False)
print(f'Val data: {len(data)} sequences')
from collections import Counter
tasks = Counter(d['task'] for d in data)
print(f'Tasks: {dict(tasks)}')
print('All good!')
"
```

---

## 4. Running Experiments

### Basic S3 training (baseline)
```bash
cd /workspace/autoresearch
/usr/bin/python3 -u train_s3.py --output_dir results/s3_adam --max_steps 3000 --no_save_model
```

### All Batch 2 experiments (sequential, ~12.5h)
```bash
bash run_new_experiments.sh
```

### Individual experiments
```bash
P="/usr/bin/python3 -u"

# Exp A: CB weighting
$P train_s3.py --output_dir results/s3_cbweight --max_steps 3000 \
  --cb_weights "100,10,1,1,10,1,1" --no_save_model

# Exp B: M-SAM
$P train_s3.py --output_dir results/s3_msam --max_steps 3000 \
  --method m_sam --sam_rho 0.05 --no_save_model

# Exp C: Skip S2
$P train_s3.py --output_dir results/s3_skip_s2 --max_steps 3000 \
  --checkpoint_mode post_s1 --lr_max 2e-4 --no_save_model

# Exp D: Authentic S2 (text-only)
$P train_s3.py --output_dir results/s2_textonly --max_steps 3000 \
  --s2_mode --no_save_model
```

### Analyzing results
```bash
$P analyze_diagnostics.py  # compare diagnostics across experiments
```

---

## 5. Code Architecture

### Core files

| File | Purpose |
|------|---------|
| `train_s3.py` | **Main training script**. S3 training with 4 method variants (baseline, grad_proj, m_sam, adaptive_lambda), checkpoint modes (published, post_s1), s2_mode, CB weighting. Full D1-D7 diagnostics. |
| `train_observe.py` | S2 observation training (earlier experiment). Contains `load_qwen2_weights()` with QKV interleaving logic. |
| `prepare_s3.py` | Preprocesses VoiceAssistant-400K → S3 training data with SNAC delay pattern. |
| `prepare_s2.py` | Preprocesses VoiceAssistant-400K → S2 training data (no delay pattern). |
| `prepare.py` | Downloads climbmix data + trains BPE tokenizer (for earlier LLM-only experiments, not used in omni work). |
| `analyze_diagnostics.py` | Loads diagnostics.json files and prints comparison tables. |
| `eval_s3.py` | Evaluation script for S3 models. |
| `run_new_experiments.sh` | Sequential runner for Batch 2 experiments. |

### Model code (external dependency)
```
/workspace/mini-omni-ref/litgpt/
  config.py   — Config dataclass (all model hyperparameters)
  model.py    — GPT class (transformer + 7 audio heads + whisper adapter)
```

The model is imported as:
```python
sys.path.insert(0, "/workspace/mini-omni-ref")
from litgpt.config import Config
from litgpt.model import GPT
```

### Data format

**8-stream parallel token format** (mini-omni's approach):
- Streams 0-6: audio codebook tokens (layershifted to non-overlapping vocab ranges)
- Stream 7: text tokens
- Each position has at most one active loss (text OR audio, never both)

**Layershift formula**: `shifted_id = raw_audio_id + 152000 + stream_index * 4160`

**SNAC delay pattern** (S3 only, not S2):
- Stream i is delayed by i positions
- Total length = n_frames + 6 (flush pipeline)
- This is how mini-omni handles the hierarchical codebook structure

---

## 6. Diagnostic Signals (D1-D7)

Computed every `diag_every` steps (default 100) on val data:

| ID | Signal | What it measures |
|----|--------|-----------------|
| D1 | ρ(t) = \|\|∇L_audio\|\| / \|\|∇L_text\|\| | Audio gradient relative strength |
| D2 | cos φ(t) = ⟨∇text, ∇audio⟩ / (\|\|∇text\|\| · \|\|∇audio\|\|) | Gradient interference (-ve = conflict) |
| D2' | Per-layer cos φ | Where interference happens (Layer 0 = embedding) |
| D4 | Basin width probing | Loss sensitivity to random perturbations |
| D5 | Parameter displacement | How far model has moved from init |
| D6 | Embedding effective rank | Text vs audio embedding diversity |
| D7 | Per-CB losses | Individual codebook performance |

**How diagnostics work**: Two separate forward+backward passes on the same val batch — one with text loss only, one with audio loss only. Compare the resulting gradients.

---

## 7. Key Findings So Far

See `EXPERIMENTS_S3_BATCH2.md` for full data tables. Summary:

1. **Audio plateau is structural, not caused by S2**: Text-only training (ExpD) produces gradient interference (cos φ = -0.30) and degrades all CB val losses. The text-audio conflict exists in the shared embedding table.

2. **S2 helps audio**: Without S2 (ExpC), text gradients explode (344 vs 11), drowning audio (ρ = 0.01). S2 reduces text gradient magnitude.

3. **Layer 0 (embedding) is the interference hotspot**: cos φ = -0.29 at layer 0, all other layers near 0.

4. **CB weighting changes interference but doesn't solve plateau**: Makes cos φ positive but hurts fine codebooks.

5. **M-SAM finds flatter region but same audio loss**: Plateau is loss landscape structure, not sharp minima.

---

## 8. Known Issues / TODO

### Bugs to fix
1. **M-SAM diagnostics zeros**: Val loader uses batch_size=1 (same as training), but diagnostics need both text and audio masks active in same batch. Fix: use separate val_loader with larger batch for diagnostics.
2. **Embedding displacement zeros**: With `tie_word_embeddings=True`, the wte parameter name check fails. Fix: check actual parameter names and handle tied weights.

### Git push info
```bash
git remote add myfork https://github.com/assassin808/autoresearch.git  # if not already set
git push myfork omni-basin-shaping
```

---

## 9. File Manifest

### In git (will survive server change)
```
train_s3.py, train_observe.py, train.py          # training scripts
prepare.py, prepare_s2.py, prepare_s3.py          # data prep scripts
analyze_diagnostics.py, eval_s3.py                 # analysis/eval
run_new_experiments.sh, run_s3_experiments.sh       # runners
EXPERIMENTS_S3_BATCH2.md                           # Batch 2 experiment report
SETUP_AND_PIPELINE.md                              # This file
FINDINGS.md, FINDINGS_combined.md, FINDINGS_obs1.md # Earlier findings
LITERATURE_SURVEY.md, LITERATURE_NOTES.md           # Literature review
REPORT_s3_dynamics.md, SUMMARY.md, PLAN.md          # Various reports
results/s3_adam/         diagnostics.json, config.json, train.log
results/s2_textonly/     diagnostics.json, config.json, train.log
results/s3_cbweight/     diagnostics.json, config.json, train.log
results/s3_skip_s2/      diagnostics.json, config.json, train.log
results/s3_msam/         diagnostics.json, config.json, train.log
```

### NOT in git (need rebuild on new server)
```
/workspace/mini-omni-ref/             # git clone https://github.com/gpt-omni/mini-omni.git
/workspace/mini-omni-ckpt/            # huggingface_hub.snapshot_download('gpt-omni/mini-omni')
/workspace/.hf_home/                  # HF cache (VoiceAssistant-400K + Qwen2-0.5B + SNAC)
/root/.cache/autoresearch/s3_data/    # prepare_s3.py output (train.pt 12GB + val.pt 231MB)
/root/.cache/huggingface/token        # HF auth token

results/s3_gradproj/                  # Batch 1 experiment (not committed)
results/s3_lambda3/                   # Batch 1 experiment (not committed)
results/obs_1/                        # Early observation experiment
results/obs_2_textonly/               # Early observation experiment
```

### Rebuild time estimate
| Step | Time |
|------|------|
| git clone (2 repos) | 1 min |
| Download mini-omni ckpt (2.6GB) | 5 min |
| Download VoiceAssistant-400K (40GB) | 30 min |
| Download Qwen2-0.5B (~1GB) | 2 min |
| Prepare S3 data (prepare_s3.py) | 20 min |
| **Total** | **~1 hour** |
