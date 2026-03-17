# Option B: Authentic S3 Training from Mini-Omni Checkpoint

## Background

Mini-omni has 3 training stages:
- **S1**: Train adapters only (Whisper MLP, TTS adapter) on ASR data. LLM frozen.
- **S2**: Train LLM on text-output tasks. Adapters frozen. NO audio output.
- **S3**: Train everything on all tasks including audio output. LR=2e-6→2e-5.

**Their checkpoint (`lit_model.pth`)** is the FINAL model after S1+S2+S3.
They do NOT publish intermediate S1 or S2 checkpoints.

## The Problem

We want to do authentic S3 training but we don't have a S2 checkpoint.

### Sub-options

**B1: Fine-tune their final checkpoint further (easiest)**
- Start from their S3-complete model
- Apply our optimizer improvements
- Pro: Working model as starting point
- Con: Not authentic S3 — model already converged

**B2: Use their checkpoint for adapter weights, reset LLM to Qwen2 (most authentic)**
- Extract trained adapter weights from lit_model.pth
- Load Qwen2-0.5B for the LLM backbone
- This simulates "end of S1" state
- Then do S2 (text-only tasks) → S3 (full tasks)
- Pro: Most authentic replication
- Con: Need to do S2 ourselves (~12h+)

**B3: Use their final checkpoint, do continued S3 with more data/longer (practical)**
- Start from their final checkpoint
- Continue S3 training with our data + optimizer methods
- Compare Adam vs our methods on same starting point
- Pro: Fair comparison of optimizer improvements
- Con: Starting from an already-trained model

### Recommendation: B2 for research, B3 for quick results

Do B3 first (quick, get working results in hours), then B2 for the paper.

---

## Plan: Option B3 (Quick Path)

### Step 1: Download checkpoint (~15 min)
```
huggingface-cli download gpt-omni/mini-omni --local-dir /workspace/mini-omni-ckpt
```
Files: lit_model.pth (2.78GB), model_config.yaml, tokenizer

### Step 2: Verify checkpoint loads correctly (~30 min)
- Load lit_model.pth into our GPT model (with post_adapter=True if needed)
- Verify text generation works (T1T2 task)
- Verify audio generation produces coherent SNAC tokens
- Measure baseline val losses on our VoiceAssistant-400K data

### Step 3: Fix data pipeline for authentic S3 (~2h)

**S3 data composition (matching theirs):**
| Dataset | Size | Task | Audio output? | Source |
|---------|------|------|--------------|--------|
| VoiceAssistant-400K | 88K (cached) | Voice QA (T1/A1→T2+A2) | Yes | HF cache |
| Text-only QA subset | 20K (from VA-400K text) | T1→T2 | No | Extract |
| ASR subset | 10K (from VA-400K) | A1→T2 | No | Extract |

**Task distribution per batch (matching their ~1:1 text:audio):**
- 50% text-output tasks (T1T2, A1T2)
- 50% audio-output tasks (T1A2, A1A2)

**Critical fixes needed:**
1. Add Whisper feature extraction for audio-input tasks
2. Apply SNAC delay pattern (stream i shifted by i+1 positions)
3. Compute text loss and audio loss SEPARATELY (mask text targets with -100 for audio tasks, mask audio targets with -100 for text tasks)
4. Use their lower LR: 2e-6→2e-5 (not 4e-4!)

### Step 4: S3 baseline training with Adam (~2-3h)
- AdamW, cosine LR 2e-6→2e-5
- Batch 2 × grad_accum 16 = effective 32
- 3000 steps (~2h)
- Log text_loss, audio_loss, per-codebook losses
- Save checkpoints every 500 steps

### Step 5: S3 with our optimizer improvements (~2-3h each)

| Exp | Optimizer | Key change |
|-----|-----------|-----------|
| s3_adam | AdamW (baseline) | Their exact config |
| s3_coupledWD | AdamW + coupled WD | WD decays with LR |
| s3_lowerWD | AdamW + WD=0.005 | Less regularization |
| s3_cb_weight | AdamW + CB hierarchy | CB1=3x, CB2=1.5x, CB3=1x loss weight |
| s3_muon_hybrid | Muon backbone + Adam embed | Our MuonHybrid from sweep43 |

### Step 6: Evaluation
- Text quality: val text loss + generate sample responses
- Audio quality: val audio loss per codebook + generate SNAC → decode to wav
- Compare optimizer methods on same starting checkpoint

---

## Plan: Option B2 (Authentic Path — for paper)

### Step 1: Extract adapter weights from final checkpoint
```python
# Load lit_model.pth
# Extract: whisper_adapter.*, post_adapter.*, post_adapter_audio_*
# These are the S1-trained components
```

### Step 2: Build S1-complete checkpoint
- Qwen2-0.5B weights for LLM backbone (transformer.h, transformer.wte, lm_head)
- Trained adapter weights from their checkpoint
- Audio portion of wte/lm_head: randomly initialized (or from their checkpoint)

### Step 3: S2 — Text adaptation (~4-6h)
**Data:** Text-output tasks only
- ASR: A1→T2 (use Whisper features)
- Text QA: T1→T2 (from VoiceAssistant-400K question→answer)
- Audio QA with text output: A1→T2

**Config:**
- LR: 2e-6→2e-4 (their S2 LR)
- Train: LLM backbone only (freeze adapters)
- Loss: Text CE only (audio streams masked)
- Steps: ~5000-10000

### Step 4: S3 — Full fine-tuning (~4-6h)
Same as B3 Step 4-5 above, but starting from our own S2 checkpoint.

### Step 5: Compare
- B2-Adam vs B2-OurMethod on same S2 starting point
- Also compare B2 vs B3 to see if starting point matters

---

## Option C (marked, not focused)

Full replication from scratch:
1. Download LibriTTS (586h), VCTK (44h) for S1
2. Download Open-Orca (2M), Moss-002-sft (1.5M) for S2
3. Train S1→S2→S3 from Qwen2-0.5B
4. Requires ~days of GPU time, massive data downloads
5. **Not our focus** — we're researching optimizers, not data pipelines

---

## Critical Engineering Fixes (apply to both B2 and B3)

### Fix 1: Separate text/audio loss computation
```python
# Text tasks (T1T2, A1T2): compute text CE, mask audio with -100
# Audio tasks (T1A2, A1A2): compute audio CE (all 7 heads), mask text with -100
# This matches GitHub issue #101 and #135
```

### Fix 2: SNAC delay pattern
```python
# Stream i is delayed by (i+1) positions relative to text stream
# When constructing training sequences:
# stream_0 tokens start 1 position after text _answer token
# stream_1 tokens start 2 positions after
# ...
# stream_6 tokens start 7 positions after
```

### Fix 3: Whisper feature injection
```python
# For audio-input tasks (A1T2, A1A2):
# 1. Run Whisper encoder on question audio
# 2. Pass through whisper_adapter MLP
# 3. Replace padding positions in embedding space
```

### Fix 4: Post-adapter architecture
```python
# Their model has post_adapter=True:
# - 6 additional transformer blocks after main backbone
# - Separate audio LM head (29120 outputs = 7×4160)
# - Text uses main lm_head, audio uses post_adapter_audio_lm_head
# Our current code doesn't use post_adapter — need to enable it
```

### Fix 5: Loss weighting for codebook hierarchy
```python
# From our obs_1 findings + Moshi's 100x semantic weight:
# CB1 (stream 0): coarsest, semantic → weight higher
# CB2 (streams 1,4): medium → weight medium
# CB3 (streams 2,3,5,6): finest → weight lowest
# SNAC mapping: [CB1, CB2, CB3, CB3, CB2, CB3, CB3]
```

---

## Known Challenges (from literature search)

1. **S3 is MUCH harder than S2** — audio generation requires 3 orders of magnitude more compute
2. **Discrete Representation Inconsistency (DRI)** — same audio → different SNAC tokens depending on context. Only 47% consistency in EnCodec, likely similar in SNAC.
3. **Catastrophic forgetting** — audio training degrades text quality. Need 50% text-only batches (Moshi's finding, matches mini-omni's approach).
4. **Single speaker is easier** — multi-speaker causes repetition and dropped words.
5. **Padding token dominance** — reduce loss weight on padding tokens.

## Timeline

| Task | Duration | Priority |
|------|----------|----------|
| Download checkpoint + verify | 1h | Immediate |
| Fix data pipeline (delays, loss masking, Whisper) | 3h | High |
| B3: Adam baseline S3 | 2-3h | High |
| B3: Optimizer comparison experiments | 6-8h | High |
| B2: S2 training | 4-6h | Medium |
| B2: S3 from own S2 | 4-6h | Medium |
| Option C | Days | Not focused |
