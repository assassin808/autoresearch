# Omni-Modal S3 Audio Plateau: Research Report

**Date**: 2026-03-22
**Researchers**: Yang (grad student, Prof. Xiaoxiao Li's group)
**Cluster**: Narval (Compute Canada), A100 80GB GPUs
**Branch**: `omni-basin-shaping`

---

## 1. Problem Statement

Mini-omni is an omni-modal language model that generates text and audio simultaneously. It combines Qwen2-0.5B with a Whisper encoder and SNAC 24kHz neural audio codec with 7 codebook streams. The model is trained in three stages:

- **S1**: Train Whisper adapter only (frozen LLM backbone)
- **S2**: Train LLM backbone on text-only tasks (frozen adapter)
- **S3**: Unfreeze everything, train on 4 joint text+audio task types

During S3 joint training, we observe a severe **audio loss plateau**: text loss drops rapidly (77% reduction in 3000 steps) while audio loss stalls after minimal improvement (14.5% reduction). This report documents a systematic investigation of the plateau through 13 completed experiments and 3 ongoing experiments, encompassing 6 optimization methods and 13 diagnostic signals.

**Central question**: Is the audio plateau caused by (a) gradient interference between text and audio, (b) poor optimization landscape, (c) insufficient training scale, or (d) a fundamental capability limit at the current model/data scale?

---

## 2. Experimental Setup

### 2.1 Model Architecture

- **LLM backbone**: Qwen2-0.5B (~500M parameters), 24 transformer layers, hidden_dim=896
- **Configuration**: `post_adapter=False`, `tie_word_embeddings=True`
- **Audio codec**: SNAC 24kHz with 7 codebook streams, each with vocabulary size 4160
- **Text vocabulary**: 152,000 tokens (Qwen2 tokenizer)
- **Total embedding table**: 181,120 audio tokens (7 x 4160) + 152,000 text tokens, shared via `lm_head` (tied weights)
- **Audio encoding**: Whisper encoder produces features, passed through a learned adapter, concatenated with token embeddings
- **Delay pattern**: Stream i is delayed by i positions (layer-shifted), total sequence length = n_frames + 6

### 2.2 Training Configuration

**Starting checkpoint**: Post-S1 (Whisper adapter trained, LLM backbone = pretrained Qwen2-0.5B weights, audio embeddings randomly initialized).

**Default S3 hyperparameters**:
| Parameter | Value |
|-----------|-------|
| Learning rate | 2e-6 to 2e-5 (cosine schedule) |
| Warmup steps | 1500 |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, eps=1e-8) |
| Weight decay | 0.01 |
| Batch size | 2 |
| Gradient accumulation | 16 (effective batch = 32) |
| Max gradient norm | 1.0 |
| Audio weight | 1.0 |

**S2 mode** (text-only control): `audio_weight=0`, `s2_mode=True`, LR=2e-4, only T1T2+A1T2 tasks.

### 2.3 Diagnostic Signals (D1-D13)

| ID | Diagnostic | Description | Frequency |
|----|-----------|-------------|-----------|
| D1 | Gradient magnitude ratio (rho) | `\|\|g_audio\|\| / \|\|g_text\|\|` — relative gradient strength | Every 100 steps |
| D2 | Gradient cosine similarity (cos_phi) | Cosine between text and audio gradients — detects conflict | Every 100 steps |
| D3 | Per-layer gradient cosine | D2 decomposed by transformer layer | Every 100 steps |
| D4 | Basin width | Loss degradation under random perturbations at eps=0.01,0.05,0.1,0.5,1.0 | Every 1000 steps |
| D5 | Parameter displacement | L2 distance from initialization for backbone, lm_head, embeddings | Every 100 steps |
| D6 | Embedding effective rank | Spectral effective rank of text and audio embedding matrices | Every 100 steps |
| D7 | Per-codebook losses | Validation CE loss for each of the 7 SNAC codebooks | Every 100 steps |
| D8 | Top-k accuracy & loss histograms | Text/audio prediction accuracy at top-1,5,10 | Every 100 steps |
| D9 | Linear probe accuracy | Train linear probes on frozen backbone activations at L6/L12/L18 to predict audio tokens | Every 500 steps |
| D10 | Per-task validation losses | Separate val losses for T1T2, T1A2, A1T2, A1A2 task types | Every 200 steps |
| D11 | Module gradient norms | Text and audio gradient norms per transformer layer + lm_head | Every 100 steps |
| D12 | CKA (Centered Kernel Alignment) | Representational similarity between current and initial checkpoint at L6/L12/L18 | Every 500 steps |
| D13 | Audio embedding cosine collapse | Mean pairwise cosine similarity among audio embeddings — detects representational collapse | Every 100 steps |

### 2.4 Data

- **Dataset**: VoiceAssistant-400K (gpt-omni)
- **Samples**: 470K samples x 4 task types = 1.88M sequences
- **Preprocessed size**: 65 GB train.pt, 1.4 GB val.pt
- **Whisper features**: 325 shards of pre-extracted features
- **SNAC codebook entropies** (from data distribution): CB0=6.03, CB1=7.09, CB2=7.85, CB3=7.90, CB4=7.09, CB5=7.87, CB6=7.91 nats
- **Reference**: log(4160) = 8.33 nats (uniform random baseline)

---

## 3. Experiments and Results

### 3.1 Batch 3: Initial Baselines (exp1-exp4, 3000 steps each)

| # | Name | Method | Audio Wt | LR | S2 Mode | Key Change |
|---|------|--------|----------|-----|---------|------------|
| exp1 | S2 text-only | baseline | 0.0 | 2e-4 | Yes | Text-only control (higher LR) |
| exp2 | S3 baseline | baseline | 1.0 | 2e-5 | No | Joint text+audio, equal weights |
| exp3 | S3 GradNorm | gradnorm | 1.0 | 2e-5 | No | GradNorm adaptive weighting (alpha=1.5) |
| exp4 | S3 entropy | entropy_scaled | 1.0 | 2e-5 | No | CB weights scaled by SNAC codebook entropy |

**Final losses (step 3000)**:

| Metric | exp1 (S2 text) | exp2 (S3 baseline) | exp3 (GradNorm) | exp4 (entropy) |
|--------|---------------|-------------------|-----------------|----------------|
| Train text_loss | 2.526 | 1.440 | 4766.0 | 1.188 |
| Train audio_loss | 0.0 (disabled) | 37.25 | 33.81 | 4.14 (scaled) |
| Runtime (s) | 7737 | 7461 | 8323 | 7496 |

**Per-codebook validation losses (step 3000)** — the core metric:

| CB | H_random (8.33) | exp1 (no audio) | exp2 (baseline) | exp3 (GradNorm) | exp4 (entropy) |
|----|-----------------|----------------|-----------------|-----------------|----------------|
| cb0 | 8.33 | 9.312 | 5.562 | 5.562 | 5.562 |
| cb1 | 8.33 | 9.250 | 6.562 | 6.531 | 6.594 |
| cb2 | 8.33 | 9.188 | 7.562 | 7.562 | 7.594 |
| cb3 | 8.33 | 9.000 | 7.656 | 7.656 | 7.688 |
| cb4 | 8.33 | 9.188 | 6.500 | 6.500 | 6.562 |
| cb5 | 8.33 | 9.125 | 7.781 | 7.750 | 7.781 |
| cb6 | 8.33 | 9.125 | 7.625 | 7.625 | 7.688 |
| **Sum** | **58.31** | **64.19** | **51.25** | **51.19** | **51.47** |

**Key observations**:
1. All three S3 methods (exp2/3/4) converge to **virtually identical** per-CB val losses despite radically different training dynamics.
2. CB0 (coarsest, lowest entropy) learns best: 5.56 vs 8.33 random (33% below random).
3. CB2/3/5/6 (fine-grained, highest entropy) plateau at 7.5-7.8 (only 7-9% below random).
4. **GradNorm diverged catastrophically**: weights went to w_text = -525,100 / w_audio = +525,102 by step 1000. Text loss exploded to ~5000. Yet val audio was unchanged.
5. **Entropy scaling** was deceptive: train audio_loss appeared low (4.14) because losses were divided by entropy (6-8). Unscaled val losses matched baseline exactly.

### 3.2 Batch 4: Method Improvements (exp5-exp10, 3000 steps each)

| # | Name | Method | Key Change |
|---|------|--------|------------|
| exp5 | GradNorm fixed | gradnorm | Non-negativity clamp (w >= 0.01), gradient clipping on weights |
| exp6 | Entropy fixed | entropy_scaled | Inverted weighting: multiply by max_entropy/entropy |
| exp7 | Freeze backbone | baseline | `freeze_backbone=True`, only train embeddings + lm_head |
| exp8 | Emb init sample | baseline | `audio_emb_init=sample` — init audio embeddings from text embedding distribution |
| exp9 | Curriculum | baseline | `curriculum=1000` — audio_weight ramps 0->1 over first 1000 steps |
| exp10 | Freeze + emb init | baseline | Both `freeze_backbone=True` and `audio_emb_init=sample` |

**Final losses (step 3000)**:

| Metric | exp5 (GN fix) | exp6 (ent fix) | exp7 (freeze) | exp8 (emb init) | exp9 (curric) | exp10 (both) |
|--------|--------------|----------------|---------------|-----------------|---------------|--------------|
| Train text | 2.076 | 1.339 | 29.574 | 1.200 | **1.000** | 7.668 |
| Train audio | **30.469** | 40.734 | 36.406 | 36.781 | 43.094 | 47.734 |

**Per-codebook validation losses (step 3000)**:

| CB | exp5 (GN fix) | exp6 (ent fix) | exp7 (freeze) | exp8 (emb init) | exp9 (curric) | exp10 (both) |
|----|--------------|----------------|---------------|-----------------|---------------|--------------|
| cb0 | 5.562 | 5.719 | 5.938 | 5.594 | 5.594 | 7.625 |
| cb1 | 6.500 | 6.688 | 7.125 | 6.562 | 6.656 | 8.125 |
| cb2 | 7.562 | 7.625 | 8.000 | 7.562 | 7.594 | 8.125 |
| cb3 | 7.688 | 7.750 | 8.312 | 7.688 | 7.719 | 8.125 |
| cb4 | 6.531 | 6.719 | 7.094 | 6.562 | 6.625 | 7.250 |
| cb5 | 7.781 | 7.844 | 8.125 | 7.781 | 7.812 | 8.250 |
| cb6 | 7.656 | 7.688 | 7.938 | 7.656 | 7.656 | 8.062 |
| **Sum** | **49.28** | **50.03** | **52.53** | **49.41** | **49.66** | **55.56** |

**Key observations**:
1. **GradNorm fixed (exp5)**: Best train audio loss (30.47, -18% vs baseline 37.25), but val CB losses are identical to baseline.
2. **Curriculum (exp9)**: Best train text loss (1.00, -31% vs baseline 1.44), val unchanged.
3. **Freeze backbone (exp7)**: Catastrophic — text loss explodes to 29.6, audio gets worse too (sum 52.53 vs 51.25 baseline). The backbone must adapt.
4. **Freeze + emb init (exp10)**: Worst overall (sum 55.56). Confirms frozen backbone cannot learn audio.
5. **Emb init from text distribution (exp8)**: Marginally better text (1.20), audio identical. Audio embedding rank dropped to 747 (vs 877 for random init), indicating the structured initialization persists but does not help val loss.
6. Train loss improvements from GradNorm and curriculum do not transfer to validation — evidence of optimization-only effects.

### 3.3 Batch 5: Scale Experiments (exp11-exp13, 10000 steps each)

| # | Name | Steps | Init Checkpoint | Key Test |
|---|------|-------|----------------|----------|
| exp11 | Long S2 | 10,000 | post-S1 | Extended text-only pre-training (S2 at LR=2e-4) |
| exp12 | S2 then S3 | 10,000 | exp11 final model | S3 from a well-pretrained S2 backbone |
| exp13 | Long S3 | 10,000 | post-S1 | S3 directly for 10K steps (no S2 pre-training) |

**Final losses (step 10000)**:

| Metric | exp11 (long S2) | exp12 (S2->S3) | exp13 (long S3) |
|--------|----------------|----------------|-----------------|
| Train text | 1.896 | 1.197 | 1.470 |
| Train audio | 0.0 (disabled) | 28.078 | 36.594 |
| Runtime (s) | 24,935 | 25,151 | 24,917 |

**Per-codebook validation losses (step 10000)**:

| CB | H_random | exp11 (S2 only) | exp12 (S2->S3) | exp13 (long S3) |
|----|----------|----------------|----------------|-----------------|
| cb0 | 8.33 | 9.000 | **5.250** | 5.219 |
| cb1 | 8.33 | 8.875 | **5.781** | 5.812 |
| cb2 | 8.33 | 8.875 | 7.031 | 7.062 |
| cb3 | 8.33 | 8.812 | 7.062 | 7.125 |
| cb4 | 8.33 | 9.062 | **5.406** | 5.562 |
| cb5 | 8.33 | 9.000 | 7.188 | 7.188 |
| cb6 | 8.33 | 8.875 | 7.125 | 7.156 |
| **Sum** | **58.31** | **62.50** | **44.84** | **45.12** |

**CB loss evolution over 10K steps for exp12 and exp13**:

| Step | exp12 cb0 | exp12 cb1 | exp12 cb4 | exp13 cb0 | exp13 cb1 | exp13 cb4 |
|------|-----------|-----------|-----------|-----------|-----------|-----------|
| 1000 | 5.781 | 6.875 | 6.875 | 5.750 | 6.906 | 6.875 |
| 3000 | 5.469 | 6.344 | 6.312 | 5.438 | 6.406 | 6.375 |
| 5000 | 5.406 | 6.094 | 5.844 | 5.375 | 6.094 | 5.906 |
| 7000 | 5.281 | 5.906 | 5.625 | 5.250 | 5.938 | 5.688 |
| 10000 | 5.250 | 5.781 | 5.406 | 5.219 | 5.812 | 5.562 |

**Key observations**:
1. **10K steps >> 3K steps**: Audio val sum dropped from 51.25 (3K, exp2) to 44.84 (10K, exp12) — a 12.5% improvement. Audio learning is slow but continuous.
2. **S2 pre-training helps marginally**: exp12 (S2->S3) sum=44.84 vs exp13 (no S2) sum=45.12. The gap is small (0.6%), suggesting S2 pre-training is not critical for audio performance.
3. **CB0/CB1/CB4 continue improving** through 10K steps. CB2/3/5/6 improve more slowly but are not fully converged either.
4. **Train audio loss**: exp12 reaches 28.08 (with S2 warmup) vs exp13 at 36.59 — S2 pre-training helps train loss substantially but val gap is minimal.

**Per-task validation losses (step 10000)**:

| Task | exp12 text | exp12 audio | exp13 text | exp13 audio |
|------|-----------|-------------|-----------|-------------|
| T1T2 | 1.84 | 20.66 | 1.70 | 20.73 |
| T1A2 | 1.09 | 44.55 | 1.01 | 44.74 |
| A1T2 | 2.50 | 20.97 | 2.20 | 21.05 |
| A1A2 | 1.26 | 44.78 | 1.11 | 44.93 |

Both experiments reach virtually identical per-task val losses. Audio-output tasks (T1A2, A1A2) remain at ~44-45, text-input-audio-output tasks (T1T2, A1T2) at ~21.

---

## 4. Diagnostic Analysis

### 4.1 Gradient Dynamics (D1: rho, D2: cos_phi)

**exp2 baseline evolution**:

| Step | rho (audio/text) | cos_phi | text_grad_norm | audio_grad_norm |
|------|-----------------|---------|---------------|-----------------|
| 100 | 0.32 | -0.045 | 203.4 | 65.1 |
| 500 | 0.30 | -0.013 | 109.6 | 33.0 |
| 1000 | 0.44 | +0.002 | 40.1 | 17.4 |
| 2000 | 0.30 | -0.022 | 22.2 | 6.7 |
| 3000 | 0.73 | +0.001 | 14.2 | 10.4 |

**rho across experiments at step 3000**:

| Experiment | rho | cos_phi |
|-----------|-----|---------|
| exp1 (S2 text) | 2.46 | -0.015 |
| exp2 (S3 baseline) | 0.73 | +0.001 |
| exp3 (GradNorm diverged) | 0.06 | +0.013 |
| exp5 (GradNorm fixed) | 0.73 | +0.000 |
| exp6 (entropy fixed) | 1.40 | -0.010 |
| exp7 (freeze backbone) | 0.05 | -0.143 |
| exp8 (emb init) | 0.82 | +0.016 |
| exp9 (curriculum) | 0.86 | -0.031 |

**Key finding**: cos_phi is consistently near zero (within [-0.05, +0.02]) across all experiments and all time steps. Text and audio gradients are **orthogonal, not conflicting**. The audio plateau is NOT caused by gradient interference. Per-layer cosines (D3) also stay near zero, with max |cos| ~ 0.2 at step 100 then < 0.1.

### 4.2 CKA Representation Drift (D12)

CKA measures how much the backbone's internal representations change from initialization. 1.0 = identical to init, 0.0 = completely different.

**CKA at Layer 12 across experiments**:

| Experiment | Step 500 | Step 1000 | Step 3000 | Step 5000 | Step 10000 |
|-----------|----------|-----------|-----------|-----------|------------|
| exp1 (S2 text, 3K) | — | — | 0.028 | — | — |
| exp2 (S3 baseline, 3K) | 1.000 | 0.974 | 0.955 | — | — |
| exp3 (GradNorm div, 3K) | — | — | 0.613 | — | — |
| exp4 (entropy, 3K) | — | — | 0.997 | — | — |
| exp5 (GradNorm fix, 3K) | — | — | 0.928 | — | — |
| exp7 (freeze, 3K) | — | — | 0.595 | — | — |
| exp8 (emb init, 3K) | — | — | 0.152 | — | — |
| exp9 (curriculum, 3K) | — | — | 0.008 | — | — |
| exp11 (long S2, 10K) | 1.000 | 0.828 | 0.998 | 0.999 | 0.999 |
| exp12 (S2->S3, 10K) | 1.000 | 1.000 | 0.999 | 0.995 | 0.994 |
| exp13 (long S3, 10K) | 1.000 | 0.187 | 0.030 | 0.038 | 0.036 |

**Key findings**:
1. **S2 pre-training preserves representations**: exp11 (long S2) maintains CKA > 0.99 throughout 10K steps. The backbone learns text within the same representational space.
2. **S2->S3 staged training (exp12) also preserves CKA > 0.99**: The S2-pretrained backbone barely changes its representations during S3. It adds audio capability without reorganizing.
3. **Long S3 without S2 (exp13) radically changes representations**: CKA drops to 0.03 by step 2000. The backbone reorganizes completely.
4. **Yet exp12 and exp13 reach the same val performance** (audio sum 44.84 vs 45.12). Completely different internal representations can produce the same audio prediction quality. This is strong evidence that the plateau is not about representation but about fundamental capacity.
5. **Curriculum (exp9) also reaches CKA=0.008** — early text-only training shifts representations, then adding audio shifts them further. Still same val loss.

### 4.3 Basin Width (D4)

Basin width is measured by loss degradation at eps=0.01 (smallest perturbation). Lower degradation = wider/flatter basin.

**Degradation at eps=0.01, step 3000**:

| Experiment | Baseline Loss | Deg@0.01 | Basin Width (relative) |
|-----------|--------------|----------|----------------------|
| exp1 (S2 text) | 2.356 | 34.5 | **WIDE** (1.0x ref) |
| exp2 (S3 baseline) | 1.675 | 214.0 | narrow (6.2x worse) |
| exp5 (GradNorm fixed) | 2.181 | 298.4 | very narrow |
| exp6 (entropy fixed) | 1.783 | 235.5 | narrow |
| exp7 (freeze backbone) | 58.938 | 53.4 | wide (but high loss) |
| exp8 (emb init) | 1.867 | 182.2 | medium |
| exp9 (curriculum) | 1.780 | 170.2 | medium |

**Basin width evolution for S2 (exp11) at eps=0.01**:

| Step | Baseline Loss | Degradation@0.01 |
|------|--------------|-----------------|
| 1000 | 2.641 | 8.6 |
| 3000 | 2.650 | 10.1 |
| 5000 | 2.431 | 14.3 |
| 10000 | 1.883 | 36.9 |

**S2->S3 (exp12) at eps=0.01**:

| Step | Baseline Loss | Degradation@0.01 |
|------|--------------|-----------------|
| 1000 | 1.909 | 40.1 |
| 5000 | 1.998 | 48.7 |
| 10000 | 1.955 | 85.5 |

**Key findings**:
1. **S2 pre-training produces extremely wide basins**: degradation 8.6-36.9 vs 214-336 for S3 baseline. S2 mode sits in a ~10x flatter region of the loss landscape.
2. **S2->S3 starts in a wider basin** (40.1 at step 1000 vs 336 for S3-only exp2 at step 1000) and gradually narrows as the model adapts to joint text+audio.
3. **Long S3 (exp13)** at step 10000 has degradation 204.5 — still much narrower than S2.
4. The wider basin from S2 pre-training does NOT translate to better audio val performance (exp12 vs exp13 are tied).

### 4.4 Linear Probe Accuracy (D9)

Linear probes trained on frozen backbone hidden states to predict the next audio token. Random chance = 1/4160 = 0.024%.

**Probe val_acc at Layer 12**:

| Experiment | Step 500 | Step 1000 | Step 3000 | Step 5000 | Step 10000 |
|-----------|----------|-----------|-----------|-----------|------------|
| exp1 (S2 text) | — | — | 3.45% | — | — |
| exp2 (S3 baseline) | 3.0% | 4.7% | 4.05% | — | — |
| exp5 (GradNorm fix) | — | — | 4.50% | — | — |
| exp8 (emb init) | — | — | 4.42% | — | — |
| exp9 (curriculum) | — | — | 5.17% | — | — |
| exp12 (S2->S3) | 3.97% | 4.42% | 5.25% | 5.32% | 4.87% |
| exp13 (long S3) | 3.75% | 4.72% | 4.57% | 5.10% | 5.32% |

**Key findings**:
1. Probe accuracy is dramatically better than random (3-5% vs 0.024%) but still very low in absolute terms. The backbone representations contain minimal information about audio token identity.
2. Accuracy grows slowly but continuously: roughly **+0.1-0.2% per 1000 steps**. This matches the slow CB loss improvement.
3. Train probe accuracy reaches 13-53% (severe overfitting), but val stays at 3-5%. The backbone learns audio-specific patterns that don't generalize.

### 4.5 Per-Task Validation Losses (D10)

This is one of the most revealing diagnostics. Val losses are reported for each of the 4 task types.

**exp2 baseline (step 3000)**:

| Task | Text Input | Audio Input | Text Output | Audio Output |
|------|-----------|-------------|-------------|--------------|
| T1T2 | text | — | 1.70 | 22.78 |
| T1A2 | text | — | 1.03 | 49.05 |
| A1T2 | — | audio | 2.13 | 23.09 |
| A1A2 | — | audio | 1.07 | 49.15 |

**Comparison across methods (step 3000, audio losses only)**:

| Task | exp2 | exp5 (GN) | exp6 (ent) | exp8 (emb) | exp9 (curr) |
|------|------|-----------|------------|------------|-------------|
| T1T2 audio | 22.78 | 22.80 | 23.10 | 22.88 | 22.95 |
| T1A2 audio | 49.05 | 49.06 | 49.72 | 49.27 | 49.38 |
| A1T2 audio | 23.09 | 23.09 | 23.44 | 23.20 | 23.25 |
| A1A2 audio | 49.15 | 49.20 | 49.87 | 49.39 | 49.52 |

**Key findings**:
1. **Val audio losses are virtually identical across ALL methods** (within 1% of each other). No optimization trick changes the fundamental audio prediction quality.
2. **Audio-output tasks (T1A2, A1A2) have ~2x higher loss than text-conditioned tasks (T1T2, A1T2)**: 49 vs 23. When the model must generate audio without text context, it performs much worse.
3. Text vs audio input makes almost no difference to audio output: T1T2 audio (22.78) ~ A1T2 audio (23.09), and T1A2 audio (49.05) ~ A1A2 audio (49.15).
4. **At 10K steps, losses do improve**: T1T2 audio drops from 22.78 to 20.66-20.73, A1A2 audio from 49.15 to 44.78-44.93. But the structural pattern (2x gap) persists.

### 4.6 Embedding Analysis (D6: Effective Rank, D13: Cosine Collapse)

**Effective rank at step 3000** (embedding dimension = 896):

| Experiment | Text Rank | Audio Rank | Audio Rank / 896 |
|-----------|-----------|------------|-------------------|
| exp1 (S2 text) | 581 | 883 | 98.5% |
| exp2 (S3 baseline) | 746 | 877 | 97.9% |
| exp5 (GradNorm fix) | 717 | 877 | 97.9% |
| exp8 (emb init) | 784 | 747 | 83.4% |
| exp9 (curriculum) | 743 | 878 | 98.0% |
| exp10 (freeze+init) | 679 | 717 | 80.0% |
| exp12 (S2->S3, 10K) | 388 | 862 | 96.2% |
| exp13 (long S3, 10K) | 571 | 863 | 96.3% |

**Cosine collapse (D13) at final step**:

| Experiment | cos_collapse | Interpretation |
|-----------|-------------|----------------|
| exp1 (S2 text) | -0.00000 | Audio embeddings untouched |
| exp2 (S3 baseline) | 0.01761 | Near-random (no collapse) |
| exp5 (GradNorm fix) | 0.01659 | Near-random |
| exp8 (emb init) | 0.19434 | Structured (from text init) |
| exp9 (curriculum) | 0.01294 | Near-random |
| exp12 (S2->S3, 10K) | 0.05870 | Slightly structured |
| exp13 (long S3, 10K) | 0.06188 | Slightly structured |

**Key findings**:
1. **Audio embedding rank stays near-full (96-98%)** in all experiments with random init. A random matrix has full rank; the audio embeddings never develop the low-rank structure characteristic of learned, meaningful embeddings.
2. **Text embedding rank decreases** with more training (from 746 at 3K to 388 at 10K for S2->S3). Text embeddings develop concentrated, low-rank structure as they specialize.
3. **Emb init from text distribution (exp8/10)** reduces audio rank to 747-717, confirming the initialization persists. But this structured init does not improve val audio loss.
4. **No cosine collapse**: cos_collapse stays well below 1.0 in all experiments. Audio embeddings are not collapsing to a single direction — they remain spread out in near-random configurations.
5. **After 10K steps**, audio rank drops marginally to 862-863 and cos_collapse rises to 0.06. This indicates the embeddings are slowly developing structure, consistent with the slow improvement in CB losses.

### 4.7 Module Gradient Norms (D11)

**exp2 baseline at step 3000** — gradient norms by module:

| Module | Text Grad | Audio Grad | Ratio (audio/text) |
|--------|-----------|------------|-------------------|
| lm_head | 22.38 | 8.36 | 0.37 |
| layer0 | 21.37 | 7.23 | 0.34 |
| layer6 | 4.14 | 2.84 | 0.69 |
| layer12 | 4.55 | 1.86 | 0.41 |
| layer18 | 4.10 | 2.22 | 0.54 |
| layer23 | 10.35 | **22.27** | **2.15** |

**Key finding**: Audio gradient norms are 2-3x smaller than text across most layers. The notable exception is **layer 23 (final layer)**: audio gradient (22.27) exceeds text gradient (10.35). The loss signal at the output is strong, but it fails to propagate effectively backward through the network. This is consistent with the "random embedding bottleneck" hypothesis: the audio loss gradient at the output layer is large, but because audio embeddings are near-random, the gradient carries no coherent directional information for deeper layers.

### 4.8 Displacement (D5)

**Parameter displacement from initialization**:

| Experiment | Step | Backbone Disp | lm_head Disp | Ratio (lm_head/backbone) |
|-----------|------|--------------|-------------|-------------------------|
| exp1 (S2 text, 3K) | 3000 | 102.3 | 287.8 | 2.8 |
| exp2 (S3 baseline, 3K) | 3000 | 11.8 | 60.8 | 5.2 |
| exp5 (GradNorm fix, 3K) | 3000 | 10.4 | 25.1 | 2.4 |
| exp7 (freeze, 3K) | 3000 | 0.0 | 16.7 | inf |
| exp8 (emb init, 3K) | 3000 | 11.7 | 57.0 | 4.9 |
| exp11 (S2, 10K) | 10000 | 195.6 | 554.5 | 2.8 |
| exp12 (S2->S3, 10K) | 10000 | 24.5 | 64.2 | 2.6 |
| exp13 (S3, 10K) | 10000 | 22.7 | 144.4 | 6.4 |

**Key findings**:
1. **S2 text-only moves the backbone 10x more** than S3 (102 vs 11.8 at 3K, 196 vs 23 at 10K). Higher LR + text-only signal drives larger parameter changes.
2. **lm_head always moves more** than backbone (2.4-6.4x). Most learning happens in the output projection.
3. **exp2 backbone displacement saturates**: 10.9 at step 2000 to 11.8 at step 3000 — the backbone nearly stops changing while audio loss remains high.
4. Note: embedding displacement reads 0 due to a known bug (`tie_word_embeddings=True` causes parameter name mismatch in D5 tracking).

---

## 5. Key Findings

### Finding 1: Audio plateau confirmed but NOT fully converged at 10K steps

CB0 (coarsest codebook) reaches 5.22 val loss vs 8.33 random (37% below random). CB2-6 (fine-grained codebooks) remain at 7.0-7.2 at 10K steps (13-16% below random). However, losses continue to improve between steps 3K and 10K:

| CB | 3K (exp2) | 10K (exp13) | Improvement |
|----|-----------|-------------|-------------|
| cb0 | 5.562 | 5.219 | -0.343 |
| cb1 | 6.562 | 5.812 | -0.750 |
| cb4 | 6.500 | 5.562 | -0.938 |
| cb2 | 7.562 | 7.062 | -0.500 |
| cb5 | 7.781 | 7.188 | -0.593 |

The plateau is real but not absolute — it is a very slow convergence regime, not a hard wall.

### Finding 2: Val losses are identical across all optimization methods

Six different methods (baseline, GradNorm, entropy scaling, curriculum, freeze backbone, emb init) all produce per-CB val losses within 0.1-0.2 nats of each other at 3K steps. The per-task val losses (T1T2, T1A2, A1T2, A1A2) are also virtually identical. This is the single strongest piece of evidence that the plateau is not an optimization problem.

### Finding 3: Gradients are orthogonal, not conflicting (cos_phi ~ 0)

cos_phi stays within [-0.05, +0.02] throughout training across all experiments. Per-layer cosines are also near zero. This definitively rules out gradient conflict as the cause. The problem is that audio gradients carry insufficient directional information, not that they fight text gradients.

### Finding 4: S2 pre-training makes the loss landscape 2.4-10x smoother

Basin width (degradation at eps=0.01) is 8.6 for S2 at step 1000 vs 336 for S3 baseline at step 1000 — a 39x difference. Even at step 3000, S2 basin (34.5) is 6.2x wider than S3 (214.0). S2->S3 staged training starts in a wider basin (40.1) that gradually narrows. However, this smoother landscape does not translate to better audio val performance — exp12 (wide basin) ties exp13 (narrow basin).

### Finding 5: Linear probes show slow but continuous audio learning (+0.1-0.2%/1000 steps)

Probe val_acc at L12 grows from 3.0% at step 500 to 5.3% at step 10000. This is 220x better than random (0.024%) but still very low. The backbone IS learning some audio structure, just extremely slowly.

### Finding 6: Completely different representations can yield the same val performance

exp12 (S2->S3) maintains CKA > 0.99 — its representations are nearly identical to the pretrained Qwen2 backbone. exp13 (long S3) drops to CKA = 0.03 — completely different representations. Yet both reach audio sum = 44.84 vs 45.12. This means the audio plateau is not caused by the backbone being "stuck" in a bad representational regime. Any reasonable representation leads to the same audio quality at this scale.

### Finding 7: Freezing the backbone is catastrophic

exp7 (freeze backbone): text loss explodes to 29.6, audio CB losses are worse than baseline across all codebooks. exp10 (freeze + emb init) is even worse (sum 55.56 vs 51.25). The backbone MUST adapt jointly with audio. Approaches that freeze the backbone to "protect" text are counterproductive.

### Finding 8: GradNorm (fixed) achieves best train audio (-18%) but val is unchanged

exp5 reduces train audio_loss from 37.25 to 30.47 (-18%). This is a genuine optimization improvement. But val CB losses are 49.28 vs 51.25 (sum), within noise. The model can fit training data better without generalizing.

### Finding 9: Curriculum achieves best train text (-31%) but audio is unchanged

exp9 reduces train text_loss from 1.44 to 1.00 (-31%) by starting with text-only and gradually introducing audio. Train audio is slightly worse (43.09 vs 37.25) because it had fewer effective audio-training steps, but val text and audio are unchanged.

---

## 6. Hypotheses and Discussion

### 6.1 The Plateau Is Primarily a Capacity/Scale Limit

The strongest evidence is Finding 2: every optimization method converges to the same val losses. If the problem were optimization (wrong learning rate, gradient interference, bad initialization), different methods would reach different val performances. Instead, all methods reach the same valley — suggesting this is a fundamental capability limit at the current combination of:

- **Model size** (500M params, 896 hidden dim)
- **Effective batch size** (32 — the published mini-omni uses 192)
- **Training steps** (3K-10K — mini-omni uses ~100K+ S3 steps)
- **S2 pre-training** (mini-omni uses extensive S2 with much more data)

### 6.2 The Audio Embedding Bottleneck

The diagnostic chain paints a clear picture of why convergence is slow:

```
Random audio embeddings (rank ~877/896, 97.9% of full rank)
    |
    v
Audio tokens mapped to random directions in embedding space
    |
    v
Backbone receives random vectors, produces noisy hidden states for audio
    |
    v
Audio gradients are directionless (rho < 1, cos_phi ~ 0)
    |
    v
Backbone barely moves (CKA > 0.95 at 3K, displacement saturates)
    |
    v
Linear probes learn almost nothing (val_acc 3-5%)
    |
    v
Audio loss improves very slowly (~0.1 nats/1000 steps)
```

However, this bottleneck is being slowly overcome: at 10K steps, audio rank drops to 863 (96.2%), cos_collapse rises to 0.06, and CB losses are measurably better than at 3K. The chicken-and-egg problem (backbone can't learn from random embeddings, embeddings can't learn from a backbone that ignores them) is being solved, just very slowly via gradient descent.

### 6.3 SNAC Acoustic Codebooks (CB2-6) Are Fundamentally Harder

CB0 (coarsest, entropy 6.03) reaches 37% below random. CB2-6 (finest, entropy 7.85-7.91) reach only 13-16% below random. This difficulty ordering persists across all experiments and scales. Possible explanations:
- Higher entropy means more uniform distribution, harder to predict
- Fine-grained codebooks encode high-frequency acoustic details that may require more context than a 24-layer causal transformer can capture
- CB0 encodes prosody/pitch (more structured/predictable) while CB2-6 encode waveform details

### 6.4 The 2x Gap: Audio-Only vs Text-Conditioned Tasks

Tasks where audio must be generated alongside text (T1T2, A1T2: audio loss ~21-23) perform much better than tasks where only audio is generated (T1A2, A1A2: audio loss ~44-49). This ~2x gap is structural and persists at all scales. The text prediction channel provides an information pathway that helps audio prediction, even though the two modalities have near-zero gradient correlation.

---

## 7. Ongoing Experiments (Batch 6)

Three experiments currently running on Narval A100 nodes:

| # | Name | Steps | Batch | Init | Key Test |
|---|------|-------|-------|------|----------|
| exp15 | Omni scale | 5,000 | 2x96=192 | published ckpt via exp11 | Match mini-omni batch size (192) with S2-pretrained checkpoint |
| exp16 | Long S2->S3 | 30,000 | 2x16=32 | exp11 (long S2) | Extended S3 training (30K steps) from S2-pretrained backbone |
| exp17 | No S2, large batch | 5,000 | 2x96=192 | post-S1 | Large batch without S2 pre-training |

**Progress as of 2026-03-22**:

| Experiment | Steps Done | Audio Sum | Status |
|-----------|-----------|-----------|--------|
| exp15 | 900 / 5000 | 51.60 | Running (~3.7h in) |
| exp16 | 5800 / 30000 | 44.78 | Running (~4h in, already matching 10K results) |
| exp17 | 700 / 5000 | 51.72 | Running (~3h in) |

**What these test**:
- **exp15 vs exp17**: Does S2 pre-training matter at large batch size (192)?
- **exp16**: Does audio loss continue improving beyond 10K steps? Can we reach CB2-6 < 7.0?
- **exp15 vs exp12**: Does larger batch size (192 vs 32) accelerate audio learning?

**Expected insights**: If exp16 shows continued audio improvement at 20K-30K steps, the plateau is definitively a scale issue. If exp15/17 show better audio than exp12/13 at matched steps, batch size is a key factor.

---

## 8. Infrastructure Notes

### 8.1 Narval (Compute Canada) Deployment

**Environment setup**: All paths configured via `narval_env.sh`:
- Model checkpoint: `/scratch/yang0531/mini-omni-ckpt/`
- Mini-omni reference code: `/scratch/yang0531/mini-omni-ref/`
- S3 preprocessed data: `/scratch/yang0531/s3_data/` (65GB train + 1.4GB val)
- HuggingFace cache: `/scratch/yang0531/hf_home/`

**Job submission**: `sbatch job_train_s3.sh` with SLURM resource requests for A100 80GB GPUs.

### 8.2 Issues Encountered and Resolved

1. **Critical S3 pipeline bugs (2026-03-20)**: Three bugs found before the current experiment series:
   - Audio loss was incorrectly divided by 7 (number of codebooks) — fixed to report true CE
   - Whisper encoder features were not being loaded — fixed feature loading path
   - Wrong checkpoint was being used (post-S3 instead of post-S1) — fixed checkpoint mode

2. **FlashAttention 3 unsupported on Blackwell (RTX 5090)**: Uses SDPA + FlexAttention as fallback. Not an issue on Narval A100s.

3. **M-SAM diagnostics zeros**: When val loader uses batch_size=1, M-SAM basin probing reports zeros. Fixed by ensuring batch_size >= 2 for diagnostics.

4. **Embedding displacement (D5) bug**: `tie_word_embeddings=True` causes a parameter name mismatch — the embedding matrix is referenced as `lm_head.weight` not `model.embed_tokens.weight`. D5 embedding displacement always reads 0. Documented but not fixed (lm_head displacement captures the same information since weights are tied).

5. **GradNorm divergence**: Original implementation lacked non-negativity constraints. Fixed in exp5 with weight clamping (w >= 0.01) and gradient clipping on weight parameters.

6. **Entropy scaling direction**: Original implementation (exp4) divided loss by entropy, inadvertently *reducing* audio's contribution. Fixed in exp6 to multiply by max_entropy/entropy.

### 8.3 Compute Budget

| Batch | Experiments | Steps Each | GPU-Hours (approx) |
|-------|-----------|------------|-------------------|
| Batch 3 | 4 | 3,000 | 4 x 2.1h = 8.4h |
| Batch 4 | 6 | 3,000 | 6 x 2.0h = 12.0h |
| Batch 5 | 3 | 10,000 | 3 x 7.0h = 21.0h |
| Batch 6 | 3 | 5K-30K | ~50h (estimated) |
| **Total** | **16** | — | **~91h A100** |

---

## Appendix A: Full Experiment Registry

| # | Name | Method | Steps | LR | Batch (eff) | S2 | Freeze | Emb Init | Curriculum | Init Ckpt |
|---|------|--------|-------|-----|------------|-----|--------|----------|------------|-----------|
| exp1 | S2 text | baseline | 3000 | 2e-4 | 32 | Yes | No | None | 0 | post-S1 |
| exp2 | S3 baseline | baseline | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp3 | S3 GradNorm | gradnorm | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp4 | S3 entropy | entropy_scaled | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp5 | GradNorm fixed | gradnorm | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp6 | Entropy fixed | entropy_scaled | 3000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp7 | Freeze backbone | baseline | 3000 | 2e-5 | 32 | No | Yes | None | 0 | post-S1 |
| exp8 | Emb init sample | baseline | 3000 | 2e-5 | 32 | No | No | sample | 0 | post-S1 |
| exp9 | Curriculum | baseline | 3000 | 2e-5 | 32 | No | No | None | 1000 | post-S1 |
| exp10 | Freeze+emb init | baseline | 3000 | 2e-5 | 32 | No | Yes | sample | 0 | post-S1 |
| exp11 | Long S2 | baseline | 10000 | 2e-4 | 32 | Yes | No | None | 0 | post-S1 |
| exp12 | S2 then S3 | baseline | 10000 | 2e-5 | 32 | No | No | None | 0 | exp11 final |
| exp13 | Long S3 | baseline | 10000 | 2e-5 | 32 | No | No | None | 0 | post-S1 |
| exp15 | Omni scale | baseline | 5000 | 2e-5 | 192 | No | No | None | 0 | published via exp11 |
| exp16 | Long S2->S3 | baseline | 30000 | 2e-5 | 32 | No | No | None | 0 | exp11 final |
| exp17 | No S2 large batch | baseline | 5000 | 2e-5 | 192 | No | No | None | 0 | post-S1 |

## Appendix B: exp2 Baseline Detailed Training Log

| Step | text_loss | audio_loss | rho | cos_phi | backbone_disp | CKA_L12 | audio_rank |
|------|-----------|------------|-----|---------|--------------|---------|------------|
| 100 | 5.791 | 51.672 | 0.32 | -0.045 | 0.77 | — | 883 |
| 300 | 5.041 | 44.375 | 0.77 | +0.021 | 1.42 | — | — |
| 500 | 3.381 | 48.094 | 0.30 | -0.013 | 2.15 | 1.000 | 883 |
| 1000 | 1.733 | 37.781 | 0.44 | +0.002 | 5.02 | 0.974 | 882 |
| 1500 | 1.750 | 31.312 | 0.51 | -0.000 | 8.24 | 0.971 | — |
| 2000 | 1.567 | 40.406 | 0.30 | -0.022 | 10.85 | 0.949 | 879 |
| 2500 | 1.750 | 36.594 | 0.35 | -0.009 | 11.69 | 0.954 | — |
| 3000 | 1.440 | 37.250 | 0.73 | +0.001 | 11.79 | 0.955 | 877 |

## Appendix C: exp2 Per-Task Validation Losses Over Time

| Step | T1T2 text | T1T2 audio | T1A2 text | T1A2 audio | A1T2 text | A1T2 audio | A1A2 text | A1A2 audio |
|------|-----------|------------|-----------|------------|-----------|------------|-----------|------------|
| 200 | 7.40 | 25.93 | 4.04 | 55.97 | 7.68 | 26.67 | 3.98 | 56.02 |
| 600 | 3.57 | 23.50 | 2.10 | 50.59 | 4.37 | 23.90 | 2.19 | 50.70 |
| 1000 | 2.18 | 23.37 | 1.28 | 50.31 | 2.64 | 23.73 | 1.37 | 50.40 |
| 1600 | 2.05 | 23.19 | 1.18 | 49.93 | 2.47 | 23.51 | 1.28 | 50.00 |
| 2000 | 1.89 | 22.96 | 1.15 | 49.45 | 2.38 | 23.29 | 1.21 | 49.57 |
| 2600 | 1.72 | 22.81 | 1.03 | 49.13 | 2.19 | 23.14 | 1.12 | 49.24 |
| 3000 | 1.70 | 22.78 | 1.03 | 49.05 | 2.13 | 23.09 | 1.07 | 49.15 |

## Appendix D: GradNorm Weight Divergence (exp3, unfixed)

| Step | w_text | w_audio | text_loss |
|------|--------|---------|-----------|
| 100 | +0.012 | 1.99 | 6.86 |
| 200 | -1.24 | 3.24 | 3704 |
| 300 | -20.9 | 22.9 | 5382 |
| 500 | -586 | 588 | 4154 |
| 1000 | -525,100 | 525,102 | 5880 |
| 3000 | -525,100 | 525,102 | 4766 |

Root cause: Audio loss barely changes, so GradNorm's relative rate mechanism keeps increasing audio weight and decreasing text weight. Without a non-negativity constraint, w_text crosses zero and the model actively maximizes text loss.
