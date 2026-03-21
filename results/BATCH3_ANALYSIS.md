# Batch 3 Analysis: Initial S3 Diagnostic Experiments

**Date**: 2026-03-20
**Branch**: `omni-basin-shaping`
**Checkpoint**: post-S1 (Whisper adapter trained, LLM backbone = pretrained Qwen2-0.5B)

---

## Setup

4 experiments, each 3000 steps from the same post-S1 checkpoint:

| Exp | Name | Method | Audio Weight | LR | Notes |
|-----|------|--------|-------------|-----|-------|
| exp1 | S2 text-only | baseline | 0.0 | 2e-4 | Text-only control (S2 mode, higher LR) |
| exp2 | S3 baseline | baseline | 1.0 | 2e-5 | Joint text+audio, equal weights |
| exp3 | S3 GradNorm | gradnorm | 1.0 | 2e-5 | GradNorm adaptive weighting (alpha=1.5) |
| exp4 | S3 entropy | entropy_scaled | 1.0 | 2e-5 | CB weights scaled by SNAC codebook entropy |

Common: batch_size=2, grad_accum=16, warmup=1500 steps, AdamW (beta 0.9/0.95), weight_decay=0.01, max_grad_norm=1.0.

---

## Results Table

### Final Losses (step 3000)

| Metric | exp1 (S2 text) | exp2 (baseline) | exp3 (GradNorm) | exp4 (entropy) |
|--------|---------------|-----------------|-----------------|----------------|
| **train text_loss** | **2.526** | **1.440** | 4766.0 | **1.188** |
| **train audio_loss** | 0.0 (disabled) | 37.25 | 33.81 | 4.14 (注: entropy-scaled) |
| **val text_loss** | 2.087 | **1.510** | 5399.0 | 1.538 |
| **val audio_loss** | 46.22 | **35.24** | 35.31 | **35.41** |
| runtime (s) | 7737 | 7461 | 8323 | 7496 |

> **Note on exp4 audio_loss**: The train audio_loss of 4.14 is misleadingly low. This is because entropy scaling divides each CB loss by its entropy (6.0--7.9), making the reported training loss ~1/7 of the true CE. The **val audio_loss** (35.41) is computed without entropy scaling and matches exp2 (35.24) -- confirming the method did NOT actually improve audio learning. (实际上 entropy scaling 只是改变了 loss 的显示值，并没有真正改变学习动态)

### Per-Codebook Val Losses (step 3000)

| CB | exp1 (no audio) | exp2 (baseline) | exp3 (GradNorm) | exp4 (entropy) |
|----|----------------|-----------------|-----------------|----------------|
| cb0 | 6.665 | **3.998** | 4.010 | 4.025 |
| cb1 | 6.594 | **4.683** | 4.710 | 4.726 |
| cb2 | 6.580 | **5.459** | 5.466 | 5.473 |
| cb3 | 6.570 | **5.469** | 5.470 | 5.481 |
| cb4 | 6.581 | **4.657** | 4.673 | 4.701 |
| cb5 | 6.541 | **5.496** | 5.501 | 5.509 |
| cb6 | 6.691 | **5.473** | 5.479 | 5.491 |

Key observation: All three S3 experiments (exp2/3/4) converge to nearly **identical** per-CB val losses. The audio plateau is robust to the weighting method. CB0 (coarsest) learns best; CB2/3/5/6 (fine-grained) plateau at ~5.5.

---

## Diagnostic Signals (exp2 Baseline)

### D1: Gradient Magnitude Ratio (rho = ||g_audio|| / ||g_text||)

| Step | rho | text_grad_norm | audio_grad_norm |
|------|-----|---------------|-----------------|
| 100 | 0.32 | 203.4 | 65.1 |
| 500 | 0.30 | 109.6 | 33.0 |
| 1000 | 0.44 | 40.1 | 17.4 |
| 2000 | 0.30 | 22.2 | 6.7 |
| 3000 | 0.73 | 14.2 | 10.4 |

Audio gradients are consistently **smaller** than text gradients (rho < 1 throughout). By step 3000, rho rises to 0.73 but audio_grad_norm has collapsed to 10.4 (vs 203.4 at start) -- the gradient signal is vanishing even though audio loss remains high at 37.

### D2: Gradient Cosine Similarity (cos_phi)

| Step | cos_phi (overall) | Range across layers |
|------|-------------------|---------------------|
| 100 | -0.045 | [-0.22, +0.07] |
| 500 | -0.013 | [-0.11, +0.07] |
| 1000 | +0.002 | [-0.06, +0.05] |
| 2000 | -0.022 | [-0.07, +0.06] |
| 3000 | +0.001 | [-0.04, +0.05] |

cos_phi hovers near **zero** throughout -- text and audio gradients are **orthogonal**, not conflicting. This means the audio plateau is NOT caused by gradient interference (梯度并不冲突，而是正交的). The per-layer cosines also stay near zero (max |cos| ~ 0.2 at step 100, then < 0.1).

### D5: Parameter Displacement

| Step | total | backbone | lm_head |
|------|-------|----------|---------|
| 100 | 1.08 | 0.77 | 0.76 |
| 500 | 9.10 | 2.15 | 8.84 |
| 1000 | 21.0 | 5.02 | 20.4 |
| 2000 | 50.3 | 10.9 | 49.3 |
| 3000 | 61.9 | 11.8 | 60.8 |

The **lm_head moves 5x more** than the backbone (60.8 vs 11.8). Backbone displacement saturates after step 2000 (from 10.9 to 11.8 in the last 1000 steps). The backbone is barely changing -- it stays near the pretrained text-optimized basin.

Note: embedding displacement = 0 throughout (known bug: `tie_word_embeddings=True` causes parameter name mismatch in D5 tracking).

### D6: Embedding Rank (effective rank of embedding matrix)

| Step | text_rank | audio_rank |
|------|-----------|------------|
| 100 | 798 | 883 |
| 500 | 681 | 883 |
| 1000 | 620 | 882 |
| 2000 | 837 | 879 |
| 3000 | 746 | 877 |

Text embedding rank fluctuates (620--838) while **audio embedding rank stays extremely high (~877-883)** and barely drops. For reference, the embedding dimension is 896, so audio_rank/dim ~ 0.98. This means the audio embeddings remain **near-random** (a random matrix has near-full rank). They never develop the low-rank structure that characterizes learned, meaningful embeddings. (音频 embedding 的秩几乎没有下降 -- 它们还是随机初始化的状态)

### D9: Linear Probe Accuracy (audio token prediction from backbone hidden states)

| Step | Layer 6 (val_acc) | Layer 12 (val_acc) | Layer 18 (val_acc) |
|------|-------------------|--------------------|--------------------|
| 500 | 3.4% | 3.0% | 3.2% |
| 1000 | 3.9% | 4.7% | 4.0% |
| 2000 | 4.3% | 4.3% | 4.1% |
| 3000 | **5.0%** | **4.1%** | **3.9%** |

Random chance for 4160 audio vocab = 0.024%. So probes do slightly better than random, but accuracy is terrible (3-5%). The backbone representations contain almost **no information** about which audio token should come next. Training accuracy reaches 13% at layer 18 (overfitting to train set) but val stays at 4%.

### D10: Per-Task Validation Losses

| Step | T1T2 text | T1T2 audio | T1A2 text | T1A2 audio | A1T2 text | A1T2 audio | A1A2 text | A1A2 audio |
|------|-----------|------------|-----------|------------|-----------|------------|-----------|------------|
| 1000 | 2.18 | 23.4 | 1.28 | 50.3 | 2.64 | 23.7 | 1.37 | 50.4 |
| 2000 | 1.89 | 23.0 | 1.15 | 49.4 | 2.38 | 23.3 | 1.21 | 49.6 |
| 3000 | **1.70** | **22.8** | **1.03** | **49.1** | **2.13** | **23.1** | **1.07** | **49.1** |

Key patterns:
- **Text loss improves** steadily across all 4 task types (2.18 -> 1.70 for T1T2)
- **Audio loss barely moves**: T1T2 audio goes 23.4 -> 22.8 in 3000 steps; A1A2 audio 50.4 -> 49.1
- Audio-output tasks (T1A2, A1A2) have ~2x higher audio loss than text-output tasks (T1T2, A1T2): **49 vs 23**
- The 49 vs 23 gap suggests audio-input tasks provide some signal to audio-output prediction through the backbone, but pure audio generation (A1A2) is essentially stuck

### D11: Module Gradient Norms (step 3000)

**Text gradients** (selecting key layers):
| Module | lm_head | layer0 | layer12 | layer23 |
|--------|---------|--------|---------|---------|
| text | 22.4 | 21.4 | 4.6 | 10.4 |
| audio | 8.4 | 7.2 | 1.9 | 22.3 |

Audio gradient norms are **2-3x smaller** than text across most layers. Exception: layer23 (final layer) has audio_grad=22.3 > text_grad=10.4 -- the last layer gets strong audio signal but it doesn't propagate backward. This is consistent with the "random embedding" hypothesis: the audio loss gradient at the output layer is large, but because the audio embeddings are random, this gradient doesn't translate into coherent parameter updates in deeper layers.

### D12: CKA (Centered Kernel Alignment vs. initial checkpoint)

| Step | Layer 6 | Layer 12 | Layer 18 |
|------|---------|----------|----------|
| 500 | 1.000 | 1.000 | 1.000 |
| 1000 | 1.000 | 0.974 | 0.979 |
| 2000 | 1.000 | 0.949 | 0.951 |
| 3000 | **1.000** | **0.955** | **0.961** |

CKA stays **extremely high** (> 0.95 everywhere). The backbone representations at every layer remain nearly identical to the initial checkpoint. The model is not learning new representations -- it's just adjusting the output head. (骨干网络几乎没有变化 -- CKA > 0.95 意味着表征空间几乎保持不变)

Compare with exp3 (GradNorm diverged): CKA layer12 = 0.613, layer18 = 0.672. GradNorm actually moved the backbone more, but in the wrong direction (text loss exploded).

### D13: Audio Embedding Cosine Collapse

| Step | cos_collapse |
|------|-------------|
| 100 | 0.00004 |
| 500 | 0.00043 |
| 1000 | 0.00227 |
| 2000 | 0.01481 |
| 3000 | **0.01761** |

The mean pairwise cosine similarity among audio embeddings rises very slowly from ~0 to 0.018. This is far from collapse (which would be ~1.0). Audio embeddings are NOT collapsing -- they remain **spread out in random directions**. (音频 embedding 没有坍缩，它们散布在随机方向上)

---

## Key Findings

### 1. Audio Plateau Confirmed

Audio val loss drops from 41.2 (step 200) to 35.2 (step 3000), a reduction of only **14.5%** in 3000 steps. Meanwhile text val loss drops from 6.6 to 1.5, a **77%** reduction. The per-CB val losses converge to cb0=4.0, cb2/3/5/6~5.5 and stop improving after ~step 1000.

For context, log(4160) = 8.33 nats (uniform random over SNAC vocab). The best CB (cb0) reaches 4.0 (better than random) but fine-grained CBs plateau at 5.5 (only 34% below random baseline).

### 2. Text and Audio Gradients Are Orthogonal, Not Conflicting

cos_phi stays within [-0.05, +0.03] throughout training. Per-layer cosines are also near zero. This rules out the "conflicting gradients" (梯度冲突) hypothesis that is commonly invoked for multi-task learning failures. The problem is not that text and audio fight each other -- it's that audio gradients carry **no useful directional information**.

### 3. Backbone Representations Barely Change (CKA > 0.95)

CKA at all tracked layers stays above 0.95. The backbone displacement saturates at 11.8 (vs 60.8 for lm_head). The pretrained text representations are essentially frozen in place. The model learns to output text better by adjusting lm_head, but the backbone doesn't reorganize to accommodate audio.

### 4. Audio Embeddings Do Not Collapse -- They Stay Random

Audio embedding rank remains at ~877/896 = 97.9% of full rank. Cosine collapse metric is 0.018 (negligible). The audio embeddings initialized randomly are barely moving. They retain their initial random structure, providing no useful information to the backbone.

### 5. Entropy Scaling Did Not Help (Implementation Issue)

exp4 train audio_loss appears low (4.14) but this is because each CB loss is divided by its entropy (6.0--7.9). The unscaled val audio_loss is 35.41, essentially identical to exp2 baseline (35.24). The entropy weighting reduces the **effective weight** of audio in the total loss (since entropy > 1 divides the loss), doing the opposite of what was intended. Should multiply by entropy^(-1) to upweight low-entropy (harder) codebooks.

### 6. GradNorm Diverged Catastrophically

GradNorm weights went exponentially negative for text: w_text went from +0.01 to -525,100 within 1000 steps (saturated at step 1000 and stayed there). This caused text_loss to explode to ~5000+. The root cause: audio loss barely changes (stuck at 35-45), so GradNorm's relative rate mechanism keeps increasing audio's weight and decreasing text's weight. Without a non-negativity constraint, w_text crosses zero and the model maximizes text loss.

Audio performance under GradNorm (val_audio=35.31) is the same as baseline (35.24) -- confirming that more weight on audio doesn't help.

---

## Evidence Chain: Why Audio Plateaus

The diagnostics paint a clear picture:

```
Random audio embeddings (rank ~877/896)
    ↓
Audio tokens mapped to random directions in embedding space
    ↓
Backbone receives random input, produces random-direction hidden states for audio
    ↓
Audio gradients are noisy/directionless (rho < 1, gradients shrink to 10)
    ↓
cos_phi ≈ 0: audio gradients are orthogonal to text (random noise is orthogonal to everything)
    ↓
Backbone barely moves (CKA > 0.95, displacement saturates)
    ↓
Linear probes learn nothing (val_acc 3-5%)
    ↓
Audio loss plateaus at 35
```

**Root cause**: The audio embeddings are randomly initialized and never learn meaningful representations. The 181,120 audio tokens (7 codebooks x 4160 vocab) form a massive, sparse embedding table. With random embeddings, the LLM backbone receives random vectors as input for audio tokens, making it impossible to learn audio-conditional patterns. The gradient from audio loss at the output is directionless when propagated through random embeddings.

**Why text works**: Text embeddings are pretrained (from Qwen2-0.5B). They carry rich semantic structure. The backbone already knows how to process text -- it just needs fine-tuning.

---

## Hypotheses for Next Steps

### H1: Audio Embedding Initialization Is the Bottleneck
The embeddings need structure before the backbone can learn from them. Options:
- Initialize audio embeddings from SNAC codebook centroids (if available)
- Use a small pretrained audio encoder to provide initial embedding structure
- Copy text embedding structure (e.g., cluster audio tokens and map to text token neighborhoods)

### H2: Staged Training Could Break the Chicken-and-Egg Problem
The backbone can't learn audio patterns from random embeddings, and embeddings can't learn from a backbone that ignores them.
- Stage A: Freeze backbone, train only audio embeddings + lm_head on audio-only data
- Stage B: Unfreeze backbone with warm audio embeddings
- This breaks the deadlock by letting embeddings develop structure first

### H3: Curriculum from Text-Rich to Audio-Rich
Start with mostly T1T2 tasks (text-to-text), gradually increase audio tasks. This keeps the backbone learning while slowly introducing audio.

---

## Next Experiments

1. **Fix GradNorm**: Add non-negativity constraint (clamp w >= 0.01) and gradient clipping on the weight parameters. Re-run.
2. **Fix entropy scaling**: Invert the weighting -- multiply by max_entropy/entropy to upweight low-entropy CBs, not downweight them.
3. **Staged S3**: Freeze backbone for first 1000 steps, train only audio embeddings + lm_head. Then unfreeze.
4. **Audio embedding init**: Initialize audio embeddings by copying nearest text embedding (based on SNAC codebook semantics or random text token mapping).
5. **Higher audio LR**: Use separate learning rate for audio embeddings (10x backbone LR) to accelerate embedding learning.
6. **Curriculum**: Start with audio_weight=0.1, linearly increase to 1.0 over 1500 steps.

---

## Appendix: Raw Numbers

### exp2 Baseline Loss Evolution

| Step | text_loss | audio_loss | rho | cos_phi | backbone_disp | CKA_L12 |
|------|-----------|------------|-----|---------|---------------|---------|
| 100 | 5.79 | 51.7 | 0.32 | -0.045 | 0.77 | -- |
| 300 | 5.04 | 44.4 | 0.77 | +0.021 | 1.42 | -- |
| 500 | 3.38 | 48.1 | 0.30 | -0.013 | 2.15 | 1.000 |
| 1000 | 1.73 | 37.8 | 0.44 | +0.002 | 5.02 | 0.974 |
| 1500 | 1.75 | 31.3 | 0.51 | -0.000 | 8.24 | 0.971 |
| 2000 | 1.57 | 40.4 | 0.30 | -0.022 | 10.85 | 0.949 |
| 2500 | 1.75 | 36.6 | 0.35 | -0.009 | 11.69 | 0.954 |
| 3000 | 1.44 | 37.3 | 0.73 | +0.001 | 11.79 | 0.955 |

### exp3 GradNorm Weight Divergence

| Step | w_text | w_audio | text_loss |
|------|--------|---------|-----------|
| 100 | +0.012 | 1.99 | 6.86 |
| 200 | -1.24 | 3.24 | 3704 |
| 300 | -20.9 | 22.9 | 5382 |
| 500 | -586 | 588 | 4154 |
| 1000 | -525,100 | 525,102 | 5880 |
| 3000 | -525,100 | 525,102 | 4766 |
