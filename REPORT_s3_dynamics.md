# S3 Training Dynamics Report

**Date**: 2026-03-18
**Branch**: `omni-basin-shaping`

---

## 1. Experiment Setup

### What we replicated
Mini-omni S3: the final multimodal fine-tuning stage where all weights are unfrozen and the model learns to generate both text and audio (7-layer SNAC tokens).

### Our setup vs mini-omni

| | Mini-omni (paper) | Our replication |
|---|---|---|
| **Starting checkpoint** | Their S2 output (unpublished) | Their final published checkpoint (post-S3) |
| **GPU** | 8×A100 | 1×RTX 5090 (32GB) |
| **Effective batch** | 192 | 32 (batch=2 × grad_accum=16) |
| **LR schedule** | Cosine, 2e-6 → 2e-5, 1500 warmup | Same |
| **Optimizer** | AdamW (inferred from litgpt) | AdamW, β=(0.9, 0.95), WD=0.01 |
| **Data** | 6M samples (ASR + Open-Orca + Moss + VoiceAssistant) | 88K VoiceAssistant-400K (4 task types × 88K) |
| **Audio input** | Whisper features injected | Placeholder padding (no Whisper) |
| **Tasks** | T1T2, T1A2, A1T2, A1A2 + ASR + RLHF | T1T2, T1A2, A1T2, A1A2 |
| **SNAC delay pattern** | Yes (stream i delayed by i positions) | Yes (fixed from earlier bug) |
| **Model architecture** | Qwen2-0.5B + whisperMLP, tie_word_embeddings=True, post_adapter=False | Same (loaded from their config) |
| **Params** | 532M (tied weights) | Same |
| **Training code** | Unpublished (based on litgpt commit d367a1199a) | Reverse-engineered from inference code + paper |

### Key differences and their impact

1. **Starting from post-S3 checkpoint, not post-S2.** We're continuing training on an already-converged model. This means the dynamics we observe are those of *continued* S3, not fresh S3. Losses start lower but plateau faster.

2. **88K vs 6M samples.** Our dataset is ~1.5% of theirs. We see the full dataset ~4× per 3000 steps. This may cause overfitting on train but shouldn't affect the gradient dynamics we're measuring.

3. **No Whisper features.** A1T2 and A1A2 tasks receive padding where Whisper features should be. This makes audio-input tasks harder (A1T2 text loss: 2.87 vs T1T2: 1.34). Does not affect T1A2 or T1T2.

4. **Batch size 32 vs 192.** Smaller batch = noisier gradients = potentially different dynamics. Our gradient norm measurements may be noisier, but ρ (ratio) should be stable.

### Bug fixes applied

1. **SNAC delay pattern** (critical): Mini-omni uses delayed parallel decoding where stream i is offset by i positions. Our earlier `prepare_s2.py` had all streams aligned. Effect: audio loss dropped from 9.45 (near random) to 4.56 on the pretrained checkpoint once fixed.

2. **tie_word_embeddings=True**: Our earlier `train_observe.py` used `tie_word_embeddings=False`. The published checkpoint uses True (wte and lm_head share weights, 532M vs 694M params).

3. **Val averaging bug** (known, affects all runs equally): `compute_losses` returns 0 for text on audio-only tasks (and vice versa). The val loop averages these zeros in, deflating text loss for audio-heavy batches. All runs share this bug so relative comparisons are valid. Corrected eval script (`eval_s3.py`) gives pretrained text=2.11 (not 1.48).

---

## 2. Baseline Results (S3 Adam, 3000 steps)

### Loss trajectories

| Step | Text loss | Audio loss | CB0 | CB4 | LR |
|------|----------|-----------|-----|-----|----|
| 0 (ckpt) | 1.48 | 4.56 | 6.00 | 3.10 | — |
| 200 | 1.60 | 3.92 | 5.04 | 2.55 | 4.4e-6 |
| 600 | 1.57 | 3.76 | 4.78 | 2.42 | 8.0e-6 |
| 1000 | 1.58 | 3.70 | 4.68 | 2.37 | 1.4e-5 |
| 1500 | 1.57 | 3.64 | 4.55 | 2.33 | 2.0e-5 (peak) |
| 2000 | 1.58 | 3.60 | 4.47 | 2.30 | 1.6e-5 |
| 2500 | 1.55 | 3.57 | 4.43 | 2.29 | 5.0e-6 |
| 3000 | 1.54 | 3.57 | 4.42 | 2.29 | 2.0e-6 |

*All losses use the buggy val metric (comparable across runs).*

**Audio plateau**: 80% of improvement happens in first 200 steps (4.56→3.92). Last 1400 steps improve only 0.06 (3.63→3.57). Audio converges rapidly then stalls.

**Text degradation**: 1.48→1.54 (+4.1%). Consistent with mini-omni paper's observation that audio training degrades text.

### Diagnostic signals (D1-D7)

| Step | ρ(t) | cos φ(t) | Displacement | Text rank | Audio rank |
|------|------|---------|-------------|-----------|-----------|
| 100 | 0.288 | -0.085 | 1.1 | 766 | 636 |
| 500 | 0.238 | -0.012 | 4.5 | 767 | 636 |
| 1000 | 0.229 | -0.069 | 11.4 | 766 | 637 |
| 1300 | 0.215 | **-0.247** | 17.0 | 766 | 638 |
| 1500 | 0.198 | **-0.285** | 21.3 | 767 | 637 |
| 2000 | 0.244 | -0.163 | 31.2 | 765 | 637 |
| 2500 | 0.208 | -0.108 | 36.6 | — | — |
| 3000 | 0.225 | -0.091 | 38.2 | 765 | 638 |

### Basin width probing (D4)

| Step | ε=0.01 degradation | ε=0.1 degradation |
|------|-------------------|------------------|
| 1000 | 4.69 | 90.4 |
| 2000 | 4.49 | 92.3 |

Basin width is **unchanged** by S3 training. No GO-style widening.

---

## 3. Dynamics Analysis

### Finding 1: ρ stays in Phase 1 throughout

ρ(t) = ‖∇L_audio‖ / ‖∇L_text‖ ranges from 0.18 to 0.29. Audio gradients are always ~4× weaker than text. **The model never enters Phase 2** (ρ ≈ 1) at this learning rate.

Implication: Text always dominates optimization. Audio gets ~20% of the effective gradient budget.

### Finding 2: Gradient interference peaks during max LR

cos φ(t) measures angle between text and audio gradients. It peaks at **-0.285** around step 1300-1500 — exactly when LR reaches its maximum (2e-5). This is the point of maximum interference.

The timeline:
- Steps 0-500: cos φ ≈ 0 (orthogonal, no interference)
- Steps 500-1000: cos φ drifts negative (-0.07)
- Steps 1000-1500: cos φ reaches **-0.29** (peak interference, coincides with peak LR)
- Steps 1500-3000: cos φ recovers to -0.09 as LR decays

This suggests interference is LR-dependent: higher LR → larger parameter updates → gradients start opposing each other.

### Finding 3: Audio plateau is NOT caused by weak gradients

λ=3 (3× audio loss weight) barely helps: audio loss 3.52 vs baseline 3.57 (-1.4%). The ρ trajectory is almost identical (0.18-0.25). Simply amplifying audio gradients does not break the plateau.

This rules out "audio gets too little gradient" as the explanation. The plateau is structural.

### Finding 4: Codebook hierarchy is consistent but all plateau together

SNAC maps 7 streams to 3 codebooks: [CB1, CB2, CB3, CB3, CB2, CB3, CB3].

| Stream | Codebook | Final val loss | Improvement from ckpt |
|--------|----------|---------------|---------------------|
| S0 | CB1 (coarsest) | 4.42 | -26.3% |
| S1 | CB2 | 3.82 | -28.2% |
| S2 | CB3 (finest) | 4.56 | -17.1% |
| S4 | CB2 | 2.29 | -26.1% |
| S6 | CB3 | 2.91 | -18.3% |

All codebooks improve by similar relative amounts (~20-28%) then plateau together. No single codebook is the bottleneck.

### Finding 5: Basin width unchanged

Perturbation experiments at steps 1000 and 2000 show identical sensitivity. S3 training does not widen the text loss basin. This falsifies the "audio as implicit GO" hypothesis from our THEORY.md.

---

## 4. Intervention Results

### λ=3 (higher audio weight)

| Metric | Baseline | λ=3 | Δ |
|--------|---------|-----|---|
| Final text | 1.54 | 1.54 | 0% |
| Final audio | 3.57 | 3.52 | -1.4% |
| Peak cos φ | -0.29 | -0.32 | Slightly more interference |
| ρ range | 0.18-0.29 | 0.18-0.27 | No change |

**Conclusion**: Loss weighting alone cannot overcome the audio plateau.

### Gradient Projection (running)

Removes destructive component of audio gradient when cos φ < 0:

g_audio' = g_audio - (g_audio · g_text / ‖g_text‖²) · g_text  (when cos φ < 0)

Early observations (step 710/3000):
- Gradient norms 10× lower (0.1-1.5 vs 3-11)
- Displacement 66% of baseline
- Val numbers not directly comparable (batch_size=1 → different val averaging)
- Still running, ~4h remaining

### Adaptive λ (pending)

Queued after GradProj.

---

## 5. Revised Hypotheses

Based on these findings, we update the theoretical framework:

### Original hypothesis (partially falsified)
> Audio gradients act as GO-style noise to widen the text basin.

**Status**: Falsified. Basin width unchanged. Audio helps text through a different mechanism (possibly gradient noise regularization, not basin geometry).

### Updated understanding

1. **The audio plateau is structural, not optimization-limited.** Higher λ doesn't help. The bottleneck is likely Discrete Representation Inconsistency (DRI) — the same audio maps to different SNAC tokens depending on context (only ~47% consistency per literature).

2. **Phase 2 may require much longer training or higher LR.** At S3's conservative LR (2e-5), ρ never reaches 1. The transition to Phase 2 may require:
   - 10-100× more steps
   - Higher peak LR
   - Or it may not happen at this model scale

3. **Gradient interference is real but mild.** cos φ peaks at -0.29, not -1.0. The gradients are mostly orthogonal with a small destructive component during peak LR.

4. **The text degradation (+4%) is likely inevitable** given the shared parameter space. Gradient projection may mitigate it (pending results).

---

## 6. Files and Reproducibility

| File | Purpose |
|------|---------|
| `train_s3.py` | S3 training with D1-D7 diagnostics, supports baseline/grad_proj/adaptive_lambda |
| `prepare_s3.py` | Data pipeline with correct SNAC delay pattern |
| `eval_s3.py` | Corrected evaluation (separate text_n/audio_n tracking) |
| `results/s3_adam/diagnostics.json` | Full D1-D7 traces for baseline |
| `results/s3_lambda3/diagnostics.json` | Full D1-D7 traces for λ=3 |
| `results/s3_gradproj/diagnostics.json` | Traces for gradient projection (in progress) |

Pretrained checkpoint: `/workspace/mini-omni-ckpt/lit_model.pth` (gpt-omni/mini-omni on HuggingFace)
