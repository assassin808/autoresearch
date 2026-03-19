# S3 Batch 2 Experiments: Gradient Dynamics Analysis

**Date**: 2026-03-19
**Branch**: `omni-basin-shaping`
**GPU**: NVIDIA RTX 5090, 32GB VRAM
**Base code**: `train_s3.py` (modified to support all 4 experiments)

## Motivation

S3 Batch 1 (baseline, lambda=3, grad_proj) all showed identical audio plateau at val_audio ~1.89.
Literature review (8 scans, 30+ papers) pointed to SNAC codec as bottleneck (no semantics, DRI).

**Core question**: Is the audio plateau caused by S2 (text-only pre-training distorting the model) or is it audio's own structural issue?

**Approach**: Run 4 experiments that isolate different variables, tracking gradient diagnostics (rho, cos_phi, per-layer cos, CB losses, displacement, embedding rank) throughout training.

---

## Experiment Configurations

### Exp A: Moshi-Style Codebook Weighting
- **File**: `results/s3_cbweight/`
- **Hypothesis**: 100:1 semantic vs acoustic weighting breaks the audio plateau by focusing learning on semantic codebook
- **SNAC stream->codebook mapping**: [CB1, CB2, CB3, CB3, CB2, CB3, CB3]
  - Stream 0 -> CB1 (coarsest, semantic) -> weight 100
  - Streams 1, 4 -> CB2 (medium) -> weight 10
  - Streams 2, 3, 5, 6 -> CB3 (finest, acoustic) -> weight 1
- **CLI**: `--cb_weights "100,10,1,1,10,1,1" --max_steps 3000 --no_save_model`
- **Changes**: Weighted audio loss = sum(cb_w[i] * loss[i]) / sum(cb_w) in all training methods
- **LR**: 2e-6 -> 2e-5 (standard S3)
- **Checkpoint**: Published post-S3

### Exp B: M-SAM (Modality-Aware SAM)
- **File**: `results/s3_msam/`
- **Hypothesis**: SAM finds flat regions where text and audio don't conflict
- **Method**: Simplified M-SAM (text always dominant since rho~0.22, skip Shapley):
  1. Forward+backward -> get combined gradient g
  2. Compute perturbation: eps = rho_sam * g / ||g||
  3. Perturb: theta' = theta + eps
  4. Forward+backward at theta' -> get g' (gradient at perturbed point)
  5. Restore theta, use g' as actual update
- **CLI**: `--method m_sam --sam_rho 0.05 --max_steps 3000 --no_save_model`
- **Batch**: batch_size=1, grad_accum=32 (2x forward passes like grad_proj)
- **Note**: Skips GradScaler (manual backward), 2x runtime
- **Checkpoint**: Published post-S3

### Exp C: Skip S2 -- Adapter-Only -> S3
- **File**: `results/s3_skip_s2/`
- **Hypothesis**: Test if S2 text adaptation is necessary or if direct S1->S3 works
- **Method**: Build post-S1 checkpoint:
  1. Load Qwen2-0.5B pretrained weights (QKV interleaving from train_observe.py)
  2. Copy whisper_adapter weights from published checkpoint (trained in S1)
  3. Randomly initialize audio embeddings (wte[152000:], std=0.02)
  4. lm_head shares wte (tie_word_embeddings=True)
- **CLI**: `--checkpoint_mode post_s1 --lr_max 2e-4 --max_steps 3000 --no_save_model`
- **LR**: 2e-6 -> 2e-4 (S2 range, higher than S3 since LLM not adapted)
- **Checkpoint**: Constructed post-S1 (vanilla Qwen2 + trained adapters)

### Exp D: Authentic S2 (Text-Only) + Audio Gradient Tracking
- **File**: `results/s2_textonly/`
- **Hypothesis**: Track audio gradient dynamics during text-only training to see if interference exists even without audio loss
- **Method**: Authentic mini-omni S2 setup:
  - Data: Filtered to T1T2 + A1T2 only (text-output tasks, ~50% of S3 data)
  - Frozen: whisper_adapter (trained in S1)
  - Trained: LLM backbone (transformer.h, wte/ln_f)
  - audio_weight=0.0 (no audio loss in training)
  - Val diagnostics run on FULL dataset (all 4 task types including audio-output)
- **CLI**: `--s2_mode --max_steps 3000 --no_save_model`
- **LR**: 2e-6 -> 2e-4 (S2 range)
- **Checkpoint**: Published post-S3

### Baseline: S3 Adam (from Batch 1)
- **File**: `results/s3_adam/`
- **Method**: Standard S3 training with AdamW
- **LR**: 2e-6 -> 2e-5, warmup 1500
- **All weights unfrozen**, published post-S3 checkpoint

---

## Runtime Summary

| Experiment | Steps | Runtime | Final text_loss | Final audio_loss |
|-----------|-------|---------|----------------|-----------------|
| Exp D (s2_textonly) | 3000 | 2.4h (8490s) | 1.972 | 4.766 |
| Exp A (s3_cbweight) | 3000 | 2.2h (8060s) | 1.739 | 3.669 |
| Exp C (s3_skip_s2) | 3000 | 2.2h (8092s) | 3.149 | 6.594 |
| Exp B (s3_msam) | 3000 | 8.9h (31962s) | 0.892 | 1.877 |
| Baseline (s3_adam) | 3000 | ~2.5h | ~0.91 | ~1.89 |

---

## Results: Gradient Dynamics Comparison

### rho (||grad_audio|| / ||grad_text||) -- Audio gradient relative strength

| step | S3_base | ExpA_cb | ExpC_skipS2 | ExpD_s2only |
|------|---------|---------|-------------|-------------|
| 100  | 0.288   | 0.319   | 0.173       | **0.560**   |
| 500  | 0.238   | 0.256   | **0.010**   | **0.511**   |
| 1000 | 0.229   | 0.238   | **0.057**   | **0.536**   |
| 1500 | 0.198   | 0.176   | **0.046**   | **0.449**   |
| 2000 | 0.244   | 0.214   | **0.048**   | **0.393**   |
| 2500 | 0.208   | 0.188   | **0.037**   | **0.599**   |
| 3000 | 0.225   | 0.181   | **0.061**   | **0.544**   |

**Findings**:
- ExpD (text-only): rho ~0.4-0.56, **2x baseline**. Audio gradients are relatively stronger when model never trains on audio. S3 training itself suppresses audio gradient relative magnitude.
- ExpC (skip S2): rho collapses to **0.01-0.06**. Un-adapted Qwen2 produces enormous text gradients (text_grad_norm=48-344 vs baseline 11-16), drowning out audio. **S2 actually helps audio by reducing text gradient magnitude.**
- ExpA (CB weight): rho declines from 0.32 to 0.18, faster than baseline. 100:1 weight concentrates gradient budget on semantic CB.

### cos_phi (gradient interference: +ve=cooperative, -ve=conflict)

| step | S3_base | ExpA_cb | ExpC_skipS2 | ExpD_s2only |
|------|---------|---------|-------------|-------------|
| 100  | -0.085  | -0.122  | -0.039      | -0.075      |
| 500  | -0.012  | +0.001  | -0.005      | -0.052      |
| 1000 | -0.069  | -0.031  | -0.044      | +0.003      |
| 1500 | -0.285  | **+0.140**| -0.013    | **-0.199**  |
| 2000 | -0.163  | **+0.084**| -0.010    | **-0.296**  |
| 2500 | -0.108  | **+0.105**| -0.003    | -0.151      |
| 3000 | -0.091  | **+0.084**| -0.002    | **-0.148**  |

**Findings**:
- S3 baseline: cos_phi consistently negative (-0.01 to -0.29), worst at step 1500. Text-audio gradients always conflict.
- ExpA (CB weight): **cos_phi flips positive after step 1000** (+0.08 to +0.14). Semantic-weighted audio gradients align with text direction.
- ExpD (text-only): cos_phi reaches **-0.30 at step 2000**. Interference exists even without any audio loss. This is structural, not caused by audio optimization.
- ExpC (skip S2): cos_phi ~0 (audio gradients too weak to meaningfully interfere).

### Gradient norms (text_grad_norm / audio_grad_norm)

| step | S3_base | ExpA_cb | ExpC_skipS2 | ExpD_s2only |
|------|---------|---------|-------------|-------------|
| 100  | 11.4/3.3 | 11.5/3.7 | **77.2/13.4** | 11.3/**6.4** |
| 500  | 11.4/2.7 | 10.6/2.7 | **343.6/3.6** | 12.0/**6.1** |
| 1000 | 11.9/2.7 | 11.4/2.7 | **57.1/3.3**  | 12.6/**6.7** |
| 2000 | 12.2/3.0 | 13.2/2.8 | **53.5/2.6**  | 21.2/**8.3** |
| 3000 | 12.4/2.8 | 15.5/2.8 | **48.3/3.0**  | 16.1/**8.7** |

**Findings**:
- ExpC text_grad_norm peaks at 344 (step 500) -- un-adapted Qwen2 is wildly unstable on this data.
- ExpD audio_grad_norm is 2-3x baseline (6-9 vs 3). Un-optimized directions have larger gradient signals.

### Per-codebook val losses (from diagnostic passes)

CB0=semantic(coarsest), CB4=medium, CB6=finest(acoustic)

| CB | S3_base 100->3000 | ExpA 100->3000 | ExpC 100->3000 | ExpD 100->3000 |
|----|-------------------|----------------|----------------|----------------|
| cb0 | 5.28->**4.41** | 5.19->**4.28** (better) | 8.06->5.59 | 5.91->**5.94** (worse!) |
| cb1 | 4.78->**3.98** | 4.84->4.03 | 8.25->6.66 | 5.44->**5.63** (worse!) |
| cb2 | 5.25->**4.84** | 5.38->5.00 (worse) | 8.56->7.53 | 5.66->**5.78** (worse!) |
| cb4 | 2.84->**2.52** | 2.92->2.63 (worse) | 8.38->6.38 | 3.25->**3.53** (worse!) |
| cb6 | 3.27->**2.97** | 3.38->3.16 (worse) | 8.69->7.44 | 3.59->**3.80** (worse!) |

**Findings**:
- ExpD: ALL CB val losses INCREASE over training. Text-only training actively degrades audio representations.
- ExpA: cb0 (semantic, 100x weight) improves more than baseline, but fine CBs (cb2-cb6) are all worse. Weight shifts capacity to semantics at expense of acoustics.
- ExpC: All CBs start near random (~8.3=ln(4160)), only reach 5.6-7.5 after 3000 steps. cb0 drops fastest (semantics easiest to learn).

### Parameter displacement from init

| step | S3_base | ExpA | ExpB_msam | ExpC_skipS2 | ExpD_s2only |
|------|---------|------|-----------|-------------|-------------|
| 500  | 4.49 | 4.65 | 3.47 | **44.6** | **26.6** |
| 1000 | 11.4 | 11.8 | 8.3 | **109.6** | **77.0** |
| 2000 | 31.2 | 31.4 | 22.2 | **279.2** | **198.0** |
| 3000 | 38.2 | 38.5 | **27.3** | **341.5** | **233.1** |

Backbone-only displacement at step 3000:
- S3_base: 14.9, ExpA: 15.4, ExpB: 12.4, ExpC: **107.6** (7.2x), ExpD: **106.8** (7.2x)

**Findings**:
- ExpC and ExpD backbone displacement nearly identical (107.6 vs 106.8), both 7x baseline.
- M-SAM displacement only 71% of baseline -- SAM finds flatter region but doesn't improve audio loss.

### Embedding effective rank

| step | S3_base txt/aud | ExpA txt/aud | ExpB txt/aud | ExpC txt/aud | ExpD txt/aud |
|------|-----------------|--------------|--------------|--------------|--------------|
| 100  | 766/636 | 767/635 | 765/636 | **598/883** | 765/636 |
| 1000 | 766/637 | 767/636 | 765/638 | **770/873** | 759/637 |
| 2000 | 765/637 | 765/636 | 765/642 | **680/842** | **686/636** |
| 3000 | 765/638 | 764/636 | 765/645 | **592/833** | **655/636** |

**Findings**:
- ExpC: Audio rank starts at 883 (random init = high entropy), slowly decreasing. Text rank collapses to 592.
- ExpD: Text rank collapses 766->655 (high LR text-only). Audio rank unchanged at 636.
- Baseline/ExpA/ExpB: All stable (~765/636-645).

### Per-layer cos_phi (step 1000 and 3000)

**Layer 0 (embedding layer) is the dominant source of interference:**
- S3 baseline: layer 0 cos_phi = -0.24 (step 1000) / -0.29 (step 3000)
- All other layers: cos_phi within +/-0.03
- ExpA: layer 0 flips from -0.13 to **+0.23** at step 3000
- ExpD: layer 0 = +0.00 (step 1000) / **-0.27** (step 3000)

---

## Known Bugs in This Run

### 1. M-SAM gradient diagnostics all zeros
- **Cause**: M-SAM uses batch_size=1. Val loader also uses batch_size=1. The diagnostic code needs a batch where both `loss_mask_text.any()` and `loss_mask_audio.any()` are True simultaneously. With batch_size=1, each sample is either text-output or audio-output, never both.
- **Fix needed**: Use larger val batch_size for diagnostic passes (e.g., batch_size=4).
- **Impact**: M-SAM has no rho/cos_phi/grad_norm/CB_val data. Displacement and embedding rank are fine.

### 2. Embedding displacement always 0.0
- **Cause**: Model uses `tie_word_embeddings=True`. The `transformer.wte.weight` parameter may not appear in `named_parameters()` with that name, or the `"wte" in name` check doesn't match. The shared weight is tracked under `lm_head` instead (baseline shows lm_head displacement = 0.45 at step 100).
- **Fix needed**: Check actual parameter names in the tied-weight model, update name checks.
- **Impact**: `text_embedding` and `audio_embedding` displacement split (added for these experiments) reports zeros. Total displacement is correct (captures all params).

---

## Conclusions

**Core question: Is audio plateau caused by S2 or audio's own issue?**

**Answer: Primarily audio's structural issue. S2 is NOT the cause -- it actually helps.**

Evidence:

1. **ExpD proves interference is structural**: Text-only training produces cos_phi = -0.30 with audio gradients and ALL CB val losses worsen. The text-audio conflict exists in the embedding space regardless of whether audio is optimized.

2. **ExpC proves S2 helps audio**: Without S2, text gradients explode (text_grad_norm = 344 vs 11), rho collapses to 0.01 (audio completely drowned). S2 adapts the LLM to reduce text gradient magnitude, creating space for audio gradients.

3. **Interference concentrates in Layer 0 (embedding)**: The shared embedding table (text + audio tokens in same wte) is where gradients conflict. All other layers are near-orthogonal.

4. **CB weighting changes interference pattern but doesn't solve plateau**: ExpA makes cos_phi positive (less conflict) but fine codebooks get worse. SNAC's fine codebooks lack learnable semantic signal -- weighting can't create it.

5. **M-SAM finds flatter region (28% less displacement) but same audio loss**: The plateau is not a sharp-minima problem. The loss landscape itself is structurally limited.

---

## Code Changes (train_s3.py)

All 4 experiments are controlled via CLI args in the same file:

### New CLI args:
- `--cb_weights "100,10,1,1,10,1,1"` -- per-codebook loss weights (Exp A)
- `--method m_sam` + `--sam_rho 0.05` -- M-SAM training (Exp B)
- `--checkpoint_mode post_s1` -- build post-S1 checkpoint (Exp C)
- `--s2_mode` -- authentic S2 text-only training (Exp D)

### New functions:
- `build_post_s1_checkpoint()` -- constructs Qwen2 + trained adapters + random audio embeddings
- M-SAM training loop block (forward-perturb-forward-restore)

### Modified:
- `OmniS3Dataset.__init__()` -- accepts `s2_mode` for T1T2+A1T2 filtering
- `compute_displacement()` -- added `text_embedding` / `audio_embedding` split (has bug, see above)
- Audio loss computation uses `cb_weights` in all methods
- Diagnostics log `cb_weighted_train` alongside raw `cb_losses_train`

### Runner script:
- `run_new_experiments.sh` -- sequential runner for all 4 experiments

---

## File Manifest

```
results/s3_adam/           # Baseline S3 (from Batch 1)
  config.json
  diagnostics.json         # 30 diagnostic snapshots (every 100 steps)
  train.log

results/s2_textonly/       # Exp D: authentic S2, text-only
  config.json
  diagnostics.json
  train.log

results/s3_cbweight/       # Exp A: Moshi-style CB weighting
  config.json
  diagnostics.json
  train.log

results/s3_skip_s2/        # Exp C: skip S2, post-S1 -> S3
  config.json
  diagnostics.json
  train.log

results/s3_msam/           # Exp B: M-SAM
  config.json
  diagnostics.json         # NOTE: rho/cos_phi/grad_norms are all 0 (bug)
  train.log

train_s3.py                # Modified: supports all 4 experiments
run_new_experiments.sh     # Sequential runner
EXPERIMENTS_S3_BATCH2.md   # This file
```
