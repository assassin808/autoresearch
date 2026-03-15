# Combined Observation Findings (obs_1 + obs_2)

## Experiments Completed

| Exp | Config | Steps | Duration | Text Val | Audio Val |
|-----|--------|-------|----------|----------|-----------|
| obs_1 | Adam baseline (omni, λ=1) | 3000 | 135min | 2.497 | 6.367 |
| obs_2 | Text-only (λ=0) | 2000 | 89min | 2.604 | 9.316 |

## Key Findings

### 1. Basin Width: Audio Does NOT Widen the Text Basin
The central hypothesis — that audio gradients act as GO-style noise to widen the
text loss basin — is **not supported** at this training scale:

| Experiment | ε=0.1 degradation (step 1000) | ε=0.1 degradation (step 2000) |
|------------|-------------------------------|-------------------------------|
| obs_1 (omni) | 96.8 | 96.3 |
| obs_2 (text-only) | 97.3 | 96.7 |

Difference: <1%. Basin geometry is identical whether or not audio is trained.

### 2. Audio HELPS Text (Implicit Regularization)
Despite not widening the basin, omni training produces **better text loss**:
- obs_1 text loss: **1.602** (train), **2.497** (val)
- obs_2 text loss: 1.651 (train), 2.604 (val)
- **Omni is 3% better on text validation loss**

This suggests audio gradients provide useful regularization through a mechanism
OTHER than basin widening — possibly through gradient noise or shared representation
learning.

### 3. Three-Phase Model Partially Confirmed
The ρ(t) trajectory in obs_1 shows clear phase structure:
- Steps 100-600: ρ ≈ 0.04-0.09 (early phase, text dominates)
- Steps 700-3000: ρ ≈ 0.14-0.31 (transition, approaching middle)

But ρ never reaches 1.0 in 3000 steps. The middle phase (ρ~1) requires much
longer training (~10K-40K steps).

### 4. Audio Embedding Rank Collapse Confirmed
| | obs_1 (omni) | obs_2 (text-only) |
|---|---|---|
| Audio rank start | 883 | 883 |
| Audio rank end | 838 | 883 |
| Audio rank Δ | **-45** | **0** |

Rank collapse is entirely caused by audio training, not by the shared backbone.
Linear decline at ~1.5 dims/100 steps.

### 5. Gradient Interference is Mild but Layer-Specific
Global cos φ ≈ -0.006 (near zero), but interference concentrates at:
- **Layer 0** (embedding-adjacent): 73% negative cos φ
- **Layer 23** (output-adjacent): 70% negative cos φ

This is consistent with the modality competition literature — interference
occurs where text and audio representations share the most parameters.

### 6. Per-Codebook Hierarchy
CB0 (coarse/semantic) learns 3.6× faster than CB6 (fine/acoustic):
- CB0: 6.2 → 4.4 (Δ=-1.8)
- CB6: 8.1 → 7.6 (Δ=-0.5)

This validates Moshi's approach of weighting semantic tokens higher.

## Revised Research Direction

The original hypothesis (audio-as-GO-noise widens text basin) is not supported.
Instead, the data suggests:

1. **Audio provides implicit regularization** that helps text (3% val improvement)
   through a mechanism that is NOT basin widening
2. **The three-phase transition occurs on much longer timescales** (10K+ steps)
3. **Gradient interference is layer-specific** — methods should target layers 0 and 23

### New Hypotheses to Test
- H_new1: Audio regularization works via gradient noise injection (similar to dropout)
- H_new2: The ρ~1 transition and text degradation only appear with longer training
- H_new3: Layer-specific gradient projection (only at layers 0,23) could help
- H_new4: Codebook-hierarchical loss weighting (CB0 >> CB6) improves audio convergence
