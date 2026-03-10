# Theoretical Hypotheses for Omni-Model Training

## Empirical Anchors (what we know for certain)

From 100+ controlled experiments on an 88M-param text+audio transformer:

1. **λ_embed ∝ 1/N_data**: Optimal embedding WD scales inversely with dataset size
   - 5.2h audio → λ*=2.5, 115h audio → λ*=0.0
   - Text (600M tokens) never needed WD at any scale

2. **Per-row WD >> uniform WD**: Separating text/audio WD gave 7% improvement

3. **All gradient manipulation failed**: H2 (upweighting), H3 (norm balancing),
   H4 (per-row LR), H5 (AdaDecay), H6 (MILES) — all worse than baseline

4. **Muon WD has different scaling than embed WD**: embed WD 2.5→0.0 (100% reduction)
   vs Muon WD 0.5→0.2 (60% reduction) with same data increase

5. **Muon LR is portable but Adam LR is not**: Muon LR=0.06 works across all configs,
   but embedding LR and WD are highly coupled to data distribution

---

## The 10 Hypotheses

### H1: Bayesian Embedding Theory — "Tokens as Posterior Inference"

**Claim**: Each embedding row e_i performs approximate Bayesian inference. The optimal
weight decay for token i is:

    λ_i* = σ²_prior / (n_i · σ²_data)

where n_i is the token count, σ²_prior is the prior precision, and σ²_data is the
gradient noise variance.

**Why this matters**: This gives a *closed-form* prediction for optimal per-token WD,
eliminating an entire hyperparameter sweep. It also predicts that WD should be
*inversely proportional to token frequency* — not just binary text/audio.

**Testable prediction**:
- Compute per-token frequency f_i from training data
- Set λ_i = C / f_i for some constant C
- This continuous WD should outperform our binary (text=0, audio=2.5) WD
- The optimal C should be dataset-size-independent

**Novel contribution**: First principled derivation of per-token regularization for
omni-models. Generalizes to any heterogeneous-frequency parameter (LoRA adapters,
mixture-of-experts routers, etc.)

---

### H2: Spectral Geometry of Multi-Modal Gradients

**Claim**: In a shared transformer, text and audio gradients occupy different subspaces
of the weight matrix. The key quantity is the *principal angle* between these subspaces:

    θ = arccos(σ_max(G_text^T · G_audio) / (||G_text|| · ||G_audio||))

When θ ≈ 0 (aligned), modalities cooperate — joint training helps both.
When θ ≈ π/2 (orthogonal), modalities are independent — no interference.
When θ ≈ π (opposing), modalities compete — joint training hurts both.

**Why gradient manipulation failed**: Our H2-H6 all tried to change gradient
*magnitudes*. But if the subspaces are nearly orthogonal (θ ≈ π/2), magnitude
changes have no effect — the modalities don't interact much in gradient space.

**Testable prediction**:
- Measure θ per layer during training
- If θ ≈ π/2 everywhere, confirms "non-interference" hypothesis
- If θ varies by layer, suggests layer-specific optimization strategies
- If θ changes over training, reveals a convergence dynamics story

**Novel contribution**: First empirical characterization of the multi-modal gradient
geometry in omni-models. Explains WHY simple approaches work and complex ones fail.

---

### H3: Muon's Implicit Modality Balancing via NS Orthogonalization

**Claim**: Newton-Schulz orthogonalization in Muon *automatically* balances modality
contributions. When NS computes U ≈ G(G^TG)^{-1/2}, it:

1. Projects out the dominant singular direction (which corresponds to the
   majority modality — text)
2. Amplifies the subdominant directions (which include audio-specific features)

This is equivalent to an implicit form of gradient balancing, making explicit
gradient manipulation (H2-H6) redundant.

**Why this is profound**: It means Muon is *inherently better* than Adam for
multi-modal training, not just for single-task training. The spectral normalization
acts as automatic modality equalization.

**Testable prediction**:
- Compare Muon vs Adam (with same effective LR) on the omni task
- Muon should show smaller text-audio convergence gap than Adam
- The singular value spectrum of Muon's momentum buffer should be flatter than
  the raw gradient spectrum (confirming the equalization effect)
- Disabling NS (using raw momentum) should restore the convergence gap

**Novel contribution**: First theoretical explanation of why Muon is especially
well-suited for multi-modal training. Could change how the field thinks about
optimizer choice for omni-models.

---

### H4: The Embedding Bottleneck Hypothesis

**Claim**: In omni-models, the embedding table is the primary bottleneck, not the
transformer backbone. Evidence:

- Embeddings are 73% of parameters (62M/88M)
- Per-row WD gave 12% val_loss improvement (Phase 3)
- All backbone modifications (Muon WD, dropout, LR) gave <2% each
- The embedding table must encode 20,482 discrete tokens into 512-dim vectors,
  but audio tokens have ~100x less data than text tokens

**Deeper theory**: The embedding table's effective rank is limited by
min(n_tokens_seen, d_model). For audio tokens with few observations,
the effective rank is much lower, leading to:
- Overfitting on observed contexts
- Poor generalization to unseen token combinations
- High sensitivity to WD (the only control on effective rank)

**Testable prediction**:
- Measure effective rank of text vs audio embedding submatrices during training
- Audio embeddings should have lower effective rank
- Adding data should increase audio embedding rank
- This rank difference should predict the optimal WD ratio

**Novel contribution**: Identifies the embedding table as the critical component
in omni-model design. Suggests architectual solutions: separate embedding spaces,
factored embeddings, or embedding dropout.

---

### H5: Phase Transition Theory of Regularization

**Claim**: The relationship between optimal WD and dataset size exhibits a
*phase transition*, not a smooth inverse curve. There exists a critical dataset
size N_c below which WD is essential (overfitting regime) and above which WD
is harmful (underfitting regime).

Our data points:
- 5.2h: deep in overfitting regime (λ*=2.5)
- 115h: at or past the transition (λ*=0.0)

The transition occurs when the number of unique contexts per token exceeds
the embedding dimension:

    N_c ≈ d_model × vocab_size / batch_diversity

**Why this matters**: It predicts that for large-scale omni-models (GPT-4o level),
WD for embeddings should be *exactly zero* — eliminating an entire hyperparameter
class. It also predicts the critical data threshold for each token type.

**Testable prediction**:
- Run the WD sweep at intermediate data sizes (20h, 50h, 80h)
- Plot λ*(N) — should show sigmoid-like transition, not 1/N
- The transition point should be predictable from d_model and vocab_size
- Text tokens (high frequency) should have already passed their transition

**Novel contribution**: A phase transition framework for regularization in
heterogeneous-frequency models. Connects to statistical physics of learning.

---

### H6: Momentum Decomposition — "Separate the Signal, Combine the Update"

**Claim**: Muon's momentum buffer m_t = β·m_{t-1} + (1-β)·g_t mixes gradients
from different modalities across time. This is suboptimal because:

1. Text gradients from step t-5 are stale for audio optimization at step t
2. The optimal momentum rate β differs between modalities (text converges
   faster → needs lower β, audio converges slower → needs higher β)

**Proposed method**: Maintain modality-decomposed momentum buffers:
    m_text = β_text · m_text + (1-β_text) · g_text
    m_audio = β_audio · m_audio + (1-β_audio) · g_audio
    m_combined = m_text + m_audio
    update = NS(m_combined)

This is cheap (just bookkeeping) and doesn't require separate backward passes.

**Key difference from dual-NS**: Dual-NS (which failed) applied NS separately
per modality. Here we combine BEFORE NS, preserving the spectral normalization
benefit while allowing modality-specific momentum dynamics.

**Testable prediction**:
- Decomposed momentum should improve val_loss by allowing different β per modality
- The optimal β_audio > β_text (slower modality needs more temporal smoothing)
- This should specifically improve early-training dynamics when gradients are noisiest

**Novel contribution**: First modality-aware momentum scheme that preserves
Muon's spectral properties. Generalizes to any multi-task Muon setting.

---

### H7: Information-Theoretic Batch Composition

**Claim**: The optimal text:audio ratio in each batch is NOT fixed but should
maximize the Fisher information of the batch:

    α*(t) = argmax_α  I_Fisher(batch(α)) / cost(batch(α))

where α is the audio fraction and I_Fisher is the Fisher information matrix trace.

**Intuition**: Early in training, audio gradients are highly informative (high
Fisher info per token) because the model knows nothing about audio. Late in
training, audio gradients become less informative as the model converges. The
optimal α should decrease over training.

BUT: our H2 (progressive upweighting) increased α over time and failed. This
suggests the opposite — early audio gradients are NOISY (low signal-to-noise),
not informative. The correct schedule may be:

    α(t) = α_base × SNR_audio(t) / SNR_text(t)

**Testable prediction**:
- Measure per-modality gradient SNR (signal/noise ratio) during training
- The SNR ratio should predict the optimal α schedule
- If SNR_audio increases over training → progressive upweighting should work
  (contradicts our result, revealing a bug in H2 implementation)
- If SNR_audio decreases → progressive downweighting should work

**Novel contribution**: First information-theoretic framework for batch
composition in multi-modal training. Replaces heuristic mixing ratios.

---

### H8: The Transformer as a Multi-Modal Routing Network

**Claim**: Attention heads in a shared transformer naturally specialize — some
heads primarily attend within text sequences, others within audio sequences,
and some cross-attend. This specialization happens WITHOUT explicit supervision.

**Deeper theory**: The softmax attention mechanism implements a soft routing
function. With sufficient capacity, the model learns to:
1. Allocate ~70% of heads to text (matching data proportion)
2. Allocate ~30% of heads to audio
3. A small number of "bridge" heads that cross-attend

The optimizer's job is NOT to balance modalities — it's to let this natural
specialization happen. Gradient manipulation (H2-H6) INTERFERED with natural
head specialization, which is why they all failed.

**Testable prediction**:
- Measure per-head attention entropy on text-only vs audio-only inputs
- Heads should cluster into text-specialist, audio-specialist, and generalist
- The ratio of specialists should correlate with data proportions
- Artificially balancing gradients should REDUCE this specialization

**Novel contribution**: Mechanistic understanding of how multi-modal transformers
self-organize. Connects to the modularity literature in neuroscience.

---

### H9: Scale-Dependent Optimizer Selection

**Claim**: Different parameter groups in an omni-model occupy different regions
of the loss landscape, requiring fundamentally different optimization strategies:

| Parameter Group | Landscape Character | Optimal Optimizer |
|----------------|-------------------|-------------------|
| Embeddings (73%) | Sparse, per-row independent | Per-row SGD with adaptive WD |
| Attention matrices | Low-rank, smooth | Muon (spectral descent) |
| MLP matrices | Full-rank, sharp | Muon with higher WD |
| Layer norms/scales | Convex, fast | Adam with high LR |

**Key insight**: We currently use Muon for matrices and Adam for everything else.
But the embedding table is NEITHER a matrix (each row is independently updated
based on which tokens appear) NOR a 1D parameter. It's a *dictionary* — a
collection of independent vectors indexed by token ID.

**Proposed method**: Design a dictionary-specific optimizer:
    For each token i that appeared in the batch:
        e_i ← e_i - lr_i · g_i / (sqrt(v_i) + eps)
    where lr_i and v_i are per-token adaptive.
    Tokens NOT in the batch get no update (not even WD).

This is "lazy Adam" but with the key insight that WD should ONLY apply to
tokens that were updated (otherwise WD shrinks unseen tokens unfairly).

**Testable prediction**:
- Lazy WD (only decay tokens in batch) should outperform global WD
- The difference should be larger with sparser audio tokens
- Per-token v_i should reveal frequency-dependent learning dynamics

**Novel contribution**: A theoretically motivated per-token optimizer for
embedding tables in heterogeneous-vocabulary models.

---

### H10: Cross-Modal Representation Pressure

**Claim**: In a shared transformer, text and audio tokens compete for
representational capacity in the residual stream. The 512-dim residual stream
must simultaneously encode:
- Text semantics (syntax, semantics, world knowledge)
- Audio acoustics (phonemes, prosody, speaker identity)
- Cross-modal alignment (text↔audio correspondence)

This creates a **representation pressure** that manifests as:
1. Audio features colonizing text subspace → text quality degrades
2. Text features dominating → audio quality suffers
3. Compromise → both suboptimal

**Connection to our findings**: The massive WD improvement (Phase 3) worked
because it REDUCED the effective dimensionality of audio representations,
freeing residual stream capacity for text. With more data (Phase 7), audio
embeddings learned more compact representations naturally, eliminating the
need for WD.

**Proposed experiment — Orthogonality Pressure (OP)**:
Add a soft constraint encouraging text and audio representations to be orthogonal
in the residual stream:

    L_OP = λ · ||mean(h_text) · mean(h_audio)^T||_F

This explicitly separates the modality subspaces, potentially allowing both
to use their full allocated dimensions without interference.

**Testable prediction**:
- Measuring cos(h_text, h_audio) per layer should show high alignment
  (representation pressure exists)
- Adding L_OP should reduce this alignment
- With L_OP, the model should achieve better text AND audio simultaneously
- The effect should be larger with smaller d_model (more pressure)

**Novel contribution**: First identification of representation pressure as a
fundamental challenge in omni-models. The OP loss provides a principled solution
that's architecture-agnostic.

---

## Priority Ranking for Testing

By expected impact × feasibility:

1. **H2: Spectral Geometry** (diagnostic — reveals ground truth about our system)
2. **H1: Bayesian Embedding** (closed-form prediction, easy to test)
3. **H3: Muon's Implicit Balancing** (explains our key finding)
4. **H4: Embedding Bottleneck** (diagnostic + architectural insight)
5. **H6: Momentum Decomposition** (concrete algorithm, moderate code change)
6. **H9: Scale-Dependent Optimizer** (lazy WD is easy to implement)
7. **H10: Representation Pressure** (requires internal probing)
8. **H5: Phase Transition** (requires multiple data scales)
9. **H7: Info-Theoretic Batch** (requires gradient SNR measurement)
10. **H8: Routing Network** (attention analysis, less actionable)

## Recommended Research Plan

**Phase A (Diagnostic)**: Run H2 + H4 probes to understand ground truth geometry.
These are pure measurement experiments — no training changes, just instrumentation.

**Phase B (Theory-Guided)**: Based on Phase A results, implement H1 (Bayesian WD)
and H3 (Muon analysis). These have the clearest theoretical predictions.

**Phase C (Novel Algorithms)**: Implement H6 (momentum decomposition) and H9
(lazy WD). These are the most likely to produce NeurIPS-quality algorithmic contributions.

**Phase D (Architecture)**: If representation pressure (H10) is confirmed,
design the orthogonality loss. This could be the "big idea" paper.
