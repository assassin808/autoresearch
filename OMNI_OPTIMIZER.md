# Omni-Optimizer: Solving Multi-Modal Convergence in Shared-Parameter Models

## Problem

In omni-models, text and audio share transformer backbone weights. Text dominates
the shared optimization path due to higher information density per token (~50×).
Audio representations get suppressed. Even Mini-Omni with 8,600h audio reports
"direct audio reasoning is highly challenging."

**The battlefield is the shared transformer matrices, NOT embeddings.**

## Approaches (8 distinct methods from different mathematical perspectives)

### A1: Alternating Pure Batches + Dual Momentum (Gradient Decomposition)
**Math**: Decompose G = G_text + G_audio by using pure-modality batches.
Maintain separate Muon momentum buffers: M_text, M_audio.
NS is applied independently per modality step.
**Status**: Implemented in sweep37.

### A2: Per-Modality Momentum/LR Schedules
**Math**: Audio steps use higher β₁ (accumulate sparse signal) and/or higher LR.
Text steps use standard settings.
**Status**: Implemented in sweep37.

### A3: Gradient Signal-to-Noise Ratio (SNR) Balancing
**Math**: Define SNR per modality as:
  SNR_m = ||E[G_m]||² / Var(G_m)
where E[G_m] is the expected gradient direction and Var is the variance.
Text has high SNR (consistent semantic gradients). Audio has low SNR (noisy acoustic gradients).
**Method**: Track per-modality gradient variance via exponential moving average.
Scale the learning rate per modality inversely with noise level:
  effective_lr_m ∝ 1 / sqrt(Var(G_m))
This is analogous to Adam's per-parameter adaptive LR, but applied per-modality
at the Muon level.
**Difference from previous**: Previous approaches scaled gradient magnitude (L2 norm).
This scales based on signal quality (SNR), which is second-order information.

### A4: Gradient Subspace Projection (Source Separation)
**Math**: At each step, project the combined gradient G onto two orthogonal subspaces:
1. Compute SVD of recent gradient history: G_history = [G_{t-k}, ..., G_t]
2. Top-k singular vectors capture the dominant optimization direction (text-dominated)
3. Remaining singular vectors capture the subordinate direction (audio-enriched)
4. Boost the subordinate subspace: G' = G + α * Proj_subordinate(G)
**Intuition**: This is blind source separation applied to gradients. Text dominates
the principal components; audio signal lives in the residual. Boosting the residual
amplifies audio without directly scaling anything by modality label.
**Key property**: No need to know which tokens are text vs audio — the method
discovers the modality decomposition from gradient statistics alone.

### A5: Riemannian Manifold-Aware Optimization
**Math**: Muon operates on the Stiefel manifold (orthogonal matrices) via NS.
But NS projects onto the manifold globally, ignoring that text and audio
gradients may lie on different tangent directions of the manifold.
**Method**: After NS orthogonalization, decompose the update U into components
along different principal curvature directions of the loss landscape:
  U = Σ_i σ_i * u_i * v_i^T  (SVD of U)
Weight the singular components by the curvature of the loss in each direction:
  U' = Σ_i (σ_i / sqrt(H_ii)) * u_i * v_i^T
where H_ii is the Hessian eigenvalue along direction i.
Directions with high curvature (typically text, sharp minima) get dampened.
Directions with low curvature (typically audio, flat regions) get boosted.
**Approximation**: Use the Gauss-Newton approximation: H_ii ≈ ||J_i||² where
J_i is the Jacobian along singular direction i. Approximate via the second
moment buffer that Muon already maintains.

### A6: Spectral Norm Induced Descent with Modality Regularization
**Math**: Standard NS targets ||U||_σ = 1. We add a regularization term:
  U = NS(G) + λ * (NS(G) - NS(G_prev))_⊥
where (·)_⊥ denotes the component orthogonal to the previous update.
**Intuition**: This encourages the optimizer to explore new directions each step.
Since text gradients are consistent (always pulling in similar directions), the
orthogonal complement is enriched with audio signal. Over time, this accumulates
a more balanced exploration of the weight space.
**Implementation**: Track previous NS output, compute orthogonal residual, add scaled.

### A7: Matrix Polar Decomposition with Modality-Conditioned Scaling
**Math**: After NS computes the polar factor U = G(G^TG)^{-1/2}, the update
is purely rotational (orthogonal). The scaling information is discarded.
**Method**: Restore partial scaling information based on modality:
  U' = U * (I + γ * diag(s_audio - s_text))
where s_audio, s_text are per-singular-direction contributions from each modality
(estimated from embedding gradient structure).
**Intuition**: NS throws away the singular values. For text (already well-represented),
this is fine. For audio (underrepresented), we want to preserve some of the original
scaling to prevent its signal from being normalized away.

### A8: Gradient Interference Score + Selective Update
**Math**: Compute per-layer "interference score":
  I_l = -min(0, cos(G_l^{t}, G_l^{t-1}))
When interference is high (gradient direction reverses — sign of modality conflict),
reduce the step size for that layer. When interference is low (consistent direction),
take normal steps.
**Intuition**: Layers with high interference are the ones where text and audio
pull weights in opposite directions. Dampening these layers prevents oscillation
and allows both modalities to make slower but more stable progress.
**Implementation**: Track gradient cosine similarity per layer, modulate LR per layer.

## Experimental Design

Each approach is implemented as a modification to the Muon optimizer in train.py.
All operate on transformer matrices (not embeddings). Results tracked in results.tsv.

Sweep 37: A1, A2 (gradient decomposition family)
Sweep 38: A3-A8 (SNR, source separation, manifold, spectral, interference)

## Success Criteria

1. Audio loss improves >2% without text degradation (Pareto improvement)
2. Improvement scales with higher audio mix ratio (proves representation competition solved)
3. Results replicate across seeds
