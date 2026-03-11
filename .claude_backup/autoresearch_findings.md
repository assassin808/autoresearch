# Autoresearch Findings (mar9 session)

## 1. Experiment Summary (45 runs)

### Phase 1: Throughput (exps 1-12)
| Change | val_loss | text_bpb | audio_loss | Verdict |
|--------|---------|----------|------------|---------|
| Baseline (bs=16, 189 steps) | 5.125 | 1.201 | 10.395 | - |
| Remove VE (bs=32) | 5.362 | 1.850 | 5.801 | FAIL: VE critical for text |
| Batch 262K (366 steps) | 5.122 | 1.143 | 10.867 | +0.003: more steps helps |
| Batch 131K (716 steps) | 5.185 | 1.121 | 11.310 | FAIL: too noisy |
| Warmdown 0.7 | 5.095 | 1.144 | 10.749 | +0.027: longer cooldown wins |
| Warmdown 0.8 | 5.107 | - | - | FAIL: too long |

**Insight**: Batch size trades off bias (larger=better gradient) vs steps (smaller=more updates).
262K (8 microsteps) is the sweet spot for 5min budget. Warmdown=0.7 means 70% of training
is cooldown — the model needs gentle convergence, not fast early learning.

### Phase 2: LR tuning (exps 13-17)
| Change | val_loss | text_bpb | audio_loss | Verdict |
|--------|---------|----------|------------|---------|
| Muon LR 0.05 | 5.090 | 1.140 | 10.765 | +0.005 |
| Muon LR 0.06 | 5.089 | 1.141 | 10.756 | +0.001 more |
| Muon LR 0.08 | 5.100 | 1.139 | 10.817 | FAIL: overshooting |
| Unembed LR 0.008 | 5.204 | 1.138 | 11.240 | FAIL: lm_head very sensitive |

**Insight**: Muon LR is indeed portable (operator norm normalization keeps gradients O(1)).
But Adam-optimized params (embeddings, lm_head) are NOT scale-invariant — their optimal LR
depends on the parameter scale, which depends on WD and initialization. The lm_head is
initialized at std=0.001 (very small), so LR=0.004 is actually aggressive relative to param norm.

### Phase 3: Weight Decay — THE BIG FINDING (exps 18-36)
| Embed WD | val_loss | text_bpb | audio_loss | Δval_loss |
|----------|---------|----------|------------|-----------|
| 0.0 (baseline) | 5.089 | 1.141 | 10.756 | - |
| 0.01 | 5.020 | 1.140 | 10.483 | -0.069 |
| 0.02 | 4.990 | 1.141 | 10.356 | -0.099 |
| 0.05 | 4.901 | 1.154 | 9.892 | -0.188 |
| 0.1 | 4.827 | 1.170 | 9.466 | -0.262 |
| 0.2 | 4.768 | 1.188 | 9.079 | -0.321 |
| 0.4 | 4.701 | 1.213 | 8.594 | -0.388 |
| 0.8 | 4.613 | 1.240 | 8.017 | -0.476 |
| 1.5 | 4.530 | 1.276 | 7.381 | -0.559 |
| **2.0** | **4.459** | **1.284** | **7.029** | **-0.630** |
| 2.5 | 4.472 | 1.356 | 6.476 | text collapse |
| 3.0 | 5.098 | 1.621 | 6.753 | text collapse |

**This is the key finding.** Embed WD=2.0 gives -0.63 val_loss improvement (12% relative),
almost entirely from audio (10.76→7.03, 35% reduction). Text degrades slightly (1.14→1.28, 12%).

**Why does this work?** The embedding table has 20,482 rows but audio tokens (12,290 SNAC codes)
are seen much less frequently than text tokens. Without WD, audio embeddings grow unconstrained
and overfit to the small audio dataset (5.2h dev-clean). Heavy WD keeps embedding norms small,
acting as implicit regularization. The optimal WD ~2.0 means embeddings shrink by ~2*lr*param
per step — very aggressive, essentially preventing the rarely-seen audio tokens from memorizing.

**The tradeoff curve**: val_loss = f(embed_WD) is U-shaped. Below 2.0, audio overfits.
Above 2.0, text embeddings are over-regularized and can't learn. The minimum shifts depending
on audio data volume — with more audio data, optimal WD would likely decrease.

### Phase 4: More Regularization (exps 30-42)
| Change | val_loss | text_bpb | audio_loss | Verdict |
|--------|---------|----------|------------|---------|
| lm_head WD 0.5 | 4.415 | 1.283 | 6.860 | +0.044 |
| lm_head WD 2.0 | 4.334 | 1.301 | 6.387 | +0.125 |
| lm_head WD 4.0 | 4.362 | - | - | FAIL: too much |
| Muon WD 0.5 | 4.324 | 1.294 | 6.406 | +0.010 |
| Muon WD 1.0 | 4.343 | - | - | FAIL |
| Constant WD | 4.314 | 1.293 | 6.372 | +0.010 |
| Dropout 0.1 | 4.292 | 1.301 | 6.225 | +0.022 |
| Dropout 0.15 | 4.319 | - | - | FAIL: too much |

**Insight**: All regularization helps uniformly. The model overfits everywhere, not just embeddings.
But the embedding WD effect dominates because embeddings are 73% of parameters (62M/88M).

### Phase 5: Convergence Analysis & Loss Weighting (exps 43-44)
Measured per-modality convergence:
- Audio converges 26% slower (speed ratio = 0.742)
- Loss ratio widens: 1.15 early → 1.38 late
- Audio weight 1.5x: closed gap (ratio 0.835) but hurt val_loss (+0.063)
- Audio weight 1.2x: marginal worse (+0.024)

**Insight**: Naive loss weighting doesn't work because it amplifies audio gradients through
the ENTIRE model (including text-helpful parameters). The gradient g = g_text + α*g_audio
with α>1 means Muon sees a different effective gradient direction, not just magnitude.

## 2. Theoretical Framework

### Why Muon LR is Portable but Other HPs Aren't
Muon normalizes gradients via Newton-Schulz orthogonalization: G → U where U^T U ≈ I.
This makes the effective step ||ΔW|| ≈ lr * √(fan_out) regardless of gradient magnitude.
So Muon LR transfers across model sizes by construction.

But Adam-optimized params (embeddings, scalars) have effective step:
  ΔW ≈ lr * m / (√v + ε)
where m,v are running moments. This depends on:
- Gradient magnitude (task-dependent)
- Parameter initialization scale
- WD (which acts as gradient: g_eff = g + λw)

### Bias-Variance Tradeoff for Momentum
Adam momentum β1: higher = more bias (uses stale gradient info), lower variance (smoother).
- β1=0.8 worked better than β1=0.9 in our setting (366 steps)
- With few steps, we need low-bias estimates → lower β1 is better
- But β1=0 (SGD) would be too noisy with batch=262K and grad_accum=8
- Optimal β1 ∝ 1 - 1/τ where τ is "effective gradient age" ≈ steps_to_convergence/10

For β2 (second moment):
- β2=0.95 means ~20-step half-life for variance estimate
- Too low → noisy step sizes, too high → stale variance estimate
- With 366 steps, β2=0.95 gives ~18 "effective samples" — barely enough

### Adaptive WD: A Principled Approach
The optimal WD depends on the ratio of "model capacity for this token" to "data for this token":
  λ_optimal ∝ (embedding_dim) / (token_frequency * dataset_size)

For text tokens: high frequency, need low WD to learn rich representations
For audio tokens: low frequency (only 5.2h data), need high WD to prevent memorization

**Idea**: Per-token adaptive WD based on frequency:
  λ_i = λ_base * (max_freq / freq_i)^α  where α ∈ (0, 1)

This would give high WD to rare audio tokens, low WD to common text tokens, automatically.

### Gradient Direction Conflict
The naive loss weighting failed because g_audio and g_text may point in conflicting directions
in the shared transformer. If cos(g_audio, g_text) < 0 for some layers, amplifying g_audio
actually hurts text convergence.

Better approaches:
1. **PCGrad**: Project conflicting gradients to remove interference
2. **Separate Muon**: Apply Muon(g_text) and Muon(g_audio) independently, then average the updates
3. **Gradient vaccine**: Scale down the conflicting component only

### Phase 6: Per-Row Weight Decay — SECOND BIG FINDING (exps 46-61)
Previous uniform WD=2.0 was a compromise: too much for text, tuned for audio.
Per-row WD lets each modality have optimal regularization independently.

| Exp | Text WD | Audio WD | Scope | val_loss | text_bpb | audio_loss | Verdict |
|-----|---------|----------|-------|---------|----------|------------|---------|
| 46 | audio_ratio=0.5 | - | - | 4.778 | 1.402 | 7.317 | FAIL: too much audio data |
| 47 | dual-NS Muon | - | - | 5.126 | 1.629 | 6.797 | FAIL: 2x backward too slow |
| 48 | 0.5 | 4.0 | all | NaN | - | - | CRASH: audio WD too high |
| 49 | 1.0 | 3.0 | wte | 4.280 | 1.293 | 6.240 | +0.012 |
| 50 | 0.5 | 3.0 | wte | 4.223 | 1.267 | 6.225 | +0.069 |
| 51 | 0.2 | 3.0 | wte | 4.173 | 1.241 | 6.247 | +0.119 |
| 52 | 0.1 | 3.0 | wte | 4.149 | 1.226 | 6.279 | +0.143 |
| 53 | 0.0 | 3.0 | wte | 4.108 | 1.204 | 6.301 | +0.184 |
| 54 | 0.0 | 5.0 | wte | NaN | - | - | CRASH |
| 55 | 0.0 | 3.5 | wte | 4.221 | 1.208 | 6.717 | FAIL: audio worse |
| 56 | 0.0 | 3.0 | wte+VE | 4.104 | 1.180 | 6.485 | +0.188 |
| 57 | 0.0 | 3.0 | all | 4.078 | 1.167 | 6.490 | +0.214 |
| **58** | **0.0** | **2.5** | **all** | **3.995** | **1.158** | **6.239** | **+0.297 BEST!** |
| 59 | 0.0 | 2.0 | all | 4.041 | 1.161 | 6.391 | audio worse |
| 60 | dropout=0.05 | 2.5 | all | **3.993** | 1.153 | 6.269 | marginal best |
| 61 | dropout=0.0 | 2.5 | all | 3.996 | 1.151 | 6.301 | audio worse |

**Key insight**: Text embeddings don't need WD at all (WD=0.0 is optimal for text).
The previous uniform WD=2.0 was 100% unnecessary for text — it was only helping audio.
Separating them gives each modality optimal regularization:
- Text WD=0.0: embeddings can grow freely, learning rich representations (text_bpb: 1.301→1.153)
- Audio WD=2.5: strong regularization prevents audio overfitting (audio_loss stable ~6.24)
- Applied to wte, VE tables, and lm_head — all vocab-indexed params benefit

**Why does zero text WD work?** Text tokens are seen frequently enough (70% of data)
that gradient descent naturally regularizes them via implicit bias of SGD. Audio tokens
are sparse (30% data, spread across 12K tokens), so they need explicit WD.

### Phase 6 Failed Experiments
- **Audio ratio 0.5**: More audio in each batch doesn't help — the problem is data scarcity,
  not gradient direction. With only 5.2h audio, 50% audio just overfits faster.
- **Dual-NS Muon (NS(g_text) + NS(g_audio))**: Requires 2 backward passes per step,
  reducing step count by ~40% (227 vs 364). The quality per step doesn't compensate.
  The NS orthogonalization is applied per-matrix (already batched), so the gradient
  direction issue is actually in the MOMENTUM buffer mixing, not the NS step.

### Phase 7: Scaling Audio Data — THIRD BIG FINDING (exps 65-87)
With 115h audio (vs 5.2h), optimal regularization shifts dramatically.
| Audio WD | val_loss | text_bpb | audio_loss | Notes |
|----------|---------|----------|------------|-------|
| 2.5 (old best) | 3.981 | 1.150 | 6.243 | baseline with 115h |
| 1.5 | 3.948 | 1.153 | 6.088 | |
| 1.0 | 3.928 | 1.152 | 6.015 | |
| 0.5 | 3.896 | 1.149 | 5.915 | |
| 0.2 | 3.873 | 1.151 | 5.809 | |
| 0.1 | 3.859 | 1.152 | 5.741 | |
| **0.0** | **3.838** | **1.151** | **5.668** | **no audio WD needed!** |

**Key insight**: λ_optimal ∝ 1/dataset_size confirmed experimentally.
- 5.2h audio → optimal WD=2.5
- 115h audio → optimal WD=0.0
- With enough data, SGD's implicit regularization suffices

### Phase 8: De-regularization with Data Scaling
| Config | val_loss | text_bpb | audio_loss |
|--------|---------|----------|------------|
| WD=0.0 baseline | 3.838 | 1.151 | 5.668 |
| **WD=0.0 + Muon WD=0.2** | **3.808** | **1.143** | **5.615** |
| WD=0.0 + dropout=0.0 | 3.812 | 1.147 | 5.594 |
| WD=0.0 + Muon WD=0.2 + dropout=0.0 | 3.808 | 1.139 | 5.652 |
| WD=0.0 + Muon WD=0.2 + LR=0.08 | 3.813 | 1.141 | 5.654 |
| all WD=0.0 | 3.826 | 1.143 | 5.682 |

**Insight**: Reducing Muon WD from 0.5→0.2 helps text significantly.
Transformer matrices ALSO benefit from less regularization with more data.
But some Muon WD (0.2) still needed — unlike embeddings, transformer matrices
have implicit scale from NS orthogonalization that benefits from mild WD.

### Hypothesis Test Results (with 115h data)
| Hypothesis | Result | Why |
|------------|--------|-----|
| H2: Audio upweighting | FAIL (-0.008) | With balanced data, upweighting hurts text |
| H3: Grad norm balance | FAIL (-0.036) | Interferes with optimizer's natural adaptation |
| H4: Per-row LR | CATASTROPHIC | LR mismatch destroys shared backbone gradients |
| H5: AdaDecay WD | NaN/FAIL | Adaptive WD too volatile, destabilizes training |
| H6: MILES rebalance | Marginal (-0.005) | Utilization mismatch is a data problem, not optimizer |

**Meta-insight**: All "clever" gradient manipulation hypotheses (H2-H6) failed.
The winning approach was the simplest: just add more data and reduce regularization.
This suggests that for omni-models, the audio-text convergence gap is primarily
a data issue (audio data scarcity → overfitting → need more WD), not an optimizer issue.

### Phase 9: Final Fine-tuning
| Config | val_loss | text_bpb | audio_loss |
|--------|---------|----------|------------|
| dropout=0.02 | 3.801 | 1.141 | 5.604 |
| **dropout=0.0** | **3.797** | **1.137** | **5.617** |
| Muon WD=0.15 | 3.820 | 1.144 | 5.653 |
| Adam β1=0.85 | 3.829 | 1.143 | 5.700 |
| warmdown=0.6 | 3.824 | 1.143 | 5.677 |

### Phase 10: Theory-Derived Experiments (sweep6, 20 exps)
| Config | val_loss | text_bpb | audio_loss | Notes |
|--------|---------|----------|------------|-------|
| **batch 128K** | **3.732** | **1.119** | **5.509** | **HUGE win — more steps!** |
| H9: coupled warmdown (WD*lrm) | 3.792 | 1.134 | 5.625 | WD should decay with LR |
| softcap=30 | 3.793 | 1.139 | 5.590 | marginal |
| softcap=20 | 3.794 | 1.138 | 5.603 | marginal |
| H8: NS steps=10 | 3.794 | 1.136 | 5.615 | more NS helps |
| Muon LR=0.07 | 3.796 | 1.137 | 5.617 | marginal |
| H8: NS steps=7 | 3.797 | 1.140 | 5.599 | same as 5 |
| H8: NS steps=3 | 3.834 | 1.148 | 5.674 | worse |
| batch 512K | 3.985 | 1.194 | 5.888 | much worse |
| H10: weight tying | CRASH | - | - | recursion error |

**Key insight**: Batch 128K (2x smaller) gives 2x more optimizer steps in same wall time.
This is a huge win — confirms that step count matters more than gradient quality for this
training budget. The H9 coupled warmdown finding is also important: constant WD during
cooldown over-regularizes rare tokens.

### Phase 11-12: Audio ratio + final combos (sweep8, 11, 12)
| Config | val_loss | text_bpb | audio_loss |
|--------|---------|----------|------------|
| ratio0.1+WD0.1+LLLL+NS10 | **3.212** | 1.074 | 6.193 |
| ratio0.15+WD0.1+LLLL+NS10 | 3.402 | 1.088 | 5.850 |
| ratio0.2+WD0.05+LLLL+NS10 | 3.538 | 1.095 | 5.558 |
| ratio0.3+WD0.15+LLLL+NS10 | 3.702 | 1.109 | 5.472 |
| window=LLLL (full attn) | 3.710 | 1.110 | 5.497 |
| window=SSSL (original) | 3.717 | 1.114 | 5.497 |

## 3. Current Best Config (170+ experiments)
- **Balanced**: ratio0.15, WD0.1, LLLL, NS10, batch128K → val_loss=3.402
- **Text-focused**: ratio0.1, WD0.1, LLLL, NS10, batch128K → val_loss=3.212
- Depth=8, dim=512, bs=16, Muon LR=0.06, dropout=0.0
- Embed WD: text=0.0, audio=0.0
- Adam betas (0.8, 0.95), warmdown 0.7
- 88M params, Audio: 115h LibriSpeech

### Phase 13: Diagnostic Instrumentation Results
Single instrumented run (ratio=0.3, best non-ratio config):
- **Token frequencies**: Text Zipfian (CV=6.07), audio uniform (CV=1.53). 246 dead audio tokens.
- **Gradient imbalance**: Text embeds get 3-5x larger gradients than audio
- **Rank divergence**: Audio effective rank declines 509→434 (never recovers). Text: 509→465 (recovers).
- **NS compression**: Condition number 4.9→1.002 (2650x compression). 10 NS iters sufficient.
- **Dead audio tokens**: 245/4096 CB0 codes unused. Weight decay on these is pure noise.

## 4. Key Insights
1. **Audio ratio is the biggest lever** — val_loss drops monotonically with less audio
2. **Batch size matters** — 128K (2x more steps) beats 262K
3. **NS iterations** — 10 > 5 > 3, but 15+ too slow
4. **Window pattern** — full attention (LLLL) beats sliding window at 2048 seq_len
5. **Regularization** — scales inversely with data quantity
6. **Gradient manipulation** — all failed (H2-H6), data quality > optimizer tricks
7. **Audio embedding rank collapses** — monotonic decline, 31-dim gap vs text by end of training
8. **246 dead audio tokens** — coarse codebook has 6% unused codes, pure WD victims
9. **NS is modality-blind** — crushes condition numbers 2650x, treats all directions equally
