# From 100 Experiments to Theory: Omni-Model Training Dynamics

## Part 1: What We Observed

88M-param shared transformer, text (600M tokens) + audio (SNAC, 5.2h→115h),
Muon optimizer, 5-minute training budget, 100+ controlled experiments.

### Pattern A: Regularization scales inversely with data — then vanishes

| Audio data | Optimal embed WD | Optimal dropout | Optimal Muon WD |
|-----------|-----------------|----------------|-----------------|
| 5.2h      | 2.5             | 0.05           | 0.5             |
| 115h      | 0.0             | 0.0            | 0.2             |

Not a gradual decline — embed WD went to exactly zero. Dropout went to exactly
zero. Only Muon WD retained any value (0.2). This is not "less regularization
is better" — it's "regularization compensates for missing data, nothing more."

### Pattern B: Text never needed regularization at any scale

Text WD=0.0 was optimal in EVERY experiment. We swept text WD from 0.0 to 1.0 —
always worse with any WD. Text has 600M tokens, ~8K vocab. Audio has 1.2M→34M
tokens, ~12K vocab. The difference is data-per-token, not anything architectural.

### Pattern C: ALL gradient manipulation failed

Five different approaches to "help" audio learn better:
- Progressive audio upweighting (H2): −0.008
- Gradient norm balancing (H3): −0.036
- Per-row adaptive LR (H4): catastrophic collapse
- AdaDecay adaptive WD (H5): NaN
- MILES utilization rebalancing (H6): −0.005

Not one of them helped. Not even marginally. Five different theoretical
motivations, five different implementations, all worse than doing nothing.

### Pattern D: Muon LR is invariant, Adam LR is not

Muon LR=0.06 worked optimally across ALL configurations — different data scales,
different WD, different batch sizes. Meanwhile, embedding LR, WD, and dropout
were all highly sensitive to the data distribution. The Muon-optimized parameters
(transformer matrices) were "set and forget." The Adam-optimized parameters
(embeddings, lm_head) required constant retuning.

### Pattern E: Embed WD and Muon WD scale differently

With 22x more audio data:
- Embed WD: 2.5 → 0.0 (complete elimination)
- Muon WD: 0.5 → 0.2 (partial reduction)

Same data change, qualitatively different response. The transformer matrices
still want SOME regularization even with abundant data. The embeddings want NONE.

### Pattern F: Per-row WD was the single biggest win

Going from uniform WD to per-row WD (text=0, audio=2.5) gave:
- val_loss: 4.292 → 3.993 (7% improvement)
- This single change was larger than ALL other hyperparameter tuning combined

The effect was entirely from giving text embeddings freedom (WD 2.0→0.0).
Audio WD barely changed (2.0→2.5). The uniform WD was over-regularizing text
to compensate for audio's need — a Pareto-suboptimal compromise.

---

## Part 2: What These Patterns Mean

### Observation 1: "Regularization is a tax on missing data"

This is the central finding. Every form of regularization (WD, dropout, even
implicit regularization from small batch size) was compensating for insufficient
audio data. When we added 22x more audio:
- Embed WD: unnecessary (audio tokens seen enough times)
- Dropout: unnecessary (model doesn't overfit with enough data)
- Even Muon WD partially unnecessary (0.5→0.2)

**This is not obvious.** The standard view is that regularization controls
model complexity — you need it because the model is "too big." Our result says:
the model is not too big, the DATA is too small for some tokens. Fix the data
imbalance, and regularization goes to zero.

**Implication**: For production omni-models, the regularization schedule should
be a function of per-token data coverage, not model size.

### Observation 2: "Gradient manipulation is a solution to a non-problem"

All five gradient manipulation methods assumed the same premise: "audio
converges slower because its gradients are somehow deficient — too small,
wrong direction, wrong scale." But the actual problem was simpler: audio
tokens didn't have enough data to learn good embeddings.

Gradient manipulation changes the DIRECTION of optimization but can't create
information that isn't in the data. With 5.2h audio, each of the 12K audio
tokens is seen ~100 times — there's simply not enough signal to learn a good
512-dim embedding, no matter how you massage the gradient.

**Implication**: Before designing clever optimizers for multi-modal convergence
gaps, check if the gap is just a data gap. If it is, no optimizer can fix it.

### Observation 3: "Muon and Adam optimize fundamentally different objects"

Muon parameters (transformer matrices) behave like continuous functions — smooth
loss landscape, scale-invariant updates, portable hyperparameters. Adam parameters
(embeddings) behave like lookup tables — sparse updates, frequency-dependent
dynamics, hyperparameters coupled to data distribution.

This is not an accident. NS orthogonalization makes Muon updates O(1) in operator
norm regardless of gradient magnitude. But Adam's update size depends on the
running gradient statistics, which are per-parameter and data-dependent.

**Implication**: We should not be using the same optimizer paradigm for both.
Embeddings need a frequency-aware optimizer. Matrices need a geometry-aware
optimizer. Muon is the right answer for matrices. Adam is a mediocre answer
for embeddings.

---

## Part 3: Hypotheses (testable, from the patterns)

### H1: Optimal per-token WD = C / token_frequency

From Pattern A+B: text (high freq) → WD=0, audio (low freq) → WD>0.
From Pattern A: more data → lower WD.

Simplest model that explains both: λ_i = C / f_i where f_i is the number
of times token i appears in training.

**Test**: Compute per-token frequencies. Set WD = C/f_i for each embedding row.
Sweep C. If this single-parameter model matches or beats our hand-tuned binary
WD, it confirms the hypothesis and gives a general recipe.

### H2: Muon implicitly equalizes modality gradients via NS

From Pattern C+D: gradient manipulation failed, Muon LR is invariant.

NS computes G → G(G^TG)^{-1/2}, which whitens the gradient singular spectrum.
If text dominates the top singular values of G, NS automatically downweights
text and upweights audio — doing exactly what H2-H6 tried to do manually.

**Test**: Log the singular value spectrum of the raw gradient vs the NS-processed
update for a few transformer layers. If NS flattens the spectrum AND the
top singular vectors correlate with text-token gradients, the hypothesis is
confirmed.

### H3: Embedding tables need a dictionary optimizer, not a matrix optimizer

From Pattern E+F: embeddings and matrices have qualitatively different scaling.

An embedding table is a collection of independent vectors, not a matrix. Each
row is updated only when its token appears. Current AdamW treats it as a dense
matrix — applying WD to ALL rows every step, even tokens not in the batch.

**Test**: Implement "lazy WD" — only decay embedding rows that were actually
updated this step. This should:
- Eliminate the need for per-row WD tuning (rare tokens naturally decay less)
- Improve audio token embeddings (they stop being decayed when not in batch)
- Be a strictly better default than global WD

### H4: The convergence gap is explained by effective sample size per token

From Pattern A+C: data quantity explains everything, gradient quality explains nothing.

Define effective sample size: ESS_i = freq_i × num_steps. The convergence
quality of token i is a function of ESS_i alone. The "convergence gap" between
text and audio is simply the gap in ESS.

**Test**: Plot val_loss contribution per token vs ESS. Should be a universal
curve (same for text and audio tokens at equal ESS). If confirmed, this means
there is NO modality-specific optimization problem — just a data imbalance problem.

### H5: Representation measurement — do modalities compete in the residual stream?

From Pattern F: the biggest single improvement came from freeing text embeddings
from audio's WD. This suggests text and audio representations may compete for
"space" in the residual stream.

**Test**: During evaluation, collect hidden states h for text-only and audio-only
inputs. Compute the principal angles between the text and audio subspaces per
layer. If they overlap significantly, there IS competition. If they're nearly
orthogonal, the transformer already separates them and competition is not an issue.

This is a pure diagnostic — the result tells us whether architectural changes
(separate residual streams, modality-specific layers) would help.

### H6: Embedding rank deficiency — audio embeddings are undertrained, not misregularized

From Pattern A+C: more data eliminated the need for WD, and no gradient trick helped.

With 5.2h audio, each of the 12K audio tokens is seen ~100 times. A 512-dim
embedding needs O(512) independent gradient updates to span its full space. At
~100 updates, the embedding lives in a low-rank subspace — it hasn't explored
most directions. WD's actual effect was preventing these undertrained embeddings
from drifting into high-norm noise directions, not "regularizing" in the
classical sense.

**Test**: Compute the effective rank (exp(entropy of normalized singular values))
of the audio embedding submatrix vs the text embedding submatrix at checkpoints
throughout training. Audio embeddings should have much lower effective rank.
If effective rank correlates with per-token frequency and with val_loss
improvement, this confirms that WD was a crude substitute for "stay low-rank
until you have enough data."

**Implication**: Replace WD with an explicit rank constraint on low-frequency
token embeddings. This is more principled — it constrains the RIGHT thing
(don't explore dimensions you can't learn) rather than a proxy (keep norms small).

### H7: The 73% embedding bottleneck — model is overparameterized in lookup, underparameterized in computation

From Pattern E+F: embeddings account for 62M/88M parameters (73%). Text and
audio modalities interact only through the shared transformer layers (26M params).

The standard transformer scaling intuition assumes embed_dim ≈ model_dim, with
most parameters in attention and FFN. But our vocab is 20K+ (8K text + 12K audio),
so embedding tables dominate. The transformer's actual "thinking capacity" is
only 26M parameters — less than a 20M-param text-only GPT2.

**Test**: Compare two configurations at fixed total parameter budget:
(a) Current: 512 embed_dim, 512 model_dim (62M embed, 26M transformer)
(b) Decoupled: 256 embed_dim → 512 model_dim via learned linear projection
    (31M embed + 0.3M projection + 26M transformer = 57M, leaving room for
    an extra transformer layer or wider FFN)

If (b) wins, it confirms the model is starved for compute capacity, not
embedding capacity. The embeddings don't need 512 dimensions — they need
enough dimensions to be LINEARLY SEPARABLE, which for 20K tokens is far
less than 512.

### H8: Newton-Schulz iteration count as an implicit modality balancer

From Pattern D: Muon LR=0.06 was invariant across all configurations.
From Pattern C: all manual gradient balancing failed.

Muon uses 5 iterations of Newton-Schulz to approximate (G^T G)^{-1/2}. This
is an APPROXIMATION — it doesn't fully converge. The approximation error is
larger when the gradient matrix has a wider singular value spread (high
condition number). Text-dominated batches likely produce wider-spread gradients
than audio-dominated batches.

**Test**: Vary NS iterations from 3 to 10. If the "implicit balancing" theory
(H2) is correct, then MORE NS iterations should make the update MORE balanced
across modalities — and performance should IMPROVE. But if the approximation
error is actually HELPING (by adding noise that acts as regularization for the
dominant modality), then more iterations could HURT.

This distinguishes between "NS is the right operation" vs "NS approximation
error is a useful regularizer" — two very different stories with different
engineering implications.

### H9: Warmdown schedule is doing double duty — LR decay + implicit data reweighting

From Pattern A: regularization's effect depends on data quantity.
From sweep results: warmdown_ratio=0.7 was optimal (70% of training in cooldown).

During warmdown, LR decreases linearly to 0. But the RELATIVE effect of WD
increases as LR decreases (WD is applied as absolute subtraction, not
LR-scaled). In the final phases of warmdown, WD dominates the update for
rare tokens — effectively erasing what they learned.

This means warmdown is implicitly reweighting the data: in late training,
frequent tokens (text) continue learning while rare tokens (audio) are
effectively frozen by WD-dominated updates. This is a HIDDEN interaction
between schedule and regularization.

**Test**: Compare three conditions at Muon WD=0.2:
(a) Standard warmdown (WD constant, LR decreases)
(b) Coupled warmdown (WD decreases proportionally with LR)
(c) Inverse warmdown (WD decreases faster than LR)

If (b) or (c) wins, it confirms that the WD/LR ratio interaction is meaningful
and that current warmdown schedules are accidentally data-reweighting.

### H10: Shared lm_head creates a modality tug-of-war in output space

From Pattern B+F: text never needed WD, but text embeddings suffered from
audio's WD being applied uniformly. The model uses weight-tied embeddings
(wte = lm_head).

Weight tying means the embedding must simultaneously serve two masters:
(1) input representation — map tokens to vectors that the transformer can process
(2) output classification — define a linear classifier that separates 20K+ tokens

For text tokens (well-covered), both objectives agree. For audio tokens
(sparse), the input objective wants embeddings that capture acoustic structure,
while the output objective wants embeddings that are well-separated from ALL
other tokens. These can conflict — the optimal input embedding for audio token
#5000 may overlap with text token embeddings, but the output head needs them
separated.

**Test**: Untie the weights — use separate embedding and lm_head matrices.
The parameter cost is +10M (another copy of the embedding table), which
increases total params from 88M to 98M. But the question is not about
parameter count — it's about whether the tug-of-war is hurting audio:
(a) If untying improves audio_loss but not text_bpb: confirms modality conflict
(b) If untying helps both: weight tying was a bad default for mixed-vocab models
(c) If untying helps neither: the conflict doesn't exist at this scale

---

## Part 4: Execution Plan

**Step 1: Diagnostic instrumentation** (ONE training run, no optimizer changes)
- Per-token gradient norms and frequencies (H1, H4)
- Singular spectrum before/after NS (H2, H8)
- Embedding effective rank per modality (H6)
- Hidden state geometry — principal angles between modalities (H5)
- WD/LR ratio tracking during warmdown (H9)

**Step 2: Quick architectural experiments** (3-4 runs)
- Untied lm_head (H10) — single flag change
- Decoupled embed_dim (H7) — requires projection layer
- NS iteration count sweep: 3, 5, 7, 10 (H8)

**Step 3: Optimizer experiments** (based on Step 1 diagnostics)
- Frequency-proportional WD: λ_i = C/f_i (H1)
- Lazy WD: only decay updated rows (H3)
- Coupled warmdown: WD decreases with LR (H9)

**Step 4: Write up.** The story:
1. 100+ controlled experiments on omni-model training dynamics
2. Six empirical patterns → three observations → ten hypotheses
3. Core finding: regularization compensates for data scarcity, not model size
4. Muon's NS as an implicit multi-modal gradient balancer
5. Practical recipes: lazy WD, frequency-proportional WD, decoupled embeddings

---

## Part 5: Experimental Results (120+ experiments)

### Hypothesis test results

| Hypothesis | Result | val_loss | Verdict |
|------------|--------|----------|---------|
| **H1: Freq-proportional WD C=0.5** | **3.783** | -0.014 | **CONFIRMED** — best embed WD scheme |
| H1: C=1.0 | 3.791 | -0.006 | also good |
| H1: C=2.0 | 3.803 | +0.006 | too much WD |
| H3: Lazy WD (audio WD=0.5) | 3.861 | +0.064 | FAIL — moot since WD=0 is optimal |
| H8: NS steps=3 | 3.834 | +0.037 | fewer NS hurts |
| H8: NS steps=7 | 3.797 | 0.000 | same as 5 |
| **H8: NS steps=10** | **3.794** | **-0.003** | **small improvement** |
| **H9: Coupled warmdown (WD*lrm)** | **3.792** | **-0.005** | **CONFIRMED** — WD should decay with LR |
| H9: sqrt-coupled | 3.813 | +0.016 | linear coupling is better |
| H10: Weight tying | CRASH | - | needs different implementation |

### Batch size is the biggest lever

| Batch size | val_loss | Steps | Verdict |
|-----------|---------|-------|---------|
| 512K | 3.985 | ~183 | too few steps |
| **128K** | **3.732** | **~732** | **BEST — 2x more steps** |
| 256K (baseline) | 3.797 | ~366 | good |

Smaller batch = more optimizer steps in fixed wall time = better. This is the
single largest improvement found in all experiments.

### Other notable findings (sweep7)

| Change | val_loss | Verdict |
|--------|----------|---------|
| **x0_lambda_init=0.05** | **3.782** | **stronger residual helps** |
| x0_lambda_init=0.2 | 3.783 | also good |
| GELU activation | 3.865 | ReLU^2 is better |
| SiLU activation | 3.866 | ReLU^2 is better |
| depth=10 AR=48 | 3.815 | worse at this budget |
| depth=6 AR=88 | 3.896 | much worse |
| depth=12 AR=40 | 3.858 | worse |
| Muon momentum=0.90 | 3.802 | slightly worse |
| Muon WD linear decay | 3.803 | worse than coupled warmdown |
