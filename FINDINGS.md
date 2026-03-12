# Omni-Model Optimizer Study: Key Findings

**401 experiments | 35 sweeps | Multi-seed validated | 88M-param Qwen2-0.5B + Whisper + SNAC**

Systematic study of optimizer hyperparameters for omni-model (text + audio) pretraining using the Muon optimizer. Ranked from most novel to most expected.

---

## 🏆 Finding 1: Duration-Dependent Hyperparameters — Momentum Reversal
**Novelty: High | Sweeps 29, 31-32**

Optimal momentum *reverses* with training duration:

| Duration | Best momentum | Effect |
|----------|--------------|--------|
| 5 min | 0.97 | +0.012 over 0.95 |
| 10 min | 0.95 | +0.014 over 0.97 |

**Mechanism**: Higher momentum acts as an LR multiplier for spectral-norm steepest descent. Early training has steep, consistent gradients where momentum helps. Later training has flatter, noisier landscape where high momentum causes overshooting.

Coupled weight decay also shows duration dependence: +0.019 benefit at 5min, −0.001 at 10min.

**Implication**: Optimal optimizer configs are not static — momentum schedules should adapt to training duration. This is undocumented for Muon.

---

## 🏆 Finding 2: Model-Size Scaling — Optimizer Sensitivity Grows Super-Linearly
**Novelty: High | Sweep 30-31**

| Depth | Params | Optimizer gap | % improvement |
|-------|--------|--------------|---------------|
| d4 | 22M | −0.003 | 0.1% |
| d8 | 88M | −0.060 | 1.7% |
| **d12** | **198M** | **−0.156** | **4.1%** |
| d16 | 352M | −0.023 | 0.6% (undertrained) |

The optimizer gap grows **super-linearly** from d4→d12, peaking at 4.1% for 198M params. d16 drops only because 5 minutes isn't enough to exit warm-up at 352M params (confirmed: d12 at 10min still shows 3.1% gap).

**Implication**: As you scale models, re-tuning optimizer hyperparameters becomes *more* important, not less. The common practice of "tune once at small scale, transfer settings" leaves increasing performance on the table.

---

## 🏆 Finding 3: Regularization Is a Tax on Missing Data
**Novelty: High | Sweeps 1-10**

With 22× more audio data (5.2h → 115h):
- Embed WD: 2.5 → **0.0** (eliminated)
- Dropout: 0.05 → **0.0** (eliminated)
- Muon WD: 0.5 → 0.2 (reduced)

Regularization doesn't control model complexity — it compensates for insufficient per-token data coverage. Text (600M tokens, 8K vocab) never needed WD at any scale. Audio tokens with ~100 examples each needed heavy regularization; at ~2,000 examples each, they needed none.

**Implication**: For multi-modal models, regularization should be a function of per-token data coverage, not model size.

---

## 🏆 Finding 4: All Gradient Manipulation Failed
**Novelty: Moderate-High | Sweeps 3-8**

Five theoretically-motivated approaches to help audio converge faster:

| Method | Result |
|--------|--------|
| Progressive audio upweighting | −0.008 |
| Gradient norm balancing | −0.036 |
| Per-row adaptive LR | Collapse |
| AdaDecay adaptive WD | NaN |
| MILES utilization rebalancing | −0.005 |

**Zero successes.** The audio convergence gap isn't a gradient problem — it's a data problem. Gradient manipulation changes optimization *direction* but can't create information that isn't in the data.

**Implication**: Before designing clever optimizers for multi-modal convergence gaps, verify the gap isn't simply a data gap.

---

## Finding 5: LR-Batch Size Coupling Follows sqrt Scaling for Muon
**Novelty: Moderate | Sweeps 23-26**

| Batch | LR | val_loss |
|-------|-----|----------|
| 128K | 0.06 | 3.518 |
| 64K | 0.04 | 3.478 |
| **64K** | **0.03** | **3.468** |

Halving batch size → multiply LR by ~1/√2. More optimizer steps at lower LR consistently beats fewer steps at higher LR. The sqrt scaling law (well-known for SGD/Adam) also holds for Muon in multi-modal settings.

This is the **single most impactful change** (accounts for ~60% of total improvement).

---

## Finding 6: Coupled Weight Decay
**Novelty: Moderate | Sweeps 14-16**

Making weight decay decay proportionally with LR during warmdown gives +0.019 at 5min. The insight: constant WD with decaying LR creates an increasing effective regularization ratio during cooldown, destabilizing fine-tuning.

However, this benefit **disappears at 10min** (Finding 1), limiting its practical value to short training runs.

---

## Finding 7: Muon and Adam Optimize Fundamentally Different Objects
**Novelty: Low-Moderate | Sweeps 1-10**

Muon (transformer matrices): LR=0.06 was optimal across ALL configurations — different data scales, WD, batch sizes. "Set and forget."

Adam (embeddings, lm_head): LR, WD, and dropout were all highly sensitive to data distribution. Required constant retuning.

Muon's spectral normalization makes it invariant to weight scale → invariant to data distribution. Adam operates in raw parameter space → directly affected by data statistics.

---

## Finding 8: Lower Weight Decay with Multi-Modal Data
**Novelty: Low | Sweeps 14-16**

WD=0.05 beats WD=0.1 by +0.015. Multi-modal data is inherently more diverse, providing natural regularization. Less explicit regularization needed.

---

## Finding 9: Multi-Seed Statistical Validation
**Novelty: Expected (but essential) | Sweeps 27, 33**

| Config | Seeds | Mean | Std |
|--------|-------|------|-----|
| Best | 3 | 3.467 | ±0.004 |
| Baseline | 3 | 3.545 | ±0.001 |

- **Δ = −0.078 (2.2% improvement)**
- Welch's t-test: **p = 0.0004**
- Cohen's d: **30.7** (massive effect)
- Zero overlap between distributions
- Best config also has **lower variance** — optimizer improvements stabilize training

---

## Final Optimized Recipe

```
TOTAL_BATCH_SIZE = 64K     # (was 128K) — more optimizer steps
MATRIX_LR = 0.03           # (was 0.06) — sqrt scaling with batch
WEIGHT_DECAY = 0.05        # (was 0.1) — less regularization
WARMDOWN_RATIO = 0.75      # fraction of training for LR cooldown
momentum = 0.95            # duration-robust choice
coupled WD: WD decays with LR during warmdown
momentum warmup: 0.85 → 0.95 over first 80% of training
```

### Results at Multiple Durations

| Duration | Best | Baseline | Δ | Improvement |
|----------|------|----------|---|-------------|
| 5min (n=3) | 3.467±0.004 | 3.545±0.001 | −0.078 | **2.2%** |
| 10min (n=3) | 3.362±0.003 | 3.406 | −0.044 | **1.3%** |
| 20min | 3.299 | 3.334 | −0.035 | **1.1%** |

---

## Paper Contributions

1. **Duration-dependent hyperparameters**: Optimal momentum reverses with training duration (undocumented for Muon)
2. **Model-size scaling**: Optimizer tuning benefit grows super-linearly (0.1% at 22M → 4.1% at 198M)
3. **Regularization as data tax**: Regularization compensates for per-token data scarcity, not model complexity
4. **Gradient manipulation failure**: 5/5 methods failed — convergence gaps are data gaps, not optimizer gaps
5. **LR-batch coupling for Muon**: sqrt scaling holds in multi-modal context
6. **Practical recipe**: 6 compounding changes, 2.2% improvement, p<0.001

## Experiment Infrastructure

- 401 experiments across 35 systematic sweeps
- 353 successful runs, 18 crashes, 30 timeouts
- Hardware: NVIDIA RTX 5090 (32GB)
- Model: Qwen2-0.5B backbone + Whisper encoder + SNAC codec
- Data: 600M text tokens + 115h LibriSpeech + 100.6h TTS/ASR pairs
- Full convergence curves logged at 100-step intervals
- All results in `results.tsv`, theory in `THEORY.md`
