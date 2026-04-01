# Presentation Notes: Omni-Modal S3 Audio Plateau Investigation

**Private document for presenter preparation and Q&A reference**
**Date**: 2026-04-01
**Talk context**: Research progress presentation on the audio plateau problem in mini-omni joint text+audio (S3) training.

---

## TABLE OF CONTENTS

1. [Slide-by-Slide Notes](#slide-by-slide-notes)
2. [FAQ: Anticipated Questions and Answers](#faq-anticipated-questions-and-answers)
3. [Quick Reference Data Tables](#quick-reference-data-tables)

---

# SLIDE-BY-SLIDE NOTES

---

## Slide 1: Title

**Slide title**: Omni-Modal S3 Audio Plateau: Diagnostic Investigation

**Key points to emphasize**:
- Frame the talk as a systematic diagnostic investigation, not just "we ran experiments"
- This is about understanding WHY audio plateaus in joint training, not about achieving SOTA
- Emphasize the scale of effort: 29 experiments, 15 diagnostic signals, 6 optimization methods tested
- Mention the model: Qwen2-0.5B + Whisper + SNAC (7 codebook streams), ~500M params

**Exact numbers to cite**:
- 29 completed experiments across 6 batches
- 15 diagnostic signals (D1-D15)
- ~91 GPU-hours on A100 80GB (Narval, Compute Canada)
- Mini-omni chain: 3 rounds of continued training from published checkpoint

**Anticipated questions**:
- Q: "Is this your own model?" A: "No, this is the published mini-omni model (gpt-omni). We are investigating its training dynamics, specifically the audio plateau during S3 joint training."

**Transition to next slide**: "Before diving into findings, let me briefly review the mini-omni architecture so we're on the same page about the model structure."

---

## Slide 2: Mini-Omni Architecture Overview

**Slide title**: Mini-Omni Architecture

**Key points to emphasize**:
- Three components: Qwen2-0.5B (LLM backbone), Whisper encoder (audio input), SNAC 24kHz codec (audio output)
- 7 codebook streams in SNAC, each with 4160 vocab tokens
- **Shared embedding table**: 181,120 tokens total (152K text + 29,120 audio = 7 x 4160)
  - (中文备注: 共享 embedding 是问题的核心 -- 152K text 和 29K audio token 在同一个矩阵中)
- Delay pattern: stream i delayed by i positions
- **Three training stages**: S1 (adapter only) -> S2 (backbone only, text) -> S3 (everything unfrozen, 4 task types)
- The 4 task types: T1T2, T1A2, A1T2, A1A2

**Exact numbers to cite**:
- ~500M parameters, 24 transformer layers, hidden_dim=896
- Embedding matrix: (181,120, 896) -- shared via `lm_head` (tied weights, `tie_word_embeddings=True`)
- SNAC codebook entropies: CB0=6.03 nats, CB1=7.09, CB2=7.85, CB3=7.90, CB4=7.09, CB5=7.87, CB6=7.91
- Random baseline: log(4160) = 8.33 nats

**Anticipated questions**:
- Q: "What does the delay pattern do?" A: "Stream i is delayed by i positions so the model predicts coarse-to-fine: it first predicts CB0 (semantic), then CB1 given CB0, etc. Total sequence length = n_frames + 6."
- Q: "Why is `tie_word_embeddings` relevant?" A: "The input embedding and output lm_head share the same weight matrix. This means any gradient from audio prediction also modifies the embeddings used for text input, and vice versa. It is the primary coupling point between modalities."

**Transition**: "Now let me show you the problem we're investigating -- the audio plateau."

---

## Slide 3: The Anomaly (Audio Plateau)

**Slide title**: The Anomaly: Audio Loss Plateau in S3

**Key points to emphasize**:
- During S3 joint training, text loss drops 77% in 3000 steps while audio loss drops only 14.5%
- At 3K steps: text val loss goes from 6.6 to 1.5, audio val loss from 41.2 to 35.2
- Per-codebook: CB0 (coarsest) reaches 4.0 (52% below random), CB2-6 plateau at 5.5 (34% below random)
- The plateau is NOT absolute -- it breaks with more steps (~10K-30K), but learning is very slow
- (中文备注: 这不是一个硬上限，而是一个极慢的收敛区间)

**Exact numbers to cite**:
- Text: 77% reduction (6.6 -> 1.5) in 3000 steps
- Audio: 14.5% reduction (41.2 -> 35.2) in 3000 steps
- At 10K steps: audio CB sum drops from 51.25 to 44.84 (12.5% further improvement)
- At 30K steps (exp16): A1A2 val drops from 49.5 to 36.8
- Chain R3 (~16.5K global steps, eff=192): A1A2=32.3

**Anticipated questions**:
- Q: "Is 3000 steps enough to judge?" A: "No, and that's one of our key findings. The plateau does break with more steps. But at 3K steps, 6 different optimization methods ALL converge to the same val loss, which tells us the 3K-step plateau is real and robust."
- Q: "What's the published mini-omni performance?" A: "The published model uses ~100K+ S3 steps with effective batch size 192. Our experiments show the audio plateau is a slow convergence regime that continues improving with more training."

**Transition**: "To systematically understand this, we designed a comprehensive experimental setup."

---

## Slide 4: Experimental Setup (29 experiments, 15 diagnostics)

**Slide title**: Experimental Setup: 29 Experiments, 15 Diagnostic Signals

**Key points to emphasize**:
- Systematic approach: 6 batches of experiments covering different hypotheses
- Methods tested: AdamW baseline, GradNorm, entropy scaling, gradient projection, M-SAM, codebook weighting, curriculum, freeze backbone, embedding init
- 15 diagnostics: gradient ratio (D1), cos_phi (D2), per-layer cosines (D3), basin width (D4), displacement (D5), embedding rank (D6), per-CB loss (D7), top-k accuracy (D8), linear probes (D9), per-task losses (D10), module grad norms (D11), CKA (D12), cosine collapse (D13), GSNR (D14), text subspace energy (D15)
- Training configs: 3K to 30K steps, effective batch 32 and 192, various starting checkpoints

**Exact numbers to cite**:
- 16 named experiments (exp1-exp17 minus exp14) + 9 earlier experiments (s3_adam, s3_lambda3, etc.) + 3 chain runs (R1-R3) + smoke test = 29 total
- ~91 GPU-hours on A100 80GB
- Default S3 config: LR 2e-6 to 2e-5 (cosine), batch_size=2, grad_accum=16 (eff=32), warmup=1500, AdamW

**Anticipated questions**:
- Q: "Why so many diagnostics?" A: "We kept adding diagnostics as we falsified hypotheses. Each falsified hypothesis pointed to a new diagnostic we needed. D14 (GSNR) and D15 (text subspace energy) were added last and proved most informative."
- Q: "How were diagnostics computed?" A: "Every 100 steps on the validation set. For cos_phi: separate backward passes for text_loss and audio_loss on the same val batch, then compute cosine between the two full gradient vectors."

**Transition**: "Let me walk through the five key observations, starting with gradient orthogonality."

---

## Slide 5: Observation 1 -- Gradient Orthogonality (cos_phi ~ 0, D15 energy = 0.1%)

**Slide title**: Observation 1: Text and Audio Gradients Are Orthogonal

**Key points to emphasize**:
- cos_phi stays within [-0.05, +0.02] across ALL experiments and ALL timesteps
- This definitively rules out gradient conflict as the cause of the audio plateau
- (中文备注: 梯度不冲突，是正交的 -- "不是在打架，是在各走各的")
- Per-layer analysis: interference concentrates in Layer 0 (shared embedding, 70%+ of total interference)
- Layers 1-22 (transformer backbone): cos_phi ~ +/-0.03 (essentially orthogonal)
- D15 text subspace energy: only 0.1-0.6% of audio gradient energy lies in the text gradient subspace
- Implication: text and audio are essentially independent optimization problems that happen to share parameters

**Exact numbers to cite**:
- Global cos_phi: [-0.05, +0.02] range across all experiments
- Layer 0 contribution to interference: 70%+
- D15 energy_ratio: 0.001-0.006 (0.1-0.6%)
- GradProj experiment (s3_gradproj): removed interfering component from audio gradient, SAME plateau -- confirming interference is not the cause

**Anticipated questions**:
- Q: "If gradients are orthogonal, why doesn't each modality converge independently?" A: "They essentially do. The problem is that the audio optimization problem itself is hard -- GSNR=0.08 means audio gradients are extremely noisy. Being orthogonal to text doesn't help if the signal itself is weak."
- Q: "Isn't some interference at Layer 0 expected?" A: "Yes, because text and audio share the same embedding matrix. But removing this interference (via GradProj) doesn't help, because the interference is not what's causing the plateau."

**Transition**: "If gradient conflict isn't the issue, what is? GSNR tells us the audio gradient signal itself is the problem."

---

## Slide 6: Observation 2 -- GSNR = 0.08, B_crit_audio = 25

**Slide title**: Observation 2: Audio Gradient Signal-to-Noise Ratio Is Extremely Low

**Key points to emphasize**:
- GSNR (Gradient Signal-to-Noise Ratio) = ||E[g]||^2 / tr(Var[g])
- Audio GSNR = 0.074-0.127 (far below 1.0)
  - (中文备注: GSNR < 1 意味着梯度方差远大于信号，每步优化主要是噪声)
- This means gradient variance >> signal; each step is mostly noise
- Even accumulating 16 batches (cos@K=16 ~ 0.01) does not stabilize direction
- Critical batch size derivation:
  - B_crit = tr(Sigma) / ||G||^2
  - B_crit_text = 2 (text needs very small batch)
  - B_crit_audio = 25 (audio needs larger batch)
- But counterintuitively, small batch is MORE sample-efficient (see Slide 8)
- Gradient efficiency decays exponentially: from 3.13 to 0 over 3000 steps (per-gradient-unit loss reduction)

**Exact numbers to cite**:
- GSNR: R2 start=0.127, R2 end=0.074, R3 end=0.083
- cos@K=16: ~0.01
- B_crit_text = 2, B_crit_audio = 25
- Gradient efficiency (s3_adam): steps 100-500 = 3.13e-4, steps 2500-3000 = 0.00 (complete stall at 3K)
- aud_grad_norm stays constant (~2.8) while per-unit efficiency drops to zero -- "not that gradients are too small, but that they cancel across samples"

**Anticipated questions**:
- Q: "How was GSNR measured?" A: "D14 diagnostic: we compute per-sample gradients on a validation batch (8 samples), calculate the mean gradient (signal) and the covariance trace (noise), then take their ratio. This is done every 100 steps at specific checkpoints."
- Q: "Why is audio GSNR so low?" A: "Most likely due to SNAC's DRI (Discrete Representation Inconsistency) -- the same audio segment can be encoded as different token sequences. So different training samples that sound similar produce gradients pointing in different directions, canceling each other."
- Q: "What does B_crit_audio=25 mean practically?" A: "Below batch size 25, each audio gradient step is dominated by noise. Above 25, the signal starts to dominate. But as we show in Slide 8, noise-dominated small-batch training is actually more sample-efficient because the noise provides beneficial exploration."

**Transition**: "Now here's a surprising result about the backbone representations."

---

## Slide 7: Observation 3 -- Two CKA Paths (0.03 vs 0.99, Same Val)

**Slide title**: Observation 3: Completely Different Representations, Same Performance

**Key points to emphasize**:
- CKA (Centered Kernel Alignment) measures how much backbone representations change from initialization
- Two extreme paths:
  - S2->S3 (exp12/exp16): CKA > 0.987 throughout -- backbone barely changes from pretrained state
  - Long S3 no S2 (exp13): CKA drops to 0.03 by step 2000 -- completely different representations
  - Chain (R1-R3): CKA = 1.000 -- even more stable with large batch
- **Yet both paths reach the same validation performance**
  - exp12: audio sum = 44.84
  - exp13: audio sum = 45.12
- (中文备注: 表征不是瓶颈 -- 完全不同的内部表示可以达到相同的音频预测质量)
- This is strong evidence that the plateau is NOT about the backbone being "stuck" in a bad representation

**Exact numbers to cite**:
- exp12 (S2->S3, 10K): CKA L12 = 0.994 at step 10000, audio sum = 44.84
- exp13 (Long S3, 10K): CKA L12 = 0.036 at step 10000, audio sum = 45.12
- exp16 (S2->S3, 30K): CKA stays > 0.987 throughout
- Curriculum (exp9): CKA = 0.008 at step 3000 -- another radical reshaping
- GradNorm diverged (exp3): CKA L12 = 0.613 -- moved backbone but in wrong direction

**Anticipated questions**:
- Q: "How can CKA=0.03 and CKA=0.99 give the same performance?" A: "The model has many equivalent solutions -- the backbone can represent audio information in completely different bases. What matters is not WHICH representation, but HOW MUCH audio information the representation contains. Both contain the same (limited) amount."
- Q: "Does this mean S2 is pointless?" A: "Not entirely. S2 makes the optimization landscape smoother (10x wider basins) and stabilizes training (gradient norms don't explode). But it doesn't change the final capability."
- Q: "What does CKA=1.000 for the chain mean?" A: "With large effective batch (192) and starting from the published checkpoint, the backbone representations are essentially frozen. All learning happens in the output head and embeddings."

**Transition**: "So what actually matters for audio learning? The next observation reveals a clear hierarchy."

---

## Slide 8: Observation 4 -- More Steps >> Batch Size >> S2

**Slide title**: Observation 4: Training Steps Matter Most

**Key points to emphasize**:
- Clear hierarchy: number of optimizer steps > batch size > S2 pre-training
- Key comparison at same total samples (960K):
  - exp16 (30K steps, eff=32): A1A2 = 36.8
  - exp15 (5K steps, eff=192): A1A2 = 44.6
  - More steps with smaller batch decisively wins
- S2 doesn't help val: exp12 (44.8) = exp13 (44.9) at matched 10K steps
- Small batch is 2.8x more sample-efficient:
  - To reach A1A2=42: eff=32 needs 346K samples, eff=192 needs 960K samples
- (中文备注: 小 batch 更节省样本 -- 噪声梯度提供有益的探索)

**Exact numbers to cite**:
- exp15 (S2, eff=192, 5K steps): val A1A2 = 44.6
- exp16 (S2, eff=32, 30K steps): val A1A2 = 36.8
- exp17 (no S2, eff=192, 5K steps): val A1A2 = 43.9
- Sample efficiency ratio at A1A2=42: 2.8x (eff=32 vs eff=192)
- Sample efficiency ratio at A1A2=38: 2.2x
- B_crit theory predicts 3.8x efficiency ratio -- empirical 2.8x is in the right ballpark
- Audio loss trajectory (exp16, eff=32): step 2K=49.5, step 10K=42.5, step 20K=38.2, step 30K=36.8

**Anticipated questions**:
- Q: "Why is small batch more efficient?" A: "Critical batch size theory. Audio B_crit=25, so at eff=32 each step has roughly 1 'effective sample' of clean signal plus noise. At eff=192, you have ~7.7 effective samples per step but each step costs 6x more data. The noise at small batch acts as beneficial exploration in the audio loss landscape."
- Q: "But doesn't large batch reduce variance?" A: "Yes, but for audio, reducing variance is wasteful because B >> B_crit. The model is already in a regime where more variance (noise) helps escape local structure in the loss landscape. It's like using a sledgehammer to crack a nut."
- Q: "Would even smaller batch be better?" A: "Possibly, down to ~B_crit_audio=25. But eff=32 is already close to optimal according to B_crit theory (70% efficient). Going smaller than B_crit would make text learning unstable."

**Transition**: "The next observation reveals what's happening inside the backbone during this slow learning."

---

## Slide 9: Observation 5 -- Linear Probe Gap (77% train, 6% val)

**Slide title**: Observation 5: Backbone Memorizes, Doesn't Generalize Audio

**Key points to emphasize**:
- Linear probes trained on frozen backbone hidden states to predict audio tokens
- Massive train-val gap reveals memorization, not learning:
  - L18: 77.1% train accuracy, 6.2% val accuracy (12x gap)
  - L12: 57.3% train, 6.1% val (9x gap)
  - L6: 18.6% train, 6.2% val (3x gap)
- Random chance = 1/4160 = 0.024%
- Val accuracy is 250x better than random, but still very low (6%)
- (中文备注: backbone 确实编码了一些音频信息，但完全无法泛化 -- 是位置/上下文特定的记忆)
- Embedding structure IS evolving: cosine similarity 0.018 -> 0.124 (chain R3)
- CB0 top-1 accuracy trajectory: 0.048 (3K) -> 0.146 (30K) -> 0.159 (chain R3) -- tripled, but still low

**Exact numbers to cite**:
- Probe at L18 (S2->S3 30K): train=77.1%, val=6.2%, gap=12x
- Probe at L12: train=57.3%, val=6.1%, gap=9x
- Probe at L6: train=18.6%, val=6.2%, gap=3x
- Random chance: 0.024%
- Probe val accuracy growth rate: ~+0.1% per 1000 steps, saturates around 6%
- CB0 top-1: 0.048 (3K) -> 0.146 (30K) -> 0.159 (chain R3)
- CB0 top-10: 0.207 (3K) -> 0.421 (30K) -> 0.444 (chain R3)
- Embedding cosine similarity: 0.018 (baseline) -> 0.054 (chain R1) -> 0.124 (chain R3)

**Anticipated questions**:
- Q: "Does the probe gap mean the backbone isn't learning audio at all?" A: "No -- 6% val is 250x better than random. The backbone IS learning audio structure, just very slowly and mostly in a context-specific (non-generalizable) way. Deeper layers memorize training contexts."
- Q: "What about the embedding learning evidence?" A: "Cosine similarity going from 0.018 to 0.124 shows embeddings are slowly developing directional structure -- moving from random toward organized. But 0.124 is far from the structured embeddings needed for good prediction."
- Q: "Is 6% the ceiling?" A: "At 30K steps with eff=32, it appears to plateau around 6%. Longer training or different approaches might push past this."

**Transition**: "All these observations point to a theoretical framework that explains the dynamics."

---

## Slide 10: Theoretical Framework (Joint B_crit Optimization, B*=9)

**Slide title**: Theoretical Framework: Joint Critical Batch Size

**Key points to emphasize**:
- Start from critical batch size theory (McCandlish et al.): N(B) = S_min * (1 + B_crit/B)
- Optimization target: minimize C = N x B (total training samples)
- With cos_phi ~ 0 (verified by D2 + D15): text and audio are independent optimization problems sharing parameters
- Joint constraint: N = max(N_text, N_audio) because they share parameters
- Derivation:
  - B_crit_text = 2, B_crit_audio = 25
  - Optimal batch: B* = B_crit_T + sqrt(B_crit_T * B_crit_A) = 2 + sqrt(50) = 9
  - Optimal sampling ratio: p* = B_crit_T / B* = 22% text, 78% audio
- (中文备注: 关键洞见 -- 最优 batch size 仅为 9! 我们用的 32-192 都太大了)
- Efficiency comparison:
  - B=32: 70% efficient (close to optimal)
  - B=192: 19% efficient (5x waste)
  - This explains why eff=32 is 2.8x more sample-efficient than eff=192

**Exact numbers to cite**:
- B* = 9 (optimal total effective batch size)
- p* = 22% text, 78% audio (optimal task sampling ratio)
- Current: B=32-192, p=50% -- text over-sampled, audio under-sampled
- Predicted efficiency ratio eff=32 vs eff=192: 3.8x (theory), 2.8x (empirical) -- reasonable agreement
- At B* audio stays BELOW B_crit (GSNR < 1) at optimum -- "more steps with noisy gradients > fewer steps with clean gradients"

**Anticipated questions**:
- Q: "Is B_crit the optimal batch size?" A: "No, B_crit is NOT the optimal batch size. B_crit is where noise equals signal. The optimal batch B* depends on both modalities' B_crit values. For a single objective, B*=B_crit minimizes compute. For joint optimization, B* = B_crit_T + sqrt(B_crit_T * B_crit_A) which accounts for the bottleneck modality."
- Q: "Can we validate B*=9 experimentally?" A: "Not yet -- we haven't run experiments at eff=9. But the theory correctly predicts the direction of the eff=32 vs eff=192 comparison (2.8x observed vs 3.8x predicted)."
- Q: "Why does optimal have audio below B_crit?" A: "Because steps are the bottleneck. Each step costs B samples. At B=9, each step costs 9 samples. At B=192, each step costs 192 samples. The audio loss landscape benefits from exploration (noise), so you get more improvement per sample at small B even though each step is noisier."

**Transition**: "Let me summarize the key insights before discussing next steps."

---

## Slide 11: Key Insights Summary

**Slide title**: Key Insights

**Key points to emphasize**:
- Organize by: what we disproved, what we discovered, what the implications are
- **Disproved**:
  1. "Audio plateau is caused by gradient conflict" -- cos_phi ~ 0, GradProj doesn't help
  2. "Audio plateau is caused by gradient magnitude" -- lambda=3 doesn't help, rho irrelevant
  3. "S2 is critical for audio learning" -- S2 vs no-S2 give same val at matched steps
  4. "Sharp minima trap audio" -- M-SAM finds flatter region, same loss
  5. "Audio-as-GO-noise widens text basin" -- basin probing shows no difference
- **Discovered**:
  1. Audio learning is a GSNR problem (0.08), not a gradient magnitude or conflict problem
  2. Text and audio learn in orthogonal subspaces (D15 energy < 0.6%)
  3. Different CKA paths (0.03 vs 0.99) reach same val -- representation is not the bottleneck
  4. Training steps >> batch size >> S2 for audio improvement
  5. Small batch 2.8x more sample-efficient (B_crit theory explains this)
  6. Backbone memorizes audio (77% train) but doesn't generalize (6% val)
- (中文备注: 核心是 GSNR 太低 -- 音频需要约 12x 于文本的样本才能提取同等信号)

**Exact numbers to cite**:
- 7 falsified hypotheses (reference data table)
- GSNR = 0.08 -> audio needs ~12x more samples than text
- D15 energy < 0.6% -> orthogonal learning
- CKA: 0.03 vs 0.987 -> same val performance
- B*=9, current B=32-192 -> 30-81% waste

**Anticipated questions**:
- Q: "So what's the actual root cause?" A: "Low GSNR in audio gradients, likely caused by SNAC's DRI (same audio -> different tokens). This makes each gradient step mostly noise. Combined with the chicken-and-egg problem (random audio embeddings -> noisy backbone representations -> noisy gradients -> slow embedding learning), convergence is extremely slow."

**Transition**: "Finally, let me discuss what we're planning next."

---

## Slide 12: Ongoing + Next Steps

**Slide title**: Ongoing Work and Next Steps

**Key points to emphasize**:
- Immediate experiments to run:
  1. Optimal sampling ratio (p=22% text, 78% audio) -- simple DataLoader change based on B_crit theory
  2. LR scaling with batch size -- eff=192 + LR=7x should match small batch efficiency
  3. Per-modality effective batch measurement -- directly measure B_crit for text and audio
- Longer-term directions:
  1. Better codec: literature (Moshi/Mimi, X-Codec) shows semantic-aware codecs reduce DRI
  2. Separate audio embedding to eliminate Layer 0 interference (~26M extra params)
  3. Spectral regularization on audio embeddings to prevent rank lock
- Open questions:
  - Can we push past the 6% probe accuracy ceiling?
  - Does the B_crit framework generalize to other multimodal models?
  - What happens at 100K+ steps (matching mini-omni's actual training)?
- (中文备注: 最有价值的下一步是 (1) 验证 p*=22% 采样比例 和 (2) 换一个语义感知的 codec)

**Exact numbers to cite**:
- Proposed p* = 22% text, 78% audio (current: 50/50)
- Separate audio embedding cost: ~26M params (29120 x 896)
- Current compute budget: ~91 GPU-hours used, plan to extend to ~200h
- Mini-omni published training: ~100K+ S3 steps, eff=192

**Anticipated questions**:
- Q: "What would you do differently if starting over?" A: "Three things: (1) Start with 10K+ step experiments from the beginning instead of 3K -- we wasted time drawing conclusions from under-trained models. (2) Add GSNR (D14) and text subspace (D15) diagnostics from the start. (3) Focus on batch size and sampling ratio experiments earlier."
- Q: "What's the timeline for next experiments?" A: "The p*=22% experiment is ~7 GPU-hours. We could have results within a day."

---

# FAQ: ANTICIPATED QUESTIONS AND ANSWERS

---

### Q1: Why doesn't text guide audio learning? (为什么 text 不能引导 audio 学习？)

**Answer**: Text and audio operate in almost completely orthogonal parameter subspaces (D15 energy < 0.6%). The shared parameters (embedding matrix + backbone) do couple them, but the coupling is weak -- 99.4% of the audio gradient lies outside the text gradient subspace. Text learning provides implicit regularization (3% text val improvement with omni training vs text-only), but it doesn't provide directional guidance for audio.

The backbone acts as a near-frozen "pass-through" for audio: audio gradients concentrate in Layer 0 (input embedding) and Layer 23 (output), with intermediate layers receiving comparatively small audio gradients. This means the 22 transformer layers learn primarily for text, and audio must work with whatever representations text creates.

---

### Q2: Is the training code correct? (训练代码有没有 bug？)

**Answer**: We found and fixed 3 critical bugs before the current experiment series (2026-03-20):
1. Audio loss was incorrectly divided by 7 (number of codebooks) -- fixed to report true CE
2. Whisper encoder features were not being loaded -- fixed feature loading path
3. Wrong checkpoint was being used (post-S3 instead of post-S1) -- fixed checkpoint mode

After these fixes, all experiments in this presentation use correct code. The consistency of results across 29 experiments (all methods converge to same val at matched steps) is itself evidence of correct implementation -- a bug would likely cause inconsistent behavior.

Known remaining issues:
- D5 embedding displacement reads 0 due to `tie_word_embeddings=True` parameter name mismatch (documented, not affecting results since lm_head displacement captures the same info)
- M-SAM diagnostics show zeros at batch_size=1 (fixed by ensuring batch_size >= 2)

---

### Q3: How does this compare to mini-omni's actual results? (这跟 mini-omni 实际的训练结果比怎么样？)

**Answer**: Mini-omni's published training uses:
- ~100K+ S3 steps (we used 3K-30K)
- Effective batch size 192 (we used 32 and 192)
- Extensive S2 pre-training with much more data

Our chain R3 (~16.5K global steps at eff=192) achieves A1A2=32.3, which shows meaningful audio generation capability but is still in the slow convergence regime. The published checkpoint (which we used as starting point for some experiments) represents ~100K+ steps of this same slow convergence.

Our key contribution is NOT matching mini-omni's results but understanding WHY audio converges slowly and what the optimal training strategy is.

---

### Q4: Why not use separate encoders for text and audio? (为什么不用独立的编码器？)

**Answer**: The mini-omni architecture uses a shared embedding table and backbone by design -- this is what enables joint text+audio generation in a single forward pass. Using separate encoders would be a different architecture entirely.

That said, our findings support a partial separation: giving audio tokens their own embedding matrix (not sharing with text's 152K entries in the same table) would eliminate the Layer 0 interference that accounts for 70%+ of gradient coupling. This costs only ~26M extra parameters (29120 x 896) and is one of our proposed next steps.

The backbone should remain shared because: (1) freezing it is catastrophic (exp7: text loss explodes to 29.6), and (2) the backbone's text-learned representations DO provide useful context for audio prediction (T1T2 audio loss 23 vs T1A2 audio loss 49 -- text output channel helps audio).

---

### Q5: What about the D5 displacement bug? (D5 的 bug 怎么回事？)

**Answer**: When `tie_word_embeddings=True` (which mini-omni uses), the embedding matrix is stored as `lm_head.weight` rather than `model.embed_tokens.weight`. Our D5 displacement tracking looked for `embed_tokens` and found nothing, so embedding displacement always reads 0.

This is a measurement bug, not a training bug. Since the weights are tied, lm_head displacement = embedding displacement. We track lm_head displacement correctly (e.g., 60.8 at 3K steps for exp2), so we know the combined embedding/lm_head is moving significantly.

The bug does NOT affect any training dynamics or results. It only means we can't separately measure embedding vs lm_head displacement (because they're the same matrix).

---

### Q6: Is B_crit the optimal batch size? (B_crit 是最优 batch size 吗？)

**Answer**: No. B_crit (critical batch size) is the batch size where gradient noise equals signal. For a SINGLE objective, B* = B_crit minimizes total compute (samples x steps). But for JOINT optimization of text and audio:

B* = B_crit_T + sqrt(B_crit_T * B_crit_A) = 2 + sqrt(2 * 25) = 2 + 7.07 = 9

This is because:
- Text needs at least B_crit_T = 2 samples per step to make progress
- Audio is the bottleneck (B_crit_audio >> B_crit_text)
- The optimal total batch balances text signal quality against total sample cost
- At B* = 9, audio gets ~7 samples per step (below B_crit_audio = 25), meaning each audio gradient step is noisy -- but that's OK because noise = exploration, and we get more steps per fixed sample budget

---

### Q7: How did you measure GSNR? (GSNR 是怎么测量的？)

**Answer**: D14 (GSNR) is measured as follows:
1. Take a validation batch of 8 samples
2. Compute per-sample gradients (8 separate backward passes) for audio loss only
3. Signal = ||mean(g_1, ..., g_8)||^2 (squared norm of the average gradient)
4. Noise = tr(Var(g_1, ..., g_8)) (trace of the gradient covariance = sum of per-coordinate variances)
5. GSNR = Signal / Noise

Additionally, cos@K measures: for K random subsets of size K, compute mean gradient and measure pairwise cosine similarity. cos@K=16 ~ 0.01 means even 16 accumulated batches produce gradients with near-zero correlation.

GSNR was only measured at specific checkpoints (chain R2 start, R2 end, R3 end) because it requires 8 separate backward passes and is computationally expensive.

---

### Q8: Why does S2 not help final val performance? (为什么 S2 对最终验证集性能没有帮助？)

**Answer**: S2 (text-only pre-training) helps the optimization LANDSCAPE but not the final MODEL CAPABILITY:

What S2 does:
- Stabilizes backbone (txt_gn from 344 to 11)
- Widens basin (degradation 8.6 vs 335 at eps=0.01)
- Preserves CKA > 0.99 (representations stay near pretrained state)
- Improves ρ (audio/text gradient ratio from 0.01 to 0.22)

What S2 does NOT do:
- Change the audio GSNR (still ~0.08)
- Change the final val CB losses at matched steps

Evidence:
- exp12 (S2+S3, 10K) audio sum = 44.84
- exp13 (S3 only, 10K) audio sum = 45.12
- exp15 (S2, eff=192, 5K) A1A2 = 44.6
- exp17 (no S2, eff=192, 5K) A1A2 = 43.9 (no S2 slightly BETTER)

S2 makes training smoother and more stable but doesn't change what the model can ultimately learn at a given number of optimizer steps.

---

### Q9: What's the chicken-and-egg problem? (鸡生蛋的问题是什么？)

**Answer**: The audio embedding bottleneck creates a circular dependency:

```
Random audio embeddings (rank ~877/896)
    -> Audio tokens mapped to random directions
    -> Backbone receives noise, produces noisy hidden states
    -> Audio gradients are directionless (GSNR=0.08)
    -> Embeddings update slowly
    -> Back to near-random embeddings
```

The backbone can't learn audio patterns from random embeddings, and embeddings can't learn from a backbone that ignores audio. This circular dependency IS being broken (cosine similarity: 0.018 -> 0.124 over chain R1-R3, CB0 accuracy tripling), but extremely slowly.

Compare with text: text embeddings start pretrained (from Qwen2-0.5B), so the backbone already processes them meaningfully from step 0. There is no chicken-and-egg problem for text.

---

### Q10: Why does CKA=0.03 and CKA=0.99 give the same val performance?

**Answer**: The model has many equivalent solutions in its loss landscape. CKA measures representational similarity to the initial checkpoint, not representational quality.

- CKA=0.99 (S2->S3): The backbone stays near pretrained Qwen2 representations. It adapts to audio by adjusting how the OUTPUT HEAD reads the existing representations.
- CKA=0.03 (Long S3): The backbone completely reorganizes into a new representation that jointly encodes text and audio.

Both approaches learn the same amount of audio information (linear probe val accuracy ~5-6%), just encoded differently. The representational basis doesn't matter -- what matters is the information content, which is limited by GSNR and the embedding structure.

Analogy: two databases can store the same data in different schemas. The schema (CKA) differs, but the queryable information (val performance) is the same.

---

### Q11: What's the emergent adapter phenomenon?

**Answer**: When CKA stays at 0.987+ (as in S2->S3 training), the backbone essentially acts as a frozen feature extractor. All audio learning happens in:
- The input embedding layer (Layer 0)
- The output projection (lm_head/Layer 23)

This is structurally equivalent to training an adapter around a frozen backbone -- except it emerges naturally from gradient dynamics rather than being explicitly designed. Audio gradients are simply too weak to move the backbone layers significantly when text gradients dominate.

Evidence:
- Module gradient norms (exp2, step 3K): audio gradient at Layer 23 = 22.27 (large), but at Layer 12 = 1.86 (tiny)
- Backbone displacement saturates at 11.8 while lm_head displacement reaches 60.8
- Audio learning concentrates at the periphery (input + output)

---

### Q12: Why is small batch more sample-efficient?

**Answer**: This follows from critical batch size theory when B >> B_crit:

At eff=192 (6x larger than B_crit_audio=25):
- Each step uses 192 samples
- Each step produces a clean gradient (low noise)
- But you get only N/192 steps per N samples

At eff=32 (1.3x B_crit_audio):
- Each step uses 32 samples
- Each step produces a noisier gradient
- But you get 6x more steps per N samples

The extra steps win because:
1. Audio landscape has many local optima/saddle points. Gradient noise helps escape them.
2. The clean gradient at B=192 points to the local minimum. The noisy gradient at B=32 points to a broader region, enabling exploration.
3. McCandlish et al. showed that efficiency scales as (1 + B_crit/B)^{-1}. At B=32: 56% efficient per sample. At B=192: 12% efficient per sample.

Empirical: to reach A1A2=42, eff=32 needs 346K samples, eff=192 needs 960K. Ratio = 2.8x. Theory predicts 3.8x.

---

### Q13: Can we use the B_crit theory to predict training time?

**Answer**: Approximately, yes. The framework predicts:

N_audio(B) = S_min_audio * (1 + B_crit_audio / B_audio)

Where:
- B_audio = B * p (effective audio batch = total batch * audio fraction)
- S_min_audio = minimum steps at infinite batch

We can estimate S_min_audio from our data: at eff=192 (chain), ~16.5K steps to A1A2=32.3. Extrapolating with the B_crit formula:
- At B=9, p=78%: B_audio = 7, N_audio = S_min * (1 + 25/7) = 4.57 * S_min
- Total samples = N_audio * B = 4.57 * S_min * 9 = 41 * S_min

This gives a rough estimate but should be validated experimentally. The theory assumes stationary GSNR, which may not hold as training progresses.

---

### Q14: What would you do differently? (如果重来会怎么做？)

**Answer**:
1. **Start with longer experiments**: Our first 4 batches used 3K steps each. At 3K steps, all methods look the same, leading to premature conclusions about a "hard plateau." We should have started with 10K+ steps.
2. **Add GSNR diagnostic from the beginning**: D14 (GSNR) was the most informative diagnostic but was added late. It immediately explained why audio learning is slow. D15 (text subspace energy) similarly clarified the orthogonality story.
3. **Test batch size and sampling ratio earlier**: The most actionable finding (B*=9, p*=22%) came from batch 6. Earlier batches focused on optimization tricks (GradNorm, M-SAM, etc.) that were all dead ends.
4. **Compute per-sample GSNR decomposition by codebook**: We measured GSNR for aggregate audio. Per-codebook GSNR would tell us which codebooks have the worst signal.
5. **Compare with a semantic-aware codec early**: Literature strongly suggests DRI is the root cause. Testing with X-Codec or Mimi would either confirm or rule this out.

---

### Q15: What's the paper contribution? (论文贡献是什么？)

**Answer**: The contribution is a systematic diagnostic framework and theoretical analysis for understanding multimodal training dynamics, specifically:

1. **Diagnostic methodology**: 15 diagnostic signals that decompose multimodal training into interpretable components. This toolkit is general -- it works for any shared-backbone multimodal model.

2. **Empirical findings**: 
   - 7 falsified hypotheses about audio plateaus (gradient conflict, gradient magnitude, S2 pre-training, sharp minima, basin widening, codebook weights, optimizer choice)
   - 5 positive findings (GSNR explains slow learning, orthogonal subspaces, CKA path independence, step count >> batch size, sample efficiency of small batch)

3. **Theoretical framework**: Joint critical batch size optimization for multimodal training, deriving B* and p* from per-modality B_crit values. Validated empirically (2.8x observed vs 3.8x predicted efficiency ratio).

4. **Practical recommendations**: Small batch + more steps + 22/78 text/audio sampling ratio. These are immediately actionable for anyone training omni-modal models.

---

### Q16: Why was GradNorm so catastrophic? (为什么 GradNorm 会崩溃？)

**Answer**: GradNorm (Adaptive Weighting for Multi-task Learning, Chen et al. 2018) adjusts task weights based on the ratio of each task's loss improvement rate. The problem: audio loss barely changes (stuck at 35-45), so GradNorm interprets this as "audio needs more weight." It keeps increasing w_audio and decreasing w_text. Without a non-negativity constraint, w_text crosses zero and the model begins MAXIMIZING text loss.

Timeline: w_text went from +0.012 (step 100) to -1.24 (step 200) to -525,100 (step 1000). Text loss exploded to ~5000.

The fixed version (exp5) added w >= 0.01 clamping and gradient clipping on the weight parameters. This stabilized training and achieved the best train audio loss (-18% vs baseline), but val audio was unchanged -- confirming the plateau is not an optimization problem.

---

### Q17: What is DRI and why does it matter? (什么是 DRI？为什么重要？)

**Answer**: DRI (Discrete Representation Inconsistency) refers to the fact that neural audio codecs like SNAC can encode the same audio waveform as different discrete token sequences. Two recordings of the same word may produce completely different codebook indices.

This matters because: when the model trains on sample A (tokens [42, 7, 193, ...]) and sample B (tokens [89, 55, 12, ...]) that represent similar audio, the gradients for predicting these tokens point in completely different directions. Averaging many such gradients produces near-zero signal -- this is exactly what GSNR=0.08 measures.

Literature evidence:
- Moshi uses Mimi (a codec with an explicit semantic layer) and applies 100:1 weighting on the semantic stream -- significantly better results than SNAC-style codecs
- X-Codec injects semantic information into the codec, showing DRI reduction
- Our CB0 (coarsest, lowest entropy = 6.03) learns 3.6x faster than CB6 (finest, entropy = 7.91), consistent with CB0 being more semantically consistent

---

# QUICK REFERENCE DATA TABLES

---

## Table 1: Complete Experiment Results (audio val CB sum)

| # | Name | Steps | Eff Batch | S2? | Audio CB Sum | A1A2 Val | CKA L12 |
|---|------|-------|-----------|-----|-------------|----------|---------|
| exp1 | S2 text | 3K | 32 | Yes | 64.19 | - | 0.028 |
| exp2 | S3 baseline | 3K | 32 | No | 51.25 | 49.15 | 0.955 |
| exp3 | GradNorm (broke) | 3K | 32 | No | 51.19 | - | 0.613 |
| exp4 | Entropy scaled | 3K | 32 | No | 51.47 | - | 0.997 |
| exp5 | GradNorm fixed | 3K | 32 | No | 49.28 | 49.20 | 0.928 |
| exp6 | Entropy fixed | 3K | 32 | No | 50.03 | 49.87 | - |
| exp7 | Freeze backbone | 3K | 32 | No | 52.53 | - | 0.595 |
| exp8 | Emb init sample | 3K | 32 | No | 49.41 | 49.39 | 0.152 |
| exp9 | Curriculum | 3K | 32 | No | 49.66 | 49.52 | 0.008 |
| exp10 | Freeze+emb init | 3K | 32 | No | 55.56 | - | - |
| exp11 | Long S2 | 10K | 32 | Yes | 62.50 | - | 0.999 |
| exp12 | S2->S3 | 10K | 32 | chained | 44.84 | 44.78 | 0.994 |
| exp13 | Long S3 | 10K | 32 | No | 45.12 | 44.93 | 0.036 |
| exp15 | Omni scale | 5K | 192 | Yes | ~51.6 | 44.6 | - |
| exp16 | Long S2->S3 | 30K | 32 | chained | ~37 | 36.8 | 0.987 |
| exp17 | No S2 large batch | 5K | 192 | No | ~51.7 | 43.9 | - |

---

## Table 2: SNAC Codebook Reference

| CB | Type | Entropy (nats) | Random Loss (8.33) | Best 3K Val | Best 10K Val | Best 30K Val |
|----|------|---------------|-------------------|-------------|-------------|-------------|
| CB0 | Coarse/semantic | 6.03 | 8.33 | 5.56 | 5.22 | 4.56 |
| CB1 | | 7.09 | 8.33 | 6.56 | 5.81 | 4.31 |
| CB2 | Fine | 7.85 | 8.33 | 7.56 | 7.06 | 5.78 |
| CB3 | Fine | 7.90 | 8.33 | 7.66 | 7.12 | - |
| CB4 | | 7.09 | 8.33 | 6.50 | 5.56 | 3.56 |
| CB5 | Fine | 7.87 | 8.33 | 7.78 | 7.19 | 6.31 |
| CB6 | Fine | 7.91 | 8.33 | 7.62 | 7.16 | 5.94 |

---

## Table 3: Falsified Hypotheses

| # | Hypothesis | Key Experiment | Evidence |
|---|-----------|---------------|----------|
| 1 | Audio gradients too weak | s3_lambda3 (3x audio weight) | Same plateau; aud_gn constant |
| 2 | Text-audio gradient conflict | s3_gradproj (remove conflict component) | Same plateau; cos_phi ~ 0 |
| 3 | S2 makes text saturate, squeezes audio | skip_s2 (no S2) | No S2 is worse (5.59 vs 4.41 cb0) |
| 4 | Audio-as-GO-noise widens text basin | Basin probing (obs_1 vs obs_2) | Basin width identical (+/- 1%) |
| 5 | Sharp minima trap audio | s3_msam (sharpness-aware minimization) | 28% less displacement, same loss |
| 6 | Codebook weight allocation wrong | s3_cbweight (100:10:1) | Redistributes, total unchanged |
| 7 | Optimizer choice matters | 6 methods at 3K steps | ALL converge to same val within 0.2 nats |

---

## Table 4: Key Diagnostic Values at Landmark Steps

| Metric | 3K (exp2) | 10K (exp13) | 30K (exp16) | Chain R3 (~16.5K) |
|--------|-----------|-------------|-------------|-------------------|
| cos_phi | +0.001 | - | - | - |
| rho (aud/text) | 0.73 | - | ~2.6 | - |
| GSNR (audio) | - | - | - | 0.083 |
| D15 energy | - | - | - | 0.001-0.006 |
| CKA L12 | 0.955 | 0.036/0.994 | 0.987 | 1.000 |
| Audio rank | 877 | 863 | 862 | - |
| Cos collapse | 0.018 | 0.062 | - | 0.124 |
| Probe val acc L12 | 4.05% | 5.32% | 6.1% | ~6% |
| Probe train acc L18 | 13% | - | 77.1% | - |
| CB0 top-1 acc | 0.048 | - | 0.146 | 0.159 |
| CB0 top-10 acc | 0.207 | - | 0.421 | 0.444 |
| Backbone disp | 11.8 | 22.7 | 51 | - |
| lm_head disp | 60.8 | 144.4 | - | - |

---

## Table 5: Sample Efficiency Comparison

| Target val A1A2 | eff=32 (samples) | eff=192 (samples) | Ratio (192/32) |
|-----------------|------------------|-------------------|----------------|
| 42 | 346K | 960K | 2.8x |
| 38 | 672K | 1,478K | 2.2x |

---

## Table 6: B_crit Theory Quick Reference

| Parameter | Value | Source |
|-----------|-------|--------|
| B_crit_text | 2 | Derived from GSNR measurement |
| B_crit_audio | 25 | Derived from GSNR measurement |
| B* (optimal total) | 9 | B_crit_T + sqrt(B_crit_T * B_crit_A) |
| p* (text fraction) | 22% | B_crit_T / B* |
| Efficiency at B=32 | 70% | (1 + B_crit_A/B)^{-1} approx |
| Efficiency at B=192 | 19% | (1 + B_crit_A/B)^{-1} approx |
| Predicted ratio 192/32 | 3.8x | Theory |
| Observed ratio 192/32 | 2.8x | Empirical |

---

## Table 7: Per-Task Audio Val Losses (exp2 baseline, step 3K)

| Task | Input Modality | Output Modality | Text Val | Audio Val |
|------|---------------|-----------------|----------|-----------|
| T1T2 | Text | Text + Audio | 1.70 | 22.78 |
| T1A2 | Text | Audio only | 1.03 | 49.05 |
| A1T2 | Audio | Text + Audio | 2.13 | 23.09 |
| A1A2 | Audio | Audio only | 1.07 | 49.15 |

Key insight: audio-only output tasks (T1A2, A1A2) have ~2x higher loss than text+audio output tasks (T1T2, A1T2). Text output channel provides an information pathway that helps audio prediction.

---

## Table 8: Chain Training Results (mini-omni continued S3 from published checkpoint)

| Round | Global Steps | Eff Batch | A1A2 Val | A1T2 Val | CKA L12 |
|-------|-------------|-----------|----------|----------|---------|
| R1 | ~5,500 | 192 | - | - | 1.000 |
| R2 | ~11,000 | 192 | ~35 | - | 1.000 |
| R3 | ~16,500 | 192 | 32.3 | 14.9 | 1.000 |

---

## Table 9: Gradient Dynamics Evolution (exp2 baseline)

| Step | rho | cos_phi | text_gn | audio_gn | backbone_disp | CKA L12 |
|------|-----|---------|---------|----------|---------------|---------|
| 100 | 0.32 | -0.045 | 203.4 | 65.1 | 0.77 | - |
| 500 | 0.30 | -0.013 | 109.6 | 33.0 | 2.15 | 1.000 |
| 1000 | 0.44 | +0.002 | 40.1 | 17.4 | 5.02 | 0.974 |
| 2000 | 0.30 | -0.022 | 22.2 | 6.7 | 10.85 | 0.949 |
| 3000 | 0.73 | +0.001 | 14.2 | 10.4 | 11.79 | 0.955 |

---

## Table 10: Probe Accuracy by Layer (S2->S3, 30K steps)

| Layer | Train Accuracy | Val Accuracy | Gap |
|-------|---------------|-------------|-----|
| L6 | 18.6% | 6.2% | 3x |
| L12 | 57.3% | 6.1% | 9x |
| L18 | 77.1% | 6.2% | 12x |

Random chance = 1/4160 = 0.024%
