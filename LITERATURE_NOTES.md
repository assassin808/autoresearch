# Literature Review Notes (Auto-collected)

Hourly scans for papers relevant to omni-model training dynamics.

**Focus areas:**
- Omni-model (text+speech) training optimization
- Kimi / Moonshot AI attention residue work
- 贾佳亚 (Jiajia Jia) group omni multimodal work (CUHK MMLab)
- Gradient conflict in multi-task / multimodal learning
- Audio codec + LM training (SNAC, EnCodec, RVQ)
- Basin shaping, SAM, loss landscape for multimodal

---

## Scan: 2026-03-18 06:20 (run #8)

### Kimi Attention Residuals (HIGH PRIORITY)

- **Attention Residuals** (2026-03-15) Kimi Team, Moonshot AI | https://arxiv.org/abs/2603.15031
  Replaces standard residual connections (fixed uniform accumulation) with softmax attention over preceding layer outputs. Each layer learns input-dependent weights to selectively aggregate earlier representations. Addresses PreNorm dilution where deeper layers' contributions get progressively diluted. Block AttnRes partitions layers into blocks for practical scalability (<2% inference overhead). Tested on Kimi Linear (48B/3B MoE, 1.4T tokens). Improvements: MMLU 73.5→74.6, GPQA 36.9→44.4, Math 53.5→57.1.
  **Relevance**: HIGH — Could fundamentally change how omni-model layers interact. In our setting, audio and text information flow through shared layers; AttnRes could allow layers to selectively attend to modality-relevant earlier representations instead of uniformly mixing. May help with the gradient dilution we observe (ρ≈0.22 means audio signal is diluted). Also: "more uniform gradient distribution across depth" directly addresses our per-layer cos φ findings.

  Code: https://github.com/MoonshotAI/Attention-Residuals

### Gradient Conflict Methods (MTL)

- **Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond** (2025-01) Weiyu Chen, Baijiong Lin et al. | https://arxiv.org/abs/2501.10945
  Comprehensive survey of gradient-based MOO for deep learning. Categorizes methods into: (1) single balanced solution, (2) finite Pareto set, (3) continuous Pareto frontier. Covers PCGrad, CAGrad, Nash-MTL, MGDA, and newer methods.
  **Relevance**: HIGH — Reference survey for all gradient conflict methods we might apply to text/audio optimization.

- **ConicGrad: Fantastic Multi-Task Gradient Updates and How to Find Them In a Cone** (2025-01) Hassanpour et al. | https://arxiv.org/abs/2502.00217
  Introduces angular constraint that confines gradient updates within a cone centered on the reference gradient. Dynamically regulates gradient directions while maintaining scalability. SOTA on supervised and RL benchmarks.
  **Relevance**: MEDIUM — Our gradient projection is a special case (removing opposing component). ConicGrad is more principled — confine audio gradient to a cone around text gradient direction. Worth trying as a replacement for our grad_proj method.

- **Proactive Gradient Conflict Mitigation via Sparse Training** (2024-11) Zhang et al. | https://arxiv.org/abs/2411.18615
  Instead of manipulating gradients post-hoc, freeze a portion of parameters so different tasks learn on different subsets. Naturally partitions the model to reduce gradient conflict. Can combine with existing gradient methods.
  **Relevance**: MEDIUM — Interesting alternative: freeze text-critical params while training audio-critical params. Could be combined with our per-layer cos φ data (layers 0 and 23 have 70% negative cos φ → freeze those for audio training).

### Omni-Model Training

- **EMOVA: Empowering Language Models to See, Hear and Speak with Vivid Emotions** (CVPR 2025) Chen et al. | https://arxiv.org/abs/2409.18042
  First omni-modal LLM with emotional speech. Key training finding: **joint training of vision-language and speech outperforms sequential training**. Uses semantic-acoustic disentangled speech tokenizer. Three-stage: (1) VL pre-alignment, (2) omni-modal text-centric alignment, (3) omni-modal instruction tuning.
  **Relevance**: HIGH — Direct evidence that joint > sequential for multimodal training. Their "semantic-acoustic disentangled" tokenizer may address DRI. Their finding that "joint training fosters mutual enhancement" contrasts with our observation of text degradation (+4%) — possibly because they disentangle semantic from acoustic tokens.

- **Qwen3-Omni Technical Report** (2025-09) Xu et al. | https://arxiv.org/abs/2509.17765
  Thinker-Talker MoE architecture. Thinker processes all modalities, Talker generates speech autoregressively with multi-codebook scheme. Lightweight causal ConvNet for streaming (234ms latency). SOTA on 32/36 audio/visual benchmarks.
  **Relevance**: MEDIUM — Thinker-Talker separation is an architectural solution to text-audio conflict (separate generation paths). Different approach from shared-backbone like mini-omni.

- **Omni-Diffusion: Unified Multimodal Understanding and Generation with Masked Discrete Diffusion** (2026-03) Li et al. | https://arxiv.org/abs/2603.06577
  Uses masked discrete diffusion (not autoregressive) to model joint distribution over text, speech, image tokens. Avoids the sequential generation bottleneck entirely.
  **Relevance**: LOW — Fundamentally different architecture (diffusion vs AR). Not directly applicable to mini-omni framework but interesting long-term direction.

### Nash Bargaining for Multi-Task

- **Multi-Task Learning as a Bargaining Game** (ICML 2022) Navon et al. | https://github.com/AvivNavon/nash-mtl
  Frames MTL as Nash bargaining: each task is a "player" and the solution maximizes the product of per-task improvements. Efficient algorithm with convergence guarantees for convex and non-convex cases.
  **Relevance**: MEDIUM — Could replace our manual λ tuning. Nash-MTL would automatically find the optimal text/audio trade-off. Implementation available on GitHub. Worth trying after current experiments.

- **MUNBa: Machine Unlearning via Nash Bargaining** (2024-11) | https://arxiv.org/abs/2411.15537
  Applies Nash bargaining to resolve gradient conflict between forgetting and preservation objectives.
  **Relevance**: LOW — Different application but validates Nash bargaining for 2-objective gradient conflicts (analogous to our text preservation vs audio learning).

---

## Ideas to Try (from this scan)

1. **ConicGrad for text-audio**: Replace our simple gradient projection with ConicGrad's cone constraint. Audio gradient must stay within angular bound of text gradient direction.

2. **Sparse training by layer**: Freeze high-conflict layers (0, 23 per our cos φ data) during audio training. Let audio learn on mid-layers where cos φ ≈ 0.

3. **AttnRes for omni-models**: Replace residual connections with depth-wise attention. May allow selective information routing per modality. Requires architecture change but the Kimi team shows it's a drop-in replacement.

4. **Nash-MTL**: Replace fixed λ with Nash bargaining solution. Available implementation. Would automatically balance text/audio.

5. **Disentangled speech tokenizer** (from EMOVA): Separate semantic and acoustic tokens before feeding to LM. May reduce DRI and improve audio loss convergence.

---

## Scan: 2026-03-18 07:20 (run #9)

### VITA-1.5 Training Strategy (HIGH PRIORITY)

- **VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction** (NeurIPS 2025 Spotlight) Fu et al. | https://arxiv.org/abs/2501.01957
  Three-stage progressive training for omni-modal. Key insight: **LLM is FROZEN during audio decoder training (Stage 3)** — this completely prevents text degradation from audio training. Stage 1: vision alignment (adapter→full VL). Stage 2: audio input (CTC loss for encoder, then joint fine-tune). Stage 3: audio output decoder trained with frozen LLM. Uses TiCodec with single codebook (size 1024) — much simpler than SNAC's 7 codebooks.
  **Relevance**: HIGH — Their solution to text degradation is radical: don't backprop audio output loss through the LLM at all. This is the opposite of mini-omni's approach (all weights unfrozen in S3). Explains why mini-omni has +4% text degradation while VITA preserves text. Trade-off: audio decoder can't leverage LLM reasoning since LLM is frozen. Also: single codebook avoids the DRI scaling problem across multiple codebooks.

### DRI Mitigation (HIGH PRIORITY)

- **Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models** (ACL 2025 Oral) Liu, Guo, Xu et al. | https://arxiv.org/abs/2409.19283
  Quantifies DRI: same audio → different tokens depending on context. Proposes two fixes applied during **codec training** (not LM training):
  1. **Slice-consistency**: encode random audio segment separately, constrain its latent to match the corresponding portion from full audio. Loss: `ℒ_slice = MSE(Z_slice, Z_full_portion)`.
  2. **Perturbation-consistency**: add imperceptible phase perturbations, constrain latent to match original. Loss: `ℒ_perception = MSE(Z_perturbed, Z_original)`.
  Results: consistency +21-36% across codebook layers. Downstream: WER 4.73→1.84 (-61%), speaker similarity 77→84%.
  **Relevance**: HIGH — DRI is likely the root cause of our audio plateau. SNAC tokens are inconsistent, creating contradictory training signals that cap audio loss. Fix is at the codec level (retrain SNAC with consistency losses), not the LM level. This explains why our optimizer experiments (λ=3, grad_proj) have limited effect — the bottleneck is in the data, not the optimization.

### Progressive Training Curriculum

- **Preparing Lessons for Progressive Training on Language Models** (AAAI 2024) | https://arxiv.org/abs/2401.09192
  Apollo method: train low layers first, then progressively expand to higher layers. Uses low-value-prioritized sampling (LVPS) and weight sharing for efficient expansion.
  **Relevance**: LOW — Layer-progressive training is interesting conceptually but our model is already pretrained. More relevant for training from scratch.

### Language-Codec for Audio LMs

- **Language-Codec: Reducing the Gaps Between Discrete Codec Representation and Speech Language Models** (ACL 2025 Oral) | https://github.com/jishengpeng/Languagecodec
  Addresses the semantic gap between codec tokens and LM needs. Adaptive dropout depths to differentially train codebooks across layers.
  **Relevance**: MEDIUM — The "adaptive dropout depths" for codebook training is interesting. Different codebook layers get different dropout → forces each layer to be independently useful. May help with our CB hierarchy convergence.

---

## Updated Ideas to Try (cumulative)

### From scan #8:
1. ConicGrad for text-audio gradient cone constraint
2. Sparse training: freeze high-conflict layers (0, 23)
3. AttnRes for depth-wise modality routing
4. Nash-MTL for automatic text/audio balancing
5. Disentangled speech tokenizer (EMOVA)

### From scan #9:
6. **Frozen LLM for audio decoder** (VITA-1.5 approach): train audio output path with LLM frozen. Zero text degradation but limits audio quality.
7. **DRI-aware codec retraining**: retrain SNAC with slice-consistency + perturbation-consistency losses. Addresses root cause of audio plateau.
8. **Single codebook** (TiCodec): simplify from 7 SNAC codebooks to 1. Reduces DRI surface area and simplifies loss computation. Trade-off: lower audio quality.
9. **Adaptive codebook dropout**: different dropout per codebook layer during codec training (Language-Codec approach).

---

## Scan: 2026-03-18 08:20 (run #0)

### Moshi Training Details (HIGH PRIORITY — exact numbers)

- **Moshi: a speech-text foundation model for real-time dialogue** (2024-10) Kyutai team | https://arxiv.org/abs/2410.00037
  Finally extracted exact training numbers:
  - **Codebook weighting: α_semantic = 100, α_acoustic = 1** (100:1 ratio!). This is in their loss Eq.7. Semantic tokens get 100× more gradient signal than acoustic tokens.
  - **Text-only batches: 50%** of training is text-only to prevent catastrophic forgetting.
  - **Padding token weight: reduced 50%** since padding dominates audio batches.
  - **Delay pattern: τ=1 or τ=2** steps between semantic and acoustic features.
  - **Inner Monologue**: text tokens prepended as prefix to audio at each frame (12.5Hz alignment from Whisper ASR). Reduces NLL 4.36→2.77. Spoken QA: 9%→26.6%.
  - **Architecture**: 7B Temporal Transformer (time) + small Depth Transformer (inter-codebook per frame).
  **Relevance**: CRITICAL — The 100:1 semantic vs acoustic weighting is the opposite of our equal weighting. In mini-omni all 7 SNAC streams get equal loss weight. Moshi concentrates 99% of audio gradient on semantic tokens. This may explain why our codebook hierarchy shows all CBs plateauing together — they should be weighted drastically differently. Also: 50% text batches matches our observation that text needs protection.

### UALM (NVIDIA)

- **UALM: Unified Audio Language Model for Understanding, Generation, and Reasoning** (2025) NVIDIA ADLR | https://research.nvidia.com/labs/adlr/UALM/
  Unifies audio understanding + generation + reasoning in single 7B model. Key trick: **upweight audio generation data** (not loss weight — data sampling weight) because generation is harder. Uses warmup stage before full fine-tuning. Matches specialized models without text degradation.
  **Relevance**: MEDIUM — Data upweighting (more audio gen samples per batch) as alternative to loss weighting. Simpler than our gradient methods. Their success "without capability degradation" suggests careful data mixing is more important than optimizer tricks.

### Omni-Model Survey

- **On The Landscape of Spoken Language Models: A Comprehensive Survey** (2025-04) | https://arxiv.org/abs/2504.08528
  Comprehensive survey of speech LMs covering training strategies, speech/text token decoding patterns, duplex dialogue, benchmarks.
  **Relevance**: LOW — Reference survey, not new methods.

---

## Key Insight from This Scan

**Moshi's 100:1 codebook weighting is a major finding we missed.** Our S3 training uses equal weight on all 7 SNAC streams. Moshi weights semantic tokens 100× more than acoustic. This means:

1. Semantic codebook (CB1 in SNAC = stream 0) should get ~100× more loss weight
2. Fine acoustic codebooks (CB3 = streams 2,3,5,6) should get ~1× weight
3. This would dramatically change ρ(t) — audio gradient would be dominated by semantic prediction

**Experiment to add**: S3 training with Moshi-style hierarchical weighting: CB1=100, CB2=10, CB3=1.

---

## Scan: 2026-03-18 09:20 (run #1)

### X-Codec: Semantic-Enriched Audio Tokens (HIGH PRIORITY)

- **X-Codec: Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model** (AAAI 2025) Ye et al. | https://arxiv.org/abs/2408.17175
  Pure acoustic codecs (EnCodec, SNAC) force LMs to "predict local fluctuations of the audio signal, which is difficult." X-Codec injects HuBERT semantic features before RVQ via linear projection, plus semantic reconstruction loss after RVQ. Result: ABX phonetic error 3.3% vs EnCodec's 17.5%. WER: 4.07% vs EnCodec 6.37%. The tokens carry semantic meaning, not just acoustic detail.
  **Relevance**: HIGH — Directly addresses why our audio loss plateaus. SNAC tokens lack semantic content → LM struggles to predict them. X-Codec-style semantic injection would make tokens more predictable. Combined with DRI mitigation (scan #9), this is the codec-level fix our optimizer experiments can't achieve.

### MinMo: 4-Stage Voice Interaction Training

- **MinMo: A Multimodal LLM for Seamless Voice Interaction** (2025-01) Alibaba FunAudioLLM | https://arxiv.org/abs/2501.06282
  8B model, 1.4M hours speech data. Four stages: (1) speech-to-text 1.2M hrs, (2) text-to-speech 170K hrs, (3) speech-to-speech 10K hrs, (4) duplex 4K hrs. Key: **LLM frozen initially, updated only via LoRA** later. Uses CosyVoice 2 discrete tokens (not SNAC/EnCodec). 100ms S2T latency.
  **Relevance**: MEDIUM — Another data point for "freeze LLM, train adapters" approach (like VITA-1.5). LoRA for LLM update is a middle ground between full freeze and full unfreeze. 1.4M hours vs our 88K samples highlights the data scale gap.

### Attention/Residual Sinks

- **A Unified View of Attention and Residual Sinks: Outlier-Driven Rescaling** (2026-01) Qiu et al. | https://arxiv.org/abs/2601.22966
  Attention sinks and residual sinks are the same phenomenon: outliers serve as rescale factors for normalization (softmax, RMSNorm), not as information carriers.
  **Relevance**: LOW — Theoretical insight about transformer internals. Related to Kimi AttnRes (both address residual accumulation issues) but no direct training optimization implications for us.

---

## Running Tally: Codec is the Bottleneck

Three scans now point to the same conclusion:
1. **DRI** (scan #9): Same audio → different SNAC tokens. Consistency fix: WER 4.73→1.84.
2. **X-Codec** (this scan): Acoustic tokens lack semantics. Semantic injection: WER 6.37→4.07.
3. **Moshi** (scan #0): 100:1 semantic vs acoustic weighting. Semantic tokens matter 100× more.

**Our audio plateau (4.56→3.57, then stuck) is fundamentally a codec/token problem, not an optimizer problem.** The tokens we're predicting are (a) inconsistent and (b) semantically empty. No amount of gradient manipulation can fix bad training targets.

### Actionable next steps (ordered by expected impact):
1. **Moshi-style CB weighting** (easiest, no architecture change): CB1=100, CB2=10, CB3=1
2. **DRI-consistent SNAC** (medium effort): retrain SNAC with slice+perturbation consistency
3. **X-Codec replacement** (high effort): replace SNAC with semantic-enriched codec
4. **Frozen LLM + LoRA** (medium effort): VITA-1.5/MinMo approach to prevent text degradation

---

## Scan: 2026-03-18 10:20 (run #2)

### MGM-Omni — 贾佳亚 Group (HIGH PRIORITY)

- **MGM-Omni: Scaling Omni LLMs to Personalized Long-Horizon Speech** (2025-09) Wang, Zhong, Peng, ..., **Jiaya Jia** | https://arxiv.org/abs/2509.25131
  From Jiaya Jia's group (HKUST/DVLAB). "Brain-mouth" dual-track architecture: MLLM ("brain") handles multimodal reasoning, SpeechLM ("mouth") handles speech token generation. Key design: **decouple reasoning from speech generation** — brain produces text, mouth converts to speech tokens via chunk-based parallel decoding (bridges text-speech token rate gap). Uses CosyVoice2 tokenizer + flow matching vocoder. Built on Qwen2.5-VL + Qwen3 LLMs. Claims "data-efficient" training (~400K hours audio). Training code not yet released.
  **Relevance**: HIGH — The brain-mouth separation is architecturally similar to VITA-1.5's frozen-LLM approach: the MLLM doesn't need to generate speech tokens directly, so no text-audio gradient conflict in the backbone. The SpeechLM is a separate module trained on text→speech. This completely avoids the problem we're studying (gradient interference in shared backbone). Worth understanding if this is strictly better or if there's a quality trade-off vs end-to-end approaches like mini-omni.

  Code: https://github.com/JIA-Lab-research/MGM-Omni (training code TBD)

### CosyVoice 2: Better Speech Codec

- **CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models** (2024-12) Alibaba FunAudioLLM | https://arxiv.org/abs/2412.10117
  Replaces VQ with Finite Scalar Quantization (FSQ) — achieves **100% codebook utilization** vs VQ's typical 23%. Removes text encoder and speaker embedding, using pretrained LLM as backbone directly. Trained on 200K hours. Streaming + non-streaming unified.
  **Relevance**: MEDIUM — FSQ achieving 100% codebook utilization is significant. SNAC/EnCodec likely have low utilization too, meaning many codebook entries are dead → effectively smaller vocab → harder prediction task. FSQ may reduce DRI by making quantization more uniform. Used by both MinMo and MGM-Omni.

### MMAudio: Joint Training for Audio Synthesis

- **MMAudio: Taming Multimodal Joint Training for High-Quality Video-to-Audio Synthesis** (CVPR 2025) | https://github.com/hkchengrex/MMAudio
  Joint training of video-to-audio with multimodal conditioning. "Taming" suggests they had to solve training instability issues.
  **Relevance**: LOW — Video-to-audio, not speech. But "taming joint training" may have relevant tricks.

---

## Emerging Pattern: Brain-Mouth Separation

Three independent groups converge on the same solution to text-audio interference:

| Model | Approach | Text degradation |
|-------|----------|-----------------|
| Mini-omni | Shared backbone, all unfrozen | +4% (our measurement) |
| VITA-1.5 | Frozen LLM for audio decoder | 0% |
| MinMo | Frozen LLM → LoRA update | ~0% |
| MGM-Omni | Separate SpeechLM ("mouth") | ~0% |
| Qwen3-Omni | Thinker-Talker MoE | ~0% |

**Mini-omni's approach (shared backbone for everything) is the outlier.** Every other successful omni-model separates text reasoning from speech generation in some way. Our gradient interference findings (cos φ = -0.29) explain why: when text and audio share all parameters, their gradients conflict.

This doesn't mean optimizer tricks are useless — but they're fighting architecture, not just optimization.

---

## Scan: 2026-03-18 11:20 (run #3)

### Gradient Methods Deep Dive

- **CAGrad: Conflict-Averse Gradient Descent** (NeurIPS 2021) Liu et al. | https://arxiv.org/abs/2110.14048
  Maximizes worst-case local improvement across tasks within a neighborhood of the average gradient. Includes GD and MGDA as special cases. **Provably converges to average loss minimum** (not just any Pareto point). Open-source: https://github.com/Cranial-XIX/CAGrad
  **Relevance**: HIGH — Direct replacement for our naive gradient projection. CAGrad finds the update that maximally improves the worst task (audio in our case) while staying near the average gradient. Our grad_proj just removes the opposing component; CAGrad optimizes the direction more carefully. Easy to implement (drop-in optimizer wrapper).

- **Aligned-MTL: Independent Component Alignment** (CVPR 2023) Senushkin et al. | https://arxiv.org/abs/2305.19000
  Aligns principal components of the gradient matrix across tasks. Uses condition number as stability criterion. **Provably converges to optimum with pre-defined task weights.** Resolves both gradient conflicts AND gradient dominance.
  **Relevance**: MEDIUM — More principled than our projection but requires SVD of gradient matrix per step. May be too expensive for 532M params. Could apply per-layer (our cos φ data shows layers 0,23 are most conflicted).

- **MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance** (2024) Wei & Hu | https://arxiv.org/abs/2405.17730
  Key finding: **standard Pareto methods FAIL in multimodal contexts** because multimodal loss has smaller gradient magnitude than unimodal. Pareto integration misleads unimodal encoder optimization. Fix: align gradient directions across objectives while amplifying magnitudes.
  **Relevance**: HIGH — Directly explains our ρ≈0.22 observation. Audio (multimodal) gradients are 4× weaker than text (dominant modality). Standard gradient balancing (like our λ=3) can't fix this because the issue is direction, not magnitude. MMPareto's insight: need to align directions AND amplify the weak modality.

- **SAMO: Sharpness-Aware Multi-Task Optimization** (2025-07) Ban et al. | https://arxiv.org/abs/2507.07883
  Combines SAM with multi-task gradient balancing. "Joint global-local perturbation": weighted average of task-specific and shared gradients as SAM perturbation direction. Finds flat regions where "changes in one objective don't significantly affect the other." Lightweight — approximates task gradients with forward passes only.
  **Relevance**: HIGH — This is exactly our Approach D (operator-norm SAM) from the meeting prep, but done more carefully. SAMO seeks flat regions where text and audio don't conflict — directly addresses the basin-shaping idea. The forward-pass-only gradient approximation makes it practical for 532M params. Should be our next experiment.

---

## Method Comparison for Our Setting

| Method | What it does | Cost | Addresses our problem? |
|--------|-------------|------|----------------------|
| Our grad_proj | Remove opposing audio component | 2× fwd+bwd | Partially — audio still plateaus |
| CAGrad | Max worst-task improvement near avg gradient | 1.5× fwd+bwd | Better — optimizes direction |
| Aligned-MTL | SVD alignment of gradient matrix | Expensive (SVD) | Yes but impractical at scale |
| MMPareto | Pareto + magnitude amplification | ~1.5× | Yes — addresses ρ≈0.22 issue |
| SAMO | SAM + multi-task perturbation | ~2× fwd | Yes — finds flat multi-task regions |
| Nash-MTL | Game-theoretic bargaining | ~2× | Yes — automatic balancing |

**Recommended priority**: SAMO > CAGrad > MMPareto > Nash-MTL

But recall: all optimizer methods face the codec bottleneck (DRI + semantic poverty). Even the best gradient method can't overcome bad training targets.

---

## Scan: 2026-03-18 12:20 (run #4)

### Codec Evolution: The Semantic-Acoustic Split

- **SpeechTokenizer: Unified Speech Tokenizer for Speech LLMs** (ICLR 2024) Zhang et al. | https://arxiv.org/abs/2308.16692
  First RVQ layer captures semantic content (distilled from HuBERT teacher), remaining layers capture acoustic/timbre. AR model generates semantic (layer 1) tokens, NAR model fills in acoustic (layers 2+). Key insight: **semantic and acoustic should be modeled separately**, not jointly predicted with equal weight.
  **Relevance**: HIGH — Directly supports Moshi's 100:1 weighting rationale. SNAC's 7 streams don't distinguish semantic from acoustic. SpeechTokenizer shows that layer 1 = semantic, layers 2+ = acoustic. Our equal weighting across all 7 SNAC streams is fundamentally wrong.

- **XY-Tokenizer: Mitigating the Semantic-Acoustic Conflict in Low-Bitrate Speech Codecs** (2025-06) Gong et al. | https://arxiv.org/abs/2506.23325
  Identifies "semantic-acoustic conflict": codecs must simultaneously preserve acoustic fidelity AND capture semantics, but these objectives compete. Uses multi-stage multi-task learning to balance both. Surpasses distillation-based methods (SpeechTokenizer, Mimi) on semantic benchmarks while matching acoustic quality.
  **Relevance**: MEDIUM — The "semantic-acoustic conflict" in the codec mirrors our "text-audio conflict" in the LM. Both are multi-objective problems where two goals compete. The codec-level fix (multi-task training of the codec itself) is complementary to LM-level fixes.

- **Factorized RVQ-GAN (HAC): Disentangled Speech Tokenization** (2025-06) Khurana et al. | https://arxiv.org/abs/2506.15456
  Three separate quantization modules: acoustic (7-layer RVQ), phonetic (VQ from HuBERT distillation), lexical (VQ from LaBSE distillation). Complete separation of acoustic, phonetic, and word-level tokens. Each module has its own codebook.
  **Relevance**: MEDIUM — Takes disentanglement to the extreme. If we replaced SNAC with HAC, the LM would predict phonetic+lexical tokens (easy, semantic) separately from acoustic tokens (hard, noisy). The audio plateau might only affect acoustic tokens, while semantic prediction converges properly.

### SNAC-Specific Notes

- **SNAC: Multi-Scale Neural Audio Codec** (NeurIPS 2024 Workshop) Siuzdak | https://arxiv.org/abs/2410.14411
  SNAC's multi-scale design: coarse tokens at low temporal resolution (broad context), fine tokens at high resolution (local detail). 3 codebooks at different frame rates → 7 streams when flattened. Trained for compression (250K iterations), NOT for LM compatibility. No semantic distillation, no consistency training.
  **Relevance**: HIGH (context) — Confirms SNAC was designed for compression, not for LM prediction. It has no semantic grounding (unlike SpeechTokenizer, X-Codec, HAC). This is the root cause: we're training an LM to predict tokens that were never designed to be predictable.

---

## Codec Landscape Summary

| Codec | Semantic? | Consistent? | Designed for LM? | Used by |
|-------|-----------|-------------|-------------------|---------|
| **SNAC** | No | No | No (compression) | Mini-omni |
| **EnCodec** | No | No (47%) | No (compression) | VALL-E, early work |
| **SpeechTokenizer** | Layer 1 only | Partial | Yes (AR+NAR) | USLM |
| **X-Codec** | Yes (HuBERT inject) | Partial | Yes | AAAI 2025 |
| **Mimi** (Moshi) | Layer 1 (distilled) | Better | Yes | Moshi |
| **CosyVoice2 FSQ** | Via LLM backbone | 100% utilization | Yes | MinMo, MGM-Omni |
| **HAC** | Separate modules | By design | Yes | Factorized RVQ-GAN |
| **XY-Tokenizer** | Multi-task balanced | Better | Yes | 2025 |

**SNAC is the worst choice for LM-based audio prediction.** Every modern codec designed for LMs addresses semantic content and/or consistency. SNAC addresses neither.

This is the strongest evidence yet that our audio plateau is a codec limitation, not an optimization problem.

---

## Scan: 2026-03-18 13:20 (run #5)

### Modality-Aware SAM (HIGH PRIORITY)

- **Modality-Aware SAM (M-SAM): Sharpness-Aware-Minimization Driven Gradient Modulation for Harmonized Multimodal Learning** (NeurIPS 2025) Nowdeh et al. | https://arxiv.org/abs/2510.24919
  Uses Shapley values to identify dominant modality per iteration, then modulates loss landscape to prioritize robustness of the dominant modality while enabling weaker modalities to contribute. Three steps: (1) identify dominant modality via Shapley, (2) decompose loss landscape favoring dominant, (3) apply modulated gradients. Model-agnostic, supports early/late fusion.
  **Relevance**: CRITICAL — This is exactly what we need. In our setting, text is the dominant modality (ρ≈0.22, text gradients 4× larger). M-SAM would identify text as dominant, then modulate the loss landscape to let audio contribute without being overshadowed. Combines SAM's flat-minima seeking with modality-aware gradient balancing. Much more principled than our naive grad_proj. **Should be our top priority experiment.**

### Multi-Task Learning via Generalization (Flat Minima)

- **Beyond Losses Reweighting: Empowering Multi-Task Learning via the Generalization Perspective** (ICCV 2025) Phan et al. | https://openaccess.thecvf.com/content/ICCV2025/papers/Phan_Beyond_Losses_Reweighting_Empowering_Multi-Task_Learning_via_the_Generalization_Perspective_ICCV_2025_paper.pdf
  Key insight: **loss reweighting is insufficient — need to address loss landscape geometry.** Combines flat gradient component (weight perturbations to penalize sharpness per task) with loss gradient component. Theorem: controlling gradient norm bounds generalization. Task-specific params get standard updates, shared params get SAM-style flat-seeking updates.
  **Relevance**: HIGH — Directly validates our finding that λ=3 (loss reweighting) doesn't help. The paper proves theoretically that loss-level adjustments can't fix geometry-level problems. The fix must operate on the landscape itself (SAM-style perturbations), not just the loss weights.

### Basin-Like Loss Landscape (Our Starting Paper)

- **Unveiling the Basin-Like Loss Landscape in Large Language Models** (2025-05) Chen et al. | https://arxiv.org/abs/2505.17646
  Original GO paper. Key claims: (1) pre-training creates basic capability basin, (2) fine-tuning creates nested specific basins, (3) GO (Gaussian perturbation during training) widens basins, (4) basin size bounds worst-case fine-tuning degradation. GO is most effective during pre-training.
  **Relevance**: MEDIUM (revisited) — Our experiments falsified the "audio as implicit GO" hypothesis (basin width unchanged). But the GO insight is still valid: wider basins resist degradation. M-SAM/SAMO are more targeted versions of GO for multi-task settings.

---

## Revised Method Priority (final)

Based on 6 scans of literature:

### Tier 1: Most promising for our setting
1. **M-SAM** — Modality-aware SAM with Shapley identification. Directly addresses ρ≈0.22 + cos φ<0.
2. **Moshi-style CB weighting** (100:1) — Zero code complexity. Largest expected impact from a single change.
3. **SAMO** — SAM + multi-task, finds flat regions where tasks don't conflict.

### Tier 2: Worth trying
4. **CAGrad** — Max worst-task improvement. Drop-in optimizer wrapper.
5. **Frozen LLM + LoRA for audio** — VITA-1.5/MinMo approach. Eliminates gradient conflict.
6. **DRI-consistent SNAC retraining** — Fix codec tokens. Addresses root cause.

### Tier 3: Requires architecture change
7. **Brain-mouth separation** (MGM-Omni) — Separate SpeechLM.
8. **X-Codec / SpeechTokenizer replacement** — Semantic-aware codec.
9. **AttnRes** (Kimi) — Depth-wise attention for modality routing.

### Key insight from literature review
The field has bifurcated:
- **Architecture camp**: separate text and speech paths (VITA-1.5, MGM-Omni, Qwen3-Omni)
- **Codec camp**: make tokens more LM-friendly (X-Codec, DRI fix, SpeechTokenizer)
- **Optimizer camp**: gradient methods (M-SAM, SAMO, CAGrad) — less explored for omni-models

Our work sits uniquely in the optimizer camp with **empirical evidence** (ρ, cos φ, plateau curves) showing WHY the optimizer alone is insufficient. This is the novel contribution: not "a better optimizer" but "a characterization of why optimizers hit a wall."

---

## Scan: 2026-03-18 14:20 (run #6)

### LLaMA-Omni Training Details

- **LLaMA-Omni: Seamless Speech Interaction with Large Language Models** (ICLR 2025) | https://arxiv.org/abs/2409.06666
  Two-stage: (1) train speech adapter + LLM (encoder frozen, no decoder), LR=2e-5. (2) train speech decoder only, LR=2e-4. Uses **CTC loss** for non-autoregressive speech token alignment — no need to pre-align speech and text. Built on Llama-3.1-8B-Instruct.
  **Relevance**: MEDIUM — CTC loss for speech output is a fundamentally different approach from mini-omni's CE on SNAC tokens. CTC handles variable-length alignment naturally, avoiding the DRI problem entirely (CTC marginalizes over all valid alignments). Worth noting as an alternative to discrete token prediction.

### Mimi Codec Deeper Details

- **Moshi/Mimi** (confirmed details from scan #0) | https://arxiv.org/abs/2410.00037
  Mimi distills WavLM into first codebook via **cosine similarity loss** between VQ vectors and WavLM vectors. This makes codebook 1 semantic. Training: Temporal Transformer (7B, time axis) + Depth Transformer (small, codebook axis). α_semantic=100, α_acoustic=1. 50% text batches. 50% padding weight.
  **Relevance**: Already noted. Confirms Mimi's approach = SpeechTokenizer + DRI mitigation in one codec.

### Llama-Mimi: Flat Sequence Modeling

- **Llama-Mimi: Exploring the Limits of Flattened Speech Language Modeling** (2025-09) Sugiura et al. | https://arxiv.org/abs/2509.14882
  Flattens all Mimi RVQ tokens into a single sequence, models autoregressively with one Transformer (no Depth Transformer). Outperforms hierarchical CSM-based models on acoustic consistency. Shows that **a simple flat approach can beat complex hierarchical architectures** when the codec tokens are good (Mimi has semantic first codebook).
  **Relevance**: MEDIUM — Implies that with a good codec (semantic + consistent), even the simplest LM approach works well. Mini-omni's elaborate 8-stream averaging may be unnecessary if the codec is right. Reinforces: fix the codec, simplify the architecture.

---

## No new ideas to add — the picture is complete.

After 7 scans (scans #0-#6), the literature review has converged. Three clear directions:

1. **Fix the codec** (highest impact, most work)
2. **Separate speech from LLM** (architecture change)
3. **Better gradient methods** (M-SAM, SAMO — diminishing returns vs codec fix)

Our unique contribution remains: **empirical characterization of the optimization wall** in shared-backbone omni-models, with D1-D7 diagnostics showing exactly where and why gradient methods fail.

---

## Scan: 2026-03-18 15:20 (run #7)

### Codebook Weighting & Alternatives

- **Continuous Audio Language Models** (2025-09) Rouard et al. (Kyutai) | https://arxiv.org/abs/2509.06926
  Avoids discrete tokens entirely — uses continuous audio VAE + consistency modeling. Generates continuous waveforms via MLP conditioned on Transformer embeddings. "Higher quality at lower computational cost" than discrete models. Released Pocket TTS (100M, real-time on laptop CPU).
  **Relevance**: MEDIUM — The nuclear option: skip discrete tokens entirely, eliminating DRI by construction. From same lab as Moshi. If discrete codecs are fundamentally limited, continuous approaches may be the future. But requires completely different architecture.

- **Representation Collapse in VQ Models** (ICCV 2025) Zhu et al. | https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_Addressing_Representation_Collapse_in_Vector_Quantized_Models_with_One_Linear_ICCV_2025_paper.pdf
  Addresses codebook collapse (dead codes) in VQ with a single linear layer fix. Related to CosyVoice2's FSQ achieving 100% utilization.
  **Relevance**: LOW — Codec training fix, not LM training.

- **ERVQ: Enhanced RVQ** (2024) — adds intra-codebook balancing (online clustering + code balancing loss) and inter-codebook diversity (SSIM loss). Eliminates codebook collapse.
  **Relevance**: LOW — Codec-level optimization.

### Summary note
This scan confirms: the Kyutai team (Moshi authors) is also exploring continuous alternatives to discrete tokens, suggesting even they recognize the fundamental limitations of discrete codecs for LM-based audio generation.

No new actionable items beyond existing priority list.

---

## Scan: 2026-03-18 16:20 (run #8, wrap)

### New: Speech-Omni-Lite (March 10, 2026)

- **Speech-Omni-Lite: Portable Speech Interfaces for Vision-Language Models** (2026-03-10) Tao et al. | https://arxiv.org/abs/2603.09627
  Adds speech to VL models with **VL backbone fully frozen**. Only trains two lightweight plug-and-play modules: speech projector + speech token generator. "Fully preserving VL performance." Uses low-cost data construction from ASR pairs (thousands of hours, not millions). Just 8 days old.
  **Relevance**: MEDIUM — Yet another frozen-backbone approach. Confirms the pattern: 6/6 successful models freeze the LLM when adding speech. Mini-omni's all-unfrozen S3 is the exception. Lightweight modules (projector + generator) are analogous to mini-omni's whisper_adapter but they don't backprop through the LLM.

No other new findings — queries wrapped to same topics as earlier scans.
