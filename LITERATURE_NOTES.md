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
