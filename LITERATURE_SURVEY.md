# Omni-Model Literature Survey

*Compiled 2026-03-14 for autoresearch project*

---

## 1. Mini-Omni (gpt-omni/mini-omni)

**Paper:** [arXiv 2408.16725](https://arxiv.org/abs/2408.16725)

### Architecture
- **Base LLM:** Qwen2-0.5B (24 transformer blocks, 896 internal dim)
- **Audio input:** Whisper-small encoder → 2-layer MLP adapter (ASR adapter)
- **Audio output:** SNAC codec (7 token layers), 6 additional transformer blocks (TTS adapter)
- **Key idea:** "Any Model Can Talk" — pre/post adapter design for rapid speech adaptation

### Training Stages
| Stage | What's Trained | What's Frozen | Data |
|-------|---------------|---------------|------|
| 1. Modality Alignment | ASR + TTS adapters only | Core LLM | ASR data: LibriTTS (586h), VCTK (44h), MLS (8000h) |
| 2. Adaptation Training | Core LLM only | Both adapters | TextQA (2M Open-Orca), AudioQA (1.5M synthesized) |
| 3. Multimodal Fine-tuning | Everything unfrozen | Nothing | VoiceAssistant-400K + all above + RLHF (367k) |

### Optimizer & Hyperparameters
- **Optimizer:** Not explicitly named (likely AdamW based on codebase)
- **LR schedule:** Cosine annealing, min=4e-6, max=4e-4 (pre-train); 4e-6 to 5e-5 (fine-tune)
- **Batch size:** 192 per step
- **Steps:** 40,000 per epoch
- **Hardware:** 8x A100

### Training Dynamics Issues
- Flattened audio tokenization caused instability
- Text-instructed parallel generation with delay: text tokens first, then SNAC tokens
- Batch parallel decoding (two parallel samples) transfers text capabilities to audio modality

### Data Scale
~8,000h speech + 2M text QA + 1.5M audio QA + 400K voice assistant

---

## 2. Mini-Omni 2

**Paper:** [arXiv 2410.11190](https://arxiv.org/abs/2410.11190)

### Architecture Changes from Mini-Omni 1
- Added **visual modality** (CLIP ViT-B/32 + single-layer LlamaMLP adapter)
- Expanded vocab: 7 x 4160 sub-LM-heads = 181,120 tokens for parallel text-audio generation
- **Interruption mechanism** using special "irq"/"n-irq" tokens for duplex conversation

### Training Stages
| Stage | What's Trained | What's Frozen | Data |
|-------|---------------|---------------|------|
| 1. Encoder Adaptation | Linear layers (connectors) only | LLM + encoders | Multi-modal alignment data |
| 2. Modality Alignment | LLM weights | Adapters frozen | Image/audio QA (text output only) |
| 3. Post-training | Everything | Nothing | All tasks retrained with audio output + interruption |

### Optimizer
- **LR:** 2e-5 to 1e-3 (adapter), 2e-6 to 2e-4 (LM), 2e-6 to 2e-5 (fine-tune)
- **Cosine scheduler**, 1500 warmup steps, batch=192

### Known Issues
- **High audio training loss** and token instability
- "Tokens for a segment of audio can vary significantly based on the content at both ends"
- Led to using Whisper semantic features instead of pure token-based audio input

### Data Scale
~5M+ samples: 638h speech, 1.5M text QA, 1.5M audio QA, 400K image QA, 400K voice assistant

---

## 3. SpeechGPT

**Paper:** [arXiv 2305.11000](https://arxiv.org/abs/2305.11000) (EMNLP 2023 Findings)

### Architecture
- **Base LLM:** LLaMA (7B)
- **Speech tokens:** HuBERT discrete tokens (501 phonetic units) added to vocabulary
- **No separate encoder/decoder** — speech handled as discrete tokens in unified vocabulary

### Training Stages (3 stages)
| Stage | Description | Details |
|-------|-------------|---------|
| 1. Modality-Adaptation Pre-training | LLM learns speech token distribution | Trains on speech data |
| 2. Cross-Modal Instruction Fine-tuning | LLM learns speech-text mapping | Paired speech-text data |
| 3. Chain-of-Modality Instruction Fine-tuning | LLM generates speech via text chain | SpeechInstruct dataset |

### Optimizer & Hyperparameters
- **Optimizer:** AdamW
- **LR:** 1e-4 (stage 1), 5e-5 (stage 2, to prevent catastrophic forgetting)
- **Weight decay:** 0.01
- **Batch size:** 16 per GPU, gradient accumulation=4
- **Warmup:** Linear over 500 steps

### Key Training Techniques
- Curriculum learning (gradually increases complexity)
- Mixed precision (FP16/FP32)
- Cross-entropy + contrastive alignment loss
- Conservative unfreezing in later stages

### Data
- SpeechInstruct dataset (custom cross-modal instruction data)
- Standard speech benchmarks (SLURP, STOP, Fluent Speech Commands)

---

## 4. VITA

**Paper:** [arXiv 2408.05211](https://arxiv.org/abs/2408.05211)

### Architecture
- **Base LLM:** Mixtral 8x7B (expanded vocab 32K → 51,747 tokens for Chinese)
- **Visual encoder:** Not specified in detail, connected via connector
- **Audio encoder:** Mel filter bank → 4x CNN downsampling → 24 transformer layers (341M params) → 2-layer MLP connector
- Audio: 25 tokens per 2 seconds
- **State tokens:** <1>, <2>, <3> distinguish query audio, noisy audio, text queries

### Training Stages
| Stage | What's Trained | What's Frozen | Data |
|-------|---------------|---------------|------|
| 1. LLM Instruction Tuning | LLM | - | 55M bilingual text corpus |
| 2. Multimodal Alignment | Connectors only | LLM + encoders | Visual: 2.75B entries; Audio: WenetSpeech (10Kh) + GigaSpeech (10Kh) + AudioSet (400K) |
| 3. Multimodal Instruction Tuning | Connectors + LLM | Encoders frozen | 5.96B total entries across modalities |

### Optimizer
- Not explicitly specified in paper

### Known Issues
- TTS-based noise simulation sometimes misclassifies inputs
- Balancing visual + audio + text required careful data ratio tuning

### Data Scale
- Audio: 20,000+ hours (Chinese + English)
- Visual: billions of entries
- Total instruction tuning: ~6B entries

---

## 5. Moshi (Kyutai)

**Paper:** [arXiv 2410.00037](https://arxiv.org/abs/2410.00037) — **Most detailed training paper in this survey**

### Architecture
- **Helium:** 7B text LLM (32 layers, dim=4096, 32 heads, SiLU-GLU, RoPE, FlashAttention)
- **Mimi codec:** SeaNet autoencoder → RVQ with 8 codebooks (2048 entries each), 1.1kbps
  - Split RVQ: 1 semantic VQ (distilled from WavLM) + 7 acoustic RVQ
  - 8-layer Transformer bottleneck (8 heads, 20s context)
  - 12.5Hz frame rate, 24kHz audio
- **Depth Transformer:** Small model for inter-codebook dependencies per timestep
- **Temporal Transformer:** Large 7B model (initialized from Helium) for temporal dependencies
- **Inner Monologue:** Time-aligned text tokens as prefix to audio tokens

### Training Stages (5 stages!)
| Stage | LR (Temporal/Depth) | Data | Duration |
|-------|---------------------|------|----------|
| 1. Helium pre-train | 3e-4 | 2.1T text tokens | 500K steps |
| 2. Moshi pre-train | 3e-5 / 2e-4 | 7M hours audio | 1M steps |
| 3. Post-training | 3e-6 / 5e-5 | Diarized audio | 100K steps |
| 4. Fisher fine-tune | 2e-6 / 4e-6 | 2000h phone conversations | 10K batches |
| 5. Instruct fine-tune | 2e-6 / 2e-6 | 20K+ hours synthetic | 30K steps |

### Optimizer & Hyperparameters
- **Optimizer:** AdamW throughout
- **Weight decay:** 0.1 (standard), 0.05 for Mimi Transformer params
- **Adam betas:** (0.9, 0.95)
- **Mimi codec training:** AdamW, LR=8e-4, batch=128, 4M steps, pure adversarial training (no reconstruction loss)

### **CRITICAL: Modality Competition Mitigations**
1. **Dual optimizer states:** Separate optimizer states for text-only vs audio batches
2. **LR scaling:** Text embeddings/layer scaling multiplied by 0.75x during audio batches
3. **Text masking:** 30% probability during audio pre-training
4. **Catastrophic forgetting prevention:** Half the batches are text-only data
5. **Semantic-acoustic separation:** Split RVQ prevents quality degradation
6. **Loss weighting:** Semantic audio tokens weighted 100x, acoustic 1x, text full weight
7. **Padding token weight:** Reduced by 50% in cross-entropy loss

### Training Stability Tricks
- Activation checkpointing with FSDP on H100s
- LayerScale in Mimi (init=0.01)
- Weight normalization in SeaNet convolutions
- Cosine LR with linear warmup
- Quantization dropout (50%) for bitrate scalability
- Data augmentation in instruct stage: random gain (-24 to +15 dB), noise (30%), echo (30%), reverb (30%)
- Text delay randomized ±0.6s during pre-training

### Hardware
- 127 DGX nodes = 1016 H100 GPUs

### Data Scale
- Text: 2.1T tokens
- Audio: 7M hours unsupervised + 2000h Fisher + 20K+ hours synthetic
- Total: Massive

---

## 6. AudioPaLM & AudioLM (Google)

### AudioPaLM
**Paper:** [arXiv 2306.12925](https://arxiv.org/abs/2306.12925)

#### Architecture
- **Base:** PaLM-2 (text LLM) with extended embedding matrix
- **Audio tokens:** Added `a` new rows to embedding matrix (from t to t+a)
- **Tokenization options tested:** w2v-BERT (1024 vocab, 25Hz), USM-v1, USM-v2 (with auxiliary ASR loss)
- Audio embeddings freshly initialized, text embeddings from PaLM-2

#### Training
- **Optimizer:** Adafactor with constant LR=5e-5, dropout=0.1
- **PaLM-2 variant:** LR ramp to 1e-4 then exponential decay to 1e-5, dropout=0.2
- **All parameters trained** (no freezing) — they found freezing hurt performance
- Loss masking on inputs; multi-task training across ASR, AST, S2ST, TTS, MT simultaneously

#### Data Scale
- 27-50K hours audio depending on task mixture

### AudioLM
**Paper:** [arXiv 2209.03143](https://arxiv.org/abs/2209.03143)
- Maps audio to discrete tokens via masked language model (for semantics) + neural codec (for acoustics)
- Hierarchical token generation: semantic → coarse acoustic → fine acoustic
- Foundation for AudioPaLM's audio token approach

---

## 7. Whisper (OpenAI)

**Paper:** [arXiv 2212.04356](https://arxiv.org/abs/2212.04356)

### Architecture
- Encoder-decoder Transformer
- Input: 16kHz audio → 80-channel log-mel spectrogram (25ms windows, 10ms stride)
- Multiple sizes: Tiny (39M) to Large-v3 (1.55B)

### Training Details
- **Optimizer:** AdamW
- **Adam betas:** (0.9, 0.999)
- **LR schedule:** Linear warmup (2048 steps) then linear decay to zero
- **Batch size:** 256 segments
- **Duration:** 1M updates (~2-3 epochs)
- **No data augmentation** (except Large-v2: SpecAugment, Stochastic Depth, BPE Dropout)
- FP16, dynamic loss scaling, activation checkpointing

### Training Dynamics Insights
- **Convergence challenge:** Very small LR needed at training start; otherwise loss plateaus quickly
- Large-scale multilingual models particularly prone to early convergence issues
- **Piecewise linear warmup** proposed by OWSM work: increase LR slowly at start, quickly later
- Standard linear warmup requires greatly reducing peak LR or extending warmup, both suboptimal
- Fine-tuning with AdamW (0.9, 0.999), linear scheduler, 500-step warmup is standard practice

### Data Scale
- **680,000 hours** of multilingual, multitask labeled audio from the internet
- Weakly supervised — labels are noisy

---

## 8. GLM-4-Voice (Zhipu AI)

**Paper:** [arXiv 2412.02612](https://arxiv.org/abs/2412.02612)

### Architecture
- **Base LLM:** GLM-4-9B (continued pre-training)
- **Speech tokenizer:** Single-codebook, supervised design
  - Built on Whisper-large-v3 encoder with pooling + VQ layers added
  - 12.5Hz frame rate, 175bps (ultra-low bitrate)
  - Separates semantic and acoustic information
- **Speech decoder:** Flow-matching based model to convert speech tokens → waveform
- **Streaming Thoughts:** Alternates text/speech token generation for low latency

### Training Stages
| Stage | Description | Data Scale |
|-------|-------------|------------|
| 1. Speech-text pre-training | Continued pre-training from GLM-4-9B | 1 trillion tokens |
| 2. Supervised fine-tuning | High-quality multi-turn conversation | Curated conversational data |

### Data Composition
- Unsupervised speech data
- Supervised ASR/TTS data
- Synthesized speech-text interleaved data (from text corpora via text-to-token model)
- Chinese + English bilingual

### Key Design Choices
- Single codebook (vs multi-codebook like SNAC/Mimi) reduces complexity
- Decoupled inference: speech→text→speech pipeline internally
- 175bps ultra-low bitrate enables treating speech almost like text tokens

### Training Details
- Optimizer, LR, weight decay: Not specified in paper
- The 1T token scale is notable — among the largest for speech-text models

---

## 9. Spirit-LM (Meta)

**Paper:** [arXiv 2402.05755](https://arxiv.org/abs/2402.05755) (TACL)

### Architecture
- **Base LLM:** LLaMA 2 (7B), continuously pre-trained
- **Speech tokens:** HuBERT (501 phonetic tokens), deduplicated
- **Two versions:**
  - **Base:** Phonetic HuBERT units only
  - **Expressive:** + pitch tokens + style tokens

### Training Approach
- **Word-level interleaving:** Speech and text randomly interleaved at word boundaries
  - e.g., `[Text]the cat [Speech][tokens] [Text]the mat`
  - Change of modality randomly triggered at word boundaries in aligned corpora
- **No separate stages** — single continuous pre-training with mixed data

### Data Scale
- Text-only: 300B tokens (from LLaMA datasets)
- Speech-only: 460K hours (30B speech tokens)
- Aligned speech+text: 110K hours (7B speech + 1.5B text tokens)
- Total speech: 570K hours
- Training budget: 100K steps / 100B tokens

### Key Findings
- **Equal modality sampling** crucial: tuned weights so model sees each modality (speech, text, speech+text) roughly equally
- **Interleaving is the primary factor** for good speech generation quality
- Outperforms parallel ASR+TTS-only and word-level transcription approaches
- Expressive version preserves emotional/prosodic information

### Optimizer
- Detailed in appendix (not fully specified in main text)
- Generation: temperature=0.8, top_p=0.95

---

## 10. Multi-Modal Gradient Conflict & Modality Competition

### PCGrad: Gradient Surgery for Multi-Task Learning
**Paper:** [arXiv 2001.06782](https://arxiv.org/abs/2001.06782) (NeurIPS 2020)

- **Problem:** Conflicting gradients between tasks (negative cosine similarity)
- **Solution:** Project each task's gradient onto the normal plane of conflicting task gradients
- **Result:** Removes interfering gradient components while preserving beneficial directions
- Model-agnostic, composable with other multi-task methods
- Showed substantial gains in both supervised and RL settings

### Cross-Modality Gradient Harmonization
**Paper:** [arXiv 2211.02077](https://arxiv.org/abs/2211.02077) (NeurIPS 2022)

- **Observation:** Even well-aligned video/audio/text data has ubiquitous gradient conflicts between CMA losses
- **Two techniques:**
  1. **Gradient realignment:** Modify CMA loss gradients so directions agree more
  2. **Gradient-based curriculum learning:** Use gradient conflict as sample noisiness indicator; prioritize less noisy samples
- Applied to VATT pre-training on HowTo100M, scaled to YouTube8M
- Achieved state-of-the-art results

### Modality Competition: What Makes Joint Training Fail
**Paper:** [ICML 2022](https://proceedings.mlr.press/v162/huang22e.html)

- **Theoretical proof:** In late-fusion networks trained with gradient descent, modalities compete and only a subset will be learned
- Uni-modal networks outperform jointly trained multi-modal networks
- Different modalities interfere with each other's gradient updates
- No solution proposed — primarily a diagnostic/theoretical paper

### Recent Work on Modality Imbalance (2024-2025)
- **OGM (On-the-fly Gradient Modulation):** Dynamically changes modality weights during training
- **Multi-Objective Optimization (MOO/MIMO):** Reformulates multi-modal learning as MOO; provides convergence guarantees; up to 20x reduction in computation
- **Adaptive Subnetwork Masking (AMSS):** Uses mutual information rates for modal significance
- **Online Logit Modulation (OLM):** Adaptive logit norm adjustment — attenuates fast-converging modalities, amplifies slow ones
- **Key finding:** Standard weight decay and uniform optimization are insufficient; modality-specific dynamic adjustments are needed

---

## 11. Muon Optimizer

**Paper:** [arXiv 2502.16982](https://arxiv.org/abs/2502.16982) + [Blog](https://kellerjordan.github.io/posts/muon/)

### How It Works
1. Compute standard SGD-momentum update
2. Apply Newton-Schulz iteration (5 steps) to orthogonalize: O_t = (M_t M_t^T)^(-1/2) M_t
3. Polynomial coefficients: (3.4445, -4.7750, 2.0315)
4. Works stably in bfloat16

### Key Differences from AdamW
- AdamW: dynamically adjusted max-of-max norms
- Muon: spectral norm constraint (static Schatten-p norm)
- Orthogonalization increases scale of "rare directions" — prevents low-rank convergence
- Higher SVD entropy in weight matrices (>90% of matrices have higher entropy vs AdamW)

### Scaling Discoveries
- **Weight decay is essential** at scale: without it, weights/outputs RMS grows beyond bf16 range
- **Per-parameter update scaling:** Multiply by √max(A,B) and scale factor 0.2 to match AdamW RMS
- These two tricks enable Muon to work out-of-the-box without hyperparameter tuning

### Performance
- **~2x compute efficiency** vs AdamW (only needs ~52% FLOPs for same performance)
- **Moonlight model:** 3B/16B MoE, 5.7T tokens
  - MMLU: 70.0% (vs Llama3.2-3B: 54.7%)
  - HumanEval: 48.1%
  - GSM8K: 77.4%
- Memory: half of distributed AdamW (single vs dual momentum buffers)

### Practical Tips
- Apply Muon to Q, K, V parameters separately (not combined QKV matrix)
- Use Nesterov momentum
- Keep AdamW for embeddings, LM head, RMSNorm (non-matrix params)
- Add weight decay (standard L2)

### Limitations
- **Optimizer mismatch:** Models pretrained with Muon but fine-tuned with AdamW show suboptimal performance (and vice versa)
- **No multi-modal applications reported** — only text LLM pretraining so far
- Only handles 2D matrix parameters; scalars/vectors still use AdamW

---

## Cross-Cutting Insights for Autoresearch

### Common Optimizer Choices
| Model | Optimizer | Weight Decay |
|-------|-----------|-------------|
| Mini-Omni | Likely AdamW | Not specified |
| SpeechGPT | AdamW | 0.01 |
| Moshi | AdamW | 0.1 (main), 0.05 (Mimi) |
| AudioPaLM | Adafactor | Not specified |
| Whisper | AdamW | Not specified |
| GLM-4-Voice | Not specified | Not specified |
| Spirit-LM | Not specified (likely AdamW per LLaMA) | Not specified |

### Training Stage Patterns
Nearly all models use 2-5 stage training:
1. **Freeze LLM, train adapters/connectors** (alignment stage)
2. **Freeze adapters, train LLM** (adaptation stage)
3. **Unfreeze everything** (fine-tuning stage)

Exception: AudioPaLM found that freezing anything hurt performance. Spirit-LM uses single continuous pre-training.

### Modality Competition Mitigations (from Moshi — most thorough)
1. Dual optimizer states for text vs audio batches
2. LR scaling (0.75x) for text during audio batches
3. 50% text-only batches to prevent catastrophic forgetting
4. Loss weighting: semantic tokens 100x, acoustic 1x
5. Text masking (30%) during audio training

### Relevance to Our Project (Mini-Omni / Qwen2-0.5B)
- Our coupled WD + WD=0.05 finding aligns with Moshi using WD=0.05 for codec params
- Moshi's dual optimizer approach could inform our gradient conflict work
- PCGrad/gradient harmonization could be applied to text vs audio loss conflicts
- Muon is promising but untested for multi-modal — potential novel contribution
- GLM-4-Voice's ultra-low bitrate single codebook is an interesting alternative to SNAC's 7 codebooks
- The modality competition literature confirms our observation that text_ratio>=0.7 is needed
