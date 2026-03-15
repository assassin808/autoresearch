# Basin Shaping in Omni-Model Training

## Thesis

Cross-modal noise (audio gradients) implicitly performs Gaussian-Optimization-style
basin widening on the text loss landscape. The training dynamics have three phases
characterized by the gradient norm ratio ρ(t) = λ‖∇L_audio‖ / ‖∇L_text‖:
- **Early (ρ≪1):** audio is noise → basin widening (beneficial)
- **Middle (ρ~1):** modality competition → text degradation (harmful)
- **Late (ρ≫1):** text frozen, audio converges (neutral)

## Two-Stage Research Plan

### Stage 1: Observation (Replicate + Measure)

**Goal:** Faithfully replicate mini-omni S2 training with full diagnostics.

#### 1a. Data Pipeline (exact mini-omni replication)

**Data source:** VoiceAssistant-400K (470K QA pairs, 42GB cached)
- Fields: `question`, `question_audio`, `answer`, `answer_snac`
- SNAC format: `# t1 t2 t3 t4 t5 t6 t7 # t1 ...` (7 tokens per frame, ~80 frames/answer)
- Audio: 24kHz, embedded in parquet

**4 task types (from inference.py):**

| Task | Input | Output | Whisper? | Text stream | Audio streams |
|------|-------|--------|----------|-------------|---------------|
| T1T2 | text question | text answer | No | `[_input_t] + q_tokens + [_eot, _answer_t] + a_tokens` | `[_pad_a]*N` |
| T1A2 | text question | audio answer | No | `[_input_t] + q_tokens + [_eot, _answer_t] + [_pad_t]*M` | `[_pad_a]*N + [_answer_a] + snac_tokens` |
| A1T2 | audio question | text answer | Yes | `[_input_t, _pad_t*T, _eot, _answer_t] + a_tokens` | `[_input_a, _pad_a*T, _eoa, _answer_a] + [_pad_a]*M` |
| A1A2 | audio question | audio answer | Yes | `[_input_t, _pad_t*T, _eot, _answer_t] + [_pad_t]*M` | `[_input_a, _pad_a*T, _eoa, _answer_a] + snac_tokens` |

**Token encoding:**
- Text: Qwen2 tokenizer (vocab=151,936 + 64 special = 152,000)
- Audio stream i: `layershift(token, i) = token + 152000 + i*4160`
- Total padded vocab: 152,000 + 7×4,160 = 181,120

**8-stream format:** Each sample is 8 parallel sequences fed to `wte()`, then averaged.

#### 1b. Model (exact mini-omni architecture)

- Qwen2-0.5B: 24 layers, dim=896, 14 heads (2 KV groups), GQA
- Embedding: `wte` (181,120 × 896), shared across all 8 streams
- Forward: `x = (wte(s0) + wte(s1) + ... + wte(s7)) / 8` → transformer → `lm_head`
- Output: `xt` (text logits, first 152K), `xa[0..6]` (7 audio logit heads, 4160 each)
- Whisper adapter: LLaMA-MLP (768 → 2432 → 896, SiLU gated)
- **No post_adapter** in public release

**S2 frozen:** whisper_adapter (and Whisper encoder, SNAC codec — external)
**S2 trained:** transformer.wte, transformer.h[0..23], transformer.ln_f, lm_head

#### 1c. Their optimizer (replicate exactly)

- **Optimizer:** AdamW
- **LR:** cosine annealing, min=4e-6, max=4e-4
- **Batch:** 192
- **Steps:** 40,000 per epoch
- **Hardware:** 8×A100

We'll scale down proportionally for 1×RTX 5090:
- Batch: 4 (with gradient accumulation to effective ~32-64)
- Steps: ~5,000-10,000 (5-10 min runs for observation, then longer)
- LR: same schedule (cosine 4e-6 → 4e-4)

#### 1d. Diagnostics (measured every N steps)

**D1. Gradient norm ratio ρ(t):**
```
ρ(t) = λ · ‖∇L_audio(θ_t)‖ / ‖∇L_text(θ_t)‖
```
Per-component: backbone, embedding, lm_head

**D2. Gradient interference cos φ(t):**
```
cos φ = ⟨∇L_text, ∇L_audio⟩ / (‖∇L_text‖ · ‖∇L_audio‖)
```
Per-layer in backbone. Also decompose into: attention Q/K/V/O, MLP fc1/fc2/proj

**D3. Per-modality loss velocity:**
```
L̇_text(t) = dL_text/dt,  L̇_audio(t) = dL_audio/dt
```
Also per-codebook: L̇_CB0, L̇_CB1, ..., L̇_CB6

**D4. Basin width probing (every 500 steps):**
Sample M=10 random directions, perturb θ by ε∈{0.01, 0.05, 0.1, 0.5, 1.0},
measure text loss. Record radius where L_text degrades by 10%.

**D5. Parameter displacement:**
```
d(t) = ‖θ_t - θ_0‖  (L2 norm of total drift)
```
Also decompose: d_backbone, d_embedding, d_lm_head

**D6. Embedding effective rank (from our prior work):**
Track Shannon entropy of singular values for text vs audio embedding subspaces.

**D7. Per-codebook loss & gradient norms:**
CB0 (coarse) vs CB6 (fine) — do they have different dynamics?

#### 1e. Experiments for Stage 1

| Exp | Config | Purpose |
|-----|--------|---------|
| obs_1 | Adam baseline, exact mini-omni HPs, 10min | Replicate + measure D1-D7 |
| obs_2 | Same but λ=0 (text only) | Pure text baseline for basin comparison |
| obs_3 | Same but λ=0.5 (equal text+audio weight) | Higher audio weight |
| obs_4 | Same but λ=2.0 (audio-heavy) | Extreme audio weight |
| obs_5 | Same but audio frozen (random SNAC, no audio loss) | Audio-as-pure-noise |

**Key questions answered by Stage 1:**
1. Do the three ρ-phases exist in practice?
2. Does basin width (D4) differ between obs_1 and obs_2?
3. Does cos φ go negative during the middle phase?
4. Which codebook (CB0-CB6) has the most gradient conflict?
5. Does obs_5 (audio-as-noise) actually widen the text basin?

### Stage 2: Methodology Validation

Based on Stage 1 observations, implement and test:

#### Method A: Adaptive λ scheduling
```
λ_{t+1} = λ_t + α · sign(ρ_t - 1) · (1 - |ρ_t - 1|)₊
```
Pushes λ to keep ρ away from 1. Completely first-order, ~free.

#### Method B: Audio-GO (structured perturbation)
```
θ_{t+1} = θ_t - η · ∇L_text(θ_t + σ · ĝ_audio)
```
Phase 1: text trains, audio provides perturbation direction (basin shaping).
Phase 2: switch to standard omni-training. Transition when basin width plateaus.

#### Method C: Gradient projection
```
g'_audio = g_audio - min(0, cos φ) · ⟨g_audio, g_text⟩/‖g_text‖² · g_text
```
Remove interfering component only when cos φ < 0. Cost: one dot product.

#### Method D: Two-timescale optimizer
```
η_text(t) = η₀·(1-t/T),  η_audio(t) = η₀·(t/T)
```
Linear handoff + gradient projection during crossover.

#### Experiments for Stage 2

| Exp | Method | Duration | Purpose |
|-----|--------|----------|---------|
| met_A1 | Adaptive λ, α=0.01 | 10min | Test ρ control |
| met_A2 | Adaptive λ, α=0.1 | 10min | More aggressive |
| met_B1 | Audio-GO, σ=0.1 | 10min | Structured perturbation |
| met_B2 | Audio-GO, σ=1.0 | 10min | Strong perturbation |
| met_C1 | Gradient projection | 10min | Interference removal |
| met_D1 | Two-timescale | 10min | Phase skip |
| met_D2 | Two-timescale + projection | 10min | Combined |
| met_best | Best from above | 20min | Scaling validation |

#### Success metrics

| Metric | Adam baseline (obs_1) | Target |
|--------|----------------------|--------|
| val_loss | TBD | <baseline |
| text_bpb | TBD | ≤baseline (don't regress) |
| audio_loss | TBD | <baseline |
| Basin width | TBD | >baseline (wider basin) |
| ρ~1 duration | TBD | shorter (faster transition) |

## Paper Contribution

**Main claim:** "In omni-model training, cross-modal gradients implicitly perform
basin shaping on the dominant modality's loss landscape. Understanding and controlling
the gradient norm ratio ρ(t) — via adaptive scheduling, structured perturbation, or
gradient projection — eliminates the catastrophic middle phase and yields better
final models with simple, first-order methods."

**Novel:** First measurement of basin geometry during multi-modal training.
First application of GO-style analysis to explain why multi-modal training sometimes
helps and sometimes hurts the dominant modality.
