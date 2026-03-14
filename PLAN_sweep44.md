# Sweep 44+ Plan: Omni-Optimizer for Real S2 Training

## 1. Mini-Omni's Real S2 Training (What We Now Know)

### Architecture (confirmed from reference code)
- Qwen2-0.5B: 24 layers, dim=896, 14 heads, GQA (2 groups)
- 8 parallel streams: 7 SNAC codebooks + 1 text
- Streams are **averaged** (not concatenated): `x = (x0+x1+...+x7)/8`
- Vocab: 152,000 text + 7×4,160 audio = 181,120 total
- Whisper adapter: LLaMA-MLP (768→896), frozen in S2

### Their training setup
- Optimizer: AdamW, LR cosine 4e-6→4e-4, batch=192, 40K steps/epoch
- S2: freeze adapters, train LLM backbone + embeddings
- Data: VoiceAssistant-400K + 8Kh speech + 2M text QA
- Loss: equal-weighted text + audio CE (audio averaged over 7 codebooks)

### Our reimplementation gaps
1. **Data scale**: We use 215h audio. They use 8,000h+ audio + 400K QA pairs
2. **Batch size**: We use 4. They use 192
3. **Steps**: We run ~5min. They run 40K+ steps
4. **No loss weighting tricks**: Moshi uses 100x semantic token weight

## 2. Key Training Dynamics Issues in Omni Models

From literature + our 390 experiments:

### Issue A: Modality Competition (confirmed by theory + Moshi)
- Text and audio gradients conflict in shared backbone
- Audio tokens get ~15x less gradient signal per token
- Only solved at scale by Moshi with 5 separate mitigations
- Our finding: gradient manipulation (PCGrad, etc.) all failed

### Issue B: Embedding Rank Collapse (our finding)
- Audio embedding effective rank: 509→434 (monotonic decline)
- 246 dead audio tokens receive zero gradient but get weight-decayed
- Text embedding rank recovers; audio never does

### Issue C: 8-Stream Averaging Dilution
- Mini-omni averages 8 streams: each stream contributes only 1/8
- Audio streams (7/8) dominate the input but text stream carries most semantic info
- This creates an inherent representation bottleneck

### Issue D: Codebook Hierarchy Ignored by Optimizer
- SNAC L0 (coarse): sparse, 6% dead codes, most important for audio quality
- SNAC L2 (fine): dense, all codes active, refinement only
- Adam treats all 7 codebook losses equally — no hierarchy awareness

### Issue E: LR-WD Coupling in Warmdown (our confirmed finding)
- Constant WD during LR decay over-regularizes rare tokens
- Coupled WD (WD decays with LR) gives consistent improvement

## 3. Optimizer Design: OmniMuon

### Core idea
A Muon-based optimizer designed specifically for omni-model training dynamics.
Novel contribution: **first application of Muon to multi-modal text+audio models**.

### Design principles (from our 390 experiments + literature)

#### P1: Modality-Aware Weight Decay
- Text embeddings: WD=0 (always optimal in our experiments)
- Audio embeddings: WD proportional to 1/sqrt(token_frequency)
- Dead audio tokens: WD=0 (no point decaying unused codes)
- Transformer matrices: mild WD (0.05), coupled with LR schedule

#### P2: Hierarchical Codebook Loss Weighting (inspired by Moshi)
- Moshi: semantic tokens 100x, acoustic 1x
- SNAC equivalent: L0 (coarse/semantic) weighted higher than L2 (fine/acoustic)
- Proposed: L0=3x, L1=1.5x, L2=1x (start mild, can sweep)

#### P3: Muon for Backbone, Modality-Aware Adam for Embeddings
- Backbone (2D matrices): Muon with NS orthogonalization
  - NS already equalizes gradient directions → modality-blind by design
  - This is GOOD: backbone should learn shared representations
- Embeddings/lm_head: Adam with per-modality hyperparameters
  - Text portion: standard Adam, no WD, moderate LR
  - Audio portion: Adam with frequency-aware WD, possibly higher LR

#### P4: Momentum Scheduling
- Start momentum high (0.97) for fast early convergence
- Decay to 0.95 for stable late training
- Our sweep29 showed momentum's effect reverses with duration

#### P5: Batch Composition Control (inspired by Moshi)
- Moshi: 50% text-only batches to prevent forgetting
- We control via audio_ratio parameter
- Optimal: text_ratio=0.7-0.8 (confirmed both by us and Moshi's approach)

## 4. Experiment Plan

### Phase 1: Baseline Validation on Real Architecture (sweep 44)
Run on actual Qwen2-0.5B mini-omni architecture with our best optimizer:

| Exp | Config | Purpose |
|-----|--------|---------|
| 1 | Adam baseline (their settings) | Reproduce mini-omni S2 |
| 2 | Our best recipe (batch64K, LR0.03, WD0.05, coupledWD) | Transfer test |
| 3 | Muon backbone + Adam embed (MuonHybrid from sweep43) | Muon transfer |
| 4 | Exp 3 + per-modality WD (text=0, audio=freq-aware) | Best of both |

### Phase 2: Codebook-Hierarchical Loss (sweep 45)
| Exp | L0 weight | L1 weight | L2 weight | Purpose |
|-----|-----------|-----------|-----------|---------|
| 1 | 1.0 | 1.0 | 1.0 | Baseline (equal) |
| 2 | 3.0 | 1.5 | 1.0 | Hierarchy-aware |
| 3 | 5.0 | 2.0 | 1.0 | Aggressive hierarchy |
| 4 | 10.0 | 3.0 | 1.0 | Moshi-inspired (100x=too much) |

### Phase 3: Dead Token Handling (sweep 46)
| Exp | Method | Purpose |
|-----|--------|---------|
| 1 | Baseline (WD on all tokens) | Control |
| 2 | WD=0 for dead audio tokens | Remove noise from unused codes |
| 3 | Reinitialize dead tokens periodically | Recycle dead codes |
| 4 | Merge dead tokens into nearest active | Compress codebook |

### Phase 4: Full OmniMuon (sweep 47)
Combine best from phases 1-3:
- MuonHybrid backbone
- Per-modality WD
- Codebook-hierarchical loss
- Dead token handling
- Momentum schedule
- Coupled WD

### Phase 5: Scaling Validation (sweep 48)
- Run best OmniMuon at 10min, 20min
- Compare gap vs Adam baseline at each duration
- Confirm improvements compound (our earlier finding)

## 5. Success Metrics

| Metric | Adam Baseline | Target (OmniMuon) |
|--------|--------------|-------------------|
| val_loss | ~3.5 (5min) | <3.4 (>3% improvement) |
| text_bpb | ~1.09 | ≤1.09 (don't regress) |
| audio_loss | ~5.5 | <5.3 (>4% improvement) |
| Training stability | σ≈0.005 | σ<0.003 |

## 6. What to Delete for Disk Space

Safe to delete now:
- `/home/dev/workspace` (14G) — old duplicate
- `/workspace/.hf_home/hub/models--kernels-community--flash-attn3` (768M) — FA3 broken on Blackwell

Keep:
- VoiceAssistant-400K (40G) — needed for scaling experiments
- HuggingFace model cache (954M) — Qwen2 weights needed
- autoresearch data cache (963M) — training data
