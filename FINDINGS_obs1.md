# Obs_1 Findings: S2 Baseline Diagnostics (3000 steps)

## Setup
- Model: Qwen2-0.5B mini-omni architecture (694M params, 24 layers)
- Data: VoiceAssistant-400K (49K train, 1K val, 4 task types)
- Optimizer: AdamW, cosine LR 4e-6→4e-4, WD=0.01, batch=32 effective
- Duration: 8102s (~2.25h), 3000 optimizer steps

## D1: Gradient Norm Ratio ρ(t)

| Phase | Steps | ρ range | Interpretation |
|-------|-------|---------|----------------|
| Early | 100-600 | 0.04-0.09 | Text dominates 10-25× |
| Transition | 700-900 | 0.15-0.31 | Audio strengthening |
| Plateau | 1000-3000 | 0.14-0.28 | Stable, never reaches 1.0 |

**Key finding:** ρ never approaches 1.0 in 3000 steps. The predicted "middle phase"
(ρ~1, modality competition) does not appear. Mini-omni trains for 40K steps/epoch —
the middle phase likely emerges later. Our 3000 steps covers only the early phase.

## D2: Gradient Interference

- Global cos φ: mean=-0.006, range [-0.03, +0.03]
- Modalities are **mostly orthogonal**, not conflicting
- **Layer 0** (embedding-adjacent) has most conflict: 73% negative cos φ
- **Layer 23** (output-adjacent) also conflicted: 70% negative cos φ
- Middle layers (4-20) are more cooperative

**Implication:** Gradient interference is concentrated at the input/output boundaries,
where text and audio representations most directly overlap.

## D4: Basin Width

| Step | ε=0.01 | ε=0.1 | ε=0.5 | ε=1.0 |
|------|--------|-------|-------|-------|
| 1000 | +4.3   | +96.8 | +495  | +1004 |
| 2000 | +5.7   | +96.3 | +494  | +1010 |
| 3000 | +7.1   | +98.2 | +496  | +1000 |

**Basin width is remarkably stable.** At ε=0.1, degradation is ~97 across all three
checkpoints. This means:
1. Audio gradients are NOT shrinking the text basin (contradicts naive expectation)
2. The text loss landscape geometry is stable during early training
3. Need obs_2 (text-only) to compare if audio WIDENS the basin

## D6: Embedding Rank

- Text: 844→719 (Δ=-125) — large restructuring, highly variable
- Audio: 883→838 (Δ=-45) — steady monotonic decline, no recovery

**Audio rank collapse confirmed:** At 1.5 dims/100 steps, rank drops linearly.
If this continues to 40K steps, audio rank would reach ~283 (out of 896 dims) —
severe collapse.

## D7: Per-Codebook Dynamics

| Codebook | Start | End | Δ | Learning rate |
|----------|-------|-----|---|---------------|
| CB0 (coarse) | 6.2 | 4.4 | -1.8 | Fastest |
| CB1 | 7.0 | 5.3 | -1.7 | |
| CB4 | 7.0 | 6.0 | -1.0 | |
| CB6 (fine) | 8.1 | 7.6 | -0.5 | Slowest |

**Coarse codebook learns ~3.6× faster than fine.** This aligns with Moshi's finding
that semantic tokens should be weighted higher (100×) than acoustic tokens.

## Implications for Methods

1. **Method A (adaptive λ):** May not be needed — ρ naturally stays below 0.3 in early training
2. **Method B (Audio-GO):** Still worth testing — basin width is stable but might be wider with structured noise
3. **Method C (gradient projection):** Limited value — cos φ is near zero, minimal conflict
4. **Method D (two-timescale):** Could accelerate the ρ transition to skip the slow early phase

**Revised hypothesis:** The three-phase model is correct but operates on a much longer
timescale than expected. At 3000 steps (7.5% of mini-omni's 40K), we only see the
early phase. The middle phase (ρ~1) likely appears around step 10K-20K.
