# Meeting Notes — April 1, 2026

## Attendees
Discussed with collaborator after presentation review.

---

## Key Directions Agreed

### Direction 1: Isolate the Source of the Problem

**Per-layer GSNR analysis**: Our current GSNR is measured globally. Need to decompose:
- Which layers have high/low GSNR?
- Is the bottleneck in embedding (wte), shallow layers (0-2), deep layers (23), or lm_head?
- This tells us WHERE the noise comes from, not just that it exists.

**TTS adapter isolation**:
- Step 1: Train with `post_adapter=True` (6 extra transformer layers + independent audio lm_head)
- Step 2: Compare GSNR with vs without TTS adapter
- If GSNR improves → the shared lm_head was the bottleneck
- If GSNR doesn't improve → the problem is deeper (embedding input side)

**Gradient clipping / batch size effects**: Examine whether gradient clipping interacts with the GSNR measurement. Large audio gradients (rho=2.6) might be clipped, artificially reducing effective signal.

### Direction 2: Data-Level Difficulty Analysis

**Core question**: Is the audio plateau caused by (a) embedding/architecture, or (b) the data itself being inherently harder to learn?

**Approach — Oracle models + clustering**:
1. Train a **pure audio model** (audio-only, no text) — this is the audio "oracle"
2. Train a **pure text model** (text-only, already have exp1/exp11) — text "oracle"
3. For each oracle, extract internal representations and perform **unsupervised clustering**
4. Identify **invariant features** — clusters that are stable, well-separated
5. These invariants define the "learnable structure" in each modality

**Pseudo-labels from clustering**:
- Cluster audio representations → pseudo-label audio tokens by acoustic/semantic similarity
- Cluster text representations → pseudo-label text tokens by semantic similarity
- These pseudo-labels can measure "how structured" each modality's representation is

**Perplexity as difficulty metric**:
- Per-sample perplexity from oracle models = difficulty score
- High perplexity samples = hard to learn
- Can stratify training data by difficulty → curriculum insights

### Direction 3: Cross-Modal Invariant Analysis

**How do text and audio representations relate?**
- In text oracle: cluster features at each layer → identify text invariants
- In audio oracle: cluster features at each layer → identify audio invariants
- Question: do text and audio share any invariant structure? At which layer?
- If shared invariants exist → these are the "bridge" features for omni training
- If no shared invariants → explains why cos_phi ≈ 0 (orthogonal)

**Linking clusters across modalities**:
- For paired data (same QA pair has text answer + audio answer):
  - Text oracle cluster ID for text answer
  - Audio oracle cluster ID for audio answer
  - Are they correlated? → Cross-modal alignment
- This creates a "correspondence map" between text and audio representation spaces

### Direction 4: Dynamics During Omni Training

**Track invariant collapse during S3 training**:
- Define clusters from oracle models (fixed reference)
- During omni (S3) training, at each checkpoint:
  - Extract representations, assign to oracle clusters
  - Measure: do clusters remain separated? Or do they collapse?
  - **Faster collapse = harder to learn** (the model is destroying structure)
  - **Stable clusters = easy to learn** (structure is maintained)

**Per-layer cluster dynamics**:
- At which layer do audio clusters collapse first?
- At which layer do text clusters remain stable?
- This gives a **layer-by-layer difficulty map** for each modality

**Connection to GSNR**:
- If clusters collapse → representations become less distinguishable → gradients become noisier → GSNR drops
- This would provide a **causal chain**: data difficulty → cluster collapse → low GSNR → slow learning

---

## Summary: Four-Part Research Plan

```
1. IDENTIFY SOURCE
   - Per-layer GSNR decomposition
   - TTS adapter isolation experiment
   - Gradient clipping interaction

2. SINGLE-MODALITY ORACLES
   - Pure audio model → cluster → invariants → pseudo-labels
   - Pure text model → cluster → invariants → pseudo-labels
   - Per-sample perplexity as difficulty metric

3. CROSS-MODAL LINKING
   - How do text/audio clusters relate?
   - Shared invariants? Correspondence map?
   - Where in the network do they align (if ever)?

4. TRAINING DYNAMICS
   - Track cluster stability during omni training
   - Which clusters collapse? → defines "hard to learn"
   - Layer-by-layer collapse speed → difficulty map
   - Connect to GSNR: collapse → noisy gradients → slow learning
```

---

## Concrete Next Experiments (after April 7 maintenance)

| Priority | Experiment | Purpose |
|----------|-----------|---------|
| P0 | Per-layer GSNR (decompose D14 by module) | Find WHERE noise comes from |
| P0 | Train with post_adapter=True | Test if TTS adapter fixes GSNR |
| P1 | Pure audio-only model (no text tasks) | Audio oracle for clustering |
| P1 | Extract representations + k-means clustering | Define invariants |
| P2 | Track cluster collapse during S3 training | Difficulty dynamics |
| P2 | Per-sample perplexity from oracles | Difficulty scoring |
| P3 | Cross-modal cluster correspondence | Text-audio linking |

---

## Key Insight from Discussion

> The question is not just "why is audio slow" but "what makes certain data hard to learn in a shared representation space." If we can define difficulty through cluster invariants and their collapse dynamics, this framework applies beyond audio — to any new modality added to an LLM.

This shifts the contribution from "diagnosing mini-omni" to "a general framework for understanding multi-modal learning difficulty."
