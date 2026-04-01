# Speaker Script: Audio Plateau in Omni-Modal Joint Training

Total time: ~8-10 minutes. Teleprompter style -- speak naturally, do not read verbatim.

---

### Slide 1: Title

Good afternoon. Today I want to walk you through our investigation of a specific training failure mode in omni-modal pretraining. We are training a joint text-plus-audio model and the core question is: **why does audio loss plateau while text loss converges normally?** We ran 29 experiments, tracked 15 diagnostics, and accumulated over 30,000 training steps to answer this. Let me start with some background on the system.

---

### Slide 2: Background -- Mini-Omni Architecture and Training

The model is Mini-Omni, built on a Qwen2-0.5B language model backbone with about 500 million parameters. It has two key additions: a Whisper speech encoder on the input side, and a **SNAC codec** -- that stands for Scalable Neural Audio Codec -- on the output side. SNAC produces 7 parallel codebook streams at 24 kHz, each with its own 4,160-token vocabulary, for a total of roughly 181,000 audio tokens sharing a single unified embedding table with the text tokens.

Training happens in three stages. **S1** trains only the Whisper adapter with the language model frozen. **S2** unfreezes the language model backbone but keeps the adapter frozen. **S3** unfreezes everything and trains on all four task combinations: text-to-text, text-to-audio, audio-to-text, and audio-to-audio. Our investigation focuses entirely on S3, where the audio plateau appears.

Now let me show you the problem.

---

### Slide 3: The Problem

Here is the problem in two charts. Both show **training loss**. On the left, text training loss drops from 9 down to 1.44 -- a 77% reduction in just 3,000 steps. Rapid, clean convergence. On the right, audio training loss starts at 58, which is near the random baseline, drops to about 37, and then just stops. That is only a **14% reduction**.

Two diagnostic numbers capture the severity. First, the audio **GSNR** -- that is the Gradient Signal-to-Noise Ratio -- measures just 0.08, meaning the gradient signal is 12.5 times weaker than the noise. Second, the cosine similarity between text and audio gradients, which we call **cos phi**, is exactly zero. Not negative, not conflicting -- just orthogonal. So the question is: is this an optimization failure, a gradient conflict, or something more fundamental?

Let me start with the answer.

---

### Slide 4: The Verdict

The good news is the plateau does break -- it just takes a long time. This chart shows the **validation audio-to-audio loss** over extended training. Starting from 49.1 at 3,000 steps, it drops to 32.3 at 16,500 steps with the chained run at effective batch 192. That is a **34% improvement** in validation loss. The CB0 top-10 prediction accuracy more than doubles, going from 20.7% to 44.4%. So audio is not stuck permanently. It is learning, just roughly **12 times slower than text**. The rest of this talk explains why, and what we can do about it.

Now let me walk through the six key findings.

---

### Slide 5: Finding 1 -- Orthogonal, Not Conflicting

Our first and perhaps most surprising finding is that text and audio gradients are not conflicting at all. The cosine similarity between them -- **cos phi** -- averages 0.004 plus or minus 0.02 across all 29 experiments and 30,000 steps. They are **orthogonal, not antagonistic**. We also projected audio gradients into the text gradient subspace using diagnostic D15 and found only 0.1% of energy there, compared to a random baseline of 0.00005%.

Why orthogonal? Because text embeddings are pretrained and structured, while audio embeddings start as random Gaussian noise. They live in completely different subspaces. The backbone features align with text representations, and audio simply cannot access them yet.

This is why PCGrad and gradient projection methods had zero effect. **There is no conflict to resolve.** The problem lies elsewhere.

Let me show you what the real problem is.

---

### Slide 6: Finding 2 -- GSNR Theory

The **GSNR** framework from McCandlish et al. gives us a quantitative handle on the problem. The formula is on the slide: GSNR at batch size B equals B times the squared gradient norm divided by the trace of the gradient covariance. This defines a critical batch size, **B-crit**, where noise equals signal. For audio, we measured GSNR of 0.08 at batch size 2, giving a B-crit of about 25 for audio versus only 2 for text. The optimal batch size, B-star, is approximately 9.

Now look at the controlled comparison in the table. Experiment 16 used effective batch 32 for 30,000 steps and reached validation loss 36.8. Experiment 15 used effective batch 192 for only 5,000 steps -- the same total number of samples seen -- and only reached 44.6. **More noisy steps beats fewer clean steps.** The effective batch 192 run wastes 81% of its compute relative to the optimal batch. This tells us the path forward is more steps at moderate batch sizes, not bigger batches.

This next finding tells us where the bottleneck actually lives.

---

### Slide 7: Finding 3 -- CKA -- Two Paths, Same Destination

**CKA** stands for Centered Kernel Alignment, and it measures how similar two sets of neural representations are. We compared two training paths and measured CKA similarity to the initial checkpoint at the middle layer. Direct S3 training completely reshapes the backbone -- CKA drops to 0.03, meaning the representations become almost unrecognizable. S2-to-S3 training keeps the backbone nearly frozen with CKA at 0.99. And yet, **both paths reach the same validation performance**: 44.9 versus 44.8.

The audio gradients even flow differently between the two paths. In direct S3, they propagate from the output layer downward. In S2-to-S3, they concentrate in layers 0 through 2, which act as **emergent adapters** -- spontaneously forming without being designed that way. The backbone is just a feature extractor. Reshape it or not, you get the same result. The bottleneck is in the embedding space, not the transformer layers.

So what does S2 pretraining actually contribute? The next finding answers that.

---

### Slide 8: Finding 4 -- Loss Landscape and Basin Width

We probed the loss landscape by adding small perturbations at epsilon equals 0.01 and measuring how much the loss degrades. The S2-to-S3 path starts in a **flat basin** with degradation around 40 and widens 4 times to about 163 over training. Direct S3 starts in a **sharp basin** at 336 and actually narrows to 204. So S2 pretraining gives you a much smoother starting point. This is its real contribution -- not better final accuracy, but a **more stable optimization trajectory**. In a flatter basin, noisy gradients are less likely to push the model into bad regions.

Now let me look at how the gradients themselves evolve during training.

---

### Slide 9: Finding 5 -- Gradient Dynamics

Here is a paradox. The left chart shows **rho** -- that is the ratio of audio gradient norms to text gradient norms -- rising from 0.6 to 2.6 over 30,000 steps. Audio gradients are getting about 4 times larger in magnitude. But the right chart shows GSNR staying completely flat at 0.08 across the entire training chain. **The gradients get louder, but they do not get clearer.** The magnitude increases because as text converges, the model finds audio relatively harder. But the signal-to-noise ratio does not improve at all. Different samples still push the audio parameters in contradictory directions, canceling each other out.

This means learning rate scaling will not help. You cannot fix a noise problem by amplifying the noise.

The last finding explains why the noise stays so high.

---

### Slide 10: Finding 6 -- Backbone Memorizes, Embeddings Learn Slowly

We trained linear probes on the backbone's intermediate representations to predict audio tokens. At layer 18, train accuracy reaches 77%, but validation accuracy is just 6.2% -- a **12x gap**. The backbone is memorizing audio patterns on the training set but cannot generalize. This is the chicken-and-egg problem: embeddings start random, so the backbone cannot learn stable audio rules, which means gradients stay noisy at GSNR 0.08, which means embeddings update slowly.

But they do learn. **CB0 top-10 accuracy** -- that is the prediction accuracy for the first codebook -- improves from 20.7% at 3,000 steps to 44.4% at 16,500 steps, a 114% improvement. Structure is forming in the embedding space, just very slowly. The embeddings need many more steps to build the kind of structure that text embeddings inherited from pretraining.

Let me pull all of this together.

---

### Slide 11: Key Insights

Six main takeaways. **One**: text and audio gradients are orthogonal, not conflicting -- gradient surgery methods are the wrong tool. **Two**: the root cause is low audio GSNR at 0.08; audio needs 12 times more samples than text per unit of learning. **Three**: the plateau does break with sufficient training -- we saw a 34% improvement in validation loss over 16,500 steps. **Four**: S2 pretraining creates a flatter basin that stabilizes optimization but does not improve the final loss. **Five**: the backbone is not the bottleneck -- CKA 0.03 and CKA 0.99 reach the same loss; the bottleneck is embedding-level GSNR. **Six**: audio gradient magnitude grows 4x but GSNR stays flat, so learning rate tricks cannot fix this.

The common thread is clear: **this is a noise-dominated learning regime**.

---

### Slide 12: Ongoing, Next Steps, and References

We are currently running a chained training experiment with the full Mini-Omni pipeline. Rounds 1 through 3 are complete at 16,500 global steps, with rounds 4 through 6 queued to reach 27,500 steps. The validation audio loss trajectory is still declining: 29.9 to 25.6 to 23.9, with no convergence yet.

Going forward, we propose four directions. First, optimal task sampling with 22% text and 78% audio, which B-crit theory predicts should improve sample efficiency. Second, LR scaling at 7x for large-batch runs to match small-batch efficiency. Third, directly measuring text GSNR, which we currently only estimate. Fourth, pushing training to 50,000-plus steps since audio is still improving.

**The bottom line**: the audio plateau is a noise-dominated learning regime with GSNR of 0.08. It is not an optimization or architecture failure. It breaks with sufficient training, and the critical batch size framework gives us actionable guidance for compute-optimal scheduling. Thank you.

---
