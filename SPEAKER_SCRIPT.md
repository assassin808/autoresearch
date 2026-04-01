# Speaker Script: Audio Plateau in Omni-Modal Joint Training

Total time: ~8-10 minutes. Read naturally, not word-for-word.

---

### Slide 1: Title

Good afternoon. Today I want to walk you through our investigation of a specific training failure mode we encountered in omni-modal pretraining. We are training a joint text-plus-audio model based on Mini-Omni -- a Qwen2-0.5B backbone with a Whisper encoder and SNAC audio codec producing 7 codebook streams. The core question is simple: why does **audio loss plateau** while text loss converges normally? We ran 29 experiments, tracked 15 diagnostics, and accumulated over 30,000 training steps to answer this. Let me show you what we found.

---

### Slide 2: The Problem

Here is the problem in two charts. On the left, text loss drops from 9 down to 1.44, a 77% reduction in just 3,000 steps. Rapid, clean convergence. On the right, audio loss starts at 58 -- which is close to the random baseline -- drops to about 37, and then just stops. That is only a **14% reduction**. Two numbers capture the severity. First, the audio GSNR is 0.08, meaning the gradient signal is 12.5 times weaker than the noise. Second, the cosine similarity between text and audio gradients is exactly zero. Not negative, not conflicting -- just orthogonal. So the question becomes: is this an optimization failure, a gradient conflict, or something more fundamental?

Let me start with the answer.

---

### Slide 3: The Verdict

The good news is the plateau does break -- it just takes a long time. This chart shows the validation audio loss over extended training. Starting from 49.1 at 3,000 steps, it drops to 32.3 at 16,500 steps with the chained run at effective batch 192. That is a **34% improvement**. The CB0 top-10 prediction accuracy more than doubles, going from 20.7% to 44.4%. So audio is not stuck permanently. It is learning, just roughly **12 times slower than text**. The rest of this talk explains why, and what we can do about it.

Now let me walk through the six key findings.

---

### Slide 4: Finding 1 -- Gradient Orthogonality

Our first and perhaps most surprising finding is that text and audio gradients are not conflicting at all. The cosine similarity between them averages 0.004 plus or minus 0.02 across all 29 experiments and 30,000 steps. They are **orthogonal, not antagonistic**. We also measured the energy of audio gradients projected into the text gradient subspace -- it is just 0.1%, compared to a random baseline of 0.00005%. Why orthogonal? Because text embeddings are pretrained and structured, while audio embeddings start as random Gaussian noise with standard deviation 0.02. They live in completely different subspaces. The backbone features align with text representations, and audio simply cannot access those learned features yet.

This is why PCGrad and gradient projection methods, which we tested in experiment 3, had zero effect. **There is no conflict to resolve.** The problem lies elsewhere.

---

### Slide 5: Finding 2 -- GSNR Theory and Steps vs. Batch

The GSNR framework from McCandlish et al. gives us a quantitative handle on the problem. GSNR at batch size B equals B times the squared gradient norm divided by the trace of the gradient covariance. This defines a critical batch size, B-crit, where noise equals signal. For audio, we measured GSNR of 0.08 at batch size 2, which gives us a **B-crit of about 25 for audio**, versus only 2 for text. The optimal batch size B-star is approximately 9.

Here is the critical insight from our controlled experiments. Look at the comparison table: experiment 16 used effective batch 32 for 30,000 steps and reached val loss 36.8. Experiment 15 used effective batch 192 for only 5,000 steps -- the same total number of samples -- and only reached 44.6. **More noisy steps beats fewer clean steps.** The effective batch 192 run wastes 81% of its compute relative to the optimal batch of 9. S2 pretraining does not actually help the final val loss: S2-to-S3 gives 44.8 versus direct S3 at 44.9. S2 only smooths the landscape.

This tells us the path forward is more steps, not bigger batches.

---

### Slide 6: Finding 3 -- Two CKA Paths

This finding really isolates where the bottleneck lives. We compared two training paths and measured CKA similarity to the initial checkpoint at the middle layer. Direct S3 training completely reshapes the backbone -- CKA drops to 0.03, meaning the representations become almost unrecognizable. S2-to-S3 training keeps the backbone nearly frozen with CKA at 0.99. And yet, **both paths reach the same validation performance**: 44.9 versus 44.8.

The audio gradients even flow differently: in direct S3 they propagate output-first from layer 23 down, while in S2-to-S3 they concentrate in layers 0 through 2, which act as spontaneous adapters. The backbone is just a feature extractor. Reshape it or not, you get the same result. **The bottleneck is in the embedding space**, not the transformer layers.

---

### Slide 7: Finding 4 -- Basin Width

We probed the loss landscape by adding small perturbations at epsilon equals 0.01 and measuring degradation. The S2-to-S3 path starts in a **flat basin** with degradation around 40 and widens 4 times to about 163 over training. Direct S3 starts in a sharp basin at 336 and actually narrows to 204. So S2 pretraining gives you a much smoother starting point. This is its real contribution -- not better final accuracy, but **a more stable optimization trajectory**. It explains why S2-to-S3 training is more robust even though it converges to the same place.

This connects back to the GSNR story: in a flatter basin, noisy gradients are less likely to push you into bad regions.

---

### Slide 8: Finding 5 -- Gradient Dynamics

Here is a paradox. The left chart shows rho, the ratio of audio to text gradient norms, rising from 0.6 to 2.6 over 30,000 steps. Audio gradients are getting bigger in magnitude, about 4 times larger. But the right chart shows GSNR staying completely flat at 0.08 across the entire chain. **The gradients get louder, but they do not get clearer.** The magnitude increases because as text converges, the model finds audio relatively harder. But the signal-to-noise ratio does not improve at all. Different samples still push the audio parameters in different directions, canceling each other out.

This means learning rate scaling will not help. You cannot fix a noise problem by amplifying the noise.

---

### Slide 9: Finding 6 -- Linear Probe and Embedding Learning

We trained linear probes on the backbone's intermediate representations to predict audio tokens. At layer 18, train accuracy reaches 77%, but validation accuracy is just 6.2% -- a **12x gap**. The backbone is memorizing audio patterns on the training set but cannot generalize. This is the chicken-and-egg problem: embeddings start random, so the backbone cannot learn stable audio rules, which means gradients stay noisy at GSNR 0.08, which means embeddings learn slowly.

But they do learn. CB0 top-10 accuracy improves from 20.7% at 3,000 steps to 44.4% at 16,500 steps -- a 114% improvement. And audio-audio cosine similarity in the embedding space gradually rises from 0 to 0.12. **Structure is forming, just very slowly.** We also found a bug: the D5 embedding displacement metric reported zero due to a parameter name mismatch with tied word embeddings. The actual displacement through the LM head is 127, confirming that training is working correctly.

---

### Slide 10: Key Insights

Let me summarize the six main takeaways. One: text and audio gradients are orthogonal, not conflicting. Gradient surgery methods are the wrong tool here. Two: the root cause is low audio GSNR at 0.08 -- audio needs 12 times more samples than text. Three: the plateau does break with sufficient training. We saw a 34% improvement over 16,500 steps. Four: S2 pretraining creates a flatter basin that widens further during S3, explaining its stabilizing effect. Five: the backbone is not the bottleneck -- CKA 0.03 and CKA 0.99 reach the same loss. **The bottleneck is embedding-level GSNR.** Six: audio gradient magnitude grows 4x but GSNR stays flat, so learning rate tricks cannot fix this.

The common thread is clear: this is a noise-dominated learning regime.

---

### Slide 11: Ongoing and Next Steps

We are currently running a chained training experiment with the full Mini-Omni pipeline. Rounds 1 through 3 are complete at 16,500 global steps, with rounds 4 through 6 queued to reach 27,500 steps. The audio loss trajectory is still declining: 29.9 to 25.6 to 23.9, with no convergence yet.

Going forward, we propose four experiments. First, optimal task sampling with 22% text and 78% audio, which B-crit theory predicts should improve efficiency. Second, LR scaling at 7x for the large-batch runs to match small-batch efficiency. Third, directly measuring text GSNR, which we currently only estimate. And fourth, pushing training to 50,000-plus steps since audio is still improving.

**The bottom line**: the audio plateau is a noise-dominated learning regime with GSNR of 0.08. It is not an optimization or architecture failure. It breaks with sufficient training, and the critical batch size framework gives us actionable guidance for compute-optimal scheduling. Thank you.

---
