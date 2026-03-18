#!/bin/bash
# S3 optimizer comparison experiments
# Each run: 3000 steps, ~133min
# Total: ~3 runs × 133min = ~7h

set -e
cd /workspace/autoresearch

STEPS=3000
PYTHON="/usr/bin/python3 -u"

echo "=== Experiment 1: λ=3 (higher audio weight) ==="
echo "Start: $(date)"
$PYTHON train_s3.py \
    --output_dir results/s3_lambda3 \
    --max_steps $STEPS \
    --audio_weight 3.0 \
    --no_save_model \
    2>&1 | tee results/s3_lambda3/train.log
echo "Done: $(date)"

echo ""
echo "=== Experiment 2: Gradient Projection ==="
echo "Start: $(date)"
$PYTHON train_s3.py \
    --output_dir results/s3_gradproj \
    --max_steps $STEPS \
    --method grad_proj \
    --no_save_model \
    2>&1 | tee results/s3_gradproj/train.log
echo "Done: $(date)"

echo ""
echo "=== Experiment 3: Adaptive λ ==="
echo "Start: $(date)"
$PYTHON train_s3.py \
    --output_dir results/s3_adaptive \
    --max_steps $STEPS \
    --method adaptive_lambda \
    --audio_weight 1.0 \
    --adaptive_alpha 0.1 \
    --no_save_model \
    2>&1 | tee results/s3_adaptive/train.log
echo "Done: $(date)"

echo ""
echo "=== All experiments complete ==="
