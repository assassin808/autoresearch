#!/bin/bash
# Run after GradProj finishes
set -e
cd /workspace/autoresearch
PYTHON="/usr/bin/python3 -u"

echo "=== Experiment 3: Adaptive λ ==="
echo "Start: $(date)"
$PYTHON train_s3.py \
    --output_dir results/s3_adaptive \
    --max_steps 3000 \
    --method adaptive_lambda \
    --audio_weight 1.0 \
    --adaptive_alpha 0.1 \
    --no_save_model \
    2>&1 | tee results/s3_adaptive/train.log
echo "Done: $(date)"
