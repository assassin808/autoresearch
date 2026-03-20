#!/bin/bash
# run_initial_experiments.sh — 4 initial experiments with enhanced diagnostics
# All start from post-S1 checkpoint, 3000 steps each
# Total: ~10h sequential on single GPU

set -euo pipefail

# Source env vars (Narval paths)
if [ -f narval_env.sh ]; then
    source narval_env.sh
fi

PYTHON="${PYTHON:-/usr/bin/python3}"
COMMON="--checkpoint_mode post_s1 --max_steps 3000 --no_save_model"

echo "============================================"
echo "Experiment 1/4: S2 text-only baseline"
echo "============================================"
$PYTHON train_s3.py $COMMON --s2_mode \
    --output_dir results/exp1_s2_text

echo ""
echo "============================================"
echo "Experiment 2/4: S3 baseline (all 4 tasks)"
echo "============================================"
$PYTHON train_s3.py $COMMON \
    --output_dir results/exp2_s3_baseline

echo ""
echo "============================================"
echo "Experiment 3/4: S3 + GradNorm"
echo "============================================"
$PYTHON train_s3.py $COMMON --method gradnorm \
    --gradnorm_alpha 1.5 \
    --output_dir results/exp3_s3_gradnorm

echo ""
echo "============================================"
echo "Experiment 4/4: S3 + entropy-scaled loss"
echo "============================================"
# cb_entropy values: fill in from analyze_codebook_entropy.py output
# Placeholder values below — run the analysis script first:
#   $PYTHON analyze_codebook_entropy.py
# Then replace with actual H_cb values (nats)
CB_ENTROPY="${CB_ENTROPY:-7.5,7.5,7.5,7.5,7.5,7.5,7.5}"
$PYTHON train_s3.py $COMMON --method entropy_scaled \
    --cb_entropy "$CB_ENTROPY" \
    --output_dir results/exp4_s3_entropy

echo ""
echo "============================================"
echo "All 4 experiments complete!"
echo "Results in: results/exp{1,2,3,4}_*/"
echo "============================================"
