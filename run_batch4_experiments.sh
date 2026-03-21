#!/bin/bash
# run_batch4_experiments.sh — Batch 4: Signal strength experiments
# Based on findings from Batch 3: audio plateau caused by directionless gradients
# from random audio embeddings, not gradient conflict.

set -euo pipefail

if [ -f narval_env.sh ]; then
    source narval_env.sh
fi

PYTHON="${PYTHON:-uv run python}"
COMMON="--checkpoint_mode post_s1 --max_steps 3000 --no_save_model"
CB_ENTROPY="6.0316,7.0948,7.9454,7.9326,7.1011,7.9515,7.9261"

echo "============================================"
echo "Exp 5/10: GradNorm (fixed, clamped weights)"
echo "============================================"
$PYTHON train_s3.py $COMMON --method gradnorm \
    --gradnorm_alpha 1.5 \
    --output_dir results/exp5_gradnorm_fixed

echo ""
echo "============================================"
echo "Exp 6/10: Entropy-scaled (fixed, text+audio normalized)"
echo "============================================"
$PYTHON train_s3.py $COMMON --method entropy_scaled \
    --cb_entropy "$CB_ENTROPY" \
    --output_dir results/exp6_entropy_fixed

echo ""
echo "============================================"
echo "Exp 7/10: Freeze backbone (only train emb+adapter+heads)"
echo "============================================"
$PYTHON train_s3.py $COMMON --freeze_backbone \
    --output_dir results/exp7_freeze_backbone

echo ""
echo "============================================"
echo "Exp 8/10: Audio emb init from text (random samples)"
echo "============================================"
$PYTHON train_s3.py $COMMON --audio_emb_init sample \
    --output_dir results/exp8_emb_init_sample

echo ""
echo "============================================"
echo "Exp 9/10: Curriculum (A1T2 first 1000 steps, then all)"
echo "============================================"
$PYTHON train_s3.py $COMMON --curriculum 1000 \
    --output_dir results/exp9_curriculum

echo ""
echo "============================================"
echo "Exp 10/10: Freeze backbone + emb init (combined)"
echo "============================================"
$PYTHON train_s3.py $COMMON --freeze_backbone --audio_emb_init sample \
    --output_dir results/exp10_freeze_emb_init

echo ""
echo "============================================"
echo "All 6 experiments complete!"
echo "Results in: results/exp{5..10}_*/"
echo "============================================"
