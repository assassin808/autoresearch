#!/bin/bash
set -e
cd /workspace/autoresearch
P="/usr/bin/python3 -u"
S=3000

echo "=== Exp D: Authentic S2 ===" && \
mkdir -p results/s2_textonly && \
$P train_s3.py --output_dir results/s2_textonly --max_steps $S --s2_mode --no_save_model 2>&1 | tee results/s2_textonly/train.log

echo "=== Exp A: CB Weighting ===" && \
mkdir -p results/s3_cbweight && \
$P train_s3.py --output_dir results/s3_cbweight --max_steps $S --cb_weights "100,10,1,1,10,1,1" --no_save_model 2>&1 | tee results/s3_cbweight/train.log

echo "=== Exp C: Skip S2 ===" && \
mkdir -p results/s3_skip_s2 && \
$P train_s3.py --output_dir results/s3_skip_s2 --max_steps $S --checkpoint_mode post_s1 --lr_max 2e-4 --no_save_model 2>&1 | tee results/s3_skip_s2/train.log

echo "=== Exp B: M-SAM ===" && \
mkdir -p results/s3_msam && \
$P train_s3.py --output_dir results/s3_msam --max_steps $S --method m_sam --sam_rho 0.05 --no_save_model 2>&1 | tee results/s3_msam/train.log
