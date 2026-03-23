#!/bin/bash
#SBATCH --account=def-xli135
#SBATCH --job-name=omni_s3
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=logs/omni_s3_%j.out
source narval_env.sh
mkdir -p logs
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1
cd /home/yang0531/autoresearch

# Mini-omni S3 config:
#   eff_batch=192 (batch=2, grad_accum=96)
#   LR: 2e-6 → 2e-5 cosine, 1500 warmup
#   Save checkpoint every 2500 steps for continuation
#
# Usage:
#   First run (from S2 checkpoint):
#     sbatch job_miniomni_s3.sh
#   Continuation (from last saved checkpoint):
#     sbatch --export=ALL,RESUME=results/omni_s3/model_final.pt job_miniomni_s3.sh

OUTPUT_DIR=results/omni_s3
STEPS=5500

if [ -n "$RESUME" ]; then
    echo "Resuming from: $RESUME"
    # Append to existing output dir with step suffix
    RUN_NUM=$(ls -d ${OUTPUT_DIR}_run* 2>/dev/null | wc -l)
    RUN_NUM=$((RUN_NUM + 1))
    OUTPUT_DIR="${OUTPUT_DIR}_run${RUN_NUM}"
    INIT_FLAG="--init_checkpoint $RESUME"
else
    echo "Starting from S2 checkpoint (exp11)"
    INIT_FLAG="--init_checkpoint results/exp11_long_s2/model_final.pt"
fi

echo "Output: $OUTPUT_DIR"
echo "Steps: $STEPS"
echo "eff_batch: 192 (2 × 96)"

uv run python train_s3.py \
    $INIT_FLAG \
    --max_steps $STEPS \
    --grad_accum 96 \
    --save_every 2500 \
    --output_dir $OUTPUT_DIR
