#!/bin/bash
#SBATCH --job-name=train_s3
#SBATCH --account=def-xli135
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/train_s3_%j.out

source narval_env.sh
mkdir -p logs

# Usage:
#   Single experiment:
#     sbatch --export=ALL,EXTRA_ARGS="--method gradnorm --output_dir results/exp3_s3_gradnorm" job_train_s3.sh
#   All 4 initial experiments:
#     sbatch --export=ALL,RUN_ALL=1 job_train_s3.sh

if [ "${RUN_ALL:-0}" = "1" ]; then
    echo "Running all 4 initial experiments..."
    bash run_initial_experiments.sh
else
    uv run python train_s3.py \
        --checkpoint_mode post_s1 \
        --max_steps 3000 \
        --no_save_model \
        --output_dir results/s3_narval \
        ${EXTRA_ARGS:-}
fi
