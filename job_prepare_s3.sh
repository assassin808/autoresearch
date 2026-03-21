#!/bin/bash
#SBATCH --job-name=prep_s3
#SBATCH --account=def-xli135
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/prep_s3_%j.out

source narval_env.sh
mkdir -p logs $S3_DATA_DIR

# No network on compute nodes - use pre-downloaded deps + tokenizer
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_OFFLINE=1

uv run python prepare_s3.py --output_dir $S3_DATA_DIR
