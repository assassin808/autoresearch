#!/bin/bash
#SBATCH --job-name=whisper_ext
#SBATCH --account=def-xli135
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/whisper_ext_%j.out

source narval_env.sh
mkdir -p logs $S3_DATA_DIR/whisper_features

# No network on compute nodes - use pre-downloaded model
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

uv run python prepare_s3.py --extract_whisper --output_dir $S3_DATA_DIR
