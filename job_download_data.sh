#!/bin/bash
#SBATCH --job-name=dl_data
#SBATCH --account=def-xli135
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/dl_data_%j.out

source narval_env.sh
mkdir -p logs $S3_DATA_DIR

echo "=== Step 1: Download dataset ==="
uv run python -c "
from datasets import load_dataset
print('Downloading VoiceAssistant-400K...')
ds = load_dataset('gpt-omni/VoiceAssistant-400K', cache_dir='$HF_HOME')
print(f'Loaded {len(ds[\"train\"])} samples')
print('Done.')
"

echo "=== Step 2: Tokenize S3 data ==="
uv run python prepare_s3.py --output_dir $S3_DATA_DIR

echo "=== Done ==="
ls -lh $S3_DATA_DIR/
