#!/bin/bash
# Environment variables for Narval (Compute Canada)
# Source this before running any scripts:  source narval_env.sh

export MINI_OMNI_REF=/scratch/yang0531/mini-omni-ref
export MINI_OMNI_CKPT=/scratch/yang0531/mini-omni-ckpt
export S3_DATA_DIR=/scratch/yang0531/s3_data
export S2_DATA_DIR=/scratch/yang0531/s2_data
export HF_HOME=/scratch/yang0531/.hf_home
export VA400K_BLOBS=/scratch/yang0531/.hf_home/hub/datasets--gpt-omni--VoiceAssistant-400K/blobs

export QWEN2_TOKENIZER=/scratch/yang0531/qwen2_tokenizer
export WHISPER_MODEL_DIR=/scratch/yang0531/whisper_models
export VA400K_SNAPSHOT=/scratch/yang0531/.hf_home/hub/datasets--gpt-omni--VoiceAssistant-400K/snapshots

# uv path
export PATH="$HOME/.local/bin:$PATH"
