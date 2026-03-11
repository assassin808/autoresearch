# Autoresearch Memory

## Credentials
- HuggingFace token: stored at /root/.cache/huggingface/token

## Hardware
- GPU: NVIDIA RTX 5090, 32 GB VRAM

## Project: autoresearch
- Location: /workspace/autoresearch
- Goal: study omni-model pretraining quality, focus on optimizer tweaks
- Reference model: gpt-omni/mini-omni (Qwen2-0.5B + Whisper + SNAC, real-time speech-to-speech)
- Branch: autoresearch/mar9
- Best config: batch128K, NS10, LLLL, MuonWD=0.1, ratio=0.15 → val_loss=3.402
- 170+ experiments total across sweeps 1-12
- Audio data: 115h LibriSpeech (train-clean-100 + dev-other + test splits)
- FA3 unsupported on Blackwell → SDPA + FlexAttention
- THEORY.md: 10 hypotheses, experimental results
- Detailed findings: see autoresearch_findings.md
- GitHub: assassin808/autoresearch, push via `git push myfork autoresearch/mar9`
