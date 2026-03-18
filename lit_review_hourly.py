#!/usr/bin/env python3
"""
Hourly literature review — designed to be invoked by Claude Code cron.
Outputs search queries and instructions for the agent to execute.
The cron hook will use WebSearch/WebFetch to find papers.
"""

import datetime
import os

SEARCH_QUERIES = [
    # Core topics
    ["omni model speech text joint training 2025", "multimodal LLM audio generation training dynamics 2025"],
    # Kimi + attention
    ["Kimi K2 attention residual moonshot AI 2025 paper", "Kimi model architecture attention mechanism 2025"],
    # 贾佳亚 group
    ["Jiajia Jia CUHK MMLab omni multimodal 2025", "VITA-1.5 omni model speech 2025"],
    # Gradient methods
    ["PCGrad CAGrad gradient conflict multi-task learning", "gradient surgery multimodal training 2024 2025"],
    # Audio codec LM
    ["discrete audio token inconsistency language model", "SNAC EnCodec speech language model training loss"],
    # SAM + basin
    ["sharpness aware minimization multimodal fine-tuning", "loss landscape basin width language model 2025"],
    # Related models
    ["Moshi duplex speech language model training details", "LLaMA-Omni SpeechGPT training optimization 2025"],
    # Codebook weighting
    ["residual vector quantization codebook weighting", "audio codec hierarchical loss multi-scale 2025"],
    # Multi-task optimization theory
    ["multi-task learning gradient alignment theory 2025", "Nash bargaining multi-objective optimization neural"],
    # Omni training curriculum
    ["multimodal model training curriculum speech text stages", "progressive training audio language model 2025"],
]

OUTPUT_FILE = "/workspace/autoresearch/LITERATURE_NOTES.md"


def get_queries_for_this_hour():
    now = datetime.datetime.now()
    idx = (now.hour + now.day * 24) % len(SEARCH_QUERIES)
    return idx, SEARCH_QUERIES[idx]


def main():
    idx, queries = get_queries_for_this_hour()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"LITERATURE_REVIEW_QUERIES={queries}")
    print(f"LITERATURE_REVIEW_TIME={now}")
    print(f"LITERATURE_REVIEW_RUN={idx}")
    print(f"OUTPUT_FILE={OUTPUT_FILE}")


if __name__ == "__main__":
    main()
