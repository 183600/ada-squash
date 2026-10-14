"""
Entry-point scripts (each is a standalone, directly invocable driver):
  - train_gpt2.py            : Phase 1 training driver (GPT-2 124M)
  - train_llama_depth_sweep.py: Phase 2 driver + 12/24/48/96-layer sweep launcher
  - ablation.py              : one-shot ablation matrix runner
  - eval_downstream.py       : downstream benchmarks + needle-in-a-haystack
  - run_needle.py            : long-context needle sweep
  - train_moe_tiny.py        : tiny 8x0.3B MoE for expert variance studies
  - check_fp8_stability.py   : simulated FP8 / bf16 training divergence
  - check_w8a8_ppl.py        : post-training W8A8 quantization PPL check
"""

from __future__ import annotations

# This file only marks the package. The implementations live beside it and are
# kept separate so that each can be invoked directly.
