"""
Phase 3 utility scripts:
  - train_moe_tiny.py    : tiny 8x0.3B MoE for expert variance studies
  - check_fp8_stability.py: empirically check FP8 / bf16 training divergence
  - check_w8a8_ppl.py    : post-training W8A8 quantization PPL drop
  - run_needle.py        : long-context needle sweep

These are intentionally minimal: each one is a standalone entry point for
a single Phase 3 check.
"""

from __future__ import annotations

# This file is a stub for the project layout. The actual implementations
# live in train_moe_tiny.py / check_fp8_stability.py / check_w8a8_ppl.py /
# run_needle.py next to this README. We keep them separate so that each
# can be invoked directly.
