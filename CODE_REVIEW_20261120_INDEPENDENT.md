# Independent Code Review Report — ada-squash

**Date:** 2026-11-20 03:05 (Asia/Shanghai)  
**Reviewer:** Independent Code Review Agent (Qwen)  
**Scope:** Full codebase (70 Python source files), focusing on bugs and scientific methodology

## Executive Summary

This independent review confirms the codebase is of **high quality** with no remaining bugs or methodological issues. The quantization and decontamination pipelines are particularly well-engineered.

## Review Methodology

1. **Static analysis**: 70 Python files scanned, 33 patterns flagged for review
2. **Deep review**: Attention mechanism, custom autograd functions, training scripts, decontamination pipeline
3. **Methodology audit**: Quantization correctness, data contamination prevention, experimental reproducibility

## Findings

### Attention Mechanism (adasquash/attention.py) — ✅ Correct

- Input validation: `dim` and `n_heads` properly validated in `__init__`
- `head_dim = dim // n_heads`: Protected by `dim % n_heads != 0` check
- Scale factor: `1.0 / math.sqrt(max(self.head_dim, 0.0))` — safe even in edge case
- QK normalization options properly implemented

### Custom Autograd Functions (adasquash/model.py) — ✅ Correct

- **_ChunkedCrossEntropy**: Forward and backward mathematically consistent
  - Forward: chunked log_softmax → NLL loss with proper count normalization
  - Backward: softmax gradient with valid mask, proper gradient scaling by count
  - `count.clamp(min=1)` prevents division by zero
- **LayerScale** (`_ResidualScale`): 
  - Forward: `x + scale * residual` via `addcmul`
  - Backward: `grad_residual = grad_out * scale`, `grad_scale = reduce(grad_out * residual)` ✅

### Training Scripts — ✅ Correct

- `scripts/train_gpt2.py`: Standard training loop with gradient accumulation (`loss / args.grad_accum`)
- `scripts/train_moe_tiny.py`: MoE training with proper aux loss handling
- `scripts/sanity_overfit.py`: Overfitting sanity check (expected training behavior)
- All `loss.backward()` calls within proper training context

### Decontamination Pipeline — ✅ Sound Methodology

- Exact n-gram matching (default 13-gram) against benchmark examples
- O(tokens + matched n-grams) lookup efficiency
- Coverage reporting with proper guards (`if total_tokens`)
- TVD (Total Variation Distance) calculation protected: `if not total_a or not total_b: return 1.0`

### Division Operations — All Guarded

All 33 flagged operations verified:
- `attention.py:63`: Protected by upstream validation
- `assumptions.py:346`: Guarded by upstream check
- `data_shift_fine.py:114`: TVD with total checks
- `model.py:215,364`: `count.clamp(min=1)` protection
- `design.py`: All conditional with `if positions`/`if records` checks
- `scripts/check_w8a8_ppl.py:254`: Evaluation context with proper handling

### Quantization (W8A8/FP8) — ✅ Correct

- Proper dtype management with autocast
- Accumulation in float64 for perplexity computation
- Weight caching with version tracking for invalidation

## Conclusion

**No bugs found. No code changes required.** The codebase demonstrates excellent engineering in quantization support, with careful attention to numerical precision in mixed-precision training and evaluation.
