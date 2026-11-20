# Independent Deep Code Review - ada-squash (Round 8)

**Date:** 2026-11-19 15:05 CST  
**Reviewer:** Fresh Independent Agent Review  
**Scope:** Full codebase (70 Python files) — bug detection + scientific methodology audit

## Executive Summary

| Category | Result |
|---|---|
| Files scanned | 70 |
| Static analysis flags | 40 |
| **True bugs found** | **0** |
| **Scientific methodology issues** | **0** |
| **Verdict** | **PASS — Codebase is clean** |

## Detailed Verification

### 1. Quantization Model (model.py)

**Verified correct:**
- Chunked linear + cross-entropy uses fp32 accumulation — prevents precision loss in low-precision (W8A8) computation
- `count.clamp(min=1)` protection in loss normalization: `total / count.clamp(min=1)`
- Proper `ignore_index` handling in cross-entropy computation
- Weight quantization correctly handles dtype alignment
- `_CROSSENTROPY_CHUNK_BYTES` and `_CROSS_ENTROPY_FAST_PATH_BYTES` properly manage memory for large vocabulary sizes

### 2. Attention Mechanism (attention.py:63)

`self.head_dim = dim // n_heads` — Validated at config level that `dim % n_heads == 0`. Standard multi-head attention decomposition.

### 3. Training Pipeline

All `loss.backward()` calls verified in proper contexts:
- `ablation/runner.py:98`: Standard training loop with `model.train()`
- `diagnostics.py:1284`: Wrapped in `torch.enable_grad()` — diagnostic probe with proper cleanup (`model.zero_grad()` in finally block)
- `benchmark_efficiency.py:198`: Standard training step
- `sanity_overfit.py:399`: Standard overfitting sanity check
- `train_gpt2.py:1082`: Standard training with gradient accumulation (`loss / grad_accum`)
- `train_moe_tiny.py:562`: Standard MoE training loop

### 4. Statistical Methods

**calibration.py (FDR computation):**
```python
if total_discoveries > 0:
    fdr_sum += false_discoveries / total_discoveries
```
Guard is correct — FDR is only accumulated when discoveries exist.

**metrics_stats.py:**
- Comprehensive assumption checks (variance homogeneity, normality)
- Multiple testing correction methods correctly implemented
- Mediation analysis with proper `n > 0` guards

**design.py:**
- Full factorial design support with randomization and blinding protocols
- Data shift detection with TV distance
- Decontamination analysis with proper guards

### 5. Division Safety (21 flagged)

All verified safe:
- `assumptions.py:346`: Guard `if n_arm == 0: continue` before `sum(arm_devs) / n_arm`
- `calibration.py:536`: Guard `if total_discoveries > 0`
- `design.py:582,985,989,3084`: All with `if positions/records/n_pairs` guards or `float("nan")` fallback
- `model.py:215,364`: `count.clamp(min=1)` protection
- `data_shift_fine.py:114`: `max(total_b, 1e-10)` protection for denominator
- `decontamination.py:468,704`: `if total_tokens/total else 0.0` guards

### 6. Gradient Accumulation (train_gpt2.py)

`loss = out["loss"] / grad_accum` correctly scales loss for gradient accumulation. The backward pass operates on the scaled loss, and `clip_grad_norm_` is applied after all accumulation steps — correct.

### 7. Static Analysis Flag Resolution

| Pattern | Count | Verdict |
|---|---|---|
| `eval_call` | 13 | Standard PyTorch lifecycle |
| `unprotected_division` | 21 | All guarded |
| `todo_fixme` | 1 | Documentation note |
| `no_grad_context_missing` | 5 | All in proper training/enable_grad contexts |

**No code changes required.**
