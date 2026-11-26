# Code Review Report - 2026-11-27 (Round 3 - Independent Deep Review)

## Reviewer: QwenGo Automated Review
## Repository: ada-squash (DyT++ / AdaSquash)
## Date: 2026-11-27 05:00 CST

## Scope

Comprehensive independent review of core modules:
- `adasquash/norms.py` (~1000 lines): DyT++ normalization family
- `adasquash/attention.py` (~500 lines): CausalSelfAttention with QK-norm
- `adasquash/model.py` (~1000 lines): GPT model with MoE, chunked CE
- `adasquash/_caching.py`: LRU scratch buffer management
- `adasquash/_validation.py`: Input validation utilities

## Findings

### Code Quality: EXCELLENT

#### 1. Generalized Softsign (norms.py)
- **Core math**: `phi_p(x) = x / (1 + |x|^p)^(1/p)` implementation is mathematically correct.
  - `_softsign_core`: `t = |scale·x|`, `lo = min(t, 1)`, `r = 1/max(t, 1)`, `u = lo·r` — the floor at `max(|x|, 1)` prevents 1/0 at zero and NaN in backward.
  - `_compute_s_and_a`: Handles both tensor and scalar `p` correctly; the `_LN2` and `_LOG2E` constants are used appropriately for the log2/exp2 fast path.
- **Backward pass**: The gradient `r * s` (where `r = 1/max(|x|,1)` and `s = (1 + u^p)^(-1/p)`) is correct for `|x| < 1` and `|x| > 1` regimes.
- **p gradient**: `_softsign_p_grad` correctly computes `d(phi)/dp` via the log-space formulation, with clamping at `_MIN_P = 1e-3` preventing degenerate exponents.

#### 2. RMSNorm (norms.py)
- **Forward**: `rstd = 1/sqrt(mean(x²) + eps)` — numerically stable with `vector_norm` + `square_` + `rsqrt_`.
- **Backward**: Correctly implements `grad_x = rstd * (grad_out * weight - x * mean(grad_out * weight * x * rstd²))`. The dtype promotion logic ensures fp32 accumulation for bf16/fp16 inputs.

#### 3. Depth-Conditioned Norms
- **Depth schedule**: `alpha_l = alpha_0 * exp(-k * depth_ratio)` with `_effective_decay_rate` flooring `k` at `k_min` to prevent negative decay (which would cause alpha to grow with depth).
- **LayerScale integration**: Properly combines depth-conditioned alpha with residual scaling.

#### 4. Attention (attention.py)
- **QK-Norm variants**: `qknorm` (RMSNorm on Q/K) and `dytqk` (QKSquash) are correctly applied before RoPE.
- **Logit statistics**: `_logit_stats` computes mean/abs_mean/max/std over the causal triangle with:
  - Row-block chunking for memory efficiency
  - float64 accumulation for numerical precision
  - Correct handling of NaN/Inf variance (reporting `float("nan")` rather than 0)
  - The `scale` factor `1/sqrt(head_dim)` is applied to all accumulated values
- **RoPE caching**: D6 fix with `data_ptr` + `_version` tracking correctly invalidates on in-place buffer updates.

#### 5. Chunked Cross-Entropy (model.py)
- **Forward**: Correctly accumulates per-chunk NLL loss and divides by valid target count. `count.clamp(min=1)` prevents division by zero when all targets are ignored (returns ~0 loss).
- **Backward**: Properly zeros gradients for ignored positions: `probs[~valid[start:end]] = 0.0`. Without this, ignored positions would incorrectly receive `softmax * grad_scale`.
- **Fused linear CE**: Correctly handles autocast dtype promotion with cached weight casts and separate accumulator dtype (`_loss_acc_dtype` promotes bf16/fp16 to fp32).
- **uint8 target handling**: `_valid_target_mask` correctly casts uint8 targets to int64 before comparing with negative `ignore_index` (e.g., -100 would become 156 in uint8).

#### 6. Numerical Stability
- **_MIN_ALPHA = 1e-4**: Prevents degenerate alpha values.
- **_MIN_P = 1e-3**: Floors the softsign exponent.
- **reciprocal-on-max(|x|, 1)**: The D2 design floor prevents 1/0 and maintains gradient flow.

### Bugs Found: NONE
### Methodology Issues Found: NONE

### Observations (cosmetic, not committed)
- In `attention.py`, the `_skip_init` function parameter name shadows the imported `torch.nn.utils.skip_init`. Functionally correct but slightly confusing naming.

## Recommendation: No changes needed. Code is production-ready.
