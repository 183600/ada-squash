# Code Review — 2026-11-27 (Independent Review)

**Reviewer**: Independent Code Audit  
**Scope**: Normalization layers (norms.py), attention (attention.py), model (model.py), metrics (metrics.py), experiment pipeline

## Summary

**No bugs or methodology issues found.**

ada-squash implements the DyT++ family of statistic-free adaptive squashing normalization layers. The custom autograd functions are correctly implemented with proper numerical stability guards.

## Detailed Findings

### 1. Normalization Layers (`adasquash/norms.py`)

- **Generalized softsign**: The core `phi_p(x) = x / (1 + |x|^p)^(1/p)` is implemented as a custom `_GeneralizedSoftsign` autograd function. The forward uses closed-form computation; the backward recomputes intermediates from saved inputs.
- **Numerical stability**: The `_softsign_core` function uses `reciprocal on max(|x|, 1)` formulation to prevent division by zero at x=0. The `_MIN_P = 1e-3` floor prevents the exponent from collapsing.
- **Depth scheduling**: `_depth_alpha(alpha_0, k, depth_ratio)` computes `alpha_0 * exp(-k * depth_ratio)` with optional `k_min` floor to prevent negative decay rates (which would cause alpha to grow with depth).
- **RMSNorm backward**: The `_RMSNorm` autograd function correctly implements the full backward pass including the correction term for the mean, with proper dtype promotion for mixed precision.
- **Per-channel norms**: `DyTC` and variants correctly handle learnable per-channel alpha with optional jitter initialization.

### 2. Attention (`adasquash/attention.py`)

- **Causal self-attention**: Uses `F.scaled_dot_product_attention` with `is_causal=True` for efficient attention computation. The logit statistics collection via `_logit_stats()` correctly computes masked statistics over the causal triangle.
- **RoPE caching**: The `_rope_tables()` method uses a `(data_ptr, shape, device, dtype)` cache key with version tracking for in-place buffer updates.
- **QK-norm variants**: Supports `qknorm` (RMSNormHead) and `dytqk` (QKSquash) with proper initialization.
- **Scratch buffer management**: The `_ce_scratch_cache` and `_CAST_WEIGHT_CACHE` use LRU eviction with byte-count limits.

### 3. Cross-Entropy (`adasquash/model.py`)

- **ChunkedCrossEntropy**: The custom `_ChunkedCrossEntropy` autograd function correctly handles `ignore_index` masking in both forward and backward. The backward explicitly sets `probs[~valid] = 0.0` for ignored positions, matching `F.cross_entropy` semantics.
- **Empty batch handling**: Returns `logits.sum() * 0.0` for empty batches (zero valid targets), producing a differentiable zero loss.
- **Type promotion**: `_loss_acc_dtype()` promotes bf16/fp16 to fp32 for accumulation stability.

### 4. Metrics (`adasquash/metrics.py`)

- **Statistical test suite**: Comprehensive pure-Python implementation including paired t-test, Welch's t-test, TOST equivalence test, meta-analysis with DerSimonian-Laird random effects, Cochran's Q, I², and prediction intervals.
- **Fragility analysis**: The `_fragility_summary()` computes leave-one-out fragility indices to assess whether significant results are driven by a small number of seeds.
- **Multiple comparison correction**: Holm-Bonferroni for FWER control and Benjamini-Hochberg for FDR control are both available.

### 5. Methodology Assessment

- **DyT family taxonomy**: The 10+ normalization variants are correctly organized with clear inheritance (`_SquashedAffine` → depth-conditioned → DyTCD/DyTD).
- **Depth reparameterization**: `learnable_k` correctly switches between `nn.Parameter` and non-persistent buffer, with `k_min` preventing negative decay.
- **Softsign p-gradient**: `_softsign_p_grad()` correctly implements the gradient of the generalized softsign w.r.t. the learnable exponent p, with `_MIN_P` guard.

## Conclusion

No actionable issues found. The codebase correctly implements a comprehensive family of normalization layers with thorough numerical stability considerations.
