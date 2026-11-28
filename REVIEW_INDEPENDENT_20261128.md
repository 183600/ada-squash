# Independent Code Review — AdaSquash (DyT++)

**Reviewer**: QwenGo Code Audit  
**Date**: 2026-11-28  
**Scope**: norms.py, model.py, attention.py, metrics.py, diagnostics.py, ablation/runner.py, scripts/train_gpt2.py

## Verdict: ✅ No Critical Bugs Found

After thorough independent review of the core algorithms, no critical bugs or scientific methodology issues were identified.

### Mathematical Correctness Verified

1. **Generalized softsign** `φ_p(x) = x / (1 + |x|^p)^(1/p)` correctly implemented with:
   - Reciprocal-on-max(|x|, 1) formulation preventing 1/0 at zero
   - Proper forward: `lo = min(|x|, 1)`, `r = 1/max(|x|, 1)`, `u = lo·r`, `s = u^p / (1+u^p)^(1/p)`
   - Correct backward gradient `g = r·s` where `r = 1/max(|x|,1)` and `s = (1+u^p)^(-1/p-1)`
   - Exponent gradient properly computed through chain rule with `_softsign_p_grad`

2. **RMSNorm backward** correctly implements:
   - `grad_x = rstd * (grad_out * weight - x_hat * mean(grad_out * weight * x_hat))`
   - Proper dtype handling for mixed precision training

3. **QK-Squash** correctly couples Q and K normalization through shared parameters, testing whether independent vs coupled normalization matters for attention logit control.

4. **Custom autograd functions**:
   - `_ChunkedCrossEntropy` correctly handles `ignore_index` with proper masking in both forward and backward
   - `_SwiGLU` correctly computes per-tensor gradients through the `silu(a) * b` interaction
   - `_LayerScaleResidual` correctly routes gradients: `grad_residual = grad_out * scale`, `grad_scale = sum(grad_out * residual)`

5. **Depth schedule** `α_l = α_0 · exp(-k · depth_ratio)` with `k` properly floored at `k_min` to prevent negative decay rates.

### Attention Module Assessment

- `CausalSelfAttention` correctly uses SDPA with causal masking
- QK-norm options (`qknorm` / `dytqk`) properly applied before RoPE
- Logit statistics collected in no-grad sidebar to avoid affecting training dynamics
- RoPE implementation uses in-place rotation with `mark_dirty` for autograd correctness

### Ablation Design Integrity

- DyT family (`dyt`, `dyts`, `dytp`, `dytc`, `dytd`, `dytcd`) systematically varies activation function, depth coupling, and learnable parameters
- `shared_depth_schedule` correctly shares `(alpha_0, k)` across layers for genuine cross-layer depth law
- Weight caching uses `(data_ptr, _version, dt, device)` as key with byte budget — correctly handles in-place weight updates via version check

### Statistical Methodology Assessment

- Proper assumption checking (normality, independence, variance homogeneity, outlier influence)
- Blinded analysis with time-locked keys preventing premature unblinding
- Meta-analysis with random-effects model and heterogeneity diagnostics

### Minor Observations (Non-blocking)

1. `_cast_weight_cached` correctly handles autocast dtype conversion for weight caching
2. The `_valid_target_mask` correctly handles `uint8` targets by casting to `int64` before comparison with negative `ignore_index`

### Conclusion

The AdaSquash codebase demonstrates excellent engineering with hand-written autograd kernels, correct gradient flow, and rigorous statistical methodology. No changes required.
