# Code Review Report — 2026-11-26 (Independent Review)

**Reviewer**: QwenGo! Code Review Agent  
**Repository**: ada-squash  
**Scope**: Full code review of model, norms, training scripts, ablation framework, and tests

## Summary

**Verdict: No new bugs or scientific methodology issues found.**

This independent review follows Round 6. All recent fixes are confirmed correct and no regressions were introduced.

## Areas Reviewed

### 1. Model Architecture (`adasquash/model.py`)
- **GPT model**: Standard decoder-only transformer with SwiGLU MLP, multiple norm families (DyT, DyTCD, etc.), and optional rotary embeddings.
- **Cross-entropy functions**: Both `_ChunkedCrossEntropy` and `_FusedLinearCrossEntropy` correctly handle `ignore_index` masking with proper gradient zeroing for ignored positions.
- **Division protection fix**: The recent fix changing `max(count, 1).clamp(min=1)` to `count.clamp(min=1)` is correct. When `count` is a 0-d tensor with value 0, Python's `max()` returns an `int`, so `.clamp()` would raise `AttributeError`. The fix preserves tensor semantics.
- **SwiGLU backward**: The `_SwiGLU` autograd function correctly computes gradients through `silu(x@W1) * (x@W3) @ W2`, with proper dtype handling for autocast.
- **LayerScale**: `_LayerScaleResidual` correctly routes gradients through the scale parameter and the residual branch.

### 2. Normalization (`adasquash/norms.py`)
- **DyTCD family**: Depth-conditioned normalization with learnable alpha_0, k, and delta. The depth reparameterization `alpha_l = alpha_0 * exp(-k * l / L)` is correctly implemented.
- **Shared depth schedule**: When `shared_depth_schedule=True`, the model correctly shares one `(alpha_0, k)` pair across all layers, making them identifiable parameters.

### 3. Ablation Framework
- **Matrix design**: The factorial ablation design properly separates main effects from interaction effects.
- **Baseline comparison**: Excess-over-Bayes-floor metrics correctly account for data entropy baselines.

### 4. Training Scripts
- **Sanity check**: Pre-flight overfit sanity check (`sanity_overfit.py`) catches gradient/loss bugs before full training.
- **Seed replication**: Multi-seed training with randomized complete-block schedule prevents order confounding.

### 5. Recent Fixes Verified
- `.clamp_min(1e-10)` artifact removal from strings/comments/paths: ✅ Correctly applied
- Division-by-zero protection in cross-entropy: ✅ Correct fix confirmed
- Test tolerance adjustments: ✅ Appropriate

## Conclusion

The codebase is production-quality with excellent numerical stability guards. The cross-entropy implementations are particularly well-engineered for memory efficiency. No new issues identified.
