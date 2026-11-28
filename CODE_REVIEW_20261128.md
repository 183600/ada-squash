# Code Review — 2026-11-28

## Summary
Independent review of ada-squash core modules (attention.py, model.py, norms.py).

## Findings

### No Critical Bugs Found

The ada-squash codebase implements dynamic normalization (DyT family) with adaptive depth-aware scaling. Key findings:

1. **Attention (attention.py)**: `CausalSelfAttention` correctly uses SDPA with causal masking. QK-norm options (`qknorm`, `dytqk`) properly applied before RoPE. Logit statistics collected in no-grad sidebar to avoid affecting training.

2. **Model (model.py)**: 
   - `_ChunkedCrossEntropy` and `_FusedLinearCrossEntropy` correctly handle `ignore_index` with proper masking in both forward and backward. The backward explicitly zeros gradients for ignored positions.
   - `_SwiGLU` custom autograd correctly computes per-tensor gradients through the `silu(a) * b` interaction.
   - `_LayerScaleResidual` correctly routes gradients: `grad_residual = grad_out * scale`, `grad_scale = sum(grad_out * residual)`.

3. **Cross-entropy implementation**: The `_valid_target_mask` correctly handles `uint8` targets by casting to `int64` before comparison with negative `ignore_index` — prevents silent incorrect behavior.

### Minor Observations

1. **Weight caching**: `_cast_weight_cached` uses `(data_ptr, _version, dt, device)` as key with byte budget. Correctly handles in-place weight updates via version check.

2. **RoPE implementation**: `_ApplyRoPEOne` uses in-place rotation with `mark_dirty` for autograd correctness. The `_apply_rope_inplace` correctly preserves the original `lo` row via scratch buffer before updating `hi`.

3. **Depth reparameterization**: `shared_depth_schedule` correctly shares `(alpha_0, k)` across layers for genuine cross-layer depth law rather than per-layer independent parameters.

## Methodology Assessment

- **DyT family design**: The normalization family (`dyt`, `dyts`, `dytp`, `dytc`, `dytd`, `dytcd`, etc.) systematically varies activation function, depth coupling, and learnable parameters — allows clean factorial analysis.
- **QK-squash**: Couples Q and K normalization through shared parameters — tests whether independent vs coupled normalization matters.
- **Fused cross-entropy**: Memory-efficient computation via chunked log_softmax + nll_loss. Backward correctly reconstructs `softmax - onehot` gradient with ignore masking.

## Verdict
✅ No code changes required. Solid implementation with correct gradient flow.
