## Code Review 2026-11-26 (Sixth Round)

### Summary

Round 6 deep code review of the ada-squash repository. Focused on:

1. **Attention mechanism**: Verified QK-norm, RoPE, scaled dot-product
   attention, and causal masking.
2. **Cross-entropy implementation**: Reviewed chunked and fused
   linear cross-entropy custom autograd functions.
3. **QK-Squash normalization**: Verified the dynamic QK normalization
   implementation.
4. **Logit statistics**: Reviewed the causal triangle logit sweep
   for numerical correctness.
5. **Ablation framework**: Checked ablation matrix design and
   contrast analysis.

### Findings

**No bugs found.** All automated warnings were verified as false
positives:

- **In-place `.zero_()`** (baseline_simple.py:319): Gradient zeroing
  in optimizer context, not in forward pass.
- **`model.eval()` without restore** (eval_downstream.py:112):
  Single-purpose evaluation script; model not reused after.

### Methodology Assessment

- **Attention**: Uses `F.scaled_dot_product_attention(q, k, v,
  is_causal=True, dropout_p=..., scale=scale)` — PyTorch's optimized
  implementation with correct causal masking.
- **RoPE**: Orthogonal rotation with proper forward/backward via
  custom `_ApplyRoPE` autograd function. Transpose rotation for
  cotangents is mathematically correct (R^T = R(-theta)).
- **Chunked cross-entropy**: Custom `_ChunkedCrossEntropy` correctly
  handles `ignore_index` masking, computes `log_softmax` in row
  blocks, and returns `None` gradients for ignored rows.
- **Fused linear CE**: `_FusedLinearCrossEntropy` performs GEMM in
  accumulator dtype with proper row masking. Gradient computation
  matches forward semantics.
- **Logit statistics**: Variance computed in float64 with proper
  NaN/Inf handling. Returns `nan` for negative variance (numerical
  artifact) rather than silently reporting zero spread.

### Conclusion

The ada-squash codebase implements clean, numerically sound attention
with proper QK normalization and efficient cross-entropy computation.
No changes required.
