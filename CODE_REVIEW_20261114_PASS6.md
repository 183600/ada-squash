# Code Review Report - ada-squash
## Pass 6: Independent Deep Source Code Audit

**Date:** 2026-11-14 05:00 UTC
**Reviewer:** QwenGo Independent Review System (Pass 6)
**Scope:** Full independent source code audit — focus on attention computation,
custom autograd functions, normalization layers, statistical methods, and
publication bias analysis.

---

## Summary

### Verdict: PREVIOUS FIXES VERIFIED — No new issues found

This pass performed direct source-code reading of all core modules, with
particular focus on:
- `adasquash/attention.py` — causal self-attention, QK-norm, RoPE
- `adasquash/norms.py` — `_RMSNorm`, `_GeneralizedSoftsign`, `_DyTCForward`,
  `QKSquash` and related autograd functions
- `adasquash/filedrawer.py` — publication bias tests (Egger's, Begg's,
  funnel plot, p-curve) — verifying Pass 5 fixes
- `adasquash/metrics_stats.py` — `student_t_cdf`, `student_t_ppf`,
  incomplete beta, chi-squared survival function

---

## Detailed Checks Performed

### 1. _RMSNorm Backward Pass (norms.py:425-475)
**Status:** ✅ CORRECT

Manual verification of the backward formula:

```
RMSNorm: out_j = x_j * rstd * w_j
where rstd = 1/sqrt(mean(x²) + eps)
```

The correct gradient is:
```
grad_x_j = rstd * w_j * (grad_out_j - x_j * rstd² * mean(grad_out * w * x))
```

The code computes:
```python
u = grad_out * rstd          # grad_out / sqrt(...)
t = u * x                    # grad_out * x / sqrt(...)
m = mean(t * weight)         # mean(grad_out * x * w / sqrt(...))
u = u * w                    # grad_out * w / sqrt(...)
corr = m * rstd²             # mean(...) * rstd²
grad_x = u - x * corr        # correct!
```

Expanding: `grad_x = w * grad_out * rstd - x * mean(grad_out * w * x * rstd) * rstd²`
which matches the analytical formula. ✅

### 2. _GeneralizedSoftsign (norms.py:363-395)
**Status:** ✅ CORRECT

- Forward: `phi_p(x) = x / (1 + |x|^p)^(1/p)` — the generalized softsign.
- Backward: Separate computation of `grad_x` and `grad_p` using the
  `_softsign_backward_pieces` helper. The `p` gradient is only computed
  when `ctx.need_p_grad` is True (set by `_save_exponent` based on whether
  `p` is a learnable parameter or a frozen float).
- Exponent storage: `_save_exponent` / `_restore_exponent` correctly handles
  both tensor and scalar exponents, with the `need_p_grad` flag preventing
  gradient computation on frozen exponents.

### 3. filedrawer.py — Pass 5 Fix Verification
**Status:** ✅ ALL THREE FIXES CORRECT

#### Fix 1: `(n + 1e-8)` → `n` denominators
Verified in `funnel_plot_asymmetry()` and `eggers_test()`:
- `mean_eff = sum(effects) / n` ✅ (was `/ (n + 1e-8)`)
- `mean_prec = sum(precisions) / n` ✅
- `cov = ... / n` ✅
- All 9 instances corrected. The early-return guard `len(data) < 3` ensures
  `n >= 3`, so `n` is always non-zero.

#### Fix 2: Egger's test normal → t-distribution
Verified in `eggers_test()`:
```python
df = n - 2
p = 2 * (1 - student_t_cdf(abs(t_stat), df))
```
✅ Correct — Egger's regression uses t(n-2), not N(0,1). The `student_t_cdf`
function is a high-quality implementation using the regularized incomplete beta
function for small df and the Cornish-Fisher normal expansion for large df.

#### Fix 3: Funnel plot correlation p-value
Verified in `generate_file_drawer_report()`:
```python
t_corr = abs(r) * math.sqrt((fn - 2) / (1.0 - r * r))
funnel_p = 2 * (1 - student_t_cdf(t_corr, fn - 2))
```
✅ Correct — tests `H₀: ρ = 0` using `t = r·√((n-2)/(1-r²))` with df = n-2.

### 4. student_t_cdf (metrics_stats.py:850-900)
**Status:** ✅ CORRECT

- Small df: Uses `_reg_incomplete_beta(0.5, 0.5*df, x)` with the identity
  `P(T ≤ t) = 1 - 0.5 * I_x(df/2, 1/2)` where `x = df/(df+t²)`.
- Large df: Switches to Cornish-Fisher normal expansion with O(1/df)
  correction term `(t³ + t)/(4·df)`, preventing the incomplete-beta
  accuracy loss at large df.
- Edge cases: `±inf` handled; `x > 0.5` switches to complementary beta
  for numerical stability; `NaN` and non-finite df properly rejected.

### 5. Attention Module (attention.py)
**Status:** ✅ CORRECT

- Uses PyTorch's `F.scaled_dot_product_attention` with `is_causal=True` —
  delegates causal masking to the optimized SDPA implementation.
- QK-norm options: `qknorm` (RMSNorm per head) and `dytqk` (Dynamic Tanh
  per-channel) — both properly applied before RoPE.
- Logit statistics collected in `torch.no_grad()` context, preventing
  diagnostic computation from affecting the training graph.

## Conclusion

The ada-squash codebase maintains high quality with correct custom autograd
implementations and sound statistical methods. The three Pass 5 fixes in
filedrawer.py are verified correct. No new bugs or methodology errors were
found. No code changes were necessary.
