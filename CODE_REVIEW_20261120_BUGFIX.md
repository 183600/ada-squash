# Code Review Report — ada-squash (Bug Fix)
**Date:** 2026-11-20
**Reviewer:** Independent Deep Review Agent
**Scope:** Full codebase (70 Python files) + Statistical Methods

## Bugs Found and Fixed

### BUG 1 (CRITICAL): `math.log(p, 0.0)` in `_approx_normal_ppf` — ValueError crash
- **File:** `adasquash/assumptions.py:164`
- **Original:** `t = math.sqrt(max(-2 * math.log(p, 0.0)))`
- **Problem:** `math.log(x, 0.0)` raises `ValueError: math domain error` because logarithm base must be > 0 and ≠ 1. The second argument to `math.log()` is the base; `0.0` is invalid.
- **Fix:** `t = math.sqrt(max(-2.0 * math.log(p), 0.0))` — use natural log (single-arg) and guard against negative results.
- **Impact:** `_approx_normal_ppf` is called by `_shapiro_wilk_statistic` (line 124) for computing normal scores. This crash would propagate to `check_normality()`, preventing all normality assumption checks.

### BUG 2 (CRITICAL): `math.log(W2 + 1e-8, 0.0)` in `_dagostino_pearson` — ValueError crash
- **File:** `adasquash/assumptions.py:193`
- **Original:** `delta = 1.0 / math.sqrt(max(math.log(W2 + 1e-8, 0.0))) if W2 > 1 else 1.0`
- **Problem:** Same `math.log(x, 0.0)` pattern. The D'Agostino (1970) skewness Z1 transform requires δ = 1/√(ln W), using natural log.
- **Fix:** `delta = 1.0 / math.sqrt(max(math.log(W2 + 1e-8), 0.0)) if W2 > 1 else 1.0`
- **Impact:** `_dagostino_pearson` is called by `check_normality()` (lines 249, 256). This crash would prevent all normality assumption checks via the D'Agostino-Pearson omnibus test.

### BUG 3 (SIGN ERROR): `_approx_normal_ppf` returned wrong sign for p < 0.5
- **File:** `adasquash/assumptions.py:167-168`
- **Original:** `return t - (c0 + c1*t + c2*t²) / (1 + d1*t + d2*t² + d3*t³)` (positive value)
- **Problem:** The Abramowitz & Stegun formula 26.2.23 gives x where Q(x) = p (complementary CDF). For p < 0.5, ppf(p) = -x (negative), but the code returned +x.
- **Fix:** Split into `x = formula; return -x`
- **Validation:** After fix, ppf(0.1) = -1.2817 (correct: scipy gives -1.2816), ppf(0.9) = +1.2817 (correct).

### BUG 4 (CODE QUALITY): Redundant duplicate validation in `CausalSelfAttention.__init__`
- **File:** `adasquash/attention.py:60-66`
- **Original:** Two identical `if dim % n_heads != 0` checks separated by `self.n_heads = n_heads`
- **Problem:** Dead code — the second check can never trigger because the first already raises ValueError.
- **Fix:** Merged into single check with the more informative error message.

## Verification

All fixes validated:
```
ppf(0.05) = -1.6452 (expected -1.6449) ✓
ppf(0.10) = -1.2817 (expected -1.2816) ✓
ppf(0.25) = -0.6742 (expected -0.6745) ✓
ppf(0.50) = +0.0000 (expected +0.0000) ✓
ppf(0.75) = +0.6742 (expected +0.6745) ✓
ppf(0.90) = +1.2817 (expected +1.2816) ✓
ppf(0.95) = +1.6452 (expected +1.6449) ✓
```

## Static Analysis Review (33 flags — all false positives after fixes)

- **eval_call (11):** All `model.eval()` correctly placed before evaluation phases.
- **unprotected_division (12):** All divisions guarded with `clamp_min`, conditionals, or `float("nan")`.
- **no_grad_context_missing (4):** All `backward()` in proper training or `enable_grad()` contexts.
- **todo_fixme (1):** Design documentation comment, not a code defect.

## Research Methodology Assessment

- **Attention mechanism:** CausalSelfAttention with proper RoPE, QK normalization options, fused linear-CE.
- **Loss computation:** Custom chunked cross-entropy autograd function with proper `ignore_index` masking and gradient flow.
- **Statistical methods:** Holm and BH corrections, factorial design, decontamination with n-gram matching.
- **Training pipeline:** Proper gradient accumulation, clipping, non-finite loss detection.

## Conclusion

Four bugs fixed (2 critical crashes, 1 sign error, 1 code quality). The codebase is otherwise well-engineered with sound statistical methodology.
