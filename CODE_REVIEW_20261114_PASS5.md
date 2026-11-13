# Code Review Report - ada-squash
## Pass 5: Independent Deep Review & Bug Fix

**Date:** 2026-11-14 04:00 UTC
**Reviewer:** QwenGo Independent Review System (Pass 5)
**Scope:** Full independent codebase review for bugs and research methodology issues

---

## Summary

### Verdict: BUGS FOUND AND FIXED in filedrawer.py

This pass performed a fully independent deep review of all 97 Python files,
verifying previous review conclusions and identifying additional issues that
earlier passes missed.

### Issues Found and Fixed

#### 1. Systematic bias from `(n + 1e-8)` denominators in meta-analysis (METHODOLOGY)

**File:** `adasquash/filedrawer.py`
**Severity:** Medium (methodology correctness)

**Problem:** Multiple functions used `n + 1e-8` as a divisor where `n` is an
integer count already guarded by early-return checks (e.g., `if len(data) < 3:
return`). This introduces a systematic downward bias in mean estimates,
variances, and covariances. While the relative bias per sample is ~1e-8/n
(negligible for large n), it is conceptually wrong for publication-quality
meta-analysis code: the denominator should be exactly n when n is known to be
non-zero.

**Locations fixed:**
- `funnel_plot_asymmetry()`: mean_eff, mean_prec, cov, var_eff, var_prec
  (5 divisions corrected)
- `eggers_test()`: mean_eff, mean_prec, se_intercept computation
  (3 divisions corrected)
- `p_curve_analysis()`: mean_p computation (1 division corrected)

**Fix:** Replaced `(n + 1e-8)` with `n` in all locations where n is guaranteed
non-zero by upstream guards.

#### 2. Egger's test used normal approximation instead of t-distribution (METHODOLOGY)

**File:** `adasquash/filedrawer.py`, `eggers_test()`
**Severity:** Medium (statistical correctness)

**Problem:** The Egger's regression test computed p-values using the standard
normal approximation (`math.erf`) instead of the correct t-distribution with
n-2 degrees of freedom. For small sample sizes (n < 30), the normal
approximation can produce materially incorrect p-values.

**Fix:** Imported `student_t_cdf` from `adasquash.metrics_stats` (which
implements an accurate incomplete-beta-based t-CDF) and replaced:
```python
# Before (incorrect):
p = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

# After (correct):
df = n - 2
p = 2 * (1 - student_t_cdf(abs(t_stat), df))
```

#### 3. Funnel plot correlation p-value used incorrect distribution (METHODOLOGY)

**File:** `adasquash/filedrawer.py`, `generate_file_drawer_report()`
**Severity:** Medium (statistical correctness)

**Problem:** The funnel plot asymmetry p-value was computed by treating the
Pearson correlation coefficient directly as a z-score:
```python
funnel_p = 2 * (1 - 0.5 * (1 + math.erf(abs(r) / math.sqrt(2))))
```
This is statistically incorrect: the correlation coefficient r does not follow
a standard normal distribution. The correct test uses t = r*sqrt((n-2)/(1-r^2))
with df = n-2.

**Fix:** Replaced with proper t-test for correlation significance:
```python
t_corr = abs(r) * math.sqrt((fn - 2) / (1.0 - r * r))
funnel_p = 2 * (1 - student_t_cdf(t_corr, fn - 2))
```

### Verification

All 651 existing tests pass after the fixes.

---

## Previous Pass Conclusions Verified

| Check Category | Previous Verdict | Pass 5 Verification |
|---|---|---|
| Loss computation | ✅ Correct | ✅ Confirmed |
| Attention mechanism | ✅ Correct | ✅ Confirmed |
| Statistical tests (core) | ✅ Sound | ✅ Confirmed |
| Decision framework | ✅ Sound | ✅ Confirmed |
| Data pipeline | ✅ No leakage | ✅ Confirmed |
| Reproducibility | ✅ Adequate | ✅ Confirmed |

## Scanner False Positives Confirmed

- **eval() usage:** No actual `eval()` builtin calls found. All scanner hits are
  function names containing "eval" (evaluate, eval_accuracy, model.eval(), etc.)
- **Division by zero:** All flagged divisions in core code are properly guarded
  with `max(n, 1)`, early returns, or epsilon guards appropriate to context.
- **Data leakage from shuffle:** Flagged locations are test files or config
  variable names, not actual data shuffling operations.

## Conclusion

The ada-squash codebase is well-engineered with careful attention to numerical
stability and statistical methodology. This pass identified and fixed 3
methodology issues in the publication bias analysis module (filedrawer.py)
that previous passes had not caught. All fixes have been verified against
the existing 651-test suite.
