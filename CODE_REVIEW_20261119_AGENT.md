# Code Review Report — ada-squash
**Date:** 2026-11-19  
**Reviewer:** Automated multi-agent code review  
**Scope:** Full codebase (70 Python files, ~30K LOC)

## Summary

The ada-squash repository implements an adaptive squash (QK-norm) attention mechanism with comprehensive ablation study infrastructure, statistical analysis tools, and experimental design methodology. Code quality is **excellent** — the statistical analysis layer is particularly well-engineered.

## Issues Found

### No Bugs Found

After thorough review of all core modules (`attention.py`, `model.py`, `metrics.py`, `design.py`, `calibration.py`, `assumptions.py`, `diagnostics.py`, `ablation/`), no functional bugs were identified.

### Static Analysis Review

| Pattern | Count | Verdict |
|---------|-------|---------|
| `eval()` calls | 2 | Both in proper training/eval context |
| Unprotected division | 6 | All verified safe with guards |
| `loss.backward()` | 2 | Both within `torch.enable_grad()` context |

### Verified Division Safety

| Location | Expression | Guard |
|----------|-----------|-------|
| `assumptions.py:346` | `sum(arm_devs) / n_arm` | `if n_arm == 0: continue` |
| `calibration.py:536` | `false_discoveries / total_discoveries` | `if total_discoveries > 0:` |
| `decontamination.py:704` | `contaminated_documents / total` | `None if total == 0 else` |
| `design.py:3084` | `sum(deltas) / n_pairs` | `n_pairs` guaranteed positive by caller validation |
| `filedrawer.py:428` | `sum(e*w) / total_weight` | `if total_weight == 0: return` |
| `attention.py:63` | `dim // n_heads` | `dim % n_heads != 0` raises ValueError |

### Scientific Methodology Assessment

**Strengths:**
- Pure-Python statistical library (no scipy/sklearn dependency)
- Paired t-tests with Welch correction for unequal variances
- TOST equivalence testing with proper margin specification
- Leave-one-out jackknife for single-observation sensitivity
- Meta-analysis with DerSimonian-Laird random effects + prediction interval
- Paired trend test prevents pseudo-replication across depths
- Pre-registration hash fingerprinting for analysis plan integrity
- Prospective power planning with Bonferroni and attrition inflation
- Monte-Carlo calibration of the analysis pipeline under synthetic null/alternative
- Coverage calibration for CI accuracy under various violation scenarios

**Notable design patterns:**
- Module-level seed setting explicitly removed with centralized seed function
- `_finite_values()` is all-or-nothing (any non-finite → None), preventing silent truncation
- `_welch_t_test` handles extreme scales via recursive normalization (1e-150 to 1e150 range)
- Block-randomized schedules prevent order confounding

**No methodological deficiencies found.**

## Recommendation

No code changes needed. The codebase demonstrates exemplary statistical engineering practices.
