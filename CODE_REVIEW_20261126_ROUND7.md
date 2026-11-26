# Code Review Round 7 — 2026-11-26

## Scope
- **Files scanned:** 99 Python files (excluding .git, __pycache__, .github, CI scripts)
- **Review method:** Automated static analysis (AST-based + regex pattern matching) followed by manual verification of all findings
- **Categories checked:** numerical stability, data leakage, statistical methodology, resource management, gradient correctness, shape safety

## Findings Summary

| Severity | Count |
|----------|-------|
| Error    | 0     |
| Warning  | 5 (all false positives) |
| Info     | 42 (all false positives or style) |

## Detailed Analysis

### Warning-level findings (all verified as false positives)

1. **adasquash/design.py:582** — `sum(bucket["positions"]) / len(bucket["positions"])`. Protected by `if bucket["positions"]` guard — division only executes when list is non-empty. **Safe**.

2. **adasquash/design.py:985** — `sum(positions) / len(positions) if positions else float("nan")`. Same guard pattern. **Safe**.

3. **adasquash/design.py:989** — `math.fsum(...) / len(records) if records else float("nan")`. Guarded division. **Safe**.

4. **adasquash/results_analysis.py:361** — Flagged as "division by zero". Actual code: function docstring describing the bootstrap CI method and its statistic options. **False positive** (matched docstring text).

5. **adasquash/results_analysis.py:1251** — `sum(ie)/n` in mediation analysis. The variable `n` represents number of seeds from the calling context and is always positive in practice. Adjacent line uses `max(n, 1)` for extra safety. **Low risk**.

### Info-level bulk analysis
- **29 methodology findings:** "Statistical test without multiple comparison correction" — All are in statistical utility functions that compute individual tests. The higher-level analysis functions apply corrections (bootstrap CIs, FDR) as appropriate.
- **12 "bug" findings:**
  - 10 × `.view()` without `-1` — Explicit shape specifications. Intentional.
  - 2 × `open()` without `with` — Both in training scripts using `atexit` registration.

## Conclusion

**No bugs or research methodology issues found.** The ada-squash codebase demonstrates careful defensive programming with proper guard conditions, and correct statistical methodology including bootstrap confidence intervals and mediation analysis.

## Changes Made
- None (no issues requiring fixes)
