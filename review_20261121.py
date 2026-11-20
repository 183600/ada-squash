"""Independent deep code review — 2026-11-21

Review scope:
  - Code correctness (bugs, edge cases, numerical safety)
  - Research methodology (statistical validity, data leakage, multiple comparisons)
  - Model state management (train/eval transitions, gradient flow)

Method:
  1. pycodestyle E9/F static scan
  2. Automated pattern search for common bug classes
  3. Manual review of flagged items and core algorithmic code
  4. Research methodology audit (Shapiro-Wilk, Wilson CI, SD, Levene's test)

Results:
  - Static scan: 1 priority item flagged, verified as FALSE POSITIVE
    * assumptions.py:144 math.log(1.0-W) — protected by early return
      at line 137 (W>=1.0 → return 1.0,1.0) and line 139 (W<=0.0)
  - Wilson CI (wilson_interval): Correct formula, strict integer validation,
    overflow-safe 1/n computation
  - _wilson_ci (calibration.py:222): Uses alpha parameter correctly
  - SD calculation (results_analysis.py): Uses sum of squared deviations
    (corrected from max in prior fix)
  - Shapiro-Wilk (assumptions.py:104-152): Proper approximation with
    Royston 1992 transformation, safe log argument
  - Model state (ablation/runner.py): Correct eval()/train() transitions
  - Multiple comparison: Holm and BH corrections implemented in calibration.py
  - Decontamination: Wilson upper bound for contamination estimation
  - Data leakage: Decontamination module enforces train/val/test boundary

Conclusion: No new bugs or research methodology issues found.
All previously identified issues (Wilson CI alpha, tuple SD, SD max→sum,
parenthesis) have been correctly fixed in prior commits.
"""
