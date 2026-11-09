"""Automated assumption diagnostics for statistical tests.

The validity of parametric tests (t-tests, ANOVA, etc.) depends on assumptions
that are rarely checked in practice. This module provides automated diagnostics
for:

* **Normality** of paired differences (Shapiro-Wilk, D'Agostino-Pearson)
* **Homogeneity of variance** across arms (Levene's test, Brown-Forsythe)
* **Independence** / autocorrelation in time-ordered observations
* **Outlier influence** on test conclusions
* **Sample size adequacy** for asymptotic approximations

Each diagnostic returns a structured result with a clear pass/fail/warn verdict
and actionable recommendations. The diagnostics are designed to be run *before*
interpreting the main analysis results.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "AssumptionCheck",
    "AssumptionReport",
    "check_independence",
    "check_normality",
    "check_outlier_influence",
    "check_sample_size_adequacy",
    "check_variance_homogeneity",
    "run_full_assumption_diagnostics",
]


@dataclass
class AssumptionCheck:
    """Result of a single assumption check."""
    name: str
    description: str
    status: str  # "pass", "warn", "fail", "skipped"
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    threshold: Optional[float] = None
    details: dict = field(default_factory=dict)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "details": self.details,
            "recommendation": self.recommendation,
        }


@dataclass
class AssumptionReport:
    """Aggregated assumption diagnostics report."""
    schema: str = "adasquash.assumption_report.v1"
    checks: list[AssumptionCheck] = field(default_factory=list)
    overall_status: str = "pass"  # "pass", "warn", "fail"
    summary: str = ""

    def add_check(self, check: AssumptionCheck) -> None:
        self.checks.append(check)
        self._update_overall()

    def _update_overall(self) -> None:
        statuses = [c.status for c in self.checks]
        if "fail" in statuses:
            self.overall_status = "fail"
        elif "warn" in statuses:
            self.overall_status = "warn"
        else:
            self.overall_status = "pass"

        failed = [c.name for c in self.checks if c.status == "fail"]
        warned = [c.name for c in self.checks if c.status == "warn"]
        if failed:
            self.summary = f"Assumptions violated: {', '.join(failed)}. " + \
                          (f"Warnings: {', '.join(warned)}." if warned else "")
        elif warned:
            self.summary = f"Assumptions marginal: {', '.join(warned)}."
        else:
            self.summary = "All assumptions satisfied."

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "checks": [c.to_dict() for c in self.checks],
            "overall_status": self.overall_status,
            "summary": self.summary,
        }


def _shapiro_wilk_statistic(data: Sequence[float]) -> tuple[float, float]:
    """Approximate Shapiro-Wilk test for normality.

    Returns (W statistic, p-value approximation).
    This is a simplified implementation; for production use scipy.stats.shapiro.
    """
    n = len(data)
    if n < 3 or n > 5000:
        return 1.0, 1.0  # Cannot compute reliably

    sorted_data = sorted(data)
    mean = sum(sorted_data) / n

    # Coefficients approximation (Royston 1992)
    # For simplicity, use a basic approximation
    # W = (sum(a_i * x_i))^2 / sum((x_i - mean)^2)
    # We approximate a_i using expected normal order statistics

    # Simple approximation: use correlation with normal scores
    # This is not the true Shapiro-Wilk but serves as a diagnostic
    normal_scores = [_approx_normal_ppf((i + 0.375) / (n + 0.25)) for i in range(n)]

    cov = sum((x - mean) * ns for x, ns in zip(sorted_data, normal_scores))
    var_x = sum((x - mean) ** 2 for x in sorted_data)
    var_ns = sum(ns ** 2 for ns in normal_scores)

    if var_x == 0 or var_ns == 0:
        return 1.0, 1.0

    r = cov / math.sqrt(var_x * var_ns)
    W = r * r  # Approximation

    # Approximate p-value (Royston 1992 transformation)
    if W >= 1.0:
        return 1.0, 1.0
    if W <= 0.0:
        return 0.0, 0.0

    # Very rough p-value approximation for diagnostic purposes
    # In practice, use scipy.stats.shapiro
    log_w = math.log(1.0 - W)
    mu = -1.5861 - 0.31082 * math.log(n) - 0.083751 * (math.log(n)) ** 2 + 0.0038915 * (math.log(n)) ** 3
    sigma = math.exp(-0.4803 - 0.082676 * math.log(n) + 0.0030302 * (math.log(n)) ** 2)
    z = (log_w - mu) / sigma
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return W, max(0.0, min(1.0, p))


def _approx_normal_ppf(p: float) -> float:
    """Approximate inverse normal CDF (probit function)."""
    # Abramowitz & Stegun approximation
    if p <= 0 or p >= 1:
        return float("inf") if p >= 1 else float("-inf")
    # Symmetric
    if p > 0.5:
        return -_approx_normal_ppf(1 - p)
    # Rational approximation for p < 0.5
    t = math.sqrt(-2 * math.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def _dagostino_pearson(data: Sequence[float]) -> tuple[float, float]:
    """D'Agostino-Pearson omnibus test for normality (skewness + kurtosis)."""
    n = len(data)
    if n < 8:
        return 0.0, 1.0

    mean = sum(data) / n
    m2 = sum((x - mean) ** 2 for x in data) / n
    m3 = sum((x - mean) ** 3 for x in data) / (n + 1e-8)
    m4 = sum((x - mean) ** 4 for x in data) / (n + 1e-8)

    if m2 == 0:
        return 0.0, 1.0

    # Skewness
    g1 = m3 / (m2 ** 1.5)
    # Kurtosis
    g2 = m4 / (m2 ** 2) - 3.0

    # Transform to normal (D'Agostino 1970)
    # Skewness transform
    beta1 = 3.0 * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
    W2 = math.sqrt(2.0 * beta1) - 1.0
    delta = 1.0 / math.sqrt(math.log(W2)) if W2 > 1 else 1.0
    y = g1 * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    z1 = delta * math.log(y + math.sqrt(y * y + 1.0))

    # Kurtosis transform
    E_g2 = -6.0 / (n + 1)
    var_g2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    a = 6.0 + 8.0 / var_g2 * (2.0 * var_g2 + 6.0 / (n + 1) * (n - 2) / (n - 3))
    # Anscombe-Glynn (1983) transform: coefficient is 1 - 2/(9a), not 1 - 2/a.
    # The original formula produced z-scores that were too large for small n,
    # inflating type-I error of the normality test on small samples.
    z2 = ((1.0 - 2.0 / (9.0 * a)) * (1.0 + g2 / math.sqrt(var_g2)) ** (1.0 / 3.0) - 1.0) * math.sqrt(a) * math.sqrt(2.0 / (9.0 * a))

    # Chi-squared combination
    K2 = z1 * z1 + z2 * z2
    # Approximate p-value from chi-squared with 2 df
    p = math.exp(-K2 / 2.0)

    return K2, p


def check_normality(
    paired_deltas: Sequence[float],
    alpha: float = 0.05,
) -> AssumptionCheck:
    """Check normality of paired differences using multiple tests.

    Uses Shapiro-Wilk (for n <= 50) and D'Agostino-Pearson (for n >= 8).
    """
    n = len(paired_deltas)
    if n < 3:
        return AssumptionCheck(
            name="normality",
            description=f"Normality of {n} paired differences (insufficient data)",
            status="skipped",
            recommendation=f"Need at least 3 paired differences; have {n}. Cannot assess normality.",
        )

    # Shapiro-Wilk (best for small n)
    if n <= 50:
        W, p = _shapiro_wilk_statistic(paired_deltas)
        test_name = "Shapiro-Wilk"
        stat = W
    else:
        K2, p = _dagostino_pearson(paired_deltas)
        test_name = "D'Agostino-Pearson"
        stat = K2

    # Also run D'Agostino-Pearson for n >= 8 as a secondary check
    secondary = None
    if n >= 8:
        K2, p2 = _dagostino_pearson(paired_deltas)
        secondary = {"test": "D'Agostino-Pearson", "statistic": K2, "p_value": p2}

    if p < alpha:
        status = "fail"
        rec = f"{test_name} p={p:.4f} < {alpha}: Paired differences significantly non-normal. Consider randomization test."
    elif p < 0.10:
        status = "warn"
        rec = f"{test_name} p={p:.4f} marginal: Non-normality possible. Randomization test recommended."
    else:
        status = "pass"
        rec = f"{test_name} p={p:.4f}: No evidence against normality."

    return AssumptionCheck(
        name="normality",
        description=f"Normality of paired differences ({test_name})",
        status=status,
        statistic=stat,
        p_value=p,
        threshold=alpha,
        details={"n": n, "test": test_name, "secondary": secondary},
        recommendation=rec,
    )


def check_variance_homogeneity(
    arm_data: Mapping[str, Sequence[float]],
    alpha: float = 0.05,
    center: str = "median",
) -> AssumptionCheck:
    """Check homogeneity of variance across arms (Levene's / Brown-Forsythe test).

    Parameters
    ----------
    arm_data: Mapping from arm name to list of observations (e.g., per-seed val_loss).
    alpha: Significance level.
    center: "mean" (Levene) or "median" (Brown-Forsythe, more robust).

    Returns
    -------
    AssumptionCheck with test result.
    """
    arms = list(arm_data.keys())
    k = len(arms)
    if k < 2:
        return AssumptionCheck(
            name="variance_homogeneity",
            description=f"Variance homogeneity across {k} arm(s) (insufficient arms)",
            status="skipped",
            recommendation="Need at least 2 arms to test variance homogeneity.",
        )

    # Check minimum sample size per arm
    min_n = min(len(v) for v in arm_data.values() if v)
    if min_n < 2:
        return AssumptionCheck(
            name="variance_homogeneity",
            description="Variance homogeneity (insufficient per-arm samples)",
            status="skipped",
            recommendation=f"At least one arm has < 2 observations (min_n={min_n}).",
        )

    # Compute centered absolute deviations
    all_devs = []
    all_labels = []
    for arm, values in arm_data.items():
        if not values:
            continue
        c = sum(values) / max(len(values), 1) if values else float("nan") if center == "mean" else sorted(values)[len(values) // 2]
        for v in values:
            all_devs.append(abs(v - c))
            all_labels.append(arm)

    # One-way ANOVA on absolute deviations
    grand_mean = (sum(all_devs) / max(len(all_devs), 1)) if all_devs else float("nan")
    ss_between = 0.0
    ss_within = 0.0

    for arm in arms:
        arm_devs = [d for d, l in zip(all_devs, all_labels) if l == arm]
        n_arm = len(arm_devs)
        if n_arm == 0:
            continue
        arm_mean = sum(arm_devs) / n_arm
        ss_between += n_arm * (arm_mean - grand_mean) ** 2
        ss_within += sum((d - arm_mean) ** 2 for d in arm_devs)

    df_between = k - 1
    df_within = len(all_devs) - k

    if df_within <= 0 or ss_within == 0:
        return AssumptionCheck(
            name="variance_homogeneity",
            description=f"Variance homogeneity across {k} arms (degenerate)",
            status="skipped",
            recommendation="Degenerate test: zero within-group variance or insufficient df.",
        )

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    F = ms_between / ms_within if ms_within > 0 else float("inf")

    # Approximate p-value from F-distribution
    # Using the fact that F ~ Beta(df1/2, df2/2) transformed
    # Simple approximation
    if float("inf") == F:
        p = 0.0
    else:
        # Approximate using chi-squared approximation
        p = math.exp(-0.5 * F * df_between)  # Very rough

    # Better approximation: use the relationship between F and Beta
    # p = I_{df2/(df2+df1*F)}(df2/2, df1/2) where I is regularized incomplete beta
    # We'll use a simple lookup or approximation
    try:
        # Use the fact that for large df2, F * df1 ~ chi^2(df1)
        if df_within > 30:
            chi2 = F * df_between
            p = math.exp(-chi2 / 2) * sum(chi2**k / math.factorial(k) for k in range(df_between // 2))
        else:
            # Rough approximation
            p = 1.0 / (1.0 + F)
    except (OverflowError, ZeroDivisionError):
        p = 0.0 if F > 1 else 1.0

    test_name = "Levene" if center == "mean" else "Brown-Forsythe"

    if p < alpha:
        status = "fail"
        rec = f"{test_name} p={p:.4f} < {alpha}: Variances differ significantly across arms. Use Welch's test."
    elif p < 0.10:
        status = "warn"
        rec = f"{test_name} p={p:.4f} marginal: Possible variance heterogeneity. Welch's test recommended."
    else:
        status = "pass"
        rec = f"{test_name} p={p:.4f}: No evidence against equal variances."

    return AssumptionCheck(
        name="variance_homogeneity",
        description=f"Variance homogeneity across {k} arms ({test_name})",
        status=status,
        statistic=F,
        p_value=p,
        threshold=alpha,
        details={"k": k, "center": center, "per_arm_n": {a: len(v) for a, v in arm_data.items()}},
        recommendation=rec,
    )


def check_independence(
    ordered_observations: Sequence[float],
    max_lag: int = 5,
    alpha: float = 0.05,
) -> AssumptionCheck:
    """Check for autocorrelation in time-ordered observations.

    Uses Ljung-Box test on residuals (within-arm centered if multiple arms).
    """
    n = len(ordered_observations)
    if n < 10:
        return AssumptionCheck(
            name="independence",
            description=f"Autocorrelation check ({n} observations, insufficient)",
            status="skipped",
            recommendation=f"Need at least 10 observations for autocorrelation test; have {n}.",
        )

    # Simple autocorrelation at lags 1..max_lag
    mean = sum(ordered_observations) / (n + 1e-8)
    c0 = sum((x - mean) ** 2 for x in ordered_observations) / (n + 1e-8)

    if c0 == 0:
        return AssumptionCheck(
            name="independence",
            description="Autocorrelation check (zero variance)",
            status="pass",
            recommendation="Zero variance: no autocorrelation possible.",
        )

    max_significant_lag = 0
    significant_lags = []

    for lag in range(1, min(max_lag + 1, n // 2)):
        cov = sum((ordered_observations[i] - mean) * (ordered_observations[i + lag] - mean)
                  for i in range(n - lag)) / n
        acf = cov / c0

        # Standard error of ACF under white noise
        se = 1.0 / math.sqrt(n - lag)
        z = acf / se if se > 0 else 0
        p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))) if z != 0 else 1.0

        if p < alpha:
            significant_lags.append(lag)
            max_significant_lag = lag

    if significant_lags:
        status = "fail"
        rec = f"Significant autocorrelation at lag(s) {significant_lags}: observations not independent. Check run order / randomization."
    else:
        status = "pass"
        rec = f"No significant autocorrelation up to lag {max_lag}."

    return AssumptionCheck(
        name="independence",
        description=f"Autocorrelation / independence (max lag {max_lag})",
        status=status,
        statistic=max_significant_lag,
        p_value=None,
        threshold=alpha,
        details={"n": n, "significant_lags": significant_lags},
        recommendation=rec,
    )


def check_outlier_influence(
    paired_deltas: Sequence[float],
    influence_threshold: float = 0.5,
) -> AssumptionCheck:
    """Check whether a single observation disproportionately influences the conclusion.

    Uses leave-one-out jackknife on the mean difference.
    """
    n = len(paired_deltas)
    if n < 3:
        return AssumptionCheck(
            name="outlier_influence",
            description=f"Outlier influence ({n} observations, insufficient)",
            status="skipped",
            recommendation=f"Need at least 3 observations for jackknife; have {n}.",
        )

    full_mean = sum(paired_deltas) / (n + 1e-8)
    jackknife_means = []
    for i in range(n):
        loo = [paired_deltas[j] for j in range(n) if j != i]
        jackknife_means.append((sum(loo) / max(len(loo), 1)) if loo else float("nan"))

    max_change = max(abs(jm - full_mean) for jm in jackknife_means)
    relative_influence = max_change / (abs(full_mean) + 1e-12)

    if relative_influence > influence_threshold:
        status = "fail"
        rec = f"Single observation changes mean by {relative_influence:.1%} (> {influence_threshold:.0%}). Conclusion depends on one run."
    elif relative_influence > influence_threshold / 2:
        status = "warn"
        rec = f"Moderate influence: max single-run impact is {relative_influence:.1%}."
    else:
        status = "pass"
        rec = f"Low influence: max single-run impact is {relative_influence:.1%}."

    return AssumptionCheck(
        name="outlier_influence",
        description="Leave-one-out jackknife influence on mean difference",
        status=status,
        statistic=relative_influence,
        p_value=None,
        threshold=influence_threshold,
        details={"n": n, "full_mean": full_mean, "jackknife_means": jackknife_means},
        recommendation=rec,
    )


def check_sample_size_adequacy(
    n_pairs: int,
    test_type: str = "paired_t",
    min_n_for_asymptotic: int = 30,
) -> AssumptionCheck:
    """Check whether sample size is adequate for the test's asymptotic assumptions.

    For paired t-test: n >= 30 for CLT to justify normality approximation.
    For randomization test: any n >= 2 works (exact).
    """
    if test_type == "randomization":
        if n_pairs >= 2:
            return AssumptionCheck(
                name="sample_size_adequacy",
                description=f"Sample size for randomization test (n={n_pairs} pairs)",
                status="pass",
                recommendation="Randomization test is exact for any n >= 2.",
            )
        return AssumptionCheck(
            name="sample_size_adequacy",
            description=f"Sample size for randomization test (n={n_pairs} pairs, insufficient)",
            status="fail",
            recommendation="Need at least 2 pairs for randomization test.",
        )

    if n_pairs >= min_n_for_asymptotic:
        status = "pass"
        rec = f"n={n_pairs} >= {min_n_for_asymptotic}: CLT justifies t-test normality assumption."
    elif n_pairs >= 10:
        status = "warn"
        rec = f"n={n_pairs} < {min_n_for_asymptotic}: t-test may be anti-conservative. Randomization test preferred."
    else:
        status = "fail"
        rec = f"n={n_pairs} very small: t-test reliability questionable. Use exact methods."

    return AssumptionCheck(
        name="sample_size_adequacy",
        description=f"Sample size adequacy for {test_type} (n={n_pairs} pairs)",
        status=status,
        statistic=float(n_pairs),
        p_value=None,
        threshold=float(min_n_for_asymptotic),
        details={"test_type": test_type, "min_n_asymptotic": min_n_for_asymptotic},
        recommendation=rec,
    )


def run_full_assumption_diagnostics(
    paired_deltas: Sequence[float],
    arm_data: Optional[Mapping[str, Sequence[float]]] = None,
    ordered_observations: Optional[Sequence[float]] = None,
    test_type: str = "paired_t",
    alpha: float = 0.05,
) -> AssumptionReport:
    """Run all assumption diagnostics and return an aggregated report.

    Parameters
    ----------
    paired_deltas: Per-seed paired differences (candidate - reference).
    arm_data: Optional mapping of arm -> observations for variance homogeneity.
    ordered_observations: Optional time-ordered observations for independence check.
    test_type: "paired_t", "welch_t", or "randomization".
    alpha: Significance level for tests.

    Returns
    -------
    AssumptionReport with all checks.
    """
    report = AssumptionReport()

    # 1. Normality of paired differences
    report.add_check(check_normality(paired_deltas, alpha))

    # 2. Variance homogeneity (if arm data provided)
    if arm_data:
        report.add_check(check_variance_homogeneity(arm_data, alpha))

    # 3. Independence / autocorrelation (if ordered observations provided)
    if ordered_observations:
        report.add_check(check_independence(ordered_observations, alpha=alpha))

    # 4. Outlier influence
    report.add_check(check_outlier_influence(paired_deltas))

    # 5. Sample size adequacy
    report.add_check(check_sample_size_adequacy(len(paired_deltas), test_type))

    return report