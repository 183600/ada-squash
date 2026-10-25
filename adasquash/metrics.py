from __future__ import annotations
"""Uncertainty-aware evaluation metrics shared by the experiment scripts.

Why this module exists
----------------------
An experiment must not decide "variant X beats the baseline" on a bare point
estimate. This module keeps the three comparisons the repo actually makes
honest:

* a held-out loss is a *sample mean*: `mean_std_stderr` reports its standard
  error so a delta smaller than the noise is visible as such;
* two arms evaluated on the *same* held-out batches are paired samples, so
  `paired_t_one_sided` (with `welch_t_one_sided` as the unpaired fallback) is
  the test used for "better by more than `margin`". The repo replicates most
  comparisons over only 2-4 seeds, so the Student-t reference distribution
  matters: at n = 3 the one-sided 5% critical value is t = 2.92, while the
  normal approximation (`paired_z_one_sided` / `welch_z_one_sided`, kept for
  large-n callers) would use z = 1.64 and declare wins from noise. The two
  t-tests also report a two-sided confidence interval for the mean difference,
  so a point estimate without a plausible range is never the whole story;
* a pass/fail rate over independent trials (needle-in-a-haystack) gets a
  Wilson interval via `wilson_interval`, which stays inside [0, 1] and is not
  degenerate at 0/1 like the normal approximation;
* token classification over a large vocabulary is evaluated with
  `macro_weighted_f1` / `f1_from_counts`, which *name* the averaging they
  use: `macro_f1` (unweighted mean over classes present), `weighted_f1`
  (support-weighted) and `micro_f1` (= accuracy). Macro and micro can differ
  wildly under class imbalance, so the caller must state which one it
  reports;
* class-imbalanced binary/multi-label decisions are scored with
  `average_precision` (PR-AUC) and `roc_auc`, and `binary_auc_report` names
  the macro/weighted averaging and reports the negatives-per-positive
  imbalance. This matters because at >= 1000 negatives per positive the ROC
  curve's false-positive rate (denominator: all negatives) grows so slowly
  that `roc_auc` stays optimistic while precision has already collapsed;
  in that regime read PR-AUC (`average_precision`), not ROC-AUC;
  `imbalance_report` flags exactly that regime for a target distribution;
* a scan that tests many rows against one reference gets `holm_adjust`, so a
  row-by-row alpha is corrected for the family of tests actually performed
  (Holm-Bonferroni), instead of reporting the best of many uncorrected draws;
* a claim that two arms are *the same* is an equivalence claim, not a
  non-significant superiority test: `tost_paired` runs the two one-sided
  tests and reports the matching interval. A small-seed effect is also
  checked for being carried by a single observation with
  `leave_one_out_jackknife`, and a factorial interaction is evaluated
  two-sided (`two_sided_p_from_t`) because antagonism is a finding too.
  `paired_randomization_test` adds the exact sign-flip counterpart of the
  t-test: it makes no normality assumption and reports the smallest p-value
  attainable at the achieved seed count, so a 3-seed t-test's reliance on its
  parametric model is visible instead of implicit;
* a dose-response claim ("the advantage grows with depth") is tested as a
  trend, not inferred from a row of significant per-depth deltas:
  `paired_trend_test` fits one slope per seed on the paired deltas and tests
  the mean slope one-sided, so repeated depths within a seed are not
  pseudo-replicated and a flat advantage is not mistaken for growth;
* "the effect replicates across model scales" is not a sign vote. Each scale
  estimates the same contrast with a different precision, and the scales can
  disagree by more than their own standard errors. `meta_analysis` pools the
  per-scale effects by inverse variance (fixed effect), adds a
  DerSimonian-Laird random-effects estimate, and reports Cochran's Q, I^2 and
  tau^2 together with a prediction interval for a *new* scale. A pooled
  number under large heterogeneity is an average of unlike effects, not
  evidence of one replicated mechanism.

Everything is pure `math` (no scipy/sklearn), so the training scripts can run
in a minimal environment.
"""
import functools
import math
import numbers
import random
from typing import Optional, Sequence

__all__ = [
    "average_precision",
    "binary_auc_report",
    "chi2_sf",
    "f1_from_counts",
    "holm_adjust",
    "imbalance_report",
    "leave_one_out_jackknife",
    "macro_weighted_f1",
    "MIN_SEEDS_FOR_SIGNIFICANCE",
    "mcnemar_test",
    "mean_ci",
    "mean_prediction_interval",
    "mean_std_stderr",
    "meta_analysis",
    "max_p_combination",
    "paired_design_summary",
    "paired_randomization_test",
    "paired_required_n",
    "paired_t_one_sided",
    "paired_trend_test",
    "paired_z_one_sided",
    "roc_auc",
    "scale_replication_summary",
    "student_t_cdf",
    "student_t_ppf",
    "tost_paired",
    "two_sided_p_from_t",
    "welch_t_one_sided",
    "welch_z_one_sided",
    "wilson_interval",
    "recall_at_k",
    "ndcg_at_k",
]

_Z_95 = 1.959963984540054
MIN_SEEDS_FOR_SIGNIFICANCE = 3
# Above this many observations the 2**n sign-flip enumeration is no
# longer a cheap exact computation; use the seeded Monte-Carlo branch.
_RANDOMIZATION_EXACT_MAX = 20
# Negatives-per-positive ratio above which the FPR denominator makes ROC-AUC
# optimistic and PR-AUC (average precision) is the appropriate summary.
_EXTREME_IMBALANCE = 1000.0


def _is_boolean_value(value) -> bool:
    """Python and NumPy booleans.

    ``isinstance(np.bool_(True), bool)`` is False; without this check a NumPy
    boolean is silently coerced to 1/0 by ``int()`` and can enter a count or a
    multiplicity family as a real observation.
    """
    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == 'numpy' and cls.__name__ in ('bool_', 'bool')


def _as_real(value) -> Optional[float]:
    """A Python/NumPy real scalar as a float, or None for non-numbers/bools.

    The helpers below must accept every real numeric scalar the analysis
    scripts can produce (NumPy integer/float scalars included) without
    admitting a boolean as 0/1 or crashing on an overflowing integer.
    """
    if _is_boolean_value(value) or not isinstance(value, numbers.Real):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _strict_int(value, name: str, minimum: Optional[int] = None) -> int:
    """An integer-valued count/id, rejecting bools and fractional values.

    ``int(1.5)`` silently changes a sample count or class id while the report
    still looks like it describes the requested measurement. Bools are not
    counts either; a caller passing ``True`` almost certainly has a bug.
    """
    if _is_boolean_value(value):
        raise ValueError(f"{name} must be an integer, got bool {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if converted != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {converted}")
    return converted


def _validate_alpha(alpha, name: str = "alpha") -> float:
    """A significance level must be a finite probability in (0, 1).

    ``student_t_ppf(1 - alpha/2, ...)`` needs a probability strictly inside
    (0, 1). Without this guard an out-of-range alpha either raises from deep
    inside the distribution helper (alpha = 0) or silently produces an
    inverted interval (alpha > 1), which looks like a measured confidence
    interval instead of a malformed request.
    """
    if _is_boolean_value(alpha):
        raise ValueError(f"{name} must be a probability, got bool {alpha!r}")
    try:
        value = float(alpha)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a probability, got {alpha!r}") from None
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            f"{name} must be finite and strictly between 0 and 1, got {alpha!r}")
    return value


def _binary_labels(labels, name: str = "labels") -> list[int]:
    """0/1 labels without truthiness coercion.

    ``int(bool("0"))`` is 1, so a malformed string label (or any other
    non-numeric value) used to turn every row into a positive. Labels must be
    the documented binary ids; booleans are accepted as their 0/1 values.
    """
    out = []
    for value in labels:
        if _is_boolean_value(value):
            out.append(int(value))
            continue
        converted = _strict_int(value, name)
        if converted not in (0, 1):
            raise ValueError(f"{name} must be 0/1, got {value!r}")
        out.append(converted)
    return out


def wilson_interval(n_success: int, n_total: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation for the small trial counts used
    by the needle evaluation, and for proportions near 0 or 1.
    """
    n_total = _strict_int(n_total, "n_total", minimum=1)
    n_success = _strict_int(n_success, "n_success", minimum=0)
    if n_success > n_total:
        # 150 successes out of 100 trials is impossible evidence, not a
        # proportion to clamp to 1. Clamping produced a valid-looking
        # interval (0.963, 1.0) for a count that never existed.
        raise ValueError(
            f"wilson_interval needs 0 <= n_success <= n_total, got "
            f"{n_success} successes out of {n_total} trials")
    p = n_success / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    centre = (p + z2 / (2.0 * n_total)) / denom
    half = (z / denom) * math.sqrt(
        p * (1.0 - p) / n_total + z2 / (4.0 * n_total * n_total)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_test(n_fixed: int, n_broke: int) -> dict:
    """Exact McNemar test on the discordant pairs of a paired comparison.

    Two classifiers evaluated on the *same* samples split into four cells:
    both right, both wrong, and the two discordant cells - `n_fixed`
    (reference wrong, candidate right) and `n_broke` (reference right,
    candidate wrong). The question "is the candidate better?" lives entirely
    in the discordant cells: under the null hypothesis that both classifiers
    have the same error rate on these samples, each discordant pair is a fair
    coin, so `n_fixed ~ Binomial(n_fixed + n_broke, 0.5)`.

    The exact p-value is computed with the regularized incomplete beta (the
    binomial CDF at p = 1/2), not the normal approximation: paired bad-case
    analysis runs on a few hundred to a few thousand scanned errors, and at
    the small discordant counts that survive there the normal tail is
    anti-conservative. `p_one_sided` is P(X >= n_fixed) and supports
    "candidate better"; `p_two_sided` doubles the smaller tail (capped at 1).
    `normal_z_continuity_corrected` is reported for reference only.

    Degenerate case `n_discordant = 0`: the models agree on every compared
    sample, so there is no evidence either way and both p-values are `None`
    (not 1.0, which would read as "no difference").
    """
    n_fixed = _strict_int(n_fixed, "n_fixed", minimum=0)
    n_broke = _strict_int(n_broke, "n_broke", minimum=0)
    n = n_fixed + n_broke
    out = {
        "n_fixed": n_fixed,
        "n_broke": n_broke,
        "n_discordant": n,
        "p_one_sided": None,
        "p_two_sided": None,
        "favors": "tied",
        "normal_z_continuity_corrected": None,
        "test": "mcnemar_exact",
    }
    if n == 0:
        out["note"] = (
            "no discordant pairs: the two arms agreed on every compared sample, "
            "so the comparison carries no evidence either way"
        )
        return out
    # P(X <= k) for X ~ Bin(n, 1/2) is I_{1/2}(n - k, k + 1); the one-sided
    # p-value P(X >= n_fixed) is the mirror image I_{1/2}(n_fixed, n_broke + 1).
    p_ge = _reg_incomplete_beta(n_fixed, n_broke + 1, 0.5) if n_fixed > 0 else 1.0
    p_le = _reg_incomplete_beta(n_broke, n_fixed + 1, 0.5) if n_broke > 0 else 1.0
    out["p_one_sided"] = p_ge
    out["p_two_sided"] = min(1.0, 2.0 * min(p_ge, p_le))
    out["favors"] = "candidate" if n_fixed > n_broke else ("reference" if n_broke > n_fixed else "tied")
    out["normal_z_continuity_corrected"] = max(
        0.0, (abs(n_fixed - n_broke) - 1.0) / math.sqrt(n))
    out["note"] = (
        "p_one_sided = P(Binomial(n_discordant, 0.5) >= n_fixed); the exact test "
        "is the small-discordant-count reference (the normal approximation is "
        "anti-conservative there)"
    )
    return out


def mean_std_stderr(values: Sequence[float]) -> Optional[dict]:
    """Sample mean, sample std (ddof=1) and standard error of the mean.

    Non-finite entries (NaN/Inf) are excluded from the statistics and counted
    in ``n_nonfinite``; a single corrupted batch must not turn the whole
    summary into `nan` (the caller can then see that the mean rests on fewer
    points instead of silently quoting a poisoned number). Values with no
    finite float image (including JSON integers too large for a double) are
    excluded the same way instead of aborting the summary. If no finite value
    remains, `None` is returned so the caller cannot mistake "no measurement"
    for a result.
    """
    raw = []
    n_unusable = 0
    for value in values:
        if value is None or _is_boolean_value(value):
            continue
        try:
            raw.append(float(value))
        except (TypeError, ValueError, OverflowError):
            n_unusable += 1
    vals = [v for v in raw if math.isfinite(v)]
    n_nonfinite = len(raw) - len(vals) + n_unusable
    n_input = len(raw) + n_unusable
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    if n < 2:
        return {"mean": mean, "std": 0.0, "stderr": 0.0, "n": n,
                "n_input": n_input, "n_nonfinite": n_nonfinite}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    return {"mean": mean, "std": std, "stderr": std / math.sqrt(n), "n": n,
            "n_input": n_input, "n_nonfinite": n_nonfinite}


def mean_ci(values: Sequence[float], alpha: float = 0.05) -> Optional[dict]:
    """Sample mean with a two-sided Student-t confidence interval.

    A held-out loss (validation or sealed test) is a sample mean over batches,
    and a sample mean without an interval hides its own resolution. This
    extends `mean_std_stderr` with ``mean +/- t_{1-alpha/2, n-1} * stderr``,
    the same small-sample reference distribution the paired/Welch tests use:
    the normal critical value would understate the interval at the small
    seed counts the comparisons run with. With n < 2 there are no degrees
    of freedom and `ci_low`/`ci_high` are `None` (the mean is still returned);
    the caller should not quote an interval it never measured.
    """
    stats = mean_std_stderr(values)
    if stats is None:
        return None
    alpha = _validate_alpha(alpha)
    out = dict(stats)
    out["ci_level"] = 1.0 - float(alpha)
    out["ci_low"] = None
    out["ci_high"] = None
    if stats["n"] >= 2:
        half = student_t_ppf(1.0 - alpha / 2.0, float(stats["n"] - 1)) * stats["stderr"]
        out["ci_low"] = stats["mean"] - half
        out["ci_high"] = stats["mean"] + half
    return out


def mean_prediction_interval(values: Sequence[float], alpha: float = 0.05) -> Optional[dict]:
    """Where a *new* run's metric would plausibly fall, not where the mean is.

    A confidence interval for the mean (`mean_ci`) narrows as the number of
    seeds grows: it says where the population mean is, not where the next
    replication will land. With three seeds a 95% CI can look reassuringly
    tight while a fourth seed still reverses the sign. The prediction
    interval uses the same Student-t reference with the extra between-run
    variance:

        mean +/- t_{1-alpha/2, n-1} * s * sqrt(1 + 1/n)

    so it stays wide at small n and converges to the run-to-run spread rather
    than to zero. This is the interval a replication claim should be read
    against.

    ``same_direction_probability`` is the model-based predictive probability
    that a new observation has the same sign as the observed mean, i.e.
    ``P(T_{n-1} <= |mean| / (s * sqrt(1 + 1/n)))``. It is a descriptive
    predictive quantity, not a p-value and not a substitute for the
    pre-specified superiority test; it assumes exchangeable, approximately
    normal seed-level measurements.

    Returns ``None`` when there is no finite observation. With ``n < 2`` the
    interval is undefined (no degrees of freedom); with ``n == 2`` it is
    defined but ``meaningful`` is False because one seed's spread cannot
    support a replication claim. A zero-variance sample reduces to a
    degenerate interval and a deterministic same-direction probability.
    """
    stats = mean_std_stderr(values)
    if stats is None:
        return None
    alpha = _validate_alpha(alpha)
    out = dict(stats)
    out.update({
        "prediction_interval_level": 1.0 - alpha,
        "prediction_low": None,
        "prediction_high": None,
        "same_direction_probability": None,
        "meaningful": bool(stats["n"] >= 3),
        "note": ("a prediction interval describes the next run, not the mean; "
                 "it is necessarily wider than the confidence interval"),
    })
    n = stats["n"]
    if n < 2:
        out["note"] = ("a prediction interval needs at least 2 observations; "
                       "with one run there is no run-to-run spread to predict from")
        return out
    mean = stats["mean"]
    std = stats["std"]
    df = float(n - 1)
    if std == 0.0:
        out.update({
            "prediction_low": mean,
            "prediction_high": mean,
            "same_direction_probability": 1.0 if mean != 0.0 else 0.5,
            "note": ("zero observed variance: the interval is degenerate and the "
                     "same-direction probability is deterministic, not evidence "
                     "that a new run cannot differ"),
        })
        return out
    scale = std * math.sqrt(1.0 + 1.0 / n)
    half = student_t_ppf(1.0 - alpha / 2.0, df) * scale
    out["prediction_low"] = mean - half
    out["prediction_high"] = mean + half
    out["same_direction_probability"] = student_t_cdf(abs(mean) / scale, df)
    return out


def max_p_combination(pvalues: Sequence[Optional[float]]) -> Optional[float]:
    """Intersection-union p-value for "all of these one-sided tests reject".

    When a reference is picked from a pool *after* seeing the losses, the
    scientific claim is "the candidate beats the best member of that pool".
    That is true only if the candidate beats **every** member, so the relevant
    null is a union ("at least one pool member is at least as good") and the
    combination is an intersection-union test (IUT):

        p_combined = max(p_1, ..., p_m)

    The IUT rejects at level alpha only when every individual test rejects,
    and it needs no independence assumption and no Bonferroni factor. It is
    conservative (max is stochastically larger than any single p), which is
    the right direction for a confirmatory claim built on a data-selected
    reference. Using the *minimum* p instead would answer a different
    question - "the candidate beats at least one pool member" - and could
    certify a candidate that loses to the selected reference.

    ``None``, boolean and non-finite/out-of-range entries are ignored (the
    same contract as `holm_adjust`); a caller that needs a *complete* family
    must validate the pool before calling, otherwise dropping a missing test
    silently weakens the "every member" claim. Returns ``None`` when no valid
    p-value remains.
    """
    valid = []
    for p in pvalues:
        if p is None or _is_boolean_value(p):
            continue
        try:
            value = float(p)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            valid.append(value)
    if not valid:
        return None
    return max(valid)


def _p_one_sided_from_z(z: float) -> float:
    """P(Z <= z) for a standard normal Z (the one-sided p-value of a z-score)."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Lentz's method)."""
    max_it, eps, fpmin = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _reg_incomplete_beta(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b) for a, b > 0 and 0 <= x <= 1."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _gamma_p_series(a: float, x: float) -> float:
    """Regularized lower incomplete gamma P(a, x) by its power series."""
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(10000):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_q_cf(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) by Lentz's continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x: float, df: float) -> float:
    """Upper-tail probability P(X >= x) for a chi-square with `df` d.o.f.

    The heterogeneity test in `meta_analysis` needs a chi-square tail, and the
    repository deliberately has no scipy dependency. This is the standard
    series / continued-fraction split for the regularized incomplete gamma:
    use the series for P when ``x/2 < a + 1`` (it converges fast there) and
    the continued fraction for Q otherwise, so neither tail loses precision.
    """
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("chi2_sf needs df > 0")
    if math.isnan(x):
        raise ValueError("chi2_sf got a NaN statistic")
    if x <= 0.0:
        return 1.0
    if math.isinf(x):
        return 0.0
    a = 0.5 * df
    half = 0.5 * x
    if half < a + 1.0:
        return min(1.0, max(0.0, 1.0 - _gamma_p_series(a, half)))
    return min(1.0, max(0.0, _gamma_q_cf(a, half)))


def student_t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom.

    Pure `math` and accurate to near machine precision across the small
    degrees-of-freedom range the seed-replicated comparisons live in (df =
    n - 1 = 1-4). This is the reference distribution for the paired and Welch
    tests below; the normal CDF (z) is only the df -> infinity limit.
    """
    if not math.isfinite(df) or df <= 0:
        raise ValueError("student_t_cdf needs df > 0")
    if math.isnan(t):
        raise ValueError("student_t_cdf got a NaN statistic")
    if t == float("inf"):
        return 1.0
    if t == float("-inf"):
        return 0.0
    tail = 0.5 * _reg_incomplete_beta(0.5 * df, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0.0 else tail


@functools.lru_cache(maxsize=512)
def student_t_ppf(p: float, df: float) -> float:
    """Inverse Student-t CDF `t : P(T <= t) = p`, by bisection on a monotone CDF.

    Used for confidence-interval critical values. The result is a pure
    function of `(p, df)`, so it is memoized.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("student_t_ppf needs 0 < p < 1")
    if not math.isfinite(df) or df <= 0:
        raise ValueError("student_t_ppf needs df > 0")
    if p == 0.5:
        # The median of every Student-t is exactly zero. Bisection can stop on
        # adjacent floats whose CDF rounds around 0.5 and return a tiny
        # nonzero value, which would make a 50% interval asymmetric.
        return 0.0
    lo, hi = -1.0, 1.0
    while student_t_cdf(lo, df) > p:
        lo *= 2.0
    while student_t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            # Adjacent floats: no later iteration can move the interval, so
            # the fixed-point 200-iteration return value is already reached.
            break
        cdf_mid = student_t_cdf(mid, df)
        if cdf_mid == p:
            return mid
        if cdf_mid < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def holm_adjust(pvalues: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Holm-Bonferroni step-down adjustment of a family of p-values.

    A matrix run scans many rows against one reference, so an uncorrected
    alpha = 0.05 makes a "significant" win almost guaranteed by chance (with
    19 rows the family-wise error rate approaches 1 - 0.95^19 ~ 62%). Holm's
    step-down is uniformly more powerful than plain Bonferroni and needs no
    assumption beyond validity of the individual p-values: sort ascending and
    multiply the k-th smallest by `m - k + 1`, then enforce monotonicity.
    `None` entries (tests that could not run) stay `None` and are not counted
    in `m`.
    """
    # Accept any real numeric scalar (including NumPy integer/float scalars),
    # not only Python int/float: dropping an unrecognised scalar returns None
    # for that test and silently removes it from the correction family. Bools
    # are still rejected, and non-numeric / non-finite values stay None.
    indexed = []
    for i, p in enumerate(pvalues):
        if p is None or _is_boolean_value(p):
            continue
        try:
            value = float(p)
        except (TypeError, ValueError, OverflowError):
            continue
        # A p-value outside [0, 1] is not a valid test result. Keeping it
        # would either make a negative value maximally significant (after the
        # running max) or silently cap one above 1 to a perfect result; both
        # turn malformed evidence into a claim. Exclude it like a non-finite
        # p-value, and leave its slot None.
        if math.isfinite(value) and 0.0 <= value <= 1.0:
            indexed.append((value, i))
    indexed.sort()
    m = len(indexed)
    adjusted: list[Optional[float]] = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def paired_z_one_sided(
    a: Sequence[float], b: Sequence[float], margin: float = 0.0
) -> dict:
    """One-sided paired z-test of H1: mean(a - b) < -margin (large-n only).

    `a` and `b` must be the same length and paired (e.g. the per-seed or
    per-batch losses of two arms measured on identical data). A negative
    `mean_delta` means `a` is better. `z = (mean_delta + margin) / stderr`,
    so a negative z supports the alternative; `p_one_sided` is P(Z <= z).

    This is the df -> infinity limit: with the 2-4 replicas the experiments
    in this repo actually run, the normal tail is anti-conservative - use
    `paired_t_one_sided`, which also reports a confidence interval.
    """
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    n = len(a)
    if n == 0:
        return {"mean_delta": None, "z": None, "p_one_sided": None, "n": 0, "paired": True}
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / n
    if n < 2:
        return {"mean_delta": mean, "z": None, "p_one_sided": None, "n": n, "paired": True}
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0.0:
        z = float("-inf") if mean + margin < 0.0 else (float("inf") if mean + margin > 0.0 else 0.0)
    else:
        z = (mean + margin) / se
    return {
        "mean_delta": mean,
        "z": z,
        "p_one_sided": _p_one_sided_from_z(z),
        "n": n,
        "paired": True,
    }


def paired_t_one_sided(
    a: Sequence[float], b: Sequence[float], margin: float = 0.0, alpha: float = 0.05
) -> dict:
    """One-sided paired Student-t test of H1: mean(a - b) < -margin.

    The small-sample reference for the paired comparisons this repo runs:
    `t = (mean_delta + margin) / stderr` against a t distribution with
    `n - 1` degrees of freedom, so at n = 3 seeds the 5% critical value is
    2.92 instead of the normal 1.64. A negative `mean_delta` means `a` is
    better. Also reports a two-sided `1 - alpha` confidence interval for
    `mean_delta` (`ci_low`, `ci_high`): a delta whose interval crosses zero
    is inside the noise, whatever its point estimate. The zero-variance case
    (`stderr == 0`) is the usual degenerate limit of the t statistic: the
    sign of `mean_delta + margin` decides p = 0 or 1.
    """
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    alpha = _validate_alpha(alpha)
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    n = len(a)
    empty = {
        "mean_delta": None, "t": None, "df": None, "p_one_sided": None,
        "ci_low": None, "ci_high": None, "ci_level": 1.0 - alpha,
        "n": n, "paired": True, "test": "paired_t",
    }
    if n == 0:
        return empty
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / n
    if n < 2:
        empty["mean_delta"] = mean
        return empty
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    df = float(n - 1)
    if se == 0.0:
        t = float("-inf") if mean + margin < 0.0 else (float("inf") if mean + margin > 0.0 else 0.0)
    else:
        t = (mean + margin) / se
    half = student_t_ppf(1.0 - alpha / 2.0, df) * se
    return {
        "mean_delta": mean,
        "t": t,
        "df": df,
        "p_one_sided": student_t_cdf(t, df),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "ci_level": 1.0 - alpha,
        "n": n,
        "paired": True,
        "test": "paired_t",
    }


def two_sided_p_from_t(t: Optional[float], df: Optional[float]) -> Optional[float]:
    """Two-sided p-value of a t statistic: ``2 * min(P(T<=t), P(T>=t))``.

    The one-sided paired tests answer "is the candidate better"; a factorial
    interaction (or an ablation null result) is a two-sided question, because
    an effect in *either* direction is informative. This helper reuses the
    same Student-t CDF as `paired_t_one_sided`, so the two reports never
    disagree about the reference distribution. Infinite statistics (the
    zero-variance limit) map to 0; unusable inputs return `None` rather than
    a misleading 1.0.
    """
    if t is None or df is None:
        return None
    t_value = _as_real(t)
    df_value = _as_real(df)
    if t_value is None or df_value is None:
        return None
    t, df = t_value, df_value
    if (math.isnan(t) or not math.isfinite(df) or df <= 0.0):
        return None
    if math.isinf(t):
        return 0.0
    lower = student_t_cdf(t, df)
    return min(1.0, 2.0 * min(lower, 1.0 - lower))


def _paired_mde(std: float, n: int, alpha: float, power: float) -> float:
    """One-sided paired-t minimum detectable effect (same formula as reports).

    ``(t_{1-alpha, n-1} + t_{power, n-1}) * std / sqrt(n)``. This is the
    standard two-point t approximation used by the comparison scripts; keeping
    one implementation means a power plan and a reported MDE never disagree.
    """
    df = float(n - 1)
    return ((student_t_ppf(1.0 - alpha, df) + student_t_ppf(power, df))
            * std / math.sqrt(n))


def _validate_design_inputs(std, alpha: float, power: float) -> tuple[float, float, float]:
    if _is_boolean_value(std) or std is None:
        raise ValueError(f"paired standard deviation must be a number, got {std!r}")
    try:
        std = float(std)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"paired standard deviation must be a number, got {std!r}") from None
    if not math.isfinite(std) or std < 0.0:
        raise ValueError(
            f"paired standard deviation must be finite and >= 0, got {std!r}")
    for name, value in (("alpha", alpha), ("power", power)):
        if _is_boolean_value(value):
            raise ValueError(f"{name} must be a probability, got {value!r}")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a probability, got {value!r}") from None
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {value!r}")
    return std, float(alpha), float(power)


def paired_required_n(std: float, effect: float, alpha: float = 0.05,
                      power: float = 0.8, max_n: int = 100000) -> Optional[int]:
    """Prospective paired-sample size for a pre-specified effect size.

    Answers the design question *before* a run: with a paired-difference
    standard deviation ``std``, how many paired observations (seeds, or
    paired batches) are needed for the one-sided paired t-test to detect an
    effect of size ``effect`` at the requested ``power`` and ``alpha``? It
    uses the same t-critical approximation as the MDE fields in the reports,
    so the plan and the reported resolution are consistent.

    ``effect`` is the *smallest effect of scientific interest* fixed before
    the experiment, not the observed effect. A post-hoc "n needed to detect
    what we happened to see" is not a design justification; the companion
    ``paired_design_summary`` keeps the current n and the required n side by
    side instead.

    Returns the smallest integer ``n >= 2``, or ``None`` when the effect is
    too small to detect within ``max_n`` pairs. A zero-variance paired
    difference needs only the minimum two observations (the test is
    deterministic).
    """
    std, alpha, power = _validate_design_inputs(std, alpha, power)
    if _is_boolean_value(effect) or effect is None:
        raise ValueError(f"effect must be a positive number, got {effect!r}")
    try:
        effect = float(effect)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"effect must be a positive number, got {effect!r}") from None
    if not math.isfinite(effect) or effect <= 0.0:
        raise ValueError(f"effect must be finite and > 0, got {effect!r}")
    max_n = _strict_int(max_n, "max_n", minimum=2)
    if std == 0.0:
        return 2
    if _paired_mde(std, max_n, alpha, power) > effect:
        return None
    if max_n == 2:
        return 2
    # The MDE decreases monotonically in n; bracket the answer, then bisect.
    lo, hi = 2, min(max_n, 4)
    while hi < max_n and _paired_mde(std, hi, alpha, power) > effect:
        lo = hi + 1
        hi = min(max_n, hi * 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if _paired_mde(std, mid, alpha, power) <= effect:
            hi = mid
        else:
            lo = mid + 1
    return lo


def paired_design_summary(std: float, n: int, margin: float = 0.0,
                          alpha: float = 0.05, power: float = 0.8) -> dict:
    """Prospective design check for one achieved paired comparison.

    The comparison reports already carry the *observed* MDE. This block adds
    the question a reader of a null result needs: with the observed paired
    standard deviation, how many pairs would the pre-specified ``margin``
    have required, and was the achieved ``n`` enough? It never changes the
    significance verdict; it only makes underpowered designs visible.

    ``margin <= 0`` means no smallest effect was pre-specified, so the
    required-n fields stay ``None`` rather than inventing a target from the
    observed delta.
    """
    std, alpha, power = _validate_design_inputs(std, alpha, power)
    n = _strict_int(n, "n", minimum=2)
    out = {
        "paired_sd": std,
        "n": n,
        "alpha": float(alpha),
        "power_target": float(power),
        # The legacy key promises 95% confidence / 80% power, so it must not
        # silently become the caller's alpha/power when those are overridden.
        "minimum_detectable_delta_95_power80": _paired_mde(std, n, 0.05, 0.8),
        "minimum_detectable_delta": _paired_mde(std, n, alpha, power),
        "minimum_detectable_alpha": float(alpha),
        "minimum_detectable_power": float(power),
        "margin": None,
        "required_pairs_for_margin": None,
        "adequately_powered_for_margin": None,
        "note": ("pass a positive pre-specified margin to compute the pairs "
                 "needed for that smallest effect of interest"),
    }
    if _is_boolean_value(margin) or margin is None:
        return out
    try:
        margin = float(margin)
    except (TypeError, ValueError, OverflowError):
        return out
    if not math.isfinite(margin) or margin <= 0.0:
        return out
    required = paired_required_n(std, margin, alpha=alpha, power=power)
    out.update({
        "margin": margin,
        "required_pairs_for_margin": required,
        "adequately_powered_for_margin": (
            None if required is None else bool(n >= required)),
        "note": ("this design reaches the margin" if required is not None
                 and n >= required else
                 "underpowered for the pre-specified margin" if required is not None
                 else "the margin is not detectable within the search cap"),
    })
    return out


def tost_paired(
    a: Sequence[float], b: Sequence[float], margin: float, alpha: float = 0.05
) -> dict:
    """Two one-sided paired t-tests (TOST) for *equivalence* within `margin`.

    A non-significant superiority test is **not** evidence that two arms are
    the same: it only says the design failed to see a difference. Several
    ablation hypotheses are equivalence claims ("explicit init 1e-5 must
    coincide with the default", "the two LayerScale settings should match"),
    so they need a test whose alternative is "the mean paired difference is
    inside ``+/- margin``":

    * H0: ``|mean(a-b)| >= margin`` (not equivalent);
    * H1: ``|mean(a-b)| <  margin`` (equivalent).

    This is the intersection-union TOST: reject H0 only when *both* the
    lower-bound and upper-bound one-sided tests reject, so
    ``p_tost = max(p_low, p_high)``. The reported interval is the matching
    ``1 - 2*alpha`` confidence interval (``mean +/- t_{1-alpha,df} * se``); it
    must lie inside ``[-margin, margin]`` for the equivalence claim. `margin`
    is a *scientific* smallest effect of interest, not a knob to tune until
    the result looks equivalent, so it is recorded next to the p-value in
    every comparison artifact.

    Returns a dict with ``equivalent=None`` (and a ``note``) when the test
    cannot run: no positive margin, fewer than two paired observations, or a
    non-finite input. The zero-variance case reduces to a deterministic
    comparison of ``|mean|`` with `margin`.

    TOST needs ``alpha < 0.5``: its interval is the ``1 - 2*alpha`` one, so a
    larger alpha would report a non-positive confidence level and a claimed
    equivalence interval that does not exist.
    """
    alpha = _validate_alpha(alpha)
    if alpha >= 0.5:
        raise ValueError(
            f"TOST needs alpha < 0.5 so the 1 - 2*alpha confidence interval "
            f"is positive, got {alpha!r}")
    out = {
        "test": "tost_paired",
        "margin": None,
        "alpha": alpha,
        "equivalent": None,
        "p_tost": None,
        "p_low": None,
        "p_high": None,
        "mean_delta": None,
        "ci_low": None,
        "ci_high": None,
        "ci_level": None,
        "n": 0,
        "df": None,
        "note": "equivalence not evaluated",
    }
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    if len(a) != len(b):
        raise ValueError(f"TOST needs equal lengths, got {len(a)} vs {len(b)}")
    n = len(a)
    out["n"] = n
    margin_value = _as_real(margin)
    if margin_value is None or not math.isfinite(margin_value) or margin_value <= 0.0:
        out["note"] = ("equivalence margin must be a positive finite number; "
                       "pass an explicit scientific margin to enable TOST")
        return out
    margin = margin_value
    out["margin"] = margin
    if n < 2:
        out["note"] = "TOST needs at least 2 paired observations"
        return out
    d = [x - y for x, y in zip(a, b)]
    mean = sum(d) / n
    if not math.isfinite(mean):
        out["note"] = "a non-finite paired difference makes the test undefined"
        return out
    var = sum((v - mean) ** 2 for v in d) / (n - 1)
    se = math.sqrt(var / n)
    df = float(n - 1)
    out.update({"mean_delta": mean, "df": df, "ci_level": 1.0 - 2.0 * alpha})
    if se == 0.0:
        equivalent = abs(mean) < margin
        out.update({
            "p_low": 0.0 if equivalent else 1.0,
            "p_high": 0.0 if equivalent else 1.0,
            "p_tost": 0.0 if equivalent else 1.0,
            "equivalent": bool(equivalent),
            "ci_low": mean,
            "ci_high": mean,
            "note": ("zero-variance paired differences: equivalence is decided "
                     "by |mean_delta| < margin"),
        })
        return out
    # p_low tests H0 `mean <= -margin` against H1 `mean > -margin`;
    # p_high tests H0 `mean >= +margin` against H1 `mean < +margin`.
    p_low = 1.0 - student_t_cdf((mean + margin) / se, df)
    p_high = student_t_cdf((mean - margin) / se, df)
    p_low = min(1.0, max(0.0, p_low))
    p_high = min(1.0, max(0.0, p_high))
    p_tost = max(p_low, p_high)
    half = student_t_ppf(1.0 - alpha, df) * se
    out.update({
        "p_low": p_low,
        "p_high": p_high,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "note": ("equivalent" if p_tost < alpha else
                 "not equivalent at this margin/alpha; this is not evidence of a difference"),
    })
    return out


def leave_one_out_jackknife(
    values: Sequence[float], labels: Optional[Sequence] = None
) -> dict:
    """How dependent is a sample mean on any single observation?

    With three seeds, one lucky or broken run can carry a delta. Dropping
    each observation in turn (the leave-one-out jackknife) shows whether the
    *sign* of the effect is a property of the set or of one point: if removing
    seed 42 reverses the direction, the claim is not seed-robust even when the
    full-sample t-test is significant. This is a descriptive robustness check
    - it never replaces the pre-specified significance test - and it is
    reported per comparison so a reader can see which seed is influential.

    ``labels`` name the observations (the caller passes seed ids); they must
    match `values` in length. Returns `None` for an empty input, and
    ``meaningful=False`` when fewer than three observations make the
    sign-change test vacuous (dropping one leaves a single point).
    """
    vals = [float(v) for v in values]
    n = len(vals)
    # Validate the labels even on the early-return paths: a caller that
    # supplies the wrong number of names has a programming error, and silently
    # ignoring it for an empty/non-finite input makes the contract depend on
    # the data value.
    names = list(range(n)) if labels is None else list(labels)
    if len(names) != n:
        raise ValueError(
            f"jackknife labels need {n} entries, got {len(names)}")
    if n == 0:
        return None
    if any(not math.isfinite(v) for v in vals):
        return {
            "n": n,
            "unit": "observation",
            "full_mean": None,
            "sign": None,
            "meaningful": False,
            "per_drop": [],
            "sign_stable": None,
            "influential_labels": [],
            "delta_range": None,
            "max_abs_change": None,
            "note": "non-finite observation(s); jackknife skipped",
        }
    full = sum(vals) / n

    def _sign(x: float) -> str:
        return "negative" if x < 0.0 else ("positive" if x > 0.0 else "zero")

    full_sign = _sign(full)
    if n < 2:
        # Dropping the only observation leaves nothing to average: there is
        # no jackknife sample, only the point estimate.
        return {
            "n": n,
            "unit": "observation",
            "full_mean": full,
            "sign": full_sign,
            "meaningful": False,
            "per_drop": [],
            "sign_stable": None,
            "influential_labels": [],
            "delta_range": None,
            "max_abs_change": None,
            "note": "a jackknife needs at least 2 observations",
        }
    per_drop = []
    for i in range(n):
        rest = vals[:i] + vals[i + 1:]
        mean = sum(rest) / len(rest)
        drop_sign = _sign(mean)
        if full_sign == "zero":
            # There is no direction to stabilise; a mean that is exactly zero
            # is not made "unstable" by any single point, it simply has no
            # sign claim. Report `sign_stable=None` instead of a misleading
            # True/False.
            keeps = True
        elif full_sign == "negative":
            keeps = mean < 0.0
        else:
            keeps = mean > 0.0
        per_drop.append({
            "dropped": names[i],
            "mean_without": mean,
            "sign": drop_sign,
            "keeps_full_sign": bool(keeps),
        })
    drop_means = [item["mean_without"] for item in per_drop]
    influential = [item["dropped"] for item in per_drop
                   if not item["keeps_full_sign"]]
    return {
        "n": n,
        "unit": "observation",
        "full_mean": full,
        "sign": full_sign,
        "meaningful": n >= 3,
        "per_drop": per_drop,
        "sign_stable": (None if full_sign == "zero" else not influential),
        "influential_labels": influential,
        "delta_range": [min(drop_means), max(drop_means)],
        "max_abs_change": max(abs(m - full) for m in drop_means),
        "note": ("the full-sample mean is zero, so there is no direction to "
                 "check for stability; the equivalence test is the relevant "
                 "claim here" if full_sign == "zero" else
                 "descriptive robustness check; dropping one observation is "
                 "not an independent replication and never certifies a result"),
    }


def paired_randomization_test(
    deltas: Sequence[float],
    margin: float = 0.0,
    alpha: float = 0.05,
    exact_max: int = 12,
    permutations: int = 10000,
    seed: int = 0,
) -> dict:
    """Exact sign-flip randomization test for a paired location effect.

    The paired t-test is the repo's certifying test, but it assumes the
    seed-level differences are approximately normal. With the 3-5 seeds the
    experiments actually run, that assumption is doing real work: the t-test
    can return ``p < 0.05`` from three consistent runs, while a
    distribution-free test cannot. This function supplies the model-free
    counterpart. It tests H0 "the paired differences are symmetric about
    ``-margin``" against H1 "the mean difference is below ``-margin``" by
    flipping the sign of each centered difference (``d_i + margin``) and
    comparing the observed mean with the randomization distribution.

    For ``n <= exact_max`` every one of the ``2**n`` sign patterns is
    enumerated, so the p-value is exact and deterministic. ``exact_max`` is
    capped at 20 so the enumeration cannot explode by accident. Larger
    samples use a Monte-Carlo sign-flip distribution with a fixed seed; the
    ``(extreme + 1) / (permutations + 1)`` estimator keeps the reported
    p-value strictly positive and is reproducible across reruns.

    The output is deliberately explicit about resolution. At ``n = 3`` the
    smallest attainable one-sided p-value is ``1/8``, so
    ``can_reach_alpha`` is False for the usual 0.05 threshold: a significant
    t-test at that replication count rests on the parametric t model, not on
    a distribution-free ordering of the signs. This check is therefore
    descriptive robustness - it never replaces or overrides the
    pre-specified paired t-test that certifies a comparison.

    Non-finite or non-numeric deltas are excluded and listed in ``excluded``
    rather than silently changing the sample. The direction convention is the
    same as the rest of the repo: negative deltas favour the candidate.
    """
    alpha = _validate_alpha(alpha)
    margin_value = _as_real(margin)
    if margin_value is None or not math.isfinite(margin_value):
        raise ValueError(f"margin must be a finite number, got {margin!r}")
    exact_max = _strict_int(exact_max, "exact_max", minimum=0)
    if exact_max > _RANDOMIZATION_EXACT_MAX:
        raise ValueError(
            f"exact_max must be <= {_RANDOMIZATION_EXACT_MAX} to bound the "
            f"2**n enumeration, got {exact_max}")
    permutations = _strict_int(permutations, "permutations", minimum=1)
    seed = _strict_int(seed, "seed", minimum=0)

    raw = list(deltas)
    values = []
    excluded = []
    for index, value in enumerate(raw):
        real = _as_real(value)
        if real is None:
            excluded.append({"index": index, "value": repr(value),
                             "reason": "not a real number"})
        elif not math.isfinite(real):
            excluded.append({"index": index, "value": real,
                             "reason": "non-finite delta"})
        else:
            values.append(real)
    n = len(values)
    out = {
        "test": "paired_sign_flip_randomization",
        "alternative": ("mean(delta) < -margin: deltas become more negative "
                        "than the practical margin (negative favours the "
                        "candidate)"),
        "assumption": ("exchangeable paired differences, symmetric about "
                       "-margin under the null; no normality assumption"),
        "alpha": alpha,
        "margin": margin_value,
        "n": n,
        "n_input": len(raw),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "n_negative": sum(1 for value in values if value < 0.0),
        "n_positive": sum(1 for value in values if value > 0.0),
        "n_zero": sum(1 for value in values if value == 0.0),
        "mean_delta": None,
        "observed_centered_mean": None,
        "basis": "undefined",
        "n_permutations_evaluated": None,
        "minimum_attainable_p_one_sided": None,
        "can_reach_alpha": None,
        "p_one_sided": None,
        "p_upper_one_sided": None,
        "p_two_sided": None,
        "monte_carlo_standard_error": None,
        "seed": None,
        "meaningful": bool(n >= MIN_SEEDS_FOR_SIGNIFICANCE),
        "note": None,
    }
    if n == 0:
        out["note"] = ("no finite paired delta: a randomization test needs "
                       "at least one observation")
        return out

    mean = math.fsum(values) / n
    centered = [value + margin_value for value in values]
    observed = math.fsum(centered) / n
    scale = max(1.0, math.fsum(abs(value) for value in centered))
    tolerance = 1e-12 * scale
    out["mean_delta"] = mean
    out["observed_centered_mean"] = observed

    if n <= exact_max:
        sums = [0.0]
        for value in centered:
            sums = sums + [total + value for total in sums]
        patterns = len(sums)
        lower = sum(1 for total in sums if total / n <= observed + tolerance)
        upper = sum(1 for total in sums if total / n >= observed - tolerance)
        p_lower = lower / patterns
        p_upper = upper / patterns
        minimum = 1.0 / patterns
        out.update({
            "basis": "exact_enumeration",
            "n_permutations_evaluated": patterns,
            "minimum_attainable_p_one_sided": minimum,
            "can_reach_alpha": bool(minimum < alpha),
            "p_one_sided": p_lower,
            "p_upper_one_sided": p_upper,
            "p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
        })
    else:
        rng = random.Random(seed)
        lower = 0
        upper = 0
        for _ in range(permutations):
            total = 0.0
            for value in centered:
                total += value if rng.random() < 0.5 else -value
            permuted = total / n
            if permuted <= observed + tolerance:
                lower += 1
            if permuted >= observed - tolerance:
                upper += 1
        p_lower = (lower + 1.0) / (permutations + 1.0)
        p_upper = (upper + 1.0) / (permutations + 1.0)
        minimum = 1.0 / (permutations + 1.0)
        out.update({
            "basis": "monte_carlo",
            "n_permutations_evaluated": permutations,
            "minimum_attainable_p_one_sided": minimum,
            "can_reach_alpha": bool(minimum < alpha),
            "p_one_sided": p_lower,
            "p_upper_one_sided": p_upper,
            "p_two_sided": min(1.0, 2.0 * min(p_lower, p_upper)),
            "monte_carlo_standard_error": math.sqrt(
                max(0.0, p_lower * (1.0 - p_lower)) / (permutations + 1.0)),
            "seed": seed,
        })

    notes = []
    if excluded:
        notes.append(f"{len(excluded)} non-finite/non-numeric delta(s) excluded")
    if n < MIN_SEEDS_FOR_SIGNIFICANCE:
        notes.append(f"fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} paired "
                     "observations: descriptive only")
    if out["can_reach_alpha"] is False:
        notes.append(
            f"with n={n} the smallest attainable one-sided p is "
            f"{out['minimum_attainable_p_one_sided']:.4g}, so no "
            f"distribution-free test can reject at alpha={alpha:g}; a "
            "smaller t-test p at this n relies on the parametric t model")
    else:
        notes.append(
            "distribution-free robustness check; the pre-specified paired "
            "t-test remains the certification test")
    out["note"] = "; ".join(notes)
    return out


def _ols_slope_and_r2(xs: Sequence[float], ys: Sequence[float]):
    """Least-squares slope of ``ys`` on ``xs`` plus the fit's R-squared.

    Returns ``None`` when every ``x`` is identical (the slope is undefined).
    R-squared is clamped to [0, 1] so a tiny negative value from floating
    point cancellation is not reported as a nonsensical fit quality.
    """
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2
                 for x, y in zip(xs, ys))
    if ss_tot == 0.0:
        r2 = 1.0 if ss_res == 0.0 else 0.0
    else:
        r2 = min(1.0, max(0.0, 1.0 - ss_res / ss_tot))
    return slope, r2


def paired_trend_test(
    doses: Sequence[float],
    per_seed_values: dict,
    alpha: float = 0.05,
) -> dict:
    """Seed-level test of a dose-response trend in paired deltas.

    A depth sweep tests each ``(depth, norm)`` cell against the reference and
    corrects those per-depth p-values, but a row of separately significant
    deltas does not establish the sweep's actual scientific claim: that the
    advantage *grows with depth*. This function tests that claim directly.

    ``doses`` are the predictor values (e.g. ``log2(depth)``) and
    ``per_seed_values`` maps a seed to its paired delta
    (candidate - reference) at each dose, aligned with ``doses``. For every
    seed with at least two distinct finite dose points, one least-squares
    slope is fitted. The seed is the unit of replication, so the slopes - not
    the pooled seed x dose points - are the observations of a one-sided
    Student-t test of H1: mean slope < 0 (the delta becomes more negative as
    the dose rises, i.e. the candidate's advantage grows).

    Pooling all points would pseudo-replicate the repeated doses within a
    seed and shrink the standard error. Fitting one slope per seed keeps the
    n = number of seeds and the df = n - 1 the repo's other comparisons use;
    at three seeds the one-sided 5% critical value remains t = 2.92. The
    slopes are most comparable when every seed covers the same dose grid; a
    caller with missing depths should intersect the seeds first (the depth
    sweep does), otherwise a seed fitted over a different range contributes a
    slope that is not on the same scale.

    A slope with two doses is a difference-in-differences, not evidence of
    linearity; the result records ``n_doses`` so that distinction is visible.
    A leave-one-seed-out jackknife of the slopes shows whether one seed
    carries the trend. ``certifiable`` is False below
    ``MIN_SEEDS_FOR_SIGNIFICANCE`` seeds or below two distinct doses, even
    when a p-value is computable (matching the rest of the repo: two seeds
    are descriptive only).

    Returns a dict with ``mean_slope_per_dose``, its two-sided
    ``slope_ci_95``, ``t``/``df``/``p_one_sided``, the per-seed slopes,
    ``n_seeds``/``n_doses``, the jackknife, and the smallest slope the design
    could detect at 80% power. Mismatched lengths or a non-finite dose raise
    ``ValueError`` (a programming error); non-finite *deltas* are dropped and
    reported in ``excluded_seeds``.
    """
    raw_doses = list(doses)
    alpha = _validate_alpha(alpha)
    dose_values = []
    for dose in raw_doses:
        value = _as_real(dose)
        if value is None or not math.isfinite(value):
            raise ValueError(f"trend doses must be finite numbers, got {dose!r}")
        dose_values.append(value)
    distinct_doses = sorted(set(dose_values))
    n_doses = len(distinct_doses)

    out = {
        "test": "paired_trend",
        "alternative": ("mean per-seed slope < 0: the candidate-reference "
                        "delta falls as the dose rises (the advantage grows)"),
        "n_doses": n_doses,
        "doses": dose_values,
        "n_seeds": 0,
        "seeds_used": [],
        "per_seed_slopes": {},
        "per_seed_r_squared": {},
        "excluded_seeds": {},
        "mean_slope_per_dose": None,
        "slope_std": None,
        "slope_ci_95": [None, None],
        "slope_ci": [None, None],
        "ci_level": 1.0 - float(alpha),
        "t": None,
        "df": None,
        "p_one_sided": None,
        "minimum_detectable_slope_95_power80": None,
        "minimum_detectable_slope": None,
        "slope_prediction_interval_95": [None, None],
        "slope_prediction_interval": [None, None],
        "slope_same_direction_probability": None,
        "mean_r_squared": None,
        "n_seeds_with_negative_slope": None,
        "n_seeds_with_positive_slope": None,
        "advantage_grows_with_dose": None,
        "certifiable": False,
        "seed_jackknife": None,
        "slope_randomization_test": None,
        "note": None,
    }
    if n_doses < 2:
        out["note"] = ("a trend needs at least two distinct doses; with one "
                       "dose there is no slope to estimate")
        return out

    labels = sorted((per_seed_values or {}), key=str)
    # Different Python keys can stringify to the same seed id (JSON gives
    # `"42"`, a direct caller may pass `42`). The per-seed output is keyed by
    # the string form, so accepting both would silently overwrite one fitted
    # slope and still report n_seeds=2. Reject the ambiguous record instead.
    string_labels = [str(label) for label in labels]
    duplicate_labels = sorted({label for label in string_labels
                               if string_labels.count(label) > 1})
    if duplicate_labels:
        raise ValueError(
            "trend values contain seed labels that collide after string "
            f"normalization: {duplicate_labels}; each seed must appear once")
    slopes = []
    used = []
    fitted_dose_counts = []
    r_squared_values = []
    for label in labels:
        try:
            values = list(per_seed_values[label])
        except TypeError:
            out["excluded_seeds"][str(label)] = "not an iterable sequence"
            continue
        if len(values) != len(dose_values):
            raise ValueError(
                f"trend values for seed {label!r} need {len(dose_values)} "
                f"entries, got {len(values)}")
        pairs = []
        dropped = []
        for position, (x, y) in enumerate(zip(dose_values, values)):
            value = _as_real(y)
            if value is not None and math.isfinite(value):
                pairs.append((x, value))
            else:
                dropped.append(position)
        if dropped:
            # The slope is still fitted on the remaining finite doses, but a
            # silently shortened grid changes what the slope means; report it
            # instead of letting the seed look complete.
            out["excluded_seeds"][str(label)] = (
                "dropped non-finite delta(s) at dose position(s) "
                f"{dropped}; slope fitted on the remaining doses")
        if len({x for x, _ in pairs}) < 2:
            out["excluded_seeds"][str(label)] = (
                "fewer than two distinct doses with finite deltas")
            continue
        fit = _ols_slope_and_r2([x for x, _ in pairs],
                                [y for _, y in pairs])
        if fit is None:
            out["excluded_seeds"][str(label)] = "undefined slope"
            continue
        slopes.append(fit[0])
        used.append(label)
        fitted_dose_counts.append(len({x for x, _ in pairs}))
        r_squared_values.append(fit[1])
        out["per_seed_slopes"][str(label)] = fit[0]
        out["per_seed_r_squared"][str(label)] = fit[1]

    n = len(slopes)
    # A seed can drop a non-finite dose and still yield a slope, so the raw
    # dose count can overstate the grid actually fitted. Report the smallest
    # fitted grid across the seeds used: a two-point slope must never be
    # presented as a three-dose trend.
    if fitted_dose_counts:
        n_doses = min(fitted_dose_counts)
        out["n_doses"] = n_doses
    out["n_seeds"] = n
    out["seeds_used"] = used
    out["n_seeds_with_negative_slope"] = sum(1 for b in slopes if b < 0.0)
    out["n_seeds_with_positive_slope"] = sum(1 for b in slopes if b > 0.0)
    out["certifiable"] = bool(n >= MIN_SEEDS_FOR_SIGNIFICANCE and n_doses >= 2)
    if n == 0:
        out["note"] = ("no seed has two distinct, finite dose points; the "
                       "trend is undefined")
        return out
    stats = mean_std_stderr(slopes)
    out["mean_slope_per_dose"] = stats["mean"]
    out["slope_std"] = stats["std"]
    slope_replication = mean_prediction_interval(slopes, alpha=alpha)
    if slope_replication is not None:
        out["slope_prediction_interval"] = [
            slope_replication["prediction_low"],
            slope_replication["prediction_high"]]
        out["slope_same_direction_probability"] = (
            slope_replication["same_direction_probability"])
    slope_replication_95 = (slope_replication if float(alpha) == 0.05
                            else mean_prediction_interval(slopes, alpha=0.05))
    if slope_replication_95 is not None:
        out["slope_prediction_interval_95"] = [
            slope_replication_95["prediction_low"],
            slope_replication_95["prediction_high"]]
    # Average over the fits themselves, not over the string-keyed output dict:
    # that keeps the mean correct even if two labels ever normalize together.
    out["mean_r_squared"] = (
        sum(r_squared_values) / len(r_squared_values)
        if r_squared_values else None)
    out["seed_jackknife"] = leave_one_out_jackknife(
        slopes, labels=[str(label) for label in used])
    out["slope_randomization_test"] = paired_randomization_test(
        slopes, alpha=alpha)
    if n < 2:
        out["note"] = ("one usable seed: a slope is computable but its "
                       "sampling uncertainty is not; descriptive only")
        return out
    test = paired_t_one_sided(slopes, [0.0] * n, alpha=alpha)
    out.update({
        "mean_slope_per_dose": test["mean_delta"],
        "t": test["t"],
        "df": test["df"],
        "p_one_sided": test["p_one_sided"],
        "slope_ci": [test["ci_low"], test["ci_high"]],
        "ci_level": test["ci_level"],
    })
    # The legacy `_95` keys are read by older reports; keep them at their
    # named 95% / 80% defaults and expose the configured-alpha interval and
    # MDE under the unsuffixed names.
    if float(alpha) == 0.05:
        out["slope_ci_95"] = [test["ci_low"], test["ci_high"]]
    elif test.get("df"):
        test_95 = paired_t_one_sided(slopes, [0.0] * n, alpha=0.05)
        out["slope_ci_95"] = [test_95["ci_low"], test_95["ci_high"]]
    if stats is not None and stats["stderr"] is not None and test["df"]:
        df = float(test["df"])
        out["minimum_detectable_slope"] = (
            (student_t_ppf(1.0 - float(alpha), df)
             + student_t_ppf(0.8, df)) * stats["stderr"])
        out["minimum_detectable_slope_95_power80"] = (
            (student_t_ppf(0.95, df) + student_t_ppf(0.8, df))
            * stats["stderr"])
    out["advantage_grows_with_dose"] = bool(
        out["certifiable"] and out["mean_slope_per_dose"] is not None
        and out["mean_slope_per_dose"] < 0.0
        and out["p_one_sided"] is not None
        and out["p_one_sided"] < alpha)
    if not out["certifiable"]:
        out["note"] = (f"only {n} usable seed(s), fewer than the required "
                       f"{MIN_SEEDS_FOR_SIGNIFICANCE}: the trend is "
                       "descriptive and cannot certify a dose-response claim")
    elif n_doses == 2:
        out["note"] = ("two doses: the slope is a difference-in-differences "
                       "of the two deltas, not evidence of a linear trend")
    else:
        out["note"] = ("one-sided t-test on the per-seed slopes; a "
                       "significant negative mean slope means the candidate's "
                       "advantage grows with the dose under this protocol")
    return out


def welch_z_one_sided(
    a: Sequence[float], b: Sequence[float], margin: float = 0.0
) -> dict:
    """Unpaired one-sided Welch z-test of H1: mean(a) - mean(b) < -margin.

    Used only when the two arms do not share the same held-out samples (so
    pairing is impossible); the paired test above is strictly more sensitive
    and preferred whenever the batch counts match. Like its paired sibling
    this is the large-n limit; `welch_t_one_sided` is the small-sample test.
    """
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    if len(a) < 2 or len(b) < 2:
        return {"mean_delta": None, "z": None, "p_one_sided": None,
                "n": [len(a), len(b)], "paired": False}
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((v - ma) ** 2 for v in a) / (len(a) - 1)
    vb = sum((v - mb) ** 2 for v in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    delta = ma - mb
    if se == 0.0:
        z = float("-inf") if delta + margin < 0.0 else (float("inf") if delta + margin > 0.0 else 0.0)
    else:
        z = (delta + margin) / se
    return {
        "mean_delta": delta,
        "z": z,
        "p_one_sided": _p_one_sided_from_z(z),
        "n": [len(a), len(b)],
        "paired": False,
    }


def welch_t_one_sided(
    a: Sequence[float], b: Sequence[float], margin: float = 0.0, alpha: float = 0.05
) -> dict:
    """Unpaired one-sided Welch t-test of H1: mean(a) - mean(b) < -margin.

    The small-sample fallback for arms that do not share their held-out
    samples. Uses the Welch-Satterthwaite degrees of freedom
    `(va/na + vb/nb)^2 / ((va/na)^2/(na-1) + (vb/nb)^2/(nb-1))`, so unequal
    variances shrink the df instead of pretending they are equal, and reports
    the same two-sided `1 - alpha` interval for the difference as the paired
    test. `n` is `[na, nb]`, `paired` is False.
    """
    a = [float(v) for v in a]
    b = [float(v) for v in b]
    alpha = _validate_alpha(alpha)
    na, nb = len(a), len(b)
    empty = {
        "mean_delta": None, "t": None, "df": None, "p_one_sided": None,
        "ci_low": None, "ci_high": None, "ci_level": 1.0 - alpha,
        "n": [na, nb], "paired": False, "test": "welch_t",
    }
    if na < 2 or nb < 2:
        return empty
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((v - ma) ** 2 for v in a) / (na - 1)
    vb = sum((v - mb) ** 2 for v in b) / (nb - 1)
    delta = ma - mb
    se = math.sqrt(va / na + vb / nb)
    df = float(max(1, na + nb - 2))
    if se > 0.0:
        spread_a, spread_b = va / na, vb / nb
        denom = spread_a ** 2 / (na - 1) + spread_b ** 2 / (nb - 1)
        if denom > 0.0:
            df = max(1.0, (spread_a + spread_b) ** 2 / denom)
    if se == 0.0:
        t = float("-inf") if delta + margin < 0.0 else (float("inf") if delta + margin > 0.0 else 0.0)
    else:
        t = (delta + margin) / se
    half = student_t_ppf(1.0 - alpha / 2.0, df) * se
    empty.update({
        "mean_delta": delta,
        "t": t,
        "df": df,
        "p_one_sided": student_t_cdf(t, df),
        "ci_low": delta - half,
        "ci_high": delta + half,
    })
    return empty


_META_EFFECT_KEYS = ('effect', 'delta', 'estimate', 'mean_delta')
_META_STDERR_KEYS = ('stderr', 'se', 'standard_error')


def _meta_study(observation, index: int):
    """One meta-analytic input record, or the reason it cannot be pooled.

    The effect and its standard error are the only two numbers the pooling
    needs. Accept the spellings the comparison artifacts already use
    (`delta`/`mean_delta`, `stderr`/`se`) and, when only a confidence
    interval is present, invert it with the interval's own level and degrees
    of freedom instead of assuming a normal 95% interval.
    """
    if not isinstance(observation, dict):
        return None, {'index': index, 'reason': 'not an object'}
    label = observation.get('label')
    if label is None:
        label = observation.get('scale')
    if label is None:
        label = observation.get('name')
    label = str(index if label is None else label)

    effect = None
    for key in _META_EFFECT_KEYS:
        value = _as_real(observation.get(key))
        if value is not None and math.isfinite(value):
            effect = value
            break
    if effect is None:
        return None, {'index': index, 'label': label,
                      'reason': 'missing or non-finite effect'}

    stderr = None
    invalid_stderr = None
    se_basis = 'reported'
    for key in _META_STDERR_KEYS:
        value = _as_real(observation.get(key))
        if value is None:
            continue
        if math.isfinite(value) and value > 0.0:
            stderr = value
            break
        if invalid_stderr is None:
            invalid_stderr = value
    if stderr is None and invalid_stderr is not None:
        # Preserve the malformed value so the final validation reports why the
        # record is unusable instead of pretending no stderr was recorded.
        stderr = invalid_stderr
    if stderr is None:
        low = _as_real(observation.get('ci_low'))
        high = _as_real(observation.get('ci_high'))
        if (low is None or high is None or not math.isfinite(low)
                or not math.isfinite(high) or high <= low):
            return None, {'index': index, 'label': label,
                          'reason': ('no finite stderr and no usable '
                                     'ci_low/ci_high to derive one')}
        level = _as_real(observation.get('ci_level'))
        if level is None or not 0.0 < level < 1.0:
            level = 0.95
        df = _as_real(observation.get('df'))
        if df is not None and (not math.isfinite(df) or df <= 0.0):
            df = None
        if df is None:
            n = _as_real(observation.get('n_seeds'))
            if n is None:
                n = _as_real(observation.get('n'))
            if n is not None and math.isfinite(n) and n >= 2.0:
                df = n - 1.0
        if df is not None and df > 0.0:
            critical = student_t_ppf(1.0 - (1.0 - level) / 2.0, df)
            se_basis = 'ci_t'
        else:
            # No df was recorded: use the normal limit only as a labelled
            # fallback. A caller that knows the design should pass df so the
            # derived stderr uses the same small-sample reference as the
            # per-scale test.
            critical = student_t_ppf(1.0 - (1.0 - level) / 2.0, 1e7)
            se_basis = 'ci_normal_approx'
        if not math.isfinite(critical) or critical <= 0.0:
            return None, {'index': index, 'label': label,
                          'reason': 'confidence level produced no finite '
                                    'critical value'}
        stderr = (high - low) / (2.0 * critical)
    if stderr is None or not math.isfinite(stderr) or stderr <= 0.0:
        return None, {'index': index, 'label': label,
                      'reason': 'stderr must be finite and > 0 for '
                                'inverse-variance pooling'}
    variance = stderr * stderr
    try:
        weight = 1.0 / variance
    except ZeroDivisionError:
        weight = float('inf')
    if variance == 0.0 or not math.isfinite(variance) or not math.isfinite(weight):
        # Squaring an extreme (but individually finite) stderr can underflow
        # to zero or overflow, and the inverse-variance weight can then be
        # infinite. Such a study carries no representable precision; exclude
        # it instead of letting the pool divide by zero or by infinity.
        return None, {'index': index, 'label': label,
                      'reason': 'stderr is too small or too large for a finite '
                                'inverse-variance weight'}
    return {
        'label': label,
        'effect': effect,
        'stderr': stderr,
        'se_basis': se_basis,
        'ci_level': _as_real(observation.get('ci_level')),
        'source_index': index,
    }, None


def meta_analysis(observations: Sequence[dict], alpha: float = 0.05,
                  min_studies: int = 2) -> dict:
    """Inverse-variance and random-effects meta-analysis of per-scale effects.

    A multi-scale claim is not a vote on signs. Each model scale estimates the
    same contrast with a different standard error, and two scales can agree in
    sign while disagreeing by far more than their own noise. Pooling by
    inverse variance answers "what is the common effect?"; the
    DerSimonian-Laird random-effects estimate answers "what is the average
    effect across the population of scales?"; and Cochran's Q / I^2 / tau^2
    answer "is there one effect to pool at all?".

    Required per observation: an effect under any of ``effect``, ``delta``,
    ``estimate`` or ``mean_delta``, and a positive standard error under
    ``stderr``, ``se`` or ``standard_error``. If the standard error is absent,
    a finite ``ci_low``/``ci_high`` pair is inverted using ``ci_level``
    (default 95%) and ``df`` (or ``n_seeds``/``n``); a missing df falls back
    to the normal limit and is labelled ``ci_normal_approx``. An optional
    ``label``/``scale``/``name`` identifies the study. Invalid records are
    excluded with a reason rather than silently shrinking the pool.

    Negative effects are the repository's convention for "candidate better".
    ``p_one_sided`` is the lower-tail probability of the pooled statistic
    (small when the pooled effect is negative), and ``p_two_sided`` doubles
    the smaller tail. Because the number of scales is small, the pooled tests
    and intervals use Student-t with ``k - 1`` degrees of freedom rather than
    the normal limit.

    The prediction interval (available with >= 3 studies) adds tau^2 to the
    squared standard error and uses ``k - 2`` degrees of freedom: it says where
    the effect at a *new* scale would plausibly land, not where the pooled mean
    is. It is deliberately wider than the confidence interval.

    This function never certifies a multi-scale claim by itself. A pooled
    estimate with substantial heterogeneity is an average of unlike effects;
    the caller must still show that the individual scales pass their own
    tests and must report the heterogeneity rather than only the pooled
    number.
    """
    alpha = _validate_alpha(alpha)
    min_studies = _strict_int(min_studies, "min_studies", minimum=2)
    studies = []
    excluded = []
    for index, observation in enumerate(observations or []):
        record, problem = _meta_study(observation, index)
        if problem is not None:
            excluded.append(problem)
        else:
            studies.append(record)

    k = len(studies)
    out = {
        'test': 'meta_analysis',
        'method': 'DerSimonian-Laird random effects; inverse-variance fixed effect',
        'effect_direction': ('negative effect favours the candidate; '
                             'positive favours the reference'),
        'alpha': alpha,
        'ci_level': 1.0 - alpha,
        'n_studies': k,
        'min_studies_for_pooling': min_studies,
        'meaningful': bool(k >= min_studies),
        'studies': studies,
        'excluded': excluded,
        'fixed_effect': None,
        'random_effects': None,
        'heterogeneity': None,
        'prediction_interval': {
            'level': 1.0 - alpha, 'low': None, 'high': None,
            'meaningful': False, 'df': None,
            'note': 'a prediction interval for a new scale needs >= 3 studies',
        },
        'n_negative': sum(1 for study in studies if study['effect'] < 0.0),
        'n_positive': sum(1 for study in studies if study['effect'] > 0.0),
        'n_zero': sum(1 for study in studies if study['effect'] == 0.0),
        'direction_consistent': None,
        'pooled_effect': None,
        'pooled_effect_source': None,
        'note': None,
    }
    if k == 0:
        out['note'] = ('no study has a finite effect and a positive stderr; '
                       'nothing can be pooled')
        return out

    # Work with weights normalized by their maximum: each individual inverse-
    # variance weight is finite (checked in `_meta_study`), but their sum can
    # still overflow for very small stderr values. Scaling preserves the
    # pooled estimate and yields a representable standard error via
    # (1/sqrt(max_weight)) / sqrt(sum_scaled).
    for study in studies:
        study['fixed_weight'] = 1.0 / study['stderr'] ** 2
    max_fixed_weight = max(study['fixed_weight'] for study in studies)
    fixed_scaled = [study['fixed_weight'] / max_fixed_weight for study in studies]
    total_scaled = math.fsum(fixed_scaled)
    fixed_pooled = math.fsum(
        weight * study['effect'] for weight, study in zip(fixed_scaled, studies)
    ) / total_scaled
    fixed_stderr = ((1.0 / math.sqrt(max_fixed_weight))
                    / math.sqrt(total_scaled))
    for study, weight in zip(studies, fixed_scaled):
        study['fixed_weight_share'] = weight / total_scaled

    out['direction_consistent'] = bool(
        out['n_zero'] == 0 and (out['n_negative'] == k or out['n_positive'] == k))

    fixed = {
        'pooled_effect': fixed_pooled,
        'stderr': fixed_stderr,
        'ci_low': None, 'ci_high': None,
        't': None, 'df': None,
        'p_one_sided': None, 'p_two_sided': None,
    }
    if k >= 2:
        df = float(k - 1)
        critical = student_t_ppf(1.0 - alpha / 2.0, df)
        fixed.update({
            'ci_low': fixed_pooled - critical * fixed_stderr,
            'ci_high': fixed_pooled + critical * fixed_stderr,
            't': fixed_pooled / fixed_stderr,
            'df': df,
        })
        fixed['p_one_sided'] = student_t_cdf(fixed['t'], df)
        fixed['p_two_sided'] = two_sided_p_from_t(fixed['t'], df)
    out['fixed_effect'] = fixed

    heterogeneity = {
        'q': None, 'df': None, 'p': None,
        'i_squared': None, 'tau_squared': None, 'tau': None,
    }
    tau_squared = 0.0
    random_pooled = fixed_pooled
    random_stderr = fixed_stderr
    if k >= 2:
        df_q = float(k - 1)
        q_scaled = math.fsum(
            weight * (study['effect'] - fixed_pooled) ** 2
            for weight, study in zip(fixed_scaled, studies))
        q = max_fixed_weight * q_scaled
        if q > 0.0:
            i_squared = (100.0 if math.isinf(q)
                         else max(0.0, (q - df_q) / q) * 100.0)
        else:
            i_squared = 0.0
        weight_sq_scaled = math.fsum(weight * weight for weight in fixed_scaled)
        tau_denominator_scaled = (
            total_scaled - weight_sq_scaled / total_scaled)
        if tau_denominator_scaled > 0.0:
            tau_squared = max(
                0.0,
                (q_scaled - df_q / max_fixed_weight) / tau_denominator_scaled)
        heterogeneity.update({
            'q': q,
            'df': df_q,
            'p': chi2_sf(q, df_q),
            'i_squared': i_squared,
            'tau_squared': tau_squared,
            'tau': math.sqrt(tau_squared),
        })
        random_weights = [1.0 / (study['stderr'] ** 2 + tau_squared)
                          for study in studies]
        max_random_weight = max(random_weights) if random_weights else 0.0
        if math.isfinite(max_random_weight) and max_random_weight > 0.0:
            random_scaled = [weight / max_random_weight
                             for weight in random_weights]
            total_random_scaled = math.fsum(random_scaled)
            random_pooled = math.fsum(
                weight * study['effect']
                for weight, study in zip(random_scaled, studies)
            ) / total_random_scaled
            random_stderr = ((1.0 / math.sqrt(max_random_weight))
                             / math.sqrt(total_random_scaled))
            for study, weight in zip(studies, random_scaled):
                study['random_weight'] = weight * max_random_weight
                study['random_weight_share'] = weight / total_random_scaled
        else:
            # An infinite tau^2 gives every study zero random weight (an
            # unbounded between-scale variance): report the unweighted mean
            # with an infinite standard error instead of dividing by zero.
            random_pooled = math.fsum(study['effect'] for study in studies) / k
            random_stderr = float('inf')
    out['heterogeneity'] = heterogeneity

    random_effects = {
        'pooled_effect': random_pooled,
        'stderr': random_stderr,
        'ci_low': None, 'ci_high': None,
        't': None, 'df': None,
        'p_one_sided': None, 'p_two_sided': None,
        'tau_squared': tau_squared,
    }
    if k >= 2:
        df = float(k - 1)
        critical = student_t_ppf(1.0 - alpha / 2.0, df)
        random_effects.update({
            'ci_low': random_pooled - critical * random_stderr,
            'ci_high': random_pooled + critical * random_stderr,
            't': random_pooled / random_stderr,
            'df': df,
        })
        random_effects['p_one_sided'] = student_t_cdf(
            random_effects['t'], df)
        random_effects['p_two_sided'] = two_sided_p_from_t(
            random_effects['t'], df)
    out['random_effects'] = random_effects
    out['pooled_effect'] = random_pooled
    out['pooled_effect_source'] = 'random_effects'

    if k >= 3:
        df_pi = float(k - 2)
        critical = student_t_ppf(1.0 - alpha / 2.0, df_pi)
        scale = math.sqrt(tau_squared + random_stderr ** 2)
        out['prediction_interval'] = {
            'level': 1.0 - alpha,
            'low': random_pooled - critical * scale,
            'high': random_pooled + critical * scale,
            'meaningful': True,
            'df': df_pi,
            'note': ('where the effect at a new scale would plausibly fall; '
                     'wider than the confidence interval for the pooled mean'),
        }

    i_squared = heterogeneity['i_squared']
    q_p = heterogeneity['p']
    if k < min_studies:
        out['note'] = (
            f'{k} usable study/studies, fewer than the {min_studies} needed '
            'to pool: the estimate is descriptive')
    elif (i_squared is not None and i_squared >= 50.0) or (
            q_p is not None and q_p < alpha):
        out['note'] = (
            'substantial between-scale heterogeneity '
            f"(I^2={i_squared:.1f}%, Q p={q_p:.3g}): the pooled number is an "
            'average of unlike effects, so report the per-scale results and '
            'this heterogeneity rather than a single replicated effect')
    else:
        out['note'] = (
            'no substantial between-scale heterogeneity detected at this '
            'alpha; the pooled estimate is descriptive and does not replace '
            'each scale passing its own pre-specified test')
    return out


def scale_replication_summary(observations: Sequence[dict],
                               min_scales: int = 2,
                               alpha: float = 0.05) -> dict:
    """Summarise whether an effect replicates across model scales.

    A single-scale result is one observation about one architecture; it is
    not evidence that the same mechanism wins at another width/depth. This
    helper is deliberately format-agnostic: callers pass one record per
    (scale, metric) with the effect sign and whether that scale's test was
    certified. It reports the direction per scale and only calls an effect
    replicated when at least `min_scales` distinct scales agree in sign.

    A sign vote is necessary but not sufficient: the scales can agree while
    differing by much more than their own noise. Each per-metric block also
    carries ``meta_analysis`` (see `meta_analysis`), which pools the effects
    by inverse variance, adds a random-effects estimate, and reports Cochran's
    Q / I^2 / tau^2 plus a prediction interval for a new scale. That block is
    only populated when the callers supply a standard error or a usable
    confidence interval; a sign-only record still contributes to the vote but
    is listed in the meta-analysis's ``excluded`` list rather than being
    assigned an invented precision.

    Required keys per observation: `scale` (str), `delta` (finite number)
    and `certified` (bool). Optional: `metric` (str), `contrast`, `stderr`
    (or `se`), `ci_low`/`ci_high` (+ `df` or `n_seeds`, `ci_level`), and
    `p`/`p_one_sided`. Records with a non-finite delta are excluded from the
    sign vote but kept in `excluded` so a failed scale is visible rather than
    silently dropped.
    """
    min_scales = _strict_int(min_scales, "min_scales", minimum=2)
    alpha = _validate_alpha(alpha)
    by_metric: dict[tuple[str, str], dict[tuple[str, str, str], dict]] = {}
    duplicates: list[dict] = []
    excluded: list[dict] = []
    for index, obs in enumerate(observations or []):
        if not isinstance(obs, dict):
            excluded.append({'index': index, 'reason': 'not an object'})
            continue
        scale = str(obs.get('scale') or '').strip()
        metric = str(obs.get('metric') or '').strip() or 'default'
        # A comparison script can emit many pairwise contrasts per scale and
        # metric (the default FP8 run has 15 pairs). Grouping only by
        # (scale, metric) overwrites all but the last pair, so the reported
        # sign can even belong to a different contrast than the certified one.
        # An explicit `contrast` keeps each pair as its own replication set;
        # callers without one retain the original (scale, metric) identity.
        contrast = str(obs.get('contrast') or '').strip()
        delta = obs.get('delta')
        if not scale:
            excluded.append({'index': index, 'reason': 'missing scale'})
            continue
        if _is_boolean_value(delta) or delta is None:
            excluded.append({'index': index, 'scale': scale, 'metric': metric,
                             'reason': f'non-finite delta {delta!r}'})
            continue
        try:
            delta_value = float(delta)
        except (TypeError, ValueError, OverflowError):
            excluded.append({'index': index, 'scale': scale, 'metric': metric,
                             'reason': f'non-finite delta {delta!r}'})
            continue
        if not math.isfinite(delta_value):
            excluded.append({'index': index, 'scale': scale, 'metric': metric,
                             'reason': f'non-finite delta {delta!r}'})
            continue
        key = (scale, metric, contrast)
        record = {
            'scale': scale,
            'metric': metric,
            'contrast': contrast or None,
            'delta': delta_value,
            # Only an explicit boolean True certifies a scale. `bool("False")`
            # is True, so a malformed/legacy string must not be promoted into
            # a multi-scale certification.
            'certified': obs.get('certified') is True,
            'n_seeds': obs.get('n_seeds'),
            # `dict.get(key, fallback)` only falls back when the key is
            # absent. A producer that emits `p_one_sided: None` alongside a
            # real `p` / `se` must not have that value silently discarded.
            'p': (obs.get('p_one_sided')
                  if obs.get('p_one_sided') is not None else obs.get('p')),
            # Precision for the meta-analytic block. These are optional: a
            # sign-only observation is still a valid vote, but meta_analysis
            # will report it as excluded instead of inventing a stderr.
            'stderr': (obs.get('stderr')
                       if obs.get('stderr') is not None else obs.get('se')),
            'ci_low': obs.get('ci_low'),
            'ci_high': obs.get('ci_high'),
            'df': obs.get('df'),
            'ci_level': obs.get('ci_level'),
        }
        bucket = by_metric.setdefault((metric, contrast), {})
        if key in bucket:
            # A duplicate is one underlying observation, not another scale.
            # Keep the first and expose both values instead of overwriting.
            duplicates.append({'scale': scale, 'metric': metric,
                               'contrast': contrast or None,
                               'kept_delta': bucket[key]['delta'],
                               'duplicate_delta': float(delta)})
            continue
        bucket[key] = record

    per_metric: dict[str, dict] = {}
    used_labels: set[str] = set()
    all_deltas: list[tuple[str, str, float]] = []
    for (metric, contrast), bucket in by_metric.items():
        label = f'{metric} [{contrast}]' if contrast else metric
        if label in used_labels:
            base = label
            suffix = 2
            while f'{base} #{suffix}' in used_labels:
                suffix += 1
            label = f'{base} #{suffix}'
        used_labels.add(label)
        records = list(bucket.values())
        signs = []
        for record in records:
            if record['delta'] < 0.0:
                signs.append('negative')
            elif record['delta'] > 0.0:
                signs.append('positive')
            else:
                signs.append('zero')
            all_deltas.append((label, record['scale'], record['delta']))
        certified = [record for record in records if record['certified']]
        certified_signs = {
            ('negative' if record['delta'] < 0.0 else
             'positive' if record['delta'] > 0.0 else 'zero')
            for record in certified}
        # Zero is a null result, not a direction. An all-zero metric must not
        # be reported as a "replicated direction": no candidate beat the
        # reference anywhere, so the sign vote has no winner to replicate.
        direction_consistent = (len(set(signs)) == 1
                                and signs[0] in ('negative', 'positive'))
        pooled = meta_analysis(records, alpha=alpha, min_studies=min_scales)
        per_metric[label] = {
            'metric': records[0]['metric'] if records else metric,
            'contrast': records[0]['contrast'] if records else None,
            'scales': sorted(record['scale'] for record in records),
            'n_scales': len(records),
            'deltas': {record['scale']: record['delta'] for record in records},
            'signs': {record['scale']: sign for record, sign in zip(records, signs)},
            'direction_consistent': direction_consistent,
            'replicated_direction': bool(direction_consistent
                                         and len(records) >= min_scales),
            'certified_scales': sorted(record['scale'] for record in certified),
            'n_certified_scales': len(certified),
            'multi_scale_certified': bool(direction_consistent
                                          and len(records) >= min_scales
                                          and len(certified) >= min_scales
                                          and len(certified_signs) == 1),
            'meta_analysis': pooled,
        }

    scales = sorted({record['scale'] for bucket in by_metric.values()
                     for record in bucket.values()})
    overall_signs = {('negative' if delta < 0.0 else 'positive' if delta > 0.0 else 'zero')
                     for _, _, delta in all_deltas}
    replicated_metrics = sorted(metric for metric, block in per_metric.items()
                                if block['replicated_direction'])
    multi_certified = sorted(metric for metric, block in per_metric.items()
                             if block['multi_scale_certified'])
    pooled_metrics = sorted(
        metric for metric, block in per_metric.items()
        if block.get('meta_analysis') and block['meta_analysis']['meaningful'])
    return {
        'n_scales': len(scales),
        'scales': scales,
        'single_scale_only': len(scales) < min_scales,
        'min_scales_for_replication': min_scales,
        'n_observations': len(all_deltas),
        'direction_signs': sorted(overall_signs),
        'direction_consistent': (len(overall_signs) == 1 and bool(all_deltas)
                                 and next(iter(overall_signs)) in ('negative', 'positive')),
        'replicated_metrics': replicated_metrics,
        'multi_scale_certified_metrics': multi_certified,
        'meta_analysed_metrics': pooled_metrics,
        'per_metric': per_metric,
        'duplicate_observations': duplicates,
        'excluded': excluded,
        'reading': (f'an effect is replicated only when >= {min_scales} distinct model '
                    'scales agree in sign; it is certified across scales only when >= '
                    f'{min_scales} scales individually pass their own test. One scale is '
                    'a scope-limited result, not a mechanism claim. The per-metric '
                    '`meta_analysis` block pools the effects by inverse variance and '
                    'reports between-scale heterogeneity (Q, I^2, tau^2); a pooled '
                    'number under substantial heterogeneity is an average of unlike '
                    'effects, not one replicated mechanism.'),
    }

def f1_from_counts(
    gold_counts: Sequence[int],
    pred_counts: Sequence[int],
    correct_counts: Sequence[int],
) -> dict:
    """Multiclass F1 from per-class counts, with the averaging named.

    The three vectors are the class marginals of the confusion matrix:
    `gold_counts[c]` support of class `c`, `pred_counts[c]` how often class
    `c` was predicted, `correct_counts[c]` their overlap (the diagonal).
    Streaming evaluations accumulate these vectors instead of the full
    `pred`/`gold` lists; the semantics are identical to `macro_weighted_f1`,
    which delegates here.

    * `macro_f1`: unweighted mean of per-class F1 over classes occurring in
      `gold` or `pred`. Every class counts once, so rare classes are not
      drowned by frequent ones.
    * `weighted_f1`: per-class F1 weighted by the class's support in `gold`.
    * `micro_f1`: total true positives over total support (== accuracy for a
      single-label argmax prediction).

    With class imbalance macro and micro can rank models differently, so call
    sites report the metric by name.
    """
    gold_counts = [_strict_int(v, "gold_counts", minimum=0) for v in gold_counts]
    pred_counts = [_strict_int(v, "pred_counts", minimum=0) for v in pred_counts]
    correct_counts = [_strict_int(v, "correct_counts", minimum=0)
                      for v in correct_counts]
    if not (len(gold_counts) == len(pred_counts) == len(correct_counts)):
        raise ValueError(
            "gold/pred/correct count vectors must have the same length, got "
            f"{len(gold_counts)}/{len(pred_counts)}/{len(correct_counts)}"
        )
    for i, (g, p, t) in enumerate(zip(gold_counts, pred_counts, correct_counts)):
        if g < 0 or p < 0 or t < 0 or t > min(g, p):
            raise ValueError(
                f"inconsistent counts for class {i}: gold={g}, pred={p}, "
                f"correct={t} (need 0 <= correct <= min(gold, pred))"
            )
    if sum(gold_counts) != sum(pred_counts):
        # The three vectors are marginals of one confusion matrix, so a
        # single-label prediction cannot add or drop examples. Unequal totals
        # make micro_f1 (= accuracy) meaningless while still looking plausible.
        raise ValueError(
            "gold/pred count vectors must describe the same examples, got "
            f"sum(gold)={sum(gold_counts)} vs sum(pred)={sum(pred_counts)}")
    tp = correct_counts
    fp = [p - t for p, t in zip(pred_counts, correct_counts)]
    fn = [g - t for g, t in zip(gold_counts, correct_counts)]
    support = sum(gold_counts)
    present = [i for i in range(len(gold_counts)) if tp[i] + fp[i] + fn[i] > 0]
    f1_by_class = {}
    for i in present:
        denom = 2 * tp[i] + fp[i] + fn[i]
        f1_by_class[i] = (2.0 * tp[i] / denom) if denom else 0.0
    macro = sum(f1_by_class.values()) / len(f1_by_class) if f1_by_class else 0.0
    weighted = (
        sum(f1_by_class[i] * (tp[i] + fn[i]) for i in present) / support
        if support else 0.0
    )
    micro = (sum(tp) / support) if support else 0.0
    # Per-class precision/recall, then macro/weighted/micro aggregates, so a
    # caller whose research goal is recall (e.g. medical screening) or
    # precision (e.g. alerting) can read the matching named quantity instead
    # of inferring it from F1.
    precision_by_class = {}
    recall_by_class = {}
    for i in present:
        denom_p = tp[i] + fp[i]
        denom_r = tp[i] + fn[i]
        precision_by_class[i] = (tp[i] / denom_p) if denom_p else 0.0
        recall_by_class[i] = (tp[i] / denom_r) if denom_r else 0.0
    macro_precision = (sum(precision_by_class.values()) / len(precision_by_class)
                       if precision_by_class else 0.0)
    macro_recall = (sum(recall_by_class.values()) / len(recall_by_class)
                    if recall_by_class else 0.0)
    weighted_precision = (
        sum(precision_by_class[i] * (tp[i] + fn[i]) for i in present) / support
        if support else 0.0
    )
    weighted_recall = (
        sum(recall_by_class[i] * (tp[i] + fn[i]) for i in present) / support
        if support else 0.0
    )
    micro_precision = (sum(tp) / (sum(tp) + sum(fp))) if (sum(tp) + sum(fp)) else 0.0
    micro_recall = (sum(tp) / (sum(tp) + sum(fn))) if (sum(tp) + sum(fn)) else 0.0
    return {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "micro_f1": micro,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "micro_precision": micro_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "micro_recall": micro_recall,
        "n_classes_present": len(present),
        "support": support,
    }


def macro_weighted_f1(pred: Sequence[int], gold: Sequence[int], num_classes: int) -> dict:
    """Multiclass F1 with the averaging explicitly named.

    Convenience wrapper around `f1_from_counts` for callers that hold the
    full prediction/target lists (small batches); see that function for the
    macro/weighted/micro semantics.
    """
    num_classes = _strict_int(num_classes, "num_classes", minimum=1)
    pred = [_strict_int(p, "pred class id") for p in pred]
    gold = [_strict_int(g, "gold class id") for g in gold]
    if len(pred) != len(gold):
        raise ValueError(f"pred/gold length mismatch: {len(pred)} vs {len(gold)}")
    gold_counts = [0] * num_classes
    pred_counts = [0] * num_classes
    correct_counts = [0] * num_classes
    for p, g in zip(pred, gold):
        if not 0 <= g < num_classes or not 0 <= p < num_classes:
            raise ValueError(f"class id out of range [0, {num_classes})")
        gold_counts[g] += 1
        pred_counts[p] += 1
        if p == g:
            correct_counts[g] += 1
    return f1_from_counts(gold_counts, pred_counts, correct_counts)


def imbalance_report(counts: Sequence[int]) -> dict:
    """Imbalance of a target-class distribution, with the metric warning.

    Multi-class accuracy is dominated by the majority class; once the
    majority:minority support ratio reaches `1000:1` the minority classes are
    essentially never predicted and accuracy (and `micro_f1`) is
    uninformative. `extreme`/`prefer_pr_auc` flag that regime; there the
    honest summaries are per-class F1 (`macro_f1`) and, for a decision score,
    PR-AUC (`average_precision`) - not accuracy and not ROC-AUC, whose false
    positive rate has a huge negative denominator and therefore rises too
    slowly to expose the collapse.
    """
    # A class count is a non-negative integer; a negative/fractional/bool
    # value is a malformed distribution, not a class to silently drop.
    all_counts = [_strict_int(c, "class count", minimum=0) for c in counts]
    positives = [c for c in all_counts if c > 0]
    if not positives:
        return {
            "n_classes_with_support": 0,
            "majority_count": None,
            "minority_count": None,
            "majority_minority_ratio": None,
            "extreme": False,
            "prefer_pr_auc": False,
            "note": "no class has support; imbalance is undefined",
        }
    majority = max(positives)
    minority = min(positives)
    ratio = majority / minority
    extreme = ratio >= _EXTREME_IMBALANCE
    return {
        "n_classes_with_support": len(positives),
        "majority_count": majority,
        "minority_count": minority,
        "majority_minority_ratio": ratio,
        "extreme": extreme,
        "prefer_pr_auc": extreme,
        "suggested_primary_metric": (
            "average_precision (PR-AUC) for the minority classes, plus macro_f1"
            if extreme else "macro_f1 (and accuracy only if the classes are balanced)"
        ),
        "note": (
            f"majority:minority = {ratio:.1f}:1"
            + (
                f" exceeds {_EXTREME_IMBALANCE:.0f}:1: accuracy/micro_f1 and "
                "ROC-AUC overstate performance; read macro_f1 and PR-AUC"
                if extreme else ""
            )
        ),
    }


def _tie_groups_ascending(scores: list, labels: list):
    """``(order, groups)`` of one ascending sort, ties collected into groups.

    `order` is the ascending permutation and `groups` the ``(start, end)``
    index pairs (inclusive) into ``order`` for each equal-score run. The
    group contents are order-insensitive downstream (average rank for ROC-AUC,
    threshold groups for AP), so neither metric depends on the tie order.
    """
    n = len(scores)
    order = sorted(range(n), key=scores.__getitem__)
    groups = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        groups.append((i, j))
        i = j + 1
    return order, groups


def _roc_auc_from_groups(n: int, n_pos: int, n_neg: int, order: list,
                         groups: list, labels: list) -> float:
    ranks = [0.0] * n
    for i, j in groups:
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
    rank_sum_pos = sum(r for r, lab in zip(ranks, labels) if lab)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision_from_groups(n_pos: int, order: list, groups: list,
                                   labels: list) -> float:
    ap = 0.0
    tp = fp = 0
    for i, j in reversed(groups):
        group = order[i:j + 1]
        group_pos = sum(labels[k] for k in group)
        tp += group_pos
        fp += len(group) - group_pos
        if group_pos:
            ap += (tp / (tp + fp)) * (group_pos / n_pos)
    return ap


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Binary ROC-AUC via the rank statistic, ties averaged (`None` if undefined).

    `labels` are 0/1 (1 = positive) and `scores` rank higher = more positive.
    Equal scores share their average rank, so a model gets no credit for an
    arbitrary ordering inside a tie.

    Caveat: ROC-AUC only counts ranking inversions between a positive and a
    negative and is insensitive to absolute probability levels. With
    catastrophic imbalance (>= ~1000 negatives per positive) the false
    positive rate moves in steps of 1/n_neg, so the curve can look strong
    even when every operating point usable in practice has low precision;
    use `average_precision` (PR-AUC) there.
    """
    scores = [float(s) for s in scores]
    labels = _binary_labels(labels)
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} vs {len(labels)}")
    if any(not math.isfinite(s) for s in scores):
        # A NaN score has no rank: Python's sort leaves it wherever it landed,
        # so the same positive/negative pair can produce different AUCs merely
        # by reversing the input list. Refuse the malformed measurement.
        raise ValueError("roc_auc scores must be finite numbers; a NaN/Inf score has no rank")
    n = len(scores)
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    order, groups = _tie_groups_ascending(scores, labels)
    return _roc_auc_from_groups(n, n_pos, n_neg, order, groups, labels)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Binary average precision (PR-AUC); `None` when there is no positive.

    Average precision is the step-wise area under the precision-recall curve:
    every threshold group contributes `precision * delta_recall`, so a
    positive ranked below many negatives drags the score down directly.
    Unlike ROC-AUC it does not reward a model merely for placing positives
    above negatives in the far tail, which is what makes it the right summary
    under extreme imbalance. Ties are treated as one threshold group
    (matching `sklearn.average_precision_score`): random order inside a tie
    earns no precision credit.
    """
    scores = [float(s) for s in scores]
    labels = _binary_labels(labels)
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} vs {len(labels)}")
    if any(not math.isfinite(s) for s in scores):
        # Same order-dependence as roc_auc: NaN has no threshold position.
        raise ValueError("average_precision scores must be finite numbers; "
                         "a NaN/Inf score has no rank")
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    order, groups = _tie_groups_ascending(scores, labels)
    return _average_precision_from_groups(n_pos, order, groups, labels)


def binary_auc_report(
    scores: Sequence[Sequence[float]],
    gold: Sequence[int],
    classes: Sequence[int],
) -> dict:
    """One-vs-rest PR-AUC/ROC-AUC per class, with named macro/weighted means.

    `scores` is N x K (row per example, column per class in `classes`) and
    `gold` the N target class ids. For every requested class the report
    contains its support, negatives-per-positive ratio, `average_precision`
    (PR-AUC) and `roc_auc`; classes with only positives or only negatives
    cannot be scored and are listed in `skipped_classes` (they are excluded
    from the means rather than silently scored as 0/1).

    `macro_average_precision` is the unweighted mean over evaluated classes
    (every class counts once, rare classes included); the `weighted_*`
    siblings weight each class by its positive support. They are reported
    under those full names because macro and weighted can disagree sharply
    under imbalance. `imbalance.extreme` (>= 1000 negatives per positive on
    any evaluated class) sets `prefer_pr_auc`; in that regime ROC-AUC is
    optimistic and PR-AUC is the metric to read.
    """
    classes = [_strict_int(c, "class id") for c in classes]
    gold = [_strict_int(g, "gold class id") for g in gold]
    rows = list(scores)
    if len(rows) != len(gold):
        raise ValueError(f"score-row/gold length mismatch: {len(rows)} vs {len(gold)}")
    # One column per class; every row must carry exactly one score per class.
    bad_width = next((len(r) for r in rows if len(r) != len(classes)), None)
    if bad_width is not None:
        raise ValueError(
            f"score matrix has row(s) with {bad_width} columns but {len(classes)} classes"
        )
    cols = list(zip(*rows)) if rows else []
    # This function ranks each class column itself (it cannot delegate to
    # `roc_auc`/`average_precision`, which score one column at a time), so it
    # needs the same finite-score contract. A NaN/Inf has no rank position:
    # Python's sort leaves it wherever it landed, making the reported AUC
    # depend on the arbitrary order of the input rows.
    try:
        cols = [[float(v) for v in col] for col in cols]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            "binary_auc_report scores must be finite numbers; a non-numeric "
            "score has no rank") from None
    if any(not math.isfinite(v) for col in cols for v in col):
        raise ValueError(
            "binary_auc_report scores must be finite numbers; a NaN/Inf score "
            "has no rank")
    records = []
    skipped = []
    for k, cls in enumerate(classes):
        col = cols[k] if cols else []
        labels = [g == cls for g in gold]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            skipped.append({
                "class": cls,
                "positives": n_pos,
                "negatives": n_neg,
                "reason": "no positives" if n_pos == 0 else "no negatives",
            })
            continue
        order, groups = _tie_groups_ascending(col, labels)
        records.append({
            "class": cls,
            "positives": n_pos,
            "negatives": n_neg,
            "negatives_per_positive": n_neg / n_pos,
            "average_precision": _average_precision_from_groups(n_pos, order, groups, labels),
            "roc_auc": _roc_auc_from_groups(len(labels), n_pos, n_neg, order, groups, labels),
        })
    support_total = sum(r["positives"] for r in records)
    macro_ap = (
        sum(r["average_precision"] for r in records) / len(records)
        if records else None
    )
    weighted_ap = (
        sum(r["average_precision"] * r["positives"] for r in records) / support_total
        if support_total else None
    )
    macro_roc = sum(r["roc_auc"] for r in records) / len(records) if records else None
    weighted_roc = (
        sum(r["roc_auc"] * r["positives"] for r in records) / support_total
        if support_total else None
    )
    ratios = [r["negatives_per_positive"] for r in records]
    extreme = any(r >= _EXTREME_IMBALANCE for r in ratios)
    return {
        "classes": records,
        "n_classes_requested": len(classes),
        "n_classes_evaluated": len(records),
        "skipped_classes": skipped,
        "macro_average_precision": macro_ap,
        "weighted_average_precision": weighted_ap,
        "macro_roc_auc": macro_roc,
        "weighted_roc_auc": weighted_roc,
        "imbalance": {
            "min_negatives_per_positive": min(ratios) if ratios else None,
            "max_negatives_per_positive": max(ratios) if ratios else None,
            "extreme": extreme,
        },
        "prefer_pr_auc": extreme,
        "averaging": (
            "macro = unweighted mean over evaluated classes (every class once); "
            "weighted = mean weighted by that class's positives"
        ),
        "reading_guide": (
            "classes with only positives or only negatives are skipped, not "
            "scored as 0/1; when prefer_pr_auc is true (>= 1000 negatives per "
            "positive) read average_precision (PR-AUC) and ignore roc_auc, "
            "whose FPR has a huge denominator and stays optimistic"
        ),
    }


def recall_at_k(ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]], k: int) -> Optional[float]:
    """Mean recall@k over queries: fraction of relevant items found in top-k.

    `ranked[q]` is the ordered list of item ids predicted for query q and
    `relevant[q]` the set of truly relevant items. Returns None when there are
    no queries. A metric suited to retrieval-style research goals (the
    prompt's "recommendation systems care about NDCG/recall") rather than to
    the classification metrics above.
    """
    ranked = list(ranked)
    relevant = list(relevant)
    if len(ranked) != len(relevant):
        raise ValueError(f"ranked/relevant query count mismatch: {len(ranked)} vs {len(relevant)}")
    # Keep the reported depth honest: silently clamping top-0/1.5 to top-1
    # reports a retrieval result that was not requested.
    k = _strict_int(k, "recall_at_k k", minimum=1)
    if not ranked:
        return None
    recalls = []
    for r, rel in zip(ranked, relevant):
        rel_set = set(rel)
        if not rel_set:
            recalls.append(0.0)
            continue
        hits = len(set(r[:k]) & rel_set)
        recalls.append(hits / len(rel_set))
    return sum(recalls) / len(recalls)


def ndcg_at_k(ranked: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]], k: int) -> Optional[float]:
    """Mean NDCG@k (binary relevance) over queries.

    Each relevant item contributes 1 / log2(rank + 1) (rank is 1-based) in
    the numerator; the denominator is the ideal DCG with all relevant items
    placed at the top. Returns None when there are no queries or no relevant
    items in a query set's union (so the metric is defined).
    """
    ranked = list(ranked)
    relevant = list(relevant)
    if len(ranked) != len(relevant):
        raise ValueError(f"ranked/relevant query count mismatch: {len(ranked)} vs {len(relevant)}")
    # top-0 (or a fractional depth) is not an evaluated depth; do not
    # silently report a different one.
    k = _strict_int(k, "ndcg_at_k k", minimum=1)
    if not ranked:
        return None
    if not any(rel for rel in relevant):
        return None
    ndcgs = []
    for r, rel in zip(ranked, relevant):
        rel_set = set(rel)
        if not rel_set:
            ndcgs.append(0.0)
            continue
        dcg = 0.0
        seen = set()
        for i, item in enumerate(r[:k]):
            # A repeated id is one relevant item, not several: scoring every
            # occurrence would let a padded/duplicated ranking exceed the
            # ideal DCG and push NDCG above 1.
            if item in rel_set and item not in seen:
                dcg += 1.0 / math.log2(i + 2)
                seen.add(item)
        ideal_hits = min(len(rel_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
        ndcgs.append(min(1.0, dcg / idcg) if idcg > 0 else 0.0)
    return sum(ndcgs) / len(ndcgs)
