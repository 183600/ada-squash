from __future__ import annotations

import functools
import math
import numbers
import random
from statistics import NormalDist
from typing import Optional, Sequence

_Z_95 = 1.959963984540054
MIN_SEEDS_FOR_SIGNIFICANCE = 3
# Above this many degrees of freedom the t distribution is indistinguishable
# from the normal limit at double precision, while the incomplete-beta CDF
# loses accuracy (and eventually overflows) for very large df.
_T_NORMAL_APPROX_DF = 1.0e7
# Above this many observations the 2**n sign-flip enumeration is no
# longer a cheap exact computation; use the seeded Monte-Carlo branch.
_RANDOMIZATION_EXACT_MAX = 20
# Above this many discordant pairs the incomplete-beta binomial tail loses
# precision (and eventually overflows); the normal approximation with a
# continuity correction is accurate to far better than the reporting precision.
_MCNEMAR_NORMAL_MIN = 100000
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
    z_value = _as_real(z)
    if z_value is None or not math.isfinite(z_value) or z_value < 0.0:
        raise ValueError(
            f"z must be a finite non-negative number, got {z!r}")
    z = z_value
    if n_success > n_total:
        # 150 successes out of 100 trials is impossible evidence, not a
        # proportion to clamp to 1. Clamping produced a valid-looking
        # interval (0.963, 1.0) for a count that never existed.
        raise ValueError(
            f"wilson_interval needs 0 <= n_success <= n_total, got "
            f"{n_success} successes out of {n_total} trials")
    z2 = z * z
    try:
        p = n_success / n_total
        denom = 1.0 + z2 / n_total
        centre = (p + z2 / (2.0 * n_total)) / denom
        half = (z / denom) * math.sqrt(
            p * (1.0 - p) / n_total + z2 / (4.0 * n_total * n_total)
        )
    except OverflowError:
        inv_n = 1 / n_total
        p = n_success / n_total
        denom = 1.0 + z2 * inv_n
        centre = (p + z2 * inv_n / 2.0) / denom
        half = (z / denom) * math.sqrt(
            p * (1.0 - p) * inv_n + z2 * inv_n * inv_n / 4.0
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
    anti-conservative. Above ``_MCNEMAR_NORMAL_MIN`` discordant pairs the
    incomplete beta loses precision and eventually overflows, so the
    continuity-corrected normal tail is used instead (its error is far below
    the reporting precision at that sample size). `p_one_sided` is
    P(X >= n_fixed) and supports "candidate better"; `p_two_sided` doubles the
    smaller tail (capped at 1). `normal_z_continuity_corrected` is reported
    for reference only.

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
    if n > _MCNEMAR_NORMAL_MIN:
        # Continuity-corrected normal tails. P(X >= n_fixed) uses the lower
        # boundary n_fixed - 1/2; P(X <= n_fixed) uses the upper boundary
        # n_fixed + 1/2. Both are evaluated through int/sqrt ratios so huge
        # Python-int counts never need conversion to float.
        z_ge = _int_over_sqrt(n_fixed - n_broke - 1, n)
        z_le = _int_over_sqrt(n_fixed - n_broke + 1, n)
        p_ge = _p_one_sided_from_z(-z_ge)
        p_le = _p_one_sided_from_z(z_le)
    else:
        # P(X <= k) for X ~ Bin(n, 1/2) is I_{1/2}(n - k, k + 1); the one-sided
        # p-value P(X >= n_fixed) is the mirror image I_{1/2}(n_fixed, n_broke + 1).
        p_ge = _reg_incomplete_beta(n_fixed, n_broke + 1, 0.5) if n_fixed > 0 else 1.0
        p_le = _reg_incomplete_beta(n_broke, n_fixed + 1, 0.5) if n_broke > 0 else 1.0
    out["p_one_sided"] = p_ge
    out["p_two_sided"] = min(1.0, 2.0 * min(p_ge, p_le))
    out["favors"] = "candidate" if n_fixed > n_broke else ("reference" if n_broke > n_fixed else "tied")
    difference = abs(n_fixed - n_broke)
    out["normal_z_continuity_corrected"] = (
        0.0 if difference <= 1 else _int_over_sqrt(difference - 1, n))
    out["note"] = (
        "p_one_sided = P(Binomial(n_discordant, 0.5) >= n_fixed); the exact test "
        "is the small-discordant-count reference (the normal approximation is "
        "anti-conservative there)"
    )
    return out


def _mean_and_std(values: Sequence[float]):
    """Mean and sample standard deviation without overflow of the sums.

    Finite observations near the float limit can overflow a sum, a squared
    deviation, or the sample variance even when both the mean and the standard
    deviation are representable. The ordinary path is kept unchanged; if it
    overflows or produces a non-finite result, the values are scaled by their
    maximum magnitude and mapped back.
    """
    n = len(values)
    if n == 0:
        return None, None
    mean = std = float("nan")
    try:
        mean = sum(values) / n
        if n < 2:
            return mean, 0.0
        variance = sum((value - mean) * (value - mean)
                       for value in values) / (n - 1)
        std = math.sqrt(variance)
        if math.isfinite(mean) and math.isfinite(std):
            return mean, std
    except OverflowError:
        pass
    if not all(math.isfinite(value) for value in values):
        return mean, std
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0, 0.0
    scaled = [value / scale for value in values]
    mean_scaled = math.fsum(scaled) / n
    if n < 2:
        return mean_scaled * scale, 0.0
    variance_scaled = math.fsum(
        (value - mean_scaled) * (value - mean_scaled)
        for value in scaled) / (n - 1)
    return mean_scaled * scale, math.sqrt(variance_scaled) * scale


def _paired_mean_std(a: Sequence[float], b: Sequence[float]):
    """Mean and sample std of paired differences without subtract overflow.

    Two finite observations near opposite float limits have a difference
    larger than the float range: ``x - y`` becomes +/-inf and the ordinary
    mean/std path then sees an infinite sample and returns NaN. Scaling both
    inputs by their common maximum magnitude keeps every difference in
    [-2, 2]; the true mean/std are mapped back afterwards, and the scaled
    triple is returned so the t/z ratio and confidence interval can be
    computed before that (possibly overflowing) mapping.
    """
    d = [x - y for x, y in zip(a, b)]
    mean, std = _mean_and_std(d)
    if math.isfinite(mean) and math.isfinite(std):
        return mean, std, None
    if not (all(math.isfinite(value) for value in a)
            and all(math.isfinite(value) for value in b)):
        return mean, std, None
    scale = max(max(abs(value) for value in a),
                max(abs(value) for value in b))
    if scale == 0.0:
        return 0.0, 0.0, None
    scaled = [x / scale - y / scale for x, y in zip(a, b)]
    mean_scaled, std_scaled = _mean_and_std(scaled)
    return (mean_scaled * scale, std_scaled * scale,
            (mean_scaled, std_scaled, scale))


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
    mean, std = _mean_and_std(vals)
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


def _int_over_sqrt(value: int, n: int) -> float:
    """``value / sqrt(n)`` for arbitrary Python ints, without float overflow.

    ``math.sqrt(n)`` raises ``OverflowError`` when ``n`` exceeds the float
    range, and ``float(value)`` can do the same for the numerator. Dividing
    the two ints directly lets Python's long true-division scale the quotient;
    only a genuinely unrepresentable result becomes an infinity.
    """
    try:
        return value / math.isqrt(n)
    except OverflowError:
        return math.copysign(math.inf, value)


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
    x_value = _as_real(x)
    if x_value is None:
        raise ValueError("chi2_sf got a non-finite statistic")
    x = x_value
    df_value = _as_real(df)
    if df_value is None or not math.isfinite(df_value) or df_value <= 0.0:
        raise ValueError("chi2_sf needs df > 0")
    df = df_value
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
    t_value = _as_real(t)
    if t_value is None:
        raise ValueError("student_t_cdf got a non-numeric statistic")
    t = t_value
    df_value = _as_real(df)
    if df_value is None or not math.isfinite(df_value) or df_value <= 0:
        raise ValueError("student_t_cdf needs df > 0")
    df = df_value
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
    p_value = _as_real(p)
    if p_value is None or not 0.0 < p_value < 1.0:
        raise ValueError("student_t_ppf needs 0 < p < 1")
    p = p_value
    df_value = _as_real(df)
    if df_value is None or not math.isfinite(df_value) or df_value <= 0:
        raise ValueError("student_t_ppf needs df > 0")
    df = df_value
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
    mean, std, scaled = _paired_mean_std(a, b)
    if n < 2:
        return {"mean_delta": mean, "z": None, "p_one_sided": None, "n": n, "paired": True}
    if scaled is not None:
        mean_test, se = scaled[0], scaled[1] / math.sqrt(n)
        margin_test = margin / scaled[2]
    else:
        mean_test, se, margin_test = mean, std / math.sqrt(n), margin
    if se == 0.0:
        z = float("-inf") if mean_test + margin_test < 0.0 else (
            float("inf") if mean_test + margin_test > 0.0 else 0.0)
    else:
        z = (mean_test + margin_test) / se
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
    mean, std, scaled = _paired_mean_std(a, b)
    if n < 2:
        empty["mean_delta"] = mean
        return empty
    df = float(n - 1)
    if scaled is not None:
        mean_test, se, ci_scale = scaled[0], scaled[1] / math.sqrt(n), scaled[2]
        margin_test = margin / ci_scale
    else:
        mean_test, se, ci_scale, margin_test = mean, std / math.sqrt(n), 1.0, margin
    if se == 0.0:
        t = float("-inf") if mean_test + margin_test < 0.0 else (
            float("inf") if mean_test + margin_test > 0.0 else 0.0)
    else:
        t = (mean_test + margin_test) / se
    half = student_t_ppf(1.0 - alpha / 2.0, df) * se
    ci_low = (mean_test - half) * ci_scale
    ci_high = (mean_test + half) * ci_scale
    return {
        "mean_delta": mean,
        "t": t,
        "df": df,
        "p_one_sided": student_t_cdf(t, df),
        "ci_low": ci_low,
        "ci_high": ci_high,
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

    An ``n`` beyond the float range is a legitimate (if absurd) power-plan
    input, not a reason to crash. Above ``_T_NORMAL_APPROX_DF`` degrees of
    freedom the t quantiles are computed from the normal limit, which is
    accurate to better than double precision there; ``1 / sqrt(n)`` is
    evaluated through its logarithm when converting ``n`` to float would
    overflow.
    """
    try:
        df = float(n - 1)
    except OverflowError:
        df = math.inf
    if not math.isfinite(df) or df > _T_NORMAL_APPROX_DF:
        normal = NormalDist()
        critical = normal.inv_cdf(1.0 - alpha) + normal.inv_cdf(power)
    else:
        critical = student_t_ppf(1.0 - alpha, df) + student_t_ppf(power, df)
    try:
        inverse_sqrt_n = 1.0 / math.sqrt(n)
    except OverflowError:
        inverse_sqrt_n = math.exp(-0.5 * math.log(n))
    return critical * std * inverse_sqrt_n


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
    mean, std, scaled = _paired_mean_std(a, b)
    if scaled is None and not math.isfinite(mean):
        out["note"] = "a non-finite paired difference makes the test undefined"
        return out
    df = float(n - 1)
    if scaled is not None:
        mean_test, se, ci_scale = scaled[0], scaled[1] / math.sqrt(n), scaled[2]
        margin_test = margin / ci_scale
    else:
        mean_test, se, ci_scale, margin_test = mean, std / math.sqrt(n), 1.0, margin
    out.update({"mean_delta": mean, "df": df, "ci_level": 1.0 - 2.0 * alpha})
    if se == 0.0:
        equivalent = abs(mean_test) < margin_test
        ci_low = ci_high = mean_test * ci_scale
        out.update({
            "p_low": 0.0 if equivalent else 1.0,
            "p_high": 0.0 if equivalent else 1.0,
            "p_tost": 0.0 if equivalent else 1.0,
            "equivalent": bool(equivalent),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "note": ("zero-variance paired differences: equivalence is decided "
                     "by |mean_delta| < margin"),
        })
        return out
    # In the scaled branch the excess over +/-margin is compared before the
    # (possibly overflowing) mapping back, so a huge finite difference stays
    # a definite non-equivalence instead of becoming NaN.
    # p_low tests H0 `mean <= -margin` against H1 `mean > -margin`;
    # p_high tests H0 `mean >= +margin` against H1 `mean < +margin`.
    p_low = 1.0 - student_t_cdf((mean_test + margin_test) / se, df)
    p_high = student_t_cdf((mean_test - margin_test) / se, df)
    p_low = min(1.0, max(0.0, p_low))
    p_high = min(1.0, max(0.0, p_high))
    p_tost = max(p_low, p_high)
    half = student_t_ppf(1.0 - alpha, df) * se
    ci_low = (mean_test - half) * ci_scale
    ci_high = (mean_test + half) * ci_scale
    out.update({
        "p_low": p_low,
        "p_high": p_high,
        "p_tost": p_tost,
        "equivalent": bool(p_tost < alpha),
        "ci_low": ci_low,
        "ci_high": ci_high,
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
    full, _ = _mean_and_std(vals)

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
        mean, _ = _mean_and_std(rest)
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


