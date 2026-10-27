from __future__ import annotations

import functools
import math
import numbers
import random
from typing import Optional, Sequence

from .metrics_stats import (MIN_SEEDS_FOR_SIGNIFICANCE,
                         _RANDOMIZATION_EXACT_MAX, _as_real,
                         _is_boolean_value, _p_one_sided_from_z, _strict_int,
                         _validate_alpha, chi2_sf, leave_one_out_jackknife,
                         mean_prediction_interval, mean_std_stderr,
                         paired_t_one_sided, student_t_cdf, student_t_ppf,
                         two_sided_p_from_t)

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


def _pool_meta_studies(studies: Sequence[dict], alpha: float) -> dict:
    """Pool validated effect/stderr records, fixed-effect and random-effects.

    Weights are normalized by their maximum so the sum of individually finite
    inverse-variance weights cannot overflow; an infinite tau^2 is reported as
    the unweighted mean with an infinite standard error rather than dividing
    by zero. The caller keeps validation and the human-readable summary.
    """
    k = len(studies)
    fixed = {
        'pooled_effect': None,
        'stderr': None,
        'ci_low': None, 'ci_high': None,
        't': None, 'df': None,
        'p_one_sided': None, 'p_two_sided': None,
    }
    heterogeneity = {
        'q': None, 'df': None, 'p': None,
        'i_squared': None, 'tau_squared': None, 'tau': None,
    }
    random_effects = {
        'pooled_effect': None,
        'stderr': None,
        'ci_low': None, 'ci_high': None,
        't': None, 'df': None,
        'p_one_sided': None, 'p_two_sided': None,
        'tau_squared': 0.0,
    }
    prediction = {
        'level': 1.0 - alpha, 'low': None, 'high': None,
        'meaningful': False, 'df': None,
        'note': 'a prediction interval for a new scale needs >= 3 studies',
    }
    if k == 0:
        return {
            'fixed_effect': fixed,
            'heterogeneity': heterogeneity,
            'random_effects': random_effects,
            'prediction_interval': prediction,
            'pooled_effect': None,
            'pooled_effect_source': 'random_effects',
        }

    for study in studies:
        study['fixed_weight'] = 1.0 / study['stderr'] ** 2
    max_fixed_weight = max(study['fixed_weight'] for study in studies)
    fixed_scaled = [study['fixed_weight'] / max_fixed_weight
                    for study in studies]
    total_scaled = math.fsum(fixed_scaled)
    fixed_pooled = math.fsum(
        weight * study['effect']
        for weight, study in zip(fixed_scaled, studies)
    ) / total_scaled
    fixed_stderr = ((1.0 / math.sqrt(max_fixed_weight))
                    / math.sqrt(total_scaled))
    for study, weight in zip(studies, fixed_scaled):
        study['fixed_weight_share'] = weight / total_scaled

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
    fixed['pooled_effect'] = fixed_pooled
    fixed['stderr'] = fixed_stderr

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
            random_pooled = math.fsum(study['effect'] for study in studies) / k
            random_stderr = float('inf')

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
    random_effects['pooled_effect'] = random_pooled
    random_effects['stderr'] = random_stderr
    random_effects['tau_squared'] = tau_squared

    if k >= 3:
        df_pi = float(k - 2)
        critical = student_t_ppf(1.0 - alpha / 2.0, df_pi)
        scale = math.sqrt(tau_squared + random_stderr ** 2)
        prediction = {
            'level': 1.0 - alpha,
            'low': random_pooled - critical * scale,
            'high': random_pooled + critical * scale,
            'meaningful': True,
            'df': df_pi,
            'note': ('where the effect at a new scale would plausibly fall; '
                     'wider than the confidence interval for the pooled mean'),
        }

    return {
        'fixed_effect': fixed,
        'heterogeneity': heterogeneity,
        'random_effects': random_effects,
        'prediction_interval': prediction,
        'pooled_effect': random_pooled,
        'pooled_effect_source': 'random_effects',
    }


def _meta_influence(studies: Sequence[dict], alpha: float, pooled: dict,
                    min_studies: int) -> dict:
    """Leave-one-scale-out re-pooling to expose single-scale dependence."""
    k = len(studies)
    block = {
        'method': 'leave-one-scale-out re-pooling',
        'meaningful': bool(k >= 3 and k >= min_studies),
        'n_scales': k,
        'full_pooled_effect': pooled['pooled_effect'],
        'full_p_one_sided': pooled['random_effects']['p_one_sided'],
        'full_i_squared': pooled['heterogeneity']['i_squared'],
        'per_drop': [],
        'pooled_effect_range': [None, None],
        'max_abs_change': None,
        'sign_stable': None,
        'significance_verdict_stable': None,
        'heterogeneity_verdict_stable': None,
        'influential_scales': [],
        'n_influential': 0,
        'note': None,
    }
    if not block['meaningful']:
        block['note'] = (
            f'leave-one-scale-out influence needs >= 3 usable scales (and '
            f'>={min_studies} for pooling); with {k} scale(s) the dropped '
            'pool is too small to re-estimate')
        return block

    full_effect = block['full_pooled_effect']
    full_p = block['full_p_one_sided']
    full_i_squared = block['full_i_squared']
    full_significant = bool(full_p is not None and full_p < alpha)
    full_substantial = bool(
        (full_i_squared is not None and full_i_squared >= 50.0)
        or (pooled['heterogeneity']['p'] is not None
            and pooled['heterogeneity']['p'] < alpha))

    for index, study in enumerate(studies):
        subset = [dict(item) for item in studies if item is not study]
        subset_pooled = _pool_meta_studies(subset, alpha)
        effect = subset_pooled['pooled_effect']
        p_one_sided = subset_pooled['random_effects']['p_one_sided']
        i_squared = subset_pooled['heterogeneity']['i_squared']
        q_p = subset_pooled['heterogeneity']['p']
        subset_substantial = bool(
            (i_squared is not None and i_squared >= 50.0)
            or (q_p is not None and q_p < alpha))
        if full_effect is None or full_effect == 0.0:
            keeps_sign = None
        elif effect is None or effect == 0.0:
            keeps_sign = False
        else:
            keeps_sign = bool((effect < 0.0) == (full_effect < 0.0))
        subset_significant = bool(p_one_sided is not None and p_one_sided < alpha)
        delta = (None if effect is None or full_effect is None
                 else effect - full_effect)
        row = {
            'dropped': study['label'],
            'pooled_effect': effect,
            'stderr': subset_pooled['random_effects']['stderr'],
            'p_one_sided': p_one_sided,
            'i_squared': i_squared,
            'tau_squared': subset_pooled['random_effects']['tau_squared'],
            'delta_from_full': delta,
            'keeps_full_sign': keeps_sign,
            'same_significance_verdict': bool(
                subset_significant == full_significant),
            'same_heterogeneity_verdict': bool(
                subset_substantial == full_substantial),
        }
        row['influential'] = bool(
            keeps_sign is False
            or not row['same_significance_verdict']
            or not row['same_heterogeneity_verdict'])
        block['per_drop'].append(row)

    effects = [row['pooled_effect'] for row in block['per_drop']
               if row['pooled_effect'] is not None]
    if effects:
        block['pooled_effect_range'] = [min(effects), max(effects)]
    deltas = [abs(row['delta_from_full']) for row in block['per_drop']
              if row['delta_from_full'] is not None]
    if deltas:
        block['max_abs_change'] = max(deltas)
    signs = [row['keeps_full_sign'] for row in block['per_drop']]
    if full_effect is None or full_effect == 0.0:
        block['sign_stable'] = None
    else:
        block['sign_stable'] = all(value is True for value in signs)
    block['significance_verdict_stable'] = all(
        row['same_significance_verdict'] for row in block['per_drop'])
    block['heterogeneity_verdict_stable'] = all(
        row['same_heterogeneity_verdict'] for row in block['per_drop'])
    block['influential_scales'] = [
        row['dropped'] for row in block['per_drop'] if row['influential']]
    block['n_influential'] = len(block['influential_scales'])

    if block['influential_scales']:
        block['note'] = (
            'removing scale(s) '
            + ', '.join(repr(label) for label in block['influential_scales'])
            + ' changes the sign, the significance verdict, or the '
            'heterogeneity classification: the pooled result is sensitive to '
            'a single scale. This is a descriptive sensitivity check, not a '
            'certification')
    else:
        block['note'] = (
            'no single scale changes the sign, the significance verdict, or '
            'the heterogeneity classification; descriptive sensitivity only, '
            'not a certification')
    return block


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

    The ``influence`` block (available with >= 3 studies) re-pools after
    dropping each scale in turn and reports whether the pooled sign, the
    significance verdict, or the heterogeneity classification depends on one
    scale. A multi-scale result that reverses when a single scale is removed
    is not a replicated effect, so this sensitivity check is reported next to
    the pooled estimate; it remains descriptive and never certifies a claim.

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
        'influence': {
            'method': 'leave-one-scale-out re-pooling',
            'meaningful': False,
            'n_scales': k,
            'full_pooled_effect': None,
            'full_p_one_sided': None,
            'full_i_squared': None,
            'per_drop': [],
            'pooled_effect_range': [None, None],
            'max_abs_change': None,
            'sign_stable': None,
            'significance_verdict_stable': None,
            'heterogeneity_verdict_stable': None,
            'influential_scales': [],
            'n_influential': 0,
            'note': 'leave-one-scale-out influence needs >= 3 usable scales',
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

    out['direction_consistent'] = bool(
        out['n_zero'] == 0 and (out['n_negative'] == k or out['n_positive'] == k))

    pooled = _pool_meta_studies(studies, alpha)
    out['fixed_effect'] = pooled['fixed_effect']
    out['heterogeneity'] = pooled['heterogeneity']
    out['random_effects'] = pooled['random_effects']
    out['prediction_interval'] = pooled['prediction_interval']
    out['pooled_effect'] = pooled['pooled_effect']
    out['pooled_effect_source'] = pooled['pooled_effect_source']
    out['influence'] = _meta_influence(studies, alpha, pooled, min_studies)

    i_squared = pooled['heterogeneity']['i_squared']
    q_p = pooled['heterogeneity']['p']
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
    assigned an invented precision. The same block also carries the
    leave-one-scale-out ``influence`` check, so a pooled result that depends
    on a single scale is visible without a separate aggregation step.

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

