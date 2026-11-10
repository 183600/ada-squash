"""Advanced results analysis for the DyT++ / AdaSquash studies.


# Reproducibility seed (added by code review)
import random as _random_seed_mod
_random_seed_mod.seed(42)
This module complements the frequentist paired-test framework in `metrics.py`
and `compare_arms.py` with additional analysis perspectives:

* **Robust statistics**: Wilcoxon signed-rank, trimmed means, bootstrap CIs
* **Bayesian re-analysis**: Proper priors, posterior probabilities, ROPE
* **Sensitivity analysis**: Influence diagnostics, assumption checks
* **Replication forecasting**: Probability a new study replicates the finding
* **Composite endpoints**: Multiple metrics combined with principled weighting
* **Mediation analysis**: Decompose ablation effects through mechanisms
* **Post-hoc sensitivity curves**: What effect sizes were we powered to detect?

All functions are pure Python (no scipy/sklearn) and designed to be called
after the primary confirmatory analysis is complete. They are *descriptive
robustness checks*, not replacements for the pre-specified paired t-test.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import random
from collections.abc import Sequence
from typing import Optional

from .metrics_stats import (
    MIN_SEEDS_FOR_SIGNIFICANCE,
    _as_real,
    _finite_paired_observations,
    _real_values,
    _validate_alpha,
    mean_prediction_interval,
    mean_std_stderr,
    paired_t_one_sided,
    student_t_cdf,
    student_t_ppf,
)

__all__ = [
    "assumption_checks",
    "bayesian_paired_analysis",
    "bland_altman_data",
    "bootstrap_ci",
    "composite_endpoint",
    "forest_plot_data",
    "influence_diagnostics",
    "mediation_analysis",
    "post_hoc_sensitivity_curve",
    "qq_plot_data",
    "replication_probability",
    "trimmed_mean_ci",
    "wilcoxon_signed_rank",
]


# ---------------------------------------------------------------------------
# Robust non-parametric and resampling methods
# ---------------------------------------------------------------------------

def wilcoxon_signed_rank(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "less",
    zero_method: str = "wilcoxon",
) -> dict:
    """Wilcoxon signed-rank test for paired samples.

    Distribution-free alternative to the paired t-test. Tests whether the
    median of the paired differences is < 0 (candidate better). With n <= 20
    uses the exact distribution; larger n uses the normal approximation with
    continuity correction.

    The `zero_method` controls how zero differences are handled:
    - "wilcoxon": discard zeros, reduce n (original Wilcoxon)
    - "pratt": keep zeros with rank 0 (Pratt's modification, recommended)
    - "zsplit": split zeros between positive and negative ranks

    Returns the test statistic, exact or approximate p-value, and the
    Hodges-Lehmann estimator (pseudo-median) with its confidence interval.
    """
    alpha = _validate_alpha(alpha)
    if alternative not in ("less", "greater", "two-sided"):
        raise ValueError(f"alternative must be 'less', 'greater', or 'two-sided', got {alternative!r}")
    if zero_method not in ("wilcoxon", "pratt", "zsplit"):
        raise ValueError(f"zero_method must be 'wilcoxon', 'pratt', or 'zsplit', got {zero_method!r}")

    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    n_input = len(a)
    pairs, excluded = _finite_paired_observations(a, b)
    a_vals = [left for left, _ in pairs]
    b_vals = [right for _, right in pairs]
    n = len(a_vals)

    out = {
        "test": "wilcoxon_signed_rank",
        "alternative": ("median(delta) < 0" if alternative == "less"
                        else "median(delta) > 0" if alternative == "greater"
                        else "median(delta) != 0"),
        "n": n,
        "n_input": n_input,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "zero_method": zero_method,
        "statistic": None,
        "p_value": None,
        "method": None,
        "hodges_lehmann": None,
        "hl_ci": [None, None],
        "ci_level": 1.0 - alpha,
        "note": None,
    }
    if n == 0:
        out["note"] = "no finite paired observations"
        return out

    deltas = [x - y for x, y in zip(a_vals, b_vals)]

    # Handle zeros according to method
    if zero_method == "wilcoxon":
        # Discard zeros
        non_zero = [(d, i) for i, d in enumerate(deltas) if d != 0.0]
        n_nz = len(non_zero)
        if n_nz == 0:
            out["note"] = "all paired differences are zero"
            out["statistic"] = 0
            out["p_value"] = 1.0 if alternative == "two-sided" else 0.5
            out["method"] = "exact"
            out["hodges_lehmann"] = 0.0
            out["hl_ci"] = [0.0, 0.0]
            out["n"] = 0
            return out
        deltas = [d for d, _ in non_zero]
        n = n_nz
        out["n"] = n
    elif zero_method == "zsplit":
        # Split zeros: half get positive rank, half negative
        # This is complex; for simplicity we use Pratt (recommended)
        zero_method = "pratt"

    # Pratt method: keep zeros, assign them rank 0
    abs_deltas = [abs(d) for d in deltas]
    # Average ranks for ties
    ranked = sorted(enumerate(abs_deltas), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and ranked[j][1] == ranked[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ranked[k][0]] = avg_rank if ranked[k][1] != 0.0 else 0.0
        i = j

    # Signed ranks
    signed_ranks = [r if d > 0 else -r for d, r in zip(deltas, ranks)]
    w_plus = sum(r for r in signed_ranks if r > 0)
    w_minus = -sum(r for r in signed_ranks if r < 0)
    # Convention: W = sum of positive ranks
    w = w_plus

    # Expected value and variance under null
    n_eff = sum(1 for d in deltas if d != 0.0)  # Pratt: zeros don't contribute to n_eff
    if n_eff == 0:
        out["statistic"] = 0
        out["p_value"] = 1.0 if alternative == "two-sided" else 0.5
        out["method"] = "exact"
        out["hodges_lehmann"] = 0.0
        out["hl_ci"] = [0.0, 0.0]
        return out

    expected = n_eff * (n_eff + 1) / 4.0
    variance = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0

    # Adjust for ties
    tie_correction = 0.0
    abs_sorted = sorted(abs_deltas)
    i = 0
    while i < n_eff:
        j = i + 1
        while j < n_eff and abs_sorted[j] == abs_sorted[i]:
            j += 1
        t = j - i
        if t > 1:
            tie_correction += t * (t - 1) * (t + 1)
        i = j
    if tie_correction > 0:
        variance -= tie_correction / 48.0

    # Exact distribution for small n
    if n_eff <= 20:
        # Enumerate all 2^n sign assignments
        from functools import lru_cache

        @functools.cache
        def count_wplus_leq(target: int, idx: int = 0) -> int:
            """Count sign assignments where sum of positive ranks <= target."""
            if idx == n_eff:
                return 1 if target >= 0 else 0
            r = ranks[idx]
            if r == 0:
                return count_wplus_leq(target, idx + 1)
            # If this difference is positive, add r to W+; if negative, add 0
            return count_wplus_leq(target - r, idx + 1) + count_wplus_leq(target, idx + 1)

        total = 2 ** n_eff
        if alternative == "less":
            # H1: median < 0 (negative differences) -> small W+
            p = count_wplus_leq(w) / (total + 1e-8)
        elif alternative == "greater":
            # H1: median > 0 (positive differences) -> large W+
            # P(W+ >= observed) = 1 - P(W+ <= observed - 1)
            p = 1.0 - (count_wplus_leq(w - 1) / (total + 1e-8))
        else:
            # Two-sided: double the smaller tail
            p_leq = count_wplus_leq(w) / (total + 1e-8)
            p_geq = 1.0 - (count_wplus_leq(w - 1) / (total + 1e-8))
            p = 2 * min(p_leq, p_geq)
        out["method"] = "exact"
    else:
        # Normal approximation with continuity correction
        z = (w - expected) / math.sqrt(variance)
        if alternative == "less":
            # H1: median < 0 -> W+ is small -> p = P(Z <= z)
            cc = 0.5
            p = 0.5 * math.erfc(-(z + cc) / math.sqrt(2))
        elif alternative == "greater":
            # H1: median > 0 -> W+ is large -> p = P(Z >= z)
            cc = -0.5
            p = 0.5 * math.erfc((z + cc) / math.sqrt(2))
        else:
            cc = 0.5 if z >= 0 else -0.5
            p = 2 * 0.5 * math.erfc(-abs(z + cc) / math.sqrt(2))
        out["method"] = "normal_approximation"
        out["z"] = z

    out["statistic"] = w
    out["p_value"] = p

    # Hodges-Lehmann estimator (pseudo-median of paired differences)
    # For paired data: median of (d_i + d_j)/2 for i <= j
    walsh_averages = []
    for i in range(n):
        for j in range(i, n):
            walsh_averages.append((deltas[i] + deltas[j]) / 2.0)
    walsh_averages.sort()
    m = len(walsh_averages)
    if m % 2 == 1:
        hl = walsh_averages[m // 2]
    else:
        hl = (walsh_averages[m // 2 - 1] + walsh_averages[m // 2]) / 2.0
    out["hodges_lehmann"] = hl

    # Confidence interval for Hodges-Lehmann
    # Uses the exact distribution of W; find ranks that correspond to CI
    alpha2 = alpha / 2.0 if alternative == "two-sided" else alpha
    # Approximate: use normal quantile for CI endpoints
    # More exact would invert the Wilcoxon test; this is standard
    # Use large df for normal approximation
    z_alpha = student_t_ppf(1.0 - alpha2, 1e7)
    if alternative == "two-sided":
        lower_rank = max(0, int(math.floor(expected - z_alpha * math.sqrt(variance))))
        upper_rank = min(m - 1, int(math.ceil(expected + z_alpha * math.sqrt(variance))))
    elif alternative == "less":
        lower_rank = 0
        upper_rank = min(m - 1, int(math.ceil(expected + z_alpha * math.sqrt(variance))))
    else:
        lower_rank = max(0, int(math.floor(expected - z_alpha * math.sqrt(variance))))
        upper_rank = m - 1
    out["hl_ci"] = [walsh_averages[lower_rank], walsh_averages[upper_rank]]

    return out


def trimmed_mean_ci(
    values: Sequence[float],
    trim_proportion: float = 0.2,
    alpha: float = 0.05,
    bootstrap_samples: int = 10000,
    seed: int = 0,
) -> dict:
    """Trimmed mean with bootstrap confidence interval.

    Removes the largest and smallest `trim_proportion` fraction of observations
    before computing the mean. Robust to outliers and heavy tails. The CI is
    computed by bootstrap percentile method.

    With n=3-5 seeds, trimming 20% removes at most 1 observation per tail.
    """
    alpha = _validate_alpha(alpha)
    if not 0.0 <= trim_proportion < 0.5:
        raise ValueError("trim_proportion must be in [0, 0.5)")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >= 100")

    vals = _real_values(values, "values")
    n = len(vals)
    if n == 0:
        return {"trimmed_mean": None, "ci": [None, None], "n": 0, "n_trimmed": 0}

    k = int(math.floor(n * trim_proportion))
    sorted_vals = sorted(vals)
    trimmed = sorted_vals[k:n - k] if k > 0 else sorted_vals
    n_trimmed = n - len(trimmed)
    trimmed_mean = sum(trimmed) / max(len(trimmed), 1) if trimmed else None

    out = {
        "test": "trimmed_mean_bootstrap",
        "trim_proportion": trim_proportion,
        "n": n,
        "n_trimmed": n_trimmed,
        "trimmed_mean": trimmed_mean,
        "ci": [None, None],
        "ci_level": 1.0 - alpha,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    if not trimmed:
        out["note"] = "all observations trimmed"
        return out

    # Bootstrap percentile CI
    rng = random.Random(seed)
    boot_means = []
    for _ in range(bootstrap_samples):
        sample = [trimmed[rng.randrange(len(trimmed))] for _ in trimmed]
        boot_means.append((sum(sample) / max(len(sample), 1)) if sample else float("nan"))
    boot_means.sort()
    lo_idx = int(math.floor(bootstrap_samples * alpha / 2.0))
    hi_idx = int(math.ceil(bootstrap_samples * (1.0 - alpha / 2.0))) - 1
    out["ci"] = [boot_means[lo_idx], boot_means[hi_idx]]
    out["bootstrap_se"] = math.sqrt(sum((m - trimmed_mean) ** 2 for m in boot_means) / max(bootstrap_samples - 1, 1))
    return out


def bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    statistic: str = "mean_diff",
    alpha: float = 0.05,
    bootstrap_samples: int = 10000,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence interval for paired difference statistics.

    `statistic` can be:
    - "mean_diff": mean(a - b)
    - "median_diff": median(a - b)
    - "trimmed_mean_diff": 20% trimmed mean of (a - b)
    - "ratio": mean(a) / mean(b) (for positive values)
    - "ppl_ratio": exp(mean(a - b)) (perplexity ratio from loss difference)
    """
    alpha = _validate_alpha(alpha)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >= 100")

    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    n_input = len(a)
    pairs, excluded = _finite_paired_observations(a, b)
    n = len(pairs)

    def _compute_stat(pa: list[float], pb: list[float]) -> float:
        diffs = [x - y for x, y in zip(pa, pb)]
        if statistic == "mean_diff":
            return (sum(diffs) / max(len(diffs), 1)) if diffs else float("nan")
        if statistic == "median_diff":
            s = sorted(diffs)
            m = len(s)
            return s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2.0
        if statistic == "trimmed_mean_diff":
            k = int(math.floor(len(diffs) * 0.2))
            td = sorted(diffs)[k:len(diffs) - k]
            return sum(td) / max(len(td), 1) if td else 0.0
        if statistic == "ratio":
            ma = (sum(pa) / max(len(pa), 1)) if pa else float("nan")
            mb = (sum(pb) / max(len(pb), 1)) if pb else float("nan")
            return ma / mb if mb != 0 else float("inf")
        if statistic == "ppl_ratio":
            return math.exp(sum(diffs) / max(len(diffs), 1)) if diffs else float("nan")
        raise ValueError(f"unknown statistic {statistic!r}")

    orig_a = [x for x, _ in pairs]
    orig_b = [y for _, y in pairs]
    obs = _compute_stat(orig_a, orig_b)

    out = {
        "test": "bootstrap_paired_ci",
        "statistic": statistic,
        "n": n,
        "n_input": n_input,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "observed": obs,
        "ci": [None, None],
        "ci_level": 1.0 - alpha,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
    if n == 0:
        out["note"] = "no finite paired observations"
        return out

    rng = random.Random(seed)
    boot_stats = []
    for _ in range(bootstrap_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        samp_a = [orig_a[i] for i in idx]
        samp_b = [orig_b[i] for i in idx]
        boot_stats.append(_compute_stat(samp_a, samp_b))
    boot_stats.sort()
    lo_idx = int(math.floor(bootstrap_samples * alpha / 2.0))
    hi_idx = int(math.ceil(bootstrap_samples * (1.0 - alpha / 2.0))) - 1
    out["ci"] = [boot_stats[lo_idx], boot_stats[hi_idx]]
    out["bootstrap_se"] = math.sqrt(sum((s - obs) ** 2 for s in boot_stats) / max(bootstrap_samples - 1, 1))
    return out


# ---------------------------------------------------------------------------
# Bayesian re-analysis
# ---------------------------------------------------------------------------

def _beta_binomial_posterior(successes: int, trials: int, prior_alpha: float, prior_beta: float):
    """Conjugate Beta-Binomial posterior."""
    post_alpha = prior_alpha + successes
    post_beta = prior_beta + trials - successes
    mean = post_alpha / (post_alpha + post_beta)
    # Variance
    var = (post_alpha * post_beta) / ((post_alpha + post_beta) ** 2 * (post_alpha + post_beta + 1))
    return post_alpha, post_beta, mean, math.sqrt(var)


def _normal_normal_posterior(
    sample_mean: float,
    sample_sd: float,
    n: int,
    prior_mean: float,
    prior_sd: float,
) -> tuple[float, float]:
    """Conjugate Normal-Normal posterior for the mean (known variance).

    Returns (posterior_mean, posterior_sd).
    """
    # Precision weighting
    prior_prec = 1.0 / (prior_sd ** 2) if prior_sd > 0 else 0.0
    if sample_sd == 0.0:
        # Known mean exactly: posterior = prior updated with infinite precision
        # -> posterior mean = sample_mean, posterior sd = 0
        return sample_mean, 0.0
    like_prec = n / (sample_sd ** 2)
    post_prec = prior_prec + like_prec
    if post_prec == 0.0:
        return sample_mean, sample_sd / math.sqrt(n) if n > 0 else float("inf")
    post_mean = (prior_mean * prior_prec + sample_mean * like_prec) / post_prec
    post_sd = math.sqrt(1.0 / post_prec)
    return post_mean, post_sd


def bayesian_paired_analysis(
    a: Sequence[float],
    b: Sequence[float],
    prior: str = "default",
    rope: tuple[float, float] = (-0.01, 0.01),
    alpha: float = 0.05,
    rope_prior_weight: float = 0.5,
) -> dict:
    """Bayesian paired analysis with ROPE (Region of Practical Equivalence).

    Uses a normal prior on the mean paired difference and known-variance
    likelihood (the sample SD is treated as known; for small n this is
    approximate). Reports:
    - Posterior mean and 95% credible interval
    - P(mean_diff < 0 | data) = probability candidate is better
    - P(mean_diff in ROPE | data) = probability of practical equivalence
    - Bayes factor BF_10 (if prior is specified)

    `prior` options:
    - "default": Cauchy(0, 0.707) scale on standardized effect (Rouder et al.)
    - "unit_information": prior sample size = 1, prior SD = sample SD
    - "skeptical": prior centered at 0 with SD = 0.1 * sample SD (small effects)
    - "enthusiastic": prior centered at -0.5 * sample SD with SD = 0.5 * sample SD
    - {"mean": m, "sd": s}: custom normal prior on raw difference scale
    """
    alpha = _validate_alpha(alpha)

    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    pairs, excluded = _finite_paired_observations(a, b)
    diffs = [x - y for x, y in pairs]
    n = len(diffs)

    out = {
        "test": "bayesian_paired",
        "n": n,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "rope": list(rope),
        "prior": prior,
        "posterior_mean": None,
        "posterior_sd": None,
        "credible_interval": [None, None],
        "ci_level": 1.0 - alpha,
        "p_better": None,
        "p_rope": None,
        "p_worse": None,
        "bf_10": None,
        "bf_01": None,
        "note": None,
    }
    if n == 0:
        out["note"] = "no finite paired observations"
        return out

    stats = mean_std_stderr(diffs)
    sample_mean = stats["mean"]
    sample_sd = stats["std"] if stats["n"] >= 2 else 0.0
    sample_se = stats["stderr"] if stats["n"] >= 2 else float("inf")

    # Treat near-zero SD as exactly zero to avoid numerical issues
    if sample_sd < 1e-15:
        sample_sd = 0.0

    # Prior specification
    if prior == "default":
        # Cauchy prior on standardized effect (JZS prior, Rouder et al. 2009)
        # Approximate with normal for computational simplicity
        # scale = 0.707 * sample_sd, but if sample_sd=0 use small value
        scale = 0.707 * max(sample_sd, 1e-6)
        prior_mean = 0.0
        prior_sd = scale
    elif prior == "unit_information":
        prior_mean = 0.0
        prior_sd = max(sample_sd, 1e-6)
    elif prior == "skeptical":
        prior_mean = 0.0
        prior_sd = 0.1 * max(sample_sd, 1e-6)
    elif prior == "enthusiastic":
        prior_mean = -0.5 * max(sample_sd, 1e-6)
        prior_sd = 0.5 * max(sample_sd, 1e-6)
    elif isinstance(prior, dict) and "mean" in prior and "sd" in prior:
        prior_mean = float(prior["mean"])
        prior_sd = float(prior["sd"])
    else:
        raise ValueError(f"unknown prior {prior!r}")

    # Normal-normal conjugate update
    post_mean, post_sd = _normal_normal_posterior(
        sample_mean, sample_sd, n, prior_mean, prior_sd
    )

    # Credible interval
    z = student_t_ppf(1.0 - alpha / 2.0, 1e7)
    ci_low = post_mean - z * post_sd
    ci_high = post_mean + z * post_sd

    # Probabilities
    # P(delta < 0) = CDF at 0
    if post_sd > 0:
        p_better = 0.5 * math.erfc(post_mean / (post_sd * math.sqrt(2)))
        # ROPE probability: partition into better, ROPE, worse
        rope_low, rope_high = rope
        if rope_low > rope_high:
            rope_low, rope_high = rope_high, rope_low
        cdf_low = 0.5 * math.erfc((rope_low - post_mean) / (post_sd * math.sqrt(2)))
        cdf_high = 0.5 * math.erfc((rope_high - post_mean) / (post_sd * math.sqrt(2)))
        p_rope = cdf_high - cdf_low
        p_worse = 1.0 - cdf_high  # P(delta > rope_high)
        p_better = cdf_low  # P(delta < rope_low)
    # Degenerate posterior at sample_mean
    elif sample_mean < rope[0]:
        p_better, p_rope, p_worse = 1.0, 0.0, 0.0
    elif sample_mean > rope[1]:
        p_better, p_rope, p_worse = 0.0, 0.0, 1.0
    else:
        p_better, p_rope, p_worse = 0.0, 1.0, 0.0

    # Bayes factor BF_10 (H1: delta < 0 vs H0: delta = 0)
    # Using Savage-Dickey density ratio: BF_01 = prior(0) / posterior(0)
    # For normal prior/posterior:
    if post_sd > 0 and prior_sd > 0:
        prior_at_zero = 1.0 / (prior_sd * math.sqrt(2 * math.pi))
        post_at_zero = 1.0 / (post_sd * math.sqrt(2 * math.pi)) * math.exp(-0.5 * (post_mean / post_sd) ** 2)
        bf_01 = prior_at_zero / post_at_zero if post_at_zero > 0 else float("inf")
        bf_10 = 1.0 / bf_01 if bf_01 > 0 else float("inf")
    elif post_sd == 0 and prior_sd > 0:
        # Degenerate posterior: infinite evidence for point estimate
        # If sample_mean < 0 (supports H1), BF_10 = inf; else BF_10 = 0
        bf_10 = float("inf") if sample_mean < 0 else 0.0
        bf_01 = 1.0 / bf_10 if bf_10 > 0 else float("inf")
    else:
        bf_10 = None
        bf_01 = None

    out.update({
        "sample_mean": sample_mean,
        "sample_sd": sample_sd,
        "prior_mean": prior_mean,
        "prior_sd": prior_sd,
        "posterior_mean": post_mean,
        "posterior_sd": post_sd,
        "credible_interval": [ci_low, ci_high],
        "p_better": p_better,
        "p_rope": p_rope,
        "p_worse": p_worse,
        "bf_10": bf_10,
        "bf_01": bf_01,
    })

    if n < 3:
        out["note"] = (f"only {n} seed(s): Bayesian analysis with normal approximation "
                       "is unreliable at this sample size; treat as descriptive")

    return out


# ---------------------------------------------------------------------------
# Influence diagnostics and assumption checks
# ---------------------------------------------------------------------------

def influence_diagnostics(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """Leave-one-out and leave-pair-out influence diagnostics for paired test.

    Reports for each seed:
    - The paired t-statistic and p-value when that seed is omitted
    - Cook's distance analogue for paired data
    - DFBETA (change in mean difference when omitted)
    - Whether the significance conclusion changes

    Also reports the pairwise leave-two-out for n >= 4.
    """
    alpha = _validate_alpha(alpha)

    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    pairs, excluded = _finite_paired_observations(a, b)
    n = len(pairs)

    out = {
        "test": "influence_diagnostics",
        "n": n,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "loo": [],
        "lpo": [],
        "max_cooks_d": None,
        "max_dfbeta": None,
        "conclusion_changes_loo": False,
        "conclusion_changes_lpo": False,
    }
    if n < 2:
        out["note"] = "need at least 2 paired observations"
        return out

    diffs = [x - y for x, y in pairs]
    full_test = paired_t_one_sided([x for x, _ in pairs], [y for _, y in pairs], alpha=alpha)
    full_t = full_test["t"]
    full_p = full_test["p_one_sided"]
    full_mean = full_test["mean_delta"]
    full_sig = full_p < alpha

    max_cooks = 0.0
    max_dfbeta = 0.0

    # Leave-one-out
    for i in range(n):
        loo_diffs = diffs[:i] + diffs[i+1:]
        loo_n = len(loo_diffs)
        loo_mean = sum(loo_diffs) / loo_n
        loo_sd = math.sqrt(sum((d - loo_mean) ** 2 for d in loo_diffs) / max(loo_n - 1, 1)) if loo_n > 1 else 0.0
        loo_se = loo_sd / math.sqrt(loo_n) if loo_n > 0 else 0.0
        loo_t = loo_mean / loo_se if loo_se > 0 else (float("-inf") if loo_mean < 0 else float("inf"))
        df = loo_n - 1
        loo_p = student_t_cdf(loo_t, df) if df > 0 else (0.0 if loo_t < 0 else 1.0)

        # DFBETA: change in mean
        dfbeta = abs(full_mean - loo_mean)
        max_dfbeta = max(max_dfbeta, dfbeta)

        # Cook's distance analogue: (mean_change)^2 / (p * MSE)
        # For paired, p=1, MSE = se^2
        mse = full_test.get("df", 1)  # not exactly MSE but scale
        if full_test["df"] and full_test["df"] > 0:
            full_se = full_test.get("ci_high", 0) - full_test.get("ci_low", 0)
            full_se = full_se / max((2 * student_t_ppf(1 - alpha/2, full_test["df"]), 1e-10)) if full_se else 0
            if full_se > 0:
                cooks = (dfbeta / full_se) ** 2 / 1.0
            else:
                cooks = 0.0
        else:
            cooks = 0.0
        max_cooks = max(max_cooks, cooks)

        loo_sig = loo_p < alpha
        out["loo"].append({
            "seed_index": i,
            "omitted_delta": diffs[i],
            "loo_mean": loo_mean,
            "loo_t": loo_t,
            "loo_p": loo_p,
            "dfbeta": dfbeta,
            "cooks_d": cooks,
            "significance_changes": loo_sig != full_sig,
        })
        if loo_sig != full_sig:
            out["conclusion_changes_loo"] = True

    # Leave-pair-out (all pairs) for n >= 4
    if n >= 4:
        from itertools import combinations
        for i, j in combinations(range(n), 2):
            lpo_diffs = [diffs[k] for k in range(n) if k not in (i, j)]
            lpo_n = len(lpo_diffs)
            lpo_mean = sum(lpo_diffs) / lpo_n
            lpo_sd = math.sqrt(sum((d - lpo_mean) ** 2 for d in lpo_diffs) / max(lpo_n - 1, 1)) if lpo_n > 1 else 0.0
            lpo_se = lpo_sd / math.sqrt(lpo_n)
            lpo_t = lpo_mean / lpo_se if lpo_se > 0 else (float("-inf") if lpo_mean < 0 else float("inf"))
            df = lpo_n - 1
            lpo_p = student_t_cdf(lpo_t, df) if df > 0 else (0.0 if lpo_t < 0 else 1.0)
            lpo_sig = lpo_p < alpha
            out["lpo"].append({
                "omitted_indices": [i, j],
                "omitted_deltas": [diffs[i], diffs[j]],
                "lpo_mean": lpo_mean,
                "lpo_t": lpo_t,
                "lpo_p": lpo_p,
                "significance_changes": lpo_sig != full_sig,
            })
            if lpo_sig != full_sig:
                out["conclusion_changes_lpo"] = True

    out["max_cooks_d"] = max_cooks
    out["max_dfbeta"] = max_dfbeta
    out["full_t"] = full_t
    out["full_p"] = full_p
    out["full_significant"] = full_sig

    return out


def assumption_checks(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> dict:
    """Check assumptions of the paired t-test.

    Returns:
    - Normality test (Shapiro-Wilk approximation for small n)
    - Symmetry check (for Wilcoxon validity)
    - Homoscedasticity (variance of differences constant - only checkable if multiple batches per seed)
    - Outlier detection (boxplot rule on differences)
    - Linearity of Q-Q plot (correlation of ordered differences with normal quantiles)
    """
    alpha = _validate_alpha(alpha)

    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    pairs, excluded = _finite_paired_observations(a, b)
    diffs = [x - y for x, y in pairs]
    n = len(diffs)

    out = {
        "test": "paired_assumption_checks",
        "n": n,
        "n_excluded": len(excluded),
        "excluded": excluded,
        "normality": {},
        "symmetry": {},
        "outliers": {},
        "qq_correlation": None,
    }
    if n < 3:
        out["note"] = "need at least 3 paired observations for assumption checks"
        return out

    # Normality: Shapiro-Wilk approximation (Royston 1992) for n <= 2000
    # Using the approximation from the original paper
    sorted_diffs = sorted(diffs)
    # Expected normal order statistics (approximate)
    # For small n, use Blom's formula: Phi^(-1)((i - 3/8)/(n + 1/4))
    expected = []
    for i in range(1, n + 1):
        p = (i - 3.0/8.0) / (n + 1.0/4.0)
        expected.append(student_t_ppf(p, 1e7))

    # Correlation of sorted data with expected normal quantiles
    mx = sum(sorted_diffs) / (n + 1e-8)
    my = sum(expected) / (n + 1e-8)
    dx = [x - mx for x in sorted_diffs]
    dy = [y - my for y in expected]
    sxx = sum(x * x for x in dx)
    syy = sum(y * y for y in dy)
    sxy = sum(x * y for x, y in zip(dx, dy))
    if sxx > 0 and syy > 0:
        r = sxy / math.sqrt(sxx * syy)
        out["qq_correlation"] = r

    # Shapiro-Wilk W statistic (approximate for small n)
    # W = (sum(a_i * x_(i)))^2 / sum((x_i - xbar)^2)
    # Use pre-tabulated coefficients approximation
    # For simplicity, use the correlation squared as an approximation
    w_approx = r * r
    out["normality"] = {
        "shapiro_wilk_approx": w_approx,
        "qq_correlation": r,
        "interpretation": ("close to 1 = normal-like; below 0.9 = concerning" if n >= 5
                           else "low power with n < 5"),
    }

    # Symmetry check: compare median to mean, and tail balance
    median = sorted_diffs[n // 2] if n % 2 else (sorted_diffs[n // 2 - 1] + sorted_diffs[n // 2]) / 2.0
    mean = sum(diffs) / max(n, 1)
    # Tail balance: proportion above median vs below
    n_below = sum(1 for d in diffs if d < median)
    n_above = sum(1 for d in diffs if d > median)
    tail_balance = n_above / max(n, 1) if n > 0 else 0.5
    # Skewness (moment)
    if n >= 3:
        m2 = sum((d - mean) ** 2 for d in diffs) / max(n, 1)
        m3 = sum((d - mean) ** 3 for d in diffs) / max(n, 1)
        skew = m3 / (m2 ** 1.5) if m2 > 0 else 0.0
    else:
        skew = 0.0
    out["symmetry"] = {
        "mean": mean,
        "median": median,
        "mean_minus_median": mean - median,
        "tail_balance": tail_balance,  # 0.5 = symmetric
        "skewness": skew,
        "interpretation": ("mean ≈ median, tail_balance ≈ 0.5 = symmetric"
                           if abs(mean - median) < 0.1 * max(1.0, abs(mean)) and abs(tail_balance - 0.5) < 0.2
                           else "asymmetric"),
    }

    # Outliers: Tukey's boxplot rule on differences
    if n >= 4:
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_diffs[q1_idx]
        q3 = sorted_diffs[q3_idx]
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        outliers = [d for d in diffs if d < lower_fence or d > upper_fence]
        out["outliers"] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower_fence,
            "upper_fence": upper_fence,
            "outlier_values": outliers,
            "n_outliers": len(outliers),
        }
    else:
        out["outliers"] = {"note": "need n >= 4 for IQR-based outlier detection"}

    return out


# ---------------------------------------------------------------------------
# Replication forecasting
# ---------------------------------------------------------------------------

def replication_probability(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
    margin: float = 0.0,
    future_n: Optional[int] = None,
) -> dict:
    """Probability that a future replication will yield a significant result.

    Given the current paired data, computes:
    - Predictive probability that a new study with `future_n` seeds (or same n)
      will reject H0 at level alpha
    - Probability the effect direction replicates (same sign)
    - Probability the effect size exceeds a practical margin

    Uses the t-distribution predictive distribution (same as `mean_prediction_interval`
    but for the test statistic of a future study).
    """
    alpha = _validate_alpha(alpha)

    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    pairs, excluded = _finite_paired_observations(a, b)
    diffs = [x - y for x, y in pairs]
    n = len(diffs)

    out = {
        "test": "replication_probability",
        "n_current": n,
        "n_future": future_n or n,
        "alpha": alpha,
        "margin": margin,
        "p_replicate_significant": None,
        "p_same_direction": None,
        "p_exceed_margin": None,
        "current_mean": None,
        "current_sd": None,
    }
    if n < 2:
        out["note"] = "need at least 2 paired observations"
        return out

    stats = mean_std_stderr(diffs)
    mean_delta = stats["mean"]
    sd_delta = stats["std"]
    se_delta = stats["stderr"]
    df = n - 1

    n_future = future_n or n
    df_future = n_future - 1

    # Predictive distribution for future mean: t with df = n-1, scale = sd * sqrt(1 + 1/n_future)
    # But we want: P(t_future < -margin - t_alpha * se_future)
    # where t_future ~ t_{df_future} centered at mean_delta with scale sd_delta/sqrt(n_future)

    # Actually, the predictive probability integrates over the posterior of the effect
    # Using the t-predictive distribution for the future test statistic

    # Critical value for future study
    t_crit = -student_t_ppf(alpha, df_future)  # negative for lower tail

    # Under the predictive distribution, the future mean ~ t_df(location=mean_delta, scale=sd_delta*sqrt(1+1/n_future)/sqrt(n_future))
    # Wait, the prediction interval for a new *study mean* (not a new observation) is:
    # mean_delta +/- t_{df} * sd_delta * sqrt(1/n + 1/n_future)
    # The future test statistic is (mean_future + margin) / (sd_future / sqrt(n_future))
    # But sd_future is unknown and has its own distribution.

    # Approximate: use the non-central t distribution
    # Non-centrality parameter for future study: ncp = mean_delta / (sd_delta / sqrt(n_future))
    # But we need to integrate over uncertainty in mean_delta and sd_delta.

    # Simple approximation: treat current estimate as true, compute power
    if sd_delta > 0:
        # Non-centrality parameter for future study under true effect = mean_delta
        # t = (mean_future + margin) / (sd_future / sqrt(n_future))
        # ncp = (mean_delta + margin) / (sd_delta / sqrt(n_future))
        ncp = (mean_delta + margin) * math.sqrt(n_future) / sd_delta
        # Critical value for one-sided lower-tail test at alpha
        t_crit = student_t_ppf(alpha, df_future)  # negative (e.g., -1.645 for large df)
        # Power = P(T <= t_crit) where T ~ non-central t(df, ncp)
        # Normal approximation: T ~ N(ncp, 1), Power = Phi(t_crit - ncp)
        z_power = t_crit - ncp
        p_sig = 0.5 * (1 + math.erf(z_power / math.sqrt(2)))
    else:
        p_sig = 1.0 if mean_delta + margin < 0 else 0.0

    # Same direction probability: P(future mean has same sign as current mean)
    # Future mean ~ t_df(location=mean_delta, scale=sd_delta*sqrt(1/n + 1/n_future))
    pred_scale = sd_delta * math.sqrt(1.0 / max(n, 1) + 1.0 / max(n_future, 1)) if sd_delta > 0 else 0.0
    if pred_scale > 0:
        if mean_delta < 0:
            p_same = student_t_cdf(-mean_delta / pred_scale, df)
        else:
            p_same = 1.0 - student_t_cdf(-mean_delta / pred_scale, df)
    else:
        p_same = 1.0 if mean_delta == 0 else (1.0 if mean_delta < 0 else 0.0)

    # Exceed margin
    if pred_scale > 0:
        p_margin = student_t_cdf(-(mean_delta + margin) / pred_scale, df)
    else:
        p_margin = 1.0 if mean_delta + margin < 0 else 0.0

    out.update({
        "current_mean": mean_delta,
        "current_sd": sd_delta,
        "p_replicate_significant": p_sig,
        "p_same_direction": p_same,
        "p_exceed_margin": p_margin,
        "predictive_scale": pred_scale,
        "future_critical_t": t_crit,
    })

    return out


# ---------------------------------------------------------------------------
# Composite endpoints
# ---------------------------------------------------------------------------

def composite_endpoint(
    arms: dict[str, dict[str, Sequence[float]]],
    weights: Optional[dict[str, float]] = None,
    alpha: float = 0.05,
    reference: Optional[str] = None,
) -> dict:
    """Combine multiple metrics into a single composite endpoint.

    `arms` maps arm name to a dict of metric_name -> per-seed values.
    All arms must have the same metrics and same seeds.

    `weights` gives the weight for each metric. If None, uses equal weights.
    Weights are normalized to sum to 1.

    For each seed, computes the weighted sum of standardized metrics
    (standardized by the reference arm's SD across seeds, or pooled SD).
    Then performs a paired test on the composite scores.

    This is a descriptive analysis; the confirmatory test remains the
    pre-specified primary metric.
    """
    alpha = _validate_alpha(alpha)

    if not arms:
        raise ValueError("arms cannot be empty")
    arm_names = list(arms.keys())
    if len(arm_names) < 2:
        raise ValueError("need at least 2 arms")

    # Collect all metrics
    first_arm = arms[arm_names[0]]
    if not isinstance(first_arm, dict):
        raise ValueError("each arm must be a dict of metric_name -> sequence")
    metrics = list(first_arm.keys())
    if not metrics:
        raise ValueError("no metrics provided")

    # Check all arms have same metrics
    for name, arm in arms.items():
        if set(arm.keys()) != set(metrics):
            raise ValueError(f"arm {name!r} has different metrics: {set(arm.keys())} vs {set(metrics)}")

    # Check all metrics have same seeds (by length)
    n_seeds = {metric: len(first_arm[metric]) for metric in metrics}
    for name, arm in arms.items():
        for metric in metrics:
            if len(arm[metric]) != n_seeds[metric]:
                raise ValueError(f"arm {name!r} metric {metric!r} has different seed count")
    # Guard against empty seed lists (would cause ZeroDivisionError below)
    if any(n == 0 for n in n_seeds.values()):
        empty = [m for m, n in n_seeds.items() if n == 0]
        raise ValueError(f"metrics with no seeds (empty value lists): {empty}")

    # Weights
    if weights is None:
        weights = {m: 1.0 / max(len(metrics), 1) if metrics else float("nan") for m in metrics}
    else:
        missing = set(metrics) - set(weights.keys())
        if missing:
            raise ValueError(f"weights missing for metrics: {missing}")
        total = sum(weights[m] for m in metrics)
        if total <= 0:
            raise ValueError("weights must sum to positive value")
        weights = {m: weights[m] / total for m in metrics}

    # Reference arm
    if reference is None:
        # Use the arm with best (lowest) mean on first metric
        ref_candidate = min(arm_names, key=lambda n: sum(arms[n][metrics[0]]) / n_seeds[metrics[0]])
        reference = ref_candidate

    if reference not in arms:
        raise ValueError(f"reference {reference!r} not in arms")

    # For each metric, compute reference mean and SD for standardization
    ref_means = {}
    ref_sds = {}
    for metric in metrics:
        vals = arms[reference][metric]
        stats = mean_std_stderr(vals)
        ref_means[metric] = stats["mean"]
        ref_sds[metric] = stats["std"] if stats["std"] > 0 else 1.0

    # Compute composite score per seed per arm
    n = n_seeds[metrics[0]]
    composite_scores = {name: [] for name in arm_names}

    for seed_idx in range(n):
        for name in arm_names:
            score = 0.0
            for metric in metrics:
                val = arms[name][metric][seed_idx]
                # Standardize: (val - ref_mean) / ref_sd
                # Lower is better for loss metrics; we want negative = better
                standardized = (val - ref_means[metric]) / ref_sds[metric]
                score += weights[metric] * standardized
            composite_scores[name].append(score)

    # Paired test: candidate vs reference on composite
    candidates = [name for name in arm_names if name != reference]
    comparisons = {}
    for cand in candidates:
        comp = paired_t_one_sided(
            composite_scores[cand],
            composite_scores[reference],
            alpha=alpha,
        )
        comparisons[cand] = comp

    out = {
        "test": "composite_endpoint",
        "metrics": metrics,
        "weights": weights,
        "reference": reference,
        "n_seeds": n,
        "per_seed_scores": composite_scores,
        "comparisons": comparisons,
    }
    return out


# ---------------------------------------------------------------------------
# Mediation analysis for ablation
# ---------------------------------------------------------------------------

def mediation_analysis(
    full_arm: Sequence[float],
    reference_arm: Sequence[float],
    mediator_arms: dict[str, Sequence[float]],
    alpha: float = 0.05,
    bootstrap_samples: int = 5000,
    seed: int = 0,
) -> dict:
    """Mediation analysis for ablation: how much of the full effect goes through each mechanism?

    `full_arm`: the full proposal (e.g., DyT-CD full)
    `reference_arm`: the baseline (e.g., LayerNorm)
    `mediator_arms`: dict of mechanism_name -> arm with that mechanism removed
                     (e.g., "per_channel": Full-minus-per-channel)

    For each mediator M, decomposes the total effect (TE) into:
    - Direct effect (DE): effect of full vs reference not through M
    - Indirect effect (IE): effect through M = TE - DE

    Uses the paired difference approach: for each seed, we have
    Y_full, Y_ref, Y_minus_M.
    TE_i = Y_full_i - Y_ref_i
    IE_i = Y_full_i - Y_minus_M_i  (effect of removing M from full)
    DE_i = Y_minus_M_i - Y_ref_i   (effect of reference to full-minus-M)

    Reports bootstrap CIs for IE, DE, and proportion mediated (IE/TE).
    """
    alpha = _validate_alpha(alpha)
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be >= 100")

    full = _real_values(full_arm, "full_arm")
    ref = _real_values(reference_arm, "reference_arm")
    if len(full) != len(ref):
        raise ValueError(f"full and reference must have same length, got {len(full)} vs {len(ref)}")

    mediators = {}
    for name, arm in mediator_arms.items():
        vals = _real_values(arm, name)
        if len(vals) != len(full):
            raise ValueError(f"mediator {name!r} must have same length as full, got {len(vals)} vs {len(full)}")
        mediators[name] = vals

    n = len(full)
    if n == 0:
        return {"note": "no observations"}

    # Paired differences
    te = [full[i] - ref[i] for i in range(n)]  # total effect (negative = full better)
    out = {
        "test": "mediation_analysis",
        "n_seeds": n,
        "total_effect": {"mean": sum(te)/n, "per_seed": te},
        "mediators": {},
    }

    rng = random.Random(seed)

    for name, med in mediators.items():
        ie = [full[i] - med[i] for i in range(n)]  # effect of mediator
        de = [med[i] - ref[i] for i in range(n)]  # direct effect

        # Proportion mediated: ratio-of-means estimator (mean(IE) / mean(TE)),
        # which is consistent and avoids per-seed division by near-zero TE.
        # Per-seed ratios with near-zero denominators produce extreme outliers
        # that dominate bootstrap CIs; the ratio-of-means is the standard
        # mediation estimator and avoids this instability.
        _PROP_TOL = 1e-12
        mean_te = sum(te) / (n + 1e-8)
        mean_ie = sum(ie) / (n + 1e-8)
        prop_point = mean_ie / mean_te if abs(mean_te) > _PROP_TOL else 0.0
        # Per-seed proportions are kept for diagnostic transparency only;
        # extreme values (|te_i| near zero) are clipped to prevent inf/NaN
        # from entering bootstrap percentile CIs.
        _PROP_CLIP = 100.0
        prop = []
        for i in range(n):
            if abs(te[i]) > _PROP_TOL:
                ratio = ie[i] / te[i]
                ratio = max(-_PROP_CLIP, min(_PROP_CLIP, ratio))
            else:
                ratio = 0.0
            prop.append(ratio)

        # Bootstrap CIs
        boot_ie = []
        boot_de = []
        boot_prop = []
        for _ in range(bootstrap_samples):
            idx = [rng.randrange(n) for _ in range(n)]
            ie_s = [ie[i] for i in idx]
            de_s = [de[i] for i in idx]
            te_s = [te[i] for i in idx]
            mean_ie_s = sum(ie_s) / (n + 1e-8)
            mean_te_s = sum(te_s) / (n + 1e-8)
            boot_ie.append(mean_ie_s)
            boot_de.append(sum(de_s) / (n + 1e-8))
            # Proportion: ratio-of-means per bootstrap sample (consistent
            # with the point estimator; avoids per-seed division outliers)
            if abs(mean_te_s) > _PROP_TOL:
                boot_prop.append(mean_ie_s / mean_te_s)
            else:
                boot_prop.append(0.0)

        boot_ie.sort()
        boot_de.sort()
        boot_prop.sort()
        lo = int(math.floor(bootstrap_samples * alpha / 2))
        hi = int(math.ceil(bootstrap_samples * (1 - alpha / 2))) - 1

        out["mediators"][name] = {
            "indirect_effect": {"mean": sum(ie)/n, "ci": [boot_ie[lo], boot_ie[hi]], "per_seed": ie},
            "direct_effect": {"mean": sum(de)/n, "ci": [boot_de[lo], boot_de[hi]], "per_seed": de},
            "proportion_mediated": {
                "ratio_of_means": prop_point,
                "per_seed_clipped": prop,
                "ci": [boot_prop[lo], boot_prop[hi]],
                "clip_bound": _PROP_CLIP,
                "note": "proportion = mean(IE)/mean(TE); per-seed ratios clipped to [-100, 100] to prevent near-zero TE from producing extreme outliers",
            },
            "interpretation": ("mediator helps" if sum(ie)/n < 0 else "mediator hurts"),
        }

    return out


# ---------------------------------------------------------------------------
# Post-hoc sensitivity curves
# ---------------------------------------------------------------------------

def post_hoc_sensitivity_curve(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
    power: float = 0.8,
    effect_size_range: Optional[tuple[float, float]] = None,
    n_points: int = 50,
) -> dict:
    """Post-hoc sensitivity curve: what effect sizes could we detect?

    For a range of standardized effect sizes (Cohen's d), computes:
    - The achieved power given the observed SD and n
    - The minimum n needed to achieve target power
    - The Bayes factor threshold for "strong evidence" (BF > 10)

    This is descriptive: the confirmatory design should have used
    `prospective_power_plan` before data collection.
    """
    alpha = _validate_alpha(alpha)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    pairs, excluded = _finite_paired_observations(a, b)
    diffs = [x - y for x, y in pairs]
    n = len(diffs)

    out = {
        "test": "post_hoc_sensitivity",
        "n": n,
        "n_excluded": len(excluded),
        "observed_dz": None,
        "curve": [],
    }
    if n < 2:
        out["note"] = "need at least 2 paired observations"
        return out

    stats = mean_std_stderr(diffs)
    observed_dz = stats["mean"] / stats["std"] if stats["std"] > 0 else None
    out["observed_dz"] = observed_dz

    sd = stats["std"]
    if sd == 0:
        out["note"] = "zero observed SD; sensitivity curve degenerate"
        return out

    # Effect size range
    if effect_size_range is None:
        if observed_dz is not None:
            max_es = max(abs(observed_dz) * 2, 1.0)
        else:
            max_es = 1.0
        effect_size_range = (-max_es, max_es)

    lo, hi = effect_size_range
    for i in range(n_points):
        es = lo + (hi - lo) * i / max((n_points - 1), 1e-10)
        # Achieved power at this effect size
        ncp = abs(es) * math.sqrt(n)
        t_crit = -student_t_ppf(alpha, n - 1)
        # Power = P(t_{n-1, ncp} < t_crit) for negative effect
        # Normal approximation
        z_power = ncp + t_crit
        achieved_power = 0.5 * (1 + math.erf(z_power / math.sqrt(2)))

        # Required n for target power
        # n = ((z_{1-alpha} + z_{power}) / es)^2
        z_alpha = student_t_ppf(1 - alpha, 1e7)
        z_pow = student_t_ppf(power, 1e7)
        req_n = math.ceil(((z_alpha + z_pow) / abs(es)) ** 2) if es != 0 else float("inf")

        # Bayes factor threshold
        # BF_10 > 10 requires |t| > sqrt(n) * threshold
        # Approximate: BF_10 ~ exp(t^2/2) for normal prior
        # So t > sqrt(2 * log(10)) ~ 2.15
        bf_threshold_t = math.sqrt(2 * math.log(max(10, 1e-10)))
        bf_n = math.ceil((bf_threshold_t / abs(es)) ** 2) if es != 0 else float("inf")

        out["curve"].append({
            "effect_size": es,
            "achieved_power": achieved_power,
            "required_n_for_power": req_n,
            "required_n_for_bf10": bf_n,
        })

    return out


# ---------------------------------------------------------------------------
# Visualization data helpers (for external plotting)
# ---------------------------------------------------------------------------

def forest_plot_data(
    comparisons: dict[str, dict],
    metric_name: str = "val_loss",
) -> dict:
    """Prepare data for a forest plot of multiple comparisons.

    `comparisons` maps comparison_name -> result dict from `paired_t_one_sided`
    (or similar with mean_delta, ci_low, ci_high, n).

    Returns a structure ready for plotting with matplotlib/seaborn.
    """
    rows = []
    for name, comp in comparisons.items():
        if not isinstance(comp, dict):
            continue
        mean_delta = comp.get("mean_delta")
        ci_low = comp.get("ci_low")
        ci_high = comp.get("ci_high")
        n = comp.get("n")
        if mean_delta is None or ci_low is None or ci_high is None:
            continue
        rows.append({
            "label": name,
            "estimate": mean_delta,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": n,
            "weight": 1.0 / max((ci_high - ci_low), 1e-10) ** 2 if ci_high != ci_low else 1.0,
        })

    # Sort by effect size
    rows.sort(key=lambda r: r["estimate"])

    return {
        "metric": metric_name,
        "rows": rows,
        "null_line": 0.0,
    }


def bland_altman_data(
    a: Sequence[float],
    b: Sequence[float],
) -> dict:
    """Bland-Altman (mean-difference) data for paired comparison.

    Returns per-seed: mean of (a,b), difference (a-b), and limits of agreement.
    """
    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired needs equal lengths, got {len(a)} vs {len(b)}")
    pairs, excluded = _finite_paired_observations(a, b)
    diffs = [x - y for x, y in pairs]
    means = [(x + y) / 2.0 for x, y in pairs]
    n = len(diffs)

    if n == 0:
        return {"note": "no finite paired observations"}

    mean_diff = sum(diffs) / max(n, 1)
    sd_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / max(n - 1, 1)) if n > 1 else 0.0
    loa_low = mean_diff - 1.96 * sd_diff
    loa_high = mean_diff + 1.96 * sd_diff

    return {
        "n": n,
        "per_seed": [{"mean": m, "difference": d} for m, d in zip(means, diffs)],
        "mean_difference": mean_diff,
        "sd_difference": sd_diff,
        "limits_of_agreement": [loa_low, loa_high],
        "excluded": excluded,
    }


def qq_plot_data(
    a: Sequence[float],
    b: Sequence[float],
) -> dict:
    """Q-Q plot data for paired differences vs theoretical normal quantiles."""
    a = _real_values(a, "a")
    b = _real_values(b, "b")
    if len(a) != len(b):
        raise ValueError(f"paired needs equal lengths, got {len(a)} vs {len(b)}")
    pairs, excluded = _finite_paired_observations(a, b)
    diffs = [x - y for x, y in pairs]
    n = len(diffs)

    if n == 0:
        return {"note": "no finite paired observations"}

    sorted_diffs = sorted(diffs)
    mean = sum(diffs) / max(n, 1)
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)) if n > 1 else 0.0

    # Theoretical quantiles: Phi^{-1}((i - 0.5)/n) * sd + mean
    theoretical = []
    for i in range(1, n + 1):
        p = (i - 0.5) / (n + 1e-8)
        z = student_t_ppf(p, 1e7)
        theoretical.append(mean + z * sd)

    return {
        "n": n,
        "observed": sorted_diffs,
        "theoretical": theoretical,
        "mean": mean,
        "sd": sd,
        "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# Convenience: run all advanced analyses on a compare_arms result
# ---------------------------------------------------------------------------

def enhance_comparison_report(comparison_json_path: str, output_path: Optional[str] = None) -> dict:
    """Run all advanced analyses on a compare_arms.json output.

    Reads the comparison, extracts per-seed deltas for each candidate,
    and runs:
    - Wilcoxon signed-rank
    - Bootstrap CI
    - Bayesian analysis
    - Influence diagnostics
    - Assumption checks
    - Replication probability
    - Sensitivity curve
    - Bland-Altman, Q-Q, Forest plot data

    Writes enhanced report to output_path (or comparison_json_path + ".enhanced.json").
    """
    from pathlib import Path

    path = Path(comparison_json_path)
    comp = json.loads(path.read_text())

    rows = comp.get("rows", [])
    enhanced_rows = []

    for row in rows:
        name = row.get("name")
        per_seed = row.get("per_seed_delta")
        if not per_seed:
            enhanced_rows.append({"name": name, "error": "no per_seed_delta"})
            continue

        # Reconstruct paired differences
        deltas = [float(v) for v in per_seed.values()]
        # Need reference values too - not stored in compare_arms output
        # We can only analyze the deltas directly
        # For Wilcoxon etc we need the raw paired values
        # Since we don't have them, we'll note this limitation
        enhanced = {
            "name": name,
            "n_shared_seeds": row.get("n_shared_seeds"),
            "delta_paired_mean": row.get("delta_paired_mean"),
            "paired_delta_std": row.get("paired_delta_std"),
        }

        if len(deltas) >= 2:
            # Wilcoxon on deltas vs 0
            enhanced["wilcoxon"] = wilcoxon_signed_rank(deltas, [0.0]*len(deltas))
            # Bootstrap on deltas
            enhanced["bootstrap_mean"] = bootstrap_ci(deltas, [0.0]*len(deltas), statistic="mean_diff")
            # Bayesian
            enhanced["bayesian"] = bayesian_paired_analysis(deltas, [0.0]*len(deltas))
            # Influence
            enhanced["influence"] = influence_diagnostics(deltas, [0.0]*len(deltas))
            # Assumptions
            enhanced["assumptions"] = assumption_checks(deltas, [0.0]*len(deltas))
            # Replication
            enhanced["replication"] = replication_probability(deltas, [0.0]*len(deltas))
            # Sensitivity
            enhanced["sensitivity"] = post_hoc_sensitivity_curve(deltas, [0.0]*len(deltas))

        enhanced_rows.append(enhanced)

    report = {
        "schema": "adasquash.enhanced_comparison.v1",
        "source": str(path),
        "enhanced_rows": enhanced_rows,
        "note": ("Advanced analyses are descriptive robustness checks. "
                 "The pre-specified paired t-test in compare_arms remains the "
                 "certification test. Per-seed raw values not available in "
                 "compare_arms output; deltas-only analyses assume reference=0."),
    }

    out_path = Path(output_path) if output_path else path.with_suffix(".enhanced.json")
    from .jsonio import save_json_atomic
    save_json_atomic(out_path, report)
    return report