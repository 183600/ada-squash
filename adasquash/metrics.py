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
  (Holm-Bonferroni), instead of reporting the best of many uncorrected draws.

Everything is pure `math` (no scipy/sklearn), so the training scripts can run
in a minimal environment.
"""
import math
from typing import Optional, Sequence

__all__ = [
    "average_precision",
    "binary_auc_report",
    "f1_from_counts",
    "holm_adjust",
    "imbalance_report",
    "macro_weighted_f1",
    "mean_std_stderr",
    "paired_t_one_sided",
    "paired_z_one_sided",
    "roc_auc",
    "student_t_cdf",
    "student_t_ppf",
    "welch_t_one_sided",
    "welch_z_one_sided",
    "wilson_interval",
]

_Z_95 = 1.959963984540054
# Negatives-per-positive ratio above which the FPR denominator makes ROC-AUC
# optimistic and PR-AUC (average precision) is the appropriate summary.
_EXTREME_IMBALANCE = 1000.0


def wilson_interval(n_success: int, n_total: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation for the small trial counts used
    by the needle evaluation, and for proportions near 0 or 1.
    """
    n_total = int(n_total)
    if n_total <= 0:
        raise ValueError("wilson_interval needs n_total >= 1")
    n_success = min(max(int(n_success), 0), n_total)
    p = n_success / n_total
    z2 = z * z
    denom = 1.0 + z2 / n_total
    centre = (p + z2 / (2.0 * n_total)) / denom
    half = (z / denom) * math.sqrt(
        p * (1.0 - p) / n_total + z2 / (4.0 * n_total * n_total)
    )
    return max(0.0, centre - half), min(1.0, centre + half)


def mean_std_stderr(values: Sequence[float]) -> Optional[dict]:
    """Sample mean, sample std (ddof=1) and standard error of the mean."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    if n < 2:
        return {"mean": mean, "std": 0.0, "stderr": 0.0, "n": n}
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    return {"mean": mean, "std": std, "stderr": std / math.sqrt(n), "n": n}


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


def student_t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with `df` degrees of freedom.

    Pure `math` and accurate to near machine precision across the small
    degrees-of-freedom range the seed-replicated comparisons live in (df =
    n - 1 = 1-4). This is the reference distribution for the paired and Welch
    tests below; the normal CDF (z) is only the df -> infinity limit.
    """
    if df <= 0:
        raise ValueError("student_t_cdf needs df > 0")
    if math.isnan(t):
        raise ValueError("student_t_cdf got a NaN statistic")
    if t == float("inf"):
        return 1.0
    if t == float("-inf"):
        return 0.0
    tail = 0.5 * _reg_incomplete_beta(0.5 * df, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0.0 else tail


def student_t_ppf(p: float, df: float) -> float:
    """Inverse Student-t CDF `t : P(T <= t) = p`, by bisection on a monotone CDF.

    Used for confidence-interval critical values.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("student_t_ppf needs 0 < p < 1")
    if df <= 0:
        raise ValueError("student_t_ppf needs df > 0")
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
        if student_t_cdf(mid, df) < p:
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
    indexed = sorted(
        (float(p), i) for i, p in enumerate(pvalues) if p is not None
    )
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
    df = float(na + nb - 2)
    if se > 0.0:
        spread_a, spread_b = va / na, vb / nb
        denom = spread_a ** 2 / (na - 1) + spread_b ** 2 / (nb - 1)
        if denom > 0.0:
            df = (spread_a + spread_b) ** 2 / denom
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


def f1_from_counts(
    gold_counts: Sequence[int],
    pred_counts: Sequence[int],
    correct_counts: Sequence[int],
) -> dict:
    """Multiclass F1 from per-class counts, with the averaging named.

    The three vectors are the class marginals of the confusion matrix:
    `gold_counts[c]` support of class `c`, `pred_counts[c]` how often class
    `c` was predicted, `correct_counts[c]` their overlap (the diagonal).
    Streaming evaluations (a 50k-token vocabulary over millions of tokens)
    accumulate these vectors instead of the full `pred`/`gold` lists; the
    semantics are identical to `macro_weighted_f1`, which delegates here.

    * `macro_f1`: unweighted mean of per-class F1 over classes occurring in
      `gold` or `pred`. Every class counts once, so rare classes are not
      drowned by frequent ones.
    * `weighted_f1`: per-class F1 weighted by the class's support in `gold`.
    * `micro_f1`: total true positives over total support (== accuracy for a
      single-label argmax prediction).

    With class imbalance macro and micro can rank models differently, so call
    sites report the metric by name.
    """
    gold_counts = [int(v) for v in gold_counts]
    pred_counts = [int(v) for v in pred_counts]
    correct_counts = [int(v) for v in correct_counts]
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
    return {
        "macro_f1": macro,
        "weighted_f1": weighted,
        "micro_f1": micro,
        "n_classes_present": len(present),
        "support": support,
    }


def macro_weighted_f1(pred: Sequence[int], gold: Sequence[int], num_classes: int) -> dict:
    """Multiclass F1 with the averaging explicitly named.

    Convenience wrapper around `f1_from_counts` for callers that hold the
    full prediction/target lists (small batches); see that function for the
    macro/weighted/micro semantics.
    """
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    pred = [int(p) for p in pred]
    gold = [int(g) for g in gold]
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
    positives = [int(c) for c in counts if int(c) > 0]
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

    Both ranking metrics only read tie *groups*, so one sort serves them:
    `order` is the ascending permutation and `groups` the ``(start, end)``
    index pairs (inclusive) into ``order`` for each equal-score run. The
    group contents are order-insensitive downstream (average rank for ROC-AUC,
    threshold groups for AP), so neither metric changes with the sort
    direction.
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
    labels = [int(bool(v)) for v in labels]
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} vs {len(labels)}")
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
    labels = [int(bool(v)) for v in labels]
    if len(scores) != len(labels):
        raise ValueError(f"score/label length mismatch: {len(scores)} vs {len(labels)}")
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
    classes = [int(c) for c in classes]
    gold = [int(g) for g in gold]
    scores = [[float(v) for v in row] for row in scores]
    if len(scores) != len(gold):
        raise ValueError(f"score-row/gold length mismatch: {len(scores)} vs {len(gold)}")
    if scores and len(scores[0]) != len(classes):
        raise ValueError(
            f"score matrix has {len(scores[0])} columns but {len(classes)} classes"
        )
    records = []
    skipped = []
    for k, cls in enumerate(classes):
        col = [row[k] for row in scores]
        labels = [1 if g == cls else 0 for g in gold]
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
        # One ordering per class serves both metrics (their tie groups are the
        # same set); going through the public helpers would sort twice and
        # re-convert the already-float column and 0/1 labels.
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
