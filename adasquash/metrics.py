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
  `paired_z_one_sided` (with `welch_z_one_sided` as the unpaired fallback) is
  the test used for "better by more than `margin`";
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
    "paired_z_one_sided",
    "roc_auc",
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
    """One-sided paired z-test of H1: mean(a - b) < -margin.

    `a` and `b` must be the same length and paired (e.g. the per-seed or
    per-batch losses of two arms measured on identical data). A negative
    `mean_delta` means `a` is better. `z = (mean_delta + margin) / stderr`,
    so a negative z supports the alternative; `p_one_sided` is P(Z <= z).
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


def welch_z_one_sided(
    a: Sequence[float], b: Sequence[float], margin: float = 0.0
) -> dict:
    """Unpaired one-sided Welch z-test of H1: mean(a) - mean(b) < -margin.

    Used only when the two arms do not share the same held-out samples (so
    pairing is impossible); the paired test above is strictly more sensitive
    and preferred whenever the batch counts match.
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
    cannot keep the full `pred`/`gold` lists in memory, so they accumulate
    these vectors instead; the semantics are identical to `macro_weighted_f1`,
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
    order = sorted(range(n), key=lambda i: scores[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for r, lab in zip(ranks, labels) if lab)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


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
    n = len(scores)
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    order = sorted(range(n), key=lambda i: -scores[i])
    ap = 0.0
    tp = fp = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        group = order[i:j + 1]
        group_pos = sum(labels[k] for k in group)
        tp += group_pos
        fp += len(group) - group_pos
        if group_pos:
            ap += (tp / (tp + fp)) * (group_pos / n_pos)
        i = j + 1
    return ap


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
        records.append({
            "class": cls,
            "positives": n_pos,
            "negatives": n_neg,
            "negatives_per_positive": n_neg / n_pos,
            "average_precision": average_precision(col, labels),
            "roc_auc": roc_auc(col, labels),
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
