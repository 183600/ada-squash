from __future__ import annotations

import functools
import math
import numbers
import random
from typing import Optional, Sequence

from .metrics_stats import (_EXTREME_IMBALANCE,
                               _binary_labels, _strict_int)

def _weighted_mean(values: Sequence[float], weights: Sequence[int],
                  total: int) -> float:
    if not total:
        return 0.0
    try:
        return sum(v * w for v, w in zip(values, weights)) / total
    except OverflowError:
        return math.fsum(v * (w / total) for v, w in zip(values, weights))


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
        if not denom:
            f1_by_class[i] = 0.0
            continue
        try:
            f1_by_class[i] = 2.0 * tp[i] / denom
        except OverflowError:
            f1_by_class[i] = (2 * tp[i]) / denom
    macro = sum(f1_by_class.values()) / len(f1_by_class) if f1_by_class else 0.0
    weighted = _weighted_mean(
        [f1_by_class[i] for i in present],
        [tp[i] + fn[i] for i in present],
        support,
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
    weighted_precision = _weighted_mean(
        [precision_by_class[i] for i in present],
        [tp[i] + fn[i] for i in present],
        support,
    )
    weighted_recall = _weighted_mean(
        [recall_by_class[i] for i in present],
        [tp[i] + fn[i] for i in present],
        support,
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
    try:
        ratio = majority / minority
    except OverflowError:
        ratio = math.inf
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


def _score_values(scores, metric: str) -> list[float]:
    """Convert scores to finite floats, rejecting unrepresentable values.

    Python integers larger than a float raise ``OverflowError`` on conversion.
    Such a score cannot be ranked, so it is a malformed measurement just like
    NaN/Inf; the caller must see a ``ValueError`` instead of an overflow from
    inside the metric.
    """
    out: list[float] = []
    for score in scores:
        if (isinstance(score, bool)
                or isinstance(score, (str, bytes, bytearray))):
            raise ValueError(
                f"{metric} scores must be finite numbers; a boolean/string "
                f"score has no rank")
        try:
            value = float(score)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"{metric} scores must be finite numbers; a non-numeric "
                f"score has no rank") from None
        if not math.isfinite(value):
            raise ValueError(
                f"{metric} scores must be finite numbers; a NaN/Inf score "
                f"has no rank")
        out.append(value)
    return out


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
    scores = _score_values(scores, "roc_auc")
    labels = _binary_labels(labels)
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
    scores = _score_values(scores, "average_precision")
    labels = _binary_labels(labels)
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
        cols = [_score_values(col, "binary_auc_report") for col in cols]
    except ValueError:
        raise ValueError(
            "binary_auc_report scores must be finite numbers; a boolean/string "
            "or non-numeric score has no rank") from None
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
    relevant_sets = [set(rel) for rel in relevant]
    if not any(relevant_sets):
        return None
    ndcgs = []
    for r, rel_set in zip(ranked, relevant_sets):
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
