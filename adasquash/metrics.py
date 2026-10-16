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
  `macro_weighted_f1`, which *names* the averaging it uses: `macro_f1`
  (unweighted mean over classes present), `weighted_f1` (support-weighted)
  and `micro_f1` (= accuracy). Macro and micro can differ wildly under class
  imbalance, so the caller must state which one it reports.

Everything is pure `math` (no scipy/sklearn), so the training scripts can run
in a minimal environment.
"""
import math
from typing import Optional, Sequence

__all__ = [
    "macro_weighted_f1",
    "mean_std_stderr",
    "paired_z_one_sided",
    "welch_z_one_sided",
    "wilson_interval",
]

_Z_95 = 1.959963984540054


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


def macro_weighted_f1(pred: Sequence[int], gold: Sequence[int], num_classes: int) -> dict:
    """Multiclass F1 with the averaging explicitly named.

    * `macro_f1`: unweighted mean of per-class F1 over classes occurring in
      `gold` or `pred` (sklearn's default labelling). Every class counts once,
      so rare classes are not drowned by frequent ones.
    * `weighted_f1`: per-class F1 weighted by the class's support in `gold`.
    * `micro_f1`: total true positives over total support (== accuracy for a
      single-label argmax prediction).

    With class imbalance macro and micro can rank models differently, so call
    sites report the metric by name.
    """
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    pred = [int(p) for p in pred]
    gold = [int(g) for g in gold]
    if len(pred) != len(gold):
        raise ValueError(f"pred/gold length mismatch: {len(pred)} vs {len(gold)}")
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    for p, g in zip(pred, gold):
        if not 0 <= g < num_classes or not 0 <= p < num_classes:
            raise ValueError(f"class id out of range [0, {num_classes})")
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    support = sum(tp) + sum(fn)
    present = [i for i in range(num_classes) if tp[i] + fp[i] + fn[i] > 0]
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
