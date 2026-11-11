"""Fine-grained distribution-shift analysis for held-out data splits.

The leakage-safe pipeline already reports a single total-variation distance
between the train and held-out token histograms (``scripts/audit_data.py``).
A single TVD hides where the divergence lives: a held-out split can match the
training distribution on high-frequency function words while diverging on
rare content tokens, or match on short documents while diverging on long
ones.  For a normalization-layer comparison, that matters because the
gradient statistics the module sees depend on which tokens and which document
lengths dominate the held-out loss.

This module splits both token frequencies and document lengths into
equal-frequency buckets and reports the TVD *within* each bucket, so a
reviewer can see exactly which part of the distribution the held-out slice
misrepresents.  All functions operate on plain ``collections.Counter`` /
length samples and return JSON-safe mappings, so they can be embedded in a
run manifest or an analysis plan without re-reading the dataset.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable


def _histogram_total(counter: Counter) -> int:
    return int(sum(counter.values()))


def _bucket_edges_for_counts(counts: list[int], n_buckets: int) -> list[int]:
    """Equal-frequency bucket edges over sorted item counts.

    Returns ``n_buckets + 1`` edges so that consecutive edges define one
    bucket.  When there are fewer distinct items than buckets, empty buckets
    are collapsed and the returned edges still define non-empty buckets.
    """
    if n_buckets < 1:
        msg = f"n_buckets must be >= 1, got {n_buckets}"
        raise ValueError(msg)
    ordered = sorted(counts)
    n = len(ordered)
    if n == 0:
        return [0]
    if n <= n_buckets:
        return list(range(n + 1))
    target = n / max(n_buckets, 1)
    edges = [0]
    for bucket in range(1, n_buckets):
        pos = bucket * target
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            edges.append(lo)
        else:
            edges.append(lo)
    edges.append(n)
    cleaned: list[int] = []
    for edge in edges:
        if not cleaned or edge > cleaned[-1]:
            cleaned.append(edge)
    if cleaned[-1] != n:
        cleaned.append(n)
    return cleaned


def _is_boolean_scalar(value) -> bool:
    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == "numpy" and cls.__name__ in ("bool_", "bool")


def token_frequency_buckets(counter: Counter, n_buckets: int = 4) -> list[set[int]]:
    """Partition token ids into ``n_buckets`` equal-frequency buckets.

    Bucket 0 holds the most frequent token ids, bucket ``n_buckets - 1`` the
    rarest.  Token ids that never appear are not assigned to any bucket.
    Returns one set of token ids per bucket.
    """
    if not isinstance(counter, Counter):
        msg = "counter must be a collections.Counter"
        raise TypeError(msg)
    if _is_boolean_scalar(n_buckets) or not isinstance(n_buckets, int):
        msg = f"n_buckets must be an integer, got {n_buckets!r}"
        raise TypeError(msg)
    if n_buckets < 1:
        msg = f"n_buckets must be >= 1, got {n_buckets}"
        raise ValueError(msg)
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    edges = _bucket_edges_for_counts([count for _, count in items], n_buckets)
    buckets: list[set[int]] = []
    for i in range(len(edges) - 1):
        bucket = {token_id for token_id, _ in items[edges[i] : edges[i + 1]]}
        buckets.append(bucket)
    return buckets


def _counter_subset(counter: Counter, token_ids: set[int]) -> Counter:
    return Counter(
        {token_id: count for token_id, count in counter.items() if token_id in token_ids}
    )


def _total_variation(a: Counter, b: Counter) -> float:
    total_a = _histogram_total(a)
    total_b = _histogram_total(b)
    if not total_a or not total_b:
        return 1.0
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / total_a - b.get(k, 0) / max(total_b, 1e-10)) for k in keys)


def token_frequency_tvd(
    train_counter: Counter,
    heldout_counter: Counter,
    n_buckets: int = 4,
) -> dict:
    """Total-variation distance within each token-frequency bucket.

    Tokens are ranked by their train-side frequency and split into
    ``n_buckets`` equal-frequency buckets.  The TVD is then computed within
    each bucket between the train and held-out histograms restricted to that
    bucket.  A bucket whose held-out mass is near zero reports a TVD of 1.0
    (maximal divergence), which flags a held-out slice that under-represents
    an entire frequency band.
    """
    if not isinstance(train_counter, Counter) or not isinstance(heldout_counter, Counter):
        msg = "train_counter and heldout_counter must be collections.Counter"
        raise TypeError(msg)
    buckets = token_frequency_buckets(train_counter, n_buckets)
    train_total = _histogram_total(train_counter)
    heldout_total = _histogram_total(heldout_counter)
    per_bucket: list[dict] = []
    for index, bucket in enumerate(buckets):
        train_sub = _counter_subset(train_counter, bucket)
        heldout_sub = _counter_subset(heldout_counter, bucket)
        train_mass = _histogram_total(train_sub) / train_total if train_total else 0.0
        heldout_mass = _histogram_total(heldout_sub) / heldout_total if heldout_total else 0.0
        per_bucket.append(
            {
                "bucket": index,
                "n_tokens": len(bucket),
                "train_mass": train_mass,
                "heldout_mass": heldout_mass,
                "mass_ratio": (heldout_mass / train_mass if train_mass else None),
                "tvd": _total_variation(train_sub, heldout_sub),
            }
        )
    return {
        "n_buckets": n_buckets,
        "train_total_tokens": train_total,
        "heldout_total_tokens": heldout_total,
        "overall_tvd": _total_variation(train_counter, heldout_counter),
        "per_bucket": per_bucket,
    }


def document_length_buckets(lengths: Iterable[int], n_buckets: int = 4) -> list[tuple[int, int]]:
    """Partition document lengths into ``n_buckets`` equal-count buckets.

    Returns a list of ``(lower, upper)`` inclusive length bounds, one per
    bucket.  Bucket 0 holds the shortest documents.  When the sample is
    smaller than ``n_buckets``, each document forms its own bucket.
    """
    if _is_boolean_scalar(n_buckets) or not isinstance(n_buckets, int):
        msg = f"n_buckets must be an integer, got {n_buckets!r}"
        raise TypeError(msg)
    if n_buckets < 1:
        msg = f"n_buckets must be >= 1, got {n_buckets}"
        raise ValueError(msg)
    ordered = sorted(int(length) for length in lengths)
    n = len(ordered)
    if n == 0:
        return []
    edges = _bucket_edges_for_counts(ordered, n_buckets)
    buckets: list[tuple[int, int]] = []
    for i in range(len(edges) - 1):
        lower = ordered[edges[i]]
        upper = ordered[edges[i + 1] - 1]
        buckets.append((lower, upper))
    return buckets


def document_length_tvd(
    train_lengths: Iterable[int],
    heldout_lengths: Iterable[int],
    n_buckets: int = 4,
) -> dict:
    """Total-variation distance within each document-length bucket.

    Document lengths are ranked and split into ``n_buckets`` equal-count
    buckets on the train side.  The held-out lengths are assigned to the same
    buckets by their length value, and the TVD is computed within each bucket
    between the train and held-out length histograms.
    """
    if _is_boolean_scalar(n_buckets) or not isinstance(n_buckets, int):
        msg = f"n_buckets must be an integer, got {n_buckets!r}"
        raise TypeError(msg)
    if n_buckets < 1:
        msg = f"n_buckets must be >= 1, got {n_buckets}"
        raise ValueError(msg)
    train_ordered = sorted(int(length) for length in train_lengths)
    heldout_ordered = sorted(int(length) for length in heldout_lengths)
    if not train_ordered:
        msg = "train_lengths must be non-empty"
        raise ValueError(msg)
    buckets = document_length_buckets(train_ordered, n_buckets)
    n_train = len(train_ordered)
    n_heldout = len(heldout_ordered)
    per_bucket: list[dict] = []
    for index, (lower, upper) in enumerate(buckets):
        train_in_bucket = sum(1 for length in train_ordered if lower <= length <= upper)
        heldout_in_bucket = sum(1 for length in heldout_ordered if lower <= length <= upper)
        train_mass = train_in_bucket / max(n_train, 1) if n_train else 0.0
        heldout_mass = heldout_in_bucket / max(n_heldout, 1) if n_heldout else 0.0
        train_sub = Counter(length for length in train_ordered if lower <= length <= upper)
        heldout_sub = Counter(length for length in heldout_ordered if lower <= length <= upper)
        per_bucket.append(
            {
                "bucket": index,
                "length_lower": lower,
                "length_upper": upper,
                "n_train_documents": train_in_bucket,
                "n_heldout_documents": heldout_in_bucket,
                "train_mass": train_mass,
                "heldout_mass": heldout_mass,
                "mass_ratio": (heldout_mass / train_mass if train_mass else None),
                "tvd": _total_variation(train_sub, heldout_sub),
            }
        )
    return {
        "n_buckets": n_buckets,
        "n_train_documents": n_train,
        "n_heldout_documents": n_heldout,
        "overall_tvd": _total_variation(Counter(train_ordered), Counter(heldout_ordered)),
        "per_bucket": per_bucket,
    }


def fine_grained_shift_report(  # noqa: PLR0913
    train_counter: Counter,
    heldout_counter: Counter,
    train_lengths: Iterable[int],
    heldout_lengths: Iterable[int],
    *,
    n_buckets: int = 4,
    top_k: int = 10,
) -> dict:
    """Fine-grained distribution-shift report for one held-out split.

    Combines the overall token TVD with per-bucket TVDs for token frequency
    and document length, plus the top-k most-shifted token ids.  Intended to
    be embedded in a run manifest or an analysis plan so that a held-out
    split's fitness for selection / final reporting is judged on its
    fine-grained fit to the training distribution, not only on an aggregate
    TVD.
    """
    if not isinstance(train_counter, Counter) or not isinstance(heldout_counter, Counter):
        msg = "train_counter and heldout_counter must be collections.Counter"
        raise TypeError(msg)
    if _is_boolean_scalar(top_k) or not isinstance(top_k, int):
        msg = f"top_k must be an integer, got {top_k!r}"
        raise TypeError(msg)
    if top_k < 1:
        msg = f"top_k must be a positive integer, got {top_k!r}"
        raise ValueError(msg)

    token_report = token_frequency_tvd(train_counter, heldout_counter, n_buckets)
    length_report = document_length_tvd(train_lengths, heldout_lengths, n_buckets)

    train_total = _histogram_total(train_counter)
    heldout_total = _histogram_total(heldout_counter)
    keys = set(train_counter) | set(heldout_counter)
    scored = []
    for token_id in keys:
        train_prob = train_counter.get(token_id, 0) / train_total if train_total else 0.0
        heldout_prob = heldout_counter.get(token_id, 0) / heldout_total if heldout_total else 0.0
        scored.append((abs(train_prob - heldout_prob), token_id, train_prob, heldout_prob))
    scored.sort(key=lambda item: (-item[0], item[1]))
    top_shifted = [
        {
            "token_id": token_id,
            "abs_shift": shift,
            "train_probability": train_prob,
            "heldout_probability": heldout_prob,
        }
        for shift, token_id, train_prob, heldout_prob in scored[:top_k]
    ]

    bucket_tvds = [bucket["tvd"] for bucket in token_report["per_bucket"]]
    length_tvds = [bucket["tvd"] for bucket in length_report["per_bucket"]]

    return {
        "schema": "adasquash.fine_grained_shift.v1",
        "n_buckets": n_buckets,
        "top_k": top_k,
        "overall_token_tvd": token_report["overall_tvd"],
        "overall_length_tvd": length_report["overall_tvd"],
        "token_frequency_buckets": token_report["per_bucket"],
        "document_length_buckets": length_report["per_bucket"],
        "max_token_bucket_tvd": max(bucket_tvds) if bucket_tvds else None,
        "max_length_bucket_tvd": max(length_tvds) if length_tvds else None,
        "top_shifted_tokens": top_shifted,
        "train_total_tokens": train_total,
        "heldout_total_tokens": heldout_total,
        "n_train_documents": len(train_lengths),
        "n_heldout_documents": len(heldout_lengths),
    }


__all__ = [
    "document_length_buckets",
    "document_length_tvd",
    "fine_grained_shift_report",
    "token_frequency_buckets",
    "token_frequency_tvd",
]
