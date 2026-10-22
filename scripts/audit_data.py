from __future__ import annotations
"""Audit the leakage-safety of the streaming data pipeline.

Reads a bounded sample of the train / validation / test splits produced by
`scripts/data_pipeline.py` and reports

* cleaning coverage: how many documents were dropped (and why) by the exact
  same cleaner the trainer uses - missing / NaN text, empty documents,
  degenerate short documents, length outliers, duplicates (exact and
  canonical) and tokenizer failures;
* value sanity: token ids out of the tokenizer's range, non-finite length
  statistics (NaN / Inf), duplicate rates per split;
* split hygiene: duplicate documents shared between any two splits - train vs
  held-out *and* validation vs test (a duplicate in two splits is a leak no
  matter how the split was drawn, and a val/test duplicate makes the
  selection split peek at the sealed final split). Duplicates are detected on
  the exact text *and* on a canonical form (Unicode NFKC + collapsed
  whitespace), so formatting variants are caught too;
* near-duplicate hygiene: exact/canonical hashes cannot see a held-out
  document that repeats most of a training document with a changed sentence
  or template.  A deterministic token 8-shingle MinHash sketch with LSH
  candidate bands estimates cross-split Jaccard similarity and reports pairs
  at or above `--near-dup-threshold` (default 0.8); pairs within
  `--near-dup-margin` below the threshold are listed as borderline rather
  than silently dropped.  The scan is bounded by `--near-dup-max-docs`, and a
  sampled/truncated scan reports itself as a lower bound rather than claiming
  the splits are clean.  A zero-hit scan also carries a one-sided confidence
  upper bound on the contamination rate (`--max-near-dup-rate` makes it a
  pre-registered failure rule), and `--append-near-dup-to-ledgers` writes the
  canonical digests of flagged held-out documents into the existing
  `--write-train-hash-file` / `--write-val-hash-file` ledgers so the trainer
  can drop them instead of only reporting them;
* group hygiene (`--group-key`): documents sharing a grouping unit (same
  user / item / source page) must not straddle any split boundary. The audit
  counts held-out documents whose group occurs in an earlier split and fails
  the run when it finds any;
* chronology: train/validation/test must be contiguous slices of the upstream
  order, so no document can sit on two sides and no future document can
  precede the training slice;
* distribution shift: total-variation distance between token frequency
  distributions of train vs validation / test, plus document-length means and
  3xIQR length-outlier counts. Outliers are also counted under the fence
  derived from the *training* lengths only, so a held-out slice is never
  trimmed with statistics it contributed to;
* data adequacy: total effective tokens and documents behind every held-out
  split, with the observed count when the split was fully read and a clearly
  labelled prefix extrapolation otherwise.  `--min-eval-tokens` /
  `--min-eval-docs` turn "the validation slice is too small to support the
  comparison" into a failed audit instead of a wide confidence interval that
  is never mentioned;
* training token budget: the estimated effective train-split token count,
  compared with `--scheduled-train-tokens` (= steps x batch x grad-accum x
  seq-len).  A loader that runs out of unique documents silently starts
  another epoch, so `--fail-on-train-wrap` refuses a run whose advertised
  token budget is really a repeated pass over the same corpus;
* evaluation budget: the number of distinct packed batches behind each
  held-out split, compared with the `--val-batches` / `--test-batches` the
  protocol requests.  A validation pass that asks for more batches than the
  split holds cycles the loader, so the same held-out windows are scored
  repeatedly and the per-batch standard error treats them as independent.
  With `--val-every` and `--max-steps`, the audit also reports how often
  each distinct validation batch is reused during checkpoint selection.
  `--fail-on-eval-reuse` / `--fail-on-eval-shortfall` turn those findings
  into failures instead of notes.

`--write-train-hash-file` emits the training-document hash ledger (exact and
canonical digests) consumed by `--exclude-hash-file` in the trainers, so
held-out splits drop any document whose text also occurs in train;
`--write-val-hash-file` emits the validation ledger consumed by
`--exclude-test-hash-file`, so the sealed test also drops documents duplicated
in the selection split. With `--group-key`, `--write-train-group-file` /
`--write-val-group-file` emit the corresponding grouping-unit ledgers consumed
by `--exclude-group-file` / `--exclude-test-group-file`, so a group lying on
an earlier side is dropped whole from the later sides. A full ledger requires
`--n-docs 0`; with a bounded sample the file covers only the sampled prefix,
and the run records `*.exhaustive = false` and warns instead of implying full
coverage.

This script only *reads* data; it never trains or writes to a split.
"""
import argparse
import json
import math
import sys
from array import array
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import wilson_interval
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest
from scripts.data_pipeline import (DocumentCleaner, canonical_document_hash, document_hash,
                                   extract_group_value, group_hash, load_document_stream)


# ---------------------------------------------------------------------------
# Near-duplicate contamination
# ---------------------------------------------------------------------------
# Exact and canonical hashes only catch a document that is byte-identical (up
# to Unicode/whitespace formatting).  A held-out document that repeats most of
# a training document with a changed sentence, an extra header or a different
# template is still contamination, but it is invisible to those hashes.  The
# audit therefore also compares token 8-shingles with a MinHash + LSH sketch:
# deterministic, bounded-memory, and independent of the tokenizer's string form.
_NEAR_DUP_MASK = (1 << 64) - 1
_NEAR_DUP_SHINGLE_BASE = 0x9E3779B185EBCA87
_NEAR_DUP_BANDS = 40
_NEAR_DUP_ROWS = 6
_NEAR_DUP_PERMUTATIONS = _NEAR_DUP_BANDS * _NEAR_DUP_ROWS
# A long FineWeb document can contain hundreds of thousands of 8-shingles; a
# full signature over all of them would dominate the audit.  Keep the bottom-k
# shingles (deterministic and unbiased with respect to shingle order); for a
# pair that is mostly the same document, the bottom-k sketches overlap in the
# same proportion, so the near-duplicate decision is preserved.
_NEAR_DUP_MAX_SHINGLES = 1024


def _splitmix64(state: int):
    """Deterministic 64-bit PRNG step used to derive the hash permutations."""
    state = (state + 0x9E3779B97F4A7C15) & _NEAR_DUP_MASK
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _NEAR_DUP_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _NEAR_DUP_MASK
    return state, (z ^ (z >> 31)) & _NEAR_DUP_MASK


def _near_dup_permutations(n: int = _NEAR_DUP_PERMUTATIONS):
    """`n` affine permutations of the 64-bit shingle-hash space.

    `a` is forced odd, so `(a*h + b) mod 2**64` is a bijection; taking the
    minimum over a document's shingles is therefore a valid MinHash of its
    shingle set, not merely a random projection.  The seed is fixed so two
    audits of the same corpus produce the same signatures.
    """
    a, b = [], []
    state = 0xAD45_9A5D_2026_0001
    while len(a) < n:
        state, r1 = _splitmix64(state)
        state, r2 = _splitmix64(state)
        a.append((r1 | 1) & _NEAR_DUP_MASK)
        b.append(r2)
    return (np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))


_NEAR_DUP_A, _NEAR_DUP_B = _near_dup_permutations()


def token_shingles(ids, shingle_size: int = 8) -> set:
    """Set of rolling 64-bit hashes of the token n-grams in one document.

    A rolling polynomial hash keeps this O(len(ids)) in Python and makes the
    result independent of the process-wide hash seed (unlike `hash(tuple)`), so
    two audit runs compare the same shingles.
    """
    k = int(shingle_size)
    if k < 1:
        raise ValueError(f"shingle_size must be >= 1, got {shingle_size}")
    tokens = [int(t) + 1 for t in ids]
    n = len(tokens)
    if n == 0:
        return set()
    k = min(k, n)
    h = 0
    for token in tokens[:k]:
        h = ((h * _NEAR_DUP_SHINGLE_BASE) + token) & _NEAR_DUP_MASK
    out = {h}
    if n == k:
        return out
    place = pow(_NEAR_DUP_SHINGLE_BASE, k - 1, 1 << 64)
    for i in range(k, n):
        outgoing = (tokens[i - k] * place) & _NEAR_DUP_MASK
        h = (((h - outgoing) * _NEAR_DUP_SHINGLE_BASE) + tokens[i]) & _NEAR_DUP_MASK
        out.add(h)
    return out


def minhash_signature(shingles) -> tuple:
    """MinHash signature (one minimum per permutation) of a shingle set."""
    if not shingles:
        return tuple([int(_NEAR_DUP_MASK)] * _NEAR_DUP_PERMUTATIONS)
    hashes = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
    best = np.full(_NEAR_DUP_PERMUTATIONS, _NEAR_DUP_MASK, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for start in range(0, hashes.size, 512):
            chunk = hashes[start:start + 512]
            values = (_NEAR_DUP_A[:, None] * chunk[None, :]
                      + _NEAR_DUP_B[:, None])
            np.minimum(best, values.min(axis=1), out=best)
    return tuple(int(v) for v in best)


def near_duplicate_record(ids, shingle_size: int = 8,
                          max_shingles: int = _NEAR_DUP_MAX_SHINGLES,
                          canonical_digest=None) -> dict:
    """Compact near-duplicate sketch of one document's token stream.

    At most `max_shingles` bottom-hash shingles are used, so a very long
    document cannot make the audit quadratic in its length.  The full count is
    still reported alongside the number actually sketched.

    `canonical_digest` is the document's canonical-text digest (the same
    digest `DocumentCleaner` looks up in a cross-split ledger).  Recording it
    next to the sketch is what lets a flagged held-out document be turned into
    an exclusion ledger entry instead of remaining a purely diagnostic hit.
    It is optional so the sketch stays usable on token streams that did not
    come from a text document.
    """
    shingles = token_shingles(ids, shingle_size)
    total_shingles = len(shingles)
    limit = int(max_shingles)
    if limit < 1:
        raise ValueError(f"max_shingles must be >= 1, got {max_shingles}")
    if total_shingles > limit:
        values = np.fromiter(shingles, dtype=np.uint64, count=total_shingles)
        values = np.partition(values, limit - 1)[:limit]
        shingles = set(int(v) for v in values)
    record = {
        "signature": minhash_signature(shingles),
        "n_tokens": len(ids),
        "n_shingles": total_shingles,
        "n_shingles_sketched": len(shingles),
    }
    if canonical_digest is not None:
        if isinstance(canonical_digest, str):
            try:
                canonical_digest = bytes.fromhex(canonical_digest)
            except ValueError:
                raise ValueError(
                    "canonical_digest must be bytes or a hex string, got "
                    f"{canonical_digest!r}") from None
        if not isinstance(canonical_digest, (bytes, bytearray)) or not canonical_digest:
            raise ValueError(
                "canonical_digest must be non-empty bytes or a hex string, got "
                f"{canonical_digest!r}")
        record["canonical_digest"] = bytes(canonical_digest).hex()
    return record


def _band_key(signature, band: int):
    start = band * _NEAR_DUP_ROWS
    return (band,) + tuple(signature[start:start + _NEAR_DUP_ROWS])


def near_duplicate_leakage(records_by_split: dict, *, threshold: float = 0.8,
                           margin: float = 0.05, shingle_size: int = 8,
                           max_examples: int = 20,
                           exhaustive_by_split: dict | None = None,
                           truncated_by_split: dict | None = None) -> dict:
    """Find held-out documents whose shingle sets overlap an earlier split.

    A document is compared only with documents from an earlier split (train
    before validation/test, validation before test), so a flagged pair is
    genuinely cross-split.  LSH bands generate candidates cheaply; the
    MinHash signature then estimates Jaccard similarity, which is what the
    threshold is applied to.  The estimate has a standard error of roughly
    `sqrt(J*(1-J)/240)` (about 0.026 at J=0.8), so a value just below the
    threshold is not proof of independence - it is a reason to inspect the
    corpus, not a certificate.
    """
    if not isinstance(records_by_split, dict):
        raise ValueError(
            f"records_by_split must be a mapping, got {type(records_by_split).__name__}")
    # A bool is not a probability: `float(True)` would silently become the
    # permissive 100% threshold and `float(False)` a valid zero margin.
    if _is_boolean_scalar(threshold):
        raise ValueError(
            f"near-duplicate threshold must be in (0, 1], got {threshold!r}")
    try:
        threshold = float(threshold)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"near-duplicate threshold must be in (0, 1], got {threshold!r}") from None
    if not (math.isfinite(threshold) and 0.0 < threshold <= 1.0):
        raise ValueError(f"near-duplicate threshold must be in (0, 1], got {threshold}")
    if _is_boolean_scalar(margin):
        raise ValueError(
            f"near-duplicate margin must be in [0, 1), got {margin!r}")
    try:
        margin = float(margin)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"near-duplicate margin must be in [0, 1), got {margin!r}") from None
    if not (math.isfinite(margin) and 0.0 <= margin < 1.0):
        raise ValueError(f"near-duplicate margin must be in [0, 1), got {margin}")
    screening = max(0.0, float(threshold) - margin)
    ordered = [name for name in ("train", "validation", "test")
               if records_by_split.get(name)]
    exhaustive_by_split = exhaustive_by_split or {}
    truncated_by_split = truncated_by_split or {}
    pairs = {}
    by_later_split = {}
    problems = []
    notes = []
    any_truncated = any(bool(truncated_by_split.get(name)) for name in ordered)
    any_sampled = any(not bool(exhaustive_by_split.get(name, True)) for name in ordered)
    # A held-out document can be similar to documents in more than one earlier
    # split.  For the contamination-rate bound the denominator is the whole
    # later split, so the flagged set must be a union over earlier splits; a
    # per-pair count would double-count the same contaminated document.
    later_union: dict = {}
    for i, earlier in enumerate(ordered):
        earlier_records = records_by_split[earlier]
        if not earlier_records:
            continue
        earlier_sigs = np.asarray([r["signature"] for r in earlier_records],
                                  dtype=np.uint64)
        index: dict = {}
        for doc_i, signature in enumerate(earlier_records):
            for band in range(_NEAR_DUP_BANDS):
                index.setdefault(_band_key(signature["signature"], band), []).append(doc_i)
        for later in ordered[i + 1:]:
            later_records = records_by_split[later]
            if not later_records:
                continue
            n_pairs = 0
            flagged_docs = set()
            flagged_digests = set()
            missing_digest_positions = set()
            candidate_pairs = 0
            examples = []
            borderline_pairs = 0
            borderline_docs = set()
            borderline_examples = []
            for later_i, later_record in enumerate(later_records):
                candidates = set()
                for band in range(_NEAR_DUP_BANDS):
                    candidates.update(index.get(
                        _band_key(later_record["signature"], band), ()))
                if not candidates:
                    continue
                candidate_pairs += len(candidates)
                candidate_ids = np.fromiter(sorted(candidates), dtype=np.int64,
                                            count=len(candidates))
                sims = (earlier_sigs[candidate_ids]
                        == np.asarray(later_record["signature"], dtype=np.uint64)).mean(axis=1)
                hits = np.flatnonzero(sims >= float(threshold))
                borderline = np.flatnonzero((sims >= screening)
                                            & (sims < float(threshold)))
                if borderline.size:
                    borderline_pairs += int(borderline.size)
                    borderline_docs.add(later_i)
                    if len(borderline_examples) < max_examples:
                        # `borderline` holds positions into `sims`, not
                        # document indices: map the best position through it
                        # and then through `candidate_ids`, or the example can
                        # name a different earlier document (and similarity)
                        # than the pair that triggered it.
                        best_pos = int(np.argmax(sims[borderline]))
                        sim_pos = int(borderline[best_pos])
                        earlier_i = int(candidate_ids[sim_pos])
                        borderline_examples.append({
                            "earlier_document_index": int(
                                earlier_records[earlier_i].get("doc_index", earlier_i)),
                            "later_document_index": int(
                                later_record.get("doc_index", later_i)),
                            "estimated_jaccard": round(float(sims[sim_pos]), 4),
                        })
                if not hits.size:
                    continue
                n_pairs += int(hits.size)
                flagged_docs.add(later_i)
                # The canonical digest is what `DocumentCleaner` checks
                # against an exclusion ledger.  A flagged document without one
                # can be reported but not remediated, so it is counted
                # separately instead of silently omitted from the ledger.
                digest = later_record.get("canonical_digest")
                if isinstance(digest, str) and digest:
                    flagged_digests.add(digest.lower())
                else:
                    missing_digest_positions.add(later_i)
                if len(examples) < max_examples:
                    # Same indirection as the borderline example above.
                    best_pos = int(np.argmax(sims[hits]))
                    sim_pos = int(hits[best_pos])
                    earlier_i = int(candidate_ids[sim_pos])
                    examples.append({
                        "earlier_split": earlier,
                        "later_split": later,
                        "earlier_document_index": int(
                            earlier_records[earlier_i].get("doc_index", earlier_i)),
                        "later_document_index": int(
                            later_record.get("doc_index", later_i)),
                        "estimated_jaccard": round(float(sims[sim_pos]), 4),
                        "earlier_n_tokens": int(earlier_records[earlier_i]["n_tokens"]),
                        "later_n_tokens": int(later_record["n_tokens"]),
                    })
            aggregate = later_union.setdefault(
                later, {"positions": set(), "doc_indices": set(),
                        "digests": set(), "missing_positions": set(),
                        "missing_doc_indices": set()})
            aggregate["positions"].update(flagged_docs)
            aggregate["doc_indices"].update(
                int(later_records[position].get("doc_index", position))
                for position in flagged_docs)
            aggregate["digests"].update(flagged_digests)
            aggregate["missing_positions"].update(missing_digest_positions)
            aggregate["missing_doc_indices"].update(
                int(later_records[position].get("doc_index", position))
                for position in missing_digest_positions)
            key = f"{earlier}->{later}"
            pairs[key] = {
                "earlier_split": earlier,
                "later_split": later,
                "earlier_documents": len(earlier_records),
                "later_documents": len(later_records),
                "candidate_pairs": int(candidate_pairs),
                "flagged_pairs": int(n_pairs),
                "flagged_later_documents": len(flagged_docs),
                "flagged_later_document_indices": sorted(
                    int(later_records[position].get("doc_index", position))
                    for position in flagged_docs),
                "flagged_later_digests": sorted(flagged_digests),
                "flagged_later_tokens": sum(
                    int(later_records[position].get("n_tokens") or 0)
                    for position in flagged_docs),
                "flagged_later_missing_digest": len(missing_digest_positions),
                "borderline_pairs": int(borderline_pairs),
                "borderline_later_documents": len(borderline_docs),
                "borderline_examples": borderline_examples,
                "examples": examples,
            }
            if n_pairs:
                problems.append(
                    f"near-duplicate documents cross {key}: {len(flagged_docs)} later "
                    f"document(s) have a token-shingle Jaccard estimate >= {threshold} "
                    f"with an earlier split ({n_pairs} matching pair(s)); exact/canonical "
                    "hash dedup cannot catch this overlap")
    n_missing_digests = 0
    for later in ordered[1:]:
        aggregate = later_union.get(later) or {
            "positions": set(), "doc_indices": set(), "digests": set(),
            "missing_positions": set(), "missing_doc_indices": set()}
        by_later_split[later] = {
            "documents": len(records_by_split[later]),
            "flagged_documents": len(aggregate["positions"]),
            "flagged_document_indices": sorted(aggregate["doc_indices"]),
            "flagged_digests": sorted(aggregate["digests"]),
            "flagged_tokens": sum(
                int(records_by_split[later][position].get("n_tokens") or 0)
                for position in aggregate["positions"]),
            "missing_digest_documents": len(aggregate["missing_positions"]),
            "missing_digest_document_indices": sorted(aggregate["missing_doc_indices"]),
        }
        n_missing_digests += len(aggregate["missing_positions"])
    if n_missing_digests:
        notes.append(
            f"{n_missing_digests} flagged held-out document(s) carry no canonical "
            "digest, so they can be reported but not turned into exclusion-ledger "
            "entries")
    n_borderline = sum(pair["borderline_pairs"] for pair in pairs.values())
    if n_borderline:
        notes.append(
            f"{n_borderline} cross-split pair(s) are within {margin:g} Jaccard of the "
            f"{threshold:g} threshold (screening >= {screening:g}); they are reported "
            "for inspection but are not counted as confirmed near-duplicate failures")
    truncated = any_truncated or any_sampled
    if truncated:
        found = any(pair["flagged_pairs"] for pair in pairs.values())
        notes.append(
            "the near-duplicate scan covered a sampled/truncated prefix (or fewer than "
            "the requested documents), so "
            + ("the flagged-pair count is a lower bound; more contaminated pairs may "
               "exist beyond the prefix" if found else
               "zero flagged pairs is a lower bound, not proof that the splits are "
               "near-duplicate free"))
    return {
        "enabled": True,
        "threshold_jaccard": float(threshold),
        "borderline_margin": margin,
        "screening_jaccard": screening,
        "estimator": "minhash_lsh",
        "permutations": _NEAR_DUP_PERMUTATIONS,
        "bands": _NEAR_DUP_BANDS,
        "rows_per_band": _NEAR_DUP_ROWS,
        "shingle_size": int(shingle_size),
        "pairs": pairs,
        "by_later_split": by_later_split,
        "problems": problems,
        "notes": notes,
        "exhaustive": not truncated,
        "lower_bound": bool(truncated),
    }


def near_duplicate_scan_bound(report: dict, *, confidence: float = 0.95,
                              max_rate=None) -> dict:
    """Quantify what a near-duplicate scan *failed to find*.

    A zero-hit scan is routinely read as "the splits are clean".  That is a
    lower bound, not a measurement: with N held-out documents and zero hits,
    the one-sided upper confidence bound on the contamination proportion is
    still finite (about 0.0007 at N=4000, 95%).  This function attaches that
    bound to each later split so the audit can state the largest
    contamination rate its sample is compatible with, and - when
    ``max_rate`` is pre-registered - refuse a scan that cannot rule out more
    contamination than the tolerable amount.

    The denominator is the number of *sketched* documents in the later split,
    so a sampled/truncated scan is a bound on the scanned prefix only; the
    report says so explicitly instead of extrapolating a prefix to the whole
    split.  ``max_rate`` therefore checks the scanned data, and passing it
    does not certify the unsampled remainder.
    """
    if not isinstance(report, dict):
        raise ValueError(f"report must be a mapping, got {type(report).__name__}")
    if _is_boolean_scalar(confidence):
        raise ValueError(f"confidence must be a probability, got bool {confidence!r}")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(
            f"confidence must be a probability, got {confidence!r}") from None
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(
            f"confidence must be finite and in (0, 1), got {confidence!r}")
    if max_rate is not None:
        if _is_boolean_scalar(max_rate):
            raise ValueError(f"max_rate must be a probability or None, got bool {max_rate!r}")
        try:
            max_rate = float(max_rate)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"max_rate must be a probability or None, got {max_rate!r}") from None
        if not math.isfinite(max_rate) or not 0.0 <= max_rate <= 1.0:
            raise ValueError(
                f"max_rate must be finite and in [0, 1], got {max_rate!r}")
    z = NormalDist().inv_cdf(confidence)
    exhaustive = bool(report.get("exhaustive"))
    scope = "audited_split" if exhaustive else "scanned_prefix"
    by_later = report.get("by_later_split")
    if not isinstance(by_later, dict):
        by_later = {}
    bounds = {}
    problems = []
    notes = []
    for later, block in sorted(by_later.items()):
        block = block if isinstance(block, dict) else {}
        documents = block.get("documents")
        flagged = block.get("flagged_documents")
        if _is_boolean_scalar(documents) or not isinstance(documents, int) or documents < 0:
            raise ValueError(
                f"by_later_split[{later!r}].documents must be a non-negative "
                f"integer, got {documents!r}")
        if _is_boolean_scalar(flagged) or not isinstance(flagged, int) or flagged < 0:
            raise ValueError(
                f"by_later_split[{later!r}].flagged_documents must be a "
                f"non-negative integer, got {flagged!r}")
        if flagged > documents:
            raise ValueError(
                f"by_later_split[{later!r}] reports {flagged} flagged documents "
                f"out of only {documents}; a subset cannot be larger than its set")
        observed = None if documents == 0 else flagged / documents
        upper = None if documents == 0 else wilson_interval(
            flagged, documents, z=z)[1]
        passes = None if max_rate is None or upper is None else upper <= max_rate
        bounds[later] = {
            "documents": documents,
            "flagged_documents": flagged,
            "observed_rate": observed,
            "upper_bound": upper,
            "confidence": confidence,
            "scope": scope,
            "passes_max_rate": passes,
        }
        if max_rate is not None:
            if documents == 0:
                problems.append(
                    f"no {later} documents were sketched, so the near-duplicate "
                    f"scan cannot certify a contamination rate <= {max_rate:g}")
            elif upper > max_rate:
                problems.append(
                    f"the near-duplicate scan of {later} saw {flagged}/{documents} "
                    f"document(s) at/above the threshold (observed {observed:.4%}); "
                    f"the one-sided {confidence:.1%} upper bound is {upper:.4%}, "
                    f"above the pre-registered max rate {max_rate:g}")
    if not bounds:
        notes.append(
            "no later-split scan counts were available, so no contamination "
            "bound could be computed; this is unknown, not clean")
        if max_rate is not None:
            problems.append(
                "no later-split near-duplicate scan counts were available, so "
                f"a contamination rate <= {max_rate:g} cannot be certified")
    if not exhaustive and bounds:
        notes.append(
            "the bound applies to the scanned prefix only; the unsampled "
            "remainder of each split has no finite upper bound from this scan")
    if any(block["flagged_documents"] == 0 for block in bounds.values()):
        notes.append(
            "zero flagged documents does not prove absence: the reported "
            "upper bound is the largest contamination rate compatible with "
            "the scanned sample at the stated confidence")
    return {
        "enabled": bool(report.get("enabled", True)),
        "confidence": confidence,
        "method": "wilson_score_upper_bound",
        "scope": scope,
        "exhaustive": exhaustive,
        # An empty split computes no upper bound, so it cannot certify a
        # population rate even though the scan was exhaustive.
        "certifies_population": bool(
            exhaustive and bounds
            and all(block["upper_bound"] is not None
                    for block in bounds.values())),
        "max_rate": max_rate,
        "by_later_split": bounds,
        "problems": problems,
        "notes": notes,
    }


def near_duplicate_ledger_additions(report: dict) -> dict:
    """Canonical digests of flagged held-out documents, split by ledger role.

    The existing cross-split ledgers are *earlier-side* lists applied to later
    splits: the training ledger is applied to validation and test, and the
    validation ledger to test only.  A flagged near-duplicate document belongs
    in the ledger of the earlier side it duplicates, so a training-ledger
    consumer drops it from both held-out splits and a validation-ledger
    consumer drops it from the sealed test split.  Only the later (held-out)
    document's digest is returned; training documents are never excluded.

    Documents without a canonical digest are counted under
    ``missing_digest_documents`` rather than silently omitted.
    """
    if not isinstance(report, dict):
        raise ValueError(f"report must be a mapping, got {type(report).__name__}")
    additions = {"train": set(), "validation": set()}
    by_later = report.get("by_later_split")
    missing = (sum(int(block.get("missing_digest_documents") or 0)
                   for block in by_later.values() if isinstance(block, dict))
               if isinstance(by_later, dict) else 0)
    pairs = report.get("pairs")
    if not isinstance(pairs, dict):
        return {"hashes": additions, "missing_digest_documents": missing}
    if not isinstance(by_later, dict):
        missing = sum(int(pair.get("flagged_later_missing_digest") or 0)
                      for pair in pairs.values() if isinstance(pair, dict))
    for pair in pairs.values():
        if not isinstance(pair, dict):
            continue
        earlier = pair.get("earlier_split")
        later = pair.get("later_split")
        if later == "validation" and earlier == "train":
            role = "train"
        elif later == "test" and earlier == "train":
            role = "train"
        elif later == "test" and earlier == "validation":
            role = "validation"
        else:
            # The ledger convention only has earlier->later sides; an unknown
            # orientation is not a reason to invent a ledger role.
            continue
        for digest in pair.get("flagged_later_digests") or []:
            try:
                raw = bytes.fromhex(str(digest))
            except (TypeError, ValueError):
                missing += 1
                continue
            if len(raw) != 16:
                # `canonical_document_hash` is a 16-byte blake2b digest. A
                # different length cannot match what DocumentCleaner looks up,
                # so writing it would create a ledger entry that never fires.
                missing += 1
                continue
            additions[role].add(raw)
    return {"hashes": additions, "missing_digest_documents": missing}


def _collect_split(name, dataset, dataset_config, tokenizer, options, n_docs, min_doc_chars, max_doc_chars, dedup, known_hashes=None, known_hashes_extra=None, known_hashes_exhaustive=True, known_hashes_extra_exhaustive=True, collect_hashes=True, group_key=None, known_groups=None, known_groups_extra=None, known_groups_exhaustive=True, known_groups_extra_exhaustive=True, collect_groups=True, exclude_hashes=None, exclude_group_hashes=None, near_dup=False, near_dup_shingle=8, near_dup_max_docs=4000, near_dup_max_shingles=_NEAR_DUP_MAX_SHINGLES):
    stream, report = load_document_stream(dataset, dataset_config, name, options)
    cleaner = DocumentCleaner(dedup=dedup, min_doc_chars=min_doc_chars, max_doc_chars=max_doc_chars)
    # The pipeline checks the cross-split group/document ledgers *before* the
    # within-split duplicate set, because a ledger drop must not make the next
    # copy of the same document look unique.  Replaying that order needs its
    # own cleaner: `cleaner` above reports the pre-ledger view (so overlaps
    # with earlier splits stay visible), while `effective_cleaner` is the split
    # the training run actually consumes.
    effective_cleaner = DocumentCleaner(
        dedup=dedup, min_doc_chars=min_doc_chars, max_doc_chars=max_doc_chars,
        exclude_hashes=exclude_hashes, group_key=group_key,
        exclude_group_hashes=exclude_group_hashes)
    # One machine-word int per document. `lengths` is the pre-ledger (raw)
    # view; `effective_lengths` is what the run actually consumes and is what
    # the held-out token-budget estimate is based on.
    lengths = array('q')
    effective_lengths = array('q')
    near_records: list = []
    near_dup_docs_seen = 0
    near_dup_truncated = False

    def _record_near_dup(doc_index, ids, text) -> None:
        nonlocal near_dup_docs_seen, near_dup_truncated
        near_dup_docs_seen += 1
        if near_dup_docs_seen > int(near_dup_max_docs):
            near_dup_truncated = True
            return
        if not ids:
            # A tokenizer that emits no tokens for a document leaves no
            # shingle to compare; an all-max signature would make every such
            # document look identical to every other one.
            return
        record = near_duplicate_record(
            ids, near_dup_shingle, near_dup_max_shingles,
            canonical_digest=canonical_document_hash(text))
        record['doc_index'] = int(doc_index)
        near_records.append(record)
    # Token ids are counted per document with `np.bincount`; a per-document
    # fallback keeps one bad document from losing the whole batch. Ids that
    # cannot go through bincount (negative) use the Python-counter path.
    # Two histograms are built: the raw one over every kept document, and the
    # *effective* one over the documents that survive the cross-split ledgers
    # the training run applies (exclude_hashes / exclude_group_hashes), i.e.
    # the distribution the model actually evaluates on.
    count_bufs: dict = {'raw': None, 'effective': None}
    legacy_bufs: dict = {'raw': Counter(), 'effective': Counter()}
    text_batch: list = []
    text_batch_effective: list = []
    hashes: set = set()
    groups: set = set()
    overlap = 0
    overlap_extra = 0
    group_overlap = 0
    group_overlap_extra = 0
    group_overlap_groups: set = set()
    group_overlap_extra_groups: set = set()
    group_value_missing = 0
    tokenizer_error_docs: set = set()
    vocab_size = int(getattr(tokenizer, 'vocab_size', 0) or 0)

    def _add_ids(ids, which='raw') -> None:
        try:
            arr = np.asarray(ids, dtype=np.int64)
        except (TypeError, ValueError, OverflowError):
            # A token id above int64 is still a token id: count it through
            # Python's arbitrary-precision Counter so the out-of-range check
            # can report it instead of aborting the whole audit.
            legacy_bufs[which].update(ids)
            return
        if arr.size == 0:
            return
        if arr.min() < 0:
            # np.bincount rejects negatives; a tokenizer that emits them
            # still gets counted here (and flagged below).
            legacy_bufs[which].update(arr.tolist())
            return
        bc = np.bincount(arr)
        counts = count_bufs[which]
        if counts is None:
            counts = np.zeros(max(bc.shape[0], vocab_size + 1), dtype=np.int64)
        elif bc.shape[0] > counts.shape[0]:
            # Grow on first sight of an out-of-vocabulary id so it can still
            # be reported by `token_ids_out_of_range`.
            counts = np.concatenate(
                [counts, np.zeros(bc.shape[0] - counts.shape[0], dtype=np.int64)])
        counts[:bc.shape[0]] += bc
        count_bufs[which] = counts

    def _flush_tokens(batch: list, which: str) -> None:
        if not batch:
            return
        texts = [text for _, text in batch]
        try:
            enc = tokenizer(texts, add_special_tokens=False)['input_ids']
        except Exception:
            enc = None
        # A batch tokenizer that returns a different number of rows cannot be
        # aligned to the documents; fall back to one call per document rather
        # than silently sketching/counting the wrong document.
        if enc is None or len(enc) != len(batch):
            for doc_index, text in batch:
                try:
                    ids = tokenizer(text, add_special_tokens=False)['input_ids']
                except Exception:
                    # The raw and effective passes see the same source
                    # document; count a document that fails either pass once,
                    # not once per tokenization pass.
                    tokenizer_error_docs.add(doc_index)
                    continue
                _add_ids(ids, which)
                if which == 'effective' and near_dup:
                    _record_near_dup(doc_index, ids, text)
        else:
            for (doc_index, text), ids in zip(batch, enc):
                _add_ids(ids, which)
                if which == 'effective' and near_dup:
                    _record_near_dup(doc_index, ids, text)
        batch.clear()

    capped = False
    documents_read = 0
    stream_iter = iter(stream)
    while True:
        # `--n-docs` bounds documents *sampled* (read), not documents kept:
        # otherwise a split with many filtered documents would silently read
        # far past the requested sample and claim a prefix it did not use.
        # Check the cap before pulling the next upstream example, so a bounded
        # audit never decodes/downloads one extra document just to discover
        # that the sample is complete.
        if n_docs and documents_read >= n_docs:
            capped = True
            break
        try:
            example = next(stream_iter)
        except StopIteration:
            break
        documents_read += 1
        doc_index = documents_read - 1
        # Call both cleaners for every example.  The raw cleaner may drop a
        # duplicate that the effective cleaner keeps (the first copy was
        # excluded by a ledger), so its verdict, or the effective verdict,
        # cannot be derived from the other one.
        text = cleaner(example)
        effective_text = effective_cleaner(example)
        if text is not None:
            lengths.append(len(text))
            h = document_hash(text)
            ch = canonical_document_hash(text)
            # The ledger holds exact *and* canonical digests; checking only
            # the exact one would undercount formatting-variant duplicates
            # that the cleaner (and therefore the actual split) does drop.
            if known_hashes is not None and (h in known_hashes or ch in known_hashes):
                overlap += 1
            if known_hashes_extra is not None and (h in known_hashes_extra or ch in known_hashes_extra):
                overlap_extra += 1
            if group_key:
                value = extract_group_value(example, group_key)
                if value is None:
                    group_value_missing += 1
                else:
                    gh = group_hash(value)
                    if known_groups is not None and gh in known_groups:
                        group_overlap += 1
                        group_overlap_groups.add(gh)
                    if known_groups_extra is not None and gh in known_groups_extra:
                        group_overlap_extra += 1
                        group_overlap_extra_groups.add(gh)
            text_batch.append((doc_index, text))
        if effective_text is not None:
            effective_lengths.append(len(effective_text))
            if collect_hashes:
                # Both digests, so a held-out formatting variant of a training
                # document is caught by the ledger the same way the cleaner
                # catches it (and the hash itself is still found either way).
                hashes.add(document_hash(effective_text))
                hashes.add(canonical_document_hash(effective_text))
            if collect_groups and group_key:
                value = extract_group_value(example, group_key)
                if value is not None:
                    groups.add(group_hash(value))
            text_batch_effective.append((doc_index, effective_text))
        if len(text_batch) >= 256:
            _flush_tokens(text_batch, 'raw')
        if len(text_batch_effective) >= 256:
            _flush_tokens(text_batch_effective, 'effective')
    # Reaching the cap does not by itself prove there is more stream: when
    # the split reports exactly the number of documents read, the bounded
    # audit has covered the whole split and its ledger is exhaustive. This
    # recovers the old for-loop result without pulling an extra document.
    nominal_count = report.get('count')
    if (capped
            and isinstance(nominal_count, int) and not isinstance(nominal_count, bool)
            and documents_read == nominal_count):
        capped = False
    _flush_tokens(text_batch, 'raw')
    _flush_tokens(text_batch_effective, 'effective')
    if count_bufs['raw'] is not None:
        tokens = Counter({int(t): int(c) for t, c in enumerate(count_bufs['raw']) if c})
    else:
        tokens = Counter()
    tokens.update(legacy_bufs['raw'])
    if count_bufs['effective'] is not None:
        tokens_effective = Counter({int(t): int(c)
                                    for t, c in enumerate(count_bufs['effective']) if c})
    else:
        tokens_effective = Counter()
    tokens_effective.update(legacy_bufs['effective'])
    stats = cleaner.report()
    # Count every document read from the stream, including dropped ones.
    stats['documents_sampled'] = int(stats.get('documents_read', len(lengths)))
    # The effective split is the one whose dedup guarantee matters. Mirror its
    # cap state explicitly: the raw cleaner may have hit its cap while the
    # effective cleaner (the split actually consumed) did not.
    stats['dedup_cap_reached'] = int(bool(
        effective_cleaner.report().get('dedup_cap_reached')))
    # Whether the counters cover the whole split or only a prefix. A stream
    # that ends naturally before --n-docs is fully read even though n_docs is
    # nonzero, so the cap is tracked by whether the loop actually broke. With
    # a truncated sample, `duplicate_overlap_with_train = 0` is a lower bound
    # on the true overlap, not evidence that the splits are disjoint.
    stats['exhaustive'] = not capped
    effective_stats = effective_cleaner.report()
    if exclude_hashes is not None or exclude_group_hashes is not None:
        stats['ledger_documents_dropped'] = (
            int(effective_stats.get('dropped_cross_split_duplicate', 0))
            + int(effective_stats.get('dropped_cross_split_group', 0)))
        stats['effective_documents_kept'] = int(effective_stats.get('documents_kept', 0))
    else:
        stats['ledger_documents_dropped'] = None
        stats['effective_documents_kept'] = stats.get('documents_kept', len(lengths))
    stats['duplicate_overlap_with_train'] = overlap if known_hashes is not None else None
    stats['duplicate_overlap_with_validation'] = overlap_extra if known_hashes_extra is not None else None
    # A sampled current split *or* a sampled known prefix both make the count
    # a lower bound: the ledger may simply not contain the later duplicate.
    stats['duplicate_overlap_is_lower_bound'] = (
        bool(capped
             or (known_hashes is not None and not known_hashes_exhaustive)
             or (known_hashes_extra is not None and not known_hashes_extra_exhaustive))
        if (known_hashes is not None or known_hashes_extra is not None) else None)
    stats['group_key'] = group_key
    if group_key:
        stats['group_value_missing'] = group_value_missing
        stats['groups_sampled'] = len(groups) if collect_groups else None
        stats['group_overlap_with_train'] = group_overlap if known_groups is not None else None
        stats['group_overlap_with_train_groups'] = (len(group_overlap_groups)
                                                    if known_groups is not None else None)
        stats['group_overlap_with_validation'] = (group_overlap_extra
                                                  if known_groups_extra is not None else None)
        stats['group_overlap_with_validation_groups'] = (len(group_overlap_extra_groups)
                                                         if known_groups_extra is not None else None)
        stats['group_overlap_is_lower_bound'] = (
            bool(capped
                 or (known_groups is not None and not known_groups_exhaustive)
                 or (known_groups_extra is not None and not known_groups_extra_exhaustive))
            if (known_groups is not None or known_groups_extra is not None) else None)
    if lengths:
        stats['doc_chars_mean'] = sum(lengths) / len(lengths)
        stats['doc_chars_min'] = min(lengths)
        stats['doc_chars_max'] = max(lengths)
        # Length outliers by the 3 x IQR fence (only a report; the pipeline's
        # min/max_doc_chars knobs are the actual cleaning policy).
        fence = length_fence(lengths)
        stats['doc_chars_q1'] = fence['q1']
        stats['doc_chars_q3'] = fence['q3']
        stats['length_outliers_3iqr'] = sum(
            1 for v in lengths if v < fence['lower'] or v > fence['upper'])
    else:
        stats['doc_chars_mean'] = stats['doc_chars_min'] = stats['doc_chars_max'] = None
        stats['doc_chars_q1'] = stats['doc_chars_q3'] = None
        stats['length_outliers_3iqr'] = None
    stats['tokens_total'] = int(sum(tokens.values()))
    stats['tokens_total_effective'] = int(sum(tokens_effective.values()))
    stats['chars_total'] = int(sum(lengths))
    stats['chars_total_effective'] = int(sum(effective_lengths))
    stats['tokens_per_char_effective'] = (
        stats['tokens_total_effective'] / stats['chars_total_effective']
        if stats['chars_total_effective'] else None)
    effective_documents = len(effective_lengths)
    stats['tokens_per_document_effective'] = (
        stats['tokens_total_effective'] / effective_documents
        if effective_documents else None)
    stats['tokenizer_errors'] = len(tokenizer_error_docs)
    vocab_limit = getattr(tokenizer, 'vocab_size', 2 ** 31)
    stats['token_ids_out_of_range'] = sum(c for t, c in tokens.items() if not 0 <= t < vocab_limit)
    stats['token_ids_out_of_range_effective'] = sum(
        c for t, c in tokens_effective.items() if not 0 <= t < vocab_limit)
    stats['distinct_token_ids'] = len(tokens)
    stats['distinct_token_ids_effective'] = len(tokens_effective)
    stats['finite_lengths'] = bool(all(math.isfinite(v) for v in (stats['doc_chars_mean'] or 0.0, stats['doc_chars_min'] or 0.0, stats['doc_chars_max'] or 0.0)))
    near_meta = {
        'enabled': bool(near_dup),
        'shingle_size': int(near_dup_shingle),
        'documents_scanned': int(near_dup_docs_seen),
        'documents_sketched': len(near_records),
        'max_documents': int(near_dup_max_docs),
        'max_shingles': int(near_dup_max_shingles),
        'truncated': bool(near_dup_truncated),
    }
    return (stats, tokens, report, hashes, lengths, groups, tokens_effective,
            near_records, near_meta)


def _linear_quantile(ordered, q):
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

def length_fence(lengths) -> dict:
    """3 x IQR fence (q1, q3, lower, upper) of a length sample."""
    ordered = sorted(lengths)
    if not ordered:
        return {'q1': None, 'q3': None, 'lower': 0, 'upper': 0}
    q1 = _linear_quantile(ordered, 0.25)
    q3 = _linear_quantile(ordered, 0.75)
    iqr = q3 - q1
    return {'q1': q1, 'q3': q3, 'lower': max(0, q1 - 3 * iqr), 'upper': q3 + 3 * iqr}


def split_token_budget(stats: dict, report: dict) -> dict:
    """Estimate how many *effective* held-out tokens one split really contains.

    A loss/PPL estimate is only as precise as the number of independent tokens
    behind it, but the split is placed by *document* fraction.  Document
    lengths vary, the cleaner/ledgers remove a non-random subset, and a
    bounded `--n-docs` sample is only a prefix - so the nominal `val_frac`
    alone cannot tell a reviewer whether the held-out slice is large enough
    for the claimed comparison.  When the split was fully read this is the
    observed token count; otherwise the mean effective tokens per kept
    document is extrapolated to the nominal split size and the basis is
    recorded (never presented as an exact count).
    """
    if not isinstance(stats, dict) or not isinstance(report, dict):
        raise ValueError('split_token_budget needs the stats and report mappings')
    sampled = _strict_int_option(
        stats.get('documents_sampled') or 0, 'documents_sampled', 0)
    effective_documents = stats.get('effective_documents_kept')
    if effective_documents is None:
        effective_documents = stats.get('documents_kept', 0)
    effective_documents = _strict_int_option(
        effective_documents or 0, 'effective_documents_kept', 0)
    tokens = _strict_int_option(
        stats.get('tokens_total_effective') or 0, 'tokens_total_effective', 0)
    nominal = report.get('count')
    if nominal is not None:
        nominal = _strict_int_option(nominal, 'nominal_documents', 0)
    result = {
        'nominal_documents': nominal,
        'documents_sampled': sampled,
        'sampled_effective_documents': effective_documents,
        'sampled_effective_tokens': tokens,
        'estimated_effective_documents': None,
        'estimated_effective_tokens': None,
        'tokens_per_effective_document': None,
        'basis': 'no_usable_documents',
        'exhaustive': bool(stats.get('exhaustive', False)),
    }
    if sampled <= 0 or effective_documents <= 0 or tokens <= 0:
        return result
    mean_tokens = tokens / effective_documents
    result['tokens_per_effective_document'] = mean_tokens
    if result['exhaustive']:
        # `exhaustive` means the collector reached the end of the split
        # stream. The observed count is then exact even when the upstream
        # stream is shorter than its reported nominal length; extrapolating
        # to `nominal` would invent documents the stream never contained.
        estimated_documents = float(effective_documents)
        result['basis'] = 'observed'
    elif nominal is not None:
        estimated_documents = nominal * (effective_documents / sampled)
        result['basis'] = 'extrapolated_from_sampled_prefix'
    else:
        estimated_documents = float(effective_documents)
        result['basis'] = 'sampled_only'
    result['estimated_effective_documents'] = estimated_documents
    result['estimated_effective_tokens'] = mean_tokens * estimated_documents
    return result


def train_wrap_status(budget: dict | None, scheduled_tokens: int) -> dict:
    """Compare the planned training-token budget with the effective train split.

    `train_gpt2.py` restarts its loader on `StopIteration`, which is the right
    behaviour for an intentionally multi-epoch run but silently turns a
    "trained on N unique tokens" claim into a repeated pass when the split is
    smaller than the schedule.  The audit reports the coverage ratio and makes
    the wrap explicit; `--fail-on-train-wrap` decides whether it is fatal.
    """
    scheduled = (0 if scheduled_tokens is None else
                 _strict_int_option(scheduled_tokens, "scheduled_tokens", 0))
    if budget is not None and not isinstance(budget, dict):
        raise ValueError(
            f"budget must be a mapping or None, got {type(budget).__name__}")
    estimated = None if budget is None else budget.get('estimated_effective_tokens')
    if estimated is not None:
        if _is_boolean_scalar(estimated):
            raise ValueError(
                f"estimated_effective_tokens must be numeric or None, got {estimated!r}")
        try:
            estimated = float(estimated)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"estimated_effective_tokens must be numeric or None, got {estimated!r}") from None
        if not math.isfinite(estimated) or estimated < 0.0:
            raise ValueError(
                "estimated_effective_tokens must be finite and >= 0, "
                f"got {estimated!r}")
    result = {
        'scheduled_tokens': scheduled or None,
        'estimated_effective_tokens': estimated,
        'basis': None if budget is None else budget.get('basis'),
        'exhaustive': None if budget is None else budget.get('exhaustive'),
        'wrap_expected': None,
        'coverage_ratio': None,
    }
    if scheduled > 0 and estimated is not None:
        result['coverage_ratio'] = estimated / scheduled
        result['wrap_expected'] = estimated < scheduled
    return result


def _is_boolean_scalar(value) -> bool:
    """Python and NumPy booleans (``np.bool_`` is not a Python bool)."""
    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == 'numpy' and cls.__name__ in ('bool_', 'bool')


def _strict_int_option(value, name: str, minimum: int) -> int:
    """An integer-valued measurement, rejecting bools and fractional values.

    ``int(2.5)`` silently becomes 2; for a sequence length, batch count or
    evaluation cadence that changes the geometry being audited while the
    report still looks like it described the requested protocol. A bool is
    not a count either.
    """
    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be an integer, got bool {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if converted != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {converted}")
    return converted


def packed_window_count(tokens, seq_len: int) -> int | None:
    """Number of fixed-length LM windows a split with `tokens` tokens can emit.

    `PackedStream` advances by `seq_len` tokens per window and needs
    `seq_len + 1` tokens to produce the next label, so a split with T usable
    tokens yields ``floor((T - 1) / seq_len)`` windows.  Those are the *unique*
    held-out windows: an evaluation that asks for more batches than this
    necessarily revisits earlier windows.  `None` means the token count is
    unknown; 0 means no window can be formed.
    """
    if tokens is None:
        return None
    if _is_boolean_scalar(tokens):
        raise ValueError(f"token count must be numeric or None, got {tokens!r}")
    try:
        tokens = int(math.floor(float(tokens)))
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"token count must be numeric or None, got {tokens!r}") from None
    seq_len = _strict_int_option(seq_len, "seq_len", 1)
    if tokens < 1:
        return 0
    return (tokens - 1) // seq_len


def eval_batch_capacity(budget: dict | None, seq_len: int, batch_size: int) -> dict:
    """How many distinct evaluation batches the effective held-out tokens hold.

    The held-out token budget is a token count, while the evaluation protocol
    asks for a number of *batches*.  Converting one into the other makes the
    pseudo-replication visible: when the requested batch count is larger than
    the distinct capacity, a validation loader cycles and the same held-out
    tokens are scored more than once.  The result keeps the budget `basis`
    next to the capacity, because a capacity extrapolated from a sampled
    prefix is an estimate, not an exact count.
    """
    seq_len = _strict_int_option(seq_len, "seq_len", 1)
    batch_size = _strict_int_option(batch_size, "batch_size", 1)
    if budget is not None and not isinstance(budget, dict):
        raise ValueError(
            f"budget must be a mapping or None, got {type(budget).__name__}")
    budget = budget or {}
    tokens = budget.get('estimated_effective_tokens')
    if tokens is not None:
        # A bool is not a token measurement: float(True) would turn it into a
        # one-token budget and report a capacity for a value the audit never
        # measured (the same rule the sibling budget helpers enforce).
        if _is_boolean_scalar(tokens):
            raise ValueError(
                f"estimated_effective_tokens must be numeric or None, got {tokens!r}")
        try:
            tokens = float(tokens)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"estimated_effective_tokens must be numeric or None, got {tokens!r}") from None
        if not math.isfinite(tokens) or tokens < 0:
            raise ValueError(
                f"estimated_effective_tokens must be finite and >= 0, got {tokens!r}")
    windows = packed_window_count(tokens, seq_len)
    unique_batches = None if windows is None else windows // batch_size
    basis = budget.get('basis')
    return {
        'estimated_effective_tokens': tokens,
        'packed_windows': windows,
        'unique_batches': unique_batches,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'tokens_per_batch': seq_len * batch_size,
        'basis': basis,
        'exhaustive': budget.get('exhaustive'),
        'approximate': basis not in (None, 'observed'),
    }


def evaluation_reuse_status(capacity: dict | None, requested_batches: int, *,
                            split: str, val_every: int = 0,
                            max_steps: int = 0) -> dict:
    """Does one evaluation pass consume more batches than the split holds?

    Validation cycles its loader (`evaluate_heldout(..., cycle=True)`), so a
    pass that requests more batches than the split has distinct batches
    re-measures earlier held-out windows.  The per-batch standard error then
    treats repeated windows as independent observations, which understates
    the uncertainty.  The sealed test pass uses `cycle=False`, so it instead
    *truncates*: its mean rests on fewer batches than requested.

    `val_every` / `max_steps` additionally describe checkpoint selection:
    every validation pass looks at the same finite held-out set again, so
    the selection statistic is the maximum over many noisy draws on reused
    data.  `selection_reuse_factor` counts how many times each distinct
    held-out batch is evaluated across all passes (None when either the
    frequency or the batch capacity is unknown).
    """
    if split not in ('validation', 'test'):
        raise ValueError(f"split must be validation/test, got {split!r}")
    requested_batches = _strict_int_option(
        requested_batches, "requested_batches", 1)
    val_every = _strict_int_option(val_every, "val_every", 0)
    max_steps = _strict_int_option(max_steps, "max_steps", 0)
    if capacity is not None and not isinstance(capacity, dict):
        raise ValueError(
            f"capacity must be a mapping or None, got {type(capacity).__name__}")
    capacity = capacity or {}
    unique = capacity.get('unique_batches')
    if unique is not None:
        if isinstance(unique, bool) or not isinstance(unique, int) or unique < 0:
            raise ValueError(f"unique_batches must be a non-negative int or None, got {unique!r}")
    tokens = capacity.get('estimated_effective_tokens')
    tokens_per_batch = capacity.get('tokens_per_batch')
    if tokens is not None:
        if _is_boolean_scalar(tokens):
            raise ValueError(
                f"estimated_effective_tokens must be numeric or None, got {tokens!r}")
        try:
            tokens = float(tokens)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"estimated_effective_tokens must be numeric or None, got {tokens!r}") from None
        if not math.isfinite(tokens) or tokens < 0.0:
            raise ValueError(
                "estimated_effective_tokens must be finite and >= 0, "
                f"got {tokens!r}")
    if tokens_per_batch is not None:
        tokens_per_batch = _strict_int_option(
            tokens_per_batch, "tokens_per_batch", 1)
    distinct = None if unique is None else min(requested_batches, unique)
    planned_tokens = (None if tokens_per_batch is None
                      else requested_batches * tokens_per_batch)
    token_coverage = (
        None if tokens is None or not planned_tokens else tokens / planned_tokens)
    if split == 'validation':
        reuses = None if unique is None else requested_batches > unique
        cycled = requested_batches - unique if reuses else 0
        cycles = (None if unique in (None, 0)
                  else ((requested_batches + unique - 1) // unique if reuses else 1))
        truncated = False
        shortfall = 0
    else:
        # The sealed pass never cycles; it consumes the unique batches once.
        reuses = False
        cycled = 0
        cycles = None
        truncated = None if unique is None else requested_batches > unique
        shortfall = requested_batches - unique if truncated else 0
    passes = None
    if split == 'validation' and val_every > 0 and max_steps > 0:
        # The trainer validates when step % val_every == 0, so a run of
        # max_steps optimizer steps makes floor(max_steps / val_every) passes
        # (not ceil: a partial final interval gets no validation at its end).
        passes = max_steps // val_every
    selection_draws = None if passes is None else passes * requested_batches
    selection_reuse_factor = (
        None if selection_draws is None or not unique else selection_draws / unique)
    return {
        'split': split,
        'requested_batches': requested_batches,
        'unique_batches': unique,
        'distinct_batches_per_pass': distinct,
        'distinct_fraction': None if distinct is None else distinct / requested_batches,
        'reuses_batches': reuses,
        'cycled_batches': cycled,
        'cycles_per_pass': cycles,
        'truncated': truncated,
        'shortfall_batches': shortfall,
        'planned_eval_tokens': planned_tokens,
        'token_coverage_ratio': token_coverage,
        'validation_passes': passes,
        'selection_batch_draws': selection_draws,
        'selection_reuse_factor': selection_reuse_factor,
        'selection_reuses_batches': (
            None if selection_reuse_factor is None else selection_reuse_factor > 1.0),
        'basis': capacity.get('basis'),
        'approximate': bool(capacity.get('approximate', False)),
    }


def _chronology_problems(reports: dict) -> list:
    """Verify the enabled split reports tile the stream in time order.

    train must start at 0 and each later enabled split must start where the
    previous one ends; anything else means a document could be counted on two
    sides (or the split is no longer chronological). A disabled split
    (`--val-frac 0` / `--test-frac 0`) is simply absent.
    """
    names = [name for name in ('train', 'validation', 'test') if name in reports]
    active = [reports[name] for name in names]
    if not active:
        return []
    if not all(
            isinstance(r, dict) and isinstance(r.get('start'), int)
            and not isinstance(r.get('start'), bool)
            and isinstance(r.get('count'), int)
            and not isinstance(r.get('count'), bool)
            and r.get('start') >= 0 and r.get('count') >= 0 for r in active):
        # Train-only with count=None is the legitimate "held-out splits
        # disabled" shape. With more than one active split, missing
        # start/count means the tiling cannot be verified at all; reporting
        # that as clean would fail open. A malformed single report (negative,
        # boolean, or a non-zero start) is not the disabled-shape exception.
        if len(active) > 1:
            return ["cannot verify chronological split tiling: one or more active "
                    "split reports lack an integer start/count"]
        report = active[0]
        if isinstance(report, dict):
            start, count = report.get('start'), report.get('count')
            # Older disabled-split reports may omit the implicit train start
            # entirely; accept that legacy shape but not a false/negative one.
            if (names[0] == 'train'
                    and count is None
                    and (start is None
                         or (isinstance(start, int) and not isinstance(start, bool)
                             and start == 0))):
                return []
        return [f"cannot verify chronological split tiling: invalid start/count "
                f"in the active split report {report!r}"]
    ranges = [(name, r['start'], r['start'] + r['count'])
              for name, r in zip(names, active)]
    if ranges[0][1] != 0:
        return [f"split ranges are not contiguous chronological slices: {ranges}"]
    for (_, _, prev_end), (name, start, _) in zip(ranges, ranges[1:]):
        if start != prev_end:
            return [f"split ranges are not contiguous chronological slices: {ranges}"]
    raw_totals = [r.get('num_documents') for r in active
                  if r.get('num_documents') is not None]
    invalid_totals = [value for value in raw_totals
                      if isinstance(value, bool) or not isinstance(value, int)]
    if invalid_totals:
        return [f"num_documents {invalid_totals[0]!r} is not an integer document "
                "total; the split tiling cannot be verified"]
    known_totals = sorted(set(raw_totals))
    if len(known_totals) > 1:
        # Reports from different runs/streams can tile by accident while
        # describing different corpora; their counts are not comparable.
        return [f"split reports were computed against different document totals: "
                f"{known_totals}"]
    # Use the total from *any* active report that recorded one. Reading only
    # the first split let a train report with no total hide a later report
    # that did know the stream length, so a prefix tiling looked complete.
    total = known_totals[0] if known_totals else None
    last_name, _, last_end = ranges[-1]
    if total is not None and last_end != total:
        return [f"split ranges do not cover the full document stream: "
                f"{last_name} ends at {last_end} but num_documents={total}"]
    return []


def _write_hash_ledger(path, hashes: set, n_docs: int, split_name: str,
                       exhaustive: bool | None=None):
    """Write a sorted one-digest-per-line ledger and describe its coverage.

    `load_hash_file` accepts this whitespace-separated form (and also reads
    JSON-list files). With a bounded `--n-docs` sample the ledger only covers
    the prefix that was read; the returned info records `exhaustive=false` and
    the caller warns, instead of implying full coverage. `exhaustive` comes
    from the collector because a split shorter than `n_docs` is fully read even
    though the cap is nonzero; when omitted, fall back to the old n_docs rule.
    """
    if not path:
        return None
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, 'w', encoding='utf-8') as f:
        for h in sorted(hashes):
            f.write(h.hex() + '\n')
    if exhaustive is None:
        exhaustive = not n_docs
    info = {
        'path': str(ledger),
        'split': split_name,
        'n_hashes': len(hashes),
        'exhaustive': exhaustive,
        'n_docs_limit': n_docs,
    }
    print(f'[audit] wrote {len(hashes)} {split_name} hashes to {ledger}'
          f'{" (exhaustive)" if exhaustive else " (SAMPLED PREFIX ONLY)"}')
    if not exhaustive:
        print(f'[audit] WARNING: the ledger covers at most the first {n_docs} '
              f'{split_name} documents; consumed as a cross-split exclude ledger it '
              f'only drops duplicates of that prefix. Re-run with --n-docs 0 for a full ledger.',
              file=sys.stderr)
    return info


def _total_variation(a: Counter, b: Counter) -> float:
    total_a, total_b = sum(a.values()), sum(b.values())
    if not total_a or not total_b:
        # An empty histogram has no distribution to compare. Returning 0.0
        # would make a split with no usable documents look perfectly matched;
        # 1.0 is the maximal divergence and trips any finite TVD cap.
        return 1.0
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / total_a - b.get(k, 0) / total_b) for k in keys)


def _top_shifted(a: Counter, b: Counter, k=10):
    total_a, total_b = sum(a.values()), sum(b.values())
    if not total_a or not total_b:
        return []
    keys = set(a) | set(b)
    scored = ((abs(a.get(t, 0) / total_a - b.get(t, 0) / total_b), t) for t in keys)
    return [{'token_id': t, 'abs_shift': d} for d, t in sorted(scored, reverse=True)[:k]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--dataset-revision', default=None, help='dataset revision (commit hash) to audit; must match the trained run, because auditing a moved upstream branch checks different data than was trained on')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--tokenizer-revision', default=None, help='tokenizer revision (commit hash) to pin')
    p.add_argument('--val-frac', type=float, default=0.005, help='must match the trained run')
    p.add_argument('--test-frac', type=float, default=0.005, help='must match the trained run')
    p.add_argument('--num-documents', type=int, default=None, help='total documents; required when the dataset reports no size')
    p.add_argument('--n-docs', type=int, default=2000, help='documents sampled per split (0 = stream every document, needed for an exhaustive hash ledger)')
    p.add_argument('--min-doc-chars', type=int, default=0)
    p.add_argument('--max-doc-chars', type=int, default=0)
    p.add_argument('--no-dedup', dest='dedup', action='store_false')
    p.add_argument('--write-train-hash-file', default=None, help='write the hashes of the sampled training documents here; feed the file to train_gpt2.py --exclude-hash-file so validation/test drop exact duplicates of training documents (use --n-docs 0 for a full ledger)')
    p.add_argument('--write-val-hash-file', default=None, help='write the hashes of the sampled validation documents here; feed the file to train_gpt2.py --exclude-test-hash-file so the sealed test drops exact duplicates of validation documents (a val/test duplicate makes the selection split and the final split non-disjoint; use --n-docs 0 for a full ledger)')
    p.add_argument('--group-key', default=None, help='optional grouping-unit field (top-level name or dotted path such as "meta.url"); held-out documents whose group occurs in an earlier split are counted as leakage (same semantics as train_gpt2.py --group-key)')
    p.add_argument('--write-train-group-file', default=None, help='write the grouping-unit hashes of the sampled training documents here; feed the file to train_gpt2.py --exclude-group-file so validation/test drop every document whose group was seen in train (requires --group-key; use --n-docs 0 for a full ledger)')
    p.add_argument('--write-val-group-file', default=None, help='write the grouping-unit hashes of the sampled validation documents here; feed the file to train_gpt2.py --exclude-test-group-file so the sealed test drops every document whose group was seen in validation (requires --group-key; use --n-docs 0 for a full ledger)')
    p.add_argument('--max-tvd', type=float, default=None, help="fail (exit 1) when train-vs-validation token total-variation distance exceeds this")
    p.add_argument('--max-test-tvd', type=float, default=None, help="fail (exit 1) when train-vs-test token total-variation distance exceeds this; the test split backs the final reported number, so unchecked drift there changes what that number measures")
    p.add_argument('--max-val-test-tvd', type=float, default=None, help="fail (exit 1) when validation-vs-test token total-variation distance exceeds this; the model is selected on validation and reported on test, so drift between the two held-out slices makes the selected checkpoint a poorer fit for the number finally reported")
    p.add_argument('--no-near-dup', dest='near_dup', action='store_false', help='disable the token-shingle MinHash near-duplicate scan (exact/canonical hash checks still run)')
    p.add_argument('--near-dup-threshold', type=float, default=0.8, help='token-shingle Jaccard similarity at/above which a cross-split pair is reported as near-duplicate contamination (default 0.8)')
    p.add_argument('--near-dup-margin', type=float, default=0.05, help='report pairs whose Jaccard estimate is within this margin below the threshold as borderline (default 0.05); they are not counted as failures')
    p.add_argument('--near-dup-shingle', type=int, default=8, help='token n-gram size used by the near-duplicate MinHash sketch (default 8)')
    p.add_argument('--near-dup-max-docs', type=int, default=4000, help='maximum documents sketched per split; the scan is then a prefix and reports itself as a lower bound (default 4000)')
    p.add_argument('--near-dup-max-shingles', type=int, default=_NEAR_DUP_MAX_SHINGLES, help='bottom-k token shingles used per document (default %(default)s); bounds the sketch cost on very long documents')
    p.add_argument('--max-near-dup-rate', type=float, default=None, help='pre-registered tolerable fraction of held-out documents with a cross-split near-duplicate (e.g. 0.001); fail when the one-sided confidence upper bound on the scanned documents exceeds it (omit for report-only; requires --near-dup)')
    p.add_argument('--append-near-dup-to-ledgers', action='store_true', help='append the canonical digests of flagged held-out near-duplicates to the --write-train-hash-file / --write-val-hash-file ledgers, so the training run drops them from validation/test instead of only reporting them (requires --near-dup and at least one ledger output)')
    p.add_argument('--min-eval-tokens', type=int, default=0, help='fail when the estimated effective tokens of validation or test fall below this floor (0 = report only); the estimate is exact when --n-docs 0, otherwise extrapolated from the sampled prefix')
    p.add_argument('--min-eval-docs', type=int, default=0, help='fail when the estimated effective document count of validation or test falls below this floor (0 = report only)')
    p.add_argument('--scheduled-train-tokens', type=int, default=0, help='planned unique training tokens (max_steps * batch_size * grad_accum * seq_len); when > 0, compare it with the estimated effective train-split budget so a silently recycled training stream is visible')
    p.add_argument('--fail-on-train-wrap', action='store_true', help='fail the audit when the scheduled training tokens exceed the estimated effective train split (the loader would start a second pass over already-seen documents)')
    p.add_argument('--seq-len', type=int, default=1024, help='packed sequence length used by the training/evaluation loaders (for the held-out batch-budget check; must match the run)')
    p.add_argument('--eval-batch-size', type=int, default=32, help='per-device batch size used by evaluation (for the held-out batch-budget check; must match the run)')
    p.add_argument('--val-batches', type=int, default=10, help='batches per validation pass requested by the run (for the held-out batch-budget check)')
    p.add_argument('--test-batches', type=int, default=20, help='batches requested by the one-shot sealed test pass (for the held-out batch-budget check)')
    p.add_argument('--val-every', type=int, default=500, help='validation frequency in optimizer steps (0 = unknown); with --max-steps it counts how often the same held-out batches are reused for checkpoint selection')
    p.add_argument('--max-steps', type=int, default=0, help='planned optimizer steps (0 = unknown); with --val-every it counts checkpoint-selection passes over the held-out data')
    p.add_argument('--fail-on-eval-reuse', action='store_true', help='fail when a validation pass must cycle over fewer distinct held-out batches than requested (repeated batches are then treated as independent by the per-batch stderr)')
    p.add_argument('--fail-on-eval-shortfall', action='store_true', help='fail when the sealed test pass requests more distinct batches than its held-out split contains (the test mean would rest on fewer batches than advertised)')
    p.add_argument('--output', default='runs/data_audit.json')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this audit tests; recorded in the report and manifest so the result is auditable as a test of a claim rather than a fishing trip')
    args = p.parse_args()
    for flag, value in (('--n-docs', args.n_docs),
                        ('--min-doc-chars', args.min_doc_chars),
                        ('--max-doc-chars', args.max_doc_chars)):
        if value < 0:
            raise SystemExit(f'{flag} {value} must be >= 0')
    if args.max_doc_chars and args.min_doc_chars > args.max_doc_chars:
        raise SystemExit(
            f'--min-doc-chars {args.min_doc_chars} exceeds --max-doc-chars '
            f'{args.max_doc_chars}; no document could pass the cleaner')
    for flag, value in (('--val-frac', args.val_frac), ('--test-frac', args.test_frac)):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise SystemExit(f'{flag} {value} must be finite and in [0, 1)')
    if args.val_frac + args.test_frac >= 1.0:
        raise SystemExit(
            f'--val-frac {args.val_frac} + --test-frac {args.test_frac} leaves no '
            'training stream; the split fractions must sum to less than 1')
    for flag, value in (('--max-tvd', args.max_tvd),
                        ('--max-test-tvd', args.max_test_tvd),
                        ('--max-val-test-tvd', args.max_val_test_tvd)):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise SystemExit(f'{flag} {value} must be finite and >= 0')
    if not (math.isfinite(args.near_dup_threshold)
            and 0.0 < args.near_dup_threshold <= 1.0):
        raise SystemExit(f'--near-dup-threshold {args.near_dup_threshold} must be in (0, 1]')
    if not (math.isfinite(args.near_dup_margin)
            and 0.0 <= args.near_dup_margin < 1.0):
        raise SystemExit(f'--near-dup-margin {args.near_dup_margin} must be in [0, 1)')
    for flag, value, minimum in (('--near-dup-shingle', args.near_dup_shingle, 1),
                                 ('--near-dup-max-docs', args.near_dup_max_docs, 1),
                                 ('--near-dup-max-shingles', args.near_dup_max_shingles, 1),
                                 ('--min-eval-tokens', args.min_eval_tokens, 0),
                                 ('--min-eval-docs', args.min_eval_docs, 0),
                                 ('--seq-len', args.seq_len, 1),
                                 ('--eval-batch-size', args.eval_batch_size, 1),
                                 ('--val-batches', args.val_batches, 1),
                                 ('--test-batches', args.test_batches, 1),
                                 ('--val-every', args.val_every, 0),
                                 ('--max-steps', args.max_steps, 0),
                                 ('--scheduled-train-tokens', args.scheduled_train_tokens, 0)):
        if value < minimum:
            raise SystemExit(f'{flag} {value} must be >= {minimum}')
    if args.max_near_dup_rate is not None:
        if not (math.isfinite(args.max_near_dup_rate)
                and 0.0 <= args.max_near_dup_rate <= 1.0):
            raise SystemExit(
                f'--max-near-dup-rate {args.max_near_dup_rate} must be finite and in [0, 1]')
        if not args.near_dup:
            raise SystemExit('--max-near-dup-rate requires the near-duplicate scan (do not pass --no-near-dup)')
    if args.append_near_dup_to_ledgers:
        if not args.near_dup:
            raise SystemExit('--append-near-dup-to-ledgers requires the near-duplicate scan (do not pass --no-near-dup)')
        if not (args.write_train_hash_file or args.write_val_hash_file):
            raise SystemExit(
                '--append-near-dup-to-ledgers requires --write-train-hash-file '
                'and/or --write-val-hash-file: there is no ledger to append to')
    if (args.write_train_group_file or args.write_val_group_file) and not args.group_key:
        raise SystemExit('--write-*-group-file requires --group-key')
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(f'audit_data needs transformers + datasets installed: {e}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    options = {'val_frac': args.val_frac, 'test_frac': args.test_frac,
               'num_documents': args.num_documents, 'dedup': args.dedup,
               'min_doc_chars': args.min_doc_chars, 'max_doc_chars': args.max_doc_chars,
               'dataset_revision': args.dataset_revision,
               'tokenizer_revision': args.tokenizer_revision}
    splits = {}
    tokens = {}
    effective_tokens = {}
    reports = {}
    lengths = {}
    near_records = {}
    near_meta = {}
    train_hashes = None
    val_hashes = None
    train_groups = None
    val_groups = None
    train_ledger_info = None
    val_ledger_info = None
    train_group_ledger_info = None
    val_group_ledger_info = None
    # A zero held-out fraction disables that split in the data pipeline; audit
    # the splits the run actually has instead of requesting empty ones.
    active_splits = [name for name in ('train', 'validation', 'test')
                     if (name == 'train'
                         or (name == 'validation' and args.val_frac > 0.0)
                         or (name == 'test' and args.test_frac > 0.0))]
    for name in active_splits:
        # Train/validation hash and group sets are kept (the exclude ledgers
        # and the overlap reference); test only needs to be compared against
        # them.  Test is checked against train *and* validation: a val/test
        # duplicate makes the selection split and the final split non-disjoint.
        if name == 'train':
            exclude_hashes = exclude_groups = None
        elif name == 'validation':
            exclude_hashes = train_hashes
            exclude_groups = train_groups if args.group_key else None
        else:
            # The sealed test split is opened by the run with the training
            # ledger *and* the validation ledger, so its effective documents
            # are the ones that survive both.
            exclude_hashes = (set(train_hashes or ()) | set(val_hashes or ()))
            exclude_groups = ((set(train_groups or ()) | set(val_groups or ()))
                              if args.group_key else None)
        (stats, tok_counter, report, hashes, split_lengths, groups, tok_effective,
         split_near_records, split_near_meta) = _collect_split(
            name, args.dataset, args.dataset_config, tokenizer, options,
            args.n_docs, args.min_doc_chars, args.max_doc_chars, args.dedup,
            known_hashes=train_hashes,
            known_hashes_extra=(val_hashes if name == 'test' else None),
            known_hashes_exhaustive=(splits['train']['exhaustive']
                                     if name in ('validation', 'test') else True),
            known_hashes_extra_exhaustive=((splits.get('validation') or {})
                                           .get('exhaustive', True)
                                           if name == 'test' else True),
            collect_hashes=(name in ('train', 'validation')),
            group_key=args.group_key,
            known_groups=train_groups,
            known_groups_extra=(val_groups if name == 'test' else None),
            known_groups_exhaustive=(splits['train']['exhaustive']
                                     if name in ('validation', 'test') else True),
            known_groups_extra_exhaustive=((splits.get('validation') or {})
                                           .get('exhaustive', True)
                                           if name == 'test' else True),
            collect_groups=(name in ('train', 'validation')),
            exclude_hashes=exclude_hashes,
            exclude_group_hashes=exclude_groups,
            near_dup=args.near_dup,
            near_dup_shingle=args.near_dup_shingle,
            near_dup_max_docs=args.near_dup_max_docs,
            near_dup_max_shingles=args.near_dup_max_shingles)
        splits[name] = stats
        near_records[name] = split_near_records
        near_meta[name] = split_near_meta
        tokens[name] = tok_counter
        effective_tokens[name] = tok_effective
        reports[name] = report
        lengths[name] = split_lengths
        if name == 'train':
            train_hashes = hashes
            train_groups = groups
            train_ledger_info = _write_hash_ledger(
                args.write_train_hash_file, hashes, args.n_docs, 'train-document',
                exhaustive=stats['exhaustive'])
            train_group_ledger_info = _write_hash_ledger(
                args.write_train_group_file, groups, args.n_docs, 'train-group',
                exhaustive=stats['exhaustive'])
        elif name == 'validation':
            val_hashes = hashes
            val_groups = groups
            val_ledger_info = _write_hash_ledger(
                args.write_val_hash_file, hashes, args.n_docs, 'validation-document',
                exhaustive=stats['exhaustive'])
            val_group_ledger_info = _write_hash_ledger(
                args.write_val_group_file, groups, args.n_docs, 'validation-group',
                exhaustive=stats['exhaustive'])
    # Held-out length outliers are also counted under the fence derived from
    # the *training* lengths alone.  Outliers only ever get dropped by
    # --min/max-doc-chars in the pipeline, i.e. by a fixed policy; keeping the
    # train-derived counts here lets a reviewer read the held-out slices
    # against statistics they did not contribute to (no test-set tuning).
    train_fence = length_fence(lengths['train']) if lengths['train'] else None
    for name in ('validation', 'test'):
        if name not in splits:
            continue
        if train_fence is not None and lengths[name]:
            splits[name]['length_outliers_train_fence'] = sum(
                1 for v in lengths[name]
                if v < train_fence['lower'] or v > train_fence['upper'])
            splits[name]['length_outliers_train_fence_fraction'] = (
                splits[name]['length_outliers_train_fence'] / len(lengths[name]))
        else:
            splits[name]['length_outliers_train_fence'] = None
            splits[name]['length_outliers_train_fence_fraction'] = None
    chronology = _chronology_problems(reports)
    empty_token_splits = [name for name in active_splits
                          if not sum(tokens[name].values())]

    def _tvd_pair(a, b, effective=False):
        if a not in tokens or b not in tokens:
            return None
        source = effective_tokens if effective else tokens
        return _total_variation(source[a], source[b])

    def _top_pair(a, b):
        if a not in tokens or b not in tokens:
            return None
        return _top_shifted(tokens[a], tokens[b])

    shift = {
        'train_vs_validation_tvd': _tvd_pair('train', 'validation'),
        'train_vs_test_tvd': _tvd_pair('train', 'test'),
        'validation_vs_test_tvd': _tvd_pair('validation', 'test'),
        # The same distances over the documents that survive the cross-split
        # ledgers: that is the distribution the trained model is actually
        # selected and reported on, and it can differ from the raw one when a
        # ledger is non-trivial.
        'train_vs_validation_tvd_effective': _tvd_pair('train', 'validation', effective=True),
        'train_vs_test_tvd_effective': _tvd_pair('train', 'test', effective=True),
        'validation_vs_test_tvd_effective': _tvd_pair('validation', 'test', effective=True),
        'top_shifted_train_vs_validation': _top_pair('train', 'validation'),
        'doc_chars_mean_delta_train_validation': (
            None if ('validation' not in splits
                     or splits['train']['doc_chars_mean'] is None
                     or splits['validation']['doc_chars_mean'] is None)
            else splits['validation']['doc_chars_mean'] - splits['train']['doc_chars_mean']),
    }
    # Data-adequacy: how many effective held-out tokens actually back the
    # validation/test numbers, and how many independent documents a selection
    # pass can draw on.  `val_frac` alone is a document fraction, not a token
    # budget, so it is reported separately for every held-out split.
    token_budgets = {name: split_token_budget(splits[name], reports[name])
                     for name in ('validation', 'test') if name in splits}
    train_budget = (split_token_budget(splits['train'], reports['train'])
                    if 'train' in splits else None)
    scheduled_train_tokens = int(args.scheduled_train_tokens or 0)
    train_wrap = train_wrap_status(train_budget, scheduled_train_tokens)
    # Evaluation-budget: the held-out token budget is not directly the number
    # of evaluation batches the protocol will see.  Convert it through the
    # packed-window geometry and compare with the requested pass size; this is
    # where a validation loader that cycles (and therefore double-counts the
    # same held-out windows in its per-batch stderr) becomes visible.
    held_out_eval = {}
    for name, requested in (('validation', args.val_batches),
                            ('test', args.test_batches)):
        if name not in token_budgets:
            continue
        capacity = eval_batch_capacity(token_budgets[name], args.seq_len,
                                       args.eval_batch_size)
        held_out_eval[name] = {
            'capacity': capacity,
            'pass': evaluation_reuse_status(
                capacity, requested, split=name,
                val_every=args.val_every, max_steps=args.max_steps),
        }
    if args.near_dup:
        near_dup_report = near_duplicate_leakage(
            near_records, threshold=args.near_dup_threshold,
            margin=args.near_dup_margin,
            shingle_size=args.near_dup_shingle,
            exhaustive_by_split={name: bool(splits[name].get('exhaustive'))
                                 for name in active_splits},
            truncated_by_split={name: bool(near_meta[name].get('truncated'))
                                for name in active_splits})
        near_dup_report['scan_by_split'] = near_meta
    else:
        near_dup_report = {
            'enabled': False,
            'threshold_jaccard': args.near_dup_threshold,
            'borderline_margin': args.near_dup_margin,
            'shingle_size': args.near_dup_shingle,
            'pairs': {},
            'problems': [],
            'notes': ['the token-shingle near-duplicate scan was disabled by --no-near-dup'],
            'exhaustive': False,
            'lower_bound': False,
            'scan_by_split': near_meta,
        }
    near_dup_bound = near_duplicate_scan_bound(
        near_dup_report, max_rate=args.max_near_dup_rate)
    near_dup_report['scan_bound'] = near_dup_bound
    if args.append_near_dup_to_ledgers:
        additions = near_duplicate_ledger_additions(near_dup_report)
        coverage = ('audited_split' if near_dup_report.get('exhaustive')
                    else 'scanned_prefix')
        for role, path, base_hashes, info, label in (
                ('train', args.write_train_hash_file, train_hashes,
                 train_ledger_info, 'train-document'),
                ('validation', args.write_val_hash_file, val_hashes,
                 val_ledger_info, 'validation-document')):
            if not path or info is None:
                continue
            base = set(base_hashes or ())
            extra = set(additions['hashes'].get(role) or ())
            added = extra - base
            if added:
                info = _write_hash_ledger(
                    path, base | extra, args.n_docs, label,
                    exhaustive=info.get('exhaustive'))
            info['near_duplicate_hashes_added'] = len(added)
            info['near_duplicate_coverage'] = coverage
            info['near_duplicate_missing_digest_documents'] = int(
                additions['missing_digest_documents'])
            if role == 'train':
                train_ledger_info = info
            else:
                val_ledger_info = info
        if additions['missing_digest_documents']:
            near_dup_report['notes'].append(
                f"{additions['missing_digest_documents']} flagged held-out "
                "document(s) had no usable canonical digest and were not added "
                "to the exclusion ledger")
    out = {'options': vars(args), 'hypothesis': args.hypothesis, 'run_metadata': collect_run_metadata(),
           'split_reports': reports, 'split_chronology_problems': chronology,
           'cleaning_and_sanity': splits, 'distribution_shift': shift,
           'held_out_token_budget': token_budgets,
           'held_out_eval_budget': held_out_eval,
           'train_token_budget': train_budget,
           'train_stream_wrap': train_wrap,
           'near_duplicate_leakage': near_dup_report,
           'train_hash_ledger': train_ledger_info,
           'validation_hash_ledger': val_ledger_info,
           'train_group_ledger': train_group_ledger_info,
           'validation_group_ledger': val_group_ledger_info,
           'train_length_fence': train_fence}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    for name, stats in splits.items():
        line = (f"[audit] {name}: sampled={stats['documents_sampled']} kept={stats.get('documents_kept', 0)} "
                f"dropped_empty={stats.get('dropped_empty', 0)} dropped_missing={stats.get('dropped_missing_text', 0)} "
                f"dropped_duplicate={stats.get('dropped_duplicate', 0)} overlap_with_train={stats['duplicate_overlap_with_train']} "
                f"overlap_with_validation={stats['duplicate_overlap_with_validation']}")
        if name != 'train':
            line += (f" effective_kept={stats['effective_documents_kept']} "
                     f"ledger_dropped={stats['ledger_documents_dropped']}")
        if args.group_key:
            line += (f" group_overlap_with_train={stats.get('group_overlap_with_train')} "
                     f"group_overlap_with_validation={stats.get('group_overlap_with_validation')}")
        print(line)
    if args.group_key:
        missing_groups = sum(int(splits[name].get('group_value_missing') or 0)
                             for name in active_splits)
        if missing_groups:
            print(f"[audit] NOTE: {missing_groups} sampled documents have no '{args.group_key}' value; "
                  "they cannot be tied to a group and are checked for document duplicates only")
    def _fmt_tvd(value):
        return 'n/a' if value is None else f'{value:.4f}'

    print(f"[audit] token TVD train-vs-val={_fmt_tvd(shift['train_vs_validation_tvd'])} "
          f"train-vs-test={_fmt_tvd(shift['train_vs_test_tvd'])} "
          f"val-vs-test={_fmt_tvd(shift['validation_vs_test_tvd'])}")
    print(f"[audit] token TVD after ledgers train-vs-val={_fmt_tvd(shift['train_vs_validation_tvd_effective'])} "
          f"train-vs-test={_fmt_tvd(shift['train_vs_test_tvd_effective'])} "
          f"val-vs-test={_fmt_tvd(shift['validation_vs_test_tvd_effective'])}")
    def _fmt_budget(value):
        return 'n/a' if value is None else f'{value:.0f}'

    if train_budget is not None:
        print(f"[audit] train token budget: ~"
              f"{_fmt_budget(train_budget['estimated_effective_tokens'])} effective tokens over ~"
              f"{_fmt_budget(train_budget['estimated_effective_documents'])} documents "
              f"({train_budget['basis']}; {train_budget['sampled_effective_tokens']} sampled)")
        if train_wrap['wrap_expected']:
            print(f"[audit] NOTE: the scheduled {scheduled_train_tokens} training tokens exceed "
                  f"the estimated ~{_fmt_budget(train_wrap['estimated_effective_tokens'])} "
                  f"effective train tokens (coverage {train_wrap['coverage_ratio']:.2f}x); "
                  "the loader will start another pass and reuse documents, so the run is "
                  "not a single-epoch token budget")
    for name, budget in token_budgets.items():
        print(f"[audit] {name} token budget: ~"
              f"{_fmt_budget(budget['estimated_effective_tokens'])} effective tokens over ~"
              f"{_fmt_budget(budget['estimated_effective_documents'])} documents "
              f"({budget['basis']}; {budget['sampled_effective_tokens']} sampled)")
    for name, block in held_out_eval.items():
        capacity, status = block['capacity'], block['pass']
        unique = capacity['unique_batches']
        if unique is None:
            print(f"[audit] {name} eval budget: unknown (the held-out token count "
                  f"could not be estimated), so a {status['requested_batches']}-batch "
                  f"pass cannot be checked against the split capacity")
            continue
        mark = '~' if capacity['approximate'] else ''
        coverage = status['distinct_fraction']
        print(f"[audit] {name} eval budget: {mark}{unique} distinct batch(es) of "
              f"{capacity['batch_size']}x{capacity['seq_len']} tokens for "
              f"{status['requested_batches']} requested "
              f"({coverage:.0%} of the request is unique data)")
        if status['reuses_batches']:
            print(f"[audit] NOTE: the {name} loader only holds {mark}{unique} distinct "
                  f"batch(es), so a {status['requested_batches']}-batch pass cycles the "
                  f"loader ({status['cycles_per_pass']} pass(es) over the same data); "
                  "the per-batch stderr then treats repeated windows as independent")
        if status['truncated']:
            print(f"[audit] NOTE: the sealed test pass requests "
                  f"{status['requested_batches']} batches but only {mark}{unique} are "
                  f"available; test_loss will be computed over {unique} batch(es), a "
                  f"shortfall of {status['shortfall_batches']}")
        factor = status['selection_reuse_factor']
        if factor is not None and factor > 1.0:
            print(f"[audit] NOTE: {status['validation_passes']} validation pass(es) x "
                  f"{status['requested_batches']} batches look at {mark}{unique} distinct "
                  f"batch(es); each held-out batch is evaluated ~{factor:.1f} times during "
                  "checkpoint selection, so the selected minimum is a noisy maximum over "
                  "reused data")
    if near_dup_report['enabled']:
        if near_dup_report['pairs']:
            for key, pair in near_dup_report['pairs'].items():
                print(f"[audit] near-dup {key}: {pair['flagged_later_documents']} later doc(s) "
                      f"at Jaccard >= {near_dup_report['threshold_jaccard']} "
                      f"({pair['candidate_pairs']} LSH candidate pair(s), "
                      f"{pair['borderline_pairs']} borderline)")
        else:
            print(f"[audit] near-dup scan: no cross-split pair at Jaccard >= "
                  f"{near_dup_report['threshold_jaccard']} in the sketched documents")
        for later, bound in near_dup_report['scan_bound']['by_later_split'].items():
            upper = bound['upper_bound']
            observed = bound['observed_rate']
            if upper is None:
                print(f"[audit] near-dup bound {later}: no document was sketched, "
                      "so no contamination rate can be bounded")
                continue
            print(f"[audit] near-dup bound {later}: {bound['flagged_documents']}/"
                  f"{bound['documents']} flagged (observed {observed:.4%}), "
                  f"one-sided {bound['confidence']:.1%} upper bound {upper:.4%} "
                  f"on the {bound['scope'].replace('_', ' ')}")
        for info_name, info in (('train', train_ledger_info),
                                ('validation', val_ledger_info)):
            if info and info.get('near_duplicate_hashes_added') is not None:
                print(f"[audit] {info_name} ledger: +"
                      f"{info['near_duplicate_hashes_added']} near-duplicate "
                      f"exclusion(s) ({info.get('near_duplicate_coverage')})")
        for note in near_dup_report['notes']:
            print(f"[audit] NOTE: {note}")
    if args.hypothesis:
        print(f"[audit][hypothesis] {args.hypothesis}")
    print(f"[audit] saved to {args.output}")
    problems = list(chronology)
    if empty_token_splits:
        problems.append('no usable tokens could be sampled from: ' + ', '.join(empty_token_splits)
                        + ' (an empty histogram makes TVD undefined, not zero)')
    for name in active_splits:
        stats = splits[name]
        if stats['length_outliers_3iqr']:
            print(f"[audit] NOTE: {name}: {stats['length_outliers_3iqr']} length outliers beyond the 3xIQR fence "
                  f"(q1={stats['doc_chars_q1']}, q3={stats['doc_chars_q3']}, max={stats['doc_chars_max']}); "
                  "use --min-doc-chars/--max-doc-chars in the training run to drop them")
        if stats.get('dedup_cap_reached'):
            print(f"[audit] NOTE: {name}: the in-split duplicate hash set reached dedup_max during the "
                  f"sample, so later complete-duplicate documents in this split were kept; the "
                  "guarantee degrades silently at that point (the cross-split ledger is unaffected)")
        train_fence_outliers = stats.get('length_outliers_train_fence')
        if train_fence_outliers:
            print(f"[audit] NOTE: {name}: {train_fence_outliers} documents "
                  f"({stats['length_outliers_train_fence_fraction']:.4%}) fall outside the 3xIQR fence "
                  f"[{train_fence['lower']}, {train_fence['upper']}] derived from the training lengths alone")
    # Every split is audited, test included: the final numbers come from test,
    # so corruption there must fail the audit just like corruption in train.
    non_finite = [name for name in active_splits
                  if not splits[name]['finite_lengths']]
    if non_finite:
        problems.append(f"non-finite document-length statistics (NaN/Inf in the data) in: {', '.join(non_finite)}")
    if any((splits[name]['duplicate_overlap_with_train'] or 0) > 0
           for name in ('validation', 'test') if name in splits):
        problems.append('duplicate documents (exact or canonical text) appear in both train and a held-out split (duplicate leakage)')
    if ('test' in splits
            and (splits['test']['duplicate_overlap_with_validation'] or 0) > 0):
        problems.append('duplicate documents (exact or canonical text) appear in both validation and test (the selection split and the sealed final split are not disjoint)')
    if args.group_key:
        straddling = []
        for name in ('validation', 'test'):
            if name not in splits:
                continue
            n_groups = splits[name].get('group_overlap_with_train_groups') or 0
            if n_groups:
                straddling.append(f"{name} shares {n_groups} group(s) with train")
        n_groups = (splits.get('test') or {}).get('group_overlap_with_validation_groups') or 0
        if n_groups:
            straddling.append(f"test shares {n_groups} group(s) with validation")
        if straddling:
            problems.append(f"grouping unit '{args.group_key}' straddles a split boundary: " + '; '.join(straddling) + ' (group leakage)')
    # A ledger written from a sampled prefix only covers that prefix; a run
    # consuming it drops duplicates/groups of the prefix and silently keeps
    # the rest, so the audit must not bless it as a verified split guarantee.
    for info in (train_ledger_info, val_ledger_info,
                 train_group_ledger_info, val_group_ledger_info):
        if info and not info.get('exhaustive'):
            problems.append(
                f"the {info['split']} ledger {info['path']} covers only the first "
                f"{info['n_docs_limit']} documents of its split, so a run consuming it "
                f"still accepts cross-split duplicates/groups beyond that prefix "
                f"(re-run with --n-docs 0 for a full ledger)")
    sampled_short = []
    for name, stats in splits.items():
        report_count = (reports.get(name) or {}).get('count')
        # A bounded `--n-docs` sample is an intentional prefix, not a stream
        # that ended early: only a naturally exhausted stream that is shorter
        # than the nominal split size makes the audit non-exhaustive.
        if (isinstance(report_count, int) and not isinstance(report_count, bool)
                and stats.get('exhaustive', True)
                and stats.get('documents_sampled', 0) < report_count):
            sampled_short.append(
                f"{name}: sampled {stats.get('documents_sampled')} of {report_count}")
    if sampled_short:
        problems.append(
            'the document stream ended before the nominal split size, so the audit '
            'covered only a prefix and cannot claim exhaustive coverage (' +
            '; '.join(sampled_short) + ')')
    empty_effective = [name for name in ('validation', 'test')
                       if (name in splits
                           and ((splits[name].get('documents_kept')
                                 and not splits[name].get('effective_documents_kept'))
                                or not sum(effective_tokens[name].values())))]
    if empty_effective:
        problems.append(
            'the cross-split ledgers drop every kept document of: '
            + ', '.join(empty_effective)
            + ' (the effective held-out split is empty; the run would evaluate on nothing)')
    out_of_range = [name for name in active_splits
                    if splits[name]['token_ids_out_of_range']
                    or splits[name].get('token_ids_out_of_range_effective')]
    if out_of_range:
        problems.append(f"token ids outside the tokenizer vocabulary range in: {', '.join(out_of_range)}")
    if (args.max_tvd is not None and shift['train_vs_validation_tvd'] is not None
            and shift['train_vs_validation_tvd'] > args.max_tvd):
        problems.append(f"train-vs-validation distribution shift TVD={shift['train_vs_validation_tvd']:.4f} > max-tvd={args.max_tvd}")
    if (args.max_test_tvd is not None and shift['train_vs_test_tvd'] is not None
            and shift['train_vs_test_tvd'] > args.max_test_tvd):
        problems.append(f"train-vs-test distribution shift TVD={shift['train_vs_test_tvd']:.4f} > max-test-tvd={args.max_test_tvd}")
    if (args.max_val_test_tvd is not None and shift['validation_vs_test_tvd'] is not None
            and shift['validation_vs_test_tvd'] > args.max_val_test_tvd):
        problems.append(f"validation-vs-test distribution shift TVD={shift['validation_vs_test_tvd']:.4f} > max-val-test-tvd={args.max_val_test_tvd}")
    for flag, cap, key, label in (
            ('--max-tvd', args.max_tvd, 'train_vs_validation_tvd_effective',
             'train-vs-validation'),
            ('--max-test-tvd', args.max_test_tvd, 'train_vs_test_tvd_effective',
             'train-vs-test'),
            ('--max-val-test-tvd', args.max_val_test_tvd, 'validation_vs_test_tvd_effective',
             'validation-vs-test')):
        value = shift.get(key)
        if cap is not None and value is not None and value > cap:
            problems.append(
                f"{label} distribution shift TVD={value:.4f} after the cross-split "
                f"ledgers > {flag}={cap}; the effective held-out distribution is the "
                "one the model is actually selected and reported on")
    # Data adequacy: a held-out split can be perfectly disjoint and still be
    # too small to select a checkpoint or to support the claimed test. Report
    # the estimated effective token count and, when the caller sets a floor,
    # fail instead of letting a tiny slice masquerade as a precise estimate.
    for name, budget in token_budgets.items():
        estimated_tokens = budget.get('estimated_effective_tokens')
        estimated_docs = budget.get('estimated_effective_documents')
        if estimated_tokens is not None and estimated_tokens < 100_000:
            print(f"[audit] NOTE: {name} has only ~{estimated_tokens:.0f} effective tokens "
                  f"({budget['basis']}); loss/PPL confidence intervals and paired tests "
                  "over so few tokens will be wide, and a validation pass may revisit "
                  "the same held-out batches")
        if args.min_eval_tokens and (estimated_tokens is None
                                     or estimated_tokens < args.min_eval_tokens):
            problems.append(
                f"{name} holds only ~"
                f"{'0' if estimated_tokens is None else f'{estimated_tokens:.0f}'} estimated "
                f"effective tokens, below --min-eval-tokens={args.min_eval_tokens} "
                f"({budget['basis']}); the held-out estimate is too small to support "
                "the comparison")
        if args.min_eval_docs and (estimated_docs is None
                                   or estimated_docs < args.min_eval_docs):
            problems.append(
                f"{name} holds only ~"
                f"{'0' if estimated_docs is None else f'{estimated_docs:.0f}'} estimated "
                f"effective documents, below --min-eval-docs={args.min_eval_docs} "
                f"({budget['basis']})")
    if args.fail_on_train_wrap and train_wrap['wrap_expected']:
        problems.append(
            f"the scheduled {scheduled_train_tokens} training tokens exceed the estimated "
            f"~{train_wrap['estimated_effective_tokens']:.0f} effective train tokens "
            f"(coverage {train_wrap['coverage_ratio']:.2f}x); the training loader would "
            "reuse documents in a second pass instead of consuming unique data")
    validation_eval = (held_out_eval.get('validation') or {}).get('pass') or {}
    test_eval = (held_out_eval.get('test') or {}).get('pass') or {}
    if args.fail_on_eval_reuse and validation_eval.get('reuses_batches'):
        problems.append(
            f"the validation pass requests {validation_eval['requested_batches']} "
            f"batches but the split holds only {validation_eval['unique_batches']} "
            f"distinct batch(es) ({validation_eval['distinct_fraction']:.0%} unique); "
            "the loader would cycle and the per-batch stderr would treat repeated "
            "held-out windows as independent observations. Lower --val-batches, "
            "raise the validation token budget, or pass --fail-on-eval-reuse only "
            "after fixing the split")
    if args.fail_on_eval_shortfall and test_eval.get('truncated'):
        problems.append(
            f"the sealed test pass requests {test_eval['requested_batches']} batches "
            f"but the split holds only {test_eval['unique_batches']} distinct batch(es), "
            f"so the test mean would rest on {test_eval['unique_batches']} batch(es) - "
            f"a shortfall of {test_eval['shortfall_batches']}. Lower --test-batches or "
            "raise the test token budget so the advertised interval is the one reported")
    problems.extend(near_dup_report.get('problems') or [])
    problems.extend((near_dup_report.get('scan_bound') or {}).get('problems') or [])
    # The report itself must carry the verdict. It was written before the
    # leakage/sanity checks ran, so a failed audit's data_audit.json used to
    # look like a clean report with no problems while only the separate
    # manifest said "failed". Rewrite it now that the complete list exists.
    out['problems'] = problems
    out['status'] = 'failed' if problems else 'finished'
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    manifest = {
        'run_metadata': out['run_metadata'],
        'hypothesis': args.hypothesis,
        'dataset': args.dataset,
        'dataset_config': args.dataset_config,
        'tokenizer': args.tokenizer,
        'options': vars(args),
        'split_reports': reports,
        'held_out_token_budget': token_budgets,
        'held_out_eval_budget': held_out_eval,
        'train_token_budget': train_budget,
        'train_stream_wrap': train_wrap,
        'near_duplicate_leakage': near_dup_report,
        'train_hash_ledger': train_ledger_info,
        'validation_hash_ledger': val_ledger_info,
        'train_group_ledger': train_group_ledger_info,
        'validation_group_ledger': val_group_ledger_info,
        'train_length_fence': train_fence,
        'problems': problems,
    }
    if problems:
        # The SystemExit below reports leakage/sanity problems; the manifest
        # must not claim a clean, finished audit.
        manifest['status'] = 'failed'
    else:
        manifest = finish_metadata(manifest, artifacts={'audit_report': file_fingerprint(args.output)})
    save_run_manifest(Path(args.output).with_name('data_manifest.json'), manifest)
    if problems:
        for problem in problems:
            print(f'[audit] PROBLEM: {problem}', file=sys.stderr)
        raise SystemExit(1)
    print('[audit] no leakage or sanity problems found in the sampled splits')


if __name__ == '__main__':
    main()
