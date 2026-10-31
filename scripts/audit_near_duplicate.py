from __future__ import annotations
import argparse
import json
import math
import numbers
import sys
from array import array
from collections import Counter
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import wilson_interval
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest
from scripts.data_pipeline import (DocumentCleaner, _is_token_row,
                                   canonical_document_hash, document_hash,
                                   extract_group_value, group_hash, load_document_stream)
from scripts.audit_validation import (_is_boolean_scalar,
                                        _strict_bool_option,
                                        _strict_int_option)

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
    k = _strict_int_option(shingle_size, "shingle_size", 1)
    # Exact integer ids only: `int(1.9)`/`int("1")` would silently sketch a
    # different document than the counter pass, which rejects malformed ids.
    raw_tokens = list(ids)
    if any(not isinstance(token, numbers.Integral)
           or isinstance(token, bool) for token in raw_tokens):
        raise ValueError(
            'token_shingles expects exact integer token ids; a float/string/bool '
            'would be silently truncated')
    tokens = [int(token) + 1 for token in raw_tokens]
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
    # `ids` is documented as a token stream; materialize once so a generator
    # can be both sketched and counted.
    ids = list(ids)
    shingles = token_shingles(ids, shingle_size)
    total_shingles = len(shingles)
    limit = _strict_int_option(max_shingles, "max_shingles", 1)
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
        if (not isinstance(canonical_digest, (bytes, bytearray))
                or len(canonical_digest) != 16):
            raise ValueError(
                "canonical_digest must be the 16-byte blake2b digest that "
                f"DocumentCleaner looks up, got {canonical_digest!r}")
        record["canonical_digest"] = bytes(canonical_digest).hex()
    return record


def _band_key(signature, band: int):
    start = band * _NEAR_DUP_ROWS
    return (band,) + tuple(signature[start:start + _NEAR_DUP_ROWS])


def _has_shingle_sketch(record: dict) -> bool:
    """Whether a near-duplicate record carries at least one shingle.

    `minhash_signature(set())` is the all-ones sentinel. Two token-less
    documents therefore share that signature and used to be reported as a
    near-duplicate pair (estimated Jaccard 1.0), which is a false
    contamination finding. An absent `n_shingles` field is checked against the
    signature itself so hand-built records are handled too.
    """
    if not isinstance(record, dict):
        return False
    n = record.get('n_shingles')
    if isinstance(n, int) and not isinstance(n, bool):
        return n > 0
    signature = record.get('signature')
    if not signature:
        return False
    try:
        return any(int(value) != _NEAR_DUP_MASK for value in signature)
    except (TypeError, ValueError, OverflowError):
        return False


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
    shingle_size = _strict_int_option(shingle_size, "shingle_size", 1)
    max_examples = _strict_int_option(max_examples, "max_examples", 0)
    if (exhaustive_by_split is not None
            and not isinstance(exhaustive_by_split, dict)):
        raise ValueError("exhaustive_by_split must be a mapping or None")
    if (truncated_by_split is not None
            and not isinstance(truncated_by_split, dict)):
        raise ValueError("truncated_by_split must be a mapping or None")
    screening = max(0.0, float(threshold) - margin)
    ordered = [name for name in ("train", "validation", "test")
               if records_by_split.get(name)]
    exhaustive_by_split = exhaustive_by_split or {}
    truncated_by_split = truncated_by_split or {}
    # A truthy string ("false") would silently mark the scan exhaustive and
    # let a sampled prefix certify the population. Validate the coverage flags
    # instead of coercing whatever the report happens to contain.
    for label, mapping in (("exhaustive_by_split", exhaustive_by_split),
                           ("truncated_by_split", truncated_by_split)):
        for name, value in mapping.items():
            _strict_bool_option(value, f"{label}[{name!r}]", allow_none=True)
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
        # Skip token-less documents: their all-ones sentinel signature is not
        # evidence that two empty documents are copies of each other.
        earlier_positions = [position for position, record in enumerate(earlier_records)
                             if _has_shingle_sketch(record)]
        earlier_sigs = np.asarray(
            [earlier_records[position]["signature"] for position in earlier_positions],
            dtype=np.uint64)
        index: dict = {}
        for signature_position, doc_i in enumerate(earlier_positions):
            signature = earlier_records[doc_i]
            for band in range(_NEAR_DUP_BANDS):
                index.setdefault(_band_key(signature["signature"], band), []).append(signature_position)
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
                if not _has_shingle_sketch(later_record):
                    continue
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
                        earlier_i = earlier_positions[int(candidate_ids[sim_pos])]
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
                    earlier_i = earlier_positions[int(candidate_ids[sim_pos])]
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


def within_split_near_duplicates(records, *, threshold: float = 0.8,
                                 margin: float = 0.05, shingle_size: int = 8,
                                 max_examples: int = 20,
                                 exhaustive: bool = True,
                                 truncated: bool = False) -> dict:
    """Find later documents that near-duplicate an earlier document in one split.

    Cross-split leakage is only one way duplication enters an experiment.
    A training corpus can contain the same content several times (mirrored
    pages, reworded templates, abridged copies), and those repetitions are
    not cross-split contamination.  They still change what a "trained on N
    tokens" claim means: repeated content receives extra optimisation
    weight, and the effective number of distinct training tokens is smaller
    than the packed-stream budget.  This function quantifies that redundancy
    on the *training* side with the same deterministic MinHash/LSH machinery
    used for cross-split leakage.

    Documents are compared only with documents that occur earlier in the
    stream.  The later copy is the one reported (and can be excluded from
    training), so the first occurrence is always kept, matching the
    chronological, order-preserving split policy.  The scan is a diagnostic:
    finding redundancy is not by itself a leakage failure, but the returned
    ``flagged_tokens`` / ``total_tokens`` ratio feeds an honest unique-token
    budget and a pre-registered ``--max-train-near-dup-rate`` gate.
    """
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
    shingle_size = _strict_int_option(shingle_size, "shingle_size", 1)
    max_examples = _strict_int_option(max_examples, "max_examples", 0)
    exhaustive = bool(_strict_bool_option(exhaustive, "exhaustive"))
    truncated = bool(_strict_bool_option(truncated, "truncated"))
    try:
        records = list(records)
    except TypeError:
        raise ValueError(
            f"records must be an iterable of near-duplicate records, got "
            f"{type(records).__name__}") from None

    def _tokens(record) -> int:
        value = record.get("n_tokens") if isinstance(record, dict) else None
        if (isinstance(value, numbers.Integral)
                and not isinstance(value, bool) and value > 0):
            return int(value)
        return 0

    signatures = np.zeros((len(records), _NEAR_DUP_PERMUTATIONS),
                          dtype=np.uint64)
    has_sketch = []
    for position, record in enumerate(records):
        if not _has_shingle_sketch(record):
            continue
        signature = record.get("signature")
        try:
            values = np.asarray(signature, dtype=np.uint64)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(
                f"record {position} has a non-numeric MinHash signature") from None
        if values.shape != (_NEAR_DUP_PERMUTATIONS,):
            raise ValueError(
                f"record {position} has {values.size} MinHash values; expected "
                f"{_NEAR_DUP_PERMUTATIONS}")
        signatures[position] = values
        has_sketch.append(position)

    screening = max(0.0, threshold - margin)
    index: dict = {}
    earlier_doc_indices = np.empty(len(has_sketch), dtype=np.int64)
    n_earlier = 0
    flagged_docs: set = set()
    flagged_digests: set = set()
    missing_digest_positions: set = set()
    borderline_docs: set = set()
    examples = []
    borderline_examples = []
    candidate_pairs = 0
    flagged_pairs = 0
    borderline_pairs = 0
    total_tokens = sum(_tokens(record) for record in records)

    for position in has_sketch:
        record = records[position]
        candidates = set()
        for band in range(_NEAR_DUP_BANDS):
            candidates.update(index.get(
                _band_key(record["signature"], band), ()))
        if candidates:
            candidate_pairs += len(candidates)
            candidate_ids = np.fromiter(sorted(candidates), dtype=np.int64,
                                        count=len(candidates))
            earlier_positions = earlier_doc_indices[:n_earlier][candidate_ids]
            sims = (signatures[earlier_positions] == signatures[position]).mean(axis=1)
            hits = np.flatnonzero(sims >= float(threshold))
            borderline = np.flatnonzero((sims >= screening)
                                        & (sims < float(threshold)))
            if borderline.size:
                borderline_pairs += int(borderline.size)
                borderline_docs.add(position)
                if len(borderline_examples) < max_examples:
                    best_pos = int(np.argmax(sims[borderline]))
                    sim_pos = int(borderline[best_pos])
                    earlier_i = int(earlier_positions[sim_pos])
                    borderline_examples.append({
                        "earlier_document_index": int(
                            records[earlier_i].get("doc_index", earlier_i)),
                        "later_document_index": int(
                            record.get("doc_index", position)),
                        "estimated_jaccard": round(float(sims[sim_pos]), 4),
                    })
            if hits.size:
                flagged_pairs += int(hits.size)
                flagged_docs.add(position)
                digest = record.get("canonical_digest")
                if isinstance(digest, str) and digest:
                    flagged_digests.add(digest.lower())
                else:
                    missing_digest_positions.add(position)
                if len(examples) < max_examples:
                    best_pos = int(np.argmax(sims[hits]))
                    sim_pos = int(hits[best_pos])
                    earlier_i = int(earlier_positions[sim_pos])
                    examples.append({
                        "earlier_document_index": int(
                            records[earlier_i].get("doc_index", earlier_i)),
                        "later_document_index": int(
                            record.get("doc_index", position)),
                        "estimated_jaccard": round(float(sims[sim_pos]), 4),
                        "earlier_n_tokens": _tokens(records[earlier_i]),
                        "later_n_tokens": _tokens(record),
                    })
        for band in range(_NEAR_DUP_BANDS):
            index.setdefault(
                _band_key(record["signature"], band), []).append(n_earlier)
        earlier_doc_indices[n_earlier] = position
        n_earlier += 1

    flagged_tokens = sum(_tokens(records[position]) for position in flagged_docs)
    flagged_document_indices = sorted(
        int(records[position].get("doc_index", position))
        for position in flagged_docs)
    missing_document_indices = sorted(
        int(records[position].get("doc_index", position))
        for position in missing_digest_positions)
    scan_exhaustive = bool(exhaustive and not truncated)
    notes = []
    if flagged_docs:
        fraction = flagged_tokens / total_tokens if total_tokens else None
        notes.append(
            f"{len(flagged_docs)} train document(s) near-duplicate an earlier "
            f"train document ({flagged_tokens} sketched token(s)"
            + (f", {fraction:.4%} of the sketched train tokens" if fraction is not None
               else "")
            + "); this is training-data redundancy, not cross-split leakage, "
              "and it reduces the number of distinct training tokens. Flagged "
              "documents are subtracted in full; when the scan is exhaustive, "
              "that makes the adjusted unique-token budget a conservative "
              "lower bound on the distinct tokens")
    if borderline_pairs:
        notes.append(
            f"{borderline_pairs} within-split pair(s) are within {margin:g} "
            f"Jaccard of the {threshold:g} threshold (screening >= {screening:g}); "
            "they are reported for inspection but are not counted as duplicates")
    if missing_digest_positions:
        notes.append(
            f"{len(missing_digest_positions)} flagged train document(s) carry no "
            "canonical digest and cannot be excluded through a hash ledger")
    if not scan_exhaustive:
        notes.append(
            "the within-split scan covered a sampled/truncated prefix, so the "
            "redundancy count and flagged-token total are lower bounds")
    return {
        "enabled": True,
        "threshold_jaccard": float(threshold),
        "borderline_margin": margin,
        "screening_jaccard": screening,
        "estimator": "minhash_lsh",
        "shingle_size": int(shingle_size),
        "documents": len(records),
        "documents_with_sketch": len(has_sketch),
        "candidate_pairs": int(candidate_pairs),
        "flagged_pairs": int(flagged_pairs),
        "flagged_documents": len(flagged_docs),
        "flagged_document_indices": flagged_document_indices,
        "flagged_digests": sorted(flagged_digests),
        "flagged_tokens": int(flagged_tokens),
        "total_tokens": int(total_tokens),
        "redundant_token_fraction": (
            flagged_tokens / total_tokens if total_tokens else None),
        "borderline_pairs": int(borderline_pairs),
        "borderline_documents": len(borderline_docs),
        "borderline_examples": borderline_examples,
        "examples": examples,
        "missing_digest_documents": len(missing_digest_positions),
        "missing_digest_document_indices": missing_document_indices,
        "exhaustive": scan_exhaustive,
        "lower_bound": not scan_exhaustive,
        "notes": notes,
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
    exhaustive = bool(_strict_bool_option(
        report.get("exhaustive"), "report.exhaustive", allow_none=True))
    enabled = bool(_strict_bool_option(
        report.get("enabled", True), "report.enabled", allow_none=True))
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
        "enabled": enabled,
        "confidence": confidence,
        "method": "wilson_score_upper_bound",
        "scope": scope,
        "exhaustive": exhaustive,
        # An empty split computes no upper bound, so it cannot certify a
        # population rate even though the scan was exhaustive.
        "certifies_population": bool(
            enabled and exhaustive and bounds
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
    pairs = report.get("pairs")
    by_later_blocks = ([block for block in by_later.values()
                        if isinstance(block, dict)]
                       if isinstance(by_later, dict) else [])
    missing = sum(int(block.get("missing_digest_documents") or 0)
                  for block in by_later_blocks)
    if not isinstance(pairs, dict):
        return {"hashes": additions, "missing_digest_documents": missing}
    # `by_later_split` is the deduplicated union over earlier splits, but a
    # partial block may omit the field. In that case fall back to the
    # pair-level counts rather than adding them on top of an already-union
    # count (the same held-out document can be flagged by both train and
    # validation).
    pair_missing = sum(int(pair.get("flagged_later_missing_digest") or 0)
                       for pair in pairs.values() if isinstance(pair, dict))
    blocks_complete = bool(by_later_blocks) and all(
        "missing_digest_documents" in block for block in by_later_blocks)
    if not blocks_complete:
        missing = pair_missing
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


def near_duplicate_train_ledger_hashes(report: dict) -> dict:
    """Canonical digests of later within-train near-duplicates.

    The training split already consumes an ``exclude_train_hash_file`` ledger
    (the benchmark-decontamination channel).  Writing the later copy of each
    redundant train document into that same ledger drops the copy while
    keeping the first occurrence, so the run's unique-token budget matches
    the audited one.  Digests that are malformed or the wrong width cannot
    match ``DocumentCleaner``'s 16-byte lookup and are counted as missing
    instead of being written as inert ledger entries.
    """
    if not isinstance(report, dict):
        raise ValueError(f"report must be a mapping, got {type(report).__name__}")
    hashes = set()
    missing = int(report.get("missing_digest_documents") or 0)
    for digest in report.get("flagged_digests") or []:
        try:
            raw = bytes.fromhex(str(digest))
        except (TypeError, ValueError):
            missing += 1
            continue
        if len(raw) != 16:
            missing += 1
            continue
        hashes.add(raw)
    return {"hashes": hashes, "missing_digest_documents": missing}


def near_duplicate_train_budget(budget: dict | None,
                                redundancy: dict | None) -> dict:
    """Subtract near-duplicate train tokens from the effective token budget.

    A packed-stream budget counts every token of every kept document.  When
    the corpus repeats the same content, the model does not see that many
    distinct tokens: the later copies re-weight content it already saw.  The
    redundancy scan measures the flagged fraction among the documents it
    actually sketched, and this function applies that fraction to the
    estimated effective train-token budget.  When either measurement is a
    prefix/sample, the adjusted budget is explicitly labelled as an
    extrapolation; an unavailable scan returns ``None`` rather than silently
    treating the corpus as duplicate-free.
    """
    if budget is not None and not isinstance(budget, dict):
        raise ValueError(
            f"budget must be a mapping or None, got {type(budget).__name__}")
    if redundancy is not None and not isinstance(redundancy, dict):
        raise ValueError(
            f"redundancy must be a mapping or None, got {type(redundancy).__name__}")
    budget = budget or {}
    redundancy = redundancy or {}
    enabled = bool(_strict_bool_option(
        redundancy.get("enabled", False), "redundancy.enabled", allow_none=True))
    total_raw = redundancy.get("total_tokens")
    flagged_raw = redundancy.get("flagged_tokens")
    total = _strict_int_option(
        0 if total_raw is None else total_raw, "redundancy.total_tokens", 0)
    flagged = _strict_int_option(
        0 if flagged_raw is None else flagged_raw, "redundancy.flagged_tokens", 0)
    if flagged > total:
        raise ValueError(
            f"redundancy.flagged_tokens={flagged} exceeds "
            f"redundancy.total_tokens={total}; a flagged subset cannot be "
            "larger than the scanned tokens")
    estimated = budget.get("estimated_effective_tokens")
    if estimated is not None:
        if _is_boolean_scalar(estimated):
            raise ValueError(
                "budget.estimated_effective_tokens must be numeric or None, "
                f"got {estimated!r}")
        if isinstance(estimated, numbers.Integral):
            estimated = int(estimated)
        else:
            try:
                estimated = float(estimated)
            except (TypeError, ValueError, OverflowError):
                raise ValueError(
                    "budget.estimated_effective_tokens must be numeric or None, "
                    f"got {estimated!r}") from None
            if not math.isfinite(estimated):
                raise ValueError(
                    "budget.estimated_effective_tokens must be finite, "
                    f"got {estimated!r}")
        if estimated < 0:
            raise ValueError(
                "budget.estimated_effective_tokens must be >= 0, "
                f"got {estimated!r}")
    budget_exhaustive = bool(_strict_bool_option(
        budget.get("exhaustive", False), "budget.exhaustive", allow_none=True))
    redundancy_exhaustive = bool(_strict_bool_option(
        redundancy.get("exhaustive", False), "redundancy.exhaustive",
        allow_none=True))
    scan_exhaustive = bool(budget_exhaustive and redundancy_exhaustive)
    result = {
        "enabled": enabled,
        "estimated_effective_tokens": estimated,
        "estimated_redundant_tokens": None,
        "estimated_unique_tokens": None,
        "redundant_token_fraction": None,
        "total_tokens": total,
        "flagged_tokens": flagged,
        "unique_tokens_are_lower_bound": False,
        "basis": "not_available",
        "exhaustive": scan_exhaustive,
    }
    if not enabled or estimated is None or total <= 0:
        return result
    fraction = Fraction(flagged, total)
    try:
        redundant = float(estimated * fraction)
    except OverflowError:
        redundant = math.inf if flagged else 0.0
    try:
        unique = float(Fraction(estimated) * (1 - fraction))
    except OverflowError:
        unique = math.inf
    result.update(
        estimated_redundant_tokens=redundant,
        estimated_unique_tokens=unique,
        redundant_token_fraction=float(fraction),
        # Subtracting every flagged document in full over-counts redundancy
        # when copies overlap each other, so the adjusted count is a lower
        # bound on the true distinct-token count.  A sampled scan extrapolates
        # instead, which is why that case is not labelled a lower bound.
        unique_tokens_are_lower_bound=bool(flagged > 0 and scan_exhaustive),
        basis=("observed" if scan_exhaustive
               else "extrapolated_from_sampled_prefix"),
    )
    return result


