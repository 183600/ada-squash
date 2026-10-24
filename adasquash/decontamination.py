"""Benchmark-contamination auditing for the leakage-safe data pipeline.

The split audit in :mod:`scripts.audit_data` answers whether *this corpus*
leaks a document across its own train/validation/test boundary.  It cannot
answer a different question: whether the pretraining corpus contains text
from the downstream benchmark that is used to evaluate the model.  A model
that has seen an MMLU question, a GSM8K solution or a HumanEval prompt during
pretraining can score above its true capability, and no amount of internal
split hygiene fixes that - the contaminated text is on the *training* side
and the benchmark is outside the corpus split entirely.

This module implements the reusable part of that audit:

* exact token n-gram matching against benchmark examples (the standard
  decontamination primitive, independent of whether the overlap is one
  copied sentence or a reproduced question/answer pair);
* a bounded-memory index that reports which benchmark(s) a training document
  overlaps and where the first match starts;
* a one-sided Wilson upper bound on the contamination rate, so a clean scan
  of a prefix is reported as a bound with a scope instead of a proof;
* canonical document digests in exactly the format consumed by
  ``--exclude-train-hash-file``, so a finding can be remediated rather than
  only reported.

The pure functions here operate on token id sequences and plain mappings;
loading the benchmark and the corpus is the job of
``scripts/audit_benchmark_contamination.py``.
"""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from statistics import NormalDist

DECONTAMINATION_VERSION = 1
DEFAULT_NGRAM_SIZE = 13

# Fields that commonly hold the evaluated text.  The extractor is deliberately
# explicit rather than "all string values": benchmark records also carry ids,
# URLs and labels, and hashing those would create spurious matches (e.g. a
# shared licence string) that are not contamination of the evaluated content.
_TEXT_FIELDS = ("text", "question", "prompt", "problem", "passage",
                "context", "article", "premise", "sentence")
_ANSWER_FIELDS = ("answer", "answers", "solution", "canonical_solution",
                  "test", "code", "output", "target")
_CHOICE_FIELDS = ("choices", "options")


def _is_bool(value) -> bool:
    return isinstance(value, bool) or type(value).__name__ in ("bool_", "bool")


def _positive_int(value, name: str) -> int:
    if _is_bool(value):
        raise ValueError(f"{name} must be a positive integer, got bool {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from None
    if converted != value or converted <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return converted


def _nonnegative_int(value, name: str) -> int:
    if _is_bool(value):
        raise ValueError(f"{name} must be a non-negative integer, got bool {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}") from None
    if converted != value or converted < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return converted


def _flatten_strings(value) -> list:
    """Strings in a nested answer/choice value, in a deterministic order."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return []
    if isinstance(value, dict):
        out = []
        for key in sorted(value, key=lambda item: str(item)):
            out.extend(_flatten_strings(value[key]))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_strings(item))
        return out
    return []


def normalize_match_text(text: str) -> str:
    """Formatting- and case-insensitive form used for benchmark fingerprints.

    Matching itself runs on token ids, not on this string; normalization is
    for the benchmark content digest and for comparing two benchmark files
    that differ only in Unicode compatibility forms or line wrapping.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def canonical_ledger_text(text: str) -> str:
    """The canonical text of `scripts.data_pipeline.canonical_text`.

    Kept byte-for-byte identical (NFKC + whitespace collapse, no case
    folding) because the digest of this form is what the training loader
    looks up in its exclusion ledger.  A different normalization here would
    produce a ledger that silently matches nothing.
    """
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFKC", text).split())


def canonical_document_hash(text: str) -> bytes:
    """16-byte blake2b digest in the data-pipeline ledger format."""
    if not isinstance(text, str):
        raise ValueError(f"text must be a string, got {type(text).__name__}")
    return hashlib.blake2b(canonical_ledger_text(text).encode("utf-8", "replace"),
                           digest_size=16).digest()


def benchmark_text(record: dict) -> str:
    """The evaluated text of one benchmark record, or raise ``ValueError``.

    A pre-built ``text`` field wins; otherwise question/prompt/passage fields
    are combined with all choices and answer/solution fields.  Including the
    answer matters: a training document can contain the worked solution
    without repeating the question verbatim, and that is still benchmark
    contamination.
    """
    if not isinstance(record, dict):
        raise ValueError(f"benchmark record must be a mapping, got {type(record).__name__}")
    if isinstance(record.get("text"), str) and record["text"].strip():
        return record["text"]
    parts: list = []
    for name in _TEXT_FIELDS:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    for name in _CHOICE_FIELDS:
        if name in record:
            parts.extend(_flatten_strings(record[name]))
    for name in _ANSWER_FIELDS:
        if name in record:
            parts.extend(_flatten_strings(record[name]))
    text = "\n".join(part for part in parts if part.strip())
    if not text.strip():
        raise ValueError(
            "benchmark record has no usable text in any of "
            f"{_TEXT_FIELDS + _CHOICE_FIELDS + _ANSWER_FIELDS}; "
            f"available keys: {sorted(map(str, record))}")
    return text


def benchmark_fingerprint(records_by_benchmark: dict) -> dict:
    """Content identity of every benchmark set, independent of record order.

    Sorting the normalized texts means re-downloading the same benchmark in a
    different order produces the same identity; changing a question, an
    option, an answer or a solution changes it.
    """
    if not isinstance(records_by_benchmark, dict) or not records_by_benchmark:
        raise ValueError("records_by_benchmark must be a non-empty mapping")
    out = {}
    combined = hashlib.sha256()
    for name in sorted(records_by_benchmark):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"benchmark name must be a non-empty string, got {name!r}")
        records = list(records_by_benchmark[name])
        texts = sorted({normalize_match_text(benchmark_text(record))
                        for record in records})
        hasher = hashlib.sha256()
        for text in texts:
            hasher.update(text.encode("utf-8", "replace"))
            hasher.update(b"\x00")
        digest = hasher.hexdigest()
        out[name] = {"documents": len(records), "content_sha256": digest}
        combined.update(name.encode("utf-8", "replace"))
        combined.update(b"\x00")
        combined.update(digest.encode("ascii"))
        combined.update(b"\x00")
    out["combined_sha256"] = combined.hexdigest()
    return out


def token_ngrams(tokens, ngram_size: int = DEFAULT_NGRAM_SIZE) -> set:
    """All distinct contiguous token n-grams of one document.

    A document shorter than ``ngram_size`` has no n-gram and therefore cannot
    match by this primitive.  That is why the audit reports the benchmark's
    token-length coverage: a set of one-token questions would be invisible to
    a 13-gram scan, and the caller must know that rather than read a zero
    match count as proof of cleanliness.
    """
    n = _positive_int(ngram_size, "ngram_size")
    if isinstance(tokens, (str, bytes)):
        raise ValueError("tokens must be a sequence of token ids, not a string")
    try:
        sequence = list(tokens)
    except TypeError:
        raise ValueError(f"tokens must be an iterable sequence, got {type(tokens).__name__}") from None
    if len(sequence) < n:
        return set()
    out = set()
    for start in range(len(sequence) - n + 1):
        ngram = tuple(sequence[start:start + n])
        try:
            hash(ngram)
        except TypeError:
            raise ValueError("token ids must be hashable (integers, not lists)") from None
        out.add(ngram)
    return out


class NgramIndex:
    """Inverted token n-gram index over one or more benchmark sets."""

    def __init__(self, ngram_size: int, ngrams_by_benchmark: dict,
                 documents_by_benchmark: dict, tokens_by_benchmark: dict,
                 documents_without_ngrams_by_benchmark: dict | None = None):
        self.ngram_size = int(ngram_size)
        self.ngrams_by_benchmark = {
            name: frozenset(ngrams) for name, ngrams in ngrams_by_benchmark.items()}
        self.documents_by_benchmark = dict(documents_by_benchmark)
        self.tokens_by_benchmark = dict(tokens_by_benchmark)
        self.documents_without_ngrams_by_benchmark = dict(
            documents_without_ngrams_by_benchmark or {})
        ngram_to_benchmarks: dict = {}
        for name, ngrams in self.ngrams_by_benchmark.items():
            for ngram in ngrams:
                ngram_to_benchmarks.setdefault(ngram, set()).add(name)
        self.ngram_to_benchmarks = ngram_to_benchmarks

    @property
    def benchmarks(self) -> list:
        return sorted(self.ngrams_by_benchmark)

    def summary(self) -> dict:
        return {
            "ngram_size": self.ngram_size,
            "benchmarks": {
                name: {
                    "documents": self.documents_by_benchmark[name],
                    "tokens": self.tokens_by_benchmark[name],
                    "distinct_ngrams": len(self.ngrams_by_benchmark[name]),
                    "documents_without_ngrams": self.documents_without_ngrams_by_benchmark.get(name),
                }
                for name in self.benchmarks
            },
            "distinct_ngrams": len(self.ngram_to_benchmarks),
        }


def build_ngram_index(benchmarks: dict, ngram_size: int = DEFAULT_NGRAM_SIZE) -> NgramIndex:
    """Build an exact n-gram index from ``{name: iterable[token_ids]}``."""
    n = _positive_int(ngram_size, "ngram_size")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("benchmarks must be a non-empty mapping of name -> token sequences")
    ngrams_by_benchmark = {}
    documents_by_benchmark = {}
    tokens_by_benchmark = {}
    documents_without_ngrams_by_benchmark = {}
    for name in sorted(benchmarks):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"benchmark name must be a non-empty string, got {name!r}")
        sequences = list(benchmarks[name])
        if not sequences:
            raise ValueError(f"benchmark {name!r} has no examples")
        ngrams = set()
        tokens = 0
        documents_without_ngrams = 0
        for sequence in sequences:
            if isinstance(sequence, (str, bytes)):
                raise ValueError(f"benchmark {name!r} token sequence must not be a string")
            try:
                ids = list(sequence)
            except TypeError:
                raise ValueError(
                    f"benchmark {name!r} contains a non-sequence token row") from None
            for token in ids:
                if _is_bool(token):
                    raise ValueError(
                        f"benchmark {name!r} contains a boolean token id {token!r}")
            tokens += len(ids)
            row_ngrams = token_ngrams(ids, n)
            if not row_ngrams:
                documents_without_ngrams += 1
            ngrams |= row_ngrams
        ngrams_by_benchmark[name] = ngrams
        documents_by_benchmark[name] = len(sequences)
        tokens_by_benchmark[name] = tokens
        documents_without_ngrams_by_benchmark[name] = documents_without_ngrams
    return NgramIndex(n, ngrams_by_benchmark, documents_by_benchmark,
                      tokens_by_benchmark, documents_without_ngrams_by_benchmark)


def find_ngram_matches(tokens, index: NgramIndex) -> dict:
    """Which benchmarks a token sequence overlaps, and where.

    One pass over the document: every n-gram is looked up in the inverted
    index, so the cost is O(tokens + matched n-grams), not O(benchmark size).
    """
    if not isinstance(index, NgramIndex):
        raise ValueError("index must be an NgramIndex")
    if isinstance(tokens, (str, bytes)):
        raise ValueError("tokens must be a sequence of token ids, not a string")
    try:
        sequence = list(tokens)
    except TypeError:
        raise ValueError(f"tokens must be an iterable sequence, got {type(tokens).__name__}") from None
    n = index.ngram_size
    stats = {
        name: {"matched_ngrams": 0, "occurrences": 0, "first_position": None}
        for name in index.benchmarks
    }
    matched_by_benchmark = {name: set() for name in index.benchmarks}
    first_position = None
    if len(sequence) >= n:
        for start in range(len(sequence) - n + 1):
            ngram = tuple(sequence[start:start + n])
            try:
                owners = index.ngram_to_benchmarks.get(ngram)
            except TypeError:
                raise ValueError(
                    "token ids must be hashable (integers, not lists)") from None
            if not owners:
                continue
            if first_position is None:
                first_position = start
            for name in owners:
                matched_by_benchmark[name].add(ngram)
                stats[name]["occurrences"] += 1
                if stats[name]["first_position"] is None:
                    stats[name]["first_position"] = start
    for name, ngrams in matched_by_benchmark.items():
        stats[name]["matched_ngrams"] = len(ngrams)
    matched = sorted(name for name, row in stats.items() if row["matched_ngrams"])
    return {
        "matched": bool(matched),
        "benchmarks": matched,
        "by_benchmark": stats,
        "first_position": first_position,
    }


def wilson_upper_bound(successes: int, trials: int, confidence: float = 0.95):
    """One-sided Wilson score upper bound for a binomial proportion.

    Returns ``None`` when nothing was scanned: an empty sample cannot bound a
    rate, and returning 0.0 would turn "no data" into "no contamination".
    """
    k = _nonnegative_int(successes, "successes")
    n = _nonnegative_int(trials, "trials")
    if k > n:
        raise ValueError(f"successes={k} cannot exceed trials={n}")
    if _is_bool(confidence) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number in (0, 1), got {confidence!r}")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be finite and in (0, 1), got {confidence!r}")
    if n == 0:
        return None
    z = NormalDist().inv_cdf(confidence)
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return min(1.0, centre + margin)


def _digest_from_document(document: dict):
    value = document.get("canonical_digest")
    if value is None:
        text = document.get("text")
        if isinstance(text, str):
            return canonical_document_hash(text)
        return None
    if isinstance(value, bytes):
        if len(value) != 16:
            raise ValueError(
                f"canonical_digest must be a 16-byte digest, got {len(value)} bytes")
        return value
    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value.strip())
        except ValueError:
            raise ValueError(f"canonical_digest is not hex: {value!r}") from None
        if len(raw) != 16:
            raise ValueError(
                f"canonical_digest must be a 16-byte digest, got {len(raw)} bytes")
        return raw
    raise ValueError(
        f"canonical_digest must be bytes/hex str or None, got {type(value).__name__}")


def scan_token_documents(documents, index: NgramIndex, *,
                         max_examples: int = 100, exhaustive: bool = True) -> dict:
    """Scan cleaned training documents against a benchmark n-gram index.

    ``documents`` yields mappings with ``tokens`` (required), and optionally
    ``doc_index``, ``text`` and ``canonical_digest``.  Every contaminated
    document contributes its canonical digest to ``contaminated_hashes`` (a
    sorted hex list) even when the example list is capped, so the returned
    ledger is complete for the documents that were actually scanned.
    """
    if not isinstance(index, NgramIndex):
        raise ValueError("index must be an NgramIndex")
    limit = _nonnegative_int(max_examples, "max_examples")
    if not isinstance(exhaustive, bool):
        raise ValueError(f"exhaustive must be a bool, got {exhaustive!r}")
    by_benchmark = {
        name: {"documents_scanned": 0, "contaminated_documents": 0,
               "matched_ngram_occurrences": 0}
        for name in index.benchmarks
    }
    documents_scanned = 0
    tokens_scanned = 0
    contaminated_documents = 0
    missing_digest_documents = 0
    examples: list = []
    hashes: set = set()
    for position, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(
                f"document {position} must be a mapping with a 'tokens' field")
        if "tokens" not in document:
            raise ValueError(f"document {position} has no 'tokens' field")
        tokens = document["tokens"]
        if isinstance(tokens, (str, bytes)):
            raise ValueError(
                f"document {position} tokens must be a sequence of token ids, "
                "not a string")
        try:
            tokens = list(tokens)
        except TypeError:
            raise ValueError(
                f"document {position} tokens must be an iterable sequence") from None
        for token in tokens:
            if _is_bool(token):
                raise ValueError(
                    f"document {position} contains a boolean token id {token!r}")
        match = find_ngram_matches(tokens, index)
        documents_scanned += 1
        tokens_scanned += len(tokens)
        for name, row in match["by_benchmark"].items():
            by_benchmark[name]["documents_scanned"] += 1
            if row["matched_ngrams"]:
                by_benchmark[name]["contaminated_documents"] += 1
                by_benchmark[name]["matched_ngram_occurrences"] += row["occurrences"]
        if not match["matched"]:
            continue
        contaminated_documents += 1
        digest = _digest_from_document(document)
        if digest is not None:
            hashes.add(digest)
        else:
            # A finding without a canonical digest cannot be written to the
            # training-exclusion ledger.  Count it instead of letting the
            # remediation look complete when it is not.
            missing_digest_documents += 1
        if len(examples) < limit:
            example = {
                "doc_index": document.get("doc_index", position),
                "benchmarks": match["benchmarks"],
                "matched_ngrams": sum(
                    row["matched_ngrams"] for row in match["by_benchmark"].values()),
                "matched_ngram_occurrences": sum(
                    row["occurrences"] for row in match["by_benchmark"].values()),
                "first_position": match["first_position"],
                "canonical_digest": None if digest is None else digest.hex(),
                "by_benchmark": {
                    name: dict(row) for name, row in match["by_benchmark"].items()
                    if row["matched_ngrams"]},
            }
            text = document.get("text")
            if isinstance(text, str):
                example["text_preview"] = " ".join(text.split())[:160]
            examples.append(example)
    total = documents_scanned
    any_row = {
        "documents": total,
        "contaminated_documents": contaminated_documents,
        "observed_rate": (None if total == 0 else contaminated_documents / total),
        "upper_bound": wilson_upper_bound(contaminated_documents, total),
    }
    for name, row in by_benchmark.items():
        row["documents"] = total
        row["observed_rate"] = (None if total == 0 else
                                row["contaminated_documents"] / total)
        row["upper_bound"] = wilson_upper_bound(row["contaminated_documents"], total)
    return {
        "version": DECONTAMINATION_VERSION,
        "ngram_size": index.ngram_size,
        "exhaustive": exhaustive,
        "lower_bound": not exhaustive,
        "documents_scanned": documents_scanned,
        "tokens_scanned": tokens_scanned,
        "contaminated_documents": contaminated_documents,
        "missing_digest_documents": missing_digest_documents,
        "contaminated_hashes": sorted(digest.hex() for digest in hashes),
        "any": any_row,
        "by_benchmark": by_benchmark,
        "examples": examples,
        "examples_truncated": contaminated_documents > len(examples),
    }


def contamination_scan_bound(report: dict, *, confidence: float = 0.95,
                             max_rate: float | None = None) -> dict:
    """Turn a scan report into a stated bound (and, optionally, a verdict).

    The bound is one-sided and applies to the documents that were actually
    scanned.  A prefix scan is labelled ``scope = "scanned_prefix"`` and never
    extrapolated to the whole corpus; that distinction is the difference
    between "we found no contamination in the sample" and "the corpus is
    clean".
    """
    if not isinstance(report, dict):
        raise ValueError("report must be a mapping")
    scanned = report.get("documents_scanned")
    scanned = _nonnegative_int(scanned, "documents_scanned")
    contaminated = report.get("contaminated_documents")
    contaminated = _nonnegative_int(contaminated, "contaminated_documents")
    if contaminated > scanned:
        raise ValueError(
            f"contaminated_documents={contaminated} cannot exceed scanned={scanned}")
    if _is_bool(confidence) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number in (0, 1), got {confidence!r}")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be finite and in (0, 1), got {confidence!r}")
    if max_rate is not None:
        if _is_bool(max_rate) or not isinstance(max_rate, (int, float)):
            raise ValueError(f"max_rate must be a number in [0, 1], got {max_rate!r}")
        max_rate = float(max_rate)
        if not math.isfinite(max_rate) or not 0.0 <= max_rate <= 1.0:
            raise ValueError(f"max_rate must be finite and in [0, 1], got {max_rate!r}")
    exhaustive = report.get("exhaustive")
    if exhaustive is not None and not isinstance(exhaustive, bool):
        raise ValueError(f"report.exhaustive must be a bool or None, got {exhaustive!r}")
    scope = "audited_corpus" if exhaustive else "scanned_prefix"
    problems: list = []
    missing_digest = report.get("missing_digest_documents")
    if missing_digest is not None:
        missing_digest = _nonnegative_int(missing_digest, "missing_digest_documents")
        if missing_digest:
            problems.append(
                f"{missing_digest} contaminated document(s) have no canonical "
                "digest, so a training-exclusion ledger cannot remove them")
    rows: dict = {}
    raw_benchmarks = report.get("by_benchmark")
    if not isinstance(raw_benchmarks, dict):
        raw_benchmarks = {}
    for name in sorted(raw_benchmarks):
        row = raw_benchmarks[name]
        if not isinstance(row, dict):
            raise ValueError(f"by_benchmark[{name!r}] must be a mapping")
        row_documents = _nonnegative_int(
            row.get("documents", scanned), f"by_benchmark[{name!r}].documents")
        flagged = _nonnegative_int(row.get("contaminated_documents", 0),
                                   f"by_benchmark[{name!r}].contaminated_documents")
        if flagged > row_documents:
            raise ValueError(
                f"by_benchmark[{name!r}] flags {flagged} documents but only "
                f"{row_documents} were scanned")
        observed = None if row_documents == 0 else flagged / row_documents
        upper = wilson_upper_bound(flagged, row_documents, confidence)
        passes = None
        if max_rate is not None:
            passes = upper is not None and upper <= max_rate
            if not passes:
                problems.append(
                    f"benchmark {name!r}: {flagged}/{row_documents} scanned training "
                    f"documents overlap; the one-sided {confidence:.1%} upper bound "
                    f"{'is undefined (nothing scanned)' if upper is None else f'{upper:.4%}'} "
                    f"does not certify <= {max_rate:.4%}")
        rows[name] = {
            "documents": row_documents,
            "contaminated_documents": flagged,
            "observed_rate": observed,
            "upper_bound": upper,
            "passes_max_rate": passes,
            "scope": scope,
        }
    upper = wilson_upper_bound(contaminated, scanned, confidence)
    passes_any = None
    if max_rate is not None:
        passes_any = upper is not None and upper <= max_rate
        if not passes_any:
            problems.append(
                f"training corpus: {contaminated}/{scanned} scanned documents overlap "
                f"a benchmark; the one-sided {confidence:.1%} upper bound "
                f"{'is undefined (nothing scanned)' if upper is None else f'{upper:.4%}'} "
                f"does not certify <= {max_rate:.4%}")
    if scanned == 0:
        problems.append(
            "no training document was scanned, so no contamination rate can be "
            "bounded")
    notes = []
    if not exhaustive:
        notes.append(
            "the scan covered a prefix only; the bound applies to the scanned "
            "documents, not to the unsampled remainder")
    if len(rows) > 1:
        notes.append(
            f"the per-benchmark bounds are marginal at confidence {confidence:.1%}; "
            f"with {len(rows)} benchmarks a simultaneous family-wise claim needs a "
            "multiplicity correction, so read the per-benchmark verdicts as "
            "screening results and the any-benchmark bound as the overall bound")
    return {
        "method": "wilson_score_upper_bound",
        "confidence": confidence,
        "scope": scope,
        "certifies_population": bool(exhaustive and upper is not None),
        "documents_scanned": scanned,
        "contaminated_documents": contaminated,
        "observed_rate": None if scanned == 0 else contaminated / scanned,
        "upper_bound": upper,
        "passes_max_rate": passes_any,
        "any": {
            "documents": scanned,
            "contaminated_documents": contaminated,
            "observed_rate": None if scanned == 0 else contaminated / scanned,
            "upper_bound": upper,
            "passes_max_rate": passes_any,
            "scope": scope,
        },
        "by_benchmark": rows,
        "notes": notes,
        "problems": problems,
    }


def contamination_ledger_hashes(report: dict) -> set:
    """Decode a scan report's ``contaminated_hashes`` into raw digests."""
    if not isinstance(report, dict):
        raise ValueError("report must be a mapping")
    values = report.get("contaminated_hashes")
    if values is None:
        return set()
    if not isinstance(values, (list, tuple, set)):
        raise ValueError("contaminated_hashes must be a sequence of hex digests")
    out = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"contaminated hash must be a hex string, got {value!r}")
        try:
            raw = bytes.fromhex(value.strip())
        except ValueError:
            raise ValueError(f"contaminated hash is not hex: {value!r}") from None
        if len(raw) != 16:
            raise ValueError(
                f"contaminated hash must be a 16-byte digest, got {len(raw)} bytes")
        out.add(raw)
    return out


def contamination_protocol(*, ngram_size: int, benchmarks: dict,
                            tokenizer: str, tokenizer_revision=None,
                            dataset: str | None = None,
                            dataset_config=None,
                            dataset_revision=None,
                            split_fractions: dict | None = None) -> dict:
    """The identity an audit result depends on, as a JSON-safe mapping."""
    n = _positive_int(ngram_size, "ngram_size")
    if not isinstance(benchmarks, dict) or not benchmarks:
        raise ValueError("benchmarks must be a non-empty mapping")
    fingerprint = benchmark_fingerprint(benchmarks)
    return {
        "version": DECONTAMINATION_VERSION,
        "ngram_size": n,
        "normalization": "unicode_nfkc_casefold_whitespace",
        "benchmarks": fingerprint,
        "tokenizer": tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "dataset_revision": dataset_revision,
        "split_fractions": split_fractions,
    }


def protocol_digest(protocol: dict) -> str:
    """Deterministic SHA-256 of a contamination protocol mapping."""
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a mapping")
    payload = json.dumps(protocol, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "DECONTAMINATION_VERSION",
    "DEFAULT_NGRAM_SIZE",
    "NgramIndex",
    "benchmark_fingerprint",
    "benchmark_text",
    "build_ngram_index",
    "canonical_document_hash",
    "canonical_ledger_text",
    "contamination_ledger_hashes",
    "contamination_protocol",
    "contamination_scan_bound",
    "find_ngram_matches",
    "normalize_match_text",
    "protocol_digest",
    "scan_token_documents",
    "token_ngrams",
    "wilson_upper_bound",
]
