from __future__ import annotations
"""Leakage-safe streaming data pipeline for the AdaSquash experiments.

Principles enforced here
------------------------
1. **Split first, touch data second.**  The upstream document stream is
   partitioned into train / validation / test *contiguous ranges of the
   upstream order* before tokenization, packing or cleaning.  Train is the
   oldest slice, then validation, then test, so a model only ever fits the
   past and is scored on the future - no information travels backwards in
   time.
2. **No shuffling across the boundary.**  The split itself is deterministic
   and order preserving.  A bounded shuffle buffer may reorder packed
   *batches* inside the training split only (the usual SGD shuffle); it can
   never move a validation or test document into training.
3. **Groups stay whole.**  A document is the grouping unit of a text corpus.
   Each split owns a disjoint set of documents and packs every window from
   its own documents only, so one document can never straddle two sides of a
   train/val/test boundary.
4. **Identical cleaning on every split.**  Missing / non-string / empty
    texts, degenerate length outliers, duplicates and tokenizer failures are
    dropped by the same code on train, validation and test - always *after*
    the split.  Duplicates are detected both on the exact text and on a
    canonical form (Unicode NFKC + collapsed whitespace), so a document that
    only differs from a training document by formatting cannot leak into a
    held-out split either.
5. **Train statistics only.**  Nothing here estimates a statistic (mean,
   variance, vocabulary, length) on validation or test data and feeds it back
   into training; the held-out splits are read-only for evaluation.
6. **Cross-split duplicates are dropped from the held-out side.**  Per-split
   dedup cannot see a document that occurs once in train and once in
   validation/test.  When an `exclude_hash_file` ledger (the exact *and*
   canonical hashes of the training documents, written by
   `scripts/audit_data.py --write-train-hash-file`) is supplied, any held-out
   document whose exact or canonical text is in that ledger is dropped from
   that held-out split - never from train itself (the ledger is derived from
   train, so excluding it there would throw the whole split away).  A
   validation-document ledger (`--exclude-test-hash-file`, written by
   `audit_data.py --write-val-hash-file`) is applied to the test split only,
   so a document cannot sit on both sides of the selection/final boundary
   either; it is ignored elsewhere.
7. **Groups stay whole across splits.**  A document is the default grouping
   unit, but some corpora contain several rows produced by the same source
   (same user, same item, same page).  With an optional `group_key` and a
   train-group-hash ledger (`--exclude-group-file`, written by
   `audit_data.py --write-train-group-file --group-key <path>`), a held-out
   document whose group occurs in the training ledger is dropped, so the
   group lies wholly on the training side; a validation-group ledger
   (`--exclude-test-group-file`) is likewise applied to the test split only,
   so no group straddles the selection/final boundary either.

`build_dataloader` deliberately keeps its historical 5-argument call
signature (the split / cleaning knobs travel through
`configure_data_options`) so stubs that monkeypatch it in tests keep
working.
"""
import base64
import hashlib
import json
import math
import numbers
import os
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from adasquash.repro import seed_worker

# Defaults chosen so that "just run the trainer" already reserves a small,
# deterministic, chronological validation and test slice instead of scoring on
# training documents.
DEFAULT_DATA_OPTIONS = {
    "split": "train",            # logical split: train / validation / test
    "val_frac": 0.005,           # fraction of the document stream held out
    "test_frac": 0.005,          # sealed until the one-shot final evaluation
    "num_documents": None,       # override when the dataset reports no size
    "dedup": True,               # drop exact duplicate documents per split
    "dedup_max": 1_000_000,      # cap on the duplicate hash set size
    "min_doc_chars": 0,          # drop degenerate (shorter than this) docs
    "max_doc_chars": 0,          # length-outlier cap, 0 = no cap
    "shuffle_buffer": 0,         # train-split batch shuffle window, 0 = off
    "seed": 42,
    # Loader parallelism: 0 keeps everything in the parent process (no
    # cross-worker scan at all).
    "num_workers": 2,
    "prefetch_factor": 2,
    "persistent_workers": True,
    # Data-version pins: HuggingFace datasets/tokenizers are mutable remote
    # artifacts, so a run that does not pin a revision cannot be reproduced
    # after the upstream branch moves. None keeps the historical "latest on
    # the default branch" behaviour; the resolved values are recorded in the
    # split reports / run manifests either way.
    "dataset_revision": None,    # e.g. a commit hash of the dataset repo
    "tokenizer_revision": None,  # e.g. a commit hash of the tokenizer repo
    # Optional ledger of training-document hashes: when set, held-out splits
    # drop exact duplicates of training documents (never applied to train).
    "exclude_hash_file": None,
    # Optional ledger of validation-document hashes: when set, the *test*
    # split additionally drops exact duplicates of validation documents, so
    # the selection split and the sealed final split stay disjoint.
    "exclude_test_hash_file": None,
    # Optional grouping key (a top-level example field name or a dotted path
    # such as "meta.url").  Rows sharing a group value are one grouping unit;
    # with a group-hash ledger the held-out splits drop every document whose
    # group was seen on an earlier side, so a group never straddles a split.
    "group_key": None,
    # Optional ledger of training-group hashes (audit_data.py
    # --write-train-group-file); applied to validation/test, never to train.
    "exclude_group_file": None,
    # Optional ledger of validation-group hashes; applied to the sealed test
    # split only, so no group straddles the selection/final boundary.
    "exclude_test_group_file": None,
}

_active_data_options = dict(DEFAULT_DATA_OPTIONS)


def _is_boolean_scalar(value) -> bool:
    """Python and NumPy booleans (``np.bool_`` is not a Python bool)."""
    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == 'numpy' and cls.__name__ in ('bool_', 'bool')


def _int_option(name: str, value, minimum: int, *, allow_none: bool = False):
    """Validate an integer-valued data option without silently coercing it.

    A negative `max_doc_chars`, for example, is truthy in `DocumentCleaner`
    and drops *every* document; an integer conversion plus a lower bound turns
    that silent data loss into a configuration error.
    """
    if value is None and allow_none:
        return None
    if _is_boolean_scalar(value):
        raise ValueError(f"data option {name} must be an integer, got bool {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"data option {name} must be an integer, got {value!r}") from None
    if converted != value:
        raise ValueError(f"data option {name} must be an integer, got {value!r}")
    if converted < minimum:
        raise ValueError(f"data option {name}={converted} must be >= {minimum}")
    return converted


def _bool_option(name: str, value) -> bool:
    """A boolean option, without truthiness coercion.

    ``bool(None)``/``bool("false")`` silently turn an absent or string-valued
    flag into the opposite of what the caller wrote (an empty YAML key would
    disable deduplication). Only an actual bool is a boolean measurement here.
    """
    if not _is_boolean_scalar(value):
        raise ValueError(f"data option {name} must be a bool, got {value!r}")
    return bool(value)


def _fraction_option(name: str, value) -> float:
    if _is_boolean_scalar(value):
        raise ValueError(f"data option {name} must be a fraction, got {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"data option {name} must be a fraction, got {value!r}") from None
    if isinstance(value, (str, bytes)):
        # A quoted/numeric-string fraction is a config error, not a number;
        # `float("0.5")` used to silently change the split protocol. Exact
        # rationals (Fraction/Decimal) are legitimate numeric values.
        raise ValueError(f"data option {name} must be a fraction, got {value!r}")
    if not math.isfinite(converted) or not 0.0 <= converted < 1.0:
        raise ValueError(f"data option {name}={converted} must be finite and in [0, 1)")
    return converted


def validate_data_options(options: dict | None = None) -> dict:
    """Return a validated copy of the split / cleaning options.

    The training and evaluation scripts validate their command-line flags, but
    `configure_data_options` is also a public entry point used by probes and
    tests.  Validating here means a probe cannot run with, say, a zero
    `dedup_max` (silently disabling deduplication) or `min_doc_chars` larger
    than `max_doc_chars` (no document can pass the cleaner) while still writing
    a manifest that looks like a normal leakage-safe run.
    """
    opts = dict(DEFAULT_DATA_OPTIONS)
    if options:
        opts.update(options)
    unknown = sorted(set(opts) - set(DEFAULT_DATA_OPTIONS))
    if unknown:
        raise KeyError(f"unknown data-pipeline option(s): {unknown}")
    split = opts.get("split")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"data option split must be train/validation/test, got {split!r}")
    opts["val_frac"] = _fraction_option("val_frac", opts.get("val_frac"))
    opts["test_frac"] = _fraction_option("test_frac", opts.get("test_frac"))
    if opts["val_frac"] + opts["test_frac"] >= 1.0:
        raise ValueError(
            f"data options val_frac={opts['val_frac']} + test_frac={opts['test_frac']} "
            "leave no training documents; the split fractions must sum to less than 1")
    opts["num_documents"] = _int_option(
        "num_documents", opts.get("num_documents"), 1, allow_none=True)
    opts["dedup"] = _bool_option("dedup", opts.get("dedup"))
    opts["dedup_max"] = _int_option("dedup_max", opts.get("dedup_max"), 1)
    opts["min_doc_chars"] = _int_option("min_doc_chars", opts.get("min_doc_chars"), 0)
    opts["max_doc_chars"] = _int_option("max_doc_chars", opts.get("max_doc_chars"), 0)
    if opts["max_doc_chars"] and opts["min_doc_chars"] > opts["max_doc_chars"]:
        raise ValueError(
            f"data options min_doc_chars={opts['min_doc_chars']} exceeds "
            f"max_doc_chars={opts['max_doc_chars']}; no document could pass the cleaner")
    opts["shuffle_buffer"] = _int_option("shuffle_buffer", opts.get("shuffle_buffer"), 0)
    opts["seed"] = _int_option("seed", opts.get("seed"), -(2 ** 63))
    if opts["seed"] >= 2 ** 64:
        # torch.Generator.manual_seed accepts the unsigned 64-bit range but
        # raises on 2**64; validating here turns a late loader crash into a
        # configuration error.
        raise ValueError(
            f"data option seed={opts['seed']} must be < {2 ** 64}")
    opts["num_workers"] = _int_option("num_workers", opts.get("num_workers"), 0)
    opts["prefetch_factor"] = _int_option("prefetch_factor", opts.get("prefetch_factor"), 1)
    opts["persistent_workers"] = _bool_option(
        "persistent_workers", opts.get("persistent_workers"))
    for name in ("dataset_revision", "tokenizer_revision"):
        value = opts.get(name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"data option {name} must be a string or None, got {value!r}")
    group_key = opts.get("group_key")
    if group_key is not None and (not isinstance(group_key, str) or not group_key.strip()):
        raise ValueError(f"data option group_key must be a non-empty string or None, got {group_key!r}")
    opts["group_key"] = group_key.strip() if isinstance(group_key, str) else None
    for name in ("exclude_hash_file", "exclude_test_hash_file",
                 "exclude_group_file", "exclude_test_group_file"):
        value = opts.get(name)
        if value is not None and not isinstance(value, (str, os.PathLike)):
            raise ValueError(f"data option {name} must be a path or None, got {value!r}")
    return opts


def configure_data_options(**overrides) -> dict:
    """Set the split / cleaning options used by the next `build_dataloader`.

    The candidate state is validated before it is committed, so a rejected
    call cannot leave the process in a half-updated configuration that a later
    `build_dataloader` would silently consume.
    """
    unknown = sorted(set(overrides) - set(DEFAULT_DATA_OPTIONS))
    if unknown:
        raise KeyError(f"unknown data-pipeline option(s): {unknown}")
    candidate = dict(_active_data_options)
    candidate.update(overrides)
    validated = validate_data_options(candidate)
    _active_data_options.clear()
    _active_data_options.update(validated)
    return dict(_active_data_options)


def get_data_options() -> dict:
    return dict(_active_data_options)


def infer_num_documents(ds) -> int | None:
    """Best-effort total document count of a streaming dataset.

    HuggingFace datasets usually publish `num_examples` in the split metadata;
    when they do not (or the object is a stub), the caller must pass
    `num_documents` explicitly so the chronological boundary can be placed.
    """
    info = getattr(ds, "info", None)
    splits = getattr(info, "splits", None)
    if not splits:
        return None
    # The streaming dataset was loaded with split="train", so only the train
    # count describes this stream. Falling back to validation/test would
    # place the chronological boundaries at a different corpus length (and
    # could select documents that are not in the stream at all).
    try:
        split_info = splits.get("train")
    except AttributeError:
        split_info = splits["train"] if "train" in splits else None
    n = getattr(split_info, "num_examples", None) if split_info is not None else None
    # bool is an int subclass; True must not become a one-document stream and
    # silently place the held-out boundary at the wrong place.
    return (int(n) if isinstance(n, numbers.Integral)
            and not isinstance(n, bool) and n > 0 else None)


_REVISION_CACHE: dict = {}


def _is_pinned_revision(revision) -> bool:
    """Whether `revision` is a full 40-hex Hub commit, not a short/ambiguous id.

    A 7-40 character hex string is also a valid branch/tag spelling and an
    ambiguous abbreviated SHA. Treating it as a concrete commit would skip the
    Hub resolution, certify a mutable name as the data version, and key the
    held-out cache on it.
    """
    text = str(revision or "")
    return len(text) == 40 and all(c in "0123456789abcdefABCDEF" for c in text)


def _resolve_hub_revision(kind: str, name: str, revision) -> str | None:
    """Resolve a dataset/tokenizer revision to a concrete Hub commit hash.

    A revision that already looks like a commit is returned as-is (there is
    nothing to move).  For "latest" / a branch / a tag the Hub's current
    `sha` is looked up once per process and cached; when the Hub is
    unreachable the requested value (or None) is returned so an offline run
    still works and the caller can decide whether a cache may be used.
    """
    requested = str(revision) if revision else None
    key = (kind, str(name), requested)
    if key in _REVISION_CACHE:
        return _REVISION_CACHE[key]
    if _is_pinned_revision(requested):
        _REVISION_CACHE[key] = requested
        return requested
    resolved = None
    try:
        from huggingface_hub import HfApi

        kwargs = {"revision": requested} if requested else {}
        api = HfApi()
        info = (api.dataset_info(name, **kwargs) if kind == "dataset"
                else api.model_info(name, **kwargs))
        resolved = getattr(info, "sha", None)
    except Exception:
        resolved = None
    if resolved is None and requested:
        resolved = requested
    _REVISION_CACHE[key] = resolved
    return resolved


def resolve_dataset_revision(dataset_name: str, revision=None) -> str | None:
    """Resolve the requested dataset revision to the concrete upstream commit.

    `revision=None` means "latest on the default branch", which can move
    between runs; the request alone therefore cannot be reproduced after the
    fact. Resolving it to the current Hub commit hash and recording that hash
    in the split report / run manifest is what makes the trained-on data
    version auditable (and, because it enters the materialized-cache key, a
    moved upstream branch cannot silently reuse a stale cache). When
    the Hub is unreachable the resolution degrades to the requested value (or
    None) instead of failing the run.
    """
    return _resolve_hub_revision("dataset", dataset_name, revision)


def resolve_tokenizer_revision(tokenizer_name: str, revision=None) -> str | None:
    """Resolve a tokenizer revision to the concrete Hub commit hash.

    The tokenizer is part of the data version just like the corpus: a run
    that records the requested branch but loads it before it is pinned can
    train and evaluate with different tokenizers.  Resolving first lets the
    loader pass the concrete commit to `from_pretrained`.
    """
    return _resolve_hub_revision("tokenizer", tokenizer_name, revision)


def split_document_boundaries(num_documents: int, val_frac: float, test_frac: float) -> dict:
    """Contiguous, chronological boundaries over `num_documents` documents.

    The three ranges are `[train][validation][test]` in upstream order: train
    is the past, validation and test are the future.  They are disjoint by
    construction, so no document (group) can appear on two sides.
    """
    # `int(1000.5)`/`int("1000")` silently changes the corpus length the
    # boundaries are placed over, and a bool is not a document count. Validate
    # the geometry instead of coercing it, exactly like the data options that
    # feed this helper.
    num_documents = _int_option("num_documents", num_documents, 1)
    val_frac = _fraction_option("val_frac", val_frac)
    test_frac = _fraction_option("test_frac", test_frac)
    if val_frac + test_frac >= 1.0:
        # Validate the requested fractions before rounding. Banker's rounding
        # can otherwise round both halves down and leave a (small) train
        # range even though the requested split assigns 100% to held-out data.
        raise ValueError(
            f"val_frac={val_frac} + test_frac={test_frac} leaves no training "
            "documents; the split fractions must sum to less than 1")
    val_n = int(round(num_documents * val_frac))
    test_n = int(round(num_documents * test_frac))
    if val_frac > 0.0:
        val_n = max(1, val_n)
    if test_frac > 0.0:
        test_n = max(1, test_n)
    train_n = num_documents - val_n - test_n
    if train_n <= 0:
        raise ValueError(
            f"val_frac={val_frac} + test_frac={test_frac} leaves no training documents "
            f"out of num_documents={num_documents}"
        )
    return {
        "train": (0, train_n),
        "validation": (train_n, val_n),
        "test": (train_n + val_n, test_n),
    }


def select_stream_split(ds, split: str, options: dict | None = None):
    """Slice `ds` to the requested logical split; return `(stream, report)`.

    `report` records the boundary actually used so runs can log - and reviewers
    can verify - which documents were held out.
    """
    opts = dict(DEFAULT_DATA_OPTIONS)
    if options:
        opts.update(options)
    if split not in ("train", "validation", "test"):
        raise ValueError(f"unknown split '{split}'")
    val_frac = _fraction_option("val_frac", opts["val_frac"])
    test_frac = _fraction_option("test_frac", opts["test_frac"])
    if val_frac + test_frac >= 1.0:
        raise ValueError(
            f"val_frac={val_frac} + test_frac={test_frac} leaves no training "
            "documents; the split fractions must sum to less than 1")
    raw_documents = opts.get("num_documents")
    num_documents = (_int_option("num_documents", raw_documents, 1)
                     if raw_documents is not None else infer_num_documents(ds))
    report = {
        "split": split,
        "num_documents": num_documents,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "dedup": _bool_option("dedup", opts["dedup"]),
        "min_doc_chars": _int_option("min_doc_chars", opts["min_doc_chars"], 0),
        "max_doc_chars": _int_option("max_doc_chars", opts["max_doc_chars"], 0),
        "dataset_revision": opts.get("dataset_revision"),
        "tokenizer_revision": opts.get("tokenizer_revision"),
    }
    if val_frac == 0.0 and test_frac == 0.0:
        if split != "train":
            raise ValueError(
                f"val_frac=0 and test_frac=0 leave no documents for the {split} split"
            )
        report.update(start=0, count=None)
        return ds, report
    if num_documents is None:
        raise ValueError(
            "the dataset does not report how many documents it contains, so the "
            "chronological train/validation/test boundary cannot be placed. Pass "
            "--num-documents <total documents>, or explicitly disable the held-out "
            "splits with --val-frac 0 --test-frac 0 (not recommended: the model then "
            "has no unbiased validation signal)."
        )
    start, count = split_document_boundaries(num_documents, val_frac, test_frac)[split]
    if count <= 0:
        raise ValueError(
            f"the {split} split is empty: num_documents={num_documents}, "
            f"val_frac={val_frac}, test_frac={test_frac}"
        )
    report.update(start=start, count=count)
    return ds.skip(start).take(count), report


def load_document_stream(dataset_name: str, dataset_config: str | None, split: str, options: dict | None = None):
    """Raw (unpacked, uncleaned) document stream for one logical split.

    The requested revision is resolved *before* the stream is opened and the
    concrete commit is passed to `load_dataset`.  Loading the mutable
    request first and recording the resolved hash afterwards can mix two
    upstream versions inside one train/validation/test split (and makes the
    recorded hash describe a different snapshot than the one actually read).
    """
    from datasets import load_dataset

    opts = dict(DEFAULT_DATA_OPTIONS)
    if options:
        opts.update(options)
    requested_revision = opts.get("dataset_revision")
    resolved_revision = resolve_dataset_revision(dataset_name, requested_revision)
    load_revision = resolved_revision or requested_revision
    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True,
                      revision=load_revision)
    stream, report = select_stream_split(ds, split, opts)
    report["dataset_revision_resolved"] = resolved_revision
    report["dataset_revision_loaded"] = load_revision
    return stream, report


def document_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).digest()


def canonical_text(text: str) -> str:
    """Formatting-insensitive form of a document used for near-duplicate checks.

    Unicode NFKC normalization plus whitespace collapsing: two documents that
    differ only in line breaks, repeated spaces or compatibility characters
    carry the same content and must not sit on two sides of a split.  The
    transformation is deliberately conservative (no case folding): it only
    removes formatting differences, never information.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


def canonical_document_hash(text: str) -> bytes:
    return hashlib.blake2b(canonical_text(text).encode("utf-8", "replace"), digest_size=16).digest()


def document_digests(text: str) -> tuple[bytes, bytes]:
    """`(exact, canonical)` digests of one document text."""
    return document_hash(text), canonical_document_hash(text)


def group_hash(value) -> bytes:
    """Stable digest of a grouping-unit identifier (user / item / source page).

    Filesystem- and memory-friendly replacement for storing raw group values
    in a ledger.  Scalars are hashed by their string form; list/tuple/set
    groups by their sorted element strings, so a document that belongs to
    several groups gets one deterministic digest per group set.
    """
    if isinstance(value, (list, tuple, set)):
        payload = json.dumps(sorted(str(v) for v in value), ensure_ascii=False)
    elif isinstance(value, str):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(payload.encode("utf-8", "replace"), digest_size=16).digest()


def extract_group_value(example, group_key: str):
    """Grouping-unit value of an example, or None when it has none.

    `group_key` is a top-level field name or a dotted path (``"meta.url"``).
    None / NaN / blank values count as "no group" and are never used for
    exclusion decisions.
    """
    if not group_key or not isinstance(example, dict):
        return None
    if group_key in example:
        value = example[group_key]
    else:
        value = example
        for part in group_key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
    if value is None:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        # Catch NumPy float scalars too (`np.float32(nan)` is not a Python
        # float); otherwise every NaN group hashes as the same "nan" group
        # and can collapse unrelated documents into one exclusion unit.
        try:
            if math.isnan(float(value)):
                return None
        except (TypeError, ValueError, OverflowError):
            pass
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _dedup_hash(example) -> bytes | None:
    """Canonical digest of the stripped document text, or None when there is none.

    Dedup keys on the *canonical* form so formatting variants of the same
    document are removed as duplicates; the ledger lookup additionally tests
    the exact digest (see `DocumentCleaner`).
    """
    raw = example.get("text") if isinstance(example, dict) else None
    if not isinstance(raw, str):
        return None
    return canonical_document_hash(raw.strip())


def _dedup_shard_owner(example, n_workers: int, hashed: bytes | None = None) -> int:
    """Which worker owns a document when dedup has to be global.

    Worker-local `seen` sets cannot see duplicates that the round-robin
    document shard handed to another worker, so with several loader workers
    duplicate documents are routed by their canonical-text hash instead: every
    formatting-equivalent copy lands in the same worker and the per-worker set
    then drops all but the first occurrence, matching the single-process
    cleaner the audit uses.

    `hashed` lets a caller that already hashed the stripped text pass the
    digest in.
    """
    if hashed is None:
        hashed = _dedup_hash(example)
    if hashed is None:
        return 0
    return int.from_bytes(hashed, "little") % n_workers


def load_hash_file(path) -> set:
    """Read a document-hash ledger written by `audit_data.py`.

    Accepts a JSON list of hex digests or whitespace/newline separated hex
    digests and returns the set of raw digest bytes. The line-oriented form
    (what the writer emits) is read incrementally; the JSON-list form (which
    must be parsed in one piece) is accepted as well.
    """
    def _digest(token) -> bytes:
        """Decode one ledger digest, rejecting malformed/wrong-length entries.

        `DocumentCleaner` looks up 16-byte blake2b digests. A ledger with a
        different digest width (or a non-hex line) would silently match
        nothing, making the run claim a cross-split exclusion it never
        applied; fail closed instead.
        """
        try:
            raw = bytes.fromhex(str(token).strip())
        except (TypeError, ValueError):
            raise ValueError(
                f"hash ledger {path} contains a non-hex digest {token!r}") from None
        if len(raw) != 16:
            raise ValueError(
                f"hash ledger {path} contains a {len(raw)}-byte digest {token!r}; "
                "expected the 16-byte digest written by scripts/audit_data.py")
        return raw

    out: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped[0] == "[":
                f.seek(0)
                entries = json.loads(f.read())
                if not isinstance(entries, list):
                    raise ValueError(
                        f"hash ledger {path} is a JSON object, not a list of digests")
                out.update(_digest(entry) for entry in entries)
                return out
            for token in line.split():
                out.add(_digest(token))
    return out


class DocumentCleaner:
    """One cleaning policy, applied identically to every split, after the split.

    Drops (with counters, so a run can report what was removed):
    missing / non-string text (NaN or None from the upstream table), empty
    documents, degenerate short documents, length outliers, duplicates
    (within the split, detected on the exact *and* canonical text), exact or
    canonical duplicates of documents listed in `exclude_hashes` (the
    training-document ledger used by held-out splits), and documents whose
    grouping unit is listed in `exclude_group_hashes` (the training-group
    ledger, so a group never straddles a split boundary).  The duplicate set
    is bounded at `dedup_max` entries; once the set is full, unseen documents
    are kept without being recorded and `dedup_cap_reached` is set in the
    report, so the silent degradation of the in-split dedup guarantee is
    visible (the cross-split ledger lookup is a set membership test and stays
    exact regardless).
    """

    def __init__(self, dedup: bool = True, dedup_max: int = 1_000_000,
                 min_doc_chars: int = 0, max_doc_chars: int = 0,
                 exclude_hashes: set | None = None,
                 group_key: str | None = None,
                 exclude_group_hashes: set | None = None):
        self.dedup = bool(dedup)
        self.dedup_max = int(dedup_max)
        self.min_doc_chars = int(min_doc_chars)
        self.max_doc_chars = int(max_doc_chars)
        # Held by reference: the ledger is shared with the caller, who must not
        # mutate it after handing it over.
        self.exclude_hashes = exclude_hashes if exclude_hashes is not None else set()
        self.group_key = group_key or None
        self.exclude_group_hashes = (exclude_group_hashes if exclude_group_hashes is not None
                                     else set())
        # `seen` stores the leading 64 bits of each canonical digest as an
        # int; the full digests are used for the cross-split ledger lookup.
        self.seen: set = set()
        self.stats: Counter = Counter()

    def _group_excluded(self, example) -> bool:
        if not (self.group_key and self.exclude_group_hashes):
            return False
        value = extract_group_value(example, self.group_key)
        if value is None:
            self.stats["group_value_missing"] += 1
            return False
        if group_hash(value) in self.exclude_group_hashes:
            self.stats["dropped_cross_split_group"] += 1
            return True
        return False

    def __call__(self, example, precomputed_hash: bytes | None = None):
        """Clean one example; `precomputed_hash`, when given, must be the
        *canonical* digest of the stripped text."""
        self.stats["documents_read"] += 1
        if not isinstance(example, dict):
            self.stats["dropped_unreadable"] += 1
            return None
        text = example.get("text")
        if text is None:
            self.stats["dropped_missing_text"] += 1
            return None
        if isinstance(text, numbers.Real) and not isinstance(text, bool):
            # A huge int/Fraction has no finite float image; it is simply a
            # non-string document, not a reason to crash the cleaner.
            try:
                is_nan = math.isnan(float(text))
            except (TypeError, ValueError, OverflowError):
                is_nan = False
            if is_nan:
                self.stats["dropped_missing_text"] += 1
                return None
        if not isinstance(text, str):
            self.stats["dropped_non_string_text"] += 1
            return None
        text = text.strip()
        if not text:
            self.stats["dropped_empty"] += 1
            return None
        if self.min_doc_chars and len(text) < self.min_doc_chars:
            self.stats["dropped_too_short"] += 1
            return None
        if self.max_doc_chars and len(text) > self.max_doc_chars:
            self.stats["dropped_too_long"] += 1
            return None
        if self._group_excluded(example):
            # This document's grouping unit was already seen on an earlier
            # side of a split boundary; keeping it would leak the group.
            return None
        if self.dedup or self.exclude_hashes:
            canonical = (precomputed_hash if precomputed_hash is not None
                         else canonical_document_hash(text))
            if self.exclude_hashes:
                exact = document_hash(text)
                if exact in self.exclude_hashes or canonical in self.exclude_hashes:
                    # Exact or formatting-equivalent duplicate of an earlier
                    # split's document: keeping it would leak that document
                    # across the split boundary.
                    self.stats["dropped_cross_split_duplicate"] += 1
                    return None
            if self.dedup:
                key = (int.from_bytes(canonical[:8], "little")
                       if isinstance(canonical, (bytes, bytearray)) else canonical)
                if key in self.seen:
                    self.stats["dropped_duplicate"] += 1
                    return None
                if len(self.seen) < self.dedup_max:
                    self.seen.add(key)
                else:
                    # The seen-hash set is full: from here on the split keeps
                    # every document. The flag makes the degradation of the
                    # dedup guarantee visible in the split report (the
                    # cross-split ledger lookup is a set membership test and
                    # stays exact regardless).
                    self.stats["dedup_cap_reached"] = 1
        self.stats["documents_kept"] += 1
        return text

    def report(self) -> dict:
        return dict(self.stats)


def _is_token_row(row) -> bool:
    """Whether one tokenizer row is a sequence of ids (not a flat scalar id)."""
    if isinstance(row, (str, bytes)):
        return False
    try:
        len(row)
    except TypeError:
        return False
    return True


def _iter_dataset_examples(ds):
    """Yield one example at a time from a stream without DataLoader sharding.

    HuggingFace `IterableDataset.__iter__` already splits its shards across
    DataLoader workers. Iterating it directly here and then applying this
    class's own worker filter would hand each worker only a fraction of its
    assigned source (and the union would lose documents). `IterableDataset.iter`
    is the non-worker-sharded iteration API, so flatten its one-example batches
    back to documents and leave the routing to `PackedStream`.
    """
    if not hasattr(ds, 'iter') or not hasattr(ds, '_ex_iterable'):
        yield from ds
        return
    for batch in ds.iter(batch_size=1):
        if not isinstance(batch, dict) or not batch:
            yield batch
            continue
        columns = list(batch.values())
        for i in range(len(columns[0])):
            yield {key: values[i] for key, values in batch.items()}


class PackedStream(IterableDataset):
    """Pack cleaned documents of one split into fixed-length LM windows.

    The buffer is private to the split: documents skipped by the split
    selector never enter it, so packed windows cannot mix splits.
    """

    tokenize_batch = 64

    def __init__(self, ds, tok, seq_len: int, cleaner: DocumentCleaner | None = None,
                 shuffle_buffer: int = 0, seed: int = 42):
        self.ds = ds
        self.tok = tok
        # A zero sequence length never advances `drain()`, so a full iteration
        # would loop forever yielding empty windows. Validate the geometry
        # here (direct callers bypass build_dataloader's checks).
        self.seq_len = _int_option("seq_len", seq_len, 1)
        eos = getattr(tok, "eos_token_id", None)
        if eos is None:
            raise ValueError(
                "PackedStream needs a tokenizer with a non-None eos_token_id: "
                "the end-of-document separator is appended between documents")
        self.cleaner = cleaner if cleaner is not None else DocumentCleaner()
        self.shuffle_buffer = _int_option("shuffle_buffer", shuffle_buffer or 0, 0)
        self.seed = _int_option("seed", seed, -(2 ** 63))
        if self.seed >= 2 ** 64:
            raise ValueError(
                f"PackedStream seed={self.seed} must be < {2 ** 64}")
        # A one-dimensional long tensor plus integer offsets. The tokenizer
        # returns Python ints, so the tensor is filled by copying small
        # per-document chunks; windows are cloned out of it.
        self.buf: torch.Tensor | None = None
        self.fill = 0
        self.pos = 0
        self.epoch = 0

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        n_workers = worker_info.num_workers if worker_info is not None else 1
        worker_seed = worker_info.seed if worker_info is not None else 0
        # Duplicate removal is per epoch: documents repeated inside one pass
        # are dropped, while a later epoch over the same corpus still sees its
        # own documents.
        self.cleaner.seen = set()
        epoch = self.epoch
        self.epoch += 1
        return self._shuffle(self._pack(worker_id, n_workers), worker_id, epoch,
                             worker_seed)

    def _shuffle(self, stream, worker_id: int, epoch: int=0, worker_seed: int=0):
        if self.shuffle_buffer <= 0:
            yield from stream
            return
        import random as _random

        # Include the epoch: otherwise a persistent worker replays the exact
        # same batch permutation every pass, which defeats shuffling after the
        # first epoch.
        rng = _random.Random(self.seed * 1000003 + worker_id + epoch * 104729
                             + worker_seed * 1000000007)
        window: list = []
        for item in stream:
            window.append(item)
            if len(window) >= self.shuffle_buffer:
                j = rng.randrange(len(window))
                window[j], window[-1] = window[-1], window[j]
                yield window.pop()
        rng.shuffle(window)
        yield from window

    def _pack(self, worker_id: int, n_workers: int):
        need = self.seq_len + 1
        buf = self.buf
        if buf is None:
            buf = torch.empty(max(4 * need, 1 << 16), dtype=torch.long)
        fill = self.fill
        pos = self.pos
        texts: list = []

        def reserve(extra: int) -> None:
            nonlocal buf
            if fill + extra <= buf.numel():
                return
            grown = torch.empty(max(buf.numel() * 2, fill + extra), dtype=torch.long)
            grown[:fill].copy_(buf[:fill])
            buf = grown

        def drain():
            nonlocal fill, pos
            while fill - pos >= need:
                # Owned clone: the ring is reused by later refills, so views
                # into it must not escape (the shuffle buffer holds windows).
                window = buf[pos:pos + need].clone()
                yield window[:-1], window[1:]
                pos += self.seq_len
            if pos:
                remaining = fill - pos
                if remaining:
                    buf[:remaining].copy_(buf[pos:fill])
                fill = remaining
                pos = 0

        def extend(batch):
            nonlocal fill
            try:
                token_lists = self.tok(batch, add_special_tokens=False)["input_ids"]
                if (len(token_lists) != len(batch)
                        or not all(_is_token_row(row) for row in token_lists)):
                    # A batch tokenizer that returns a different number of
                    # rows (or a flat list of scalar ids) cannot be aligned
                    # to the documents; silently dropping/pulling tokens
                    # would change the corpus.
                    raise ValueError(
                        f'tokenizer returned a non-matrix result for '
                        f'{len(batch)} document(s)')
            except Exception:
                token_lists = []
                for text in batch:
                    try:
                        token_lists.append(self.tok(text, add_special_tokens=False)["input_ids"])
                    except Exception:
                        self.cleaner.stats["dropped_tokenization_error"] += 1
            eos = self.tok.eos_token_id
            flat_ids: list = []
            for ids in token_lists:
                if len(ids) == 0:
                    continue
                flat_ids.extend(ids)
                flat_ids.append(eos)
            if not flat_ids:
                return
            reserve(len(flat_ids))
            chunk = torch.tensor(flat_ids, dtype=torch.long)
            buf[fill:fill + chunk.numel()].copy_(chunk)
            fill += chunk.numel()

        shard_by_hash = self.cleaner.dedup and n_workers > 1
        i = 0
        for ex in _iter_dataset_examples(self.ds):
            ex_hash = None
            if shard_by_hash:
                ex_hash = _dedup_hash(ex)
                if _dedup_shard_owner(ex, n_workers, ex_hash) != worker_id:
                    continue
            elif i % n_workers != worker_id:
                i += 1
                continue
            i += 1
            text = self.cleaner(ex, ex_hash)
            if text is None:
                continue
            texts.append(text)
            # Keep accumulating until the tokenizer batch is full or the ring
            # already holds a drainable window (`fill - pos` is the number of
            # *unread* tokens; `len(buf)` would be the tensor's capacity).
            if len(texts) < self.tokenize_batch and fill - pos < need:
                continue
            extend(texts)
            texts = []
            yield from drain()
        if texts:
            extend(texts)
            yield from drain()
        # Keep the partially filled ring (and its read cursor) across epochs
        # of the same stream object.
        self.buf = buf
        self.fill = fill
        self.pos = pos


class _JsonlDocumentStream(IterableDataset):
    """Re-iterable raw-document stream backed by a materialized cache file.

    The cache stores the raw `text` value with its type: a `bytes` document is
    base64-tagged and restored as `bytes` instead of being silently decoded to
    `str`.  Without that, a document the streaming cleaner drops as
    non-string would be kept when the held-out slice is served from cache -
    the same corpus would then clean differently with and without the cache.
    """

    def __init__(self, path: Path):
        self.path = str(path)

    @staticmethod
    def _restore(record):
        if not isinstance(record, dict):
            return record
        text = record.get("text")
        if record.get("text_encoding") == "base64" and isinstance(text, str):
            try:
                return {"text": base64.b64decode(text)}
            except (ValueError, TypeError):
                return {"text": text}
        return record

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                yield self._restore(json.loads(line))


_VERIFIED_CACHES: dict = {}


def _cache_matches(cache_path: Path, fingerprint: dict) -> bool:
    """Verify a materialized held-out cache against its own `.meta.json`.

    A cache is reused only when its byte size and sha256 still match the
    fingerprint written at materialization time: a truncated/partially written
    or edited file would otherwise silently feed different documents into
    validation/test while the run manifest claims the pinned fingerprint. The
    sha256 is computed once per (path, size, mtime) per process, so repeated
    loader builds in one run (checkpoint sweeps) do not re-hash the file.
    """
    expected_bytes = fingerprint.get("bytes")
    sha = fingerprint.get("sha256")
    try:
        st = cache_path.stat()
    except OSError:
        return False
    try:
        expected_size = None if expected_bytes is None else int(expected_bytes)
    except (TypeError, ValueError, OverflowError):
        return False
    if (expected_size is not None
            and (isinstance(expected_bytes, bool)
                 or expected_size != expected_bytes
                 or expected_size < 0)):
        # A bool/fractional/string byte count is not a verifiable fingerprint;
        # `int(10.5)`/`int(True)` used to truncate into a plausible size.
        return False
    if expected_size is not None and st.st_size != expected_size:
        return False
    # The size alone cannot identify the payload: an edited cache with the same
    # length would pass. Version-2 sidecars always carry the sha256, so a
    # missing/empty digest means this is not a verifiable fingerprint.
    if not isinstance(sha, str) or not sha:
        return False
    stamp = (sha, st.st_size, st.st_mtime_ns)
    if _VERIFIED_CACHES.get(str(cache_path)) == stamp:
        return True
    hasher = hashlib.sha256()
    try:
        with open(cache_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                hasher.update(chunk)
    except OSError:
        return False
    if hasher.hexdigest() != sha:
        return False
    if len(_VERIFIED_CACHES) >= 8:
        _VERIFIED_CACHES.clear()
    _VERIFIED_CACHES[str(cache_path)] = stamp
    return True


def _split_cache_dir() -> Path | None:
    """Where materialized held-out slices are stored.

    `ADASQUASH_DATA_CACHE` overrides the root; the values `0`/`off`/`none`
    disable materialization entirely (the loader then re-plays the streaming
    split on every build, which is the historical behaviour).
    """
    env = os.environ.get("ADASQUASH_DATA_CACHE")
    if env is not None:
        if env.strip().lower() in ("", "0", "off", "none", "false"):
            return None
        return Path(env)
    try:
        return Path.home() / ".cache" / "adasquash" / "splits"
    except Exception:
        return None


def _materialized_split_stream(stream_ds, dataset_name: str, dataset_config, split: str,
                               report: dict):
    """Cache the held-out slice's raw documents once; serve later builds locally.

    `select_stream_split` implements the chronological split with
    `ds.skip(start)`, which iterates (and decodes/downloads) every preceding
    document. Materializing the raw text of the slice on first use keeps the
    exact same documents in the exact same order - the cleaner still runs
    afterwards, so cleaning/dedup semantics are unchanged - while subsequent
    builds read a local file. Returns `(stream, cache_path)` or
    `(stream_ds, None)` when caching is disabled/unavailable.

    The cache key carries the *resolved* dataset revision (recorded in
    `report`), so a cache materialized from the default branch before it moved
    is not silently reused afterwards.  When the revision is neither pinned
    nor resolvable (offline Hub lookup), no cache is read or written at all:
    the cached documents and the current training stream could then come from
    different upstream commits, and a stale held-out slice could silently
    overlap what training now sees. The written payload is fingerprinted
    (sha256 + document count) into a `.meta.json` sidecar and echoed into
    `report['materialized_cache_fingerprint']`, so a run manifest pins the
    concrete cached content, not only its name; before any reuse the payload
    is verified against that fingerprint *and* against the declared split
    `count`, and a cache that fails either check is rebuilt instead of
    silently serving different or truncated held-out documents.  The payload
    preserves the type of the raw `text` field (`bytes` included), so the
    cached and directly streamed splits clean identically.
    """
    start, count = report.get("start"), report.get("count")
    if not start or not count:
        # No prefix to skip.
        return stream_ds, None
    root = _split_cache_dir()
    if root is None:
        return stream_ds, None
    resolved_revision = report.get("dataset_revision_resolved")
    if not _is_pinned_revision(resolved_revision):
        # Without a resolved commit the cache key would not pin the upstream
        # revision, so this split is streamed directly and no cache is used.
        print(f"[data] {split} cache disabled: the dataset revision is neither pinned "
              f"nor resolvable, so a cached slice could silently come from a different "
              f"upstream commit than the training stream", file=sys.stderr)
        report["materialized_cache_disabled_reason"] = "dataset revision unresolved"
        return stream_ds, None
    key = json.dumps(
        {
            "version": 3,
            "dataset": dataset_name,
            "dataset_config": dataset_config,
            "split": split,
            "dataset_revision": report.get("dataset_revision"),
            "dataset_revision_resolved": resolved_revision,
            "num_documents": report.get("num_documents"),
            "val_frac": report.get("val_frac"),
            "test_frac": report.get("test_frac"),
            "start": start,
            "count": count,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
    safe_name = str(dataset_name).replace("/", "__")
    cache_path = root / f"{safe_name}-{split}-{digest}.jsonl"
    meta_path = cache_path.with_name(cache_path.name + ".meta.json")
    if cache_path.exists():
        fingerprint = _read_cache_fingerprint(meta_path)
        cached_documents = None if fingerprint is None else fingerprint.get("documents")
        if (fingerprint is not None
                and fingerprint.get("cache_key") == key
                and _cache_matches(cache_path, fingerprint)
                and isinstance(cached_documents, int)
                and not isinstance(cached_documents, bool)
                and cached_documents == int(count)):
            report["materialized_cache_fingerprint"] = fingerprint
            return _JsonlDocumentStream(cache_path), cache_path
        # A cache without a matching sidecar, or one holding a different
        # number of documents than the declared split range, cannot be
        # trusted (truncated/edited payload, moved boundary, or a foreign
        # file under the same key): rebuild it from the upstream stream
        # rather than silently changing the held-out documents behind the
        # manifest's fingerprint.
        print(f"[data] held-out cache {cache_path} failed its fingerprint check "
              f"(missing/foreign .meta.json, changed content, or document count "
              f"{cached_documents} != {count}); rebuilding it", file=sys.stderr)
        report["materialized_cache_rebuilt"] = True
    root.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
    hasher = hashlib.sha256()
    n_documents = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for example in stream_ds:
                if isinstance(example, dict):
                    # Only `text` is read downstream; its type is preserved
                    # exactly (None / NaN / bytes / non-string included) so
                    # the cleaner's drop counters stay identical with and
                    # without the cache. Raw bytes are not directly
                    # JSON-serializable, so they travel base64-tagged and are
                    # restored as bytes by `_JsonlDocumentStream`.
                    text = example.get("text")
                    if isinstance(text, (bytes, bytearray)):
                        line = json.dumps(
                            {"text": base64.b64encode(bytes(text)).decode("ascii"),
                             "text_encoding": "base64"},
                            ensure_ascii=False) + "\n"
                    else:
                        line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                else:
                    line = "null\n"
                f.write(line)
                hasher.update(line.encode("utf-8"))
                n_documents += 1
        if n_documents != int(count):
            # The declared chronological slice could not be filled: the
            # upstream stream is shorter than its reported document count (or
            # the boundary was moved). Serving the short slice would silently
            # evaluate on a different range than the manifest claims.
            raise ValueError(
                f"the {split} split expected {count} documents from the upstream "
                f"stream but only {n_documents} were available; refusing to "
                f"materialize a truncated held-out cache. Check that "
                f"--num-documents matches the actual stream and that the dataset "
                f"revision is unchanged.")
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    fingerprint = {
        "path": str(cache_path),
        "version": 3,
        # The full identity that produced the payload. Size+sha alone only
        # prove that the sidecar describes *this file*, not that the file is
        # the split/dataset/revision the current key asked for: a foreign but
        # self-consistent cache copied under this path would otherwise pass.
        "cache_key": key,
        "dataset": dataset_name,
        "split": split,
        "dataset_revision_resolved": resolved_revision,
        "documents": n_documents,
        "sha256": hasher.hexdigest(),
        "bytes": cache_path.stat().st_size,
    }
    tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
    tmp_meta.write_text(json.dumps(fingerprint, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_meta, meta_path)
    try:
        st = cache_path.stat()
        if len(_VERIFIED_CACHES) >= 8:
            _VERIFIED_CACHES.clear()
        _VERIFIED_CACHES[str(cache_path)] = (fingerprint["sha256"], st.st_size, st.st_mtime_ns)
    except OSError:
        pass
    report["materialized_cache_fingerprint"] = fingerprint
    return _JsonlDocumentStream(cache_path), cache_path


def _read_cache_fingerprint(meta_path: Path):
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


_TOKENIZER_CACHE: dict = {}
_TOKENIZER_CACHE_MAX_ENTRIES = 8


def _cached_tokenizer(name: str, revision):
    """Process-wide `AutoTokenizer` cache keyed by (name, resolved revision).

    Checkpoint scans call `build_dataloader` once per checkpoint; without the
    cache each call repeats the tokenizer resolution (network/cache lookup and
    config parsing) even though the tokenizer identity is fixed by the model.
    The requested revision is resolved to a concrete commit before loading, so
    the tokenizer used to train and the one used to evaluate cannot come from
    different snapshots of a moving branch.  The cache is a plain dict keyed
    by the pinned name, bounded like the other scratch caches, and
    `pad_token_id` is still normalized per call.
    """
    requested = str(revision) if revision else None
    resolved = resolve_tokenizer_revision(name, requested)
    load_revision = resolved or requested
    key = (name, load_revision)
    tok = _TOKENIZER_CACHE.get(key)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, revision=load_revision)
        if len(_TOKENIZER_CACHE) >= _TOKENIZER_CACHE_MAX_ENTRIES:
            _TOKENIZER_CACHE.clear()
        _TOKENIZER_CACHE[key] = tok
    return tok


def build_dataloader(dataset_name: str, dataset_config: str | None, seq_len: int,
                     batch_size: int, tokenizer: str = "gpt2"):
    """Build the DataLoader for the split currently configured.

    Kept at five arguments on purpose - the split / cleaning knobs are read
    from `configure_data_options`; existing callers and test stubs keep
    working unchanged. The explicit generator and `worker_init_fn` make the
    loader's base seed - and therefore every worker's RNG - a function of the
    configured seed alone, not of how much global RNG the caller happened to
    consume before building it. The optional `dataset_revision` /
    `tokenizer_revision` options pin the mutable remote artifacts; a missing
    `dataset_revision` is resolved to the current upstream commit hash and
    both the request and the resolved hash are copied into the split report
    (data-version tracking), the latter also keying the held-out cache.
    """
    opts = validate_data_options(get_data_options())
    seq_len = _int_option("seq_len", seq_len, 1)
    batch_size = _int_option("batch_size", batch_size, 1)
    if (opts.get("exclude_group_file") or opts.get("exclude_test_group_file")) and not opts.get("group_key"):
        # Without the key the pipeline cannot extract a group value, so the
        # supplied ledger would be silently ignored and the held-out splits
        # would claim group hygiene they never applied.
        raise ValueError(
            "exclude_group_file/exclude_test_group_file require group_key: the "
            "group hash ledger is keyed by the value at --group-key, and without "
            "it no held-out document can be matched to a listed group")
    split = opts["split"]
    group_key = opts.get("group_key")
    # The validation->test ledgers only have a source side when a validation
    # split exists. With val_frac=0 the only boundary is train<->test, and
    # demanding the (unusable) second ledger would refuse a valid protocol.
    has_val_split = float(opts.get("val_frac") or 0.0) > 0.0
    if group_key and split in ("validation", "test"):
        missing_group_ledgers = []
        if not opts.get("exclude_group_file"):
            missing_group_ledgers.append(
                "--exclude-group-file (train<->held-out groups)")
        if split == "test" and has_val_split and not opts.get("exclude_test_group_file"):
            missing_group_ledgers.append(
                "--exclude-test-group-file (validation<->test groups)")
        if missing_group_ledgers:
            # `group_key` alone excludes nothing: `DocumentCleaner` only drops
            # cross-split groups when an earlier-side ledger lists them.
            # Proceeding would leave sibling documents of training/selection
            # groups inside this held-out split while the run still looks like
            # it enforces group-disjoint evaluation.
            raise ValueError(
                f"group_key '{group_key}' is set but the {split} split is "
                f"missing {missing_group_ledgers}: without those ledger(s) "
                f"documents whose group already appeared on an earlier side "
                f"would enter this split. Write them with "
                f"scripts/audit_data.py --group-key {group_key} "
                f"(--write-train-group-file / --write-val-group-file), or drop "
                f"--group-key if groups do not need to stay whole.")
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    tok = _cached_tokenizer(tokenizer, opts.get("tokenizer_revision"))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    # Resolve "latest"/branch/tag requests *before* opening the stream: the
    # concrete commit is what train, validation and test must all be read
    # from. Recording the resolved hash only after the load can describe a
    # snapshot other than the one actually consumed.
    requested_revision = opts.get("dataset_revision")
    resolved_revision = resolve_dataset_revision(dataset_name, requested_revision)
    load_revision = resolved_revision or requested_revision
    stream_ds = load_dataset(dataset_name, dataset_config, split="train",
                             streaming=True, revision=load_revision)
    stream_ds, report = select_stream_split(stream_ds, opts["split"], opts)
    # Pin "latest" to a concrete upstream commit before anything is cached or
    # trained on, and carry the hash into the split report (data-version
    # tracking). Degrades to None when the Hub is unreachable.
    report["dataset_revision_resolved"] = resolved_revision
    report["dataset_revision_loaded"] = load_revision
    # Group ledgers follow the same earlier-side-only rule: the training-group
    # ledger is applied to both held-out splits, the validation-group ledger
    # to the sealed test split only, and neither is ever applied to train.
    report["group_key"] = opts.get("group_key") or None
    applied_train_groups = opts.get("exclude_group_file") if opts["split"] != "train" else None
    applied_test_groups = (opts.get("exclude_test_group_file")
                           if opts["split"] == "test" and has_val_split else None)
    group_files = [p for p in (applied_train_groups, applied_test_groups) if p]
    group_exclusion_active = bool(opts.get("group_key") and group_files)
    exclude_group_hashes: set = set()
    for path in group_files:
        exclude_group_hashes |= load_hash_file(path)
    report["exclude_group_file"] = str(applied_train_groups) if applied_train_groups else None
    report["exclude_test_group_file"] = str(applied_test_groups) if applied_test_groups else None
    if group_files:
        report["exclude_group_files"] = [str(p) for p in group_files]
        report["exclude_groups"] = len(exclude_group_hashes)
    if opts["split"] in ("validation", "test"):
        if group_exclusion_active:
            # The materialized cache stores only `text`; the group field the
            # exclusion decision needs is gone by the time the cleaner runs on
            # a cached stream. Caching this split would therefore silently
            # serve documents whose group is listed in an earlier-side ledger.
            print(f"[data] {opts['split']} cache disabled: group exclusion on "
                  f"'{opts.get('group_key')}' needs the raw group field, which the "
                  f"text-only cache does not preserve", file=sys.stderr)
            report["materialized_cache_disabled_reason"] = (
                "group exclusion is active; the text-only cache cannot apply "
                "the group ledger")
        else:
            stream_ds, cache_path = _materialized_split_stream(
                stream_ds, dataset_name, dataset_config, opts["split"], report)
            if cache_path is not None:
                report["materialized_cache"] = str(cache_path)
    report["tokenizer"] = tokenizer
    report["tokenizer_revision"] = opts.get("tokenizer_revision")
    # The requested revision alone cannot pin a mutable upstream tokenizer:
    # record the concrete commit the load resolved to (None for a local path).
    # Some tokenizer objects do not expose `_commit_hash`, so retain the
    # resolved key from the same cache lookup instead of falling back to the
    # mutable branch/tag request.
    requested_tok_revision = opts.get("tokenizer_revision")
    resolved_tok_revision = (
        resolve_tokenizer_revision(tokenizer, requested_tok_revision)
        or requested_tok_revision)
    report["tokenizer_revision_resolved"] = (
        getattr(tok, "_commit_hash", None) or resolved_tok_revision)
    report["shuffle_buffer"] = int(opts["shuffle_buffer"]) if opts["split"] == "train" else 0
    # The train-document ledger is consumed by both held-out splits; the
    # validation-document ledger is consumed by the sealed test split only, so
    # a document cannot sit on both sides of the selection/final boundary
    # either.  Both are never applied to train (they are derived from earlier
    # splits; excluding them there would throw the split away).
    applied_train_ledger = opts.get("exclude_hash_file") if opts["split"] != "train" else None
    # The validation-document ledger has no source side when there is no
    # validation split; applying it to test would drop documents by hashes
    # that cannot correspond to this run's selection split.
    applied_test_ledger = (opts.get("exclude_test_hash_file")
                           if opts["split"] == "test" and has_val_split else None)
    exclude_files = [p for p in (applied_train_ledger, applied_test_ledger) if p]
    exclude_hashes: set = set()
    for path in exclude_files:
        exclude_hashes |= load_hash_file(path)
    report["exclude_hash_file"] = str(applied_train_ledger) if applied_train_ledger else None
    report["exclude_test_hash_file"] = str(applied_test_ledger) if applied_test_ledger else None
    if exclude_files:
        report["exclude_hash_files"] = [str(p) for p in exclude_files]
        report["exclude_hashes"] = len(exclude_hashes)
    cleaner = DocumentCleaner(
        dedup=opts["dedup"],
        dedup_max=opts["dedup_max"],
        min_doc_chars=opts["min_doc_chars"],
        max_doc_chars=opts["max_doc_chars"],
        exclude_hashes=exclude_hashes,
        group_key=opts.get("group_key"),
        exclude_group_hashes=exclude_group_hashes,
    )
    stream = PackedStream(
        stream_ds,
        tok,
        seq_len,
        cleaner=cleaner,
        shuffle_buffer=report["shuffle_buffer"],
        seed=opts["seed"],
    )
    generator = torch.Generator()
    generator.manual_seed(int(opts["seed"]))
    num_workers = max(0, int(opts.get("num_workers", 2) or 0))
    loader_kwargs: dict = {}
    if num_workers > 0:
        # Workers stay alive across epochs (no per-epoch tokenizer/cleaner
        # rebuild) and fetch ahead independently of the parent's consumption.
        loader_kwargs["persistent_workers"] = bool(opts.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = max(1, int(opts.get("prefetch_factor", 2) or 2))
    loader = DataLoader(stream, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=torch.cuda.is_available(),
                        worker_init_fn=seed_worker, generator=generator, **loader_kwargs)
    report["num_workers"] = num_workers
    loader.split_report = report
    return loader


__all__ = [
    "DEFAULT_DATA_OPTIONS",
    "DocumentCleaner",
    "PackedStream",
    "build_dataloader",
    "canonical_document_hash",
    "canonical_text",
    "configure_data_options",
    "document_digests",
    "document_hash",
    "extract_group_value",
    "get_data_options",
    "group_hash",
    "infer_num_documents",
    "load_document_stream",
    "load_hash_file",
    "resolve_dataset_revision",
    "resolve_tokenizer_revision",
    "select_stream_split",
    "split_document_boundaries",
    "validate_data_options",
]
