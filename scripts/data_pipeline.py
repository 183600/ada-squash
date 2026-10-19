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
   texts, degenerate length outliers, exact duplicates and tokenizer
   failures are dropped by the same code on train, validation and test -
   always *after* the split.
5. **Train statistics only.**  Nothing here estimates a statistic (mean,
   variance, vocabulary, length) on validation or test data and feeds it back
   into training; the held-out splits are read-only for evaluation.
6. **Cross-split duplicates are dropped from the held-out side.**  Per-split
   dedup cannot see a document that occurs once in train and once in
   validation/test.  When an `exclude_hash_file` ledger (the hashes of the
   training documents, written by `scripts/audit_data.py
   --write-train-hash-file`) is supplied, any held-out document whose exact
   text is in that ledger is dropped from that held-out split - never from
   train itself (the ledger is derived from train, so excluding it there would
   throw the whole split away).

`build_dataloader` deliberately keeps its historical 5-argument call
signature (the split / cleaning knobs travel through
`configure_data_options`) so stubs that monkeypatch it in tests keep
working.
"""
import hashlib
import json
import math
import os
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
}

_active_data_options = dict(DEFAULT_DATA_OPTIONS)


def configure_data_options(**overrides) -> dict:
    """Set the split / cleaning options used by the next `build_dataloader`."""
    unknown = sorted(set(overrides) - set(DEFAULT_DATA_OPTIONS))
    if unknown:
        raise KeyError(f"unknown data-pipeline option(s): {unknown}")
    for key, value in overrides.items():
        _active_data_options[key] = value
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
    for name in ("train", "validation", "test"):
        try:
            split_info = splits.get(name)
        except AttributeError:
            split_info = splits[name] if name in splits else None
        n = getattr(split_info, "num_examples", None) if split_info is not None else None
        if isinstance(n, int) and n > 0:
            return n
    return None


_REVISION_CACHE: dict = {}


def resolve_dataset_revision(dataset_name: str, revision=None) -> str | None:
    """Resolve the requested dataset revision to the concrete upstream commit.

    `revision=None` means "latest on the default branch", which can move
    between runs; the request alone therefore cannot be reproduced after the
    fact. Resolving it to the current Hub commit hash and recording that hash
    in the split report / run manifest is what makes the trained-on data
    version auditable (and, because it enters the materialized-cache key, a
    moved upstream branch can no longer silently reuse the old cache). When
    the Hub is unreachable the resolution degrades to the requested value (or
    None) instead of failing the run.
    """
    if revision:
        return str(revision)
    key = ("revision", str(dataset_name))
    if key in _REVISION_CACHE:
        return _REVISION_CACHE[key]
    resolved = None
    try:
        from huggingface_hub import HfApi

        resolved = getattr(HfApi().dataset_info(dataset_name), "sha", None)
    except Exception:
        resolved = None
    _REVISION_CACHE[key] = resolved
    return resolved


def split_document_boundaries(num_documents: int, val_frac: float, test_frac: float) -> dict:
    """Contiguous, chronological boundaries over `num_documents` documents.

    The three ranges are `[train][validation][test]` in upstream order: train
    is the past, validation and test are the future.  They are disjoint by
    construction, so no document (group) can appear on two sides.
    """
    num_documents = int(num_documents)
    if num_documents <= 0:
        raise ValueError(f"num_documents must be positive, got {num_documents}")
    val_frac, test_frac = float(val_frac), float(test_frac)
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0):
        raise ValueError(f"val_frac/test_frac must be in [0, 1), got {val_frac}/{test_frac}")
    val_n = int(round(num_documents * val_frac))
    test_n = int(round(num_documents * test_frac))
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
    val_frac, test_frac = float(opts["val_frac"]), float(opts["test_frac"])
    num_documents = opts.get("num_documents")
    num_documents = int(num_documents) if num_documents is not None else infer_num_documents(ds)
    report = {
        "split": split,
        "num_documents": num_documents,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "dedup": bool(opts["dedup"]),
        "min_doc_chars": int(opts["min_doc_chars"]),
        "max_doc_chars": int(opts["max_doc_chars"]),
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
    """Raw (unpacked, uncleaned) document stream for one logical split."""
    from datasets import load_dataset

    opts = dict(DEFAULT_DATA_OPTIONS)
    if options:
        opts.update(options)
    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True,
                      revision=opts.get("dataset_revision"))
    stream, report = select_stream_split(ds, split, opts)
    report["dataset_revision_resolved"] = resolve_dataset_revision(
        dataset_name, opts.get("dataset_revision"))
    return stream, report


def document_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).digest()


def _dedup_hash(example) -> bytes | None:
    """Digest of the stripped document text, or None when the example has none."""
    raw = example.get("text") if isinstance(example, dict) else None
    if not isinstance(raw, str):
        return None
    return document_hash(raw.strip())


def _dedup_shard_owner(example, n_workers: int, hashed: bytes | None = None) -> int:
    """Which worker owns a document when dedup has to be global.

    Worker-local `seen` sets cannot see duplicates that the round-robin
    document shard handed to another worker, so with several loader workers
    identical documents are routed by their cleaned-text hash instead: every
    copy lands in the same worker and the per-worker set then drops all but
    the first occurrence, matching the single-process cleaner the audit uses.

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
    (what the writer now emits) is read incrementally so a million-document
    ledger never materializes a single multi-hundred-MB string; the legacy
    JSON-list form (which must be parsed in one piece) is still accepted.
    """
    out: set = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped[0] == "[":
                f.seek(0)
                entries = json.loads(f.read())
                if isinstance(entries, list):
                    out.update(bytes.fromhex(str(entry).strip()) for entry in entries)
                return out
            for token in line.split():
                out.add(bytes.fromhex(token))
    return out


class DocumentCleaner:
    """One cleaning policy, applied identically to every split, after the split.

    Drops (with counters, so a run can report what was removed):
    missing / non-string text (NaN or None from the upstream table), empty
    documents, degenerate short documents, length outliers, exact
    duplicates (within the split), and exact duplicates of documents listed
    in `exclude_hashes` (the training-document ledger used by held-out
    splits).  The duplicate set is bounded at `dedup_max` entries; beyond
    that the cleaner keeps dropping documents but stops recording new hashes.
    """

    def __init__(self, dedup: bool = True, dedup_max: int = 1_000_000,
                 min_doc_chars: int = 0, max_doc_chars: int = 0,
                 exclude_hashes: set | None = None):
        self.dedup = bool(dedup)
        self.dedup_max = int(dedup_max)
        self.min_doc_chars = int(min_doc_chars)
        self.max_doc_chars = int(max_doc_chars)
        # Held by reference: the ledger is read once and may be large, so the
        # caller must not mutate it after handing it over.
        self.exclude_hashes = exclude_hashes if exclude_hashes is not None else set()
        # `seen` stores the leading 64 bits of each digest as an int instead
        # of boxed `bytes` objects; collisions only matter at a negligible
        # 2^-64 scale and the full digest is still used for the cross-split
        # ledger lookup.
        self.seen: set = set()
        self.stats: Counter = Counter()

    def __call__(self, example, precomputed_hash: bytes | None = None):
        """Clean one example; `precomputed_hash`, when given, must be the
        digest of the stripped text."""
        self.stats["documents_read"] += 1
        if not isinstance(example, dict):
            self.stats["dropped_unreadable"] += 1
            return None
        text = example.get("text")
        if text is None or (isinstance(text, float) and math.isnan(text)):
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
        if self.dedup or self.exclude_hashes:
            h = precomputed_hash if precomputed_hash is not None else document_hash(text)
            if h in self.exclude_hashes:
                # Exact duplicate of a training document: keeping it in a
                # held-out split would leak that document across the split
                # boundary.
                self.stats["dropped_cross_split_duplicate"] += 1
                return None
            if self.dedup:
                key = int.from_bytes(h[:8], "little") if isinstance(h, (bytes, bytearray)) else h
                if key in self.seen:
                    self.stats["dropped_duplicate"] += 1
                    return None
                if len(self.seen) < self.dedup_max:
                    self.seen.add(key)
        self.stats["documents_kept"] += 1
        return text

    def report(self) -> dict:
        return dict(self.stats)


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
        self.seq_len = int(seq_len)
        self.cleaner = cleaner if cleaner is not None else DocumentCleaner()
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.seed = int(seed)
        # A one-dimensional long tensor plus integer offsets: the old Python
        # list boxed every token and rebuilt each window element by element
        # (`torch.tensor(list_slice)`) with an O(remaining) `del buf[:pos]`
        # shift per refill. The tokenizer returns Python ints, so the tensor
        # is filled by copying small per-document chunks; windows are cloned
        # out of it with a memcpy.
        self.buf: torch.Tensor | None = None
        self.fill = 0
        self.pos = 0

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        n_workers = worker_info.num_workers if worker_info is not None else 1
        # Duplicate removal is per epoch: documents repeated inside one pass
        # are dropped, while a later epoch over the same corpus still sees its
        # own documents.
        self.cleaner.seen = set()
        return self._shuffle(self._pack(worker_id, n_workers), worker_id)

    def _shuffle(self, stream, worker_id: int):
        if self.shuffle_buffer <= 0:
            yield from stream
            return
        import random as _random

        rng = _random.Random(self.seed * 1000003 + worker_id)
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
            except Exception:
                token_lists = []
                for text in batch:
                    try:
                        token_lists.append(self.tok(text, add_special_tokens=False)["input_ids"])
                    except Exception:
                        self.cleaner.stats["dropped_tokenization_error"] += 1
            eos = self.tok.eos_token_id
            for ids in token_lists:
                if not ids:
                    continue
                chunk = torch.tensor(ids, dtype=torch.long)
                reserve(chunk.numel() + 1)
                buf[fill:fill + chunk.numel()].copy_(chunk)
                fill += chunk.numel()
                buf[fill] = eos
                fill += 1

        shard_by_hash = self.cleaner.dedup and n_workers > 1
        i = 0
        for ex in self.ds:
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
    """Re-iterable raw-document stream backed by a materialized cache file."""

    def __init__(self, path: Path):
        self.path = str(path)

    def __iter__(self):
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


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
    is not silently reused afterwards. The written payload is fingerprinted
    (sha256 + document count) into a `.meta.json` sidecar and echoed into
    `report['materialized_cache_fingerprint']`, so a run manifest pins the
    concrete cached content, not only its name.
    """
    start, count = report.get("start"), report.get("count")
    if not start or not count:
        # start == 0 means there is no prefix to skip; nothing to gain.
        return stream_ds, None
    root = _split_cache_dir()
    if root is None:
        return stream_ds, None
    resolved_revision = report.get("dataset_revision_resolved")
    key = json.dumps(
        {
            "version": 2,
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
        if fingerprint is not None:
            report["materialized_cache_fingerprint"] = fingerprint
        return _JsonlDocumentStream(cache_path), cache_path
    root.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
    hasher = hashlib.sha256()
    n_documents = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for example in stream_ds:
                if isinstance(example, dict):
                    # Only `text` is read downstream; typing is preserved
                    # exactly (None / NaN / non-string included) so the
                    # cleaner's drop counters stay identical.
                    line = json.dumps({"text": example.get("text")}, ensure_ascii=False) + "\n"
                else:
                    line = "null\n"
                f.write(line)
                hasher.update(line.encode("utf-8"))
                n_documents += 1
        os.replace(tmp, cache_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    fingerprint = {
        "path": str(cache_path),
        "version": 2,
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
    report["materialized_cache_fingerprint"] = fingerprint
    return _JsonlDocumentStream(cache_path), cache_path


def _read_cache_fingerprint(meta_path: Path):
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


_TOKENIZER_CACHE: dict = {}
_TOKENIZER_CACHE_MAX_ENTRIES = 8


def _cached_tokenizer(name: str, revision):
    """Process-wide `AutoTokenizer` cache keyed by (name, revision).

    Checkpoint scans call `build_dataloader` once per checkpoint; without the
    cache each call repeats the tokenizer resolution (network/cache lookup and
    config parsing) even though the tokenizer identity is fixed by the model.
    The cache is a plain dict keyed by the revision-pinned name, bounded like
    the other scratch caches, and `pad_token_id` is still normalized per call.
    """
    key = (name, revision)
    tok = _TOKENIZER_CACHE.get(key)
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
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
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    opts = get_data_options()
    tok = _cached_tokenizer(tokenizer, opts.get("tokenizer_revision"))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    stream_ds = load_dataset(dataset_name, dataset_config, split="train",
                             streaming=True, revision=opts.get("dataset_revision"))
    stream_ds, report = select_stream_split(stream_ds, opts["split"], opts)
    # Pin "latest" to a concrete upstream commit before anything is cached or
    # trained on, and carry the hash into the split report (data-version
    # tracking). Degrades to None when the Hub is unreachable.
    report["dataset_revision_resolved"] = resolve_dataset_revision(
        dataset_name, opts.get("dataset_revision"))
    if opts["split"] in ("validation", "test"):
        stream_ds, cache_path = _materialized_split_stream(
            stream_ds, dataset_name, dataset_config, opts["split"], report)
        if cache_path is not None:
            report["materialized_cache"] = str(cache_path)
    report["tokenizer"] = tokenizer
    report["shuffle_buffer"] = int(opts["shuffle_buffer"]) if opts["split"] == "train" else 0
    exclude_hashes: set = set()
    exclude_file = opts.get("exclude_hash_file")
    if exclude_file:
        if opts["split"] == "train":
            # The ledger is the set of training-document hashes; applying it
            # to the train split would drop every document.  Stay silent on
            # train and let validation/test consume it.
            report["exclude_hash_file"] = None
        else:
            exclude_hashes = load_hash_file(exclude_file)
            report["exclude_hash_file"] = str(exclude_file)
            report["exclude_hashes"] = len(exclude_hashes)
    cleaner = DocumentCleaner(
        dedup=opts["dedup"],
        dedup_max=opts["dedup_max"],
        min_doc_chars=opts["min_doc_chars"],
        max_doc_chars=opts["max_doc_chars"],
        exclude_hashes=exclude_hashes,
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
    "configure_data_options",
    "document_hash",
    "get_data_options",
    "infer_num_documents",
    "load_document_stream",
    "load_hash_file",
    "resolve_dataset_revision",
    "select_stream_split",
    "split_document_boundaries",
]
