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

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    return select_stream_split(ds, split, options or get_data_options())


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

    `hashed` lets a caller that hashed the stripped text already (to hand the
    digest to the cleaner) share it instead of hashing the document twice.
    """
    if hashed is None:
        hashed = _dedup_hash(example)
    if hashed is None:
        return 0
    return int.from_bytes(hashed, "little") % n_workers


def load_hash_file(path) -> set:
    """Read a document-hash ledger written by `audit_data.py`.

    Accepts a JSON list of hex digests or whitespace/newline separated hex
    digests and returns the set of raw digest bytes.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        entries = json.loads(text)
    except json.JSONDecodeError:
        entries = None
    if not isinstance(entries, list):
        entries = text.split()
    return {bytes.fromhex(str(entry).strip()) for entry in entries}


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
        self.exclude_hashes = set(exclude_hashes) if exclude_hashes else set()
        self.seen: set = set()
        self.stats: Counter = Counter()

    def __call__(self, example, precomputed_hash: bytes | None = None):
        """Clean one example; `precomputed_hash` must be the digest of the
        stripped text and is reused instead of hashing the document again."""
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
                if h in self.seen:
                    self.stats["dropped_duplicate"] += 1
                    return None
                if len(self.seen) < self.dedup_max:
                    self.seen.add(h)
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
        self.buf: list = []
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
        buf = self.buf
        pos = self.pos
        need = self.seq_len + 1
        texts: list = []

        def drain():
            nonlocal pos
            while len(buf) - pos >= need:
                window = torch.tensor(buf[pos:pos + need], dtype=torch.long)
                x = window[:-1]
                y = window[1:]
                pos += self.seq_len
                yield (x, y)
            del buf[:pos]
            pos = 0

        def extend(batch):
            try:
                token_lists = self.tok(batch, add_special_tokens=False)["input_ids"]
            except Exception:
                token_lists = []
                for text in batch:
                    try:
                        token_lists.append(self.tok(text, add_special_tokens=False)["input_ids"])
                    except Exception:
                        self.cleaner.stats["dropped_tokenization_error"] += 1
            for ids in token_lists:
                if not ids:
                    continue
                buf.extend(ids)
                buf.append(self.tok.eos_token_id)

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
            if len(texts) < self.tokenize_batch and len(buf) - pos < need:
                continue
            extend(texts)
            texts = []
            yield from drain()
        if texts:
            extend(texts)
            yield from drain()
        self.pos = pos


def build_dataloader(dataset_name: str, dataset_config: str | None, seq_len: int,
                     batch_size: int, tokenizer: str = "gpt2"):
    """Build the DataLoader for the split currently configured.

    Kept at five arguments on purpose - the split / cleaning knobs are read
    from `configure_data_options`; existing callers and test stubs keep
    working unchanged. The explicit generator and `worker_init_fn` make the
    loader's base seed - and therefore every worker's RNG - a function of the
    configured seed alone, not of how much global RNG the caller happened to
    consume before building it.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    opts = get_data_options()
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    stream_ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    stream_ds, report = select_stream_split(stream_ds, opts["split"], opts)
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
    loader = DataLoader(stream, batch_size=batch_size, num_workers=2,
                        pin_memory=torch.cuda.is_available(),
                        worker_init_fn=seed_worker, generator=generator)
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
    "select_stream_split",
    "split_document_boundaries",
]
