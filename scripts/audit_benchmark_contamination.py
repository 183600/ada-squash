#!/usr/bin/env python3
"""Audit downstream-benchmark contamination of the pretraining corpus.

`scripts/audit_data.py` proves that the corpus's own train/validation/test
splits do not leak into each other.  This script asks the complementary
question that internal split hygiene cannot answer: does the *training* text
contain the questions, choices, answers or solutions of the benchmarks the
model will later be evaluated on?  Such overlap inflates a downstream score
regardless of how clean the corpus's internal split is.

Method
------
1. Load one or more benchmark sets (local JSON/JSONL files or the built-in
   MMLU / GSM8K / HumanEval / DROP presets), pinning their revision where the
   source supports it.
2. Tokenize the evaluated text with the *same* tokenizer (and revision) the
   run uses, and build an exact token n-gram index (default 13-grams, the
   usual LM-decontamination primitive).
3. Stream the training split through the same split/cleaning policy as
   training and scan each cleaned document against that index.
4. Report per-benchmark overlap rates with a one-sided Wilson upper bound,
   and optionally write the canonical digests of contaminated training
   documents to a ledger.  Pass that ledger to
   ``train_gpt2.py --exclude-train-hash-file`` so the training split - and
   only the training split - drops them.

The scan is a measurement, not a proof.  A bounded ``--n-docs`` sample is
reported as a prefix bound, and a zero-hit scan of a small sample is still
compatible with a non-zero contamination rate.

Example
-------
    python scripts/audit_benchmark_contamination.py \\
        --benchmark-preset mmlu --benchmark-preset gsm8k \\
        --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT \\
        --dataset-revision <commit> --tokenizer gpt2 --tokenizer-revision <commit> \\
        --n-docs 0 --max-contamination-rate 0.0005 \\
        --write-train-exclusion-file runs/ledgers/benchmark_train_exclude.hash \\
        --output runs/benchmark_contamination.json

Then every training arm adds:

    --exclude-train-hash-file runs/ledgers/benchmark_train_exclude.hash

so the comparison still sees the same data protocol on both sides.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The package imports below are deliberately lightweight: `datasets` and
# `transformers` are imported inside the loaders so `--help` and the unit-test
# suite work in the CPU-only CI environment.
from adasquash.decontamination import (
    DEFAULT_NGRAM_SIZE,
    benchmark_text,
    build_ngram_index,
    contamination_ledger_hashes,
    contamination_protocol,
    contamination_scan_bound,
    protocol_digest,
    scan_token_documents,
)
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest
from scripts.data_pipeline import (
    DocumentCleaner,
    canonical_document_hash as pipeline_canonical_document_hash,
    load_document_stream,
)

# Presets only fix the source identifiers; the user still has to pass a
# revision if the benchmark snapshot should be reproducible.  A preset that
# silently downloads "latest" is recorded as such in the report.
_PRESETS = {
    "mmlu": {"dataset": "cais/mmlu", "config": "all", "split": "test"},
    "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test"},
    "humaneval": {"dataset": "openai/openai_humaneval", "config": None, "split": "test"},
    "drop": {"dataset": "ucinlp/drop", "config": None, "split": "validation"},
}


def _load_benchmark_file(path: Path) -> list:
    """Read a benchmark file: a JSON list/JSON object or JSONL records."""
    if not path.exists():
        raise ValueError(f"benchmark file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    # JSONL is the common case for preprocessed benchmarks, but a pretty
    # printed JSON array starts with '[' too.  Try JSON first when the whole
    # file parses; fall back to one-record-per-line.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # A single-record JSON file (or a one-line JSONL file, which is
        # indistinguishable to `json.loads`) is one record, not a wrapper.
        record_fields = set(data) & {
            "text", "question", "prompt", "problem", "passage", "context",
            "article", "premise", "sentence", "choices", "options",
            "answer", "answers", "solution", "canonical_solution", "test",
            "code", "output", "target",
        }
        if record_fields:
            records = [data]
        else:
            # A wrapper such as {"data": [...]} is accepted only when it has
            # a single obvious record list; arbitrary mappings would hide the
            # actual examples.
            keys = [key for key, value in data.items() if isinstance(value, list)]
            if len(keys) != 1:
                raise ValueError(
                    f"benchmark file {path} is a JSON object without exactly one "
                    f"record list (keys with lists: {keys})")
            records = data[keys[0]]
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"benchmark file {path} line {line_number} is not JSON: {exc}") from None
            records.append(record)
    if not isinstance(records, list) or not records:
        raise ValueError(f"benchmark file {path} contains no records")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"benchmark file {path} record {index} is not a JSON object")
        # Validate the text extraction now, so a malformed record fails at
        # load time instead of after the expensive corpus scan starts.
        benchmark_text(record)
    return records


def _load_preset(name: str, revision=None, split=None) -> tuple[list, dict]:
    """Load a built-in benchmark preset through HuggingFace datasets."""
    preset = _PRESETS.get(name)
    if preset is None:
        raise ValueError(f"unknown benchmark preset {name!r}; choose from {sorted(_PRESETS)}")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise SystemExit(
            "benchmark presets need the `datasets` package; install "
            "requirements.txt or pass --benchmark name=path.jsonl") from exc
    kwargs = {"split": split or preset["split"], "revision": revision}
    if preset["config"] is not None:
        dataset = load_dataset(preset["dataset"], preset["config"], **kwargs)
    else:
        dataset = load_dataset(preset["dataset"], **kwargs)
    records = [dict(row) for row in dataset]
    meta = {
        "source": "preset",
        "preset": name,
        "dataset": preset["dataset"],
        "config": preset["config"],
        "split": split or preset["split"],
        "requested_revision": revision,
        "fingerprint": getattr(dataset, "_fingerprint", None),
    }
    return records, meta


def _parse_benchmark_spec(spec: str):
    """Parse ``name=path`` or a bare preset name."""
    if "=" in spec:
        name, value = spec.split("=", 1)
        name, value = name.strip(), value.strip()
        if not name or not value:
            raise ValueError(f"benchmark spec must be name=path, got {spec!r}")
        return name, value
    if spec.strip() in _PRESETS:
        return spec.strip(), None
    raise ValueError(
        f"benchmark spec {spec!r} is neither a preset {sorted(_PRESETS)} nor "
        "name=path.jsonl")


def _load_benchmarks(specs, presets, revision, split):
    """Return ``(records_by_benchmark, sources)`` with unique names."""
    records_by_benchmark: dict = {}
    sources: dict = {}
    for spec in specs:
        name, path = _parse_benchmark_spec(spec)
        if name in records_by_benchmark:
            raise ValueError(f"duplicate benchmark name {name!r}")
        records_by_benchmark[name] = _load_benchmark_file(Path(path))
        sources[name] = {
            "source": "file",
            "path": str(Path(path).resolve()),
            "fingerprint": file_fingerprint(path),
        }
    for name in presets:
        if name in records_by_benchmark:
            raise ValueError(f"duplicate benchmark name {name!r}")
        records, meta = _load_preset(name, revision=revision, split=split)
        records_by_benchmark[name] = records
        sources[name] = meta
    if not records_by_benchmark:
        raise ValueError("at least one --benchmark or --benchmark-preset is required")
    return records_by_benchmark, sources


def _validate_token_row(row) -> list:
    """Validate one token-id row returned by a tokenizer."""
    if not isinstance(row, list):
        raise ValueError("tokenizer returned a non-list token row")
    for token in row:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"tokenizer returned a non-integer token id {token!r}")
    return row


def _tokenize_texts(texts, tokenizer) -> list:
    """Tokenize a batch of strings, validating the row count and ids."""
    encoded = tokenizer(list(texts), add_special_tokens=False)
    rows = encoded.get("input_ids") if isinstance(encoded, dict) else None
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise ValueError(
            f"tokenizer returned {None if rows is None else len(rows)} rows for "
            f"{len(texts)} text documents")
    return [_validate_token_row(row) for row in rows]


def _iter_training_documents(args):
    """Return the cleaned training stream, its split report and the cleaner.

    The split selector and cleaner are the training pipeline's own; using a
    local reimplementation here would audit a different corpus than the run
    trains on.  Documents are yielded lazily so a full-corpus scan does not
    hold the corpus in memory.
    """
    options = {
        "split": "train",
        "val_frac": args.val_frac,
        "test_frac": args.test_frac,
        "num_documents": args.num_documents,
        "dedup": args.dedup,
        "min_doc_chars": args.min_doc_chars,
        "max_doc_chars": args.max_doc_chars,
        "dataset_revision": args.dataset_revision,
        "tokenizer_revision": args.tokenizer_revision,
    }
    stream, report = load_document_stream(
        args.dataset, args.dataset_config, "train", options)
    cleaner = DocumentCleaner(
        dedup=args.dedup,
        min_doc_chars=args.min_doc_chars,
        max_doc_chars=args.max_doc_chars,
    )
    return stream, report, cleaner


def _write_ledger(path: Path, hashes: set, *, exhaustive: bool, n_docs: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for digest in sorted(hashes):
            f.write(digest.hex() + "\n")
    info = {
        "path": str(path),
        "n_hashes": len(hashes),
        "exhaustive": exhaustive,
        "n_docs_limit": n_docs,
        "purpose": "training-side benchmark decontamination",
    }
    if not exhaustive:
        print(
            f"[decontam] WARNING: the exclusion ledger covers only the first "
            f"{n_docs} scanned training documents; contamination outside that "
            f"prefix is not removed. Re-run with --n-docs 0 for a full ledger.",
            file=sys.stderr)
    return info


def _scan_training(args, tokenizer, index):
    """Stream cleaned training documents through the benchmark index.

    The split selector and cleaner are the training pipeline's own; using a
    local reimplementation here would audit a different corpus than the run
    trains on.  Documents are tokenized in small batches but yielded one at a
    time so the scan can report a stable index and a canonical digest.
    """
    stream, split_report, cleaner = _iter_training_documents(args)
    documents_scanned = 0
    documents_read = 0
    tokenization_errors = 0
    capped = False
    pending_texts: list = []
    pending_indices: list = []

    def tokenize_pending():
        """Tokenize and yield one pending batch, aligned with its texts."""
        nonlocal pending_texts, pending_indices, tokenization_errors
        texts, indices = pending_texts, pending_indices
        pending_texts, pending_indices = [], []
        try:
            rows = _tokenize_texts(texts, tokenizer)
        except Exception:
            rows = []
            for text in texts:
                try:
                    row = tokenizer(text, add_special_tokens=False)["input_ids"]
                    rows.append(_validate_token_row(row))
                except Exception:
                    tokenization_errors += 1
                    rows.append(None)
        for position, text, row in zip(indices, texts, rows):
            if row is None:
                continue
            yield {
                "doc_index": position,
                "tokens": row,
                "text": text,
                "canonical_digest": pipeline_canonical_document_hash(text),
            }

    def documents():
        nonlocal documents_scanned, documents_read, capped
        for example in stream:
            if args.n_docs and documents_scanned >= args.n_docs:
                capped = True
                break
            documents_read += 1
            text = cleaner(example)
            if text is None:
                continue
            if args.n_docs and documents_scanned >= args.n_docs:
                capped = True
                break
            pending_texts.append(text)
            pending_indices.append(documents_scanned)
            documents_scanned += 1
            if len(pending_texts) >= args.tokenize_batch_size:
                yield from tokenize_pending()
        if pending_texts:
            yield from tokenize_pending()

    report_scan = scan_token_documents(
        documents(),
        index,
        max_examples=args.max_examples,
        # A bounded n_docs scan is exhaustive only when the stream ended
        # before the cap; otherwise the result describes a prefix.
        exhaustive=not capped,
    )
    report_scan["documents_read"] = documents_read
    report_scan["tokenization_errors"] = tokenization_errors
    report_scan["n_docs_limit"] = int(args.n_docs)
    report_scan["cleaning_stats"] = cleaner.report()
    return report_scan, split_report


def _validate_args(args) -> None:
    for flag, value in (("--n-docs", args.n_docs),
                        ("--tokenize-batch-size", args.tokenize_batch_size),
                        ("--max-examples", args.max_examples),
                        ("--ngram-size", args.ngram_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SystemExit(f"{flag} must be a non-negative integer")
    if args.tokenize_batch_size < 1:
        raise SystemExit("--tokenize-batch-size must be >= 1")
    if args.ngram_size < 1:
        raise SystemExit("--ngram-size must be >= 1")
    for flag, value in (("--val-frac", args.val_frac), ("--test-frac", args.test_frac)):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise SystemExit(f"{flag} must be finite and in [0, 1)")
    if args.val_frac + args.test_frac >= 1.0:
        raise SystemExit("--val-frac + --test-frac must leave a training stream")
    if args.max_contamination_rate is not None:
        if not (math.isfinite(args.max_contamination_rate)
                and 0.0 <= args.max_contamination_rate <= 1.0):
            raise SystemExit("--max-contamination-rate must be finite and in [0, 1]")
    if args.min_doc_chars < 0 or args.max_doc_chars < 0:
        raise SystemExit("--min-doc-chars/--max-doc-chars must be >= 0")
    if args.max_doc_chars and args.min_doc_chars > args.max_doc_chars:
        raise SystemExit("--min-doc-chars cannot exceed --max-doc-chars")
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Audit and optionally remediate downstream-benchmark "
                    "contamination of the training corpus.")
    p.add_argument("--benchmark", action="append", default=[],
                   help="benchmark source as name=path.jsonl (repeatable); "
                        "records may hold a prebuilt 'text' field or "
                        "question/choices/answer-style fields")
    p.add_argument("--benchmark-preset", action="append", default=[],
                   choices=sorted(_PRESETS),
                   help="built-in benchmark preset (repeatable): "
                        + ", ".join(sorted(_PRESETS)))
    p.add_argument("--benchmark-revision", default=None,
                   help="revision/commit requested for every preset (recorded; "
                        "None means the upstream default branch)")
    p.add_argument("--benchmark-split", default=None,
                   help="override the evaluation split of every preset (default: "
                        "the preset's test/validation split)")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--dataset-revision", default=None,
                   help="dataset revision to audit; should be the run's resolved commit")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--tokenizer-revision", default=None)
    p.add_argument("--val-frac", type=float, default=0.005)
    p.add_argument("--test-frac", type=float, default=0.005)
    p.add_argument("--num-documents", type=int, default=None,
                   help="total documents in the stream; required when the "
                        "dataset reports no size")
    p.add_argument("--n-docs", type=int, default=2000,
                   help="cleaned training documents to scan (0 = whole training "
                        "split, needed for an exhaustive ledger)")
    p.add_argument("--min-doc-chars", type=int, default=0)
    p.add_argument("--max-doc-chars", type=int, default=0)
    p.add_argument("--no-dedup", dest="dedup", action="store_false")
    p.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE,
                   help=f"token n-gram size for exact overlap (default {DEFAULT_NGRAM_SIZE})")
    p.add_argument("--tokenize-batch-size", type=int, default=64)
    p.add_argument("--max-examples", type=int, default=100,
                   help="contaminated-document examples kept in the report "
                        "(counts and hashes are never truncated)")
    p.add_argument("--max-contamination-rate", type=float, default=None,
                   help="pre-registered tolerable fraction of scanned training "
                        "documents overlapping any benchmark; fail when the "
                        "one-sided confidence upper bound exceeds it")
    p.add_argument("--fail-on-any-contamination", action="store_true",
                   help="fail when any scanned training document overlaps a "
                        "benchmark, even below --max-contamination-rate")
    p.add_argument("--write-train-exclusion-file", default=None,
                   help="write the canonical digests of contaminated training "
                        "documents here; pass it to train_gpt2.py "
                        "--exclude-train-hash-file")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="confidence level of the contamination upper bound")
    p.add_argument("--output", default="runs/benchmark_contamination.json")
    p.add_argument("--hypothesis", default=None,
                   help="the scientific claim this audit supports; recorded in "
                        "the report manifest")
    args = p.parse_args(argv)
    _validate_args(args)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise SystemExit(
            "audit_benchmark_contamination needs the `transformers` package; "
            "install requirements.txt") from exc

    records_by_benchmark, sources = _load_benchmarks(
        args.benchmark, args.benchmark_preset, args.benchmark_revision,
        args.benchmark_split)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision)
    tokenized = {}
    for name, records in records_by_benchmark.items():
        texts = [benchmark_text(record) for record in records]
        tokenized[name] = _tokenize_texts(texts, tokenizer)
    index = build_ngram_index(tokenized, ngram_size=args.ngram_size)
    scan, split_report = _scan_training(args, tokenizer, index)
    bound = contamination_scan_bound(
        scan, confidence=args.confidence,
        max_rate=args.max_contamination_rate)
    # The ledger is deliberately derived from the scan's canonical hashes,
    # not from the text again: the training pipeline's ledger lookup uses
    # exactly those digests.
    ledger_hashes = contamination_ledger_hashes(scan)
    ledger_info = None
    if args.write_train_exclusion_file:
        ledger_info = _write_ledger(
            Path(args.write_train_exclusion_file), ledger_hashes,
            exhaustive=scan["exhaustive"], n_docs=args.n_docs)

    # Prefer the concrete revisions the loaders resolved over the mutable
    # branch/tag the user requested: a protocol digest built from "latest"
    # would be the same string for two different corpus snapshots.
    resolved_dataset = (split_report.get("dataset_revision_resolved")
                        or args.dataset_revision)
    resolved_tokenizer = (getattr(tokenizer, "_commit_hash", None)
                          or args.tokenizer_revision)
    protocol = contamination_protocol(
        ngram_size=args.ngram_size,
        benchmarks=records_by_benchmark,
        tokenizer=args.tokenizer,
        tokenizer_revision=resolved_tokenizer,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        dataset_revision=resolved_dataset,
        split_fractions={"val_frac": args.val_frac,
                         "test_frac": args.test_frac},
    )
    protocol["requested_dataset_revision"] = args.dataset_revision
    protocol["requested_tokenizer_revision"] = args.tokenizer_revision
    # The digest covers the measurement geometry and benchmark content, not
    # the local source paths or the derived index summary (those are recorded
    # separately so two machines can compare the digest of the same audit).
    protocol["digest"] = protocol_digest(protocol)
    protocol["digest_scope"] = (
        "core protocol fields: ngram_size, normalization, benchmark content, "
        "tokenizer, resolved/requested dataset revisions and split fractions")
    protocol["benchmark_sources"] = sources
    protocol["index"] = index.summary()
    problems = list(bound.get("problems") or [])
    if args.fail_on_any_contamination and scan["contaminated_documents"]:
        problems.append(
            f"{scan['contaminated_documents']} training document(s) overlap a "
            "downstream benchmark; --fail-on-any-contamination was requested")
    if scan["documents_scanned"] == 0:
        problems.append(
            "no cleaned training document was scanned; check --n-docs, the split "
            "fraction and the dataset revision")
    # A benchmark whose examples are all shorter than the n-gram size cannot
    # contribute a single match.  Say so instead of reporting a clean result.
    short = []
    for name, row in index.summary()["benchmarks"].items():
        if row["documents"] and row["distinct_ngrams"] == 0:
            short.append(name)
    if short:
        problems.append(
            "benchmark(s) " + ", ".join(sorted(short))
            + f" have no {args.ngram_size}-token n-gram (every example is shorter "
            f"than {args.ngram_size} tokens), so an n-gram scan cannot detect "
            "their contamination; lower --ngram-size or inspect them separately")
    report = {
        "schema": "adasquash.benchmark_contamination.v1",
        "hypothesis": args.hypothesis,
        "status": "failed" if problems else "finished",
        "protocol": protocol,
        "benchmarks": {
            name: {
                "documents": len(records),
                "tokens": sum(len(row) for row in tokenized[name]),
                "distinct_ngrams": index.summary()["benchmarks"][name]["distinct_ngrams"],
                "documents_without_ngrams": index.summary()["benchmarks"][name][
                    "documents_without_ngrams"],
                "source": sources[name],
            }
            for name, records in records_by_benchmark.items()
        },
        "training_split": split_report,
        "scan": scan,
        "scan_bound": bound,
        "train_exclusion_ledger": ledger_info,
        "problems": problems,
        "run_metadata": collect_run_metadata(extra={"hypothesis": args.hypothesis}),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    manifest = {
        "run_metadata": report["run_metadata"],
        "hypothesis": args.hypothesis,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "tokenizer": args.tokenizer,
        "options": vars(args),
        "protocol": protocol,
        "scan": scan,
        "scan_bound": bound,
        "train_exclusion_ledger": ledger_info,
        "problems": problems,
    }
    if problems:
        manifest["status"] = "failed"
    else:
        manifest = finish_metadata(
            manifest, artifacts={"benchmark_contamination_report": file_fingerprint(out_path)})
    save_run_manifest(out_path.with_name("benchmark_contamination_manifest.json"), manifest)

    print(f"[decontam] scanned {scan['documents_scanned']} training document(s), "
          f"{scan['tokens_scanned']} tokens, against {len(records_by_benchmark)} benchmark(s)")
    def _bound_text(value):
        return "undefined" if value is None else f"{value:.4%}"

    if scan["documents_scanned"]:
        print(f"[decontam] overlap: {scan['contaminated_documents']}/"
              f"{scan['documents_scanned']} scanned training documents "
              f"({scan['contaminated_documents'] / scan['documents_scanned']:.4%}); "
              f"one-sided {args.confidence:.1%} bound "
              f"{_bound_text(bound['upper_bound'])}"
              f" ({bound['scope']})")
    for name, row in sorted(bound["by_benchmark"].items()):
        print(f"[decontam]   {name}: {row['contaminated_documents']}/"
              f"{row['documents']} documents, upper bound "
              f"{_bound_text(row['upper_bound'])}")
    if ledger_info:
        print(f"[decontam] wrote {ledger_info['n_hashes']} training-exclusion "
              f"hash(es) to {ledger_info['path']}")
    print(f"[decontam] saved to {out_path}")
    if problems:
        for problem in problems:
            print(f"[decontam] PROBLEM: {problem}", file=sys.stderr)
        return 1
    if scan["contaminated_documents"]:
        print(f"[decontam] audit completed with "
              f"{scan['contaminated_documents']} contaminated training document(s); "
              "no pre-registered threshold was exceeded (use "
              "--fail-on-any-contamination or --max-contamination-rate to gate)")
    else:
        print("[decontam] no benchmark overlap found in the scanned training documents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
