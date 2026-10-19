from __future__ import annotations
"""Audit the leakage-safety of the streaming data pipeline.

Reads a bounded sample of the train / validation / test splits produced by
`scripts/data_pipeline.py` and reports

* cleaning coverage: how many documents were dropped (and why) by the exact
  same cleaner the trainer uses - missing / NaN text, empty documents,
  degenerate short documents, length outliers, exact duplicates and
  tokenizer failures;
* value sanity: token ids out of the tokenizer's range, non-finite length
  statistics (NaN / Inf), duplicate rates per split;
* split hygiene: exact duplicate documents shared between any two splits -
  train vs held-out *and* validation vs test (a duplicate in two splits is a
  leak no matter how the split was drawn, and a val/test duplicate makes the
  selection split peek at the sealed final split);
* chronology: train/validation/test must be contiguous slices of the upstream
  order, so no document can sit on two sides and no future document can
  precede the training slice;
* distribution shift: total-variation distance between token frequency
  distributions of train vs validation / test, plus document-length means and
  3xIQR length-outlier counts.

`--write-train-hash-file` emits the training-document hash ledger consumed by
`--exclude-hash-file` in the trainers, so held-out splits drop any document
whose exact text also occurs in train; `--write-val-hash-file` emits the
validation ledger consumed by `--exclude-test-hash-file`, so the sealed test
also drops documents duplicated in the selection split. A full ledger requires
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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest
from scripts.data_pipeline import DocumentCleaner, document_hash, load_document_stream


def _collect_split(name, dataset, dataset_config, tokenizer, options, n_docs, min_doc_chars, max_doc_chars, dedup, known_hashes=None, known_hashes_extra=None, collect_hashes=True):
    stream, report = load_document_stream(dataset, dataset_config, name, options)
    cleaner = DocumentCleaner(dedup=dedup, min_doc_chars=min_doc_chars, max_doc_chars=max_doc_chars)
    # One machine-word int per document.
    lengths = array('q')
    # Token ids are counted in a fixed `vocab_size + 1` numpy array, one
    # `np.bincount` per tokenized document. Documents are tokenized in
    # batches, with a per-document fallback so one bad document is still
    # counted (and skipped) instead of losing the whole batch. Ids that
    # cannot go through bincount (negative) use the Python-counter path.
    counts = None
    text_batch: list = []
    legacy_tokens: Counter = Counter()
    hashes: set = set()
    overlap = 0
    overlap_extra = 0
    tokenizer_errors = 0
    vocab_size = int(getattr(tokenizer, 'vocab_size', 0) or 0)

    def _add_ids(ids) -> None:
        nonlocal counts
        try:
            arr = np.asarray(ids, dtype=np.int64)
        except (TypeError, ValueError):
            legacy_tokens.update(ids)
            return
        if arr.size == 0:
            return
        if arr.min() < 0:
            # np.bincount rejects negatives; a tokenizer that emits them
            # still gets counted here (and flagged below).
            legacy_tokens.update(arr.tolist())
            return
        bc = np.bincount(arr)
        if counts is None:
            counts = np.zeros(max(bc.shape[0], vocab_size + 1), dtype=np.int64)
        elif bc.shape[0] > counts.shape[0]:
            # Grow on first sight of an out-of-vocabulary id so it can still
            # be reported by `token_ids_out_of_range`.
            counts = np.concatenate(
                [counts, np.zeros(bc.shape[0] - counts.shape[0], dtype=np.int64)])
        counts[:bc.shape[0]] += bc

    def _flush_tokens() -> None:
        nonlocal tokenizer_errors
        if not text_batch:
            return
        try:
            enc = tokenizer(text_batch, add_special_tokens=False)['input_ids']
        except Exception:
            enc = None
        if enc is None:
            for t in text_batch:
                try:
                    ids = tokenizer(t, add_special_tokens=False)['input_ids']
                except Exception:
                    tokenizer_errors += 1
                    continue
                _add_ids(ids)
        else:
            for ids in enc:
                _add_ids(ids)
        text_batch.clear()

    for example in stream:
        if n_docs and len(lengths) >= n_docs:
            break
        text = cleaner(example)
        if text is None:
            # Count tokenizer failures too by trying only kept-looking docs is
            # impossible here; the cleaner's own counters cover the rest.
            continue
        lengths.append(len(text))
        h = document_hash(text)
        if collect_hashes:
            hashes.add(h)
        if known_hashes is not None and h in known_hashes:
            overlap += 1
        if known_hashes_extra is not None and h in known_hashes_extra:
            overlap_extra += 1
        text_batch.append(text)
        if len(text_batch) >= 256:
            _flush_tokens()
    _flush_tokens()
    if counts is not None:
        tokens = Counter({int(t): int(c) for t, c in enumerate(counts) if c})
    else:
        tokens = Counter()
    tokens.update(legacy_tokens)
    stats = cleaner.report()
    # Documents actually read from the stream, including the ones the cleaner
    # dropped; `len(lengths)` would always equal `documents_kept`.
    stats['documents_sampled'] = int(stats.get('documents_read', len(lengths)))
    # Whether the counters cover the whole split (--n-docs 0) or a prefix: with
    # a bounded sample, `duplicate_overlap_with_train = 0` is a lower bound on
    # the true overlap, not evidence that the splits are disjoint.
    stats['exhaustive'] = not n_docs
    stats['duplicate_overlap_with_train'] = overlap if known_hashes is not None else None
    stats['duplicate_overlap_with_validation'] = overlap_extra if known_hashes_extra is not None else None
    stats['duplicate_overlap_is_lower_bound'] = bool(n_docs) if (known_hashes is not None or known_hashes_extra is not None) else None
    if lengths:
        stats['doc_chars_mean'] = sum(lengths) / len(lengths)
        stats['doc_chars_min'] = min(lengths)
        stats['doc_chars_max'] = max(lengths)
        # Length outliers by the 3 x IQR fence (only a report; the pipeline's
        # min/max_doc_chars knobs are the actual cleaning policy).
        ordered = sorted(lengths)
        q1 = ordered[len(ordered) // 4]
        q3 = ordered[(3 * len(ordered)) // 4]
        iqr = q3 - q1
        lower, upper = max(0, q1 - 3 * iqr), q3 + 3 * iqr
        stats['doc_chars_q1'] = q1
        stats['doc_chars_q3'] = q3
        stats['length_outliers_3iqr'] = sum(1 for v in lengths if v < lower or v > upper)
    else:
        stats['doc_chars_mean'] = stats['doc_chars_min'] = stats['doc_chars_max'] = None
        stats['doc_chars_q1'] = stats['doc_chars_q3'] = None
        stats['length_outliers_3iqr'] = None
    stats['tokenizer_errors'] = tokenizer_errors
    stats['token_ids_out_of_range'] = sum(c for t, c in tokens.items() if not 0 <= t < getattr(tokenizer, 'vocab_size', 2 ** 31))
    stats['distinct_token_ids'] = len(tokens)
    stats['finite_lengths'] = bool(all(math.isfinite(v) for v in (stats['doc_chars_mean'] or 0.0, stats['doc_chars_min'] or 0.0, stats['doc_chars_max'] or 0.0)))
    return stats, tokens, report, hashes


def _chronology_problems(reports: dict) -> list:
    """Verify the three split reports tile the stream in time order.

    train must start at 0, validation must start where train ends and test
    must start where validation ends; anything else means a document could be
    counted on two sides (or the split is no longer chronological).
    """
    train, validation, test = (reports.get(name) for name in ('train', 'validation', 'test'))
    if not all(isinstance(r, dict) and isinstance(r.get('start'), int) and isinstance(r.get('count'), int) for r in (train, validation, test)):
        return []
    train_end = train['start'] + train['count']
    validation_end = validation['start'] + validation['count']
    if train['start'] != 0 or validation['start'] != train_end or test['start'] != validation_end:
        return [f"split ranges are not contiguous chronological slices: train=[{train['start']},{train_end}) validation=[{validation['start']},{validation_end}) test=[{test['start']},{test['start']+test['count']})"]
    return []


def _write_hash_ledger(path, hashes: set, n_docs: int, split_name: str):
    """Write a sorted one-digest-per-line ledger and describe its coverage.

    `load_hash_file` accepts this whitespace-separated form (and also reads
    JSON-list files). With a bounded `--n-docs` sample the ledger only covers
    the prefix that was read; the returned info records `exhaustive=false` and
    the caller warns, instead of implying full coverage.
    """
    if not path:
        return None
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, 'w', encoding='utf-8') as f:
        for h in sorted(hashes):
            f.write(h.hex() + '\n')
    exhaustive = not n_docs
    info = {
        'path': str(ledger),
        'split': split_name,
        'n_hashes': len(hashes),
        'exhaustive': exhaustive,
        'n_docs_limit': n_docs,
    }
    print(f'[audit] wrote {len(hashes)} {split_name}-document hashes to {ledger}'
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
        return 0.0
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
    p.add_argument('--max-tvd', type=float, default=None, help="fail (exit 1) when train-vs-validation token total-variation distance exceeds this")
    p.add_argument('--max-test-tvd', type=float, default=None, help="fail (exit 1) when train-vs-test token total-variation distance exceeds this; the test split backs the final reported number, so unchecked drift there changes what that number measures")
    p.add_argument('--max-val-test-tvd', type=float, default=None, help="fail (exit 1) when validation-vs-test token total-variation distance exceeds this; the model is selected on validation and reported on test, so drift between the two held-out slices makes the selected checkpoint a poorer fit for the number finally reported")
    p.add_argument('--output', default='runs/data_audit.json')
    args = p.parse_args()
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
    reports = {}
    train_hashes = None
    val_hashes = None
    train_ledger_info = None
    val_ledger_info = None
    for name in ('train', 'validation', 'test'):
        # Train/validation hash sets are kept (the exclude ledgers and the
        # overlap reference); test only needs to be compared against them.
        # Test is checked against train *and* validation: a val/test duplicate
        # makes the selection split and the final split non-disjoint.
        stats, tok_counter, report, hashes = _collect_split(
            name, args.dataset, args.dataset_config, tokenizer, options,
            args.n_docs, args.min_doc_chars, args.max_doc_chars, args.dedup,
            known_hashes=train_hashes,
            known_hashes_extra=(val_hashes if name == 'test' else None),
            collect_hashes=(name in ('train', 'validation')))
        splits[name] = stats
        tokens[name] = tok_counter
        reports[name] = report
        if name == 'train':
            train_hashes = hashes
            train_ledger_info = _write_hash_ledger(
                args.write_train_hash_file, hashes, args.n_docs, 'train')
        elif name == 'validation':
            val_hashes = hashes
            val_ledger_info = _write_hash_ledger(
                args.write_val_hash_file, hashes, args.n_docs, 'validation')
    chronology = _chronology_problems(reports)
    shift = {
        'train_vs_validation_tvd': _total_variation(tokens['train'], tokens['validation']),
        'train_vs_test_tvd': _total_variation(tokens['train'], tokens['test']),
        'validation_vs_test_tvd': _total_variation(tokens['validation'], tokens['test']),
        'top_shifted_train_vs_validation': _top_shifted(tokens['train'], tokens['validation']),
        'doc_chars_mean_delta_train_validation': (
            None if splits['train']['doc_chars_mean'] is None or splits['validation']['doc_chars_mean'] is None
            else splits['validation']['doc_chars_mean'] - splits['train']['doc_chars_mean']),
    }
    out = {'options': vars(args), 'run_metadata': collect_run_metadata(),
           'split_reports': reports, 'split_chronology_problems': chronology,
           'cleaning_and_sanity': splits, 'distribution_shift': shift,
           'train_hash_ledger': train_ledger_info,
           'validation_hash_ledger': val_ledger_info}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    for name, stats in splits.items():
        print(f"[audit] {name}: sampled={stats['documents_sampled']} kept={stats.get('documents_kept', 0)} "
              f"dropped_empty={stats.get('dropped_empty', 0)} dropped_missing={stats.get('dropped_missing_text', 0)} "
              f"dropped_duplicate={stats.get('dropped_duplicate', 0)} overlap_with_train={stats['duplicate_overlap_with_train']} "
              f"overlap_with_validation={stats['duplicate_overlap_with_validation']}")
    print(f"[audit] token TVD train-vs-val={shift['train_vs_validation_tvd']:.4f} "
          f"train-vs-test={shift['train_vs_test_tvd']:.4f} "
          f"val-vs-test={shift['validation_vs_test_tvd']:.4f}")
    print(f"[audit] saved to {args.output}")
    problems = list(chronology)
    for name in ('train', 'validation', 'test'):
        stats = splits[name]
        if stats['length_outliers_3iqr']:
            print(f"[audit] NOTE: {name}: {stats['length_outliers_3iqr']} length outliers beyond the 3xIQR fence "
                  f"(q1={stats['doc_chars_q1']}, q3={stats['doc_chars_q3']}, max={stats['doc_chars_max']}); "
                  "use --min-doc-chars/--max-doc-chars in the training run to drop them")
        if stats.get('dedup_cap_reached'):
            print(f"[audit] NOTE: {name}: the in-split duplicate hash set reached dedup_max during the "
                  f"sample, so later complete-duplicate documents in this split were kept; the "
                  "guarantee degrades silently at that point (the cross-split ledger is unaffected)")
    # Every split is audited, test included: the final numbers come from test,
    # so corruption there must fail the audit just like corruption in train.
    non_finite = [name for name in ('train', 'validation', 'test')
                  if not splits[name]['finite_lengths']]
    if non_finite:
        problems.append(f"non-finite document-length statistics (NaN/Inf in the data) in: {', '.join(non_finite)}")
    if (splits['validation']['duplicate_overlap_with_train'] or 0) > 0 or (splits['test']['duplicate_overlap_with_train'] or 0) > 0:
        problems.append('exact duplicate documents appear in both train and a held-out split (group/duplicate leakage)')
    if (splits['test']['duplicate_overlap_with_validation'] or 0) > 0:
        problems.append('exact duplicate documents appear in both validation and test (the selection split and the sealed final split are not disjoint)')
    out_of_range = [name for name in ('train', 'validation', 'test')
                    if splits[name]['token_ids_out_of_range']]
    if out_of_range:
        problems.append(f"token ids outside the tokenizer vocabulary range in: {', '.join(out_of_range)}")
    if args.max_tvd is not None and shift['train_vs_validation_tvd'] > args.max_tvd:
        problems.append(f"train-vs-validation distribution shift TVD={shift['train_vs_validation_tvd']:.4f} > max-tvd={args.max_tvd}")
    if args.max_test_tvd is not None and shift['train_vs_test_tvd'] > args.max_test_tvd:
        problems.append(f"train-vs-test distribution shift TVD={shift['train_vs_test_tvd']:.4f} > max-test-tvd={args.max_test_tvd}")
    if args.max_val_test_tvd is not None and shift['validation_vs_test_tvd'] > args.max_val_test_tvd:
        problems.append(f"validation-vs-test distribution shift TVD={shift['validation_vs_test_tvd']:.4f} > max-val-test-tvd={args.max_val_test_tvd}")
    manifest = {
        'run_metadata': out['run_metadata'],
        'dataset': args.dataset,
        'dataset_config': args.dataset_config,
        'tokenizer': args.tokenizer,
        'options': vars(args),
        'split_reports': reports,
        'train_hash_ledger': train_ledger_info,
        'validation_hash_ledger': val_ledger_info,
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
