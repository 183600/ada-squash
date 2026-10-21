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
  trimmed with statistics it contributed to.

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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest
from scripts.data_pipeline import (DocumentCleaner, canonical_document_hash, document_hash,
                                   extract_group_value, group_hash, load_document_stream)


def _collect_split(name, dataset, dataset_config, tokenizer, options, n_docs, min_doc_chars, max_doc_chars, dedup, known_hashes=None, known_hashes_extra=None, collect_hashes=True, group_key=None, known_groups=None, known_groups_extra=None, collect_groups=True, exclude_hashes=None, exclude_group_hashes=None):
    stream, report = load_document_stream(dataset, dataset_config, name, options)
    cleaner = DocumentCleaner(dedup=dedup, min_doc_chars=min_doc_chars, max_doc_chars=max_doc_chars)
    # One machine-word int per document.
    lengths = array('q')
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
    tokenizer_errors = 0
    ledger_dropped = 0
    effective_documents = 0
    vocab_size = int(getattr(tokenizer, 'vocab_size', 0) or 0)

    def _add_ids(ids, which='raw') -> None:
        try:
            arr = np.asarray(ids, dtype=np.int64)
        except (TypeError, ValueError):
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

    def _flush_tokens() -> None:
        nonlocal tokenizer_errors
        if not text_batch:
            return
        try:
            enc = tokenizer(text_batch, add_special_tokens=False)['input_ids']
        except Exception:
            enc = None
        if enc is None:
            for t, effective in zip(text_batch, text_batch_effective):
                try:
                    ids = tokenizer(t, add_special_tokens=False)['input_ids']
                except Exception:
                    tokenizer_errors += 1
                    continue
                _add_ids(ids, 'raw')
                if effective:
                    _add_ids(ids, 'effective')
        else:
            for ids, effective in zip(enc, text_batch_effective):
                _add_ids(ids, 'raw')
                if effective:
                    _add_ids(ids, 'effective')
        text_batch.clear()
        text_batch_effective.clear()

    capped = False
    documents_read = 0
    for example in stream:
        # `--n-docs` bounds documents *sampled* (read), not documents kept:
        # otherwise a split with many filtered documents would silently read
        # far past the requested sample and claim a prefix it did not use.
        if n_docs and documents_read >= n_docs:
            capped = True
            break
        documents_read += 1
        text = cleaner(example)
        if text is None:
            # Count tokenizer failures too by trying only kept-looking docs is
            # impossible here; the cleaner's own counters cover the rest.
            continue
        lengths.append(len(text))
        h = document_hash(text)
        ch = canonical_document_hash(text)
        if collect_hashes:
            # Both digests, so a held-out formatting variant of a training
            # document is caught by the ledger the same way the cleaner
            # catches it (and the hash itself is still found either way).
            hashes.add(h)
            hashes.add(ch)
        # The ledger holds exact *and* canonical digests; checking only the
        # exact one would undercount formatting-variant duplicates that the
        # cleaner (and therefore the actual held-out split) does drop.
        dropped_by_ledger = bool(
            exclude_hashes and (h in exclude_hashes or ch in exclude_hashes))
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
                if collect_groups:
                    groups.add(gh)
                if exclude_group_hashes and gh in exclude_group_hashes:
                    dropped_by_ledger = True
                if known_groups is not None and gh in known_groups:
                    group_overlap += 1
                    group_overlap_groups.add(gh)
                if known_groups_extra is not None and gh in known_groups_extra:
                    group_overlap_extra += 1
                    group_overlap_extra_groups.add(gh)
        if dropped_by_ledger:
            ledger_dropped += 1
        else:
            effective_documents += 1
        text_batch.append(text)
        text_batch_effective.append(not dropped_by_ledger)
        if len(text_batch) >= 256:
            _flush_tokens()
    _flush_tokens()
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
    # Whether the counters cover the whole split or only a prefix. A stream
    # that ends naturally before --n-docs is fully read even though n_docs is
    # nonzero, so the cap is tracked by whether the loop actually broke. With
    # a truncated sample, `duplicate_overlap_with_train = 0` is a lower bound
    # on the true overlap, not evidence that the splits are disjoint.
    stats['exhaustive'] = not capped
    stats['ledger_documents_dropped'] = (ledger_dropped
                                         if exclude_hashes is not None or exclude_group_hashes is not None
                                         else None)
    stats['effective_documents_kept'] = (effective_documents
                                         if exclude_hashes is not None or exclude_group_hashes is not None
                                         else stats.get('documents_kept', len(lengths)))
    stats['duplicate_overlap_with_train'] = overlap if known_hashes is not None else None
    stats['duplicate_overlap_with_validation'] = overlap_extra if known_hashes_extra is not None else None
    stats['duplicate_overlap_is_lower_bound'] = bool(capped) if (known_hashes is not None or known_hashes_extra is not None) else None
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
            bool(capped) if (known_groups is not None or known_groups_extra is not None) else None)
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
    stats['tokenizer_errors'] = tokenizer_errors
    stats['token_ids_out_of_range'] = sum(c for t, c in tokens.items() if not 0 <= t < getattr(tokenizer, 'vocab_size', 2 ** 31))
    stats['distinct_token_ids'] = len(tokens)
    stats['finite_lengths'] = bool(all(math.isfinite(v) for v in (stats['doc_chars_mean'] or 0.0, stats['doc_chars_min'] or 0.0, stats['doc_chars_max'] or 0.0)))
    return stats, tokens, report, hashes, lengths, groups, tokens_effective


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
    test_end = test['start'] + test['count']
    if train['start'] != 0 or validation['start'] != train_end or test['start'] != validation_end:
        return [f"split ranges are not contiguous chronological slices: train=[{train['start']},{train_end}) validation=[{validation['start']},{validation_end}) test=[{test['start']},{test_end})"]
    total = train.get('num_documents')
    if isinstance(total, int):
        counts_match = all(r.get('num_documents') == total
                           for r in (validation, test))
        if not counts_match or test_end != total:
            return [f"split ranges do not cover the full document stream: "
                    f"test ends at {test_end} but num_documents={total}"]
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
    p.add_argument('--output', default='runs/data_audit.json')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this audit tests; recorded in the report and manifest so the result is auditable as a test of a claim rather than a fishing trip')
    args = p.parse_args()
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
    train_hashes = None
    val_hashes = None
    train_groups = None
    val_groups = None
    train_ledger_info = None
    val_ledger_info = None
    train_group_ledger_info = None
    val_group_ledger_info = None
    for name in ('train', 'validation', 'test'):
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
        stats, tok_counter, report, hashes, split_lengths, groups, tok_effective = _collect_split(
            name, args.dataset, args.dataset_config, tokenizer, options,
            args.n_docs, args.min_doc_chars, args.max_doc_chars, args.dedup,
            known_hashes=train_hashes,
            known_hashes_extra=(val_hashes if name == 'test' else None),
            collect_hashes=(name in ('train', 'validation')),
            group_key=args.group_key,
            known_groups=train_groups,
            known_groups_extra=(val_groups if name == 'test' else None),
            collect_groups=(name in ('train', 'validation')),
            exclude_hashes=exclude_hashes,
            exclude_group_hashes=exclude_groups)
        splits[name] = stats
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
    empty_token_splits = [name for name in ('train', 'validation', 'test')
                          if not sum(tokens[name].values())]
    shift = {
        'train_vs_validation_tvd': _total_variation(tokens['train'], tokens['validation']),
        'train_vs_test_tvd': _total_variation(tokens['train'], tokens['test']),
        'validation_vs_test_tvd': _total_variation(tokens['validation'], tokens['test']),
        # The same distances over the documents that survive the cross-split
        # ledgers: that is the distribution the trained model is actually
        # selected and reported on, and it can differ from the raw one when a
        # ledger is non-trivial.
        'train_vs_validation_tvd_effective': _total_variation(
            effective_tokens['train'], effective_tokens['validation']),
        'train_vs_test_tvd_effective': _total_variation(
            effective_tokens['train'], effective_tokens['test']),
        'validation_vs_test_tvd_effective': _total_variation(
            effective_tokens['validation'], effective_tokens['test']),
        'top_shifted_train_vs_validation': _top_shifted(tokens['train'], tokens['validation']),
        'doc_chars_mean_delta_train_validation': (
            None if splits['train']['doc_chars_mean'] is None or splits['validation']['doc_chars_mean'] is None
            else splits['validation']['doc_chars_mean'] - splits['train']['doc_chars_mean']),
    }
    out = {'options': vars(args), 'hypothesis': args.hypothesis, 'run_metadata': collect_run_metadata(),
           'split_reports': reports, 'split_chronology_problems': chronology,
           'cleaning_and_sanity': splits, 'distribution_shift': shift,
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
                             for name in ('train', 'validation', 'test'))
        if missing_groups:
            print(f"[audit] NOTE: {missing_groups} sampled documents have no '{args.group_key}' value; "
                  "they cannot be tied to a group and are checked for document duplicates only")
    print(f"[audit] token TVD train-vs-val={shift['train_vs_validation_tvd']:.4f} "
          f"train-vs-test={shift['train_vs_test_tvd']:.4f} "
          f"val-vs-test={shift['validation_vs_test_tvd']:.4f}")
    print(f"[audit] token TVD after ledgers train-vs-val={shift['train_vs_validation_tvd_effective']:.4f} "
          f"train-vs-test={shift['train_vs_test_tvd_effective']:.4f} "
          f"val-vs-test={shift['validation_vs_test_tvd_effective']:.4f}")
    if args.hypothesis:
        print(f"[audit][hypothesis] {args.hypothesis}")
    print(f"[audit] saved to {args.output}")
    problems = list(chronology)
    if empty_token_splits:
        problems.append('no usable tokens could be sampled from: ' + ', '.join(empty_token_splits)
                        + ' (an empty histogram makes TVD undefined, not zero)')
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
        train_fence_outliers = stats.get('length_outliers_train_fence')
        if train_fence_outliers:
            print(f"[audit] NOTE: {name}: {train_fence_outliers} documents "
                  f"({stats['length_outliers_train_fence_fraction']:.4%}) fall outside the 3xIQR fence "
                  f"[{train_fence['lower']}, {train_fence['upper']}] derived from the training lengths alone")
    # Every split is audited, test included: the final numbers come from test,
    # so corruption there must fail the audit just like corruption in train.
    non_finite = [name for name in ('train', 'validation', 'test')
                  if not splits[name]['finite_lengths']]
    if non_finite:
        problems.append(f"non-finite document-length statistics (NaN/Inf in the data) in: {', '.join(non_finite)}")
    if (splits['validation']['duplicate_overlap_with_train'] or 0) > 0 or (splits['test']['duplicate_overlap_with_train'] or 0) > 0:
        problems.append('duplicate documents (exact or canonical text) appear in both train and a held-out split (duplicate leakage)')
    if (splits['test']['duplicate_overlap_with_validation'] or 0) > 0:
        problems.append('duplicate documents (exact or canonical text) appear in both validation and test (the selection split and the sealed final split are not disjoint)')
    if args.group_key:
        straddling = []
        for name in ('validation', 'test'):
            n_groups = splits[name].get('group_overlap_with_train_groups') or 0
            if n_groups:
                straddling.append(f"{name} shares {n_groups} group(s) with train")
        n_groups = splits['test'].get('group_overlap_with_validation_groups') or 0
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
    empty_effective = [name for name in ('validation', 'test')
                       if splits[name].get('documents_kept')
                       and not splits[name].get('effective_documents_kept')]
    if empty_effective:
        problems.append(
            'the cross-split ledgers drop every kept document of: '
            + ', '.join(empty_effective)
            + ' (the effective held-out split is empty; the run would evaluate on nothing)')
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
        'hypothesis': args.hypothesis,
        'dataset': args.dataset,
        'dataset_config': args.dataset_config,
        'tokenizer': args.tokenizer,
        'options': vars(args),
        'split_reports': reports,
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
