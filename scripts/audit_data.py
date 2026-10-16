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
* split hygiene: exact duplicate documents shared between train and the
  held-out splits (a duplicate in two splits is a leak no matter how the
  split was drawn);
* distribution shift: total-variation distance between token frequency
  distributions of train vs validation / test, plus document-length means.

This script only *reads* data; it never trains or writes to a split.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.data_pipeline import DocumentCleaner, document_hash, load_document_stream


def _collect_split(name, dataset, dataset_config, tokenizer, options, n_docs, min_doc_chars, max_doc_chars, dedup, known_hashes=None):
    stream, report = load_document_stream(dataset, dataset_config, name, options)
    cleaner = DocumentCleaner(dedup=dedup, min_doc_chars=min_doc_chars, max_doc_chars=max_doc_chars)
    lengths = []
    tokens = Counter()
    hashes: set = set()
    overlap = 0
    tokenizer_errors = 0
    for example in stream:
        if len(lengths) >= n_docs:
            break
        text = cleaner(example)
        if text is None:
            # Count tokenizer failures too by trying only kept-looking docs is
            # impossible here; the cleaner's own counters cover the rest.
            continue
        lengths.append(len(text))
        h = document_hash(text)
        hashes.add(h)
        if known_hashes is not None and h in known_hashes:
            overlap += 1
        try:
            tokens.update(tokenizer(text, add_special_tokens=False)['input_ids'])
        except Exception:
            tokenizer_errors += 1
    stats = cleaner.report()
    stats['documents_sampled'] = len(lengths)
    stats['duplicate_overlap_with_train'] = overlap if known_hashes is not None else None
    if lengths:
        stats['doc_chars_mean'] = sum(lengths) / len(lengths)
        stats['doc_chars_min'] = min(lengths)
        stats['doc_chars_max'] = max(lengths)
    else:
        stats['doc_chars_mean'] = stats['doc_chars_min'] = stats['doc_chars_max'] = None
    stats['tokenizer_errors'] = tokenizer_errors
    stats['token_ids_out_of_range'] = sum(c for t, c in tokens.items() if not 0 <= t < getattr(tokenizer, 'vocab_size', 2 ** 31))
    stats['distinct_token_ids'] = len(tokens)
    stats['finite_lengths'] = bool(all(math.isfinite(v) for v in (stats['doc_chars_mean'] or 0.0, stats['doc_chars_min'] or 0.0, stats['doc_chars_max'] or 0.0)))
    return stats, tokens, report, hashes


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
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--val-frac', type=float, default=0.005, help='must match the trained run')
    p.add_argument('--test-frac', type=float, default=0.005, help='must match the trained run')
    p.add_argument('--num-documents', type=int, default=None, help='total documents; required when the dataset reports no size')
    p.add_argument('--n-docs', type=int, default=2000, help='documents sampled per split')
    p.add_argument('--min-doc-chars', type=int, default=0)
    p.add_argument('--max-doc-chars', type=int, default=0)
    p.add_argument('--no-dedup', dest='dedup', action='store_false')
    p.add_argument('--max-tvd', type=float, default=None, help="fail (exit 1) when train-vs-validation token total-variation distance exceeds this")
    p.add_argument('--output', default='runs/data_audit.json')
    args = p.parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(f'audit_data needs transformers + datasets installed: {e}')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    options = {'val_frac': args.val_frac, 'test_frac': args.test_frac,
               'num_documents': args.num_documents, 'dedup': args.dedup,
               'min_doc_chars': args.min_doc_chars, 'max_doc_chars': args.max_doc_chars}
    splits = {}
    tokens = {}
    reports = {}
    train_hashes = None
    for name in ('train', 'validation', 'test'):
        stats, tok_counter, report, hashes = _collect_split(
            name, args.dataset, args.dataset_config, tokenizer, options,
            args.n_docs, args.min_doc_chars, args.max_doc_chars, args.dedup,
            known_hashes=train_hashes)
        splits[name] = stats
        tokens[name] = tok_counter
        reports[name] = report
        if name == 'train':
            train_hashes = hashes
    shift = {
        'train_vs_validation_tvd': _total_variation(tokens['train'], tokens['validation']),
        'train_vs_test_tvd': _total_variation(tokens['train'], tokens['test']),
        'top_shifted_train_vs_validation': _top_shifted(tokens['train'], tokens['validation']),
        'doc_chars_mean_delta_train_validation': (
            None if splits['train']['doc_chars_mean'] is None or splits['validation']['doc_chars_mean'] is None
            else splits['validation']['doc_chars_mean'] - splits['train']['doc_chars_mean']),
    }
    out = {'options': vars(args), 'split_reports': reports, 'cleaning_and_sanity': splits, 'distribution_shift': shift}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2, default=str))
    for name, stats in splits.items():
        print(f"[audit] {name}: sampled={stats['documents_sampled']} kept={stats.get('documents_kept', 0)} "
              f"dropped_empty={stats.get('dropped_empty', 0)} dropped_missing={stats.get('dropped_missing_text', 0)} "
              f"dropped_duplicate={stats.get('dropped_duplicate', 0)} overlap_with_train={stats['duplicate_overlap_with_train']}")
    print(f"[audit] token TVD train-vs-val={shift['train_vs_validation_tvd']:.4f} "
          f"train-vs-test={shift['train_vs_test_tvd']:.4f}")
    print(f"[audit] saved to {args.output}")
    problems = []
    if not splits['train']['finite_lengths'] or not splits['validation']['finite_lengths']:
        problems.append('non-finite document-length statistics (NaN/Inf in the data)')
    if (splits['validation']['duplicate_overlap_with_train'] or 0) > 0 or (splits['test']['duplicate_overlap_with_train'] or 0) > 0:
        problems.append('exact duplicate documents appear in both train and a held-out split (group/duplicate leakage)')
    if splits['train']['token_ids_out_of_range'] or splits['validation']['token_ids_out_of_range']:
        problems.append('token ids outside the tokenizer vocabulary range')
    if args.max_tvd is not None and shift['train_vs_validation_tvd'] > args.max_tvd:
        problems.append(f"train-vs-validation distribution shift TVD={shift['train_vs_validation_tvd']:.4f} > max-tvd={args.max_tvd}")
    if problems:
        for problem in problems:
            print(f'[audit] PROBLEM: {problem}', file=sys.stderr)
        raise SystemExit(1)
    print('[audit] no leakage or sanity problems found in the sampled splits')


if __name__ == '__main__':
    main()
