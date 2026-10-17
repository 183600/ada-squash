from __future__ import annotations
"""Bad-case analysis: print the mispredictions and say *why* they are wrong.

Why this script exists
----------------------
Aggregate metrics say *how often* the model is wrong; new ideas usually come
from reading *which* wrong cases and *why*. For every mispredicted token this
script records the gold/predicted token, the model's confidence, the gold
rank, and a small taxonomy of causes:

* ``ambiguous_topk`` - gold was in the model's top-k but not top-1: a
  ranking/tie problem, not a knowledge-of-nothing problem;
* ``prev_context_ambiguous`` - other targets follow the same preceding token
  with at least the same frequency: the context itself cannot determine the
  label, so part of the "error" is inherent entropy (and in labeled corpora a
  candidate label-noise signal);
* ``rare_gold`` - the gold token is in the long tail of the corpus: a data
  imbalance / long-tail issue, not a bug;
* ``repetition_bias`` - the prediction occurs in the preceding context while
  the gold does not: the classic copy-the-context failure mode;
* ``high_confidence_wrong`` - the model is confident and wrong: a blind spot,
  the most useful class for a new hypothesis;
* ``unexplained`` - none of the above.

It also reports the baselines the accuracy must be read against
(majority-class, uniform-random, copy-the-current-token, and the
``bigram_majority_accuracy`` rule "predict the most frequent successor of the
current token") and an oracle ``topk_accuracy`` that bounds how much of the
error is a ranking problem.

Metrics
-------
Target tokens over a 50k vocabulary are strongly imbalanced, so accuracy and
``micro_f1`` are dominated by the frequent classes and are reported together
with the metrics that can disagree with them:

* ``macro_f1`` / ``weighted_f1`` / ``micro_f1`` from
  :func:`adasquash.metrics.f1_from_counts` - all three averaging modes are
  named explicitly (macro and micro can rank models very differently);
* one-vs-rest PR-AUC (``average_precision``) and ROC-AUC for a bounded set of
  target classes taken from the first batch (half by frequency, half sampled
  from its long tail), accumulated over the scanned tokens and capped by
  ``--auc-max-tokens``. ``prefer_pr_auc`` is set when some class has >= 1000
  negatives per positive; in that regime ROC-AUC is optimistic (its FPR
  denominator is the huge negative count) and PR-AUC is the number to read.

Corpora
-------
* ``--corpus markov`` (default): the repo's synthetic Markov walk - no
  dataset download needed, deterministic under ``--seed``, and the ambiguity
  structure is known by construction (the next state has 3 possible jumps);
* ``--corpus dataset``: the leakage-safe validation split through
  ``scripts.data_pipeline.py``; pass the data flags that match the trained
  run.
"""
import argparse
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from adasquash.metrics import binary_auc_report, f1_from_counts, imbalance_report
from adasquash.repro import collect_run_metadata, save_run_manifest, set_all_seeds
from adasquash.utils import markov_batch
from scripts.eval_downstream import load_ckpt


def _markov_batches(args, cfg, device):
    rng = torch.Generator(device=device).manual_seed(args.seed)
    n_states = max(2, min(int(cfg.vocab_size), 64))
    for _ in range(args.n_batches):
        x, y = markov_batch(rng, cfg.ignore_index, args.batch_size, args.seq_len, device, n_states=n_states)
        yield x, y


def _dataset_batches(args, model, device):
    from scripts.train_gpt2 import open_data_split

    namespace = argparse.Namespace(
        dataset=args.dataset, dataset_config=args.dataset_config, tokenizer=args.tokenizer,
        val_frac=args.val_frac, test_frac=args.test_frac, num_documents=args.num_documents,
        dedup=not args.no_dedup, min_doc_chars=args.min_doc_chars, max_doc_chars=args.max_doc_chars,
        exclude_hash_file=args.exclude_hash_file, train_shuffle_buffer=0,
        seed=args.seed, seq_len=args.seq_len, batch_size=args.batch_size,
    )
    loader = open_data_split(args.split, namespace)
    report = getattr(loader, 'split_report', None)
    if report:
        print(f'[errors] data split report: {report}')
    n = 0
    for x, y in loader:
        yield x.to(device), y.to(device)
        n += 1
        if n >= args.n_batches:
            break


def _decode(ids, tokenizer):
    if tokenizer is None:
        return '#' + ' #'.join(str(int(i)) for i in ids)
    return tokenizer.decode(ids, skip_special_tokens=False)


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def analyze(model, batches, args, tokenizer=None, device='cpu') -> dict:
    """Run the model over `batches` and build the error taxonomy report."""
    cfg = model.cfg
    ignore = int(cfg.ignore_index)
    vocab = int(cfg.vocab_size)
    target_counts = torch.zeros(vocab, dtype=torch.long)
    # Confusion-matrix marginals for the named F1 report: the three per-class
    # counts are accumulated instead of the full pred/gold lists.
    pred_counts = torch.zeros(vocab, dtype=torch.long)
    correct_counts = torch.zeros(vocab, dtype=torch.long)
    # Ranking-metric diagnostics: the class set is fixed from the first batch
    # (half by frequency, half sampled from its tail), then every scanned
    # token contributes one score per chosen class, up to --auc-max-tokens.
    auc_class_ids: list = []
    auc_scores: list = []
    auc_labels: list = []
    auc_seen = 0
    auc_max_tokens = max(0, int(getattr(args, 'auc_max_tokens', 0)))
    batches = iter(batches)
    first_batch = None
    if int(getattr(args, 'auc_classes', 0)) > 0:
        first_batch = next(batches, None)
        if first_batch is not None:
            _, y0 = first_batch
            counts0 = torch.bincount(y0[y0.ne(ignore)].long().cpu(), minlength=vocab)
            present0 = torch.nonzero(counts0).flatten().tolist()
            k = min(int(args.auc_classes), len(present0))
            n_head = (k + 1) // 2
            head = sorted(present0, key=lambda c: (-int(counts0[c]), c))[:n_head]
            head_set = set(head)
            tail_pool = [c for c in present0 if c not in head_set]
            rng = random.Random(int(getattr(args, 'seed', 0)))
            tail = rng.sample(tail_pool, min(k - len(head), len(tail_pool)))
            auc_class_ids = sorted(head + tail)
    prev_gold_keys = []          # encoded (prev_id * vocab + gold_id) over all tokens
    n_tokens = n_correct = 0
    topk = max(1, int(args.topk))
    # A k above the number of classes cannot be measured: top-k with k >= V is
    # the whole distribution. Clamp to vocab - 1 so the report key names the
    # radius actually analyzed.
    analyzed_topk = min(topk, max(1, vocab - 1))
    topk_hits = 0
    gold_prob_sum = 0.0
    gold_rank_sum = 0.0
    n_errors = 0
    n_scanned = 0
    error_scan = []              # capped records, classified after the pass
    copy_input_correct = 0
    n_positional = 0

    if first_batch is not None:
        batches = itertools.chain([first_batch], batches)
    for x, y in batches:
        with torch.no_grad():
            logits, _ = model(x, return_logits=True)
        mask = y.ne(ignore)
        prev = x
        valid = mask
        n_tokens += int(valid.sum().item())
        target_counts += torch.bincount(y[valid].cpu(), minlength=vocab)
        keys = (prev[valid].long().cpu() * vocab + y[valid].long().cpu())
        prev_gold_keys.append(keys)
        if tuple(x.shape) == tuple(y.shape):
            copy_input_correct += int((x[valid] == y[valid]).sum().item())
        B, T, V = logits.shape
        chunk = max(1, int(args.eval_chunk))
        for t0 in range(0, T, chunk):
            t1 = min(T, t0 + chunk)
            lc = logits[:, t0:t1].detach().float()
            m = mask[:, t0:t1]
            if not bool(m.any()):
                continue
            probs = torch.softmax(lc, dim=-1)
            gold = y[:, t0:t1]
            pred = probs.argmax(-1)
            n_correct += int((pred[m] == gold[m]).sum().item())
            pred_counts += torch.bincount(pred[m].long().cpu(), minlength=vocab)
            correct_counts += torch.bincount(gold[m & (pred == gold)].long().cpu(), minlength=vocab)
            gold_p = probs.gather(-1, gold.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            gold_prob_sum += float(gold_p[m].sum().item())
            k = analyzed_topk
            top = probs.topk(k, dim=-1)
            # 1-based rank; ties count above (>= counts the gold token itself,
            # every strictly-larger probability and every tie, so a tie cannot
            # flatter the model's ranking).
            gold_rank = (probs >= gold_p.unsqueeze(-1)).sum(-1)
            topk_hits += int((top.indices == gold.unsqueeze(-1)).any(-1)[m].sum().item())
            gold_rank_sum += float(gold_rank[m].clamp(max=args.rank_clip).sum().item())
            n_positional += int(m.sum().item())
            wrong = m & (pred != gold)
            n_errors += int(wrong.sum().item())
            if n_scanned < args.max_scan_errors:
                idx = wrong.nonzero(as_tuple=False)
                for b, tt in idx[: args.max_scan_errors - n_scanned].tolist():
                    n_scanned += 1
                    error_scan.append({
                        'batch_row': b,
                        'position': t0 + tt,
                        'gold': int(gold[b, tt]),
                        'pred': int(pred[b, tt]),
                        'gold_prob': float(gold_p[b, tt]),
                        'pred_prob': float(probs[b, tt, pred[b, tt]].item()),
                        'gold_rank': int(min(gold_rank[b, tt].item(), args.rank_clip)),
                        'gold_in_topk': bool((top.indices[b, tt] == gold[b, tt]).any().item()),
                        'prev': int(prev[b, t0 + tt]),
                        'context': [int(v) for v in x[b, max(0, t0 + tt - args.context_tokens):t0 + tt + 1].tolist()],
                    })
            if auc_class_ids:
                sel = probs[..., auc_class_ids][m]
                take = int(sel.shape[0])
                if auc_max_tokens > 0:
                    take = min(take, auc_max_tokens - auc_seen)
                if take > 0:
                    auc_scores.append(sel[:take].detach().cpu())
                    auc_labels.append(gold[m][:take].detach().cpu())
                    auc_seen += take
            del probs, top, lc

    if n_tokens == 0:
        raise SystemExit('[errors] no valid target tokens in the selected batches')

    prev_gold = Counter()
    if prev_gold_keys:
        keys = torch.cat(prev_gold_keys)
        uniq, counts = torch.unique(keys, return_counts=True)
        for key, count in zip(uniq.tolist(), counts.tolist()):
            prev_gold[key] = count
    prev_majority = {}
    for key, count in prev_gold.items():
        p = key // vocab
        g = key % vocab
        if p not in prev_majority or count > prev_majority[p][1]:
            prev_majority[p] = (g, count)
    positive_counts = [int(c) for c in target_counts.tolist() if c > 0]
    rare_threshold = _quantile(positive_counts, args.rare_quantile)
    rare_ids = {i for i, c in enumerate(target_counts.tolist()) if 0 < c <= (rare_threshold or 0)}

    categories = Counter()
    primary = Counter()
    examples = []
    for rec in error_scan:
        gold, pred, prev = rec['gold'], rec['pred'], rec['prev']
        flags = {}
        flags['ambiguous_topk'] = bool(rec['gold_in_topk'])
        maj = prev_majority.get(prev)
        flags['prev_context_ambiguous'] = bool(
            maj is not None and maj[0] != gold and maj[1] >= prev_gold.get(prev * vocab + gold, 0)
            and prev_gold.get(prev * vocab + gold, 0) > 0
        )
        flags['rare_gold'] = gold in rare_ids
        flags['repetition_bias'] = bool(pred in rec['context'] and gold not in rec['context'])
        flags['high_confidence_wrong'] = bool(rec['pred_prob'] >= args.confidence)
        for name, value in flags.items():
            if value:
                categories[name] += 1
        if flags['ambiguous_topk']:
            cause = 'label_ambiguity_topk'
        elif flags['prev_context_ambiguous']:
            cause = 'context_ambiguity'
        elif flags['rare_gold']:
            cause = 'long_tail_target'
        elif flags['high_confidence_wrong'] and flags['repetition_bias']:
            cause = 'model_blind_spot_copying'
        elif flags['high_confidence_wrong']:
            cause = 'model_blind_spot_high_confidence'
        elif flags['repetition_bias']:
            cause = 'model_copying_bias'
        else:
            cause = 'unexplained'
        primary[cause] += 1
        record = dict(rec)
        record['flags'] = flags
        record['primary_cause'] = cause
        examples.append(record)

    examples.sort(key=lambda r: (-r['pred_prob'], r['gold_rank']))
    shown = examples[: args.max_examples]
    for rec in shown:
        rec['context_text'] = _decode(rec['context'], tokenizer)
        rec['gold_text'] = _decode([rec['gold']], tokenizer)
        rec['pred_text'] = _decode([rec['pred']], tokenizer)
    majority_acc = float(target_counts.max().item() / n_tokens) if n_tokens else 0.0
    # The context-majority rule ("predict the most frequent successor of the
    # current token"): a tuned-then-scored trivial baseline, not a strawman.
    bigram_correct = sum(count for _, count in prev_majority.values())
    bigram_acc = (bigram_correct / n_tokens) if n_tokens else 0.0
    f1 = f1_from_counts(target_counts.tolist(), pred_counts.tolist(), correct_counts.tolist())
    imbalance = imbalance_report(target_counts.tolist())
    auc_report = None
    if auc_class_ids and auc_scores:
        auc_report = binary_auc_report(
            torch.cat(auc_scores, dim=0).tolist(),
            torch.cat(auc_labels, dim=0).tolist(),
            auc_class_ids,
        )
    return {
        'summary': {
            'n_tokens': n_tokens,
            'n_errors': n_errors,
            'error_rate': n_errors / n_tokens,
            'accuracy': n_correct / n_tokens,
            'micro_f1': f1['micro_f1'],
            'macro_f1': f1['macro_f1'],
            'weighted_f1': f1['weighted_f1'],
            'f1_averaging': ('macro = unweighted mean over classes present; '
                             'weighted = support-weighted; micro = accuracy'),
            'imbalance': imbalance,
            'prefer_pr_auc': imbalance['prefer_pr_auc'],
            f'top{analyzed_topk}_accuracy': topk_hits / n_tokens,
            'mean_gold_prob': gold_prob_sum / n_tokens,
            'mean_gold_rank_capped': gold_rank_sum / n_tokens,
            'rank_cap': args.rank_clip,
            'majority_class_accuracy': majority_acc,
            'bigram_majority_accuracy': bigram_acc,
            'uniform_random_accuracy': 1.0 / vocab,
            'copy_input_accuracy': (copy_input_correct / n_tokens) if n_positional else None,
            'auc_classes': auc_class_ids,
            'auc_tokens_scored': auc_seen,
            'macro_average_precision': None if auc_report is None else auc_report['macro_average_precision'],
            'weighted_average_precision': None if auc_report is None else auc_report['weighted_average_precision'],
            'macro_roc_auc': None if auc_report is None else auc_report['macro_roc_auc'],
            'weighted_roc_auc': None if auc_report is None else auc_report['weighted_roc_auc'],
            'errors_scanned': n_scanned,
            'errors_total': n_errors,
        },
        'auc_report': auc_report,
        'cause_distribution': dict(primary.most_common()),
        'flag_counts': dict(categories.most_common()),
        'reading_guide': {
            'label_ambiguity_topk': 'gold is in top-k: improve ranking/calibration, not representation',
            'context_ambiguity': 'same preceding token is followed by other targets at least as often: inherent entropy / candidate label noise',
            'long_tail_target': f'gold among the bottom {args.rare_quantile:.0%} of target frequencies: data imbalance',
            'model_blind_spot_high_confidence': 'confident and wrong: the best source of new hypotheses',
            'model_blind_spot_copying': 'confident wrong copy of the context: check position/attention/repetition',
            'model_copying_bias': 'prediction copied from the context: check repetition handling',
            'unexplained': 'no flag fired: inspect the printed context manually',
        },
        'rarest_target_threshold': rare_threshold,
        'examples': shown,
        'note': (
            f'causes are disjoint per error and reported over the first {args.max_scan_errors} '
            'scanned errors; flag_counts overlap because one error can carry several flags'
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--corpus', choices=['markov', 'dataset'], default='markov')
    p.add_argument('--split', choices=['validation', 'test', 'train'], default='validation')
    p.add_argument('--n-batches', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--eval-chunk', type=int, default=64, help='positions per softmax chunk')
    p.add_argument('--topk', type=int, default=5, help='oracle top-k radius; values below 1 are clamped to 1 and values above vocab-1 to vocab-1 so the report key always matches the analyzed k')
    p.add_argument('--confidence', type=float, default=0.8, help='pred_prob at/above which a wrong prediction counts as confident')
    p.add_argument('--rare-quantile', type=float, default=0.1, help='bottom fraction of target frequencies treated as rare')
    p.add_argument('--rank-clip', type=int, default=100)
    p.add_argument('--context-tokens', type=int, default=32, help='preceding tokens kept/printed per error')
    p.add_argument('--max-scan-errors', type=int, default=2000)
    p.add_argument('--max-examples', type=int, default=20)
    p.add_argument('--auc-classes', type=int, default=8, help='target classes for the one-vs-rest PR-AUC/ROC-AUC diagnostics: half the most frequent classes in the first batch, half sampled from its long tail; 0 disables')
    p.add_argument('--auc-max-tokens', type=int, default=200000, help='cap on scored tokens for the AUC diagnostics (deterministic first-N subsample; 0 = no cap)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--output', default=None)
    # dataset-corpus options (must match the trained run's split protocol)
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--val-frac', type=float, default=0.005)
    p.add_argument('--test-frac', type=float, default=0.005)
    p.add_argument('--num-documents', type=int, default=None)
    p.add_argument('--min-doc-chars', type=int, default=0)
    p.add_argument('--max-doc-chars', type=int, default=0)
    p.add_argument('--no-dedup', action='store_true')
    p.add_argument('--exclude-hash-file', default=None)
    args = p.parse_args()
    args.topk = max(1, int(args.topk))
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(args.seed)
    model, model_args = load_ckpt(args.ckpt, device=device)
    # Keep the CLI value identical to the analyzed radius (see `analyze`) so
    # the printed/`config`-recorded topk names the key that is in the summary.
    args.topk = max(1, min(int(args.topk), max(1, int(model.cfg.vocab_size) - 1)))
    print(f"[errors] loaded {args.ckpt} | norm={model_args['norm']} | corpus={args.corpus} | device={device}")

    tokenizer = None
    if args.corpus == 'markov':
        batches = _markov_batches(args, model.cfg, device)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        batches = _dataset_batches(args, model, device)

    report = analyze(model, batches, args, tokenizer=tokenizer, device=device)
    report['config'] = vars(args)
    report['model_args'] = {k: v for k, v in model_args.items() if isinstance(v, (str, int, float, bool, type(None)))}
    report['run_metadata'] = collect_run_metadata(seed=args.seed)

    s = report['summary']
    print(f"[errors] tokens={s['n_tokens']} errors={s['n_errors']} "
          f"accuracy={s['accuracy']:.4f} error_rate={s['error_rate']:.4f} "
          f"top{args.topk}_accuracy={s[f'top{args.topk}_accuracy']:.4f} "
          f"macro_f1={s['macro_f1']:.4f} weighted_f1={s['weighted_f1']:.4f} "
          f"majority_baseline={s['majority_class_accuracy']:.4f} "
          f"bigram_baseline={s['bigram_majority_accuracy']:.4f} "
          f"uniform_baseline={s['uniform_random_accuracy']:.2e}")
    if s['imbalance']['prefer_pr_auc']:
        print(f"[errors] WARNING: target imbalance {s['imbalance']['note']}; "
              f"accuracy/micro_f1 overstate - read macro_f1 and the PR-AUC block")
    if s['macro_average_precision'] is not None:
        print(f"[errors] one-vs-rest over {len(s['auc_classes'])} classes "
              f"({s['auc_tokens_scored']} tokens): "
              f"macro_pr_auc={s['macro_average_precision']:.4f} "
              f"weighted_pr_auc={s['weighted_average_precision']:.4f} "
              f"macro_roc_auc={s['macro_roc_auc']:.4f} "
              f"weighted_roc_auc={s['weighted_roc_auc']:.4f} "
              f"prefer_pr_auc={s['prefer_pr_auc']}")
    print(f"[errors] cause distribution (over {s['errors_scanned']} scanned errors): "
          f"{report['cause_distribution']}")
    for rec in report['examples']:
        flags = ','.join(k for k, v in rec['flags'].items() if v) or 'none'
        print(f"[errors] pos={rec['position']:>5d} gold={rec['gold_text']!r}(p={rec['gold_prob']:.3f}, rank={rec['gold_rank']}) "
              f"pred={rec['pred_text']!r}(p={rec['pred_prob']:.3f}) cause={rec['primary_cause']} flags={flags}")
        print(f"[errors]   context: ...{rec['context_text']}")
    out = args.output or str(Path(args.ckpt).parent / f'error_analysis_{args.corpus}.json')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    save_run_manifest(Path(out).with_name('error_analysis_manifest.json'), {
        'run_metadata': report['run_metadata'], 'ckpt': args.ckpt,
        'corpus': args.corpus, 'config': vars(args), 'summary': s,
    })
    print(f'[errors] report -> {out}')


if __name__ == '__main__':
    main()
