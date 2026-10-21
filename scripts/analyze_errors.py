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

Paired mode (``--reference-ckpt``)
----------------------------------
A single-model taxonomy says how often a model is wrong; the decision "did
the candidate improve?" is a property of the difference. With
``--reference-ckpt`` both checkpoints consume the *identical* batches and the
report becomes a paired bad-case analysis: exact ``fixed``/``broken`` counts,
the exact McNemar test on the discordant tokens, the same taxonomy applied to
each side (what kind of error did the candidate repair, what kind did it
introduce), the shared blind spots (both arms confidently wrong - label-noise
/ inherent-ambiguity candidates) and the parameter/FLOP cost of both sides.
See :func:`analyze_pair`.

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
import heapq
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from adasquash.diagnostics import parameter_finite_report
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.metrics import binary_auc_report, f1_from_counts, imbalance_report, mcnemar_test
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds
from adasquash.utils import markov_batch
from scripts.eval_downstream import load_ckpt


def _decode(token_ids, tokenizer) -> str:
    ids = [int(i) for i in token_ids]
    if tokenizer is None:
        return " ".join(str(i) for i in ids)
    try:
        return tokenizer.decode(
            ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return " ".join(str(i) for i in ids)


def _markov_batches(args, cfg, device):
    rng = torch.Generator(device=device).manual_seed(args.seed)
    n_states = max(2, min(int(cfg.vocab_size), 64))
    for _ in range(args.n_batches):
        x, y = markov_batch(rng, cfg.ignore_index, args.batch_size, args.seq_len, device, n_states=n_states)
        yield x, y


def _dataset_batches(args, model, device, report_sink: dict | None = None):
    from scripts.train_gpt2 import open_data_split

    namespace = argparse.Namespace(
        dataset=args.dataset, dataset_config=args.dataset_config, tokenizer=args.tokenizer,
        dataset_revision=args.dataset_revision, tokenizer_revision=args.tokenizer_revision,
        val_frac=args.val_frac, test_frac=args.test_frac, num_documents=args.num_documents,
        dedup=not args.no_dedup, min_doc_chars=args.min_doc_chars, max_doc_chars=args.max_doc_chars,
        exclude_hash_file=args.exclude_hash_file,
        exclude_test_hash_file=args.exclude_test_hash_file,
        group_key=args.group_key, exclude_group_file=args.exclude_group_file,
        exclude_test_group_file=args.exclude_test_group_file, train_shuffle_buffer=0,
        num_workers=getattr(args, 'num_workers', 2),
        seed=args.seed, seq_len=args.seq_len, batch_size=args.batch_size,
    )
    loader = open_data_split(args.split, namespace)
    report = getattr(loader, 'split_report', None)
    if report:
        if report_sink is not None:
            # Keep the resolved revisions/boundaries in the bad-case report:
            # otherwise the analyzed documents cannot be audited against the
            # checkpoint's training protocol after the fact.
            report_sink['split_report'] = dict(report)
        print(f'[errors] data split report: {report}')
    n = 0
    for x, y in loader:
        yield x.to(device), y.to(device)
        n += 1
        if n >= args.n_batches:
            break


def data_protocol_mismatches(args, model_args: dict) -> dict:
    """Training-vs-analysis data-protocol differences that would invalidate pairing.

    A bad-case report is only about the trained run if it reads the same
    documents. The checkpoint's `args` records the split-defining flags, so
    the analysis can compare its own settings with them instead of silently
    sampling a different revision/fraction/ledger. Missing keys in old
    checkpoints are treated as unknown, never as a match.
    """
    current = {
        'dataset': args.dataset,
        'dataset_config': args.dataset_config,
        'dataset_revision': args.dataset_revision,
        'tokenizer': args.tokenizer,
        'tokenizer_revision': args.tokenizer_revision,
        'val_frac': args.val_frac,
        'test_frac': args.test_frac,
        'num_documents': args.num_documents,
        'dedup': not args.no_dedup,
        'min_doc_chars': args.min_doc_chars,
        'max_doc_chars': args.max_doc_chars,
        'exclude_hash_file': args.exclude_hash_file,
        'exclude_test_hash_file': args.exclude_test_hash_file,
        'group_key': args.group_key,
        'exclude_group_file': args.exclude_group_file,
        'exclude_test_group_file': args.exclude_test_group_file,
    }
    out = {}
    for key, value in current.items():
        # An old checkpoint may predate a split-defining flag. Absence is
        # unknown, not agreement: silently skipping it would pair an analysis
        # on one data protocol with a checkpoint trained on another.
        checkpoint_value = model_args[key] if key in model_args else '<missing>'
        if checkpoint_value != value:
            out[key] = {'checkpoint': checkpoint_value, 'analysis': value}
    return out


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
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


def _prev_majority(prev_gold_counts, vocab):
    """Bigram statistics: (prev, gold) counts and the majority successor per `prev`.

    Kept model-independent and shared by the single-model and paired passes, so
    both classify the same context-ambiguity evidence. The input is the
    device-side `(unique keys, counts)` histogram.
    """
    prev_gold = Counter()
    if prev_gold_counts is not None:
        uniq, counts = prev_gold_counts
        for key, count in zip(uniq.tolist(), counts.tolist()):
            prev_gold[key] = count
    prev_majority = {}
    for key, count in prev_gold.items():
        p = key // vocab
        g = key % vocab
        if p not in prev_majority or count > prev_majority[p][1]:
            prev_majority[p] = (g, count)
    return prev_gold, prev_majority


def _rare_ids(target_counts, rare_quantile):
    """Token ids in the bottom `rare_quantile` of the observed target frequencies."""
    if rare_quantile <= 0:
        # Zero explicitly disables the long-tail bucket; the quantile at 0 is
        # the minimum observed count and would otherwise still flag tokens.
        return set(), None
    positive_counts = [int(c) for c in target_counts.tolist() if c > 0]
    if not positive_counts:
        return set(), None
    rare_threshold = _quantile(positive_counts, rare_quantile)
    return {i for i, c in enumerate(target_counts.tolist())
            if 0 < c <= rare_threshold}, rare_threshold


def _error_flags(rec, prev_gold, prev_majority, rare_ids, vocab, confidence):
    """The cause flags of one error record (model-side fields live in `rec`).

    Pure: every flag names a *checkable* property of the error, so the same
    function serves a single model, both arms of a paired comparison and any
    future per-token re-analysis.
    """
    gold, pred, prev = rec['gold'], rec['pred'], rec['prev']
    flags = {}
    flags['ambiguous_topk'] = bool(rec['gold_in_topk'])
    maj = prev_majority.get(prev)
    gold_here = prev_gold.get(prev * vocab + gold, 0)
    flags['prev_context_ambiguous'] = bool(
        maj is not None and maj[0] != gold and maj[1] >= gold_here and gold_here > 0
    )
    flags['rare_gold'] = gold in rare_ids
    flags['repetition_bias'] = bool(pred in rec['context'] and gold not in rec['context'])
    flags['high_confidence_wrong'] = bool(rec['pred_prob'] >= confidence)
    return flags


def _primary_cause(flags):
    """One disjoint cause per error, with the overlap kept in `flags`."""
    if flags['ambiguous_topk']:
        return 'label_ambiguity_topk'
    if flags['prev_context_ambiguous']:
        return 'context_ambiguity'
    if flags['rare_gold']:
        return 'long_tail_target'
    if flags['high_confidence_wrong'] and flags['repetition_bias']:
        return 'model_blind_spot_copying'
    if flags['high_confidence_wrong']:
        return 'model_blind_spot_high_confidence'
    if flags['repetition_bias']:
        return 'model_copying_bias'
    return 'unexplained'


def _classify_scan(error_scan, prev_gold, prev_majority, vocab, rare_ids, confidence):
    """Attach flags and a primary cause to every scanned error record.

    Returns ``(categories, primary, examples)``; `categories` counts the
    overlapping flags, `primary` the disjoint primary causes.
    """
    categories = Counter()
    primary = Counter()
    examples = []
    for rec in error_scan:
        flags = _error_flags(rec, prev_gold, prev_majority, rare_ids, vocab, confidence)
        for name, value in flags.items():
            if value:
                categories[name] += 1
        cause = _primary_cause(flags)
        primary[cause] += 1
        record = dict(rec)
        record['flags'] = flags
        record['primary_cause'] = cause
        examples.append(record)
    return categories, primary, examples


def analyze(model, batches, args, tokenizer=None, device='cpu') -> dict:
    """Run the model over `batches` and build the error taxonomy report."""
    cfg = model.cfg
    ignore = int(cfg.ignore_index)
    vocab = int(cfg.vocab_size)
    target_counts = torch.zeros(vocab, dtype=torch.long)
    # Confusion-matrix marginals for the named F1 report: per-class
    # predict/gold/correct counts.
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
    # All scan counters are device tensors. Each batch's `(key, count)` bigram
    # histogram is merged after the pass.
    dev = next(model.parameters()).device
    prev_gold_parts: list = []   # per-batch (unique keys, counts)
    # The fixed AUC class columns live on the model's device.
    auc_ids_t = torch.as_tensor(auc_class_ids, device=dev) if auc_class_ids else None
    n_tokens_t = torch.zeros((), dtype=torch.long, device=dev)
    n_correct_t = torch.zeros((), dtype=torch.long, device=dev)
    n_errors_t = torch.zeros((), dtype=torch.long, device=dev)
    n_positional_t = torch.zeros((), dtype=torch.long, device=dev)
    topk_hits_t = torch.zeros((), dtype=torch.long, device=dev)
    gold_prob_sum_t = torch.zeros((), dtype=torch.float64, device=dev)
    gold_rank_sum_t = torch.zeros((), dtype=torch.float64, device=dev)
    target_counts_t = torch.zeros(vocab, dtype=torch.long, device=dev)
    pred_counts_t = torch.zeros(vocab, dtype=torch.long, device=dev)
    correct_counts_t = torch.zeros(vocab, dtype=torch.long, device=dev)
    copy_input_correct_t = torch.zeros((), dtype=torch.long, device=dev)
    topk = max(1, int(args.topk))
    # A k above the number of classes cannot be measured: top-k with k >= V is
    # the whole distribution. Clamp to vocab - 1 so the report key names the
    # radius actually analyzed.
    analyzed_topk = min(topk, max(1, vocab - 1))
    n_scanned = 0
    error_scan = []              # capped records, classified after the pass
    # Printed examples are selected across the *whole* pass, not just its first
    # `--max-scan-errors` mistakes: a confident blind spot that appears after
    # the scan cap is exactly the case this analysis exists to surface. The
    # heap keeps the most confident records seen so far; the bounded taxonomy
    # sample above stays sequential so its cause distribution is representative.
    example_limit = max(0, int(args.max_examples))
    example_heap: list = []
    example_counter = 0

    if first_batch is not None:
        batches = itertools.chain([first_batch], batches)
    fwd_chunk = max(0, int(getattr(args, 'forward_batch_chunk', 0) or 0))
    for x, y in batches:
        mask = y.ne(ignore)
        prev = x
        valid = mask
        n_tokens_t += valid.sum()
        target_counts_t += torch.bincount(y[valid].long(), minlength=vocab)
        keys = prev[valid].long() * vocab + y[valid].long()
        prev_gold_parts.append(torch.unique(keys, return_counts=True))
        if tuple(x.shape) == tuple(y.shape):
            copy_input_correct_t += (x[valid] == y[valid]).sum()
        B, T = x.shape
        chunk = max(1, int(args.eval_chunk))
        step = fwd_chunk if fwd_chunk > 0 else B
        # Forward the batch in row blocks. Every model row is independent, so
        # the statistics are unchanged (integer statistics are identical;
        # float accumulators can differ at the last ulp because the reduction
        # grouping changes with the block size).
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            mask_b = mask[b0:b1]
            y_b = y[b0:b1]
            # `norm_f` and `lm_head` are per-position, so the vocabulary
            # projection runs one position chunk at a time: `--eval-chunk`
            # bounds the logits buffer.
            with torch.no_grad():
                hidden = model.forward_hidden(x[b0:b1])
            for t0 in range(0, T, chunk):
                t1 = min(T, t0 + chunk)
                m = mask_b[:, t0:t1]
                v_idx = m.nonzero(as_tuple=False)
                if v_idx.numel() == 0:
                    continue
                with torch.no_grad():
                    lc = model.lm_head(model.norm_f(hidden[:, t0:t1])).detach().float()
                # Score only the valid rows.
                probs = torch.softmax(lc[m], dim=-1)
                gold = y_b[:, t0:t1][m]
                pred = probs.argmax(-1)
                n_correct_t += (pred == gold).sum()
                pred_counts_t += torch.bincount(pred, minlength=vocab)
                correct_counts_t += torch.bincount(gold[pred == gold], minlength=vocab)
                gold_p = probs.gather(-1, gold.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                gold_prob_sum_t += gold_p.sum().double()
                k = analyzed_topk
                top = probs.topk(k, dim=-1)
                # 1-based rank; ties count above (>= counts the gold token
                # itself, every strictly-larger probability and every tie, so
                # a tie cannot flatter the model's ranking).
                gold_rank = (probs >= gold_p.unsqueeze(-1)).sum(-1)
                topk_hits_t += (top.indices == gold.unsqueeze(-1)).any(-1).sum()
                gold_rank_sum_t += gold_rank.clamp(max=args.rank_clip).sum().double()
                n_positional_t += pred.numel()
                wrong = pred != gold
                n_errors_t += wrong.sum()

                def _error_record(vi: int) -> dict:
                    b, tt = v_idx[vi].tolist()
                    return {
                        'batch_row': b0 + b,
                        'position': t0 + tt,
                        'gold': int(gold[vi]),
                        'pred': int(pred[vi]),
                        'gold_prob': float(gold_p[vi]),
                        'pred_prob': float(probs[vi, pred[vi]]),
                        'gold_rank': int(min(int(gold_rank[vi]), args.rank_clip)),
                        'gold_in_topk': bool((top.indices[vi] == gold[vi]).any()),
                        'prev': int(prev[b0 + b, t0 + tt]),
                        'context': [int(v) for v in x[b0 + b, max(0, t0 + tt - args.context_tokens):t0 + tt + 1].tolist()],
                    }

                if n_scanned < args.max_scan_errors or example_limit > 0:
                    idx = wrong.nonzero(as_tuple=False).flatten()
                    if n_scanned < args.max_scan_errors:
                        for vi in idx[: args.max_scan_errors - n_scanned].tolist():
                            n_scanned += 1
                            error_scan.append(_error_record(vi))
                    if example_limit > 0 and idx.numel():
                        # Keep the `example_limit` most confident mistakes over
                        # the full pass (a min-heap pops the weakest).
                        wrong_pred_prob = probs.gather(
                            -1, pred.unsqueeze(-1)).squeeze(-1)[wrong]
                        take = min(example_limit, int(wrong_pred_prob.numel()))
                        for pos in torch.topk(wrong_pred_prob, take).indices.tolist():
                            record = _error_record(int(idx[pos]))
                            heapq.heappush(
                                example_heap,
                                (record['pred_prob'], example_counter, record))
                            example_counter += 1
                            if len(example_heap) > example_limit:
                                heapq.heappop(example_heap)
                if auc_ids_t is not None:
                    # `index_select` picks the scored columns directly.
                    sel = probs.index_select(-1, auc_ids_t)
                    take = int(sel.shape[0])
                    if auc_max_tokens > 0:
                        take = min(take, auc_max_tokens - auc_seen)
                    if take > 0:
                        auc_scores.append(sel[:take].detach().cpu())
                        auc_labels.append(gold[:take].detach().cpu())
                        auc_seen += take
                del probs, top, lc
            del hidden

    n_tokens = int(n_tokens_t.item())
    if n_tokens == 0:
        raise SystemExit('[errors] no valid target tokens in the selected batches')
    n_correct = int(n_correct_t.item())
    n_errors = int(n_errors_t.item())
    n_positional = int(n_positional_t.item())
    topk_hits = int(topk_hits_t.item())
    gold_prob_sum = float(gold_prob_sum_t.item())
    gold_rank_sum = float(gold_rank_sum_t.item())
    copy_input_correct = int(copy_input_correct_t.item())
    target_counts = target_counts_t.cpu()
    pred_counts = pred_counts_t.cpu()
    correct_counts = correct_counts_t.cpu()

    # One merge of the per-batch bigram histograms (integer counts, exact).
    prev_gold_hist = None
    if prev_gold_parts:
        merged_keys = torch.cat([p[0] for p in prev_gold_parts])
        merged_counts = torch.cat([p[1] for p in prev_gold_parts])
        uniq, inv = torch.unique(merged_keys, return_inverse=True)
        counts = torch.zeros_like(uniq)
        counts.index_add_(0, inv, merged_counts)
        prev_gold_hist = (uniq, counts)
        del merged_keys, merged_counts, inv, counts, prev_gold_parts
    prev_gold, prev_majority = _prev_majority(prev_gold_hist, vocab)
    rare_ids, rare_threshold = _rare_ids(target_counts, args.rare_quantile)
    categories, primary, _ = _classify_scan(
        error_scan, prev_gold, prev_majority, vocab, rare_ids, args.confidence)
    _, _, example_records = _classify_scan(
        [record for _, _, record in example_heap], prev_gold, prev_majority,
        vocab, rare_ids, args.confidence)

    example_records.sort(key=lambda r: (-r['pred_prob'], r['gold_rank']))
    shown = example_records[:example_limit]
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
            'scanned errors; flag_counts overlap because one error can carry several flags. '
            f'Printed examples are the {example_limit} most confident errors over the whole '
            'pass, not the first ones, so a late confident blind spot is not hidden by the '
            'taxonomy scan cap'
        ),
    }


def _chunk_stats(logits_chunk, gold, topk, mask):
    """Per-position prediction statistics of one chunk, for pairing two models.

    Rank and top-k membership are only evaluated on the `mask` (valid target)
    positions; ignore_index positions never contribute a statistic.
    """
    probs = torch.softmax(logits_chunk.detach().float(), dim=-1)
    pred = probs.argmax(-1)
    gold_p = probs.gather(-1, gold.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    pred_p = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)
    gold_rank = torch.zeros_like(gold)
    gold_in_topk = torch.zeros_like(gold, dtype=torch.bool)
    if bool(mask.any()):
        probs_m = probs[mask]
        gold_rank[mask] = (probs_m >= gold_p[mask].unsqueeze(-1)).sum(-1)
        top = probs_m.topk(topk, dim=-1)
        gold_in_topk[mask] = (top.indices == gold[mask].unsqueeze(-1)).any(-1)
    return {
        'pred': pred,
        'gold_prob': gold_p,
        'pred_prob': pred_p,
        'gold_rank': gold_rank,
        'gold_in_topk': gold_in_topk,
        'ok': pred == gold,
    }


def _side_record(stats, b, tt, rank_clip):
    """One model's per-token record at (b, tt), detached to plain Python."""
    return {
        'pred': int(stats['pred'][b, tt]),
        'pred_prob': float(stats['pred_prob'][b, tt]),
        'gold_prob': float(stats['gold_prob'][b, tt]),
        'gold_rank': int(min(stats['gold_rank'][b, tt].item(), rank_clip)),
        'gold_in_topk': bool(stats['gold_in_topk'][b, tt]),
    }


def _side_error_input(rec, side_name):
    """Reshape a paired record into the single-model error-record schema."""
    side = rec[side_name]
    return {
        'gold': rec['gold'], 'pred': side['pred'], 'prev': rec['prev'],
        'context': rec['context'], 'gold_in_topk': side['gold_in_topk'],
        'pred_prob': side['pred_prob'],
    }


def _bump(cause_summary, cause, field):
    cause_summary.setdefault(cause, Counter())[field] += 1


def analyze_pair(reference_model, candidate_model, batches, args, tokenizer=None, device='cpu') -> dict:
    """Paired bad-case analysis of a candidate against a reference checkpoint.

    A single-model error taxonomy says *how often* a model is wrong; the
    decision "did the change help?" is a statement about the *difference* of
    two models on the same samples. Both models consume the identical batches
    here (the tensors are shared, not re-drawn), so every token is a paired
    observation with four possible outcomes: both right, both wrong, fixed
    (reference wrong / candidate right) and broken (reference right /
    candidate wrong).

    The report is built for hypothesis generation, in this order:

    * the exact McNemar test on the discordant pairs (`fixed`, `broken`):
      only the tokens where the two models disagree carry information about
      which one is better, and at the few-hundred-error scale this analysis
      runs at the exact test is the honest reference;
    * each error is classified with the same taxonomy as the single-model
      analysis, so `fixed_by_reference_cause` names *what kind* of error the
      candidate repairs and `broken_by_candidate_cause` names the new ones;
    * the ``shared blind spots`` - tokens both models get confidently wrong,
      often with the same wrong prediction - are printed separately: these
      are the strongest label-noise / inherent-ambiguity candidates, because
      two independently trained arms agreeing on a wrong answer is not a
      property of either model's mechanism;
    * the cost side (parameters, analytic FLOPs) of both checkpoints travels
      in the same report, so a fixed-minus-broken count is read next to what
      the candidate costs.

    The AUC block is not duplicated here: it is a single-model ranking
    diagnostic, and it would not sharpen the pairing.
    """
    ref_cfg, cand_cfg = reference_model.cfg, candidate_model.cfg
    if int(ref_cfg.vocab_size) != int(cand_cfg.vocab_size):
        raise SystemExit(
            f"[pair] the checkpoints disagree on vocab_size ({int(ref_cfg.vocab_size)} vs "
            f"{int(cand_cfg.vocab_size)}): their predicted classes are not comparable, "
            "so a paired analysis would be meaningless")
    if int(ref_cfg.ignore_index) != int(cand_cfg.ignore_index):
        raise SystemExit(
            f"[pair] the checkpoints disagree on ignore_index ({int(ref_cfg.ignore_index)} vs "
            f"{int(cand_cfg.ignore_index)}): the two models would be scored on different tokens")
    ignore = int(ref_cfg.ignore_index)
    vocab = int(ref_cfg.vocab_size)
    topk = min(max(1, int(args.topk)), max(1, vocab - 1))
    max_scan = max(0, int(args.max_scan_errors))
    dev = next(reference_model.parameters()).device
    prev_gold_parts: list = []   # per-batch (unique keys, counts), merged once
    target_counts_t = torch.zeros(vocab, dtype=torch.long, device=dev)
    n_tokens_t = torch.zeros((), dtype=torch.long, device=dev)
    n_both_right_t = torch.zeros((), dtype=torch.long, device=dev)
    n_fixed_t = torch.zeros((), dtype=torch.long, device=dev)
    n_broken_t = torch.zeros((), dtype=torch.long, device=dev)
    n_both_wrong_t = torch.zeros((), dtype=torch.long, device=dev)
    n_both_wrong_confident_t = torch.zeros((), dtype=torch.long, device=dev)
    copy_input_correct_t = torch.zeros((), dtype=torch.long, device=dev)
    n_positional_t = torch.zeros((), dtype=torch.long, device=dev)
    scans = {'fixed': [], 'broken': [], 'both_wrong': []}
    scanned = {'fixed': 0, 'broken': 0, 'both_wrong': 0}
    # The printed examples are the most confident records over the *whole*
    # paired pass, not just the first `--max-scan-errors` per bucket. The
    # bounded `scans` above still define the cause taxonomies.
    example_limit = max(0, int(args.max_examples))
    example_heaps = {'fixed': [], 'broken': [], 'both_wrong': []}
    example_counters = {'fixed': 0, 'broken': 0, 'both_wrong': 0}
    fwd_chunk = max(0, int(getattr(args, 'forward_batch_chunk', 0) or 0))
    for x, y in batches:
        mask = y.ne(ignore)
        n_tokens_t += mask.sum()
        target_counts_t += torch.bincount(y[mask].long(), minlength=vocab)
        keys = x[mask].long() * vocab + y[mask].long()
        prev_gold_parts.append(torch.unique(keys, return_counts=True))
        if tuple(x.shape) == tuple(y.shape):
            copy_input_correct_t += (x[mask] == y[mask]).sum()
        n_positional_t += mask.sum()
        B, T = x.shape
        chunk = max(1, int(args.eval_chunk))
        step = fwd_chunk if fwd_chunk > 0 else B
        for b0 in range(0, B, step):
            b1 = min(B, b0 + step)
            x_b, y_b, mask_b = x[b0:b1], y[b0:b1], mask[b0:b1]
            bounds = [(t0, min(T, t0 + chunk)) for t0 in range(0, T, chunk)
                      if bool(mask_b[:, t0:min(T, t0 + chunk)].any())]
            if not bounds:
                continue
            # Reference and candidate never hold a full logits block at the
            # same time: the vocabulary projection is done one position chunk
            # at a time (on top of the block hidden states) and each chunk's
            # logits are dropped before the next forward runs.
            ref_stats = []
            with torch.no_grad():
                h_ref = reference_model.forward_hidden(x_b)
                for t0, t1 in bounds:
                    logits_ref = reference_model.lm_head(reference_model.norm_f(h_ref[:, t0:t1]))
                    ref_stats.append(_chunk_stats(logits_ref, y_b[:, t0:t1], topk, mask_b[:, t0:t1]))
                    del logits_ref
            del h_ref
            cand_stats = []
            with torch.no_grad():
                h_cand = candidate_model.forward_hidden(x_b)
                for t0, t1 in bounds:
                    logits_cand = candidate_model.lm_head(candidate_model.norm_f(h_cand[:, t0:t1]))
                    cand_stats.append(_chunk_stats(logits_cand, y_b[:, t0:t1], topk, mask_b[:, t0:t1]))
                    del logits_cand
            del h_cand
            for (t0, t1), stats_ref, stats_cand in zip(bounds, ref_stats, cand_stats):
                m = mask_b[:, t0:t1]
                gold = y_b[:, t0:t1]
                selectors = {
                    'fixed': m & ~stats_ref['ok'] & stats_cand['ok'],
                    'broken': m & stats_ref['ok'] & ~stats_cand['ok'],
                    'both_wrong': m & ~stats_ref['ok'] & ~stats_cand['ok'],
                }
                n_both_right_t += (m & stats_ref['ok'] & stats_cand['ok']).sum()
                n_fixed_t += selectors['fixed'].sum()
                n_broken_t += selectors['broken'].sum()
                n_both_wrong_t += selectors['both_wrong'].sum()
                # Both arms confident and agreeing on the same wrong token.
                n_both_wrong_confident_t += (
                    selectors['both_wrong']
                    & (stats_ref['pred'] == stats_cand['pred'])
                    & (torch.minimum(stats_ref['pred_prob'], stats_cand['pred_prob'])
                       >= args.confidence)).sum()
                def _pair_record(b: int, tt: int) -> dict:
                    pos = t0 + tt
                    return {
                        'batch_row': b0 + b,
                        'position': pos,
                        'gold': int(gold[b, tt]),
                        'prev': int(x_b[b, pos]),
                        'context': [int(v) for v in x_b[b, max(0, pos - args.context_tokens):pos + 1].tolist()],
                        'reference': _side_record(stats_ref, b, tt, args.rank_clip),
                        'candidate': _side_record(stats_cand, b, tt, args.rank_clip),
                    }

                for status, selector in selectors.items():
                    if max_scan > scanned[status]:
                        remaining = max_scan - scanned[status]
                        for b, tt in selector.nonzero(as_tuple=False)[:remaining].tolist():
                            scanned[status] += 1
                            scans[status].append(_pair_record(b, tt))
                    if example_limit > 0 and bool(selector.any()):
                        # Confidence of the side whose error the bucket is
                        # about. For shared blind spots use the weaker of the
                        # two, so both models are confidently wrong.
                        if status == 'fixed':
                            metric = stats_ref['pred_prob']
                        elif status == 'broken':
                            metric = stats_cand['pred_prob']
                        else:
                            metric = torch.minimum(
                                stats_ref['pred_prob'], stats_cand['pred_prob'])
                        flat_selector = selector.reshape(-1)
                        flat_metric = metric.reshape(-1).masked_fill(
                            ~flat_selector, float('-inf'))
                        take = min(example_limit, int(flat_metric.numel()))
                        for flat in torch.topk(flat_metric, take).indices.tolist():
                            if not bool(flat_selector[flat]):
                                continue
                            b, tt = divmod(flat, selector.shape[1])
                            record = _pair_record(b, tt)
                            heapq.heappush(
                                example_heaps[status],
                                (float(metric[b, tt]), example_counters[status], record))
                            example_counters[status] += 1
                            if len(example_heaps[status]) > example_limit:
                                heapq.heappop(example_heaps[status])
                del stats_ref, stats_cand, selectors

    n_tokens = int(n_tokens_t.item())
    if n_tokens == 0:
        raise SystemExit('[pair] no valid target tokens in the selected batches')
    n_both_right = int(n_both_right_t.item())
    n_fixed = int(n_fixed_t.item())
    n_broken = int(n_broken_t.item())
    n_both_wrong = int(n_both_wrong_t.item())
    n_both_wrong_confident = int(n_both_wrong_confident_t.item())
    copy_input_correct = int(copy_input_correct_t.item())
    n_positional = int(n_positional_t.item())
    target_counts = target_counts_t.cpu()

    # One merge of the per-batch bigram histograms (integer counts, exact).
    prev_gold_hist = None
    if prev_gold_parts:
        merged_keys = torch.cat([p[0] for p in prev_gold_parts])
        merged_counts = torch.cat([p[1] for p in prev_gold_parts])
        uniq, inv = torch.unique(merged_keys, return_inverse=True)
        counts = torch.zeros_like(uniq)
        counts.index_add_(0, inv, merged_counts)
        prev_gold_hist = (uniq, counts)
        del merged_keys, merged_counts, inv, counts, prev_gold_parts
    prev_gold, prev_majority = _prev_majority(prev_gold_hist, vocab)
    rare_ids, rare_threshold = _rare_ids(target_counts, args.rare_quantile)
    cause_summary: dict = {}
    reference_cause = Counter()
    candidate_cause = Counter()
    examples = {'fixed': [], 'broken': [], 'shared_blind_spots': []}

    def _classify_one(rec, side_name):
        flags = _error_flags(_side_error_input(rec, side_name), prev_gold,
                             prev_majority, rare_ids, vocab, args.confidence)
        return flags, _primary_cause(flags)

    for rec in scans['fixed']:
        flags, cause = _classify_one(rec, 'reference')
        reference_cause[cause] += 1
        _bump(cause_summary, cause, 'reference_errors')
        _bump(cause_summary, cause, 'fixed')
        rec['reference_flags'] = flags
        rec['reference_cause'] = cause
        examples['fixed'].append(rec)
    for rec in scans['broken']:
        flags, cause = _classify_one(rec, 'candidate')
        candidate_cause[cause] += 1
        _bump(cause_summary, cause, 'candidate_errors')
        _bump(cause_summary, cause, 'broken')
        rec['candidate_flags'] = flags
        rec['candidate_cause'] = cause
        examples['broken'].append(rec)
    for rec in scans['both_wrong']:
        ref_flags, ref_cause = _classify_one(rec, 'reference')
        cand_flags, cand_cause = _classify_one(rec, 'candidate')
        reference_cause[ref_cause] += 1
        candidate_cause[cand_cause] += 1
        _bump(cause_summary, ref_cause, 'reference_errors')
        _bump(cause_summary, cand_cause, 'candidate_errors')
        rec['reference_flags'] = ref_flags
        rec['candidate_flags'] = cand_flags
        rec['reference_cause'] = ref_cause
        rec['candidate_cause'] = cand_cause
        examples['shared_blind_spots'].append(rec)

    # Printed examples come from the full-pass confidence heaps. Records that
    # were already in the bounded taxonomy scan keep their flags; the rest are
    # classified here without touching the cause counts.
    for status, heap in example_heaps.items():
        records = [record for _, _, record in heap]
        for rec in records:
            if status in ('fixed', 'broken'):
                side = 'reference' if status == 'fixed' else 'candidate'
                if f'{side}_cause' not in rec:
                    flags, cause = _classify_one(rec, side)
                    rec[f'{side}_flags'] = flags
                    rec[f'{side}_cause'] = cause
            else:
                if 'reference_cause' not in rec:
                    flags, cause = _classify_one(rec, 'reference')
                    rec['reference_flags'] = flags
                    rec['reference_cause'] = cause
                if 'candidate_cause' not in rec:
                    flags, cause = _classify_one(rec, 'candidate')
                    rec['candidate_flags'] = flags
                    rec['candidate_cause'] = cause
        examples[status if status != 'both_wrong' else 'shared_blind_spots'] = records

    examples['fixed'].sort(key=lambda r: (-r['reference']['pred_prob'], r['reference']['gold_rank']))
    examples['broken'].sort(key=lambda r: (-r['candidate']['pred_prob'], r['candidate']['gold_rank']))
    examples['shared_blind_spots'].sort(
        key=lambda r: (-min(r['reference']['pred_prob'], r['candidate']['pred_prob']),
                       r['reference']['gold_rank']))
    shown = {status: records[:example_limit] for status, records in examples.items()}
    for records in shown.values():
        for rec in records:
            rec['context_text'] = _decode(rec['context'], tokenizer)
            rec['gold_text'] = _decode([rec['gold']], tokenizer)
            rec['reference']['pred_text'] = _decode([rec['reference']['pred']], tokenizer)
            rec['candidate']['pred_text'] = _decode([rec['candidate']['pred']], tokenizer)

    def _cost(model):
        breakdown = parameter_breakdown(model)
        flops = analytic_flops_per_token(model.cfg, seq_len=int(args.seq_len))
        return {
            'n_params': breakdown['total'],
            'norm_share_of_total': breakdown['norm_share_of_total'],
            'parameter_breakdown': breakdown['by_role'],
            'flops_per_token_forward': flops['forward_per_token'],
        }

    cost = {'reference': _cost(reference_model), 'candidate': _cost(candidate_model)}
    ref_params, cand_params = cost['reference']['n_params'], cost['candidate']['n_params']
    ref_flops, cand_flops = (cost['reference']['flops_per_token_forward'],
                             cost['candidate']['flops_per_token_forward'])
    cost['params_ratio_candidate_over_reference'] = (cand_params / ref_params) if ref_params else None
    cost['flops_ratio_candidate_over_reference'] = (cand_flops / ref_flops) if ref_flops else None
    cost['note'] = ('parameters and analytic matmul FLOPs at the analysis seq_len; norm layers '
                    'are elementwise so they barely move FLOPs')

    bigram_correct = sum(count for _, count in prev_majority.values())
    mcnemar = mcnemar_test(n_fixed, n_broken)
    summary = {
        'n_tokens': n_tokens,
        'reference_accuracy': (n_tokens - n_fixed - n_both_wrong) / n_tokens,
        'candidate_accuracy': (n_tokens - n_broken - n_both_wrong) / n_tokens,
        'accuracy_delta_candidate_minus_reference': (n_fixed - n_broken) / n_tokens,
        'fixed': n_fixed,
        'broken': n_broken,
        'both_wrong': n_both_wrong,
        'both_right': n_both_right,
        'net_fixed_minus_broken': n_fixed - n_broken,
        'fixed_per_1000_tokens': 1000.0 * n_fixed / n_tokens,
        'broken_per_1000_tokens': 1000.0 * n_broken / n_tokens,
        'mcnemar': mcnemar,
        'shared_blind_spot_confident': n_both_wrong_confident,
        'reference_errors_total': n_fixed + n_both_wrong,
        'candidate_errors_total': n_broken + n_both_wrong,
        'majority_class_accuracy': float(target_counts.max().item()) / n_tokens,
        'bigram_majority_accuracy': bigram_correct / n_tokens,
        'uniform_random_accuracy': 1.0 / vocab,
        'copy_input_accuracy': (copy_input_correct / n_tokens) if n_positional else None,
        'topk': topk,
        'errors_scanned': scanned,
    }
    return {
        'summary': summary,
        'cost': cost,
        'cause_summary': {c: dict(v) for c, v in sorted(cause_summary.items())},
        'reference_cause_distribution': dict(reference_cause.most_common()),
        'candidate_cause_distribution': dict(candidate_cause.most_common()),
        'reading_guide': {
            'fixed': ('reference errors the candidate repaired, classified by why the '
                      'reference failed: this names what the candidate mechanism buys'),
            'broken': ('errors the candidate introduced, classified by the candidate cause: '
                       'this is the candidate\'s new failure mode'),
            'shared_blind_spots': ('both arms wrong on the same token; when both are confident '
                                   '(min pred_prob >= --confidence) and predict the same wrong '
                                   'token, suspect label noise / inherent ambiguity rather than '
                                   'a model-specific defect - that class is not fixable by either '
                                   'mechanism'),
            'mcnemar': 'exact paired test over the discordant (fixed/broken) tokens only',
            'cost': ('parameters / analytic FLOPs of both checkpoints next to the error delta'),
        },
        'examples': shown,
        'rarest_target_threshold': rare_threshold,
        'note': (
            f'discordant counts (fixed/broken) are exact over all {n_tokens} scored tokens; '
            f'the taxonomy covers the first {max_scan} scanned records per bucket '
            f'(fixed={scanned["fixed"]}, broken={scanned["broken"]}, '
            f'both_wrong={scanned["both_wrong"]}); printed examples are the '
            f'{example_limit} most confident records per bucket over the whole pass, '
            'so a late confident blind spot is not hidden by the scan cap. '
            'The AUC block is not run in pair mode.'
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True, help='checkpoint to analyze; with --reference-ckpt this is the candidate')
    p.add_argument('--reference-ckpt', default=None, help='second checkpoint: when given, the analysis becomes a paired reference-vs-candidate bad-case comparison on identical batches (fixed/broken errors, exact McNemar, shared blind spots) - see analyze_pair')
    p.add_argument('--corpus', choices=['markov', 'dataset'], default='markov')
    p.add_argument('--split', choices=['validation', 'test', 'train'], default='validation')
    p.add_argument('--n-batches', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--eval-chunk', type=int, default=64, help='positions per softmax chunk')
    p.add_argument('--forward-batch-chunk', type=int, default=4, help='rows per forward pass: the (rows, seq, vocab) logits block is forwarded in slices of at most this many rows, bounding the analysis peak; each model row is independent, so the statistics are unchanged (0 = one pass over the whole batch)')
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
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this error analysis tests; recorded in the report and manifest so the result is auditable as a test of a claim rather than a fishing trip')
    # dataset-corpus options (must match the trained run's split protocol)
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--dataset-revision', default=None, help='pin the dataset revision (commit hash) exactly as the trained run did; without it the analysis can read a different upstream snapshot')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--tokenizer-revision', default=None, help='pin the tokenizer revision (commit hash) exactly as the trained run did')
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--val-frac', type=float, default=0.005)
    p.add_argument('--test-frac', type=float, default=0.005)
    p.add_argument('--num-documents', type=int, default=None)
    p.add_argument('--min-doc-chars', type=int, default=0)
    p.add_argument('--max-doc-chars', type=int, default=0)
    p.add_argument('--no-dedup', action='store_true')
    p.add_argument('--exclude-hash-file', default=None)
    p.add_argument('--exclude-test-hash-file', default=None)
    p.add_argument('--group-key', default=None, help='grouping-unit field of the trained run (see train_gpt2.py --group-key)')
    p.add_argument('--exclude-group-file', default=None)
    p.add_argument('--exclude-test-group-file', default=None)
    p.add_argument('--allow-sealed-test', action='store_true', help="permit inspecting the sealed test split; reading its error cases and iterating turns test into an implicit validation set, so use --split validation unless this is the single final diagnosis")
    args = p.parse_args()
    args.topk = max(1, int(args.topk))
    if args.n_batches < 1:
        p.error(f'--n-batches {args.n_batches} must be >= 1')
    if args.corpus == 'dataset' and args.split == 'test' and (not args.allow_sealed_test):
        p.error("--corpus dataset --split test inspects the sealed test split; reading its "
                "error cases and iterating turns test into an implicit validation set. Use "
                "--split validation, or pass --allow-sealed-test for the single final diagnosis.")
    if args.corpus == 'dataset' and args.split == 'test':
        print("[errors] WARNING: inspecting the sealed test split (--allow-sealed-test); "
              "this must remain the single final diagnosis.", file=sys.stderr)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(args.seed)
    model, model_args = load_ckpt(args.ckpt, device=device)
    # Keep the CLI value identical to the analyzed radius (see `analyze`) so
    # the printed/`config`-recorded topk names the key that is in the summary.
    args.topk = max(1, min(int(args.topk), max(1, int(model.cfg.vocab_size) - 1)))
    print(f"[errors] loaded {args.ckpt} | norm={model_args['norm']} | corpus={args.corpus} | device={device}")
    protocol_mismatches = ({}
                           if args.corpus != 'dataset'
                           else data_protocol_mismatches(args, model_args))
    reference_model = reference_args = None
    reference_protocol_mismatches = {}
    if args.reference_ckpt:
        reference_model, reference_args = load_ckpt(args.reference_ckpt, device=device)
        print(f"[pair] reference {args.reference_ckpt} | norm={reference_args['norm']}")
        if args.corpus == 'dataset':
            reference_protocol_mismatches = data_protocol_mismatches(args, reference_args)
    protocol_checks = [('checkpoint', protocol_mismatches)]
    if reference_model is not None:
        protocol_checks.append(('reference checkpoint', reference_protocol_mismatches))
    for label, mismatches in protocol_checks:
        if not mismatches:
            continue
        print(f'[errors] WARNING: the analysis data protocol differs from the {label}\'s '
              'training protocol; the samples are not the ones the model was selected on:',
              file=sys.stderr)
        for key, diff in mismatches.items():
            print(f"[errors]   {key}: checkpoint={diff['checkpoint']!r} "
                  f"analysis={diff['analysis']!r}", file=sys.stderr)
        print('[errors] pass the checkpoint\'s data flags (including --dataset-revision / '
              '--tokenizer-revision) so the bad cases come from the same held-out documents.',
              file=sys.stderr)

    # A checkpoint's weights themselves must be finite before any result is
    # read out of it.
    checkpoint_health = {'candidate': parameter_finite_report(model)}
    if reference_model is not None:
        checkpoint_health['reference'] = parameter_finite_report(reference_model)
    for label, health in checkpoint_health.items():
        if not health['finite']:
            print(f"[errors] ERROR: the {label} checkpoint has "
                  f"{health['n_nonfinite_parameters']} non-finite parameter tensor(s) "
                  f"and {health['n_nonfinite_buffers']} non-finite buffer(s): "
                  f"{health['nonfinite_parameters'][:5]}")
            print(f"[errors] {health['recommendation']}", file=sys.stderr)
            raise SystemExit(1)
        print(f"[errors] {label} checkpoint weights finite ({health['n_parameters']} tensors checked)")

    tokenizer = None
    data_split: dict = {}
    if args.corpus == 'markov':
        batches = _markov_batches(args, model.cfg, device)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                  revision=args.tokenizer_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        batches = _dataset_batches(args, model, device, report_sink=data_split)

    if reference_model is not None:
        report = analyze_pair(reference_model, model, batches, args,
                              tokenizer=tokenizer, device=device)
        report['model_args'] = {k: v for k, v in model_args.items()
                                if isinstance(v, (str, int, float, bool, type(None)))}
        report['reference_model_args'] = {k: v for k, v in reference_args.items()
                                          if isinstance(v, (str, int, float, bool, type(None)))}
        out = args.output or str(Path(args.ckpt).parent / f'error_analysis_pair_{args.corpus}.json')
    else:
        report = analyze(model, batches, args, tokenizer=tokenizer, device=device)
        report['model_args'] = {k: v for k, v in model_args.items()
                                if isinstance(v, (str, int, float, bool, type(None)))}
        out = args.output or str(Path(args.ckpt).parent / f'error_analysis_{args.corpus}.json')
    report['config'] = vars(args)
    # Resolved data identity of the batches the taxonomy was computed on
    # (revisions, boundary points, hashes), captured from the loader while the
    # batches were consumed - a bad-case report without it cannot be paired
    # with the trained run it is meant to explain.
    report['data_split'] = data_split.get('split_report')
    report['data_protocol_mismatches'] = protocol_mismatches
    if reference_model is not None:
        report['reference_data_protocol_mismatches'] = reference_protocol_mismatches
    report['checkpoint_health'] = checkpoint_health
    report['run_metadata'] = collect_run_metadata(seed=args.seed)
    report['hypothesis'] = args.hypothesis

    if reference_model is not None:
        s = report['summary']
        m = s['mcnemar']
        print(f"[pair] tokens={s['n_tokens']} reference_accuracy={s['reference_accuracy']:.4f} "
              f"candidate_accuracy={s['candidate_accuracy']:.4f} "
              f"delta={s['accuracy_delta_candidate_minus_reference']:+.4f}")
        print(f"[pair] fixed={s['fixed']} ({s['fixed_per_1000_tokens']:.1f}/1k tok) "
              f"broken={s['broken']} ({s['broken_per_1000_tokens']:.1f}/1k tok) "
              f"net={s['net_fixed_minus_broken']:+d} both_wrong={s['both_wrong']} "
              f"both_right={s['both_right']}")
        p1 = 'n/a' if m['p_one_sided'] is None else f"{m['p_one_sided']:.4g}"
        p2 = 'n/a' if m['p_two_sided'] is None else f"{m['p_two_sided']:.4g}"
        print(f"[pair] exact McNemar on {m['n_discordant']} discordant tokens: "
              f"p_one_sided={p1} p_two_sided={p2} favors={m['favors']}")
        if m['p_two_sided'] is not None and m['p_two_sided'] < 0.05 and m['favors'] == 'reference':
            print("[pair] WARNING: the candidate is significantly WORSE than the reference on the "
                  "discordant tokens; read the BROKEN examples before proposing the change as an improvement")
        elif m['p_one_sided'] is not None and m['p_one_sided'] >= 0.05:
            print("[pair] NOTE: the fixed-minus-broken difference is not significant at 5%; "
                  "the point estimate is within what coin-flipping the discordant tokens explains")
        cost = report['cost']
        print(f"[pair] cost: params={cost['reference']['n_params']} -> "
              f"{cost['candidate']['n_params']} "
              f"(x{cost['params_ratio_candidate_over_reference']:.4f}), "
              f"flops/token={cost['reference']['flops_per_token_forward']:.4g} -> "
              f"{cost['candidate']['flops_per_token_forward']:.4g} "
              f"(x{cost['flops_ratio_candidate_over_reference']:.4f})")
        print(f"[pair] reference causes (scanned reference errors): "
              f"{report['reference_cause_distribution']}")
        print(f"[pair] candidate causes (scanned candidate errors): "
              f"{report['candidate_cause_distribution']}")
        print(f"[pair] shared blind spots: {s['both_wrong']} tokens, "
              f"{s['shared_blind_spot_confident']} with both arms confident "
              f"(candidate label noise / inherent ambiguity)")
        for status, label in (('fixed', 'FIXED'), ('broken', 'BROKEN'),
                              ('shared_blind_spots', 'SHARED-BLIND-SPOT')):
            for rec in report['examples'][status]:
                ref, cand = rec['reference'], rec['candidate']
                print(f"[pair] {label} pos={rec['position']:>5d} gold={rec['gold_text']!r} "
                      f"ref_pred={ref['pred_text']!r}(p={ref['pred_prob']:.3f}, rank={ref['gold_rank']}) "
                      f"cand_pred={cand['pred_text']!r}(p={cand['pred_prob']:.3f}, rank={cand['gold_rank']}) "
                      f"gold_p_ref={ref['gold_prob']:.3f} gold_p_cand={cand['gold_prob']:.3f}")
                print(f"[pair]   context: ...{rec['context_text']}")
    else:
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
        print(f"[errors] suspected label-noise / context-ambiguity cases: "
              f"{report['flag_counts'].get('prev_context_ambiguous', 0)}; "
              f"confident blind spots: "
              f"{report['flag_counts'].get('high_confidence_wrong', 0)}")
        for rec in report['examples']:
            flags = ','.join(k for k, v in rec['flags'].items() if v) or 'none'
            print(f"[errors] pos={rec['position']:>5d} gold={rec['gold_text']!r}(p={rec['gold_prob']:.3f}, rank={rec['gold_rank']}) "
                  f"pred={rec['pred_text']!r}(p={rec['pred_prob']:.3f}) cause={rec['primary_cause']} flags={flags}")
            print(f"[errors]   context: ...{rec['context_text']}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2, default=str))
    manifest = {
        'run_metadata': report['run_metadata'], 'ckpt': args.ckpt,
        'hypothesis': args.hypothesis,
        'corpus': args.corpus, 'config': vars(args),
        'summary': report['summary'], 'checkpoint_health': checkpoint_health,
    }
    if args.reference_ckpt:
        manifest['reference_ckpt'] = args.reference_ckpt
    manifest = finish_metadata(manifest, artifacts={'report': file_fingerprint(out)})
    save_run_manifest(Path(out).with_name('error_analysis_manifest.json'), manifest)
    if args.hypothesis:
        print(f'[errors][hypothesis] {args.hypothesis}')
    print(f'[errors] report -> {out}')


if __name__ == '__main__':
    main()
