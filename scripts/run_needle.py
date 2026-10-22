from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash.metrics import holm_adjust, mcnemar_test, mean_ci, paired_t_one_sided, wilson_interval
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds
from scripts.eval_downstream import (DEFAULT_HAYSTACK, DEFAULT_QUESTION, load_ckpt,
                                       needle_decode_overhead, needle_in_haystack)

# One placement draw is one run-level sample: a single seed's Wilson interval
# describes only the trial noise inside that draw, so a "retains needle
# accuracy" claim needs the placement dimension replicated too. `--seeds`
# repeats the whole evaluation once per placement seed and reports the
# cross-seed Mean +/- Std (and a t interval over the seed means), keeping the
# per-seed records and the per-trial pass/fail vectors. Two checkpoints scored
# with the same seeds see the *same* placements, so `--compare-to` pairs them:
# trial-level exact McNemar on the pooled discordant trials plus a paired t on
# the per-seed accuracies, instead of eyeballing two accuracies whose intervals
# overlap. A comparison certified at the placement level also needs >= 3
# shared placement seeds (2 seeds have df=1, where the t interval is too wide
# to be a useful decision rule), and the per-context p-values are Holm-adjusted
# before any context is called a win: scanning 4 context lengths at alpha=0.05
# without correction gives a ~19% chance of at least one false positive.
DEFAULT_SEEDS = (0, 1, 2)
MIN_SHARED_PLACEMENT_SEEDS = 3
# Training-recipe fields that must match before a checkpoint delta can be
# attributed to the norm/quantization treatment. Norm internals
# (`norm`, `qk_norm`, `alpha_init`, `k_init`, `p_init`, `no_learnable_*`,
# `no_layerscale`, ...) are deliberately absent: they *are* the treatment.
# Training seeds are absent too: independent seeds are the replication unit of
# the training-run variance this comparison does not cover.
# Shape overrides are optional: the trainer only records them when the
# corresponding ADASQUASH_*_OVERRIDE env var was set, so two normal
# checkpoints both lack all three. Absence from both therefore means "no
# override", not "unknown".
OPTIONAL_PROTOCOL_KEYS = (
    '_depth_override', '_dim_override', '_heads_override',
)
# Concrete Hub commits recorded next to the mutable requested revision. They
# are compared only when at least one checkpoint records them, so older
# checkpoints still fall back to the requested-revision comparison above.
RESOLVED_PROTOCOL_KEYS = ('dataset_revision_resolved', 'tokenizer_revision_resolved')
TRAINING_PROTOCOL_KEYS = (
    'model', *OPTIONAL_PROTOCOL_KEYS,
    'dataset', 'dataset_config', 'dataset_revision', 'tokenizer', 'tokenizer_revision',
    'val_frac', 'test_frac', 'num_documents', 'min_doc_chars', 'max_doc_chars',
    'dedup', 'group_key', 'exclude_hash_file', 'exclude_test_hash_file',
    'exclude_group_file', 'exclude_test_group_file', 'train_shuffle_buffer',
    'seq_len', 'batch_size', 'grad_accum', 'max_steps', 'lr', 'min_lr_ratio',
    'warmup', 'weight_decay', 'beta1', 'beta2', 'grad_clip', 'bf16',
    'num_workers', 'gradient_checkpointing', 'tie_embeddings',
    'val_every', 'val_batches', 'early_stop_patience', 'nonfinite_policy',
    'test_batches',
)

def comparison_protocol_guard(candidate_args: dict, reference_args: dict) -> dict:
    """Check that two checkpoints were trained under one comparable recipe.

    A checkpoint pair can differ in the norm (the treatment) but not in the
    data snapshot, step count, sequence length or optimizer recipe, otherwise
    a "needle accuracy delta" is a confound. `_checkpoint_step` is handled
    separately: when both checkpoints record it and the steps differ, the
    weights are not the same point in training and the pair is marked
    incomparable. A missing step is reported (and certification requires both
    steps to be known), not silently treated as equal.
    """
    mismatches = {}
    missing = '<missing>'
    for key in TRAINING_PROTOCOL_KEYS:
        # A field absent from *both* checkpoints is unknown, not equal: an old
        # or incomplete checkpoint must not certify as protocol-compatible.
        # Optional fields are the exception: both absent means neither run
        # used an override, while present-vs-absent is still a real mismatch.
        if key not in candidate_args and key not in reference_args:
            if key not in OPTIONAL_PROTOCOL_KEYS:
                mismatches[key] = {'candidate': missing, 'reference': missing}
            continue
        candidate_value = candidate_args.get(key, missing)
        reference_value = reference_args.get(key, missing)
        if candidate_value != reference_value:
            mismatches[key] = {'candidate': candidate_value,
                               'reference': reference_value}
    for key in RESOLVED_PROTOCOL_KEYS:
        if key not in candidate_args and key not in reference_args:
            continue
        candidate_value = candidate_args.get(key, missing)
        reference_value = reference_args.get(key, missing)
        if candidate_value != reference_value:
            # A concrete commit on one side and none (or a different commit)
            # on the other means the data/tokenizer identity is not proven
            # equal, even when both requested revisions were None.
            mismatches[key] = {'candidate': candidate_value,
                               'reference': reference_value}
    candidate_step = candidate_args.get('_checkpoint_step')
    reference_step = reference_args.get('_checkpoint_step')
    steps_known = candidate_step is not None and reference_step is not None
    if steps_known and candidate_step != reference_step:
        mismatches['_checkpoint_step'] = {'candidate': candidate_step,
                                          'reference': reference_step}
    if steps_known:
        step_basis = f'both checkpoints record step {candidate_step}'
    else:
        step_basis = (f'checkpoint step not recorded for '
                      f'{"candidate" if candidate_step is None else "reference"} '
                      '- the compared weights are not verifiably at the same training step')
    return {
        'ok': not mismatches,
        'mismatches': mismatches or None,
        'keys_compared': list(TRAINING_PROTOCOL_KEYS),
        'keys_excluded_as_treatment': [
            'norm', 'qk_norm', 'qk_alpha_init', 'alpha_init', 'alpha_0',
            'k_init', 'p_init', 'no_learnable_p', 'no_learnable_k',
            'no_learnable_delta', 'layerscale_init', 'no_layerscale', 'no_beta',
            'seed', 'seeds'],
        'candidate_checkpoint_step': candidate_step,
        'reference_checkpoint_step': reference_step,
        'checkpoint_steps_known': steps_known,
        'checkpoint_step_basis': step_basis,
    }

def _print_protocol_guard(guard: dict) -> None:
    steps = (f"candidate step={guard['candidate_checkpoint_step']}, "
             f"reference step={guard['reference_checkpoint_step']}")
    if guard['ok']:
        print(f'[needle][protocol] comparable recipes ({steps}; '
              f'{len(guard["keys_compared"])} protocol fields matched)')
    else:
        print(f'[needle][protocol] recipe mismatch: {guard["mismatches"]} ({steps})',
              file=sys.stderr)
    if not guard['checkpoint_steps_known']:
        print(f"[needle][protocol] {guard['checkpoint_step_basis']}; a certified "
              'comparison requires both steps to be known', file=sys.stderr)

def _checkpoint_tokenizer(ckpt_args: dict, fallback_name: str):
    """The tokenizer identity the checkpoint was trained with, else the CLI default.

    Use the concrete commit recorded at training time when present: loading
    the mutable requested branch could tokenize the needle differently from
    the run that produced the weights.
    """
    name = ckpt_args.get('tokenizer') or fallback_name
    revision = (ckpt_args.get('tokenizer_revision_resolved')
                or ckpt_args.get('tokenizer_revision'))
    return name, revision

def evaluate_model(model, tok, args, device, seeds, trained_seq_len):
    """One checkpoint's needle sweep over `args.ctx_lens` x placement seeds."""
    results = []
    skipped = []
    overhead = needle_decode_overhead(tok, args.secret)
    # The haystack builder needs at least one repeat of the filler plus the
    # needle and question; a smaller requested length cannot be evaluated at
    # all, so report it as skipped instead of letting the first seed raise.
    min_context = (len(tok(args.secret, add_special_tokens=False)['input_ids'])
                   + len(tok(DEFAULT_QUESTION, add_special_tokens=False)['input_ids'])
                   + len(tok(DEFAULT_HAYSTACK, add_special_tokens=False)['input_ids']))
    for L in args.ctx_lens:
        if L < min_context:
            reason = (f'{L} is too short to hold the needle + question + one haystack '
                      f'repeat (minimum {min_context} tokens)')
            print(f'[skip] L={L}: {reason}')
            skipped.append({'ctx_len': L, 'reason': reason,
                            'model_max_seq_len': int(model.cfg.max_seq_len),
                            'scoring_overhead': overhead,
                            'minimum_context': min_context})
            continue
        if L + overhead > model.cfg.max_seq_len:
            reason = (f'{L} + {overhead} scoring positions for the secret exceed '
                      f'max_seq_len={model.cfg.max_seq_len}; RoPE scaling needed')
            print(f'[skip] L={L} (+{overhead} scoring positions for the secret) > max_seq_len={model.cfg.max_seq_len}, RoPE scaling needed - skipping for now.')
            skipped.append({'ctx_len': L, 'reason': reason,
                            'model_max_seq_len': int(model.cfg.max_seq_len),
                            'scoring_overhead': overhead})
            continue
        per_seed = []
        for placement_seed in seeds:
            set_all_seeds(placement_seed)
            res = needle_in_haystack(model, tok, needle=args.secret, context_len=L, n_trials=args.trials, device=device, seed=placement_seed, return_counts=True, decode_batch=args.decode_batch)
            per_seed.append({
                'seed': placement_seed,
                'accuracy': res['accuracy'],
                'n_correct': res['n_correct'], 'n_trials': res['n_trials'],
                'trial_correct': res['trial_correct'],
                'ci95_low': res['ci95_low'], 'ci95_high': res['ci95_high'],
                'failures': res['failures'],
                'failure_causes': res['failure_causes'],
                'failure_confidence': res['failure_confidence'],
            })
        accs = [r['accuracy'] for r in per_seed]
        n_correct = sum(r['n_correct'] for r in per_seed)
        n_trials = sum(r['n_trials'] for r in per_seed)
        if len(per_seed) == 1:
            # Legacy single-seed shape: the Wilson interval is exactly what the
            # single run supports - its trial noise - and nothing about
            # placement variability.
            first = per_seed[0]
            record = {'ctx_len': L, 'acc': first['accuracy'],
                      'n_correct': first['n_correct'], 'n_trials': first['n_trials'],
                      'ci95_low': first['ci95_low'], 'ci95_high': first['ci95_high'],
                      'ci_basis': 'wilson_within_single_placement_draw',
                      'acc_std': None, 'acc_ci95_low': None, 'acc_ci95_high': None,
                      'pooled_wilson_low': None, 'pooled_wilson_high': None,
                      'failures': first['failures'],
                      'failure_causes': first['failure_causes'],
                      'failure_confidence': first['failure_confidence']}
            print(f"[needle] L={L:5d} acc={record['acc']:.3f} "
                  f"({record['n_correct']}/{record['n_trials']}, 95% CI "
                  f"[{record['ci95_low']:.3f}, {record['ci95_high']:.3f}], trial noise only - "
                  f"single placement draw; pass --seeds for placement-level replication)")
        else:
            ci = mean_ci(accs)
            low, high = wilson_interval(n_correct, n_trials)
            record = {'ctx_len': L, 'acc': ci['mean'], 'n_correct': n_correct, 'n_trials': n_trials,
                      'ci95_low': ci['ci_low'], 'ci95_high': ci['ci_high'],
                      'ci_basis': 't_across_seed_means',
                      'acc_std': ci['std'], 'acc_ci95_low': ci['ci_low'], 'acc_ci95_high': ci['ci_high'],
                      'pooled_wilson_low': low, 'pooled_wilson_high': high,
                      'failures': [{**f, 'seed': r['seed']} for r in per_seed for f in r['failures']],
                      'failure_causes': dict(Counter(f['cause'] for r in per_seed for f in r['failures'])),
                      'failure_confidence': dict(Counter(f['confidence'] for r in per_seed for f in r['failures']))}
            print(f"[needle] L={L:5d} acc={record['acc']:.3f} +/- {record['acc_std']:.3f} "
                  f"(cross-seed CI [{record['ci95_low']:.3f}, {record['ci95_high']:.3f}], n_seeds={len(per_seed)}; "
                  f"pooled Wilson [{low:.3f}, {high:.3f}] covers trial noise only)")
        record['n_seeds'] = len(per_seed)
        record['seeds'] = list(seeds)
        record['per_seed'] = per_seed
        record['trial_correct'] = [c for r in per_seed for c in r['trial_correct']]
        record['exceeds_trained_seq_len'] = (trained_seq_len is not None and L + overhead > trained_seq_len)
        # Bad-case records: a bare accuracy cannot tell "never found the needle"
        # (retrieval_miss) from "found it, then lost the thread" (decode_drift/
        # context_copy) or from a confident wrong answer; print decoded
        # examples and keep every failure record in the JSON.
        if record['failure_causes']:
            print(f"[needle] L={L:5d} failure causes: {record['failure_causes']} "
                  f"confidence: {record['failure_confidence']}")
            for f in record['failures'][:max(0, args.max_failure_examples)]:
                def _tok(i):
                    try:
                        return repr(tok.decode([int(i)]))
                    except Exception:
                        return f'id={i}'
                print(f"[needle] L={L:5d}   trial={f['trial']} needle_pos={f['needle_position']} "
                      f"failed at secret token {f['failure_step'] + 1}/{f['n_target_tokens']} "
                      f"({f['cause']}/{f['confidence']}): expected {_tok(f['expected_id'])} "
                      f"(rank {f['expected_rank']}, p={f['expected_prob']:.4f}) got "
                      f"{_tok(f['predicted_id'])} (p={f['predicted_prob']:.4f}, "
                      f"margin={f['logit_margin_over_expected']:+.2f}, "
                      f"in_context={f['predicted_in_context']})")
        if record['exceeds_trained_seq_len']:
            print(f"[needle][warn] L={L} + {overhead} scoring positions exceeds the "
                  f"trained seq_len={trained_seq_len}: this is positional extrapolation, "
                  f"not measured in-distribution long-context ability")
        results.append(record)
    return results, skipped

def compare_needle(results, reference_results, alpha=0.05, protocol=None):
    """Paired candidate-vs-reference tests on the shared placements.

    Both arms were scored with the same seeds, so within a seed the trial
    sequence is the same placement sequence: trial-level outcomes are paired
    and pooled across seeds into an exact McNemar test (`n_fixed` = reference
    missed, candidate hit). The per-seed accuracies are additionally tested
    with a paired t (`H1: mean(reference - candidate) < 0`, so a negative
    `mean_delta` means the candidate is better), which weights each placement
    draw equally instead of each trial.

    `protocol` is the `comparison_protocol_guard` verdict. A pair whose
    training recipes (data, steps, seq_len, optimizer) differ, or whose
    checkpoint steps are not verifiably identical, is reported but never
    certified: placement-level pairing cannot remove a training-recipe
    confound.
    """
    ref_by_ctx = {r['ctx_len']: r for r in reference_results}
    tests = []
    for cand in results:
        ref = ref_by_ctx.get(cand['ctx_len'])
        if ref is None:
            tests.append({'ctx_len': cand['ctx_len'], 'testable': False,
                          'reason': 'context length was not evaluated for the reference'})
            continue
        ref_seeds = {r['seed']: r for r in ref['per_seed']}
        n_fixed = n_broke = 0
        paired_accs = []
        paired_seeds = []
        unpaired = []
        for cand_seed in cand['per_seed']:
            ref_seed = ref_seeds.get(cand_seed['seed'])
            if ref_seed is None:
                unpaired.append(cand_seed['seed'])
                continue
            c_vec = list(cand_seed['trial_correct'])
            r_vec = list(ref_seed['trial_correct'])
            if len(c_vec) != len(r_vec):
                unpaired.append(cand_seed['seed'])
                continue
            n_fixed += sum((not rv) and bool(cv) for rv, cv in zip(r_vec, c_vec))
            n_broke += sum(bool(rv) and (not cv) for rv, cv in zip(r_vec, c_vec))
            paired_accs.append((ref_seed['accuracy'], cand_seed['accuracy']))
            paired_seeds.append(cand_seed['seed'])
        t = {'ctx_len': cand['ctx_len'], 'testable': bool(paired_accs),
             'paired_seeds': paired_seeds,
             'unpaired_seeds': unpaired,
             'candidate_acc': cand['acc'], 'reference_acc': ref['acc'],
             'mcnemar': mcnemar_test(n_fixed, n_broke)}
        if not paired_accs:
            # Always carry a reason: consumers print it for non-testable rows.
            t['reason'] = ('no shared placement seeds with matching trial vectors '
                           'between candidate and reference')
        if paired_accs:
            ref_accs = [a for a, _ in paired_accs]
            cand_accs = [b for _, b in paired_accs]
            t.update(paired_t_one_sided(ref_accs, cand_accs, alpha=alpha))
            t['direction'] = 'negative mean_delta means the candidate (--ckpt) is better than --compare-to'
        tests.append(t)
    paired_holm = holm_adjust([t.get('p_one_sided') for t in tests])
    mcnemar_holm = holm_adjust([t.get('mcnemar', {}).get('p_one_sided') for t in tests])
    # No protocol verdict means the training recipe and checkpoint step were
    # never verified. That is unknown, not equal: fail closed and report the
    # comparison descriptively rather than certifying a confounded delta.
    protocol_ok = bool(protocol.get('ok')) if isinstance(protocol, dict) else False
    steps_verified = (bool(protocol.get('checkpoint_steps_known'))
                      if isinstance(protocol, dict) else False)
    for t, p_holm, m_holm in zip(tests, paired_holm, mcnemar_holm):
        n_shared = len(t.get('paired_seeds') or [])
        t['n_shared_placement_seeds'] = n_shared
        t['p_one_sided_holm'] = p_holm
        t['protocol_ok'] = protocol_ok
        t['checkpoint_steps_verified'] = steps_verified
        t['certifiable'] = bool(t.get('testable')
                                and n_shared >= MIN_SHARED_PLACEMENT_SEEDS
                                and protocol_ok and steps_verified)
        delta = t.get('mean_delta')
        t['candidate_better'] = bool(delta is not None and delta < 0.0)
        t['significant_holm'] = bool(
            t['certifiable'] and t['candidate_better']
            and p_holm is not None and p_holm < alpha
        )
        if not t['certifiable']:
            reasons = []
            if n_shared < MIN_SHARED_PLACEMENT_SEEDS:
                reasons.append(f'{n_shared} shared placement seed(s) < '
                               f'{MIN_SHARED_PLACEMENT_SEEDS}')
            if not protocol_ok:
                reasons.append('training recipes differ')
            if not steps_verified:
                reasons.append('checkpoint steps not verifiably identical')
            t['significance_basis'] = (
                'descriptive only: ' + ('; '.join(reasons) or 'not testable')
                + ' - the placement-level t/McNemar cannot certify this delta')
        else:
            t['significance_basis'] = (
                f'placement-level paired test on {n_shared} shared placement seeds with '
                'identical training recipes and checkpoint steps; training-run variance '
                'is not covered by this comparison')
        m = t.get('mcnemar')
        if isinstance(m, dict):
            m['p_one_sided_holm'] = m_holm
            m['candidate_better'] = m.get('n_fixed', 0) > m.get('n_broke', 0)
            m['significant_holm'] = bool(
                t['certifiable'] and m['candidate_better']
                and m_holm is not None and m_holm < alpha
            )
    return {'alpha': alpha,
            'unit_of_replication': 'placement draw within each fixed training checkpoint',
            'min_shared_placement_seeds': MIN_SHARED_PLACEMENT_SEEDS,
            'multiple_comparison': ('Holm-Bonferroni over the evaluated context lengths, '
                                    'applied separately to the paired-t and exact-McNemar families'),
            'certification': ('a context is certified only when the two checkpoints share '
                              f'>= {MIN_SHARED_PLACEMENT_SEEDS} placement seeds, were trained '
                              'under identical data/step/sequence/optimizer recipes, both record '
                              'the same checkpoint step, and the Holm-adjusted one-sided p < '
                              'alpha; anything else is reported as descriptive only, and no '
                              'training-run variance is inferred'),
            'pairing': ('arms share placement seeds, so trial outcomes are paired within a seed; '
                        'McNemar is exact on the pooled discordant trials, and the paired t weights '
                        'each placement draw equally'),
            'tests': tests}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--compare-to', default=None, help='reference checkpoint to compare against with the same placement seeds; adds paired McNemar + per-seed paired t to the payload (the comparison --ckpt vs --compare-to). The pair must share the training recipe (data, seq_len, steps, optimizer) and checkpoint step or the comparison is refused')
    p.add_argument('--allow-protocol-mismatch', action='store_true', help='report a comparison whose training recipe or checkpoint step does not match, but never certify it (the payload records the mismatched fields)')
    p.add_argument('--ctx-lens', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    p.add_argument('--trials', type=int, default=20, help='independent placements per seed; the Wilson interval covers these trials only, the cross-seed Mean +/- Std covers placement draws')
    p.add_argument('--decode-batch', type=int, default=8, help='independent trials decoded per forward pass (the per-trial prefix is still recomputed every step)')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--secret', default='The magic number is 98765.')
    p.add_argument('--output', default=None)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; when given it overrides --seeds (one placement draw - the Wilson interval then says nothing about placement variability)')
    p.add_argument('--seeds', nargs='+', type=int, default=None, help='placement seeds; the whole evaluation is repeated once per seed and reported as cross-seed Mean +/- Std with a t interval over the seed means (default: 0 1 2, the minimum for a placement-level comparison)')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this evaluation tests (e.g. "DyT-CD retains needle accuracy at 4k tokens"); recorded in the manifest so a pass/fail is auditable as a test rather than a fishing trip')
    p.add_argument('--max-failure-examples', type=int, default=3, help='bad-case samples to print per context length (all failures are kept in the JSON)')
    args = p.parse_args()
    if args.trials < 1:
        p.error(f'--trials {args.trials} must be >= 1: a trial count of zero measures nothing')
    if args.decode_batch < 1:
        p.error(f'--decode-batch {args.decode_batch} must be >= 1')
    requested_ctx_lens = list(args.ctx_lens)
    if not requested_ctx_lens:
        p.error('--ctx-lens must not be empty: a sweep with no context length measures nothing')
    for ctx_len in requested_ctx_lens:
        if ctx_len < 1:
            p.error(f'--ctx-lens entries must be >= 1, got {ctx_len}')
    args.ctx_lens = list(dict.fromkeys(int(ctx_len) for ctx_len in requested_ctx_lens))
    if len(args.ctx_lens) != len(requested_ctx_lens):
        print(f'[needle][warn] duplicate context lengths removed: {requested_ctx_lens} -> '
              f'{args.ctx_lens} (a repeated length is the same hypothesis, not an '
              'independent test in the Holm family)', file=sys.stderr)
    if args.compare_to and not args.hypothesis:
        p.error('--compare-to requires --hypothesis "..." so the claim, direction and context '
                'family are registered before the paired comparison is run, not selected after '
                'seeing which context length moved')
    seeds = [args.seed] if args.seed is not None else (list(args.seeds) if args.seeds else list(DEFAULT_SEEDS))
    if not seeds:
        p.error('--seeds must not be empty')
    raw_seeds = seeds
    seeds = list(dict.fromkeys(int(s) for s in seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[needle][warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              '(a repeated seed reruns the identical placement draw, it is not '
              'an independent replication)', file=sys.stderr)
    if len(seeds) < 3:
        print(f'[needle][warn] {len(seeds)} placement seed(s): one placement draw is one sample; pass >= 3 seeds '
              'so one lucky needle position cannot decide the result', file=sys.stderr)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seeds[0])
    model, model_args = load_ckpt(args.ckpt, device=device)
    model_max_seq_len = int(model.cfg.max_seq_len)
    trained_seq_len = model_args.get('seq_len')
    trained_seq_len = int(trained_seq_len) if isinstance(trained_seq_len, int) else None
    if trained_seq_len is None:
        print('[needle][warn] checkpoint args carry no seq_len: whether a context length was inside '
              'the training distribution cannot be checked')
    from transformers import AutoTokenizer
    tok_name, tok_revision = _checkpoint_tokenizer(model_args, args.tokenizer)
    tok = AutoTokenizer.from_pretrained(tok_name, revision=tok_revision)
    tok_revision_resolved = getattr(tok, '_commit_hash', None) or tok_revision
    results, skipped = evaluate_model(model, tok, args, device, seeds, trained_seq_len)
    comparison = None
    if args.compare_to:
        # The candidate's results are plain Python records, so the model is
        # released before the reference is loaded.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ref_model, ref_args = load_ckpt(args.compare_to, device=device)
        # The reference's own training context length decides which context
        # lengths are in-distribution for it; reusing the candidate's would
        # mislabel the reference's in/out-of-distribution slices.
        ref_trained_seq_len = ref_args.get('seq_len')
        ref_trained_seq_len = (int(ref_trained_seq_len)
                               if isinstance(ref_trained_seq_len, int) else None)
        ref_tok_name, ref_tok_revision = _checkpoint_tokenizer(ref_args, args.tokenizer)
        if (ref_tok_name, ref_tok_revision) != (tok_name, tok_revision):
            raise SystemExit(
                f'[needle] tokenizer mismatch: candidate uses {tok_name!r}@{tok_revision!r}, '
                f'reference uses {ref_tok_name!r}@{ref_tok_revision!r}. Paired placement '
                'comparisons require identical tokenization; re-train with one tokenizer '
                'or evaluate the checkpoints separately.')
        protocol = comparison_protocol_guard(model_args, ref_args)
        _print_protocol_guard(protocol)
        if not protocol['ok'] and not args.allow_protocol_mismatch:
            raise SystemExit(
                '[needle] refusing to compare checkpoints trained under different recipes: '
                f"{protocol['mismatches']}. A recipe/step difference confounds the norm effect; "
                're-train both arms under one protocol, compare ckpts at the same step, or pass '
                '--allow-protocol-mismatch to get a descriptive (never certified) comparison.')
        ref_results, _ = evaluate_model(ref_model, tok, args, device, seeds, ref_trained_seq_len)
        comparison = compare_needle(results, ref_results, alpha=0.05, protocol=protocol)
        comparison['protocol_guard'] = protocol
        comparison['candidate'] = args.ckpt
        comparison['reference'] = args.compare_to
        for t in comparison['tests']:
            if not t.get('testable'):
                print(f"[needle][compare] L={t['ctx_len']}: not testable "
                      f"({t.get('reason', 'no reason recorded')})")
                continue
            m = t['mcnemar']
            verdict = m['favors']
            if m['p_one_sided'] is None:
                verdict = 'no discordant trials - no evidence either way'
            mcnemar_p = m['p_one_sided'] if m['p_one_sided'] is None else round(m['p_one_sided'], 4)
            mcnemar_holm = m.get('p_one_sided_holm')
            mcnemar_holm = 'n/a' if mcnemar_holm is None else f'{mcnemar_holm:.4f}'
            head = (f"[needle][compare] L={t['ctx_len']}: candidate {t['candidate_acc']:.3f} vs reference "
                    f"{t['reference_acc']:.3f}; McNemar fixed={m['n_fixed']} broke={m['n_broke']} "
                    f"p={mcnemar_p} p_holm={mcnemar_holm} "
                    f"(favors {verdict}); paired t delta={t['mean_delta']:+.4f}")
            if t['p_one_sided'] is None:
                print(f"{head} (n_seeds={t['n_shared_placement_seeds']}: no paired t interval; "
                      f"{t['significance_basis']})")
            else:
                verdict_t = 'CERTIFIED candidate better' if t['significant_holm'] else 'not certified'
                print(f"{head} 95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
                      f"p={t['p_one_sided']:.4f} p_holm={t['p_one_sided_holm']:.4f} "
                      f"(n_seeds={t['n_shared_placement_seeds']}, {verdict_t}; "
                      f"{t['significance_basis']})")
        print(f"[needle][compare] {comparison['multiple_comparison']}; "
              f"{comparison['certification']}")
    out = args.output or Path(args.ckpt).parent / 'needle_results.json'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    # Skipped context lengths are part of the result, not a console aside: a
    # JSON holding only the lengths that ran silently reads as "the model held
    # up to the maximum requested length" when the longest ones were never
    # measured.
    payload = {
        'ckpt': args.ckpt,
        'model_max_seq_len': model_max_seq_len,
        'trained_seq_len': trained_seq_len,
        'tokenizer': tok_name,
        'tokenizer_revision': tok_revision,
        'tokenizer_revision_resolved': tok_revision_resolved,
        'secret': args.secret,
        'trials': args.trials,
        'placement_seeds': list(seeds),
        'ctx_lens_requested': requested_ctx_lens,
        # Only lengths that actually produced a result; skipped lengths are
        # listed separately and must not look measured in the evaluated set.
        'ctx_lens_evaluated': [r['ctx_len'] for r in results],
        'runs': results,
        'skipped': skipped,
        'ci_note': ('ci95_low/high are the cross-seed t interval when >= 2 placement seeds are run '
                    '(ci_basis="t_across_seed_means"); with one seed they are the Wilson interval '
                    '(ci_basis="wilson_within_single_placement_draw"), which covers trial noise within '
                    'that one placement draw only. A --compare-to win additionally requires >= 3 '
                    'shared placement seeds and a Holm-adjusted one-sided p < alpha'),
    }
    if comparison is not None:
        payload['compare_to'] = args.compare_to
        payload['reference_tokenizer'] = ref_tok_name
        payload['reference_tokenizer_revision'] = ref_tok_revision
        payload['comparison'] = comparison
    Path(out).write_text(json.dumps(payload, indent=2))
    manifest = collect_run_metadata(seed=seeds[0])
    manifest['hyperparameters'] = vars(args)
    manifest['placement_seeds'] = list(seeds)
    manifest['tokenizer'] = tok_name
    manifest['tokenizer_revision'] = tok_revision
    manifest['tokenizer_revision_resolved'] = tok_revision_resolved
    if args.hypothesis:
        manifest['hypothesis'] = args.hypothesis
        print(f'[hypothesis] {args.hypothesis}')
    else:
        print('[hypothesis] none stated: pass --hypothesis "..." so this evaluation is auditable as a test of a claim, not a sweep')
    manifest['results_file'] = str(out)
    manifest['trained_seq_len'] = trained_seq_len
    manifest['skipped_ctx_lens'] = [s['ctx_len'] for s in skipped]
    manifest['evaluated_ctx_lens'] = [r['ctx_len'] for r in results]
    if args.compare_to:
        manifest['compare_to'] = args.compare_to
    if results:
        manifest = finish_metadata(manifest, artifacts={'results': file_fingerprint(out)})
    else:
        # The SystemExit below reports that nothing was measured; the manifest
        # must not claim a finished evaluation.
        manifest['status'] = 'failed'
    save_run_manifest(Path(out).with_name('needle_manifest.json'), manifest)
    print(f'[done] saved to {out} ({len(results)} context length(s) evaluated, '
          f'{len(skipped)} skipped: {[s["ctx_len"] for s in skipped]})')
    if not results:
        print('[needle][error] no requested context length could be evaluated; nothing was measured',
              file=sys.stderr)
        raise SystemExit(2)
if __name__ == '__main__':
    main()
