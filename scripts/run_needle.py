from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash.metrics import mcnemar_test, mean_ci, paired_t_one_sided, wilson_interval
from adasquash.repro import collect_run_metadata, save_run_manifest, set_all_seeds
from scripts.eval_downstream import load_ckpt, needle_decode_overhead, needle_in_haystack

# One placement draw is one run-level sample: a single seed's Wilson interval
# describes only the trial noise inside that draw, so a "retains needle
# accuracy" claim needs the placement dimension replicated too. `--seeds`
# repeats the whole evaluation once per placement seed and reports the
# cross-seed Mean +/- Std (and a t interval over the seed means), keeping the
# per-seed records and the per-trial pass/fail vectors. Two checkpoints scored
# with the same seeds see the *same* placements, so `--compare-to` pairs them:
# trial-level exact McNemar on the pooled discordant trials plus a paired t on
# the per-seed accuracies, instead of eyeballing two accuracies whose intervals
# overlap.
DEFAULT_SEED = 0

def evaluate_model(model, tok, args, device, seeds, trained_seq_len):
    """One checkpoint's needle sweep over `args.ctx_lens` x placement seeds."""
    results = []
    skipped = []
    overhead = needle_decode_overhead(tok, args.secret)
    for L in args.ctx_lens:
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
        record['exceeds_trained_seq_len'] = (trained_seq_len is not None and L > trained_seq_len)
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
            print(f"[needle][warn] L={L} exceeds the trained seq_len={trained_seq_len}: this is "
                  f"positional extrapolation, not measured in-distribution long-context ability")
        results.append(record)
    return results, skipped

def compare_needle(results, reference_results, alpha=0.05):
    """Paired candidate-vs-reference tests on the shared placements.

    Both arms were scored with the same seeds, so within a seed the trial
    sequence is the same placement sequence: trial-level outcomes are paired
    and pooled across seeds into an exact McNemar test (`n_fixed` = reference
    missed, candidate hit). The per-seed accuracies are additionally tested
    with a paired t (`H1: mean(reference - candidate) < 0`, so a negative
    `mean_delta` means the candidate is better), which weights each placement
    draw equally instead of each trial.
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
        if paired_accs:
            ref_accs = [a for a, _ in paired_accs]
            cand_accs = [b for _, b in paired_accs]
            t.update(paired_t_one_sided(ref_accs, cand_accs, alpha=alpha))
            t['direction'] = 'negative mean_delta means the candidate (--ckpt) is better than --compare-to'
        tests.append(t)
    return {'alpha': alpha,
            'pairing': ('arms share placement seeds, so trial outcomes are paired within a seed; '
                        'McNemar is exact on the pooled discordant trials, and the paired t weights '
                        'each placement draw equally'),
            'tests': tests}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--compare-to', default=None, help='reference checkpoint to compare against with the same placement seeds; adds paired McNemar + per-seed paired t to the payload (the comparison --ckpt vs --compare-to)')
    p.add_argument('--ctx-lens', nargs='+', type=int, default=[512, 1024, 2048, 4096])
    p.add_argument('--trials', type=int, default=20, help='independent placements per seed; the Wilson interval covers these trials only, the cross-seed Mean +/- Std covers placement draws')
    p.add_argument('--decode-batch', type=int, default=8, help='independent trials decoded per forward pass (bounds peak memory; the per-trial prefix is still recomputed every step)')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--secret', default='The magic number is 98765.')
    p.add_argument('--output', default=None)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; when given it overrides --seeds (one placement draw - the Wilson interval then says nothing about placement variability)')
    p.add_argument('--seeds', nargs='+', type=int, default=None, help='placement seeds; the whole evaluation is repeated once per seed and reported as cross-seed Mean +/- Std with a t interval over the seed means (default: [0], the legacy single draw)')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this evaluation tests (e.g. "DyT-CD retains needle accuracy at 4k tokens"); recorded in the manifest so a pass/fail is auditable as a test rather than a fishing trip')
    p.add_argument('--max-failure-examples', type=int, default=3, help='bad-case samples to print per context length (all failures are kept in the JSON)')
    args = p.parse_args()
    seeds = [args.seed] if args.seed is not None else (list(args.seeds) if args.seeds else [DEFAULT_SEED])
    if not seeds:
        p.error('--seeds must not be empty')
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
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    results, skipped = evaluate_model(model, tok, args, device, seeds, trained_seq_len)
    comparison = None
    if args.compare_to:
        # The candidate's results are plain Python records, so the model is
        # released before the reference is loaded: two checkpoints resident at
        # once would double the parameter footprint for no benefit.
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
        ref_results, _ = evaluate_model(ref_model, tok, args, device, seeds, ref_trained_seq_len)
        comparison = compare_needle(results, ref_results, alpha=0.05)
        comparison['candidate'] = args.ckpt
        comparison['reference'] = args.compare_to
        for t in comparison['tests']:
            if not t.get('testable'):
                print(f"[needle][compare] L={t['ctx_len']}: not testable ({t['reason']})")
                continue
            m = t['mcnemar']
            verdict = m['favors']
            if m['p_one_sided'] is None:
                verdict = 'no discordant trials - no evidence either way'
            head = (f"[needle][compare] L={t['ctx_len']}: candidate {t['candidate_acc']:.3f} vs reference "
                    f"{t['reference_acc']:.3f}; McNemar fixed={m['n_fixed']} broke={m['n_broke']} "
                    f"p={m['p_one_sided'] if m['p_one_sided'] is None else round(m['p_one_sided'], 4)} "
                    f"(favors {verdict}); paired t delta={t['mean_delta']:+.4f}")
            if t['p_one_sided'] is None:
                print(f"{head} (n_seeds={t['n']}: one paired seed, no t interval)")
            else:
                print(f"{head} 95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
                      f"p={t['p_one_sided']:.4f} (n_seeds={t['n']})")
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
        'secret': args.secret,
        'trials': args.trials,
        'placement_seeds': list(seeds),
        'ctx_lens_requested': list(args.ctx_lens),
        'runs': results,
        'skipped': skipped,
        'ci_note': ('ci95_low/high are the cross-seed t interval when >= 2 placement seeds are run '
                    '(ci_basis="t_across_seed_means"); with one seed they are the Wilson interval '
                    '(ci_basis="wilson_within_single_placement_draw"), which covers trial noise within '
                    'that one placement draw only'),
    }
    if comparison is not None:
        payload['compare_to'] = args.compare_to
        payload['comparison'] = comparison
    Path(out).write_text(json.dumps(payload, indent=2))
    manifest = collect_run_metadata(seed=seeds[0])
    manifest['hyperparameters'] = vars(args)
    manifest['placement_seeds'] = list(seeds)
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
    save_run_manifest(Path(out).with_name('needle_manifest.json'), manifest)
    print(f'[done] saved to {out} ({len(results)} context length(s) evaluated, '
          f'{len(skipped)} skipped: {[s["ctx_len"] for s in skipped]})')
    if not results:
        print('[needle][error] no requested context length could be evaluated; nothing was measured',
              file=sys.stderr)
        raise SystemExit(2)
if __name__ == '__main__':
    main()
