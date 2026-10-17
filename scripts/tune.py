#!/usr/bin/env python3
"""Compute-matched hyper-parameter tuning: never compare against an untuned baseline.

Scientific protocol (why this script exists)
--------------------------------------------
"The new norm beats LayerNorm/RMSNorm" is only meaningful if the baseline has
been tuned with the *same compute and the same selection protocol* as the
proposal. A single fixed recipe can make the simplest baseline look weak
enough that any added machinery appears to win - the strawman failure mode.

This driver therefore:

* gives every arm (baseline or variant) the identical compute - same
  `--max-steps`, batch size, sequence length, corpus generator and seed set -
  while sweeping the learning rate, the hyper-parameter that dominates
  short-run optimization;
* selects the learning rate per arm on a *held-out* metric only: the mean
  `val_loss` over fresh synthetic batches from a generator the optimizer
  never sees (`ablation.run_one` computes it); the sealed real-data test
  split is never touched;
* runs each trial through `ablation.run_one`, so the tuning protocol and the
  ablation protocol cannot silently drift apart;
* compares each variant's best trial against the best tuned baseline with a
  one-sided paired Student-t test over the shared seeds (>= 2 shared seeds
  required to certify a win). The t reference distribution is the right
  small-sample choice: over the default three seeds a two-sided 95% critical
  value is 4.30, not the normal 1.96. Each delta is also reported with a
  two-sided 95% confidence interval. A point-estimate win below `--margin`
  is reported but flagged as not significant, and with only one shared seed
  the same run's batches are tested for description only (batch noise is not
  run-to-run noise): such an arm is never certified and stays out of the Holm
  family. Because every variant arm is tested against the *same* baseline,
  the seed-replicated p-values are a family and are Holm-Bonferroni adjusted
  before an arm is called significantly better.
  The default selection set is three seeds: with fewer, the lr-grid means
  are too noisy for a small delta to mean anything.

Selection bias: picking the best learning rate on `--seeds` and then quoting
that same run overstates the effect (winner's curse). Pass `--confirm-seeds`
(split off before tuning) to score the selected learning rates on fresh seeds;
the comparison then uses the confirm runs, and the output records which split
the decision was made on.

Usage
-----
    python3 scripts/tune.py --arms ln rmsnorm dyt dytcd \
        --lrs 3e-4 6e-4 1.2e-3 --seeds 42 43 --confirm-seeds 44 45 \
        --output-dir runs/tune
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import holm_adjust, mean_std_stderr, paired_t_one_sided, welch_t_one_sided
from adasquash.repro import collect_run_metadata, git_revision, save_run_manifest
from scripts.ablation import run_one


def cached_trial_matches(summary: dict, arm: str, lr: float, args) -> bool:
    """Whether a cached trial ran under the *same protocol* as this invocation.

    The trial path (`arm/lr/seed`) does not encode `--max-steps`,
    `--batch-size`, `--seq-len` or `--weight-decay`, so reusing a summary
    without checking those would silently compare and select under two
    different recipes.
    """
    if not isinstance(summary, dict):
        return False
    controls = summary.get('controls')
    if not isinstance(controls, dict):
        return False
    optimizer = controls.get('optimizer') or {}
    data = controls.get('data') or {}
    cfg = summary.get('cfg')
    return bool(
        isinstance(cfg, dict) and cfg.get('norm_name') == arm
        and summary.get('model') == args.model
        and optimizer.get('lr') == float(lr)
        and optimizer.get('weight_decay') == args.weight_decay
        and data.get('batch_size') == args.batch_size
        and data.get('seq_len') == args.seq_len
        and controls.get('train_steps') == args.max_steps
        and summary.get('n_val_batches') == max(1, args.val_batches)
    )


def run_trial(arm: str, lr: float, seed: int, args, base: Path) -> dict:
    """One (arm, lr, seed) run, reusing the ablation runner verbatim."""
    trial_args = copy.copy(args)
    trial_args.lr = float(lr)
    trial_args.seed = int(seed)
    run_dir = base / 'trials' / arm / f'lr{lr:g}' / f'seed{seed}'
    if (run_dir / 'summary.json').exists() and (not args.force):
        try:
            cached = json.loads((run_dir / 'summary.json').read_text())
            if cached_trial_matches(cached, arm, lr, args):
                print(f'[tune][skip] {arm} lr={lr:g} seed={seed} (summary exists, --force to rerun)')
                return cached
            print(f'[tune][stale] {arm} lr={lr:g} seed={seed}: cached trial ran under a different '
                  f'protocol (max_steps/batch/seq/weight_decay/model); retraining and overwriting')
        except (OSError, json.JSONDecodeError):
            pass
    summary = run_one(
        f'{arm}|lr={lr:g}|seed={seed}', {'norm_name': arm}, trial_args, run_dir, seed=seed,
        hypothesis=(f'Compute-matched lr-sweep cell for arm {arm}: baselines and variants get the '
                    f'identical grid and compute, so the selected lr makes the later arm '
                    f'comparison a test against a tuned reference, not against a strawman.'),
    )
    return summary


def _trial_record(summary: dict, lr: float, seed: int) -> dict:
    return {
        'lr': float(lr),
        'seed': int(seed),
        'val_loss': summary.get('val_loss'),
        'val_loss_std': summary.get('val_loss_std'),
        'val_loss_stderr': summary.get('val_loss_stderr'),
        'val_losses': summary.get('val_losses'),
        'final_loss': summary.get('final_loss'),
    }


def tune_arm(arm: str, args, base: Path) -> dict:
    """Sweep the learning-rate grid for one arm over the selection seeds."""
    trials = []
    for lr in args.lrs:
        for seed in args.seeds:
            summary = run_trial(arm, lr, seed, args, base)
            trials.append(_trial_record(summary, lr, seed))
    lr_grid = {}
    for lr in args.lrs:
        losses = [t['val_loss'] for t in trials
                  if t['lr'] == lr and isinstance(t['val_loss'], (int, float))]
        stats = mean_std_stderr(losses)
        if stats is not None:
            lr_grid[f'{lr:g}'] = stats
    if not lr_grid:
        return {'arm': arm, 'trials': trials, 'lr_grid': {}, 'best': None}
    best_lr_str = min(lr_grid, key=lambda k: lr_grid[k]['mean'])
    best = {'lr': float(best_lr_str), **lr_grid[best_lr_str]}
    return {'arm': arm, 'trials': trials, 'lr_grid': lr_grid, 'best': best}


def add_confirm_trials(arm_result: dict, args, base: Path) -> dict:
    """Score the selected learning rate on the held-back confirm seeds."""
    if arm_result.get('best') is None or not args.confirm_seeds:
        return arm_result
    lr = arm_result['best']['lr']
    confirm = []
    for seed in args.confirm_seeds:
        summary = run_trial(arm_result['arm'], lr, seed, args, base)
        confirm.append(_trial_record(summary, lr, seed))
    arm_result['confirm_trials'] = confirm
    losses = [t['val_loss'] for t in confirm if isinstance(t['val_loss'], (int, float))]
    arm_result['confirm'] = {'lr': lr, 'stats': mean_std_stderr(losses)}
    return arm_result


def _trial_by_seed(trials, lr):
    return {t['seed']: t for t in trials if t['lr'] == lr}


def _split_basis(r: dict, use_confirm: bool):
    """(mean, std, n) of an arm's selected lr on the comparison split.

    The delta and p-value are already computed on the confirm split when it is
    available; the reported absolute losses must come from the same split,
    otherwise the printed val_loss and the printed delta disagree.
    """
    if use_confirm:
        stats = (r.get('confirm') or {}).get('stats')
        if stats:
            return stats.get('mean'), stats.get('std'), stats.get('n')
    return r['best']['mean'], r['best'].get('std'), r['best'].get('n')


def compare(tuned: dict, baseline_arms, margin: float, alpha: float) -> dict | None:
    """Best variant trials vs the best tuned baseline, paired where possible.

    Only the paired test on >= 2 shared seeds certifies an arm as better:
    with exactly one shared seed the same run's held-out batches are tested
    for description only (batch noise is not run-to-run noise), and with no
    shared seed no test runs at all. Only the seed-replicated rows enter the
    Holm family.
    """
    usable = {arm: r for arm, r in tuned.items() if r.get('best') is not None}
    baselines = {arm: r for arm, r in usable.items() if arm in set(baseline_arms)}
    if not baselines:
        return None
    ref_arm = min(baselines, key=lambda a: baselines[a]['best']['mean'])
    ref = baselines[ref_arm]
    # The confirm split is only a valid basis when *every* compared arm has
    # confirm statistics; a mixed basis would compare some arms on fresh seeds
    # and others on the seeds their lr was selected on.
    confirm_ready = {a for a, r in usable.items() if (r.get('confirm') or {}).get('stats')}
    confirm_ran = {a for a, r in usable.items() if r.get('confirm_trials')}
    use_confirm = len(confirm_ready) == len(usable) and bool(usable)
    split = 'confirm_seeds' if use_confirm else 'selection_seeds'
    ref_mean, ref_std, ref_n = _split_basis(ref, use_confirm)
    out = {
        'reference_arm': ref_arm,
        'reference_lr': ref['best']['lr'],
        'reference_val_loss': ref_mean,
        'reference_val_loss_std': ref_std,
        'reference_n_seeds': ref_n,
        'reference_selection_val_loss': ref['best']['mean'],
        'reference_split': split,
        'confirm_ignored_arms': sorted(confirm_ran - confirm_ready),
        'margin': margin,
        'alpha': alpha,
        'rows': [],
    }
    def _split_trials(r):
        return (r.get('confirm_trials') or []) if use_confirm else r['trials']

    ref_seed_losses = {
        t['seed']: t['val_loss'] for t in _split_trials(ref)
        if t['lr'] == ref['best']['lr'] and isinstance(t['val_loss'], (int, float))
    }
    for arm, r in usable.items():
        if arm in baselines:
            continue
        arm_seed_losses = {
            t['seed']: t['val_loss'] for t in _split_trials(r)
            if t['lr'] == r['best']['lr'] and isinstance(t['val_loss'], (int, float))
        }
        common = sorted(set(arm_seed_losses) & set(ref_seed_losses))
        arm_vals = [arm_seed_losses[s] for s in common]
        ref_vals = [ref_seed_losses[s] for s in common]
        delta = (mean_std_stderr(arm_vals)['mean'] - mean_std_stderr(ref_vals)['mean']
                 if arm_vals else None)
        arm_mean, arm_std, arm_n = _split_basis(r, use_confirm)
        entry = {
            'arm': arm,
            'best_lr': r['best']['lr'],
            'val_loss': arm_mean,
            'val_loss_std': arm_std,
            'n_seeds': arm_n,
            'selection_val_loss': r['best']['mean'],
            'delta_vs_reference': delta,
            'n_paired_seeds': len(common),
            'beats_reference': bool(delta is not None and delta < -margin),
            'beats_reference_significantly': False,
            'test': None,
            'test_basis': None,
            'replicated_across_seeds': False,
            'significance_basis': None,
            'p_one_sided': None,
            'p_one_sided_holm': None,
            't': None,
            'df': None,
            'delta_ci_95': None,
        }
        test = None
        if len(common) >= 2:
            test = paired_t_one_sided(arm_vals, ref_vals, margin=margin)
            entry['test_basis'] = 'seed_means'
            entry['replicated_across_seeds'] = True
        elif len(common) == 1:
            # One shared seed: test that run's held-out batches for
            # *description only*. Batch-level noise is not run-to-run noise,
            # so a single run can never certify that an arm generalizes better
            # (and it must not enter the Holm family).
            seed = common[0]
            a_trial = _trial_by_seed(_split_trials(r), r['best']['lr']).get(seed)
            b_trial = _trial_by_seed(_split_trials(ref), ref['best']['lr']).get(seed)
            a_batches = a_trial.get('val_losses') if a_trial else None
            b_batches = b_trial.get('val_losses') if b_trial else None
            test = None
            entry['test_basis'] = f'val_batches_single_seed{seed}'
            if (isinstance(a_batches, list) and isinstance(b_batches, list)
                    and len(a_batches) >= 2 and len(b_batches) >= 2):
                if len(a_batches) == len(b_batches):
                    test = paired_t_one_sided(a_batches, b_batches, margin=margin)
                else:
                    test = welch_t_one_sided(a_batches, b_batches, margin=margin)
                    entry['test_basis'] = f'val_batches_single_seed{seed}_unpaired'
            entry['significance_basis'] = (
                'descriptive only: one shared seed, so the batch-level test cannot separate '
                'the arm effect from run-to-run noise; not certified and not counted in the '
                'Holm family')
        else:
            entry['test_basis'] = 'no_shared_seeds' if (arm_seed_losses and ref_seed_losses) else 'insufficient_seeds'
        if test is not None:
            entry['test'] = test['test']
            entry['p_one_sided'] = test['p_one_sided']
            entry['t'] = test['t']
            entry['df'] = test['df']
            entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
            entry['n'] = test['n']
        out['rows'].append(entry)
    # Every seed-replicated arm is tested against the same reference, so those
    # raw p-values are a family of tests; without the Holm step-down the chance
    # of some arm looking significant grows with the number of arms. Single-run
    # batch tests stay out of the family (and can never be called significant).
    family = [e for e in out['rows'] if e['replicated_across_seeds']]
    adjusted = holm_adjust([e['p_one_sided'] for e in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['beats_reference_significantly'] = bool(
            entry['beats_reference'] and p_holm is not None and p_holm < alpha
        )
    out['rows_tested_descriptively_only'] = [e['arm'] for e in out['rows']
                                             if e['test'] is not None and not e['replicated_across_seeds']]
    out['multiple_comparison'] = (
        f'Holm-Bonferroni over the {sum(p is not None for p in adjusted)} seed-replicated arm(s) '
        f'against the same reference; `beats_reference_significantly` uses the adjusted p-value. '
        f'Single-run batch-level tests are descriptive only '
        f'({out["rows_tested_descriptively_only"]}).'
    )
    out['rows_with_fewer_than_3_seeds'] = sorted(
        e['arm'] for e in out['rows'] if e['n_paired_seeds'] < 3)
    return out


def print_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[tune][compare] no tuned baseline arm present; comparison skipped')
        return
    ref_std = comparison.get('reference_val_loss_std')
    ref_std_str = '' if ref_std is None else f" +/- {ref_std:.4f}"
    print(f"[tune][compare] reference: arm={comparison['reference_arm']} "
          f"lr={comparison['reference_lr']:g} "
          f"val_loss={comparison['reference_val_loss']:.4f}{ref_std_str} "
          f"n_seeds={comparison.get('reference_n_seeds')} "
          f"(split={comparison['reference_split']}; "
          f"selection-seed mean={comparison['reference_selection_val_loss']:.4f})")
    for entry in comparison['rows']:
        p = entry['p_one_sided']
        p_holm = entry.get('p_one_sided_holm')
        p_str = 'n/a' if p is None else f'{p:.3g}'
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        std = entry.get('val_loss_std')
        std_str = '' if std is None else f" +/- {std:.4f}"
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        delta = entry.get('delta_vs_reference')
        delta_str = 'n/a' if delta is None else f'{delta:+.4f}'
        basis = entry.get('test_basis') or '-'
        note = ' [descriptive only, not certified]' if entry.get('significance_basis') else ''
        print(f"[tune][compare] {entry['arm']:>12s} lr={entry['best_lr']:g} "
              f"val_loss={entry['val_loss']:.4f}{std_str} "
              f"delta={delta_str}{ci_str} p(one-sided)={p_str} "
              f"p(holm)={p_holm_str} paired_seeds={entry['n_paired_seeds']} "
              f"test={entry.get('test') or '-'}@{basis} "
              f"beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}{note}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[tune][compare] NOTE: single-run batch-level tests (fewer than 2 shared seeds) "
              f"are described but never certified: {comparison['rows_tested_descriptively_only']}")
    if comparison.get('confirm_ignored_arms'):
        print(f"[tune][compare] confirm runs ignored (missing confirm stats for "
              f"{comparison['confirm_ignored_arms']}); comparison is on selection seeds")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[tune][compare] WARNING: <3 paired seeds for "
              f"{comparison['rows_with_fewer_than_3_seeds']}: "
              f"effect and noise cannot be separated reliably")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['gpt2-124m', 'gpt2-350m', 'llama-1b'], default='gpt2-124m')
    p.add_argument('--arms', nargs='+', required=True, help='arm names (norm names) to tune, lr-swept with equal compute')
    p.add_argument('--baselines', nargs='+', default=['ln', 'rmsnorm'], help='arms that are the simple references')
    p.add_argument('--lrs', nargs='+', type=float, default=[3e-4, 6e-4, 1.2e-3])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44], help='selection seeds for the lr grid; the default is 3 so each lr-grid mean rests on independent replications rather than a single lucky run')
    p.add_argument('--confirm-seeds', nargs='+', type=int, default=[], help='held-back seeds used only to score the selected lr (avoids winner\'s-curse comparisons)')
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--stats-every', type=int, default=10 ** 9)
    p.add_argument('--val-batches', type=int, default=20, help='held-out synthetic batches per trial (selection metric)')
    p.add_argument('--margin', type=float, default=0.0, help='minimum val_loss improvement over the tuned baseline to count as a win')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the one-sided test')
    p.add_argument('--output-dir', default='runs/tune')
    p.add_argument('--force', action='store_true', help='rerun trials even if a summary already exists')
    args = p.parse_args()
    if not args.lrs:
        p.error('--lrs must not be empty')
    if not args.seeds:
        p.error('--seeds must not be empty')
    overlap = set(args.seeds) & set(args.confirm_seeds)
    if overlap:
        p.error(f'--confirm-seeds must be disjoint from --seeds (selection would leak into confirmation): {sorted(overlap)}')
    if len(args.seeds) < 3:
        print(f'[tune][warn] {len(args.seeds)} selection seed(s): the lr grid means are noisy; '
              f'use >= 3 (and --confirm-seeds) before treating a small delta as real')
    if not args.confirm_seeds:
        print('[tune][warn] no --confirm-seeds: the comparison quotes the same runs the lr was '
              'selected on (winner\'s curse); treat the p-values as optimistic')
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    tuned = {}
    for arm in args.arms:
        print(f'\n[tune] arm={arm} lrs={args.lrs} seeds={args.seeds}')
        tuned[arm] = tune_arm(arm, args, base)
        if tuned[arm]['best'] is not None:
            b = tuned[arm]['best']
            print(f"[tune] {arm}: best lr={b['lr']:g} val_loss={b['mean']:.4f} "
                  f"(stderr={b['stderr']:.4f} over {b['n']} runs)")
        tuned[arm] = add_confirm_trials(tuned[arm], args, base)
        if tuned[arm].get('confirm'):
            c = tuned[arm]['confirm']
            stats = c['stats']
            if stats:
                print(f"[tune] {arm}: confirm on seeds={args.confirm_seeds} lr={c['lr']:g} "
                      f"val_loss={stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})")
            else:
                print(f"[tune] {arm}: confirm produced no usable losses")
    comparison = compare(tuned, args.baselines, args.margin, args.alpha)
    print_comparison(comparison)
    out = {
        'protocol': {
            'model': args.model,
            'arms': list(args.arms),
            'baselines': list(args.baselines),
            'lrs': list(args.lrs),
            'selection_seeds': list(args.seeds),
            'confirm_seeds': list(args.confirm_seeds),
            'max_steps': args.max_steps,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'weight_decay': args.weight_decay,
            'val_batches': args.val_batches,
            'selection_metric': 'held-out synthetic val_loss (mean over val batches)',
            'compute_matched': 'every arm/seed/lr trial runs the identical max_steps x batch_size x seq_len',
            'test_split': 'untouched: all losses come from the synthetic held-out generator',
        },
        'arms': tuned,
        'comparison': comparison,
        'run_metadata': collect_run_metadata(),
    }
    with open(base / 'tuning_summary.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    manifest = collect_run_metadata(seed=None)
    manifest['git'] = git_revision()
    manifest['hyperparameters'] = vars(args)
    manifest['arms'] = list(args.arms)
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[tune][done] saved to {base / "tuning_summary.json"}')


if __name__ == '__main__':
    main()
