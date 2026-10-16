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
  one-sided paired z-test (same seeds -> paired samples, same held-out
  batches within a seed). A point-estimate win below `--margin` is reported
  but flagged as not significant.

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

from adasquash.metrics import mean_std_stderr, paired_z_one_sided, welch_z_one_sided
from adasquash.repro import collect_run_metadata, git_revision, save_run_manifest
from scripts.ablation import run_one


def run_trial(arm: str, lr: float, seed: int, args, base: Path) -> dict:
    """One (arm, lr, seed) run, reusing the ablation runner verbatim."""
    trial_args = copy.copy(args)
    trial_args.lr = float(lr)
    trial_args.seed = int(seed)
    run_dir = base / 'trials' / arm / f'lr{lr:g}' / f'seed{seed}'
    if (run_dir / 'summary.json').exists() and (not args.force):
        print(f'[tune][skip] {arm} lr={lr:g} seed={seed} (summary exists, --force to rerun)')
        try:
            return json.loads((run_dir / 'summary.json').read_text())
        except (OSError, json.JSONDecodeError):
            pass
    summary = run_one(f'{arm}|lr={lr:g}|seed={seed}', {'norm_name': arm},
                      trial_args, run_dir)
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


def compare(tuned: dict, baseline_arms, margin: float, alpha: float) -> dict | None:
    """Best variant trials vs the best tuned baseline, paired where possible."""
    usable = {arm: r for arm, r in tuned.items() if r.get('best') is not None}
    baselines = {arm: r for arm, r in usable.items() if arm in set(baseline_arms)}
    if not baselines:
        return None
    ref_arm = min(baselines, key=lambda a: baselines[a]['best']['mean'])
    ref = baselines[ref_arm]
    use_confirm = bool(ref.get('confirm_trials')) and all(
        r.get('confirm_trials') for r in usable.values()
    )
    split = 'confirm_seeds' if use_confirm else 'selection_seeds'
    out = {
        'reference_arm': ref_arm,
        'reference_lr': ref['best']['lr'],
        'reference_val_loss': ref['best']['mean'],
        'reference_split': split,
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
        entry = {
            'arm': arm,
            'best_lr': r['best']['lr'],
            'val_loss': r['best']['mean'],
            'delta_vs_reference': delta,
            'n_paired_seeds': len(common),
            'beats_reference': bool(delta is not None and delta < -margin),
            'beats_reference_significantly': False,
            'test': None,
            'p_one_sided': None,
        }
        if len(common) >= 2:
            test = paired_z_one_sided(arm_vals, ref_vals, margin=margin)
        else:
            # One shared seed (or none): fall back to the shared held-out
            # batches of the single paired run, then to the unpaired test.
            a_trial = _trial_by_seed(_split_trials(r), r['best']['lr']).get(common[0]) if common else None
            b_trial = _trial_by_seed(_split_trials(ref), ref['best']['lr']).get(common[0]) if common else None
            a_batches = a_trial.get('val_losses') if a_trial else None
            b_batches = b_trial.get('val_losses') if b_trial else None
            if (isinstance(a_batches, list) and isinstance(b_batches, list)
                    and len(a_batches) == len(b_batches) and len(a_batches) >= 2):
                test = paired_z_one_sided(a_batches, b_batches, margin=margin)
            else:
                test = welch_z_one_sided(arm_vals, ref_vals, margin=margin)
        if test is not None:
            entry['test'] = 'paired_z_one_sided' if test['paired'] else 'welch_z_one_sided'
            entry['p_one_sided'] = test['p_one_sided']
            entry['z'] = test['z']
            entry['n'] = test['n']
            entry['beats_reference_significantly'] = bool(
                entry['beats_reference'] and test['p_one_sided'] is not None
                and test['p_one_sided'] < alpha
            )
        out['rows'].append(entry)
    return out


def print_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[tune][compare] no tuned baseline arm present; comparison skipped')
        return
    print(f"[tune][compare] reference: arm={comparison['reference_arm']} "
          f"lr={comparison['reference_lr']:g} val_loss={comparison['reference_val_loss']:.4f} "
          f"(decided on {comparison['reference_split']})")
    for entry in comparison['rows']:
        p = entry['p_one_sided']
        p_str = 'n/a' if p is None else f'{p:.3g}'
        print(f"[tune][compare] {entry['arm']:>12s} lr={entry['best_lr']:g} "
              f"delta={entry['delta_vs_reference']:+.4f} p(one-sided)={p_str} "
              f"beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=['gpt2-124m', 'gpt2-350m', 'llama-1b'], default='gpt2-124m')
    p.add_argument('--arms', nargs='+', required=True, help='arm names (norm names) to tune, lr-swept with equal compute')
    p.add_argument('--baselines', nargs='+', default=['ln', 'rmsnorm'], help='arms that are the simple references')
    p.add_argument('--lrs', nargs='+', type=float, default=[3e-4, 6e-4, 1.2e-3])
    p.add_argument('--seeds', nargs='+', type=int, default=[42], help='selection seeds for the lr grid')
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
            print(f"[tune] {arm}: confirm on seeds={args.confirm_seeds} "
                  f"val_loss={stats['mean']:.4f}" if stats else
                  f"[tune] {arm}: confirm produced no usable losses")
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
