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
  one-sided paired Student-t test over the shared seeds (>= 3 shared seeds
  required to certify a win). The t reference distribution is the right
  small-sample choice: over the default three seeds a two-sided 95% critical
  value is 4.30, not the normal 1.96. Each delta is also reported with a
  two-sided 95% confidence interval. A point-estimate win below `--margin`
  is reported but flagged as not significant; two shared seeds are tested for
  description only (df = 1), and with only one shared seed the same run's
  batches are tested for description only (batch noise is not run-to-run
  noise). Such arms are never certified and stay out of the Holm family.
  Because every variant arm is tested against the *same* baseline, the
  seed-replicated p-values are a family and are Holm-Bonferroni adjusted
  before an arm is called significantly better.
  The default selection set is three seeds: with fewer, the lr-grid means
  are too noisy for a small delta to mean anything. An lr cell whose diverged
  seeds were dropped (NaN/Inf val_loss) has a mean over a *subset* of the
  seeds, so it never wins selection against a cell with full seed coverage
  (`selection_pool = complete_coverage`); survivor-only cells are listed in
  `missing_seeds` and warned about instead.

Selection bias: picking the best learning rate on `--seeds` and then quoting
that same run overstates the effect (winner's curse). Pass `--confirm-seeds`
(split off before tuning) to score the selected learning rates on fresh seeds;
the comparison then uses the confirm runs, and the output records which split
the decision was made on. Without `--confirm-seeds` the p-values are still
reported for description, but no arm is certified as significantly better:
the rows carry `beats_reference_significantly = false` and a
`significance_basis` saying the comparison ran on selection seeds.

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
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust, mean_std_stderr,
                               paired_t_one_sided, student_t_ppf, welch_t_one_sided)
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds
from scripts.ablation import run_one


def _lr_key(lr: float) -> str:
    """Exact, filename-safe identity for a learning rate in the grid.

    ``format(lr, 'g')`` rounds to 6 significant digits: two distinct lrs can
    share a cache directory / grid key, and the selected rounded value no
    longer matches the trial records. ``repr`` round-trips the float exactly.
    """
    return repr(float(lr))


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


def run_trial(arm: str, lr: float, seed: int, args, base: Path, run_metadata: dict | None = None) -> dict:
    """One (arm, lr, seed) run, reusing the ablation runner verbatim."""
    trial_args = copy.copy(args)
    trial_args.lr = float(lr)
    trial_args.seed = int(seed)
    run_dir = base / 'trials' / arm / f'lr{_lr_key(lr)}' / f'seed{seed}'
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
        run_metadata=run_metadata,
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
        # The trial's final train/val regime: a trial still flat on both
        # curves is an underfit/stalled optimization cell whose val_loss
        # describes that regime, not the norm; selecting the lr grid must not
        # hide it.
        'curve_classification': summary.get('curve_classification'),
        'curve_generalization_gap': (summary.get('curve_diagnosis') or {}).get('generalization_gap'),
    }


def _usable_loss(value) -> bool:
    """Whether a trial's val_loss may enter the lr-grid selection.

    `isinstance(value, (int, float))` alone admits NaN/Inf: a diverged trial
    would then poison the seed mean (NaN) or win the `min` (Inf), so selection
    must happen on finite losses only.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def tune_arm(arm: str, args, base: Path, run_metadata: dict | None = None) -> dict:
    """Sweep the learning-rate grid for one arm over the selection seeds."""
    trials = []
    for lr in args.lrs:
        for seed in args.seeds:
            summary = run_trial(arm, lr, seed, args, base, run_metadata=run_metadata)
            trials.append(_trial_record(summary, lr, seed))
    n_selection_seeds = len(args.seeds) or 1
    lr_grid = {}
    for lr in args.lrs:
        cell = [t for t in trials if t['lr'] == lr]
        losses = [t['val_loss'] for t in cell if _usable_loss(t['val_loss'])]
        stats = mean_std_stderr(losses)
        if stats is not None:
            stats = dict(stats)
            # A cell whose diverged seeds were dropped has a mean over a
            # *subset* of the seeds; letting it win against a cell with full
            # coverage is survivorship bias (the unstable lr looks good
            # because its failures are not in its own mean).
            stats['missing_seeds'] = sorted(
                t['seed'] for t in cell if not _usable_loss(t['val_loss']))
            stats['complete_coverage'] = not stats['missing_seeds']
            stats['n_selection_seeds'] = n_selection_seeds
            stats['curve_classification_counts'] = dict(Counter(
                t['curve_classification'] for t in cell if t.get('curve_classification')))
            stats['curve_classifications_by_seed'] = {
                t['seed']: t.get('curve_classification') for t in cell}
            stats['lr'] = float(lr)
            lr_grid[_lr_key(lr)] = stats
    if not lr_grid:
        return {'arm': arm, 'trials': trials, 'lr_grid': {}, 'best': None}
    complete = {k: v for k, v in lr_grid.items() if v['complete_coverage']}
    pool, pool_name = (complete, 'complete_coverage') if complete else (lr_grid, 'partial_coverage')
    if complete and len(complete) < len(lr_grid):
        print(f"[tune][warn] {arm}: lr cell(s) "
              f"{sorted(set(lr_grid) - set(complete))} lack a usable val_loss for every "
              f"selection seed ({n_selection_seeds} required); they are excluded from lr "
              f"selection rather than winning on a survivor-only mean. Rerun with --force "
              f"or add seeds to fill them in.")
    best_lr_key = min(pool, key=lambda k: pool[k]['mean'])
    best = {**pool[best_lr_key], 'selection_pool': pool_name}
    return {'arm': arm, 'trials': trials, 'lr_grid': lr_grid, 'best': best}


def add_confirm_trials(arm_result: dict, args, base: Path, run_metadata: dict | None = None) -> dict:
    """Score the selected learning rate on the held-back confirm seeds."""
    if arm_result.get('best') is None or not args.confirm_seeds:
        return arm_result
    lr = arm_result['best']['lr']
    confirm = []
    for seed in args.confirm_seeds:
        summary = run_trial(arm_result['arm'], lr, seed, args, base, run_metadata=run_metadata)
        confirm.append(_trial_record(summary, lr, seed))
    arm_result['confirm_trials'] = confirm
    losses = [t['val_loss'] for t in confirm if _usable_loss(t['val_loss'])]
    missing = sorted(t['seed'] for t in confirm if not _usable_loss(t['val_loss']))
    arm_result['confirm'] = {'lr': lr, 'stats': mean_std_stderr(losses),
                             'missing_seeds': missing, 'complete_coverage': not missing}
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


def _describe_paired_effect(entry: dict, arm_vals, ref_vals, common, test,
                            alpha: float, power: float=0.8) -> None:
    """Add seed-direction and resolution without changing certification.

    Sign counts expose an effect carried by one or two seeds, and the MDE
    states the smallest loss difference this seed count could detect at 80%
    power. Both are descriptive: `beats_reference_significantly` still needs
    the confirm split and the Holm-adjusted p-value.
    """
    entry.update({
        'n_seeds_favoring': None,
        'n_seeds_tied': None,
        'n_seeds_against': None,
        'per_seed_delta': None,
        'paired_delta_std': None,
        'minimum_detectable_delta_95_power80': None,
    })
    if len(common) < 2 or len(arm_vals) != len(ref_vals):
        return
    deltas = [a - b for a, b in zip(arm_vals, ref_vals)]
    stats = mean_std_stderr(deltas)
    entry.update({
        'n_seeds_favoring': sum(value < 0.0 for value in deltas),
        'n_seeds_tied': sum(value == 0.0 for value in deltas),
        'n_seeds_against': sum(value > 0.0 for value in deltas),
        'per_seed_delta': {str(seed): value for seed, value in zip(common, deltas)},
        'paired_delta_std': None if stats is None else stats['std'],
    })
    if test is None or not test.get('df') or test.get('ci_low') is None:
        return
    half = (test['ci_high'] - test['ci_low']) / 2.0
    critical = student_t_ppf(1.0 - alpha / 2.0, float(test['df']))
    if critical > 0.0:
        stderr = half / critical
        entry['minimum_detectable_delta_95_power80'] = (
            student_t_ppf(1.0 - alpha, float(test['df']))
            + student_t_ppf(power, float(test['df']))
        ) * stderr


def compare(tuned: dict, baseline_arms, margin: float, alpha: float, certify: bool = True) -> dict | None:
    """Best variant trials vs the best tuned baseline, paired where possible.

    Only the paired test on >= 3 shared seeds certifies an arm as better:
    with two shared seeds the paired test is descriptive only (df = 1), with
    exactly one shared seed the same run's held-out batches are tested for
    description only (batch noise is not run-to-run noise), and with no shared
    seed no test runs at all. Only the seed-replicated rows enter the Holm
    family. `certify=False` (no held-back confirm seeds were provided)
    keeps the p-values for description but withholds every significance call:
    a lr selected on `--seeds` and then tested on those same runs is the
    winner's curse.
    """
    usable = {arm: r for arm, r in tuned.items() if r.get('best') is not None}
    baselines = {arm: r for arm, r in usable.items() if arm in set(baseline_arms)}
    if not baselines:
        return None
    # The confirm split is only a valid basis when *every* compared arm has
    # complete confirm statistics; a mixed basis would compare some arms on
    # fresh seeds and others on the seeds their lr was selected on. A
    # survivor-only confirm mean (a diverged seed silently dropped) would let
    # the arm that diverged look better, the same survivorship bias the lr
    # grid excludes via `complete_coverage`.
    confirm_ready = {a for a, r in usable.items()
                     if ((r.get('confirm') or {}).get('stats')
                         and not (r.get('confirm') or {}).get('missing_seeds'))}
    confirm_ran = {a for a, r in usable.items() if r.get('confirm_trials')}
    confirm_ignored = sorted(confirm_ran - confirm_ready)
    confirm_ignored_missing_seeds = {
        a: (usable[a].get('confirm') or {}).get('missing_seeds') for a in confirm_ignored}
    use_confirm = len(confirm_ready) == len(usable) and bool(usable)
    split = 'confirm_seeds' if use_confirm else 'selection_seeds'
    # Prefer a baseline that actually reached the confirm split. Choosing the
    # lowest selection mean regardless of confirm coverage can crown a
    # single-split baseline and silently make every certification impossible.
    confirm_baselines = [arm for arm in baselines if arm in confirm_ready]
    reference_pool = confirm_baselines if (certify and confirm_baselines) else list(baselines)
    ref_arm = min(reference_pool, key=lambda a: baselines[a]['best']['mean'])
    point_best_arm = min(baselines, key=lambda a: baselines[a]['best']['mean'])
    ref = baselines[ref_arm]
    # Only a comparison on held-back seeds can certify: without --confirm-seeds
    # the same runs selected the lr and then "proved" it, which is the
    # winner's curse regardless of how small the p-value looks.
    certified = bool(certify and use_confirm)
    ref_mean, ref_std, ref_n = _split_basis(ref, use_confirm)
    out = {
        'reference_arm': ref_arm,
        'reference_lr': ref['best']['lr'],
        'reference_val_loss': ref_mean,
        'reference_val_loss_std': ref_std,
        'reference_n_seeds': ref_n,
        'reference_selection_val_loss': ref['best']['mean'],
        'reference_curve_classification_counts': ref['best'].get('curve_classification_counts'),
        'reference_split': split,
        'reference_selection': {
            'rule': ('lowest selection mean among confirm-ready baselines'
                     if (certify and confirm_baselines)
                     else 'lowest selection mean (no confirm-ready baseline to prefer)'),
            'confirm_ready_baselines': sorted(confirm_baselines),
            'unconfirmed_baselines': sorted(set(baselines) - set(confirm_baselines)),
            'reference_can_certify': ref_arm in confirm_ready,
            'reference_is_point_estimate_best': ref_arm == point_best_arm,
        },
        'point_estimate_best_baseline': {
            'arm': point_best_arm,
            'lr': baselines[point_best_arm]['best']['lr'],
            'selection_val_loss': baselines[point_best_arm]['best']['mean'],
        },
        'certify_requested': bool(certify),
        'confirm_ignored_arms': confirm_ignored,
        'confirm_ignored_missing_seeds': confirm_ignored_missing_seeds,
        'certified': certified,
        'certification': (
            'confirm_seeds: significance tested on seeds held back from lr selection'
            if certified else
            ("none: --confirm-seeds was provided but unusable for "
             f"{confirm_ignored or 'the compared arms'} (incomplete or missing confirm "
             "statistics), so the comparison fell back to selection seeds and the p-values "
             "are descriptive only (winner's curse)"
             if certify else
             "none: no held-back --confirm-seeds, so the p-values are descriptive "
             "only and no arm is certified as significantly better (winner's curse)")),
        'margin': margin,
        'alpha': alpha,
        'rows': [],
    }
    def _split_trials(r):
        return (r.get('confirm_trials') or []) if use_confirm else r['trials']

    ref_seed_losses = {
        t['seed']: t['val_loss'] for t in _split_trials(ref)
        if t['lr'] == ref['best']['lr'] and _usable_loss(t['val_loss'])
    }
    for arm, r in usable.items():
        if arm in baselines:
            continue
        arm_seed_losses = {
            t['seed']: t['val_loss'] for t in _split_trials(r)
            if t['lr'] == r['best']['lr'] and _usable_loss(t['val_loss'])
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
            'curve_classification_counts': r['best'].get('curve_classification_counts'),
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
            'n_seeds_favoring': None,
            'n_seeds_tied': None,
            'n_seeds_against': None,
            'per_seed_delta': None,
            'paired_delta_std': None,
            'minimum_detectable_delta_95_power80': None,
        }
        test = None
        if len(common) >= 2:
            test = paired_t_one_sided(arm_vals, ref_vals, margin=margin)
            entry['test_basis'] = 'seed_means'
            entry['replicated_across_seeds'] = len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
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
        _describe_paired_effect(entry, arm_vals, ref_vals, common, test, alpha)
        if (entry['test'] is not None and not entry['replicated_across_seeds']
                and entry['significance_basis'] is None):
            entry['significance_basis'] = (
                f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                f'{MIN_SEEDS_FOR_SIGNIFICANCE}; the p-value is reported but the arm is not certified')
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
            certified and entry['beats_reference'] and p_holm is not None and p_holm < alpha
        )
        if not certified:
            if certify:
                entry['significance_basis'] = (
                    "descriptive only: --confirm-seeds was provided but unusable for "
                    f"{confirm_ignored or 'the compared arms'} (incomplete or missing confirm "
                    "statistics), so the comparison ran on the seeds the lr was selected on "
                    "(winner's curse); the p-value is reported but not certified")
            else:
                entry['significance_basis'] = (
                    "descriptive only: no --confirm-seeds, so the comparison ran on the seeds "
                    "the lr was selected on (winner's curse); the p-value is reported but not certified")
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


def _off_regime(regimes) -> dict:
    """Seeds of the tuned cell whose final train/val regime is not healthy."""
    return {k: v for k, v in (regimes or {}).items()
            if k not in ('healthy', 'insufficient_data')}


def print_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[tune][compare] no tuned baseline arm present; comparison skipped')
        return
    ref_std = comparison.get('reference_val_loss_std')
    ref_std_str = '' if ref_std is None else f" +/- {ref_std:.4f}"
    ref_note = ''
    ref_off = _off_regime(comparison.get('reference_curve_classification_counts'))
    if ref_off:
        ref_note = f" [curve regimes {ref_off}: the reference lr was selected in a non-healthy regime]"
    print(f"[tune][compare] reference: arm={comparison['reference_arm']} "
          f"lr={comparison['reference_lr']:g} "
          f"val_loss={comparison['reference_val_loss']:.4f}{ref_std_str} "
          f"n_seeds={comparison.get('reference_n_seeds')} "
          f"(split={comparison['reference_split']}; "
          f"selection-seed mean={comparison['reference_selection_val_loss']:.4f}){ref_note}")
    selection = comparison.get('reference_selection') or {}
    if selection:
        print(f"[tune][compare] reference selection: {selection.get('rule')}; "
              f"point-estimate best={comparison.get('point_estimate_best_baseline')}; "
              f"reference can certify={selection.get('reference_can_certify')}")
    if comparison.get('certified') is False:
        if comparison.get('certify_requested'):
            ignored = comparison.get('confirm_ignored_arms')
            missing = comparison.get('confirm_ignored_missing_seeds') or {}
            missing_str = (f' (missing seed(s) '
                           f'{ {a: s for a, s in missing.items() if s} })'
                           if any(missing.values()) else '')
            print("[tune][compare] NOT CERTIFIED: --confirm-seeds was provided but unusable"
                  f"{f' for {ignored}' if ignored else ''} (incomplete or missing confirm "
                  f"statistics){missing_str}, so the comparison fell back to selection seeds "
                  "(winner's curse); no arm is called significant")
        else:
            print("[tune][compare] NOT CERTIFIED: no --confirm-seeds, so these p-values compare runs "
                  "that also selected the lr (winner's curse); no arm is called significant")
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
        direction = ''
        if entry.get('n_seeds_favoring') is not None:
            direction = f" seeds_favoring={entry['n_seeds_favoring']}/{entry['n_paired_seeds']}"
            if entry.get('n_seeds_tied'):
                direction += f" tied={entry['n_seeds_tied']}"
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f" mde80={mde:.4f}"
        off = _off_regime(entry.get('curve_classification_counts'))
        if off:
            note += (f" [curve regimes {off}: the arm's best lr was picked in a non-healthy "
                     f"regime - read the trial curves before attributing the delta to the norm]")
        print(f"[tune][compare] {entry['arm']:>12s} lr={entry['best_lr']:g} "
              f"val_loss={entry['val_loss']:.4f}{std_str} "
              f"delta={delta_str}{ci_str} p(one-sided)={p_str} "
              f"p(holm)={p_holm_str} paired_seeds={entry['n_paired_seeds']}{direction} "
              f"test={entry.get('test') or '-'}@{basis}{mde_str} "
              f"beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}{note}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[tune][compare] NOTE: tests with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared "
              f"seeds are described but never certified: "
              f"{comparison['rows_tested_descriptively_only']}")
    if comparison.get('confirm_ignored_arms'):
        print(f"[tune][compare] confirm runs ignored (incomplete confirm coverage for "
              f"{comparison['confirm_ignored_arms']}, missing seeds "
              f"{comparison.get('confirm_ignored_missing_seeds')}); comparison is on selection seeds")
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
    p.add_argument('--curve-every', type=int, default=0, help='held-out curve-monitoring period in steps per trial (0 = auto: max_steps//5; negative = disabled); forwarded to ablation.run_one, so every trial records the train/val regime its val_loss was selected in')
    p.add_argument('--curve-batches', type=int, default=5, help='held-out batches per curve-monitoring point')
    p.add_argument('--margin', type=float, default=0.0, help='minimum val_loss improvement over the tuned baseline to count as a win')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the one-sided test')
    p.add_argument('--output-dir', default='runs/tune')
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='run the small-batch overfit sanity check once per swept norm before the grid (default: on; a broken gradient/loss path is caught before the grid runs)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the pre-flight overfit sanity check')
    p.add_argument('--force', action='store_true', help='rerun trials even if a summary already exists')
    args = p.parse_args()
    if not args.lrs:
        p.error('--lrs must not be empty')
    if any(not math.isfinite(lr) or lr <= 0.0 for lr in args.lrs):
        p.error(f'--lrs must be finite and positive, got {args.lrs}')
    if len(set(args.lrs)) != len(args.lrs):
        p.error(f'--lrs contains duplicates: {args.lrs}; a repeated lr reruns the identical '
                f'cells, so it inflates n and fakes replication (std would collapse toward 0)')
    if not args.seeds:
        p.error('--seeds must not be empty')
    if len(set(args.seeds)) != len(args.seeds):
        p.error(f'--seeds contains duplicates: {args.seeds}; a repeated seed reruns the identical '
                f'run, so it inflates n and fakes replication (std would collapse toward 0)')
    if len(set(args.confirm_seeds)) != len(args.confirm_seeds):
        p.error(f'--confirm-seeds contains duplicates: {args.confirm_seeds}; confirmation seeds must '
                f'be independent runs')
    overlap = set(args.seeds) & set(args.confirm_seeds)
    if overlap:
        p.error(f'--confirm-seeds must be disjoint from --seeds (selection would leak into confirmation): {sorted(overlap)}')
    if len(args.seeds) < 3:
        print(f'[tune][warn] {len(args.seeds)} selection seed(s): the lr grid means are noisy; '
              f'use >= 3 (and --confirm-seeds) before treating a small delta as real')
    if not args.confirm_seeds:
        print('[tune][warn] no --confirm-seeds: the comparison quotes the same runs the lr was '
              'selected on (winner\'s curse); treat the p-values as optimistic')
    if args.seeds:
        set_all_seeds(args.seeds[0])
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    sanity_report_path = None
    if args.sanity_check:
        # One preflight for the grid: `run_one` trains every arm through the
        # same model/loss plumbing, so a single per-norm overfit check catches
        # a broken gradient/loss path before the grid runs. The architecture
        # comes from the same table the ablation rows use, so the check
        # exercises this model shape.
        from scripts.sanity_overfit import run_preflight
        from scripts.train_gpt2 import _MODEL_SHAPES
        n_layers, dim, n_heads = _MODEL_SHAPES.get(args.model, _MODEL_SHAPES['gpt2-124m'])
        run_preflight(sorted(set(args.arms)) or ['dytcd'], args.seeds[0], base,
                      arch_args=['--dim', str(dim), '--n-layers', str(n_layers),
                                 '--n-heads', str(n_heads)])
        sanity_report_path = base / 'sanity_overfit.json'
    # One metadata snapshot for the whole grid: the snapshot does not change
    # while the grid runs.
    run_metadata = collect_run_metadata(seed=None)
    tuned = {}
    for arm in args.arms:
        print(f'\n[tune] arm={arm} lrs={args.lrs} seeds={args.seeds}')
        tuned[arm] = tune_arm(arm, args, base, run_metadata=run_metadata)
        if tuned[arm]['best'] is not None:
            b = tuned[arm]['best']
            print(f"[tune] {arm}: best lr={b['lr']:g} val_loss={b['mean']:.4f} "
                  f"(stderr={b['stderr']:.4f} over {b['n']} runs)")
            regimes = b.get('curve_classification_counts') or {}
            if regimes:
                print(f"[tune] {arm}: best-lr curve regimes {regimes}")
            else:
                print(f"[tune] {arm}: best-lr trials have no curve diagnosis (cached "
                      f"pre-monitoring summaries); rerun with --force to record the regimes")
            off = _off_regime(regimes)
            if off:
                print(f"[tune][warn] {arm}: lr={b['lr']:g} was selected while "
                      f"{sum(off.values())} trial(s) were in {off}: the selected val_loss "
                      f"describes that regime (underfit/stalled/overfit/diverging), not "
                      f"just the norm; check the trial curves and the lr grid before "
                      f"quoting this arm's win")
        tuned[arm] = add_confirm_trials(tuned[arm], args, base, run_metadata=run_metadata)
        if tuned[arm].get('confirm'):
            c = tuned[arm]['confirm']
            stats = c['stats']
            if stats:
                print(f"[tune] {arm}: confirm on seeds={args.confirm_seeds} lr={c['lr']:g} "
                      f"val_loss={stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})")
            else:
                print(f"[tune] {arm}: confirm produced no usable losses")
    comparison = compare(tuned, args.baselines, args.margin, args.alpha,
                         certify=bool(args.confirm_seeds))
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
            'curve_monitoring': {
                'every': args.curve_every, 'batches': args.curve_batches,
                'note': ('per-trial train/val regime recorded by ablation.run_one; curve '
                         'batches come from a generator the lr selection never uses, so the '
                         'monitoring does not bias the grid'),
            },
            'test_split': 'untouched: all losses come from the synthetic held-out generator',
        },
        'arms': tuned,
        'comparison': comparison,
        'run_metadata': run_metadata,
    }
    with open(base / 'tuning_summary.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    manifest = dict(run_metadata)
    manifest['seed'] = None
    manifest['hyperparameters'] = vars(args)
    manifest['arms'] = list(args.arms)
    manifest['seeds'] = list(args.seeds)
    manifest['confirm_seeds'] = list(args.confirm_seeds)
    manifest['certified'] = bool(comparison and comparison.get('certified'))
    manifest['reference_arm'] = comparison.get('reference_arm') if comparison else None
    # Compact per-arm outcome next to the full summary file: which lr was
    # selected and what it scored, so a completed grid is readable from the
    # manifest without parsing every trial record.
    manifest['selected_lr_by_arm'] = {
        arm: ({'lr': r['best']['lr'], 'val_loss': r['best']['mean'],
               'stderr': r['best']['stderr'], 'n': r['best']['n']}
              if r.get('best') else None)
        for arm, r in tuned.items()
    }
    manifest = finish_metadata(
        manifest,
        artifacts={
            'tuning_summary': file_fingerprint(base / 'tuning_summary.json'),
            'sanity_report': (file_fingerprint(sanity_report_path) if sanity_report_path
                              else {'enabled': False}),
        },
    )
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[tune][done] saved to {base / "tuning_summary.json"}')


if __name__ == '__main__':
    main()
