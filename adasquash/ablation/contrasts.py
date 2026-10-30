from __future__ import annotations
import dataclasses
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
import torch
from adasquash import build_gpt_124m, count_parameters, ActivationStats, LossSpikeCounter, markov_batch
from adasquash.attention import clear_logit_scratch_cache
from adasquash.design import (classify_run_exclusions,
                               design_readiness_audit,
                               exclusion_policy_fingerprint, normalize_exclusion_policy,
                               order_drift_report,
                               randomized_complete_block_schedule,
                               run_exclusion_audit,
                               schedule_balance_report, simulate_paired_design)
from adasquash.diagnostics import diagnose_train_val_curve, triage_nonfinite
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust,
                               leave_one_out_jackknife, mean_prediction_interval,
                               mean_std_stderr, max_p_combination,
                               paired_design_summary, paired_randomization_test,
                               paired_t_one_sided, ratio_effect_report,
                               student_t_ppf, tost_paired, two_sided_p_from_t,
                               welch_t_one_sided)
from adasquash.model import build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache, clear_cast_weight_cache
from adasquash.repro import artifact_record, collect_run_metadata, file_fingerprint, finish_metadata, git_revision, save_run_manifest, set_all_seeds
from scripts.train_gpt2 import (_explicit_arg_names, apply_config_to_parser,
                                build_optimizer, load_yaml_config)

from .common import (_basis_counted_in_family, _finite_metric,
                     _paired_effect_summary, _paired_row_test, _per_seed_deltas,
                     _robustness_for_deltas, _robustness_note, _row_n_seeds,
                     _row_std, _run_identity, _safe_exp, _seed_means)
from .matrix import (FACTORIAL_SPEC, LOO_SPEC, QK_AXIS_SPEC,
                     THREE_WAY_FACTORIAL_SPEC, _parse_three_way_cell_key)

def leave_one_out_comparison(summaries: list, spec: dict | None=None, margin: float=0.0,
                             alpha: float=0.05,
                             equivalence_margin: float=0.0) -> dict | None:
    """Full vs each Full-minus-X row: does removing mechanism X hurt?

    `check_leave_one_out` proves the *structure* of the leave-one-out matrix -
    the reference row differs from Full in exactly the declared knobs. This
    function adds the statistical half: for every declared mechanism it runs
    the same one-sided paired test as `baseline_comparison`, of H1 "Full has
    lower held-out val_loss than the X-free row". Because all mechanisms are
    tested against the same Full row, the per-mechanism p-values are
    Holm-Bonferroni adjusted before a mechanism is called significantly
    helpful; a point-estimate win inside the seed noise remains visible as
    `mechanism_helps=True, mechanism_helps_significantly=False`. This is what
    backs the claim that each axis of the main proposal is load-bearing on
    its own, not merely correlated with the others. The percentage PPL cost
    of a removal also carries a paired interval (`ppl_delta_pct_ci_95`):
    PPL = exp(loss), so exponentiating the paired loss interval gives the
    uncertainty for the ratio of geometric-mean PPLs. Returns None when no
    declared contrast has both rows with a usable val_loss.
    """
    spec = LOO_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    parent_of = {r['name']: r['reused_from'] for r in summaries
                 if isinstance(r, dict) and 'name' in r and isinstance(r.get('reused_from'), str)}
    rows: list = []
    for full_name, removals in spec.items():
        full = usable.get(full_name)
        for mechanism, info in removals.items():
            ref_name = info['row']
            ref = usable.get(ref_name)
            entry = {
                'full': full_name,
                'mechanism': mechanism,
                'reference_row': ref_name,
                'declared_diff': sorted(info['diff']),
                'full_val_loss': None if full is None else full['val_loss'],
                'full_val_loss_std': None if full is None else _row_std(full),
                'reference_val_loss': None if ref is None else ref['val_loss'],
                'reference_val_loss_std': None if ref is None else _row_std(ref),
                'n_seeds_full': None if full is None else _row_n_seeds(full),
                'n_seeds_reference': None if ref is None else _row_n_seeds(ref),
                'full_curve_classification_counts': None if full is None else full.get('curve_classification_counts'),
                'reference_curve_classification_counts': None if ref is None else ref.get('curve_classification_counts'),
                'run_identity_full': None if full is None else _run_identity(full, parent_of),
                'run_identity_reference': None if ref is None else _run_identity(ref, parent_of),
                'delta_full_minus_reference': None,
                # Shared-seed effect of the paired test (the certifying
                # statistic); may differ from the all-seed mean difference.
                'delta_paired_mean': None,
                'delta_over_max_seed_std': None,
                'within_seed_noise': None,
                'ppl_full': None,
                'ppl_reference': None,
                'ppl_delta_pct': None,
                'ppl_ratio_paired': None,
                'ppl_ratio_ci_95': None,
                'ppl_delta_pct_paired': None,
                'ppl_delta_pct_ci_95': None,
                'ppl_interval_basis': None,
                'mechanism_helps': False,
                'mechanism_helps_significantly': False,
                'test': None,
                'test_basis': None,
                'paired_seeds': [],
                'replicated_across_seeds': False,
                'significance_basis': None,
                'excluded_from_family': None,
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
                'seed_jackknife': None,
                'equivalence': None,
            }
            if full is None:
                entry['skipped'] = f"full row '{full_name}' has no usable val_loss"
                rows.append(entry)
                continue
            if ref is None:
                entry['skipped'] = f"Full-minus-X row '{ref_name}' has no usable val_loss"
                rows.append(entry)
                continue
            if entry['run_identity_full'] == entry['run_identity_reference']:
                entry['skipped'] = (f"'{full_name}' and '{ref_name}' are the same run "
                                    f"(reused summary); the contrast is vacuous")
                rows.append(entry)
                continue
            entry['delta_full_minus_reference'] = full['val_loss'] - ref['val_loss']
            entry['ppl_full'] = _safe_exp(full['val_loss'])
            entry['ppl_reference'] = _safe_exp(ref['val_loss'])
            if (math.isfinite(entry['ppl_full']) and math.isfinite(entry['ppl_reference'])
                    and entry['ppl_reference'] > 0.0):
                entry['ppl_delta_pct'] = (100.0 * (entry['ppl_full'] - entry['ppl_reference'])
                                          / entry['ppl_reference'])
            seed_stds = [value for value in (_row_std(full), _row_std(ref))
                         if _finite_metric(value)]
            if seed_stds:
                denom = max(seed_stds)
                if denom > 0.0:
                    entry['delta_over_max_seed_std'] = (
                        abs(entry['delta_full_minus_reference']) / denom)
                    entry['within_seed_noise'] = bool(
                        abs(entry['delta_full_minus_reference']) < denom)
            entry['mechanism_helps'] = bool(entry['delta_full_minus_reference'] < -margin)
            test, basis, common = _paired_row_test(full, ref, margin=margin,
                                                   alpha=alpha)
            entry['test_basis'] = basis
            entry['paired_seeds'] = common
            entry['replicated_across_seeds'] = _basis_counted_in_family(basis, len(common))
            if test is not None:
                entry['test'] = test['test']
                entry['p_one_sided'] = test['p_one_sided']
                entry['t'] = test['t']
                entry['df'] = test['df']
                entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
                entry['delta_paired_mean'] = test['mean_delta']
                entry['n'] = test['n']
                if all(_finite_metric(test.get(key)) for key in
                       ('mean_delta', 'ci_low', 'ci_high')):
                    effect = ratio_effect_report(
                        test['mean_delta'], test['ci_low'], test['ci_high'])
                    entry.update({
                        'ppl_ratio_paired': effect['ratio'],
                        'ppl_ratio_ci_95': effect['ratio_ci'],
                        'ppl_delta_pct_paired': effect['percent_change'],
                        'ppl_delta_pct_ci_95': effect['percent_change_ci'],
                        'ppl_interval_basis': effect['note'],
                    })
                # Keep the mechanism direction on the same shared-seed
                # estimate the p-value tests, not the all-seed mean.
                if test['mean_delta'] is not None:
                    entry['mechanism_helps'] = bool(test['mean_delta'] < -margin)
            entry.update(_paired_effect_summary(full, ref, common, test, alpha=alpha,
                                               margin=margin))
            entry.update(_robustness_for_deltas(
                _per_seed_deltas(full, ref, common),
                alpha=alpha, equivalence_margin=equivalence_margin, margin=margin))
            if entry['test'] is not None and not entry['replicated_across_seeds']:
                entry['significance_basis'] = (
                    f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                    f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the test cannot certify the mechanism '
                    'effect; not counted in the Holm family')
            rows.append(entry)
    if not rows:
        return None
    # A contrast is identified by the underlying runs, not the row names: an
    # alias pair would otherwise appear as two independent mechanism tests,
    # and a run could even be contrasted with itself via aliases. Each
    # underlying contrast enters the Holm family exactly once.
    contrasts_by_key: dict[tuple, list] = {}
    for entry in rows:
        if entry.get('skipped') or entry.get('run_identity_full') is None:
            continue
        key = (entry['run_identity_full'], entry['run_identity_reference'])
        contrasts_by_key.setdefault(key, []).append(entry)
    for aliases in contrasts_by_key.values():
        # Same rule as the baseline aliases: pick the usable/testable alias,
        # not whichever one happened to be written first.
        canonical = min(
            aliases,
            key=lambda entry: (
                0 if entry['replicated_across_seeds'] else 1,
                0 if entry['test'] is not None else 1,
                -len(entry.get('paired_seeds') or []),
                entry['mechanism'],
            ),
        )
        for entry in aliases:
            if entry is not canonical:
                entry['excluded_from_family'] = (
                    f"duplicate of mechanism '{canonical['mechanism']}' "
                    "(same underlying runs, not independent evidence)")
    family = [entry for entry in rows
              if entry['replicated_across_seeds'] and entry['excluded_from_family'] is None]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        paired_delta = entry.get('delta_paired_mean')
        entry['mechanism_helps_significantly'] = bool(
            paired_delta is not None and paired_delta < -margin
            and p_holm is not None and p_holm < alpha
        )
    thin = sorted({name for entry in rows for name in (entry['full'], entry['reference_row'])
                   if 1 <= _row_n_seeds(usable.get(name, {})) < 3})
    tested_descriptively = [entry['mechanism'] for entry in rows
                            if entry['test'] is not None and not entry['replicated_across_seeds']]
    return {
        'protocol': ('Full vs Full-minus-X: one-sided paired test of H1 "Full has lower '
                     'held-out val_loss than the X-free row"; only the paired test on >= 3 '
                     f'shared per-seed means certifies significance (two or one shared seed(s) '
                     'are descriptive only), and each underlying contrast is counted once even '
                     'when rows are aliases'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'multiple_comparison': (f'Holm-Bonferroni over the '
                                f'{sum(e["p_one_sided"] is not None for e in family)} seed-replicated '
                                f'mechanism(s); `mechanism_helps_significantly` uses the adjusted '
                                f'p-value'),
        'rows': rows,
        'rows_tested_descriptively_only': tested_descriptively,
        'rows_with_fewer_than_3_seeds': thin,
    }

def qk_axis_comparison(summaries: list, spec: dict | None=None, margin: float=0.0,
                       alpha: float=0.05,
                       equivalence_margin: float=0.0) -> dict | None:
    """Attribute the independent QK-compression axis on every host norm.

    The matrix contains a clean QK-off/on pair and a QK-Norm comparator on
    LN, RMSNorm and the full DyT-CD stack. This function tests each pair with
    the same seed-paired one-sided test used everywhere else (`H1: on/off
    lowers held-out loss`) and then reports whether the *direction* of the
    QK effect replicates across hosts. A single host's point estimate is not
    evidence that the attention-logits axis generalizes; a host whose seeds
    do not share the required replication count stays descriptive.
    """
    spec = QK_AXIS_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    rows: list = []
    for host, info in spec.items():
        pair_specs = (
            ('qk_vs_off', info['on'], info['off'],
             'the fixed tanh QK compression lowers loss on this host'),
            ('qknorm_vs_qk', info['comparator'], info['on'],
             'the standard per-head QK-Norm comparator lowers loss relative to the fixed tanh QK compression'),
        )
        for contrast, cand_name, ref_name, hypothesis in pair_specs:
            cand, ref = usable.get(cand_name), usable.get(ref_name)
            entry = {
                'host': host, 'contrast': contrast,
                'candidate_row': cand_name, 'reference_row': ref_name,
                'hypothesis': hypothesis,
                'candidate_val_loss': None if cand is None else cand['val_loss'],
                'reference_val_loss': None if ref is None else ref['val_loss'],
                'delta_candidate_minus_reference': None,
                # Shared-seed effect of the paired test (the certifying
                # statistic); may differ from the all-seed mean difference.
                'delta_paired_mean': None,
                'effect_helps': False,
                'effect_helps_significantly': False,
                'test': None, 'test_basis': None, 'paired_seeds': [],
                'replicated_across_seeds': False, 'significance_basis': None,
                'p_one_sided': None, 'p_one_sided_holm': None,
                't': None, 'df': None, 'delta_ci_95': None,
                'n_seeds_favoring': None, 'n_seeds_tied': None,
                'n_seeds_against': None, 'per_seed_delta': None,
                'paired_delta_std': None,
                'minimum_detectable_delta_95_power80': None,
                'seed_jackknife': None,
                'equivalence': None,
            }
            if cand is None or ref is None:
                missing = [name for name, row in ((cand_name, cand), (ref_name, ref))
                           if row is None]
                entry['skipped'] = f"row(s) {missing} have no usable val_loss"
                rows.append(entry)
                continue
            entry['delta_candidate_minus_reference'] = cand['val_loss'] - ref['val_loss']
            entry['effect_helps'] = bool(entry['delta_candidate_minus_reference'] < -margin)
            test, basis, common = _paired_row_test(cand, ref, margin=margin,
                                                   alpha=alpha)
            entry['test_basis'] = basis
            entry['paired_seeds'] = common
            entry['replicated_across_seeds'] = _basis_counted_in_family(basis, len(common))
            if test is not None:
                entry.update({
                    'test': test['test'], 'p_one_sided': test['p_one_sided'],
                    't': test['t'], 'df': test['df'],
                    'delta_ci_95': [test['ci_low'], test['ci_high']],
                    'delta_paired_mean': test['mean_delta'],
                    'n': test['n'],
                })
                # The paired test is the certifying statistic: its effect is
                # the shared-seed mean. Keep the all-seed difference only as
                # descriptive context; otherwise `effect_helps` and the
                # cross-host direction can contradict a certified paired win
                # when the arms' seed sets differ.
                entry['effect_helps'] = bool(test['mean_delta'] < -margin)
            entry.update(_paired_effect_summary(cand, ref, common, test, alpha=alpha,
                                               margin=margin))
            entry.update(_robustness_for_deltas(
                _per_seed_deltas(cand, ref, common),
                alpha=alpha, equivalence_margin=equivalence_margin, margin=margin))
            if entry['test'] is not None and not entry['replicated_across_seeds']:
                entry['significance_basis'] = (
                    f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                    f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the QK effect cannot be certified; '
                    'not counted in the Holm family')
            rows.append(entry)
    if not rows:
        return None
    family = [entry for entry in rows
              if entry['replicated_across_seeds'] and not entry.get('skipped')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        paired_delta = entry.get('delta_paired_mean')
        entry['effect_helps_significantly'] = bool(
            paired_delta is not None and paired_delta < -margin
            and p_holm is not None and p_holm < alpha)
    # Cross-host replication of the QK-on vs QK-off direction. Only hosts
    # with a seed-replicated test count as independent replications; a host
    # tested on one run is reported but cannot make the axis "generalize".
    replicated_qk = [entry for entry in rows
                     if entry['contrast'] == 'qk_vs_off'
                     and entry['replicated_across_seeds']
                     and entry.get('delta_paired_mean') is not None]
    deltas = [entry['delta_paired_mean'] for entry in replicated_qk]
    signs = {('negative' if delta < 0 else 'positive' if delta > 0 else 'zero')
             for delta in deltas}
    # An all-zero set is a null result, not a replicated direction: no host
    # shows a QK effect, so there is no sign to agree on.
    direction_consistent = (len(signs) == 1 and bool(deltas)
                            and next(iter(signs)) in ('negative', 'positive'))
    certified_hosts = sorted(entry['host'] for entry in replicated_qk
                             if entry['effect_helps_significantly'])
    cross_host = {
        'hosts_with_replicated_test': sorted(entry['host'] for entry in replicated_qk),
        'host_deltas': {entry['host']: entry['delta_paired_mean']
                        for entry in replicated_qk},
        'host_delta_basis': 'shared-seed paired mean (same statistic as the test)',
        'direction_signs': sorted(signs),
        'direction_consistent_across_hosts': direction_consistent,
        'hosts_certified_individually': certified_hosts,
        'certified_at_multiple_hosts': (len(certified_hosts) >= 2
                                        and direction_consistent),
        'reading': ('the attention-logits axis is independent of the norm family: a QK effect '
                    'claimed as general must have the same direction on at least two hosts, and '
                    'is only certified at multiple hosts when >= 2 hosts pass the Holm-adjusted '
                    'paired test; a single host is a scope-limited result'),
    }
    return {
        'protocol': ('QK compression is isolated by switching only `qk_norm` on each host; '
                     'one-sided paired test H1 "on/comparator lowers held-out loss", '
                     f'certified only with >= {MIN_SEEDS_FOR_SIGNIFICANCE} shared per-seed means, '
                     'Holm-adjusted across all QK pairs'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'cross_host_replication': cross_host,
        'rows': rows,
        'rows_tested_descriptively_only': [
            f"{entry['host']}:{entry['contrast']}" for entry in rows
            if entry.get('test') is not None and not entry['replicated_across_seeds']],
    }

def print_qk_axis_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[qk] no QK-axis pair available; nothing to test')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[qk] {entry['host']}/{entry['contrast']}: SKIPPED ({entry['skipped']})")
            continue
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm = entry.get('p_one_sided_holm')
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        note = '' if entry['replicated_across_seeds'] else ' [descriptive only, not certified]'
        print(f"[qk] {entry['host']:>8s} {entry['contrast']:>14s}: "
              f"delta={entry['delta_candidate_minus_reference']:+.4f}{ci_str} "
              f"p={p_str} p(holm)={p_holm_str} helps={entry['effect_helps']} "
              f"significant={entry['effect_helps_significantly']}{note}"
              f"{_robustness_note(entry)}")
    cross = comparison.get('cross_host_replication') or {}
    print(f"[qk] cross-host: {cross.get('host_deltas')} signs={cross.get('direction_signs')} "
          f"consistent={cross.get('direction_consistent_across_hosts')} "
          f"certified_hosts={cross.get('hosts_certified_individually')} "
          f"multi_host={cross.get('certified_at_multiple_hosts')}")
    if not cross.get('certified_at_multiple_hosts'):
        print('[qk] NOTE: the QK axis is not certified at multiple hosts; report any effect '
              'with the host it was measured on (single-host scope), not as a general result')

def print_leave_one_out_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[loo] no leave-one-out contrast available; nothing to test')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[loo] {entry['mechanism']}: SKIPPED ({entry['skipped']})")
            continue
        full_std = entry.get('full_val_loss_std')
        ref_std = entry.get('reference_val_loss_std')
        full_std_str = '' if full_std is None else f' +/- {full_std:.4f}'
        ref_std_str = '' if ref_std is None else f' +/- {ref_std:.4f}'
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm_str = 'n/a' if entry.get('p_one_sided_holm') is None else f"{entry['p_one_sided_holm']:.3g}"
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        note = ''
        if entry.get('excluded_from_family'):
            note = f" [{entry['excluded_from_family']}]"
        elif entry.get('significance_basis'):
            note = ' [descriptive only, not certified]'
        direction = ''
        if entry.get('n_seeds_favoring') is not None:
            direction = (f" seeds_favoring={entry['n_seeds_favoring']}/{len(entry.get('paired_seeds') or [])}")
            if entry.get('n_seeds_tied'):
                direction += f" tied={entry['n_seeds_tied']}"
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f" mde80={mde:.4f}"
        ratio = entry.get('delta_over_max_seed_std')
        ratio_str = '' if ratio is None else f" delta/seed_std={ratio:.2f}"
        ppl = entry.get('ppl_delta_pct')
        ppl_str = '' if ppl is None else f" dPPL={ppl:+.2f}%"
        ppl_ci = entry.get('ppl_delta_pct_ci_95')
        if ppl_ci is not None and all(_finite_metric(value) for value in ppl_ci):
            ppl_str += f" (paired ci [{ppl_ci[0]:+.1f}, {ppl_ci[1]:+.1f}]%)"
        power_warn = ''
        if entry.get('within_seed_noise'):
            power_warn += ' [DELTA WITHIN SEED NOISE]'
        if mde is not None and entry.get('delta_full_minus_reference') is not None:
            if abs(entry['delta_full_minus_reference']) < mde:
                power_warn += ' [UNDERPOWERED: |delta| < mde80]'
        regime_note = ''
        for label, key in (('full', 'full_curve_classification_counts'),
                           ('ref', 'reference_curve_classification_counts')):
            regimes = entry.get(key)
            off_regime = {k: v for k, v in (regimes or {}).items()
                          if k not in ('healthy', 'insufficient_data')}
            if off_regime:
                regime_note += f" [{label} curve regimes {off_regime}]"
        print(f"[loo] {entry['mechanism']:>22s}: {entry['full']} "
              f"val_loss={entry['full_val_loss']:.4f}{full_std_str} vs "
              f"{entry['reference_row']} val_loss={entry['reference_val_loss']:.4f}{ref_std_str} "
              f"delta(full-ref)={entry['delta_full_minus_reference']:+.4f} {ci_str}"
              f"{ppl_str}{ratio_str} p(one-sided)={p_str} p(holm)={p_holm_str} "
              f"test={entry.get('test') or '-'}@{entry.get('test_basis') or '-'}{direction}{mde_str} "
              f"removing_hurts={entry['mechanism_helps']} "
              f"significant={entry['mechanism_helps_significantly']}{note}{regime_note}{power_warn}"
              f"{_robustness_note(entry)}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[loo] NOTE: tests with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are described "
              f"but never certified: {comparison['rows_tested_descriptively_only']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[loo] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def _conditional_effect_entry(base: dict, effect: str, factor: str, context: dict,
                              gate_open: bool, per_seed: dict, common: list,
                              margin: float, alpha: float, formula: str) -> dict:
    """One simple effect of `factor` inside a fixed context of the other axes.

    Simple effects answer "does this factor help *here*?", which the pooled
    main effect and the interaction term cannot: a main effect can be zero
    while the factor helps in one arm and hurts in the other. They are only
    certified when the corresponding interaction survives its two-sided Holm
    family; otherwise they stay descriptive and say so explicitly.
    """
    per_seed = {str(seed): value for seed, value in per_seed.items()}
    values = [per_seed[str(seed)] for seed in common]
    test = paired_t_one_sided(values, [0.0] * len(values), margin=margin,
                              alpha=alpha)
    replicated = len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
    mean_delta = sum(values) / len(values)
    entry = dict(
        base, effect=effect, kind='simple_effect', term_factors=[factor],
        conditional_factor=factor, conditional_context=dict(context),
        analysis_role='conditional_follow_up',
        interaction_gate_open=bool(gate_open),
        conditional_test_status=('tested_follow_up' if gate_open
                                 else 'descriptive_no_interaction'),
        formula=formula, shared_seeds=common, n_seeds=len(common),
        delta_seed_mean=mean_delta,
        per_seed_delta=dict(per_seed),
        n_seeds_favoring=sum(value < 0.0 for value in values),
        n_seeds_tied=sum(value == 0.0 for value in values),
        n_seeds_against=sum(value > 0.0 for value in values),
        effect_helps=bool(mean_delta < -margin),
        effect_helps_significantly=False,
        test=test['test'],
        test_basis=('seed_means' if replicated
                    else f'seed_means_{len(common)}_descriptive'),
        replicated_across_seeds=replicated,
        p_one_sided=test['p_one_sided'],
        p_one_sided_holm=None,
        p_two_sided=None,
        p_two_sided_holm=None,
        interaction_direction=None,
        interaction_differs_significantly=None,
        t=test['t'], df=test['df'],
        delta_ci_95=[test['ci_low'], test['ci_high']],
        n=test['n'])
    effect_row = {'per_seed': [
        {'seed': int(seed), 'val_loss': per_seed[str(seed)]}
        for seed in common]}
    zero_row = {'per_seed': [
        {'seed': int(seed), 'val_loss': 0.0} for seed in common]}
    entry.update(_paired_effect_summary(
        effect_row, zero_row, common, test, alpha=alpha, margin=margin))
    entry.update(_robustness_for_deltas(
        {str(seed): float(per_seed[str(seed)]) for seed in common},
        alpha=alpha, equivalence_margin=0.0, margin=margin))
    paired_std = entry.get('paired_delta_std')
    if (isinstance(paired_std, (int, float)) and paired_std > 0.0
            and entry.get('delta_seed_mean') is not None):
        entry['delta_over_paired_std'] = abs(mean_delta) / paired_std
        entry['within_paired_seed_noise'] = bool(abs(mean_delta) < paired_std)
    else:
        entry['delta_over_paired_std'] = None
        entry['within_paired_seed_noise'] = None
    return entry


def _finalize_conditional_effects(rows: list, conditional_rows: list,
                                  alpha: float) -> list:
    """Gate simple effects on the interaction family and Holm-test the open ones.

    A factor's simple effects are confirmatory follow-ups only when at least
    one interaction involving that factor differs significantly in the
    two-sided interaction family. That gate is what stops "simple effects"
    from becoming an unregistered second look at every cell: when the axes
    are additive, the conditional estimates are descriptive.
    """
    interactions = [entry for entry in rows
                    if entry.get('kind') == 'interaction'
                    and not entry.get('skipped')]
    for entry in conditional_rows:
        factor = entry['conditional_factor']
        gate_open = any(
            row.get('design') == entry.get('design')
            and factor in (row.get('term_factors') or ())
            and row.get('interaction_differs_significantly')
            for row in interactions)
        entry['interaction_gate_open'] = bool(gate_open)
        entry['conditional_test_status'] = ('tested_follow_up' if gate_open
                                            else 'descriptive_no_interaction')
    follow_up = [entry for entry in conditional_rows
                 if entry.get('interaction_gate_open')
                 and entry.get('replicated_across_seeds')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in follow_up])
    for entry, p_holm in zip(follow_up, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['effect_helps_significantly'] = bool(
            entry['effect_helps'] and p_holm is not None and p_holm < alpha)
    return follow_up


def _conditional_effects_summary(conditional_rows: list, follow_up: list) -> dict:
    by_factor: dict = {}
    for entry in conditional_rows:
        factor = f"{entry.get('design')}:{entry['conditional_factor']}"
        info = by_factor.setdefault(factor, {
            'gate_open': False,
            'n_simple_effects': 0,
            'n_tested_follow_up': 0,
            'n_descriptive_gate_closed': 0,
            'n_descriptive_under_replicated': 0,
        })
        info['n_simple_effects'] += 1
        if entry.get('interaction_gate_open'):
            info['gate_open'] = True
            if entry.get('replicated_across_seeds'):
                info['n_tested_follow_up'] += 1
            else:
                info['n_descriptive_under_replicated'] += 1
        else:
            info['n_descriptive_gate_closed'] += 1
    return {
        'protocol': (
            'simple effects of each factor within every level/context of the '
            'other factor(s). They are pre-registered conditional follow-ups: '
            'a factor\'s simple effects are Holm-tested only when at least one '
            'interaction involving that factor survives the two-sided '
            'interaction family; otherwise they are descriptive '
            '(conditional_test_status=descriptive_no_interaction). They are '
            'not counted as new study-level confirmatory claims.'),
        'factors': by_factor,
        'n_simple_effects': len(conditional_rows),
        'n_tested_follow_up': len(follow_up),
        'n_descriptive_gate_closed': sum(
            not entry.get('interaction_gate_open')
            for entry in conditional_rows),
        'n_descriptive_under_replicated': sum(
            entry.get('interaction_gate_open')
            and not entry.get('replicated_across_seeds')
            for entry in conditional_rows),
        'multiple_comparison': (
            f'Holm-Bonferroni over {len(follow_up)} interaction-gated simple '
            'effect(s) using the one-sided p-value'),
    }


def factorial_comparison(summaries: list, spec: dict | None=None, margin: float=0.0,
                         alpha: float=0.05,
                         equivalence_margin: float=0.0) -> dict | None:
    """Paired main effects and interaction for each declared 2x2 factorial.

    The interaction is the difference of the factor's effect in the two arms
    of the other factor, computed per seed:
    ``(both - depth) - (per_channel - neither)``. A negative interaction means
    the two mechanisms together lower loss by more than the sum of their
    separate effects (synergy); a positive one means they overlap or interfere.
    Testing the interaction is what lets the report distinguish "Full wins
    because either axis helps" from "the combination is special". Every term
    uses the same shared seed set across all four cells and the same
    small-sample paired t-test as the other comparisons; only >= 3 shared
    seeds enter the Holm family. The interaction-gated simple effects of each
    factor inside each level of the other factor are appended as conditional
    follow-ups, so a pooled main effect cannot hide an effect that reverses
    sign across contexts. Returns None when no design has all cells.
    """
    spec = FACTORIAL_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    rows: list = []
    for design_name, design in spec.items():
        cells = design['cells']
        factor_names = list(design['factors'])
        missing = [row for row in cells.values() if row not in usable]
        base_entry = {
            'design': design_name,
            'cells': dict(cells),
            'context': design.get('context'),
            'shared_seeds': [],
        }
        if missing:
            entry = dict(base_entry, effect=None, kind='design', skipped=(
                f"cell row(s) {sorted(missing)} have no usable val_loss"))
            rows.append(entry)
            continue
        seed_values = {cell: _seed_means(usable[row]) for cell, row in cells.items()}
        common = sorted(set.intersection(*(set(values) for values in seed_values.values())))
        if len(common) < 2:
            entry = dict(base_entry, effect=None, kind='design', shared_seeds=common,
                         skipped=(f"only {len(common)} seed(s) are shared by all four "
                                  f"cells; a paired effect needs at least 2"))
            rows.append(entry)
            continue
        effects = {}
        for factor in factor_names:
            pairs = design['factors'][factor].get('pairs') or ()
            if len(pairs) != 2:
                raise ValueError(
                    f"factorial '{design_name}', factor '{factor}': expected "
                    f"exactly 2 on/off pairs, got {len(pairs)}")
            arm_values = {}
            for off_cell, on_cell in pairs:
                arm_values[f'{off_cell}->{on_cell}'] = [
                    seed_values[on_cell][seed] - seed_values[off_cell][seed]
                    for seed in common
                ]
            arm_means = [sum(values) / len(values) for values in arm_values.values()]
            main_values = [sum(vals) / len(vals) for vals in zip(*arm_values.values())]
            effects[factor] = {
                'kind': 'main_effect',
                'term_factors': [factor],
                'formula': 'mean over both arms of the other factor: on - off',
                'arm_effects': {name: {'mean_delta': sum(vals) / len(vals),
                                       'per_seed_delta': dict(zip(map(str, common), vals))}
                                for name, vals in arm_values.items()},
                'per_seed_values': dict(zip(map(str, common), main_values)),
                'mean_arm_deltas': arm_means,
                'delta_seed_mean': sum(main_values) / len(main_values),
                'per_seed_delta': dict(zip(map(str, common), main_values)),
            }
        if len(factor_names) == 2:
            first, second = factor_names
            first_arms = list(effects[first]['arm_effects'].items())
            interaction_values = [
                on - off for off, on in zip(
                    first_arms[0][1]['per_seed_delta'].values(),
                    first_arms[1][1]['per_seed_delta'].values())
            ]
            effects[f'{first}_x_{second}'] = {
                'kind': 'interaction',
                'term_factors': [first, second],
                'formula': (f'effect of {first} with {second} on minus effect of {first} '
                            f'with {second} off; negative = synergy'),
                'per_seed_values': dict(zip(map(str, common), interaction_values)),
                'delta_seed_mean': sum(interaction_values) / len(interaction_values),
                'per_seed_delta': dict(zip(map(str, common), interaction_values)),
            }
        for effect_name, effect in effects.items():
            values = list(effect['per_seed_values'].values())
            test = paired_t_one_sided(values, [0.0] * len(values), margin=margin, alpha=alpha)
            # A two-sided test of "the effect is non-zero" is the right
            # question for an interaction: antagonism (positive interaction)
            # is as reportable as synergy, and the one-sided synergy test
            # alone would silently ignore it. The one-sided test stays the
            # certification for "helps", so the existing claim is unchanged.
            two_sided_test = paired_t_one_sided(
                values, [0.0] * len(values), margin=0.0, alpha=alpha)
            p_two_sided = two_sided_p_from_t(
                two_sided_test['t'], two_sided_test['df'])
            replicated = len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
            entry = dict(base_entry, effect=effect_name, kind=effect['kind'],
                         term_factors=list(effect.get('term_factors') or ()),
                         formula=effect['formula'], shared_seeds=common,
                         n_seeds=len(common),
                         delta_seed_mean=effect['delta_seed_mean'],
                         per_seed_delta=effect['per_seed_delta'],
                         n_seeds_favoring=sum(value < 0.0 for value in values),
                         n_seeds_tied=sum(value == 0.0 for value in values),
                         n_seeds_against=sum(value > 0.0 for value in values),
                         effect_helps=bool(effect['delta_seed_mean'] < -margin),
                         effect_helps_significantly=False,
                         test=test['test'],
                         test_basis=('seed_means' if replicated
                                     else f'seed_means_{len(common)}_descriptive'),
                         replicated_across_seeds=replicated,
                         p_one_sided=test['p_one_sided'],
                         p_one_sided_holm=None,
                         p_two_sided=p_two_sided,
                         p_two_sided_holm=None,
                         interaction_differs_significantly=None,
                         interaction_direction=None,
                         t=test['t'],
                         df=test['df'],
                         delta_ci_95=[test['ci_low'], test['ci_high']],
                         n=test['n'])
            if effect['kind'] == 'interaction':
                delta = effect['delta_seed_mean']
                if delta < -margin:
                    entry['interaction_direction'] = 'synergy'
                elif delta > margin:
                    entry['interaction_direction'] = 'antagonism'
                else:
                    entry['interaction_direction'] = 'additive_or_within_margin'
            effect_row = {'per_seed': [{'seed': int(seed), 'val_loss': value}
                                       for seed, value in effect['per_seed_values'].items()]}
            zero_row = {'per_seed': [{'seed': int(seed), 'val_loss': 0.0}
                                     for seed in effect['per_seed_values']]}
            entry.update(_paired_effect_summary(effect_row, zero_row, common, test,
                                              alpha=alpha, margin=margin))
            entry.update(_robustness_for_deltas(
                {str(seed): float(value)
                 for seed, value in effect['per_seed_values'].items()},
                alpha=alpha, equivalence_margin=equivalence_margin, margin=margin))
            paired_std = entry.get('paired_delta_std')
            if (isinstance(paired_std, (int, float)) and paired_std > 0.0
                    and entry.get('delta_seed_mean') is not None):
                entry['delta_over_paired_std'] = (abs(entry['delta_seed_mean'])
                                                 / paired_std)
                entry['within_paired_seed_noise'] = bool(
                    abs(entry['delta_seed_mean']) < paired_std)
            else:
                entry['delta_over_paired_std'] = None
                entry['within_paired_seed_noise'] = None
            if 'arm_effects' in effect:
                entry['arm_effects'] = effect['arm_effects']
            rows.append(entry)
    if not rows:
        return None
    family = [entry for entry in rows
              if entry.get('replicated_across_seeds') and not entry.get('skipped')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['effect_helps_significantly'] = bool(
            entry['effect_helps'] and p_holm is not None and p_holm < alpha)
    # Interactions answer a two-sided question ("is the combination more or
    # less than the sum of its parts?"), so they get their own Holm family
    # over the two-sided p-values. A significant positive interaction is an
    # antagonism finding, not a failed synergy claim.
    interaction_family = [entry for entry in family
                          if entry.get('kind') == 'interaction']
    interaction_adjusted = holm_adjust(
        [entry.get('p_two_sided') for entry in interaction_family])
    for entry, p_holm in zip(interaction_family, interaction_adjusted):
        entry['p_two_sided_holm'] = p_holm
        entry['interaction_differs_significantly'] = bool(
            p_holm is not None and p_holm < alpha)
    conditional_rows: list = []
    for main in rows:
        if main.get('skipped') or main.get('kind') != 'main_effect':
            continue
        design = spec[main['design']]
        factor = main['effect']
        others = [name for name in design['factors'] if name != factor]
        if len(others) != 1:
            continue
        other = others[0]
        other_pairs = design['factors'][other].get('pairs') or ()
        for index, (off_cell, on_cell) in enumerate(
                design['factors'][factor].get('pairs') or ()):
            other_level = ('off', 'on')[index] if index < 2 else str(index)
            for pair_index, pair in enumerate(other_pairs):
                if len(pair) != 2 or pair_index >= 2:
                    continue
                if off_cell == pair[0]:
                    other_level = 'off'
                    break
                if off_cell == pair[1]:
                    other_level = 'on'
                    break
            arm = (main.get('arm_effects') or {}).get(f'{off_cell}->{on_cell}')
            if not arm:
                continue
            conditional_rows.append(_conditional_effect_entry(
                dict(base_entry, design=main['design'],
                     cells=dict(design['cells']),
                     context=design.get('context')),
                effect=f'{factor}@{other}={other_level}', factor=factor,
                context={other: other_level}, gate_open=False,
                per_seed=arm['per_seed_delta'], common=main['shared_seeds'],
                margin=margin, alpha=alpha,
                formula=(f'effect of {factor} on - off within '
                         f'{other}={other_level} (simple effect)')))
    follow_up = _finalize_conditional_effects(rows, conditional_rows, alpha)
    rows.extend(conditional_rows)
    conditional_summary = _conditional_effects_summary(
        conditional_rows, follow_up)
    tested_descriptively = [entry['effect'] for entry in rows
                            if entry.get('p_one_sided') is not None
                            and not entry.get('replicated_across_seeds')]
    return {
        'protocol': ('2x2 factorial per_seed paired terms on the seeds shared by all '
                     'four cells; only tests with >= '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are certified, and all '
                     'main effects and interactions form one Holm family'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'multiple_comparison': (f'Holm-Bonferroni over {len(family)} seed-replicated '
                                'factorial term(s); `effect_helps_significantly` uses the '
                                'adjusted one-sided p-value'),
        'interaction_multiple_comparison': (
            f'Holm-Bonferroni over {len(interaction_family)} seed-replicated interaction '
            'term(s) using the two-sided p-value; `interaction_differs_significantly` '
            'covers synergy and antagonism'),
        'conditional_effects': conditional_summary,
        'designs': spec,
        'rows': rows,
        'rows_tested_descriptively_only': tested_descriptively,
    }

def print_factorial_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[factorial] no 2x2 factorial design declared; interaction test skipped')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[factorial] {entry['design']}: SKIPPED ({entry['skipped']})")
            continue
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm_str = ('n/a' if entry['p_one_sided_holm'] is None
                      else f"{entry['p_one_sided_holm']:.3g}")
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f"mde80={mde:.4f} "
        ratio = entry.get('delta_over_paired_std')
        ratio_str = '' if ratio is None else f"delta/paired_std={ratio:.2f} "
        power_warn = ''
        if entry.get('within_paired_seed_noise'):
            power_warn += ' [DELTA WITHIN PAIRED SEED NOISE]'
        if mde is not None and entry.get('delta_seed_mean') is not None:
            if abs(entry['delta_seed_mean']) < mde:
                power_warn += ' [UNDERPOWERED: |delta| < mde80]'
        note = '' if entry['replicated_across_seeds'] else ' [descriptive only, not certified]'
        if entry.get('kind') == 'simple_effect':
            note += (f" [simple effect; interaction gate "
                     f"{'open' if entry.get('interaction_gate_open') else 'closed'}; "
                     f"{entry.get('conditional_test_status')}]")
        interaction = ''
        if entry.get('kind') == 'interaction':
            p2 = entry.get('p_two_sided')
            p2_holm = entry.get('p_two_sided_holm')
            interaction = (f" interaction={entry.get('interaction_direction')} "
                           f"p(two-sided)={'n/a' if p2 is None else f'{p2:.3g}'} "
                           f"p2(holm)={'n/a' if p2_holm is None else f'{p2_holm:.3g}'} "
                           f"differs={entry.get('interaction_differs_significantly')}")
        print(f"[factorial] {entry['design']} {entry['kind']} {entry['effect']}: "
              f"delta={entry['delta_seed_mean']:+.4f} {ci_str}"
              f"seeds_favoring={entry['n_seeds_favoring']}/{entry['n_seeds']} "
              f"p(one-sided)={p_str} p(holm)={p_holm_str} {mde_str}{ratio_str}"
              f"helps={entry['effect_helps']} "
              f"significant={entry['effect_helps_significantly']}{note}{power_warn}"
              f"{interaction}{_robustness_note(entry)}")
    conditional = comparison.get('conditional_effects') or {}
    if conditional:
        open_factors = sorted(
            factor for factor, info in (conditional.get('factors') or {}).items()
            if (info or {}).get('gate_open'))
        print(f"[factorial] conditional effects: {conditional.get('n_tested_follow_up')} "
              f"tested follow-up(s) (gate open for {open_factors or 'no factor'}), "
              f"{conditional.get('n_descriptive_gate_closed')} descriptive "
              f"(gate closed)")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[factorial] NOTE: terms with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared "
              f"seeds are described but never certified: "
              f"{comparison['rows_tested_descriptively_only']}")

def _factorial_term_coefficients(factors, levels: dict, cells: dict,
                                 term_factors) -> dict:
    """Orthogonal (sum-to-zero) coefficients for one factorial term.

    Levels are coded -1 / +1 in declaration order. A term over `t` of the
    `k` factors multiplies the cell's product code by ``2**-(k-t)``, which
    averages over the factors not in the term. For a 2x2x2 design the
    resulting coefficients are the usual ``+1/-1`` contrasts for interactions
    and ``+/-1/4`` for main effects, without hard-coding each formula.
    """
    term = set(term_factors)
    unknown = term - set(factors)
    if unknown:
        raise ValueError(f'factorial term names unknown factor(s) {sorted(unknown)}')
    if not term:
        raise ValueError('a factorial term must name at least one factor')
    scale = 1.0 / (2 ** (len(factors) - len(term)))
    out = {}
    for cell_key in cells:
        assignment = _parse_three_way_cell_key(cell_key)
        code = 1
        for factor in term:
            factor_levels = levels[factor]
            if assignment.get(factor) not in factor_levels:
                raise ValueError(
                    f'factorial cell {cell_key!r} has no valid level for {factor!r}')
            code *= 1 if assignment[factor] == factor_levels[1] else -1
        out[cell_key] = code * scale
    return out

def three_way_factorial_comparison(summaries: list, spec: dict | None=None,
                                   margin: float=0.0, alpha: float=0.05,
                                   equivalence_margin: float=0.0) -> dict | None:
    """Paired main effects and all interactions of a 2x2x2 ablation design.

    The two 2x2 reports estimate per-channel x depth interactions separately
    under generalized softsign and under tanh. This function pools the eight
    cells into one 2x2x2 design and reports:

    * the per-channel and depth main effects, averaged over both activation
      families (more efficient than either 2x2 estimate alone);
    * all three two-way interactions, including the per-channel x activation
      and depth x activation terms that no existing report estimates; and
    * the three-way interaction, i.e. whether the per-channel x depth
      interaction itself differs between the activation families.

    Every term is a per-seed linear contrast of the eight cell means (levels
    coded -1/+1), tested with the same paired t reference as the rest of the
    module; only >= 3 shared seeds can certify. Interactions are two-sided
    (synergy and antagonism are both findings), and each pairwise interaction
    also carries a TOST equivalence test. The orthogonality block therefore
    only returns a verdict when a positive `equivalence_margin` was
    pre-specified: "no significant interaction" is not evidence that the axes
    are additive, whereas an equivalence test within a stated loss margin is.
    Simple effects of each factor in every context of the other two are added
    as interaction-gated conditional follow-ups.
    """
    spec = THREE_WAY_FACTORIAL_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r
              and _finite_metric(r.get('val_loss'))}
    rows: list = []
    design_reports: dict = {}
    conditional_rows: list = []
    for design_name, design in spec.items():
        factors = tuple(design.get('factors') or ())
        levels = design.get('levels') or {}
        cells = design.get('cells') or {}
        base = {'design': design_name, 'factors': list(factors),
                'design_type': design.get('design_type', '2x2x2'),
                'context': design.get('context'), 'shared_seeds': []}
        if len(factors) != 3:
            rows.append(dict(base, effect=None, kind='design', skipped=(
                f'design must have exactly three factors, got {list(factors)}')))
            continue
        missing = [key for key, row in cells.items() if row not in usable]
        if missing:
            rows.append(dict(base, effect=None, kind='design', skipped=(
                f"cell row(s) {sorted({cells[key] for key in missing})} have no "
                f"usable val_loss")))
            continue
        seed_values = {key: _seed_means(usable[row]) for key, row in cells.items()}
        common = sorted(set.intersection(
            *(set(values) for values in seed_values.values())))
        if len(common) < 2:
            rows.append(dict(base, effect=None, kind='design',
                             shared_seeds=common, skipped=(
                                 f'only {len(common)} seed(s) are shared by all '
                                 f'eight cells; a paired contrast needs at least 2')))
            continue
        design_reports[design_name] = {
            'design_type': design.get('design_type', '2x2x2'),
            'factors': list(factors),
            'levels': {factor: list(levels[factor]) for factor in factors},
            'cells': dict(cells),
            'shared_seeds': common,
        }

        def _term_values(term_factors) -> dict:
            coefficients = _factorial_term_coefficients(
                factors, levels, cells, term_factors)
            return {
                seed: sum(coefficient * seed_values[key][seed]
                          for key, coefficient in coefficients.items())
                for seed in common
            }

        terms: list[tuple[tuple, str, str]] = []
        for factor in factors:
            if factor in ('per_channel', 'depth'):
                terms.append(((factor,), 'main_effect',
                              'mean over both levels of the other two factors: '
                              'on - off'))
        for first, second in itertools.combinations(factors, 2):
            others = [name for name in factors if name not in (first, second)]
            terms.append(((first, second), 'interaction',
                          f'{first} x {second} interaction, averaged over '
                          f'{others[0] if others else "the remaining factor"}'))
        if len(factors) == 3:
            terms.append((tuple(factors), 'interaction',
                          'three-way interaction: change in the '
                          f'{factors[0]} x {factors[1]} interaction between '
                          f'the two levels of {factors[2]}'))
        for term_factors, kind, formula in terms:
            per_seed = _term_values(term_factors)
            values = [per_seed[seed] for seed in common]
            test = paired_t_one_sided(values, [0.0] * len(values),
                                      margin=margin, alpha=alpha)
            two_sided_test = paired_t_one_sided(
                values, [0.0] * len(values), margin=0.0, alpha=alpha)
            p_two_sided = two_sided_p_from_t(
                two_sided_test['t'], two_sided_test['df'])
            replicated = len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
            effect = '_x_'.join(term_factors)
            entry = dict(
                base, effect=effect, kind=kind, term_factors=list(term_factors),
                # Only the interaction terms are added to the pre-registered
                # plan. The pooled main effects are supporting estimates: the
                # same factors are already pre-registered inside each 2x2
                # design, and registering a third, highly correlated estimate
                # would inflate the study family without adding a new claim.
                analysis_role=('supporting_pooled_main_effect'
                               if kind == 'main_effect'
                               else 'cross_axis_interaction'),
                formula=formula, shared_seeds=common, n_seeds=len(common),
                delta_seed_mean=sum(values) / len(values),
                per_seed_delta={str(seed): value
                                for seed, value in per_seed.items()},
                n_seeds_favoring=sum(value < 0.0 for value in values),
                n_seeds_tied=sum(value == 0.0 for value in values),
                n_seeds_against=sum(value > 0.0 for value in values),
                effect_helps=bool(sum(values) / len(values) < -margin),
                effect_helps_significantly=False,
                test=test['test'],
                test_basis=('seed_means' if replicated
                            else f'seed_means_{len(common)}_descriptive'),
                replicated_across_seeds=replicated,
                p_one_sided=test['p_one_sided'],
                p_one_sided_holm=None,
                p_two_sided=p_two_sided,
                p_two_sided_holm=None,
                interaction_direction=None,
                interaction_differs_significantly=None,
                orthogonal_within_margin=None,
                t=test['t'], df=test['df'],
                delta_ci_95=[test['ci_low'], test['ci_high']],
                n=test['n'])
            if kind == 'interaction':
                delta = entry['delta_seed_mean']
                if delta < -margin:
                    entry['interaction_direction'] = 'negative'
                elif delta > margin:
                    entry['interaction_direction'] = 'positive'
                else:
                    entry['interaction_direction'] = 'additive_or_within_margin'
            effect_row = {'per_seed': [
                {'seed': int(seed), 'val_loss': value}
                for seed, value in per_seed.items()]}
            zero_row = {'per_seed': [
                {'seed': int(seed), 'val_loss': 0.0} for seed in per_seed]}
            entry.update(_paired_effect_summary(
                effect_row, zero_row, common, test, alpha=alpha, margin=margin))
            entry.update(_robustness_for_deltas(
                {str(seed): float(value) for seed, value in per_seed.items()},
                alpha=alpha, equivalence_margin=equivalence_margin, margin=margin))
            paired_std = entry.get('paired_delta_std')
            if (isinstance(paired_std, (int, float)) and paired_std > 0.0
                    and entry.get('delta_seed_mean') is not None):
                entry['delta_over_paired_std'] = (
                    abs(entry['delta_seed_mean']) / paired_std)
                entry['within_paired_seed_noise'] = bool(
                    abs(entry['delta_seed_mean']) < paired_std)
            else:
                entry['delta_over_paired_std'] = None
                entry['within_paired_seed_noise'] = None
            if kind == 'interaction':
                equivalence = entry.get('equivalence') or {}
                if equivalence.get('enabled'):
                    entry['orthogonal_within_margin'] = bool(
                        equivalence.get('equivalent'))
            rows.append(entry)
        for factor in factors:
            others = [name for name in factors if name != factor]
            off_level, on_level = levels[factor]
            for combo in itertools.product(
                    *(levels[name] for name in others)):
                context = dict(zip(others, combo))
                off_assignment = dict(context, **{factor: off_level})
                on_assignment = dict(context, **{factor: on_level})
                off_key = '|'.join(
                    f'{name}={off_assignment[name]}' for name in factors)
                on_key = '|'.join(
                    f'{name}={on_assignment[name]}' for name in factors)
                per_seed = {
                    seed: (seed_values[on_key][seed]
                           - seed_values[off_key][seed])
                    for seed in common}
                context_label = '|'.join(
                    f'{name}={context[name]}' for name in others)
                conditional_rows.append(_conditional_effect_entry(
                    dict(base, design=design_name, cells=dict(cells)),
                    effect=f'{factor}@{context_label}', factor=factor,
                    context=context, gate_open=False, per_seed=per_seed,
                    common=common, margin=margin, alpha=alpha,
                    formula=(f'effect of {factor} on - off within '
                             + ', '.join(f'{name}={context[name]}'
                                         for name in others)
                             + ' (simple effect)')))
    if not rows:
        return None
    family = [entry for entry in rows
              if entry.get('replicated_across_seeds') and not entry.get('skipped')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['effect_helps_significantly'] = bool(
            entry['effect_helps'] and p_holm is not None and p_holm < alpha)
    interaction_family = [entry for entry in family
                          if entry.get('kind') == 'interaction']
    interaction_adjusted = holm_adjust(
        [entry.get('p_two_sided') for entry in interaction_family])
    for entry, p_holm in zip(interaction_family, interaction_adjusted):
        entry['p_two_sided_holm'] = p_holm
        entry['interaction_differs_significantly'] = bool(
            p_holm is not None and p_holm < alpha)
    follow_up = _finalize_conditional_effects(rows, conditional_rows, alpha)
    rows.extend(conditional_rows)
    conditional_summary = _conditional_effects_summary(
        conditional_rows, follow_up)
    pairwise = [entry for entry in interaction_family
                if len(entry.get('term_factors') or ()) == 2]
    triple = [entry for entry in interaction_family
              if len(entry.get('term_factors') or ()) == 3]
    interactions_block = {
        entry['effect']: {
            'delta_seed_mean': entry['delta_seed_mean'],
            'interaction_direction': entry['interaction_direction'],
            'differs_significantly': entry['interaction_differs_significantly'],
            'equivalent': (entry.get('equivalence') or {}).get('equivalent'),
            'orthogonal_within_margin': entry.get('orthogonal_within_margin'),
        }
        for entry in pairwise + triple
    }
    if equivalence_margin and equivalence_margin > 0.0:
        if pairwise:
            pairwise_verdict = (
                all(entry.get('orthogonal_within_margin') is not None
                    for entry in pairwise)
                and all(entry.get('orthogonal_within_margin')
                        for entry in pairwise))
            orthogonality_note = (
                'pairwise axes are orthogonal within +/- '
                f'{equivalence_margin:g} only when every pairwise interaction '
                'TOST passes; a non-significant interaction alone is not '
                'equivalence')
        else:
            pairwise_verdict = None
            orthogonality_note = 'no pairwise interaction could be tested'
        if triple:
            triple_equiv = (triple[0].get('equivalence') or {}).get('equivalent')
        else:
            triple_equiv = None
    else:
        pairwise_verdict = None
        triple_equiv = None
        orthogonality_note = (
            'no positive --equivalence-margin was pre-specified, so the axes '
            'cannot be claimed orthogonal from non-significant interactions; '
            'the interaction estimates are reported descriptively')
    tested_descriptively = [entry['effect'] for entry in rows
                            if entry.get('p_one_sided') is not None
                            and not entry.get('replicated_across_seeds')]
    return {
        'schema': 'adasquash.three_way_factorial.v1',
        'protocol': (
            '2x2x2 per-seed orthogonal contrasts on the seeds shared by all '
            'eight cells; levels coded -1/+1, main effects averaged over the '
            'other two factors; interactions are two-sided and Holm-corrected '
            'as their own family. Only >= '
            f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds can certify. The four '
            'interaction terms are pre-registered; the pooled main effects '
            'carry analysis_role=supporting_pooled_main_effect and are not '
            'counted as new confirmatory claims.'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'designs': design_reports,
        'rows': rows,
        'multiple_comparison': (
            f'Holm-Bonferroni over {len(family)} seed-replicated term(s) '
            'using the one-sided p-value for the directional `helps` claim'),
        'interaction_multiple_comparison': (
            f'Holm-Bonferroni over {len(interaction_family)} seed-replicated '
            'interaction(s) using the two-sided p-value; covers negative and '
            'positive interactions'),
        'conditional_effects': conditional_summary,
        'orthogonality': {
            'margin': equivalence_margin,
            'interactions': interactions_block,
            'axes_orthogonal_within_margin': pairwise_verdict,
            'three_way_interaction_absent_within_margin': triple_equiv,
            'note': orthogonality_note,
        },
        'reading_guide': (
            'levels are coded -1/+1 in declaration order (per_channel off/on, '
            'depth off/on, activation softsign/tanh). A negative main effect '
            'means switching the factor on lowers val_loss. A negative '
            'two-way interaction means the combined effect is more negative '
            'than the sum of the separate effects (synergy); a positive one '
            'means antagonism. For an interaction involving activation, '
            'negative means the effect is stronger under tanh than softsign '
            '(coded +1); for the three-way term, negative means the '
            'per-channel x depth interaction is more negative under tanh than '
            'under softsign. `orthogonal_within_margin` is a TOST verdict and '
            'is None when no positive equivalence margin was registered.'),
        'rows_tested_descriptively_only': tested_descriptively,
    }

def print_three_way_factorial_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[factorial3] no 2x2x2 factorial design declared; '
              'cross-axis interaction test skipped')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[factorial3] {entry['design']}: SKIPPED ({entry['skipped']})")
            continue
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm = ('n/a' if entry['p_one_sided_holm'] is None
                  else f"{entry['p_one_sided_holm']:.3g}")
        p2 = entry.get('p_two_sided')
        p2_holm = entry.get('p_two_sided_holm')
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        note = ('' if entry['replicated_across_seeds']
                else ' [descriptive only, not certified]')
        if entry.get('kind') == 'simple_effect':
            note += (f" [simple effect; interaction gate "
                     f"{'open' if entry.get('interaction_gate_open') else 'closed'}; "
                     f"{entry.get('conditional_test_status')}]")
        interaction = ''
        if entry.get('kind') == 'interaction':
            interaction = (
                f" direction={entry.get('interaction_direction')} "
                f"p(two-sided)={'n/a' if p2 is None else f'{p2:.3g}'} "
                f"p2(holm)={'n/a' if p2_holm is None else f'{p2_holm:.3g}'} "
                f"differs={entry.get('interaction_differs_significantly')} "
                f"orthogonal_within_margin={entry.get('orthogonal_within_margin')}")
        print(f"[factorial3] {entry['effect']} ({entry['kind']}): "
              f"delta={entry['delta_seed_mean']:+.4f} {ci_str}"
              f"seeds_favoring={entry['n_seeds_favoring']}/{entry['n_seeds']} "
              f"p(one-sided)={p_str} p(holm)={p_holm} "
              f"helps={entry['effect_helps']} "
              f"significant={entry['effect_helps_significantly']}{note}"
              f"{interaction}{_robustness_note(entry)}")
    orthogonality = comparison.get('orthogonality') or {}
    print(f"[factorial3] orthogonality: axes_orthogonal_within_margin="
          f"{orthogonality.get('axes_orthogonal_within_margin')} "
          f"three_way_absent_within_margin="
          f"{orthogonality.get('three_way_interaction_absent_within_margin')} "
          f"margin={orthogonality.get('margin')}")
    if orthogonality.get('note'):
        print(f"[factorial3] note: {orthogonality['note']}")
    conditional = comparison.get('conditional_effects') or {}
    if conditional:
        open_factors = sorted(
            factor for factor, info in (conditional.get('factors') or {}).items()
            if (info or {}).get('gate_open'))
        print(f"[factorial3] conditional effects: "
              f"{conditional.get('n_tested_follow_up')} tested follow-up(s) "
              f"(gate open for {open_factors or 'no factor'}), "
              f"{conditional.get('n_descriptive_gate_closed')} descriptive "
              f"(gate closed)")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[factorial3] NOTE: terms with fewer than "
              f"{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are described but "
              f"never certified: {comparison['rows_tested_descriptively_only']}")

