from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust,
                               max_p_combination, paired_t_one_sided,
                               ratio_effect_report, student_t_ppf)

from .common import (ROOT, _basis_counted_in_family, _finite_metric,
                     _is_seed_id, _paired_effect_summary, _paired_row_test,
                     _per_seed_deltas, _robustness_for_deltas, _robustness_note,
                     _row_n_seeds, _row_std, _run_identity, _safe_exp,
                     _seed_key_id, _seed_means)
from .runner import _PROTOCOL_MODEL_VOCAB, _protocol_outlier_id

def baseline_comparison(summaries: list, baseline_names, margin: float=0.0, alpha: float=0.05,
                        equivalence_margin: float=0.0) -> dict | None:
    """Every row's held-out val_loss against the best reference baseline.

    The reference is the best (lowest val_loss) *replicated* row among
    `baseline_names` when one exists: a lucky single-seed baseline must not
    become the bar, because no comparison against it could ever be certified
    and its point estimate is the noisiest. The point-estimate best among all
    baselines is still recorded (`point_estimate_best_baseline`), so the
    selection cannot hide a lower unreplicated reference. Significance uses
    the one-sided paired
    Student-t test on the per-seed means whenever both rows share >= 2 seeds,
    but only >= 3 shared seeds can certify significance (2 seeds have df = 1).
    With exactly one shared seed the same run's held-out batches are tested
    for description only - batch noise is not run-to-run noise - and with no
    shared seed no test runs at all.
    Alias rows (`reused_from`) are the same underlying run and are counted
    once, both in the report and in the Holm family. The t reference distribution matters
    at this replication count: at n = 3, df = 2, the one-sided 5% critical
    value is 2.92, not the normal 1.64. `margin` is the smallest loss
    improvement treated as meaningful, and every entry carries a two-sided
    95% interval for the delta (`delta_ci_95`) so a win smaller than its own
    precision is visible as such. The PPL improvement also carries a paired
    interval (`ppl_delta_pct_ci_95`): because PPL = exp(loss), the exp of the
    paired loss interval is the uncertainty for the ratio of geometric-mean
    PPLs, rather than an interval-less percentage computed from arithmetic
    PPL means. Because every row is tested against the
    *same* reference, the per-row p-values are a family of tests and are
    Holm-Bonferroni adjusted over the matrix before anything is called
    significant: `beats_reference_significantly` requires the adjusted
    p < alpha, and the raw one-sided p stays available as `p_one_sided`.
    Returns None when no baseline row has a usable val_loss.
    """
    baseline_names = set(baseline_names)
    usable = [r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))]
    baselines = [r for r in usable if r['name'] in baseline_names]
    if not baselines:
        return None
    parent_of = {r['name']: r['reused_from'] for r in summaries
                 if isinstance(r, dict) and 'name' in r and isinstance(r.get('reused_from'), str)}
    replicated_baselines = [r for r in baselines
                            if len(_seed_means(r)) >= MIN_SEEDS_FOR_SIGNIFICANCE]
    selection_pool = replicated_baselines or baselines
    selection_pool_names = sorted(r['name'] for r in selection_pool)
    reference = min(selection_pool, key=lambda r: r['val_loss'])
    point_best = min(baselines, key=lambda r: r['val_loss'])
    ref_seeds = _seed_means(reference)
    ref_stderr = reference.get('val_loss_seed_stderr')
    if ref_stderr is None:
        ref_stderr = reference.get('val_loss_stderr')
    out = {
        'reference': reference['name'],
        'reference_val_loss': reference['val_loss'],
        'reference_val_loss_std': _row_std(reference),
        'reference_val_loss_stderr': ref_stderr,
        'reference_n_seeds': _row_n_seeds(reference),
        'reference_seeds': sorted(ref_seeds),
        'reference_excluded_seeds': reference.get('excluded_seeds') or [],
        'reference_n_seeds_excluded': reference.get('n_seeds_excluded', 0),
        'reference_selection': {
            'rule': ('lowest val_loss among baselines with >= '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} seeds'
                     if replicated_baselines else
                     'lowest val_loss (no baseline reaches '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} seeds; certification unavailable)'),
            'replicated_baselines': sorted(r['name'] for r in replicated_baselines),
            'unreplicated_baselines': sorted(r['name'] for r in baselines
                                             if r not in replicated_baselines),
            'reference_can_certify': len(ref_seeds) >= MIN_SEEDS_FOR_SIGNIFICANCE,
            'reference_is_point_estimate_best': reference['name'] == point_best['name'],
            'data_dependent': True,
            'pool': selection_pool_names,
            'pool_size': len(selection_pool_names),
            'adjustment': ('intersection-union max-p over the baseline pool: a '
                           'candidate must beat every eligible baseline, not '
                           'only the best-looking one'),
            'p_family_basis': 'p_inferential (= selection-combined p)',
        },
        'point_estimate_best_baseline': {
            'name': point_best['name'],
            'val_loss': point_best['val_loss'],
            'n_seeds': _row_n_seeds(point_best),
            'seeds': sorted(_seed_means(point_best)),
        },
        'baseline_rows': {r['name']: r['val_loss'] for r in baselines},
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'protocol': ('seeds are the unit of replication: mean +/- std across >= 3 seeds; '
                     'only the paired test on >= 3 shared per-seed means certifies significance; '
                     'two or one shared seed(s) are descriptive only, alias rows '
                     '(reused summaries) are counted once, and a seed excluded by the '
                     'pre-specified attrition policy is reported in excluded_seeds instead '
                     'of silently reducing n_seeds'),
        'rows': [],
    }
    for row in usable:
        if row['name'] in baseline_names:
            continue
        delta = row['val_loss'] - reference['val_loss']
        entry = {
            'name': row['name'],
            'hypothesis': row.get('hypothesis'),
            'run_identity': _run_identity(row, parent_of),
            'val_loss': row['val_loss'],
            'val_loss_std': _row_std(row),
            'n_seeds': _row_n_seeds(row),
            'excluded_seeds': row.get('excluded_seeds') or [],
            'n_seeds_excluded': row.get('n_seeds_excluded', 0),
            'exclusion_status_by_seed': row.get('exclusion_status_by_seed'),
            'delta_vs_reference': delta,
            # The paired test is the certifying statistic; its effect is the
            # shared-seed mean, which can differ from the difference of the
            # arms' own all-seed means when their seed sets differ.
            'delta_paired_mean': None,
            # The prompt's scale check: a loss/PPL gap is only readable next
            # to the run-to-run spread it is being compared against. These are
            # descriptive warnings; the paired t-test remains the certification.
            'delta_over_max_seed_std': None,
            'within_seed_noise': None,
            'ppl_reference': _safe_exp(reference['val_loss']),
            'ppl_row': None,
            'ppl_delta': None,
            'ppl_delta_pct': None,
            'ppl_ratio_paired': None,
            'ppl_ratio_ci_95': None,
            'ppl_delta_pct_paired': None,
            'ppl_delta_pct_ci_95': None,
            'ppl_interval_basis': None,
            'beats_reference': bool(delta < -margin),
            'beats_reference_significantly': False,
            'test': None,
            'test_basis': None,
            'paired_seeds': [],
            'replicated_across_seeds': False,
            'significance_basis': None,
            'excluded_from_family': None,
            'p_one_sided': None,
            'p_one_sided_selection_adjusted': None,
            'p_inferential': None,
            'selection_adjustment_complete': True,
            'pool_contrasts': {},
            'beats_every_pool_member': None,
            'delta_vs_pool_worst_case': None,
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
            # Per-seed held-out losses let the calibrated count-model
            # comparison test the same seeds (the count model is fitted on
            # seed s and scored on the seed s+1 stream, exactly like the row).
            'per_seed_val_loss': _seed_means(row),
            # Curve regimes of the row's seeds (None for pre-monitoring
            # summaries): a delta measured while one arm is underfit or
            # overfitting is annotated, so it cannot be read as pure norm
            # effect. It does not alter the significance test - the regime is
            # itself a property of the norm at the fixed step count.
            'curve_classification_counts': row.get('curve_classification_counts'),
        }
        row_std = _row_std(row)
        ref_std = _row_std(reference)
        seed_stds = [value for value in (row_std, ref_std)
                     if _finite_metric(value)]
        if seed_stds:
            denom = max(seed_stds)
            if denom > 0.0:
                entry['delta_over_max_seed_std'] = abs(delta) / denom
                entry['within_seed_noise'] = bool(abs(delta) < denom)
        entry['ppl_row'] = _safe_exp(row['val_loss'])
        entry['ppl_delta'] = entry['ppl_row'] - entry['ppl_reference']
        if math.isfinite(entry['ppl_reference']) and entry['ppl_reference'] > 0.0:
            entry['ppl_delta_pct'] = 100.0 * entry['ppl_delta'] / entry['ppl_reference']
        test, basis, common = _paired_row_test(row, reference, margin=margin,
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
            # The direction must come from the same paired shared-seed
            # estimate as the p-value. The all-seed mean can have the
            # opposite sign when the arms have different seed sets, which
            # made a row both "does not beat" and "significantly beats".
            if test['mean_delta'] is not None:
                entry['beats_reference'] = bool(test['mean_delta'] < -margin)
        entry.update(_paired_effect_summary(row, reference, common, test, alpha=alpha,
                                           margin=margin))
        entry.update(_robustness_for_deltas(
            _per_seed_deltas(row, reference, common),
            alpha=alpha, equivalence_margin=equivalence_margin, margin=margin))
        # The reference baseline is itself the lowest-loss member of a pool
        # chosen after seeing the matrix. The claim "this row beats the
        # baselines" is therefore a claim against the best of that pool, and
        # it needs the candidate to beat every eligible baseline. Combine the
        # per-baseline one-sided p-values with the intersection-union max,
        # which is valid under arbitrary dependence and does not reward
        # picking the most favourable baseline.
        pool_pvalues: list[float] = []
        pool_deltas: list[float] = []
        pool_contrasts: dict[str, dict] = {}
        expected_comparators = 0
        selection_complete = True
        for baseline_row in selection_pool:
            baseline_name = baseline_row['name']
            if baseline_name == entry['name']:
                continue
            expected_comparators += 1
            if baseline_name == reference['name'] and test is not None:
                member_test = test
                member_basis = basis
                member_common = common
            else:
                member_test, member_basis, member_common = _paired_row_test(
                    row, baseline_row, margin=margin, alpha=alpha)
            record = {
                'n_shared_seeds': len(member_common),
                'test_basis': member_basis,
                'comparable': (member_basis == 'seed_means'
                               and len(member_common) >= MIN_SEEDS_FOR_SIGNIFICANCE),
            }
            if member_test is not None:
                record['delta_paired_mean'] = member_test.get('mean_delta')
                record['p_one_sided'] = member_test.get('p_one_sided')
            if not record['comparable']:
                selection_complete = False
                record.setdefault(
                    'reason',
                    'fewer than the required shared seeds for a seed-level test')
                pool_contrasts[baseline_name] = record
                continue
            pool_contrasts[baseline_name] = record
            if member_test is None or not _finite_metric(member_test.get('p_one_sided')):
                selection_complete = False
            else:
                pool_pvalues.append(member_test['p_one_sided'])
            if member_test is None or member_test.get('mean_delta') is None:
                selection_complete = False
            else:
                pool_deltas.append(member_test['mean_delta'])
        entry['pool_contrasts'] = pool_contrasts
        entry['selection_adjustment_complete'] = bool(
            selection_complete and expected_comparators > 0
            and len(pool_pvalues) == expected_comparators)
        entry['p_one_sided_selection_adjusted'] = (
            max_p_combination(pool_pvalues)
            if entry['selection_adjustment_complete'] else None)
        entry['p_inferential'] = (
            entry['p_one_sided_selection_adjusted']
            if entry['selection_adjustment_complete'] else None)
        if pool_deltas:
            entry['delta_vs_pool_worst_case'] = max(pool_deltas)
        entry['beats_every_pool_member'] = bool(
            entry['selection_adjustment_complete']
            and len(pool_deltas) == expected_comparators
            and all(value < -margin for value in pool_deltas))
        if (entry['replicated_across_seeds']
                and not entry['selection_adjustment_complete']
                and entry['excluded_from_family'] is None):
            entry['excluded_from_family'] = (
                'reference-selection pool is incomplete: the candidate was '
                'not tested against every eligible baseline, so the '
                '"beats the baselines" claim is not certifiable')
        if entry['test'] is not None and not entry['replicated_across_seeds']:
            entry['significance_basis'] = (
                f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the test cannot certify the norm effect; '
                'not counted in the Holm family')
        out['rows'].append(entry)
    # Alias rows (identical effective config, `reused_from`) are the *same*
    # run under a second name: they are reported, but a run must not be
    # counted twice as independent evidence nor inflate the Holm family size.
    baseline_identity: dict[str, str] = {}
    for row in baselines:
        baseline_identity.setdefault(_run_identity(row, parent_of), row['name'])
    rows_by_identity: dict[str, list] = {}
    for entry in out['rows']:
        identity = entry.get('run_identity') or entry['name']
        rows_by_identity.setdefault(identity, []).append(entry)
    for identity, aliases in rows_by_identity.items():
        canonical = baseline_identity.get(identity)
        if canonical is None:
            # Pick the canonical alias by evidence quality, not by summary
            # order: an ineligible one-seed alias must not shadow a later
            # replicated row that shares the same run identity.
            canonical = min(
                aliases,
                key=lambda entry: (
                    0 if entry['replicated_across_seeds'] else 1,
                    0 if entry['test'] is not None else 1,
                    -len(entry.get('paired_seeds') or []),
                    entry['name'],
                ),
            )['name']
        for entry in aliases:
            if entry['name'] != canonical:
                entry['excluded_from_family'] = (
                    f'duplicate run of {canonical} (reused summary, not '
                    'independent evidence)')
    # Every seed-replicated row above tests the same reference, so its raw
    # p-values form a family and the per-row alpha must be corrected
    # (Holm-Bonferroni step-down) before a row is reported as a significant
    # win. Rows tested on a single run, duplicates of another row, or rows
    # whose test could not run stay out of the family (and cannot be reported
    # significant).
    family = [entry for entry in out['rows']
              if entry['replicated_across_seeds']
              and entry['excluded_from_family'] is None
              and entry.get('p_inferential') is not None]
    adjusted = holm_adjust([entry['p_inferential'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        # Match the effect estimate to the test that produced the p-value:
        # `delta_paired_mean` is the shared-seed mean, while
        # `delta_vs_reference` differences the arms' own all-seed means.
        paired_delta = entry.get('delta_paired_mean')
        entry['beats_reference_significantly'] = bool(
            paired_delta is not None and paired_delta < -margin
            and entry.get('beats_every_pool_member') is True
            and p_holm is not None and p_holm < alpha
        )
    described_only = [entry['name'] for entry in out['rows']
                      if entry['test'] is not None and not entry['replicated_across_seeds']]
    out['rows_tested_descriptively_only'] = described_only
    out['multiple_comparison'] = (
        f'Holm-Bonferroni over the {sum(p is not None for p in adjusted)} seed-replicated row(s) '
        f'against the best of the eligible baseline pool; each row first passes the '
        f'intersection-union test against every pool member, so '
        f'`beats_reference_significantly` requires beating all of them, not just '
        f'the most favourable one. Single-run batch-level tests are descriptive '
        f'only ({described_only}) and duplicates/aliases are counted once.'
    )
    thin = {r['name'] for r in [reference, *baselines] if _row_n_seeds(r) < 3}
    out['rows_with_fewer_than_3_seeds'] = sorted(thin | {e['name'] for e in out['rows'] if e['n_seeds'] < 3})
    return out

def print_baseline_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[compare] no baseline row present; comparison skipped (include a baseline row to interpret the matrix)')
        return
    ref_std = comparison.get('reference_val_loss_std')
    ref_std_str = '' if ref_std is None else f" +/- {ref_std:.4f}"
    print(f"[compare] reference baseline: {comparison['reference']} "
          f"val_loss={comparison['reference_val_loss']:.4f}{ref_std_str} "
          f"(seeds={comparison.get('reference_seeds')}; "
          f"all baselines: { {k: round(v, 4) for k, v in comparison['baseline_rows'].items()} })")
    selection = comparison.get('reference_selection') or {}
    if selection:
        print(f"[compare] reference selection: {selection.get('rule')}; "
              f"pool={selection.get('pool')}; "
              f"point-estimate best={comparison.get('point_estimate_best_baseline')}; "
              f"reference can certify={selection.get('reference_can_certify')}")
        if selection.get('data_dependent') and selection.get('pool_size', 0) > 1:
            print('[compare] NOTE: this reference was selected after seeing the '
                  'losses; a candidate must beat every pool member '
                  '(intersection-union max-p), not only the best-looking one')
    calibration_status = comparison.get('calibration_status')
    if calibration_status == 'missing':
        print('[compare] calibration: MISSING - rows are ranked only against each other; '
              'regenerate with scripts/baseline_simple.py to read them against the task '
              'Bayes limits (see simple_baseline.how_to_generate in the JSON)')
    elif calibration_status == 'explicitly_disabled':
        print('[compare] calibration: explicitly disabled (--no-simple-baselines)')
    tuning = comparison.get('baseline_tuning') or {}
    if tuning.get('certified_against_tuned_reference'):
        print(f"[compare] baseline tuning: CERTIFIED against {tuning.get('reference_row')!r} "
              f"at the matrix recipe (source: {tuning.get('source')})")
    elif tuning.get('status') == 'not_requested':
        print('[compare] baseline tuning: not requested - results are vs fixed-recipe '
              'baselines, not a certified tuned baseline (see baseline_tuning.how_to_generate)')
    else:
        why = tuning.get('why_not_certified') or [tuning.get('status') or 'missing']
        print('[compare] baseline tuning: NOT CERTIFIED - '
              f"{'; '.join(str(item) for item in why)}")
    for entry in comparison['rows']:
        p = entry['p_one_sided']
        p_holm = entry.get('p_one_sided_holm')
        p_inf = entry.get('p_inferential')
        p_str = 'n/a' if p is None else f'{p:.3g}'
        p_inf_str = ('' if p_inf is None or p_inf == p
                     else f' p(sel)={p_inf:.3g}')
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        randomization = entry.get('randomization_test') or {}
        rand_p = randomization.get('p_one_sided')
        rand_str = '' if rand_p is None else f' p(sign-flip)={rand_p:.3g}'
        std = entry.get('val_loss_std')
        std_str = '' if std is None else f" +/- {std:.4f}"
        basis = entry.get('test_basis') or '-'
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        pi = entry.get('delta_prediction_interval_95')
        pi_str = ('' if pi is None or pi[0] is None
                  else f" pi95=[{pi[0]:+.4f}, {pi[1]:+.4f}]")
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
        if mde is not None and entry.get('delta_vs_reference') is not None:
            if abs(entry['delta_vs_reference']) < mde:
                power_warn += ' [UNDERPOWERED: |delta| < mde80]'
        regimes = entry.get('curve_classification_counts')
        regime_note = ''
        if regimes:
            off_regime = {k: v for k, v in regimes.items()
                          if k not in ('healthy', 'insufficient_data')}
            if off_regime:
                regime_note = (f" [curve regimes {off_regime}: the delta was not measured "
                               f"in a healthy falling-falling regime - read the curve block]")
        print(f"[compare] {entry['name']:>32s} val_loss={entry['val_loss']:.4f}{std_str} "
              f"delta={entry['delta_vs_reference']:+.4f}{ci_str}{pi_str}{ppl_str}{ratio_str} "
              f"p(one-sided)={p_str}{p_inf_str} p(holm)={p_holm_str}"
              f"{rand_str} test={entry.get('test') or '-'}@{basis}{direction}{mde_str} "
              f"beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}{note}{regime_note}{power_warn}"
              f"{_robustness_note(entry)}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[compare] NOTE: tests with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are "
              f"described but never certified: {comparison['rows_tested_descriptively_only']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[compare] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def _simple_baseline_protocol_mismatches(protocol, args) -> list:
    """Fields where a calibration report describes a different task/protocol."""
    if not isinstance(protocol, dict):
        return ['protocol']
    expected = {
        'corpus': 'markov_batch',
        'n_states': 64,
        'max_jump': 3,
        'outlier_channel': _protocol_outlier_id(args.model) is not None,
        'outlier_id': _protocol_outlier_id(args.model),
        'vocab_size': _PROTOCOL_MODEL_VOCAB.get(args.model),
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
    }
    mismatches = [f'{key}={protocol.get(key)!r} != {want!r}'
                  for key, want in expected.items() if protocol.get(key) != want]
    # Seeds are the pairing unit of the count-model test: a calibration fitted
    # on different seeds cannot be paired with these rows even when the task
    # and stream offsets are identical.
    recorded_seeds = protocol.get('seeds')
    current_seeds = getattr(args, 'seeds', None)
    if current_seeds is not None:
        # Normalize only genuine integer seed ids; `int(True)`/`int(42.9)`
        # must not make a calibration fitted on 1/42 look like the current
        # 42/1 seed set. Duplicates are not a set of independent runs.
        recorded = None
        if (isinstance(recorded_seeds, list)
                and all(_is_seed_id(seed) for seed in recorded_seeds)):
            recorded_values = [int(seed) for seed in recorded_seeds]
            if len(set(recorded_values)) == len(recorded_values):
                recorded = set(recorded_values)
        current = None
        current_values = list(current_seeds)
        if all(_is_seed_id(seed) for seed in current_values):
            current_values = [int(seed) for seed in current_values]
            if len(set(current_values)) == len(current_values):
                current = set(current_values)
        if recorded is None or current is None or recorded != current:
            mismatches.append(
                f'seeds={recorded_seeds!r} != '
                f'{sorted(current) if current is not None else "<malformed>"!r}')
    return mismatches

def load_simple_baselines(path, args=None) -> dict | None:
    """Load the Bayes-limit calibration written by scripts/baseline_simple.py.

    Returns None - with a warning for a malformed or stale file - when the
    path is absent or does not look like a calibration report, so a stale side
    file can never abort the matrix; it only removes the floor annotation.
    When `args` is given, the recorded corpus/vocabulary/batch/held-out
    protocol must match this invocation: annotating rows with limits measured
    on another task would silently misstate how much signal they captured.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[compare] WARNING: ignoring unreadable simple-baseline file {path}: {exc}")
        return None
    limits = data.get('bayes_limits') if isinstance(data, dict) else None
    tuned = data.get('tuned_count_baseline') if isinstance(data, dict) else None
    # A bool/NaN/Inf is not a measured cross-entropy: `float(True)` would
    # otherwise be accepted as 1.0 nat and silently annotate every row with a
    # fabricated Bayes floor.
    if not isinstance(limits, dict) or not _finite_metric(limits.get('memoryless_nats')):
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: missing bayes_limits.memoryless_nats")
        return None
    if not isinstance(tuned, dict) or not _finite_metric(tuned.get('mean_ce_nats')):
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: missing tuned_count_baseline.mean_ce_nats")
        return None
    ceiling = limits.get('full_context_nats')
    if ceiling is not None and not _finite_metric(ceiling):
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: "
              f"bayes_limits.full_context_nats is not finite")
        return None
    predictor = tuned.get('predictor')
    if predictor is not None and not isinstance(predictor, str):
        # The predictor is a dict key into each seed record; a list/dict would
        # make `_count_seed_ce` raise on an unhashable lookup.
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: "
              f"tuned_count_baseline.predictor is not a string")
        return None
    if args is not None:
        mismatches = _simple_baseline_protocol_mismatches(data.get('protocol'), args)
        if mismatches:
            print(f"[compare] WARNING: ignoring simple-baseline file {path}: "
                  f"it describes a different protocol ({'; '.join(mismatches)})")
            return None
    return data

def run_simple_baseline_calibration(path, seeds, args) -> dict | None:
    """Run the cheap count-model/Bayes ladder for a missing calibration.

    The ladder fits on the same synthetic generator as the ablation rows
    (one stream per seed), tunes the add-k smoothing on a disjoint stream and
    scores on the held-out stream, so its numbers are directly comparable.
    The vocabulary, outlier id, batch shape and held-out batch count must be
    the invocation's protocol: scoring a different stream would attach a
    Bayes floor/ceiling that describes a different task.
    A failed subprocess is an error: continuing would silently drop the only
    reference that says whether a row learned the task.
    """
    import subprocess
    vocab_size = _PROTOCOL_MODEL_VOCAB.get(args.model)
    if vocab_size is None:
        raise SystemExit(f'[compare] ERROR: no calibration vocabulary is registered '
                         f'for model {args.model!r}')
    cmd = [sys.executable, str(ROOT / 'scripts' / 'baseline_simple.py'),
           '--output', str(path), '--seeds', *[str(int(seed)) for seed in seeds],
           '--vocab-size', str(int(vocab_size)),
           '--batch-size', str(int(args.batch_size)),
           '--seq-len', str(int(args.seq_len)),
           '--val-batches', str(max(1, int(getattr(args, 'val_batches', 20))))]
    print(f"[compare] calibration missing: running {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise SystemExit(f'[compare] ERROR: auto simple-baseline calibration failed '
                         f'(exit {proc.returncode}); fix the ladder run or pass '
                         f'--simple-baselines with an existing report')
    data = load_simple_baselines(path, args)
    if data is None:
        raise SystemExit(f'[compare] ERROR: simple-baseline calibration wrote {path} but it '
                         f'does not parse as a calibration report')
    return data


def _baseline_tuning_protocol_mismatches(protocol: dict, args, seeds) -> list:
    """Fields where a tune.py report describes a different matrix recipe."""
    expected = {
        'model': args.model,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
    }
    mismatches = [f'{key}={protocol.get(key)!r} != {want!r}'
                  for key, want in expected.items() if protocol.get(key) != want]
    recorded = protocol.get('selection_seeds')
    normalized = None
    if isinstance(recorded, list) and all(_is_seed_id(seed) for seed in recorded):
        normalized = [int(seed) for seed in recorded]
    expected = (sorted(int(seed) for seed in seeds)
                if all(_is_seed_id(seed) for seed in seeds) else None)
    if (normalized is None or expected is None
            or len(set(normalized)) != len(normalized)
            or sorted(normalized) != expected):
        mismatches.append(f'selection_seeds={recorded!r} != {expected!r} '
                          '(the baseline lr must be selected on the same independent runs)')
    return mismatches


def _tuning_cell(arm: dict, lr: float, weight_decay: float):
    """The recorded (lr, wd) cell of a tune.py arm, or None."""
    grid = arm.get('lr_grid')
    if not isinstance(grid, dict):
        return None
    for cell in grid.values():
        if (isinstance(cell, dict)
                and cell.get('lr') == float(lr)
                and cell.get('weight_decay') == float(weight_decay)):
            return cell
    return None


def load_baseline_tuning(path, args, rows: dict, seeds) -> dict | None:
    """Load the compute-matched tune.py report and certify the baseline recipe.

    A matrix at one fixed recipe can only claim a win over a *tuned* baseline
    if the baseline's selected ``(lr, weight_decay)`` cell is that same recipe
    and the selection was confirmed on held-back seeds. The report is also
    rejected when its task/protocol (model, steps, batch/sequence, held-out
    batch count or selection seeds) differs from this matrix: a baseline tuned
    on another task cannot certify this comparison.

    Returns ``None`` (with a warning) for a missing, unreadable or
    protocol-mismatched file. `main` turns that into a hard error under
    ``--require-baseline-tuning``; otherwise the comparison is still written
    but explicitly marked as not certified against a tuned reference.
    """
    path = Path(path)
    if not path.exists():
        print(f'[tune-cert] WARNING: baseline tuning report {path} does not exist')
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[tune-cert] WARNING: cannot read baseline tuning report {path}: {exc}')
        return None
    if not isinstance(data, dict):
        print(f'[tune-cert] WARNING: {path} is not a tuning summary object')
        return None
    protocol = data.get('protocol') or {}
    arms = data.get('arms') or {}
    if not isinstance(protocol, dict) or not isinstance(arms, dict):
        print(f'[tune-cert] WARNING: {path} has no protocol/arms block')
        return None
    mismatches = _baseline_tuning_protocol_mismatches(protocol, args, seeds)
    if mismatches:
        print(f'[tune-cert] WARNING: ignoring {path}: it describes a different matrix '
              f'recipe ({"; ".join(mismatches)})')
        return None
    weight_decays = protocol.get('weight_decays')
    if not isinstance(weight_decays, list):
        weight_decays = [protocol.get('weight_decay')]
    recipe_in_grid = (any(isinstance(wd, (int, float)) and float(wd) == args.weight_decay
                          for wd in weight_decays)
                      and any(isinstance(lr, (int, float)) and float(lr) == args.lr
                              for lr in (protocol.get('lrs') or [])))
    # A malformed `comparison` block (list/string/scalar) must not crash the
    # loader that promises a None/warning for a stale report; only an explicit
    # boolean True can certify the tuned baseline.
    comparison = data.get('comparison')
    tuning_confirmed = bool(
        isinstance(comparison, dict) and comparison.get('certified') is True)
    per_baseline = {}
    for row_name in args.baselines:
        norm_name = (rows.get(row_name) or {}).get('norm_name')
        arm = arms.get(norm_name) if norm_name else None
        # A malformed arm (list/string) is treated like a missing one; the
        # guarded `best` read alone still left `.get` calls in the entry
        # builder crashing on a truthy non-dict.
        arm = arm if isinstance(arm, dict) else {}
        best = arm.get('best')
        confirm = arm.get('confirm')
        confirm = confirm if isinstance(confirm, dict) else {}
        confirm_stats = confirm.get('stats')
        confirm_stats = confirm_stats if isinstance(confirm_stats, dict) else {}
        entry = {
            'row': row_name,
            'norm': norm_name,
            'found': isinstance(best, dict),
            'selected_lr': None if not isinstance(best, dict) else best.get('lr'),
            'selected_weight_decay': None if not isinstance(best, dict) else best.get('weight_decay'),
            'selected_val_loss': None if not isinstance(best, dict) else best.get('mean'),
            'selection_pool': None if not isinstance(best, dict) else best.get('selection_pool'),
            'selection_missing_seeds': (None if not isinstance(best, dict)
                                        else best.get('missing_seeds')),
            'selection_complete': bool(
                isinstance(best, dict)
                and best.get('selection_pool') == 'complete_coverage'
                and best.get('missing_seeds') == []),
            'confirm_complete': bool(confirm.get('complete_coverage')),
            'confirm_val_loss': confirm_stats.get('mean'),
            'recipe_in_grid': recipe_in_grid,
            'fixed_recipe_cell_val_loss': None,
            'recipe_is_selected_cell': False,
        }
        if isinstance(best, dict):
            entry['recipe_is_selected_cell'] = (
                entry['selected_lr'] == args.lr
                and entry['selected_weight_decay'] == args.weight_decay)
            cell = _tuning_cell(arm, args.lr, args.weight_decay)
            if cell is not None:
                entry['fixed_recipe_cell_val_loss'] = cell.get('mean')
        per_baseline[row_name] = entry
    found = [entry for entry in per_baseline.values() if entry['found']]
    if not found:
        print(f'[tune-cert] WARNING: {path} has no tuned arm for any baseline '
              f'{args.baselines}')
        return None
    # Rank by the confirm split when every candidate has one, otherwise by the
    # selection grid mean (the same preference tune.py itself uses).
    if all(entry['confirm_complete'] and _finite_metric(entry['confirm_val_loss'])
           for entry in found):
        best_row = min(found, key=lambda entry: entry['confirm_val_loss'])['row']
    else:
        best_row = min(found, key=lambda entry: (entry['selected_val_loss']
                                                 if _finite_metric(entry['selected_val_loss'])
                                                 else float('inf')))['row']
    missing_baselines = sorted(entry['row'] for entry in per_baseline.values()
                               if not entry['found'])
    all_at_recipe = (not missing_baselines
                     and all(entry['recipe_is_selected_cell'] and entry['selection_complete']
                             for entry in found))
    block = {
        'source': str(path),
        'protocol_checked': True,
        'recipe': {'lr': args.lr, 'weight_decay': args.weight_decay,
                   'max_steps': args.max_steps, 'batch_size': args.batch_size,
                   'seq_len': args.seq_len,
                   'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
                   'seeds': list(seeds)},
        'recipe_in_tuning_grid': recipe_in_grid,
        'tuning_confirmed_on_held_back_seeds': tuning_confirmed,
        'per_baseline': per_baseline,
        'best_tuned_baseline_row': best_row,
        'missing_baseline_arms': missing_baselines,
        'all_baselines_selected_this_recipe': all_at_recipe,
        'certified': bool(recipe_in_grid and tuning_confirmed and all_at_recipe),
        'reading': ('a matrix at one fixed recipe may claim a win over a tuned baseline only '
                    'when every compared baseline selected that exact (lr, weight_decay) cell '
                    'and the selection was confirmed on seeds held back from the grid; '
                    'otherwise the comparison is against the fixed-recipe baseline, not its '
                    'tuned optimum'),
    }
    if not block['certified']:
        reasons = []
        if not recipe_in_grid:
            reasons.append('the matrix recipe is not in the tuning grid')
        if not tuning_confirmed:
            reasons.append('tune.py did not certify on held-back --confirm-seeds')
        if missing_baselines:
            reasons.append(f'no tuned arm for baseline(s) {missing_baselines}')
        if not all_at_recipe and not missing_baselines:
            incomplete = [entry['row'] for entry in found if not entry['selection_complete']]
            if incomplete:
                reasons.append('baseline selection used a partial-coverage (survivor-only) '
                               f'cell for {incomplete}')
            if any(not entry['recipe_is_selected_cell'] for entry in found):
                reasons.append('at least one baseline selected a different '
                               '(lr, weight_decay) cell')
        block['why_not_certified'] = reasons
        print(f'[tune-cert] WARNING: {path} does not certify the fixed recipe: '
              f'{"; ".join(reasons)}')
    else:
        print(f'[tune-cert] certified: baselines {sorted(per_baseline)} were tuned at '
              f'lr={args.lr:g} wd={args.weight_decay:g} and confirmed on held-back seeds '
              f'({path})')
    return block


def attach_baseline_tuning(comparison: dict, tuning: dict) -> dict:
    """Tie the tuning certificate to the reference baseline actually used."""
    block = dict(tuning)
    reference = comparison.get('reference')
    entry = (tuning.get('per_baseline') or {}).get(reference)
    block['reference_row'] = reference
    block['reference_baseline_tuned_at_this_recipe'] = bool(
        entry and entry.get('recipe_is_selected_cell'))
    block['certified_against_tuned_reference'] = bool(
        tuning.get('certified') and block['reference_baseline_tuned_at_this_recipe'])
    if not block['certified_against_tuned_reference']:
        print(f'[compare] WARNING: the reference baseline {reference!r} was not certified at '
              f'this matrix recipe; the significance rows below compare against a fixed-recipe '
              f'baseline, not necessarily its tuned optimum (see baseline_tuning)')
    else:
        print(f'[compare] tuned-baseline certificate: reference {reference!r} selected '
              f'lr={tuning["recipe"]["lr"]:g} wd={tuning["recipe"]["weight_decay"]:g} '
              f'and was confirmed on held-back seeds')
    return block


def _count_seed_ce(records: list, predictor: str | None) -> dict:
    """Per-seed CE of the tuned count model on the shared held-out stream."""
    out = {}
    if not predictor:
        return out
    seen: set[int] = set()
    for record in records or []:
        if not isinstance(record, dict):
            continue
        seed = record.get('seed')
        # A boolean/fractional/string seed is not an independent replication
        # id; ``int(True)``/``int(42.9)`` would silently pair the count model
        # with a different ablation run (and can overwrite a real seed's
        # entry after coercion).
        if not _is_seed_id(seed):
            continue
        seed = int(seed)
        if seed in seen:
            # Two records for one seed are the same run; returning a partial
            # set would understate the pairing denominator.
            return {}
        seen.add(seed)
        predictors = record.get('predictors')
        if predictors is None:
            predictors = {}
        if not isinstance(predictors, dict):
            continue
        value = predictors.get(predictor)
        if _finite_metric(value):
            out[seed] = float(value)
    return out


def _mde_from_test(test: dict | None, alpha: float = 0.05, power: float = 0.8):
    if (test is None or not test.get('df')
            or not _finite_metric(test.get('ci_low'))
            or not _finite_metric(test.get('ci_high'))):
        return None
    half = (test['ci_high'] - test['ci_low']) / 2.0
    critical = student_t_ppf(1.0 - alpha / 2.0, float(test['df']))
    if critical <= 0.0:
        return None
    stderr = half / critical
    return ((student_t_ppf(1.0 - alpha, float(test['df']))
             + student_t_ppf(power, float(test['df']))) * stderr)


def attach_simple_baseline_calibration(comparison: dict, floor_data: dict, source,
                                       alpha: float = 0.05) -> dict:
    """Annotate every comparison row with its distance to the task's Bayes limits.

    `baseline_comparison` answers "which row is best among these
    transformers"; this annotation answers "how much of the task's learnable
    signal did the row capture". Both are needed: a row can win the matrix
    while still sitting far above the ceiling, which means the comparison
    ranked under-trained models. The ceiling is the exact HMM-filter CE when
    the corpus carries the outlier channel (the transformer reads the full
    observed prefix), reported next to the closed-form memoryless walk floor.
    `excess_over_bayes_ceiling_nats` = val_loss - ceiling (0 = Bayes optimal),
    `excess_over_tuned_simple_nats` = distance to the tuned count model, and
    `signal_captured_fraction` rescales the remaining gap against the
    uniform-over-vocabulary strawman.

    The count model and the transformer consume the same generator stream for
    each seed (fit on s, scored on s+1), so their per-seed CEs are paired:
    `beats_tuned_simple_significantly` is set only by a one-sided paired
    t-test over shared seeds (>= 3) after a Holm correction across rows. A
    point-estimate win inside the seed noise remains visible as
    `beats_tuned_simple_baseline=True, beats_tuned_simple_significantly=False`.
    """
    limits = floor_data['bayes_limits']
    floor = float(limits['memoryless_nats'])
    ceiling = limits.get('full_context_nats')
    ceiling = float(ceiling) if _finite_metric(ceiling) else None
    tuned = floor_data['tuned_count_baseline']
    tuned_ce = float(tuned['mean_ce_nats'])
    tuned_predictor = tuned.get('predictor')
    seed_ce = _count_seed_ce(floor_data.get('per_seed') or [], tuned_predictor)
    protocol = floor_data.get('protocol') or {}
    vocab_size = protocol.get('vocab_size')
    log_vocab = math.log(vocab_size) if isinstance(vocab_size, (int, float)) and vocab_size > 0 else None
    annotated = []
    for entry in comparison['rows']:
        val_loss = entry.get('val_loss')
        if not _finite_metric(val_loss):
            # NaN/Inf/bool is not a measured held-out loss; annotating it
            # would turn an invalid row into a finite-looking Bayes gap.
            continue
        entry['excess_over_bayes_floor_nats'] = val_loss - floor
        if ceiling is not None:
            entry['excess_over_bayes_ceiling_nats'] = val_loss - ceiling
        entry['excess_over_tuned_simple_nats'] = val_loss - tuned_ce
        entry['beats_tuned_simple_baseline'] = bool(val_loss < tuned_ce)
        if log_vocab is not None and ceiling is not None and log_vocab > ceiling:
            entry['signal_captured_fraction'] = (log_vocab - val_loss) / (log_vocab - ceiling)
        row_seeds = {}
        for seed, value in (entry.get('per_seed_val_loss') or {}).items():
            # JSON object keys are strings; only a canonical integer spelling
            # names a seed. ``int("42.9")`` used to crash the annotation and
            # ``int(True)`` paired the row with run 1.
            seed_id = _seed_key_id(seed)
            if seed_id is not None and _finite_metric(value):
                row_seeds[seed_id] = float(value)
        common = sorted(set(row_seeds) & set(seed_ce))
        test = None
        entry.update({
            'beats_tuned_simple_significantly': False,
            'n_paired_seeds_tuned_simple': len(common),
            'p_one_sided_tuned_simple': None,
            'p_one_sided_tuned_simple_holm': None,
            'delta_seed_mean_tuned_simple': None,
            'delta_ci_95_tuned_simple': None,
            'minimum_detectable_delta_tuned_simple': None,
            'simple_baseline_test_basis': None,
        })
        if len(common) >= 2:
            test = paired_t_one_sided([row_seeds[seed] for seed in common],
                                      [seed_ce[seed] for seed in common],
                                      alpha=alpha)
            entry.update({
                'p_one_sided_tuned_simple': test['p_one_sided'],
                'delta_seed_mean_tuned_simple': test['mean_delta'],
                'delta_ci_95_tuned_simple': [test['ci_low'], test['ci_high']],
                'minimum_detectable_delta_tuned_simple': _mde_from_test(test, alpha=alpha),
                'simple_baseline_test_basis': (
                    'seed_means' if len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
                    else f'descriptive only: {len(common)} shared seeds, fewer than the '
                         f'required {MIN_SEEDS_FOR_SIGNIFICANCE}'),
            })
        elif len(common) == 1:
            entry['simple_baseline_test_basis'] = (
                'descriptive only: one shared seed, so a paired count-model test '
                'cannot separate an arm effect from run-to-run noise')
        else:
            entry['simple_baseline_test_basis'] = 'no_shared_seeds'
        annotated.append(entry)
    family = [entry for entry in annotated
              if entry.get('n_paired_seeds_tuned_simple', 0) >= MIN_SEEDS_FOR_SIGNIFICANCE
              and entry.get('p_one_sided_tuned_simple') is not None
              and entry.get('excluded_from_family') is None]
    adjusted = holm_adjust([entry['p_one_sided_tuned_simple'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_tuned_simple_holm'] = p_holm
        # The certifying test is paired on the shared seeds, so its direction
        # must come from that paired delta, not the all-seed point estimate
        # (an extra seed can move the pooled mean across the tuned CE and
        # suppress a genuine paired win).
        paired_delta = entry.get('delta_seed_mean_tuned_simple')
        entry['beats_tuned_simple_significantly'] = bool(
            paired_delta is not None and paired_delta < 0.0
            and p_holm is not None and p_holm < alpha)
    block = {
        'source': str(source),
        'memoryless_floor_nats': floor,
        'full_context_ceiling_nats': ceiling,
        'tuned_simple_predictor': tuned_predictor,
        'tuned_simple_ce_nats': tuned_ce,
        'tuned_simple_std_ce_nats': tuned.get('std_ce_nats'),
        'uniform_vocab_strawman_nats': log_vocab,
        'calibration_seeds': protocol.get('seeds'),
        'hypothesis': floor_data.get('hypothesis'),
        'paired_test': ('one-sided paired t-test over the seeds shared by the row and the '
                        'tuned count model; only >= '
                        f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds enter the Holm family'),
        'reading': ('excess_over_bayes_ceiling_nats is the gap to the exact full-context Bayes predictor '
                    '(hmm_bayes) and the number that says whether a row learned the task; '
                    'beats_tuned_simple_baseline only says it beat a tuned count model on the point '
                    'estimate, beats_tuned_simple_significantly is the seed-paired test, and '
                    'signal_captured_fraction rescales the gap against uniform-over-vocab'),
        'rows': [{key: entry.get(key) for key in (
            'name', 'val_loss', 'excess_over_bayes_ceiling_nats', 'excess_over_bayes_floor_nats',
            'excess_over_tuned_simple_nats', 'beats_tuned_simple_baseline',
            'beats_tuned_simple_significantly', 'n_paired_seeds_tuned_simple',
            'p_one_sided_tuned_simple', 'p_one_sided_tuned_simple_holm',
            'delta_seed_mean_tuned_simple', 'delta_ci_95_tuned_simple',
            'minimum_detectable_delta_tuned_simple', 'simple_baseline_test_basis',
            'signal_captured_fraction',
        )} for entry in annotated],
    }
    if annotated:
        best = min(annotated, key=lambda entry: entry.get('excess_over_bayes_ceiling_nats', float('inf')))
        ceiling_str = 'n/a' if ceiling is None else f"{ceiling:.4f}"
        best_gap = best.get('excess_over_bayes_ceiling_nats')
        best_str = 'n/a' if best_gap is None else f"{best_gap:+.4f}"
        print(f"[compare] Bayes calibration ({source}): memoryless floor {floor:.4f} nats, full-context "
              f"ceiling {ceiling_str} nats, tuned simple baseline {tuned_ce:.4f} nats "
              f"({tuned_predictor}); closest row to the ceiling: {best['name']} "
              f"{best_str} nats; rows beating the tuned simple baseline: "
              f"{sum(1 for entry in annotated if entry.get('beats_tuned_simple_baseline'))}/{len(annotated)} "
              f"(significantly: "
              f"{sum(1 for entry in annotated if entry.get('beats_tuned_simple_significantly'))})")
    return block

