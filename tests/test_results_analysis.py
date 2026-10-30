"""Results-analysis checks: replication intervals and reference selection.

Two failure modes are pinned here:

* a confidence interval for the *mean* is not a prediction interval for the
  next seed; with three runs the latter is much wider, and a claim that does
  not say which one it reports is ambiguous;
* selecting the reference/baseline after seeing the losses (the default in
  both comparison scripts) makes "the candidate beats the reference" a
  selected comparison. The reproducible answer is not to test only the
  best-looking member: the candidate must beat every eligible pool member
  (intersection-union max-p), and that combined p-value is what enters the
  certification family.

The same logic applies to the endpoint: a value selected on the evaluation
data (best_val_loss) or measured without a verified sealed-batch pairing
(test_loss) may be reported, but only the fixed-step final_val_loss can
certify from aggregate.json. PPL ratios inherit the paired loss interval,
because exponentiating a difference of arithmetic-mean PPLs would be a
different, interval-less estimand.

These are pure-Python, synthetic-seed tests; no checkpoint is needed.
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.metrics import (max_p_combination, mean_ci,
                               mean_prediction_interval, ratio_effect_report,
                               student_t_ppf)
from adasquash.repro import TRAINING_RECIPE_KEYS

SEEDS = (42, 43, 44)


def _aggregate(values):
    """A minimal Phase-1 aggregate with complete, cross-seed provenance."""
    ledger = {'sha256': 'ledger', 'bytes': 10}
    version = {'name': 'corpus', 'revision': 'v1'}
    return {
        'norm': 'synthetic',
        'model': 'gpt2-124m',
        'seeds': sorted(values),
        'complete': True,
        'per_seed': [
            {'seed': seed, 'final_val_loss': value, 'final_val_step': 1000}
            for seed, value in sorted(values.items())
        ],
        'shapes_by_seed': {
            str(seed): {'n_layers': 12, 'dim': 768, 'n_heads': 12}
            for seed in values
        },
        'data_versions_by_seed': {str(seed): dict(version) for seed in values},
        'data_ledgers_by_seed': {str(seed): dict(ledger) for seed in values},
    }


def _aggregate_metric(values, metric):
    """The same complete aggregate, with one additional per-seed metric."""
    agg = _aggregate(values)
    for record in agg['per_seed']:
        record[metric] = values[record['seed']]
    return agg


def _manifest(agg):
    hyper = {key: 'synthetic' for key in TRAINING_RECIPE_KEYS}
    return {
        'status': 'finished',
        'seeds': list(agg['seeds']),
        'aggregate': agg,
        'hyperparameters': hyper,
    }


def _arms():
    base_fast = _aggregate({seed: 2.0 for seed in SEEDS})
    base_slow = _aggregate({seed: 2.1 for seed in SEEDS})
    candidate = _aggregate({seed: 1.5 for seed in SEEDS})
    arms = {'base-fast': base_fast, 'base-slow': base_slow,
            'candidate': candidate}
    return arms, {name: _manifest(agg) for name, agg in arms.items()}


# ---------------------------------------------------------------------------
# Prediction intervals
# ---------------------------------------------------------------------------
def test_prediction_interval_is_wider_than_the_confidence_interval():
    values = [1.0, 2.0, 3.0]
    ci = mean_ci(values)
    pi = mean_prediction_interval(values)
    assert pi['n'] == 3
    assert pi['meaningful'] is True
    assert pi['prediction_low'] < ci['ci_low'] < ci['ci_high'] < pi['prediction_high']
    # The exact small-sample factor is sqrt(1 + 1/n) on top of the CI half-width.
    half = student_t_ppf(0.975, 2.0) * math.sqrt(1.0 + 1.0 / 3.0)
    assert pi['prediction_low'] == pytest.approx(2.0 - half, rel=1e-12)
    assert pi['prediction_high'] == pytest.approx(2.0 + half, rel=1e-12)
    # A positive mean plus a between-seed spread: the next seed is more likely
    # than not to stay positive, but not certain.
    assert 0.5 < pi['same_direction_probability'] < 1.0


def test_prediction_interval_refuses_to_invent_a_spread():
    one = mean_prediction_interval([2.0])
    assert one['prediction_low'] is None
    assert one['prediction_high'] is None
    assert one['meaningful'] is False

    constant = mean_prediction_interval([2.0, 2.0, 2.0])
    assert constant['prediction_low'] == 2.0
    assert constant['prediction_high'] == 2.0
    assert constant['same_direction_probability'] == 1.0

    zero = mean_prediction_interval([0.0, 0.0, 0.0])
    assert zero['same_direction_probability'] == 0.5
    assert zero['meaningful'] is True


def test_prediction_interval_ignores_non_finite_values_but_reports_them():
    out = mean_prediction_interval([1.0, float('nan'), 3.0])
    assert out['n'] == 2
    assert out['n_nonfinite'] == 1
    assert out['meaningful'] is False


# ---------------------------------------------------------------------------
# Selection-aware combination
# ---------------------------------------------------------------------------
def test_max_p_combination_is_an_intersection_union_not_a_corrected_min():
    # "all tests reject" is the max of the one-sided p-values; the minimum
    # would answer "at least one test rejects" and must never be used here.
    assert max_p_combination([0.01, 0.04, None]) == 0.04
    assert max_p_combination([0.4, 0.3, 0.2]) == 0.4
    assert max_p_combination([0.9, 0.8]) == 0.9
    assert max_p_combination([]) is None
    assert max_p_combination([None, float('nan'), 1.5, -0.1]) is None


# ---------------------------------------------------------------------------
# compare_arms: explicit reference vs selected reference
# ---------------------------------------------------------------------------
def test_explicit_reference_uses_the_raw_paired_p_value():
    from scripts.compare_arms import compare_arms

    arms, manifests = _arms()
    out = compare_arms(arms, reference='base-fast', manifests=manifests,
                       require_recipe=True)
    row = [entry for entry in out['rows']
           if entry['name'] == 'candidate'][0]
    assert out['reference_selection']['data_dependent'] is False
    assert row['p_one_sided_selection_adjusted'] is None
    assert row['p_inferential'] == row['p_one_sided']
    assert row['certified'] is True


def test_selected_reference_requires_beating_every_pool_member():
    from scripts.compare_arms import compare_arms

    arms = {
        # The candidate beats the weak baseline but not the stronger one.
        'weak': _aggregate({seed: 2.0 for seed in SEEDS}),
        'strong': _aggregate({seed: 1.9 for seed in SEEDS}),
        'candidate': _aggregate({seed: 1.95 for seed in SEEDS}),
    }
    manifests = {name: _manifest(agg) for name, agg in arms.items()}
    out = compare_arms(arms, reference='weak',
                       reference_selection='best-of-pool',
                       reference_pool=['weak', 'strong'],
                       manifests=manifests, require_recipe=True)
    row = [entry for entry in out['rows']
           if entry['name'] == 'candidate'][0]
    assert out['reference_selection']['data_dependent'] is True
    assert out['reference_selection']['pool'] == ['weak', 'strong']
    assert row['p_one_sided'] == 0.0       # beats the weak baseline
    assert row['p_one_sided_selection_adjusted'] == pytest.approx(1.0)
    # The Holm family must use the selection-combined p, not the raw win
    # against the weak member.
    assert row['p_one_sided_holm'] == pytest.approx(1.0)
    assert row['beats_every_pool_member'] is False
    assert row['certified'] is False


def test_selected_reference_certifies_only_when_every_member_is_beaten():
    from scripts.compare_arms import compare_arms

    arms, manifests = _arms()
    out = compare_arms(arms, reference='base-fast',
                       reference_selection='best-of-pool',
                       reference_pool=['base-fast', 'base-slow'],
                       manifests=manifests, require_recipe=True)
    row = [entry for entry in out['rows']
           if entry['name'] == 'candidate'][0]
    assert row['beats_every_pool_member'] is True
    assert row['p_one_sided_selection_adjusted'] == pytest.approx(
        row['pool_contrasts']['base-fast']['p_one_sided'])
    assert row['p_inferential'] == row['p_one_sided_selection_adjusted']
    assert row['certified'] is True


def test_default_reference_is_flagged_as_data_selected():
    from scripts.compare_arms import compare_arms

    arms, manifests = _arms()
    out = compare_arms(arms, manifests=manifests, require_recipe=True)
    assert out['reference'] == 'candidate'  # lowest mean -> reference
    assert out['reference_selection']['data_dependent'] is True
    assert set(out['reference_selection']['pool']) == set(arms)
    assert 'intersection-union' in out['reference_selection']['adjustment']


def test_reference_pool_must_contain_the_named_reference():
    from scripts.compare_arms import compare_arms

    arms, manifests = _arms()
    with pytest.raises(ValueError, match='not in reference_pool'):
        compare_arms(arms, reference='base-fast',
                     reference_selection='best-of-pool',
                     reference_pool=['base-slow'], manifests=manifests)


# ---------------------------------------------------------------------------
# ablation baseline_comparison: same selection problem, same answer
# ---------------------------------------------------------------------------
def _row(name, values):
    vals = [values[seed] for seed in SEEDS]
    return {
        'name': name,
        'val_loss': sum(vals) / len(vals),
        'n_seeds': len(vals),
        'val_loss_seed_std': statistics.stdev(vals),
        'val_loss_seed_stderr': statistics.stdev(vals) / math.sqrt(len(vals)),
        'per_seed': [{'seed': seed, 'val_loss': values[seed]}
                     for seed in SEEDS],
    }


def test_ablation_baseline_selection_uses_the_same_pool_rule():
    from scripts.ablation import baseline_comparison

    weak = _row('weak', {seed: 2.0 for seed in SEEDS})
    strong = _row('strong', {seed: 1.9 for seed in SEEDS})
    candidate = _row('candidate', {seed: 1.95 for seed in SEEDS})
    out = baseline_comparison([weak, strong, candidate], ['weak', 'strong'])
    selection = out['reference_selection']
    assert selection['data_dependent'] is True
    assert selection['pool'] == ['strong', 'weak']
    row = out['rows'][0]
    assert row['p_one_sided'] == 1.0  # selected reference is the stronger one
    assert row['p_one_sided_selection_adjusted'] == pytest.approx(1.0)
    assert row['beats_every_pool_member'] is False
    assert row['beats_reference_significantly'] is False


def test_ablation_certifies_a_candidate_that_beats_every_baseline():
    from scripts.ablation import baseline_comparison

    base_a = _row('base-a', {seed: 2.0 for seed in SEEDS})
    base_b = _row('base-b', {seed: 2.1 for seed in SEEDS})
    candidate = _row('candidate', {seed: 1.5 for seed in SEEDS})
    out = baseline_comparison([base_a, base_b, candidate],
                              ['base-a', 'base-b'])
    row = out['rows'][0]
    assert row['beats_every_pool_member'] is True
    assert row['p_one_sided_selection_adjusted'] == 0.0
    assert row['p_inferential'] == 0.0
    assert row['p_one_sided_holm'] == 0.0
    assert row['beats_reference_significantly'] is True
    assert row['delta_prediction_interval_95'] is not None


def test_trend_test_reports_a_slope_replication_interval():
    from adasquash.metrics import paired_trend_test

    out = paired_trend_test([0.0, 1.0, 2.0], {
        42: [0.0, -1.0, -2.0],
        43: [0.0, -1.1, -2.2],
        44: [0.0, -0.9, -1.8],
    })
    low, high = out['slope_prediction_interval_95']
    assert low is not None and high is not None
    assert low < out['mean_slope_per_dose'] < high
    assert 0.5 < out['slope_same_direction_probability'] <= 1.0
    assert out['slope_prediction_interval_95'][0] < out['slope_ci_95'][0]


# ---------------------------------------------------------------------------
# Endpoint/estimand policy: a loss-shaped number is not automatically a
# confirmatory endpoint
# ---------------------------------------------------------------------------
def test_final_val_loss_is_the_certifiable_aggregate_endpoint():
    from scripts.compare_arms import compare_arms

    arms, manifests = _arms()
    out = compare_arms(arms, reference='base-fast', manifests=manifests,
                       require_recipe=True)
    policy = out['metric_policy']
    assert out['metric'] == 'final_val_loss'
    assert policy['data_dependent'] is False
    assert policy['certifiable_from_aggregate'] is True
    row = [entry for entry in out['rows'] if entry['name'] == 'candidate'][0]
    assert row['metric_certifiable_from_aggregate'] is True
    assert row['certifiable'] is True


def test_best_val_loss_cannot_certify_its_own_selection():
    from scripts.compare_arms import compare_arms

    arms = {
        'reference': _aggregate_metric({seed: 2.0 for seed in SEEDS},
                                       'best_val_loss'),
        'candidate': _aggregate_metric({seed: 1.0 for seed in SEEDS},
                                       'best_val_loss'),
    }
    manifests = {name: _manifest(agg) for name, agg in arms.items()}
    out = compare_arms(arms, reference='reference', metric='best_val_loss',
                       manifests=manifests, require_recipe=True)
    row = out['rows'][0]
    assert out['metric_policy']['data_dependent'] is True
    assert out['metric_policy']['certifiable_from_aggregate'] is False
    assert row['p_one_sided'] == 0.0       # numerically significant ...
    assert row['certifiable'] is False     # ... but selected on the same data
    assert row['certified'] is False
    assert row['p_one_sided_holm'] is None
    assert 'selected extreme' in row['certification_basis']


def test_aggregate_test_loss_requires_the_sealed_pairing_audit():
    from scripts.compare_arms import compare_arms

    arms = {
        'reference': _aggregate_metric({seed: 2.0 for seed in SEEDS},
                                       'test_loss'),
        'candidate': _aggregate_metric({seed: 1.0 for seed in SEEDS},
                                       'test_loss'),
    }
    manifests = {name: _manifest(agg) for name, agg in arms.items()}
    out = compare_arms(arms, reference='reference', metric='test_loss',
                       manifests=manifests, require_recipe=True)
    row = out['rows'][0]
    assert out['metric_policy']['certifiable_from_aggregate'] is False
    assert row['p_one_sided'] == 0.0
    assert row['certifiable'] is False
    assert row['certified'] is False
    assert 'compare_sealed_tests.py' in row['certification_basis']


# ---------------------------------------------------------------------------
# Headline PPL improvement carries the interval of its own estimand
# ---------------------------------------------------------------------------
def test_ratio_effect_report_is_the_exp_of_the_log_interval():
    out = ratio_effect_report(-0.20, -0.30, -0.10)
    assert out['ratio'] == pytest.approx(math.exp(-0.20))
    assert out['ratio_ci'] == pytest.approx([math.exp(-0.30), math.exp(-0.10)])
    assert out['percent_change'] == pytest.approx(
        100.0 * (math.exp(-0.20) - 1.0))
    assert out['percent_change_ci'] == pytest.approx([
        100.0 * (math.exp(-0.30) - 1.0),
        100.0 * (math.exp(-0.10) - 1.0)])
    assert 'geometric-mean' in out['note']


def test_ratio_effect_report_rejects_a_malformed_interval():
    with pytest.raises(ValueError, match='ci_low'):
        ratio_effect_report(0.0, 0.1, -0.1)
    with pytest.raises(ValueError, match='inside'):
        ratio_effect_report(0.5, -0.1, 0.1)
    with pytest.raises(ValueError, match='finite'):
        ratio_effect_report(float('nan'), -0.1, 0.1)


def test_compare_arms_reports_the_paired_ppl_interval():
    from scripts.compare_arms import compare_arms

    arms = {
        'reference': _aggregate({42: 2.00, 43: 2.10, 44: 1.95}),
        'candidate': _aggregate({42: 1.80, 43: 2.00, 44: 1.90}),
    }
    manifests = {name: _manifest(agg) for name, agg in arms.items()}
    out = compare_arms(arms, reference='reference', manifests=manifests,
                       require_recipe=True)
    row = out['rows'][0]
    low, high = row['delta_ci_95']
    assert row['ppl_ratio_ci_95'] == pytest.approx(
        [math.exp(low), math.exp(high)])
    assert row['ppl_delta_pct_ci_95'] == pytest.approx(
        [100.0 * (math.exp(low) - 1.0), 100.0 * (math.exp(high) - 1.0)])
    assert row['ppl_delta_pct_paired'] == pytest.approx(
        100.0 * (math.exp(row['delta_paired_mean']) - 1.0))
    assert 'geometric-mean' in row['ppl_interval_basis']


def test_ablation_reports_the_paired_ppl_interval():
    from scripts.ablation import baseline_comparison

    baseline = _row('baseline', {42: 2.00, 43: 2.10, 44: 1.95})
    candidate = _row('candidate', {42: 1.80, 43: 2.00, 44: 1.90})
    out = baseline_comparison([baseline, candidate], ['baseline'])
    row = out['rows'][0]
    low, high = row['delta_ci_95']
    assert row['ppl_ratio_ci_95'] == pytest.approx(
        [math.exp(low), math.exp(high)])
    assert row['ppl_delta_pct_ci_95'] == pytest.approx(
        [100.0 * (math.exp(low) - 1.0), 100.0 * (math.exp(high) - 1.0)])
    assert 'geometric-mean' in row['ppl_interval_basis']


def test_leave_one_out_reports_the_paired_ppl_interval():
    from scripts.ablation import leave_one_out_comparison

    full = _row('dytcd_full', {42: 1.40, 43: 1.41, 44: 1.39})
    removed = _row('dytcd_no_perchannel', {42: 1.55, 43: 1.54, 44: 1.56})
    spec = {'dytcd_full': {
        'per_channel_delta': {
            'row': 'dytcd_no_perchannel', 'diff': ('learnable_delta',)}}}
    out = leave_one_out_comparison([full, removed], spec=spec)
    entry = out['rows'][0]
    low, high = entry['delta_ci_95']
    assert entry['ppl_ratio_ci_95'] == pytest.approx(
        [math.exp(low), math.exp(high)])
    assert entry['ppl_delta_pct_ci_95'] == pytest.approx(
        [100.0 * (math.exp(low) - 1.0), 100.0 * (math.exp(high) - 1.0)])
    assert 'geometric-mean' in entry['ppl_interval_basis']
