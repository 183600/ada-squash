"""Pre-specified group-sequential monitoring and optional-stopping control.

The ablation layer already freezes contrasts, margins and attrition before
training. These tests pin the missing running-study contract: when a study
looks at the accumulating seeds, the allowed look counts and alpha-spending
thresholds are part of the same pre-registration. Looking after every new seed
and stopping when an unadjusted ``p < alpha`` is optional stopping, so a
result must cross the threshold frozen for that look, not merely the nominal
study alpha.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.sequential import (alpha_spent, audit_paired_contrasts,
                                  group_sequential_plan, sequential_monitor,
                                  sequential_paired_audit)
from scripts.ablation import build_analysis_plan


def _plan(max_pairs=5, n_looks=2, **kwargs):
    return group_sequential_plan(max_pairs, n_looks=n_looks, **kwargs)


def test_alpha_spending_is_monotone_and_spends_the_family_budget():
    for spending in ('obrien-fleming', 'pocock'):
        spent = [alpha_spent(0.05, fraction, spending=spending)
                 for fraction in (0.1, 0.25, 0.5, 0.75, 1.0)]
        assert all(0.0 <= value <= 0.05 for value in spent)
        assert spent == sorted(spent)
        assert spent[-1] == pytest.approx(0.05)
    assert (alpha_spent(0.05, 0.25, spending='obrien-fleming')
            < alpha_spent(0.05, 0.25, spending='pocock'))


@pytest.mark.parametrize('fraction', [0.0, -0.1, 1.01, float('nan')])
def test_alpha_spending_rejects_invalid_information_fractions(fraction):
    with pytest.raises(ValueError):
        alpha_spent(0.05, fraction)


@pytest.mark.parametrize('spending', ['equal', '', None, 1])
def test_alpha_spending_rejects_unknown_methods(spending):
    with pytest.raises(ValueError):
        alpha_spent(0.05, 0.5, spending=spending)


def test_plan_freezes_looks_family_alpha_and_incremental_thresholds():
    plan = _plan(max_pairs=5, n_looks=2, alpha=0.05, n_tests=2)
    assert plan['schema'] == 'adasquash.sequential_plan.v1'
    assert plan['look_pairs'] == [3, 5]
    assert plan['family_alpha'] == pytest.approx(0.025)
    assert plan['n_tests'] == 2
    assert plan['n_looks'] == 2
    assert plan['looks'][-1]['cumulative_alpha'] == pytest.approx(0.025)
    assert sum(look['incremental_alpha'] for look in plan['looks']) == pytest.approx(0.025)
    assert plan['looks'][0]['incremental_alpha'] < plan['looks'][-1]['incremental_alpha']
    assert all(look['rejection_possible'] for look in plan['looks'])


def test_plan_accepts_explicit_looks_and_defaults_are_evenly_spread():
    explicit = _plan(max_pairs=10, n_looks=3, look_pairs=[3, 6, 10])
    assert explicit['look_pairs'] == [3, 6, 10]
    automatic = _plan(max_pairs=10, n_looks=4)
    assert automatic['look_pairs'] == [3, 5, 7, 10]


@pytest.mark.parametrize('kwargs', [
    {'max_pairs': 4, 'n_looks': 4},
    {'max_pairs': 5, 'n_looks': 2, 'look_pairs': [4, 3, 5]},
    {'max_pairs': 5, 'n_looks': 2, 'look_pairs': [2, 5]},
    {'max_pairs': 5, 'n_looks': 2, 'look_pairs': [3, 4]},
    {'max_pairs': 5, 'n_looks': 2, 'look_pairs': '3,5'},
    {'max_pairs': 3, 'n_looks': 2},
    {'max_pairs': 5, 'n_looks': 2, 'spending': 'equal'},
    {'max_pairs': 5, 'n_looks': 2, 'alpha': 1.0},
])
def test_plan_rejects_malformed_or_unspendable_looks(kwargs):
    with pytest.raises(ValueError):
        group_sequential_plan(**kwargs)


def test_monitor_distinguishes_a_frozen_crossing_from_an_unadjusted_peek():
    plan = _plan(max_pairs=5, n_looks=2)
    first_threshold = plan['looks'][0]['nominal_p_threshold']
    assert 0.0 < first_threshold < 0.05

    continuing = sequential_monitor(plan, [0.5])
    assert continuing['status'] == 'continue'
    assert continuing['controlled_significant'] is False
    assert continuing['remaining_looks'] == 1

    stopped = sequential_monitor(plan, [first_threshold / 2.0])
    assert stopped['status'] == 'stopped'
    assert stopped['controlled_significant'] is True
    assert stopped['first_crossing_look'] == 1

    peek = sequential_monitor(plan, [(first_threshold + 0.05) / 2.0])
    assert peek['status'] == 'continue'
    assert peek['unadjusted_significant'] is True
    assert peek['optional_stopping_only'] is True
    assert 'not certified' in peek['note']

    final_null = sequential_monitor(plan, [0.5, 0.4])
    assert final_null['status'] == 'not_significant_at_final_look'
    assert final_null['optional_stopping_only'] is False


def test_monitor_rejects_extending_the_plan_after_the_fact():
    plan = _plan(max_pairs=5, n_looks=2)
    with pytest.raises(ValueError):
        sequential_monitor(plan, [0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        sequential_monitor(plan, [])
    with pytest.raises(ValueError):
        sequential_monitor(plan, [1.5])
    broken = dict(plan)
    broken['looks'] = [dict(plan['looks'][0], nominal_p_threshold=1.5),
                       plan['looks'][1]]
    with pytest.raises(ValueError):
        sequential_monitor(broken, [0.5])


def test_paired_audit_tests_only_the_frozen_prefixes():
    plan = _plan(max_pairs=5, n_looks=2)
    stopped = sequential_paired_audit([-1.0, -1.2, -0.9, -1.1, -1.0], plan)
    assert stopped['paired_deltas_available'] == 5
    assert stopped['analyzed_looks'] == 2
    assert stopped['controlled_significant'] is True
    assert [look['planned_pairs'] for look in stopped['looks']] == [3, 5]

    continuing = sequential_paired_audit([0.1, 0.2, 0.0], plan)
    assert continuing['status'] == 'continue'
    assert continuing['analyzed_looks'] == 1
    assert continuing['remaining_looks'] == 1


def test_paired_audit_rejects_design_mismatches_and_partial_first_looks():
    plan = _plan(max_pairs=5, n_looks=2)
    with pytest.raises(ValueError):
        sequential_paired_audit([0.0] * 6, plan)
    with pytest.raises(ValueError):
        sequential_paired_audit([0.1, 0.2], plan)
    with pytest.raises(ValueError):
        sequential_paired_audit([0.1, 0.2, float('nan')], plan)


def test_contrast_audit_summarises_stopped_continuing_and_empty_families():
    plan = _plan(max_pairs=5, n_looks=2)
    report = audit_paired_contrasts(
        {'strong': [-1.0, -1.2, -0.9], 'null': [0.1, 0.2, 0.0]}, plan)
    assert report['n_contrasts'] == 2
    assert report['n_stopped'] == 1
    assert report['stopped_contrasts'] == ['strong']
    assert report['n_continuing'] == 1
    assert report['continuing_contrasts'] == ['null']
    assert set(report['contrasts']) == {'strong', 'null'}

    empty = audit_paired_contrasts({}, plan)
    assert empty['n_contrasts'] == 0
    assert empty['n_stopped'] == 0
    assert 'no contrast' in empty['note']


def _analysis_inputs():
    rows = {'ln_baseline': {'norm_name': 'ln'},
            'dytcd_full': {'norm_name': 'dytcd'}}
    hypotheses = {name: 'h' for name in rows}
    matrix_report = {'leave_one_out': {}, 'factorial': {},
                     'three_way_factorial': {}, 'qk_axis': {}}
    return rows, hypotheses, matrix_report


def test_analysis_plan_without_looks_is_byte_compatible_with_the_old_plan():
    rows, hypotheses, matrix_report = _analysis_inputs()
    implicit = build_analysis_plan(rows, hypotheses, matrix_report,
                                   ['ln_baseline'], 'gpt2-124m',
                                   [42, 43, 44, 45, 46])
    explicit = build_analysis_plan(rows, hypotheses, matrix_report,
                                   ['ln_baseline'], 'gpt2-124m',
                                   [42, 43, 44, 45, 46], sequential_looks=0)
    assert 'sequential_monitoring' not in implicit
    assert implicit['fingerprint'] == explicit['fingerprint']


def test_analysis_plan_freezes_looks_and_family_size_in_the_fingerprint():
    rows, hypotheses, matrix_report = _analysis_inputs()
    baseline = build_analysis_plan(rows, hypotheses, matrix_report,
                                   ['ln_baseline'], 'gpt2-124m',
                                   [42, 43, 44, 45, 46])
    monitored = build_analysis_plan(rows, hypotheses, matrix_report,
                                    ['ln_baseline'], 'gpt2-124m',
                                    [42, 43, 44, 45, 46],
                                    sequential_looks=3)
    assert monitored['sequential_monitoring']['look_pairs'] == [3, 4, 5]
    assert (monitored['sequential_monitoring']['n_tests']
            == sum(1 for entry in monitored['contrasts']
                   if entry.get('tier') == 1))
    assert monitored['fingerprint'] != baseline['fingerprint']

    pocock = build_analysis_plan(rows, hypotheses, matrix_report,
                                 ['ln_baseline'], 'gpt2-124m',
                                 [42, 43, 44, 45, 46],
                                 sequential_looks=3,
                                 sequential_spending='pocock')
    assert (pocock['sequential_monitoring']['spending'] == 'pocock')
    assert pocock['fingerprint'] != monitored['fingerprint']


@pytest.mark.parametrize('looks', [-1, True, '3', 1.5])
def test_analysis_plan_rejects_invalid_sequential_look_counts(looks):
    rows, hypotheses, matrix_report = _analysis_inputs()
    with pytest.raises(ValueError):
        build_analysis_plan(rows, hypotheses, matrix_report,
                            ['ln_baseline'], 'gpt2-124m',
                            [42, 43, 44, 45, 46],
                            sequential_looks=looks)


def test_analysis_plan_rejects_unknown_spending_with_looks():
    rows, hypotheses, matrix_report = _analysis_inputs()
    with pytest.raises(ValueError):
        build_analysis_plan(rows, hypotheses, matrix_report,
                            ['ln_baseline'], 'gpt2-124m',
                            [42, 43, 44, 45, 46],
                            sequential_looks=2, sequential_spending='equal')


def test_cli_exposes_the_monitoring_flags():
    from adasquash.ablation.cli import build_arg_parser
    parser, _, _, _ = build_arg_parser(
        ['--sequential-looks', '2', '--sequential-spending', 'pocock'])
    args = parser.parse_args(['--sequential-looks', '2',
                              '--sequential-spending', 'pocock'])
    assert args.sequential_looks == 2
    assert args.sequential_spending == 'pocock'
    defaults = parser.parse_args([])
    assert defaults.sequential_looks == 0
    assert defaults.sequential_spending == 'obrien-fleming'
