"""Prospective power planning under multiplicity and run attrition.

The design-readiness audit already checked whether the nominal seed count
reaches a smallest effect. These tests pin the two corrections that decide
whether that check is honest about the study that will actually run:

* the plan is sized at the Bonferroni family level for the number of
  pre-registered primary claims, not at the per-test alpha;
* a complete pair needs both arms, so ``a`` run-level attrition leaves
  ``n * (1 - a)^2`` expected pairs, and the required *planned* pair count is
  inflated accordingly;
* the readiness report exposes the family-adjusted plan before launch and
  fails closed when multiplicity plus attrition leave too few pairs;
* the attrition assumption is frozen in the pre-registration fingerprint.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.design import (design_readiness_audit,
                              prospective_power_plan,
                              randomized_complete_block_schedule,
                              schedule_balance_report,
                              schedule_fingerprint_status,
                              tost_power_plan)
from scripts.ablation import build_analysis_plan


# ---------------------------------------------------------------------------
# Multiplicity and attrition arithmetic
# ---------------------------------------------------------------------------
def test_family_adjusted_plan_needs_more_pairs_than_the_per_test_plan():
    single = prospective_power_plan(0.10, 27, 0.05, n_tests=1)
    family = prospective_power_plan(0.10, 27, 0.05, n_tests=8)
    assert single['family_alpha'] == pytest.approx(0.05)
    assert family['family_alpha'] == pytest.approx(0.00625)
    assert single['required_pairs_unadjusted'] == 27
    assert family['required_pairs_unadjusted'] == 27
    assert family['required_pairs_family_adjusted'] > single['required_pairs_family_adjusted']
    assert single['adequately_powered_family_adjusted'] is True
    assert family['adequately_powered_family_adjusted'] is False
    assert family['shortfall_pairs'] == (
        family['required_planned_pairs'] - family['n_pairs'])


def test_family_adjusted_required_n_is_the_smallest_adequate_plan():
    plan = prospective_power_plan(0.10, 48, 0.05, n_tests=8)
    required = plan['required_pairs_family_adjusted']
    assert required == 48
    assert plan['adequately_powered_after_attrition'] is True
    before = prospective_power_plan(0.10, required - 1, 0.05, n_tests=8)
    assert before['adequately_powered_after_attrition'] is False


def test_attrition_inflates_the_planned_pairs_and_reduces_the_effective_n():
    plan = prospective_power_plan(0.10, 60, 0.05, n_tests=8,
                                  expected_attrition=0.10)
    required = plan['required_pairs_family_adjusted']
    assert plan['complete_pair_survival'] == pytest.approx(0.81)
    assert plan['expected_analyzable_pairs'] == 48
    assert plan['required_planned_pairs'] == 60
    assert required == 48
    assert plan['required_planned_pairs'] >= required
    assert plan['adequately_powered_after_attrition'] is True
    # Loss of pairs widens the resolution even though the nominal n is fixed.
    assert (plan['minimum_detectable_delta_unadjusted']
            <= plan['minimum_detectable_delta_family_adjusted']
            <= plan['minimum_detectable_delta_after_attrition'])


def test_high_attrition_can_make_a_nominally_adequate_design_underpowered():
    plan = prospective_power_plan(0.10, 48, 0.05, n_tests=8,
                                  expected_attrition=0.20)
    assert plan['expected_analyzable_pairs'] == 30
    assert plan['adequately_powered_family_adjusted'] is True
    assert plan['adequately_powered_after_attrition'] is False
    assert plan['required_planned_pairs'] == 75
    assert plan['shortfall_pairs'] == 27


@pytest.mark.parametrize('overrides', [
    {'paired_sd': -0.1},
    {'paired_sd': float('nan')},
    {'n_pairs': 1},
    {'margin': 0.0},
    {'margin': -0.05},
    {'alpha': 0.0},
    {'power': 1.0},
    {'n_tests': 0},
    {'n_tests': True},
    {'expected_attrition': -0.1},
    {'expected_attrition': 1.0},
    {'max_pairs': 1},
])
def test_planning_inputs_are_validated(overrides):
    kwargs = dict(paired_sd=0.1, n_pairs=30, margin=0.05)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        prospective_power_plan(**kwargs)


# ---------------------------------------------------------------------------
# Readiness integration
# ---------------------------------------------------------------------------
def _schedule(n_arms, n_seeds):
    return randomized_complete_block_schedule(
        [f'arm{i}' for i in range(n_arms)],
        [str(seed) for seed in range(n_seeds)], seed=2)


def _audit(seeds, *, n_tests, attrition, sd=0.10, margin=0.05):
    rows = {'arm0': {}, 'arm1': {}}
    hypotheses = {name: 'h' for name in rows}
    schedule = _schedule(len(rows), len(seeds))
    return design_readiness_audit(
        seeds, rows, hypotheses, schedule=schedule,
        expected_units=schedule['order'], margin=margin,
        assumed_paired_sd=sd, identity_controls=[('arm0', 'arm1')],
        schedule_arm_of=lambda unit: unit.rsplit('::rep=', 1)[0],
        n_confirmatory_tests=n_tests, expected_attrition=attrition)


def test_tost_power_plan_uses_equivalence_power_and_attrition():
    plan = tost_power_plan(0.10, 12, equivalence_margin=0.10, n_tests=4,
                           expected_attrition=0.10)
    assert plan['n_tests'] == 4
    assert plan['family_alpha'] == pytest.approx(0.0125)
    assert plan['required_pairs_family_adjusted'] > plan[
        'required_pairs_unadjusted']
    assert plan['required_planned_pairs'] >= plan[
        'required_pairs_family_adjusted']
    assert plan['expected_analyzable_pairs'] == 9
    assert plan['schema'] == 'adasquash.tost_power.v1'


def test_readiness_checks_equivalence_power_separately_from_superiority():
    rows = {'arm0': {}, 'arm1': {}}
    seeds = list(range(12))
    schedule = _schedule(len(rows), len(seeds))
    report = design_readiness_audit(
        seeds, rows, {name: 'h' for name in rows}, schedule=schedule,
        expected_units=schedule['order'], equivalence_margin=0.10,
        assumed_paired_sd=0.10, identity_controls=[('arm0', 'arm1')],
        schedule_arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    by_id = {check['id']: check for check in report['checks']}
    assert 'power' not in by_id
    assert by_id['equivalence_power']['status'] == 'pass'
    assert report['planned_power'] is None
    assert report['planned_equivalence_power']['n_tests'] == 1


def test_readiness_plans_at_the_family_level_and_reports_the_assumptions():
    report = _audit(list(range(27)), n_tests=8, attrition=0.10)
    by_id = {check['id']: check for check in report['checks']}
    assert by_id['power']['status'] == 'fail'
    # 21 analyzable pairs still clear the three-seed replication floor; the
    # failure here is resolution, not replication.
    assert by_id['attrition']['status'] == 'pass'
    assert report['ready'] is False
    plan = report['planned_power']
    assert plan['n_tests'] == 8
    assert plan['family_alpha'] == pytest.approx(0.00625)
    assert plan['required_pairs_family_adjusted'] == 48
    assert plan['expected_analyzable_pairs'] == 21
    assert plan['required_planned_pairs'] == 60
    assert report['n_confirmatory_tests'] == 8
    assert report['expected_attrition'] == pytest.approx(0.10)


def test_readiness_passes_a_plan_with_enough_seeds_after_attrition():
    report = _audit(list(range(60)), n_tests=8, attrition=0.10)
    by_id = {check['id']: check for check in report['checks']}
    assert by_id['power']['status'] == 'pass'
    assert by_id['attrition']['status'] == 'pass'
    assert report['ready'] is True
    assert report['n_warn'] == 0
    assert report['planned_power']['adequately_powered_after_attrition'] is True


def test_readiness_rejects_invalid_attrition_and_multiplicity_inputs():
    bad_attrition = _audit(list(range(10)), n_tests=1, attrition=1.0)
    bad_tests = _audit(list(range(10)), n_tests=0, attrition=0.0)
    for report in (bad_attrition, bad_tests):
        by_id = {check['id']: check for check in report['checks']}
        assert report['ready'] is False
        assert by_id['power']['status'] == 'fail'
    assert {check['id'] for check in bad_attrition['checks']
            if check['status'] == 'fail'} >= {'attrition'}


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------
def test_expected_attrition_is_part_of_the_analysis_plan_fingerprint():
    rows = {'ln_baseline': {'norm_name': 'ln'},
            'dytcd_full': {'norm_name': 'dytcd'}}
    hypotheses = {name: 'h' for name in rows}
    matrix_report = {'leave_one_out': {}, 'factorial': {}, 'qk_axis': {}}
    first = build_analysis_plan(rows, hypotheses, matrix_report,
                                ['ln_baseline'], 'gpt2-124m', [42, 43, 44],
                                expected_attrition=0.0)
    changed = build_analysis_plan(rows, hypotheses, matrix_report,
                                  ['ln_baseline'], 'gpt2-124m', [42, 43, 44],
                                  expected_attrition=0.10)
    assert first['expected_attrition'] == 0.0
    assert changed['expected_attrition'] == pytest.approx(0.10)
    assert first['fingerprint'] != changed['fingerprint']


@pytest.mark.parametrize('bad', [-0.1, 1.0, float('nan'), True])
def test_analysis_plan_rejects_invalid_attrition(bad):
    rows = {'ln_baseline': {'norm_name': 'ln'},
            'dytcd_full': {'norm_name': 'dytcd'}}
    with pytest.raises(ValueError):
        build_analysis_plan(rows, {name: 'h' for name in rows},
                            {'leave_one_out': {}, 'factorial': {}, 'qk_axis': {}},
                            ['ln_baseline'], 'gpt2-124m', [42, 43, 44],
                            expected_attrition=bad)


# ---------------------------------------------------------------------------
# Frozen allocation and complete-block structure
# ---------------------------------------------------------------------------
def test_schedule_fingerprint_verifies_and_detects_an_edited_order():
    schedule = randomized_complete_block_schedule(
        ['a', 'b', 'c'], ['42', '43'], seed=4)
    ok = schedule_fingerprint_status(schedule)
    assert ok['verified'] is True
    assert ok['recomputed'] == schedule['fingerprint']

    edited = dict(schedule)
    edited['order'] = list(reversed(schedule['order']))
    status = schedule_fingerprint_status(edited)
    assert status['verified'] is False
    assert status['recomputed'] != schedule['fingerprint']


def test_balance_confirms_a_complete_block_design():
    schedule = randomized_complete_block_schedule(
        ['a', 'b', 'c'], ['42', '43', '44'], seed=3)
    report = schedule_balance_report(
        schedule, arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    assert report['complete_block_design'] is True
    assert report['balanced'] is True
    assert set(report['block_arm_counts']) == {'0', '1', '2'}
    assert all(counts == {'a': 1, 'b': 1, 'c': 1}
               for counts in report['block_arm_counts'].values())


def test_balance_flags_an_rcbd_that_is_not_crossed():
    # Three arms, three nominal blocks, every arm run three times -- but each
    # arm is missing from one block and duplicated in another. Per-arm block
    # coverage and the total run count both look balanced, so only the
    # arm x block crossing check can see this.
    layout = [('a', 0), ('a', 0), ('b', 0),
              ('b', 1), ('b', 1), ('c', 1),
              ('c', 2), ('c', 2), ('a', 2)]
    entries = [{'unit': unit, 'run_id': f'{unit}::rep={position}',
                'replicate': str(position), 'block': block,
                'position': position}
               for position, (unit, block) in enumerate(layout)]
    schedule = {
        'schema': 'adasquash.rcbd_schedule.v1',
        'n_units': len(entries),
        'n_blocks': 3,
        'block_sizes': [3, 3, 3],
        'entries': entries,
        'order': [entry['run_id'] for entry in entries],
        'fingerprint': 'unused-by-balance-report',
    }
    report = schedule_balance_report(
        schedule, arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    assert report['complete_block_design'] is False
    assert report['balanced'] is False
    assert any('does not cover every arm in every block exactly once' in warning
               for warning in report['warnings'])


def test_readiness_fails_when_the_frozen_schedule_is_edited():
    schedule = randomized_complete_block_schedule(
        ['arm0', 'arm1'], ['0', '1', '2'], seed=1)
    edited = dict(schedule)
    edited['order'] = list(reversed(schedule['order']))
    rows = {'arm0': {}, 'arm1': {}}
    report = design_readiness_audit(
        [0, 1, 2], rows, {name: 'h' for name in rows}, schedule=edited,
        expected_units=edited['order'], margin=0.05,
        identity_controls=[('arm0', 'arm1')],
        schedule_arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    by_id = {check['id']: check for check in report['checks']}
    assert by_id['schedule_integrity']['status'] == 'fail'
    assert report['ready'] is False
