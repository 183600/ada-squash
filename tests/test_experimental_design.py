"""Checks for the pre-run experimental-design layer.

The comparison reports are covered elsewhere. These tests pin the pieces that
have to be right *before* a launch:

* the execution schedule is deterministic, complete and balanced;
* an RCBD gives every arm one run per time block and every seed once per arm;
* order-drift detection survives a real arm effect (within-arm centering) but
  catches a within-arm monotone drift;
* the readiness audit refuses a design without replication/hypotheses/a
  complete schedule/an identity control and flags an underpowered plan;
* the analysis pipeline itself has the advertised type-I error and power on
  synthetic null/alternative data;
* the pre-registration fingerprint moves when the randomized order moves.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.design import (block_randomized_schedule,
                              design_readiness_audit, order_drift_report,
                              randomized_complete_block_schedule,
                              schedule_balance_report,
                              simulate_paired_design)
from scripts.ablation import (aggregate_seed_summaries, build_analysis_plan,
                              run_order_audit)

SEEDS = [42, 43, 44]


# ---------------------------------------------------------------------------
# Simple block schedule
# ---------------------------------------------------------------------------
def test_block_schedule_is_deterministic_complete_and_blocked():
    units = [f'run{i}' for i in range(11)]
    first = block_randomized_schedule(units, block_size=4, seed=7)
    second = block_randomized_schedule(units, block_size=4, seed=7)
    assert first['order'] == second['order']
    assert first['fingerprint'] == second['fingerprint']
    assert sorted(first['order']) == sorted(units)
    assert [entry['position'] for entry in first['entries']] == list(range(11))
    assert first['block_sizes'] == [4, 4, 3]
    assert first['n_blocks'] == 3
    # Each block's positions are contiguous and its members stay together.
    for block in range(3):
        positions = [entry['position'] for entry in first['entries']
                     if entry['block'] == block]
        assert positions == list(range(min(positions), max(positions) + 1))


def test_block_schedule_changes_with_the_seed():
    units = [f'run{i}' for i in range(12)]
    assert (block_randomized_schedule(units, block_size=4, seed=1)['order']
            != block_randomized_schedule(units, block_size=4, seed=2)['order'])


@pytest.mark.parametrize('kwargs', [
    {'units': [], 'block_size': 2},
    {'units': ['a', 'a'], 'block_size': 2},
    {'units': ['a'], 'block_size': 0},
    {'units': ['a'], 'block_size': 2, 'seed': True},
])
def test_block_schedule_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        block_randomized_schedule(**kwargs)


# ---------------------------------------------------------------------------
# Randomized complete block design
# ---------------------------------------------------------------------------
def test_rcbd_gives_one_run_per_arm_per_block_and_every_seed_once():
    schedule = randomized_complete_block_schedule(
        ['a', 'b', 'c'], ['42', '43', '44'], seed=5)
    assert schedule['n_units'] == 9
    assert schedule['n_blocks'] == 3
    assert schedule['block_sizes'] == [3, 3, 3]
    assert sorted(schedule['order']) == sorted(
        f'{unit}::rep={rep}' for unit in ('a', 'b', 'c')
        for rep in ('42', '43', '44'))
    for unit in ('a', 'b', 'c'):
        reps = sorted(entry['replicate'] for entry in schedule['entries']
                      if entry['unit'] == unit)
        assert reps == ['42', '43', '44']
        blocks = [entry['block'] for entry in schedule['entries']
                  if entry['unit'] == unit]
        assert blocks == [0, 1, 2]
    for block in range(3):
        assert sorted(entry['unit'] for entry in schedule['entries']
                      if entry['block'] == block) == ['a', 'b', 'c']
    balance = schedule_balance_report(
        schedule, arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    assert balance['balanced'] is True
    assert all(bucket['n_blocks'] == 3 for bucket in balance['per_arm'].values())


def test_rcbd_changes_with_the_seed_but_keeps_the_design_balanced():
    a = randomized_complete_block_schedule(['a', 'b'], ['1', '2'], seed=0)
    b = randomized_complete_block_schedule(['a', 'b'], ['1', '2'], seed=1)
    assert a['order'] != b['order']
    assert a['n_blocks'] == b['n_blocks'] == 2


def test_balance_report_flags_an_arm_confined_to_one_block():
    entries = []
    # Arm 'stuck' runs only in block 0; 'spread' appears in both blocks.
    for position, (unit, block) in enumerate([
            ('stuck::rep=1', 0), ('spread::rep=1', 0),
            ('stuck::rep=2', 0), ('spread::rep=2', 0),
            ('other1', 0), ('other2', 0),
            ('spread::rep=3', 1), ('other3', 1),
            ('other4', 1), ('other5', 1)]):
        entries.append({'unit': unit, 'block': block, 'position': position})
    schedule = {'schema': 'test', 'n_units': 10, 'n_blocks': 2,
                'block_sizes': [6, 4], 'entries': entries,
                'order': [entry['unit'] for entry in entries],
                'positions': {entry['unit']: entry['position']
                              for entry in entries},
                'fingerprint': 'test'}
    report = schedule_balance_report(
        schedule, arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    assert report['balanced'] is False
    assert any('only one time block' in warning for warning in report['warnings'])


# ---------------------------------------------------------------------------
# Within-arm order drift
# ---------------------------------------------------------------------------
def test_order_drift_is_not_flagged_for_a_real_arm_effect():
    rng = random.Random(11)
    observations = []
    for position in range(15):
        arm = ('a', 'b', 'c')[position % 3]
        base = {'a': 2.0, 'b': 2.6, 'c': 3.2}[arm]
        observations.append({
            'unit': f'{arm}-{position}', 'arm': arm, 'position': position,
            'value': base + rng.gauss(0.0, 0.01),
        })
    report = order_drift_report(observations, permutations=500, seed=3)
    assert report['status'] == 'tested'
    assert report['drift_detected'] is False
    assert report['n_arms'] == 3


def test_order_drift_detects_a_within_arm_monotone_trend():
    rng = random.Random(12)
    observations = []
    for arm, base in (('a', 2.0), ('b', 2.4), ('c', 2.8)):
        for position in range(5):
            observations.append({
                'unit': f'{arm}-{position}', 'arm': arm, 'position': position,
                'value': base + 0.20 * position + rng.gauss(0.0, 0.005),
            })
    report = order_drift_report(observations, permutations=500, seed=4)
    assert report['status'] == 'tested'
    assert report['spearman_rho'] > 0.5
    assert report['p_two_sided'] < 0.05
    assert report['drift_detected'] is True
    assert report['direction'] == 'later_runs_worse'


def test_order_drift_excludes_singletons_and_reports_insufficient_data():
    observations = [
        {'arm': 'a', 'position': 0, 'value': 1.0},
        {'arm': 'b', 'position': 1, 'value': 1.0},
        {'arm': 'c', 'position': 2, 'value': 1.0},
        {'arm': 'd', 'position': 3, 'value': 1.0},
    ]
    report = order_drift_report(observations, permutations=100)
    assert report['status'] == 'insufficient_data'
    assert report['excluded']['singleton_arm'] == 4

    # Two runs in each of three arms is enough to attempt a test; the
    # singleton arm is excluded rather than contributing a zero residual.
    observations += [
        {'arm': 'a', 'position': 4, 'value': 1.1},
        {'arm': 'b', 'position': 5, 'value': 0.9},
        {'arm': 'c', 'position': 6, 'value': 1.0},
    ]
    report = order_drift_report(observations, permutations=100)
    assert report['status'] == 'tested'
    assert report['excluded']['singleton_arm'] == 1


# ---------------------------------------------------------------------------
# Pre-flight readiness audit
# ---------------------------------------------------------------------------
def _ready_schedule(rows=('a', 'b'), seeds=('42', '43', '44')):
    return randomized_complete_block_schedule(list(rows), list(seeds), seed=1)


def test_design_readiness_passes_a_complete_well_powered_design():
    rows = {'a': {}, 'b': {}}
    hypotheses = {'a': 'h', 'b': 'h'}
    schedule = _ready_schedule()
    report = design_readiness_audit(
        SEEDS, rows, hypotheses, schedule=schedule,
        expected_units=schedule['order'], margin=0.05,
        assumed_paired_sd=0.01, identity_controls=[('a', 'b')],
        schedule_arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    assert report['ready'] is True
    assert report['n_fail'] == 0
    assert report['n_warn'] == 0
    by_id = {check['id']: check for check in report['checks']}
    assert by_id['power']['status'] == 'pass'
    assert by_id['schedule']['status'] == 'pass'
    assert by_id['controls']['status'] == 'pass'


def test_design_readiness_fails_missing_hypothesis_replication_and_schedule():
    rows = {'a': {}, 'b': {}}
    schedule = _ready_schedule()
    report = design_readiness_audit(
        [42, 42], rows, {'a': 'h'}, schedule=schedule,
        expected_units=['only-one-unit'])
    assert report['ready'] is False
    failed = {check['id'] for check in report['checks']
              if check['status'] == 'fail'}
    assert {'replication', 'hypotheses', 'schedule'} <= failed


def test_design_readiness_warns_but_does_not_fail_without_a_smallest_effect():
    rows = {'a': {}, 'b': {}}
    schedule = _ready_schedule()
    report = design_readiness_audit(
        SEEDS, rows, {'a': 'h', 'b': 'h'}, schedule=schedule,
        expected_units=schedule['order'], margin=0.0,
        equivalence_margin=0.0, identity_controls=[('a', 'b')])
    by_id = {check['id']: check for check in report['checks']}
    assert by_id['effect_size']['status'] == 'warn'
    assert report['ready'] is True


def test_design_readiness_fails_an_underpowered_plan():
    rows = {'a': {}, 'b': {}}
    schedule = _ready_schedule()
    report = design_readiness_audit(
        SEEDS, rows, {'a': 'h', 'b': 'h'}, schedule=schedule,
        expected_units=schedule['order'], margin=0.01,
        assumed_paired_sd=0.20, identity_controls=[('a', 'b')])
    by_id = {check['id']: check for check in report['checks']}
    assert by_id['power']['status'] == 'fail'
    assert report['ready'] is False


# ---------------------------------------------------------------------------
# Simulation calibration of the analysis pipeline
# ---------------------------------------------------------------------------
def test_simulated_null_keeps_type_i_error_near_alpha_and_power_rises():
    null = simulate_paired_design(n_pairs=3, effect=0.0, paired_sd=0.05,
                                  reps=1500, seed=21)
    assert null['empirical_type_i_error'] is not None
    assert null['empirical_type_i_error'] <= 0.08
    assert null['empirical_power'] is None
    assert null['undefined_p_rate'] == 0.0

    alternative = simulate_paired_design(n_pairs=3, effect=-0.20,
                                         paired_sd=0.05, reps=1500, seed=22)
    assert alternative['empirical_power'] > 0.8
    assert alternative['empirical_type_i_error'] is None
    assert alternative['mean_estimate'] < 0.0
    assert abs(alternative['mean_bias']) < 0.01


def test_simulated_equivalence_reports_the_tost_rate():
    out = simulate_paired_design(n_pairs=3, effect=0.0, paired_sd=0.02,
                                 reps=500, equivalence_margin=0.10, seed=23)
    assert out['empirical_equivalence_rate'] is not None
    assert out['empirical_equivalence_rate'] > 0.8


@pytest.mark.parametrize('kwargs', [
    {'n_pairs': 1},
    {'n_pairs': 3, 'reps': 10},
    {'n_pairs': 3, 'paired_sd': -0.1},
    {'n_pairs': 3, 'effect': float('nan')},
    {'n_pairs': 3, 'alpha': 1.0},
    {'n_pairs': 3, 'equivalence_margin': -0.1},
])
def test_simulated_design_validates_inputs(kwargs):
    with pytest.raises(ValueError):
        simulate_paired_design(**kwargs)


# ---------------------------------------------------------------------------
# Pre-registration and the run summaries
# ---------------------------------------------------------------------------
def _plan(schedule):
    rows = {'a': {'norm_name': 'dyt'}, 'b': {'norm_name': 'dytcd'}}
    hypotheses = {'a': 'h', 'b': 'h'}
    report = {'leave_one_out': {}, 'factorial': {}, 'qk_axis': {}}
    return build_analysis_plan(rows, hypotheses, report, ['a'], 'gpt2-124m',
                               SEEDS, equivalence_margin=0.02,
                               run_schedule=schedule)


def test_analysis_plan_fingerprint_includes_the_randomized_schedule():
    first = _ready_schedule()
    second = randomized_complete_block_schedule(['a', 'b'],
                                                ['42', '43', '44'], seed=2)
    plan_a = _plan(first)
    plan_b = _plan(second)
    assert plan_a['run_schedule']['fingerprint'] == first['fingerprint']
    assert plan_a['fingerprint'] != plan_b['fingerprint']
    assert _plan(first)['fingerprint'] == plan_a['fingerprint']


def _seed_summary(name, seed, value, position):
    return {
        'name': name, 'seed': seed, 'val_loss': value,
        'val_losses': [value], 'final_loss': value,
        'mean_last_100_loss': value, 'n_loss_spikes': 0,
        'elapsed_s': 1.0, 'tokens_processed': 100,
        'tokens_per_second': 100.0, 'peak_device_memory_mb': None,
        'run_order': {'position': position, 'block': position // 3,
                      'run_id': f'{name}::rep={seed}', 'randomized': True,
                      'schedule_fingerprint': 'abc', 'schedule_seed': 0},
    }


def test_aggregate_and_order_audit_carry_the_run_positions():
    summaries = []
    position = 0
    for name, base in (('a', 2.0), ('b', 2.5), ('c', 3.0)):
        per_seed = []
        for seed in SEEDS:
            per_seed.append(_seed_summary(name, seed, base + seed / 1000.0,
                                          position))
            position += 1
        summaries.append(aggregate_seed_summaries(name, per_seed))
    by_name = {row['name']: row for row in summaries}
    assert set(by_name['a']['run_order_by_seed']) == {'42', '43', '44'}
    audit = run_order_audit(summaries)
    assert audit['status'] == 'tested'
    assert audit['n_rows_with_order'] == 3
    assert audit['n_rows_without_order'] == 0
    assert audit['n_observations'] == 9


def test_order_audit_skips_alias_rows_and_reports_missing_order():
    summaries = [
        aggregate_seed_summaries('a', [_seed_summary('a', seed, 2.0 + seed / 1000,
                                                      i)
                                       for i, seed in enumerate(SEEDS)]),
        {'name': 'alias', 'reused_from': 'a', 'per_seed': [
            {'seed': seed, 'val_loss': 2.0,
             'run_order': {'position': i, 'block': 0}}
            for i, seed in enumerate(SEEDS)]},
        aggregate_seed_summaries('b', [_seed_summary('b', seed, 2.5 + seed / 1000,
                                                      i + 3)
                                       for i, seed in enumerate(SEEDS)]),
        {'name': 'legacy', 'per_seed': [
            {'seed': seed, 'val_loss': 3.0} for seed in SEEDS]},
    ]
    audit = run_order_audit(summaries)
    assert audit['status'] == 'tested'
    assert audit['n_rows_with_order'] == 2
    assert audit['n_rows_without_order'] == 1
