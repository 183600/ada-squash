"""Pre-specified run exclusions and attrition accounting.

A comparison is only as good as its denominator. These tests pin the design
layer that decides, before outcomes are seen, which runs may leave the
analysis and then audits every planned run against that rule:

* the default policy admits only protocol failures (missing run, recipe
  mismatch, non-finite loss) and never a finite-but-unlucky seed;
* an outcome-dependent exclude rule is rejected instead of being allowed to
  drop bad results after the fact;
* a run whose curve diverged but whose loss is finite is retained with a
  sensitivity flag, not excluded;
* an arm left below the replication minimum after exclusions fails closed;
* differential attrition across arms is a warning even when each exclusion
  is individually allowed;
* the policy fingerprint is part of the analysis plan, so changing what may
  be dropped changes the pre-registration;
* the row aggregate keeps excluded seeds visible instead of only shrinking
  ``n_seeds``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.design import (classify_run_exclusions, default_exclusion_policy,
                              design_readiness_audit,
                              exclusion_policy_fingerprint,
                              normalize_exclusion_policy,
                              randomized_complete_block_schedule,
                              run_exclusion_audit)
from scripts.ablation import (aggregate_seed_summaries, baseline_comparison,
                              build_analysis_plan)

SEEDS = [42, 43, 44]


def _run(run_id, arm, seed, loss, **extra):
    record = {'run_id': run_id, 'arm': arm, 'seed': seed, 'val_loss': loss}
    record.update(extra)
    return record


def test_default_policy_is_stable_and_rejects_outcome_dependent_exclusions():
    policy = default_exclusion_policy()
    canonical = normalize_exclusion_policy(policy)
    assert canonical['schema'] == 'adasquash.exclusion_policy.v1'
    assert {rule['id'] for rule in canonical['rules']} >= {
        'missing_run', 'protocol_mismatch', 'non_finite_loss'}
    assert exclusion_policy_fingerprint(policy) == exclusion_policy_fingerprint(canonical)
    # Reordering rules does not change the meaning or the fingerprint.
    shuffled = dict(policy, rules=list(reversed(policy['rules'])))
    assert exclusion_policy_fingerprint(shuffled) == exclusion_policy_fingerprint(policy)

    bad = dict(policy, rules=list(policy['rules']) + [{
        'id': 'worst_seed', 'action': 'exclude', 'outcome_blind': False,
        'description': 'drop the seed with the largest loss',
    }])
    with pytest.raises(ValueError, match='outcome-dependent'):
        normalize_exclusion_policy(bad)

    missing = dict(policy, rules=[rule for rule in policy['rules']
                                  if rule['id'] != 'non_finite_loss'])
    with pytest.raises(ValueError, match='mandatory protocol rule'):
        normalize_exclusion_policy(missing)

    disguised = dict(policy, rules=list(policy['rules']) + [{
        'id': 'val_loss_outlier', 'action': 'exclude', 'outcome_blind': True,
        'description': 'also drop a finite loss just above the rest',
    }])
    with pytest.raises(ValueError, match='forbidden'):
        normalize_exclusion_policy(disguised)


def test_classification_separates_protocol_failures_from_unlucky_curves():
    policy = default_exclusion_policy()
    runs = [
        _run('a::1', 'a', 1, 2.00),
        _run('a::2', 'a', 2, float('nan')),
        _run('a::3', 'a', 3, 2.05, protocol_mismatch=True),
        _run('b::1', 'b', 1, 2.10, curve_classification='diverging'),
        _run('b::2', 'b', 2, None, present=False),
    ]
    classified = {item['run_id']: item for item in classify_run_exclusions(runs, policy)}
    assert classified['a::1']['status'] == 'analyzed'
    assert classified['a::2']['status'] == 'excluded'
    assert 'non_finite_loss' in classified['a::2']['reasons']
    assert classified['a::3']['status'] == 'excluded'
    assert 'protocol_mismatch' in classified['a::3']['reasons']
    # A diverged curve is a sensitivity flag, not a licence to drop a finite run.
    assert classified['b::1']['status'] == 'sensitivity'
    assert classified['b::1']['val_loss'] == pytest.approx(2.10)
    assert classified['b::2']['status'] == 'excluded'
    assert 'missing_run' in classified['b::2']['reasons']


def test_clean_audit_has_equal_planned_and_analyzed_counts():
    runs = [_run(f'{arm}::{seed}', arm, seed, 2.0 + seed / 100.0)
            for arm in ('a', 'b') for seed in SEEDS]
    audit = run_exclusion_audit(runs)
    assert audit['ready'] is True
    assert audit['n_planned'] == audit['n_analyzed'] == 6
    assert audit['n_excluded'] == 0
    assert audit['warnings'] == []
    assert all(bucket['planned'] == bucket['analyzed'] == 3
               for bucket in audit['per_arm'].values())


def test_audit_fails_closed_when_exclusions_leave_too_few_pairs():
    runs = [
        _run('a::42', 'a', 42, 2.0),
        _run('a::43', 'a', 43, float('nan')),
        _run('a::44', 'a', 44, None, present=False),
        _run('b::42', 'b', 42, 2.1),
        _run('b::43', 'b', 43, 2.2),
        _run('b::44', 'b', 44, 2.3),
    ]
    audit = run_exclusion_audit(runs)
    assert audit['ready'] is False
    assert audit['n_planned'] == 6
    assert audit['n_analyzed'] == 4
    assert audit['n_excluded'] == 2
    assert audit['per_arm']['a']['n_seeds_analyzed'] == 1
    assert any('below the certification minimum' in item
               for item in audit['violations'])


def test_differential_attrition_is_a_warning_even_when_each_drop_is_allowed():
    runs = [
        _run('a::42', 'a', 42, 2.0),
        _run('a::43', 'a', 43, float('nan')),
        _run('a::44', 'a', 44, None, present=False),
        _run('b::42', 'b', 42, 2.1),
        _run('b::43', 'b', 43, 2.2),
        _run('b::44', 'b', 44, 2.3),
    ]
    audit = run_exclusion_audit(runs)
    assert any('differential attrition' in item for item in audit['warnings'])
    assert audit['attrition_balanced'] is False


def test_duplicate_run_and_seed_are_structural_failures():
    runs = [
        _run('a::42', 'a', 42, 2.0),
        _run('a::42', 'a', 42, 2.0),
        _run('a::43', 'a', 43, 2.1),
        _run('a::44', 'a', 44, 2.2),
    ]
    audit = run_exclusion_audit(runs)
    assert audit['ready'] is False
    joined = '; '.join(audit['violations'])
    assert 'duplicate run id' in joined
    assert 'duplicate seed' in joined


def test_analysis_plan_fingerprint_covers_the_exclusion_policy():
    rows = {'ln_baseline': {'norm_name': 'ln'},
            'dytcd_full': {'norm_name': 'dytcd'}}
    hypotheses = {name: 'h' for name in rows}
    matrix_report = {'leave_one_out': {}, 'factorial': {}, 'qk_axis': {}}
    first = build_analysis_plan(rows, hypotheses, matrix_report,
                                ['ln_baseline'], 'gpt2-124m', SEEDS)
    changed_policy = default_exclusion_policy()
    changed_policy['rules'].append({
        'id': 'underfit_curve', 'action': 'sensitivity', 'outcome_blind': False,
        'description': 'retain but flag a still-underfitting run',
    })
    second = build_analysis_plan(rows, hypotheses, matrix_report,
                                 ['ln_baseline'], 'gpt2-124m', SEEDS,
                                 exclusion_policy=changed_policy)
    assert first['exclusion_policy_fingerprint'] != second['exclusion_policy_fingerprint']
    assert first['fingerprint'] != second['fingerprint']
    assert first['exclusion_policy']['schema'] == 'adasquash.exclusion_policy.v1'


def test_aggregate_keeps_excluded_seeds_in_the_per_seed_table():
    summaries = [
        {'seed': 42, 'val_loss': 2.0, 'val_losses': [2.0]},
        {'seed': 43, 'val_loss': 2.2, 'val_losses': [2.2]},
        {'seed': 44, 'val_loss': float('nan'), 'val_losses': [float('nan')]},
    ]
    agg = aggregate_seed_summaries('dytcd_full', summaries)
    assert agg['n_seeds'] == 2
    assert len(agg['per_seed']) == 3
    assert agg['val_loss'] == pytest.approx(2.1)
    assert agg['excluded_seeds'] == ['44']
    assert agg['n_seeds_excluded'] == 1
    assert agg['exclusion_status_by_seed']['42'] == 'analyzed'
    assert agg['exclusion_status_by_seed']['44'] == 'excluded'
    assert 'non_finite_loss' in agg['exclusion_reasons_by_seed']['44']


def test_design_readiness_records_the_exclusion_policy():
    rows = {'a': {}, 'b': {}}
    hypotheses = {'a': 'h', 'b': 'h'}
    schedule = randomized_complete_block_schedule(
        ['a', 'b'], [str(seed) for seed in SEEDS], seed=1)
    report = design_readiness_audit(
        SEEDS, rows, hypotheses, schedule=schedule,
        expected_units=schedule['order'], margin=0.05,
        assumed_paired_sd=0.01, identity_controls=[('a', 'b')],
        schedule_arm_of=lambda unit: unit.rsplit('::rep=', 1)[0])
    check = {item['id']: item for item in report['checks']}['exclusions']
    assert check['status'] == 'pass'
    assert report['exclusion_policy_fingerprint']
    assert report['ready'] is True
    assert report['n_warn'] == 0

    broken = default_exclusion_policy()
    broken['rules'] = list(broken['rules']) + [{
        'id': 'drop_bad_run', 'action': 'exclude', 'outcome_blind': False,
        'description': 'drop a run after seeing its loss',
    }]
    report = design_readiness_audit(
        SEEDS, rows, hypotheses, schedule=schedule,
        expected_units=schedule['order'], margin=0.05,
        assumed_paired_sd=0.01, identity_controls=[('a', 'b')],
        schedule_arm_of=lambda unit: unit.rsplit('::rep=', 1)[0],
        exclusion_policy=broken)
    check = {item['id']: item for item in report['checks']}['exclusions']
    assert check['status'] == 'fail'
    assert report['ready'] is False


def test_baseline_comparison_surfaces_attrition_next_to_the_delta():
    baseline = aggregate_seed_summaries('ln_baseline', [
        {'seed': seed, 'val_loss': 2.0} for seed in SEEDS])
    candidate = aggregate_seed_summaries('dytcd_full', [
        {'seed': 42, 'val_loss': 1.9},
        {'seed': 43, 'val_loss': 1.95},
        {'seed': 44, 'val_loss': float('nan')},
    ])
    comparison = baseline_comparison([baseline, candidate], ['ln_baseline'])
    entry = {row['name']: row for row in comparison['rows']}['dytcd_full']
    assert entry['n_seeds'] == 2
    assert entry['excluded_seeds'] == ['44']
    assert entry['n_seeds_excluded'] == 1
    assert entry['exclusion_status_by_seed']['44'] == 'excluded'
