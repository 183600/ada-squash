"""Methodology checks for comparison experiments and ablations.

These tests pin the properties the ablation report now relies on:

* equivalence (TOST) is *not* the same as "the superiority test was not
  significant", and refuses to run without a positive margin;
* the leave-one-seed-out jackknife exposes an effect carried by one run;
* a 2x2 factorial reports a two-sided interaction (antagonism included), not
  only the one-sided synergy p-value;
* the pre-registered analysis plan is stable and only its contrasts enter the
  study-level gatekeeping correction, so an unregistered row cannot become a
  confirmed claim after the fact.

They are pure-python / small synthetic summaries, so they stay fast and do
not depend on a trained checkpoint.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.metrics import (leave_one_out_jackknife, tost_paired,
                               two_sided_p_from_t)
from scripts.ablation import (baseline_comparison, build_analysis_plan,
                              factorial_comparison,
                              study_level_correction)

SEEDS = [42, 43, 44]


def _row(name, values, **extra):
    """A minimal seed-aggregate summary in the shape the comparisons expect."""
    items = sorted(values.items())
    vals = [value for _, value in items]
    mean = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    row = {
        'name': name,
        'val_loss': mean,
        'per_seed': [{'seed': seed, 'val_loss': value} for seed, value in items],
        'n_seeds': len(items),
        'val_loss_seed_std': std,
        'val_loss_seed_stderr': std / (len(items) ** 0.5),
    }
    row.update(extra)
    return row


def _const(name, value, **extra):
    return _row(name, {seed: value for seed in SEEDS}, **extra)


# ---------------------------------------------------------------------------
# TOST equivalence
# ---------------------------------------------------------------------------
def test_tost_refuses_undefined_requests():
    a = [1.0, 1.0, 1.0]
    b = [1.0, 1.0, 1.0]
    for margin in (0.0, -0.1):
        out = tost_paired(a, b, margin)
        assert out['equivalent'] is None
        assert out['p_tost'] is None
        assert 'positive' in out['note']
    out = tost_paired([1.0], [1.0], 0.1)
    assert out['equivalent'] is None
    assert '2 paired' in out['note']


def test_tost_distinguishes_equivalence_from_non_significance():
    # Small, consistent differences: equivalent within a 0.02 margin.
    close = tost_paired([1.000, 1.005, 0.998], [1.0, 1.0, 1.0], margin=0.02)
    assert close['equivalent'] is True
    assert close['p_tost'] < 0.05
    assert close['ci_low'] > -0.02 and close['ci_high'] < 0.02
    # A real difference is not equivalent, even though it is small in absolute
    # terms: the margin is the scientific claim, not a rescaled p-value.
    far = tost_paired([1.2, 1.21, 1.19], [1.0, 1.0, 1.0], margin=0.02)
    assert far['equivalent'] is False
    assert far['p_tost'] > 0.05


def test_tost_zero_variance_is_deterministic():
    same = tost_paired([1.0] * 3, [1.0] * 3, margin=0.01)
    assert same['equivalent'] is True
    assert same['p_tost'] == 0.0
    shifted = tost_paired([1.1] * 3, [1.0] * 3, margin=0.01)
    assert shifted['equivalent'] is False
    assert shifted['p_tost'] == 1.0


def test_two_sided_p_from_t_is_symmetric_and_bounded():
    assert two_sided_p_from_t(0.0, 2.0) == pytest.approx(1.0)
    assert two_sided_p_from_t(3.0, 2.0) == pytest.approx(
        two_sided_p_from_t(-3.0, 2.0))
    assert 0.0 < two_sided_p_from_t(3.0, 2.0) < 1.0
    assert two_sided_p_from_t(float('inf'), 2.0) == 0.0
    assert two_sided_p_from_t(float('-inf'), 2.0) == 0.0
    assert two_sided_p_from_t(None, 2.0) is None
    assert two_sided_p_from_t(1.0, 0.0) is None


# ---------------------------------------------------------------------------
# Jackknife robustness
# ---------------------------------------------------------------------------
def test_jackknife_reports_a_stable_sign():
    out = leave_one_out_jackknife([-0.10, -0.11, -0.09], labels=['42', '43', '44'])
    assert out['sign'] == 'negative'
    assert out['sign_stable'] is True
    assert out['influential_labels'] == []
    assert out['meaningful'] is True
    assert {item['dropped'] for item in out['per_drop']} == {'42', '43', '44'}


def test_jackknife_flags_the_seed_carrying_the_effect():
    # Two mildly negative seeds and one large positive seed: the full-sample
    # mean is positive, but the negative evidence is the two consistent runs.
    out = leave_one_out_jackknife([-0.01, -0.02, 0.50], labels=[42, 43, 44])
    assert out['sign'] == 'positive'
    assert out['sign_stable'] is False
    assert out['influential_labels'] == [44]
    dropped_44 = [item for item in out['per_drop'] if item['dropped'] == 44][0]
    assert dropped_44['mean_without'] < 0.0


def test_jackknife_marks_two_observations_as_not_meaningful():
    out = leave_one_out_jackknife([-0.1, 0.2], labels=['a', 'b'])
    assert out['n'] == 2
    assert out['meaningful'] is False
    assert out['sign_stable'] is False


# ---------------------------------------------------------------------------
# Comparison reports carry the new blocks
# ---------------------------------------------------------------------------
def test_baseline_comparison_adds_equivalence_and_jackknife():
    baseline = _const('ln_baseline', 2.0)
    candidate = _row('dytcd_full', {42: 1.5, 43: 1.52, 44: 1.49})
    near = _row('dytc_near_baseline', {42: 2.001, 43: 1.999, 44: 2.000})
    out = baseline_comparison([baseline, candidate, near], ['ln_baseline'],
                              equivalence_margin=0.02)
    by_name = {entry['name']: entry for entry in out['rows']}
    winner = by_name['dytcd_full']
    assert winner['beats_reference_significantly'] is True
    assert winner['seed_jackknife']['sign_stable'] is True
    # A large superiority difference is *not* equivalent within a tiny margin.
    assert winner['equivalence']['enabled'] is True
    assert winner['equivalence']['equivalent'] is False
    # A row that really matches the baseline can be called equivalent, which
    # "not significant" alone would never establish.
    assert by_name['dytc_near_baseline']['equivalence']['equivalent'] is True
    assert out['equivalence_margin'] == 0.02


def test_leave_one_out_marks_an_explicit_default_as_equivalent():
    from scripts.ablation import leave_one_out_comparison

    full = _const('dytcd_full', 1.50)
    explicit = _row('dytcd_ls1e5', {42: 1.501, 43: 1.499, 44: 1.500})
    spec = {'dytcd_full': {
        'layerscale_explicit_vs_default': {
            'row': 'dytcd_ls1e5', 'diff': ('layerscale_init',)}}}
    out = leave_one_out_comparison([full, explicit], spec=spec,
                                   equivalence_margin=0.02)
    entry = [row for row in out['rows']
             if row['mechanism'] == 'layerscale_explicit_vs_default'][0]
    assert entry['equivalence']['enabled'] is True
    assert entry['equivalence']['equivalent'] is True
    # The full-sample mean is exactly zero, so there is no sign to stabilise;
    # the jackknife says so rather than pretending the near-zero deltas have
    # a direction.
    assert entry['seed_jackknife']['sign_stable'] is None
    assert entry['seed_jackknife']['influential_labels'] == []


def test_leave_one_out_comparison_reports_robustness():
    from scripts.ablation import leave_one_out_comparison

    full = _row('dytcd_full', {42: 1.40, 43: 1.41, 44: 1.39})
    removed = _row('dytcd_no_perchannel', {42: 1.55, 43: 1.54, 44: 1.56})
    out = leave_one_out_comparison([full, removed], equivalence_margin=0.01)
    entry = [row for row in out['rows']
             if row['mechanism'] == 'per_channel_delta'][0]
    assert entry['mechanism_helps_significantly'] is True
    assert entry['seed_jackknife']['meaningful'] is True
    assert entry['equivalence']['enabled'] is True
    # Removing the mechanism made the model clearly worse, so the two arms
    # are not equivalent within a 0.01 loss margin.
    assert entry['equivalence']['equivalent'] is False


# ---------------------------------------------------------------------------
# Factorial interaction: two-sided, both directions
# ---------------------------------------------------------------------------
def _factorial_spec(design='per_channel_x_depth_softsign'):
    return {
        design: {
            'context': 'synthetic',
            'cells': {'neither': 'neither', 'per_channel': 'per_channel',
                      'depth': 'depth', 'both': 'both'},
            'factors': {
                'per_channel': {
                    'diff': ('learnable_delta',),
                    'pairs': (('neither', 'per_channel'),
                              ('depth', 'both')),
                },
                'depth': {
                    'diff': ('k_init', 'learnable_k'),
                    'pairs': (('neither', 'depth'),
                              ('per_channel', 'both')),
                },
            },
        },
    }


def _factorial_cells(both_value):
    return [
        _const('neither', 1.80),
        _const('per_channel', 1.70),
        _const('depth', 1.60),
        _const('both', both_value),
    ]


def test_factorial_reports_synergy_as_a_two_sided_interaction():
    # per_channel effect = -0.10 (flat) / -0.20 (depth on); depth effect =
    # -0.20 (flat) / -0.30 (per-channel on): interaction = -0.10 (synergy).
    out = factorial_comparison(_factorial_cells(1.40), spec=_factorial_spec())
    interaction = [row for row in out['rows']
                   if row['kind'] == 'interaction'][0]
    assert interaction['interaction_direction'] == 'synergy'
    assert interaction['p_two_sided'] is not None
    assert interaction['interaction_differs_significantly'] is True


def test_factorial_reports_antagonism_instead_of_hiding_it():
    # Making the "both" cell worse than the additive prediction yields a
    # positive interaction; the one-sided synergy test must not call it a
    # success, while the two-sided interaction test still detects it.
    out = factorial_comparison(_factorial_cells(1.60), spec=_factorial_spec())
    interaction = [row for row in out['rows']
                   if row['kind'] == 'interaction'][0]
    assert interaction['interaction_direction'] == 'antagonism'
    assert interaction['p_two_sided'] is not None
    assert interaction['interaction_differs_significantly'] is True
    assert interaction['effect_helps'] is False
    assert interaction['effect_helps_significantly'] is False


# ---------------------------------------------------------------------------
# Pre-registered plan + study-level gatekeeping
# ---------------------------------------------------------------------------
def _plan_and_comparison(primary_values, secondary_value, extra_row=None):
    rows = {
        'ln_baseline': {'norm_name': 'ln'},
        'dytcd_full': {'norm_name': 'dytcd'},
        'dytcd_no_perchannel': {'norm_name': 'dytcd', 'learnable_delta': False},
    }
    hypotheses = {name: f'hypothesis for {name}' for name in rows}
    matrix_report = {
        'leave_one_out': {'dytcd_full': {
            'per_channel_delta': 'dytcd_no_perchannel'}},
        'factorial': {},
        'qk_axis': {},
    }
    plan = build_analysis_plan(rows, hypotheses, matrix_report,
                               ['ln_baseline'], 'gpt2-124m', SEEDS,
                               alpha=0.05, margin=0.0,
                               equivalence_margin=0.02)
    summaries = [_const('ln_baseline', 2.0),
                 _row('dytcd_full', primary_values),
                 _const('dytcd_no_perchannel', secondary_value)]
    if extra_row is not None:
        summaries.append(extra_row)
    comparison = baseline_comparison(summaries, ['ln_baseline'],
                                     equivalence_margin=0.02)
    return plan, comparison


def test_plan_is_stable_and_tiers_the_main_proposal_as_primary():
    plan_a, _ = _plan_and_comparison({42: 1.5, 43: 1.5, 44: 1.5}, 1.6)
    plan_b, _ = _plan_and_comparison({42: 1.5, 43: 1.5, 44: 1.5}, 1.6)
    assert plan_a['fingerprint'] == plan_b['fingerprint']
    by_id = {contrast['id']: contrast for contrast in plan_a['contrasts']}
    assert by_id['baseline:dytcd_full']['tier'] == 1
    assert by_id['loo:dytcd_full:per_channel_delta']['tier'] == 1
    assert by_id['baseline:dytcd_no_perchannel']['tier'] == 2


def test_plan_fingerprint_changes_with_the_scientific_margin():
    rows = {'ln_baseline': {'norm_name': 'ln'},
            'dytcd_full': {'norm_name': 'dytcd'}}
    hyps = {name: 'h' for name in rows}
    report = {'leave_one_out': {}, 'factorial': {}, 'qk_axis': {}}
    a = build_analysis_plan(rows, hyps, report, ['ln_baseline'], 'm', SEEDS,
                            equivalence_margin=0.01)
    b = build_analysis_plan(rows, hyps, report, ['ln_baseline'], 'm', SEEDS,
                            equivalence_margin=0.02)
    assert a['fingerprint'] != b['fingerprint']


def test_study_level_gate_opens_when_the_primary_survives():
    plan, comparison = _plan_and_comparison({42: 1.5, 43: 1.5, 44: 1.5}, 1.6)
    study = study_level_correction({'baseline': comparison}, plan)
    assert study['primary_gate_open'] is True
    assert study['n_confirmed_primary'] >= 1
    confirmed = {row['id'] for row in study['rows']
                 if row['confirmed_at_study_level']}
    assert 'baseline:dytcd_full' in confirmed
    # The secondary row is constant and far from the baseline, so once the
    # gate opens its own Holm-corrected test confirms it too.
    assert 'baseline:dytcd_no_perchannel' in confirmed


def test_study_level_gate_closes_and_blocks_secondary_claims():
    # A noisy primary row: the point estimate is better than the baseline,
    # but the paired t-test cannot certify it. The secondary row is a clean
    # winner; gatekeeping still refuses to promote it.
    plan, comparison = _plan_and_comparison(
        {42: 1.50, 43: 2.20, 44: 1.60}, 1.60)
    study = study_level_correction({'baseline': comparison}, plan)
    assert study['primary_gate_open'] is False
    primary = [row for row in study['rows'] if row['tier'] == 'primary']
    assert all(row['confirmed_at_study_level'] is False for row in primary)
    secondary = [row for row in study['rows']
                 if row['id'] == 'baseline:dytcd_no_perchannel'][0]
    assert secondary['confirmed_at_study_level'] is False
    assert secondary['p_study_holm'] is None
    assert 'not tested' in secondary['gate_status']


def test_unregistered_rows_cannot_become_confirmed_claims():
    sneaky = _const('dytcd_sneaky', 1.0)
    plan, comparison = _plan_and_comparison(
        {42: 1.5, 43: 1.5, 44: 1.5}, 1.6, extra_row=sneaky)
    study = study_level_correction({'baseline': comparison}, plan)
    assert 'baseline:dytcd_sneaky' in study['unregistered_contrasts']
    row = [entry for entry in study['rows']
           if entry['id'] == 'baseline:dytcd_sneaky'][0]
    assert row['tier'] == 'unregistered'
    assert row['confirmed_at_study_level'] is False
    assert 'pre-registered analysis plan' in row['excluded_from_study_family']


def test_unresolved_plan_contrasts_are_reported_not_silently_dropped():
    plan, comparison = _plan_and_comparison({42: 1.5, 43: 1.5, 44: 1.5}, 1.6)
    # The plan also contains the leave-one-out contrast, but no LOO report is
    # supplied here: it must show up as unresolved, not disappear.
    study = study_level_correction({'baseline': comparison}, plan)
    assert 'loo:dytcd_full:per_channel_delta' in study['unresolved_plan_contrasts']


def test_global_correction_can_demote_a_locally_significant_claim():
    # Three primary contrasts each pass their own family-local alpha = 0.05,
    # but the study-level Holm correction multiplies the smallest p by 3 and
    # no primary claim survives. The gate must then close, not quietly keep
    # the family-local verdicts.
    plan = {
        'fingerprint': 'test-plan',
        'contrasts': [
            {'id': f'baseline:row{i}', 'family': 'baseline',
             'kind': 'superiority', 'tier': 1, 'tier_role': 'primary'}
            for i in range(3)
        ],
    }
    rows = [{
        'name': f'row{i}',
        'p_one_sided': 0.03,
        'p_one_sided_holm': 0.03,
        'beats_reference': True,
        'beats_reference_significantly': True,
        'replicated_across_seeds': True,
        'excluded_from_family': None,
        'delta_vs_reference': -0.1,
        'test_basis': 'seed_means',
        'hypothesis': 'h',
    } for i in range(3)]
    study = study_level_correction({'baseline': {'rows': rows}}, plan)
    assert study['primary_gate_open'] is False
    assert study['n_confirmed_at_study_level'] == 0
    assert {entry['id'] for entry in study['downgraded_by_study_level']} == {
        f'baseline:row{i}' for i in range(3)}


def test_focused_secondary_only_run_is_not_blocked_by_a_missing_primary_family():
    # A QK-only or single-axis run has no primary tier at all. That is not a
    # failed gate - the secondary family should still be corrected and tested.
    plan, _ = _plan_and_comparison({42: 1.5, 43: 1.5, 44: 1.5}, 1.6)
    summaries = [_const('ln_baseline', 2.0),
                 _const('dytcd_no_perchannel', 1.6)]
    comparison = baseline_comparison(summaries, ['ln_baseline'],
                                     equivalence_margin=0.02)
    study = study_level_correction({'baseline': comparison}, plan)
    assert study['primary_gate_status'] == 'not_applicable_no_primary_tests'
    secondary = [row for row in study['rows']
                 if row['id'] == 'baseline:dytcd_no_perchannel'][0]
    assert secondary['p_study_holm'] is not None
    assert secondary['confirmed_at_study_level'] is True


# ---------------------------------------------------------------------------
# Phase-1 arm comparison gets the same two robustness blocks
# ---------------------------------------------------------------------------
def test_compare_arms_reports_equivalence_and_jackknife():
    from scripts.compare_arms import compare_arms

    def aggregate(values):
        return {
            'norm': 'test',
            'model': 'gpt2-124m',
            'per_seed': [{'seed': seed, 'final_val_loss': value,
                          'final_val_step': 100}
                         for seed, value in sorted(values.items())],
            'shapes_by_seed': {str(seed): {'n_layers': 12, 'dim': 768,
                                           'n_heads': 12}
                               for seed in values},
            'data_versions_by_seed': {str(seed): {'name': 'corpus',
                                                  'revision': 'v1'}
                                      for seed in values},
        }

    arms = {
        'reference': aggregate({42: 2.0, 43: 2.0, 44: 2.0}),
        'candidate': aggregate({42: 2.001, 43: 1.999, 44: 2.000}),
    }
    out = compare_arms(arms, reference='reference', equivalence_margin=0.02,
                       require_recipe=False)
    row = out['rows'][0]
    assert row['equivalence']['enabled'] is True
    assert row['equivalence']['equivalent'] is True
    # Small, sign-alternating differences: no stable direction to report.
    assert row['seed_jackknife']['sign'] == 'zero'
    assert row['seed_jackknife']['sign_stable'] is None

    clear = {
        'reference': aggregate({42: 2.0, 43: 2.0, 44: 2.0}),
        'candidate': aggregate({42: 1.5, 43: 1.52, 44: 1.49}),
    }
    out = compare_arms(clear, reference='reference', equivalence_margin=0.02,
                       require_recipe=False)
    row = out['rows'][0]
    assert row['equivalence']['equivalent'] is False
    assert row['seed_jackknife']['sign_stable'] is True
