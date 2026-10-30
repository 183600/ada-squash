"""Checks for the 2x2x2 cross-axis factorial and tested orthogonality.

The two 2x2 designs in the ablation matrix estimate a per-channel x depth
interaction separately under generalized softsign and under tanh. They cannot
tell whether that interaction is the same in both activation families, and
neither can they support the proposal's "three orthogonal axes" claim: a
non-significant interaction is not evidence of additivity. These tests pin the
new 2x2x2 report:

* the shipped matrix really forms a complete, clean 2x2x2 design;
* the orthogonal contrasts recover known additive effects exactly;
* a real interaction is reported two-sided and the three-way term isolates an
  activation-specific interaction;
* orthogonality is a TOST claim, unavailable without a pre-specified margin;
* the new interaction contrasts are pre-registered and reachable by the
  study-level gatekeeping correction.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation import (THREE_WAY_FACTORIAL_SPEC, build_analysis_plan,
                              check_three_way_factorial,
                              load_ablations_from_config,
                              study_level_correction,
                              three_way_factorial_comparison,
                              validate_matrix)

SEEDS = [42, 43, 44]
LEVELS = {'per_channel': ('off', 'on'), 'depth': ('off', 'on'),
          'activation': ('softsign', 'tanh')}
FACTORS = ('per_channel', 'depth', 'activation')


def _const(name, value):
    return {
        'name': name,
        'val_loss': value,
        'per_seed': [{'seed': seed, 'val_loss': value} for seed in SEEDS],
        'n_seeds': len(SEEDS),
        'val_loss_seed_std': 0.0,
        'val_loss_seed_stderr': 0.0,
    }


def _cell_name(assignment):
    return 'cell_' + '_'.join(assignment[factor] for factor in FACTORS)


def _synthetic_design():
    cells = {
        '|'.join(f'{factor}={assignment[factor]}' for factor in FACTORS):
        _cell_name(assignment)
        for assignment in (
            dict(zip(FACTORS, combo))
            for combo in itertools.product(*(LEVELS[factor] for factor in FACTORS)))
    }
    spec = {'synthetic': {
        'design_type': '2x2x2',
        'context': 'synthetic additive/interaction fixture',
        'factors': FACTORS,
        'levels': LEVELS,
        'cells': cells,
    }}
    return spec, cells


def _row_values(spec, value_of):
    cells = spec['synthetic']['cells']
    rows = []
    for key, name in cells.items():
        assignment = {}
        for part in key.split('|'):
            factor, _, level = part.partition('=')
            assignment[factor] = level
        rows.append(_const(name, value_of(assignment)))
    return rows


def _additive(base=2.0, per_channel=-0.1, depth=-0.2, activation=-0.05):
    def value(assignment):
        return (base
                + (per_channel if assignment['per_channel'] == 'on' else 0.0)
                + (depth if assignment['depth'] == 'on' else 0.0)
                + (activation if assignment['activation'] == 'tanh' else 0.0))
    return value


def _entry(report, effect):
    return [row for row in report['rows'] if row['effect'] == effect][0]


# ---------------------------------------------------------------------------
# The shipped matrix is a complete, clean 2x2x2
# ---------------------------------------------------------------------------
def test_shipped_matrix_forms_a_complete_clean_2x2x2():
    meta = load_ablations_from_config(
        str(ROOT / 'configs' / 'ablation-matrix.yaml'),
        include_metadata=True)
    rows = {name: value['cfg'] for name, value in meta.items()}
    hypotheses = {name: value['hypothesis'] for name, value in meta.items()}
    report = check_three_way_factorial(rows, 'gpt2-124m')
    design = report['per_channel_x_depth_x_activation']
    assert design['design_type'] == '2x2x2'
    assert len(design['cells']) == 8
    assert set(design['switches']) == set(FACTORS)
    for factor in FACTORS:
        # Every factor is switched in all four contexts of the other two.
        assert len(design['switches'][factor]) == 4
        assert all(item['diff'] for item in design['switches'][factor].values())
    # The matrix validation exposes the 2x2x2 report next to the 2x2 reports.
    matrix = validate_matrix(rows, hypotheses, 'gpt2-124m')
    assert 'three_way_factorial' in matrix
    assert matrix['three_way_factorial'] == report


def test_three_way_check_rejects_a_missing_cell():
    meta = load_ablations_from_config(
        str(ROOT / 'configs' / 'ablation-matrix.yaml'),
        include_metadata=True)
    rows = {name: value['cfg'] for name, value in meta.items()}
    rows.pop('dytcd_no_softsign')
    with pytest.raises(ValueError, match='references absent row'):
        check_three_way_factorial(rows, 'gpt2-124m')


def test_three_way_check_rejects_a_confounded_factor_switch():
    meta = load_ablations_from_config(
        str(ROOT / 'configs' / 'ablation-matrix.yaml'),
        include_metadata=True)
    rows = {name: value['cfg'] for name, value in meta.items()}
    # `dytcd_full` is the per-channel-on / depth-on / softsign cell. Turning
    # LayerScale off there would make the per-channel switch at depth-on
    # differ in `learnable_delta` *and* `use_layerscale`, so the interaction
    # would no longer be attributable to the declared factor.
    rows['dytcd_full'] = dict(rows['dytcd_full'], use_layerscale=False)
    with pytest.raises(ValueError, match='expected one of'):
        check_three_way_factorial(rows, 'gpt2-124m')


# ---------------------------------------------------------------------------
# Orthogonal contrasts: additive and interaction models
# ---------------------------------------------------------------------------
def test_three_way_recovers_pure_additive_effects():
    spec, _ = _synthetic_design()
    report = three_way_factorial_comparison(
        _row_values(spec, _additive()), spec=spec, equivalence_margin=0.02)
    per_channel = _entry(report, 'per_channel')
    depth = _entry(report, 'depth')
    assert per_channel['kind'] == 'main_effect'
    assert per_channel['delta_seed_mean'] == pytest.approx(-0.1)
    assert depth['delta_seed_mean'] == pytest.approx(-0.2)
    for effect in ('per_channel_x_depth', 'per_channel_x_activation',
                   'depth_x_activation', 'per_channel_x_depth_x_activation'):
        entry = _entry(report, effect)
        assert entry['kind'] == 'interaction'
        assert entry['delta_seed_mean'] == pytest.approx(0.0)
        assert entry['interaction_direction'] == 'additive_or_within_margin'
        assert entry['orthogonal_within_margin'] is True
        assert entry['interaction_differs_significantly'] is False
    assert report['orthogonality']['axes_orthogonal_within_margin'] is True
    assert (report['orthogonality']
            ['three_way_interaction_absent_within_margin']) is True


def test_three_way_isolates_an_activation_specific_interaction():
    def value(assignment):
        base = _additive()(assignment)
        if (assignment['per_channel'] == 'on'
                and assignment['depth'] == 'on'
                and assignment['activation'] == 'softsign'):
            return base - 0.10
        return base

    spec, _ = _synthetic_design()
    report = three_way_factorial_comparison(
        _row_values(spec, value), spec=spec, equivalence_margin=0.02)
    # The interaction is -0.10 under softsign and 0 under tanh, so the
    # pooled per-channel x depth term is -0.05 and the three-way contrast is
    # the full +0.10 (tanh minus softsign).
    assert _entry(report, 'per_channel_x_depth')['delta_seed_mean'] == pytest.approx(-0.05)
    triple = _entry(report, 'per_channel_x_depth_x_activation')
    assert triple['delta_seed_mean'] == pytest.approx(0.10)
    assert triple['interaction_direction'] == 'positive'
    assert triple['interaction_differs_significantly'] is True
    assert triple['orthogonal_within_margin'] is False
    # Cross-family interactions are nonzero because the per-channel effect
    # differs between activation families in this fixture.
    assert _entry(report, 'per_channel_x_activation')['delta_seed_mean'] == pytest.approx(0.05)
    assert report['orthogonality']['axes_orthogonal_within_margin'] is False


def test_three_way_reports_antagonism_two_sidedly():
    def value(assignment):
        extra = 0.10 if (assignment['per_channel'] == 'on'
                         and assignment['depth'] == 'on') else 0.0
        return _additive()(assignment) + extra

    spec, _ = _synthetic_design()
    report = three_way_factorial_comparison(
        _row_values(spec, value), spec=spec, equivalence_margin=0.02)
    interaction = _entry(report, 'per_channel_x_depth')
    assert interaction['delta_seed_mean'] == pytest.approx(0.10)
    assert interaction['interaction_direction'] == 'positive'
    assert interaction['interaction_differs_significantly'] is True
    # A positive interaction is an antagonism finding, not a failed synergy
    # claim; the one-sided "helps" certification must stay false.
    assert interaction['effect_helps'] is False
    assert interaction['effect_helps_significantly'] is False


def test_simple_effects_are_gated_per_factor_in_the_2x2x2():
    # Only per_channel x depth interacts (and only when both are on). The
    # per_channel and depth gates open; activation has no surviving
    # interaction, so its simple effects remain descriptive.
    def value(assignment):
        extra = 0.10 if (assignment['per_channel'] == 'on'
                         and assignment['depth'] == 'on') else 0.0
        return _additive()(assignment) + extra

    spec, _ = _synthetic_design()
    report = three_way_factorial_comparison(
        _row_values(spec, value), spec=spec, equivalence_margin=0.02)
    simple = {row['effect']: row for row in report['rows']
              if row.get('kind') == 'simple_effect'}
    assert len(simple) == 12
    assert simple['per_channel@depth=off|activation=softsign'][
        'delta_seed_mean'] == pytest.approx(-0.10)
    assert simple['per_channel@depth=on|activation=softsign'][
        'delta_seed_mean'] == pytest.approx(0.0)
    assert simple['depth@per_channel=off|activation=tanh'][
        'delta_seed_mean'] == pytest.approx(-0.20)
    assert simple['depth@per_channel=on|activation=tanh'][
        'delta_seed_mean'] == pytest.approx(-0.10)
    for factor in ('per_channel', 'depth'):
        assert report['conditional_effects']['factors'][
            f'synthetic:{factor}']['gate_open'] is True
    assert (report['conditional_effects']['factors']['synthetic:activation']
            ['gate_open']) is False
    for effect, entry in simple.items():
        if effect.startswith('activation@'):
            assert entry['interaction_gate_open'] is False
            assert entry['conditional_test_status'] == (
                'descriptive_no_interaction')
            assert entry['p_one_sided_holm'] is None
        else:
            assert entry['interaction_gate_open'] is True
            assert entry['conditional_test_status'] == 'tested_follow_up'
            assert entry['p_one_sided_holm'] is not None
    assert report['conditional_effects']['n_tested_follow_up'] == 8
    assert report['conditional_effects']['n_descriptive_gate_closed'] == 4


def test_orthogonality_is_not_inferred_without_an_equivalence_margin():
    spec, _ = _synthetic_design()
    report = three_way_factorial_comparison(
        _row_values(spec, _additive()), spec=spec, equivalence_margin=0.0)
    assert report['orthogonality']['axes_orthogonal_within_margin'] is None
    assert report['orthogonality']['three_way_interaction_absent_within_margin'] is None
    assert 'no positive --equivalence-margin' in report['orthogonality']['note']
    for effect in ('per_channel_x_depth', 'per_channel_x_activation',
                   'depth_x_activation', 'per_channel_x_depth_x_activation'):
        assert _entry(report, effect)['orthogonal_within_margin'] is None


# ---------------------------------------------------------------------------
# Pre-registration and study-level integration
# ---------------------------------------------------------------------------
def test_cross_axis_interactions_are_pre_registered_and_reach_study_level():
    spec, cells = _synthetic_design()
    rows = {name: {'norm_name': 'dytcd'} for name in cells.values()}
    hypotheses = {name: f'hypothesis for {name}' for name in rows}
    report = three_way_factorial_comparison(
        _row_values(spec, _additive()), spec=spec, equivalence_margin=0.02)
    matrix_report = {
        'leave_one_out': {}, 'factorial': {}, 'qk_axis': {},
        'three_way_factorial': {
            'synthetic': {'factors': list(FACTORS), 'cells': cells},
        },
    }
    plan = build_analysis_plan(
        rows, hypotheses, matrix_report, [], 'gpt2-124m', SEEDS,
        equivalence_margin=0.02)
    ids = {entry['id'] for entry in plan['contrasts']}
    for first, second in itertools.combinations(FACTORS, 2):
        assert f'factorial3:synthetic:{first}_x_{second}' in ids
    assert ('factorial3:synthetic:per_channel_x_depth_x_activation'
            in ids)
    # Main effects are already registered by the 2x2 designs; the pooled
    # 2x2x2 main effects must not be double-counted in the study family.
    assert 'factorial3:synthetic:per_channel' not in ids
    assert 'factorial3:synthetic:depth' not in ids
    # Every simple effect is pre-registered as an interaction-gated follow-up,
    # but explicitly not counted in the study-level family.
    simple = [entry for entry in plan['contrasts']
              if entry['kind'] == 'simple_effect']
    assert len(simple) == 12
    for entry in simple:
        assert entry['counted_in_study_family'] is False
        assert entry['tier'] == 2
        assert entry['tier_role'] == 'conditional_follow_up'
        assert entry['family'] == 'conditional_follow_up'

    study = study_level_correction({'factorial3': report}, plan)
    assert study['n_conditional_follow_up'] == 12
    assert study['n_in_study_family'] == 4
    simple_records = [entry for entry in study['rows']
                      if entry['tier'] == 'conditional_follow_up']
    assert len(simple_records) == 12
    assert all(entry['confirmed_at_study_level'] is False
               for entry in simple_records)
    assert all('conditional follow-up' in entry['excluded_from_study_family']
               for entry in simple_records)
    unresolved = set(study['unresolved_plan_contrasts'])
    for first, second in itertools.combinations(FACTORS, 2):
        assert f'factorial3:synthetic:{first}_x_{second}' not in unresolved
    assert ('factorial3:synthetic:per_channel_x_depth_x_activation'
            not in unresolved)
    # The synthetic interactions are exactly zero; they are resolved (tested)
    # but must not be promoted to confirmed findings.
    assert 'factorial3:synthetic:per_channel_x_depth' not in (
        study['study_level_significant'])
