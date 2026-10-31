"""Ordered ablation sweeps: clean coverage, trend and winner reporting.

The shipped matrix contains two ordered hyper-parameter sweeps (the fixed
softsign exponent p and the DyT alpha initialization). Treating those rows as
an unordered bag loses the actual claim ("p=2 is the interior optimum",
"alpha materially changes the run") and makes a post-hoc winner look
confirmatory. These tests pin the structural check, the paired level tests,
the optional monotone trend and the analysis-plan integration.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.ablation import (DOSE_RESPONSE_SPEC, build_analysis_plan,
                              check_dose_response, dose_response_comparison,
                              load_ablations_from_config,
                              study_level_correction, validate_matrix)

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (42, 43, 44)


def _matrix():
    meta = load_ablations_from_config(
        str(ROOT / 'configs' / 'ablation-matrix.yaml'), include_metadata=True)
    rows = {name: dict(value['cfg']) for name, value in meta.items()}
    hypotheses = {name: value['hypothesis'] for name, value in meta.items()}
    return rows, hypotheses


def _row(name, values):
    return {
        'name': name,
        'val_loss': sum(values.values()) / len(values),
        'per_seed': [{'seed': seed, 'val_loss': value}
                     for seed, value in values.items()],
        'seeds': list(values),
        'n_seeds': len(values),
        'val_loss_seed_std': 0.01,
    }


def _summaries():
    return [
        _row('dytc_p2_softsign', {42: 2.00, 43: 2.00, 44: 2.00}),
        _row('dytc_p1_softsign', {42: 2.06, 43: 2.05, 44: 2.07}),
        _row('dytc_p4_softsign', {42: 2.03, 43: 2.04, 44: 2.02}),
        _row('dytc_hardtanh_eq', {42: 2.12, 43: 2.11, 44: 2.13}),
        _row('dyt_alpha_0.5', {42: 1.50, 43: 1.50, 44: 1.50}),
        _row('dyt_alpha_0.3', {42: 1.53, 43: 1.52, 44: 1.54}),
        _row('dyt_alpha_1.0', {42: 1.46, 43: 1.45, 44: 1.47}),
    ]


def test_shipped_sweeps_are_clean_and_expose_the_expected_dose_grid():
    rows, hypotheses = _matrix()
    report = check_dose_response(rows, 'gpt2-124m')
    assert set(report) == {'softsign_p', 'dyt_alpha_init'}
    p_sweep = report['softsign_p']
    assert p_sweep['reference'] == 'dytc_p2_softsign'
    assert [level['dose'] for level in p_sweep['levels']] == [
        1.0, 2.0, 4.0, 64.0]
    assert [level['row'] for level in p_sweep['levels']] == [
        'dytc_p1_softsign', 'dytc_p2_softsign', 'dytc_p4_softsign',
        'dytc_hardtanh_eq']
    assert p_sweep['trend']['alternative'] == 'increasing_loss'
    assert p_sweep['trend']['dose_transform'] == 'log2'
    alpha = report['dyt_alpha_init']
    assert alpha['reference'] == 'dyt_alpha_0.5'
    assert alpha['trend'] is None
    # `validate_matrix` carries the report next to the other design checks.
    matrix = validate_matrix(rows, hypotheses, 'gpt2-124m')
    assert matrix['dose_response'] == report


def test_confounded_alias_duplicate_or_unordered_sweeps_are_rejected():
    rows, _ = _matrix()
    confounded = {name: dict(cfg) for name, cfg in rows.items()}
    confounded['dytc_p4_softsign']['alpha_jitter'] = 0.1
    with pytest.raises(ValueError, match='differ in'):
        check_dose_response(confounded, 'gpt2-124m')

    aliased = {name: dict(cfg) for name, cfg in rows.items()}
    aliased['dytc_p4_softsign'] = dict(aliased['dytc_p1_softsign'])
    with pytest.raises(ValueError, match='alias levels'):
        check_dose_response(aliased, 'gpt2-124m')

    missing = {name: dict(cfg) for name, cfg in rows.items()}
    missing.pop('dytc_p4_softsign')
    with pytest.raises(ValueError, match='absent row'):
        check_dose_response(missing, 'gpt2-124m')

    bad_spec = copy.deepcopy(DOSE_RESPONSE_SPEC)
    bad_spec['softsign_p']['levels'] = tuple(
        reversed(bad_spec['softsign_p']['levels']))
    with pytest.raises(ValueError, match='strictly increasing'):
        check_dose_response(rows, 'gpt2-124m', spec=bad_spec)

    duplicate_dose = copy.deepcopy(DOSE_RESPONSE_SPEC)
    duplicate_dose['softsign_p']['levels'][2]['dose'] = 2.0
    with pytest.raises(ValueError, match='repeats dose'):
        check_dose_response(rows, 'gpt2-124m', spec=duplicate_dose)

    multi_knob = copy.deepcopy(DOSE_RESPONSE_SPEC)
    multi_knob['softsign_p']['declared_diff'] = ('p_init', 'alpha_jitter')
    with pytest.raises(ValueError, match='exactly one'):
        check_dose_response(rows, 'gpt2-124m', spec=multi_knob)

    wrong_axis = copy.deepcopy(DOSE_RESPONSE_SPEC)
    wrong_axis['softsign_p']['axis'] = 'alpha_init'
    with pytest.raises(ValueError, match='does not match'):
        check_dose_response(rows, 'gpt2-124m', spec=wrong_axis)


def test_level_tests_trend_and_post_hoc_winner_are_reported_separately():
    out = dose_response_comparison(_summaries())
    p_sweep = out['designs']['softsign_p']
    by_row = {entry['row']: entry for entry in p_sweep['levels']}
    assert by_row['dytc_p2_softsign']['is_reference'] is True
    assert by_row['dytc_p1_softsign']['delta_paired_mean'] > 0.0
    assert by_row['dytc_p1_softsign']['effect_helps'] is False
    assert by_row['dytc_p1_softsign']['p_one_sided_holm'] == pytest.approx(1.0)
    assert by_row['dytc_p4_softsign']['delta_paired_mean'] > 0.0
    assert by_row['dytc_hardtanh_eq']['delta_paired_mean'] > 0.0

    trend = p_sweep['trend']
    assert trend['doses'] == pytest.approx([1.0, 2.0, 6.0])
    assert trend['mean_slope_per_dose'] > 0.0
    assert trend['p_one_sided'] < 0.05
    assert trend['direction_observed'] is True
    assert trend['trend_detected'] is True
    assert trend['trend_significant'] is True
    assert trend['certifiable'] is True

    best = p_sweep['best_level']
    selection = p_sweep['best_level_selection']
    assert best['row'] == 'dytc_p2_softsign'
    assert best['is_reference'] is True
    assert selection['post_hoc'] is True
    assert selection['stable_across_seeds'] is True
    assert selection['is_interior_level'] is True
    assert selection['n_seeds_compared'] == 3

    alpha = out['designs']['dyt_alpha_init']
    assert alpha['best_level']['row'] == 'dyt_alpha_1.0'
    assert alpha['best_level_selection']['is_interior_level'] is False
    assert alpha['best_level_selection']['stable_across_seeds'] is True
    assert alpha['trend'] is None
    assert len([entry for entry in out['rows']
                if not entry['is_reference']]) == 5
    assert len(out['trends']) == 1


def test_missing_levels_and_aliases_are_skipped_not_silently_dropped():
    summaries = [row for row in _summaries()
                 if row['name'] != 'dytc_p4_softsign']
    out = dose_response_comparison(summaries)
    p_sweep = out['designs']['softsign_p']
    by_row = {entry['row']: entry for entry in p_sweep['levels']}
    assert 'no usable val_loss' in by_row['dytc_p4_softsign']['skipped']
    assert 'no usable val_loss' in p_sweep['trend']['skipped']
    assert p_sweep['trend']['certifiable'] is False

    aliased = _summaries()
    for row in aliased:
        if row['name'] == 'dytc_p4_softsign':
            row['reused_from'] = 'dytc_p2_softsign'
    out = dose_response_comparison(aliased)
    by_row = {entry['row']: entry
              for entry in out['designs']['softsign_p']['levels']}
    assert 'same run' in by_row['dytc_p4_softsign']['skipped']


def test_two_seed_sweeps_stay_descriptive():
    summaries = [row for row in _summaries()
                 if row['name'].startswith('dytc_')]
    for row in summaries:
        row['per_seed'] = row['per_seed'][:2]
        row['seeds'] = row['seeds'][:2]
        row['n_seeds'] = 2
        row['val_loss'] = sum(item['val_loss']
                              for item in row['per_seed']) / 2
    out = dose_response_comparison(summaries)
    p_sweep = out['designs']['softsign_p']
    level = [entry for entry in p_sweep['levels']
             if entry['row'] == 'dytc_p1_softsign'][0]
    assert level['replicated_across_seeds'] is False
    assert level['p_one_sided'] is not None
    assert level['p_one_sided_holm'] is None
    assert level['effect_helps_significantly'] is False
    assert p_sweep['trend']['certifiable'] is False
    assert p_sweep['trend']['trend_significant'] is False
    assert p_sweep['best_level_selection']['stable_across_seeds'] is None


def test_best_level_needs_a_shared_seed_set():
    summaries = [row for row in _summaries()
                 if row['name'].startswith('dytc_')]
    for row in summaries:
        if row['name'] == 'dytc_p4_softsign':
            row['per_seed'] = [
                {'seed': seed, 'val_loss': 2.03 + index * 0.01}
                for index, seed in enumerate((45, 46, 47))]
            row['seeds'] = [45, 46, 47]
            row['val_loss'] = 2.04
    out = dose_response_comparison(summaries)
    p_sweep = out['designs']['softsign_p']
    assert p_sweep['best_level'] is None
    selection = p_sweep['best_level_selection']
    assert selection['n_seeds_compared'] == 0
    assert selection['common_seeds'] == []
    assert 'no shared seed set' in selection['note']


def test_sweep_contrasts_are_pre_registered_and_fingerprinted():
    rows, hypotheses = _matrix()
    matrix_report = validate_matrix(rows, hypotheses, 'gpt2-124m')
    plan = build_analysis_plan(
        rows, hypotheses, matrix_report, ['ln_baseline', 'rmsnorm_baseline'],
        'gpt2-124m', list(SEEDS))
    by_id = {entry['id']: entry for entry in plan['contrasts']}
    expected = {
        'dose_response:softsign_p:dytc_p1_softsign',
        'dose_response:softsign_p:dytc_p4_softsign',
        'dose_response:softsign_p:dytc_hardtanh_eq',
        'dose_response:softsign_p:trend',
        'dose_response:dyt_alpha_init:dyt_alpha_0.3',
        'dose_response:dyt_alpha_init:dyt_alpha_1.0',
    }
    assert expected <= set(by_id)
    assert 'dose_response:softsign_p:dytc_p2_softsign' not in by_id
    assert by_id['dose_response:softsign_p:trend']['kind'] == 'trend'
    assert by_id['dose_response:softsign_p:trend']['sided'] == 'one_sided'
    for contrast_id in expected:
        assert by_id[contrast_id]['family'] == 'dose_response'
        assert by_id[contrast_id]['tier'] == 2

    changed = copy.deepcopy(matrix_report)
    changed['dose_response']['softsign_p']['levels'][2]['dose'] = 5.0
    changed_plan = build_analysis_plan(
        rows, hypotheses, changed, ['ln_baseline', 'rmsnorm_baseline'],
        'gpt2-124m', list(SEEDS))
    assert changed_plan['fingerprint'] != plan['fingerprint']

    out = dose_response_comparison(_summaries())
    study = study_level_correction({'dose_response': out}, plan)
    study_by_id = {entry['id']: entry for entry in study['rows']}
    assert expected <= set(study_by_id)
    assert not [contrast_id for contrast_id
                in study['unresolved_plan_contrasts']
                if contrast_id.startswith('dose_response:')]
    assert study_by_id['dose_response:softsign_p:trend'][
        'confirmed_at_study_level'] is True
    assert study_by_id['dose_response:softsign_p:dytc_p1_softsign'][
        'family'] == 'dose_response'
