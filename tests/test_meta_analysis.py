"""Results-analysis checks for multi-scale evidence synthesis.

A multi-scale claim used to be a sign vote: if every scale's delta pointed the
same way, the direction was called replicated. That ignores how precisely each
scale measured the effect and whether the scales agree beyond their own noise.
These tests pin the inverse-variance / random-effects meta-analysis that now
accompanies the vote:

* a homogeneous set of studies collapses to one common effect, with zero
  between-study variance;
* two disagreeing studies produce Cochran's Q, I^2 and tau^2, and the
  random-effects interval is wider than the fixed-effect one;
* a prediction interval needs three studies and is wider than the confidence
  interval for the pooled mean;
* leaving one scale out re-pools the rest, flags a scale that reverses the
  sign or changes the significance/heterogeneity verdict, and stays quiet
  when the pooled result is not single-scale dependent;
* a precision supplied only as a confidence interval is inverted with that
  interval's own level and degrees of freedom, not assumed to be normal;
* malformed records are excluded with a reason, never assigned an invented
  precision;
* the scale-replication report forwards per-scale precision and mirrors a
  reversed pair's interval along with its sign-flipped delta.

These are pure-Python synthetic studies, so no checkpoint or GPU is needed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.metrics import (chi2_sf, meta_analysis,
                               scale_replication_summary, student_t_cdf,
                               student_t_ppf)
from scripts.check_scale_replication import (_observation_from_pair,
                                             _observations_from_moe)


# ---------------------------------------------------------------------------
# Distribution helper
# ---------------------------------------------------------------------------
def test_chi2_survival_matches_known_critical_values():
    assert chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-12)
    assert chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-12)
    assert chi2_sf(50.0, 2) == pytest.approx(1.3887943864964e-11, rel=1e-9)
    assert chi2_sf(0.0, 3) == 1.0
    assert chi2_sf(float('inf'), 3) == 0.0
    with pytest.raises(ValueError):
        chi2_sf(1.0, 0.0)


# ---------------------------------------------------------------------------
# Closed-form two-study case
# ---------------------------------------------------------------------------
def test_two_study_pooling_has_the_closed_form_solution():
    out = meta_analysis([
        {'label': 'small', 'effect': 1.0, 'stderr': 1.0},
        {'label': 'large', 'effect': 3.0, 'stderr': 1.0},
    ])
    fixed = out['fixed_effect']
    random_effects = out['random_effects']
    heterogeneity = out['heterogeneity']

    assert out['n_studies'] == 2
    assert out['meaningful'] is True
    assert fixed['pooled_effect'] == pytest.approx(2.0)
    assert fixed['stderr'] == pytest.approx(math.sqrt(0.5))
    # Q = (1-2)^2 + (3-2)^2 = 2 at df = 1; I^2 = (2-1)/2 = 50%; the
    # DerSimonian-Laird tau^2 is (Q-df)/(sum w - sum w^2/sum w) = 1.
    assert heterogeneity['q'] == pytest.approx(2.0)
    assert heterogeneity['df'] == pytest.approx(1.0)
    assert heterogeneity['i_squared'] == pytest.approx(50.0)
    assert heterogeneity['tau_squared'] == pytest.approx(1.0)
    assert random_effects['pooled_effect'] == pytest.approx(2.0)
    assert random_effects['stderr'] == pytest.approx(1.0)
    assert out['pooled_effect'] == pytest.approx(2.0)
    assert out['pooled_effect_source'] == 'random_effects'
    # Positive pooled effect: the lower-tail probability is > 0.5 and the
    # two-sided p is the smaller (upper) tail doubled.
    expected_p = student_t_cdf(random_effects['t'], 1.0)
    assert random_effects['p_one_sided'] == pytest.approx(expected_p)
    assert random_effects['p_two_sided'] == pytest.approx(
        min(1.0, 2.0 * (1.0 - expected_p)))
    assert out['direction_consistent'] is True
    # Two studies cannot support a prediction interval for a third scale.
    assert out['prediction_interval']['meaningful'] is False
    assert out['prediction_interval']['low'] is None


def test_homogeneous_studies_collapse_to_the_common_effect():
    out = meta_analysis([
        {'label': 'a', 'effect': -2.0, 'stderr': 0.5},
        {'label': 'b', 'effect': -2.0, 'stderr': 0.5},
        {'label': 'c', 'effect': -2.0, 'stderr': 0.5},
    ])
    fixed = out['fixed_effect']
    random_effects = out['random_effects']
    heterogeneity = out['heterogeneity']

    assert fixed['pooled_effect'] == pytest.approx(-2.0)
    assert fixed['stderr'] == pytest.approx(0.5 / math.sqrt(3.0))
    assert heterogeneity['q'] == pytest.approx(0.0)
    assert heterogeneity['i_squared'] == pytest.approx(0.0)
    assert heterogeneity['tau_squared'] == pytest.approx(0.0)
    assert random_effects['pooled_effect'] == pytest.approx(fixed['pooled_effect'])
    assert random_effects['stderr'] == pytest.approx(fixed['stderr'])
    # The advantage is negative and precise at df = 2, so the one-sided
    # lower-tail test rejects the no-effect null.
    assert random_effects['p_one_sided'] < 0.05
    assert random_effects['p_two_sided'] < 0.05

    prediction = out['prediction_interval']
    assert prediction['meaningful'] is True
    assert prediction['low'] < random_effects['ci_low']
    assert prediction['high'] > random_effects['ci_high']


def test_heterogeneity_widens_the_random_effects_interval_and_is_reported():
    out = meta_analysis([
        {'label': 'a', 'effect': -1.0, 'stderr': 0.2},
        {'label': 'b', 'effect': 0.0, 'stderr': 0.2},
        {'label': 'c', 'effect': 1.0, 'stderr': 0.2},
    ])
    fixed = out['fixed_effect']
    random_effects = out['random_effects']
    heterogeneity = out['heterogeneity']

    assert heterogeneity['q'] == pytest.approx(50.0)
    assert heterogeneity['i_squared'] == pytest.approx(96.0)
    assert heterogeneity['tau_squared'] == pytest.approx(0.96)
    assert heterogeneity['p'] < 1e-9
    assert random_effects['stderr'] > fixed['stderr']
    random_width = random_effects['ci_high'] - random_effects['ci_low']
    fixed_width = fixed['ci_high'] - fixed['ci_low']
    assert random_width > fixed_width
    # The pooled point estimate is zero, but the heterogeneity note must stop
    # a reader from calling that "one replicated mechanism".
    assert 'heterogeneity' in out['note']
    assert out['n_negative'] == 1 and out['n_positive'] == 1
    assert out['direction_consistent'] is False


# ---------------------------------------------------------------------------
# Leave-one-scale-out influence
# ---------------------------------------------------------------------------
def test_leave_one_scale_out_flags_a_single_dominant_scale():
    out = meta_analysis([
        {'label': 'a', 'effect': -1.0, 'stderr': 0.1},
        {'label': 'b', 'effect': -0.9, 'stderr': 0.1},
        {'label': 'c', 'effect': 3.0, 'stderr': 0.1},
    ])
    influence = out['influence']

    assert influence['meaningful'] is True
    assert influence['n_scales'] == 3
    assert len(influence['per_drop']) == 3
    assert influence['influential_scales'] == ['c']
    assert influence['n_influential'] == 1
    # The outlier pulls the full pool positive; dropping it restores the
    # negative direction of the other two scales and removes the heterogeneity.
    assert out['pooled_effect'] > 0.0
    dropped = next(row for row in influence['per_drop']
                   if row['dropped'] == 'c')
    assert dropped['pooled_effect'] < 0.0
    assert dropped['keeps_full_sign'] is False
    assert dropped['same_significance_verdict'] is False
    assert dropped['same_heterogeneity_verdict'] is False
    assert dropped['influential'] is True
    assert influence['sign_stable'] is False
    assert influence['significance_verdict_stable'] is False
    assert influence['heterogeneity_verdict_stable'] is False
    assert influence['max_abs_change'] == pytest.approx(1.3166666666666667)
    assert influence['pooled_effect_range'][0] == pytest.approx(-0.95)
    assert influence['pooled_effect_range'][1] == pytest.approx(1.05)
    assert 'sensitive to a single scale' in influence['note']


def test_leave_one_scale_out_is_quiet_for_a_stable_replication():
    out = meta_analysis([
        {'label': 'a', 'effect': -0.50, 'stderr': 0.05},
        {'label': 'b', 'effect': -0.52, 'stderr': 0.06},
        {'label': 'c', 'effect': -0.51, 'stderr': 0.04},
    ])
    influence = out['influence']

    assert influence['meaningful'] is True
    assert influence['influential_scales'] == []
    assert influence['n_influential'] == 0
    assert influence['sign_stable'] is True
    assert influence['significance_verdict_stable'] is True
    assert influence['heterogeneity_verdict_stable'] is True
    assert all(not row['influential'] for row in influence['per_drop'])
    assert 'descriptive sensitivity only' in influence['note']


def test_leave_one_scale_out_needs_three_scales():
    out = meta_analysis([
        {'label': 'small', 'effect': -0.2, 'stderr': 0.1},
        {'label': 'large', 'effect': -0.3, 'stderr': 0.1},
    ])
    influence = out['influence']

    assert influence['meaningful'] is False
    assert influence['per_drop'] == []
    assert influence['influential_scales'] == []
    assert '>= 3 usable scales' in influence['note']


# ---------------------------------------------------------------------------
# Precision recovered from a confidence interval
# ---------------------------------------------------------------------------
def test_ci_only_study_is_inverted_with_its_own_level_and_df():
    stderr = 1.0
    df = 2.0
    half = student_t_ppf(0.975, df) * stderr
    out = meta_analysis([
        {'label': 'ci-only', 'effect': 1.0, 'ci_low': 1.0 - half,
         'ci_high': 1.0 + half, 'df': df, 'ci_level': 0.95},
    ])
    assert out['n_studies'] == 1
    study = out['studies'][0]
    assert study['stderr'] == pytest.approx(stderr)
    assert study['se_basis'] == 'ci_t'
    # A one-study pool is descriptive, not a meta-analytic claim.
    assert out['meaningful'] is False
    assert 'descriptive' in out['note']


def test_zero_or_missing_precision_is_excluded_not_invented():
    out = meta_analysis([
        {'label': 'ok', 'effect': -0.2, 'stderr': 0.1},
        None,
        {'label': 'nan effect', 'effect': float('nan'), 'stderr': 0.1},
        {'label': 'zero stderr', 'effect': -0.1, 'stderr': 0.0},
        {'label': 'no precision', 'effect': -0.1},
        {'label': 'boolean effect', 'effect': True, 'stderr': 0.1},
        {'label': 'inverted ci', 'effect': -0.1, 'ci_low': 1.0,
         'ci_high': 0.0, 'df': 2.0},
    ])
    assert out['n_studies'] == 1
    assert out['studies'][0]['label'] == 'ok'
    assert len(out['excluded']) == 6
    reasons = '; '.join(item['reason'] for item in out['excluded'])
    assert 'non-finite effect' in reasons
    assert 'stderr' in reasons
    assert 'ci_low/ci_high' in reasons


def test_empty_input_returns_a_typed_empty_summary():
    out = meta_analysis([])
    assert out['n_studies'] == 0
    assert out['meaningful'] is False
    assert out['pooled_effect'] is None
    assert out['fixed_effect'] is None
    assert out['influence']['meaningful'] is False
    assert out['influence']['per_drop'] == []
    assert 'nothing can be pooled' in out['note']


# ---------------------------------------------------------------------------
# Scale-replication integration
# ---------------------------------------------------------------------------
def test_scale_replication_carries_pooled_effect_and_heterogeneity():
    summary = scale_replication_summary([
        {'scale': '124m', 'delta': -0.10, 'stderr': 0.02, 'certified': True},
        {'scale': '350m', 'delta': -0.12, 'stderr': 0.03, 'certified': True},
        {'scale': '1b', 'delta': -0.11, 'stderr': 0.04, 'certified': False},
    ], min_scales=2)
    block = summary['per_metric']['default']
    pooled = block['meta_analysis']
    assert block['n_scales'] == 3
    assert pooled['n_studies'] == 3
    assert pooled['meaningful'] is True
    assert pooled['direction_consistent'] is True
    assert pooled['pooled_effect'] < 0.0
    assert pooled['random_effects']['ci_high'] < 0.0
    assert pooled['heterogeneity']['i_squared'] is not None
    assert pooled['prediction_interval']['meaningful'] is True
    assert summary['meta_analysed_metrics'] == ['default']
    # The sign vote is unchanged: two certified scales agree, so the
    # multi-scale certification still follows its own rule.
    assert block['replicated_direction'] is True
    assert block['multi_scale_certified'] is True


def test_scale_replication_sign_vote_survives_without_precision():
    summary = scale_replication_summary([
        {'scale': 'a', 'delta': -0.10, 'certified': True},
        {'scale': 'b', 'delta': -0.20, 'certified': True},
    ])
    block = summary['per_metric']['default']
    assert block['replicated_direction'] is True
    pooled = block['meta_analysis']
    assert pooled['n_studies'] == 0
    assert len(pooled['excluded']) == 2
    assert summary['meta_analysed_metrics'] == []


def test_reversed_pair_mirrors_the_confidence_interval_with_the_delta():
    observation = _observation_from_pair({
        'a': 'reference', 'b': 'candidate', 'a_scale': 's1',
        'metric': 'loss', 'mean_delta': 0.2,
        'ci_low': 0.1, 'ci_high': 0.3, 'df': 5, 'n_shared_seeds': 3,
        'significant': True,
    }, 'candidate', 'reference', 'loss')
    assert observation['orientation'] == 'reversed'
    assert observation['delta'] == pytest.approx(-0.2)
    # B - A in [0.1, 0.3] is A - B in [-0.3, -0.1], not [-0.1, 0.3].
    assert observation['ci_low'] == pytest.approx(-0.3)
    assert observation['ci_high'] == pytest.approx(-0.1)
    assert observation['certified'] is False


def test_explicit_false_or_empty_exclusion_reason_keeps_the_observation():
    base = {
        'a': 'candidate', 'b': 'reference', 'a_scale': 's1',
        'metric': 'loss', 'mean_delta': -0.1, 'same_scale': True,
        'significant': True,
    }
    for flag in (False, '', None):
        entry = {**base, 'excluded_from_family': flag}
        observation = _observation_from_pair(
            entry, 'candidate', 'reference', 'loss')
        assert observation is not None
        assert observation['delta'] == pytest.approx(-0.1)

    entry = {**base, 'excluded_from_family': 'protocol mismatch'}
    assert _observation_from_pair(
        entry, 'candidate', 'reference', 'loss') is None


def test_moe_explicit_false_exclusion_reason_keeps_the_row():
    def row(flag):
        return {
            'norm': 'candidate', 'reference': 'reference', 'scale': 's1',
            'metric': 'loss', 'delta_vs_reference': -0.1,
            'beats_reference_significantly': True, 'same_scale': True,
            'excluded_from_family': flag,
        }

    observations = _observations_from_moe(
        {'rows': [row(False), row('protocol mismatch'), row(None)]},
        'candidate', 'reference', 'loss')
    assert len(observations) == 2
    assert all(item['delta'] == pytest.approx(-0.1)
               for item in observations)
