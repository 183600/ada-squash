"""Distribution-free paired inference: exact sign-flip randomization test.

The paired t-test is the certification test in this repository, but with the
3-5 seeds the experiments run its normal-theory p-value can be much smaller
than any distribution-free test could produce. These tests pin the sign-flip
counterpart: the exact enumeration, the Monte-Carlo branch, the reported
resolution floor, exclusion of non-finite deltas, and integration into the
comparison/trend report rows.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.metrics import (paired_randomization_test,
                               paired_trend_test)

SEEDS = (42, 43, 44)


def _aggregate(values):
    return {
        'norm': 'synthetic',
        'model': 'gpt2-124m',
        'seeds': sorted(values),
        'complete': True,
        'per_seed': [
            {'seed': seed, 'final_val_loss': value,
             'final_val_step': 1000}
            for seed, value in sorted(values.items())
        ],
    }


def test_exact_three_seed_test_cannot_reach_alpha():
    out = paired_randomization_test([-0.1, -0.11, -0.09])
    assert out['basis'] == 'exact_enumeration'
    assert out['n'] == 3
    assert out['n_permutations_evaluated'] == 8
    assert out['minimum_attainable_p_one_sided'] == pytest.approx(1 / 8)
    assert out['can_reach_alpha'] is False
    assert out['p_one_sided'] == pytest.approx(1 / 8)
    assert out['p_upper_one_sided'] == pytest.approx(1.0)
    assert out['p_two_sided'] == pytest.approx(0.25)
    assert out['meaningful'] is True
    assert 'relies on the parametric t model' in out['note']


def test_exact_five_seed_test_can_reach_alpha_and_is_two_sided_twice_one_sided():
    out = paired_randomization_test([-0.1] * 5)
    assert out['n_permutations_evaluated'] == 32
    assert out['p_one_sided'] == pytest.approx(1 / 32)
    assert out['p_two_sided'] == pytest.approx(1 / 16)
    assert out['minimum_attainable_p_one_sided'] == pytest.approx(1 / 32)
    assert out['can_reach_alpha'] is True


def test_mixed_and_zero_deltas_use_the_non_strict_boundary():
    out = paired_randomization_test([-0.1, 0.0, 1.0, 0.9], exact_max=12)
    assert out['n_negative'] == 1
    assert out['n_positive'] == 2
    assert out['n_zero'] == 1
    assert out['observed_centered_mean'] == pytest.approx(0.45)
    # The observed mean is positive, so the lower tail contains almost every
    # sign pattern. The upper tail contains the all-positive pattern and the
    # pattern that flips only the -0.1 (which reproduces the observed mean
    # exactly), doubled by the zero delta's sign symmetry.
    assert out['p_one_sided'] == pytest.approx(7 / 8)
    assert out['p_upper_one_sided'] == pytest.approx(1 / 4)
    assert out['p_two_sided'] == pytest.approx(1 / 2)


def test_margin_shifts_the_centered_deltas():
    deltas = [-0.1, -0.11, -0.09]
    zero = paired_randomization_test(deltas, margin=0.05)
    wide = paired_randomization_test(deltas, margin=0.2)
    assert zero['observed_centered_mean'] == pytest.approx(-0.05)
    assert zero['p_one_sided'] == pytest.approx(1 / 8)
    assert wide['observed_centered_mean'] == pytest.approx(0.1)
    assert wide['p_one_sided'] == pytest.approx(1.0)
    assert wide['p_upper_one_sided'] == pytest.approx(1 / 8)


def test_positive_deltas_are_a_negative_alternative_failure():
    out = paired_randomization_test([0.1, 0.11, 0.09])
    assert out['p_one_sided'] == pytest.approx(1.0)
    assert out['p_upper_one_sided'] == pytest.approx(1 / 8)
    assert out['p_two_sided'] == pytest.approx(0.25)


def test_monte_carlo_branch_is_seeded_and_reproducible():
    deltas = [-0.01 * (index + 1) for index in range(15)]
    first = paired_randomization_test(
        deltas, exact_max=12, permutations=500, seed=7)
    second = paired_randomization_test(
        deltas, exact_max=12, permutations=500, seed=7)
    assert first['basis'] == 'monte_carlo'
    assert first['seed'] == 7
    assert first['n_permutations_evaluated'] == 500
    assert first['p_one_sided'] == second['p_one_sided']
    assert first['p_upper_one_sided'] == second['p_upper_one_sided']
    assert first['p_one_sided'] >= 1 / 501
    assert first['p_one_sided'] <= 1.0
    assert first['monte_carlo_standard_error'] is not None
    assert first['minimum_attainable_p_one_sided'] == pytest.approx(1 / 501)


def test_non_finite_and_non_numeric_deltas_are_excluded_not_fabricated():
    out = paired_randomization_test([-0.1, float('nan'), 'x', -0.2, -0.15])
    assert out['n_input'] == 5
    assert out['n'] == 3
    assert out['n_excluded'] == 2
    assert [item['index'] for item in out['excluded']] == [1, 2]
    assert out['mean_delta'] == pytest.approx(-0.15)
    assert 'excluded' in out['note']

    empty = paired_randomization_test([])
    assert empty['n'] == 0
    assert empty['p_one_sided'] is None
    assert empty['basis'] == 'undefined'
    assert empty['meaningful'] is False


@pytest.mark.parametrize('kwargs', [
    {'alpha': 0.0},
    {'alpha': 1.0},
    {'margin': float('inf')},
    {'margin': float('nan')},
    {'exact_max': -1},
    {'exact_max': 1.5},
    {'exact_max': True},
    {'exact_max': 21},
    {'permutations': 0},
    {'seed': -1},
])
def test_invalid_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        paired_randomization_test([-20, -21, -19], **kwargs)


def test_robustness_block_carries_the_randomization_result_and_margin():
    from scripts.ablation import _robustness_for_deltas

    deltas = {str(seed): value for seed, value in
              zip(SEEDS, [-0.1, -0.11, -0.09])}
    out = _robustness_for_deltas(deltas, margin=0.05)
    test = out['randomization_test']
    assert test['test'] == 'paired_sign_flip_randomization'
    assert test['margin'] == pytest.approx(0.05)
    assert test['n'] == 3
    assert test['p_one_sided'] == pytest.approx(1 / 8)
    assert test['can_reach_alpha'] is False


def test_compare_arms_reports_the_sign_flip_p_value():
    from scripts.compare_arms import compare_arms

    arms = {
        'reference': _aggregate({seed: 2.0 for seed in SEEDS}),
        'candidate': _aggregate({seed: 2.0 + delta for seed, delta in
                                 zip(SEEDS, [-0.1, -0.11, -0.09])}),
    }
    out = compare_arms(arms, reference='reference',
                       equivalence_margin=0.2, require_recipe=False)
    row = [entry for entry in out['rows']
           if entry['name'] == 'candidate'][0]
    assert row['randomization_test']['p_one_sided'] == pytest.approx(1 / 8)
    assert row['randomization_test']['can_reach_alpha'] is False
    assert row['equivalence']['enabled'] is True


def test_trend_test_reports_a_slope_randomization_result():
    doses = [1, 2, 3, 4]
    per_seed = {
        42: [0.0, -0.010, -0.021, -0.033],
        43: [0.0, -0.009, -0.020, -0.031],
        44: [0.0, -0.012, -0.019, -0.034],
    }
    out = paired_trend_test(doses, per_seed)
    test = out['slope_randomization_test']
    assert test is not None
    assert test['n'] == 3
    assert test['basis'] == 'exact_enumeration'
    assert test['p_one_sided'] == pytest.approx(1 / 8)
    assert test['can_reach_alpha'] is False
    # The slopes are negative and consistent, so the upper tail should not.
    assert test['p_upper_one_sided'] == pytest.approx(1.0)
