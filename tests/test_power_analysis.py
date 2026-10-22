"""Prospective power / sample-size planning for paired seed comparisons.

The comparison reports already carry the *observed* minimum detectable effect.
A null result still needs the design question: with the observed paired SD,
how many pairs would the pre-specified smallest effect of interest require?
These tests pin the planning helper and its two report integrations:

* the required n is the smallest n whose MDE reaches the target (and n-1 does
  not), so a reported "adequately powered" flag is not off by one;
* a larger target effect never needs more pairs;
* a zero-variance difference needs only the minimum two pairs;
* no target (margin <= 0) leaves the required-n fields ``None`` rather than
  inventing a target from the observed effect;
* the ablation and Phase-1 comparison reports expose the block.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.metrics import paired_design_summary, paired_required_n
from scripts.ablation import baseline_comparison
from scripts.compare_arms import compare_arms

SEEDS = [42, 43, 44]


def _row(name, values, **extra):
    items = sorted(values.items())
    vals = [value for _, value in items]
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
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


def test_required_n_is_the_smallest_n_that_reaches_the_target():
    std, target = 0.10, 0.08
    n = paired_required_n(std, target)
    assert n is not None and n >= 2
    at_n = paired_design_summary(std, n, margin=target)
    assert at_n['adequately_powered_for_margin'] is True
    assert at_n['minimum_detectable_delta_95_power80'] <= target
    if n > 2:
        before = paired_design_summary(std, n - 1, margin=target)
        assert before['adequately_powered_for_margin'] is False
        assert before['minimum_detectable_delta_95_power80'] > target


def test_a_larger_effect_never_needs_more_pairs():
    small = paired_required_n(0.10, 0.02)
    large = paired_required_n(0.10, 0.10)
    assert small is not None and large is not None
    assert large <= small
    # A target so small it is unreachable under the search cap is reported as
    # None, not as an arbitrary huge n.
    assert paired_required_n(0.10, 1e-9, max_n=1000) is None


def test_zero_variance_needs_only_the_minimum_two_pairs():
    assert paired_required_n(0.0, 0.05) == 2
    out = paired_design_summary(0.0, 2, margin=0.05)
    assert out['minimum_detectable_delta_95_power80'] == 0.0
    assert out['adequately_powered_for_margin'] is True


def test_no_pre_specified_margin_leaves_the_target_unset():
    out = paired_design_summary(0.1, 3, margin=0.0)
    assert out['margin'] is None
    assert out['required_pairs_for_margin'] is None
    assert out['adequately_powered_for_margin'] is None
    assert 'pre-specified margin' in out['note']


@pytest.mark.parametrize('kwargs', [
    {'std': -0.1, 'effect': 0.1},
    {'std': float('nan'), 'effect': 0.1},
    {'std': 0.1, 'effect': 0.0},
    {'std': 0.1, 'effect': -0.1},
    {'std': 0.1, 'effect': 0.1, 'alpha': 0.0},
    {'std': 0.1, 'effect': 0.1, 'power': 1.0},
])
def test_planning_inputs_are_validated(kwargs):
    std = kwargs.pop('std')
    effect = kwargs.pop('effect')
    with pytest.raises(ValueError):
        paired_required_n(std, effect, **kwargs)


def test_ablation_report_exposes_the_design_block():
    baseline = _row('ln_baseline', {seed: 2.0 for seed in SEEDS})
    noisy = _row('dytcd_full', {42: 1.90, 43: 1.95, 44: 2.05})
    out = baseline_comparison([baseline, noisy], ['ln_baseline'],
                              margin=0.05, alpha=0.05)
    entry = out['rows'][0]
    design = entry['design_power']
    assert design['paired_sd'] > 0.0
    assert design['margin'] == 0.05
    assert design['required_pairs_for_margin'] is not None
    assert isinstance(design['adequately_powered_for_margin'], bool)


def test_compare_arms_report_exposes_the_design_block():
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
        'candidate': aggregate({42: 1.9, 43: 1.95, 44: 2.05}),
    }
    out = compare_arms(arms, reference='reference', margin=0.05,
                       require_recipe=False)
    design = out['rows'][0]['design_power']
    assert design['required_pairs_for_margin'] is not None
    assert design['margin'] == 0.05
