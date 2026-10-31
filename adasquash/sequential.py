"""Pre-specified group-sequential monitoring for seed-replicated experiments.

The comparison layer answers a statistical question after a fixed number of
seeds exists. A running study has a second, equally important question before
that point: *when is it legitimate to look?* Adding seeds until the paired
t-test crosses ``p < 0.05`` and then reporting that p-value is optional
stopping; the nominal alpha no longer controls the error rate, and a study
that looked after every seed can find a "significant" difference under a true
null far more often than 5% of the time. The problem is acute here because
the unit of replication is a training seed (three to five seeds are typical),
so every additional run is a new look at a small sample.

This module makes the stopping rule part of the pre-registration instead of a
post-hoc decision:

``alpha_spent``
    A Lan-DeMets-style alpha-spending function (O'Brien-Fleming or Pocock),
    evaluated at an information fraction.
``group_sequential_plan``
    Freezes the maximum number of seed pairs, the interim looks (in pairs),
    the family-level alpha and the per-look nominal p-value thresholds.
``sequential_monitor``
    Applies that plan to the p-values observed at the pre-specified looks and
    reports whether the spending threshold was crossed. An unadjusted
    ``p < alpha`` at an early look is reported separately: it is exactly the
    quantity that optional stopping inflates, not a licence to stop.
``sequential_paired_audit``
    Computes the look-wise p-values from an ordered sequence of paired seed
    deltas using the same paired t-test as the final analysis, then feeds
    them to ``sequential_monitor``. The seed order must be fixed before
    outcomes are seen (ascending seed id is the default convention in the
    callers).
``audit_paired_contrasts``
    Runs that audit for several named contrasts and summarises how many would
    stop under the plan versus only under an unadjusted per-look threshold.

The per-look thresholds are *incremental* spending levels and are applied to
the stagewise p-value; by the union bound their sum controls the family-wise
error rate without requiring a full multivariate-normal boundary computation.
That is deliberately conservative relative to a group-sequential boundary that
exploits the correlation between looks: a result that crosses this plan is
valid, while a result that only crosses an unadjusted per-look alpha is not
certified.

A monitoring plan is only useful when it is fixed before the results. The
ablation plan embeds it in ``analysis_plan.json`` and therefore in the
pre-registration fingerprint; changing the number of looks, their seed counts
or the spending function is a design change and moves the fingerprint.
"""
from __future__ import annotations

import math
from statistics import NormalDist
from typing import Mapping, Optional, Sequence

from .metrics_stats import (MIN_SEEDS_FOR_SIGNIFICANCE, _as_real,
                            _strict_int, _validate_alpha,
                            paired_t_one_sided)

_SPENDING_FUNCTIONS = ('obrien-fleming', 'pocock')


def _finite_fraction(value, name: str) -> float:
    fraction = _as_real(value)
    if fraction is None or not math.isfinite(fraction):
        raise ValueError(f'{name} must be a finite number, got {value!r}')
    return fraction


def _validate_spending(spending: str) -> str:
    if not isinstance(spending, str):
        raise ValueError(
            f'spending must be one of {_SPENDING_FUNCTIONS}, got {spending!r}')
    normalized = spending.strip().lower().replace('_', '-')
    if normalized not in _SPENDING_FUNCTIONS:
        raise ValueError(
            f'spending must be one of {_SPENDING_FUNCTIONS}, got {spending!r}')
    return normalized


def alpha_spent(alpha: float, information_fraction: float,
                *, spending: str = 'obrien-fleming') -> float:
    """Cumulative alpha spent by an information fraction in ``(0, 1]``.

    ``alpha`` is the total one-sided error budget at the final look. The
    O'Brien-Fleming spending function spends little early and almost all of
    the budget at the end, which is appropriate when the earliest looks are
    least informative. Pocock spending is closer to equal per look. Both
    functions are monotone and evaluate to exactly ``alpha`` at ``t = 1``.
    """
    alpha_value = _validate_alpha(alpha)
    t = _finite_fraction(information_fraction, 'information_fraction')
    if not 0.0 < t <= 1.0:
        raise ValueError(
            f'information_fraction must be in (0, 1], got {information_fraction!r}')
    method = _validate_spending(spending)
    if t == 1.0:
        return alpha_value
    if method == 'obrien-fleming':
        z = NormalDist().inv_cdf(1.0 - alpha_value)
        spent = 1.0 - NormalDist().cdf(z / math.sqrt(t))
    else:
        spent = alpha_value * math.log1p((math.e - 1.0) * t)
    return min(alpha_value, max(0.0, spent))


def _default_look_pairs(max_pairs: int, n_looks: int,
                        min_pairs: int) -> list[int]:
    if n_looks == 1:
        return [max_pairs]
    if max_pairs <= min_pairs:
        raise ValueError(
            f'{max_pairs} planned pair(s) support only one look at the '
            f'certification minimum of {min_pairs}; plan more seeds or use '
            f'one look')
    span = max_pairs - min_pairs
    if n_looks - 1 > span:
        raise ValueError(
            f'{n_looks} looks need {n_looks - 1} distinct additional seed '
            f'pair(s) above {min_pairs}, but only {span} are available before '
            f'{max_pairs}')
    pairs = [min_pairs + int(math.floor(i * span / (n_looks - 1)))
             for i in range(n_looks - 1)]
    pairs.append(max_pairs)
    if len(set(pairs)) != len(pairs) or any(
            left >= right for left, right in zip(pairs, pairs[1:])):
        raise ValueError(
            f'could not place {n_looks} distinct looks between {min_pairs} '
            f'and {max_pairs} seed pairs')
    return pairs


def group_sequential_plan(max_pairs: int, *, n_looks: int = 2,
                          look_pairs: Optional[Sequence[int]] = None,
                          alpha: float = 0.05, n_tests: int = 1,
                          spending: str = 'obrien-fleming',
                          min_pairs: int = MIN_SEEDS_FOR_SIGNIFICANCE) -> dict:
    """Freeze a pre-specified sequential monitoring plan.

    ``max_pairs`` is the maximum number of complete seed pairs the study will
    ever analyse; every test in the plan is an analysis of the first
    ``look_pairs[i]`` pairs in a pre-specified seed order. ``n_tests`` applies
    a Bonferroni family correction across the primary claims, so the spending
    plan for each claim uses ``alpha / n_tests``. ``min_pairs`` defaults to
    the repository's certification minimum: a look below three shared seeds
    cannot certify a contrast even if its p-value is small.

    ``look_pairs`` may be supplied explicitly; otherwise the looks are spread
    as evenly as possible between ``min_pairs`` and ``max_pairs``. The final
    look must be at ``max_pairs``. The returned ``nominal_p_threshold`` of a
    look is the incremental alpha spent at that look, not the cumulative
    alpha and not the unadjusted study alpha.
    """
    max_value = _strict_int(max_pairs, 'max_pairs',
                            minimum=MIN_SEEDS_FOR_SIGNIFICANCE)
    looks_requested = _strict_int(n_looks, 'n_looks', minimum=1)
    tests = _strict_int(n_tests, 'n_tests', minimum=1)
    minimum = _strict_int(min_pairs, 'min_pairs',
                          minimum=MIN_SEEDS_FOR_SIGNIFICANCE)
    if minimum > max_value:
        raise ValueError(
            f'min_pairs={minimum} exceeds max_pairs={max_value}')
    alpha_value = _validate_alpha(alpha)
    method = _validate_spending(spending)
    try:
        family_alpha = alpha_value / tests
    except OverflowError:
        raise ValueError(
            f'n_tests={tests!r} is too large: alpha / n_tests exceeds the '
            f'float range') from None
    if family_alpha == 0.0:
        raise ValueError(
            f'n_tests={tests!r} is too large: alpha / n_tests underflows to '
            f'zero')

    if look_pairs is None:
        pairs = _default_look_pairs(max_value, looks_requested, minimum)
    else:
        if isinstance(look_pairs, (str, bytes, bytearray)) or isinstance(
                look_pairs, Mapping):
            raise ValueError(
                f'look_pairs must be a sequence of integer pair counts, got '
                f'{look_pairs!r}')
        try:
            pairs = [_strict_int(value, 'look_pairs entry', minimum=2)
                     for value in look_pairs]
        except TypeError:
            raise ValueError(
                f'look_pairs must be a sequence of integer pair counts, got '
                f'{look_pairs!r}') from None
        if not pairs:
            raise ValueError('look_pairs must contain at least one look')
        if len(set(pairs)) != len(pairs) or any(
                left >= right for left, right in zip(pairs, pairs[1:])):
            raise ValueError(
                f'look_pairs must be strictly increasing, got {look_pairs!r}')
        if pairs[0] < minimum:
            raise ValueError(
                f'the first look must analyze at least {minimum} pairs, got '
                f'{pairs[0]}')
        if pairs[-1] != max_value:
            raise ValueError(
                f'the final look must analyze all {max_value} planned pairs, '
                f'got {pairs[-1]}')

    cumulative = [alpha_spent(family_alpha, value / max_value,
                              spending=method) for value in pairs]
    cumulative[-1] = family_alpha
    if any(right < left for left, right in zip(cumulative, cumulative[1:])):
        raise ValueError(
            'the alpha-spending function is not monotone at the requested '
            'looks')
    increments = [cumulative[0]] + [
        right - left for left, right in zip(cumulative, cumulative[1:])]
    looks = []
    for index, (pair_count, spent, increment) in enumerate(
            zip(pairs, cumulative, increments), start=1):
        looks.append({
            'look': index,
            'planned_pairs': pair_count,
            'information_fraction': pair_count / max_value,
            'cumulative_alpha': spent,
            'incremental_alpha': increment,
            'nominal_p_threshold': increment,
            'rejection_possible': bool(increment > 0.0),
        })
    return {
        'schema': 'adasquash.sequential_plan.v1',
        'method': ('pre-specified alpha spending with incremental stagewise '
                   'thresholds; the union bound over looks and tests controls '
                   'the family-wise error rate'),
        'spending': method,
        'alpha': alpha_value,
        'n_tests': tests,
        'family_alpha': family_alpha,
        'max_pairs': max_value,
        'min_pairs': minimum,
        'n_looks': len(looks),
        'look_pairs': list(pairs),
        'looks': looks,
        'note': ('look only at the pre-specified seed counts; a smaller '
                 'unadjusted p-value at an earlier look is not a stopping '
                 'signal unless it crosses that look\'s nominal threshold'),
    }


def _validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise ValueError(f'plan must be a sequential plan mapping, got {plan!r}')
    looks = plan.get('looks')
    if not isinstance(looks, list) or not looks:
        raise ValueError('plan must contain a non-empty looks list')
    alpha_value = _validate_alpha(plan.get('alpha'))
    normalized = []
    for index, look in enumerate(looks, start=1):
        if not isinstance(look, dict):
            raise ValueError(f'plan look {index} is not a mapping')
        pairs = _strict_int(look.get('planned_pairs'),
                            f'look {index} planned_pairs', minimum=2)
        threshold = _as_real(look.get('nominal_p_threshold'))
        if (threshold is None or not math.isfinite(threshold)
                or not 0.0 <= threshold <= 1.0):
            raise ValueError(
                f'look {index} nominal_p_threshold must be a finite '
                f'probability, got {look.get("nominal_p_threshold")!r}')
        normalized.append({'planned_pairs': pairs,
                           'nominal_p_threshold': threshold})
    pair_counts = [look['planned_pairs'] for look in normalized]
    if any(left >= right for left, right in zip(pair_counts, pair_counts[1:])):
        raise ValueError(
            f'plan look pair counts must be strictly increasing, got '
            f'{pair_counts!r}')
    return {'alpha': alpha_value, 'looks': normalized, 'raw': plan}


def sequential_monitor(plan: dict, p_values: Sequence[float]) -> dict:
    """Apply a pre-specified spending plan to the observed look p-values.

    ``p_values`` must be the stagewise p-values for the first
    ``len(p_values)`` planned looks, in order. A look is certified when its
    p-value is at most that look's incremental spending threshold. The output
    also records whether any look would have been called significant at the
    unadjusted study alpha; when that happens without a spending-threshold
    crossing, the result is labelled as the optional-stopping pattern that the
    plan exists to prevent.
    """
    checked = _validate_plan(plan)
    if isinstance(p_values, (str, bytes, bytearray)) or isinstance(
            p_values, Mapping):
        raise ValueError(
            f'p_values must be a sequence of probabilities, got {p_values!r}')
    try:
        values = list(p_values)
    except TypeError:
        raise ValueError(
            f'p_values must be a sequence of probabilities, got {p_values!r}') from None
    if not values:
        raise ValueError('p_values must contain at least one observed look')
    if len(values) > len(checked['looks']):
        raise ValueError(
            f'{len(values)} p-value(s) supplied for a {len(checked["looks"])}-'
            f'look plan; the plan must not be extended after the fact')
    probabilities = []
    for index, value in enumerate(values, start=1):
        probability = _as_real(value)
        if probability is None or not math.isfinite(probability):
            raise ValueError(
                f'look {index} p-value must be a finite probability, got '
                f'{value!r}')
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f'look {index} p-value must be in [0, 1], got {value!r}')
        probabilities.append(probability)

    records = []
    first_crossing = None
    uncorrected_looks = []
    for index, (probability, look) in enumerate(
            zip(probabilities, checked['looks']), start=1):
        threshold = look['nominal_p_threshold']
        crossed = probability <= threshold
        uncorrected = probability <= checked['alpha']
        if crossed and first_crossing is None:
            first_crossing = index
        if uncorrected:
            uncorrected_looks.append(index)
        records.append({
            'look': index,
            'planned_pairs': look['planned_pairs'],
            'p_one_sided': probability,
            'nominal_p_threshold': threshold,
            'crossed': bool(crossed),
            'unadjusted_significant': bool(uncorrected),
        })

    analyzed = len(probabilities)
    total = len(checked['looks'])
    if first_crossing is not None:
        status = 'stopped'
    elif analyzed < total:
        status = 'continue'
    else:
        status = 'not_significant_at_final_look'
    controlled = first_crossing is not None
    uncorrected_any = bool(uncorrected_looks)
    optional_stopping_only = uncorrected_any and not controlled
    if optional_stopping_only:
        note = ('an unadjusted look crossed the study alpha but no look '
                'crossed its pre-specified spending threshold; this is the '
                'optional-stopping pattern the plan controls for, so the '
                'result is not certified')
    elif controlled:
        note = (f'look {first_crossing} crossed its pre-specified spending '
                'threshold; stop and do not analyse later looks')
    elif status == 'continue':
        note = ('no pre-specified look has crossed; continue only to the '
                'next planned look, not to an arbitrary later seed count')
    else:
        note = ('the pre-specified plan did not reject at its final look; '
                'adding looks after seeing this result is optional stopping, '
                'not a significance test')
    return {
        'schema': 'adasquash.sequential_monitor.v1',
        'status': status,
        'controlled_significant': bool(controlled),
        'first_crossing_look': first_crossing,
        'unadjusted_significant': uncorrected_any,
        'unadjusted_significant_looks': uncorrected_looks,
        'optional_stopping_only': bool(optional_stopping_only),
        'analyzed_looks': analyzed,
        'planned_looks': total,
        'remaining_looks': total - analyzed,
        'looks': records,
        'note': note,
    }


def sequential_paired_audit(deltas: Sequence[float], plan: dict,
                            *, margin: float = 0.0) -> dict:
    """Look-wise paired-test audit of an ordered sequence of seed deltas.

    The deltas must be ordered by the pre-registered seed order (the callers
    use ascending seed id), not sorted by outcome. Each planned look tests the
    first ``planned_pairs`` deltas with the same one-sided paired t-test used
    by the final comparison, so the audit answers "what would the pre-specified
    monitoring plan have concluded at each look?" without rerunning models.

    A look whose planned prefix is not yet available is simply not evaluated;
    ``status`` then remains ``continue``. More deltas than ``max_pairs`` is a
    design mismatch and is rejected rather than silently truncated.
    """
    checked = _validate_plan(plan)
    max_pairs = plan.get('max_pairs')
    max_pairs_value = _strict_int(max_pairs, 'plan max_pairs', minimum=2)
    if isinstance(deltas, (str, bytes, bytearray)) or isinstance(
            deltas, Mapping):
        raise ValueError(
            f'deltas must be an ordered sequence of numbers, got {deltas!r}')
    try:
        values = list(deltas)
    except TypeError:
        raise ValueError(
            f'deltas must be an ordered sequence of numbers, got {deltas!r}') from None
    if len(values) > max_pairs_value:
        raise ValueError(
            f'{len(values)} paired delta(s) exceed the plan maximum of '
            f'{max_pairs_value}; the plan was frozen for fewer seeds')
    finite = []
    for index, value in enumerate(values, start=1):
        delta = _as_real(value)
        if delta is None or not math.isfinite(delta):
            raise ValueError(
                f'paired delta {index} must be a finite number, got {value!r}')
        finite.append(delta)
    margin_value = _as_real(margin)
    if margin_value is None or not math.isfinite(margin_value):
        raise ValueError(f'margin must be a finite number, got {margin!r}')

    details = []
    p_values = []
    for look in checked['looks']:
        pairs = look['planned_pairs']
        if len(finite) < pairs:
            break
        prefix = finite[:pairs]
        test = paired_t_one_sided(
            prefix, [0.0] * pairs, margin=margin_value,
            alpha=checked['alpha'])
        probability = test.get('p_one_sided')
        if probability is None:
            break
        p_values.append(probability)
        details.append({
            'look': len(details) + 1,
            'planned_pairs': pairs,
            'n': test.get('n'),
            'mean_delta': test.get('mean_delta'),
            'df': test.get('df'),
            'p_one_sided': probability,
            'nominal_p_threshold': look['nominal_p_threshold'],
            'crossed': bool(probability <= look['nominal_p_threshold']),
        })
    if not p_values:
        raise ValueError(
            'not enough paired deltas to evaluate the first planned look')
    monitored = sequential_monitor(plan, p_values)
    monitored['looks'] = details
    monitored['paired_deltas_available'] = len(finite)
    monitored['plan_max_pairs'] = max_pairs_value
    monitored['margin'] = margin_value
    monitored['descriptive_only'] = (
        'this audit uses the realized seed order; the monitoring plan must '
        'have been frozen before the results, and it never replaces the final '
        'pre-specified comparison')
    return monitored


def audit_paired_contrasts(contrasts: Mapping[str, Sequence[float]],
                           plan: dict, *, margin: float = 0.0) -> dict:
    """Run :func:`sequential_paired_audit` for several named contrasts."""
    if not isinstance(contrasts, Mapping):
        raise ValueError(
            f'contrasts must map names to delta sequences, got {contrasts!r}')
    results = {}
    for name in sorted(contrasts):
        results[str(name)] = sequential_paired_audit(
            contrasts[name], plan, margin=margin)
    stopped = [name for name, result in results.items()
               if result.get('controlled_significant')]
    uncorrected_only = [name for name, result in results.items()
                        if result.get('optional_stopping_only')]
    continuing = [name for name, result in results.items()
                  if result.get('status') == 'continue']
    final_null = [name for name, result in results.items()
                  if result.get('status') == 'not_significant_at_final_look']
    if not results:
        note = 'no contrast had an ordered per-seed delta sequence to audit'
    elif uncorrected_only:
        note = ('at least one contrast crosses an unadjusted per-look alpha '
                'but not its pre-specified spending threshold; those results '
                'are not certified')
    elif stopped:
        note = ('the pre-specified spending plan was crossed; the look at '
                'which it happened is recorded per contrast')
    elif continuing:
        note = 'no pre-specified look has crossed yet; continue to the next look only'
    else:
        note = ('the pre-specified plan did not reject at its final look; do '
                'not add looks after seeing this result')
    return {
        'schema': 'adasquash.sequential_audit.v1',
        'plan': plan,
        'n_contrasts': len(results),
        'n_stopped': len(stopped),
        'stopped_contrasts': stopped,
        'n_optional_stopping_only': len(uncorrected_only),
        'optional_stopping_only_contrasts': uncorrected_only,
        'n_continuing': len(continuing),
        'continuing_contrasts': continuing,
        'n_not_significant_at_final_look': len(final_null),
        'not_significant_at_final_look_contrasts': final_null,
        'contrasts': results,
        'note': note,
    }
