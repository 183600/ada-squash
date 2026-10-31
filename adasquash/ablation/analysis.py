from __future__ import annotations

import hashlib
import itertools
import json

from adasquash.design import exclusion_policy_fingerprint, normalize_exclusion_policy
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, _validate_alpha,
                               benjamini_hochberg_adjust, holm_adjust)
from adasquash.sequential import group_sequential_plan

from .common import _finite_metric, _is_seed_id
from .matrix import (PRIMARY_FACTORIAL_DESIGN, PRIMARY_MAIN_ROW,
                     _parse_three_way_cell_key)


def _factorial_levels(info: dict, factor_names) -> dict:
    """Two levels per factor, from the report or inferred from cell keys."""
    info = info or {}
    declared = dict(info.get('levels') or {})
    cells = info.get('cells') or {}
    levels: dict = {}
    for factor in factor_names:
        values = [str(value) for value in (declared.get(factor) or ())]
        if len(values) != 2:
            values = []
            for key in cells:
                value = _parse_three_way_cell_key(key).get(factor)
                if value is not None and value not in values:
                    values.append(value)
        if len(values) != 2:
            return {}
        levels[factor] = values
    return levels

def build_analysis_plan(rows: dict, hypotheses: dict, matrix_report: dict,
                        baseline_names, model_name: str, seeds,
                        alpha: float=0.05, margin: float=0.0,
                        equivalence_margin: float=0.0,
                        run_schedule: dict | None=None,
                        exclusion_policy: dict | None=None,
                        expected_attrition: float=0.0,
                        sequential_looks: int=0,
                        sequential_spending: str='obrien-fleming') -> dict:
    """Freeze the confirmatory contrast registry *before* any training starts.

    A comparison experiment is only confirmatory if the set of tests is fixed
    in advance. Discovering an extra row after seeing the results and then
    correcting only the surviving family is a well-known way to manufacture a
    "significant" ablation. This function derives every mechanism / baseline /
    factorial / 2x2x2 interaction / QK contrast from the already-validated
    matrix (so it cannot name a contrast the matrix does not isolate), records
    the stated
    hypothesis and test direction, and hashes the whole registry. The hash is
    written next to the results; `study_level_correction` only admits tests
    whose id appears here, and everything else is reported as unregistered
    (exploratory) even when its p-value is small.

    The baseline reference is a *rule* ("best replicated baseline"), not a
    single row name, because which baseline is strongest is a result of the
    run; the realized reference is recorded in `baseline_comparison.json`.
    ``sequential_looks`` freezes an optional group-sequential monitoring plan
    (see `adasquash.sequential`) at the same time: looking at the seeds after
    every run and stopping when p < alpha is optional stopping, so the look
    counts, family alpha and spending function are part of this fingerprint
    whenever monitoring is requested. The default of zero looks adds no key
    and leaves existing pre-registration fingerprints unchanged.
    The exclusion policy is normalized and hashed into the same fingerprint:
    a rerun that quietly changes which runs may be dropped is a changed
    experiment, not a re-analysis. The expected run-attrition assumption used
    to size the design is frozen in the same payload: it is the denominator
    the planned power was computed against, so changing it after seeing how
    many runs were lost would be a post-hoc design change.
    """
    baseline_set = set(baseline_names or [])
    contrasts: list = []
    seen_directional_pairs: set[tuple] = set()

    def _add(entry: dict) -> None:
        # The same candidate/reference contrast can be described by more than
        # one report family (the QK axis is both a leave-one-out removal and a
        # host-level QK contrast). It is one test, so the pre-registration and
        # the study-level Holm family must count it once; otherwise the
        # duplicated p-value doubles its weight and inflates the family size.
        if entry.get('kind') in ('superiority', 'mechanism_removal'):
            pair = (entry.get('candidate'), entry.get('reference'))
            if pair[0] is not None and pair[1] is not None:
                if pair in seen_directional_pairs:
                    return
                seen_directional_pairs.add(pair)
        contrasts.append(entry)

    for name in sorted(rows):
        if name in baseline_set:
            continue
        if not str(hypotheses.get(name) or '').strip():
            # `validate_matrix` already rejects this, but a direct call must
            # not silently drop a contrast from the pre-registration: a plan
            # that omits a row would let the missing hypothesis disappear and
            # later be reported as unregistered rather than as an error.
            raise ValueError(
                f"ablation row '{name}' has no stated scientific hypothesis; "
                "the analysis plan must not silently omit it")
        _add({
            'id': f'baseline:{name}',
            'family': 'baseline',
            'kind': 'superiority',
            'candidate': name,
            'reference': 'best_replicated_baseline',
            'metric': 'val_loss',
            'sided': 'one_sided',
            'hypothesis': hypotheses.get(name),
        })
    qk_axis_report = matrix_report.get('qk_axis') or {}
    qk_on_off_pairs = {
        (info.get('on'), info.get('off')) for info in qk_axis_report.values()
        if isinstance(info, dict)}
    for full_name, mechanisms in sorted(
            (matrix_report.get('leave_one_out') or {}).items()):
        for mechanism, reference in sorted((mechanisms or {}).items()):
            if (full_name, reference) in qk_on_off_pairs:
                # Canonicalised by the host-level QK-axis contrast below: the
                # LOO report still describes the same test, but the plan must
                # not pre-register it twice.
                continue
            _add({
                'id': f'loo:{full_name}:{mechanism}',
                'family': 'leave_one_out',
                'kind': 'mechanism_removal',
                'candidate': full_name,
                'reference': reference,
                'metric': 'val_loss',
                'sided': 'one_sided',
                'hypothesis': (f'removing {mechanism} from {full_name} raises '
                               f'held-out val_loss'),
            })
    for design, info in sorted((matrix_report.get('factorial') or {}).items()):
        factor_names = list((info or {}).get('factors') or {})
        for factor in factor_names:
            _add({
                'id': f'factorial:{design}:{factor}',
                'family': 'factorial',
                'kind': 'main_effect',
                'design': design,
                'factor': factor,
                'metric': 'val_loss',
                'sided': 'one_sided',
                'hypothesis': f'{factor} lowers held-out val_loss in {design}',
            })
        if len(factor_names) == 2:
            first, second = factor_names
            _add({
                'id': f'factorial:{design}:{first}_x_{second}',
                'family': 'factorial',
                'kind': 'interaction',
                'design': design,
                'factor': f'{first}_x_{second}',
                'metric': 'val_loss',
                'sided': 'two_sided',
                'hypothesis': (f'{first} and {second} interact (synergy or '
                               f'antagonism) in {design}'),
            })
            for factor, other in ((first, second), (second, first)):
                for level in ('off', 'on'):
                    _add({
                        'id': f'factorial:{design}:{factor}@{other}={level}',
                        'family': 'conditional_follow_up',
                        'kind': 'simple_effect',
                        'design': design,
                        'factor': factor,
                        'conditional_context': {other: level},
                        'metric': 'val_loss',
                        'sided': 'one_sided',
                        # Pre-registered, but a conditional follow-up must not
                        # double-count a contrast already in the study family
                        # (e.g. Full vs Full-minus-per-channel is the
                        # per-channel simple effect at depth=on).
                        'counted_in_study_family': False,
                        'hypothesis': (f'{factor} lowers held-out val_loss '
                                       f'within {other}={level} in {design} '
                                       f'(interaction-gated follow-up)'),
                    })
    # The 2x2x2 design adds the two interactions that involve the activation
    # family and the three-way term. Main effects are deliberately not
    # re-registered: they are already pre-registered by the two 2x2 designs,
    # and registering the pooled estimate as well would count the same
    # mechanism twice in the study-level Holm family. The pooled 2x2x2
    # interaction terms are different estimands from the within-family 2x2
    # interactions, so they are new tests, not duplicates.
    for design, info in sorted(
            (matrix_report.get('three_way_factorial') or {}).items()):
        factor_names = list((info or {}).get('factors') or [])
        for first, second in itertools.combinations(factor_names, 2):
            _add({
                'id': f'factorial3:{design}:{first}_x_{second}',
                'family': 'factorial3',
                'kind': 'interaction',
                'design': design,
                'factor': f'{first}_x_{second}',
                'metric': 'val_loss',
                'sided': 'two_sided',
                'hypothesis': (f'{first} and {second} interact when the third '
                               f'axis is pooled over in {design}'),
            })
        if len(factor_names) == 3:
            first, second, third = factor_names
            _add({
                'id': (f'factorial3:{design}:{first}_x_{second}_x_{third}'),
                'family': 'factorial3',
                'kind': 'interaction',
                'design': design,
                'factor': f'{first}_x_{second}_x_{third}',
                'metric': 'val_loss',
                'sided': 'two_sided',
                'hypothesis': (f'the {first} x {second} interaction differs '
                               f'between the two levels of {third} in {design}'),
            })
        factor_levels = _factorial_levels(info, factor_names)
        if factor_levels:
            for factor in factor_names:
                others = [name for name in factor_names if name != factor]
                for combo in itertools.product(
                        *(factor_levels[name] for name in others)):
                    context = dict(zip(others, combo))
                    label = '|'.join(
                        f'{name}={context[name]}' for name in others)
                    _add({
                        'id': f'factorial3:{design}:{factor}@{label}',
                        'family': 'conditional_follow_up',
                        'kind': 'simple_effect',
                        'design': design,
                        'factor': factor,
                        'conditional_context': context,
                        'metric': 'val_loss',
                        'sided': 'one_sided',
                        'counted_in_study_family': False,
                        'hypothesis': (f'{factor} lowers held-out val_loss '
                                       f'within {label} in {design} '
                                       f'(interaction-gated follow-up)'),
                    })
    for host, info in sorted((matrix_report.get('qk_axis') or {}).items()):
        info = info or {}
        _add({
            'id': f'qk:{host}:qk_vs_off',
            'family': 'qk_axis',
            'kind': 'superiority',
            'candidate': info.get('on'),
            'reference': info.get('off'),
            'metric': 'val_loss',
            'sided': 'one_sided',
            'hypothesis': f'tanh QK compression lowers held-out val_loss on {host}',
        })
        _add({
            'id': f'qk:{host}:qknorm_vs_qk',
            'family': 'qk_axis',
            'kind': 'comparator',
            'candidate': info.get('comparator'),
            'reference': info.get('on'),
            'metric': 'val_loss',
            'sided': 'one_sided',
            'hypothesis': (f'per-head QK-Norm beats the tanh QK compression on '
                           f'{host}'),
        })
    # Hierarchical gatekeeping: the main proposal's own superiority and
    # mechanism contrasts are the primary family. Only if at least one of
    # them survives its Holm correction may the secondary contrasts (single
    # axis variants, control factorials, QK axis, hyper-parameter sweeps) be
    # tested at full alpha. Gatekeeping keeps the study-wide false-positive
    # rate at alpha without making every exploratory sweep compete with the
    # primary claim for the same alpha budget.
    for entry in contrasts:
        if entry.get('counted_in_study_family') is False:
            entry['tier'] = 2
            entry['tier_role'] = 'conditional_follow_up'
            continue
        primary = (
            (entry['family'] == 'baseline'
             and entry.get('candidate') == PRIMARY_MAIN_ROW)
            or (entry['family'] == 'leave_one_out'
                and entry.get('candidate') == PRIMARY_MAIN_ROW)
            or (entry['family'] == 'factorial'
                and entry.get('design') == PRIMARY_FACTORIAL_DESIGN)
        )
        entry['tier'] = 1 if primary else 2
        entry['tier_role'] = 'primary' if primary else 'secondary'
    # The exclusion policy is a design decision too: changing which runs may
    # be dropped changes the analysed sample, so it belongs in the same
    # fingerprint as the contrasts and the randomized schedule.
    exclusion_policy = normalize_exclusion_policy(exclusion_policy)
    if (not _finite_metric(expected_attrition)
            or not 0.0 <= float(expected_attrition) < 1.0):
        raise ValueError(
            f'expected_attrition must be a finite probability in [0, 1), '
            f'got {expected_attrition!r}')
    if not _finite_metric(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError(
            f'analysis-plan alpha must be a finite number strictly between '
            f'0 and 1, got {alpha!r}')
    if not _finite_metric(margin) or float(margin) < 0.0:
        raise ValueError(
            f'analysis-plan margin must be a finite number >= 0, '
            f'got {margin!r}')
    if not _finite_metric(equivalence_margin) or float(equivalence_margin) < 0.0:
        raise ValueError(
            f'analysis-plan equivalence_margin must be a finite number >= 0, '
            f'got {equivalence_margin!r}')
    if float(equivalence_margin) > 0.0 and float(alpha) >= 0.5:
        raise ValueError(
            'analysis-plan equivalence_margin > 0 requires alpha < 0.5 so '
            f'the TOST interval 1 - 2*alpha is positive, got alpha={alpha!r}')
    seeds = list(seeds or [])
    if not all(_is_seed_id(seed) for seed in seeds):
        raise ValueError(
            f'analysis-plan seeds must be integer ids, got {seeds!r}')
    normalized_seeds = [int(seed) for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError(
            f'analysis-plan seeds must be unique, got {seeds!r}')
    if isinstance(sequential_looks, bool) or not isinstance(sequential_looks, int) \
            or sequential_looks < 0:
        raise ValueError(
            f'sequential_looks must be a non-negative integer, got '
            f'{sequential_looks!r}')
    sequential_plan = None
    if sequential_looks:
        primary_tests = max(
            1, sum(1 for entry in contrasts if entry.get('tier') == 1))
        sequential_plan = group_sequential_plan(
            len(normalized_seeds), n_looks=sequential_looks,
            alpha=float(alpha), n_tests=primary_tests,
            spending=sequential_spending)
    payload = {
        'model': model_name,
        'seeds': normalized_seeds,
        'alpha': float(alpha),
        'margin': float(margin),
        'equivalence_margin': float(equivalence_margin),
        'baseline_pool': sorted(baseline_set),
        'expected_attrition': float(expected_attrition),
        'exclusion_policy': exclusion_policy,
        'exclusion_policy_fingerprint': exclusion_policy_fingerprint(exclusion_policy),
        # The contrast *ids* alone do not identify the experiment: two
        # matrices can reuse the same row names and hypotheses while changing
        # a model knob (e.g. alpha_init), or can map a factorial design to
        # different cells. Hash those definitions too, or a post-hoc model
        # change would not move the pre-registration fingerprint.
        'row_configs': {name: rows.get(name) for name in sorted(rows)},
        'matrix_cells': {
            key: matrix_report.get(key)
            for key in ('leave_one_out', 'factorial', 'three_way_factorial',
                        'qk_axis')
        },
        # The randomized execution order is part of the pre-registration: a
        # schedule chosen after seeing which arms look good is a design
        # degree of freedom exactly like choosing the contrasts afterwards.
        'run_schedule': run_schedule,
        'contrasts': contrasts,
    }
    if sequential_plan is not None:
        payload['sequential_monitoring'] = sequential_plan
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
    return {
        'schema': 'adasquash.analysis_plan.v1',
        'fingerprint': fingerprint,
        'n_contrasts': len(contrasts),
        'contrast_ids': [entry['id'] for entry in contrasts],
        **payload,
        'protocol': (
            'confirmatory contrasts are fixed before training and identified by '
            '`id`; the study-level Holm correction admits only these ids, so a '
            'realized test absent from this plan is exploratory and cannot be '
            'counted as a confirmed claim'),
    }


def _finite_p(value):
    return float(value) if _finite_metric(value) else None


def _conditional_follow_up_reason(entry: dict) -> str:
    status = entry.get('conditional_test_status')
    gate = entry.get('interaction_gate_open')
    return (
        'conditional follow-up (interaction-gated simple effect); '
        'pre-registered but not counted as a new study-level confirmatory '
        f'claim (interaction gate={"open" if gate else "closed"}, '
        f'conditional_test_status={status})')


def study_level_correction(comparisons: dict, plan: dict | None,
                           alpha: float=0.05) -> dict | None:
    """Study-level gatekeeping over the pre-registered confirmatory contrasts.

    The per-family reports each correct within their own family (baseline
    rows, leave-one-out mechanisms, factorial terms, QK pairs). That is the
    right correction for a claim scoped to one family, but the study as a
    whole makes all of those claims at once - and they reuse the same seeds,
    so the tests are correlated. This function pools the pre-registered
    confirmatory p-values into a primary tier (the main proposal's superiority,
    its mechanisms and its main 2x2 factorial) and a secondary tier. The
    primary tier is Holm-corrected first; the secondary tier (including the
    cross-axis 2x2x2 interactions) is corrected and tested only if at least
    one primary contrast survives. Gatekeeping controls the study-wide
    false-positive rate while keeping exploratory single-axis / control / QK
    contrasts from competing with the primary claim for the same alpha budget.
    The family-local adjusted p stays next to the study-level one, and every
    demoted contrast (or secondary contrast not tested because the gate closed)
    is listed explicitly rather than silently dropped. Unregistered contrasts
    remain exploratory, but their p-values get Benjamini-Hochberg q-values so
    the screening set can be read at a stated false-discovery rate instead of
    an uncorrected alpha; those discoveries are explicitly labelled
    hypothesis-generating rather than confirmed.

    Interactions enter with their two-sided p-value (either synergy or
    antagonism is a finding); main effects and superiority contrasts enter
    with the one-sided p-value that matches their directional claim. Tests
    that touch fewer than `MIN_SEEDS_FOR_SIGNIFICANCE` seeds, duplicate runs
    (aliases) and ids absent from the plan are excluded from the family and
    reported with the reason. A focused run with no primary contrast gets
    `primary_gate_status = not_applicable_no_primary_tests` (the secondary
    tier is still tested) rather than a misleading "gate closed". Returns
    `None` without a plan.
    """
    if not isinstance(plan, dict):
        return None
    alpha = _validate_alpha(alpha)
    planned = {entry['id']: entry for entry in plan.get('contrasts', [])
               if isinstance(entry, dict) and entry.get('id')}
    records: list = []

    def _record(cid, family, p, family_p, claim_direction, family_significant,
                *, hypothesis=None, replicated=False, test_basis=None,
                effect=None, excluded_reason=None):
        records.append({
            'id': cid,
            'family': family,
            'p_one_sided_or_two_sided': _finite_p(p),
            'p_family_holm': _finite_p(family_p),
            'claim_direction_observed': bool(claim_direction),
            'family_local_significant': bool(family_significant),
            'replicated_across_seeds': bool(replicated),
            'test_basis': test_basis,
            'effect': effect,
            'hypothesis': hypothesis,
            'excluded_from_study_family': excluded_reason,
            'p_study_holm': None,
            'q_value_bh': None,
            'exploratory_fdr_significant': False,
            'confirmed_at_study_level': False,
            'downgraded_by_study_level': False,
        })

    baseline = comparisons.get('baseline') or {}
    for entry in baseline.get('rows', []) or []:
        name = entry.get('name')
        if not name:
            continue
        _record(
            f'baseline:{name}', 'baseline',
            (entry.get('p_inferential')
             if entry.get('p_inferential') is not None
             else entry.get('p_one_sided')),
            entry.get('p_one_sided_holm'), entry.get('beats_reference'),
            entry.get('beats_reference_significantly'),
            hypothesis=entry.get('hypothesis'),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            # The certifying p-value is the paired shared-seed test, so the
            # study-level effect must be that same paired delta. Falling back
            # to the all-seed difference can even reverse its sign.
            effect=(entry.get('delta_paired_mean')
                    if entry.get('delta_paired_mean') is not None
                    else entry.get('delta_vs_reference')),
            excluded_reason=entry.get('excluded_from_family'))
    loo = comparisons.get('leave_one_out') or {}
    for entry in loo.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('full'):
            continue
        _record(
            f"loo:{entry['full']}:{entry['mechanism']}", 'leave_one_out',
            entry.get('p_one_sided'), entry.get('p_one_sided_holm'),
            entry.get('mechanism_helps'),
            entry.get('mechanism_helps_significantly'),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            effect=(entry.get('delta_paired_mean')
                    if entry.get('delta_paired_mean') is not None
                    else entry.get('delta_full_minus_reference')),
            excluded_reason=entry.get('excluded_from_family'))
    factorial = comparisons.get('factorial') or {}
    for entry in factorial.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('design'):
            continue
        is_interaction = entry.get('kind') == 'interaction'
        is_simple = (entry.get('kind') == 'simple_effect'
                     or entry.get('analysis_role') == 'conditional_follow_up')
        _record(
            f"factorial:{entry['design']}:{entry['effect']}",
            'conditional_follow_up' if is_simple else 'factorial',
            entry.get('p_two_sided') if is_interaction else entry.get('p_one_sided'),
            (entry.get('p_two_sided_holm') if is_interaction
             else entry.get('p_one_sided_holm')),
            (entry.get('interaction_direction') not in
             (None, 'additive_or_within_margin'))
            if is_interaction else entry.get('effect_helps'),
            (entry.get('interaction_differs_significantly') if is_interaction
             else entry.get('effect_helps_significantly')),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            effect=entry.get('delta_seed_mean'),
            excluded_reason=(_conditional_follow_up_reason(entry)
                             if is_simple else None))
    factorial3 = comparisons.get('factorial3') or {}
    for entry in factorial3.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('design'):
            continue
        is_interaction = entry.get('kind') == 'interaction'
        is_simple = (entry.get('kind') == 'simple_effect'
                     or entry.get('analysis_role') == 'conditional_follow_up')
        _record(
            f"factorial3:{entry['design']}:{entry['effect']}",
            'conditional_follow_up' if is_simple else 'factorial3',
            entry.get('p_two_sided') if is_interaction else entry.get('p_one_sided'),
            (entry.get('p_two_sided_holm') if is_interaction
             else entry.get('p_one_sided_holm')),
            (entry.get('interaction_direction') not in
             (None, 'additive_or_within_margin'))
            if is_interaction else entry.get('effect_helps'),
            (entry.get('interaction_differs_significantly') if is_interaction
             else entry.get('effect_helps_significantly')),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            effect=entry.get('delta_seed_mean'),
            excluded_reason=(_conditional_follow_up_reason(entry)
                             if is_simple else None))
    qk = comparisons.get('qk_axis') or {}
    for entry in qk.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('host'):
            continue
        _record(
            f"qk:{entry['host']}:{entry['contrast']}", 'qk_axis',
            entry.get('p_one_sided'), entry.get('p_one_sided_holm'),
            entry.get('effect_helps'), entry.get('effect_helps_significantly'),
            hypothesis=entry.get('hypothesis'),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            # `delta_paired_mean` is always present on a QK row (None until
            # the paired test runs), so `dict.get(key, fallback)` would drop
            # a finite descriptive candidate-reference delta on untested rows.
            effect=(entry.get('delta_paired_mean')
                    if entry.get('delta_paired_mean') is not None
                    else entry.get('delta_candidate_minus_reference')))

    registered: list = []
    unregistered: list = []
    for record in records:
        planned_entry = planned.get(record['id'])
        if planned_entry is None:
            record['excluded_from_study_family'] = (
                'not present in the pre-registered analysis plan '
                '(exploratory/unregistered)')
            unregistered.append(record['id'])
        else:
            record['plan_kind'] = planned_entry.get('kind')
            registered.append(record)
    unregistered_records = [record for record in records
                            if record['id'] in set(unregistered)
                            and record['p_one_sided_or_two_sided'] is not None]
    unregistered_q = benjamini_hochberg_adjust(
        [record['p_one_sided_or_two_sided']
         for record in unregistered_records])
    for record, q_value in zip(unregistered_records, unregistered_q):
        record['q_value_bh'] = q_value
        record['exploratory_fdr_significant'] = bool(
            q_value is not None and q_value < alpha)
    exploratory_fdr = {
        'method': 'benjamini-hochberg-step-up',
        'alpha': float(alpha),
        'n_tested': len(unregistered_records),
        'n_discoveries': sum(
            record['exploratory_fdr_significant']
            for record in unregistered_records),
        'discoveries': sorted(
            record['id'] for record in unregistered_records
            if record['exploratory_fdr_significant']),
        'note': ('q-values control the false-discovery rate among the '
                 'exploratory/unregistered contrasts only; they are '
                 'hypothesis-generating and do not become confirmatory claims. '
                 'The BH guarantee assumes independence or positive regression '
                 'dependence'),
    }
    resolved_ids = {record['id'] for record in registered}
    unresolved = sorted(set(planned) - resolved_ids)
    eligible = [record for record in registered
                if record.get('excluded_from_study_family') is None
                and (planned.get(record['id']) or {}).get(
                    'counted_in_study_family', True) is not False
                and record.get('replicated_across_seeds')
                and record.get('p_one_sided_or_two_sided') is not None]
    eligible_ids = {record['id'] for record in eligible}
    for record in registered:
        if record['id'] not in eligible_ids:
            if record.get('excluded_from_study_family') is None:
                if (planned.get(record['id']) or {}).get(
                        'counted_in_study_family', True) is False:
                    record['excluded_from_study_family'] = (
                        'pre-registered conditional follow-up '
                        '(counted_in_study_family=False); its parent factorial '
                        'term already carries the study-level test')
                elif not record.get('replicated_across_seeds'):
                    record['excluded_from_study_family'] = (
                        f'fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds; '
                        'descriptive only')
                elif record.get('p_one_sided_or_two_sided') is None:
                    record['excluded_from_study_family'] = 'no testable p-value'
    planned_primary_ids = {
        contrast_id for contrast_id, entry in planned.items()
        if (entry or {}).get('tier', 1) == 1}
    resolved_primary = [record for record in registered
                        if record['id'] in planned_primary_ids]
    primary = [record for record in eligible
               if record['id'] in planned_primary_ids]
    secondary = [record for record in eligible
                 if (planned.get(record['id']) or {}).get('tier', 1) == 2]
    primary_adjusted = holm_adjust(
        [record['p_one_sided_or_two_sided'] for record in primary])
    for record, p_holm in zip(primary, primary_adjusted):
        record['tier'] = 'primary'
        record['p_study_holm'] = p_holm
        record['confirmed_at_study_level'] = bool(
            record['claim_direction_observed'] and p_holm is not None
            and p_holm < alpha)
    primary_gate_open = any(
        record['confirmed_at_study_level'] for record in primary)
    # A focused run (e.g. only the QK rows) has no resolved primary contrast
    # at all. That is not a failed gate: the gate is only meaningful when the
    # primary family was part of the study. But a primary contrast that *was*
    # planned and resolved yet became ineligible (e.g. under-replicated) is a
    # closed gate, not an absent one: it must not license secondary claims.
    if not planned_primary_ids or not resolved_primary:
        primary_gate_status = 'not_applicable_no_primary_tests'
    elif not primary or not primary_gate_open:
        primary_gate_status = 'closed'
    else:
        primary_gate_status = 'open'
    secondary_allowed = primary_gate_status in ('open',
                                                'not_applicable_no_primary_tests')
    if secondary_allowed:
        secondary_adjusted = holm_adjust(
            [record['p_one_sided_or_two_sided'] for record in secondary])
    else:
        secondary_adjusted = [None] * len(secondary)
    for record, p_holm in zip(secondary, secondary_adjusted):
        record['tier'] = 'secondary'
        record['p_study_holm'] = p_holm
        record['confirmed_at_study_level'] = bool(
            secondary_allowed and record['claim_direction_observed']
            and p_holm is not None and p_holm < alpha)
        if not secondary_allowed:
            record['gate_status'] = (
                'secondary family not tested: no primary contrast survived '
                'its gatekeeping correction, so the study has no confirmed '
                'primary claim to license secondary claims')
    for record in records:
        if 'tier' not in record:
            planned_entry = planned.get(record['id'])
            if planned_entry is None:
                record['tier'] = 'unregistered'
            elif planned_entry.get('counted_in_study_family', True) is False:
                record['tier'] = 'conditional_follow_up'
            else:
                record['tier'] = ('primary' if planned_entry.get('tier', 1) == 1
                                  else 'secondary')
        record['downgraded_by_study_level'] = bool(
            record['family_local_significant']
            and not record['confirmed_at_study_level']
            and record.get('tier') != 'conditional_follow_up')
    secondary_gate_closed = sorted(
        record['id'] for record in secondary if not secondary_allowed)
    local_significant = sorted(record['id'] for record in registered
                               if record.get('family_local_significant'))
    study_significant = sorted(record['id'] for record in primary + secondary
                               if record.get('confirmed_at_study_level'))
    downgraded = [
        {'id': record['id'],
         'family': record['family'],
         'tier': record.get('tier'),
         'p_family_holm': record['p_family_holm'],
         'p_study_holm': record['p_study_holm']}
        for record in primary + secondary if record['downgraded_by_study_level']
    ]
    records.sort(key=lambda record: record['id'])
    return {
        'schema': 'adasquash.study_level_tests.v1',
        'analysis_plan_fingerprint': plan.get('fingerprint'),
        'alpha': float(alpha),
        'n_planned_contrasts': len(planned),
        'n_registered_tests': len(registered),
        'n_in_study_family': len(primary) + len(secondary),
        'n_primary_tests': len(primary),
        'n_secondary_tests': len(secondary),
        'n_conditional_follow_up': sum(
            record.get('tier') == 'conditional_follow_up'
            for record in records),
        'primary_gate_open': bool(primary_gate_open),
        'primary_gate_status': primary_gate_status,
        'n_confirmed_primary': sum(
            record.get('confirmed_at_study_level') for record in primary),
        'n_confirmed_secondary': sum(
            record.get('confirmed_at_study_level') for record in secondary),
        'n_confirmed_at_study_level': len(study_significant),
        'family_local_significant': local_significant,
        'study_level_significant': study_significant,
        'downgraded_by_study_level': downgraded,
        'secondary_not_tested_gate_closed': secondary_gate_closed,
        'unregistered_contrasts': sorted(unregistered),
        'exploratory_fdr': exploratory_fdr,
        'unresolved_plan_contrasts': unresolved,
        'protocol': (
            'hierarchical Holm gatekeeping over the pre-registered confirmatory '
            'contrasts. Tier 1 (primary) = the main proposal vs the best baseline, '
            'its leave-one-out mechanisms, and the main softsign factorial. Only '
            'if >= 1 primary contrast survives does the secondary tier (single-axis '
            'variants, control factorials, QK axis, hyper-parameter sweeps) get '
            'tested at alpha; both tiers are Holm-corrected within themselves. '
            '`confirmed_at_study_level` additionally requires the claimed '
            'direction. Interactions use the two-sided p-value, everything else '
            'the one-sided p-value matching its directional hypothesis. '
            'Family-local adjusted p-values are retained, so a demotion by the '
            'study-level correction is visible rather than silent. '
            'Exploratory/unregistered contrasts are never promoted: they receive '
            'Benjamini-Hochberg q-values for a stated FDR and remain '
            'hypothesis-generating. Pre-registered interaction-gated simple '
            'effects are conditional follow-ups: they are excluded from the '
            'study family and never counted as new confirmatory claims'),
        'rows': records,
    }


def print_study_level_correction(comparison: dict | None) -> None:
    if comparison is None:
        print('[study] no analysis plan; study-level multiplicity correction skipped')
        return
    print(f"[study] {comparison['n_in_study_family']}/{comparison['n_planned_contrasts']} "
          f"pre-registered contrast(s) tested "
          f"(primary={comparison['n_primary_tests']}, "
          f"secondary={comparison['n_secondary_tests']}, "
          f"conditional_follow_up={comparison.get('n_conditional_follow_up', 0)}); "
          f"family-local significant={len(comparison['family_local_significant'])}, "
          f"study-level confirmed={comparison['n_confirmed_at_study_level']} "
          f"(primary gate {comparison.get('primary_gate_status', 'open').upper()}; "
          f"plan {str(comparison.get('analysis_plan_fingerprint'))[:12]})")
    for entry in comparison.get('downgraded_by_study_level') or []:
        print(f"[study] DOWNGRADED {entry['id']}: family p_holm="
              f"{entry['p_family_holm']} -> study p_holm={entry['p_study_holm']} "
              f"(not confirmed once the whole study is corrected)")
    if comparison.get('secondary_not_tested_gate_closed'):
        print(f"[study] secondary contrasts not tested (primary gate closed): "
              f"{comparison['secondary_not_tested_gate_closed']}")
    if comparison.get('unregistered_contrasts'):
        ids = comparison['unregistered_contrasts']
        preview = f"{ids[:8]}{' ...' if len(ids) > 8 else ''}"
        print(f"[study] unregistered/exploratory ({len(ids)}, not in the plan, "
              f"not counted): {preview}")
        fdr = comparison.get('exploratory_fdr') or {}
        print(f"[study] exploratory BH-FDR: {fdr.get('n_discoveries', 0)}/"
              f"{fdr.get('n_tested', 0)} q<{fdr.get('alpha', comparison['alpha'])} "
              f"(hypothesis-generating only)")
    if comparison.get('unresolved_plan_contrasts'):
        ids = comparison['unresolved_plan_contrasts']
        preview = f"{ids[:8]}{' ...' if len(ids) > 8 else ''}"
        print(f"[study] pre-registered but not realized ({len(ids)}; full list in "
              f"study_level_tests.json): {preview}")


