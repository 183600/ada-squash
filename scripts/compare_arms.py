from __future__ import annotations
"""Seed-paired comparison of Phase-1 arm aggregates.

`train_gpt2.py --seeds` writes one `<output-dir>/aggregate.json` per arm with
Mean +/- Std across independent seed runs. Those files are the raw material
for the Phase-1 claim, but two separate Means and Stds do not by themselves
say whether a gap is larger than run-to-run noise. This tool loads two or
more aggregates, pairs them by training seed, and reports:

* Mean +/- Std (and stderr) per arm, with the same completion/data-version
  checks the aggregate already records;
* the paired one-sided Student-t test of H1 "candidate has lower held-out
  loss than the reference", with a 95% CI, seed-direction counts and the
  minimum detectable delta at 80% power;
* the ratio of the delta to the across-seed std (the scale check: a 0.01
  loss gap under a 0.05 seed std is not a result);
* a leave-one-seed-out jackknife (does the sign survive dropping any one
  seed?) and, with `--equivalence-margin <loss>`, a TOST equivalence test, so
  "the two arms are the same" is a tested claim rather than a non-significant
  superiority result;
* a 95% *prediction* interval for the next seed's paired delta, alongside the
  confidence interval for the mean: the CI says where the mean is, the PI says
  how far a replication can still land;
* both loss and perplexity (`exp(loss)`) deltas, because the papers quote
  PPL while the seed variance lives in loss;
* a protocol audit across arms: identical model shape, fixed final step and
  resolved data version. If the arms differ on those, the comparison is
  reported but never certified, no matter how small the p-value;
* a reference-selection audit. The default reference is the best-looking arm,
  and an explicit `--reference` is normally fixed in advance; when
  `--reference-selection best-of-pool` (or `--reference-pool`) declares a
  data-selected reference, the candidate must beat every pool member
  (intersection-union max-p) before entering the Holm family. Reporting only
  the comparison against the winner would make the claim conditional on the
  selection.

Only the paired test on >= 3 shared per-seed means is certifiable; two shared
seeds are descriptive only (df = 1). p-values are Holm-adjusted over the
candidate family against one reference. This is the Phase-1 counterpart of
`scripts/compare_sealed_tests.py`, which pairs the sealed test split.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust,
                               leave_one_out_jackknife, mean_prediction_interval,
                               mean_std_stderr, max_p_combination,
                               paired_design_summary, paired_randomization_test,
                               paired_t_one_sided, student_t_ppf,
                               tost_design_summary, tost_paired)
from adasquash.repro import compare_train_streams, training_recipe_mismatches

_LOSS_METRICS = ('final_val_loss', 'best_val_loss', 'test_loss')


def _finite(value) -> bool:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)):
        return False
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        # A JSON integer can exceed the finite float range; it is not a
        # usable loss, but it must not crash the comparison.
        return False
    return math.isfinite(converted)


def _finite_number(value):
    """The finite float value of a real scalar, or None for anything else."""
    if not _finite(value):
        return None
    return float(value)


def _seed_values(agg: dict, metric: str) -> dict[int, float]:
    """{seed: finite metric} from one aggregate's `per_seed` records."""
    out: dict[int, float] = {}
    per_seed = agg.get('per_seed') if isinstance(agg, dict) else None
    if isinstance(per_seed, list):
        for record in per_seed:
            if not isinstance(record, dict):
                continue
            seed = record.get('seed')
            if (not isinstance(seed, int) or isinstance(seed, bool)):
                continue
            value = record.get(metric)
            if _finite(value):
                out[int(seed)] = float(value)
    return out


def _seed_record_problems(agg: dict) -> list[str]:
    """Structural problems in an aggregate's per-seed records.

    Two records for one seed are one underlying run, not two replications;
    collapsing them in `_seed_values` could otherwise lift a two-seed arm over
    the three-seed certification threshold. A malformed seed id is unknown,
    never a seed to skip silently.
    """
    per_seed = agg.get('per_seed') if isinstance(agg, dict) else None
    if not isinstance(per_seed, list):
        return []
    seen = set()
    duplicates = set()
    malformed = []
    for record in per_seed:
        if not isinstance(record, dict):
            malformed.append(repr(record))
            continue
        seed = record.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int):
            malformed.append(repr(seed))
            continue
        if seed in seen:
            duplicates.add(seed)
        seen.add(seed)
    problems = []
    if duplicates:
        problems.append(f'duplicate per-seed id(s) {sorted(duplicates)}')
    if malformed:
        problems.append(f'malformed per-seed id(s) {malformed}')
    declared = agg.get('seeds')
    if declared is not None:
        if not isinstance(declared, list):
            problems.append('declared seed list is not a list')
        else:
            malformed_declared = [value for value in declared
                                  if isinstance(value, bool)
                                  or not isinstance(value, int)]
            if malformed_declared:
                problems.append(
                    f'malformed declared seed id(s) {malformed_declared}')
            else:
                declared_set = set(declared)
                if len(declared_set) != len(declared):
                    problems.append(
                        f'duplicate declared seed id(s) {sorted(declared)}')
                missing = sorted(declared_set - seen)
                extra = sorted(seen - declared_set)
                if missing or extra:
                    problems.append(
                        'per-seed records do not cover the declared seed set '
                        f'(missing {missing}, extra {extra})')
    return problems


def _safe_exp(value):
    """exp(loss) for PPL reporting; a finite but huge loss stays visible."""
    try:
        return math.exp(float(value))
    except (OverflowError, ValueError):
        return math.inf


def _per_seed_ids(agg: dict) -> set[str]:
    """Seeds named by the aggregate's per-seed records, normalized to str."""
    per_seed = agg.get('per_seed') if isinstance(agg, dict) else None
    out = set()
    if isinstance(per_seed, list):
        for record in per_seed:
            if not isinstance(record, dict):
                continue
            seed = record.get('seed')
            if isinstance(seed, int) and not isinstance(seed, bool):
                out.add(str(seed))
    return out


def _provenance_covers_seeds(versions: dict, agg: dict) -> bool:
    """Every measured seed must have an entry; JSON keys may be strings."""
    seeds = _per_seed_ids(agg)
    if not seeds:
        return False
    return {str(key) for key in versions} == seeds


def _data_version_signature(agg: dict):
    versions = agg.get('data_versions_by_seed')
    if not isinstance(versions, dict) or not versions:
        return None
    # A partial map is unknown provenance, not a shorter list of versions. The
    # aggregate writer emits one entry per seed; a stale/hand-written record
    # must not certify when one seed's snapshot was never resolved.
    if not _provenance_covers_seeds(versions, agg):
        return None
    values = list(versions.values())
    if any(value is None for value in values):
        return None
    try:
        signatures = {json.dumps(value, sort_keys=True) for value in values}
    except (TypeError, ValueError):
        # A non-JSON value (e.g. a set) is unknown provenance, not a reason to
        # crash the comparison or let two malformed arms compare equal.
        return None
    if len(signatures) != 1:
        return 'MIXED'
    return next(iter(signatures))


def _data_ledger_signature(agg: dict):
    """Cross-split ledger identity recorded for every seed, or None if unknown.

    The data *version* says which snapshot the loader requested; the ledger
    content says which documents that snapshot actually held out after
    deduplication/exclusion. Two arms can share a version and still validate
    on different documents if a ledger file changed, so this must be part of
    the cross-arm protocol identity.
    """
    ledgers = agg.get('data_ledgers_by_seed')
    if not isinstance(ledgers, dict) or not ledgers:
        return None
    if not _provenance_covers_seeds(ledgers, agg):
        return None
    values = list(ledgers.values())
    # A missing per-seed ledger record is unknown, not an empty ledger.
    if any(not isinstance(value, dict) for value in values):
        return None
    try:
        signatures = {json.dumps(value, sort_keys=True) for value in values}
    except (TypeError, ValueError):
        # Non-JSON ledger content is unknown, not a comparable identity.
        return None
    if len(signatures) != 1:
        return 'MIXED'
    return next(iter(signatures))


_SHAPE_KEYS = ('n_layers', 'dim', 'n_heads')


def _shape_signature(agg: dict):
    shapes = agg.get('shapes_by_seed')
    if not isinstance(shapes, dict) or not shapes:
        return None
    # A shape record with a missing/None field is unknown, not a shape. The
    # aggregate writer only emits complete shapes, so this is the malformed /
    # stale-record path; it must not compare equal to another unknown shape.
    # Only the architecture keys define the controlled shape: `shapes_by_seed`
    # also carries `n_params`, which legitimately differs between two norm
    # variants (DyT-CD has a few more parameters). Folding it into the
    # signature would reject every cross-norm comparison as a "shape" change.
    if any(not isinstance(shape, dict)
           or any(not isinstance(shape.get(key), int)
                  or isinstance(shape.get(key), bool)
                  or shape.get(key) <= 0
                  for key in _SHAPE_KEYS)
           for shape in shapes.values()):
        return None
    if not _provenance_covers_seeds(shapes, agg):
        return None
    signatures = {
        json.dumps({key: shape[key] for key in _SHAPE_KEYS}, sort_keys=True)
        for shape in shapes.values()}
    if len(signatures) != 1:
        return 'MIXED'
    return next(iter(signatures))


def _signature_unknown(value) -> bool:
    return value is None or value == 'MIXED'


def _model_selector(agg: dict):
    """The aggregate's model selector, or None when it is malformed.

    The selector is compared across arms and used as a set member. A list/dict
    spelling is unhashable (crashing the comparison) and is not a known model
    identity, so it must fail closed as unknown rather than certify.
    """
    model = agg.get('model') if isinstance(agg, dict) else None
    return model if isinstance(model, str) and model else None


def _final_step(agg: dict):
    per_seed = agg.get('per_seed') if isinstance(agg, dict) else None
    if not isinstance(per_seed, list):
        return None
    steps = []
    for record in per_seed:
        if not isinstance(record, dict):
            continue
        step = record.get('final_val_step')
        # A bool/float/list spelling is not a step count. Comparison against
        # the set form let `1` and `True` collapse into one value, so whether a
        # malformed aggregate looked consistent depended on record order.
        if not isinstance(step, int) or isinstance(step, bool):
            return None
        steps.append(step)
    if not steps or len(set(steps)) != 1:
        return None
    step = steps[0]
    return step if step > 0 else None


def _train_stream_for_seed(streams, seed):
    """A per-seed stream record from an aggregate loaded from JSON.

    `aggregate.json` is written by `json.dump`, so integer seed keys come back
    as strings; a direct `mapping.get(42)` would silently treat a recorded
    stream as missing. Accept either spelling.
    """
    if not isinstance(streams, dict):
        return None
    value = streams.get(seed)
    if value is None:
        value = streams.get(str(seed))
    return value


def _arm_issues(name: str, agg: dict, metric: str) -> list[str]:
    issues = []
    if not isinstance(agg, dict):
        return [f'arm {name!r} is not a JSON object']
    values = _seed_values(agg, metric)
    if len(values) < MIN_SEEDS_FOR_SIGNIFICANCE:
        issues.append(f'arm {name!r} has {len(values)} finite {metric} seed(s), '
                      f'fewer than {MIN_SEEDS_FOR_SIGNIFICANCE}')
    if agg.get('missing_seeds'):
        issues.append(f"arm {name!r} has missing seeds {agg['missing_seeds']}")
    if agg.get('invalid_seeds'):
        issues.append(f"arm {name!r} has invalid seeds {agg['invalid_seeds']}")
    for problem in _seed_record_problems(agg):
        issues.append(f'arm {name!r} has {problem}')
    # Only an explicit True certifies completeness. An aggregate written by an
    # older/malformed aggregator without the flag is unknown, not complete;
    # `is False` let such an arm through and certify a partial run.
    if agg.get('complete') is not True:
        issues.append(f"arm {name!r} was not marked complete by its aggregator")
    if agg.get('data_ledgers_complete') is False:
        issues.append(f'arm {name!r} mixes missing/known cross-split ledger identities')
    if agg.get('data_versions_consistent') is False:
        issues.append(f'arm {name!r} mixes resolved data versions across seeds')
    if agg.get('shapes_consistent') is False:
        issues.append(f'arm {name!r} mixes model shapes across seeds')
    return issues


def _manifest_replication_problem(agg: dict, manifest: dict | None) -> str | None:
    """Why a sibling replication manifest cannot certify this aggregate.

    A recipe block alone does not tie the loaded `aggregate.json` to the run
    that produced it: the manifest can be failed/unfinished, describe a
    different seed set, or belong to a stale aggregate at the same path.
    Certification must fail closed in all of those cases.
    """
    if not isinstance(manifest, dict):
        return 'run_manifest.json is missing or unreadable'
    status = manifest.get('status')
    if status != 'finished':
        return (f"run_manifest.json status is {status!r}, not 'finished'; the "
                'replication did not complete cleanly')
    recorded_seeds = manifest.get('seeds')
    aggregate_seeds = agg.get('seeds') if isinstance(agg, dict) else None
    if not isinstance(recorded_seeds, list) or not isinstance(aggregate_seeds, list):
        return 'run_manifest.json does not record the replicated seed set'
    def _strict_seed_list(values):
        out = []
        for seed in values:
            # A JSON float/bool is not a seed id; `int(42.5)` would silently
            # truncate it and `int(True)` would accept a boolean as one seed.
            if isinstance(seed, bool) or not isinstance(seed, int):
                return None
            out.append(int(seed))
        if len(set(out)) != len(out):
            return None
        return sorted(out)

    manifest_seed_set = _strict_seed_list(recorded_seeds)
    aggregate_seed_set = _strict_seed_list(aggregate_seeds)
    if manifest_seed_set is None or aggregate_seed_set is None:
        return 'run_manifest.json has a malformed seed list'
    if manifest_seed_set != aggregate_seed_set:
        return (f'run_manifest.json seeds {manifest_seed_set} differ from the '
                f'aggregate seeds {aggregate_seed_set}')
    recorded_aggregate = manifest.get('aggregate')
    if not isinstance(recorded_aggregate, dict):
        return 'run_manifest.json does not record the aggregate it produced'
    if recorded_aggregate != agg:
        return ('run_manifest.json aggregate differs from the loaded aggregate.json; '
                'the files are stale or belong to different runs')
    return None


def compare_arms(arms: dict[str, dict], reference: str | None=None,
                 metric: str='final_val_loss', alpha: float=0.05,
                 margin: float=0.0, manifests: dict[str, dict] | None=None,
                 require_recipe: bool=True,
                 equivalence_margin: float=0.0,
                 require_train_stream_identity: bool=False,
                 reference_selection: str | None=None,
                 reference_pool: list[str] | None=None) -> dict:
    """Paired, Holm-adjusted comparison of arm aggregates.

    `arms` maps a display name to an aggregate.json object. The reference is
    the explicitly named arm, or the best arm with >= 3 seeds; each other arm
    is tested against it on the seeds both arms actually share. The result
    separates "delta < 0" (point estimate) from "certified" (>= 3 shared
    seeds, clean arms, identical protocol, and Holm-adjusted p < alpha).

    Reference selection is itself a result-analysis decision. When the
    reference was chosen *after* seeing the losses (`reference=None`, or
    `reference_selection='best-of-pool'`), the reported p-value is for the
    best-looking member of a pool, not for a pre-specified comparator.
    `p_one_sided_selection_adjusted` applies an intersection-union max-p
    over that pool (`max_p_combination`), and the certification family uses that
    adjusted value; `p_one_sided` remains the raw test against the selected
    reference so both quantities are auditable. An explicit `reference` is
    treated as pre-specified unless `reference_selection='best-of-pool'` (or
    `reference_pool`) says otherwise.
    """
    if not isinstance(arms, dict) or len(arms) < 2:
        raise ValueError('compare_arms needs at least two named aggregates')
    if metric not in _LOSS_METRICS:
        raise ValueError(f'metric must be one of {_LOSS_METRICS}')
    alpha_value = _finite_number(alpha)
    if alpha_value is None or not 0.0 < alpha_value < 1.0:
        raise ValueError('alpha must be finite and strictly between 0 and 1')
    margin_value = _finite_number(margin)
    if margin_value is None or margin_value < 0.0:
        raise ValueError('margin must be finite and >= 0')
    equivalence_value = _finite_number(equivalence_margin)
    if equivalence_value is None or equivalence_value < 0.0:
        raise ValueError('equivalence_margin must be finite and >= 0 '
                         '(0 disables TOST)')
    alpha, margin, equivalence_margin = (
        alpha_value, margin_value, equivalence_value)
    if equivalence_margin > 0.0 and alpha >= 0.5:
        raise ValueError(
            'a positive equivalence_margin requires alpha < 0.5 so the TOST '
            f'confidence interval 1 - 2*alpha is positive, got alpha={alpha!r}')
    if reference_selection not in (None, 'prespecified', 'best-of-pool'):
        raise ValueError(
            "reference_selection must be None, 'prespecified' or 'best-of-pool', "
            f'got {reference_selection!r}')
    if reference_pool is not None:
        if (not isinstance(reference_pool, (list, tuple))
                or not reference_pool
                or any(not isinstance(name, str) or not name for name in reference_pool)):
            raise ValueError('reference_pool must be a non-empty list of arm names')
        if reference_selection == 'prespecified':
            raise ValueError(
                'reference_pool declares a data-selected reference pool; it '
                "cannot be combined with reference_selection='prespecified'")

    prepared = {}
    for name, agg in arms.items():
        vals = _seed_values(agg, metric)
        stats = mean_std_stderr(list(vals.values()))
        prepared[name] = {
            'name': name,
            'norm': agg.get('norm') if isinstance(agg, dict) else None,
            'model': _model_selector(agg),
            'values': vals,
            'mean': None if stats is None else stats['mean'],
            'std': None if stats is None or stats['n'] < 2 else stats['std'],
            'stderr': None if stats is None or stats['n'] < 2 else stats['stderr'],
            'n_seeds': len(vals),
            'seeds': sorted(vals),
            'issues': _arm_issues(name, agg, metric),
            'curve_classification_counts': (agg.get('curve_classification_counts')
                                            if isinstance(agg, dict) else None),
            'train_stream_by_seed': (agg.get('train_stream_by_seed')
                                     if isinstance(agg, dict)
                                     and isinstance(agg.get('train_stream_by_seed'), dict)
                                     else {}),
        }
    manifests = manifests or {}
    manifest_problems: dict[str, str | None] = {}
    for name in list(prepared):
        manifest = manifests.get(name) if isinstance(manifests, dict) else None
        hyper = manifest.get('hyperparameters') if isinstance(manifest, dict) else None
        prepared[name]['recipe_recorded'] = isinstance(hyper, dict)
        manifest_problems[name] = _manifest_replication_problem(arms[name], manifest)
    reference_was_explicit = reference is not None
    default_selection_pool: list[str] = []
    if reference is not None:
        if reference not in prepared:
            raise ValueError(f'reference {reference!r} is not one of {sorted(prepared)}')
        if reference_selection == 'prespecified' and reference_pool is not None:
            raise ValueError('a pre-specified reference cannot also declare a pool')
    else:
        if reference_selection == 'prespecified':
            raise ValueError(
                "reference_selection='prespecified' requires an explicit reference")
        if reference_pool is not None:
            missing = [name for name in reference_pool if name not in prepared]
            if missing:
                raise ValueError(
                    f'reference_pool names unknown arm(s) {missing}; known: '
                    f'{sorted(prepared)}')
            pool = [prepared[name] for name in dict.fromkeys(reference_pool)
                    if prepared[name]['mean'] is not None]
        else:
            replicated = [row for row in prepared.values()
                          if row['n_seeds'] >= MIN_SEEDS_FOR_SIGNIFICANCE
                          and row['mean'] is not None]
            pool = replicated or [row for row in prepared.values()
                                  if row['mean'] is not None]
        if not pool:
            raise ValueError('no arm in the reference pool has a usable '
                             f'{metric} mean')
        reference = min(pool, key=lambda row: row['mean'])['name']
        default_selection_pool = sorted(row['name'] for row in pool)
    ref = prepared[reference]
    if ref['mean'] is None:
        raise ValueError(f'reference arm {reference!r} has no usable {metric}')
    reference_data_dependent = bool(
        not reference_was_explicit
        or reference_selection == 'best-of-pool'
        or reference_pool is not None)

    # Protocol identity across arms: only the normalization may differ. The
    # model shape, fixed final step and resolved data version are the
    # controlled variables the Phase-1 configs promise to hold constant.
    signatures = {name: {'shape': _shape_signature(agg),
                         'final_step': _final_step(agg),
                         'data_version': _data_version_signature(agg),
                         'data_ledger': _data_ledger_signature(agg),
                         'model': prepared[name]['model']}
                  for name, agg in arms.items()}
    shape_values = {sig['shape'] for sig in signatures.values()}
    step_values = {sig['final_step'] for sig in signatures.values()}
    data_values = {sig['data_version'] for sig in signatures.values()}
    ledger_values = {sig['data_ledger'] for sig in signatures.values()}
    model_values = {sig['model'] for sig in signatures.values()}
    protocol_problems = []
    def _has_unknown(values) -> bool:
        return any(_signature_unknown(value) for value in values)

    if len(shape_values) != 1 or _has_unknown(shape_values):
        protocol_problems.append(f'model shapes differ or are unknown: {sorted(map(str, shape_values))}')
    if len(step_values) != 1 or _has_unknown(step_values):
        protocol_problems.append(f'final step differs or is unknown: {sorted(map(str, step_values))}')
    if len(data_values) != 1 or _has_unknown(data_values):
        protocol_problems.append(f'data versions differ or are unknown: {sorted(map(str, data_values))}')
    if len(ledger_values) != 1 or _has_unknown(ledger_values):
        protocol_problems.append(
            'cross-split ledger identities differ or are unknown: '
            f'{sorted(map(str, ledger_values))}')
    if len(model_values) != 1 or _has_unknown(model_values):
        protocol_problems.append(f'model selectors differ or are unknown: {sorted(map(str, model_values))}')
    protocol_ok = not protocol_problems

    # The selection pool is the set of arms the reported reference was chosen
    # from. For the default (best replicated arm) it is the same set the
    # selection rule used, restricted to arms that share the reference's
    # controlled protocol: an arm at another scale could not have supplied a
    # like-for-like reference. When the caller names the pool explicitly we
    # keep it verbatim, and a missing/incomparable member makes the
    # selection-adjusted claim fail closed rather than silently shrink m.
    selection_pool: list[str] = []
    if reference_data_dependent:
        if reference_pool is not None:
            selection_pool = list(dict.fromkeys(reference_pool))
            missing = [name for name in selection_pool if name not in prepared]
            if missing:
                raise ValueError(
                    f'reference_pool names unknown arm(s) {missing}; known: '
                    f'{sorted(prepared)}')
            if reference not in selection_pool:
                raise ValueError(
                    f'reference {reference!r} is not in reference_pool '
                    f'{selection_pool}; a data-selected reference must belong '
                    'to the pool it was selected from')
        elif default_selection_pool:
            selection_pool = [
                name for name in default_selection_pool
                if signatures[name] == signatures[reference]]
            if reference not in selection_pool:
                selection_pool.append(reference)
        else:
            selection_pool = [
                name for name in prepared
                if name == reference
                or (prepared[name]['n_seeds'] >= MIN_SEEDS_FOR_SIGNIFICANCE
                    and prepared[name]['mean'] is not None
                    and signatures[name] == signatures[reference])]

    rows = []
    for name, cand in prepared.items():
        if name == reference:
            continue
        # Protocol identity is a property of *this pair*: a third arm at a
        # different scale must not invalidate the clean reference-vs-candidate
        # contrasts, and a candidate at a different scale must not be
        # certified against the reference.
        pair_problems = []
        for field, label in (('shape', 'model shape'), ('final_step', 'final step'),
                             ('data_version', 'data version'),
                             ('data_ledger', 'cross-split ledger identity'),
                             ('model', 'model selector')):
            candidate_value = signatures[name][field]
            reference_value = signatures[reference][field]
            if (_signature_unknown(candidate_value)
                    or _signature_unknown(reference_value)
                    or candidate_value != reference_value):
                pair_problems.append(
                    f'{label} differs or is unknown: {candidate_value} vs '
                    f'{reference_value}')
        cand_hyper = ((manifests.get(name) or {}).get('hyperparameters')
                      if isinstance(manifests.get(name), dict) else None)
        ref_hyper = ((manifests.get(reference) or {}).get('hyperparameters')
                     if isinstance(manifests.get(reference), dict) else None)
        recipe_mismatches = None
        if isinstance(cand_hyper, dict) and isinstance(ref_hyper, dict):
            recipe_mismatches = training_recipe_mismatches(ref_hyper, cand_hyper)
            if recipe_mismatches:
                pair_problems.append(
                    'training recipe differs: '
                    + json.dumps(recipe_mismatches, sort_keys=True))
        else:
            # A missing recipe is unknown, not equal. `--allow-missing-recipe`
            # permits the comparison to be produced, but it can never certify:
            # the optimizer/schedule/token budget was not checked.
            pair_problems.append(
                'training recipe (optimizer/schedule/token budget) was not recorded '
                'for one or both arms; the comparison is descriptive only'
                + ('' if require_recipe else ' (--allow-missing-recipe)'))
        for role in (name, reference):
            problem = manifest_problems.get(role)
            if problem:
                pair_problems.append(f'run manifest for {role!r}: {problem}')
        common = sorted(set(cand['values']) & set(ref['values']))
        # Pair the arms at the data level, not only at the config level: for
        # every shared training seed, the input batches must be the same
        # windows in the same order. A missing record is unknown, not equal.
        stream_checks = {}
        for seed in common:
            check = compare_train_streams(
                _train_stream_for_seed(ref['train_stream_by_seed'], seed),
                _train_stream_for_seed(cand['train_stream_by_seed'], seed))
            stream_checks[str(seed)] = check
            if check.get('recorded') and not check.get('comparable'):
                pair_problems.append(
                    f'train stream for seed {seed} is not comparable: '
                    + '; '.join(check.get('problems') or [check.get('reason')]))
            elif check.get('comparable') and not check.get('identical'):
                pair_problems.append(
                    f'train stream differs for seed {seed} ('
                    + ', '.join(check.get('problems') or ['unknown difference'])
                    + ')')
            elif (require_train_stream_identity
                  and not (check.get('comparable') and check.get('identical'))):
                pair_problems.append(
                    f'train stream identity was not established for seed {seed}; '
                    'the arms may have consumed different data batches')
        pair_protocol_ok = not pair_problems
        entry = {
            'name': name,
            'norm': cand['norm'],
            'mean': cand['mean'],
            'std': cand['std'],
            'stderr': cand['stderr'],
            'n_seeds': cand['n_seeds'],
            'seeds': cand['seeds'],
            'issues': list(cand['issues']),
            'reference': reference,
            'reference_mean': ref['mean'],
            'reference_std': ref['std'],
            'reference_n_seeds': ref['n_seeds'],
            'shared_seeds': common,
            'n_shared_seeds': len(common),
            'delta_vs_reference': (None if cand['mean'] is None
                                   else cand['mean'] - ref['mean']),
            'delta_paired_mean': None,
            'paired_delta_std': None,
            'delta_over_seed_std': None,
            'delta_paired_over_delta_std': None,
            'within_seed_noise': None,
            # The seeds are the run-to-run variance; a loss delta that is
            # smaller than the arms' own spread is not a result.
            'ppl_reference': None if ref['mean'] is None else _safe_exp(ref['mean']),
            'ppl_arm': None if cand['mean'] is None else _safe_exp(cand['mean']),
            'ppl_delta': None,
            'ppl_delta_pct': None,
            'test': None,
            'test_basis': None,
            'p_one_sided': None,
            'p_one_sided_selection_adjusted': None,
            'p_inferential': None,
            'selection_adjustment_complete': (not reference_data_dependent),
            'selection_pool_used': [],
            'pool_contrasts': {},
            'delta_vs_pool_worst_case': None,
            'beats_every_pool_member': None,
            'p_one_sided_holm': None,
            't': None,
            'df': None,
            'delta_ci_95': None,
            'delta_prediction_interval_95': None,
            'delta_same_direction_probability': None,
            'n_seeds_favoring': None,
            'n_seeds_tied': None,
            'n_seeds_against': None,
            'per_seed_delta': None,
            'minimum_detectable_delta_95_power80': None,
            'design_power': None,
            'seed_jackknife': None,
            'randomization_test': None,
            'equivalence': {
                'enabled': False, 'test': 'tost_paired', 'margin': None,
                'equivalent': None, 'p_tost': None,
                'note': ('pass --equivalence-margin <loss> to test whether the '
                         'arms are equivalent; a non-significant difference is '
                         'not equivalence')},
            'protocol_problems': pair_problems,
            'train_stream_identity': {
                'checked': bool(stream_checks),
                'by_seed': stream_checks,
                'required': bool(require_train_stream_identity),
            },
            'recipe_checked': (isinstance(cand_hyper, dict)
                               and isinstance(ref_hyper, dict)),
            'recipe_mismatches': recipe_mismatches,
            'certifiable': False,
            'certified': False,
            'certification_basis': None,
        }
        if cand['mean'] is not None and ref['mean'] is not None:
            entry['ppl_delta'] = entry['ppl_arm'] - entry['ppl_reference']
            if entry['ppl_reference']:
                entry['ppl_delta_pct'] = (100.0 * entry['ppl_delta']
                                          / entry['ppl_reference'])
            seed_stds = [value for value in (cand['std'], ref['std'])
                         if _finite(value)]
            if seed_stds:
                denom = max(seed_stds)
                if denom > 0.0:
                    entry['delta_over_seed_std'] = abs(entry['delta_vs_reference']) / denom
                    entry['within_seed_noise'] = bool(
                        abs(entry['delta_vs_reference']) < denom)
        if len(common) >= 2:
            cand_vals = [cand['values'][seed] for seed in common]
            ref_vals = [ref['values'][seed] for seed in common]
            deltas = [a - b for a, b in zip(cand_vals, ref_vals)]
            delta_stats = mean_std_stderr(deltas)
            entry['delta_paired_mean'] = delta_stats['mean']
            entry['paired_delta_std'] = delta_stats['std']
            entry['n_seeds_favoring'] = sum(value < 0.0 for value in deltas)
            entry['n_seeds_tied'] = sum(value == 0.0 for value in deltas)
            entry['n_seeds_against'] = sum(value > 0.0 for value in deltas)
            entry['per_seed_delta'] = {str(seed): value
                                       for seed, value in zip(common, deltas)}
            replication = mean_prediction_interval(deltas, alpha=alpha)
            if replication is not None:
                entry['delta_prediction_interval_95'] = [
                    replication['prediction_low'],
                    replication['prediction_high']]
                entry['delta_same_direction_probability'] = (
                    replication['same_direction_probability'])
            if delta_stats['std'] and delta_stats['std'] > 0.0:
                entry['delta_paired_over_delta_std'] = (
                    abs(delta_stats['mean']) / delta_stats['std'])
            test = paired_t_one_sided(cand_vals, ref_vals, margin=margin, alpha=alpha)
            entry.update({
                'test': test['test'],
                'test_basis': 'seed_means',
                'p_one_sided': test['p_one_sided'],
                't': test['t'],
                'df': test['df'],
                'delta_ci_95': [test['ci_low'], test['ci_high']],
                'n': test['n'],
            })
            if test.get('df'):
                # The test is H1 mean(delta) < -margin. The smallest raw
                # effect that clears both the margin and the noise is
                # margin + (t_alpha + t_power) * stderr, not the margin-free
                # MDE (which would understate the required effect).
                entry['minimum_detectable_delta_95_power80'] = (
                    margin
                    + (student_t_ppf(1.0 - alpha, float(test['df']))
                       + student_t_ppf(0.8, float(test['df'])))
                    * delta_stats['stderr']
                )
                # Prospective design check against the pre-specified margin.
                # The observed paired SD is the planning value; the target is
                # the scientific margin, not the observed delta, so this does
                # not turn a null result into a post-hoc claim.
                entry['design_power'] = paired_design_summary(
                    delta_stats['std'], len(common), margin=margin,
                    alpha=alpha, power=0.8)
            if reference_data_dependent:
                # The reported reference was selected from a pool. Test the
                # candidate against every pool member that could actually
                # have been the reference, and combine the p-values with the
                # intersection-union max-p. Reporting only the comparison
                # against the best-looking member would hide the selection.
                used_pool: list[str] = []
                pool_contrasts: dict[str, dict] = {}
                pool_pvalues: list[float] = []
                pool_deltas: list[float] = []
                complete = True
                for member in selection_pool:
                    if member == name:
                        pool_contrasts[member] = {
                            'comparable': False,
                            'reason': 'the candidate is this pool member',
                        }
                        continue
                    if member == reference:
                        used_pool.append(member)
                        pool_contrasts[member] = {
                            'comparable': True,
                            'selected_reference': True,
                            'n_shared_seeds': len(common),
                            'delta_paired_mean': entry.get('delta_paired_mean'),
                            'p_one_sided': entry.get('p_one_sided'),
                        }
                        if _finite(entry.get('p_one_sided')):
                            pool_pvalues.append(entry['p_one_sided'])
                        if entry.get('delta_paired_mean') is not None:
                            pool_deltas.append(entry['delta_paired_mean'])
                        else:
                            complete = False
                        continue
                    other = prepared[member]
                    same_protocol = signatures[member] == signatures[reference]
                    common_m = sorted(set(cand['values']) & set(other['values']))
                    record = {
                        'comparable': bool(
                            same_protocol and
                            len(common_m) >= MIN_SEEDS_FOR_SIGNIFICANCE),
                        'same_protocol': same_protocol,
                        'n_shared_seeds': len(common_m),
                    }
                    if not record['comparable']:
                        record['reason'] = (
                            'not protocol-comparable to the selected reference'
                            if not same_protocol else
                            f'only {len(common_m)} shared seed(s); a '
                            f'certifying paired test needs '
                            f'{MIN_SEEDS_FOR_SIGNIFICANCE}')
                        complete = False
                        pool_contrasts[member] = record
                        continue
                    member_values = [other['values'][seed] for seed in common_m]
                    candidate_values = [cand['values'][seed] for seed in common_m]
                    member_test = paired_t_one_sided(
                        candidate_values, member_values, margin=margin, alpha=alpha)
                    member_deltas = [a - b for a, b in
                                     zip(candidate_values, member_values)]
                    record.update({
                        'delta_paired_mean': member_test['mean_delta'],
                        'p_one_sided': member_test['p_one_sided'],
                        'delta_ci_95': [member_test['ci_low'], member_test['ci_high']],
                    })
                    used_pool.append(member)
                    pool_contrasts[member] = record
                    if not _finite(member_test.get('p_one_sided')):
                        complete = False
                    else:
                        pool_pvalues.append(member_test['p_one_sided'])
                    if member_test['mean_delta'] is None:
                        complete = False
                    else:
                        pool_deltas.append(member_test['mean_delta'])
                entry['selection_pool_used'] = used_pool
                entry['pool_contrasts'] = pool_contrasts
                expected_comparators = len(
                    [member for member in selection_pool if member != name])
                entry['selection_adjustment_complete'] = bool(
                    complete and len(pool_pvalues) == expected_comparators)
                entry['p_one_sided_selection_adjusted'] = (
                    max_p_combination(pool_pvalues)
                    if entry['selection_adjustment_complete'] else None)
                entry['p_inferential'] = (
                    entry['p_one_sided_selection_adjusted']
                    if entry['selection_adjustment_complete'] else None)
                if pool_deltas:
                    # Largest candidate-minus-baseline delta = smallest
                    # advantage. A claim that survives only against the
                    # most favourable pool member is visible here.
                    entry['delta_vs_pool_worst_case'] = max(pool_deltas)
                entry['beats_every_pool_member'] = bool(
                    entry['selection_adjustment_complete']
                    and len(pool_deltas) == expected_comparators
                    and all(delta < -margin for delta in pool_deltas))
            else:
                entry['p_inferential'] = entry.get('p_one_sided')
            both_clean = (not cand['issues'] and not ref['issues']
                          and len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE)
            selection_ok = (not reference_data_dependent
                            or entry['selection_adjustment_complete'])
            entry['certifiable'] = bool(both_clean and pair_protocol_ok and selection_ok)
            if not pair_protocol_ok:
                entry['certification_basis'] = (
                    'not certifiable: this pair differs on a controlled protocol field '
                    f'({"; ".join(pair_problems)})')
            elif reference_data_dependent and not selection_ok:
                entry['certification_basis'] = (
                    'not certifiable: the data-selected reference pool could not be '
                    'tested completely, so the reported comparison is conditional on '
                    'the reference selection')
            elif len(common) < MIN_SEEDS_FOR_SIGNIFICANCE:
                entry['certification_basis'] = (
                    f'descriptive only: {len(common)} shared seed(s), fewer than '
                    f'{MIN_SEEDS_FOR_SIGNIFICANCE}; batch/step noise is not run-to-run noise')
            elif cand['issues'] or ref['issues']:
                entry['certification_basis'] = (
                    'descriptive only: incomplete/invalid seed coverage - '
                    + '; '.join(cand['issues'] + ref['issues']))
        else:
            entry['test_basis'] = 'insufficient_shared_seeds'
            entry['certification_basis'] = (
                f'{len(common)} shared seed(s); a paired test needs at least 2, '
                f'and only >= {MIN_SEEDS_FOR_SIGNIFICANCE} can certify')
        if common:
            deltas = [cand['values'][seed] - ref['values'][seed]
                      for seed in common]
            entry['seed_jackknife'] = leave_one_out_jackknife(
                deltas, labels=[str(seed) for seed in common])
            if len(common) >= 2:
                entry['randomization_test'] = paired_randomization_test(
                    deltas, margin=margin, alpha=alpha)
        if equivalence_margin > 0.0 and len(common) >= 2:
            tost = tost_paired(
                [cand['values'][seed] for seed in common],
                [ref['values'][seed] for seed in common],
                margin=equivalence_margin, alpha=alpha)
            tost['enabled'] = True
            paired_summary = mean_std_stderr(deltas)
            if (paired_summary is not None
                    and paired_summary.get('std') is not None):
                tost['design'] = tost_design_summary(
                    paired_summary['std'], int(tost['n']),
                    equivalence_margin, alpha=alpha)
            entry['equivalence'] = tost
        rows.append(entry)

    # One reference means the candidate p-values are a family; adjust them
    # before calling any arm a significant improvement.
    # Only a clean, >= 3-shared-seed pair is admissible to the Holm family:
    # a two-seed or cross-protocol p-value is descriptive and must not shrink
    # the adjusted p of a certifiable contrast. When the reference was
    # selected after seeing the losses, the family p-value is the
    # selection-adjusted one, not the raw comparison against the winner.
    family = [entry for entry in rows
              if entry.get('certifiable') and entry.get('p_inferential') is not None]
    for entry, p_holm in zip(family, holm_adjust([e['p_inferential'] for e in family])):
        entry['p_one_sided_holm'] = p_holm
        # The test that produced `p_one_sided` is paired on the *shared* seeds,
        # so its effect estimate is `delta_paired_mean`. Comparing the adjusted
        # p-value against the difference of the arms' own all-seed means mixes
        # two different quantities (and the two can even differ in sign when
        # the arm seed sets are not identical).
        paired_delta = entry.get('delta_paired_mean')
        entry['certified'] = bool(
            entry['certifiable']
            and paired_delta is not None
            and paired_delta < -margin
            and p_holm is not None and p_holm < alpha)

    return {
        'metric': metric,
        'alpha': alpha,
        'margin': margin,
        'equivalence_margin': equivalence_margin,
        'reference': reference,
        'reference_mean': ref['mean'],
        'reference_std': ref['std'],
        'reference_n_seeds': ref['n_seeds'],
        'arms': prepared,
        'rows': rows,
        'protocol': {
            'ok': protocol_ok,
            'problems': protocol_problems,
            'shape': sorted(map(str, shape_values)),
            'final_step': sorted(map(str, step_values)),
            'data_version': sorted(map(str, data_values)),
            'model': sorted(map(str, model_values)),
            'require_recipe': bool(require_recipe),
            'require_train_stream_identity': bool(require_train_stream_identity),
            'recipe_recorded': {name: prepared[name]['recipe_recorded']
                                for name in prepared},
            'reading': ('only the normalization may differ across Phase-1 arms; shape, '
                        'fixed final step, resolved data version and the recorded '
                        'optimizer/schedule/token recipe must be identical, the shared '
                        'seed must have consumed the same training stream, and each arm '
                        'needs >= 3 clean seeds'),
        },
        'reference_selection': {
            'data_dependent': reference_data_dependent,
            'rule': ('explicitly named reference (treated as pre-specified)'
                     if not reference_data_dependent else
                     'lowest held-out mean among replicated arms'
                     if not reference_was_explicit else
                     'caller-declared selection from a reference pool'),
            'pool': list(selection_pool),
            'pool_size': len(selection_pool),
            'adjustment': ('intersection-union max-p over the selection pool; '
                           'the certification family uses p_inferential'
                           if reference_data_dependent else
                           'none: the reference is pre-specified'),
            'p_family_basis': ('p_inferential (= selection-adjusted p)'
                               if reference_data_dependent else
                               'p_one_sided (= raw paired test)'),
            'reading': ('a data-selected reference makes the reported comparison '
                        'conditional on which arm looked best; the max-p '
                        'intersection-union requires the candidate to beat '
                        'every pool member, not just the winner'),
        },
        'multiple_comparison': (
            f'Holm-Bonferroni over the {len(family)} candidate arm(s) tested '
            f'against {reference!r}; `certified` requires >= '
            f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds, a clean protocol, a '
            f'complete reference-selection pool when the reference was '
            f'data-selected, and the adjusted p < {alpha}'),
        'reading': ('delta < 0 means the candidate has lower loss than the reference. '
                    '`delta_over_seed_std` compares the gap with the larger of the '
                    'two across-seed stds; `within_seed_noise` is a warning, not a '
                    'p-value. PPL deltas use exp(loss) and are reported for the '
                    'paper-facing number, while the test is run on loss (its natural '
                    'variance scale).'),
    }


def print_comparison(comparison: dict) -> None:
    ref = comparison['reference']
    print(f"[arms] metric={comparison['metric']} reference={ref} "
          f"mean={comparison['reference_mean']:.4f} "
          f"+/- {comparison['reference_std'] if comparison['reference_std'] is not None else 'n/a'}")
    protocol = comparison['protocol']
    if not protocol['ok']:
        # Certification is decided per reference/candidate pair. A third arm
        # at another scale can make the cross-arm view inconsistent without
        # invalidating a clean pair, so do not claim that no row can certify.
        print('[arms] WARNING: cross-arm protocol mismatch; only pairs that share '
              'every controlled field can certify: '
              + '; '.join(protocol['problems']))
    selection = comparison.get('reference_selection') or {}
    if selection.get('data_dependent') and selection.get('pool_size', 0) > 1:
        print(f"[arms] WARNING: reference {ref!r} was selected from "
              f"{selection.get('pool')} after seeing the losses; the "
              f"selection-adjusted p is reported alongside the raw one")
    for entry in comparison['rows']:
        delta = entry['delta_vs_reference']
        std = entry.get('std')
        p = entry.get('p_one_sided')
        p_holm = entry.get('p_one_sided_holm')
        p_inf = entry.get('p_inferential')
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        pi = entry.get('delta_prediction_interval_95')
        pi_str = ('' if pi is None or pi[0] is None
                  else f" pi95=[{pi[0]:+.4f}, {pi[1]:+.4f}]")
        p_str = 'n/a' if p is None else f'{p:.3g}'
        p_inf_str = ('' if p_inf is None or p_inf == p
                     else f' p(sel)={p_inf:.3g}')
        ph_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        ratio = entry.get('delta_over_seed_std')
        ratio_str = '' if ratio is None else f' delta/seed_std={ratio:.2f}'
        ppl = entry.get('ppl_delta_pct')
        ppl_str = '' if ppl is None else f' dPPL={ppl:+.2f}%'
        noise = ' [WITHIN SEED NOISE]' if entry.get('within_seed_noise') else ''
        note = '' if entry.get('certifiable') else f" [{entry.get('certification_basis')}]"
        if (selection.get('data_dependent') and selection.get('pool_size', 0) > 1
                and entry.get('beats_every_pool_member') is False):
            note += ' [does not beat every reference-pool member]'
        jackknife = entry.get('seed_jackknife') or {}
        if jackknife.get('meaningful') and jackknife.get('sign_stable') is False:
            note += (f" [SEED-SENSITIVE: dropping "
                     f"{jackknife.get('influential_labels')} flips the sign]")
        equivalence = entry.get('equivalence') or {}
        if equivalence.get('enabled'):
            if equivalence.get('equivalent'):
                note += f" [EQUIVALENT within {equivalence['margin']:g}]"
            else:
                note += f" [NOT equivalent within {equivalence['margin']:g}]"
        randomization = entry.get('randomization_test') or {}
        rand_p = randomization.get('p_one_sided')
        rand_str = '' if rand_p is None else f' p(sign-flip)={rand_p:.3g}'
        if randomization.get('can_reach_alpha') is False:
            floor = randomization.get('minimum_attainable_p_one_sided')
            if floor is not None:
                note += (f" [distribution-free p floor {floor:.3g} at "
                         f"n={randomization.get('n')}: sign-flip test cannot "
                         "reject without the t model]")
        mean_str = ('n/a' if entry.get('mean') is None
                    else f"{entry['mean']:.4f}")
        delta_str = ('n/a' if delta is None else f"{delta:+.4f}")
        print(f"[arms] {entry['name']:>24s} mean={mean_str}"
              + ('' if std is None else f" +/- {std:.4f}")
              + f" delta={delta_str}{ci_str}{pi_str}{ppl_str}{ratio_str}"
              + f" p={p_str}{p_inf_str} p(holm)={ph_str}{rand_str}"
              + f" certified={entry['certified']}{noise}{note}")
    print('[arms] ' + comparison['multiple_comparison'])


def _default_name(path: Path) -> str:
    # `runs/gpt2-124m-ln/aggregate.json` -> `gpt2-124m-ln`
    return path.parent.name or path.stem


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('aggregates', nargs='+', help='aggregate.json files from train_gpt2.py --seeds')
    p.add_argument('--names', nargs='+', default=None,
                   help='one display name per aggregate (default: the parent directory name)')
    p.add_argument('--reference', default=None,
                   help='arm name to compare against (default: best replicated arm)')
    p.add_argument('--reference-selection', default=None,
                   choices=('prespecified', 'best-of-pool'),
                   help='declare whether the reference was fixed before seeing the '
                        'results (prespecified) or chosen from a pool after seeing '
                        'them (best-of-pool). The default reference is treated as '
                        'data-selected; an explicit --reference is treated as '
                        'pre-specified unless this flag says otherwise.')
    p.add_argument('--reference-pool', nargs='+', default=None,
                   help='arm names the reference was selected from; enables the '
                        'intersection-union selection correction and requires '
                        '--reference to be one of them')
    p.add_argument('--metric', default='final_val_loss', choices=_LOSS_METRICS,
                   help='held-out metric to compare (default: the fixed-step final validation loss)')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='one-sided significance level (Holm-adjusted over candidates)')
    p.add_argument('--margin', type=float, default=0.0,
                   help='minimum loss improvement required to count as better')
    p.add_argument('--equivalence-margin', type=float, default=0.0,
                   help='TOST equivalence margin in loss for "the arms are the same" '
                        'claims; 0 disables equivalence testing, and a non-significant '
                        'difference is never relabelled as equivalence')
    p.add_argument('--require-train-stream-identity', action='store_true',
                   help='fail closed when the per-seed training-stream fingerprint is '
                        'missing or not comparable; the fingerprint proves the paired '
                        'arms consumed the same token windows in the same order')
    p.add_argument('--allow-missing-recipe', action='store_true',
                   help='do not fail closed when a sibling run_manifest.json (the record of '
                        'optimizer/schedule/token budget) is missing; the comparison is then '
                        'reported as recipe-unchecked and cannot certify a norm effect')
    p.add_argument('--output', default=None,
                   help='write the full comparison JSON here (default: <first aggregate dir>/arm_comparison.json)')
    args = p.parse_args(argv)
    if args.names is not None and len(args.names) != len(args.aggregates):
        p.error('--names must provide exactly one name per aggregate')
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        p.error('--alpha must be finite and strictly between 0 and 1')
    if not math.isfinite(args.margin) or args.margin < 0.0:
        p.error('--margin must be finite and >= 0')
    if not math.isfinite(args.equivalence_margin) or args.equivalence_margin < 0.0:
        p.error('--equivalence-margin must be finite and >= 0 (0 disables TOST)')
    if args.equivalence_margin > 0.0 and args.alpha >= 0.5:
        p.error('--equivalence-margin requires --alpha < 0.5 so the TOST '
                'confidence interval 1 - 2*alpha is positive')
    paths = [Path(path) for path in args.aggregates]
    names = args.names or [_default_name(path) for path in paths]
    if len(set(names)) != len(names):
        p.error(f'duplicate arm names: {names}; pass --names to disambiguate')
    if args.reference_pool is not None:
        unknown = sorted(set(args.reference_pool) - set(names))
        if unknown:
            p.error(f'--reference-pool names unknown arm(s) {unknown}; known: {names}')
        if args.reference is not None and args.reference not in args.reference_pool:
            p.error('--reference must be one of the --reference-pool members')
    if args.reference_selection == 'prespecified' and args.reference is None:
        p.error('--reference-selection prespecified requires --reference')
    arms = {}
    manifests = {}
    for name, path in zip(names, paths):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            p.error(f'cannot read aggregate {path}: {exc}')
        if not isinstance(data, dict) or not isinstance(data.get('per_seed'), list):
            p.error(f'{path} is not a train_gpt2.py seed aggregate (missing per_seed)')
        arms[name] = data
        manifest_path = path.parent / 'run_manifest.json'
        try:
            manifests[name] = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            manifests[name] = None
        if manifests[name] is None and not args.allow_missing_recipe:
            print(f'[arms][warn] {name}: no readable {manifest_path}; the optimizer/'
                  'schedule/token recipe cannot be verified. Pass --allow-missing-recipe '
                  'to compare anyway (the result will not be certifiable).',
                  file=sys.stderr)
    comparison = compare_arms(arms, reference=args.reference, metric=args.metric,
                              alpha=args.alpha, margin=args.margin,
                              manifests=manifests,
                              require_recipe=not args.allow_missing_recipe,
                              equivalence_margin=args.equivalence_margin,
                              require_train_stream_identity=args.require_train_stream_identity,
                              reference_selection=args.reference_selection,
                              reference_pool=args.reference_pool)
    print_comparison(comparison)
    output = Path(args.output) if args.output else paths[0].parent / 'arm_comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'[arms] wrote {output}')


if __name__ == '__main__':
    main()
