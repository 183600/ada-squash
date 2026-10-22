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
* both loss and perplexity (`exp(loss)`) deltas, because the papers quote
  PPL while the seed variance lives in loss;
* a protocol audit across arms: identical model shape, fixed final step and
  resolved data version. If the arms differ on those, the comparison is
  reported but never certified, no matter how small the p-value.

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
                               mean_std_stderr, paired_t_one_sided,
                               student_t_ppf)
from adasquash.repro import training_recipe_mismatches

_LOSS_METRICS = ('final_val_loss', 'best_val_loss', 'test_loss')


def _finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


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


def _data_version_signature(agg: dict):
    versions = agg.get('data_versions_by_seed')
    if not isinstance(versions, dict) or not versions:
        return None
    values = list(versions.values())
    if any(value is None for value in values):
        return None
    return json.dumps(values[0], sort_keys=True)


def _shape_signature(agg: dict):
    shapes = agg.get('shapes_by_seed')
    if not isinstance(shapes, dict) or not shapes:
        return None
    signatures = {json.dumps(shape, sort_keys=True) for shape in shapes.values()}
    if len(signatures) != 1:
        return 'MIXED'
    return next(iter(signatures))


def _final_step(agg: dict):
    per_seed = agg.get('per_seed') if isinstance(agg, dict) else None
    if not isinstance(per_seed, list):
        return None
    steps = {record.get('final_val_step') for record in per_seed
             if isinstance(record, dict)}
    if len(steps) != 1:
        return None
    step = next(iter(steps))
    return step if isinstance(step, int) and not isinstance(step, bool) else None


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
    if agg.get('data_versions_consistent') is False:
        issues.append(f'arm {name!r} mixes resolved data versions across seeds')
    if agg.get('shapes_consistent') is False:
        issues.append(f'arm {name!r} mixes model shapes across seeds')
    return issues


def compare_arms(arms: dict[str, dict], reference: str | None=None,
                 metric: str='final_val_loss', alpha: float=0.05,
                 margin: float=0.0, manifests: dict[str, dict] | None=None,
                 require_recipe: bool=True) -> dict:
    """Paired, Holm-adjusted comparison of arm aggregates.

    `arms` maps a display name to an aggregate.json object. The reference is
    the explicitly named arm, or the best arm with >= 3 seeds; each other arm
    is tested against it on the seeds both arms actually share. The result
    separates "delta < 0" (point estimate) from "certified" (>= 3 shared
    seeds, clean arms, identical protocol, and Holm-adjusted p < alpha).
    """
    if not isinstance(arms, dict) or len(arms) < 2:
        raise ValueError('compare_arms needs at least two named aggregates')
    if metric not in _LOSS_METRICS:
        raise ValueError(f'metric must be one of {_LOSS_METRICS}')
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError('alpha must be finite and strictly between 0 and 1')

    prepared = {}
    for name, agg in arms.items():
        vals = _seed_values(agg, metric)
        stats = mean_std_stderr(list(vals.values()))
        prepared[name] = {
            'name': name,
            'norm': agg.get('norm') if isinstance(agg, dict) else None,
            'model': agg.get('model') if isinstance(agg, dict) else None,
            'values': vals,
            'mean': None if stats is None else stats['mean'],
            'std': None if stats is None or stats['n'] < 2 else stats['std'],
            'stderr': None if stats is None else stats['stderr'],
            'n_seeds': len(vals),
            'seeds': sorted(vals),
            'issues': _arm_issues(name, agg, metric),
            'curve_classification_counts': (agg.get('curve_classification_counts')
                                            if isinstance(agg, dict) else None),
        }
    manifests = manifests or {}
    for name in list(prepared):
        manifest = manifests.get(name) if isinstance(manifests, dict) else None
        hyper = manifest.get('hyperparameters') if isinstance(manifest, dict) else None
        prepared[name]['recipe_recorded'] = isinstance(hyper, dict)
    if reference is not None:
        if reference not in prepared:
            raise ValueError(f'reference {reference!r} is not one of {sorted(prepared)}')
    else:
        replicated = [row for row in prepared.values()
                      if row['n_seeds'] >= MIN_SEEDS_FOR_SIGNIFICANCE
                      and row['mean'] is not None]
        pool = replicated or [row for row in prepared.values()
                              if row['mean'] is not None]
        if not pool:
            raise ValueError('no arm has a usable mean for the requested metric')
        reference = min(pool, key=lambda row: row['mean'])['name']
    ref = prepared[reference]
    if ref['mean'] is None:
        raise ValueError(f'reference arm {reference!r} has no usable {metric}')

    # Protocol identity across arms: only the normalization may differ. The
    # model shape, fixed final step and resolved data version are the
    # controlled variables the Phase-1 configs promise to hold constant.
    signatures = {name: {'shape': _shape_signature(agg),
                         'final_step': _final_step(agg),
                         'data_version': _data_version_signature(agg),
                         'model': prepared[name]['model']}
                  for name, agg in arms.items()}
    shape_values = {sig['shape'] for sig in signatures.values()}
    step_values = {sig['final_step'] for sig in signatures.values()}
    data_values = {sig['data_version'] for sig in signatures.values()}
    model_values = {sig['model'] for sig in signatures.values()}
    protocol_problems = []
    if len(shape_values) != 1 or 'MIXED' in shape_values or None in shape_values:
        protocol_problems.append(f'model shapes differ or are unknown: {sorted(map(str, shape_values))}')
    if len(step_values) != 1 or None in step_values:
        protocol_problems.append(f'final step differs or is unknown: {sorted(map(str, step_values))}')
    if len(data_values) != 1 or None in data_values:
        protocol_problems.append(f'data versions differ or are unknown: {sorted(map(str, data_values))}')
    if len(model_values) != 1 or None in model_values:
        protocol_problems.append(f'model selectors differ or are unknown: {sorted(map(str, model_values))}')
    protocol_ok = not protocol_problems

    rows = []
    for name, cand in prepared.items():
        if name == reference:
            continue
        # Protocol identity is a property of *this pair*: a third arm at a
        # different scale must not invalidate the clean reference-vs-candidate
        # contrasts, and a candidate at a different scale must not be
        # certified against the reference.
        pair_problems = []
        if signatures[name]['shape'] != signatures[reference]['shape']:
            pair_problems.append(
                f"model shape differs: {signatures[name]['shape']} vs "
                f"{signatures[reference]['shape']}")
        if signatures[name]['final_step'] != signatures[reference]['final_step']:
            pair_problems.append(
                f"final step differs: {signatures[name]['final_step']} vs "
                f"{signatures[reference]['final_step']}")
        if signatures[name]['data_version'] != signatures[reference]['data_version']:
            pair_problems.append(
                f"data version differs: {signatures[name]['data_version']} vs "
                f"{signatures[reference]['data_version']}")
        if signatures[name]['model'] != signatures[reference]['model']:
            pair_problems.append(
                f"model selector differs: {signatures[name]['model']} vs "
                f"{signatures[reference]['model']}")
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
        elif require_recipe:
            pair_problems.append(
                'training recipe (optimizer/schedule/token budget) was not recorded '
                'for one or both arms; pass --allow-missing-recipe only for an '
                'exploratory comparison')
        pair_protocol_ok = not pair_problems
        common = sorted(set(cand['values']) & set(ref['values']))
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
            'ppl_reference': None if ref['mean'] is None else math.exp(ref['mean']),
            'ppl_arm': None if cand['mean'] is None else math.exp(cand['mean']),
            'ppl_delta': None,
            'ppl_delta_pct': None,
            'test': None,
            'test_basis': None,
            'p_one_sided': None,
            'p_one_sided_holm': None,
            't': None,
            'df': None,
            'delta_ci_95': None,
            'n_seeds_favoring': None,
            'n_seeds_tied': None,
            'n_seeds_against': None,
            'per_seed_delta': None,
            'minimum_detectable_delta_95_power80': None,
            'protocol_problems': pair_problems,
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
                entry['minimum_detectable_delta_95_power80'] = (
                    student_t_ppf(1.0 - alpha, float(test['df']))
                    + student_t_ppf(0.8, float(test['df']))
                ) * delta_stats['stderr']
            both_clean = (not cand['issues'] and not ref['issues']
                          and len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE)
            entry['certifiable'] = bool(both_clean and pair_protocol_ok)
            if not pair_protocol_ok:
                entry['certification_basis'] = (
                    'not certifiable: this pair differs on a controlled protocol field '
                    f'({"; ".join(pair_problems)})')
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
        rows.append(entry)

    # One reference means the candidate p-values are a family; adjust them
    # before calling any arm a significant improvement.
    # Only a clean, >= 3-shared-seed pair is admissible to the Holm family:
    # a two-seed or cross-protocol p-value is descriptive and must not shrink
    # the adjusted p of a certifiable contrast.
    family = [entry for entry in rows
              if entry.get('certifiable') and entry['p_one_sided'] is not None]
    for entry, p_holm in zip(family, holm_adjust([e['p_one_sided'] for e in family])):
        entry['p_one_sided_holm'] = p_holm
        entry['certified'] = bool(
            entry['certifiable']
            and entry['delta_vs_reference'] is not None
            and entry['delta_vs_reference'] < -margin
            and p_holm is not None and p_holm < alpha)

    return {
        'metric': metric,
        'alpha': alpha,
        'margin': margin,
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
            'recipe_recorded': {name: prepared[name]['recipe_recorded']
                                for name in prepared},
            'reading': ('only the normalization may differ across Phase-1 arms; shape, '
                        'fixed final step, resolved data version and the recorded '
                        'optimizer/schedule/token recipe must be identical, and each '
                        'arm needs >= 3 clean seeds'),
        },
        'multiple_comparison': (
            f'Holm-Bonferroni over the {len(family)} candidate arm(s) tested '
            f'against {reference!r}; `certified` requires >= '
            f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds, a clean protocol, and '
            f'the adjusted p < {alpha}'),
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
        print('[arms] WARNING: protocol mismatch across arms; no certification is possible: '
              + '; '.join(protocol['problems']))
    for entry in comparison['rows']:
        delta = entry['delta_vs_reference']
        std = entry.get('std')
        p = entry.get('p_one_sided')
        p_holm = entry.get('p_one_sided_holm')
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        p_str = 'n/a' if p is None else f'{p:.3g}'
        ph_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        ratio = entry.get('delta_over_seed_std')
        ratio_str = '' if ratio is None else f' delta/seed_std={ratio:.2f}'
        ppl = entry.get('ppl_delta_pct')
        ppl_str = '' if ppl is None else f' dPPL={ppl:+.2f}%'
        noise = ' [WITHIN SEED NOISE]' if entry.get('within_seed_noise') else ''
        note = '' if entry.get('certifiable') else f" [{entry.get('certification_basis')}]"
        mean_str = ('n/a' if entry.get('mean') is None
                    else f"{entry['mean']:.4f}")
        delta_str = ('n/a' if delta is None else f"{delta:+.4f}")
        print(f"[arms] {entry['name']:>24s} mean={mean_str}"
              + ('' if std is None else f" +/- {std:.4f}")
              + f" delta={delta_str}{ci_str}{ppl_str}{ratio_str}"
              + f" p={p_str} p(holm)={ph_str} certified={entry['certified']}"
              + f"{noise}{note}")
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
    p.add_argument('--metric', default='final_val_loss', choices=_LOSS_METRICS,
                   help='held-out metric to compare (default: the fixed-step final validation loss)')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='one-sided significance level (Holm-adjusted over candidates)')
    p.add_argument('--margin', type=float, default=0.0,
                   help='minimum loss improvement required to count as better')
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
    paths = [Path(path) for path in args.aggregates]
    names = args.names or [_default_name(path) for path in paths]
    if len(set(names)) != len(names):
        p.error(f'duplicate arm names: {names}; pass --names to disambiguate')
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
                              require_recipe=not args.allow_missing_recipe)
    print_comparison(comparison)
    output = Path(args.output) if args.output else paths[0].parent / 'arm_comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'[arms] wrote {output}')


if __name__ == '__main__':
    main()
