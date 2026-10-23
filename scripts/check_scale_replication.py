from __future__ import annotations
"""Combine per-scale comparison artifacts into a multi-scale verdict.

Every experiment script compares norms at *one* model shape. A direction that
is certified at one width/depth is a scope-limited result: the prompt's
scientific-method requirement is that FP8/W8A8/MoE claims be repeated at more
than one scale before they are generalized. This tool reads one comparison
artifact per scale and asks the same question for each metric: does the
candidate beat the reference at every scale, is the sign consistent, and is
the effect certified at >= 2 independent scales?

A sign vote is not enough when the scales estimate the effect with different
precision. Each per-metric block also carries an inverse-variance /
DerSimonian-Laird random-effects meta-analysis (Cochran's Q, I^2, tau^2 and a
prediction interval for a new scale). The pooled estimate is descriptive: it
does not replace the requirement that each scale pass its own pre-specified
test, and substantial heterogeneity means the pooled number is an average of
unlike effects rather than one replicated mechanism.

Inputs are the JSON files the experiment scripts already write:

* ``--kind fp8``  -> ``comparison.json`` from check_fp8_stability.py
* ``--kind w8a8`` -> ``ppl_comparison.json`` from check_w8a8_ppl.py
* ``--kind moe``  -> ``norm_comparison.json`` from train_moe_tiny.py
* ``--kind generic`` -> a JSON list of observation records (or an object with
  an ``observations`` list): each record needs ``scale``, ``delta`` and
  ``certified``, plus optional ``metric``/``p_one_sided`` and a precision for
  the meta-analysis (``stderr``, or ``ci_low``/``ci_high`` with ``df`` or
  ``n_seeds``). A record without a precision still counts in the sign vote but
  is listed as excluded from the pooled estimate.

The pair to follow is given with ``--pair A B`` (A is the candidate/treated
arm, B the reference). For FP8/W8A8 the comparison's ``a``/``b`` labels must
match in the candidate->reference orientation; reversed pairs are reported but
not credited as certified because their one-sided p-value tests the opposite
hypothesis. For MoE, ``--pair`` names the norm and baseline.
"""
import argparse
import json
import math
import numbers
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import scale_replication_summary


def _finite(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _reported_p(primary, fallback):
    """The Holm p when it is a real value, else the unadjusted p.

    A producer may emit the `p_*_holm` key with a null value for a row that
    was excluded from its family; that null is not a p-value and must not hide
    the raw one.
    """
    return primary if _finite(primary) else fallback


def _match_quality(label, target: str) -> int:
    """2 = exact/arm label, 1 = same basename, 0 = no match."""
    label = str(label or '')
    target = str(target or '')
    if label == target or label.startswith(target + '@'):
        return 2
    if Path(label).name == Path(target).name:
        return 1
    return 0


def _pair_orientation(a, b, candidate: str, reference: str):
    """Candidate/reference orientation, or None when the pair is ambiguous.

    Exact/arm-label matches outrank a basename match. A basename-only match is
    inherently ambiguous when both arms are, e.g., ``ckpt_best.pt`` from
    different directories: treating that as forward would inherit the
    candidate's certification for a reversed pair. Fail closed on ambiguity.
    """
    forward = (_match_quality(a, candidate), _match_quality(b, reference))
    reversed_ = (_match_quality(a, reference), _match_quality(b, candidate))
    forward_exact = min(forward) == 2
    reversed_exact = min(reversed_) == 2
    if forward_exact and reversed_exact:
        if forward == reversed_:
            return None
        return ('candidate_vs_reference' if forward > reversed_
                else 'reversed')
    if forward_exact:
        return 'candidate_vs_reference'
    if reversed_exact:
        return 'reversed'
    # A basename match in one orientation necessarily leaves the other
    # orientation with a zero (distinct basenames cannot match both ways), so
    # requiring both orientations to be complete made the basename tier
    # unreachable. Fail closed only when neither orientation is complete or
    # both are equally good (e.g. both names are `ckpt_best.pt`).
    if (min(forward) == 0 and min(reversed_) == 0) or forward == reversed_:
        return None
    return 'candidate_vs_reference' if forward > reversed_ else 'reversed'


def _pair_rows_from_fp8_w8a8(data, kind: str) -> list[dict]:
    tests = data.get('tests') if kind == 'fp8' else data.get('pairwise_tests')
    if not isinstance(tests, list):
        raise ValueError(f'{kind} comparison has no test list')
    return list(tests)


def _observation_from_pair(entry: dict, candidate: str, reference: str,
                           metric: str | None) -> dict | None:
    if not isinstance(entry, dict):
        # A malformed/null row is not an observation; skip it instead of
        # aborting the entire multi-scale report with an AttributeError.
        return None
    a, b = entry.get('a'), entry.get('b')
    if metric is not None and entry.get('metric') != metric:
        return None
    orientation = _pair_orientation(a, b, candidate, reference)
    if orientation is None:
        return None
    # A cross-scale pair is a deliberately descriptive comparison, not an
    # observation of the same mechanism at another scale: folding its delta
    # into the sign vote would let two different architectures masquerade as
    # two replications. Same for a row the producing script excluded from its
    # own family.
    if entry.get('same_scale') is False or entry.get('excluded_from_family') is not None:
        return None
    delta = entry.get('mean_delta')
    if orientation == 'reversed' and _finite(delta):
        delta = -float(delta)
    ci_low = entry.get('ci_low')
    ci_high = entry.get('ci_high')
    if orientation == 'reversed' and _finite(ci_low) and _finite(ci_high):
        # The interval belongs to the same mean difference as the delta, so a
        # reversed pair must mirror it around zero as well: a lower bound of
        # -0.3 for B-A is an upper bound of +0.3 for A-B.
        ci_low, ci_high = -float(ci_high), -float(ci_low)
    return {
        'scale': entry.get('a_scale') or entry.get('scale') or 'unknown',
        'metric': entry.get('metric') or metric or 'default',
        # The delta below is always candidate - reference (reversed pairs are
        # sign-flipped). Use the requested pair identity, not the artifact's
        # display labels: checkpoint paths differ per scale, so label-based
        # contrasts could never match across scales.
        'contrast': f'{candidate} vs {reference}',
        'delta': delta,
        # A reversed pair tests the opposite one-sided hypothesis; keep it as a
        # descriptive sign observation but never inherit its certification.
        # Only an explicit boolean True certifies a scale. `bool("False")`
        # is True, so a malformed/legacy string must not inherit a
        # certification the producer never asserted.
        'certified': (entry.get('significant') is True
                      if orientation == 'candidate_vs_reference' else False),
        'p_one_sided': (_reported_p(entry.get('p_holm'), entry.get('p_one_sided'))
                        if orientation == 'candidate_vs_reference' else None),
        'n_seeds': (entry.get('n_shared_train_seeds')
                    if entry.get('n_shared_train_seeds') not in (None, 0)
                    else entry.get('n_shared_seeds', entry.get('n_seeds'))),
        # Precision for the inverse-variance meta-analysis. Keep the CI even
        # for a reversed pair (mirrored above) so its width still contributes.
        'stderr': entry.get('stderr', entry.get('se')),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'df': entry.get('df'),
        'ci_level': entry.get('ci_level'),
        'orientation': orientation,
    }


def _observations_from_moe(data: dict, candidate: str, reference: str,
                           metric: str | None) -> list[dict]:
    rows = data.get('rows')
    if not isinstance(rows, list):
        raise ValueError('moe comparison has no rows list')
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if metric is not None and row.get('metric') != metric:
            continue
        orientation = _pair_orientation(row.get('norm'), row.get('reference'),
                                        candidate, reference)
        if orientation is None:
            continue
        if row.get('same_scale') is False or row.get('excluded_from_family') is not None:
            # Cross-scale or otherwise excluded rows are not independent
            # observations of the same-scale effect; including them would
            # manufacture a replication.
            continue
        delta = row.get('delta_vs_reference')
        if orientation == 'reversed' and _finite(delta):
            delta = -float(delta)
        ci = row.get('delta_ci_95')
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            ci_low, ci_high = ci
        else:
            ci_low, ci_high = None, None
        if orientation == 'reversed' and _finite(ci_low) and _finite(ci_high):
            ci_low, ci_high = -float(ci_high), -float(ci_low)
        out.append({
            'scale': row.get('scale') or data.get('scale') or 'unknown',
            'metric': row.get('metric') or metric or 'default',
            # Same canonical candidate/reference identity as the sign-flipped
            # delta, so separate per-scale artifacts share one contrast id.
            'contrast': f'{candidate} vs {reference}',
            'delta': delta,
            'certified': ((row.get('beats_reference_significantly') is True)
                          if orientation == 'candidate_vs_reference' else False),
            'p_one_sided': (_reported_p(row.get('p_one_sided_holm'),
                                        row.get('p_one_sided'))
                            if orientation == 'candidate_vs_reference' else None),
            'n_seeds': (len(row['paired_seeds'])
                        if isinstance(row.get('paired_seeds'), list) else None),
            'stderr': row.get('stderr', row.get('se')),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'df': row.get('df'),
            'ci_level': row.get('ci_level'),
            'orientation': orientation,
        })
    return out


def load_comparison(kind: str, path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f'[scale] cannot read {path}: {exc}')
    # A generic input is either an object with an `observations` list or the
    # bare list of observation records (the documented form). The kind-specific
    # comparison artifacts are always objects.
    if kind != 'generic' and not isinstance(data, dict):
        raise SystemExit(f'[scale] {path} is not a comparison object')
    return data


def build_observations(kind: str, scales: list[tuple[str, Path]],
                       candidate: str, reference: str,
                       metric: str | None) -> tuple[list[dict], list[dict]]:
    observations = []
    sources = []
    seen_labels = set()
    seen_paths = set()
    for label, path in scales:
        # Two --scale labels that point at the same file are one underlying
        # run, not two independent scales. Deduplicate by resolved path (and
        # label) before the artifact's own scale is prefixed, otherwise the
        # same result can manufacture a multi-scale replication.
        resolved = path.resolve()
        if label in seen_labels or resolved in seen_paths:
            print(f'[scale] ignoring duplicate scale input {label}={path} '
                  f'(same label or artifact already provided); one run is one scale',
                  file=sys.stderr)
            continue
        seen_labels.add(label)
        seen_paths.add(resolved)
        data = load_comparison(kind, path)
        if kind == 'generic':
            records = data.get('observations') if isinstance(data, dict) else data
            if not isinstance(records, list):
                raise SystemExit(f'[scale] generic file {path} has no observations list')
            for record in records:
                if not isinstance(record, dict):
                    continue
                item = dict(record)
                if not item.get('scale'):
                    item['scale'] = label
                if metric is not None and item.get('metric', 'default') != metric:
                    continue
                observations.append(item)
        elif kind in ('fp8', 'w8a8'):
            for entry in _pair_rows_from_fp8_w8a8(data, kind):
                item = _observation_from_pair(entry, candidate, reference, metric)
                if item is None:
                    continue
                # Keep the artifact's own model-scale identity. The CLI label
                # is only the source name: prefixing it made two copies of the
                # same run under different labels look like two independent
                # scales and manufactured a multi-scale certification.
                observations.append(item)
        elif kind == 'moe':
            for item in _observations_from_moe(data, candidate, reference, metric):
                observations.append(item)
        else:
            raise SystemExit(f'[scale] unknown kind {kind!r}')
        sources.append({'scale': label, 'path': str(path)})
    return observations, sources


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--kind', required=True, choices=['fp8', 'w8a8', 'moe', 'generic'])
    p.add_argument('--scale', action='append', default=[],
                   metavar='LABEL=PATH',
                   help='one comparison artifact per scale; repeat for each scale')
    p.add_argument('--pair', nargs=2, metavar=('CANDIDATE', 'REFERENCE'),
                   default=None,
                   help='candidate/treated arm and reference arm to follow '
                        '(required for fp8/w8a8/moe; generic files can omit it)')
    p.add_argument('--metric', default=None,
                   help='metric to follow (default: every metric present)')
    p.add_argument('--min-scales', type=int, default=2,
                   help='distinct scales required before an effect is called replicated')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='significance level for the pooled CI and heterogeneity test')
    p.add_argument('--output', default=None,
                   help='write the full JSON report here')
    args = p.parse_args(argv)
    if args.min_scales < 2:
        p.error('--min-scales must be >= 2; one scale cannot replicate')
    if not 0.0 < args.alpha < 1.0:
        p.error('--alpha must be strictly between 0 and 1')
    if not args.scale:
        p.error('at least one --scale LABEL=PATH is required')
    if args.kind != 'generic' and args.pair is None:
        p.error(f'--kind {args.kind} needs --pair CANDIDATE REFERENCE')
    scales = []
    for spec in args.scale:
        if '=' not in spec:
            p.error(f'--scale {spec!r} must be LABEL=PATH')
        label, raw = spec.split('=', 1)
        if not label or not raw:
            p.error(f'--scale {spec!r} must be LABEL=PATH')
        scales.append((label, Path(raw)))
    candidate, reference = (args.pair or ('candidate', 'reference'))
    observations, sources = build_observations(args.kind, scales, candidate,
                                               reference, args.metric)
    summary = scale_replication_summary(observations, min_scales=args.min_scales,
                                        alpha=args.alpha)
    ci_label = f'{100.0 * (1.0 - args.alpha):g}%'
    report = {
        'kind': args.kind,
        'pair': {'candidate': candidate, 'reference': reference},
        'metric': args.metric,
        'sources': sources,
        'observations': observations,
        **summary,
    }
    print(f"[scale] {args.kind}: {candidate} vs {reference} over "
          f"{summary['n_scales']} scale(s): {summary['scales']}")
    for metric, block in sorted(summary['per_metric'].items()):
        print(f"[scale]   {metric}: deltas={block['deltas']} "
              f"replicated={block['replicated_direction']} "
              f"certified_scales={block['certified_scales']} "
              f"multi_scale_certified={block['multi_scale_certified']}")
        pooled = block.get('meta_analysis') or {}
        random_effects = pooled.get('random_effects') or {}
        heterogeneity = pooled.get('heterogeneity') or {}
        if pooled.get('n_studies', 0) >= 2 and random_effects.get('ci_low') is not None:
            prediction = pooled.get('prediction_interval') or {}
            pi = ('n/a' if prediction.get('low') is None else
                  f"[{prediction['low']:+.4f}, {prediction['high']:+.4f}]")
            print(f"[scale]     meta: pooled={random_effects['pooled_effect']:+.4f} "
                  f"CI{ci_label}=[{random_effects['ci_low']:+.4f}, "
                  f"{random_effects['ci_high']:+.4f}] "
                  f"k={pooled['n_studies']} "
                  f"I^2={heterogeneity.get('i_squared'):.1f}% "
                  f"Q_p={heterogeneity.get('p'):.3g} "
                  f"tau={heterogeneity.get('tau'):.4g} "
                  f"prediction={pi}")
            if pooled.get('excluded'):
                print(f"[scale]     meta excluded (no usable precision): "
                      f"{[item.get('label') for item in pooled['excluded']]}",
                      file=sys.stderr)
    if summary['single_scale_only']:
        print(f"[scale] NOT a multi-scale result: only {summary['n_scales']} scale(s); "
              f"at least {args.min_scales} are required. Do not extrapolate the "
              'single-scale direction to another width/depth.')
    elif summary['multi_scale_certified_metrics']:
        print(f"[scale] multi-scale claim supported for "
              f"{summary['multi_scale_certified_metrics']}")
    elif summary['replicated_metrics']:
        print(f"[scale] direction replicated but not individually certified at >= "
              f"{args.min_scales} scales for {summary['replicated_metrics']}; report it "
              'as a consistent sign, not a certified multi-scale win')
    else:
        print('[scale] direction does not replicate across the supplied scales; '
              'the single-scale claim must not be generalized')
    # The sign vote and the pooled estimate answer different questions. Say so
    # when they disagree instead of letting the optimistic one stand alone.
    heterogeneous = []
    unresolved = []
    for metric in summary['meta_analysed_metrics']:
        pooled = summary['per_metric'][metric].get('meta_analysis') or {}
        random_effects = pooled.get('random_effects') or {}
        heterogeneity = pooled.get('heterogeneity') or {}
        if (heterogeneity.get('i_squared') is not None
                and (heterogeneity['i_squared'] >= 50.0
                     or (heterogeneity.get('p') is not None
                         and heterogeneity['p'] < args.alpha))):
            heterogeneous.append(metric)
        if (random_effects.get('ci_low') is not None
                and random_effects['ci_low'] <= 0.0 <= random_effects['ci_high']):
            unresolved.append(metric)
    if heterogeneous:
        print(f"[scale] WARNING: substantial between-scale heterogeneity for "
              f"{heterogeneous}; the pooled estimate is an average of unlike "
              'effects, so do not report it as one replicated mechanism.')
    if unresolved:
        print(f"[scale] WARNING: the pooled {ci_label} interval includes zero for "
              f"{unresolved}; the direction vote is not a resolved pooled effect.")
    if summary['excluded']:
        print(f"[scale] excluded observations: {summary['excluded']}", file=sys.stderr)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'[scale] wrote {output}')


if __name__ == '__main__':
    main()
