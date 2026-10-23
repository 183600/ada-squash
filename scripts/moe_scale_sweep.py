from __future__ import annotations
"""Run the MoE norm study at several model scales and audit replication.

A MoE load-balancing / val_ce result at one (dim, depth, expert count) is a
single-scale observation. This launcher runs the *identical* `train_moe_tiny`
recipe at two or more scales, then combines each scale's
`norm_comparison.json` with `scale_replication_summary`: a norm effect is
called replicated only when the sign agrees across scales, and certified
across scales only when >= 2 scales individually pass their Holm-adjusted
test. The per-scale artifacts stay on disk, so a reader can inspect each
comparison rather than trusting the pooled verdict.

Usage (everything after `--` is forwarded verbatim to train_moe_tiny.py):

    python3 scripts/moe_scale_sweep.py \
        --scale small:256:4:4:4 --scale medium:512:8:8:8 \
        --pair dytcd rmsnorm --output-dir runs/moe-scale \
        -- --norms ln rmsnorm dytcd --baselines ln rmsnorm \
           --seeds 42 43 44 --max-steps 500 --hypothesis "..."
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.check_scale_replication import build_observations

# Small, scientifically sensible defaults: same expert count, increasing
# width and depth. The launcher still requires >= 2 scales.
DEFAULT_SCALES = ('small:256:4:4:4', 'medium:512:8:8:4')
_SCALE_RE = re.compile(r'^([A-Za-z0-9_.-]+):(\d+):(\d+):(\d+):(\d+)$')
_FORBIDDEN_PASSTHROUGH = ('--dim', '--n-layers', '--n-heads', '--n-experts',
                          '--output-dir')


def parse_scale(spec: str) -> dict:
    match = _SCALE_RE.match(spec.strip())
    if not match:
        raise ValueError(f'scale {spec!r} must be NAME:DIM:LAYERS:HEADS:EXPERTS')
    name, dim, layers, heads, experts = match.groups()
    values = {'name': name, 'dim': int(dim), 'n_layers': int(layers),
              'n_heads': int(heads), 'n_experts': int(experts)}
    for key in ('dim', 'n_layers', 'n_heads', 'n_experts'):
        if values[key] < 1:
            raise ValueError(f'scale {spec!r} has non-positive {key}')
    if values['dim'] % values['n_heads']:
        raise ValueError(f'scale {spec!r}: dim must be divisible by n_heads')
    return values


def build_command(python: str, script: Path, scale: dict, passthrough: list[str],
                  output_dir: Path) -> list[str]:
    return [python, str(script), *passthrough,
            '--dim', str(scale['dim']),
            '--n-layers', str(scale['n_layers']),
            '--n-heads', str(scale['n_heads']),
            '--n-experts', str(scale['n_experts']),
            '--output-dir', str(output_dir)]


def summarise_scales(scales: list[dict], base: Path, candidate: str,
                     reference: str, metric: str | None = None,
                     min_scales: int = 2, alpha: float = 0.05) -> dict:
    entries = []
    for scale in scales:
        path = base / scale['name'] / 'norm_comparison.json'
        if not path.exists():
            raise SystemExit(f'[moe-scale] missing per-scale comparison {path}; '
                             'the scale run did not finish')
        entries.append((scale['name'], path))
    observations, sources = build_observations(
        'moe', entries, candidate, reference, metric)
    if not observations:
        # The child comparison may have selected a different baseline, so an
        # unmatched --pair would otherwise produce an empty report and exit 0
        # ("not multi-scale") instead of saying the requested contrast was
        # never run.
        raise SystemExit(
            f'[moe-scale] no observation found for pair {candidate!r} vs '
            f'{reference!r}' + (f' on metric {metric!r}' if metric else '')
            + '; the per-scale comparisons do not contain that contrast')
    from adasquash.metrics import scale_replication_summary
    summary = scale_replication_summary(observations, min_scales=min_scales,
                                        alpha=alpha)
    return {
        'candidate': candidate,
        'reference': reference,
        'metric': metric,
        'sources': sources,
        # `scales` below is the summary's observed scale-id list; keep the
        # actual (dim/layers/heads/experts) definitions under a distinct key so
        # the report cannot lose the sweep axis it ran.
        'requested_scales': scales,
        'observations': observations,
        **summary,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scale', action='append', default=None,
                   metavar='NAME:DIM:LAYERS:HEADS:EXPERTS',
                   help='a model scale to run; repeat for each scale (default: '
                        + ' '.join(DEFAULT_SCALES) + ')')
    p.add_argument('--pair', nargs=2, metavar=('CANDIDATE', 'REFERENCE'),
                   required=True,
                   help='the norm and baseline whose cross-scale replication is audited')
    p.add_argument('--metric', default=None,
                   help='metric to follow in every comparison (default: every metric)')
    p.add_argument('--min-scales', type=int, default=2,
                   help='distinct scales required before calling the effect replicated')
    p.add_argument('--alpha', type=float, default=0.05,
                   help='significance level for the pooled CI and heterogeneity test')
    p.add_argument('--output-dir', default='runs/moe-scale',
                   help='base directory; each scale writes <base>/<name>/')
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--dry-run', action='store_true',
                   help='print the child commands without running them')
    args, passthrough = p.parse_known_args(argv)
    if passthrough and passthrough[0] == '--':
        passthrough = passthrough[1:]
    if args.min_scales < 2:
        p.error('--min-scales must be >= 2; one scale cannot replicate')
    if not 0.0 < args.alpha < 1.0:
        p.error('--alpha must be strictly between 0 and 1')
    bad = [token for token in passthrough
           if token.split('=', 1)[0] in _FORBIDDEN_PASSTHROUGH]
    if bad:
        p.error(f'pass-through args may not set the sweep axis/output: {bad}')
    scale_specs = args.scale or list(DEFAULT_SCALES)
    try:
        scales = [parse_scale(spec) for spec in scale_specs]
    except ValueError as exc:
        p.error(str(exc))
    names = [scale['name'] for scale in scales]
    if len(set(names)) != len(names):
        p.error(f'duplicate scale names: {names}')
    # Two labels for the same architecture are one scale, not two independent
    # replications. Reject the duplicate before launching any child run.
    shapes = [(scale['dim'], scale['n_layers'], scale['n_heads'],
               scale['n_experts']) for scale in scales]
    if len(set(shapes)) != len(shapes):
        p.error(f'duplicate model shape(s) across --scale entries: {shapes}; '
                'one architecture is one scale')
    if len(scales) < args.min_scales:
        p.error(f'{len(scales)} scale(s) supplied but --min-scales={args.min_scales}; '
                'a single scale cannot support a multi-scale claim')
    base = Path(args.output_dir)
    script = ROOT / 'scripts' / 'train_moe_tiny.py'
    commands = []
    for scale in scales:
        out = base / scale['name']
        commands.append(build_command(args.python, script, scale, passthrough, out))
    if args.dry_run:
        for command in commands:
            print('[moe-scale] ' + ' '.join(command))
        return
    base.mkdir(parents=True, exist_ok=True)
    for scale, command in zip(scales, commands):
        print(f"[moe-scale] running {scale['name']} "
              f"(dim={scale['dim']}, L={scale['n_layers']}, "
              f"H={scale['n_heads']}, E={scale['n_experts']})")
        proc = subprocess.run(command)
        if proc.returncode != 0:
            raise SystemExit(f"[moe-scale] scale {scale['name']} failed "
                             f"(exit {proc.returncode}); no cross-scale verdict is written")
    report = summarise_scales(scales, base, args.pair[0], args.pair[1],
                              metric=args.metric, min_scales=args.min_scales,
                              alpha=args.alpha)
    output = base / 'scale_replication.json'
    with open(output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[moe-scale] {report['candidate']} vs {report['reference']} over "
          f"{report['n_scales']} scale(s): {report['scales']}")
    for metric, block in sorted(report['per_metric'].items()):
        print(f"[moe-scale]   {metric}: deltas={block['deltas']} "
              f"replicated={block['replicated_direction']} "
              f"multi_scale_certified={block['multi_scale_certified']}")
    if report['single_scale_only']:
        print('[moe-scale] NOT multi-scale: increase --scale count before generalizing')
    elif report['multi_scale_certified_metrics']:
        print('[moe-scale] multi-scale claim supported for '
              f"{report['multi_scale_certified_metrics']}")
    elif report['replicated_metrics']:
        print('[moe-scale] direction replicated but not individually certified at >= '
              f"{args.min_scales} scales for {report['replicated_metrics']}; report a "
              'consistent sign, not a certified multi-scale win')
    else:
        print('[moe-scale] direction does not replicate across scales; the single-scale '
              'claim must not be generalized')
    print(f'[moe-scale] wrote {output}')


if __name__ == '__main__':
    main()
