from __future__ import annotations
"""Efficiency/accuracy trade-off benchmark for the norm variants.

A point-estimate loss win is only half the evidence; the other half is what
the win costs. This script measures, per normalization variant on the same
backbone, shape and device:

* parameters total and by role (norm share of the model), via
  `adasquash.efficiency.parameter_breakdown`;
* analytic matmul FLOPs per token (forward and training estimate) and, when
  torch's `FlopCounterMode` is available, the counter's measurement of the
  actual forward;
* forward latency and full training-step latency per token (warmup + repeated
  measurements, mean/std, on CUDA with events and synchronization). A single
  serial pass in fixed norm order lets a slow thermal/clock drift land on the
  norms measured later, so the norms are interleaved round-robin over
  `--rounds` passes (alternating forward and reverse order), each norm's
  estimate is the mean of its per-round means, and the exact measurement
  order is recorded in the report (`measurement_order`, `latency_by_round`);
* peak CUDA memory for inference and for a training step (null on CPU: the
  process RSS cannot be attributed to one model).

With `--summaries runs/ablation/all_summaries.json` plus a reference and a
candidate row, it joins the measured cost to the held-out loss and applies
`efficiency_verdict`: a below-margin gain or an above-maximum slowdown is
reported as *not viable*, with the cost-per-loss-point printed explicitly.
`--require-viable` turns that into a non-zero exit status for CI-style use.

The elementwise norm math is not a matmul, so FLOPs do not see it; the
wall-clock latency does. Read the two together.
"""
import argparse
import dataclasses
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from adasquash.efficiency import (
    DEFAULT_MAX_COST_RATIO,
    DEFAULT_MIN_LOSS_GAIN,
    analytic_flops_per_token,
    efficiency_verdict,
    parameter_breakdown,
)
from adasquash.attention import clear_logit_scratch_cache
from adasquash.model import (
    build_gpt_124m,
    build_gpt_350m,
    build_llama_tiny_1b,
    clear_cast_weight_cache,
    clear_ce_scratch_cache,
)
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds

_BUILDERS = {'gpt2-124m': build_gpt_124m, 'gpt2-350m': build_gpt_350m, 'llama-1b': build_llama_tiny_1b}


def _is_cuda(device: str) -> bool:
    return torch.device(device).type == 'cuda'


def _sync(device: str) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(device)


def measure_timed(fn, iters: int, warmup: int, device: str) -> dict:
    n_iters = max(1, iters)
    for _ in range(max(0, warmup)):
        fn()
    if _is_cuda(device):
        # Event burst: every iteration is enqueued back-to-back and the host
        # syncs once at the end.
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        _sync(device)
        for i in range(n_iters):
            starts[i].record()
            fn()
            ends[i].record()
        _sync(device)
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        times = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        'mean_ms': statistics.fmean(times),
        'median_ms': statistics.median(times),
        'std_ms': statistics.pstdev(times) if len(times) > 1 else 0.0,
        'n_iters': len(times),
    }


def measure_peak_memory(fn, device: str):
    if not _is_cuda(device):
        return None
    # Dropping the scratch caches first keeps the first measured norm
    # comparable to the rest.
    clear_ce_scratch_cache()
    clear_logit_scratch_cache()
    clear_cast_weight_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fn()
    _sync(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)


def measured_forward_flops(fn, tokens: int):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    try:
        with FlopCounterMode(display=False) as counter:
            fn()
        return float(counter.get_total_flops()) / max(1, tokens)
    except Exception:
        return None


def benchmark_norm(model_name: str, norm: str, args, device: str, measure_heavy: bool=True) -> dict:
    # Drop the previous norm's autocast casts before its replacement is built:
    # the cache pins the old model's weights until the next forward re-casts.
    clear_cast_weight_cache()
    # Build on the benchmark device and skip the constructor init pass;
    # `reset_parameters(args.seed)` below is the single seeded init.
    with torch.device(device):
        model = _BUILDERS[model_name](norm_name=norm, init=False)
    model.reset_parameters(args.seed)
    model.to(device)
    model.train()
    vocab = int(model.cfg.vocab_size)
    # Common random numbers: every norm is measured on the *same* tokens. A
    # per-norm draw from the global RNG would depend on how much RNG the norm's
    # init consumed (parameter counts differ), so the timed steps would not be
    # input-comparable across norms.
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    x = torch.randint(0, vocab, (args.batch_size, args.seq_len), generator=gen, device=device)
    y = torch.randint(0, vocab, (args.batch_size, args.seq_len), generator=gen, device=device)
    tokens = args.batch_size * args.seq_len

    autocast_ctx = (
        torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        if (args.bf16 and _is_cuda(device)) else torch.autocast(device_type='cpu', enabled=False)
    )

    def forward():
        with torch.inference_mode(), autocast_ctx:
            model(x, targets=None, return_logits=True)

    def train_step():
        model.zero_grad(set_to_none=True)
        with autocast_ctx:
            loss = model(x, y, return_logits=False)[1]
        loss.backward()
        if args.include_optimizer:
            opt.step()

    from scripts.train_gpt2 import build_optimizer
    opt = build_optimizer(model, args.lr, 0.1, 0.9, 0.95)
    fwd_timing = measure_timed(forward, args.iters, args.warmup, device)
    train_timing = measure_timed(train_step, args.iters, args.warmup, device)
    # Memory and FlopCounterMode are the same for every round of the same
    # norm; measuring them once is enough.
    if measure_heavy:
        fwd_mem = measure_peak_memory(forward, device)
        train_mem = measure_peak_memory(train_step, device)
    else:
        fwd_mem = train_mem = None
    model.zero_grad(set_to_none=True)
    flops = analytic_flops_per_token(model.cfg, seq_len=args.seq_len)
    measured = measured_forward_flops(forward, tokens) if measure_heavy else None
    breakdown = parameter_breakdown(model)
    return {
        'norm': norm,
        'n_params': breakdown['total'],
        'parameter_breakdown': breakdown,
        'norm_share_of_total': breakdown['norm_share_of_total'],
        'analytic_flops': flops,
        'measured_forward_flops_per_token': measured,
        'forward_latency_ms_per_token': fwd_timing['mean_ms'] / tokens,
        'forward_latency_ms': fwd_timing,
        'training_latency_ms_per_token': train_timing['mean_ms'] / tokens,
        'training_latency_ms': train_timing,
        'forward_tokens_per_second': tokens / (fwd_timing['mean_ms'] / 1000.0),
        'training_tokens_per_second': tokens / (train_timing['mean_ms'] / 1000.0),
        'peak_inference_memory_mb': fwd_mem,
        'peak_training_memory_mb': train_mem,
    }


_LATENCY_MEAN_KEYS = ('forward_latency_ms_per_token', 'training_latency_ms_per_token',
                      'forward_tokens_per_second', 'training_tokens_per_second')


def aggregate_benchmark_rounds(rows: list, norms: list, rounds: int) -> list:
    """One row per norm, latency estimates averaged over the interleaved rounds.

    Each round measured the same norm in a different position of the order, so
    the per-round means are (approximately) independent samples of the
    position-dependent drift. Averaging them cancels a *linear* drift to first
    order and keeps every norm's estimate resting on the same number of
    iterations; `latency_by_round` is retained so the raw spread can be
    audited instead of hidden behind the cross-round mean.
    """
    out = []
    for norm in norms:
        norm_rows = [r for r in rows if r['norm'] == norm]
        base = dict(norm_rows[0])
        for key in _LATENCY_MEAN_KEYS:
            base[key] = sum(r[key] for r in norm_rows) / len(norm_rows)
        base['forward_latency_ms'] = {
            'mean_ms': sum(r['forward_latency_ms']['mean_ms'] for r in norm_rows) / len(norm_rows),
            'median_ms': sum(r['forward_latency_ms']['median_ms'] for r in norm_rows) / len(norm_rows),
            'std_ms': sum(r['forward_latency_ms']['std_ms'] for r in norm_rows) / len(norm_rows),
            'n_iters': norm_rows[0]['forward_latency_ms']['n_iters'],
            'n_rounds': len(norm_rows),
        }
        base['training_latency_ms'] = {
            'mean_ms': sum(r['training_latency_ms']['mean_ms'] for r in norm_rows) / len(norm_rows),
            'median_ms': sum(r['training_latency_ms']['median_ms'] for r in norm_rows) / len(norm_rows),
            'std_ms': sum(r['training_latency_ms']['std_ms'] for r in norm_rows) / len(norm_rows),
            'n_iters': norm_rows[0]['training_latency_ms']['n_iters'],
            'n_rounds': len(norm_rows),
        }
        base['latency_by_round'] = [{'round': r['round'],
                                     'forward_ms': r['forward_latency_ms'],
                                     'training_ms': r['training_latency_ms']}
                                    for r in norm_rows]
        base['peak_inference_memory_mb'] = norm_rows[0]['peak_inference_memory_mb']
        base['peak_training_memory_mb'] = norm_rows[0]['peak_training_memory_mb']
        out.append(base)
    return out


def load_losses(summaries_path: str) -> dict:
    data = json.loads(Path(summaries_path).read_text())
    rows = data if isinstance(data, list) else (data.get('rows') or data.get('summaries') or [])
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get('name') is not None and isinstance(row.get('val_loss'), (int, float)):
            out[row['name']] = float(row['val_loss'])
    return out


def load_row_cfgs(summaries_path: str) -> dict:
    """{row name: builder kwargs} for ablation rows that record their config."""
    data = json.loads(Path(summaries_path).read_text())
    rows = data if isinstance(data, list) else (data.get('rows') or data.get('summaries') or [])
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get('name') is not None and isinstance(row.get('cfg'), dict):
            out[row['name']] = row['cfg']
    return out


def _cfg_key(model_name: str, cfg_kwargs: dict) -> str:
    """Canonical identity of the model a config builds (like the ablation runner)."""
    with torch.device('meta'):
        cfg = _BUILDERS[model_name](**cfg_kwargs).cfg
    return json.dumps(dataclasses.asdict(cfg), sort_keys=True)


def load_significance(comparison_path) -> dict:
    """Per-row `beats_reference_significantly` from baseline_comparison.json.

    A point-estimate gain that the ablation runner could not call significant
    (Holm-adjusted) is weak evidence, so the trade-off verdict must be able
    to see it - otherwise a noise-level delta could be reported as a costed
    win.
    """
    if not comparison_path or not Path(comparison_path).exists():
        return {}
    data = json.loads(Path(comparison_path).read_text())
    out = {}
    for row in (data.get('rows') or []):
        if isinstance(row, dict) and row.get('name') is not None:
            out[row['name']] = {
                'beats_reference_significantly': row.get('beats_reference_significantly'),
                'p_one_sided': row.get('p_one_sided'),
                'p_one_sided_holm': row.get('p_one_sided_holm'),
                'delta_vs_reference': row.get('delta_vs_reference'),
                'n_seeds': row.get('n_seeds'),
            }
    return out


def print_table(results: list, reference: str | None) -> None:
    header = f"{'norm':>8s} {'params':>10s} {'norm%':>7s} {'fwd ms/tok':>11s} {'train ms/tok':>13s} {'tok/s':>10s}"
    print(header)
    print('-' * len(header))
    for r in results:
        print(f"{r['norm']:>8s} {r['n_params']:>10d} {100.0 * r['norm_share_of_total']:>6.3f}% "
              f"{r['forward_latency_ms_per_token']:>11.4f} {r['training_latency_ms_per_token']:>13.4f} "
              f"{r['forward_tokens_per_second']:>10.1f}")
    if reference and reference in {r['norm'] for r in results}:
        ref = next(r for r in results if r['norm'] == reference)
        print(f"\n[cost] ratios vs reference '{reference}' (1.0 = identical):")
        for r in results:
            if r['norm'] == reference:
                continue
            print(f"[cost] {r['norm']:>8s} params={r['n_params'] / ref['n_params']:.4f}x "
                  f"fwd_latency={r['forward_latency_ms_per_token'] / ref['forward_latency_ms_per_token']:.4f}x "
                  f"train_latency={r['training_latency_ms_per_token'] / ref['training_latency_ms_per_token']:.4f}x")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', choices=sorted(_BUILDERS), default='gpt2-124m')
    p.add_argument('--norms', nargs='+', default=['ln', 'rmsnorm', 'dyt', 'dytc', 'dytd', 'dytcd'])
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--rounds', type=int, default=3, help='interleaved passes over the norms; each pass measures every norm once and alternates forward/reverse order, so a monotone thermal/clock drift is spread over all norms instead of loading onto the ones measured last. Each norm reports the mean of its per-round means (>= 1; more rounds = more drift cancellation)')
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--include-optimizer', action='store_true', help='include the optimizer step in the training latency')
    p.add_argument('--bf16', action='store_true', help='run the timed passes under bf16 autocast (CUDA only)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default=None)
    p.add_argument('--reference', default='ln', help='reference norm for the cost ratios')
    # cost-adjusted accuracy comparison (optional)
    p.add_argument('--summaries', default=None, help='ablation all_summaries.json to join held-out losses with the measured cost')
    p.add_argument('--comparison', default=None, help='baseline_comparison.json for significance gates; defaults to the sibling of --summaries')
    p.add_argument('--reference-name', default='ln_baseline', help='baseline row in --summaries')
    p.add_argument('--candidate', nargs='+', default=[], help='candidate row names in --summaries to judge against the reference')
    p.add_argument('--min-loss-gain', type=float, default=DEFAULT_MIN_LOSS_GAIN)
    p.add_argument('--max-cost-ratio', type=float, default=DEFAULT_MAX_COST_RATIO)
    p.add_argument('--require-viable', action='store_true', help='exit 1 when a candidate is not viable at the given margins')
    p.add_argument('--output', default='runs/efficiency/efficiency.json')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this benchmark tests (e.g. "DyT-CD reaches a lower loss per point of training cost than RMSNorm at the same lr"); recorded in the report and manifest so the tradeoff verdict is auditable against a claim')
    args = p.parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    set_all_seeds(args.seed)
    print(f'[bench] model={args.model} device={device} batch={args.batch_size} seq={args.seq_len} '
          f'iters={args.iters} warmup={args.warmup} bf16={args.bf16 and _is_cuda(device)}')
    if args.hypothesis:
        print(f'[hypothesis] {args.hypothesis}')
    else:
        print('[hypothesis] none stated: pass --hypothesis "..." so this tradeoff verdict is auditable as a test of a claim, not a sweep')
    rounds = max(1, int(args.rounds))
    if rounds != args.rounds:
        print(f'[bench][warn] --rounds={args.rounds} clamped to {rounds}')
    measurement_order = []
    raw_rows = []
    for r_i in range(rounds):
        order = list(args.norms) if r_i % 2 == 0 else list(reversed(args.norms))
        measurement_order.append(order)
        print(f'[bench] round {r_i + 1}/{rounds} order: {order}')
        for norm in order:
            row = benchmark_norm(args.model, norm, args, device, measure_heavy=(r_i == 0))
            row['round'] = r_i
            raw_rows.append(row)
    results = aggregate_benchmark_rounds(raw_rows, args.norms, rounds)
    for r in results:
        print(f"[bench] {r['norm']:>8s}: params={r['n_params']} norm_share={100.0 * r['norm_share_of_total']:.4f}% "
              f"fwd={r['forward_latency_ms_per_token']:.4f} ms/tok train={r['training_latency_ms_per_token']:.4f} ms/tok "
              f"measured_flops={r['measured_forward_flops_per_token'] and round(r['measured_forward_flops_per_token'])} "
              f"({r['forward_latency_ms']['n_rounds']} round(s) averaged)")
    print_table(results, args.reference)

    tradeoff = None
    if args.summaries:
        losses = load_losses(args.summaries)
        if args.reference_name not in losses:
            print(f"[tradeoff] FAIL: reference row '{args.reference_name}' not in {args.summaries}", file=sys.stderr)
            raise SystemExit(1)
        comparison_path = args.comparison
        if comparison_path is None:
            sibling = Path(args.summaries).with_name('baseline_comparison.json')
            comparison_path = str(sibling) if sibling.exists() else None
        significance = load_significance(comparison_path)
        if significance:
            print(f"[tradeoff] significance gate -> {comparison_path}")
        # The candidate cost is measured here for the same model/norm; the
        # reference cost is the --reference norm's measurement. The benchmark
        # only builds the base norm, so a row whose trained architecture
        # differs beyond that base (qk_norm / activation / parameterization
        # variants) gets a proxy cost: flagged per row instead of silently
        # substituting the model.
        by_norm = {r['norm']: r for r in results}
        cand_cfgs = load_row_cfgs(args.summaries)
        rows = []
        for cand in args.candidate:
            cand_loss = losses.get(cand)
            if cand_loss is None:
                rows.append({'candidate': cand, 'error': 'no val_loss in summaries'})
                continue
            cand_norm = cand.split('_')[0].replace('-', '')
            cand_norm = 'ln' if cand_norm == 'ln' else cand_norm
            cand_bench = by_norm.get(cand_norm)
            ref_bench = by_norm.get(args.reference)
            if cand_bench is None or ref_bench is None:
                rows.append({'candidate': cand, 'error': f'missing benchmark for norm {cand_norm!r} or reference {args.reference!r}'})
                continue
            cand_cfg = cand_cfgs.get(cand)
            cost_is_proxy = None
            if isinstance(cand_cfg, dict):
                try:
                    cost_is_proxy = (_cfg_key(args.model, cand_cfg)
                                     != _cfg_key(args.model, {'norm_name': cand_norm}))
                except Exception:
                    cost_is_proxy = bool(set(cand_cfg) - {'norm_name'})
                if cost_is_proxy:
                    print(f"[tradeoff][warn] {cand}: the benchmark has no row for its exact "
                          f"architecture; cost is measured on the base '{cand_norm}' model "
                          f"(proxy cost)")
            loss_improvement = losses[args.reference_name] - cand_loss
            cost_ratio = cand_bench['training_latency_ms_per_token'] / ref_bench['training_latency_ms_per_token']
            verdict = efficiency_verdict(loss_improvement, cost_ratio,
                                         min_loss_gain=args.min_loss_gain,
                                         max_cost_ratio=args.max_cost_ratio,
                                         cost_name='training_latency')
            # A gain the seed-level ablation could not call significant is
            # weak evidence: block viability until a longer/multi-seed run
            # clears the significance bar.
            sig = significance.get(cand)
            if sig:
                verdict['statistically_significant'] = sig['beats_reference_significantly']
                verdict['p_one_sided_holm'] = sig['p_one_sided_holm']
                verdict['n_seeds'] = sig['n_seeds']
                if sig['beats_reference_significantly'] is False:
                    verdict['viable'] = False
                    verdict['reason'] += ('; ablation comparison on the same seeds does not call the '
                                          'gain significant (Holm-adjusted), so the point estimate is weak evidence')
            rows.append({
                'candidate': cand,
                'candidate_norm': cand_norm,
                'candidate_cfg': cand_cfg,
                'reference_row': args.reference_name,
                'reference_norm': args.reference,
                'candidate_val_loss': cand_loss,
                'reference_val_loss': losses[args.reference_name],
                'significance': sig,
                'cost_model_norm': cand_norm,
                'cost_is_proxy': cost_is_proxy,
                'verdict': verdict,
            })
        tradeoff = {
            'summaries': args.summaries,
            'comparison': comparison_path,
            'reference_row': args.reference_name,
            'reference_norm': args.reference,
            'min_loss_gain': args.min_loss_gain,
            'max_cost_ratio': args.max_cost_ratio,
            'rows': rows,
            'cost_model': ("each candidate's cost is measured on the base norm taken "
                           "from its row name; rows whose trained architecture differs "
                           "beyond that base (e.g. qk_norm or activation variants) record "
                           "cost_is_proxy=true and their cost ratio is a proxy"),
        }
        for row in rows:
            if row.get('error'):
                print(f"[tradeoff] {row['candidate']}: {row['error']}")
                continue
            v = row['verdict']
            sig_str = ''
            if v.get('statistically_significant') is not None:
                sig_str = f" significant={v['statistically_significant']} (p_holm={v['p_one_sided_holm']})"
            proxy_str = ' [proxy cost]' if row.get('cost_is_proxy') else ''
            print(f"[tradeoff] {row['candidate']}: val_loss={row['candidate_val_loss']:.4f} vs "
                  f"{row['reference_row']}={row['reference_val_loss']:.4f} "
                  f"gain={v['loss_improvement']:+.4f} cost={v['cost_ratio']:.3f}x "
                  f"cost/loss_point={v['cost_per_loss_point'] and round(v['cost_per_loss_point'], 3)} "
                  f"viable={v['viable']}{sig_str} ({v['reason']}){proxy_str}")
        if args.require_viable:
            not_viable = [row['candidate'] for row in rows if row.get('verdict', {}).get('viable') is not True]
            if not_viable:
                raise SystemExit(f'[tradeoff] --require-viable: {len(not_viable)} candidate(s) not viable '
                                 f'or not evaluable: {", ".join(not_viable)}')

    report = {
        'model': args.model,
        'device': device,
        'hypothesis': args.hypothesis,
        'config': vars(args),
        'results': results,
        'tradeoff': tradeoff,
        'rounds': rounds,
        'measurement_order': measurement_order,
        'method': {
            'latency': (f"per norm, the mean of {rounds} interleaved round mean(s) over {args.iters} timed iterations "
                        f"after {args.warmup} warmup each; rounds alternate forward/reverse norm order so drift is "
                        f"not confounded with norm identity (raw per-round timings in latency_by_round); "
                        f"per token = per step / (batch*seq)"),
            'flops': 'analytic = matmul FLOPs per token; measured = torch FlopCounterMode forward',
            'memory': 'peak CUDA allocated; null on CPU',
            'inputs': ('common random numbers: one Generator seeded with --seed draws the '
                       'timed x/y for every norm, so a slower step is not a different input'),
        },
        'run_metadata': collect_run_metadata(seed=args.seed),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    save_run_manifest(out.with_name('efficiency_manifest.json'), finish_metadata({
        'run_metadata': report['run_metadata'], 'hypothesis': args.hypothesis,
        'config': vars(args), 'results': results, 'tradeoff': tradeoff,
    }, artifacts={'efficiency_report': file_fingerprint(out)}))
    print(f'[bench] report -> {out}')


if __name__ == '__main__':
    main()
