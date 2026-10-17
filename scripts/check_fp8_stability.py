from __future__ import annotations
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from adasquash.metrics import holm_adjust, mean_std_stderr, paired_t_one_sided
from adasquash.model import MLP, build_gpt_124m
from adasquash.repro import collect_run_metadata, save_run_manifest, set_all_seeds
from adasquash.utils import KurtosisMeter, markov_batch
FP8_E4M3_MAX = 448.0
DEFAULT_SEEDS = (42, 43, 44)
# One probe draw is one sample: these per-norm quantities are aggregated as
# Mean +/- Std over the `--seeds` replications, with the per-seed records kept
# so an unstable seed cannot hide behind a favorable mean. Cross-norm
# differences are additionally tested for significance in `compare_fp8_norms`
# (paired across the shared seeds, Holm-adjusted over metrics x pairs), because
# a "more stable" claim resting on a mean difference inside the seed spread is
# not a result; the comparison is saved next to the results as
# `comparison.json`.
_FP8_SEED_METRIC_KEYS = (
    'max_abs_pre_norm', 'max_abs_post_norm', 'mean_excess_kurtosis_pre_norm',
    'mean_overflow_rate_e4m3', 'max_overflow_rate_e4m3',
    'mean_underflow_rate_e4m3', 'mean_rel_quant_err_e4m3',
    'logit_rel_divergence',
)

def fp8_min_normal(exp_bits: int) -> float:
    return 2.0 ** (1 - (2 ** (exp_bits - 1) - 1))

def to_fp8_e4m3(x: torch.Tensor, axis: int=-1) -> torch.Tensor:
    return simulate_fp8(x, max_val=448.0, exp_bits=4, mantissa_bits=3)

def to_fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    return simulate_fp8(x, max_val=57344.0, exp_bits=5, mantissa_bits=2)

def simulate_fp8(x: torch.Tensor, max_val: float, exp_bits: int, mantissa_bits: int) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.float()
    levels = float(2 ** mantissa_bits)
    a = x.clamp(-max_val, max_val)
    min_exp = 2 - (1 << exp_bits - 1) - mantissa_bits
    mant, exp = torch.frexp(a)
    exp.sub_(1 + mantissa_bits)
    mant.mul_(2.0 * levels)
    mant.round_()
    mant.masked_fill_(exp < min_exp, 0.0)
    return torch.ldexp(mant, exp).to(in_dtype)

class FP8LinearSim:

    def __init__(self, module: nn.Module, quant):
        self.module = module
        self.quant = quant
        self.orig_forward = module.forward
        with torch.no_grad():
            self.qw = quant(module.weight)

        def fwd(x):
            y = F.linear(self.quant(x), self.qw, module.bias)
            return self.quant(y)
        module.forward = fwd

class FP8SwiGLUSim:

    def __init__(self, mlp: MLP, quant):
        self.mlp = mlp
        self.quant = quant
        self.orig_forward = mlp.forward
        with torch.no_grad():
            self.qw1 = quant(mlp.w1.weight)
            self.qw2 = quant(mlp.w2.weight)
            self.qw3 = quant(mlp.w3.weight)

        def fwd(x):
            q = self.quant
            x_q = q(x)
            a = q(F.linear(x_q, self.qw1))
            b = q(F.linear(x_q, self.qw3))
            h = F.silu(a)
            h.mul_(b)
            return mlp.drop(q(F.linear(q(h), self.qw2)))
        mlp.forward = fwd

    @property
    def module(self):
        return self.mlp

@contextmanager
def fp8_forward(model: nn.Module, quant):
    patched = []
    try:
        for m in model.modules():
            if isinstance(m, MLP):
                patched.append(FP8SwiGLUSim(m, quant))
            elif isinstance(m, nn.Linear):
                patched.append(FP8LinearSim(m, quant))
        yield
    finally:
        for p in patched:
            p.module.forward = p.orig_forward

def measure_fp8_divergence(model, x, norm_name: str):
    handles = []
    quant_errors = []
    ranges = []
    fp8_e4m3_min = fp8_min_normal(4)

    def _zero_like_scalar(ref: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), dtype=ref.dtype, device=ref.device)

    def make_stat_hook():

        def hook(module, inp, out):
            x_in = inp[0] if isinstance(inp, tuple) and inp else None
            cls_name = module.__class__.__name__
            has_in = torch.is_tensor(x_in)
            abs_out = out.abs()
            abs_in = x_in.abs() if has_in else None
            q = to_fp8_e4m3(out)
            scalars = [abs_out.max(), abs_out.mean(), (abs_out > FP8_E4M3_MAX).float().mean(), (abs_out < fp8_e4m3_min).float().mean() if out.numel() else _zero_like_scalar(abs_out), q.sub_(out).abs_().mean(), abs_in.max() if has_in else _zero_like_scalar(abs_out), _excess_kurtosis(x_in) if has_in else _zero_like_scalar(abs_out)]
            max_abs_out, mean_abs_out, overflow, underflow, err, max_abs_in, kurtosis_in = torch.stack([s.detach().reshape(()).double() for s in scalars]).tolist()
            denom = mean_abs_out + 1e-06
            ranges.append({'module': cls_name, 'max_abs_in': max_abs_in, 'max_abs_out': max_abs_out, 'kurtosis_in': kurtosis_in})
            quant_errors.append({'module': cls_name, 'max_abs': max_abs_out, 'overflow_rate': overflow, 'underflow_rate': underflow, 'rel_err': err / denom})
        return hook
    first_bad = []
    first_over = []

    def make_block_hook(i):

        def hook(module, inp, out):
            y = out[0] if isinstance(out, tuple) else out
            all_finite = torch.isfinite(y).all()
            first_bad.append((i, all_finite))
            first_over.append((i, y.abs().amax()))
        return hook
    with torch.no_grad():
        logits_fp32, _ = model(x)
    for name, m in model.named_modules():
        if m.__class__.__name__ in {'LayerNorm', 'RMSNorm', 'DyT', 'DyTC', 'DyTD', 'DyTCD', 'DyTCDTanh'}:
            handles.append(m.register_forward_hook(make_stat_hook()))
    block_handles = [blk.register_forward_hook(make_block_hook(i)) for i, blk in enumerate(model.blocks)]
    with fp8_forward(model, to_fp8_e4m3):
        with torch.no_grad():
            logits_fp8, _ = model(x)
    for h in block_handles:
        h.remove()
    for h in handles:
        h.remove()
    bad_flags = torch.stack([f for _, f in first_bad]).tolist() if first_bad else []
    over_vals = torch.stack([v for _, v in first_over]).tolist() if first_over else []
    first_bad_idx = next((i for i, ok in zip([i for i, _ in first_bad], bad_flags) if not ok), None)
    first_over_idx = next((i for i, v in zip([i for i, _ in first_over], over_vals) if v > FP8_E4M3_MAX), None)
    max_abs_fp32 = logits_fp32.abs().max()
    finite = torch.isfinite(logits_fp8).all()
    if finite:
        rel = (logits_fp8.sub_(logits_fp32).abs_().max() / max_abs_fp32.clamp(min=1e-12)).item()
        logit_nan = False
    else:
        rel = float('inf')
        logit_nan = True
    if not quant_errors:
        return {'norm': norm_name, 'n_layers': int(getattr(model.cfg, 'n_layers', 0)), 'n_norm_hooks': 0, 'max_abs_pre_norm': 0.0, 'max_abs_post_norm': 0.0, 'mean_excess_kurtosis_pre_norm': 0.0, 'mean_overflow_rate_e4m3': 0.0, 'max_overflow_rate_e4m3': 0.0, 'mean_underflow_rate_e4m3': 0.0, 'mean_rel_quant_err_e4m3': 0.0, 'logit_rel_divergence': rel, 'logit_nan_under_fp8': logit_nan, 'first_nonfinite_layer': first_bad_idx, 'first_overflow_layer': first_over_idx}
    mean_overflow = sum((e['overflow_rate'] for e in quant_errors)) / len(quant_errors)
    max_overflow = max((e['overflow_rate'] for e in quant_errors))
    mean_underflow = sum((e['underflow_rate'] for e in quant_errors)) / len(quant_errors)
    mean_rel_err = sum((e['rel_err'] for e in quant_errors)) / len(quant_errors)
    max_max = max((e['max_abs'] for e in quant_errors))
    max_in = max((r['max_abs_in'] for r in ranges)) if ranges else 0.0
    mean_kurt = sum((r['kurtosis_in'] for r in ranges)) / len(ranges) if ranges else 0.0
    return {'norm': norm_name, 'n_layers': int(getattr(model.cfg, 'n_layers', 0)), 'n_norm_hooks': len(quant_errors), 'max_abs_pre_norm': max_in, 'max_abs_post_norm': max_max, 'mean_excess_kurtosis_pre_norm': mean_kurt, 'mean_overflow_rate_e4m3': mean_overflow, 'max_overflow_rate_e4m3': max_overflow, 'mean_underflow_rate_e4m3': mean_underflow, 'mean_rel_quant_err_e4m3': mean_rel_err, 'logit_rel_divergence': rel, 'logit_nan_under_fp8': logit_nan, 'first_nonfinite_layer': first_bad_idx, 'first_overflow_layer': first_over_idx}

def _excess_kurtosis(x: torch.Tensor):
    with torch.no_grad():
        flat = x.detach().flatten()
        if flat.numel() < 4:
            return flat.new_zeros((), dtype=torch.float64)
        meter = _Float64KurtosisMeter()
        step = _Float64KurtosisMeter._chunk
        for start in range(0, flat.numel(), step):
            meter.update(flat[start:start + step])
        return flat.new_zeros((), dtype=torch.float64) + meter.result()

class _Float64KurtosisMeter(KurtosisMeter):
    acc_dtype = torch.float64

def aggregate_fp8_seed_rows(norm_name: str, per_seed: list) -> dict:
    """Mean +/- Std of one norm's FP8 metrics over independent probes.

    Each seed is one independent probe draw (and, for a fresh init, one
    independent model initialization). The first record supplies the static
    fields (`n_layers`, `n_norm_hooks`); every metric in
    `_FP8_SEED_METRIC_KEYS` becomes `<key>` (mean) and `<key>_std` (sample
    std across seeds). `logit_nan_under_fp8`/`first_nonfinite_layer` are not
    averaged - a failure in any seed is a failure of the norm - so they are
    reported with `_any` / `_earliest` semantics.
    """
    if not per_seed:
        raise ValueError('aggregate_fp8_seed_rows needs at least one seed record')
    agg = {k: v for k, v in per_seed[0].items() if not k.startswith('__')}
    agg.pop('seed', None)
    agg['norm'] = norm_name
    agg['seeds'] = [r.get('seed') for r in per_seed]
    agg['n_seeds'] = len(per_seed)
    for key in _FP8_SEED_METRIC_KEYS:
        vals = [r[key] for r in per_seed if isinstance(r.get(key), (int, float))]
        stats = mean_std_stderr(vals)
        if stats is None:
            continue
        agg[key] = stats['mean']
        agg[key + '_std'] = stats['std'] if stats['n'] >= 2 else None
    nonfinite = [r['first_nonfinite_layer'] for r in per_seed
                 if isinstance(r.get('first_nonfinite_layer'), int)]
    overflowing = [r['first_overflow_layer'] for r in per_seed
                   if isinstance(r.get('first_overflow_layer'), int)]
    agg['first_nonfinite_layer'] = min(nonfinite) if nonfinite else None
    agg['first_nonfinite_layer_earliest'] = agg['first_nonfinite_layer']
    agg['first_overflow_layer'] = min(overflowing) if overflowing else None
    agg['first_overflow_layer_earliest'] = agg['first_overflow_layer']
    agg['logit_nan_under_fp8'] = any(bool(r.get('logit_nan_under_fp8')) for r in per_seed)
    agg['per_seed'] = per_seed
    return agg

# The seed-varying metrics on which "norm A is more FP8-stable than norm B" is
# actually decided. All three share the same direction: lower is more stable.
# `first_overflow_layer` / `first_nonfinite_layer` are event counts, not
# continuous per-seed quantities, so they are reported with `_any` /
# `_earliest` semantics instead of being averaged or tested.
FP8_COMPARISON_METRICS = ('logit_rel_divergence', 'mean_overflow_rate_e4m3',
                          'mean_underflow_rate_e4m3')

def _seed_metric_values(agg: dict, key: str) -> list:
    return [float(r[key]) for r in agg.get('per_seed', [])
            if isinstance(r.get(key), (int, float))]

def _shared_seed_samples(agg_a: dict, agg_b: dict, key: str) -> tuple[list, list, list]:
    """Per-seed values of `key` for the seeds both norms measured, paired by seed.

    For a fixed seed the probe draw (and, in random-init mode, the model
    initialization seed) is identical for every norm, so the two norms'
    records are paired samples - the same reason the ablation runner pairs its
    rows by seed.
    """
    a_by_seed = {r.get('seed'): r.get(key) for r in agg_a.get('per_seed', [])}
    b_by_seed = {r.get('seed'): r.get(key) for r in agg_b.get('per_seed', [])}
    shared = sorted(s for s in set(a_by_seed) & set(b_by_seed)
                    if isinstance(a_by_seed[s], (int, float))
                    and isinstance(b_by_seed[s], (int, float)))
    return shared, [float(a_by_seed[s]) for s in shared], [float(b_by_seed[s]) for s in shared]

def compare_fp8_norms(results: list, metrics: tuple=FP8_COMPARISON_METRICS, alpha: float=0.05) -> dict:
    """Paired, Holm-adjusted significance of one norm's FP8 metrics vs another's.

    For every metric and every unordered norm pair the test is the one-sided
    paired Student-t `H1: mean(A - B) < 0`, so a negative `mean_delta` means A
    is more FP8-stable on that metric. Pairs with fewer than 2 shared seeds
    are reported without a p-value rather than silently tested unpaired.
    p-values are Holm-adjusted across the whole metric x pair family, so
    scanning six norms does not hand a "significant" win to the luckiest pair.
    `main` writes this next to the results as `comparison.json`, mirroring the
    ablation runner's `leave_one_out_comparison.json`.
    """
    labels = [str(r.get('norm') or r.get('ckpt') or f'run{i}') for i, r in enumerate(results)]
    tests = []
    for metric in metrics:
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                shared, av, bv = _shared_seed_samples(results[i], results[j], metric)
                t = {'a': labels[i], 'b': labels[j], 'metric': metric,
                     'n_shared_seeds': len(shared), 'shared_seeds': shared,
                     'direction': f'negative mean_delta means A has lower {metric} (more FP8-stable)'}
                if len(shared) < 2:
                    t['mean_delta'] = (sum(av) - sum(bv)) / len(shared) if shared else None
                    t['p_one_sided'] = None
                    t['reason'] = 'need >= 2 seeds measured for both norms to pair a test'
                else:
                    t.update(paired_t_one_sided(av, bv, alpha=alpha))
                tests.append(t)
    for t, p_adj in zip(tests, holm_adjust([t.get('p_one_sided') for t in tests])):
        t['p_holm'] = p_adj
        t['significant'] = bool(p_adj is not None and p_adj < alpha)
    ranking = {}
    for metric in metrics:
        rows = []
        for label, r in zip(labels, results):
            vals = _seed_metric_values(r, metric)
            stats = mean_std_stderr(vals)
            rows.append({'norm': label, 'n_seeds': len(vals),
                         'mean': stats['mean'] if stats else None,
                         'std': stats['std'] if stats and stats['n'] >= 2 else None})
        rows.sort(key=lambda row: (row['mean'] is None, row['mean']))
        ranking[metric] = rows
    return {'metrics': list(metrics), 'alpha': alpha,
            'pairing': 'per-seed probe draws are identical across norms (and identical init seeds in random-init mode)',
            'ranking': ranking, 'tests': tests}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--norms', nargs='+', default=['ln', 'rmsnorm', 'dyt', 'dytc', 'dytd', 'dytcd'])
    p.add_argument('--probe', default='markov', choices=['markov', 'random'], help="probe corpus. 'random' is uniform token ids: it has no next-token structure, so the activation magnitudes below describe embedding noise rather than a trained model's activations (kept only for parity)")
    p.add_argument('--ckpt', nargs='+', default=None, help='one checkpoint per --norms entry; when given, the probe runs on the trained weights instead of a fresh random init')
    p.add_argument('--n-states', type=int, default=64, help="state count of the synthetic 'markov' probe")
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS), help='each norm is probed once per seed (fresh probe, and fresh init in random-init mode) and reported as Mean +/- Std across seeds; >= 3 recommended so one probe draw cannot decide a comparison')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the paired cross-norm tests in comparison.json (Holm-adjusted over metrics x pairs)')
    p.add_argument('--output', default='runs/fp8_check/results.json')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this check tests (e.g. "DyT-CD limits fp8 activation blowup versus RMSNorm at matched init"); recorded in the run manifest so the result can be audited against the claim it was meant to support')
    args = p.parse_args()
    if args.ckpt and len(args.ckpt) != len(args.norms):
        p.error(f'--ckpt expects {len(args.norms)} paths (one per --norms), got {len(args.ckpt)}')
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if not seeds:
        p.error('--seeds must not be empty')
    if len(seeds) < 3:
        print(f'[fp8][warn] {len(seeds)} seed(s): one probe draw is one sample; use >= 3 independent probes before treating a metric difference as real')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seeds[0])

    def make_probe(rng, cfg):
        if args.probe == 'random':
            return torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), generator=rng, device=device)
        x, _ = markov_batch(rng, cfg.ignore_index, args.batch_size, args.seq_len, device, n_states=args.n_states)
        return x
    results = []
    for i, norm in enumerate(args.norms):
        if args.ckpt:
            from scripts.eval_downstream import load_ckpt
        else:
            print(f'[fp8] WARNING: {norm} probed at a fresh random init. layerscale_init=1e-5 leaves the residual stream at the embedding table, so every norm reports the same pre-norm statistics and this row says nothing about the layer. Pass --ckpt <path> (one per --norms entry) to measure trained weights.', file=sys.stderr)
        per_seed = []
        for seed in seeds:
            set_all_seeds(seed)
            rng = torch.Generator(device=device).manual_seed(seed)
            if args.ckpt:
                model, _ = load_ckpt(args.ckpt[i], device=device)
            else:
                model = build_gpt_124m(norm_name=norm).to(device)
                model.reset_parameters(seed)
            model.eval()
            x = make_probe(rng, model.cfg)
            r = measure_fp8_divergence(model, x, norm)
            r['seed'] = seed
            per_seed.append(r)
        r = aggregate_fp8_seed_rows(norm, per_seed)
        if args.ckpt:
            r['ckpt'] = args.ckpt[i]
        print(f'[fp8] {norm}: {r}')
        results.append(r)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    comparison = compare_fp8_norms(results, alpha=args.alpha)
    comparison_path = out_path.with_name('comparison.json')
    comparison_path.write_text(json.dumps(comparison, indent=2))
    for t in comparison['tests']:
        if t.get('p_one_sided') is None:
            print(f"[fp8][compare] {t['a']} vs {t['b']} on {t['metric']}: {t['reason']}")
            continue
        verdict = 'A more stable' if t['significant'] and t['mean_delta'] < 0 else ('B more stable' if t['significant'] else 'inside the noise')
        print(f"[fp8][compare] {t['a']} vs {t['b']} on {t['metric']}: delta={t['mean_delta']:+.4g}, 95% CI [{t['ci_low']:+.4g}, {t['ci_high']:+.4g}], Holm p={t['p_holm']:.4f} (n={t['n_shared_seeds']}) -> {verdict}")
    manifest = collect_run_metadata(seed=seeds[0])
    manifest['hyperparameters'] = vars(args)
    manifest['seeds'] = seeds
    if args.hypothesis:
        manifest['hypothesis'] = args.hypothesis
        print(f'[hypothesis] {args.hypothesis}')
    else:
        print('[hypothesis] none stated: pass --hypothesis "..." so this check is auditable as a test of a claim, not a sweep')
    manifest['results_file'] = str(out_path)
    manifest['comparison_file'] = str(comparison_path)
    save_run_manifest(out_path.with_name('run_manifest.json'), manifest)
    print(f'[done] saved to {out_path}')
if __name__ == '__main__':
    main()
