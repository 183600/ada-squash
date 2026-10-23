from __future__ import annotations
import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust, mean_std_stderr,
                               paired_t_one_sided, scale_replication_summary)
from adasquash.model import (MLP, build_gpt_124m, build_gpt_350m,
                             build_llama_tiny_1b)
from adasquash.repro import (TRAINING_RECIPE_KEYS, collect_run_metadata, file_fingerprint,
                             finish_metadata, save_run_manifest, set_all_seeds,
                             training_recipe_mismatches)
from adasquash.utils import KurtosisMeter, markov_batch
from scripts.data_pipeline import _is_pinned_revision
FP8_E4M3_MAX = 448.0
DEFAULT_SEEDS = (42, 43, 44)
# The FP8 probe must not certify a mechanism from one architecture. These are
# the builder targets accepted by --models (random-init mode); each produces a
# distinct shape label in comparison.json's scale_replication block.
_FP8_MODEL_BUILDERS = {
    'gpt2-124m': build_gpt_124m,
    'gpt2-350m': build_gpt_350m,
    'llama-1b': build_llama_tiny_1b,
}
# One probe draw is one sample: these per-norm quantities are aggregated as
# Mean +/- Std over the `--seeds` replications, with the per-seed records kept
# so an unstable seed cannot hide behind a favorable mean.
#
# Cross-norm certification needs both dimensions to be replicated:
# * the training run (initialization / data order) is the run-to-run unit. A
#   `--ckpt` entry may be a comma-separated list of checkpoint paths (training
#   seeds) for one norm; when two norms share >= 3 *training* seeds the
#   comparison pairs their per-training-seed probe means and certifies
#   significance on that basis. Two shared training seeds are descriptive
#   only (df = 1).
# * a single trained checkpoint per norm (the common case) leaves only probe
#   noise in the denominator: `compare_fp8_norms(trained=True)` then reports
#   the probe-paired tests as descriptive and never calls them significant,
#   because "norm A is more FP8-stable" is a claim about training runs, not
#   about one draw of probe batches on one trained model. In random-init mode
#   each seed re-initializes the model too, so the paired test does cover
#   independent (init, probe) draws and is certifiable once there are at least
#   three of them - but it still says nothing about trained weights. Two
#   independent draws (df = 1) are descriptive only, like two training seeds.
# The comparison is saved next to the results as `comparison.json`.
_FP8_SEED_METRIC_KEYS = (
    'max_abs_pre_norm', 'max_abs_post_norm', 'mean_excess_kurtosis_pre_norm',
    'mean_overflow_rate_e4m3', 'max_overflow_rate_e4m3',
    'mean_underflow_rate_e4m3', 'mean_subnormal_rate_e4m3',
    'mean_quantized_zero_rate_e4m3', 'mean_rel_quant_err_e4m3',
    'logit_rel_divergence',
)

def fp8_min_normal(exp_bits: int) -> float:
    return 2.0 ** (1 - (2 ** (exp_bits - 1) - 1))

def to_fp8_e4m3(x: torch.Tensor, axis: int=-1, inplace: bool=False) -> torch.Tensor:
    return simulate_fp8(x, max_val=448.0, exp_bits=4, mantissa_bits=3, inplace=inplace)

def to_fp8_e5m2(x: torch.Tensor, inplace: bool=False) -> torch.Tensor:
    return simulate_fp8(x, max_val=57344.0, exp_bits=5, mantissa_bits=2, inplace=inplace)

# The quantizer identity travels with the callable so range diagnostics can ask
# which format/scaling was actually applied instead of assuming e4m3.
to_fp8_e4m3.max_val = 448.0
to_fp8_e4m3.min_normal = fp8_min_normal(4)
to_fp8_e4m3.scaling = 'none'
to_fp8_e5m2.max_val = 57344.0
to_fp8_e5m2.min_normal = fp8_min_normal(5)
to_fp8_e5m2.scaling = 'none'

def make_fp8_quant(fmt: str='e4m3', scaling: str='none'):
    """Return a quantizer with the requested format and scaling recipe.

    ``scaling='none'`` is the raw-value worst case: values beyond the format
    range clip, which is a range stress test, not a prediction of real FP8
    training. ``scaling='dynamic-tensor'`` rescales each tensor by its amax
    into the full representable range before rounding and rescales back after,
    the per-tensor dynamic recipe real FP8 stacks use. Keeping the two
    explicit prevents a clipped stress-test number from being quoted as a
    training-stability prediction.
    """
    if fmt == 'e5m2':
        base, maxv = (to_fp8_e5m2, 57344.0)
    elif fmt == 'e4m3':
        base, maxv = (to_fp8_e4m3, 448.0)
    else:
        raise ValueError(f"unknown fp8 format {fmt!r} (e4m3 | e5m2)")
    if scaling == 'none':
        return base
    if scaling != 'dynamic-tensor':
        raise ValueError(f"unknown scaling {scaling!r} (none | dynamic-tensor)")

    def q(x: torch.Tensor, inplace: bool=False) -> torch.Tensor:
        # A detached amax is a per-tensor statistic; the quantization is
        # already a forward-only simulation, and using detect would break
        # the straight-through-free demo path.
        amax = x.detach().abs().amax().clamp_(min=1e-12)
        scale = maxv / amax
        out = base(x * scale)
        out = out / scale
        if inplace and x.dtype == torch.float32:
            x.copy_(out)
            return x
        return out
    # The metric hook needs to know that the overflow/underflow thresholds
    # apply to the *scaled* tensor, not the raw activation. Without this the
    # rates describe a quantization recipe that was not applied.
    q.scaling = scaling
    q.max_val = maxv
    q.min_normal = fp8_min_normal(4 if fmt == 'e4m3' else 5)
    return q

# Reusable frexp/mask scratch keyed by (shape, device).
_FP8_SCRATCH: dict = {}

def _fp8_scratch(shape, device):
    key = (tuple(shape), str(device))
    got = _FP8_SCRATCH.get(key)
    # A scratch set allocated under torch.inference_mode() is an inference
    # tensor; reusing it as an `out=` target outside that mode raises. It is
    # only scratch data, so rebuild a normal set for the grad-enabled caller.
    if (got is not None and not torch.is_inference_mode_enabled()
            and any(t.is_inference() for t in got)):
        got = None
    if got is None:
        got = (torch.empty(shape, dtype=torch.float32, device=device),
               torch.empty(shape, dtype=torch.int32, device=device),
               torch.empty(shape, dtype=torch.bool, device=device))
        _FP8_SCRATCH[key] = got
    return got

def simulate_fp8(x: torch.Tensor, max_val: float, exp_bits: int, mantissa_bits: int, inplace: bool=False) -> torch.Tensor:
    """Round `x` to the fp8 grid (fp32 math throughout).

    `inplace=True` means the caller hands over ownership of an fp32 `x` (e.g. a
    fresh `F.linear` output) and gets the result written back into it.
    """
    in_dtype = x.dtype
    if inplace and in_dtype == torch.float32:
        a = x
    else:
        a = x.float().clone()
    levels = float(2 ** mantissa_bits)
    a.clamp_(-max_val, max_val)
    # This simulation intentionally flushes everything below the min *normal*
    # to zero: the repo's fixed test contract treats the subnormal range as
    # underflow (see tests/test_norms.py). Keep the quantizer semantics stable;
    # the subnormal share is reported separately by the metric hook where it
    # can be read without redefining the format's zero threshold.
    min_exp = 2 - (1 << exp_bits - 1) - mantissa_bits
    mant, exp, low = _fp8_scratch(a.shape, a.device)
    torch.frexp(a, out=(mant, exp))
    exp.sub_(1 + mantissa_bits)
    mant.mul_(2.0 * levels)
    mant.round_()
    torch.lt(exp, min_exp, out=low)
    mant.masked_fill_(low, 0.0)
    torch.ldexp(mant, exp, out=a)
    return a.to(in_dtype)

class FP8LinearSim:

    def __init__(self, module: nn.Module, quant):
        self.module = module
        self.quant = quant
        self.orig_forward = module.forward
        with torch.no_grad():
            self.qw = quant(module.weight)

        def fwd(x):
            y = F.linear(self.quant(x), self.qw, module.bias)
            # The rounding writes into the freshly allocated output.
            return self.quant(y, inplace=True)
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
            # Every other tensor here is owned by this closure.
            a = q(F.linear(x_q, self.qw1), inplace=True)
            b = q(F.linear(x_q, self.qw3), inplace=True)
            h = F.silu(a)
            h.mul_(b)
            return mlp.drop(q(F.linear(q(h, inplace=True), self.qw2), inplace=True))
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

def measure_fp8_divergence(model, x, norm_name: str, quant=None):
    handles = []
    quant_errors = []
    ranges = []
    quant = quant or to_fp8_e4m3
    quant_max = float(getattr(quant, 'max_val', FP8_E4M3_MAX))
    quant_min = float(getattr(quant, 'min_normal', fp8_min_normal(4)))
    quant_dynamic = getattr(quant, 'scaling', None) == 'dynamic-tensor'

    def _zero_scalar(device) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float64, device=device)

    def make_stat_hook():

        def hook(module, inp, out):
            # The norm hooks collect tensors on the device; they are converted
            # after the forward.
            x_in = inp[0] if isinstance(inp, tuple) and inp else None
            cls_name = module.__class__.__name__
            has_in = torch.is_tensor(x_in)
            zero = _zero_scalar(out.device)
            numel = out.numel()
            abs_out = out.abs()
            max_abs_out = torch.maximum(out.max(), -out.min())
            mean_abs_out = abs_out.mean()
            # With dynamic per-tensor scaling the quantizer first maps the
            # tensor's amax into the format range. The overflow/underflow
            # thresholds therefore apply to that scaled image; using raw
            # `abs_out` would report 100% underflow for tiny activations even
            # though the applied quantizer loses nothing.
            if quant_dynamic and numel:
                amax = out.detach().abs().amax().clamp_(min=1e-12)
                # Do the rescale in float64: `448 * (448/448)` in float32 can
                # round just above the format maximum and report a phantom
                # overflow on a quantizer that by construction never clips.
                # Clamp the scaled image to the format max for the same reason
                # in float64: amax*(max/amax) can still round to
                # 448.00000000000006 and be counted as an overflow.
                scaled_abs = (abs_out.double()
                              * (quant_max / amax.double())).clamp_(max=quant_max)
                threshold_max = quant_max
            else:
                scaled_abs = abs_out
                threshold_max = quant_max
            if numel:
                overflow = torch.count_nonzero(scaled_abs > threshold_max).to(torch.float64) / numel
                underflow = torch.count_nonzero(scaled_abs < quant_min).to(torch.float64) / numel
                # A subnormal is representable but below the min normal; a
                # quantized zero is a value the format could not represent at
                # all. Report them separately so "underflow" does not conflate
                # representable subnormals with a true flush to zero.
                subnormal = torch.count_nonzero(
                    (scaled_abs > 0) & (scaled_abs < quant_min)
                ).to(torch.float64) / numel
            else:
                overflow = underflow = subnormal = zero
            quantized = quant(out)
            if numel:
                quantized_zero = torch.count_nonzero(
                    (quantized == 0) & (out != 0)
                ).to(torch.float64) / numel
            else:
                quantized_zero = zero
            err = quantized.sub_(out).abs_().mean()
            max_abs_in = torch.maximum(x_in.max(), -x_in.min()) if has_in else zero
            kurtosis_in = _excess_kurtosis(x_in) if has_in else zero
            stat = torch.stack([max_abs_in.detach(), max_abs_out.detach(), mean_abs_out.detach(),
                                overflow.detach(), underflow.detach(), subnormal.detach(),
                                quantized_zero.detach(), err.detach(),
                                kurtosis_in.detach()]).double()
            pending_stats.append((cls_name, stat))
        return hook
    pending_stats = []
    first_bad = []
    first_over = []

    def make_block_hook(i):

        def hook(module, inp, out):
            y = out[0] if isinstance(out, tuple) else out
            all_finite = torch.isfinite(y).all()
            first_bad.append((i, all_finite))
            # Overflow is measured against the range of the quantizer that
            # actually ran. Dynamic per-tensor scaling first maps the block's
            # amax into the format range, so the raw max overstates overflow
            # (in fact it cannot overflow unless the input is non-finite).
            max_abs = torch.maximum(y.max(), -y.min())
            if quant_dynamic and y.numel():
                amax = y.detach().abs().amax().clamp_(min=1e-12)
                max_abs = (max_abs.double()
                           * (quant_max / amax.double())).clamp_(max=quant_max)
            first_over.append((i, max_abs))
        return hook
    with torch.no_grad():
        logits_fp32, _ = model(x)
    for name, m in model.named_modules():
        if m.__class__.__name__ in {'LayerNorm', 'RMSNorm', 'DyT', 'DyTC', 'DyTD', 'DyTDTanh', 'DyTCD', 'DyTCDTanh'}:
            handles.append(m.register_forward_hook(make_stat_hook()))
    block_handles = [blk.register_forward_hook(make_block_hook(i)) for i, blk in enumerate(model.blocks)]
    with fp8_forward(model, quant):
        with torch.no_grad():
            logits_fp8, _ = model(x)
    for h in block_handles:
        h.remove()
    for h in handles:
        h.remove()
    if pending_stats:
        stat_vals = torch.stack([s for _, s in pending_stats]).tolist()
        for (cls_name, _), vals in zip(pending_stats, stat_vals):
            (max_abs_in, max_abs_out, mean_abs_out, overflow, underflow,
             subnormal, quantized_zero, err, kurtosis_in) = vals
            denom = mean_abs_out + 1e-06
            ranges.append({'module': cls_name, 'max_abs_in': max_abs_in, 'max_abs_out': max_abs_out, 'kurtosis_in': kurtosis_in})
            quant_errors.append({'module': cls_name, 'max_abs': max_abs_out, 'overflow_rate': overflow, 'underflow_rate': underflow, 'subnormal_rate': subnormal, 'quantized_zero_rate': quantized_zero, 'rel_err': err / denom})
    bad_flags = torch.stack([f for _, f in first_bad]).tolist() if first_bad else []
    over_vals = torch.stack([v for _, v in first_over]).tolist() if first_over else []
    first_bad_idx = next((i for i, ok in zip([i for i, _ in first_bad], bad_flags) if not ok), None)
    first_over_idx = next((i for i, v in zip([i for i, _ in first_over], over_vals)
                           if v > quant_max), None)
    # Compare against the fp32 logits row-block by row-block.
    max_abs_fp32 = torch.maximum(logits_fp32.max(), -logits_fp32.min()).cpu()
    logits_fp32_cpu = logits_fp32.detach().to('cpu')
    del logits_fp32
    rows = logits_fp8.shape[0]
    row_bytes = max(1, int(logits_fp8[0].numel())) * 4
    block_rows = max(1, (1 << 24) // row_bytes)
    finite = True
    max_diff = None
    for b0 in range(0, rows, block_rows):
        b1 = min(rows, b0 + block_rows)
        blk = logits_fp8[b0:b1]
        if not bool(torch.isfinite(blk).all()):
            finite = False
            break
        diff = (blk.to('cpu') - logits_fp32_cpu[b0:b1]).abs().max()
        max_diff = diff if max_diff is None else torch.maximum(max_diff, diff)
    del logits_fp32_cpu
    if finite:
        # Division in fp32.
        rel = float(max_diff / max_abs_fp32.clamp(min=1e-12))
        logit_nan = False
    else:
        rel = float('inf')
        logit_nan = True
    if not quant_errors:
        return {'norm': norm_name, 'n_layers': int(getattr(model.cfg, 'n_layers', 0)), 'n_norm_hooks': 0, 'max_abs_pre_norm': 0.0, 'max_abs_post_norm': 0.0, 'mean_excess_kurtosis_pre_norm': 0.0, 'mean_overflow_rate_e4m3': 0.0, 'max_overflow_rate_e4m3': 0.0, 'mean_underflow_rate_e4m3': 0.0, 'mean_subnormal_rate_e4m3': 0.0, 'mean_quantized_zero_rate_e4m3': 0.0, 'mean_rel_quant_err_e4m3': 0.0, 'logit_rel_divergence': rel, 'logit_nan_under_fp8': logit_nan, 'first_nonfinite_layer': first_bad_idx, 'first_overflow_layer': first_over_idx}
    mean_overflow = sum((e['overflow_rate'] for e in quant_errors)) / len(quant_errors)
    max_overflow = max((e['overflow_rate'] for e in quant_errors))
    mean_underflow = sum((e['underflow_rate'] for e in quant_errors)) / len(quant_errors)
    mean_subnormal = sum((e['subnormal_rate'] for e in quant_errors)) / len(quant_errors)
    mean_quantized_zero = sum((e['quantized_zero_rate'] for e in quant_errors)) / len(quant_errors)
    mean_rel_err = sum((e['rel_err'] for e in quant_errors)) / len(quant_errors)
    max_max = max((e['max_abs'] for e in quant_errors))
    max_in = max((r['max_abs_in'] for r in ranges)) if ranges else 0.0
    mean_kurt = sum((r['kurtosis_in'] for r in ranges)) / len(ranges) if ranges else 0.0
    return {'norm': norm_name, 'n_layers': int(getattr(model.cfg, 'n_layers', 0)), 'n_norm_hooks': len(quant_errors), 'max_abs_pre_norm': max_in, 'max_abs_post_norm': max_max, 'mean_excess_kurtosis_pre_norm': mean_kurt, 'mean_overflow_rate_e4m3': mean_overflow, 'max_overflow_rate_e4m3': max_overflow, 'mean_underflow_rate_e4m3': mean_underflow, 'mean_subnormal_rate_e4m3': mean_subnormal, 'mean_quantized_zero_rate_e4m3': mean_quantized_zero, 'mean_rel_quant_err_e4m3': mean_rel_err, 'logit_rel_divergence': rel, 'logit_nan_under_fp8': logit_nan, 'first_nonfinite_layer': first_bad_idx, 'first_overflow_layer': first_over_idx}

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
        # Keep the numeric draws (including non-finite ones) for the census;
        # `mean_std_stderr` filters non-finite values itself. Filtering them
        # out here first made `_n_nonfinite` and `nonfinite_metric_counts`
        # always zero, silently hiding the diverged probes the aggregate is
        # supposed to expose.
        raw = [r.get(key) for r in per_seed
               if isinstance(r.get(key), (int, float))
               and not isinstance(r.get(key), bool)]
        stats = mean_std_stderr(raw)
        if stats is None:
            # Every probe draw was non-finite: report "unmeasured" instead of
            # leaving the first seed's NaN in the aggregate. A probe that
            # overflowed is a numerical failure to triage, not a result.
            agg[key] = None
            agg[key + '_std'] = None
            agg[key + '_n_finite'] = 0
            agg[key + '_n_nonfinite'] = len(raw)
            if raw:
                agg.setdefault('nonfinite_metric_counts', {})[key] = len(raw)
            continue
        agg[key] = stats['mean']
        agg[key + '_std'] = stats['std'] if stats['n'] >= 2 else None
        agg[key + '_n_finite'] = stats['n']
        agg[key + '_n_nonfinite'] = stats['n_nonfinite']
        if stats['n_nonfinite']:
            agg.setdefault('nonfinite_metric_counts', {})[key] = stats['n_nonfinite']
    nonfinite = [r['first_nonfinite_layer'] for r in per_seed
                 if isinstance(r.get('first_nonfinite_layer'), int)
                 and not isinstance(r.get('first_nonfinite_layer'), bool)]
    overflowing = [r['first_overflow_layer'] for r in per_seed
                   if isinstance(r.get('first_overflow_layer'), int)
                   and not isinstance(r.get('first_overflow_layer'), bool)]
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

def _finite_metric(value) -> bool:
    """Whether a probe metric may enter a paired test.

    A diverged probe can leave NaN/Inf in the per-seed record. The aggregate
    reports those draws as excluded, but the paired tests must exclude them
    too: a single NaN makes the t statistic NaN and aborts the whole
    comparison instead of testing the finite draws.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))

def _seed_id(value):
    """A seed key as an int, or None for a boolean/fractional/foreign key.

    JSON turns an integer mapping key into its string form, so ``"42"`` is the
    same seed as ``42``; ``"42.0"``, ``"042"`` and booleans are not.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if str(converted) == value:
            return converted
    return None

def _seed_metric_values(agg: dict, key: str) -> list:
    return [float(r[key]) for r in agg.get('per_seed', [])
            if _finite_metric(r.get(key))]

def _train_seed_metric_values(agg: dict, key: str) -> dict:
    """Per-training-seed mean of `key` (averaged over that seed's checkpoints)."""
    out = {}
    for train_seed, metrics in (agg.get('per_train_seed') or {}).items():
        seed = _seed_id(train_seed)
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if seed is not None and _finite_metric(value):
            out[seed] = float(value)
    return out

def _shared_seed_samples(agg_a: dict, agg_b: dict, key: str) -> tuple[list, list, list]:
    """Per-seed values of `key` for the seeds both norms measured, paired by seed.

    For a fixed seed the probe draw (and, in random-init mode, the model
    initialization seed) is identical for every norm, so the two norms'
    records are paired samples - the same reason the ablation runner pairs its
    rows by seed. A seed with a non-finite metric on either side is not a
    usable pair and is excluded rather than poisoning the paired t statistic.
    A malformed seed field is unknown and is skipped, never coerced.
    """
    a_by_seed = {_seed_id(r.get('seed')): r.get(key)
                 for r in agg_a.get('per_seed', [])
                 if _seed_id(r.get('seed')) is not None}
    b_by_seed = {_seed_id(r.get('seed')): r.get(key)
                 for r in agg_b.get('per_seed', [])
                 if _seed_id(r.get('seed')) is not None}
    shared = sorted(s for s in set(a_by_seed) & set(b_by_seed)
                    if _finite_metric(a_by_seed[s])
                    and _finite_metric(b_by_seed[s]))
    return shared, [float(a_by_seed[s]) for s in shared], [float(b_by_seed[s]) for s in shared]

_TREATMENT_KEYS = (
    'norm', 'qk_norm', 'qk_alpha_init', 'alpha_0', 'k_init', 'alpha_init',
    'p_init', 'learnable_p', 'learnable_k', 'learnable_delta', 'beta',
    'use_layerscale', 'layerscale_init', 'alpha_jitter',
)


def _treatment_record(ckpt_args, model) -> dict:
    """The normalization treatment recorded by one checkpoint.

    This is deliberately separate from `_protocol_record`: the treatment is
    the one thing that *should* differ between comparison arms, but every
    checkpoint inside one arm must share it. Checkpoint groups are accepted as
    comma-separated lists, so without this check an ``ln`` checkpoint could be
    averaged with two ``rmsnorm`` checkpoints and the mixed group certified as
    a single norm.
    """
    args = ckpt_args if isinstance(ckpt_args, dict) else {}
    cfg = getattr(model, 'cfg', None)

    def recorded(name, cfg_name=None, default=None):
        if name in args:
            return args[name]
        return getattr(cfg, cfg_name or name, default)

    def negated(name, cfg_name, default):
        # Training checkpoints store `--no-learnable-p` / `--no-layerscale`
        # flags; the model config stores the positive attribute.
        if name in args:
            return not bool(args[name])
        return getattr(cfg, cfg_name, default)

    return {
        'norm': recorded('norm', 'norm_name'),
        'qk_norm': recorded('qk_norm'),
        'qk_alpha_init': recorded('qk_alpha_init'),
        'alpha_0': recorded('alpha_0'),
        'k_init': recorded('k_init'),
        'alpha_init': recorded('alpha_init'),
        'p_init': recorded('p_init'),
        'learnable_p': negated('no_learnable_p', 'learnable_p', True),
        'learnable_k': negated('no_learnable_k', 'learnable_k', True),
        'learnable_delta': negated('no_learnable_delta', 'learnable_delta', True),
        'beta': recorded('beta', default=True),
        'use_layerscale': negated('no_layerscale', 'use_layerscale', True),
        'layerscale_init': recorded('layerscale_init'),
        'alpha_jitter': recorded('alpha_jitter'),
    }


def _treatment_mismatches(records) -> dict:
    """Per-key differing treatment values inside one checkpoint group."""
    values = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in _TREATMENT_KEYS:
            values.setdefault(key, set()).add(repr(record.get(key)))
    return {key: sorted(seen) for key, seen in values.items() if len(seen) > 1}


def _protocol_record(ckpt_args: dict, model) -> dict:
    """Training recipe and shape recorded by one checkpoint."""
    return {
        # Keep absent keys absent: `training_recipe_mismatches` treats a
        # missing key as unknown, but materializing it as None made two
        # unrecorded recipes compare equal.
        'hyperparameters': {key: ckpt_args[key]
                            for key in TRAINING_RECIPE_KEYS
                            if key in ckpt_args},
        # The requested dataset/tokenizer revision may be None/branch while
        # the loader actually resolved a concrete commit. Each checkpoint's
        # probe is built from its own resolved revision, so the protocol guard
        # must compare these too: otherwise probes from different corpora are
        # paired as if they were identical.
        'resolved_revisions': {
            'dataset_revision_resolved': ckpt_args.get('dataset_revision_resolved'),
            'tokenizer_revision_resolved': ckpt_args.get('tokenizer_revision_resolved'),
        },
        'shape': {'model': ckpt_args.get('model'),
                  'n_layers': int(getattr(model.cfg, 'n_layers', 0)),
                  'dim': int(getattr(model.cfg, 'dim', 0)),
                  'n_heads': int(getattr(model.cfg, 'n_heads', 0))},
        'checkpoint': {'step': ckpt_args.get('_checkpoint_step')},
    }

def _protocol_pair_mismatches(a: dict, b: dict) -> dict:
    """Differences in the training recipe or model shape of two checkpoints."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {'protocol': ['<missing>', '<missing>']}
    out = training_recipe_mismatches(a.get('hyperparameters'), b.get('hyperparameters'))
    revisions_a = a.get('resolved_revisions') or {}
    revisions_b = b.get('resolved_revisions') or {}
    for key in ('dataset_revision_resolved', 'tokenizer_revision_resolved'):
        value_a = revisions_a.get(key, a.get(key))
        value_b = revisions_b.get(key, b.get(key))
        # Equal is not provenance: two runs on the mutable default branch can
        # both record None when the Hub was unreachable. Only two identical
        # concrete commits prove the checkpoints share one data/tokenizer
        # snapshot; unknown/mismatched revisions cannot certify a norm delta.
        if (not _is_pinned_revision(value_a)
                or not _is_pinned_revision(value_b)
                or value_a != value_b):
            out[f'data.{key}'] = [value_a, value_b]
    shape_a = a.get('shape') or {}
    shape_b = b.get('shape') or {}
    for key in ('model', 'n_layers', 'dim', 'n_heads'):
        value_a = shape_a.get(key)
        value_b = shape_b.get(key)
        if key == 'model':
            valid_a = isinstance(value_a, str) and bool(value_a.strip())
            valid_b = isinstance(value_b, str) and bool(value_b.strip())
        else:
            valid_a = (isinstance(value_a, int) and not isinstance(value_a, bool)
                       and value_a > 0)
            valid_b = (isinstance(value_b, int) and not isinstance(value_b, bool)
                       and value_b > 0)
        # Unknown is not equal: two arms that both omit a shape field have
        # not proven that they share an architecture.
        if not valid_a or not valid_b or value_a != value_b:
            out[f'shape.{key}'] = [value_a, value_b]
    step_a = (a.get('checkpoint') or {}).get('step')
    step_b = (b.get('checkpoint') or {}).get('step')
    valid_step_a = (isinstance(step_a, int) and not isinstance(step_a, bool)
                    and step_a >= 0)
    valid_step_b = (isinstance(step_b, int) and not isinstance(step_b, bool)
                    and step_b >= 0)
    if not valid_step_a or not valid_step_b or step_a != step_b:
        out['checkpoint.step'] = [step_a, step_b]
    return out

def _result_scale(result: dict, index: int) -> str:
    """A stable architecture label for one probe result.

    Checkpoint probes carry the full protocol shape; random-init probes carry
    the builder's model name and shape. The label is what the cross-scale
    summary groups on, so it must distinguish widths/depths without depending
    on the norm being probed. Results without either (e.g. hand-built test
    fixtures) share the single `unknown` scale rather than inventing one
    scale per row.
    """
    protocol = result.get('protocol') if isinstance(result, dict) else None
    shape = protocol.get('shape') if isinstance(protocol, dict) else None
    shape = shape or (result.get('shape') if isinstance(result, dict) else None)
    if isinstance(shape, dict) and shape:
        return json.dumps(shape, sort_keys=True)
    model = result.get('model') if isinstance(result, dict) else None
    n_layers = result.get('n_layers') if isinstance(result, dict) else None
    if model is not None or n_layers is not None:
        return f"{model or 'model'}:L{n_layers}"
    return 'unknown'

def compare_fp8_norms(results: list, metrics: tuple=FP8_COMPARISON_METRICS, alpha: float=0.05,
                      trained: bool=False) -> dict:
    """Paired, Holm-adjusted significance of one norm's FP8 metrics vs another's.

    The test family is the one-sided paired Student-t `H1: mean(A - B) < 0`,
    so a negative `mean_delta` means A is more FP8-stable on that metric; pairs
    with fewer than 2 shared samples are reported without a p-value rather
    than silently tested unpaired. p-values are Holm-adjusted across the whole
    metric x pair family, so scanning six norms does not hand a "significant"
    win to the luckiest pair. `main` writes this next to the results as
    `comparison.json`, mirroring the ablation runner's
    `leave_one_out_comparison.json`.

    What counts as a sample depends on how the check was run:

    * `trained=True` (checkpoints were probed): the run-to-run unit is the
      *training* seed, and a test is certifiable (`basis='train_seed_means'`)
      only when both norms provide >= 3 checkpoints with shared training
      seeds, i.e. when the Delta is observed across independent training runs;
      two shared seeds are described but not certified (df = 1).
      With a single trained checkpoint per norm the paired test degenerates to
      probe-noise pairing on one model each: it is still computed, but labeled
      `basis='probe_paired_single_train_run'`, `certifiable=False`, and
      `significant` stays False so the report cannot certify a cross-norm
      winner from repeated probes of one run per arm.
    * `trained=False` (fresh random init): each seed re-initializes the model
      as well as the probe, so `basis='independent_random_inits'` pairs
      independent (init, probe) draws. It is certifiable only with >= 3
      shared draws; two (df = 1) are descriptive. The standing caveat is that
      a random init says nothing about trained weights.
    """
    labels = [str(r.get('label') or r.get('norm') or r.get('ckpt') or f'run{i}') for i, r in enumerate(results)]
    # The cross-scale contrast identity is the norm pair, not the display
    # label: in multi-model mode the label carries an `@model` suffix, which
    # made the "same" test at two scales look like two different contrasts and
    # silently disabled the replication report.
    norm_ids = [str(r.get('norm') or str(labels[i]).split('@', 1)[0])
                for i, r in enumerate(results)]
    tests = []
    for metric in metrics:
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                a_train = _train_seed_metric_values(results[i], metric)
                b_train = _train_seed_metric_values(results[j], metric)
                shared_train = sorted(set(a_train) & set(b_train))
                shared, av, bv = _shared_seed_samples(results[i], results[j], metric)
                a_scale = _result_scale(results[i], i)
                b_scale = _result_scale(results[j], j)
                t = {'a': labels[i], 'b': labels[j], 'metric': metric,
                     'a_norm': norm_ids[i], 'b_norm': norm_ids[j],
                     'a_scale': a_scale, 'b_scale': b_scale,
                     'same_scale': a_scale == b_scale,
                     'direction': f'negative mean_delta means A has lower {metric} (more FP8-stable)',
                     'a_train_seeds': sorted(a_train), 'b_train_seeds': sorted(b_train)}
                if a_scale != b_scale:
                    # A cross-scale pair confounds the norm with width/depth;
                    # record the number but never certify it as a norm effect.
                    t.update({'basis': 'cross_scale', 'certifiable': False,
                              'significant': False, 'p_one_sided': None,
                              'mean_delta': ((sum(av) - sum(bv)) / len(shared)
                                             if shared else None),
                              'n_shared_seeds': len(shared),
                              'shared_seeds': shared,
                              'reason': ('the two arms are different model shapes; a '
                                         'cross-scale delta is not a same-scale norm effect '
                                         '(run both norms at each scale and use the '
                                         'scale_replication block)')})
                    tests.append(t)
                    continue
                if trained:
                    arm_problems = [f'{labels[idx]} mixes training recipes across its checkpoints'
                                    for idx, arm in ((i, results[i]), (j, results[j]))
                                    if arm.get('protocol_consistent') is False]
                    if arm_problems:
                        t.update({'basis': 'protocol_mismatch', 'certifiable': False,
                                  'mean_delta': None, 'p_one_sided': None,
                                  'n_shared_train_seeds': len(shared_train),
                                  'shared_train_seeds': shared_train,
                                  'reason': '; '.join(arm_problems)})
                        tests.append(t)
                        continue
                    protocol_mismatches = _protocol_pair_mismatches(
                        results[i].get('protocol'), results[j].get('protocol'))
                    if protocol_mismatches:
                        t.update({'basis': 'protocol_mismatch', 'certifiable': False,
                                  'mean_delta': None, 'p_one_sided': None,
                                  'n_shared_train_seeds': len(shared_train),
                                  'shared_train_seeds': shared_train,
                                  'protocol_mismatches': protocol_mismatches,
                                  'reason': ('training recipe/model shape differs between norms, '
                                             'so the delta is not attributable to the norm: '
                                             + json.dumps(protocol_mismatches, sort_keys=True))})
                        tests.append(t)
                        continue
                if trained and len(shared_train) >= 2:
                    t['basis'] = 'train_seed_means'
                    t['certifiable'] = len(shared_train) >= MIN_SEEDS_FOR_SIGNIFICANCE
                    t['n_shared_train_seeds'] = len(shared_train)
                    t['shared_train_seeds'] = shared_train
                    t['n_shared_seeds'] = len(shared)
                    t['shared_seeds'] = shared
                    t.update(paired_t_one_sided([a_train[s] for s in shared_train],
                                                [b_train[s] for s in shared_train], alpha=alpha))
                else:
                    t['basis'] = ('probe_paired_single_train_run' if trained
                                  else 'independent_random_inits')
                    # In random-init mode each seed is an independent (init,
                    # probe) draw, so it is a legitimate run-to-run unit - but
                    # only once there are at least three of them. Two draws
                    # give df=1 and must stay descriptive, exactly like two
                    # shared training seeds.
                    t['certifiable'] = (not trained) and len(shared) >= MIN_SEEDS_FOR_SIGNIFICANCE
                    t['n_shared_train_seeds'] = len(shared_train)
                    t['shared_train_seeds'] = shared_train
                    t['n_shared_seeds'] = len(shared)
                    t['shared_seeds'] = shared
                    if len(shared) < 2:
                        t['mean_delta'] = (sum(av) - sum(bv)) / len(shared) if shared else None
                        t['p_one_sided'] = None
                        t['reason'] = 'need >= 2 shared samples to pair a test'
                    else:
                        t.update(paired_t_one_sided(av, bv, alpha=alpha))
                    if not trained and len(shared) < MIN_SEEDS_FOR_SIGNIFICANCE:
                        t['reason'] = '; '.join(filter(None, (
                            t.get('reason'),
                            f'only {len(shared)} shared probe draw(s): fewer than the required '
                            f'{MIN_SEEDS_FOR_SIGNIFICANCE} independent (init, probe) draws, so '
                            'the test is descriptive only')))
                    if trained:
                        # Append, never replace: with too few shared probe
                        # samples *and* one training run per arm, both reasons
                        # apply to this pair.
                        t['reason'] = '; '.join(filter(None, (
                            t.get('reason'),
                            f'only {len(shared_train)} shared training seed(s): fewer than the '
                            'required 3 cannot certify the norm against run-to-run noise, so the '
                            'test is descriptive only; provide >= 3 checkpoints (comma-separated in '
                            '--ckpt) with distinct training seeds per norm to certify')))
                tests.append(t)
    for t, p_adj in zip(tests, holm_adjust([t.get('p_one_sided') for t in tests])):
        t['p_holm'] = p_adj
        t['significant'] = bool(t.get('certifiable') and p_adj is not None and p_adj < alpha)
    scale_obs = []
    for t in tests:
        if t.get('same_scale') and t.get('mean_delta') is not None:
            # A Holm p-value can be explicitly None for a row excluded from
            # its family; `dict.get(key, fallback)` only falls back on a
            # missing key, so choose the raw p explicitly.
            p_holm = t.get('p_holm')
            scale_obs.append({'scale': t.get('a_scale'), 'metric': t.get('metric'),
                              'contrast': f"{t.get('a_norm')} vs {t.get('b_norm')}",
                              'delta': t.get('mean_delta'),
                              'certified': t.get('significant') is True,
                              'p_one_sided': (p_holm if p_holm is not None
                                              else t.get('p_one_sided')),
                              'n_seeds': (t.get('n_shared_train_seeds')
                                          if t.get('basis') == 'train_seed_means'
                                          else t.get('n_shared_seeds')),
                              # Width of the same paired test, forwarded so
                              # the multi-scale report can pool effects by
                              # inverse variance instead of voting on signs.
                              'ci_low': t.get('ci_low'),
                              'ci_high': t.get('ci_high'),
                              'df': t.get('df'),
                              'ci_level': t.get('ci_level')})
    scale_replication = scale_replication_summary(scale_obs, alpha=alpha)
    ranking = {}
    for metric in metrics:
        rows = []
        for label, r in zip(labels, results):
            vals = _seed_metric_values(r, metric)
            stats = mean_std_stderr(vals)
            train_vals = list(_train_seed_metric_values(r, metric).values())
            train_stats = mean_std_stderr(train_vals)
            rows.append({'norm': label, 'n_seeds': len(vals),
                         'n_train_seeds': len(train_vals),
                         'mean': stats['mean'] if stats else None,
                         'std': stats['std'] if stats and stats['n'] >= 2 else None,
                         'mean_across_train_seeds': train_stats['mean'] if train_stats else None,
                         'std_across_train_seeds': (train_stats['std']
                                                    if train_stats and train_stats['n'] >= 2 else None)})
        rows.sort(key=lambda row: (row['mean'] is None, row['mean']))
        ranking[metric] = rows
    return {'metrics': list(metrics), 'alpha': alpha, 'trained': trained,
            'pairing': ('probe draws are paired across norms by probe seed and (when available) by training seed; '
                        'certification requires >= 3 shared training seeds per arm and otherwise stays descriptive'),
            'scale_replication': scale_replication,
            'ranking': ranking, 'tests': tests}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--norms', nargs='+', default=['ln', 'rmsnorm', 'dyt', 'dytc', 'dytd', 'dytcd'])
    p.add_argument('--models', nargs='+', default=['gpt2-124m'],
                   choices=sorted(_FP8_MODEL_BUILDERS),
                   help='random-init architectures to probe (one run per model; each norm is '
                        'measured at every model and comparison.json records whether a '
                        'cross-norm direction replicated across scales). Not used with --ckpt, '
                        'where the checkpoint defines the scale.')
    p.add_argument('--probe', default='markov', choices=['markov', 'random'], help="probe corpus. 'random' is uniform token ids: it has no next-token structure, so the activation magnitudes below describe embedding noise rather than a trained model's activations (kept only for parity)")
    p.add_argument('--scaling', default='none', choices=['none', 'dynamic-tensor'],
                   help="quantization scaling. 'none' (default) is the raw-value worst case: "
                        "values beyond the format range clip, a stress test of range pressure, "
                        "NOT a prediction of real FP8 training. 'dynamic-tensor' rescales each "
                        "tensor to the full range first, like real FP8 recipes; compare against "
                        "that run before drawing a training-stability conclusion.")
    p.add_argument('--ckpt', nargs='+', default=None, help='one entry per --norms entry; a plain checkpoint path probes the trained weights instead of a fresh random init. An entry may be a comma-separated list of checkpoints (ideally distinct training seeds of the same arm), which are all probed and averaged per probe seed; cross-norm certification then pairs training seeds, so >= 3 shared training seeds across the compared norms are required for comparison.json to certify a winner')
    p.add_argument('--n-states', type=int, default=64, help="state count of the synthetic 'markov' probe")
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS), help='each norm is probed once per seed (fresh probe, and fresh init in random-init mode) and reported as Mean +/- Std across seeds; >= 3 recommended so one probe draw cannot decide a comparison')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the paired cross-norm tests in comparison.json (Holm-adjusted over metrics x pairs)')
    p.add_argument('--output', default='runs/fp8_check/results.json')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this check tests (e.g. "DyT-CD limits fp8 activation blowup versus RMSNorm at matched init"); recorded in the run manifest so the result can be audited against the claim it was meant to support')
    args = p.parse_args()
    if not args.norms:
        p.error('--norms must not be empty')
    if len(set(args.norms)) != len(args.norms):
        p.error(f'--norms contains duplicates: {args.norms}; a repeated norm is the same '
                'row and would add duplicate tests to the Holm family')
    if args.ckpt and len(args.ckpt) != len(args.norms):
        p.error(f'--ckpt expects {len(args.norms)} paths (one per --norms), got {len(args.ckpt)}')
    if args.ckpt and len(args.models) > 1:
        p.error('--models is for random-init probing; with --ckpt the checkpoint defines the '
                'model shape, so pass at most the default --models value')
    for flag, value in (('--batch-size', args.batch_size), ('--seq-len', args.seq_len)):
        if value < 1:
            p.error(f'{flag} {value} must be >= 1')
    if args.n_states < 2:
        p.error(f'--n-states {args.n_states} must be >= 2: a one-state probe has a '
                'deterministic successor and measures no next-token structure')
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        p.error(f'--alpha {args.alpha} must be finite and strictly between 0 and 1')
    raw_seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if not raw_seeds:
        p.error('--seeds must not be empty')
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[fp8][warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed reruns the identical probe, not an independent sample)')
    if len(seeds) < 3:
        print(f'[fp8][warn] {len(seeds)} seed(s): one probe draw is one sample; use >= 3 independent probes before treating a metric difference as real')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seeds[0])
    fp8_quant = make_fp8_quant('e4m3', args.scaling)
    print(f"[fp8] scope: FORWARD-ONLY simulation (no FP8 gradients or optimizer states) "
          f"with scaling={args.scaling!r}; scaling='none' is a range stress test, not a "
          "real FP8-training prediction", file=sys.stderr)

    def make_probe(rng, cfg):
        if args.probe == 'random':
            return torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), generator=rng, device=device)
        if args.n_states > int(cfg.vocab_size):
            raise SystemExit(
                f'[fp8] --n-states {args.n_states} exceeds the model vocabulary '
                f'({int(cfg.vocab_size)}); markov_batch would emit out-of-range token ids')
        x, _ = markov_batch(rng, cfg.ignore_index, args.batch_size, args.seq_len, device, n_states=args.n_states)
        return x
    ckpt_groups = None
    if args.ckpt:
        ckpt_groups = [[path for path in str(entry).split(',') if path] for entry in args.ckpt]
        if any(not group for group in ckpt_groups):
            p.error('--ckpt entries must not be empty')
        for group in ckpt_groups:
            if len(set(group)) != len(group):
                p.error(f'--ckpt arm {group} repeats a checkpoint; a repeated path is the '
                        'same trained run and would be weighted twice instead of adding evidence')
        normalized_groups = [tuple(sorted(group)) for group in ckpt_groups]
        if len(set(normalized_groups)) != len(normalized_groups):
            p.error('--ckpt contains the same checkpoint set for two norms; the same weights '
                    'cannot support a cross-norm difference')
        # A partially shared checkpoint set is also a self-comparison on the
        # shared runs: their paired deltas are exactly zero, which shrinks the
        # variance estimate and contaminates the cross-norm contrast. Compare
        # resolved paths so ./a and a count as the same run.
        seen_paths: dict[str, int] = {}
        for group_index, group in enumerate(ckpt_groups):
            for path in group:
                key = str(Path(path).resolve())
                previous = seen_paths.get(key)
                if previous is not None:
                    where = ('within one norm' if previous == group_index
                             else f'in norms {previous + 1} and {group_index + 1}')
                    p.error(f'--ckpt checkpoint {path} appears more than once ({where}); '
                            'each training run may contribute to exactly one norm')
                seen_paths[key] = group_index
    results = []
    # Random-init mode can sweep several architectures in one invocation;
    # checkpoint mode keeps one arm list (the checkpoint defines the scale).
    model_names = [None] if args.ckpt else list(args.models)
    multi_model = len(model_names) > 1
    for arm_index, (model_name, norm) in enumerate(
            (model_name, norm) for model_name in model_names for norm in args.norms):
        i = args.norms.index(norm)
        # A multi-model arm gets a unique label so the comparison cannot
        # silently pair the same norm name across two architectures.
        label = f'{norm}@{model_name}' if multi_model else norm
        arm_model = model_name
        arm_cfg = None
        if not args.ckpt:
            print(f'[fp8] WARNING: {label} probed at a fresh random init. layerscale_init=1e-5 leaves the residual stream at the embedding table, so every norm reports the same pre-norm statistics and this row says nothing about the layer. Pass --ckpt <path> (one per --norms entry) to measure trained weights.', file=sys.stderr)
        else:
            from scripts.eval_downstream import load_ckpt
        per_ckpt = []
        for ckpt_path in (ckpt_groups[i] if ckpt_groups else [None]):
            train_seed = None
            ckpt_model = None
            if ckpt_path is not None:
                ckpt_model, ckpt_args = load_ckpt(ckpt_path, device=device)
                raw_seed = (ckpt_args or {}).get('seed')
                train_seed = _seed_id(raw_seed)
                arm_model = (ckpt_args or {}).get('model')
                arm_cfg = getattr(ckpt_model, 'cfg', None)
            per_seed_ckpt = []
            for seed in seeds:
                set_all_seeds(seed)
                rng = torch.Generator(device=device).manual_seed(seed)
                if ckpt_model is not None:
                    model = ckpt_model
                else:
                    model = _FP8_MODEL_BUILDERS[model_name](
                        norm_name=norm, init=False).to(device)
                    model.reset_parameters(seed)
                    arm_cfg = model.cfg
                model.eval()
                x = make_probe(rng, model.cfg)
                r = measure_fp8_divergence(model, x, norm, quant=fp8_quant)
                r['scaling'] = args.scaling
                r['forward_only'] = True
                r['seed'] = seed
                per_seed_ckpt.append(r)
            per_ckpt.append({'ckpt': ckpt_path, 'train_seed': train_seed,
                             'protocol': _protocol_record(ckpt_args, ckpt_model)
                             if ckpt_model is not None else None,
                             'treatment': _treatment_record(ckpt_args, ckpt_model)
                             if ckpt_model is not None else None,
                             'per_seed': per_seed_ckpt})
        # A comma-separated group is one arm only when every checkpoint in it
        # is the same treatment; otherwise its per-seed means mix norms before
        # the cross-norm comparison ever starts.
        treatment_mismatches = _treatment_mismatches(
            c.get('treatment') for c in per_ckpt)
        if treatment_mismatches:
            p.error(
                f'--ckpt group {label!r} mixes treatment values '
                f'{json.dumps(treatment_mismatches, sort_keys=True)}; one '
                'checkpoint group must contain a single normalization treatment')
        # A multi-checkpoint norm is one treatment only if its checkpoints
        # share the training recipe and model shape; keep any disagreement
        # visible instead of averaging it into the arm.
        base_protocol = next((c.get('protocol') for c in per_ckpt
                              if c.get('protocol') is not None), None)
        arm_protocol_mismatches = []
        for c in per_ckpt:
            if c.get('protocol') is None or c.get('protocol') is base_protocol:
                # Comparing the base checkpoint with itself is not a
                # disagreement: a single arm with an old/incomplete protocol
                # must not be marked inconsistent (the same reason the
                # w8a8 path loops over the remaining checkpoints).
                continue
            mismatches = _protocol_pair_mismatches(base_protocol, c.get('protocol'))
            if mismatches:
                arm_protocol_mismatches.append({'ckpt': c.get('ckpt'),
                                                'mismatches': mismatches})
        # Norm-level probe records: each probe seed keeps one row whose metrics
        # are the mean over this norm's checkpoints (probe seed is the shared
        # draw; checkpoints are replicates of the arm).
        per_seed = []
        for j, seed in enumerate(seeds):
            record = {k: v for k, v in per_ckpt[0]['per_seed'][j].items() if not k.startswith('__')}
            for key in _FP8_SEED_METRIC_KEYS:
                vals = [c['per_seed'][j][key] for c in per_ckpt
                        if _finite_metric(c['per_seed'][j].get(key))]
                if vals:
                    record[key] = sum(vals) / len(vals)
            flags = [c['per_seed'][j] for c in per_ckpt]
            nonfinite = [f['first_nonfinite_layer'] for f in flags
                         if isinstance(f.get('first_nonfinite_layer'), int)
                         and not isinstance(f.get('first_nonfinite_layer'), bool)]
            overflowing = [f['first_overflow_layer'] for f in flags
                           if isinstance(f.get('first_overflow_layer'), int)
                           and not isinstance(f.get('first_overflow_layer'), bool)]
            record['first_nonfinite_layer'] = min(nonfinite) if nonfinite else None
            record['first_overflow_layer'] = min(overflowing) if overflowing else None
            record['logit_nan_under_fp8'] = any(bool(f.get('logit_nan_under_fp8')) for f in flags)
            record['seed'] = seed
            record['n_checkpoints'] = len(per_ckpt)
            per_seed.append(record)
        r = aggregate_fp8_seed_rows(norm, per_seed)
        # The probe-seed row uses the mean over finite checkpoint values, so a
        # mixed group can hide its diverged checkpoint from the arm census.
        # Re-attach the count of non-finite checkpoint draws per metric.
        arm_nonfinite = {}
        for c in per_ckpt:
            for rec in c.get('per_seed') or []:
                for key in _FP8_SEED_METRIC_KEYS:
                    value = rec.get(key)
                    if (isinstance(value, (int, float))
                            and not isinstance(value, bool)
                            and not math.isfinite(float(value))):
                        arm_nonfinite[key] = arm_nonfinite.get(key, 0) + 1
        if arm_nonfinite:
            counts = dict(r.get('nonfinite_metric_counts') or {})
            for key, count in arm_nonfinite.items():
                r[key + '_n_nonfinite'] = max(
                    int(r.get(key + '_n_nonfinite') or 0), count)
                counts[key] = max(int(counts.get(key) or 0), count)
            r['nonfinite_metric_counts'] = counts
        train_seeds = sorted({c['train_seed'] for c in per_ckpt if c['train_seed'] is not None})
        r['train_seeds'] = train_seeds
        r['n_train_seeds'] = len(train_seeds)
        r['label'] = label
        r['model'] = arm_model
        if arm_cfg is not None:
            r['shape'] = {'n_layers': int(arm_cfg.n_layers),
                          'dim': int(arm_cfg.dim),
                          'n_heads': int(arm_cfg.n_heads)}
        if args.ckpt:
            r['protocol'] = base_protocol
            r['protocol_consistent'] = not arm_protocol_mismatches
            r['protocol_mismatches'] = arm_protocol_mismatches
        per_train_seed = {}
        for c in per_ckpt:
            if c['train_seed'] is None:
                continue
            bucket = per_train_seed.setdefault(c['train_seed'], {key: [] for key in _FP8_SEED_METRIC_KEYS})
            for rec in c['per_seed']:
                for key in _FP8_SEED_METRIC_KEYS:
                    value = rec.get(key)
                    if _finite_metric(value):
                        bucket[key].append(float(value))
        r['per_train_seed'] = {ts: {key: sum(vals) / len(vals) for key, vals in bucket.items() if vals}
                               for ts, bucket in per_train_seed.items()}
        if arm_protocol_mismatches:
            print(f'[fp8][warn] {norm}: checkpoints disagree on the training recipe/model '
                  f'shape; cross-norm tests for this arm are descriptive only: '
                  f'{json.dumps(arm_protocol_mismatches, sort_keys=True)}', file=sys.stderr)
        if ckpt_groups:
            r['ckpt'] = list(ckpt_groups[i])
            r['ckpts'] = list(ckpt_groups[i])
            if ckpt_groups[i] != [None] and len(train_seeds) < MIN_SEEDS_FOR_SIGNIFICANCE:
                print(f'[fp8][warn] {norm}: {len(ckpt_groups[i])} checkpoint(s) but {len(train_seeds)} distinct '
                      'training seed(s); cross-norm certification needs >= 3 independent training runs per arm '
                      '(pass comma-separated checkpoints with distinct training seeds)', file=sys.stderr)
        print(f'[fp8] {norm}: {r}')
        results.append(r)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    comparison = compare_fp8_norms(results, alpha=args.alpha, trained=bool(args.ckpt))
    comparison_path = out_path.with_name('comparison.json')
    comparison_path.write_text(json.dumps(comparison, indent=2))
    scale = comparison.get('scale_replication') or {}
    if scale.get('single_scale_only'):
        print(f"[fp8][scale] WARNING: this comparison covers {scale.get('n_scales', 0)} "
              f"model scale(s) ({scale.get('scales')}); a direction that is certified at one "
              'scale is a scope-limited result, not a multi-scale mechanism claim. Rerun with '
              '--models at another width/depth (or add checkpoints from another scale) and '
              'combine the per-scale comparison files.')
    elif scale.get('multi_scale_certified_metrics'):
        print(f"[fp8][scale] replicated across scales: "
              f"{scale.get('multi_scale_certified_metrics')} over {scale.get('scales')}")
    else:
        print(f"[fp8][scale] scales={scale.get('scales')} "
              f"replicated_metrics={scale.get('replicated_metrics')} "
              f"(no metric certified at >= 2 scales yet)")
    for t in comparison['tests']:
        if t.get('p_one_sided') is None:
            print(f"[fp8][compare] {t['a']} vs {t['b']} on {t['metric']}: {t['reason']}")
            continue
        verdict = 'A more stable' if t['significant'] and t['mean_delta'] < 0 else ('B more stable' if t['significant'] else 'inside the noise')
        if not t.get('certifiable'):
            if t.get('basis') == 'independent_random_inits':
                verdict = (f'descriptive only ({t.get("n_shared_seeds", 0)} shared '
                           'probe/init draw(s); >= 3 required)')
            else:
                verdict = (f'descriptive only ({t.get("n_shared_train_seeds", 0)} shared '
                           'training seed(s); >= 3 required)')
        print(f"[fp8][compare] {t['a']} vs {t['b']} on {t['metric']}: delta={t['mean_delta']:+.4g}, 95% CI [{t['ci_low']:+.4g}, {t['ci_high']:+.4g}], Holm p={t['p_holm']:.4f} (n={t['n_shared_seeds']}, basis={t['basis']}) -> {verdict}")
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
    manifest = finish_metadata(manifest, artifacts={
        'results': file_fingerprint(out_path),
        'comparison': file_fingerprint(comparison_path),
    })
    save_run_manifest(out_path.with_name('run_manifest.json'), manifest)
    print(f'[done] saved to {out_path}')
if __name__ == '__main__':
    main()
