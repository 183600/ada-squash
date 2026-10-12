"""
FP8 stability check: simulate FP8 / bf16 training divergence on the
same GPT-2 124M model with different norm layers.

We measure:
  - forward divergence: max|logits_fp8 - logits_fp32| / max|logits_fp32|,
    where the FP8 pass quantize-dequantizes the input and output of every
    GEMM (the SwiGLU MLP included)
  - activation overflow rate under FP8-E4M3
  - per-norm activation range and pre-norm excess kurtosis
  - the "FP8 trainability" curve: the first block whose output goes
    non-finite

The probe corpus is the same decision `check_w8a8_ppl.py` makes: the
headline metrics here are activation *magnitudes* (overflow rate,
underflow rate, max|x|, kurtosis), and uniform random token ids drive the
network with embedding noise rather than with anything a trained model
actually sees. `--probe markov` (default) walks a Markov chain, so the
activation statistics are measured on a corpus with real next-token
structure; `--probe random` reproduces the old behaviour.

This script is a CPU/GPU pure-python simulation - it does NOT need real
FP8 hardware.

Usage:
    python scripts/check_fp8_stability.py --norms ln rmsnorm dyt dytc dytd dytcd
"""

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

from adasquash.model import MLP, build_gpt_124m
from adasquash.utils import markov_batch

# Largest finite magnitude of the FP8-E4M3 format used throughout this file.
FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# FP8 simulation
# ---------------------------------------------------------------------------
def fp8_min_normal(exp_bits: int) -> float:
    """Smallest *normal* (non-subnormal) magnitude representable in an FP
    format with `exp_bits` exponent bits.

    The exponent bias is ``2**(exp_bits-1) - 1`` and normal exponents start
    at 1, so the smallest normal magnitude is ``2**(1 - bias)``:

    E4M3 (exp_bits=4) -> 2**-6  = 0.015625
    E5M2 (exp_bits=5) -> 2**-14 = 6.103515625e-05
    """
    return 2.0 ** (1 - (2 ** (exp_bits - 1) - 1))


def to_fp8_e4m3(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """Simulate per-tensor FP8-E4M3 quantization. Range: +-448."""
    return simulate_fp8(x, max_val=448.0, exp_bits=4, mantissa_bits=3)

def to_fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-tensor FP8-E5M2 quantization. Range: +-57344."""
    return simulate_fp8(x, max_val=57344.0, exp_bits=5, mantissa_bits=2)

def simulate_fp8(x: torch.Tensor, max_val: float, exp_bits: int, mantissa_bits: int) -> torch.Tensor:
    """
    Emulate FP8 by clamping + rounding to the nearest representable value.
    This is a coarse approximation but reproduces the *overflow / underflow*
    behaviour that's the actual subject of the stability check.

    Algebraically it is `sign(x) * round(|x| * L / 2**exp) * 2**exp / L`
    with `L = 2**mantissa_bits` and `exp = floor(log2(|x|))`. `frexp`
    supplies `|x| == m * 2**e` with `m` in [0.5, 1), so `2*|m|` is exactly
    `|x| / 2**exp` and the whole evaluation reduces to integer-scaled
    rounding plus one `ldexp` - which also folds the trailing `/ L` into the
    exponent, because `L` is a power of two. The signed mantissa that `frexp`
    returns carries the sign through for free.

    Flush-to-zero is decided on the same exponent: a value is sub-normal
    exactly when `floor(log2(|x|)) < 2 - 2**(exp_bits-1)`.
    """
    # Keep the caller's dtype: rebinding `x` to the fp32 copy would make the
    # return type always fp32, so under a bf16 autocast (the regime FP8
    # training actually runs in) every simulated tensor would be silently
    # promoted back to fp32 and the accumulation precision would not be what
    # the caller believes it is.
    in_dtype = x.dtype
    x = x.float()
    levels = float(2 ** mantissa_bits)
    a = x.clamp(-max_val, max_val)
    # Smallest normal magnitude, expressed directly as an exponent threshold
    # on the shifted `exp` below (the shift is `1 + mantissa_bits`).
    min_exp = (2 - (1 << (exp_bits - 1))) - mantissa_bits
    mant, exp = torch.frexp(a)
    exp.sub_(1 + mantissa_bits)
    # Round `2 * m * L` onto the FP8 grid; `m * 2 * L` is exact in fp32 for
    # every supported `mantissa_bits`.
    mant.mul_(2.0 * levels)
    mant.round_()
    # Sub-normals are not representable; frexp maps them (and zero) below the
    # threshold, so a single masked fill covers both.
    mant.masked_fill_(exp < min_exp, 0.0)
    return torch.ldexp(mant, exp).to(in_dtype)


# ---------------------------------------------------------------------------
# FP8 forward simulation (quantize -> dequantize, a.k.a. "fake quant")
# ---------------------------------------------------------------------------
class FP8LinearSim:
    """Wrap a module so its matmuls run on FP8-quantized operands.

    This mirrors how FP8 training works on real hardware: the *inputs* and
    *outputs* of every GEMM are cast to FP8 (quantized-dequantized, since
    the simulation runs on fp32 tensors) while the accumulation itself stays
    in higher precision. Measuring only the norms' outputs cannot observe how
    much error actually accumulates through the network, and cannot answer the
    "which layer first produces Inf/NaN" question at all.

    Both GEMM operands are quantized: the activation argument *and* the
    weight. `F.linear(x, W)` is an fp8 x fp8 product on real hardware, so
    quantizing only `x` would leave this an A8-only simulation and would
    systematically understate the error - especially for the weight term,
    which is constant across the probe batch and therefore free to re-use.
    """

    def __init__(self, module: nn.Module, quant):
        self.module = module
        self.quant = quant
        self.orig_forward = module.forward

        def fwd(x):
            # `nn.Linear.forward` is exactly `F.linear(x, self.weight,
            # self.bias)`, so the quantized weight is injected here: it is not
            # reachable through the argument list, which takes only `input`.
            y = F.linear(self.quant(x), self.quant(module.weight), module.bias)
            return self.quant(y)

        module.forward = fwd


class FP8SwiGLUSim:
    """FP8 simulation for the SwiGLU MLP.

    `MLP.forward` goes through `_SwiGLU.apply`, which calls `F.linear`
    directly, so this wraps `F.linear` itself rather than the child
    `nn.Linear` modules, and covers the MLP weights too.
    """

    def __init__(self, mlp: MLP, quant):
        self.mlp = mlp
        self.quant = quant
        self.orig_forward = mlp.forward

        def fwd(x):
            q = self.quant
            w1, w2, w3 = q(mlp.w1.weight), q(mlp.w2.weight), q(mlp.w3.weight)
            x_q = q(x)
            a = q(F.linear(x_q, w1))
            b = q(F.linear(x_q, w3))
            h = q(F.silu(a) * b)
            return mlp.drop(q(F.linear(h, w2)))

        mlp.forward = fwd

    @property
    def module(self):
        return self.mlp


@contextmanager
def fp8_forward(model: nn.Module, quant):
    """Run `model`'s forward with every GEMM on FP8-quantized operands."""
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
    """FP8 forward-divergence and trainability stats for one model.

    Returns the relative divergence of the FP8-simulated logits against the
    fp32 logits, per-norm activation-range statistics before/after each
    norm, the first block whose residual stream leaves the FP8 range, and
    the first block whose residual stream goes non-finite.
    """
    handles = []
    quant_errors = []
    ranges = []
    fp8_e4m3_min = fp8_min_normal(4)

    def make_stat_hook():
        """One hook feeding both the range table and the quant-error table."""
        def hook(module, inp, out):
            x_in = inp[0] if isinstance(inp, tuple) and inp else None
            cls_name = module.__class__.__name__
            has_in = torch.is_tensor(x_in)
            abs_out = out.abs()
            max_abs_out = abs_out.max().item()
            abs_in = x_in.abs() if has_in else None
            ranges.append({
                "module": cls_name,
                "max_abs_in": abs_in.max().item() if has_in else 0.0,
                "max_abs_out": max_abs_out,
                "kurtosis_in": _excess_kurtosis(x_in) if has_in else 0.0,
            })
            # out is the post-norm activation
            q = to_fp8_e4m3(out)
            denom = abs_out.mean().item() + 1e-6
            quant_errors.append({
                "module": cls_name,
                "max_abs": max_abs_out,
                "overflow_rate": (abs_out > FP8_E4M3_MAX).float().mean().item(),
                "underflow_rate": (abs_out < fp8_e4m3_min).float().mean().item() if out.numel() else 0.0,
                "rel_err": (q - out).abs().mean().item() / denom,
            })
        return hook

    first_bad = []
    first_over = []

    def make_block_hook(i):
        def hook(module, inp, out):
            y = out[0] if isinstance(out, tuple) else out
            if not torch.isfinite(y).all():
                first_bad.append(i)
                return
            # `simulate_fp8` saturates anything above `max_val` back down to
            # +-448, so an overflowing residual stream is finite by the time
            # it leaves the wrapper and a finite-check can never see it. The
            # magnitude itself is the thing FP8 cannot represent, so it is
            # recorded here as its own failure mode.
            if y.abs().amax().item() > FP8_E4M3_MAX:
                first_over.append(i)
        return hook

    # Reference logits in fp32 (no hooks: we only want to characterize the
    # activations actually seen by the FP8 GEMMs in the pass below, and
    # attaching here would double-count every norm).
    with torch.no_grad():
        logits_fp32, _ = model(x)

    for name, m in model.named_modules():
        if m.__class__.__name__ in {"LayerNorm", "RMSNorm", "DyT", "DyTC", "DyTD", "DyTCD"}:
            handles.append(m.register_forward_hook(make_stat_hook()))

    block_handles = [
        blk.register_forward_hook(make_block_hook(i))
        for i, blk in enumerate(model.blocks)
    ]
    with fp8_forward(model, to_fp8_e4m3):
        with torch.no_grad():
            logits_fp8, _ = model(x)
    for h in block_handles:
        h.remove()

    for h in handles:
        h.remove()

    # Relative forward divergence of the logits.
    finite = torch.isfinite(logits_fp8).all()
    if finite:
        rel = ((logits_fp8 - logits_fp32).abs().max()
               / logits_fp32.abs().max().clamp(min=1e-12)).item()
        logit_nan = False
    else:
        rel = float("inf")
        logit_nan = True

    if not quant_errors:
        # A model with no hookable norm modules: emit the same key set as the
        # normal path so rows in results.json stay homogeneous and downstream
        # `row[key]` lookups cannot KeyError halfway through a sweep.
        return {
            "norm": norm_name,
            "n_layers": int(getattr(model.cfg, "n_layers", 0)),
            "n_norm_hooks": 0,
            "max_abs_pre_norm": 0.0,
            "max_abs_post_norm": 0.0,
            "mean_excess_kurtosis_pre_norm": 0.0,
            "mean_overflow_rate_e4m3": 0.0,
            "max_overflow_rate_e4m3": 0.0,
            "mean_underflow_rate_e4m3": 0.0,
            "mean_rel_quant_err_e4m3": 0.0,
            "logit_rel_divergence": rel,
            "logit_nan_under_fp8": logit_nan,
            "first_nonfinite_layer": first_bad[0] if first_bad else None,
            "first_overflow_layer": first_over[0] if first_over else None,
        }

    mean_overflow = sum(e["overflow_rate"] for e in quant_errors) / len(quant_errors)
    max_overflow = max(e["overflow_rate"] for e in quant_errors)
    mean_underflow = sum(e["underflow_rate"] for e in quant_errors) / len(quant_errors)
    mean_rel_err = sum(e["rel_err"] for e in quant_errors) / len(quant_errors)
    max_max = max(e["max_abs"] for e in quant_errors)
    max_in = max(r["max_abs_in"] for r in ranges) if ranges else 0.0
    mean_kurt = (sum(r["kurtosis_in"] for r in ranges) / len(ranges)) if ranges else 0.0

    return {
        "norm": norm_name,
        # The block count, not the number of norm hooks: there are two norms
        # per block plus the final norm, so reporting the hook count here
        # labelled it `n_layers` while printing (2L+1).
        "n_layers": int(getattr(model.cfg, "n_layers", 0)),
        "n_norm_hooks": len(quant_errors),
        "max_abs_pre_norm": max_in,
        "max_abs_post_norm": max_max,
        "mean_excess_kurtosis_pre_norm": mean_kurt,
        "mean_overflow_rate_e4m3": mean_overflow,
        "max_overflow_rate_e4m3": max_overflow,
        "mean_underflow_rate_e4m3": mean_underflow,
        "mean_rel_quant_err_e4m3": mean_rel_err,
        "logit_rel_divergence": rel,
        "logit_nan_under_fp8": logit_nan,
        "first_nonfinite_layer": first_bad[0] if first_bad else None,
        "first_overflow_layer": first_over[0] if first_over else None,
    }


def _excess_kurtosis(x: torch.Tensor) -> float:
    """Excess kurtosis of a tensor, in fp64 for stability."""
    xd = x.detach().double().flatten()
    if xd.numel() < 4:
        return 0.0
    # `torch.var_mean` returns `(variance, mean)` in that order.
    var, mean = torch.var_mean(xd, unbiased=False)
    if var <= 0:
        return 0.0
    d2 = xd - mean
    d2.mul_(d2)
    d2.mul_(d2)
    return (d2.mean() / (var * var) - 3.0).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norms", nargs="+", default=["ln", "rmsnorm", "dyt", "dytc", "dytd", "dytcd"])
    p.add_argument("--probe", default="markov",
                   choices=["markov", "random"],
                   help="probe corpus. 'random' is uniform token ids: it has "
                        "no next-token structure, so the activation magnitudes "
                        "below describe embedding noise rather than a trained "
                        "model's activations (kept only for parity)")
    p.add_argument("--ckpt", nargs="+", default=None,
                   help="one checkpoint per --norms entry; when given, the "
                        "probe runs on the trained weights instead of a "
                        "fresh random init")
    p.add_argument("--n-states", type=int, default=64,
                   help="state count of the synthetic 'markov' probe")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/fp8_check/results.json")
    args = p.parse_args()

    if args.ckpt and len(args.ckpt) != len(args.norms):
        p.error(f"--ckpt expects {len(args.norms)} paths (one per --norms), got {len(args.ckpt)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    def make_probe(rng, cfg):
        if args.probe == "random":
            return torch.randint(
                0, cfg.vocab_size, (args.batch_size, args.seq_len),
                generator=rng, device=device,
            )
        # A real walk, not independent draws: the activation statistics
        # measured below have to be the ones a trained model sees.
        x, _ = markov_batch(
            rng, cfg.ignore_index, args.batch_size, args.seq_len, device,
            n_states=args.n_states,
        )
        return x

    results = []
    for i, norm in enumerate(args.norms):
        # Re-seed *both* streams per norm: otherwise every norm gets a
        # different random init AND a different probe batch, and the
        # overflow/underflow comparison is confounded by both.
        torch.manual_seed(args.seed)
        rng = torch.Generator(device=device).manual_seed(args.seed)
        if args.ckpt:
            from scripts.eval_downstream import load_ckpt
            # A freshly initialised model has no trained outlier channels and
            # `layerscale_init=1e-5` pins the residual stream to the embedding
            # table, so every norm looks identical and the metric reads
            # embedding noise. Probing a trained checkpoint is the only way
            # this experiment can measure what it claims to measure.
            model, _ = load_ckpt(args.ckpt[i], device=device)
            model.eval()
        else:
            model = build_gpt_124m(norm_name=norm).to(device)
            model.reset_parameters(args.seed)
            model.eval()
        x = make_probe(rng, model.cfg)
        r = measure_fp8_divergence(model, x, norm)
        if args.ckpt:
            r["ckpt"] = args.ckpt[i]
        print(f"[fp8] {norm}: {r}")
        results.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] saved to {out_path}")


if __name__ == "__main__":
    main()
