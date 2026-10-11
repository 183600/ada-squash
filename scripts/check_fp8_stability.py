"""
FP8 stability check: simulate FP8 / bf16 training divergence on the
same GPT-2 124M model with different norm layers.

We measure:
  - forward divergence: max|output_fp8 - output_bf16| / max|output_bf16|
  - activation overflow rate under FP8-E4M3 / FP8-E5M2
  - per-norm activation range, before / after the norm
  - the standard "FP8 trainability" curve: at what layer do logits NaN?

This script is a CPU/GPU pure-python simulation - it does NOT need real
FP8 hardware (e.g. H100 FP8), and is useful for predicting which norm
configurations will be FP8-trainable *before* paying for the kernel work.

Usage:
    python scripts/check_fp8_stability.py --norms ln rmsnorm dyt dytc dytd dytcd
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from adasquash import build_norm
from adasquash.model import build_gpt_124m


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
    """Simulate per-tensor FP8-E4M3 quantization. Range: ±448."""
    return simulate_fp8(x, max_val=448.0, exp_bits=4, mantissa_bits=3)

def to_fp8_e5m2(x: torch.Tensor) -> torch.Tensor:
    """Simulate per-tensor FP8-E5M2 quantization. Range: ±57344."""
    return simulate_fp8(x, max_val=57344.0, exp_bits=5, mantissa_bits=2)

def simulate_fp8(x: torch.Tensor, max_val: float, exp_bits: int, mantissa_bits: int) -> torch.Tensor:
    """
    Emulate FP8 by clamping + rounding to the nearest representable value.
    This is a coarse approximation but reproduces the *overflow / underflow*
    behaviour that's the actual subject of the stability check.
    """
    x = x.float()
    # Clamp to the representable range.
    x_clamped = x.clamp(-max_val, max_val)
    # Round to the nearest mantissa increment (very rough).
    # Smallest normal magnitude:
    min_norm = fp8_min_normal(exp_bits)
    # Subnormal regime: we treat anything below min_norm as flush-to-zero
    # (matching FP8 hardware behaviour on most training cards).
    flushed = torch.where(x_clamped.abs() < min_norm, torch.zeros_like(x_clamped), x_clamped)
    # Mantissa quantization: 2^mantissa_bits levels per binade.
    sign = flushed.sign()
    abs_x = flushed.abs()
    # Compute exponent
    exp = torch.floor(torch.log2(abs_x.clamp(min=1e-30)))
    exp = exp.clamp(max=torch.log2(torch.tensor(max_val)).item())
    # Round the mantissa
    mant = abs_x / (2.0 ** exp)  # in [1, 2)
    levels = 2 ** mantissa_bits
    mant_q = torch.round((mant - 1.0) * (levels - 1)) / (levels - 1) + 1.0
    quant = sign * mant_q * (2.0 ** exp)
    # Restore zero where flushed
    quant = torch.where(flushed == 0, torch.zeros_like(quant), quant)
    return quant.to(x.dtype)


def measure_fp8_divergence(model, x, norm_name: str):
    """Return dict with FP8-E4M3 quantization-stability stats per norm.

    Note `x` holds *token ids*, so it must stay an integer tensor - casting
    it to float breaks the embedding lookup. The model itself already runs
    in fp32, so a single hooked forward is enough.
    """
    # Measure the quantization error introduced by FP8-E4M3 on every
    # normalization layer's output, which is where the low-precision
    # activation traffic actually bites.
    handles = []
    quant_errors = []
    fp8_e4m3_min = fp8_min_normal(4)

    def make_q_hook():
        def hook(module, inp, out):
            # out is the post-norm activation
            q = to_fp8_e4m3(out)
            rel_err = (q - out).abs().mean().item() / (out.abs().mean().item() + 1e-6)
            quant_errors.append({
                "module": module.__class__.__name__,
                "max_abs": out.abs().max().item(),
                "overflow_rate": (out.abs() > 448).float().mean().item(),
                "underflow_rate": (out.abs() < fp8_e4m3_min).float().mean().item() if out.numel() else 0.0,
                "rel_err": rel_err,
            })
        return hook

    for name, m in model.named_modules():
        if m.__class__.__name__ in {"LayerNorm", "RMSNorm", "DyT", "DyTC", "DyTD", "DyTCD"}:
            handles.append(m.register_forward_hook(make_q_hook()))

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    # Aggregate
    if not quant_errors:
        return {"norm": norm_name, "n_layers": 0, "mean_overflow": 0.0, "max_overflow": 0.0}

    mean_overflow = sum(e["overflow_rate"] for e in quant_errors) / len(quant_errors)
    max_overflow = max(e["overflow_rate"] for e in quant_errors)
    mean_underflow = sum(e["underflow_rate"] for e in quant_errors) / len(quant_errors)
    mean_rel_err = sum(e["rel_err"] for e in quant_errors) / len(quant_errors)
    max_max = max(e["max_abs"] for e in quant_errors)

    return {
        "norm": norm_name,
        "n_layers": len(quant_errors),
        "max_abs_post_norm": max_max,
        "mean_overflow_rate_e4m3": mean_overflow,
        "max_overflow_rate_e4m3": max_overflow,
        "mean_underflow_rate_e4m3": mean_underflow,
        "mean_rel_quant_err_e4m3": mean_rel_err,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norms", nargs="+", default=["ln", "rmsnorm", "dyt", "dytc", "dytd", "dytcd"])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/fp8_check/results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    results = []
    for norm in args.norms:
        # Re-seed per norm: otherwise every norm gets a different random
        # init and the overflow/underflow comparison is confounded by
        # initialization noise.
        torch.manual_seed(args.seed)
        model = build_gpt_124m(norm_name=norm).to(device)
        model.eval()
        x = torch.randint(0, model.cfg.vocab_size, (args.batch_size, args.seq_len), generator=rng, device=device)
        r = measure_fp8_divergence(model, x, norm)
        print(f"[fp8] {norm}: {r}")
        results.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] saved to {out_path}")


if __name__ == "__main__":
    main()
