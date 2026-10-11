"""
Post-training W8A8 quantization PPL drop check.

Naive round-to-nearest W8A8 quantization.

We use a per-channel weight quantization (axis=1) and per-token
activation quantization (axis=-1), which is the same scheme as
llm.int8() minus the mixed-precision detour - the detour exists
precisely to handle the outliers we're trying to suppress.

Usage:
    python scripts/check_w8a8_ppl.py --ckpts runs/gpt2-124m-ln/ckpt_final.pt runs/gpt2-124m-rmsnorm/ckpt_final.pt runs/gpt2-124m-dytcd/ckpt_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from adasquash import GPTConfig, GPT
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b
from scripts.eval_downstream import load_ckpt


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def quantize_weight_per_channel(w: torch.Tensor, n_bits: int = 8):
    """Per-output-channel (axis=0) symmetric int quantization.
    w: [out, in] -> [out, in]"""
    mx = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    q_max = (2 ** (n_bits - 1)) - 1
    scale = mx / q_max
    q = torch.round(w / scale).clamp(-q_max - 1, q_max)
    return q * scale, scale


def quantize_activation_per_token(x: torch.Tensor, n_bits: int = 8):
    """Per-token (last-dim) symmetric int quantization.
    x: [..., C] -> [..., C]"""
    mx = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    q_max = (2 ** (n_bits - 1)) - 1
    scale = mx / q_max
    q = torch.round(x / scale).clamp(-q_max - 1, q_max)
    return q * scale


def wrap_linear_w8a8(linear: nn.Linear):
    """Replace a linear's forward with W8A8 quantized versions."""
    original_forward = linear.forward

    def quantized_forward(x):
        # weight quantization happens lazily once.
        if not hasattr(linear, "_q_weight"):
            qw, _ = quantize_weight_per_channel(linear.weight.data)
            linear._q_weight = qw
        x_q = quantize_activation_per_token(x)
        return torch.nn.functional.linear(x_q, linear._q_weight, linear.bias)

    linear.forward = quantized_forward
    return linear


def wrap_model_w8a8(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            wrap_linear_w8a8(m)
    return model


# ---------------------------------------------------------------------------
# PPL measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_ce_sum(model, batch, target, device="cuda"):
    """Summed (not averaged) cross-entropy over `target` tokens.

    Returning the raw sum lets several probe batches be pooled into a
    single PPL estimate - averaging per-batch PPLs would weight every
    batch equally regardless of its token count. Padding targets are
    masked exactly like training (ignore_index), so the padded final
    position does not inflate the PPL, and only valid tokens are counted.
    """
    model.eval()
    batch = batch.to(device)
    target = target.to(device)
    logits, _ = model(batch, target)
    # logits: [B, T, V]
    ce = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        target.view(-1),
        reduction="sum",
        ignore_index=model.cfg.pad_token_id,
    )
    n = (target != model.cfg.pad_token_id).sum().item()
    return ce.item(), n


def compute_ppl(model, batch, target, device="cuda"):
    ce, n = compute_ce_sum(model, batch, target, device=device)
    return torch.exp(torch.tensor(ce / n)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/w8a8/results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    results = []
    for ckpt_path in args.ckpts:
        # Re-seed per checkpoint so every checkpoint is evaluated on the
        # *same* probe batches; otherwise the cross-checkpoint ppl_drop
        # comparison also carries the variance of a different eval sample.
        rng = torch.Generator(device=device).manual_seed(args.seed)
        # Load fp32 model
        model, _ = load_ckpt(ckpt_path, device=device)
        # Build a *copy* of the model for quantization so we can compare.
        # The cleanest way is a deepcopy, but state_dict round-trip is
        # safer for our custom modules.
        import copy
        qmodel = copy.deepcopy(model)
        qmodel.to(device)
        wrap_model_w8a8(qmodel)

        ce_fp = ce_q = 0.0
        n_tok = 0
        for _ in range(max(1, args.n_batches)):
            # Build random data.
            x = torch.randint(
                0, model.cfg.vocab_size, (args.batch_size, args.seq_len),
                generator=rng, device=device,
            )
            y = torch.cat(
                [x[:, 1:], torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)],
                dim=1,
            )
            a, n = compute_ce_sum(model, x, y, device=device)
            b, _ = compute_ce_sum(qmodel, x, y, device=device)
            ce_fp += a
            ce_q += b
            n_tok += n
        ppl_fp = math.exp(ce_fp / n_tok)
        ppl_q = math.exp(ce_q / n_tok)

        r = {
            "ckpt": ckpt_path,
            "ppl_fp": ppl_fp,
            "ppl_w8a8": ppl_q,
            "ppl_drop": ppl_q - ppl_fp,
            "ppl_drop_pct": 100.0 * (ppl_q - ppl_fp) / ppl_fp,
        }
        print(f"[w8a8] {ckpt_path}: {r}")
        results.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] saved to {out_path}")


if __name__ == "__main__":
    main()
