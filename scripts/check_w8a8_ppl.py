"""
Post-training W8A8 quantization PPL drop check.

Naive round-to-nearest W8A8 quantization.

We use a per-channel weight quantization (axis=0) and per-token
activation quantization (axis=-1), which is the same scheme as
llm.int8() minus the mixed-precision detour - the detour exists
precisely to handle the outliers we're trying to suppress.

The probe corpus matters as much as the quantizer: on uniform random
tokens there is no next-token signal, both models sit at
ppl ~= vocab_size, and the reported drop is noise (it can even be
negative). `--probe markov` (default) uses the synthetic learnable corpus
so the metric has resolution; `--probe dataset` uses real text.

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
import torch.nn.functional as F

from adasquash import GPT, markov_batch
from adasquash.model import MLP
from scripts.eval_downstream import load_ckpt


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def quantize_weight_per_channel(w: torch.Tensor, n_bits: int = 8):
    """Per-output-channel (axis=0) symmetric int quantization.

    w: [out, in] -> [out, in]

    The operation is idempotent: a matrix already on the int8 grid has
    ``max|w| / 127`` for a scale and integral ``w / scale``, so re-quantizing
    reproduces it exactly. That is what lets a weight shared by several
    accessors (a tied embedding / head) be quantized more than once without
    the copies drifting apart.
    """
    mx = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-6)
    q_max = (2 ** (n_bits - 1)) - 1
    scale = mx / q_max
    # saturation clamp and the dequantizing scale all run on it.
    q = w / scale
    q.round_()
    q.clamp_(-q_max - 1, q_max)
    q.mul_(scale)
    return q, scale


def quantize_activation_per_token(x: torch.Tensor, n_bits: int = 8):
    """Per-token (last-dim) symmetric int quantization.
    x: [..., C] -> [..., C]"""
    mx = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-6)
    q_max = (2 ** (n_bits - 1)) - 1
    scale = mx / q_max
    q = x / scale
    q.round_()
    q.clamp_(-q_max - 1, q_max)
    q.mul_(scale)
    return q


def _install_quantized_weight(linear: nn.Module, qw: torch.Tensor):
    """Make `qw` the weight every consumer of this module sees.

    Two consumers exist. The wrapped forward reads `_q_weight`. Any caller
    that reaches for the matrix directly reads `linear.weight` - and
    `GPT.forward(return_logits=False)` does exactly that, because the fused
    loss takes `lm_head.weight` as an argument and never calls
    `lm_head.forward`. Replacing the parameter in place as well is what keeps
    the output projection quantized on that path; otherwise that branch would
    still read the fp32 matrix and the run would measure a partly-quantized
    network.
    """
    with torch.no_grad():
        linear.weight.copy_(qw)
    linear._q_weight = qw


def wrap_linear_w8a8(linear: nn.Linear):
    """Replace a linear's forward with W8A8 quantized versions."""
    qw, _ = quantize_weight_per_channel(linear.weight.data)
    _install_quantized_weight(linear, qw)

    def quantized_forward(x):
        x_q = quantize_activation_per_token(x)
        return torch.nn.functional.linear(x_q, qw, linear.bias)

    linear.forward = quantized_forward
    return linear


def wrap_swiglu_w8a8(mlp):
    """Quantize a SwiGLU `MLP`.

    `MLP.forward` calls `_SwiGLU.apply` with the raw `nn.Linear.weight`
    tensors, so wrapping the child `nn.Linear` modules (as
    `wrap_linear_w8a8` does) never takes effect: every SwiGLU weight would
    stay fp32 and the reported W8A8 PPL drop would be measured on a model
    that is only partly quantized - the MLP holds the majority of the
    arithmetic. Wrapping the child forwards would then also leave the SwiGLU
    *hidden* activation (`silu(w1 x) * (w3 x)`, the most outlier-prone
    tensor in the network) in fp32 while every other GEMM input is int8. The
    module is therefore wrapped at once, quantizing each GEMM input in turn.

    The weights are quantized once, here: they do not change afterwards, and
    `quantize_weight_per_channel` is idempotent.
    """
    weights = []
    for lin in (mlp.w1, mlp.w2, mlp.w3):
        qw, _ = quantize_weight_per_channel(lin.weight.data)
        _install_quantized_weight(lin, qw)
        weights.append(qw)
    qa = quantize_activation_per_token

    def quantized_forward(x):
        x_q = qa(x)
        a = qa(F.linear(x_q, weights[0]))
        b = qa(F.linear(x_q, weights[2]))
        h = F.silu(a)
        h.mul_(b)
        return mlp.drop(F.linear(qa(h), weights[1]))

    mlp.forward = quantized_forward
    return mlp


def wrap_embedding_w8a8(emb: nn.Embedding):
    """Quantize an embedding lookup.

    GPT-2 ties `lm_head.weight` to `tok_emb.weight`. `wrap_model_w8a8`
    reaches `lm_head` as an `nn.Linear` and gives it a quantized *copy*,
    while `nn.Embedding` keeps reading the original fp32 tensor - so the
    same weight would be consumed at two different precisions inside one
    forward. Quantizing the lookup keeps the two sides consistent.
    """
    qw, _ = quantize_weight_per_channel(emb.weight.data)

    def quantized_forward(idx):
        return F.embedding(idx, qw)

    with torch.no_grad():
        emb.weight.copy_(qw)
    emb._q_weight = qw
    emb.forward = quantized_forward
    return emb


def wrap_model_w8a8(model: nn.Module):
    """Wrap `model` (in place) so every GEMM runs W8A8.

    The weights are replaced, not just shadowed behind an overridden
    `forward`: consumers that read a weight matrix directly - most
    importantly `fused_cross_entropy_mean`, which takes `lm_head.weight` as
    an argument - would otherwise keep using the fp32 copy. Wrap a copy
    unless the fp32 weights are throwaway.

    `model.modules()` is recursive, so an `MLP` that `wrap_swiglu_w8a8` has
    already taken over is *still* descended into afterwards and its three
    `quantized_forward` closures are never called because `MLP.forward` has
    been replaced wholesale. The sub-modules of a wrapped `MLP` are therefore
    skipped explicitly - continuing the loop does not, because the generator
    recurses on its own.

    A tied `lm_head` is *not* skipped: the embedding branch only puts the
    shared matrix on the int8 grid, while the linear branch is what
    quantizes that GEMM's activation.
    """
    handled: set[int] = set()
    for m in model.modules():
        if isinstance(m, MLP):
            wrap_swiglu_w8a8(m)
            handled.update(id(child) for child in (m.w1, m.w2, m.w3))
            continue
        if isinstance(m, nn.Embedding):
            wrap_embedding_w8a8(m)
            continue
        if isinstance(m, nn.Linear):
            if id(m) in handled:
                continue
            wrap_linear_w8a8(m)
    # The `lm_head` wrapper only covers `nn.Linear.forward`, but
    # `fused_cross_entropy_mean` is handed `lm_head.weight` directly and never
    # calls it (see `_install_quantized_weight`), so on the measurement path
    # the output projection would otherwise receive an fp32 activation:
    # A16xW8 for the largest GEMM in the model and a `ppl_drop` measured on a
    # partly quantized network. `norm_f`'s output is that projection's input
    # and nothing else consumes it, so the per-token quantizer is installed on
    # it instead - a hook that returns a value replaces the module's output.
    norm_f = getattr(model, "norm_f", None)
    if isinstance(norm_f, nn.Module) and isinstance(
        getattr(model, "lm_head", None), nn.Module
    ):
        norm_f.register_forward_hook(
            lambda m, inp, out: quantize_activation_per_token(out)
        )
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
    n = (target != model.cfg.ignore_index).sum().item()
    if n == 0:
        raise ValueError("probe batch contains no valid target tokens")
    # The model's own loss is already the mean over the non-ignored targets
    # (see `cross_entropy_mean`), so scaling it back up by `n` gives the sum
    # over those targets.
    loss = model(batch, target, return_logits=False)[1]
    return loss.item() * n, n


def compute_ppl(model, batch, target, device="cuda"):
    ce, n = compute_ce_sum(model, batch, target, device=device)
    return math.exp(ce / n)


def probe_token_coverage(batches) -> float:
    """Fraction of probe positions holding a *distinct* token id.

    A uniformly random probe over a large vocabulary is almost entirely
    distinct (each id shows up about once), so no predictive model - bigram,
    trigram or transformer - can beat chance on it and every PPL sits at the
    vocab_size ceiling. A corpus with real next-token structure repeats
    tokens, and the coverage falls by orders of magnitude (a 64-state Markov
    chain is <= 0.03; natural text is typically <= 0.15).

    This is a property of the probe alone, so it separates "the corpus is
    degenerate" (a hard error here) from "the checkpoint is untrained"
    (reported separately as a warning).

    Measured exactly rather than estimated: an empirical bigram conditional
    entropy would be badly biased downward at this sample size, where most
    (prev, next) pairs are seen zero or one times.
    """
    n_pos = 0
    ids: set = set()
    for x in batches:
        flat = x.reshape(-1)
        ids.update(flat.tolist())
        n_pos += flat.numel()
    return len(ids) / max(1, n_pos)


# ---------------------------------------------------------------------------
# Probe batches
# ---------------------------------------------------------------------------
def _probe_random(rng, vocab_size, batch_size, seq_len, device):
    """Uniform random tokens.

    They carry no next-token signal: every PPL comes out at ~vocab_size and
    the reported `ppl_drop_pct` is noise (it can even come out negative).
    """
    return torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=rng, device=device
    )


def _probe_markov(rng, batch_size, seq_len, device, n_states=64):
    """Learnable synthetic corpus (see `adasquash.utils.markov_batch`)."""
    x, _ = markov_batch(rng, -100, batch_size, seq_len, device, n_states=n_states)
    return x


def make_dataset_probe(dataset, dataset_config, tokenizer, batch_size, seq_len):
    """Real text, packed into fixed-length streams.

    Returns a stateful zero-argument callable: the streaming dataset is
    opened once and cycled.
    """
    from scripts.train_gpt2 import build_dataloader
    loader = build_dataloader(
        dataset, dataset_config, seq_len, batch_size, tokenizer=tokenizer
    )
    it = iter(loader)

    def _next():
        nonlocal it
        try:
            return next(it)[0]
        except StopIteration:
            it = iter(loader)
            return next(it)[0]

    return _next


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--probe", default="markov",
                   choices=["markov", "dataset", "random"],
                   help="probe corpus. 'random' is uniform tokens: it has no "
                        "next-token signal, so ppl_fp ~= vocab_size and the "
                        "PPL drop is unmeasurable (kept only for parity)")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--tokenizer", default="gpt2",
                   help="tokenizer for --probe dataset; a checkpoint that "
                        "recorded the one it was trained with wins")
    p.add_argument("--n-states", type=int, default=64,
                   help="state count of the synthetic 'markov' probe")
    p.add_argument("--max-probe-coverage", type=float, default=0.5,
                   help="reject a probe whose tokens are more than this "
                        "fraction distinct (uniform noise is ~1.0)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--n-batches", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="runs/w8a8/results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    def make_probe(rng, cfg, dataset_probe):
        if args.probe == "random":
            return _probe_random(rng, cfg.vocab_size, args.batch_size,
                                 args.seq_len, device)
        if args.probe == "markov":
            return _probe_markov(rng, args.batch_size, args.seq_len, device,
                                 n_states=args.n_states)
        return dataset_probe()

    results = []
    for ckpt_path in args.ckpts:
        # Re-seed per checkpoint so every checkpoint is evaluated on the
        # *same* probe batches; otherwise the cross-checkpoint ppl_drop
        # comparison also carries the variance of a different eval sample.
        rng = torch.Generator(device=device).manual_seed(args.seed)
        # Load fp32 model
        model, ckpt_args = load_ckpt(ckpt_path, device=device)
        # The probe corpus is tokenized with the checkpoint's own tokenizer:
        # `train_gpt2.py` records the one it was launched with, and feeding a
        # 50k-class GPT-2 tokenizer to the depth sweep's 32k-vocabulary model
        # is an index error out of the embedding (or, for the ids that happen
        # to fit, a silently differently tokenized corpus).
        #
        # The dataset probe is (re)built per checkpoint for the same reason
        # `rng` is re-seeded: `make_dataset_probe` is a *stateful* stream, so
        # building it once above would let checkpoint #2 resume where #1
        # stopped and the two checkpoints would be compared on two different
        # eval samples. `markov` / `random` are pure functions of `rng` and
        # are unaffected.
        dataset_probe = (
            make_dataset_probe(
                args.dataset, args.dataset_config,
                ckpt_args.get("tokenizer") or args.tokenizer,
                args.batch_size, args.seq_len,
            )
            if args.probe == "dataset" else None
        )

        # Build every probe batch up front, so the fp32 and the quantized
        # measurement see exactly the same data without either model having
        # to exist while the other runs.
        batches = []
        for _ in range(max(1, args.n_batches)):
            x = make_probe(rng, model.cfg, dataset_probe)
            batches.append(x)

        # The cleanest way to compare the two precisions is to quantize a
        # copy, but measuring the fp32 model first and then wrapping *the
        # same* model in place keeps the comparison bit-for-bit the same:
        # `wrap_model_w8a8` is deterministic and the batches are already
        # materialized.
        ce_fp = 0.0
        n_tok = 0
        for x in batches:
            # Shift-by-one targets; the final position has no successor and is
            # padded with `cfg.ignore_index` (out of vocabulary) rather than
            # token 0, which is a real GPT-2 token ("!") and would otherwise
            # be counted as a scored prediction.
            y = torch.full(
                (args.batch_size, args.seq_len), model.cfg.ignore_index,
                dtype=torch.long, device=device,
            )
            y[:, :-1] = x[:, 1:]
            a, n = compute_ce_sum(model, x, y, device=device)
            ce_fp += a
            n_tok += n

        wrap_model_w8a8(model)
        ce_q = 0.0
        for x in batches:
            y = torch.full(
                (args.batch_size, args.seq_len), model.cfg.ignore_index,
                dtype=torch.long, device=device,
            )
            y[:, :-1] = x[:, 1:]
            b, _ = compute_ce_sum(model, x, y, device=device)
            ce_q += b
        ppl_fp = math.exp(ce_fp / n_tok)
        ppl_q = math.exp(ce_q / n_tok)

        # A probe with no next-token signal pins the CE near log(vocab_size)
        # for *both* models, so the difference is pure noise. Measured on the
        # probe itself, so an untrained checkpoint is not mistaken for a
        # broken corpus.
        coverage = probe_token_coverage(batches)
        if coverage > args.max_probe_coverage:
            raise ValueError(
                f"probe='{args.probe}' carries no next-token signal: "
                f"{coverage:.0%} of its tokens are distinct, so nothing "
                f"repeats and no model can beat chance. Use "
                f"--probe markov|dataset (--max-probe-coverage overrides)."
            )

        r = {
            "ckpt": ckpt_path,
            "probe": args.probe,
            "ppl_fp": ppl_fp,
            "ppl_w8a8": ppl_q,
            "ppl_drop": ppl_q - ppl_fp,
            "ppl_drop_pct": 100.0 * (ppl_q - ppl_fp) / ppl_fp,
            "probe_token_coverage": coverage,
        }
        # Distinct from the hard check above: a high PPL here means the
        # *checkpoint* is untrained, which is the user's call, not an error.
        if ppl_fp >= 0.5 * model.cfg.vocab_size:
            print(f"[w8a8] WARNING: {ckpt_path} ppl_fp={ppl_fp:.1f} is at the "
                  f"vocab_size={model.cfg.vocab_size} ceiling - the probe is "
                  f"fine (coverage={coverage:.1%}) but this checkpoint appears "
                  f"untrained, so ppl_drop_pct is not meaningful yet",
                  file=sys.stderr)
        print(f"[w8a8] {ckpt_path}: {r}")
        results.append(r)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[done] saved to {out_path}")


if __name__ == "__main__":
    main()
