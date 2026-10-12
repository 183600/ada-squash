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
    def quantized_forward(x):
        # weight quantization happens lazily once.
        if not hasattr(linear, "_q_weight"):
            qw, _ = quantize_weight_per_channel(linear.weight.data)
            linear._q_weight = qw
        x_q = quantize_activation_per_token(x)
        return torch.nn.functional.linear(x_q, linear._q_weight, linear.bias)

    linear.forward = quantized_forward
    return linear


def wrap_swiglu_w8a8(mlp):
    """Quantize a SwiGLU `MLP`.

    `MLP.forward` calls `_SwiGLU.apply` with the raw `nn.Linear.weight`
    tensors, so wrapping the child `nn.Linear` modules (as
    `wrap_linear_w8a8` does) never takes effect: every SwiGLU weight would
    stay fp32 and the reported W8A8 PPL drop would be measured on a model
    that is only partly quantized - the MLP holds the majority of the
    parameters and is exactly where activation outliers live. The whole
    module is therefore wrapped at once.

    `_SwiGLU.apply` also performs the `w2` GEMM internally, so routing the
    forward through it would leave the SwiGLU *hidden* activation
    (`silu(w1 x) * (w3 x)`, the most outlier-prone tensor in the network)
    in fp32 while every other GEMM input is int8. The forward is therefore
    written out explicitly, quantizing each GEMM input in turn.
    """
    def quantized_forward(x):
        weights = []
        for lin in (mlp.w1, mlp.w2, mlp.w3):
            if not hasattr(lin, "_q_weight"):
                qw, _ = quantize_weight_per_channel(lin.weight.data)
                lin._q_weight = qw
            weights.append(lin._q_weight)
        qa = quantize_activation_per_token
        x_q = qa(x)
        a = qa(F.linear(x_q, weights[0]))
        b = qa(F.linear(x_q, weights[2]))
        h = qa(F.silu(a) * b)
        return mlp.drop(F.linear(h, weights[1]))

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
    def quantized_forward(idx):
        if not hasattr(emb, "_q_weight"):
            emb._q_weight, _ = quantize_weight_per_channel(emb.weight.data)
        return F.embedding(idx, emb._q_weight)

    emb.forward = quantized_forward
    return emb


def wrap_model_w8a8(model: nn.Module):
    for m in model.modules():
        if isinstance(m, MLP):
            wrap_swiglu_w8a8(m)
        elif isinstance(m, nn.Embedding):
            wrap_embedding_w8a8(m)
        elif isinstance(m, nn.Linear):
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
    n = (target != model.cfg.ignore_index).sum().item()
    if n == 0:
        raise ValueError("probe batch contains no valid target tokens")
    # The model's own loss is already the mean over the non-ignored targets
    # (see `cross_entropy_mean`), so scaling it back up by `n` gives the sum
    # over those targets.
    logits, loss = model(batch, target)
    del logits
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
    opened once and cycled, rather than re-instantiated (and re-sharded
    from scratch) on every batch.
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
    p.add_argument("--tokenizer", default="gpt2")
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

    dataset_probe = (
        make_dataset_probe(args.dataset, args.dataset_config, args.tokenizer,
                           args.batch_size, args.seq_len)
        if args.probe == "dataset" else None
    )

    def make_probe(rng, cfg):
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
        batches = []
        for _ in range(max(1, args.n_batches)):
            x = make_probe(rng, model.cfg)
            batches.append(x)
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
            b, _ = compute_ce_sum(qmodel, x, y, device=device)
            ce_fp += a
            ce_q += b
            n_tok += n
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
