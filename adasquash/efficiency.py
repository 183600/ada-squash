from __future__ import annotations
"""Efficiency accounting: parameters, FLOPs and the accuracy/cost verdict.

* `parameter_breakdown` splits a model's parameters by role (embedding /
  attention / MLP / normalization / LayerScale / LM head), counting tied
  storage once;
* `analytic_flops_per_token` derives the matmul FLOPs per token from the
  config (forward and backward), including the quadratic T x T attention
  score/context matmuls - the norm layers are elementwise and add no matmul
  FLOPs;
* `efficiency_verdict` marks a gain as viable only when it clears the
  practical margin and the cost ratio stays within the allowed maximum.

The functions are pure (config/model in, dict out) so ablation and sweep
runners can attach the numbers to their summaries.
"""
from collections import defaultdict
from typing import Optional

__all__ = [
    "DEFAULT_MIN_LOSS_GAIN",
    "DEFAULT_MAX_COST_RATIO",
    "analytic_flops_per_token",
    "efficiency_verdict",
    "mlp_hidden_size",
    "parameter_breakdown",
]

DEFAULT_MIN_LOSS_GAIN = 0.01
DEFAULT_MAX_COST_RATIO = 1.25


def mlp_hidden_size(dim: int, mlp_ratio: float) -> int:
    """Hidden width of the SwiGLU MLP, exactly as `model.MLP` computes it."""
    hidden = int(int(dim * mlp_ratio) * 2 / 3)
    return hidden + 7 & ~7


def analytic_flops_per_token(cfg, backward: bool = False, seq_len: int | None = None) -> dict:
    """Matmul FLOPs per token of the GPT decoder defined by `cfg`.

    Counts every weight-matrix multiplication (attention qkv/proj, SwiGLU
    w1/w2/w3, LM head) as 2 x multiply-adds *plus* the attention score and
    context matmuls: ``Q @ K^T`` and ``softmax(QK^T) @ V`` contribute
    ``4 * d * T^2`` FLOPs per layer, i.e. ``4 * d * T`` per token.

    `seq_len` is the context length the model is actually run at (defaults to
    `cfg.max_seq_len`; pass the trained length when it differs). The score
    count is the dense T x T convention; the causal mask serves only the lower
    triangle, so a causal-exact count is about half - the dense convention is
    the conservative upper bound and is applied identically to every cell.

    `backward=True` returns the training-step estimate: forward plus backward,
    with the multiplier used recorded as ``backward_factor``.
    """
    d = int(cfg.dim)
    n_layers = int(cfg.n_layers)
    vocab = int(cfg.vocab_size)
    h = mlp_hidden_size(d, float(cfg.mlp_ratio))
    T = int(seq_len) if seq_len is not None else int(getattr(cfg, "max_seq_len", 0) or 0)
    T = max(0, T)
    attn = float(n_layers * 8 * d * d)          # 4 d x d matmuls/token/layer
    attn_scores = float(4 * n_layers * d * T)   # QK^T + AV, 4 d T^2 per layer / T
    mlp = float(n_layers * 6 * d * h)           # 3 d x h matmuls/token/layer
    lm_head = float(2 * d * vocab)
    forward = attn + attn_scores + mlp + lm_head
    return {
        "attention": attn,
        "attention_scores": attn_scores,
        "mlp": mlp,
        "lm_head": lm_head,
        "forward_per_token": forward,
        "training_per_token": 3.0 * forward,
        "backward_factor": 3.0,
        "mlp_hidden": h,
        "seq_len": T,
        "note": (
            "matmul FLOPs per token (2 x MACs), including the quadratic attention "
            "score/context matmuls counted densely (causal masking roughly halves "
            "them; the dense convention is the conservative upper bound applied to "
            "every cell). Elementwise ops (softsign/tanh, LayerScale, softmax, "
            "SiLU) are not counted, so norm variants differ here only through the "
            "shared backbone"
        ),
    }


def _param_bucket(name: str) -> str:
    if name.startswith("_orig_mod."):  # torch.compile wrapper
        name = name[len("_orig_mod."):]
    if name.startswith("tok_emb"):
        return "embedding"
    if name.startswith("pos_emb"):
        return "positional_embedding"
    if name.startswith("lm_head"):
        return "lm_head"
    if "norm" in name:
        return "norm"
    if ".ls1" in name or ".ls2" in name or name.endswith(".ls1") or name.endswith(".ls2"):
        return "layerscale"
    if ".attn." in name or name.startswith("attn."):
        return "attention"
    if ".mlp." in name or name.startswith("mlp."):
        return "mlp"
    return "other"


def parameter_breakdown(model) -> dict:
    """Per-role parameter counts, tied storage counted once.

    Returns ``{'total', 'by_role', 'norm_share_of_total', 'tied_storage_skipped'}``.
    Tied embeddings share one tensor between `tok_emb.weight` and
    `lm_head.weight`; the second reference is counted under
    `tied_storage_skipped` rather than double-counting it.
    """
    by_role: dict = defaultdict(int)
    seen: set = set()
    skipped = 0
    for name, p in model.named_parameters(remove_duplicate=False):
        key = p.data_ptr()
        if key in seen:
            skipped += int(p.numel())
            continue
        seen.add(key)
        by_role[_param_bucket(name)] += int(p.numel())
    total = sum(by_role.values())
    norm = by_role.get("norm", 0)
    return {
        "total": total,
        "by_role": dict(sorted(by_role.items())),
        "norm_share_of_total": (norm / total) if total else 0.0,
        "tied_storage_skipped": skipped,
    }


def efficiency_verdict(
    loss_improvement: Optional[float],
    cost_ratio: Optional[float],
    min_loss_gain: float = DEFAULT_MIN_LOSS_GAIN,
    max_cost_ratio: float = DEFAULT_MAX_COST_RATIO,
    cost_name: str = "compute",
) -> dict:
    """Is `loss_improvement` worth `cost_ratio` times the baseline cost?

    `loss_improvement` is positive when the candidate's loss is lower than the
    reference's (a *better* model); `cost_ratio` is candidate/reference (e.g.
    latency, FLOPs or parameter ratio; 1.0 = free). A candidate is viable only
    when the gain clears `min_loss_gain` **and** the cost ratio stays within
    `max_cost_ratio`. `cost_per_loss_point` (cost_ratio per unit of gain) makes
    the exchange rate explicit.
    """
    verdict = {
        "loss_improvement": loss_improvement,
        "cost_ratio": cost_ratio,
        "cost_name": cost_name,
        "min_loss_gain": float(min_loss_gain),
        "max_cost_ratio": float(max_cost_ratio),
        "gain_meaningful": None,
        "cost_acceptable": None,
        "viable": None,
        "cost_per_loss_point": None,
        "reason": "",
    }
    if loss_improvement is None or cost_ratio is None:
        verdict["reason"] = "missing loss or cost measurement; nothing to judge"
        return verdict
    gain = float(loss_improvement)
    cost = float(cost_ratio)
    verdict["gain_meaningful"] = gain >= float(min_loss_gain)
    verdict["cost_acceptable"] = cost <= float(max_cost_ratio)
    verdict["viable"] = bool(verdict["gain_meaningful"] and verdict["cost_acceptable"])
    if gain > 0:
        verdict["cost_per_loss_point"] = cost / gain
    reasons = []
    if not verdict["gain_meaningful"]:
        reasons.append(
            f"loss gain {gain:+.4f} is below the practical margin {min_loss_gain:g}"
        )
    if not verdict["cost_acceptable"]:
        reasons.append(
            f"cost ratio {cost:.3f}x exceeds the allowed maximum {max_cost_ratio:.3f}x"
        )
    verdict["reason"] = "; ".join(reasons) if reasons else (
        f"gain {gain:+.4f} clears {min_loss_gain:g} at cost {cost:.3f}x within {max_cost_ratio:.3f}x"
    )
    return verdict
