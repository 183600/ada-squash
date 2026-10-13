"""
GPT-style decoder-only Transformer with configurable normalization,
depth-conditioned LayerScale, and QK-squashable attention.

The model is deliberately compact (124M to 1.1B params) and dependency-free
so that the *normalization strategy* is the only experimental variable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import build_norm, NORM_REGISTRY, _sum_leading_dims
from .attention import CausalSelfAttention


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class LayerConfig:
    """Per-block configuration. Allows different norm names per layer if
    a future ablation needs that."""
    norm: str = "dytcd"  # one of NORM_REGISTRY
    qk_norm: Optional[str] = None  # None | "qknorm" | "dytqk"
    use_layerscale: bool = True
    layerscale_init: float = 1e-5  # tiny init for stability


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_seq_len: int = 1024
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Norm & attention
    norm_name: str = "dytcd"
    qk_norm: Optional[str] = None
    # Initial alpha_q / alpha_k for `qk_norm="dytqk"`. Threaded through here
    # so the squashing strength of variant C is configurable.
    qk_alpha_init: float = 1.0
    use_layerscale: bool = True
    layerscale_init: float = 1e-5
    # DyT-family specific
    alpha_0: float = 0.9
    k_init: float = 1.0
    alpha_init: float = 0.5  # for plain DyT and DyT-C
    p_init: float = 2.0
    learnable_p: bool = True
    learnable_k: bool = True
    beta: bool = True  # output affine bias for DyT-family norms
    alpha_jitter: float = 0.05  # per-channel alpha init jitter for DyTC
    gradient_checkpointing: bool = False
    # Misc
    pad_token_id: int = 0
    # Label id excluded from the loss. MUST be outside [0, vocab_size) so it
    # can never collide with a real token: with the GPT-2 vocabulary id 0 is
    # "!", an ordinary token that appears in ordinary prose. Packed streaming
    # corpora contain no padding at all, so the training pipeline leaves
    # nothing to mask and this value should never appear.
    ignore_index: int = -100
    tie_word_embeddings: bool = True
    rotary: bool = False


# ---------------------------------------------------------------------------
# Norm-construction helpers
# ---------------------------------------------------------------------------
_CROSSENTROPY_CHUNK_BYTES = 1 << 26
_CROSS_ENTROPY_CHUNK_ROWS_CACHE: dict[int, int] = {}


def _cross_entropy_chunk_rows(vocab_size: int) -> int:
    """Rows of the [N, V] logit block handled per chunk."""
    if vocab_size not in _CROSS_ENTROPY_CHUNK_ROWS_CACHE:
        row_bytes = max(1, vocab_size) * 4
        _CROSS_ENTROPY_CHUNK_ROWS_CACHE[vocab_size] = max(1, _CROSSENTROPY_CHUNK_BYTES // row_bytes)
    return _CROSS_ENTROPY_CHUNK_ROWS_CACHE[vocab_size]


def build_norm_kwargs(cfg: "GPTConfig", layer_idx: int = 0, is_final_norm: bool = False) -> dict:
    """Collect the kwargs a given `norm_name` accepts from a GPTConfig.

    The result is passed to `build_norm`, which forwards it to the class
    constructor; norms that do not declare one of these kwargs (LN, RMSNorm,
    plain DyT) simply ignore the extras only when they are not supplied, so
    each family gets exactly the subset it supports.

    Using one helper for the blocks *and* the final norm guarantees that a
    configured hyper-parameter (alpha_init / alpha_0 / k_init / p_init) is
    applied to every norm of the family that accepts it.

    This deliberately reads `cfg` afresh on every call and keeps no
    module-level state, so that each model in a multi-model process gets
    its own correct configuration.

    Args:
        cfg: Model configuration
        layer_idx: Layer index for depth-conditioned norms
        is_final_norm: If True, this is the final norm (after all blocks).
                       For depth-conditioned norms, uses depth_ratio=1.0.
    """
    kwargs: dict = {}
    is_dyt_family = cfg.norm_name in ("dyt", "dytd", "dytc", "dytcd")
    if cfg.norm_name in ("dytd", "dytcd"):
        if is_final_norm:
            # Final norm sits after all layers; use depth_ratio = 1.0
            depth_ratio = 1.0
            layer_idx = cfg.n_layers  # for logging consistency
        else:
            depth_ratio = float(layer_idx) / float(cfg.n_layers)
        kwargs = dict(
            layer_idx=layer_idx,
            num_layers=cfg.n_layers,
            alpha_0=cfg.alpha_0,
            k_init=cfg.k_init,
            learnable_k=cfg.learnable_k,
        )
    if cfg.norm_name == "dytcd":
        kwargs.update(p_init=cfg.p_init, learnable_p=cfg.learnable_p)
    if cfg.norm_name == "dytc":
        kwargs.update(
            p_init=cfg.p_init, learnable_p=cfg.learnable_p,
            alpha_init=cfg.alpha_init,
            alpha_jitter=cfg.alpha_jitter,
        )
    if cfg.norm_name == "dyt":
        kwargs.update(alpha_init=cfg.alpha_init)
    if is_dyt_family and hasattr(cfg, 'beta') and cfg.beta is not None:
        kwargs.update(beta=cfg.beta)
    return kwargs


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
class _ChunkedCrossEntropy(torch.autograd.Function):
    """
    Mean cross-entropy over a [N, V] logit block, evaluated in row chunks.

    The per-row math is identical to `F.cross_entropy`; only the
    reduction over rows is split.

    The backward writes into a private gradient buffer rather than reusing
    the saved logit block: `GPT.forward(..., return_logits=True)` hands the
    very same tensor back to its caller, so overwriting it in place would
    leave that caller holding gradient values stamped over its logits after
    `loss.backward()`. Within that buffer the probabilities are built
    directly, so no second `[chunk, V]` block is ever live.
    """

    @staticmethod
    def forward(ctx, logits, targets, ignore_index, chunk_rows):
        total = None
        for start in range(0, targets.numel(), chunk_rows):
            chunk_loss = F.cross_entropy(
                logits[start:start + chunk_rows],
                targets[start:start + chunk_rows],
                ignore_index=ignore_index,
                reduction="sum",
            )
            total = chunk_loss if total is None else total + chunk_loss
        count = targets.ne(ignore_index).sum()
        ctx.save_for_backward(logits, targets)
        ctx.ignore_index = ignore_index
        ctx.chunk_rows = chunk_rows
        ctx.count = count
        return total / count

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        # A buffer of our own: `logits` belongs to whoever produced it and
        # may still be reading it (see the class docstring).
        grad_logits = torch.empty_like(logits)
        grad_scale = (grad_out / ctx.count).to(logits.dtype)
        chunk_rows = ctx.chunk_rows
        for start in range(0, targets.numel(), chunk_rows):
            end = min(start + chunk_rows, targets.numel())
            chunk_targets = targets[start:end]
            # The softmax lands straight in the gradient slice it belongs in:
            # a separate probability buffer plus a copy into `grad_logits`
            # would be two extra passes over the largest block in the model.
            probs = grad_logits[start:end]
            torch.softmax(logits[start:end], dim=-1, out=probs)
            valid = chunk_targets.ne(ctx.ignore_index)
            # subtract the one-hot at the gold column (ignored rows are
            # remapped to column 0 first, then scaled to zero below)
            gold = chunk_targets.masked_fill(~valid, 0).unsqueeze(-1)
            keep = valid.unsqueeze(-1).to(probs.dtype)
            # Per-row scatter source and per-row scale: a masked row adds 0
            # instead of -1 and is then multiplied by 0, which zeroes it
            # without a second full-size `masked_fill_` pass. The scale is
            # applied *after* the subtraction so the row reads
            # (softmax - onehot) * grad_out / count, as autograd does.
            probs.scatter_add_(-1, gold, keep.neg())
            probs.mul_(keep.mul_(grad_scale))
        return grad_logits, None, None, None


def cross_entropy_mean(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_rows: Optional[int] = None,
) -> torch.Tensor:
    """Mean cross-entropy over a [N, V] logit block, evaluated in row chunks."""
    if chunk_rows is None:
        chunk_rows = _cross_entropy_chunk_rows(logits.size(-1))
    if logits.size(0) <= chunk_rows:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)
    return _ChunkedCrossEntropy.apply(
        logits, targets, int(ignore_index), int(chunk_rows)
    )


def _loss_acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype for the loss / softmax: at least fp32."""
    return torch.promote_types(dtype, torch.float32)


def _autocast_dtype(device_type: str, *dtypes: torch.dtype) -> torch.dtype:
    """The dtype a projection over these operands would actually run in."""
    fallback = dtypes[0]
    for dt in dtypes[1:]:
        fallback = torch.promote_types(fallback, dt)
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except TypeError:
        # torch < 2.4 only knows the CUDA spelling.
        if device_type == "cuda" and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return fallback


class _FusedLinearCrossEntropy(torch.autograd.Function):
    """
    Mean cross-entropy of `hidden @ weight.T` against `targets`, fused with
    the projection that produces the logits.

    The forward builds one row chunk of logits at a time, and the backward
    re-derives each chunk from the `[N, C]` hidden states that the rest of
    the graph already keeps. See `fused_cross_entropy_mean`.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index, chunk_rows):
        # Cast once, up front: the projections then run on the tensors that
        # are also the ones the backward replays.
        dt = _autocast_dtype(hidden.device.type, hidden.dtype, weight.dtype)
        h = hidden.to(dt)
        w = weight.to(dt)
        acc = _loss_acc_dtype(dt)
        total = None
        for start in range(0, targets.numel(), chunk_rows):
            end = min(start + chunk_rows, targets.numel())
            # The chunk's logits are this node's own GEMM output - the cast
            # allocates a private copy whenever it changes the dtype - so the
            # log-softmax runs in place on them instead of adding a second
            # `[chunk, V]` block to the peak.
            chunk_logits = F.linear(h[start:end], w).to(acc)
            torch.log_softmax(chunk_logits, -1, out=chunk_logits)
            # `log_softmax` + `nll_loss` is what `F.cross_entropy` runs; the
            # pair is spelled out only so the first half can write in place.
            chunk = F.nll_loss(
                chunk_logits, targets[start:end],
                ignore_index=ignore_index, reduction="sum",
            )
            total = chunk if total is None else total + chunk
        count = targets.ne(ignore_index).sum()
        ctx.save_for_backward(h, w, targets)
        ctx.ignore_index = ignore_index
        ctx.chunk_rows = chunk_rows
        ctx.count = count
        ctx.compute_dtype = dt
        ctx.in_dtypes = (hidden.dtype, weight.dtype)
        return total / count

    @staticmethod
    def backward(ctx, grad_out):
        h, w, targets = ctx.saved_tensors
        # The backward runs outside autocast; the operands were already cast
        # in the forward, so the chunk projection replays bit for bit.
        dt = ctx.compute_dtype
        acc = _loss_acc_dtype(dt)
        need_h = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        # `empty_like`, not `zeros_like`: every slice of `grad_h` and the
        # first chunk of `grad_w` are written directly.
        grad_h = torch.empty_like(h) if need_h else None
        grad_w = torch.empty_like(w) if need_w else None
        grad_scale = (grad_out / ctx.count).to(dt)
        chunk_rows = ctx.chunk_rows
        first = True
        for start in range(0, targets.numel(), chunk_rows):
            end = min(start + chunk_rows, targets.numel())
            hc = h[start:end]
            tc = targets[start:end]
            # Same in-place log-softmax as the forward: the recomputed chunk
            # is the only owner of its logits, and folding the softmax into
            # that buffer halves the block this loop keeps live.
            probs = F.linear(hc, w).to(acc)
            torch.softmax(probs, -1, out=probs)
            valid = tc.ne(ctx.ignore_index)
            # subtract the one-hot at the gold column (ignored rows are
            # remapped to column 0 first, then scaled to zero below)
            gold = tc.masked_fill(~valid, 0).unsqueeze(-1)
            keep = valid.unsqueeze(-1).to(probs.dtype)
            # Per-row scatter source and per-row scale: a masked row adds 0
            # instead of -1 and is then multiplied by 0, which zeroes it
            # without a second full-size `masked_fill_` pass. The scale is
            # applied *after* the subtraction, so a scored row reads
            # (softmax - onehot) * grad_out / count, as autograd does.
            probs.scatter_add_(-1, gold, keep.neg())
            probs.mul_(keep.mul_(grad_scale))
            g = probs.to(dt)
            if need_h:
                # Each slice is written, not accumulated: `grad_h` is filled
                # one row block at a time and `grad_scale` is already folded
                # into `g`, so there is nothing to add to.
                torch.mm(g, w, out=grad_h[start:end])
            if need_w:
                # `grad_w` on the other hand is accumulated over every chunk,
                # so only its first write has to skip the addition.
                if first:
                    torch.mm(g.t(), hc, out=grad_w)
                else:
                    grad_w.addmm_(g.t(), hc)
            first = False
        hidden_dtype, weight_dtype = ctx.in_dtypes
        if need_h and grad_h.dtype != hidden_dtype:
            grad_h = grad_h.to(hidden_dtype)
        if need_w and grad_w.dtype != weight_dtype:
            grad_w = grad_w.to(weight_dtype)
        return grad_h, grad_w, None, None, None


def fused_cross_entropy_mean(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_rows: Optional[int] = None,
) -> torch.Tensor:
    """
    Mean cross-entropy of the `[N, V]` logits implied by `hidden @ weight.T`.

    Equivalent to `cross_entropy_mean(F.linear(hidden, weight), targets)`,
    including the gradient w.r.t. both `hidden` and `weight`, but without
    ever holding the full logit block. `hidden` is `[..., C]`; `targets` is
    the matching flat `[N]` block.
    """
    n = targets.numel()
    if hidden.dim() != 2 or hidden.size(0) != n:
        hidden = hidden.reshape(n, -1)
    if chunk_rows is None:
        chunk_rows = _cross_entropy_chunk_rows(weight.size(0))
    if n <= chunk_rows:
        return F.cross_entropy(
            F.linear(hidden, weight), targets, ignore_index=ignore_index
        )
    return _FusedLinearCrossEntropy.apply(
        hidden, weight, targets, int(ignore_index), int(chunk_rows)
    )


class _LayerScaleResidual(torch.autograd.Function):
    """
    Fused LayerScale + residual add: x + scale * residual
    """
    @staticmethod
    def forward(ctx, x, residual, scale):
        # x: [B, T, C], residual: [B, T, C], scale: [C]
        ctx.save_for_backward(residual, scale)
        return torch.addcmul(x, residual, scale)

    @staticmethod
    def backward(ctx, grad_out):
        residual, scale = ctx.saved_tensors
        # grad_x = grad_out
        # grad_residual = grad_out * scale
        # grad_scale = (grad_out * residual).sum(dim=(0, 1))
        # `_sum_leading_dims` rather than a bare `sum(dim=red_dims)`: for a
        # 1-D residual the empty dim tuple would reduce the `[C]` gradient
        # to a scalar and autograd would reject it (see the helper).
        grad_scale = _sum_leading_dims(grad_out * residual)
        grad_residual = grad_out * scale
        return grad_out, grad_residual, grad_scale


class Block(nn.Module):
    """
    Pre-norm Transformer block (norm-dependent residual).

    For DyT-D / DyT-CD the residual path uses a LayerScale-style gate
        x = x + lambda_l * attn_residual
        x = x + lambda_l * mlp_residual
    so that the depth-conditioned squashing inside the norm interacts
    cleanly with the residual stream.
    """

    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # Note: norm_kwargs only used by DyT-D / DyT-CD; other norms ignore
        # the extra kwargs silently.
        norm_kwargs = build_norm_kwargs(cfg, layer_idx, is_final_norm=False)

        self.norm1 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.attn = CausalSelfAttention(
            cfg.dim, cfg.n_heads,
            qk_norm=cfg.qk_norm,
            alpha_init=cfg.qk_alpha_init,
            dropout=cfg.dropout,
        )
        self.norm2 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.mlp = MLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio), dropout=cfg.dropout)

        # LayerScale - separate gammas for attn and mlp
        self.use_layerscale = cfg.use_layerscale
        if self.use_layerscale:
            self.ls1 = nn.Parameter(
                torch.full((cfg.dim,), cfg.layerscale_init)
            )
            self.ls2 = nn.Parameter(
                torch.full((cfg.dim,), cfg.layerscale_init)
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        # Pre-norm style.
        h = self.norm1(x)
        attn_out, logit_stats = self.attn(
            h, freqs_cis=freqs_cis, return_stats=return_stats
        )
        if self.use_layerscale:
            x = _LayerScaleResidual.apply(x, attn_out, self.ls1)
        else:
            x = x + attn_out

        h = self.norm2(x)
        mlp_out = self.mlp(h)
        if self.use_layerscale:
            x = _LayerScaleResidual.apply(x, mlp_out, self.ls2)
        else:
            x = x + mlp_out
        return x, logit_stats


class _SwiGLU(torch.autograd.Function):
    """
    w2(silu(w1 x) * (w3 x)) handled as a single autograd node.

    Fused forward + backward. Only saves x, a, b, w1, w2, w3 (weights are
    shared). The backward recomputes silu(a) and the gated product
    analytically.

    `a` and `b` are the irreducible part: `grad_a` needs `a` for the SiLU
    derivative and `grad_b` needs `silu(a)`, so neither can be dropped.
    """

    @staticmethod
    def forward(ctx, x, w1, w2, w3):
        # The projections run on a flattened [N, C] view of the activation, so
        # every reduction below is a 2-D GEMM. `reshape` is a view for the
        # contiguous [B, T, C] the block always hands in.
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        a = F.linear(x2, w1)
        b = F.linear(x2, w3)
        h = F.silu(a)
        h.mul_(b)
        y = F.linear(h, w2)
        # Save only what's needed: x, a, b for backward, weights for grad
        ctx.save_for_backward(x2, a, b, w1, w2, w3)
        ctx.compute_dtype = y.dtype
        ctx.in_dtypes = (x.dtype, w1.dtype, w2.dtype, w3.dtype)
        ctx.lead = lead
        return y.view(*lead, -1)

    @staticmethod
    def backward(ctx, grad_out):
        x, a, b, w1, w2, w3 = ctx.saved_tensors
        dt = ctx.compute_dtype
        if w1.dtype != dt:
            w1, w2, w3 = w1.to(dt), w2.to(dt), w3.to(dt)
        if x.dtype != dt:
            x = x.to(dt)

        # grad_h = grad_out @ w2^T  [N, H]
        grad_out = grad_out.reshape(-1, grad_out.shape[-1])
        grad_h = F.linear(grad_out, w2.t())

        # The three hidden-sized buffers this pass needs are kept to a
        # minimum by the order below: `prod` is dropped before the last one
        # is allocated, and `sa` is folded into both of its consumers so
        # that `grad_a`, the gated product and one of {grad_b, sa} are the
        # only hidden-sized tensors alive at any point.
        #
        # grad_a = grad_h * b * silu'(a) = silu_backward(grad_h * b, a).
        # `torch.ops.aten.silu_backward` computes the SiLU derivative.
        prod = grad_h * b
        grad_a = torch.ops.aten.silu_backward(prod, a)
        del prod

        # This is h from the forward; only grad_w2 reads it.
        sa = F.silu(a)
        # grad_b = grad_h * silu(a); the product takes over grad_h's buffer.
        grad_b = grad_h.mul_(sa)
        # grad_a needs `a` for the SiLU derivative and grad_b needs
        # `silu(a)`, so neither of those two can be dropped.
        gated = sa.mul_(b)
        del sa

        grad_x = grad_w1 = grad_w2 = grad_w3 = None
        if ctx.needs_input_grad[0]:
            grad_x = F.linear(grad_a, w1.t())
            grad_x.addmm_(grad_b, w3)
        if ctx.needs_input_grad[1]:
            grad_w1 = grad_a.t() @ x
        if ctx.needs_input_grad[2]:
            grad_w2 = grad_out.t() @ gated
        if ctx.needs_input_grad[3]:
            grad_w3 = grad_b.t() @ x

        xd, w1d, w2d, w3d = ctx.in_dtypes
        if grad_x is not None:
            if grad_x.dtype != xd:
                grad_x = grad_x.to(xd)
            grad_x = grad_x.view(*ctx.lead, -1)
        if grad_w1 is not None and grad_w1.dtype != w1d:
            grad_w1 = grad_w1.to(w1d)
        if grad_w2 is not None and grad_w2.dtype != w2d:
            grad_w2 = grad_w2.to(w2d)
        if grad_w3 is not None and grad_w3.dtype != w3d:
            grad_w3 = grad_w3.to(w3d)
        return grad_x, grad_w1, grad_w2, grad_w3


class MLP(nn.Module):
    """SwiGLU MLP. Hidden dim is 2/3 * (ratio * dim)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU: w2(silu(w1 x) * w3 x)
        hidden = int(hidden * 2 / 3)
        hidden = (hidden + 7) & ~7
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(
            _SwiGLU.apply(x, self.w1.weight, self.w2.weight, self.w3.weight)
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim) if not cfg.rotary else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [Block(cfg, i) for i in range(cfg.n_layers)]
        )
        self.norm_f = build_norm(cfg.norm_name, cfg.dim, **build_norm_kwargs(cfg, cfg.n_layers, is_final_norm=True))
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        # init
        self.apply(self._init_weights)

        # RoPE freqs buffer
        if cfg.rotary:
            head_dim = cfg.dim // cfg.n_heads
            freqs = 1.0 / (
                10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
            )
            t = torch.arange(cfg.max_seq_len).float()
            freqs = torch.outer(t, freqs)  # [T, hd//2]
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.freqs_cis = None

        # Pre-computed position ids for non-rotary mode
        if not cfg.rotary:
            self.register_buffer(
                "_pos_ids", torch.arange(cfg.max_seq_len), persistent=False
            )
        else:
            self._pos_ids = None

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def reset_parameters(self, seed: Optional[int] = None):
        """Re-run (re-)initialization of the shared Linear / Embedding
        weights from a fixed seed. Norm-owned parameters (alpha, delta,
        gamma, ...) are left untouched.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.apply(self._init_weights)
        return self

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        last_position_only: bool = False,
        return_logits: bool = True,
    ):
        """
        Args:
            input_ids:          [B, T]
            targets:            [B, T] next-token ids; `None` skips the loss
            return_stats:       collect per-layer attention logit statistics
            last_position_only: score only the final position
            return_logits:      return the [B, T, V] logit block. Set it
                                to False on the training path: the loss is
                                then fused with the output projection and the
                                returned logits are `None`.
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            x.add_(self.pos_emb(self._pos_ids[:T].unsqueeze(0)))
        x = self.drop(x)

        freqs = self.freqs_cis[:T] if self.freqs_cis is not None else None

        all_logit_stats = []
        if self.cfg.gradient_checkpointing and self.training:
            # Gradient checkpointing: recompute block activations during backward
            for block in self.blocks:
                x, ls = torch.utils.checkpoint.checkpoint(
                    block, x, freqs, return_stats, use_reentrant=False
                )
                if return_stats:
                    all_logit_stats.append(ls)
        else:
            for block in self.blocks:
                x, ls = block(x, freqs_cis=freqs, return_stats=return_stats)
                if return_stats:
                    all_logit_stats.append(ls)
        x = self.norm_f(x)
        if last_position_only:
            # Decoding / scoring only the final position never needs the full
            # [B, T, V] logit block.
            x = x[:, -1:]

        logits = None
        loss = None
        if targets is None:
            if return_logits:
                logits = self.lm_head(x)
        else:
            flat = (targets[:, -1:] if last_position_only else targets).reshape(-1)
            if return_logits:
                logits = self.lm_head(x)
                loss = cross_entropy_mean(
                    logits.view(-1, logits.size(-1)),
                    flat,
                    ignore_index=self.cfg.ignore_index,
                )
            else:
                loss = fused_cross_entropy_mean(
                    x.reshape(-1, x.size(-1)),
                    self.lm_head.weight,
                    flat,
                    ignore_index=self.cfg.ignore_index,
                )

        if not return_stats:
            return logits, loss
        return logits, loss, all_logit_stats

    def last_position_logits(
        self, input_ids: torch.Tensor, return_stats: bool = False
    ) -> torch.Tensor:
        """Logits for the final position only, shaped `[B, V]`."""
        out = self.forward(
            input_ids, targets=None, return_stats=return_stats,
            last_position_only=True,
        )
        return out[0][:, -1]

    @torch.no_grad()
    def num_params(self, trainable_only: bool = False) -> int:
        s = 0
        for p in self.parameters():
            if trainable_only and not p.requires_grad:
                continue
            s += p.numel()
        return s

    @torch.no_grad()
    def norm_param_count(self) -> dict:
        """Return per-norm parameter count."""
        out = {}
        for name, m in self.named_modules():
            # Pick out the top-level normalization modules. `norm_f` is a
            # direct child of GPT, so its name carries no `blocks.<i>.` prefix
            # - matching it with the leading-dot pattern below would drop the
            # final norm from the accounting.
            if name == "norm_f" or name.endswith((".norm1", ".norm2")):
                cls = m.__class__.__name__
                out.setdefault(cls, 0)
                for p in m.parameters(recurse=False):
                    out[cls] += p.numel()
        return out


# ---------------------------------------------------------------------------
# Convenience builders for the experiment matrix
# ---------------------------------------------------------------------------
def build_gpt_124m(norm_name: str = "dytcd", **over) -> GPT:
    """GPT-2 small (124M). dim=768, L=12, H=12."""
    cfg = GPTConfig(
        vocab_size=50304, dim=768, n_layers=12, n_heads=12,
        max_seq_len=1024, mlp_ratio=4.0,
        norm_name=norm_name, **over,
    )
    return GPT(cfg)


def build_gpt_350m(norm_name: str = "dytcd", **over) -> GPT:
    """GPT-2 medium (350M). dim=1024, L=24, H=16."""
    cfg = GPTConfig(
        vocab_size=50304, dim=1024, n_layers=24, n_heads=16,
        max_seq_len=1024, mlp_ratio=4.0,
        norm_name=norm_name, **over,
    )
    return GPT(cfg)


def build_llama_tiny_1b(norm_name: str = "dytcd", **over) -> GPT:
    """LLaMA-tiny ~1.1B params, matches the xIELU / LLaMA-tiny recipe."""
    # `setdefault` so an explicit `tie_word_embeddings` in `over` takes
    # precedence over the LLaMA-tiny default.
    over.setdefault("tie_word_embeddings", False)
    over.setdefault("rotary", True)
    cfg = GPTConfig(
        vocab_size=32000, dim=2048, n_layers=24, n_heads=16,
        max_seq_len=2048, mlp_ratio=8.0 / 3.0,
        norm_name=norm_name, **over,
    )
    return GPT(cfg)


__all__ = [
    "GPTConfig",
    "LayerConfig",
    "Block",
    "MLP",
    "GPT",
    "build_norm_kwargs",
    "cross_entropy_mean",
    "fused_cross_entropy_mean",
    "build_gpt_124m",
    "build_gpt_350m",
    "build_llama_tiny_1b",
]
