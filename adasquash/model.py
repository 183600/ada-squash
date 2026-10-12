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

from .norms import build_norm, NORM_REGISTRY
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
    use_layerscale: bool = True
    layerscale_init: float = 1e-5
    # DyT-family specific
    alpha_0: float = 0.9
    k_init: float = 1.0
    alpha_init: float = 0.5  # for plain DyT and DyT-C
    p_init: float = 2.0
    learnable_p: bool = True
    learnable_k: bool = True
    # Misc
    pad_token_id: int = 0
    tie_word_embeddings: bool = True
    rotary: bool = False


# ---------------------------------------------------------------------------
# Norm-construction helpers
# ---------------------------------------------------------------------------
def build_norm_kwargs(cfg: "GPTConfig", layer_idx: int = 0) -> dict:
    """Collect the kwargs a given `norm_name` accepts from a GPTConfig.

    The result is passed to `build_norm`, which forwards it to the class
    constructor; norms that do not declare one of these kwargs (LN, RMSNorm,
    plain DyT) simply ignore the extras only when they are not supplied, so
    each family gets exactly the subset it supports.

    Using one helper for the blocks *and* the final norm guarantees that a
    configured hyper-parameter (alpha_init / alpha_0 / k_init / p_init) is
    honoured everywhere, instead of silently reverting to the class default
    on `norm_f`.
    """
    kwargs: dict = {}
    if cfg.norm_name in ("dytd", "dytcd"):
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
        )
    if cfg.norm_name == "dyt":
        kwargs.update(alpha_init=cfg.alpha_init)
    return kwargs


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------
_CROSSENTROPY_CHUNK_BYTES = 1 << 26  # ~64MB of fp32 logits per chunk


def _cross_entropy_chunk_rows(vocab_size: int) -> int:
    """Rows of the [N, V] logit block handled per chunk."""
    row_bytes = max(1, vocab_size) * 4
    return max(1, _CROSSENTROPY_CHUNK_BYTES // row_bytes)


class _ChunkedCrossEntropy(torch.autograd.Function):
    """
    Mean cross-entropy over a [N, V] logit block, evaluated in row chunks.

    The per-row math is identical to `F.cross_entropy`; only the
    reduction over rows is split. The [N, V] log-probabilities that the
    fused kernel would otherwise have to keep alive for backward are
    dropped chunk by chunk and the gradient is rebuilt from a single
    softmax per chunk, so the step never holds more than one chunk's
    worth of extra logits at a time.

    The backward pass overwrites the saved logit block in place with the
    gradient (see `backward`), so the graph must not be traversed twice
    with `retain_graph=True`.
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
        # The saved logit block doubles as the gradient buffer: each chunk
        # is fully reduced into `probs` before its slice is overwritten, and
        # no other node needs the original logit values (the producing
        # matmul consumes only its inputs and this gradient vector). That
        # keeps the step from ever holding a second [N, V] tensor.
        grad_logits = logits
        grad_scale = (grad_out / ctx.count).to(logits.dtype)
        chunk_rows = ctx.chunk_rows
        for start in range(0, targets.numel(), chunk_rows):
            end = min(start + chunk_rows, targets.numel())
            chunk_logits = logits[start:end]
            chunk_targets = targets[start:end]
            valid = chunk_targets.ne(ctx.ignore_index)
            probs = torch.softmax(chunk_logits, dim=-1)
            # subtract the one-hot at the gold column (ignored rows are
            # remapped to column 0 first, then zeroed out below)
            gold = chunk_targets.masked_fill(~valid, 0).unsqueeze(-1)
            probs.scatter_add_(-1, gold, torch.full_like(gold, -1.0, dtype=probs.dtype))
            probs.mul_(grad_scale)
            probs.masked_fill_(~valid.unsqueeze(-1), 0.0)
            grad_logits[start:end] = probs
        return grad_logits, None, None, None


def cross_entropy_mean(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    chunk_rows: Optional[int] = None,
) -> torch.Tensor:
    """Mean cross-entropy with a bounded log-probability footprint."""
    if chunk_rows is None:
        chunk_rows = _cross_entropy_chunk_rows(logits.size(-1))
    if logits.size(0) <= chunk_rows:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)
    return _ChunkedCrossEntropy.apply(
        logits, targets, int(ignore_index), int(chunk_rows)
    )


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
        norm_kwargs = build_norm_kwargs(cfg, layer_idx)

        self.norm1 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.attn = CausalSelfAttention(
            cfg.dim, cfg.n_heads,
            qk_norm=cfg.qk_norm,
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
            attn_out.mul_(self.ls1)
        attn_out.add_(x)
        x = attn_out

        h = self.norm2(x)
        mlp_out = self.mlp(h)
        if self.use_layerscale:
            mlp_out.mul_(self.ls2)
        mlp_out.add_(x)
        x = mlp_out
        return x, logit_stats


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
        # swiglu
        h1 = self.w1(x)
        h3 = self.w3(x)
        h = F.silu(h1)
        h.mul_(h3)
        return self.drop(self.w2(h))


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
        self.norm_f = build_norm(cfg.norm_name, cfg.dim, **build_norm_kwargs(cfg, cfg.n_layers - 1))
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
        self.register_buffer(
            "_pos_ids", torch.arange(cfg.max_seq_len), persistent=False
        )

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
    ):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = self._pos_ids[:T].unsqueeze(0)
            x.add_(self.pos_emb(pos))
        x = self.drop(x)

        freqs = None
        if self.freqs_cis is not None:
            freqs = self.freqs_cis[:T]

        all_logit_stats = []
        for block in self.blocks:
            x, ls = block(x, freqs_cis=freqs, return_stats=return_stats)
            if return_stats:
                all_logit_stats.append(ls)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            loss = cross_entropy_mean(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.cfg.pad_token_id,
            )

        if not return_stats:
            return logits, loss
        return logits, loss, all_logit_stats

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
            # Pick out the top-level normalization modules.
            if name.endswith((".norm1", ".norm2", ".norm_f")):
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
    cfg = GPTConfig(
        vocab_size=32000, dim=2048, n_layers=24, n_heads=16,
        max_seq_len=2048, mlp_ratio=8.0 / 3.0,
        norm_name=norm_name, rotary=True,
        tie_word_embeddings=False,
        **over,
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
    "build_gpt_124m",
    "build_gpt_350m",
    "build_llama_tiny_1b",
]
