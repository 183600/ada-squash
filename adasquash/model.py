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
# Block
# ---------------------------------------------------------------------------
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
        norm_kwargs = {}
        if cfg.norm_name in ("dytd", "dytcd"):
            norm_kwargs = dict(
                layer_idx=layer_idx,
                num_layers=cfg.n_layers,
                alpha_0=cfg.alpha_0,
                k_init=cfg.k_init,
                learnable_k=cfg.learnable_k,
            )
        if cfg.norm_name == "dytcd":
            norm_kwargs.update(
                p_init=cfg.p_init, learnable_p=cfg.learnable_p,
            )
        if cfg.norm_name == "dytc":
            norm_kwargs.update(
                p_init=cfg.p_init, learnable_p=cfg.learnable_p,
                alpha_init=cfg.alpha_init,
            )
        if cfg.norm_name == "dyt":
            norm_kwargs.update(dict(alpha_init=cfg.alpha_init))

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
        # Pre-norm style
        h = self.norm1(x)
        attn_out, logit_stats = self.attn(
            h, freqs_cis=freqs_cis, return_stats=return_stats
        )
        if self.use_layerscale:
            x = x + self.ls1 * attn_out
        else:
            x = x + attn_out

        h = self.norm2(x)
        mlp_out = self.mlp(h)
        if self.use_layerscale:
            x = x + self.ls2 * mlp_out
        else:
            x = x + mlp_out
        return x, logit_stats


class MLP(nn.Module):
    """SwiGLU MLP. Hidden dim is 2/3 * (ratio * dim)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU: w2(silu(w1 x) * w3 x)
        hidden = int(hidden * 2 / 3)
        # round up to multiple of 8 for fp8 / TensorCore friendliness
        hidden = (hidden + 7) & ~7
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # swiglu: h1 and h3 are fresh intermediates, so the SiLU can be
        # applied in-place (saves one [.., hidden] activation buffer).
        h1 = self.w1(x)
        h3 = self.w3(x)
        return self.drop(self.w2(F.silu(h1, inplace=True) * h3))


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
        self.norm_f = build_norm(cfg.norm_name, cfg.dim)
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

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
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
            loss = F.cross_entropy(
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
        """Return per-norm parameter count, useful for the 'free-ness' story."""
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
    "build_gpt_124m",
    "build_gpt_350m",
    "build_llama_tiny_1b",
]
