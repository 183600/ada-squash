"""
Causal self-attention with optional QK-squashing (DyT++ variant C).

The attention implementation is deliberately simple and dependency-free
so that the *normalization* layer's effect on logit magnitudes and
gradient flow is the only variable being measured.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import QKSquash


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Args:
        dim:         embedding dim
        n_heads:     number of heads
        qk_norm:     None | "qknorm" (RMSNorm on Q and K) | "dytqk" (squash)
        qk_norm_dim: per-head dimension if qk_norm is set (default: head_dim)
        alpha_init:  initial scalar for QKSquash when qk_norm="dytqk"
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        qk_norm: Optional[str] = None,
        alpha_init: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        self.qk_norm = qk_norm
        self._rope_cache: Optional[tuple] = None
        self._tril_cache: Optional[tuple] = None

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        if qk_norm is None:
            self.q_squash = None
            self.k_squash = None
            self.q_rms = None
            self.k_rms = None
        elif qk_norm == "qknorm":
            self.q_squash = None
            self.k_squash = None
            self.q_rms = RMSNormHead(self.head_dim)
            self.k_rms = RMSNormHead(self.head_dim)
        elif qk_norm == "dytqk":
            self.q_squash = QKSquash(alpha_q_init=alpha_init, alpha_k_init=alpha_init)
            self.k_squash = self.q_squash  # share scalar pair
            self.q_rms = None
            self.k_rms = None
        else:
            raise ValueError(f"Unknown qk_norm='{qk_norm}'")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        """
        Args
            x:            [B, T, C]
            freqs_cis:    optional RoPE freqs [T, head_dim//2, 2]  (complex)
            return_stats: if True, compute logit statistics for monitoring
        Returns
            y:         [B, T, C]
            logit_stats: dict of logit statistics (empty unless requested)
        """
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.q_rms is not None:
            q = self.q_rms(q)
            k = self.k_rms(k)
        elif self.q_squash is not None:
            q, k = self.q_squash(q, k)

        if freqs_cis is not None:
            cos, sin = self._rope_tables(freqs_cis)
            q, k = apply_rope(q, k, freqs_cis, cos, sin)

        scale = 1.0 / math.sqrt(self.head_dim)

        # logit monitoring (useful for the QK-squash ablation). Monitoring
        # only; disabled by default.
        stats = {}
        if return_stats:
            with torch.no_grad():
                logits = torch.matmul(q, k.transpose(-1, -2))
                logits.mul_(scale)
                # Only the causal triangle is ever read, so it is selected
                # directly rather than masking the upper triangle with -inf.
                # `masked_select` (not `logits[mask]`): the mask is [T, T]
                # while the logits are [B, H, T, T], and boolean *indexing*
                # would apply it to the leading two dims instead of the last
                # two. `masked_select` broadcasts, which is what is meant here.
                keep = self._tril_mask(T, x.device)
                valid = logits.masked_select(keep)
                if valid.numel() > 1:
                    abs_logits = valid.abs()
                    logit_abs_mean = abs_logits.mean().item()
                    logit_abs_max = abs_logits.amax().item()
                    logit_std = valid.std().item()
                else:
                    logit_abs_mean = logit_abs_max = logit_std = 0.0
            stats = {
                "logit_abs_mean": logit_abs_mean,
                "logit_abs_max": logit_abs_max,
                "logit_std": logit_std,
            }

        # is_causal=True is mathematically equivalent to the manual triu
        # mask.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=scale,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj_drop(self.proj(out))
        return out, stats

    def _tril_mask(self, T: int, device) -> torch.Tensor:
        """Lower-triangular boolean mask `[T, T]`, cached per (T, device)."""
        cache = self._tril_cache
        if cache is None or cache[0] != (T, device):
            mask = torch.ones(T, T, device=device, dtype=torch.bool).tril()
            cache = ((T, device), mask)
            self._tril_cache = cache
        return cache[1]

    def _rope_tables(self, freqs_cis: torch.Tensor):
        """Return the expanded (cos, sin) rotary tables, cached per
        (source storage, shape, device, dtype). The tables never require
        grad, so the cache is a plain attribute and does not show up in the
        state dict.

        The key uses the source tensor's storage pointer rather than its
        Python identity: the caller passes `self.freqs_cis[:T]`, i.e. a fresh
        view of the same buffer on every forward, so an identity test would
        miss on every call. Two different frequency tables of the same length
        (e.g. a re-scaled RoPE base, or a different model sharing the module)
        still get distinct storage pointers. The source is kept alive by the
        cache entry so its storage pointer stays valid.
        """
        key = (
            freqs_cis.data_ptr(),
            freqs_cis.shape,
            freqs_cis.device,
            freqs_cis.dtype,
        )
        cache = self._rope_cache
        if cache is None or cache[0] != key:
            f_real = freqs_cis.real
            f_imag = freqs_cis.imag
            cos = torch.cat([f_real, f_real], dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
            sin = torch.cat([f_imag, f_imag], dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
            self._rope_cache = (key, freqs_cis, cos, sin)
        return self._rope_cache[2], self._rope_cache[3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class RMSNormHead(nn.Module):
    """RMSNorm over the head_dim (QK-Norm)."""

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x.mul_(self.weight)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
):
    """Apply rotary positional embeddings. q,k: [B,H,T,hd].

    `cos`/`sin` may be passed pre-expanded ([1,1,T,hd]); otherwise they are
    derived from `freqs_cis` on the fly.
    """
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    if cos is None or sin is None:
        f_real = freqs_cis.real
        f_imag = freqs_cis.imag
        cos = torch.cat([f_real, f_real], dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.cat([f_imag, f_imag], dim=-1).unsqueeze(0).unsqueeze(0)

    # Cast the embeddings rather than the queries: q/k keep their own dtype,
    # so the rotary and the attention scores derived from it are computed in
    # that dtype. The embeddings are [1, 1, T, hd] and are cast once here.
    if cos.dtype != q.dtype:
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)

    q_out = q * cos
    q_out.add_(rotate_half(q).mul_(sin))
    k_out = k * cos
    k_out.add_(rotate_half(k).mul_(sin))
    return q_out, k_out


__all__ = ["CausalSelfAttention", "RMSNormHead"]
