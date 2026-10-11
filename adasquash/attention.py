"""
Causal self-attention with optional QK-squashing (DyT++ variant C).

The attention implementation is deliberately simple and dependency-free
(no FlashAttention, no sdpa) so that the *normalization* layer's effect on
logit magnitudes and gradient flow is the only variable being measured.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import QKSquash, build_norm


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
                # Keep only the causal-visible (finite) logits for the
                # statistics; the -inf masked entries would otherwise make
                # every aggregate (mean/max/std) inf/nan.
                causal_mask = torch.triu(
                    torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
                )
                valid = logits.masked_fill(causal_mask, float("-inf"))
                valid = valid[valid != float("-inf")]
                if valid.numel():
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

        # Use SDPA for memory-efficient attention (FlashAttention / mem-efficient).
        # is_causal=True is mathematically equivalent to the manual triu mask
        # and lets SDPA pick the most efficient fused kernel.
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=scale,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj_drop(self.proj(out))
        return out, stats

    def _rope_tables(self, freqs_cis: torch.Tensor):
        """Return the expanded (cos, sin) rotary tables, cached per
        (length, device, dtype). The tables never require grad, so the cache
        is a plain attribute and does not show up in the state dict."""
        key = (freqs_cis.shape[0], freqs_cis.device, freqs_cis.dtype)
        cache = self._rope_cache
        if cache is None or cache[0] != key:
            # freqs_cis: [T, hd//2] complex (exp(i*theta)); take real/imag as
            # the rotation factors and expand to [T, hd] to match the head
            # dim, then broadcast-shape them to [1, 1, T, hd].
            f_real = freqs_cis.real
            f_imag = freqs_cis.imag
            cos = torch.cat([f_real, f_real], dim=-1).unsqueeze(0).unsqueeze(0)
            sin = torch.cat([f_imag, f_imag], dim=-1).unsqueeze(0).unsqueeze(0)
            cache = (key, cos, sin)
            self._rope_cache = cache
        return cache[1], cache[2]


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
        var = (x * x).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


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

    # freqs_cis: [T, hd//2] complex (exp(i*theta)); take real/imag as the
    # rotation factors and expand to [T, hd] to match the head dim.
    if cos is None or sin is None:
        f_real = freqs_cis.real
        f_imag = freqs_cis.imag
        cos = torch.cat([f_real, f_real], dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.cat([f_imag, f_imag], dim=-1).unsqueeze(0).unsqueeze(0)
    q_d = q.float()
    k_d = k.float()
    q_out = q_d * cos
    rot = rotate_half(q_d)
    rot.mul_(sin)
    q_out.add_(rot)
    k_out = k_d * cos
    rot = rotate_half(k_d)
    rot.mul_(sin)
    k_out.add_(rot)
    return q_out.to(q.dtype), k_out.to(k.dtype)


__all__ = ["CausalSelfAttention", "RMSNormHead"]
