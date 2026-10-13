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

from .norms import QKSquash, _RMSNorm


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
        self.scale = 1.0 / math.sqrt(self.head_dim)

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

        # Any QK transform reallocates Q and K. From that moment `v` is the
        # only slice still aliasing the `[B, T, 3C]` projection buffer, and
        # SDPA keeps `v` for its backward, so the whole 3C buffer would be
        # pinned alive for the rest of the layer. Materialising `v` hands it
        # a private C-sized storage instead and lets the projection buffer
        # go. Without a transform Q, K and V all share one storage and the
        # buffer is already minimal, so the copy is skipped there.
        qk_fresh = False
        if self.q_rms is not None:
            q = self.q_rms(q)
            k = self.k_rms(k)
            qk_fresh = True
        elif self.q_squash is not None:
            q, k = self.q_squash(q, k)
            qk_fresh = True

        if freqs_cis is not None:
            cos, sin = self._rope_tables(freqs_cis, q.dtype)
            q, k = apply_rope(q, k, freqs_cis, cos, sin)
            qk_fresh = True

        if qk_fresh:
            v = v.contiguous()

        scale = self.scale

        # logit monitoring (useful for the QK-squash ablation). Monitoring
        # only; disabled by default.
        stats = {}
        if return_stats:
            with torch.no_grad():
                stats = self._logit_stats(q, k, scale, T, x.device)

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
            mask = torch.ones(T, T, device=device, dtype=torch.bool).tril_()
            cache = ((T, device), mask)
            self._tril_cache = cache
        return cache[1]

    _LOGITS_CHUNK_BYTES = 1 << 26

    def _logit_stats(
        self, q: torch.Tensor, k: torch.Tensor, scale: float, T: int, device
    ) -> dict:
        """Causal-logit monitor, evaluated over query-row blocks.

        Each `[B, H, R, T]` block is reduced to a few scalars before the
        next one is computed. The reductions run in float64 on the device
        (accumulated as 0-dim tensors) and are only read back once at the
        end.

        The upper triangle is zeroed rather than gathered with
        `masked_select`: a zero adds nothing to any of the sums and cannot
        raise the absolute maximum, so the reductions see exactly the causal
        entries the gather would have handed over.
        """
        B, H = q.shape[0], q.shape[1]
        keep = self._tril_mask(T, device)
        rows = max(
            1, self._LOGITS_CHUNK_BYTES // max(1, B * H * T * q.element_size())
        )
        # `keep` is the full lower triangle, so its population is T(T+1)/2.
        n_valid = B * H * (T * (T + 1) // 2)
        if n_valid <= 1:
            return {"logit_abs_mean": 0.0, "logit_abs_max": 0.0, "logit_std": 0.0}

        acc_abs = torch.zeros((), dtype=torch.float64, device=device)
        acc_x = torch.zeros((), dtype=torch.float64, device=device)
        acc_sq = torch.zeros((), dtype=torch.float64, device=device)
        peak = torch.zeros((), dtype=q.dtype, device=device)
        # Complement of the causal triangle, built once rather than re-derived
        # for every block below.
        drop = ~keep
        k_t = k.transpose(-1, -2)
        for start in range(0, T, rows):
            end = min(start + rows, T)
            block = torch.matmul(q[:, :, start:end], k_t)
            block.mul_(scale)
            block.masked_fill_(drop[start:end], 0.0)
            # Signed sum first: the variance needs `sum(x)`, and the absolute
            # pass below would have thrown the signs away.
            acc_x = acc_x + block.sum(dtype=torch.float64)
            block.abs_()
            peak = torch.maximum(peak, block.amax())
            acc_abs = acc_abs + block.sum(dtype=torch.float64)
            block.square_()
            acc_sq = acc_sq + block.sum(dtype=torch.float64)
        total = float(acc_abs)
        signed = float(acc_x)
        mean_abs = total / n_valid
        var = (float(acc_sq) - signed * signed / n_valid) / (n_valid - 1)
        return {
            "logit_abs_mean": mean_abs,
            "logit_abs_max": float(peak),
            "logit_std": math.sqrt(var) if var > 0.0 else 0.0,
        }

    def _rope_tables(self, freqs_cis: torch.Tensor, dtype: torch.dtype):
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

        The expanded tables live in the source's own dtype and the cast to
        the activation dtype is memoised beside them, so a mixed-precision
        model does not re-allocate and re-copy the tables on every layer of
        every step.
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
            self._rope_cache = (key, freqs_cis, {}, cos, sin)
        tables = self._rope_cache[2]
        entry = tables.get(dtype)
        if entry is None:
            cos, sin = self._rope_cache[3], self._rope_cache[4]
            if cos.dtype != dtype:
                cos, sin = cos.to(dtype), sin.to(dtype)
            entry = (cos, sin)
            tables[dtype] = entry
        return entry


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
        # The same fused kernel the backbone norms use, rather than the
        # `pow(2).mean()` composition. The math is identical.
        return _RMSNorm.apply(x, self.weight, self.eps)


def _rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate one `[B, H, T, hd]` embedding with pre-expanded tables.

    The textbook pairing (`rotate_half(x) = cat([-x2, x1])`, then
    `x * cos + rotate_half(x) * sin`) is written as two in-place `addcmul`s
    over the two half-views of the output.
    """
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    # out[..., :half] = x_lo * cos_lo - x_hi * sin_lo
    # out[..., half:] = x_hi * cos_hi + x_lo * sin_hi
    out = x * cos
    out[..., :half].addcmul_(x_hi, sin[..., :half], value=-1.0)
    out[..., half:].addcmul_(x_lo, sin[..., half:], value=1.0)
    return out


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

    return _rope_half(q, cos, sin), _rope_half(k, cos, sin)


__all__ = ["CausalSelfAttention", "RMSNormHead"]
