from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import QKSquash, _RMSNorm

class CausalSelfAttention(nn.Module):

    def __init__(self, dim: int, n_heads: int, qk_norm: Optional[str]=None, alpha_init: float=1.0, dropout: float=0.0):
        super().__init__()
        assert dim % n_heads == 0, 'dim must be divisible by n_heads'
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
        elif qk_norm == 'qknorm':
            self.q_squash = None
            self.k_squash = None
            self.q_rms = RMSNormHead(self.head_dim)
            self.k_rms = RMSNormHead(self.head_dim)
        elif qk_norm == 'dytqk':
            self.q_squash = QKSquash(alpha_q_init=alpha_init, alpha_k_init=alpha_init)
            self.k_squash = self.q_squash
            self.q_rms = None
            self.k_rms = None
        else:
            raise ValueError(f"Unknown qk_norm='{qk_norm}'")

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor]=None, return_stats: bool=False):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        del qkv
        if self.q_rms is not None:
            q = self.q_rms(q)
            k = self.k_rms(k)
        elif self.q_squash is not None:
            q, k = self.q_squash(q, k)
        if freqs_cis is not None:
            cos, sin = self._rope_tables(freqs_cis, q.dtype)
            q, k = apply_rope(q, k, freqs_cis, cos, sin)
        scale = self.scale
        stats = {}
        if return_stats:
            with torch.no_grad():
                stats = self._logit_stats(q, k, scale, T, x.device)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_drop.p if self.training else 0.0, scale=scale)
        out = out.transpose(1, 2).reshape(B, T, C)
        if self.proj_drop.p > 0.0:
            out = self.proj_drop(self.proj(out))
        else:
            out = self.proj(out)
        return (out, stats)

    def _tril_mask(self, T: int, device) -> torch.Tensor:
        cache = self._tril_cache
        if cache is None or cache[0] != (T, device):
            ij = torch.arange(T, device=device)
            mask = ij[:, None] >= ij[None, :]
            cache = ((T, device), mask, ~mask)
            self._tril_cache = cache
        return cache[1]

    def _tril_pair(self, T: int, device):
        cache = self._tril_cache
        if cache is None or cache[0] != (T, device):
            self._tril_mask(T, device)
            cache = self._tril_cache
        if len(cache) >= 3:
            return (cache[1], cache[2])
        keep = cache[1]
        return (keep, ~keep)
    _LOGITS_CHUNK_BYTES = 1 << 26

    def _logit_stats(self, q: torch.Tensor, k: torch.Tensor, scale: float, T: int, device) -> dict:
        B, H = (q.shape[0], q.shape[1])
        keep = self._tril_mask(T, device)
        rows = max(1, self._LOGITS_CHUNK_BYTES // max(1, B * H * T * q.element_size()))
        n_valid = B * H * (T * (T + 1) // 2)
        if n_valid <= 1:
            return {'logit_abs_mean': 0.0, 'logit_abs_max': 0.0, 'logit_std': 0.0}
        acc_abs = torch.zeros((), dtype=torch.float64, device=device)
        acc_x = torch.zeros((), dtype=torch.float64, device=device)
        acc_sq = torch.zeros((), dtype=torch.float64, device=device)
        peak = torch.zeros((), dtype=q.dtype, device=device)
        drop = self._tril_pair(T, device)[1]
        k_t = k.transpose(-1, -2)
        for start in range(0, T, rows):
            end = min(start + rows, T)
            block = torch.matmul(q[:, :, start:end], k_t)
            block.mul_(scale)
            block.masked_fill_(drop[start:end], 0.0)
            acc_x.add_(block.sum(dtype=torch.float64))
            block.abs_()
            torch.maximum(peak, block.amax(), out=peak)
            acc_abs.add_(block.sum(dtype=torch.float64))
            block.square_()
            acc_sq.add_(block.sum(dtype=torch.float64))
        total = float(acc_abs)
        signed = float(acc_x)
        mean_abs = total / n_valid
        var = (float(acc_sq) - signed * signed / n_valid) / (n_valid - 1)
        return {'logit_abs_mean': mean_abs, 'logit_abs_max': float(peak), 'logit_std': math.sqrt(var) if var > 0.0 else 0.0}

    def _rope_tables(self, freqs_cis: torch.Tensor, dtype: torch.dtype):
        key = (freqs_cis.data_ptr(), freqs_cis.shape, freqs_cis.device, freqs_cis.dtype)
        cache = self._rope_cache
        if cache is None or cache[0] != key:
            f_real = freqs_cis.real
            f_imag = freqs_cis.imag
            cos = torch.cat([f_real, f_real], dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
            sin = torch.cat([f_imag, f_imag], dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
            self._rope_cache = (key, None, {}, cos, sin)
        tables = self._rope_cache[2]
        entry = tables.get(dtype)
        if entry is None:
            cos, sin = (self._rope_cache[3], self._rope_cache[4])
            if cos.dtype != dtype:
                cos, sin = (cos.to(dtype).contiguous(), sin.to(dtype).contiguous())
            entry = (cos, sin)
            tables[dtype] = entry
        return entry

class RMSNormHead(nn.Module):

    def __init__(self, head_dim: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNorm.apply(x, self.weight, self.eps)

def _rope_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = x_lo * cos[..., :half] - x_hi * sin[..., :half]
    out[..., half:] = x_hi * cos[..., half:] + x_lo * sin[..., half:]
    return out

def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor, cos: Optional[torch.Tensor]=None, sin: Optional[torch.Tensor]=None):
    if cos is None or sin is None:
        f_real = freqs_cis.real
        f_imag = freqs_cis.imag
        cos = torch.cat([f_real, f_real], dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
        sin = torch.cat([f_imag, f_imag], dim=-1).unsqueeze(0).unsqueeze(0).contiguous()
    if cos.dtype != q.dtype:
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
    return (_rope_half(q, cos, sin), _rope_half(k, cos, sin))
__all__ = ['CausalSelfAttention', 'RMSNormHead']
