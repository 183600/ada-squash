from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import QKSquash, _RMSNorm
_TRIL_MASK_CACHE: dict = {}
_ROPE_TABLE_CACHE: dict = {}
_CACHE_MAX_ENTRIES = 8

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
        self._logit_buf: Optional[tuple] = None
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
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        del qkv
        if self.q_rms is not None:
            q = self.q_rms(q)
            k = self.k_rms(k)
        elif self.q_squash is not None:
            q, k = self.q_squash(q, k)
        if freqs_cis is not None:
            cos, sin = self._rope_tables(freqs_cis, q.dtype)
            q, k = _apply_rope_(q, k, freqs_cis, cos, sin)
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
        key = (T, device)
        cache = self._tril_cache
        if cache is not None and cache[0] == key:
            return cache[1]
        mask = _TRIL_MASK_CACHE.get(key)
        if mask is None:
            ij = torch.arange(T, device=device)
            mask = ij[:, None] < ij[None, :]
            if len(_TRIL_MASK_CACHE) >= _CACHE_MAX_ENTRIES:
                _TRIL_MASK_CACHE.clear()
            _TRIL_MASK_CACHE[key] = mask
        self._tril_cache = (key, mask)
        return mask
    _LOGITS_CHUNK_BYTES = 1 << 27

    def _logit_scratch(self, B: int, H: int, R: int, T: int, q: torch.Tensor) -> torch.Tensor:
        key = (B, H, R, T, q.dtype, q.device)
        cache = self._logit_buf
        if cache is None or cache[0] != key:
            cache = (key, torch.empty(B, H, R, T, dtype=q.dtype, device=q.device))
            self._logit_buf = cache
        return cache[1]

    def _logit_stats(self, q: torch.Tensor, k: torch.Tensor, scale: float, T: int, device) -> dict:
        B, H = (q.shape[0], q.shape[1])
        rows = max(1, self._LOGITS_CHUNK_BYTES // max(1, B * H * T * q.element_size()))
        n_valid = B * H * (T * (T + 1) // 2)
        if n_valid <= 1:
            return {'logit_abs_mean': 0.0, 'logit_abs_max': 0.0, 'logit_std': 0.0}
        reduce_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        acc_abs = torch.zeros((), dtype=torch.float64, device=device)
        acc_x = torch.zeros((), dtype=torch.float64, device=device)
        acc_sq = torch.zeros((), dtype=torch.float64, device=device)
        peak_sq = torch.zeros((), dtype=q.dtype, device=device)
        drop = self._tril_mask(T, device)
        k_t = k.transpose(-1, -2)
        buf = self._logit_scratch(B, H, min(rows, T), T, q)
        block = None
        for start in range(0, T, rows):
            end = min(start + rows, T)
            block = torch.matmul(q[:, :, start:end], k_t, out=buf[:, :, :end - start])
            block.mul_(scale)
            block.masked_fill_(drop[start:end], 0.0)
            acc_x.add_(block.sum(dtype=reduce_dtype))
            block.abs_()
            acc_abs.add_(block.sum(dtype=reduce_dtype))
            block.square_()
            acc_sq.add_(block.sum(dtype=reduce_dtype))
            torch.maximum(peak_sq, block.amax(), out=peak_sq)
        del k_t, block
        total = float(acc_abs)
        signed = float(acc_x)
        mean_abs = total / n_valid
        var = (float(acc_sq) - signed * signed / n_valid) / (n_valid - 1)
        return {'logit_abs_mean': mean_abs, 'logit_abs_max': math.sqrt(float(peak_sq)), 'logit_std': math.sqrt(var) if var > 0.0 else 0.0}

    def _rope_tables(self, freqs_cis: torch.Tensor, dtype: torch.dtype):
        key = (freqs_cis.data_ptr(), freqs_cis.shape, freqs_cis.device, freqs_cis.dtype)
        cache = self._rope_cache
        entry = cache[1] if cache is not None and cache[0] == key else None
        if entry is None:
            entry = _ROPE_TABLE_CACHE.get(key)
            if entry is None:
                cos = freqs_cis.real.unsqueeze(0).unsqueeze(0).contiguous()
                sin = freqs_cis.imag.unsqueeze(0).unsqueeze(0).contiguous()
                entry = (freqs_cis, cos, sin, {})
                if len(_ROPE_TABLE_CACHE) >= _CACHE_MAX_ENTRIES:
                    _ROPE_TABLE_CACHE.clear()
                _ROPE_TABLE_CACHE[key] = entry
            self._rope_cache = (key, entry)
        tables = entry[3]
        got = tables.get(dtype)
        if got is None:
            cos, sin = (entry[1], entry[2])
            if cos.dtype != dtype:
                cos, sin = (cos.to(dtype).contiguous(), sin.to(dtype).contiguous())
            got = (cos, sin)
            tables[dtype] = got
        return got

class RMSNormHead(nn.Module):

    def __init__(self, head_dim: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNorm.apply(x, self.weight, self.eps)

def _rope_halves(cos: torch.Tensor, sin: torch.Tensor, half: int):
    if cos.shape[-1] == half:
        return (cos, cos, sin, sin)
    return (cos[..., :half], cos[..., half:], sin[..., :half], sin[..., half:])

def _rope_into(x: torch.Tensor, lo_c: torch.Tensor, hi_c: torch.Tensor, lo_s: torch.Tensor, hi_s: torch.Tensor, transpose: bool=False, out: Optional[torch.Tensor]=None):
    """(x_lo*lo_c - x_hi*lo_s, x_hi*hi_c + x_lo*hi_s) written into ``out``.

    The step order is the same op sequence as the plain-autograd composition,
    so the values are bit-identical. ``transpose`` applies the transposed
    (inverse) rotation ``[[lo_c, hi_s], [-lo_s, hi_c]]``, which is what a
    cotangent of this orthogonal map has to be rotated by: the ``lo`` row
    picks up ``hi_s`` and the ``hi`` row picks up ``-lo_s`` (a no-op on the
    half-width split, where both sin halves are the same tensor).
    """
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    out_lo = out[..., :half]
    out_hi = out[..., half:]
    torch.mul(x_lo, lo_c, out=out_lo)
    torch.mul(x_hi, hi_c, out=out_hi)
    lo_sign, hi_sign = (1.0, -1.0) if transpose else (-1.0, 1.0)
    if transpose:
        lo_s, hi_s = hi_s, lo_s
    torch.addcmul(out_lo, x_hi, lo_s, value=lo_sign, out=out_lo)
    torch.addcmul(out_hi, x_lo, hi_s, value=hi_sign, out=out_hi)
    return out

def _rope_inplace_(x: torch.Tensor, lo_c: torch.Tensor, hi_c: torch.Tensor, lo_s: torch.Tensor, hi_s: torch.Tensor):
    """In-place ``_rope_into``: same op order per element, same rounding.

    The ``lo`` row is updated first (reading the original ``hi`` row) and a
    half-width clone carries the original ``lo`` row into the ``hi`` update.
    """
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    tmp = x_lo.clone()
    x_lo.mul_(lo_c)
    x_lo.addcmul_(x_hi, lo_s, value=-1)
    x_hi.mul_(hi_c)
    x_hi.addcmul_(tmp, hi_s, value=1)
    return x

class _ApplyRoPE(torch.autograd.Function):
    """Rotary embedding for Q and K as one autograd node.

    The rotation is orthogonal, so a cotangent is rotated by the transpose
    (R^T = R(-theta)) and depends on nothing but itself and the (cached)
    cos/sin tables; the backward applies that transposed rotation to each
    incoming cotangent with the same half/whole split as the forward.
    """

    @staticmethod
    def forward(ctx, q, k, cos, sin):
        half = q.shape[-1] // 2
        lo_c, hi_c, lo_s, hi_s = _rope_halves(cos, sin, half)
        ctx.save_for_backward(cos, sin)
        qo = _rope_into(q, lo_c, hi_c, lo_s, hi_s)
        ko = _rope_into(k, lo_c, hi_c, lo_s, hi_s)
        return (qo, ko)

    @staticmethod
    def backward(ctx, grad_qo, grad_ko):
        cos, sin = ctx.saved_tensors
        grad_q = grad_k = None
        if ctx.needs_input_grad[0]:
            half = grad_qo.shape[-1] // 2
            lo_c, hi_c, lo_s, hi_s = _rope_halves(cos, sin, half)
            grad_q = _rope_into(grad_qo, lo_c, hi_c, lo_s, hi_s, transpose=True)
        if ctx.needs_input_grad[1]:
            half = grad_ko.shape[-1] // 2
            lo_c, hi_c, lo_s, hi_s = _rope_halves(cos, sin, half)
            grad_k = _rope_into(grad_ko, lo_c, hi_c, lo_s, hi_s, transpose=True)
        return (grad_q, grad_k, None, None)

class _ApplyRoPEOne(torch.autograd.Function):
    """In-place rotation of one Q/K tensor, for single-use inputs.

    ``CausalSelfAttention`` hands its freshly projected (or freshly normed) q/k
    over exactly once, so the rotation is applied directly to that storage. A
    single output keeps the autograd engine's view bookkeeping legal for the
    raw path, where q/k are stride views of the shared qkv buffer.
    """

    @staticmethod
    def forward(ctx, x, cos, sin):
        half = x.shape[-1] // 2
        lo_c, hi_c, lo_s, hi_s = _rope_halves(cos, sin, half)
        ctx.save_for_backward(cos, sin)
        _rope_inplace_(x, lo_c, hi_c, lo_s, hi_s)
        ctx.mark_dirty(x)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        half = grad_out.shape[-1] // 2
        lo_c, hi_c, lo_s, hi_s = _rope_halves(cos, sin, half)
        return (_rope_into(grad_out, lo_c, hi_c, lo_s, hi_s, transpose=True), None, None)

def _apply_rope_(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor, cos: Optional[torch.Tensor]=None, sin: Optional[torch.Tensor]=None):
    if cos is None or sin is None:
        cos = freqs_cis.real.unsqueeze(0).unsqueeze(0).contiguous()
        sin = freqs_cis.imag.unsqueeze(0).unsqueeze(0).contiguous()
    if cos.dtype != q.dtype:
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
    return (_ApplyRoPEOne.apply(q, cos, sin), _ApplyRoPEOne.apply(k, cos, sin))

def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor, cos: Optional[torch.Tensor]=None, sin: Optional[torch.Tensor]=None):
    if cos is None or sin is None:
        cos = freqs_cis.real.unsqueeze(0).unsqueeze(0).contiguous()
        sin = freqs_cis.imag.unsqueeze(0).unsqueeze(0).contiguous()
    if cos.dtype != q.dtype:
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
    return _ApplyRoPE.apply(q, k, cos, sin)
__all__ = ['CausalSelfAttention', 'RMSNormHead']
