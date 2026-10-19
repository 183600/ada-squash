from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import skip_init
from .norms import QKSquash, _RMSNorm
_TRIL_MASK_CACHE: dict = {}
_ROPE_TABLE_CACHE: dict = {}
_LOGIT_SCRATCH_CACHE: dict = {}
_CACHE_MAX_ENTRIES = 8
_LOGIT_SCRATCH_MAX_ENTRIES = 2
_ROPE_HALF_SCRATCH_CACHE: dict = {}
_ROPE_HALF_SCRATCH_MAX_ENTRIES = 4

def _rope_half_scratch(x_lo: torch.Tensor) -> torch.Tensor:
    """Shared half-width buffer for the rotary rotation.

    Keyed by (shape, dtype, device) like the other aux caches (never by
    `id()`); an inference-mode buffer is rebuilt for a grad-mode caller.
    """
    key = (tuple(x_lo.shape), x_lo.dtype, x_lo.device)
    buf = _ROPE_HALF_SCRATCH_CACHE.get(key)
    if buf is not None and buf.is_inference() and (not torch.is_inference_mode_enabled()):
        buf = None
    if buf is None:
        buf = torch.empty(x_lo.shape, dtype=x_lo.dtype, device=x_lo.device)
        if len(_ROPE_HALF_SCRATCH_CACHE) >= _ROPE_HALF_SCRATCH_MAX_ENTRIES:
            _ROPE_HALF_SCRATCH_CACHE.clear()
        _ROPE_HALF_SCRATCH_CACHE[key] = buf
    return buf

def clear_logit_scratch_cache() -> None:
    """Drop the cached `_logit_scratch` buffers when a stats window ends."""
    _LOGIT_SCRATCH_CACHE.clear()

def _skip_init(module_cls, *args, **kwargs):
    """``torch.nn.utils.skip_init`` that honors the ambient default device.

    ``scripts/ablation.py`` builds whole models under
    ``with torch.device('meta')`` just to read their config, so the module
    is kept on the ambient default device.
    """
    device = torch.get_default_device()
    if device.type == 'meta':
        with torch.device('meta'):
            return module_cls(*args, **kwargs)
    return skip_init(module_cls, *args, **kwargs, device=device)

def _make_linear(in_features: int, out_features: int, bias: bool, skip: bool) -> nn.Linear:
    """``nn.Linear`` whose constructor's kaiming pass can be skipped.

    ``GPT.__init__`` re-initializes every Linear through ``_init_weights``
    right after construction, so the constructor's draw is overwritten; when
    the caller says so, build the module with uninitialized storage instead
    (``skip_init``) and let that pass be the only initialization. Standalone
    construction (tests, helper modules) keeps the default initialization.
    """
    if skip:
        return _skip_init(nn.Linear, in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)

class CausalSelfAttention(nn.Module):

    def __init__(self, dim: int, n_heads: int, qk_norm: Optional[str]=None, alpha_init: float=1.0, dropout: float=0.0, skip_init: bool=False):
        super().__init__()
        assert dim % n_heads == 0, 'dim must be divisible by n_heads'
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dim = dim
        self.qk_norm = qk_norm
        self._rope_cache: Optional[tuple] = None
        self._tril_cache: Optional[tuple] = None
        self.qkv = _make_linear(dim, 3 * dim, False, skip_init)
        self.proj = _make_linear(dim, dim, False, skip_init)
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
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        if freqs_cis is None or self.q_squash is not None or self.q_rms is not None:
            # `unbind` and `qkv[:, :, i]` select the same three tensors; the
            # raw rotary path below still needs single views because
            # _ApplyRoPEOne edits them.
            q, k, v = qkv.unbind(2)
        else:
            q, k, v = (qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2])
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
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
    # Row-block cap for the causal logit sweep: the sweep walks the lower
    # triangle in row blocks of at most this many rows.
    _LOGITS_BLOCK_ROWS = 128

    def _logit_scratch(self, B: int, H: int, R: int, T: int, dtype: torch.dtype, device) -> torch.Tensor:
        key = (B, H, R, T, dtype, device)
        buf = _LOGIT_SCRATCH_CACHE.get(key)
        if buf is None:
            # Flat scratch for the whole sweep; each row block views a
            # contiguous prefix of the shape it needs.
            buf = torch.empty(B * H * R * T, dtype=dtype, device=device)
            if len(_LOGIT_SCRATCH_CACHE) >= _LOGIT_SCRATCH_MAX_ENTRIES:
                _LOGIT_SCRATCH_CACHE.clear()
            _LOGIT_SCRATCH_CACHE[key] = buf
        return buf

    def _logit_stats(self, q: torch.Tensor, k: torch.Tensor, scale: float, T: int, device) -> dict:
        B, H = (q.shape[0], q.shape[1])
        reduce_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        # The sweep can need two row-block images (the q-dtype matmul output and
        # its reduce-dtype copy). Budget both together so the stats window never
        # holds 1.5x the advertised scratch just because autocast halves the
        # logit element size.
        row_bytes = q.element_size()
        if reduce_dtype != q.dtype:
            row_bytes += torch.finfo(reduce_dtype).bits // 8
        budget_rows = max(1, self._LOGITS_CHUNK_BYTES // max(1, B * H * T * row_bytes))
        rows = min(T, self._LOGITS_BLOCK_ROWS, budget_rows)
        n_valid = B * H * (T * (T + 1) // 2)
        if n_valid <= 1:
            return {'logit_abs_mean': 0.0, 'logit_abs_max': 0.0, 'logit_std': 0.0}
        acc_abs = torch.zeros((), dtype=torch.float64, device=device)
        acc_x = torch.zeros((), dtype=torch.float64, device=device)
        acc_sq = torch.zeros((), dtype=torch.float64, device=device)
        # The max statistic is kept in the reduce dtype: squaring a bf16/fp16
        # logit can overflow (|x| > ~1.8e19) to inf, exactly when a blown-up
        # logit is the thing the monitor is looking for.
        peak = torch.zeros((), dtype=reduce_dtype, device=device)
        drop = self._tril_mask(T, device)
        k_t = k.transpose(-1, -2)
        buf = self._logit_scratch(B, H, min(rows, T), T, q.dtype, q.device)
        # Reductions run on a matching-precision image of the block.
        work = buf if reduce_dtype == q.dtype else self._logit_scratch(B, H, min(rows, T), T, reduce_dtype, device)
        # Scaled statistics: `s*sum(x)`, `s*sum|x|`, `s*max|x|`, `s^2*sum(x^2)`,
        # accumulated in float64.
        scale_sq = scale * scale
        block = None
        for start in range(0, T, rows):
            end = min(start + rows, T)
            # Only keys up to `end` are causal for this row block; later
            # columns would be masked to zero. The kept values are unchanged
            # (each column is the same reduction over D).
            block = torch.matmul(q[:, :, start:end], k_t[..., :end],
                                 out=buf[:B * H * (end - start) * end].view(B, H, end - start, end))
            # Only the square [start:end) part of the row block can be causally
            # masked: every earlier column is below the diagonal for all rows
            # of this block. The kept logits are unchanged.
            block[:, :, :, start:end].masked_fill_(drop[start:end, start:end], 0.0)
            if work is buf:
                blockf = block
            else:
                blockf = work[:B * H * (end - start) * end].view(B, H, end - start, end)
                blockf.copy_(block)
            acc_x.add_(blockf.sum().mul_(scale))
            blockf.abs_()
            acc_abs.add_(blockf.sum().mul_(scale))
            torch.maximum(peak, blockf.amax().mul_(scale).to(reduce_dtype), out=peak)
            blockf.square_()
            acc_sq.add_(blockf.sum().mul_(scale_sq))
        del k_t, block
        total, signed, sq, peak_v = torch.stack(
            (acc_abs, acc_x, acc_sq, peak.to(torch.float64))).tolist()
        mean_abs = total / n_valid
        var = (sq - signed * signed / n_valid) / (n_valid - 1)
        return {'logit_abs_mean': mean_abs, 'logit_abs_max': peak_v, 'logit_std': math.sqrt(var) if var > 0.0 else 0.0}

    def _rope_tables(self, freqs_cis: torch.Tensor, dtype: torch.dtype):
        key = (freqs_cis.data_ptr(), freqs_cis.shape, freqs_cis.device, freqs_cis.dtype)
        cache = self._rope_cache
        entry = cache[1] if cache is not None and cache[0] == key else None
        if entry is not None and entry[1].is_inference() and (not torch.is_inference_mode_enabled()):
            # A table built under `inference_mode` cannot be saved for
            # backward; a later grad-mode forward through this module has to
            # get normal tables.
            entry = None
        if entry is None:
            key, entry = _rope_table_entry(freqs_cis)
            self._rope_cache = (key, entry)
        return _rope_pair_for_dtype(entry, dtype)

class RMSNormHead(nn.Module):

    def __init__(self, head_dim: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNorm.apply(x, self.weight, self.eps)

def _rope_table_entry(freqs_cis: torch.Tensor):
    """``(key, entry)`` for ``freqs_cis`` from the shared table cache.

    The key is the ``(data_ptr, shape, device, dtype)`` tuple D6 fixes; the
    entry keeps a strong reference to ``freqs_cis`` so the storage pointer can
    never be recycled into a key collision while the entry is cached. Both the
    attention modules and the standalone `apply_rope` go through here.
    """
    key = (freqs_cis.data_ptr(), freqs_cis.shape, freqs_cis.device, freqs_cis.dtype)
    entry = _ROPE_TABLE_CACHE.get(key)
    if entry is not None and entry[1].is_inference() and (not torch.is_inference_mode_enabled()):
        # Inference-mode tables are immutable constants: they cannot be saved
        # for backward, so grad-mode callers need a normal rebuild (the key
        # and the values are unchanged).
        entry = None
    if entry is None:
        cos = freqs_cis.real.unsqueeze(0).unsqueeze(0).contiguous()
        sin = freqs_cis.imag.unsqueeze(0).unsqueeze(0).contiguous()
        entry = (freqs_cis, cos, sin, {})
        if len(_ROPE_TABLE_CACHE) >= _CACHE_MAX_ENTRIES:
            _ROPE_TABLE_CACHE.clear()
        _ROPE_TABLE_CACHE[key] = entry
    return (key, entry)

def _rope_pair_for_dtype(entry, dtype: torch.dtype):
    """``(cos, sin)`` of an entry in ``dtype``; the cast pair is cached too."""
    tables = entry[3]
    got = tables.get(dtype)
    if got is not None and got[0].is_inference() and (not torch.is_inference_mode_enabled()):
        got = None
    if got is None:
        cos, sin = (entry[1], entry[2])
        if cos.dtype != dtype:
            cos, sin = (cos.to(dtype).contiguous(), sin.to(dtype).contiguous())
        got = (cos, sin)
        tables[dtype] = got
    return got

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
    """Apply the ``_rope_into`` rotation to ``x`` itself.

    The ``lo`` row is updated first (reading the original ``hi`` row) and a
    half-width clone carries the original ``lo`` row into the ``hi`` update.
    """
    half = x.shape[-1] // 2
    x_lo = x[..., :half]
    x_hi = x[..., half:]
    tmp = _rope_half_scratch(x_lo)
    tmp.copy_(x_lo)
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
    """Rotary rotation of one Q/K tensor, applied to ``x`` itself.

    A single returned tensor keeps the autograd engine's view bookkeeping
    legal for the raw path, where q/k are stride views of the shared qkv
    buffer.
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

def _resolve_rope_tables(freqs_cis: torch.Tensor, cos: Optional[torch.Tensor], sin: Optional[torch.Tensor], dtype: torch.dtype):
    """``(cos, sin)`` in ``dtype``, from the caller or the shared cache.

    A caller-supplied pair is only cast (never re-expanded); otherwise the
    shared table cache serves both the expansion and the per-dtype casts.
    """
    if cos is None or sin is None:
        _, entry = _rope_table_entry(freqs_cis)
        return _rope_pair_for_dtype(entry, dtype)
    if cos.dtype != dtype:
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
    return (cos, sin)

def _apply_rope_(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor, cos: Optional[torch.Tensor]=None, sin: Optional[torch.Tensor]=None):
    cos, sin = _resolve_rope_tables(freqs_cis, cos, sin, q.dtype)
    return (_ApplyRoPEOne.apply(q, cos, sin), _ApplyRoPEOne.apply(k, cos, sin))

def apply_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor, cos: Optional[torch.Tensor]=None, sin: Optional[torch.Tensor]=None):
    cos, sin = _resolve_rope_tables(freqs_cis, cos, sin, q.dtype)
    return _ApplyRoPE.apply(q, k, cos, sin)
__all__ = ['CausalSelfAttention', 'RMSNormHead']
