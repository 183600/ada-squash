from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .norms import build_norm, _reduce_products
from .attention import CausalSelfAttention, _make_linear, _skip_init

@dataclass
class LayerConfig:
    norm: str = 'dytcd'
    qk_norm: Optional[str] = None
    use_layerscale: bool = True
    layerscale_init: float = 1e-05

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    max_seq_len: int = 1024
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    norm_name: str = 'dytcd'
    qk_norm: Optional[str] = None
    qk_alpha_init: float = 1.0
    use_layerscale: bool = True
    layerscale_init: float = 1e-05
    alpha_0: float = 0.9
    k_init: float = 1.0
    alpha_init: float = 0.5
    p_init: float = 2.0
    learnable_p: bool = True
    learnable_k: bool = True
    learnable_delta: bool = True
    beta: bool = True
    alpha_jitter: float = 0.05
    gradient_checkpointing: bool = False
    pad_token_id: int = 0
    ignore_index: int = -100
    tie_word_embeddings: bool = True
    rotary: bool = False
_CROSSENTROPY_CHUNK_BYTES = 1 << 27
_CROSS_ENTROPY_FAST_PATH_BYTES = 1 << 23
_CROSS_ENTROPY_CHUNK_ROWS_CACHE: dict[int, int] = {}
_CE_SCRATCH_CACHE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
# At most one shape pair is live at a time: the fp32 accumulator plus (under
# autocast) its low-precision cast twin.
_CE_SCRATCH_MAX_ENTRIES = 2

def _cross_entropy_chunk_rows(vocab_size: int) -> int:
    if vocab_size not in _CROSS_ENTROPY_CHUNK_ROWS_CACHE:
        row_bytes = max(1, vocab_size) * 4
        _CROSS_ENTROPY_CHUNK_ROWS_CACHE[vocab_size] = max(1, _CROSSENTROPY_CHUNK_BYTES // row_bytes)
    return _CROSS_ENTROPY_CHUNK_ROWS_CACHE[vocab_size]

_FUSED_LINEAR_CHUNK_ROWS_CACHE: dict[tuple, int] = {}

def _fused_linear_chunk_rows(chunk_rows: int, vocab: int, dt: torch.dtype, acc: torch.dtype) -> int:
    """Cap a fused-CE chunk at the row-block size.

    The fused loss can hold the low-precision logits plus their accumulator
    image at the same time, so the chunk accounts for both.
    """
    key = (vocab, dt, acc)
    if key not in _FUSED_LINEAR_CHUNK_ROWS_CACHE:
        element_bytes = torch.finfo(acc).bits // 8
        if dt != acc:
            element_bytes += torch.finfo(dt).bits // 8
        cap = max(1, _CROSSENTROPY_CHUNK_BYTES // max(1, vocab * element_bytes))
        _FUSED_LINEAR_CHUNK_ROWS_CACHE[key] = cap
    return min(chunk_rows, _FUSED_LINEAR_CHUNK_ROWS_CACHE[key])

def _ce_scratch(rows: int, vocab: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Reusable ``(rows, vocab)`` logit scratch for the chunked losses.

    The buffers are private to the forward/backward sweeps (never returned,
    never saved for backward) and keyed by shape, dtype and device.
    """
    key = (rows, vocab, dtype, device)
    buf = _CE_SCRATCH_CACHE.get(key)
    if buf is not None and buf.is_inference() and (not torch.is_inference_mode_enabled()):
        buf = None
    if buf is None:
        buf = torch.empty(rows, vocab, dtype=dtype, device=device)
        if len(_CE_SCRATCH_CACHE) >= _CE_SCRATCH_MAX_ENTRIES:
            _CE_SCRATCH_CACHE.popitem(last=False)
        _CE_SCRATCH_CACHE[key] = buf
    else:
        _CE_SCRATCH_CACHE.move_to_end(key)
    return buf

def clear_ce_scratch_cache() -> None:
    """Release the cached CE scratch buffers (model switch / run end)."""
    _CE_SCRATCH_CACHE.clear()

# Autocast weight casts for the custom kernels, keyed by identity + version.
# `_version` bumps on every in-place update, so an optimizer step can never
# leave a stale cast behind. Each entry keeps a strong reference to its
# source weight (the guard the RoPE table cache uses): the storage address
# in the key can then never be recycled by a *different* model's parameter,
# and the identity check rejects a same-address / same-version collision
# outright.
_CAST_WEIGHT_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
_CAST_WEIGHT_CACHE_BYTES = 0
_CAST_WEIGHT_CACHE_MAX_BYTES = 1 << 28

def clear_cast_weight_cache() -> None:
    """Release the cached autocast weight casts (model switch / run end)."""
    global _CAST_WEIGHT_CACHE_BYTES
    _CAST_WEIGHT_CACHE.clear()
    _CAST_WEIGHT_CACHE_BYTES = 0

def _cast_entry_bytes(entry: tuple) -> int:
    cast = entry[1]
    return cast.numel() * cast.element_size()

def _cast_weight_cached(w: torch.Tensor, dt: torch.dtype) -> torch.Tensor:
    global _CAST_WEIGHT_CACHE_BYTES
    if w.dtype == dt:
        return w
    key = (w.data_ptr(), w._version, dt, w.device)
    entry = _CAST_WEIGHT_CACHE.get(key)
    if entry is not None and entry[0] is w and entry[1].shape == w.shape:
        if not (entry[1].is_inference() and (not torch.is_inference_mode_enabled())):
            return entry[1]
        # An inference-mode cast cannot be saved for backward; a grad-mode
        # forward has to rebuild it (the key and the values are unchanged).
    nbytes = w.numel() * (torch.finfo(dt).bits // 8)
    if nbytes > _CAST_WEIGHT_CACHE_MAX_BYTES:
        # A single cast larger than the cache capacity (e.g. the 1B CE
        # weight) is used directly instead of being cached.
        return w.to(dt)
    if entry is not None:
        # Stale same-key entry (collided identity or an inference cast):
        # the replacement gives back the bytes it shadows.
        _CAST_WEIGHT_CACHE_BYTES -= _cast_entry_bytes(entry)
    if _CAST_WEIGHT_CACHE_BYTES + nbytes > _CAST_WEIGHT_CACHE_MAX_BYTES:
        _CAST_WEIGHT_CACHE.clear()
        _CAST_WEIGHT_CACHE_BYTES = 0
    got = w.to(dt)
    _CAST_WEIGHT_CACHE[key] = (w, got)
    _CAST_WEIGHT_CACHE_BYTES += nbytes
    return got

def build_norm_kwargs(cfg: 'GPTConfig', layer_idx: int=0, is_final_norm: bool=False) -> dict:
    kwargs: dict = {}
    is_dyt_family = cfg.norm_name in ('dyt', 'dytd', 'dytd_tanh', 'dytc', 'dytcd', 'dytcd_tanh')
    if cfg.norm_name in ('dytd', 'dytd_tanh', 'dytcd', 'dytcd_tanh'):
        if is_final_norm:
            depth_ratio = 1.0
            layer_idx = cfg.n_layers
        else:
            depth_ratio = float(layer_idx) / float(cfg.n_layers)
        kwargs = dict(layer_idx=layer_idx, num_layers=cfg.n_layers, alpha_0=cfg.alpha_0, k_init=cfg.k_init, learnable_k=cfg.learnable_k)
    if cfg.norm_name in ('dytcd', 'dytcd_tanh'):
        kwargs.update(learnable_delta=cfg.learnable_delta)
    if cfg.norm_name == 'dytcd':
        kwargs.update(p_init=cfg.p_init, learnable_p=cfg.learnable_p)
    if cfg.norm_name == 'dytc':
        kwargs.update(p_init=cfg.p_init, learnable_p=cfg.learnable_p, alpha_init=cfg.alpha_init, alpha_jitter=cfg.alpha_jitter)
    if cfg.norm_name == 'dyt':
        kwargs.update(alpha_init=cfg.alpha_init)
    if is_dyt_family and hasattr(cfg, 'beta') and (cfg.beta is not None):
        kwargs.update(beta=cfg.beta)
    return kwargs

class _ChunkedCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, targets, ignore_index, chunk_rows):
        total = None
        n = targets.numel()
        buf_rows = min(chunk_rows, n)
        buf = None
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            if buf is None:
                buf = _ce_scratch(buf_rows, logits.size(-1), logits.dtype, logits.device)
            log_probs = buf[:end - start]
            torch.log_softmax(logits[start:end], dim=-1, out=log_probs)
            chunk_loss = F.nll_loss(log_probs, targets[start:end], ignore_index=ignore_index, reduction='sum')
            total = chunk_loss if total is None else total.add_(chunk_loss)
        count = targets.ne(ignore_index).sum()
        ctx.save_for_backward(logits, targets)
        ctx.ignore_index = ignore_index
        ctx.chunk_rows = chunk_rows
        ctx.count = count
        return total / count

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        grad_logits = torch.empty_like(logits)
        grad_scale = (grad_out / ctx.count.clamp(min=1)).to(logits.dtype)
        chunk_rows = ctx.chunk_rows
        # The row mask and the scatter source depend only on the targets.
        valid = targets.ne(ctx.ignore_index)
        gold = targets.masked_fill(~valid, 0).unsqueeze(-1)
        keep = valid.unsqueeze(-1).to(logits.dtype)
        neg_keep = keep.neg()
        keep.mul_(grad_scale)
        neg_keep.mul_(grad_scale)
        for start in range(0, targets.numel(), chunk_rows):
            end = min(start + chunk_rows, targets.numel())
            probs = grad_logits[start:end]
            torch.softmax(logits[start:end], dim=-1, out=probs)
            probs.mul_(keep[start:end])
            probs.scatter_add_(-1, gold[start:end], neg_keep[start:end])
        return (grad_logits, None, None, None)

def cross_entropy_mean(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int=-100, chunk_rows: Optional[int]=None) -> torch.Tensor:
    # `F.cross_entropy` accepts logits of shape (..., V) with matching target
    # shape (...,). The chunked node slices the leading dimension, so flatten
    # the leading dims first; otherwise a valid 3-D call is sliced on dim 0
    # but its targets are not, and nll_loss sees mismatched shapes.
    if logits.dim() > 2:
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
    if chunk_rows is None:
        chunk_rows = _cross_entropy_chunk_rows(logits.size(-1))
    if logits.size(0) == 0:
        return F.cross_entropy(logits, targets, ignore_index=ignore_index)
    return _ChunkedCrossEntropy.apply(logits, targets, int(ignore_index), int(chunk_rows))

def _loss_acc_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.promote_types(dtype, torch.float32)

def _autocast_dtype(device_type: str, *dtypes: torch.dtype) -> torch.dtype:
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except TypeError:
        if device_type == 'cuda' and torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    fallback = dtypes[0]
    for dt in dtypes[1:]:
        fallback = torch.promote_types(fallback, dt)
    return fallback

class _FusedLinearCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, weight, targets, ignore_index, chunk_rows):
        dt = _autocast_dtype(hidden.device.type, hidden.dtype, weight.dtype)
        w = _cast_weight_cached(weight, dt)
        acc = _loss_acc_dtype(dt)
        chunk_rows = _fused_linear_chunk_rows(chunk_rows, w.shape[0], dt, acc)
        w_t = w.t()
        total = None
        n = targets.numel()
        buf_rows = min(chunk_rows, n)
        logits = None
        cast_buf = None
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            rows = end - start
            if logits is None or logits.size(0) < rows:
                logits = _ce_scratch(buf_rows, w.shape[0], acc, hidden.device)
                cast_buf = None if acc == dt else _ce_scratch(buf_rows, w.shape[0], dt, hidden.device)
            buf = logits[:rows]
            hc = hidden[start:end]
            if hc.dtype != dt:
                hc = hc.to(dt)
            if cast_buf is None:
                torch.mm(hc, w_t, out=buf)
                torch.log_softmax(buf, -1, out=buf)
            else:
                cast = cast_buf[:rows]
                torch.mm(hc, w_t, out=cast)
                torch.log_softmax(cast, -1, dtype=acc, out=buf)
            chunk = F.nll_loss(buf, targets[start:end], ignore_index=ignore_index, reduction='sum')
            total = chunk if total is None else total.add_(chunk)
        count = targets.ne(ignore_index).sum()
        # Backward consumes the same cast weight the forward GEMMs use; the
        # original dtype is kept in `in_dtypes` for the returned gradient.
        ctx.save_for_backward(hidden, w, targets)
        ctx.ignore_index = ignore_index
        ctx.chunk_rows = chunk_rows
        ctx.count = count
        ctx.compute_dtype = dt
        ctx.in_dtypes = (hidden.dtype, weight.dtype)
        return total / count

    @staticmethod
    def backward(ctx, grad_out):
        h, w, targets = ctx.saved_tensors
        dt = ctx.compute_dtype
        acc = _loss_acc_dtype(dt)
        need_h = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        _, weight_dtype = ctx.in_dtypes
        chunk_rows = ctx.chunk_rows
        grad_h = torch.empty_like(h) if need_h else None
        grad_h_tmp = None
        if need_h and h.dtype != dt:
            # The GEMM produces a low-precision tile while the caller needs
            # the gradient in the input dtype; each chunk is written through
            # a small temporary in the GEMM dtype.
            grad_h_tmp = torch.empty((min(chunk_rows, targets.numel()), h.shape[1]), dtype=dt, device=h.device)
        # The weight gradient accumulates into its final buffer from the first
        # chunk on: a low-precision weight keeps a low-precision accumulator,
        # while an fp32/fp64 weight (the autocast case `w = weight.to(bf16)`)
        # accumulates directly in `acc`.
        grad_w_dtype = None
        grad_w_tmp = None
        grad_w = None
        if need_w:
            weight_dtype = ctx.in_dtypes[1]
            low_precision = weight_dtype in (torch.bfloat16, torch.float16)
            grad_w_dtype = weight_dtype if low_precision else acc
            grad_w = torch.empty_like(w, dtype=grad_w_dtype)
        grad_scale = (grad_out / ctx.count.clamp(min=1)).to(dt)
        w_t = w.t()
        first = True
        n = targets.numel()
        buf_rows = min(chunk_rows, n)
        probs = None
        cast_buf = None
        # Target-only row mask and scatter source.
        valid = targets.ne(ctx.ignore_index)
        gold = targets.masked_fill(~valid, 0).unsqueeze(-1)
        keep = valid.unsqueeze(-1).to(acc)
        neg_keep = keep.neg()
        keep.mul_(grad_scale)
        neg_keep.mul_(grad_scale)
        for start in range(0, n, chunk_rows):
            end = min(start + chunk_rows, n)
            rows = end - start
            hc = h[start:end]
            if hc.dtype != dt:
                hc = hc.to(dt)
            if probs is None or probs.size(0) < rows:
                probs = _ce_scratch(buf_rows, w.shape[0], acc, h.device)
                cast_buf = None if acc == dt else _ce_scratch(buf_rows, w.shape[0], dt, h.device)
            buf = probs[:rows]
            if cast_buf is None:
                torch.mm(hc, w_t, out=buf)
                torch.softmax(buf, -1, out=buf)
            else:
                cast = cast_buf[:rows]
                torch.mm(hc, w_t, out=cast)
                torch.softmax(cast, -1, dtype=acc, out=buf)
            buf.mul_(keep[start:end])
            buf.scatter_add_(-1, gold[start:end], neg_keep[start:end])
            if need_h:
                g = buf if acc == dt else cast.copy_(buf)
                if grad_h_tmp is None:
                    torch.mm(g, w, out=grad_h[start:end])
                else:
                    hgrad = grad_h_tmp[:rows]
                    torch.mm(g, w, out=hgrad)
                    grad_h[start:end].copy_(hgrad)
            if need_w:
                # An fp32 weight takes the accumulator-dtype probabilities in
                # `buf`, even under autocast. A low-precision weight uses the
                # low-precision `g` when the input gradient is requested.
                w_probs = g if (need_h and acc != dt) else buf
                if w_probs.dtype == grad_w.dtype and hc.dtype == grad_w.dtype:
                    if first:
                        torch.mm(w_probs.t(), hc, out=grad_w)
                    else:
                        grad_w.addmm_(w_probs.t(), hc)
                elif grad_w.dtype == acc:
                    # Mixed input dtypes cannot feed a single GEMM; cast `hc`
                    # so both operands share the accumulator dtype.
                    hcf = hc.to(acc)
                    if first:
                        torch.mm(buf.t(), hcf, out=grad_w)
                    else:
                        grad_w.addmm_(buf.t(), hcf)
                    del hcf
                else:
                    # A low-precision weight gradient needs a temporary
                    # product for each vocab block and a converted `hc`.
                    hcf = hc.to(acc)
                    block = max(1, (_CROSSENTROPY_CHUNK_BYTES // 4)
                                // max(1, hcf.shape[1] * hcf.element_size()))
                    for v0 in range(0, w.shape[0], block):
                        v1 = min(w.shape[0], v0 + block)
                        if grad_w_tmp is None or grad_w_tmp.size(0) < v1 - v0:
                            grad_w_tmp = torch.empty(
                                (min(block, w.shape[0]), hcf.shape[1]),
                                dtype=acc, device=w.device)
                        tmp = grad_w_tmp[:v1 - v0]
                        torch.mm(buf[:, v0:v1].t(), hcf, out=tmp)
                        if first:
                            grad_w[v0:v1].copy_(tmp)
                        else:
                            grad_w[v0:v1].add_(tmp)
                    del hcf
            first = False
        if need_w and grad_w.dtype != weight_dtype:
            grad_w = grad_w.to(weight_dtype)
        return (grad_h, grad_w, None, None, None)

def fused_cross_entropy_mean(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, ignore_index: int=-100, chunk_rows: Optional[int]=None) -> torch.Tensor:
    n = targets.numel()
    if hidden.dim() != 2 or hidden.size(0) != n:
        hidden = hidden.reshape(n, -1)
    # `F.cross_entropy`/`nll_loss` expect a 1-D target whose length matches
    # the flattened rows. The logits-path wrapper flattens leading dims, and
    # the fused path must do the same: `hidden` is reshaped above, but a 2-D
    # target left as-is made the chunked nll_loss reject the batch even though
    # the same call is valid through `cross_entropy_mean`.
    if targets.dim() != 1:
        targets = targets.reshape(-1)
    if chunk_rows is None:
        chunk_rows = _cross_entropy_chunk_rows(weight.size(0))
    # The plain pair is taken only when the whole block fits in a single
    # chunk; an explicit small `chunk_rows` still selects the fused node.
    fast_rows = max(1, _CROSS_ENTROPY_FAST_PATH_BYTES // max(1, weight.size(0) * hidden.element_size()))
    if n <= min(fast_rows, chunk_rows):
        dt = _autocast_dtype(hidden.device.type, hidden.dtype, weight.dtype)
        if hidden.dtype != dt:
            hidden = hidden.to(dt)
        if weight.dtype != dt:
            weight = weight.to(dt)
        return F.cross_entropy(F.linear(hidden, weight), targets, ignore_index=ignore_index)
    return _FusedLinearCrossEntropy.apply(hidden, weight, targets, int(ignore_index), int(chunk_rows))

class _LayerScaleResidual(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, residual, scale):
        ctx.save_for_backward(residual, scale)
        return torch.addcmul(x, residual, scale)

    @staticmethod
    def backward(ctx, grad_out):
        residual, scale = ctx.saved_tensors
        grad_scale = (_reduce_products(grad_out, residual)
                      if ctx.needs_input_grad[2] else None)
        grad_residual = grad_out.mul(scale)
        return (grad_out, grad_residual, grad_scale)

class Block(nn.Module):

    def __init__(self, cfg: GPTConfig, layer_idx: int, skip_init: bool=False):
        super().__init__()
        self.layer_idx = layer_idx
        norm_kwargs = build_norm_kwargs(cfg, layer_idx, is_final_norm=False)
        self.norm1 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.attn = CausalSelfAttention(cfg.dim, cfg.n_heads, qk_norm=cfg.qk_norm, alpha_init=cfg.qk_alpha_init, dropout=cfg.dropout, skip_init=skip_init)
        self.norm2 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.mlp = MLP(cfg.dim, int(cfg.dim * cfg.mlp_ratio), dropout=cfg.dropout, skip_init=skip_init)
        self.use_layerscale = cfg.use_layerscale
        if self.use_layerscale:
            self.ls1 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale_init))
            self.ls2 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale_init))

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor]=None, return_stats: bool=False):
        h = self.norm1(x)
        attn_out, logit_stats = self.attn(h, freqs_cis=freqs_cis, return_stats=return_stats)
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
        return (x, logit_stats)

class _SwiGLU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w1, w2, w3):
        lead = x.shape[:-1]
        in_dtypes = (x.dtype, w1.dtype, w2.dtype, w3.dtype)
        dt = _autocast_dtype(x.device.type, x.dtype, w1.dtype, w2.dtype, w3.dtype)
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != dt:
            x2 = x2.to(dt)
        # The backward consumes the same cast weight copies as the forward
        # GEMMs (from the version-keyed cache).
        w1c = _cast_weight_cached(w1, dt)
        w2c = _cast_weight_cached(w2, dt)
        w3c = _cast_weight_cached(w3, dt)
        a = F.linear(x2, w1c)
        b = F.linear(x2, w3c)
        h = F.silu(a, inplace=True)
        h.mul_(b)
        del b
        y = F.linear(h, w2c)
        ctx.save_for_backward(x2, w1c, w2c, w3c)
        ctx.compute_dtype = y.dtype
        ctx.in_dtypes = in_dtypes
        ctx.lead = lead
        return y.view(*lead, -1)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2, w3 = ctx.saved_tensors
        dt = ctx.compute_dtype
        if x.dtype != dt:
            x = x.to(dt)
        grad_out = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = grad_w1 = grad_w2 = grad_w3 = None
        do_x = ctx.needs_input_grad[0]
        do_w1 = ctx.needs_input_grad[1]
        do_w2 = ctx.needs_input_grad[2]
        do_w3 = ctx.needs_input_grad[3]
        need_a = do_x or do_w1
        need_b = do_x or do_w3
        grad_h = grad_out @ w2 if (need_a or need_b) else None
        grad_a = grad_b = None
        a = None
        b = None
        if need_a or need_b or do_w2:
            a = F.linear(x, w1)
            b = F.linear(x, w3)
        if need_a:
            grad_a = torch.ops.aten.silu_backward(grad_h, a)
            grad_a.mul_(b)
        h = None
        if need_b or do_w2:
            h = F.silu(a, inplace=True)
        if need_b:
            grad_h.mul_(h)
            grad_b = grad_h
        del grad_h
        if do_w2:
            h.mul_(b)
            b = None
            grad_w2 = grad_out.t() @ h
        else:
            b = None
        del a, h
        if do_x:
            grad_x = grad_a @ w1
            grad_x.addmm_(grad_b, w3)
        if do_w1:
            grad_w1 = grad_a.t() @ x
            del grad_a
        if do_w3:
            grad_w3 = grad_b.t() @ x
            del grad_b
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
        return (grad_x, grad_w1, grad_w2, grad_w3)

class MLP(nn.Module):

    def __init__(self, dim: int, hidden: int, dropout: float=0.0, skip_init: bool=False):
        super().__init__()
        hidden = int(hidden * 2 / 3)
        hidden = hidden + 7 & ~7
        self.w1 = _make_linear(dim, hidden, False, skip_init)
        self.w2 = _make_linear(hidden, dim, False, skip_init)
        self.w3 = _make_linear(dim, hidden, False, skip_init)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _SwiGLU.apply(x, self.w1.weight, self.w2.weight, self.w3.weight)
        if self.drop.p > 0.0:
            y = self.drop(y)
        return y

class GPT(nn.Module):

    def __init__(self, cfg: GPTConfig, init: bool=True):
        super().__init__()
        self.cfg = cfg
        # The constructor's own kaiming / normal passes are always overwritten
        # by the `apply(self._init_weights)` below, so every Linear / Embedding
        # is built with `skip_init` and the apply pass is the single
        # initialization (norm layers keep their own init untouched).
        # `init=False` skips that pass for callers that immediately re-init
        # (`reset_parameters(seed)`) or load a checkpoint into the module: the
        # build-time pass would be a full second parameter initialization that
        # is then overwritten.
        self.tok_emb = _skip_init(nn.Embedding, cfg.vocab_size, cfg.dim)
        self.pos_emb = _skip_init(nn.Embedding, cfg.max_seq_len, cfg.dim) if not cfg.rotary else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i, skip_init=True) for i in range(cfg.n_layers)])
        self.norm_f = build_norm(cfg.norm_name, cfg.dim, **build_norm_kwargs(cfg, cfg.n_layers, is_final_norm=True))
        self.lm_head = _skip_init(nn.Linear, cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        if init:
            self.apply(self._init_weights)
        if cfg.rotary:
            head_dim = cfg.dim // cfg.n_heads
            freqs = 1.0 / 10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)
            t = torch.arange(cfg.max_seq_len).float()
            freqs = torch.outer(t, freqs)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
            self.register_buffer('freqs_cis', freqs_cis, persistent=False)
        else:
            self.freqs_cis = None
        if not cfg.rotary:
            self.register_buffer('_pos_ids', torch.arange(cfg.max_seq_len), persistent=False)
        else:
            self._pos_ids = None

    def _apply(self, fn, recurse: bool=True):
        """Move/cast the module while keeping the RoPE table complex.

        `nn.Module.to(dtype)` applies the conversion to every buffer, which
        turns the complex `freqs_cis` into a real tensor and discards its
        imaginary part; the next rotary forward then fails at `.imag`. A
        complex RoPE table is an auxiliary constant, not a model parameter, so
        hide it from the generic pass and rebuild it from its real/imaginary
        parts in the destination dtype/device. Non-RoPE models have no such
        buffer and take the normal path unchanged.
        """
        freqs = self._buffers.pop('freqs_cis', None)
        result = super()._apply(fn, recurse)
        if freqs is not None:
            real = fn(freqs.real)
            imag = fn(freqs.imag)
            if real.dtype in (torch.bfloat16, torch.float16):
                # torch.complex has no bfloat16/complex32 support on every
                # backend; the per-call RoPE cast to the activation dtype
                # still happens in attention._rope_pair_for_dtype.
                real = real.float()
                imag = imag.float()
            self.register_buffer('freqs_cis', torch.complex(real, imag),
                                 persistent=False)
        return result

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            if m.weight is self.tok_emb.weight:
                # lm_head shares the input embedding's storage (tied): `apply`
                # already initialized that storage through `tok_emb`.
                return
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def reset_parameters(self, seed: Optional[int]=None):
        if seed is not None:
            # Parameters are usually allocated on the accelerator (callers build
            # inside `torch.device('cuda')`), so `nn.init.normal_` draws from the
            # CUDA generator. Seed it too: otherwise the same reset seed gives
            # different weights depending on how much CUDA RNG the constructor
            # (e.g. norm alpha jitter) consumed before this call.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.apply(self._init_weights)
        return self

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor]=None, return_stats: bool=False, last_position_only: bool=False, return_logits: bool=True, return_hidden: bool=False):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        if self.pos_emb is not None:
            pe = self.pos_emb.weight[:T]
            x.add_(pe if pe.dtype == x.dtype else pe.to(x.dtype))
        if self.drop.p > 0.0:
            x = self.drop(x)
        freqs = self.freqs_cis[:T] if self.freqs_cis is not None else None
        all_logit_stats = []
        if self.cfg.gradient_checkpointing and self.training:
            # The ambient RNG state only has to be preserved for the
            # recomputation when a dropout draw consumes it.
            preserve_rng = self.drop.p > 0.0
            for block in self.blocks:
                x, ls = torch.utils.checkpoint.checkpoint(block, x, freqs, return_stats, use_reentrant=False, preserve_rng_state=preserve_rng)
                if return_stats:
                    all_logit_stats.append(ls)
        else:
            for block in self.blocks:
                x, ls = block(x, freqs_cis=freqs, return_stats=return_stats)
                if return_stats:
                    all_logit_stats.append(ls)
        if last_position_only:
            x = x[:, -1:]
        if return_hidden:
            # Pre-`norm_f` block output; `norm_f` and `lm_head` are
            # per-position, so callers can project position chunks.
            if return_stats:
                return (x, None, all_logit_stats)
            return (x, None)
        x = self.norm_f(x)
        logits = None
        loss = None
        if targets is None:
            if return_logits:
                logits = self.lm_head(x)
        else:
            flat = (targets[:, -1:] if last_position_only else targets).reshape(-1)
            if return_logits:
                logits = self.lm_head(x)
                loss = cross_entropy_mean(logits.view(-1, logits.size(-1)), flat, ignore_index=self.cfg.ignore_index)
            else:
                loss = fused_cross_entropy_mean(x.reshape(-1, x.size(-1)), self.lm_head.weight, flat, ignore_index=self.cfg.ignore_index)
        if not return_stats:
            return (logits, loss)
        return (logits, loss, all_logit_stats)

    def forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Final block output (pre-`norm_f`) for chunked logits projection."""
        out = self.forward(input_ids, targets=None, return_hidden=True)
        return out[0]

    def last_position_logits(self, input_ids: torch.Tensor, return_stats: bool=False) -> torch.Tensor:
        out = self.forward(input_ids, targets=None, return_stats=return_stats, last_position_only=True)
        return out[0][:, -1]

    @torch.no_grad()
    def num_params(self, trainable_only: bool=False) -> int:
        s = 0
        for p in self.parameters():
            if trainable_only and (not p.requires_grad):
                continue
            s += p.numel()
        return s

    @torch.no_grad()
    def norm_param_count(self) -> dict:
        out = {}
        for name, m in self.named_modules():
            if name == 'norm_f' or name.endswith(('.norm1', '.norm2')):
                cls = m.__class__.__name__
                out.setdefault(cls, 0)
                for p in m.parameters(recurse=False):
                    out[cls] += p.numel()
        return out

def build_gpt_124m(norm_name: str='dytcd', init: bool=True, **over) -> GPT:
    cfg = GPTConfig(vocab_size=50304, dim=768, n_layers=12, n_heads=12, max_seq_len=1024, mlp_ratio=4.0, norm_name=norm_name, **over)
    return GPT(cfg, init=init)

def build_gpt_350m(norm_name: str='dytcd', init: bool=True, **over) -> GPT:
    cfg = GPTConfig(vocab_size=50304, dim=1024, n_layers=24, n_heads=16, max_seq_len=1024, mlp_ratio=4.0, norm_name=norm_name, **over)
    return GPT(cfg, init=init)

def build_llama_tiny_1b(norm_name: str='dytcd', init: bool=True, **over) -> GPT:
    over.setdefault('tie_word_embeddings', False)
    over.setdefault('rotary', True)
    cfg = GPTConfig(vocab_size=32000, dim=2048, n_layers=24, n_heads=16, max_seq_len=2048, mlp_ratio=8.0 / 3.0, norm_name=norm_name, **over)
    return GPT(cfg, init=init)
__all__ = ['GPTConfig', 'LayerConfig', 'Block', 'MLP', 'GPT', 'build_norm_kwargs', 'cross_entropy_mean', 'fused_cross_entropy_mean', 'clear_ce_scratch_cache', 'build_gpt_124m', 'build_gpt_350m', 'build_llama_tiny_1b']
