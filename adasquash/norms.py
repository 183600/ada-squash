from __future__ import annotations
import math
from typing import Optional
import torch
import torch.nn as nn
NORM_REGISTRY: dict[str, type] = {}
_MIN_ALPHA = 1e-4
_MIN_P = 1e-3
_LOG2E = math.log2(math.e)
_LN2 = math.log(2.0)
_TINY_LOG_CACHE: dict = {}

def _log_tiny(dtype: torch.dtype) -> float:
    """``log(finfo(dtype).tiny)`` per dtype: the softsign p-gradient floor.

    The constant only depends on the dtype; the cache keyed by dtype returns
    the same value every time.
    """
    v = _TINY_LOG_CACHE.get(dtype)
    if v is None:
        v = math.log(torch.finfo(dtype).tiny)
        _TINY_LOG_CACHE[dtype] = v
    return v

def _as_scalar_exponent(p):
    """``p`` as a Python float when it is a one-element CPU tensor.

    Other tensors (including every CUDA tensor) are returned unchanged.
    """
    if isinstance(p, torch.Tensor) and p.device.type == 'cpu' and p.numel() == 1:
        return float(p)
    return p

def _as_scalar_exponent_inplace(g: torch.Tensor, p) -> torch.Tensor:
    """``g ** (p + 1)`` written in place."""
    if isinstance(p, torch.Tensor):
        return g.log2_().mul_(p + 1).exp2_()
    return g.pow_(p + 1)

def _scale_needs_sign_fix(scale: torch.Tensor) -> bool:
    """Whether the compensating `±1` multiply has to run.

    Multiplying by `+1.0` is exact, so when no `scale` entry is negative the
    correction is a no-op; non-CPU tensors keep the multiply unconditionally.
    `signbit` (rather than `< 0`) is what makes `-0.0` count as negative.
    """
    if scale.device.type != 'cpu':
        return True
    return bool(torch.signbit(scale).any())

def _sum_leading_dims(t: torch.Tensor) -> torch.Tensor:
    if t.ndim <= 1:
        return t.clone()
    return t.sum(dim=tuple(range(t.ndim - 1)))
_REDUCE_CHUNK_ELEMS = 1 << 22

def _reduce_products(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim <= 1:
        return a * b
    if a.shape[-1] == 0:
        return a.new_zeros(a.shape[-1])
    if a.ndim == 2 and a.is_contiguous() and b.is_contiguous() and a.numel() <= _REDUCE_CHUNK_ELEMS:
        # Larger shapes take the chunked branches below.
        acc = torch.float32 if a.dtype in (torch.bfloat16, torch.float16) else None
        return (a * b).sum(dim=0, dtype=acc).to(a.dtype)
    if a.is_contiguous() and b.is_contiguous():
        rows = a.numel() // a.shape[-1]
        step = max(1, _REDUCE_CHUNK_ELEMS // max(1, a.shape[-1]))
        if rows <= step:
            return _sum_leading_dims(a * b)
        acc_dtype = torch.promote_types(a.dtype, torch.float32) if a.dtype in (torch.bfloat16, torch.float16) else a.dtype
        a2 = a.view(rows, -1)
        b2 = b.view(rows, -1)
        total = torch.zeros(a2.shape[1], dtype=acc_dtype, device=a.device)
        for start in range(0, rows, step):
            part = (a2[start:start + step] * b2[start:start + step]).sum(dim=0, dtype=acc_dtype)
            total.add_(part)
        if total.dtype != a.dtype:
            total = total.to(a.dtype)
        return total
    if a.numel() > _REDUCE_CHUNK_ELEMS:
        rows = a.numel() // a.shape[-1]
        step = max(1, _REDUCE_CHUNK_ELEMS // max(1, a.shape[-1]))
        acc_dtype = torch.promote_types(a.dtype, torch.float32) if a.dtype in (torch.bfloat16, torch.float16) else a.dtype
        a2 = a.reshape(rows, -1)
        b2 = b.reshape(rows, -1)
        total = torch.zeros(a2.shape[1], dtype=acc_dtype, device=a.device)
        for start in range(0, rows, step):
            part = (a2[start:start + step] * b2[start:start + step]).sum(dim=0, dtype=acc_dtype)
            total.add_(part)
        if total.dtype != a.dtype:
            total = total.to(a.dtype)
        return total
    return _sum_leading_dims(a * b)

def _dot_all(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.is_contiguous() and b.is_contiguous():
        return torch.dot(a.reshape(-1), b.reshape(-1))
    return (a * b).sum()

def register_norm(name: str):

    def _decorator(cls):
        NORM_REGISTRY[name] = cls
        cls._norm_name = name
        return cls
    return _decorator

def _softsign_core(xs: torch.Tensor, p, scale=None, scale_nonneg: bool=False):
    """Compute shared intermediates for softsign forward/backward.

    Returns (t, lo, r, u) where:
      t = |xs| or |xs * scale|
      lo = clamp(t, max=1)
      r = 1 / clamp(t, min=1)
      u = lo * r = min(|x|, 1) / max(|x|, 1)  (always <= 1)
    """
    p = _as_scalar_exponent(p)
    if scale is None:
        t = xs.abs()
    else:
        t = xs.mul(scale).abs_()
    lo = t.clamp(max=1.0)
    r = t.clamp_(min=1.0).reciprocal_()
    u = lo.mul(r)
    return t, lo, r, u, p


def _compute_s_from_u(u: torch.Tensor, p) -> torch.Tensor:
    """Compute s = (u^p + 1)^(-1/p) from u, handling tensor and scalar p."""
    if isinstance(p, torch.Tensor):
        neg_log2e_p = -_LOG2E / p.to(torch.promote_types(u.dtype, p.dtype))
        return u.log2_().mul_(p).exp2_().log1p_().mul_(neg_log2e_p).exp2_()
    return u.pow_(p).add_(1.0).pow_(-1.0 / p)


def _softsign_forward(xs: torch.Tensor, p, scale=None, scale_nonneg: bool=False) -> torch.Tensor:
    t, lo, r, u, p = _softsign_core(xs, p, scale, scale_nonneg)
    s = _compute_s_from_u(u, p)
    out = s.mul_(lo).copysign_(xs)
    if scale is not None and (not scale_nonneg) and _scale_needs_sign_fix(scale):
        out.mul_(torch.where(torch.signbit(scale), -1.0, 1.0))
    return out


def _softsign_backward_pieces(xs: torch.Tensor, p, need_u: bool, need_phi: bool=True, need_g: bool=True, scale=None, scale_nonneg: bool=False):
    t, lo, r, u, p = _softsign_core(xs, p, scale, scale_nonneg)
    a = None
    if need_u:
        # `lo` only feeds `phi`; when that output is not requested it is not
        # needed separately.
        if isinstance(p, torch.Tensor):
            q = u.log2_().mul_(p)
            a = q.mul(_LN2).clamp_(min=p * _log_tiny(q.dtype))
            s = q.exp2_().log1p_().mul_(-_LOG2E / p.to(torch.promote_types(q.dtype, p.dtype))).exp2_()
        else:
            s = torch.pow(u, p).add_(1.0).pow_(-1.0 / p)
            a = u.log2_().mul_(p * _LN2).clamp_(min=p * _log_tiny(u.dtype))
    else:
        s = u
        if isinstance(p, torch.Tensor):
            s.log2_().mul_(p).exp2_().log1p_().mul_(-_LOG2E / p.to(torch.promote_types(s.dtype, p.dtype))).exp2_()
        else:
            s.pow_(p).add_(1.0).pow_(-1.0 / p)
    phi = None
    if need_phi:
        if need_g:
            phi = lo.mul_(s).copysign_(xs)
        else:
            phi = s.mul_(lo).copysign_(xs)
        if scale is not None and (not scale_nonneg) and _scale_needs_sign_fix(scale):
            phi.mul_(torch.where(torch.signbit(scale), -1.0, 1.0))
    if not need_g:
        return (None, phi, a)
    return (r.mul_(s), phi, a)

def _softsign_p_grad(grad_phi: torch.Tensor, a: torch.Tensor, p: torch.Tensor, p_raw: torch.Tensor, gamma: Optional[torch.Tensor]=None) -> torch.Tensor:
    """Gradient w.r.t. ``p`` from the cotangent of the squashed output.

    ``a`` is the clamped ``p * ln(u)`` from ``_softsign_backward_pieces``; it is
    overwritten with the ``a * sigmoid(a)`` factor and consumed by the end of
    the call.
    """
    if p.numel() != 1 or p_raw.numel() != 1:
        raise ValueError(
            "the generalized-softsign exponent must be a scalar (one element), "
            f"got p.shape={tuple(p.shape)}")
    p_scalar = p.reshape(())
    p_raw_scalar = p_raw.reshape(())
    if gamma is not None:
        grad_phi.mul_(gamma)
    sig = a.sigmoid()
    a.mul_(sig)
    sig.neg_().log1p_().neg_()
    k = sig.sub_(a)
    acc = torch.float32 if a.dtype in (torch.bfloat16, torch.float16) else None
    if acc is None:
        red = _dot_all(grad_phi, k)
    else:
        red = (grad_phi * k).sum(dtype=acc)
    # Out-of-place with a 0-dim p: an in-place division by a one-element p
    # would try to write a [1] broadcast back into the 0-dim reduction. For a
    # one-element tensor the scalar reduction is still the correct value; it
    # is reshaped to p's shape only for autograd's returned gradient.
    red = red / (p_scalar * p_scalar)
    red = red * (p_raw_scalar > _MIN_P)
    if red.dtype != p.dtype:
        red = red.to(p.dtype)
    return red.reshape(p.shape)

def build_norm(name: str, dim: int, **kwargs) -> nn.Module:
    if name not in NORM_REGISTRY:
        raise KeyError(f"Unknown norm '{name}'. Registered: {list(NORM_REGISTRY)}")
    return NORM_REGISTRY[name](dim, **kwargs)

class _SquashedAffine(nn.Module):
    pass

class _GeneralizedSoftsign(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, p):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            p_raw = p
            p = p.clamp(min=_MIN_P)
        else:
            # Frozen p (`learnable_p=False`) is stored as a Python float, so
            # the tensor-only clamp above would let p <= 0 reach the closed
            # form: p=0 divides by zero and p<0 makes |x|^p explode at zero.
            # D2's floor applies to every representation of the exponent.
            p = max(float(p), _MIN_P)
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x]
        if p_is_tensor:
            saved.append(p_raw)
        else:
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        return _softsign_forward(x, p)

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x = saved[0]
        if ctx.p_is_tensor:
            p_raw = saved[1]
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value
        grad_x = grad_p = None
        need_x = ctx.needs_input_grad[0]
        need_p = ctx.needs_input_grad[1] and ctx.need_p_grad
        if need_x or need_p:
            g, phi, a = _softsign_backward_pieces(x, p, need_p, need_p, need_x)
        if need_x:
            grad_x = _as_scalar_exponent_inplace(g, _as_scalar_exponent(p)).mul_(grad_out)
        if need_p:
            grad_p = _softsign_p_grad(phi.mul_(grad_out), a, p, p_raw)
        return (grad_x, grad_p)

def generalized_softsign(x: torch.Tensor, p) -> torch.Tensor:
    return _GeneralizedSoftsign.apply(x, p)

@register_norm("ln")
class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)

@register_norm("rmsnorm")
class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNorm.apply(x, self.weight, self.eps)

class _RMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, eps):
        nrm = torch.linalg.vector_norm(x, 2, dim=-1, keepdim=True)
        rstd = nrm.square_().div_(x.shape[-1]).add_(eps).rsqrt_()
        ctx.save_for_backward(x, rstd, weight)
        out = x * rstd
        if out.dtype == weight.dtype:
            out.mul_(weight)
        else:
            out = out * weight
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, rstd, weight = ctx.saved_tensors
        need_x = ctx.needs_input_grad[0]
        need_w = ctx.needs_input_grad[1]
        grad_x = None
        grad_weight = None
        if need_x:
            u = grad_out.mul(rstd)
            t = u.mul(x)
            if need_w:
                grad_weight = _sum_leading_dims(t)
            if t.dtype == weight.dtype:
                m = t.mul_(weight).mean(dim=-1, keepdim=True)
            else:
                m = (t * weight).mean(dim=-1, keepdim=True)
            del t
            if u.dtype == weight.dtype:
                u.mul_(weight)
            else:
                u = u * weight
            if m.dtype == rstd.dtype:
                corr = m.mul_(rstd).mul_(rstd)
            else:
                corr = m * rstd * rstd
            out_dtype = torch.promote_types(u.dtype, torch.promote_types(x.dtype, corr.dtype))
            if out_dtype == u.dtype:
                u.addcmul_(x, corr, value=-1)
            else:
                u = torch.addcmul(u, x, corr, value=-1)
            grad_x = u
        elif need_w:
            grad_weight = _sum_leading_dims(grad_out.mul(rstd).mul_(x))
        if grad_x is not None and grad_x.dtype != x.dtype:
            grad_x = grad_x.to(x.dtype)
        if grad_weight is not None and grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)
        return (grad_x, grad_weight, None)

@register_norm("dytc")
class DyTC(_SquashedAffine):

    def __init__(self, dim: int, alpha_init: float=0.5, p_init: float=2.0, learnable_p: bool=True, beta: bool=True, alpha_jitter: float=0.05):
        super().__init__()
        alpha = torch.full((dim,), float(alpha_init))
        if alpha_jitter > 0:
            alpha = alpha + alpha_jitter * torch.randn(dim) * alpha_init
        self.alpha = nn.Parameter(alpha)
        self.p = nn.Parameter(torch.tensor(float(p_init))) if learnable_p else float(p_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTCForward.apply(x, self.alpha, self.p, self.gamma, self.beta)

class _DyTCForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, p, gamma, beta):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            p_raw = p
            p = p.clamp(min=_MIN_P)
        else:
            # Frozen p is a Python float; apply the same D2 floor as tensor p.
            p = max(float(p), _MIN_P)
        phi = _softsign_forward(x, p, alpha)
        if beta is not None:
            out = phi.mul_(gamma).add_(beta)
        else:
            out = phi.mul_(gamma)
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x, alpha, gamma, beta]
        if p_is_tensor:
            saved.append(p_raw)
        else:
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, alpha, gamma, beta = saved[:4]
        p_is_tensor = ctx.p_is_tensor
        p_raw = saved[4] if p_is_tensor else None
        if p_is_tensor:
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value
        need_softsign = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[2]
        need_gamma_grad = ctx.needs_input_grad[3]
        need_phi = need_p_grad or need_gamma_grad
        phi = None
        a = None
        if need_softsign or need_phi:
            g, phi, a = _softsign_backward_pieces(x, p, need_p_grad, need_phi, need_softsign, alpha)
        p_val = _as_scalar_exponent(p)
        grad_p = grad_gamma = None
        if need_phi:
            grad_phi = phi.mul_(grad_out)
            if need_gamma_grad:
                grad_gamma = _sum_leading_dims(grad_phi)
            if need_p_grad:
                grad_p = _softsign_p_grad(grad_phi, a, p, p_raw, gamma)
            del phi, a, grad_phi
        grad_x = grad_alpha = None
        if need_softsign:
            grad_x_scaled = _as_scalar_exponent_inplace(g, p_val)
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            if ctx.needs_input_grad[1]:
                grad_alpha = _reduce_products(grad_x_scaled, x)
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha)
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha, grad_p, grad_gamma, grad_beta)

@register_norm("dytd")
class DyTD(_SquashedAffine):

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, learnable_k: bool=True, beta: bool=True, k_min: Optional[float]=0.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)
        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer('k', torch.tensor(float(k_init)))
        # Depth-decay rate floor. ``None`` restores the old unconstrained
        # behaviour (k may go negative and alpha_l then *grows* with depth);
        # the default 0.0 keeps the schedule non-increasing in depth, so the
        # "depth-conditioned" claim holds not only at init but for every
        # point the optimizer visits.
        self.k_min = k_min
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def _k_effective(self):
        k = self.k
        if self.k_min is None:
            return k
        if isinstance(k, torch.Tensor):
            return k.clamp(min=self.k_min)
        return max(float(k), float(self.k_min))

    @property
    def alpha_l(self) -> torch.Tensor:
        return self.alpha_0 * torch.exp(-self._k_effective() * self._depth_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTDForward.apply(x, self.alpha_0, self._k_effective(), self.gamma, self.beta, self._depth_ratio)

class _DyTDForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha_0, k, gamma, beta, depth_ratio):
        exp_kdr = torch.exp(k.mul(-depth_ratio))
        a = alpha_0 * exp_kdr
        phi = x.mul(a).tanh_()
        if beta is not None:
            out = phi.mul_(gamma).add_(beta)
        else:
            out = phi.mul_(gamma)
        ctx.save_for_backward(x, alpha_0, k, gamma, beta, exp_kdr.detach())
        ctx.depth_ratio = depth_ratio
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha_0, k, gamma, beta, exp_kdr = ctx.saved_tensors
        depth_ratio = ctx.depth_ratio
        a = alpha_0 * exp_kdr
        phi = x.mul(a).tanh_()
        w = phi * grad_out
        grad_gamma = None
        if ctx.needs_input_grad[3]:
            grad_gamma = _sum_leading_dims(w)
        w.mul_(phi)
        del phi
        torch.sub(grad_out, w, out=w)
        w.mul_(gamma)
        need_schedule = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        grad_a = _dot_all(w, x) if need_schedule else None
        w.mul_(a)
        grad_x = w
        grad_alpha_0 = grad_a * exp_kdr if ctx.needs_input_grad[1] else None
        grad_k = -grad_a * a * depth_ratio if ctx.needs_input_grad[2] else None
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha_0, grad_k, grad_gamma, grad_beta, None)

@register_norm("dytd_tanh")
class DyTDTanh(nn.Module):
    """Ablation-only comparator: DyT-D with `tanh` (plain autograd).

    Same depth schedule as `DyTD` (`alpha_l = alpha_0 * exp(-k * l/L)`),
    same `gamma`/`beta`, same LayerScale residual (it lives in `Block`):
    the only change is using plain autograd `tanh` instead of the fused
    `_DyTDForward` kernel. This is a control for the factorial design,
    not part of the research core.

    Frozen `k` is supported via `learnable_k=False` (registers a buffer).
    """

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, learnable_k: bool=True, beta: bool=True, k_min: Optional[float]=0.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)
        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer('k', torch.tensor(float(k_init)))
        self.k_min = k_min
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def _k_effective(self):
        k = self.k
        if self.k_min is None:
            return k
        if isinstance(k, torch.Tensor):
            return k.clamp(min=self.k_min)
        return max(float(k), float(self.k_min))

    @property
    def alpha_l(self) -> torch.Tensor:
        return self.alpha_0 * torch.exp(-self._k_effective() * self._depth_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.tanh(self.alpha_l * x)
        if self.beta is not None:
            return phi * self.gamma + self.beta
        return phi * self.gamma

@register_norm("dyt")
class DyT(_SquashedAffine):

    def __init__(self, dim: int, alpha_init: float=0.6, beta: bool=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTForward.apply(x, self.alpha, self.gamma, self.beta)

class _DyTForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha, gamma, beta):
        phi = x.mul(alpha).tanh_()
        if beta is not None:
            out = phi.mul_(gamma).add_(beta)
        else:
            out = phi.mul_(gamma)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha, gamma, beta = ctx.saved_tensors
        phi = x.mul(alpha).tanh_()
        w = phi * grad_out
        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = _sum_leading_dims(w)
        w.mul_(phi)
        del phi
        torch.sub(grad_out, w, out=w)
        w.mul_(gamma)
        grad_alpha = _dot_all(w, x) if ctx.needs_input_grad[1] else None
        w.mul_(alpha)
        grad_x = w
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[3]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha, grad_gamma, grad_beta)

@register_norm("dytcd")
class DyTCD(_SquashedAffine):

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, learnable_k: bool=True, p_init: float=2.0, learnable_p: bool=True, beta: bool=True, learnable_delta: bool=True, k_min: Optional[float]=0.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)
        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer('k', torch.tensor(float(k_init)))
        self.k_min = k_min
        if learnable_delta:
            self.delta = nn.Parameter(torch.zeros(dim))
        else:
            # Per-channel axis off for the controlled ablation: `delta` is a
            # zero buffer rather than a parameter, so `alpha_c == alpha_l`
            # exactly. The kernel is untouched - it only sees `delta.requires_
            # grad == False` and returns no gradient for it - which keeps the
            # depth schedule, the softsign activation and LayerScale identical
            # to the full model. (Frozen `k` / `p` already use this pattern.)
            self.register_buffer('delta', torch.zeros(dim))
        if learnable_p:
            self.p = nn.Parameter(torch.tensor(float(p_init)))
        else:
            self.register_buffer('p', torch.tensor(float(p_init)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def _k_effective(self):
        k = self.k
        if self.k_min is None:
            return k
        if isinstance(k, torch.Tensor):
            return k.clamp(min=self.k_min)
        return max(float(k), float(self.k_min))

    @property
    def alpha_per_channel(self) -> torch.Tensor:
        scale = self.alpha_0 * torch.exp(-self._k_effective() * self._depth_ratio)
        return torch.addcmul(scale, scale, self.delta).clamp_(min=_MIN_ALPHA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTCDForward.apply(x, self.alpha_0, self._k_effective(), self.delta, self.p, self.gamma, self.beta, self._depth_ratio)

class _DyTCDForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha_0, k, delta, p, gamma, beta, depth_ratio):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            p_raw = p
            p = p.clamp(min=_MIN_P)
        else:
            # A frozen exponent buffer can still be a plain float on external
            # call paths; keep the D2 floor for both representations.
            p = max(float(p), _MIN_P)
        exp_kdr = torch.exp(k.mul(-depth_ratio))
        scale = alpha_0 * exp_kdr
        alpha_c = torch.addcmul(scale, scale, delta).clamp_(min=_MIN_ALPHA)
        phi = _softsign_forward(x, p, alpha_c, scale_nonneg=True)
        if beta is not None:
            out = phi.mul_(gamma).add_(beta)
        else:
            out = phi.mul_(gamma)
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x, alpha_0, k, delta, gamma, beta, alpha_c, exp_kdr.detach()]
        if p_is_tensor:
            saved.append(p_raw)
        else:
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        ctx.depth_ratio = depth_ratio
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, alpha_0, k, delta, gamma, beta, alpha_c = saved[:7]
        exp_kdr = saved[7]
        depth_ratio = ctx.depth_ratio
        p_is_tensor = ctx.p_is_tensor
        p_raw = saved[8] if p_is_tensor else None
        if p_is_tensor:
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value
        need_softsign = ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[4]
        need_gamma_grad = ctx.needs_input_grad[5]
        need_phi = need_p_grad or need_gamma_grad
        phi = None
        a = None
        if need_softsign or need_phi:
            g, phi, a = _softsign_backward_pieces(x, p, need_p_grad, need_phi, need_softsign, alpha_c, scale_nonneg=True)
        p_val = _as_scalar_exponent(p)
        grad_p = grad_gamma = None
        if need_phi:
            grad_phi = phi.mul_(grad_out)
            if need_gamma_grad:
                grad_gamma = _sum_leading_dims(grad_phi)
            if need_p_grad:
                grad_p = _softsign_p_grad(grad_phi, a, p, p_raw, gamma)
            del phi, a, grad_phi
        grad_x = grad_alpha_0 = grad_k = grad_delta = None
        if need_softsign:
            grad_x_scaled = _as_scalar_exponent_inplace(g, p_val)
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            need_alpha_chain = ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]
            if need_alpha_chain:
                grad_alpha_c = _reduce_products(grad_x_scaled, x)
                grad_alpha_c.mul_(alpha_c > _MIN_ALPHA)
                need_delta = ctx.needs_input_grad[3]
                if need_delta or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                    scale = alpha_0 * exp_kdr
                    if need_delta:
                        grad_delta = grad_alpha_c * scale
                    if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                        grad_scale = grad_alpha_c * (1.0 + delta)
                        if ctx.needs_input_grad[1]:
                            grad_alpha_0 = grad_scale * exp_kdr
                        if ctx.needs_input_grad[2]:
                            grad_k = -grad_scale * scale * depth_ratio
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha_c)
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[6]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha_0, grad_k, grad_delta, grad_p, grad_gamma, grad_beta, None)

@register_norm("dytcd_tanh")
class DyTCDTanh(nn.Module):
    """Ablation-only comparator: DyT-CD with `tanh` instead of the softsign family.

    Same per-channel x depth alpha as `DyTCD` (`alpha_c,l = alpha_0 *
    exp(-k*l/L) * (1 + delta)`, clamped at `_MIN_ALPHA`), same `gamma`/`beta`,
    same LayerScale residual (it lives in `Block`): the only change is the
    squash. This is what makes the activation family (generalized softsign vs
    tanh) a leave-one-out axis of the main proposal - the `dytd_full`
    reference cannot isolate it because it also removes the per-channel
    factor.

    `tanh` primitives are only used here; the fused-kernel family above stays
    untouched. Like `LayerNorm`, this comparator is plain autograd (it is a
    control, not part of the research core), so its gradients are torch's own.
    """

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, learnable_k: bool=True, beta: bool=True, learnable_delta: bool=True, k_min: Optional[float]=0.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)
        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer('k', torch.tensor(float(k_init)))
        self.k_min = k_min
        if learnable_delta:
            self.delta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer('delta', torch.zeros(dim))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def _k_effective(self):
        k = self.k
        if self.k_min is None:
            return k
        if isinstance(k, torch.Tensor):
            return k.clamp(min=self.k_min)
        return max(float(k), float(self.k_min))

    @property
    def alpha_per_channel(self) -> torch.Tensor:
        scale = self.alpha_0 * torch.exp(-self._k_effective() * self._depth_ratio)
        return torch.addcmul(scale, scale, self.delta).clamp_(min=_MIN_ALPHA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.tanh(self.alpha_per_channel * x)
        if self.beta is not None:
            return phi * self.gamma + self.beta
        return phi * self.gamma
@register_norm("dyts")
class DyTS(_SquashedAffine):
    """Scalar-alpha generalized softsign: the missing cell of the
    {scalar, per-channel} x {tanh, softsign} factorial.

    Identical to DyTC except alpha is a single scalar, so ``dyt -> dyts``
    isolates the activation family and ``dyts -> dytc`` isolates the
    per-channel axis (single-variable contrasts the old matrix mixed).
    """

    def __init__(self, dim: int, alpha_init: float=0.5, p_init: float=2.0, learnable_p: bool=True, beta: bool=True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.p = nn.Parameter(torch.tensor(float(p_init))) if learnable_p else float(p_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTSForward.apply(x, self.alpha, self.p, self.gamma, self.beta)

class _DyTSForward(torch.autograd.Function):
    """DyTS forward: same math as _DyTCForward but with a scalar alpha, so
    the alpha gradient is a full reduction (_dot_all) instead of per-channel."""

    @staticmethod
    def forward(ctx, x, alpha, p, gamma, beta):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            p_raw = p
            p = p.clamp(min=_MIN_P)
        phi = _softsign_forward(x, p, alpha)
        if beta is not None:
            out = phi.mul_(gamma).add_(beta)
        else:
            out = phi.mul_(gamma)
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x, alpha, gamma, beta]
        if p_is_tensor:
            saved.append(p_raw)
        else:
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, alpha, gamma, beta = saved[:4]
        p_is_tensor = ctx.p_is_tensor
        p_raw = saved[4] if p_is_tensor else None
        if p_is_tensor:
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value
        need_softsign = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[2]
        need_gamma_grad = ctx.needs_input_grad[3]
        need_phi = need_p_grad or need_gamma_grad
        phi = None
        a = None
        if need_softsign or need_phi:
            g, phi, a = _softsign_backward_pieces(x, p, need_p_grad, need_phi, need_softsign, alpha)
        p_val = p if p_is_tensor else float(p)
        grad_p = grad_gamma = None
        if need_phi:
            grad_phi = phi.mul_(grad_out)
            if need_gamma_grad:
                grad_gamma = _sum_leading_dims(grad_phi)
            if need_p_grad:
                grad_p = _softsign_p_grad(grad_phi, a, p, p_raw, gamma)
            del phi, a, grad_phi
        grad_x = grad_alpha = None
        if need_softsign:
            grad_x_scaled = g.log2_().mul_(p_val + 1).exp2_()
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            if ctx.needs_input_grad[1]:
                grad_alpha = _dot_all(grad_x_scaled, x)
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha)
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha, grad_p, grad_gamma, grad_beta)

@register_norm("dytp")
class DyTP(_SquashedAffine):
    """Per-channel-alpha tanh: the other missing cell of the
    {scalar, per-channel} x {tanh, softsign} factorial.

    Identical to DyT except alpha is a per-channel vector, so ``dyt -> dytp``
    isolates the per-channel axis under the *tanh* family.
    """

    def __init__(self, dim: int, alpha_init: float=0.5, beta: bool=True, alpha_jitter: float=0.0):
        super().__init__()
        alpha = torch.full((dim,), float(alpha_init))
        if alpha_jitter > 0:
            alpha = alpha + alpha_jitter * torch.randn(dim) * alpha_init
        self.alpha = nn.Parameter(alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTPForward.apply(x, self.alpha, self.gamma, self.beta)

class _DyTPForward(torch.autograd.Function):
    """DyTP forward: same math as _DyTForward but with a per-channel alpha,
    so the alpha gradient is a per-channel reduction (_reduce_products)."""

    @staticmethod
    def forward(ctx, x, alpha, gamma, beta):
        phi = x.mul(alpha).tanh_()
        if beta is not None:
            out = phi.mul_(gamma).add_(beta)
        else:
            out = phi.mul_(gamma)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha, gamma, beta = ctx.saved_tensors
        phi = x.mul(alpha).tanh_()
        w = phi * grad_out
        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = _sum_leading_dims(w)
        w.mul_(phi)
        torch.sub(grad_out, w, out=w)
        w.mul_(gamma)
        grad_alpha = _reduce_products(w, x) if ctx.needs_input_grad[1] else None
        del phi
        grad_x = w.mul_(alpha) if ctx.needs_input_grad[0] else None
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[3]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha, grad_gamma, grad_beta)

@register_norm("dytdi")
class DyTDI(DyT):
    """Depth-*init* control for DyT-D: a plain DyT whose scalar alpha is
    *initialized* at the DyT-D schedule value alpha_0 * exp(-k l / L) and
    then learned freely, with no (alpha_0, k) reparameterization.

    Against ``dytd`` this isolates "depth prior at init + plain
    parameterization" from "depth-decay reparameterization": if dytd wins
    only via a better-conditioned init, dytdi matches it.
    """

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, beta: bool=True):
        ratio = float(layer_idx) / float(max(1, num_layers))
        super().__init__(dim, alpha_init=float(alpha_0) * math.exp(-float(k_init) * ratio), beta=beta)

@register_norm("dytcdi")
class DyTCDI(DyTC):
    """Depth-*init* control for DyT-CD: DyT-C (per-channel alpha, generalized
    softsign, learnable p) whose per-channel alpha is *initialized* at the
    DyT-CD schedule value alpha_0 * exp(-k l / L) (delta = 0) and then
    learned freely.

    Against ``dytcd`` this isolates the depth-decay reparameterization from
    the depth prior at init, activation family and per-channel axis held
    fixed.
    """

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, p_init: float=2.0, learnable_p: bool=True, beta: bool=True):
        ratio = float(layer_idx) / float(max(1, num_layers))
        super().__init__(dim, alpha_init=float(alpha_0) * math.exp(-float(k_init) * ratio), p_init=p_init, learnable_p=learnable_p, beta=beta, alpha_jitter=0.0)

class _QKSquashTanh(torch.autograd.Function):
    """tanh(alpha * Q) and tanh(alpha * K) as one autograd node.

    Same closed form as ``torch.tanh(q * alpha_q)`` (D4): the backward
    re-derives the squashed output from the raw Q/K, so the node returns both
    the per-tensor values and the alpha_q / alpha_k gradients while keeping
    the public tanh semantics.
    """

    @staticmethod
    def forward(ctx, q, k, alpha_q, alpha_k):
        qo = q.mul(alpha_q)
        qo.tanh_()
        ko = k.mul(alpha_k)
        ko.tanh_()
        ctx.save_for_backward(q, k, alpha_q, alpha_k)
        return (qo, ko)

    @staticmethod
    def _side(g, x, alpha, need_x, need_alpha):
        xo = x.mul(alpha)
        xo.tanh_()
        gx = torch.ops.aten.tanh_backward(g, xo)
        grad_alpha = _reduce_products(gx, x) if need_alpha else None
        return (gx.mul_(alpha) if need_x else None), grad_alpha

    @staticmethod
    def backward(ctx, grad_qo, grad_ko):
        q, k, alpha_q, alpha_k = ctx.saved_tensors
        need_q, need_k = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need_aq, need_ak = ctx.needs_input_grad[2], ctx.needs_input_grad[3]
        grad_q = grad_k = grad_alpha_q = grad_alpha_k = None
        if need_q or need_aq:
            grad_q, grad_alpha_q = _QKSquashTanh._side(
                grad_qo, q, alpha_q, need_q, need_aq)
        if need_k or need_ak:
            grad_k, grad_alpha_k = _QKSquashTanh._side(
                grad_ko, k, alpha_k, need_k, need_ak)
        return (grad_q, grad_k, grad_alpha_q, grad_alpha_k)

class QKSquash(nn.Module):

    def __init__(self, alpha_q_init: float=1.0, alpha_k_init: float=1.0):
        super().__init__()
        self.alpha_q = nn.Parameter(torch.tensor(float(alpha_q_init)))
        self.alpha_k = nn.Parameter(torch.tensor(float(alpha_k_init)))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return _QKSquashTanh.apply(q, k, self.alpha_q, self.alpha_k)
__all__ = ['NORM_REGISTRY', 'register_norm', 'build_norm', 'generalized_softsign', 'LayerNorm', 'RMSNorm', 'DyT', 'DyTS', 'DyTP', 'DyTC', 'DyTD', 'DyTDTanh', 'DyTDI', 'DyTCD', 'DyTCDTanh', 'DyTCDI', 'QKSquash']
