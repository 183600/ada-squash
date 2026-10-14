from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
NORM_REGISTRY: dict[str, type] = {}
_MIN_ALPHA = 1e-4
_MIN_P = 1e-3

def _sum_leading_dims(t: torch.Tensor) -> torch.Tensor:
    if t.ndim <= 1:
        return t.clone()
    return t.sum(dim=tuple(range(t.ndim - 1)))
_REDUCE_CHUNK_ELEMS = 1 << 20

def _reduce_products(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim <= 1:
        return a * b
    if a.ndim == 2 and a.is_contiguous() and b.is_contiguous():
        acc = torch.float32 if a.dtype in (torch.bfloat16, torch.float16) else None
        return (a * b).sum(dim=0, dtype=acc).to(a.dtype)
    rows = a.numel() // a.shape[-1]
    step = max(1, _REDUCE_CHUNK_ELEMS // max(1, a.shape[-1]))
    if rows <= step or not (a.is_contiguous() and b.is_contiguous()):
        return _sum_leading_dims(a * b)
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

def _softsign_forward(xs: torch.Tensor, p) -> torch.Tensor:
    t = xs.abs()
    lo = t.clamp(max=1.0)
    hi = t.clamp_(min=1.0).reciprocal_()
    u = hi.mul_(lo)
    s = u.log_().mul_(p).exp_().log1p_().mul_(-1.0 / p).exp_()
    return lo.mul_(s).copysign_(xs)

def _softsign_backward_pieces(xs: torch.Tensor, p, need_u: bool, need_phi: bool=True, need_g: bool=True):
    t = xs.abs()
    lo = t.clamp(max=1.0)
    r = t.clamp_(min=1.0).reciprocal_()
    a = None
    if need_u:
        log_u = lo.mul(r).log_() if need_phi else lo.mul_(r).log_()
        log_u.mul_(p)
        s = log_u.exp().log1p_().mul_(-1.0 / p).exp_()
        a = log_u.clamp_(min=p * math.log(torch.finfo(log_u.dtype).tiny))
    else:
        s = lo.mul(r) if need_phi else lo.mul_(r)
        s = s.log_().mul_(p).exp_().log1p_().mul_(-1.0 / p).exp_()
    phi = lo.mul_(s).copysign_(xs) if need_phi else None
    if not need_g:
        return (None, phi, a)
    return (r.mul_(s), phi, a)

def _softsign_p_grad(grad_phi: torch.Tensor, a: torch.Tensor, p: torch.Tensor, p_raw: torch.Tensor) -> torch.Tensor:
    sig = a.sigmoid()
    a_sig = a.mul_(sig)
    sig.neg_().log1p_().neg_()
    k = sig.sub_(a_sig)
    acc = torch.float32 if a.dtype in (torch.bfloat16, torch.float16) else None
    red = grad_phi.mul_(k).sum(dtype=acc)
    red = red.div_(p * p)
    red.mul_(p_raw > _MIN_P)
    if red.dtype != p.dtype:
        red = red.to(p.dtype)
    return red

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
            grad_x = g.log_().mul_(p + 1).exp_().mul_(grad_out)
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
        rstd = torch.rsqrt(nrm.square_().div_(x.shape[-1]).add_(eps))
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
            u = grad_out * rstd
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
            grad_weight = _sum_leading_dims((grad_out * rstd).mul(x))
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
        x_scaled = x * alpha
        phi = _softsign_forward(x_scaled, p)
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi.mul_(gamma)
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x_scaled, alpha, gamma, beta]
        if p_is_tensor:
            saved.append(p_raw)
        else:
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x_scaled, alpha, gamma, beta = saved[:4]
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
            g, phi, a = _softsign_backward_pieces(x_scaled, p, need_p_grad, need_phi, need_softsign)
        p_val = p if p_is_tensor else float(p)
        grad_p = grad_gamma = None
        if need_phi:
            grad_phi = phi.mul_(grad_out)
            if need_gamma_grad:
                grad_gamma = _sum_leading_dims(grad_phi)
            if need_p_grad:
                grad_p = _softsign_p_grad(grad_phi.mul_(gamma), a, p, p_raw)
            del phi, a, grad_phi
        grad_x = grad_alpha = None
        if need_softsign:
            grad_x_scaled = g.log_().mul_(p_val + 1).exp_()
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            if ctx.needs_input_grad[1]:
                grad_alpha = _reduce_products(grad_x_scaled, x_scaled).div_(alpha)
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha)
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha, grad_p, grad_gamma, grad_beta)

@register_norm("dytd")
class DyTD(_SquashedAffine):

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, learnable_k: bool=True, beta: bool=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)
        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer('k', torch.tensor(float(k_init)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    @property
    def alpha_l(self) -> torch.Tensor:
        return self.alpha_0 * torch.exp(-self.k * self._depth_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTDForward.apply(x, self.alpha_0, self.k, self.gamma, self.beta, self._depth_ratio)

class _DyTDForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha_0, k, gamma, beta, depth_ratio):
        exp_kdr = torch.exp(-k * depth_ratio)
        a = alpha_0 * exp_kdr
        phi = a.mul(x)
        phi.tanh_()
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi * gamma
        ctx.save_for_backward(x, alpha_0, k, gamma, beta, exp_kdr.detach())
        ctx.depth_ratio = depth_ratio
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha_0, k, gamma, beta, exp_kdr = ctx.saved_tensors
        depth_ratio = ctx.depth_ratio
        a = alpha_0 * exp_kdr
        phi = a.mul(x)
        phi.tanh_()
        w = phi * grad_out
        grad_gamma = None
        if ctx.needs_input_grad[3]:
            grad_gamma = _sum_leading_dims(w)
        w.mul_(phi)
        torch.sub(grad_out, w, out=w)
        w.mul_(gamma)
        grad_a = _dot_all(w, x)
        w.mul_(a)
        del phi
        grad_x = w
        grad_alpha_0 = grad_a * exp_kdr
        grad_k = -grad_a * a * depth_ratio
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha_0, grad_k, grad_gamma, grad_beta, None)

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
        phi = alpha.mul(x)
        phi.tanh_()
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi * gamma
        ctx.save_for_backward(x, alpha, gamma, beta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha, gamma, beta = ctx.saved_tensors
        phi = alpha.mul(x)
        phi.tanh_()
        w = phi * grad_out
        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = _sum_leading_dims(w)
        w.mul_(phi)
        torch.sub(grad_out, w, out=w)
        w.mul_(gamma)
        grad_alpha = _dot_all(w, x)
        w.mul_(alpha)
        del phi
        grad_x = w
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[3]:
            grad_beta = _sum_leading_dims(grad_out)
        return (grad_x, grad_alpha, grad_gamma, grad_beta)

@register_norm("dytcd")
class DyTCD(_SquashedAffine):

    def __init__(self, dim: int, layer_idx: int=0, num_layers: int=12, alpha_0: float=0.9, k_init: float=1.0, learnable_k: bool=True, p_init: float=2.0, learnable_p: bool=True, beta: bool=True):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)
        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer('k', torch.tensor(float(k_init)))
        self.delta = nn.Parameter(torch.zeros(dim))
        if learnable_p:
            self.p = nn.Parameter(torch.tensor(float(p_init)))
        else:
            self.register_buffer('p', torch.tensor(float(p_init)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    @property
    def alpha_per_channel(self) -> torch.Tensor:
        scale = self.alpha_0 * torch.exp(-self.k * self._depth_ratio)
        return torch.addcmul(scale, scale, self.delta).clamp_(min=_MIN_ALPHA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTCDForward.apply(x, self.alpha_0, self.k, self.delta, self.p, self.gamma, self.beta, self._depth_ratio)

class _DyTCDForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha_0, k, delta, p, gamma, beta, depth_ratio):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            p_raw = p
            p = p.clamp(min=_MIN_P)
        exp_kdr = torch.exp(-k * depth_ratio)
        scale = alpha_0 * exp_kdr
        alpha_c = torch.addcmul(scale, scale, delta).clamp_(min=_MIN_ALPHA)
        x_scaled = x * alpha_c
        phi = _softsign_forward(x_scaled, p)
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi.mul_(gamma)
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x_scaled, alpha_0, k, delta, gamma, beta, alpha_c, exp_kdr.detach()]
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
        x_scaled, alpha_0, k, delta, gamma, beta, alpha_c = saved[:7]
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
            g, phi, a = _softsign_backward_pieces(x_scaled, p, need_p_grad, need_phi, need_softsign)
        p_val = p if p_is_tensor else float(p)
        grad_p = grad_gamma = None
        if need_phi:
            grad_phi = phi.mul_(grad_out)
            if need_gamma_grad:
                grad_gamma = _sum_leading_dims(grad_phi)
            if need_p_grad:
                grad_p = _softsign_p_grad(grad_phi.mul_(gamma), a, p, p_raw)
            del phi, a, grad_phi
        grad_x = grad_alpha_0 = grad_k = grad_delta = None
        if need_softsign:
            grad_x_scaled = g.log_().mul_(p_val + 1).exp_()
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            need_alpha_chain = ctx.needs_input_grad[1] or ctx.needs_input_grad[2] or ctx.needs_input_grad[3]
            if need_alpha_chain:
                grad_alpha_c = _reduce_products(grad_x_scaled, x_scaled).div_(alpha_c)
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

class QKSquash(nn.Module):

    def __init__(self, alpha_q_init: float=1.0, alpha_k_init: float=1.0):
        super().__init__()
        self.alpha_q = nn.Parameter(torch.tensor(float(alpha_q_init)))
        self.alpha_k = nn.Parameter(torch.tensor(float(alpha_k_init)))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q = q * self.alpha_q
        k = k * self.alpha_k
        q = torch.tanh(q)
        k = torch.tanh(k)
        return (q, k)
__all__ = ['NORM_REGISTRY', 'register_norm', 'build_norm', 'generalized_softsign', 'LayerNorm', 'RMSNorm', 'DyT', 'DyTC', 'DyTD', 'DyTCD', 'QKSquash']
