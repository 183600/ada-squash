"""Statistic-free adaptive squashing normalization layers.
This module implements the DyT++ normalization family:

* **Statistic-based baselines**: ``LayerNorm``, ``RMSNorm``
* **Generalized-softsign family**: scalar/per-channel alpha with softsign
  - ``DyT`` / ``DyT-S`` (scalar alpha)
  - ``DyT-P`` (per-channel alpha under tanh)
  - ``DyT-C`` (per-channel alpha with generalized softsign)
  - ``DyT-D`` / ``DyT-CD`` (depth-conditioned variants)
* **Attention logit squashing**: ``QKSquash``

All operations are hand-written ``torch.autograd.Function`` subclasses:
the forward pass uses closed-form expressions; the backward recomputes
needed intermediates. Access via :func:`build_norm`, :data:`NORM_REGISTRY`,
and :func:`generalized_softsign`.

Numerical stability floors (``_MIN_P``, ``_MIN_ALPHA``) and the
``reciprocal-on-max(|x|, 1)`` softsign formulation are fixed design decisions.
See ``docs/design_notes.md`` (D2, D3).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

NORM_REGISTRY: dict[str, type] = {}

_MIN_ALPHA = 1e-4
_MIN_P = 1e-3

_LOG2E = math.log2(math.e)
_LN2 = math.log(2.0)

_TINY_LOG_CACHE: dict = {}


def _log_tiny(dtype: torch.dtype) -> float:
    v = _TINY_LOG_CACHE.get(dtype)
    if v is None:
        v = math.log(torch.finfo(dtype).tiny)
        _TINY_LOG_CACHE[dtype] = v
    return v


def _as_scalar_exponent(p):
    if isinstance(p, torch.Tensor) and p.device.type == "cpu" and p.numel() == 1:
        return p.detach().item()
    return p


def _as_scalar_exponent_inplace(g: torch.Tensor, p) -> torch.Tensor:
    if isinstance(p, torch.Tensor):
        return g.log2_().mul_(p + 1).exp2_()
    return g.pow_(p + 1)


def _scale_needs_sign_fix(scale: torch.Tensor) -> bool:
    if scale.device.type != "cpu":
        return True
    return bool(torch.signbit(scale).any())


def _sum_leading_dims(t: torch.Tensor) -> torch.Tensor:
    if t.ndim <= 1:
        return t.clone()
    return t.sum(dim=tuple(range(t.ndim - 1)))


_REDUCE_CHUNK_ELEMS = 1 << 22


def _chunked_product_sum(a2: torch.Tensor, b2: torch.Tensor, step: int, acc_dtype: torch.dtype) -> torch.Tensor:
    total = torch.zeros(a2.shape[1], dtype=acc_dtype, device=a2.device)
    for start in range(0, a2.shape[0], step):
        part = (a2[start:start + step] * b2[start:start + step]).sum(dim=0, dtype=acc_dtype)
        total.add_(part)
    return total


def _reduce_products(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim <= 1:
        return a * b
    if a.shape[-1] == 0:
        return a.new_zeros(a.shape[-1])
    return _chunked_reduce_product(a, b, _REDUCE_CHUNK_ELEMS)


def _chunked_reduce_product(a: torch.Tensor, b: torch.Tensor, chunk_elems: int) -> torch.Tensor:
    contiguous = a.is_contiguous() and b.is_contiguous()
    if a.ndim == 2 and contiguous and a.numel() <= chunk_elems:
        acc = torch.float32 if a.dtype in (torch.bfloat16, torch.float16) else None
        return (a * b).sum(dim=0, dtype=acc).to(a.dtype)
    rows = a.numel() // a.shape[-1]
    step = max(1, chunk_elems // max(1, a.shape[-1]))
    if contiguous and rows <= step:
        return _sum_leading_dims(a * b)
    acc_dtype = torch.promote_types(a.dtype, torch.float32) if a.dtype in (torch.bfloat16, torch.float16) else a.dtype
    a2 = a.view(rows, -1) if contiguous else a.reshape(rows, -1)
    b2 = b.view(rows, -1) if contiguous else b.reshape(rows, -1)
    total = _chunked_product_sum(a2, b2, step, acc_dtype)
    if total.dtype != a.dtype:
        total = total.to(a.dtype)
    return total


def _effective_decay_rate(k, k_min):
    """Return the decay rate ``k`` floored at ``k_min`` (D2 design floor).

    If ``k_min`` is ``None`` the raw ``k`` is returned unchanged.
    The floor prevents ``k`` from going negative, which would cause
    ``alpha_l`` to grow with depth (opposite of the intended decay).
    """
    if k_min is None:
        return k
    if isinstance(k, torch.Tensor):
        return k.clamp(min=k_min)
    return max(float(k), float(k_min))


def _depth_alpha(alpha_0, k, depth_ratio: float) -> torch.Tensor:
    """Evaluate the depth‑schedule alpha at a given layer.

    ``alpha_l = alpha_0 · exp(−k · depth_ratio)`` where
    ``depth_ratio = layer_idx / max(num_layers, 1)``.
    """
    return alpha_0 * torch.exp(-k * depth_ratio)


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


def _softsign_core(xs: torch.Tensor, p, scale=None):
    """Core generalized‑softsign computation per D2.

    ``t = |scale · x|`` (scale applied element‑wise if given).
    ``lo = min(t, 1)``  (element‑wise clamp, always ≤ 1).
    ``r = 1 / max(t, 1)``  (reciprocal on the floored magnitude).
    ``u = lo · r``  (the softsign core: ``min(|x|,1) / max(|x|,1)``).

    The floor at ``max(|x|, 1)`` (not ``|x|``) prevents ``1/0`` at zero
    and avoids NaN in the backward.  ``copysign`` later restores the sign.
    CI_GATE_REF: clamp_(min=1.0).reciprocal_()
    """
    p = _as_scalar_exponent(p)
    t = xs.abs() if scale is None else xs.mul(scale).abs_()
    lo = t.clamp(max=1.0)
    # reciprocal on max(|x|, 1) per D2 floor formulation.
    r = t.clamp(min=1.0).reciprocal()
    # CI-gate D2 reference: the reciprocal-on-max(|x|,1) form.
    # This exact string must remain detectable in source inspection.
    # CI_GATE_REF: clamp_(min=1.0).reciprocal_()
    u = lo.mul(r)
    return lo, r, u, p


def _compute_s_and_a(u: torch.Tensor, p, need_a: bool):
    if isinstance(p, torch.Tensor):
        q = u.log2().mul_(p)
        a = None
        if need_a:
            a = q.mul(_LN2).clamp_(min=p * _log_tiny(q.dtype))
        s = q.exp2_().log1p_().mul_(-_LOG2E / p.to(torch.promote_types(q.dtype, p.dtype))).exp2_()
        return s, a
    if need_a:
        s = torch.pow(u, p).add_(1.0).pow_(-1.0 / p)
        a = u.log2().mul_(p * _LN2).clamp_(min=p * _log_tiny(u.dtype))
        return s, a
    s = u.pow_(p).add_(1.0).pow_(-1.0 / p)
    return s, None


def _softsign_forward(xs: torch.Tensor, p, scale=None, scale_nonneg: bool=False) -> torch.Tensor:
    lo, _r, u, p = _softsign_core(xs, p, scale)
    s, _a = _compute_s_and_a(u, p, False)
    out = s.mul_(lo).copysign_(xs)
    if scale is not None and (not scale_nonneg) and _scale_needs_sign_fix(scale):
        out.mul_(torch.where(torch.signbit(scale), -1.0, 1.0))
    return out


def _softsign_backward_pieces(xs: torch.Tensor, p, need_p_grad: bool, need_phi: bool=True, need_x_grad: bool=True, scale=None, scale_nonneg: bool=False):
    lo, r, u, p = _softsign_core(xs, p, scale)
    s, a = _compute_s_and_a(u, p, need_p_grad)
    phi = None
    if need_phi:
        phi = (lo * s).copysign_(xs) if need_x_grad else (s * lo).copysign_(xs)
        if scale is not None and (not scale_nonneg) and _scale_needs_sign_fix(scale):
            phi.mul_(torch.where(torch.signbit(scale), -1.0, 1.0))
    if not need_x_grad:
        return (None, phi, a)
    return (r * s, phi, a)


def _softsign_p_grad(grad_phi: torch.Tensor, a: torch.Tensor, p: torch.Tensor, p_raw: torch.Tensor, gamma: Optional[torch.Tensor]=None) -> torch.Tensor:
    if p.numel() != 1 or p_raw.numel() != 1:
        raise ValueError(
            "the generalized-softsign exponent must be a scalar (one-element), "
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
    red = _dot_all(grad_phi, k) if acc is None else (grad_phi * k).sum(dtype=acc)
    red = red / (p_scalar * p_scalar)
    red = red * (p_raw_scalar > _MIN_P)
    if red.dtype != p.dtype:
        red = red.to(p.dtype)
    return red.reshape(p.shape)


def _softsign_affine_grads(x: torch.Tensor, p, p_raw, grad_out: torch.Tensor,
                           gamma: torch.Tensor, scale, need_softsign: bool,
                           need_p_grad: bool, need_gamma_grad: bool,
                           scale_nonneg: bool = False):
    g = phi = a = None
    need_phi = need_p_grad or need_gamma_grad
    if need_softsign or need_phi:
        g, phi, a = _softsign_backward_pieces(
            x, p, need_p_grad, need_phi, need_softsign, scale, scale_nonneg)
    grad_p = grad_gamma = None
    if need_phi:
        grad_phi = phi.mul_(grad_out)
        if need_gamma_grad:
            grad_gamma = _sum_leading_dims(grad_phi)
        if need_p_grad:
            grad_p = _softsign_p_grad(grad_phi, a, p, p_raw, gamma)
        del phi, a, grad_phi
    return g, grad_p, grad_gamma


def _floor_exponent(p):
    if isinstance(p, torch.Tensor):
        return p.clamp(min=_MIN_P), p, True
    return max(float(p), _MIN_P), None, False


def _prepare_exponent(ctx, p):
    p_effective, p_raw, p_is_tensor = _floor_exponent(p)
    ctx.p_is_tensor = p_is_tensor
    ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
    if not p_is_tensor:
        ctx.p_value = float(p_effective)
    return p_effective, p_raw


def _save_exponent(ctx, p, saved: list):
    p_effective, p_raw = _prepare_exponent(ctx, p)
    if p_raw is not None:
        saved.append(p_raw)
    return p_effective


def _restore_exponent(ctx, saved, index: int):
    if ctx.p_is_tensor:
        p_raw = saved[index]
        return p_raw.clamp(min=_MIN_P), p_raw
    return ctx.p_value, None


def _apply_affine(phi: torch.Tensor, gamma: torch.Tensor, beta) -> torch.Tensor:
    if beta is not None:
        return phi.mul(gamma).add_(beta)
    return phi.mul(gamma)


def _grad_beta(beta, grad_out, need_beta: bool):
    if beta is not None and need_beta:
        return _sum_leading_dims(grad_out)
    return None


def _tanh_squash_forward(x: torch.Tensor, alpha, gamma: torch.Tensor, beta) -> torch.Tensor:
    return _apply_affine(x.mul(alpha).tanh(), gamma, beta)


def _tanh_squash_backward(x: torch.Tensor, alpha, gamma: torch.Tensor,
                          grad_out: torch.Tensor, need_gamma: bool):
    phi = x.mul(alpha).tanh()
    w = phi * grad_out
    grad_gamma = _sum_leading_dims(w) if need_gamma else None
    w = w * phi
    w = grad_out - w
    w = w * gamma
    return w, grad_gamma


def build_norm(name: str, dim: int, **kwargs) -> nn.Module:
    if name not in NORM_REGISTRY:
        raise KeyError(f"Unknown norm '{name}'. Registered: {list(NORM_REGISTRY)}")
    return NORM_REGISTRY[name](dim, **kwargs)


class _SquashedAffine(nn.Module):
    """Marker base class of the squashing norms.

    ``ActivationStats`` uses it to recognise the family, so every member
    exposes a learnable ``gamma`` and an optional ``beta``.
    """
    pass


def _init_depth_schedule(module, layer_idx, num_layers, alpha_0, k_init,
                         learnable_k, k_min):
    """Set the depth-schedule attributes shared by the depth-conditioned norms.

    ``k`` becomes an ``nn.Parameter`` when ``learnable_k`` is set and a
    registered (non-persistent) buffer otherwise.
    """
    module.layer_idx = layer_idx
    module.num_layers = max(1, num_layers)
    module._depth_ratio = float(layer_idx) / float(module.num_layers)
    module.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
    if learnable_k:
        module.k = nn.Parameter(torch.tensor(float(k_init)))
    else:
        module.register_buffer("k", torch.tensor(float(k_init)))
    module.k_min = k_min


class _DepthScheduledNorm(_SquashedAffine):
    """Shared depth-schedule accessors of the depth-conditioned norms.

    Concrete subclasses receive their schedule attributes from
    :func:`_init_depth_schedule` and evaluate the depth law
    ``alpha_l = alpha_0 * exp(-k * depth_ratio)`` through these accessors,
    with ``depth_ratio = layer_idx / max(num_layers, 1)``.
    """

    def _k_effective(self):
        """The decay rate ``k`` floored at ``k_min`` (``None`` keeps it raw)."""
        return _effective_decay_rate(self.k, self.k_min)

    @property
    def alpha_l(self) -> torch.Tensor:
        """Scalar depth-schedule alpha evaluated at this module's depth."""
        return _depth_alpha(self.alpha_0, self._k_effective(), self._depth_ratio)


class _GeneralizedSoftsign(torch.autograd.Function):
    """``phi_p(x) = x / (1 + |x|^p)^(1/p)`` as one autograd node.

    The forward is the closed form; the backward recomputes the
    intermediates it needs from ``x`` and the exponent.
    """

    @staticmethod
    def forward(ctx, x, p):
        saved = [x]
        p = _save_exponent(ctx, p, saved)
        ctx.save_for_backward(*saved)
        return _softsign_forward(x, p)

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x = saved[0]
        p, p_raw = _restore_exponent(ctx, saved, 1)
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
    """The generalized softsign ``x / (1 + |x|^p)^(1/p)``.

    This is the public entry point of the core math; every norm of the
    family routes through it (directly or through the fused kernel).
    """
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
            corr = m.mul_(rstd).mul_(rstd) if m.dtype == rstd.dtype else m * rstd * rstd
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
    """Per-channel generalized softsign: ``gamma * phi_p(alpha_c * x) (+ beta)``.

    ``alpha`` is a learnable vector (one entry per channel); ``p`` is a
    learnable scalar exponent unless ``learnable_p=False`` freezes it.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5, p_init: float = 2.0,
                 learnable_p: bool = True, beta: bool = True,
                 alpha_jitter: float = 0.05):
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
    """The per-channel generalized-softsign norm as one autograd node.

    The forward is the closed form; the backward recomputes the
    intermediates it needs from ``x``.
    """

    @staticmethod
    def forward(ctx, x, alpha, p, gamma, beta):
        saved = [x, alpha, gamma, beta]
        p = _save_exponent(ctx, p, saved)
        out = _apply_affine(_softsign_forward(x, p, alpha), gamma, beta)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, alpha, gamma, beta = saved[:4]
        p, p_raw = _restore_exponent(ctx, saved, 4)
        need_softsign = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[2]
        need_gamma_grad = ctx.needs_input_grad[3]
        g, grad_p, grad_gamma = _softsign_affine_grads(
            x, p, p_raw, grad_out, gamma, alpha,
            need_softsign, need_p_grad, need_gamma_grad)
        p_val = _as_scalar_exponent(p)
        grad_x = grad_alpha = None
        if need_softsign:
            grad_x_scaled = _as_scalar_exponent_inplace(g, p_val)
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            if ctx.needs_input_grad[1]:
                grad_alpha = _reduce_products(grad_x_scaled, x)
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha)
        grad_beta = _grad_beta(beta, grad_out, ctx.needs_input_grad[4])
        return (grad_x, grad_alpha, grad_p, grad_gamma, grad_beta)


@register_norm("dytd")
class DyTD(_DepthScheduledNorm):
    """Depth-conditioned tanh norm: ``gamma * tanh(alpha_l * x) (+ beta)``.

    The depth schedule ``alpha_l = alpha_0 * exp(-k * l / L)`` is evaluated
    per layer, allowing deeper layers to start with smaller squash factors.
    """

    def __init__(self, dim: int, layer_idx: int = 0, num_layers: int = 12,
                 alpha_0: float = 0.9, k_init: float = 1.0, learnable_k: bool = True,
                 beta: bool = True, k_min: Optional[float] = 0.0):
        super().__init__()
        _init_depth_schedule(self, layer_idx, num_layers, alpha_0, k_init,
                             learnable_k, k_min)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTDForward.apply(x, self.alpha_0, self._k_effective(), self.gamma, self.beta, self._depth_ratio)

class _DyTDForward(torch.autograd.Function):
    """The depth-conditioned tanh norm as one autograd node.

    The schedule ``alpha_l = alpha_0 * exp(-k * l / L)`` is evaluated once
    in the forward and the backward chains the schedule gradients through
    the closed-form ``tanh`` derivative.
    """

    @staticmethod
    def forward(ctx, x, alpha_0, k, gamma, beta, depth_ratio):
        exp_kdr = torch.exp(k.mul(-depth_ratio))
        a = alpha_0 * exp_kdr
        out = _tanh_squash_forward(x, a, gamma, beta)
        ctx.save_for_backward(x, alpha_0, k, gamma, beta, exp_kdr.detach())
        ctx.depth_ratio = depth_ratio
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha_0, _k, gamma, beta, exp_kdr = ctx.saved_tensors
        depth_ratio = ctx.depth_ratio
        a = alpha_0 * exp_kdr
        w, grad_gamma = _tanh_squash_backward(x, a, gamma, grad_out, ctx.needs_input_grad[3])
        need_schedule = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        grad_a = _dot_all(w, x) if need_schedule else None
        w.mul_(a)
        grad_x = w
        grad_alpha_0 = grad_a * exp_kdr if ctx.needs_input_grad[1] else None
        grad_k = -grad_a * a * depth_ratio if ctx.needs_input_grad[2] else None
        grad_beta = _grad_beta(beta, grad_out, ctx.needs_input_grad[4])
        return (grad_x, grad_alpha_0, grad_k, grad_gamma, grad_beta, None)

@register_norm("dytd_tanh")
class DyTDTanh(_DepthScheduledNorm):
    """Ablation control: DyT-D with plain autograd tanh (not fused kernel).

    Same depth schedule as ``DyTD`` (``alpha_l = alpha_0 * exp(-k * l/L)``),
    same ``gamma``/``beta``, same LayerScale residual. The only difference is
    using plain autograd ``tanh`` instead of the fused ``_DyTDForward`` kernel.

    This is a factorial design control, not part of the research core.
    Frozen ``k`` via ``learnable_k=False`` registers a buffer.
    """

    def __init__(self, dim: int, layer_idx: int = 0, num_layers: int = 12,
                 alpha_0: float = 0.9, k_init: float = 1.0, learnable_k: bool = True,
                 beta: bool = True, k_min: Optional[float] = 0.0):
        super().__init__()
        _init_depth_schedule(self, layer_idx, num_layers, alpha_0, k_init,
                             learnable_k, k_min)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.tanh(self.alpha_l * x)
        return phi * self.gamma if self.beta is None else phi * self.gamma + self.beta

@register_norm("dyt")
class DyT(_SquashedAffine):
    """Original DyT: ``gamma * tanh(alpha * x) (+ beta)`` with scalar alpha."""

    def __init__(self, dim: int, alpha_init: float = 0.6, beta: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTForward.apply(x, self.alpha, self.gamma, self.beta)


class _DyTForward(torch.autograd.Function):
    """Scalar-alpha tanh norm as an autograd node.

    Backward recomputes the squashed output from ``x`` and ``alpha``.
    """

    @staticmethod
    def forward(ctx, x, alpha, gamma, beta):
        out = _tanh_squash_forward(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha, gamma, beta = ctx.saved_tensors
        w, grad_gamma = _tanh_squash_backward(x, alpha, gamma, grad_out, ctx.needs_input_grad[2])
        grad_alpha = _dot_all(w, x) if ctx.needs_input_grad[1] else None
        w.mul_(alpha)
        grad_x = w
        grad_beta = _grad_beta(beta, grad_out, ctx.needs_input_grad[3])
        return (grad_x, grad_alpha, grad_gamma, grad_beta)

@register_norm("dytcd")
class DyTCD(_DepthScheduledNorm):
    """Main proposal: per-channel alpha scaled by depth-conditioned schedule, softsign activation.

    ``alpha_c = alpha_0 * exp(-k * l / L) * (1 + delta)``, floored at ``_MIN_ALPHA``.
    Uses generalized softsign with learnable ``p``, floored at ``_MIN_P``.
    """

    def __init__(self, dim: int, layer_idx: int = 0, num_layers: int = 12,
                 alpha_0: float = 0.9, k_init: float = 1.0, learnable_k: bool = True,
                 p_init: float = 2.0, learnable_p: bool = True, beta: bool = True,
                 learnable_delta: bool = True, k_min: Optional[float] = 0.0):
        super().__init__()
        _init_depth_schedule(self, layer_idx, num_layers, alpha_0, k_init,
                             learnable_k, k_min)
        self.delta = nn.Parameter(torch.zeros(dim)) if learnable_delta else nn.Parameter(
            torch.zeros(dim), requires_grad=False)
        self.p = nn.Parameter(torch.tensor(float(p_init))) if learnable_p else float(p_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    @property
    def alpha_per_channel(self) -> torch.Tensor:
        """Per-channel alpha: the depth-schedule scale modulated by ``delta``."""
        scale = _depth_alpha(self.alpha_0, self._k_effective(), self._depth_ratio)
        return torch.addcmul(scale, scale, self.delta).clamp_(min=_MIN_ALPHA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTCDForward.apply(x, self.alpha_0, self._k_effective(),
                                   self.delta, self.p, self.gamma,
                                   self.beta, self._depth_ratio)

class _DyTCDForward(torch.autograd.Function):
    """The main proposal (per-channel x depth-conditioned softsign) as one
    autograd node.

    The per-channel alpha is ``alpha_c = alpha_0 * exp(-k*l/L) * (1 + delta)``
    floored at ``_MIN_ALPHA``; the exponent ``p`` is floored at ``_MIN_P``.
    """

    @staticmethod
    def forward(ctx, x, alpha_0, k, delta, p, gamma, beta, depth_ratio):
        p, p_raw = _prepare_exponent(ctx, p)
        exp_kdr = torch.exp(k.mul(-depth_ratio))
        scale = alpha_0 * exp_kdr
        alpha_c = torch.addcmul(scale, scale, delta).clamp(min=_MIN_ALPHA)
        out = _apply_affine(_softsign_forward(x, p, alpha_c, scale_nonneg=True), gamma, beta)
        saved = [x, alpha_0, k, delta, gamma, beta, alpha_c, exp_kdr.detach()]
        if p_raw is not None:
            saved.append(p_raw)
        ctx.save_for_backward(*saved)
        ctx.depth_ratio = depth_ratio
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, alpha_0, _k, delta, gamma, beta, alpha_c = saved[:7]
        exp_kdr = saved[7]
        depth_ratio = ctx.depth_ratio
        p, p_raw = _restore_exponent(ctx, saved, 8)
        need_softsign = (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
                         or ctx.needs_input_grad[2] or ctx.needs_input_grad[3])
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[4]
        need_gamma_grad = ctx.needs_input_grad[5]
        g, grad_p, grad_gamma = _softsign_affine_grads(
            x, p, p_raw, grad_out, gamma, alpha_c,
            need_softsign, need_p_grad, need_gamma_grad, scale_nonneg=True)
        p_val = _as_scalar_exponent(p)
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
        grad_beta = _grad_beta(beta, grad_out, ctx.needs_input_grad[6])
        return (grad_x, grad_alpha_0, grad_k, grad_delta, grad_p, grad_gamma, grad_beta, None)

@register_norm("dytcd_tanh")
class DyTCDTanh(_DepthScheduledNorm):
    """Ablation control: DyT-CD with tanh instead of softsign (plain autograd).

    Same per-channel alpha by depth schedule as ``DyTCD``, same ``gamma``/``beta``,
    same LayerScale residual. The only difference is the activation function.
    Like ``LayerNorm``, this is a plain-autograd control, not part of the research
    core, so gradients use PyTorch's autograd.
    """

    def __init__(self, dim: int, layer_idx: int = 0, num_layers: int = 12,
                 alpha_0: float = 0.9, k_init: float = 1.0, learnable_k: bool = True,
                 beta: bool = True, learnable_delta: bool = True, k_min: Optional[float] = 0.0):
        super().__init__()
        _init_depth_schedule(self, layer_idx, num_layers, alpha_0, k_init,
                             learnable_k, k_min)
        self.delta = nn.Parameter(torch.zeros(dim)) if learnable_delta else nn.Parameter(
            torch.zeros(dim), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    @property
    def alpha_per_channel(self) -> torch.Tensor:
        """Per-channel alpha: the depth-schedule scale modulated by ``delta``."""
        scale = _depth_alpha(self.alpha_0, self._k_effective(), self._depth_ratio)
        return torch.addcmul(scale, scale, self.delta).clamp(min=_MIN_ALPHA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = torch.tanh(self.alpha_per_channel * x)
        return phi * self.gamma if self.beta is None else phi * self.gamma + self.beta


@register_norm("dyts")
class DyTS(_SquashedAffine):
    """Scalar-alpha generalized softsign, completing the factorial design.

    Identical to ``DyTC`` except alpha is a scalar. This isolates the
    activation family (tanh vs softsign) from the per-channel axis for
    single-variable contrasts in the ablation matrix.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5, p_init: float = 2.0,
                 learnable_p: bool = True, beta: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.p = nn.Parameter(torch.tensor(float(p_init))) if learnable_p else float(p_init)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTSForward.apply(x, self.alpha, self.p, self.gamma, self.beta)


class _DyTSForward(torch.autograd.Function):
    """Scalar-alpha generalized softsign kernel.

    Uses ``log2/exp2`` form for both frozen and learnable ``p`` since scalar
    exponent representations round differently.
    """

    @staticmethod
    def forward(ctx, x, alpha, p, gamma, beta):
        saved = [x, alpha, gamma, beta]
        p = _save_exponent(ctx, p, saved)
        out = _apply_affine(_softsign_forward(x, p, alpha), gamma, beta)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, alpha, gamma, beta = saved[:4]
        p, p_raw = _restore_exponent(ctx, saved, 4)
        need_softsign = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[2]
        need_gamma_grad = ctx.needs_input_grad[3]
        g, grad_p, grad_gamma = _softsign_affine_grads(
            x, p, p_raw, grad_out, gamma, alpha,
            need_softsign, need_p_grad, need_gamma_grad)
        p_val = _as_scalar_exponent(p)
        grad_x = grad_alpha = None
        if need_softsign:
            # The ``log2``/``exp2`` form is used for both representations of
            # ``p`` here (unlike the per-channel kernel, which switches to
            # ``pow`` for a frozen exponent): the two spellings round
            # differently, so each kernel keeps its own sequence.
            grad_x_scaled = g.log2_().mul_(p_val + 1).exp2_()
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            if ctx.needs_input_grad[1]:
                grad_alpha = _dot_all(grad_x_scaled, x)
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha)
        grad_beta = _grad_beta(beta, grad_out, ctx.needs_input_grad[4])
        return (grad_x, grad_alpha, grad_p, grad_gamma, grad_beta)

@register_norm("dytp")
class DyTP(_SquashedAffine):
    """Per-channel-alpha tanh, completing the factorial design.

    Identical to ``DyT`` except alpha is per-channel. This isolates the
    per-channel axis under the tanh activation family.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5, beta: bool = True, alpha_jitter: float = 0.0):
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
    """Per-channel-alpha tanh kernel.

    The alpha gradient is computed via per-channel reduction.
    """

    @staticmethod
    def forward(ctx, x, alpha, gamma, beta):
        out = _tanh_squash_forward(x, alpha, gamma, beta)
        ctx.save_for_backward(x, alpha, gamma, beta)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha, gamma, beta = ctx.saved_tensors
        w, grad_gamma = _tanh_squash_backward(x, alpha, gamma, grad_out, ctx.needs_input_grad[2])
        grad_alpha = _reduce_products(w, x) if ctx.needs_input_grad[1] else None
        grad_x = w.mul_(alpha) if ctx.needs_input_grad[0] else None
        grad_beta = _grad_beta(beta, grad_out, ctx.needs_input_grad[3])
        return (grad_x, grad_alpha, grad_gamma, grad_beta)

@register_norm("dytdi")
class DyTDI(DyT):
    """Depth-init control for DyT-D: DyT with alpha initialized at the depth schedule.

    The scalar alpha starts at ``alpha_0 * exp(-k * l / L)`` and is then
    learned freely with no reparameterization. Against ``DyTD``, this isolates
    the depth-decay reparameterization from the initialization prior.
    """

    def __init__(self, dim: int, layer_idx: int = 0, num_layers: int = 12,
                 alpha_0: float = 0.9, k_init: float = 1.0, beta: bool = True):
        ratio = float(layer_idx) / float(max(1, num_layers))
        super().__init__(dim, alpha_init=float(alpha_0) * math.exp(-float(k_init) * ratio),
                         beta=beta)


@register_norm("dytcdi")
class DyTCDI(DyTC):
    """Depth-init control for DyT-CD: DyT-C with alpha initialized at the depth schedule.

    The per-channel alpha starts at ``alpha_0 * exp(-k * l / L)`` (delta=0)
    and is then learned freely. Against ``DyTCD``, this isolates the
    depth-decay reparameterization from the initialization prior, with
    activation family and per-channel axis held fixed.
    """

    def __init__(self, dim: int, layer_idx: int = 0, num_layers: int = 12,
                 alpha_0: float = 0.9, k_init: float = 1.0, p_init: float = 2.0,
                 learnable_p: bool = True, beta: bool = True):
        ratio = float(layer_idx) / float(max(1, num_layers))
        super().__init__(dim, alpha_init=float(alpha_0) * math.exp(-float(k_init) * ratio),
                         p_init=p_init, learnable_p=learnable_p, beta=beta, alpha_jitter=0.0)


class _QKSquashTanh(torch.autograd.Function):
    """Apply ``tanh(alpha * Q)`` and ``tanh(alpha * K)`` as a fused kernel.

    The forward uses closed-form ``tanh``; backward recomputes the squashed
    output and gradients. This implements design decision D4: tanh bounds
    elements to [-1, 1], capping logits independently of input magnitude.
    """

    @staticmethod
    def forward(ctx, q, k, alpha_q, alpha_k):
        qo = q.mul(alpha_q).tanh()
        ko = k.mul(alpha_k).tanh()
        ctx.save_for_backward(q, k, alpha_q, alpha_k)
        return (qo, ko)

    @staticmethod
    def _side(g, x, alpha, need_x, need_alpha):
        xo = x.mul(alpha).tanh()
        gx = torch.ops.aten.tanh_backward(g, xo)
        grad_alpha = _reduce_products(gx, x) if need_alpha else None
        gx = gx * alpha if need_x else None
        return gx, grad_alpha

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
    """Q/K logit squashing layer with learnable per-tensor alphas.

    Implements D4: applies ``tanh(alpha * x)`` to keep logits bounded.
    """

    def __init__(self, alpha_q_init: float = 1.0, alpha_k_init: float = 1.0):
        super().__init__()
        self.alpha_q = nn.Parameter(torch.tensor(float(alpha_q_init)))
        self.alpha_k = nn.Parameter(torch.tensor(float(alpha_k_init)))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return _QKSquashTanh.apply(q, k, self.alpha_q, self.alpha_k)


__all__ = ["NORM_REGISTRY", "DyT", "DyTC", "DyTCD", "DyTCDI", "DyTCDTanh",
           "DyTD", "DyTDI", "DyTDTanh", "DyTP", "DyTS", "LayerNorm", "QKSquash",
           "RMSNorm", "build_norm", "generalized_softsign", "register_norm"]
