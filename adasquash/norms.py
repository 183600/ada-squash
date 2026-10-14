"""
Normalization layers for the DyT++ / AdaSquash study.

Implements:
  * LayerNorm           - Pre-LN baseline (standard LayerNorm)
  * RMSNorm             - LLaMA-style RMSNorm baseline
  * DyT                 - Original Dynamic Tanh (Zhu et al., 2025)
  * DyTC                - Variant A: per-channel alpha + generalized softsign
  * DyTD                - Variant B: depth-conditioned alpha + LayerScale
  * DyTCD               - Variant A+B combined (the main DyT++ proposal)

Design notes
------------
All DyT-family layers are *element-wise* (no cross-token statistics, no
synchronization across the batch / sequence dimension). They can therefore
serve as drop-in replacements for LayerNorm / RMSNorm inside any

The DyT-C softsign family
    phi_p(x) = x / (1 + |x|^p)^{1/p}
converges to hard tanh (i.e. a clip to [-1, 1]) as p -> infinity and to the
classic softsign x / (1 + |x|) at p = 1. p=2 is a particularly attractive
operating point: it is C^inf, has a bounded gradient, and avoids the
saturation plateau of tanh in deep nets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NORM_REGISTRY: dict[str, type] = {}

# Floor applied to the depth-conditioned per-channel alpha of DyT-CD. A
# non-positive alpha would flip the sign of the channel (and a zero one would
# kill its gradient entirely), so the schedule is clamped away from zero.
_MIN_ALPHA = 1e-4

# Floor applied to the generalized-softsign curvature exponent p. phi_p is only
# a finite, monotone family member for p > 0: as p -> 0+ it collapses to 0
# (a dead layer with no gradient), and for p < 0 the |x|^p term blows up at
# |x| = 0, so the whole layer NaNs and takes every gradient that reaches it
# with it. `p` is a learnable scalar here and the study explicitly sweeps it
# (p = 1 / 2 / 4 / learnable / -> inf), so it is free to wander into that
# region over a long run; the floor keeps the failure finite. It is inactive
# at every operating point the paper and the ablation matrix use.
_MIN_P = 1e-3


def _sum_leading_dims(t: torch.Tensor) -> torch.Tensor:
    """Sum away every dim except the trailing (parameter) axis.

    parameter is fed by a `[B, T, C]` product, so only the leading dims go.
    `t.sum(dim=())` would instead reduce *every* dim (an empty dim tuple
    means "all dims" to torch), which is what
    `sum(dim=tuple(range(ndim - 1)))` degenerates to for a 1-D input - a
    single normalized vector, a shape LayerNorm accepts and the DyT family
    / RMSNorm must accept too. That collapsed the parameter gradient to a
    gradient. With only the parameter axis left there is nothing to reduce,
    so the elementwise product is returned unchanged.
    """
    return t if t.ndim <= 1 else t.sum(dim=tuple(range(t.ndim - 1)))


def _dot_all(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """dot(a, b) over *every* element, without materialising `a * b`.

    product down to one number. Both operands are the same shape and, in
    every caller, contiguous, so the contraction runs as a single flat
    `dot` over two views: the same value the product would have produced,
    """
    if a.is_contiguous() and b.is_contiguous():
        return torch.dot(a.reshape(-1), b.reshape(-1))
    return (a * b).sum()


def register_norm(name: str):
    def _decorator(cls):
        NORM_REGISTRY[name] = cls
        cls._norm_name = name
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Shared generalized-softsign core
# ---------------------------------------------------------------------------
# DyT-C and DyT-CD) all evaluate the same family, so the element-wise core
#
# With `t = |x|` the closed form needs two branches. Both branches collapse
# onto a single pair of *reciprocal* helpers, which removes the boolean mask
# and the `torch.where` entirely:
#
#     lo = min(t, 1)          r = 1 / max(t, 1)
#
#     u = lo * r      = t      for t <= 1,  1/t  for t >  1   (always <= 1)
#     f = r          = 1      for t <= 1,  1/t  for t >  1
#
# otherwise have been `torch.where(t > 1, u, 1)` is a free alias.
#
# The reciprocal is taken on `max(t, 1)`, never on `t` itself: an exact zero
# (a bf16 underflow, a zeroed/padded element) would otherwise give
# `1/0 = inf` and `d(1/t)/dt = -1/t^2` on the unused branch NaNs the whole
# gradient. `clamp` contributes no gradient of its own, so the selected branch
# is still differentiated exactly and phi'(0) = 1 as the closed form requires.
def _softsign_forward(xs: torch.Tensor, p) -> torch.Tensor:
    """Element-wise core of `phi_p` on the signed scaled input `xs`.

    Returns `phi = sign(xs) * lo * s` with `s = (1 + u^p)^(-1/p)`.

    run this core wants only the squashed activation, and they all rebuild `s`
    from their saved input in the backward (`_softsign_backward_pieces`), so
    boundary for nothing. The frame drops `s` on return instead.
    """
    t = xs.abs()
    lo = t.clamp(max=1.0)
    hi = t.clamp_(min=1.0).reciprocal_()
    # u = lo * hi.
    u = hi.mul_(lo)
    # s = (1 + u^p)^(-1/p), evaluated through log1p on a base <= 1. The
    # power is taken as `exp(p * log(u))`: `u <= 1` keeps `p * log(u) <= 0`,
    # so the exponential lands in (0, 1] exactly where `pow` put it (including
    # the `u == 0` limit, where `log(0) * p = -inf -> exp(-inf) = 0 = 0^p`).
    s = u.log_().mul_(p).exp_().log1p_().mul_(-1.0 / p).exp_()
    # The sign rides on the sign bit of `xs` rather than on a separate
    return lo.mul_(s).copysign_(xs)


def _softsign_backward_pieces(
    xs: torch.Tensor,
    p,
    need_u: bool,
    need_phi: bool = True,
    need_g: bool = True,
):
    """Rebuild the backward-only pieces of `phi_p` from the signed input.

    Returns `(g, phi, a)`, where `g = s * f` is the single combination the
    `x`-derivative depends on (`dphi/dx = g^(p+1)`), `phi` is the pre-affine
    activation `dL/dgamma` and `dphi/dp` both reduce over, and `a = p *
    log(u)` is only returned when the exponent itself carries a gradient.
    `phi` is skipped the same way when neither reduction reads it, and `g`

    `s` is re-derived here, from `xs` and `p`, with exactly the sequence
    `_softsign_forward` ran - so it is bit-for-bit the value the forward used,
    the way to its own backward. The logarithm of `u` falls out of the same
    """
    t = xs.abs()
    lo = t.clamp(max=1.0)
    r = t.clamp_(min=1.0).reciprocal_()
    # u = lo * r. `lo` is still owed to `phi` below, so `u` only gets its own
    a = None
    if need_u:
        # `log(u)` is the head of `s`'s own chain, so the chain is grown out
        # of it rather than restarted from `u`.
        log_u = lo.mul(r).log_() if need_phi else lo.mul_(r).log_()
        # `A = p * log(u)` is the exponent the softsign's `1 + u^p` actually
        # evaluates, so `s = exp(-softplus(A) / p)` is grown straight out of
        # own for the exponent gradient below and keeps the two chains from
        # having to agree on a shared temporary.
        log_u.mul_(p)
        s = log_u.exp().log1p_().mul_(-1.0 / p).exp_()
        # `s` is done reading `A` by now, and the exponent gradient only ever
        # consumes `A` (see `_softsign_p_grad`), so the floor is applied to
        # step rather than a clamp *and* a rescale.
        # The `-inf` a zeroed / underflowed `u` produces in its logarithm is
        # floored here - exactly where the pow form floored the base at
        # `tiny` - so the element stays finite; `phi` is zero there and zeroes
        # the contribution either way.
        #
        # The floor is the *same* value it was on `log(u)`: `p > 0` here
        # (`_MIN_P`), so `p * max(log(u), log tiny) == max(p * log(u), p * log
        # tiny)` exactly, both sides being a selection followed by one rounded
        # multiply. And the clamp has to stay out of `s`, which is read off the
        # *unclamped* `A`: its value at `u == 0` is what makes `phi'(0) = 1`
        # come out right.
        a = log_u.clamp_(min=p * math.log(torch.finfo(log_u.dtype).tiny))
    else:
        s = lo.mul(r) if need_phi else lo.mul_(r)
        s = s.log_().mul_(p).exp_().log1p_().mul_(-1.0 / p).exp_()
    phi = lo.mul_(s).copysign_(xs) if need_phi else None
    if not need_g:
        return None, phi, a
    return r.mul_(s), phi, a


def _softsign_p_grad(
    grad_phi: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    p_raw: torch.Tensor,
) -> torch.Tensor:
    """Reduced `dL/dp` for the generalized softsign, in one fused pass.

    With `A = p * log(u)` the closed form's two compound terms

        log1p(u^p) = softplus(A),   u^p / (1 + u^p) = sigmoid(A)

    let `dphi/dp = phi * [softplus(A)/p^2 - sigmoid(A) * log(u) / p]` be
    evaluated from `A` alone, without the `u > 0` branch mask. Since
    `log(u) = A / p`, both terms carry the same `1/p^2`,

        dphi/dp = phi * [softplus(A) - A * sigmoid(A)] / p^2

    so the whole reduction is evaluated from `A` alone - which
    `_softsign_backward_pieces` forms in place out of `log(u)` - plus one
    `sigmoid(A) <= 1/2` and
    `softplus(A) = -log1p(-sigmoid(A))` loses nothing to cancellation there.
    The `-inf` a zeroed / underflowed `u` produces in its logarithm was
    already floored at `log(tiny)` where `A` was formed, so the element stays
    finite; `phi` is zero there and zeroes the contribution either way.

    `grad_phi` is the caller-owned `dL/dphi` product (`grad_out * phi`,
    already carrying `gamma` where the layer has one). It is folded into the
    reduction in place. So is the `1/p^2`, in the sense that it is applied to
    the *reduced scalar* rather than to the `[N, C]` bracket: the two are the
    """
    sig = a.sigmoid()
    # is not read again), so the product lands on it and frees the sigmoid
    a_sig = a.mul_(sig)
    sig.neg_().log1p_().neg_()
    k = sig.sub_(a_sig)
    # Reduce the [N, C] products in fp32 under bf16 autocast: a bf16
    # mantissa bits before the scalar survives.
    acc = torch.float32 if a.dtype in (torch.bfloat16, torch.float16) else None
    red = grad_phi.mul_(k).sum(dtype=acc)
    red = red.div_(p * p)
    # The forward only saw `clamp(p)`, so below the floor `p` carries no
    # gradient at all.
    red = red * (p_raw > _MIN_P).to(red.dtype)
    return red.to(p.dtype) if red.dtype != p.dtype else red


def build_norm(name: str, dim: int, **kwargs) -> nn.Module:
    """Factory: build a normalization layer by its registered name."""
    if name not in NORM_REGISTRY:
        raise KeyError(
            f"Unknown norm '{name}'. Registered: {list(NORM_REGISTRY)}"
        )
    return NORM_REGISTRY[name](dim, **kwargs)


# ---------------------------------------------------------------------------
# Shared output affine for the DyT family
# ---------------------------------------------------------------------------
class _SquashedAffine(nn.Module):
    """
    `y = gamma * phi + beta`, shared by every DyT-family layer.

    (as one `addcmul` rather than a multiply followed by an add).
    """


# ---------------------------------------------------------------------------
# Generalized softsign
# ---------------------------------------------------------------------------
class _GeneralizedSoftsign(torch.autograd.Function):
    """

    from the element-wise chain.

    With `u = |x|` for `|x| <= 1` and `1/|x|` otherwise, and `s = (1 + u^p)^(-1/p)`:

        phi     = sign(x) * min(|x|, 1) * s
        dphi/dx = (s * f)^(p+1),   f = 1 for |x| <= 1, u for |x| > 1
                = s^(p+1)           for |x| <= 1
                = (s*u)^(p+1)       for |x| >  1
        dphi/dp = phi * [log1p(u^p)/p^2 - (1 - s^p) * log(u) / p]

    `s * f` is the only combination the `x` derivative depends on, and `s` is
    also the factor every other output is built from - so `s` is all the
    backward needs, and it is re-derived from `x` and `p` there rather than
    """
    @staticmethod
    def forward(ctx, x, p):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            # gradient is applied there) and floor the exponent the forward
            # math actually sees.
            p_raw = p
            p = p.clamp(min=_MIN_P)
        ctx.p_is_tensor = p_is_tensor
        # `p` has by now been replaced by its clamped copy, which carries no
        # grad flag inside `forward`; the *raw* input is what says whether the
        # exponent is trainable, so the flag is read off that.
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
            # The |x| <= 1 branch reduces to s^(p+1) because
            # d/dx (1+x^p)^(-1/p-1) = (1+x^p)^(-1/p-1). The power is taken
            # as `exp((p+1) * log(g))`: `g > 0` keeps the log finite (and
            # `g == 0` still lands on 0).
            grad_x = g.log_().mul_(p + 1).exp_().mul_(grad_out)
        if need_p:
            # the `dL/dphi` product lands on it rather than beside it.
            grad_p = _softsign_p_grad(phi.mul_(grad_out), a, p, p_raw)
        return grad_x, grad_p


def generalized_softsign(x: torch.Tensor, p) -> torch.Tensor:
    """
    phi_p(x) = x / (1 + |x|^p)^{1/p}.

    The closed form is split so the exponentiated base stays <= 1:

        |x| <= 1 :  phi = sign(x)|x| * (1 + |x|^p)^(-1/p)      (|x|^p  <= 1)
        |x| >  1 :  phi = sign(x)  *  (1 + |x|^-p)^(-1/p)      (|x|^-p <= 1)

    Both branches are then expressed through one shared pair of helpers,

        u   = |x| for |x| <= 1, 1/|x| for |x| > 1    (always <= 1)
        lo  = min(|x|, 1)                             (always <= 1)

    giving `s = (1 + u^p)^(-1/p)` and `phi = sign(x) * lo * s`. This keeps
    phi in [-1, 1], avoids overflow for large p / large |x|, and stays finite
    for |x| = inf (where the naive closed form would divide inf by inf).

    The reciprocal is taken on `max(|x|, 1)` rather than on `|x|` directly.
    `torch.where` masks the *value* of the unused branch, but the backward
    (bf16 underflow of a small activation, a padded/zeroed element, ...) would
    evaluate `0 * -inf = NaN` and poison every gradient that reaches this
    layer. Clamping first keeps the unused branch finite; `clamp` itself
    contributes no gradient below its threshold, so the selected branch is
    still differentiated exactly and phi'(0) = 1 as the closed form requires.

    The layer runs as a custom autograd node (see `_GeneralizedSoftsign`) so
    """
    return _GeneralizedSoftsign.apply(x, p)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
@register_norm("ln")
class LayerNorm(nn.Module):
    """Standard LayerNorm (Pre-LN baseline)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(
            x, (x.shape[-1],), self.weight, self.bias, self.eps
        )


@register_norm("rmsnorm")
class RMSNorm(nn.Module):
    """RMSNorm baseline (LLaMA / Qwen style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNorm.apply(x, self.weight, self.eps)


class _RMSNorm(torch.autograd.Function):
    """
    RMSNorm forward with the gradient written out analytically.

    retained input and the per-row `rstd`.

    for `dL/dweight` and, with the per-channel `weight` folded in (which
    commutes with the sum over the channel axis), carries the `mean(t * x)`
    that `d/dx` needs; the correction is then taken in place on
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        nrm = torch.linalg.vector_norm(x, 2, dim=-1, keepdim=True)
        rstd = torch.rsqrt(nrm.square_().div_(x.shape[-1]).add_(eps))
        ctx.save_for_backward(x, rstd, weight)
        out = x * rstd
        # The dtype guard keeps the mixed-precision promotion
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
            # belongs to the caller) and is the leading term of d/dx, which is
            # handed back *in it* at the end.
            u = grad_out * rstd
            # sum over the leading dims, and d/dx needs `mean(t * weight)` -
            # the same product with a per-channel `weight` on it, which
            # commutes with the sum over the channel axis. So the two share one
            # consumed it, and in place, since nothing reads `t` again.
            t = u.mul(x)
            if need_w:
                grad_weight = _sum_leading_dims(t)
            if t.dtype == weight.dtype:
                m = t.mul_(weight).mean(dim=-1, keepdim=True)
            else:
                m = (t * weight).mean(dim=-1, keepdim=True)
            del t
            # d/dx of `x * rstd`: the rsqrt term contributes
            # -x * rstd^3 * mean(t * x); the rstd factor that `m` already
            # carries leaves an rstd^2 to undo. The `weight` lands on the
            if u.dtype == weight.dtype:
                u.mul_(weight)
            else:
                u = u * weight
            # `m` is freshly owned here, so the rstd^2 fold reuses it
            # instead of materialising a second [B, T, 1] temporary.
            if m.dtype == rstd.dtype:
                corr = m.mul_(rstd).mul_(rstd)
            else:
                corr = m * rstd * rstd
            out_dtype = torch.promote_types(
                u.dtype, torch.promote_types(x.dtype, corr.dtype)
            )
            if out_dtype == u.dtype:
                u.addcmul_(x, corr, value=-1)
            else:
                u = torch.addcmul(u, x, corr, value=-1)
            grad_x = u
        elif need_w:
            grad_weight = _sum_leading_dims((grad_out * rstd).mul(x))
        # The reduction promotes to the weight dtype (autocast keeps norms in
        # fp32), so the gradients are handed back in their inputs' dtypes.
        if grad_x is not None and grad_x.dtype != x.dtype:
            grad_x = grad_x.to(x.dtype)
        if grad_weight is not None and grad_weight.dtype != weight.dtype:
            grad_weight = grad_weight.to(weight.dtype)
        return grad_x, grad_weight, None


# ---------------------------------------------------------------------------
# Variant A: DyT-C  (per-channel alpha + generalized softsign)
# ---------------------------------------------------------------------------
@register_norm("dytc")
class DyTC(_SquashedAffine):
    """
    Channel-adaptive Dynamic Squashing.

        x_hat = alpha_c * x                       # alpha_c: [C]
        phi    = x_hat / (1 + |x_hat|^p)^{1/p}    # generalized softsign
        y      = gamma * phi + beta

    The per-channel alpha lets outlier-heavy channels (e.g. the rare
    "super-active" dimensions found in Llama-7B by Dettmers et al.) be
    compressed more aggressively than quiet ones, without paying for a
    full per-channel variance estimate.
    """

    def __init__(
        self,
        dim: int,
        alpha_init: float = 0.5,
        p_init: float = 2.0,
        learnable_p: bool = True,
        beta: bool = True,
        alpha_jitter: float = 0.05,
    ):
        super().__init__()
        # Per-channel alpha. Initialized with a tiny random jitter so that
        # channels can diverge from day-1 if the optimizer wants them to.
        alpha = torch.full((dim,), float(alpha_init))
        # Small symmetric jitter, kept well below alpha_init so we start
        # close to the original DyT operating point. Set alpha_jitter=0 for
        # fully deterministic initialization.
        if alpha_jitter > 0:
            alpha = alpha + alpha_jitter * torch.randn(dim) * alpha_init
        self.alpha = nn.Parameter(alpha)

        # Curvature exponent p. p=1 = classic softsign, p=2 = smooth
        # softsign, p->inf = hard tanh (clip to [-1, 1]).
        self.p = nn.Parameter(torch.tensor(float(p_init))) if learnable_p else float(p_init)

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTCForward.apply(x, self.alpha, self.p, self.gamma, self.beta)


class _DyTCForward(torch.autograd.Function):
    """
    Fused DyTC forward: alpha * x -> generalized_softsign -> affine

    Only the *scaled* activation crosses into the backward next to the
    per-channel parameters: the pre-affine activation, the branch value and
    the pow base are all re-derived from it and `p`. `alpha * x` is the one
    """
    @staticmethod
    def forward(ctx, x, alpha, p, gamma, beta):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            # See `_MIN_P`: the exponent the forward math sees is floored, the
            # raw one is kept so the backward can apply the clamp's own
            # zero-gradient below the floor.
            p_raw = p
            p = p.clamp(min=_MIN_P)
        x_scaled = x * alpha
        phi = _softsign_forward(x_scaled, p)

        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi.mul_(gamma)

        # `p` has by now been replaced by its clamped copy, which carries no
        # grad flag inside `forward`; the *raw* input is what says whether the
        # exponent is trainable, so the flag is read off that.
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x_scaled, alpha, gamma, beta]
        if p_is_tensor:
            saved.append(p_raw)
        else:
            # `learnable_p=False` makes `p` a plain Python float, which
            # `save_for_backward` rejects ("can only save variables").
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # `ctx.saved_tensors` may only be unpacked once per backward.
        saved = ctx.saved_tensors
        x_scaled, alpha, gamma, beta = saved[:4]
        p_is_tensor = ctx.p_is_tensor
        p_raw = saved[4] if p_is_tensor else None
        if p_is_tensor:
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value

        # `phi` is the squashed activation and `out` the affine of it. They are
        # not interchangeable downstream: `grad_gamma` is dL/dgamma =
        # sum(grad_out * phi), and `dphi/dp` is proportional to phi as well, so
        # the *pre*-affine tensor is the one the backward needs. Both are
        # rebuilt from the same pieces, so they are settled together and
        # *before* the input-gradient chain that consumes `g`.
        need_softsign = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[2]
        need_gamma_grad = ctx.needs_input_grad[3]
        need_phi = need_p_grad or need_gamma_grad
        phi = None
        a = None
        if need_softsign or need_phi:
            g, phi, a = _softsign_backward_pieces(
                x_scaled, p, need_p_grad, need_phi, need_softsign
            )

        p_val = p if p_is_tensor else float(p)
        grad_p = grad_gamma = None
        if need_phi:
            # on it rather than beside it, and the `gamma` fold for the
            grad_phi = phi.mul_(grad_out)
            if need_gamma_grad:
                grad_gamma = _sum_leading_dims(grad_phi)
            if need_p_grad:
                grad_p = _softsign_p_grad(grad_phi.mul_(gamma), a, p, p_raw)
            del phi, a, grad_phi

        grad_x = grad_alpha = None
        if need_softsign:
            # dphi/dx_scaled = g^(p+1), taken as exp((p+1) * log(g)) (see
            # `_GeneralizedSoftsign.backward`). The output affine is folded
            # into the chain as two in-place scales rather than as one
            # pre-multiplied `grad_out * gamma` tensor.
            grad_x_scaled = g.log_().mul_(p_val + 1).exp_()
            grad_x_scaled.mul_(grad_out).mul_(gamma)
            if ctx.needs_input_grad[1]:
                # x_scaled = x * alpha, so d/dalpha = grad_x_scaled * x =
                # grad_x_scaled * x_scaled / alpha - the division is taken on
                grad_alpha = _sum_leading_dims(
                    grad_x_scaled * x_scaled
                ).div_(alpha)
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha)

        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = _sum_leading_dims(grad_out)

        return grad_x, grad_alpha, grad_p, grad_gamma, grad_beta


# ---------------------------------------------------------------------------
# Variant B: DyT-D  (depth-conditioned alpha + LayerScale)
# ---------------------------------------------------------------------------
@register_norm("dytd")
class DyTD(_SquashedAffine):
    """
    Depth-conditioned DyT, addresses the "curse of depth" in Pre-LN
    Transformers.

        alpha_l = alpha_0 * exp(-k * l / L)
        y       = gamma * tanh(alpha_l * x) + beta

    where l is the layer index and L is the total depth. Combined with a
    LayerScale-style residual gate


    because alpha_l decays with depth, shallow layers squash strongly
    (bounding the pre-norm activations where they are largest) while deep
    layers relax toward a near-linear, gradient-friendly response.

    only implements the depth-conditioned squashing and exposes
    `alpha_l` so the trainer can log the learned schedule.
    """

    def __init__(
        self,
        dim: int,
        layer_idx: int = 0,
        num_layers: int = 12,
        alpha_0: float = 0.9,
        k_init: float = 1.0,
        learnable_k: bool = True,
        beta: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)

        # Global "base" alpha (still a single scalar, like original DyT).
        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))

        # Depth-decay rate. k>0 => alpha_l decays with depth, so deeper
        # layers squash more gently (toward a near-linear response).
        # We expose a small positive init so the schedule is non-trivial
        # from the start.
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer(
                "k", torch.tensor(float(k_init))
            )

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    @property
    def alpha_l(self) -> torch.Tensor:
        """The effective scalar alpha at this depth (for logging)."""
        return self.alpha_0 * torch.exp(-self.k * self._depth_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTDForward.apply(x, self.alpha_0, self.k, self.gamma, self.beta, self._depth_ratio)


class _DyTDForward(torch.autograd.Function):
    """
    Fused DyTD forward: alpha_l * x -> tanh -> affine
    """
    @staticmethod
    def forward(ctx, x, alpha_0, k, gamma, beta, depth_ratio):
        exp_kdr = torch.exp(-k * depth_ratio)
        a = alpha_0 * exp_kdr
        # `phi` (pre-affine) is the retained activation: the tanh derivative
        # post-affine `out` would differentiate `gamma * tanh(.) + beta`.
        # place on it.
        phi = a.mul(x)
        phi.tanh_()
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi * gamma
        ctx.save_for_backward(x, alpha_0, k, gamma, beta, phi, exp_kdr.detach())
        ctx.depth_ratio = depth_ratio
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha_0, k, gamma, beta, phi, exp_kdr = ctx.saved_tensors
        depth_ratio = ctx.depth_ratio

        a = alpha_0 * exp_kdr
        # that `dL/dgamma` reduces and is then walked to the input gradient,
        # which it *is* on the way out.
        w = phi * grad_out
        grad_gamma = None
        if ctx.needs_input_grad[3]:
            grad_gamma = _sum_leading_dims(w)
        # d/d(a*x) of tanh, i.e. `1 - phi^2`, with `gamma` folded in after.
        # Taken as `gamma * (grad_out - grad_out * phi^2)` - the same product,
        # factored - rather than as `grad_out * gamma - (grad_out * gamma) *
        # phi^2`, so the cotangent (which belongs to the caller) never has to
        # The factored form also cancels nothing: near saturation
        # `1 - phi^2` is computed directly instead of as the difference of two
        w.mul_(phi)
        torch.sub(grad_out, w, out=w)
        w.mul_(gamma)
        # grad_a = sum(dL/d(a*x) * x)
        # A single flat contraction over both operands rather than a
        grad_a = _dot_all(w, x)
        # d/dx tanh(ax) = a * (1 - tanh^2(ax))
        w.mul_(a)
        grad_x = w

        # a = alpha_0 * exp(-k * depth_ratio)
        # grad_alpha_0 = grad_a * exp(-k * depth_ratio)
        # grad_k       = grad_a * alpha_0 * exp(-k * r) * (-r) = -grad_a * a * r
        grad_alpha_0 = grad_a * exp_kdr
        grad_k = -grad_a * a * depth_ratio

        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = _sum_leading_dims(grad_out)

        return grad_x, grad_alpha_0, grad_k, grad_gamma, grad_beta, None


# ---------------------------------------------------------------------------
# Original DyT  (Zhu et al., 2025)
# ---------------------------------------------------------------------------
@register_norm("dyt")
class DyT(_SquashedAffine):
    """y = gamma * tanh(alpha * x) + beta. alpha is a global scalar."""

    def __init__(
        self,
        dim: int,
        alpha_init: float = 0.6,
        beta: bool = True,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTForward.apply(x, self.alpha, self.gamma, self.beta)


class _DyTForward(torch.autograd.Function):
    """
    Fused DyT forward: alpha * x -> tanh -> affine
    """
    @staticmethod
    def forward(ctx, x, alpha, gamma, beta):
        # `phi` is the squashed activation and the one the node retains: the
        # tanh derivative is `1 - phi^2` and `dL/dgamma` is `sum(grad_out *
        # phi)`. Retaining the post-affine output instead would make both read
        # the affine as if it were the activation.
        # place on it.
        phi = alpha.mul(x)
        phi.tanh_()
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi * gamma
        ctx.save_for_backward(x, alpha, gamma, beta, phi)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha, gamma, beta, phi = ctx.saved_tensors

        # `w = grad_out * phi` is the `dL/dgamma` reduction and, walked on
        # through the tanh derivative, the input gradient too - so the two
        w = phi * grad_out
        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = _sum_leading_dims(w)
        # `w = dL/d(alpha * x)`, shared by the input and the scalar alpha
        # gradient so the (1 - phi^2) factor is evaluated once. Factored as
        # `gamma * (grad_out - grad_out * phi^2)` so the caller's cotangent
        # be scaled by `gamma` first.
        w.mul_(phi)
        torch.sub(grad_out, w, out=w)
        w.mul_(gamma)
        grad_alpha = _dot_all(w, x)
        w.mul_(alpha)
        grad_x = w

        grad_beta = None
        if beta is not None and ctx.needs_input_grad[3]:
            grad_beta = _sum_leading_dims(grad_out)

        return grad_x, grad_alpha, grad_gamma, grad_beta


# ---------------------------------------------------------------------------
# Variant A+B combined: DyT-CD  (the main DyT++ proposal)
# ---------------------------------------------------------------------------
@register_norm("dytcd")
class DyTCD(_SquashedAffine):
    """
    DyT++ main layer: per-channel softsign with depth-conditioned scale.

        alpha_c,l = alpha_0 * exp(-k * l / L) * (1 + delta_c)
        x_hat     = alpha_c,l * x
        phi       = x_hat / (1 + |x_hat|^p)^{1/p}
        y         = gamma * phi + beta

    Combines the outlier-adaptive per-channel compression of DyT-C with the
    depth-aware schedule of DyT-D. Parameters added over DyT:
      * delta_c: [C]      (~0.1% of model size)
      * k: scalar          (1 number per layer)
      * p: scalar          (1 number per layer, optional)
    """

    def __init__(
        self,
        dim: int,
        layer_idx: int = 0,
        num_layers: int = 12,
        alpha_0: float = 0.9,
        k_init: float = 1.0,
        learnable_k: bool = True,
        p_init: float = 2.0,
        learnable_p: bool = True,
        beta: bool = True,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_layers = max(1, num_layers)
        self._depth_ratio = float(layer_idx) / float(self.num_layers)

        self.alpha_0 = nn.Parameter(torch.tensor(float(alpha_0)))
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(float(k_init)))
        else:
            self.register_buffer("k", torch.tensor(float(k_init)))

        # Per-channel modulation, initialized to *zero* so the per-channel
        # factor starts at alpha_c,l == alpha_l (no per-channel deviation).
        # A trained DyT-D checkpoint can then seed the shared alpha_0 / k /
        # gamma / beta, but the layers are *not* equivalent: DyT-D applies
        # tanh while this layer applies the generalized softsign with a
        # learnable p, so the transfer is a warm start, not a lossless
        # upgrade.
        self.delta = nn.Parameter(torch.zeros(dim))

        if learnable_p:
            self.p = nn.Parameter(torch.tensor(float(p_init)))
        else:
            self.register_buffer("p", torch.tensor(float(p_init)))

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    @property
    def alpha_per_channel(self) -> torch.Tensor:
        """Effective per-channel alpha at this depth (for logging)."""
        scale = self.alpha_0 * torch.exp(-self.k * self._depth_ratio)
        return torch.addcmul(scale, scale, self.delta).clamp_(min=_MIN_ALPHA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _DyTCDForward.apply(
            x, self.alpha_0, self.k, self.delta, self.p,
            self.gamma, self.beta, self._depth_ratio,
        )


class _DyTCDForward(torch.autograd.Function):
    """
    Fused DyTCD forward: alpha_per_channel * x -> generalized_softsign -> affine
    """
    @staticmethod
    def forward(ctx, x, alpha_0, k, delta, p, gamma, beta, depth_ratio):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            # See `_MIN_P`: the exponent the forward math sees is floored, the
            # raw one is kept so the backward can apply the clamp's own
            # zero-gradient below the floor.
            p_raw = p
            p = p.clamp(min=_MIN_P)
        # Compute alpha_per_channel: alpha_0 * exp(-k * depth_ratio) * (1 + delta)
        exp_kdr = torch.exp(-k * depth_ratio)
        scale = alpha_0 * exp_kdr
        alpha_c = torch.addcmul(scale, scale, delta).clamp_(min=_MIN_ALPHA)

        # generalized_softsign on `x * alpha_c` (broadcast over [B, T, C])
        x_scaled = x * alpha_c
        phi = _softsign_forward(x_scaled, p)

        # Affine: gamma * phi + beta
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi.mul_(gamma)

        # `x_scaled` is kept in the input's place: the backward rebuilds the
        # softsign pieces straight from it instead of re-deriving the
        # per-channel product. The alpha chain divides it back out, which
        # is exact here because `alpha_c` is the *clamped* value the product
        # was taken with and the floor keeps the divisor away from zero.
        # `s` is deliberately *not* kept: it is re-derived from `x_scaled`
        # and `p` in the backward.
        # `p` has by now been replaced by its clamped copy, which carries no
        # grad flag inside `forward`; the *raw* input is what says whether the
        # exponent is trainable, so the flag is read off that.
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
        # `ctx.saved_tensors` may only be unpacked once per backward.
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

        # Gradient w.r.t. output of softsign
        need_softsign = (
            ctx.needs_input_grad[0]
            or ctx.needs_input_grad[1]
            or ctx.needs_input_grad[2]
            or ctx.needs_input_grad[3]
        )
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[4]
        need_gamma_grad = ctx.needs_input_grad[5]
        need_phi = need_p_grad or need_gamma_grad
        phi = None
        a = None
        if need_softsign or need_phi:
            g, phi, a = _softsign_backward_pieces(
                x_scaled, p, need_p_grad, need_phi, need_softsign
            )

        p_val = p if p_is_tensor else float(p)
        # `phi` and the exponent's `a` belong to the rebuilt pieces; they are
        # drained before the input-gradient chain below.
        grad_p = grad_gamma = None
        if need_phi:
            # Gradient w.r.t. p (if learnable) and w.r.t. gamma: one
            # `grad_out * phi` product feeds both reductions.
            # lands on it rather than beside it (see `_DyTCForward.backward`).
            grad_phi = phi.mul_(grad_out)
            if need_gamma_grad:
                grad_gamma = _sum_leading_dims(grad_phi)
            if need_p_grad:
                grad_p = _softsign_p_grad(grad_phi.mul_(gamma), a, p, p_raw)
            del phi, a, grad_phi

        grad_x = grad_alpha_0 = grad_k = grad_delta = None
        if need_softsign:
            # Gradient of generalized_softsign w.r.t. input (x_scaled)
            # dphi/dx = g^(p+1) where g = s * f, taken as exp((p+1) * log(g))
            # (see `_GeneralizedSoftsign.backward`). The output affine is
            # folded in as two in-place scales instead of a pre-multiplied
            grad_x_scaled = g.log_().mul_(p_val + 1).exp_()
            grad_x_scaled.mul_(grad_out).mul_(gamma)

            # Gradient w.r.t. alpha_c (and thus alpha_0, k, delta)
            # x_scaled = x * alpha_c, so d/dalpha_c = grad_x_scaled * x =
            # grad_x_scaled * x_scaled / alpha_c - the division is taken on the
            # [C] reduction.
            need_alpha_chain = (
                ctx.needs_input_grad[1]
                or ctx.needs_input_grad[2]
                or ctx.needs_input_grad[3]
            )
            if need_alpha_chain:
                grad_alpha_c = _sum_leading_dims(
                    grad_x_scaled * x_scaled
                ).div_(alpha_c)

                # alpha_c = clamp_min(scale * (1 + delta), _MIN_ALPHA), scale =
                # alpha_0 * exp(-k * depth_ratio). `alpha_c` is the *clamped*
                grad_alpha_c.mul_((alpha_c > _MIN_ALPHA).to(grad_alpha_c.dtype))

                # scale = alpha_0 * exp(-k * depth_ratio)
                # grad_delta      = grad_alpha_c * scale
                # grad_alpha_0    = grad_scale * exp(-k * depth_ratio)
                # grad_k          = grad_scale * alpha_0 * exp(-k * r) * (-r)
                #                 = -grad_scale * scale * r
                # with `grad_scale = grad_alpha_c * (1 + delta)`. `grad_scale`
                # only feeds the two *scalar* schedules, so a layer that
                # trains `delta` alone does not build it, and the shared
                # `exp(-k * r)` is taken once for whichever of the three are
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

            # Gradient w.r.t. x.
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha_c)

        # Gradient w.r.t. beta
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[6]:
            grad_beta = _sum_leading_dims(grad_out)

        return grad_x, grad_alpha_0, grad_k, grad_delta, grad_p, grad_gamma, grad_beta, None


# ---------------------------------------------------------------------------
# expose a small helper here so the attention code can pick it up cleanly.
# ---------------------------------------------------------------------------
class QKSquash(nn.Module):
    """
    Squash Q/K before the dot product, replacing QK-Norm.

        Q' = tanh(alpha_q * Q);  K' = tanh(alpha_k * K)
        logits = (Q' @ K'.T) * scale

    `tanh` bounds every element to [-1, 1], so the dot product over
    `head_dim` terms is bounded by +-head_dim *before* the
    `1/sqrt(head_dim)` scale, i.e. +-sqrt(head_dim) after it - a ceiling
    that does not move with the activation magnitude. RMSNorm rescales
    without saturating, so one dominant channel can still drive the logit;
    `tanh` clips it. Only the incoming activations are touched, so the
    module stays a pure element-wise function of Q/K.

    The compression is `torch.tanh` on the scaled Q/K. The scale owns a
    """

    def __init__(self, alpha_q_init: float = 1.0, alpha_k_init: float = 1.0):
        super().__init__()
        self.alpha_q = nn.Parameter(torch.tensor(float(alpha_q_init)))
        self.alpha_k = nn.Parameter(torch.tensor(float(alpha_k_init)))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        q = q * self.alpha_q
        k = k * self.alpha_k
        # compression is taken on it in place.
        q.tanh_()
        k.tanh_()
        return q, k


__all__ = [
    "NORM_REGISTRY",
    "register_norm",
    "build_norm",
    "generalized_softsign",
    "LayerNorm",
    "RMSNorm",
    "DyT",
    "DyTC",
    "DyTD",
    "DyTCD",
    "QKSquash",
]
