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
Transformer block.

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
import torch.nn.functional as F

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


def register_norm(name: str):
    def _decorator(cls):
        NORM_REGISTRY[name] = cls
        cls._norm_name = name
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Shared generalized-softsign core
# ---------------------------------------------------------------------------
# The three hand-written kernels below (the standalone `generalized_softsign`,
# DyT-C and DyT-CD) all evaluate the same family, so the element-wise core
# lives here rather than being open-coded three times.
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
# `r` doubles as the derivative's branch factor, so the kernel that would
# otherwise have been `torch.where(t > 1, u, 1)` is a free alias.
#
# The reciprocal is taken on `max(t, 1)`, never on `t` itself: an exact zero
# (a bf16 underflow, a zeroed/padded element) would otherwise give
# `1/0 = inf` and `d(1/t)/dt = -1/t^2` on the unused branch NaNs the whole
# gradient. `clamp` contributes no gradient of its own, so the selected branch
# is still differentiated exactly and phi'(0) = 1 as the closed form requires.
def _softsign_forward(xs: torch.Tensor, p):
    """Element-wise core of `phi_p` on the signed scaled input `xs`.

    Returns `(s, phi)` with `phi = sign(xs) * lo * s` and `s = (1 + u^p)^(-1/p)`.
    `s` is what the backward keeps: every other quantity here is a product of
    it with a cheap element-wise function of the input, so the node retains one
    activation instead of the three the branch split would otherwise pin down
    (the output, the branch value and the base of the pow chain).
    """
    # `lo` and `hi` are carved out of a single `abs()` buffer: after `lo` has
    # been materialized in full, the buffer is free to be reused in place.
    t = xs.abs()
    lo = t.clamp(max=1.0)
    hi = t.clamp_(min=1.0).reciprocal_()
    u = lo.mul(hi)
    # s = (1 + u^p)^(-1/p), evaluated through log1p on a base <= 1.
    s = u.pow(p).log1p_().mul_(-1.0 / p).exp_()
    # The sign rides on the sign bit of `xs` rather than on a separate
    # `torch.sign` pass, so `|x| = inf` still yields +-1 instead of inf * 0.
    phi = lo.mul_(s).copysign_(xs)
    return s, phi


def _softsign_backward_pieces(
    xs: torch.Tensor, s: torch.Tensor, need_u: bool, need_phi: bool = True
):
    """Rebuild the backward-only pieces of `phi_p` from the retained pair.

    Returns `(g, phi, u)`, where `g = s * f` is the single combination the
    `x`-derivative depends on (`dphi/dx = g^(p+1)`), `phi` is the pre-affine
    activation `dL/dgamma` and `dphi/dp` both reduce over, and `u` is only
    built when the exponent itself carries a gradient. `phi` is skipped the
    same way when neither reduction reads it. `t` is carved in place exactly
    as in the forward.
    """
    t = xs.abs()
    lo = t.clamp(max=1.0)
    r = t.clamp_(min=1.0).reciprocal_()
    phi = lo.mul(s).copysign_(xs) if need_phi else None
    u = lo.mul_(r) if need_u else None
    return r.mul_(s), phi, u


def _softsign_p_grad(
    grad_softsign: torch.Tensor,
    phi: torch.Tensor,
    u: torch.Tensor,
    p: torch.Tensor,
    p_raw: torch.Tensor,
) -> torch.Tensor:
    """Reduced `dL/dp` for the generalized softsign, in one fused pass.

    With `A = p * log(u)` the closed form's two compound terms

        log1p(u^p) = softplus(A),   u^p / (1 + u^p) = sigmoid(A)

    let `dphi/dp = phi * [softplus(A)/p^2 - sigmoid(A) * log(u) / p]` be
    evaluated without ever materializing `u^p`, its reciprocal, or the
    `u > 0` branch mask as full-size tensors: `pow` + `log1p` + `reciprocal`
    + `where` collapse onto two fused kernels. The `log` is taken on `u`
    clamped off zero exactly as the `pow` form clamped it, so a zeroed /
    underflowed element stays finite; `phi` is zero there and zeroes the
    contribution either way.
    """
    log_u = u.clamp(min=torch.finfo(u.dtype).tiny).log()
    a = p * log_u
    k = F.softplus(a).div_(p * p)
    k.addcmul_(log_u, torch.sigmoid(a), value=-(1.0 / p))
    # Reduce the [N, C] products in fp32 under bf16 autocast: a bf16
    # accumulator over a whole activation block cancels most of its own
    # mantissa bits before the scalar survives.
    acc = torch.float32 if u.dtype in (torch.bfloat16, torch.float16) else None
    # `grad_softsign * phi` is a fresh buffer, so the second factor can be
    # folded into it in place instead of pinned as a second activation.
    red = (grad_softsign * phi).mul_(k).sum(dtype=acc)
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

    The affine itself is evaluated inside each layer's fused autograd kernel
    (as one `addcmul` rather than a multiply followed by an add).
    """


# ---------------------------------------------------------------------------
# Generalized softsign
# ---------------------------------------------------------------------------
class _GeneralizedSoftsign(torch.autograd.Function):
    """
    Autograd kernel behind `generalized_softsign`.

    The forward pass is the split closed form (see the public wrapper below).
    The backward pass is written out analytically instead of being recorded
    from the element-wise chain: the chain reuses every intermediate - the
    absolute value, the clamped reciprocal, both `where` outputs, the powered
    base and the denominator - which pins down eight full activations per
    layer for the whole backward pass.

    With `u = |x|` for `|x| <= 1` and `1/|x|` otherwise, and `s = (1 + u^p)^(-1/p)`:

        phi     = sign(x) * min(|x|, 1) * s
        dphi/dx = (s * f)^(p+1),   f = 1 for |x| <= 1, u for |x| > 1
                = s^(p+1)           for |x| <= 1
                = (s*u)^(p+1)       for |x| >  1
        dphi/dp = phi * [log1p(u^p)/p^2 - (1 - s^p) * log(u) / p]

    `s * f` is the only combination the `x` derivative depends on, and `s` is
    also the factor every other output is built from, so `s` alone is what the
    node retains - one activation per layer instead of the derivative factor,
    the output and the base of the pow chain. `g`, `phi` and `u` are cheap
    element-wise products of it that the backward rebuilds from the retained
    input, which the caller was holding anyway.
    """
    @staticmethod
    def forward(ctx, x, p):
        p_is_tensor = isinstance(p, torch.Tensor)
        if p_is_tensor:
            # Keep the raw `p` for the backward pass (the clamp's own zero
            # gradient is applied there) and floor the exponent the forward
            # math actually sees.
            p_raw = p
            p = p.clamp(min=_MIN_P)
        s, out = _softsign_forward(x, p)

        ctx.p_is_tensor = p_is_tensor
        # `p` has by now been replaced by its clamped copy, which carries no
        # grad flag inside `forward`; the *raw* input is what says whether the
        # exponent is trainable, so the flag is read off that.
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x, s]
        if p_is_tensor:
            saved.append(p_raw)
        else:
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        x, s = saved[0], saved[1]
        if ctx.p_is_tensor:
            p_raw = saved[2]
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value

        grad_x = grad_p = None
        need_x = ctx.needs_input_grad[0]
        need_p = ctx.needs_input_grad[1] and ctx.need_p_grad
        if need_x or need_p:
            g, phi, u = _softsign_backward_pieces(x, s, need_p, need_p)
        if need_x:
            # The |x| <= 1 branch reduces to s^(p+1) because
            # d/dx (1+x^p)^(-1/p-1) = (1+x^p)^(-1/p-1). `g` is the private
            # buffer the rebuild handed over, so the exponent runs in place.
            grad_x = g.pow_(p + 1).mul_(grad_out)
        if need_p:
            grad_p = _softsign_p_grad(grad_out, phi, u, p, p_raw)
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

    so only a single pow chain is materialized: `s = (1 + u^p)^(-1/p)` and
    `phi = sign(x) * lo * s`. This keeps phi in [-1, 1], avoids overflow for
    large p / large |x|, and stays finite for |x| = inf (where the naive
    closed form would divide inf by inf).

    The reciprocal is taken on `max(|x|, 1)` rather than on `|x|` directly.
    `torch.where` masks the *value* of the unused branch, but the backward
    pass still runs `d/dt (1/t) = -1/t^2` on it, so an exact zero
    (bf16 underflow of a small activation, a padded/zeroed element, ...) would
    evaluate `0 * -inf = NaN` and poison every gradient that reaches this
    layer. Clamping first keeps the unused branch finite; `clamp` itself
    contributes no gradient below its threshold, so the selected branch is
    still differentiated exactly and phi'(0) = 1 as the closed form requires.

    The layer runs as a custom autograd node (see `_GeneralizedSoftsign`) so
    the backward pass can evaluate the closed-form derivatives directly.
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

    The obvious composition (`x.pow(2)`, `rsqrt`, two multiplies) makes
    autograd retain both the input and the normalized activation - the latter
    solely so the weight gradient can reduce over it. The backward pass below
    rebuilds that product from the retained input and the per-row `rstd`
    instead, so only one full activation per layer is kept.

    The forward reduction and the backward products are arranged so that no
    full-size tensor is built twice: the sum of squares comes out of the
    fused norm reduction, and `grad_out * rstd * weight` - the leading term
    of `grad_x` and the pre-scaled operand behind `mean(t * x)` - is formed
    once and used for both reductions.
    """

    @staticmethod
    def forward(ctx, x, weight, eps):
        # `vector_norm` reduces x*x without materializing the squares first.
        nrm = torch.linalg.vector_norm(x, 2, dim=-1, keepdim=True)
        rstd = torch.rsqrt(nrm.square_().div_(x.shape[-1]).add_(eps))
        ctx.save_for_backward(x, rstd, weight)
        return x * rstd * weight

    @staticmethod
    def backward(ctx, grad_out):
        x, rstd, weight = ctx.saved_tensors
        # `gw = grad_out * rstd * weight` is both the leading term of d/dx
        # and the pre-scaled operand behind mean(t * x), so the reduction is
        # taken over `gw * x` rather than over a separately scaled copy.
        gu = grad_out * rstd
        gw = gu * weight
        grad_x = None
        if ctx.needs_input_grad[0]:
            # d/dx of `x * rstd`: the rsqrt term contributes
            # -x * rstd^3 * mean(t * x); the rstd factor that `m` already
            # carries leaves an rstd^2 to undo.
            m = (gw * x).mean(dim=-1, keepdim=True)
            grad_x = torch.addcmul(gw, x, m * rstd * rstd, value=-1)
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = (gu * x).sum(dim=tuple(range(x.ndim - 1)))
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

    Only `s` (see `_softsign_forward`) crosses into the backward next to the
    input itself: the pre-affine activation, the branch value and the pow base
    are all element-wise products of `s` that the backward rebuilds from `x`
    and `alpha`, so a layer holds one activation here instead of three.
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
        s, phi = _softsign_forward(x_scaled, p)

        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi.mul_(gamma)

        # `p` has by now been replaced by its clamped copy, which carries no
        # grad flag inside `forward`; the *raw* input is what says whether the
        # exponent is trainable, so the flag is read off that.
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x, alpha, gamma, beta, s]
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
        x, alpha, gamma, beta, s = saved[:5]
        p_is_tensor = ctx.p_is_tensor
        p_raw = saved[5] if p_is_tensor else None
        if p_is_tensor:
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value

        # `phi` is the squashed activation and `out` the affine of it. They are
        # not interchangeable downstream: `grad_gamma` is dL/dgamma =
        # sum(grad_out * phi), and `dphi/dp` is proportional to phi as well, so
        # the *pre*-affine tensor is the one the backward needs. Both are
        # rebuilt from `s`; keeping the post-affine `out` instead would
        # silently differentiate the affine.
        grad_softsign = grad_out * gamma
        need_softsign = ctx.needs_input_grad[0] or ctx.needs_input_grad[1]
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[2]
        need_gamma_grad = ctx.needs_input_grad[3]
        phi = None
        g = u = None
        if need_softsign or need_p_grad or need_gamma_grad:
            g, phi, u = _softsign_backward_pieces(
                x * alpha, s, need_p_grad, need_p_grad or need_gamma_grad
            )

        p_val = p if p_is_tensor else float(p)
        grad_x = grad_alpha = None
        if need_softsign:
            # dphi/dx_scaled = g^(p+1); the exponent runs in place on `g`,
            # which the rebuild handed over as a private buffer.
            grad_x_scaled = g.pow_(p_val + 1).mul_(grad_softsign)
            if ctx.needs_input_grad[1]:
                grad_alpha = (grad_x_scaled * x).sum(dim=tuple(range(x.ndim - 1)))
            if ctx.needs_input_grad[0]:
                # `grad_x_scaled` has already fed the alpha reduction, so its
                # buffer can become the input gradient instead of allocating
                # a second block for it.
                grad_x = grad_x_scaled.mul_(alpha)

        grad_p = None
        if need_p_grad:
            grad_p = _softsign_p_grad(grad_softsign, phi, u, p, p_raw)

        grad_gamma = None
        if need_gamma_grad:
            grad_gamma = (grad_out * phi).sum(dim=tuple(range(phi.ndim - 1)))

        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))

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

        x_out = x + lambda_l * Block(y)

    because alpha_l decays with depth, shallow layers squash strongly
    (bounding the pre-norm activations where they are largest) while deep
    layers relax toward a near-linear, gradient-friendly response.

    The full LayerScale is applied in the model's Block.forward; this module
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
        a = alpha_0 * torch.exp(-k * depth_ratio)
        # `phi` (pre-affine) is the retained activation: the tanh derivative
        # reads `1 - phi^2` and `dL/dgamma` reads `phi`, so saving the
        # post-affine `out` would differentiate `gamma * tanh(.) + beta`.
        phi = torch.tanh(a * x)
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi * gamma
        ctx.save_for_backward(x, alpha_0, k, gamma, beta, phi)
        ctx.depth_ratio = depth_ratio
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha_0, k, gamma, beta, phi = ctx.saved_tensors
        depth_ratio = ctx.depth_ratio

        a = alpha_0 * torch.exp(-k * depth_ratio)
        grad_tanh = grad_out * gamma
        # d/d(a*x) of tanh, i.e. `1 - phi^2`, taken once and reused by both
        # the input gradient and the scalar `a` gradient. Evaluated as
        # `grad_tanh - grad_tanh * phi^2` so the product stays one private
        # buffer that the input gradient can then take over in place.
        w = phi * phi
        w.mul_(grad_tanh)
        torch.sub(grad_tanh, w, out=w)
        # grad_a = sum(dL/d(a*x) * x)
        grad_a = (w * x).sum()
        # d/dx tanh(ax) = a * (1 - tanh^2(ax))
        w.mul_(a)
        grad_x = w

        # a = alpha_0 * exp(-k * depth_ratio)
        # grad_alpha_0 = grad_a * exp(-k * depth_ratio)
        # grad_k       = grad_a * alpha_0 * exp(-k * r) * (-r) = -grad_a * a * r
        grad_alpha_0 = grad_a * torch.exp(-k * depth_ratio)
        grad_k = -grad_a * a * depth_ratio

        grad_gamma = None
        if ctx.needs_input_grad[3]:
            grad_gamma = (grad_out * phi).sum(dim=tuple(range(phi.ndim - 1)))

        grad_beta = None
        if beta is not None and ctx.needs_input_grad[4]:
            grad_beta = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))

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
        phi = torch.tanh(alpha * x)
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi * gamma
        ctx.save_for_backward(x, alpha, gamma, beta, phi)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, alpha, gamma, beta, phi = ctx.saved_tensors

        grad_tanh = grad_out * gamma
        # `w = dL/d(alpha * x)`, shared by the input and the scalar alpha
        # gradient so the (1 - phi^2) factor is evaluated once. Same
        # single-buffer arrangement as `_DyTDForward.backward`.
        w = phi * phi
        w.mul_(grad_tanh)
        torch.sub(grad_tanh, w, out=w)
        grad_alpha = (w * x).sum()
        w.mul_(alpha)
        grad_x = w

        grad_gamma = None
        if ctx.needs_input_grad[2]:
            grad_gamma = (grad_out * phi).sum(dim=tuple(range(phi.ndim - 1)))

        grad_beta = None
        if beta is not None and ctx.needs_input_grad[3]:
            grad_beta = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))

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
        scale = alpha_0 * torch.exp(-k * depth_ratio)
        alpha_c = torch.addcmul(scale, scale, delta).clamp_(min=_MIN_ALPHA)

        # generalized_softsign on `x * alpha_c` (broadcast over [B, T, C])
        x_scaled = x * alpha_c
        s, phi = _softsign_forward(x_scaled, p)

        # Affine: gamma * phi + beta
        if beta is not None:
            out = torch.addcmul(beta, phi, gamma)
        else:
            out = phi.mul_(gamma)

        # `x_scaled` is kept in the input's place: it is the same [B, T, C]
        # footprint the unscaled input would have cost, and the backward can
        # rebuild the softsign pieces straight from it instead of re-deriving
        # the per-channel product. The alpha chain divides it back out, which
        # is exact here because `alpha_c` is the *clamped* value the product
        # was taken with and the floor keeps the divisor away from zero.
        # `alpha_c` is kept so the backward does not have to re-derive the
        # clamped per-channel product either.
        # `p` has by now been replaced by its clamped copy, which carries no
        # grad flag inside `forward`; the *raw* input is what says whether the
        # exponent is trainable, so the flag is read off that.
        ctx.p_is_tensor = p_is_tensor
        ctx.need_p_grad = p_is_tensor and p_raw.requires_grad
        saved = [x_scaled, alpha_0, k, delta, gamma, beta, s, alpha_c]
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
        x_scaled, alpha_0, k, delta, gamma, beta, s, alpha_c = saved[:8]
        depth_ratio = ctx.depth_ratio
        p_is_tensor = ctx.p_is_tensor
        p_raw = saved[8] if p_is_tensor else None
        if p_is_tensor:
            p = p_raw.clamp(min=_MIN_P)
        else:
            p = ctx.p_value

        # Gradient w.r.t. output of softsign
        grad_softsign = grad_out * gamma

        need_softsign = (
            ctx.needs_input_grad[0]
            or ctx.needs_input_grad[1]
            or ctx.needs_input_grad[2]
            or ctx.needs_input_grad[3]
        )
        need_p_grad = ctx.need_p_grad and ctx.needs_input_grad[4]
        need_gamma_grad = ctx.needs_input_grad[5]
        phi = None
        u = None
        if need_softsign or need_p_grad or need_gamma_grad:
            g, phi, u = _softsign_backward_pieces(
                x_scaled, s, need_p_grad, need_p_grad or need_gamma_grad
            )

        p_val = p if p_is_tensor else float(p)
        grad_x = grad_alpha_0 = grad_k = grad_delta = None
        if need_softsign:
            # Gradient of generalized_softsign w.r.t. input (x_scaled)
            # dphi/dx = g^(p+1) where g = s * f; the exponent runs in place on
            # the private buffer the rebuild handed over.
            grad_x_scaled = g.pow_(p_val + 1).mul_(grad_softsign)

            # Gradient w.r.t. alpha_c (and thus alpha_0, k, delta)
            # x_scaled = x * alpha_c, so d/dalpha_c = grad_x_scaled * x =
            # grad_x_scaled * x_scaled / alpha_c - the division is taken on the
            # [C] reduction rather than on a second full-size block.
            need_alpha_chain = (
                ctx.needs_input_grad[1]
                or ctx.needs_input_grad[2]
                or ctx.needs_input_grad[3]
            )
            if need_alpha_chain:
                grad_alpha_c = (grad_x_scaled * x_scaled).sum(
                    dim=tuple(range(x_scaled.ndim - 1))
                ).div_(alpha_c)

                # alpha_c = clamp_min(scale * (1 + delta), _MIN_ALPHA), scale =
                # alpha_0 * exp(-k * depth_ratio). `alpha_c` is the *clamped*
                # value, so it cannot be divided back into `scale` on the
                # channels the clamp is holding: the raw product is re-derived
                # instead, and the clamp's own zero-gradient is applied.
                scale = alpha_0 * torch.exp(-k * depth_ratio)
                one_plus_delta = 1.0 + delta
                live = (scale * one_plus_delta >= _MIN_ALPHA).to(grad_alpha_c.dtype)
                grad_alpha_c = grad_alpha_c * live

                grad_scale = grad_alpha_c * one_plus_delta

                # scale = alpha_0 * exp(-k * depth_ratio)
                # grad_alpha_0 = grad_scale * exp(-k * depth_ratio)
                # grad_k = grad_scale * alpha_0 * exp(-k * r) * (-r) = -grad_scale * scale * r
                if ctx.needs_input_grad[3]:
                    grad_delta = grad_alpha_c * scale
                if ctx.needs_input_grad[1]:
                    grad_alpha_0 = grad_scale * torch.exp(-k * depth_ratio)
                if ctx.needs_input_grad[2]:
                    grad_k = -grad_scale * scale * depth_ratio

            # Gradient w.r.t. x. `grad_x_scaled` has already fed the alpha
            # chain, so its buffer can become the input gradient in place.
            if ctx.needs_input_grad[0]:
                grad_x = grad_x_scaled.mul_(alpha_c)

        # Gradient w.r.t. p (if learnable)
        grad_p = None
        if need_p_grad:
            grad_p = _softsign_p_grad(grad_softsign, phi, u, p, p_raw)

        # Gradient w.r.t. gamma
        grad_gamma = None
        if need_gamma_grad:
            grad_gamma = (grad_out * phi).sum(dim=tuple(range(phi.ndim - 1)))

        # Gradient w.r.t. beta
        grad_beta = None
        if beta is not None and ctx.needs_input_grad[6]:
            grad_beta = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))

        return grad_x, grad_alpha_0, grad_k, grad_delta, grad_p, grad_gamma, grad_beta, None


# ---------------------------------------------------------------------------
# QK-squashing (Variant C) - lives inside the Attention module, but we
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
    """

    def __init__(self, alpha_q_init: float = 1.0, alpha_k_init: float = 1.0):
        super().__init__()
        self.alpha_q = nn.Parameter(torch.tensor(float(alpha_q_init)))
        self.alpha_k = nn.Parameter(torch.tensor(float(alpha_k_init)))

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return torch.tanh(self.alpha_q * q), torch.tanh(self.alpha_k * k)


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
