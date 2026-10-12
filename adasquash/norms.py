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

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
NORM_REGISTRY: dict[str, type] = {}


def register_norm(name: str):
    def _decorator(cls):
        NORM_REGISTRY[name] = cls
        cls._norm_name = name
        return cls

    return _decorator


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

    Written as one `addcmul` rather than as a multiply followed by an
    in-place add: the add would rewrite the squashed activation, which is
    retained by the softsign node for its own backward pass, so it can only
    run out of place - which costs a second full-size tensor. `addcmul`
    evaluates `beta + phi * gamma` in a single pass over the activation.
    """

    def _affine(self, x: torch.Tensor) -> torch.Tensor:
        if self.beta is None:
            return x * self.gamma
        return torch.addcmul(self.beta, x, self.gamma)


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

    `s * f` is the only combination the `x` derivative depends on, so that
    single product is what the node retains - one activation per layer
    instead of the input, the denominator and the branch mask. `u` and the
    output are added on top only when `p` itself carries a gradient, which
    is the one case that needs them.
    """
    @staticmethod
    def forward(ctx, x, p):
        t = x.abs()
        mask = t > 1.0
        # `lo` pins the output magnitude, `hi` keeps the reciprocal branch
        # finite; both are needed to evaluate the two branches below.
        lo = t.clamp(max=1.0)
        hi = t.clamp(min=1.0)
        hi.reciprocal_()
        u = torch.where(mask, hi, t)
        # s = (1 + u^p)^(-1/p), evaluated through log1p on a base <= 1.
        s = u.pow(p).log1p_().mul_(-1.0 / p).exp_()
        g = torch.where(mask, u, 1.0)
        g.mul_(s)
        # sign(x) is carried by the sign bit of x rather than by a separate
        # `torch.sign` pass, so |x| = inf still yields +-1 instead of inf*0.
        out = lo.mul_(s).copysign_(x)

        ctx.p_is_tensor = isinstance(p, torch.Tensor)
        ctx.need_p_grad = ctx.p_is_tensor and p.requires_grad
        saved = []
        if ctx.need_p_grad:
            saved.extend((out, u))
        saved.append(g)
        if ctx.p_is_tensor:
            saved.append(p)
        else:
            ctx.p_value = float(p)
        ctx.save_for_backward(*saved)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved = ctx.saved_tensors
        i = 0
        out = u = None
        if ctx.need_p_grad:
            out, u = saved[0], saved[1]
            i = 2
        g = saved[i]
        p = saved[i + 1] if ctx.p_is_tensor else ctx.p_value

        grad_x = grad_p = None
        need_x = ctx.needs_input_grad[0]
        need_p = ctx.needs_input_grad[1] and ctx.need_p_grad
        if need_x:
            # The |x| <= 1 branch reduces to s^(p+1) because
            # d/dx (1+x^p)^(-1/p-1) = (1+x^p)^(-1/p-1).
            grad_x = g.pow(p + 1).mul(grad_out)
        if need_p:
            a = u.pow(p)
            inv = a.add(1.0).reciprocal()  # = 1 / (1 + u^p) = s^p
            logu = u.clamp(min=torch.finfo(u.dtype).tiny).log()
            k = a.log1p() / (p * p) - (1.0 - inv) * logu / p
            # Reduce the [N, C] products in fp32 under bf16 autocast: a bf16
            # accumulator over a whole activation block cancels most of its
            # own mantissa bits before the scalar survives.
            acc = torch.float32 if u.dtype in (torch.bfloat16, torch.float16) else None
            red = (grad_out * out * k).sum(dtype=acc)
            grad_p = red.to(p.dtype) if ctx.p_is_tensor and red.dtype != p.dtype else red
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
        x = torch.tanh(self.alpha * x)
        return self._affine(x)


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
    ):
        super().__init__()
        # Per-channel alpha. Initialized with a tiny random jitter so that
        # channels can diverge from day-1 if the optimizer wants them to.
        alpha = torch.full((dim,), float(alpha_init))
        # Small symmetric jitter, kept well below alpha_init so we start
        # close to the original DyT operating point.
        alpha = alpha + 0.05 * torch.randn(dim) * alpha_init
        self.alpha = nn.Parameter(alpha)

        # Curvature exponent p. p=1 = classic softsign, p=2 = smooth
        # softsign, p->inf = hard tanh (clip to [-1, 1]).
        self.p = nn.Parameter(torch.tensor(float(p_init))) if learnable_p else float(p_init)

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.alpha
        return self._affine(generalized_softsign(x, self.p))


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
        a = self.alpha_l
        return self._affine(torch.tanh(a * x))


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

        # Per-channel modulation, initialized to *zero* so the layer starts
        # exactly as DyT-D would. This guarantees a smooth optimization
        # trajectory when the channel branch is added on top of a trained
        # DyT-D checkpoint, and makes ablation easy.
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
        # `scale * (1 + delta)` as a single `addcmul`, then clamp the result
        # in place: `addcmul` saves its two inputs, never its output, so the
        # bound costs no extra tensor.
        return torch.addcmul(scale, scale, self.delta).clamp_(min=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_c = self.alpha_per_channel  # [C]
        # Broadcast over [B, T, C]
        x = x * alpha_c

        # `p` may be a learnable scalar tensor or a buffer; both broadcast.
        return self._affine(generalized_softsign(x, self.p))


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
