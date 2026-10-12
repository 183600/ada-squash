"""
Normalization layers for the DyT++ / AdaSquash study.

Implements:
  * LayerNorm           - Pre-LN baseline (standard LayerNorm)
  * RMSNorm             - LLaMA-style RMSNorm baseline
  * DyT                 - Original Dynamic Tanh (Zhou et al., 2025)
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
converges to tanh as p -> infinity and to identity (with gentle clipping) as
p -> 1. p=2 is a particularly attractive operating point: it is C^inf, has a
bounded gradient, and avoids the saturation plateau of tanh in deep nets.
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
# Generalized softsign
# ---------------------------------------------------------------------------
def generalized_softsign(x: torch.Tensor, p) -> torch.Tensor:
    """
    phi_p(x) = x / (1 + |x|^p)^{1/p}.

    The closed form is split so the exponentiated base stays <= 1:

        |x| <= 1 :  phi = sign(x)|x| * (1 + |x|^p)^(-1/p)      (|x|^p  <= 1)
        |x| >  1 :  phi = sign(x)  *  (1 + |x|^-p)^(-1/p)      (|x|^-p <= 1)

    Both branches are then expressed through one shared pair of helpers,

        u   = |x| for |x| <= 1, 1/|x| for |x| > 1    (always <= 1)
        num = x for |x| <= 1, sign(x) for |x| > 1

    so only a single pow chain is materialized. This keeps phi in [-1, 1],
    avoids overflow for large p / large |x|, and stays finite for |x| = inf
    (where the naive closed form would divide inf by inf).
    """
    t = x.abs()
    w = t.clamp(min=1.0)
    mask = t > 1.0
    # Pow base: |x| (<= 1) below the threshold, 1/|x| above it. w == 1 keeps
    # the reciprocal well defined for the elements that do not need it.
    u = torch.where(mask, w.reciprocal(), t)
    # Numerator: |x| signed below the threshold, plain sign above it.
    num = torch.where(mask, torch.sign(x), x)
    return num / (u.pow(p) + 1.0).pow(1.0 / p)


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
        var = (x * x).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


# ---------------------------------------------------------------------------
# Original DyT  (Zhou et al., 2025)
# ---------------------------------------------------------------------------
@register_norm("dyt")
class DyT(nn.Module):
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
        x = x * self.gamma
        if self.beta is not None:
            x.add_(self.beta)
        return x


# ---------------------------------------------------------------------------
# Variant A: DyT-C  (per-channel alpha + generalized softsign)
# ---------------------------------------------------------------------------
@register_norm("dytc")
class DyTC(nn.Module):
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

        # Curvature exponent p. p=2 = softsign, p->inf = hard tanh.
        self.p = nn.Parameter(torch.tensor(float(p_init))) if learnable_p else float(p_init)

        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim)) if beta else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alpha * x
        # `p` may be a learnable scalar tensor or a plain float; both
        # broadcast against `x`.
        x = generalized_softsign(x, self.p)
        x = x * self.gamma
        if self.beta is not None:
            x.add_(self.beta)
        return x


# ---------------------------------------------------------------------------
# Variant B: DyT-D  (depth-conditioned alpha + LayerScale)
# ---------------------------------------------------------------------------
@register_norm("dytd")
class DyTD(nn.Module):
    """
    Depth-conditioned DyT, addresses the "curse of depth" in Pre-LN
    Transformers.

        alpha_l = alpha_0 * exp(-k * l / L)
        y       = gamma * tanh(alpha_l * x) + beta

    where l is the layer index and L is the total depth. Combined with a
    LayerScale-style residual gate

        x_out = x + lambda_l * Block(y)

    this lets the shallow layers stay close to identity (preserving
    gradient flow) while the deep layers apply progressively stronger
    squashing.

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

        # Depth-decay rate. k>0 => deeper layers compress harder.
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
        x = torch.tanh(a * x)
        x = x * self.gamma
        if self.beta is not None:
            x.add_(self.beta)
        return x


# ---------------------------------------------------------------------------
# Variant A+B combined: DyT-CD  (the main DyT++ proposal)
# ---------------------------------------------------------------------------
@register_norm("dytcd")
class DyTCD(nn.Module):
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
        # delta may be negative, in which case the channel gets *less*
        # compression than the global alpha. Clamp to keep alpha > 0.
        return (scale * (1.0 + self.delta)).clamp(min=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_c = self.alpha_per_channel  # [C]
        # Broadcast over [B, T, C]
        x = alpha_c * x

        # `p` may be a learnable scalar tensor or a buffer; both broadcast.
        x = generalized_softsign(x, self.p)
        x = x * self.gamma
        if self.beta is not None:
            x.add_(self.beta)
        return x


# ---------------------------------------------------------------------------
# QK-squashing (Variant C) - lives inside the Attention module, but we
# expose a small helper here so the attention code can pick it up cleanly.
# ---------------------------------------------------------------------------
class QKSquash(nn.Module):
    """
    Squash Q/K before the dot product, replacing QK-Norm.

        Q' = tanh(alpha_q * Q);  K' = tanh(alpha_k * K)
        logits = (Q' @ K'.T) * scale
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
