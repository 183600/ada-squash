"""
Stability / efficiency metrics for the DyT++ study.

Provides:
  * ActivationStats       - aggregated max|x| and kurtosis over all
                             activations that pass through a norm layer
  * LossSpikeCounter      - running count of loss spikes (relative jump)
  * KurtosisMeter         - online kurtosis estimator (Welford-style)
  * MaxAbsMeter           - online max |x| estimator
  * measure_activation_stats - one-shot hook-driven measurement pass
  * count_parameters     - param accounting helpers
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Per-tensor online meters
# ---------------------------------------------------------------------------
class MaxAbsMeter:
    """Track running max(|x|) per channel-magnitude bucket."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = -float("inf")
        self.count = 0

    def update(self, x: torch.Tensor):
        with torch.no_grad():
            v = x.abs().amax().item()
        if v > self.value:
            self.value = v
        self.count += 1


class KurtosisMeter:
    """
    Online kurtosis (excess) estimator following the Welford recurrence.
    The metric is computed over all flattened activations seen since the
    last reset. Lower kurtosis => fewer outliers, which is what we want
    from a stable squashing layer.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M4 = 0.0
        self._M3 = 0.0

    def update(self, x: torch.Tensor):
        # Detach + cast to f32 for numerically stable accumulation,
        # then delegate to a vectorized Welford update (Pébay 2008).
        x = x.detach().to(torch.float32).flatten()
        self._bulk_update(x)

    def _bulk_update(self, x: torch.Tensor):
        """Batched Welford update."""
        # Use the formulas from:
        #  Pébay, "Formulas for Robust, One-Pass Parallel Computation of
        #  Covariances and Arbitrary-Order Statistical Moments", 2008.
        new_n = self.n + x.numel()
        if new_n == self.n:
            return
        x_mean_t = x.mean()
        x_var_t = x.var(unbiased=False)
        xc = x - x_mean_t
        xc2 = xc * xc
        x_m3_t = (xc * xc2).sum()
        x_m4_t = (xc2 * xc2).sum()
        x_mean, x_var, x_m3, x_m4 = torch.stack(
            [x_mean_t, x_var_t, x_m3_t, x_m4_t]
        ).tolist()
        x_m2 = x_var * x.numel()
        n_a, n_b = self.n, x.numel()
        n = n_a + n_b
        delta = x_mean - self.mean
        if n_a == 0:
            self.mean = x_mean
            self.M2 = x_m2
            self._M3 = x_m3
            self.M4 = x_m4
            self.n = n
            return
        delta_ab = delta * delta
        delta_ab2 = delta_ab * delta_ab
        # M2 update
        new_M2 = self.M2 + x_m2 + delta_ab * n_a * n_b / n
        # M3 update
        new_M3 = (
            self._M3 + x_m3
            + delta_ab * delta * (n_a * n_b * (n_a - n_b) / (n * n))
            + 3.0 * delta * (n_a * x_m2 - n_b * self.M2) / n
        )
        # M4 update
        new_M4 = (
            self.M4 + x_m4
            + delta_ab2 * (n_a * n_b * (n_a * n_a - n_a * n_b + n_b * n_b) / (n ** 3))
            + 6.0 * delta_ab * (n_a * n_a * x_m2 + n_b * n_b * self.M2) / (n * n)
            + 4.0 * delta * (n_a * x_m3 - n_b * self._M3) / n
        )
        self.mean = (self.mean * n_a + x_mean * n_b) / n
        self.M2 = new_M2
        self._M3 = new_M3
        self.M4 = new_M4
        self.n = n

    def result(self) -> float:
        if self.n < 2:
            return 0.0
        var = self.M2 / self.n
        if var < 1e-12:
            return 0.0
        # Excess kurtosis = kurt - 3
        return (self.M4 / self.n) / (var * var) - 3.0


class ActivationStats:
    """
    Aggregator: register forward hooks on normalization modules.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.handles: list = []
        self.max_abs: Dict[str, MaxAbsMeter] = defaultdict(MaxAbsMeter)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else None
            if x is None or not torch.is_tensor(x):
                return
            self.max_abs[name].update(x)
        return hook

    def attach(self):
        self.handles = []
        self.max_abs = defaultdict(MaxAbsMeter)
        for name, m in self.model.named_modules():
            cls_name = m.__class__.__name__
            if cls_name in {
                "LayerNorm", "RMSNorm", "DyT", "DyTC", "DyTD", "DyTCD",
            }:
                self.handles.append(m.register_forward_hook(self._make_hook(name)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def summary(self) -> dict:
        out = {}
        for name in self.max_abs:
            out[name] = {
                "max_abs": self.max_abs[name].value,
            }
        if out:
            all_max = [v["max_abs"] for v in out.values() if v["max_abs"] != -float("inf")]
            out["__summary__"] = {
                "mean_max_abs": sum(all_max) / max(1, len(all_max)),
                "max_max_abs": max(all_max) if all_max else 0.0,
            }
        return out


# ---------------------------------------------------------------------------
# Loss spike counter
# ---------------------------------------------------------------------------
class LossSpikeCounter:
    """
    """

    def __init__(self, ema_beta: float = 0.99, rel_threshold: float = 0.20, abs_threshold: float = 0.05):
        self.ema_beta = ema_beta
        self.rel_threshold = rel_threshold
        self.abs_threshold = abs_threshold
        self.ema = None
        self.n_spikes = 0
        self.history: list = []

    def update(self, loss_value: float) -> bool:
        is_spike = False
        if self.ema is not None:
            rel_jump = (loss_value - self.ema) / max(self.ema, 1e-6)
            abs_jump = loss_value - self.ema
            is_spike = (
                rel_jump > self.rel_threshold
                and abs_jump > self.abs_threshold
            )
            if is_spike:
                self.n_spikes += 1
        if self.ema is None:
            self.ema = loss_value
        else:
            self.ema = self.ema_beta * self.ema + (1 - self.ema_beta) * loss_value
        self.history.append((loss_value, self.ema, is_spike))
        return is_spike

    def reset(self):
        self.ema = None
        self.n_spikes = 0
        self.history = []


# ---------------------------------------------------------------------------
# One-shot measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_activation_stats(model: nn.Module, batch: torch.Tensor) -> dict:
    """
    Run a single forward pass on `batch`.
    """
    stats = ActivationStats(model)
    stats.attach()
    out = model(batch)
    if isinstance(out, tuple):
        out = out[0]
    s = stats.summary()
    stats.detach()
    return s


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(
        p.numel()
        for p in model.parameters()
        if (not trainable_only) or p.requires_grad
    )


__all__ = [
    "ActivationStats",
    "LossSpikeCounter",
    "KurtosisMeter",
    "MaxAbsMeter",
    "measure_activation_stats",
    "count_parameters",
]
