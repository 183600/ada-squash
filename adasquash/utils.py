"""
Stability metrics for the DyT++ study.

Provides:
   * ActivationStats       - aggregated max|x| and kurtosis over all
   * LossSpikeCounter      - running count of loss spikes (relative jump)
   * KurtosisMeter         - online kurtosis estimator (Welford-style)
   * MaxAbsMeter           - online max |x| estimator
   * markov_batch          - learnable synthetic corpus for the offline
                             experiments (ablations / MoE / W8A8 probe)
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
            v = x.abs().max().item()
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

    # Elements per bulk update: the recurrence is fed in slices regardless of
    # the batch the run is configured for. The parallel-moment merge below is
    # exact, so slicing changes nothing but the summation order.
    _chunk = 1 << 22

    # Datatype the moments are accumulated in; fp32 is ample for the bf16 /
    # fp32 activations sampled during training. Callers measuring activations
    # large enough for the fourth moment to overflow can raise this to
    # float64.
    acc_dtype = torch.float32

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M4 = 0.0
        self._M3 = 0.0

    def update(self, x: torch.Tensor):
        # Detach + widen for numerically stable accumulation,
        # then delegate to a vectorized Welford update (Pébay 2008).
        x = x.detach().flatten()
        if x.numel() <= self._chunk:
            self._bulk_update(x.to(self.acc_dtype))
            return
        # Slice first, widen per slice.
        for start in range(0, x.numel(), self._chunk):
            self._bulk_update(x[start:start + self._chunk].to(self.acc_dtype))

    def _bulk_update(self, x: torch.Tensor):
        """Batched Welford update."""
        # Use the formulas from:
        #  Covariances and Arbitrary-Order Statistical Moments", 2008.
        n_b = x.numel()
        new_n = self.n + n_b
        if new_n == self.n:
            return
        # `torch.var_mean` returns `(variance, mean)` in that order.
        x_var_t, x_mean_t = torch.var_mean(x, unbiased=False)
        xc = x - x_mean_t
        xc2 = xc * xc
        x_m3_t = torch.linalg.vecdot(xc, xc2)
        x_m4_t = torch.linalg.vecdot(xc2, xc2)
        x_mean, x_var, x_m3, x_m4 = torch.stack(
            [x_mean_t, x_var_t, x_m3_t, x_m4_t]
        ).tolist()
        x_m2 = x_var * n_b
        n_a, n_b_ = self.n, n_b
        n = n_a + n_b_
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
        new_M2 = self.M2 + x_m2 + delta_ab * n_a * n_b_ / n
        # M3 update
        new_M3 = (
            self._M3 + x_m3
            + delta_ab * delta * (n_a * n_b_ * (n_a - n_b_) / (n * n))
            + 3.0 * delta * (n_a * x_m2 - n_b_ * self.M2) / n
        )
        # M4 update
        new_M4 = (
            self.M4 + x_m4
            + delta_ab2 * (n_a * n_b_ * (n_a * n_a - n_a * n_b_ + n_b_ * n_b_) / (n ** 3))
            + 6.0 * delta_ab * (n_a * n_a * x_m2 + n_b_ * n_b_ * self.M2) / (n * n)
            + 4.0 * delta * (n_a * x_m3 - n_b_ * self._M3) / n
        )
        self.mean = (self.mean * n_a + x_mean * n_b_) / n
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
        self.kurtosis: Dict[str, KurtosisMeter] = defaultdict(KurtosisMeter)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) and len(inp) > 0 else None
            if x is None or not torch.is_tensor(x):
                return
            self.max_abs[name].update(x)
            self.kurtosis[name].update(x)
        return hook

    def attach(self, reset: bool = True):
        """Hook every norm module.

        `reset=False` keeps the meters accumulated so far, so a caller that
        re-attaches on a sampling window can pool statistics over the whole
        run instead of reporting only the last window. `MaxAbsMeter` takes a
        running maximum, so pooling is exact; `KurtosisMeter` merges moments,
        so the pooled value is the kurtosis over all samples seen.

        Any hook registered by a previous `attach` is removed first, and
        `detach` clears the whole handle list, so a re-attach neither doubles
        the meters nor leaves the abandoned hooks writing into them.
        """
        self.detach()
        if reset:
            self.max_abs = defaultdict(MaxAbsMeter)
            self.kurtosis = defaultdict(KurtosisMeter)
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
                "excess_kurtosis": self.kurtosis[name].result(),
            }
        if out:
            all_max = [v["max_abs"] for v in out.values() if v["max_abs"] != -float("inf")]
            all_kurt = [
                v["excess_kurtosis"] for v in out.values()
                if math.isfinite(v["excess_kurtosis"])
            ]
            peak = max(all_max) if all_max else 0.0
            out["__summary__"] = {
                "mean_max_abs": sum(all_max) / max(1, len(all_max)),
                "max_max_abs": peak,
                # Explicit alias for the metric name used in the README table:
                # the largest pre-norm activation magnitude seen by any norm.
                "max_abs_pre_norm": peak,
                "mean_excess_kurtosis_pre_norm": (
                    sum(all_kurt) / len(all_kurt) if all_kurt else 0.0
                ),
            }
        return out


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------
def markov_batch(
    rng: torch.Generator,
    ignore_index: int,
    batch_size: int,
    seq_len: int,
    device: str,
    n_states: int = 64,
    outlier_id: Optional[int] = None,
    max_jump: int = 3,
):
    """Synthetic corpus with *learnable* structure.

    Each row is an actual walk of a Markov chain over `n_states` hidden
    states:

        s_{t+1} = (s_t + j_{t+1}) % n_states,   j_t ~ U{0, ..., max_jump-1}

    and the emitted token is the state itself, so the target at position `t`
    is a deterministic function of the token at position `t-1` restricted to
    `max_jump` successors.

    Two failure modes this avoids:

    * Uniformly random token ids carry *no* next-token signal, so the
      optimal loss is `log(vocab_size)` no matter what the model does and
      the loss column cannot rank anything.
    * Drawing each position independently (`state_t ~ U`, no walk) is no
      better: the sum of two independent draws is still independent of the
      previous position, which silently reproduces the first failure while
      looking like a chain. Here the attainable full-vocab floor is
      `log(vocab_size) - (log(n_states) - log(max_jump))`, i.e. strictly
      below the uniform baseline for any `vocab_size > n_states`.

    `outlier_id` designates one token whose embedding row has been scaled
    up (see `scripts.ablation.install_outlier_channel`). Positions are drawn
    from it on a few percent of steps, which reproduces the "super-active
    dimension" regime the per-channel axis is meant to handle: an i.i.d.
    corpus never produces such a channel, so the ablations could not
    exercise the component under test.

    Returns `(x, y)` with `y[:, -1] == ignore_index`: the final position has
    no successor and is masked out of the loss.
    """
    start = torch.randint(0, n_states, (batch_size, 1), generator=rng, device=device)
    if seq_len > 1:
        # The walk is cumulative: s_t = (s_0 + sum_{i<=t} j_i) mod n, and the
        # whole reduction collapses into one cumsum over the jump draw.
        jumps = torch.randint(
            0, max_jump, (batch_size, seq_len - 1), generator=rng, device=device
        )
        x = torch.empty((batch_size, seq_len), dtype=start.dtype, device=device)
        x[:, :1] = start
        torch.cumsum(jumps, dim=1, out=x[:, 1:])
        del jumps
        x[:, 1:].add_(start).remainder_(n_states)
    else:
        x = start
    y = torch.full(
        (batch_size, seq_len), ignore_index, dtype=torch.long, device=device
    )
    y[:, :-1] = x[:, 1:]
    if outlier_id is not None:
        spike = torch.rand((batch_size, seq_len), generator=rng, device=device) < 0.03
        x[spike] = outlier_id
    return x, y


# ---------------------------------------------------------------------------
# Loss spike counter
# ---------------------------------------------------------------------------
class LossSpikeCounter:
    """
    Counts loss spikes: jumps away from an EMA of the loss that are large
    both in absolute and in relative terms.
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

    def state_dict(self) -> dict:
        """The part of the counter a checkpoint must carry to survive a
        `--resume`: `ema` is the baseline the next update compares against,
        so dropping it makes the first post-resume steps blind to a jump
        away from the pre-resume loss scale, and `n_spikes` is the running total.
        """
        return {"ema": self.ema, "n_spikes": self.n_spikes}

    def load_state_dict(self, state: Optional[dict]) -> bool:
        """Adopt a previously saved counter state.

        Returns True when a usable state was actually restored, so the
        caller can log it instead of silently starting a fresh count. A
        mapping without a finite `ema` restores nothing: the spike test is
        relative to that baseline, so adopting the count without it would
        keep counting spikes against a freshly seeded EMA.
        """
        if not isinstance(state, dict):
            return False
        ema = state.get("ema")
        if not (isinstance(ema, (int, float)) and math.isfinite(ema)):
            return False
        self.ema = float(ema)
        self.n_spikes = int(state.get("n_spikes") or 0)
        return True


# ---------------------------------------------------------------------------
# One-shot measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_activation_stats(model: nn.Module, batch: torch.Tensor) -> dict:
    """One hook-driven measurement pass over `batch`, summarised as a dict."""
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
    "markov_batch",
    "count_parameters",
]
