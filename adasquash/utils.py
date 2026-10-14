from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
import torch.nn as nn

class MaxAbsMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = -float('inf')
        self.count = 0

    def update(self, x: torch.Tensor):
        with torch.no_grad():
            v = x.abs().max().item()
        if v > self.value:
            self.value = v
        self.count += 1

class KurtosisMeter:
    _chunk = 1 << 22
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
        x = x.detach().flatten()
        if x.numel() <= self._chunk:
            self._bulk_update(x.to(self.acc_dtype))
            return
        for start in range(0, x.numel(), self._chunk):
            self._bulk_update(x[start:start + self._chunk].to(self.acc_dtype))

    def _bulk_update(self, x: torch.Tensor):
        n_b = x.numel()
        new_n = self.n + n_b
        if new_n == self.n:
            return
        x_var_t, x_mean_t = torch.var_mean(x, unbiased=False)
        xc = x - x_mean_t
        xc2 = xc * xc
        x_m3_t = torch.linalg.vecdot(xc, xc2)
        x_m4_t = torch.linalg.vecdot(xc2, xc2)
        x_mean, x_var, x_m3, x_m4 = torch.stack([x_mean_t, x_var_t, x_m3_t, x_m4_t]).tolist()
        x_m2 = x_var * n_b
        n_a, n_b_ = (self.n, n_b)
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
        new_M2 = self.M2 + x_m2 + delta_ab * n_a * n_b_ / n
        new_M3 = self._M3 + x_m3 + delta_ab * delta * (n_a * n_b_ * (n_a - n_b_) / (n * n)) + 3.0 * delta * (n_a * x_m2 - n_b_ * self.M2) / n
        new_M4 = self.M4 + x_m4 + delta_ab2 * (n_a * n_b_ * (n_a * n_a - n_a * n_b_ + n_b_ * n_b_) / n ** 3) + 6.0 * delta_ab * (n_a * n_a * x_m2 + n_b_ * n_b_ * self.M2) / (n * n) + 4.0 * delta * (n_a * x_m3 - n_b_ * self._M3) / n
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
        return self.M4 / self.n / (var * var) - 3.0

class ActivationStats:

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

    def attach(self, reset: bool=True):
        self.detach()
        if reset:
            self.max_abs = defaultdict(MaxAbsMeter)
            self.kurtosis = defaultdict(KurtosisMeter)
        for name, m in self.model.named_modules():
            cls_name = m.__class__.__name__
            if cls_name in {'LayerNorm', 'RMSNorm', 'DyT', 'DyTC', 'DyTD', 'DyTCD'}:
                self.handles.append(m.register_forward_hook(self._make_hook(name)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def summary(self) -> dict:
        out = {}
        for name in self.max_abs:
            out[name] = {'max_abs': self.max_abs[name].value, 'excess_kurtosis': self.kurtosis[name].result()}
        if out:
            all_max = [v['max_abs'] for v in out.values() if v['max_abs'] != -float('inf')]
            all_kurt = [v['excess_kurtosis'] for v in out.values() if math.isfinite(v['excess_kurtosis'])]
            peak = max(all_max) if all_max else 0.0
            out['__summary__'] = {'mean_max_abs': sum(all_max) / max(1, len(all_max)), 'max_max_abs': peak, 'max_abs_pre_norm': peak, 'mean_excess_kurtosis_pre_norm': sum(all_kurt) / len(all_kurt) if all_kurt else 0.0}
        return out

def markov_batch(rng: torch.Generator, ignore_index: int, batch_size: int, seq_len: int, device: str, n_states: int=64, outlier_id: Optional[int]=None, max_jump: int=3):
    start = torch.randint(0, n_states, (batch_size, 1), generator=rng, device=device)
    if seq_len > 1:
        jumps = torch.randint(0, max_jump, (batch_size, seq_len - 1), generator=rng, device=device)
        x = torch.empty((batch_size, seq_len), dtype=start.dtype, device=device)
        x[:, :1] = start
        torch.cumsum(jumps, dim=1, out=x[:, 1:])
        del jumps
        x[:, 1:].add_(start).remainder_(n_states)
    else:
        x = start
    y = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long, device=device)
    y[:, :-1] = x[:, 1:]
    if outlier_id is not None:
        spike = torch.rand((batch_size, seq_len), generator=rng, device=device) < 0.03
        x[spike] = outlier_id
    return (x, y)

class LossSpikeCounter:

    def __init__(self, ema_beta: float=0.99, rel_threshold: float=0.2, abs_threshold: float=0.05):
        self.ema_beta = ema_beta
        self.rel_threshold = rel_threshold
        self.abs_threshold = abs_threshold
        self.ema = None
        self.n_spikes = 0
        self.history: list = []

    def update(self, loss_value: float) -> bool:
        is_spike = False
        if self.ema is not None:
            rel_jump = (loss_value - self.ema) / max(self.ema, 1e-06)
            abs_jump = loss_value - self.ema
            is_spike = rel_jump > self.rel_threshold and abs_jump > self.abs_threshold
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
        return {'ema': self.ema, 'n_spikes': self.n_spikes}

    def load_state_dict(self, state: Optional[dict]) -> bool:
        if not isinstance(state, dict):
            return False
        ema = state.get('ema')
        if not (isinstance(ema, (int, float)) and math.isfinite(ema)):
            return False
        self.ema = float(ema)
        self.n_spikes = int(state.get('n_spikes') or 0)
        return True

@torch.no_grad()
def measure_activation_stats(model: nn.Module, batch: torch.Tensor) -> dict:
    stats = ActivationStats(model)
    stats.attach()
    out = model(batch)
    if isinstance(out, tuple):
        out = out[0]
    s = stats.summary()
    stats.detach()
    return s

def count_parameters(model: nn.Module, trainable_only: bool=False) -> int:
    return sum((p.numel() for p in model.parameters() if not trainable_only or p.requires_grad))
__all__ = ['ActivationStats', 'LossSpikeCounter', 'KurtosisMeter', 'MaxAbsMeter', 'measure_activation_stats', 'markov_batch', 'count_parameters']
