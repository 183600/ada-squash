from __future__ import annotations
import math
from collections import defaultdict, deque, OrderedDict
from typing import Optional, Dict
import torch
import torch.nn as nn

class MaxAbsMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self._peak = None
        self.count = 0

    @property
    def value(self) -> float:
        return -float('inf') if self._peak is None else float(self._peak)

    @staticmethod
    def _abs_max(x: torch.Tensor) -> torch.Tensor:
        """max(|x|)."""
        lo, hi = torch.aminmax(x)
        if not x.is_floating_point():
            # Negating the dtype minimum overflows in place (int8 -128 -> -128,
            # int64 min -> itself). Promote the extrema before taking the
            # absolute value; the meter's public value is a Python float.
            hi_f = hi.to(torch.float64)
            return torch.maximum(hi_f, -lo.to(torch.float64))
        lo.neg_()
        return torch.maximum(hi, lo, out=hi)

    def update(self, x: torch.Tensor):
        with torch.no_grad():
            xd = x.detach().reshape(-1)
            v = self._abs_max(xd) if xd.numel() else None
            if v is not None:
                if not torch.isfinite(v):
                    # A non-finite activation is not a finite peak: record it
                    # as NaN so the corruption is visible instead of being
                    # masked by a finite running maximum.
                    v = torch.full((), float('nan'), dtype=v.dtype, device=v.device)
                # A peak tensor created under torch.inference_mode() cannot be
                # updated in place once the meter is used outside it. Cloning
                # drops the inference flag while keeping the running maximum.
                if (self._peak is not None and self._peak.is_inference()
                        and not torch.is_inference_mode_enabled()):
                    self._peak = self._peak.clone()
                if self._peak is None:
                    self._peak = v.clone()
                elif self._peak.dtype != v.dtype or self._peak.device != v.device:
                    self._peak = self._peak.to(dtype=torch.promote_types(self._peak.dtype, v.dtype), device=v.device)
                torch.maximum(self._peak, v, out=self._peak)
        self.count += 1

# One workspace pair per (name, device, dtype), shared by every meter: the
# sampling hooks run one at a time, so a single pair serves all norms.
_KURTOSIS_WORKSPACES: OrderedDict = OrderedDict()
_KURTOSIS_WORKSPACES_MAX_ENTRIES = 8

def release_kurtosis_workspaces() -> None:
    """Release the shared KurtosisMeter scratch buffers (window / run end)."""
    _KURTOSIS_WORKSPACES.clear()

def _kurtosis_workspace_get_or_create(key: tuple, n: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Get or create a workspace buffer with LRU eviction."""
    buf = _KURTOSIS_WORKSPACES.get(key)
    # A workspace allocated under torch.inference_mode() becomes an inference
    # tensor; reusing it as an `out=` target outside that mode raises. It is
    # only scratch, so drop it and allocate a normal buffer instead.
    stale_inference = (buf is not None and buf.is_inference()
                       and not torch.is_inference_mode_enabled())
    if buf is None or buf.numel() < n or stale_inference:
        buf = torch.empty(n, dtype=dtype, device=device)
        if len(_KURTOSIS_WORKSPACES) >= _KURTOSIS_WORKSPACES_MAX_ENTRIES:
            # LRU eviction: remove oldest entry (first inserted)
            _KURTOSIS_WORKSPACES.popitem(last=False)
        _KURTOSIS_WORKSPACES[key] = buf
    else:
        # Move to end (most recently used)
        _KURTOSIS_WORKSPACES.move_to_end(key)
    return buf

class KurtosisMeter:
    # The per-chunk workspace is bounded and reused across samples.
    _chunk = 1 << 22
    # Accumulate the moments in float64. The fourth central moment of a small
    # activation (e.g. randn * 1e-12) underflows float32 to zero while the
    # variance is still representable, which made the scale-invariant
    # excess-kurtosis ratio collapse to -3. Float64 keeps both moments for
    # the full activation-magnitude range the diagnostics see.
    acc_dtype = torch.float64

    def __init__(self):
        self.reset()

    def _workspace(self, name: str, like: torch.Tensor) -> torch.Tensor:
        """A reusable buffer at least as large as `like` (a view is returned)."""
        n = like.numel()
        key = (name, like.device, like.dtype)
        buf = _kurtosis_workspace_get_or_create(key, n, like.dtype, like.device)
        return buf[:n].view(like.shape)

    def reset(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M4 = 0.0
        self._M3 = 0.0
        self._pending_stats: list = []
        self._pending_counts: list = []

    def update(self, x: torch.Tensor):
        xd = x.detach().reshape(-1)
        n = xd.numel()
        if n == 0:
            return
        parts = []
        counts = []
        for start in range(0, n, self._chunk):
            chunk = xd[start:start + self._chunk]
            if chunk.dtype != self.acc_dtype:
                chunk = chunk.to(self.acc_dtype)
            count = chunk.numel()
            # `torch.var_mean` accumulates the variance in fp32; the autocast
            # list casts `torch.linalg.vecdot` to the active low-precision
            # dtype.
            x_var_t, x_mean_t = torch.var_mean(chunk, unbiased=False)
            xc = torch.sub(chunk, x_mean_t, out=self._workspace('xc', chunk))
            xc2 = torch.mul(xc, xc, out=self._workspace('xc2', chunk))
            parts.append(torch.stack([x_mean_t, x_var_t, torch.linalg.vecdot(xc, xc2), torch.linalg.vecdot(xc2, xc2)]))
            counts.append(count)
        self._pending_stats.append(torch.stack(parts))
        self._pending_counts.extend(counts)

    def _flush(self):
        if not self._pending_stats:
            return
        pending = self._pending_stats[0] if len(self._pending_stats) == 1 else torch.cat(self._pending_stats, dim=0)
        for stats, count in zip(pending.tolist(), self._pending_counts):
            self._merge(*stats, count)
        self._pending_stats = []
        self._pending_counts = []

    def _bulk_update(self, x: torch.Tensor):
        x_var_t, x_mean_t = torch.var_mean(x, unbiased=False)
        xc = x - x_mean_t
        xc2 = xc * xc
        x_m3_t = torch.linalg.vecdot(xc, xc2)
        x_m4_t = torch.linalg.vecdot(xc2, xc2)
        x_mean, x_var, x_m3, x_m4 = torch.stack([x_mean_t, x_var_t, x_m3_t, x_m4_t]).tolist()
        self._merge(x_mean, x_var, x_m3, x_m4, x.numel())

    def _merge(self, x_mean: float, x_var: float, x_m3: float, x_m4: float, n_b: int):
        new_n = self.n + n_b
        if new_n == self.n:
            return
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
        self._flush()
        if self.n == 0:
            # No observation at all is the empty aggregate, not a sample whose
            # spread is zero.
            return 0.0
        if self.n < 2:
            # A single observation has no sample variance, hence no
            # standardized fourth moment. Reporting 0.0 here would label a
            # one-element activation as exactly Gaussian, the same masking the
            # constant/non-finite branch below avoids.
            return float('nan')
        var = self.M2 / self.n
        # Excess kurtosis is scale-invariant. An absolute variance floor (the
        # old `var < 1e-12`) reports heavy-tailed but small-magnitude
        # activations as perfectly Gaussian; only a zero/non-finite variance
        # has no standardized fourth moment. Returning 0.0 there labelled a
        # NaN/Inf activation (and a constant one) as exactly Gaussian, hiding
        # the corruption this meter exists to expose.
        if not math.isfinite(var) or var == 0.0:
            return float('nan')
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
        from .norms import LayerNorm, RMSNorm, _SquashedAffine
        for name, m in self.model.named_modules():
            if isinstance(m, (LayerNorm, RMSNorm, _SquashedAffine)):
                self.handles.append(m.register_forward_hook(self._make_hook(name)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        # The accumulators stay (pooling across windows).
        release_kurtosis_workspaces()

    def summary(self) -> dict:
        out = {}
        for name in self.max_abs:
            out[name] = {'max_abs': self.max_abs[name].value, 'excess_kurtosis': self.kurtosis[name].result()}
        if out:
            all_max = [v['max_abs'] for v in out.values() if v['max_abs'] != -float('inf')]
            all_kurt = [v['excess_kurtosis'] for v in out.values()]
            peak = (float('nan') if any(math.isnan(v) for v in all_max)
                    else (max(all_max) if all_max else 0.0))
            # A non-finite kurtosis means the activation had no standardized
            # fourth moment (NaN/Inf input, or a constant tensor). Averaging
            # only the finite entries would report a clean number and hide
            # exactly the corruption the meter exposes.
            mean_kurt = (float('nan')
                         if any(not math.isfinite(v) for v in all_kurt)
                         else (sum(all_kurt) / len(all_kurt) if all_kurt else 0.0))
            out['__summary__'] = {'mean_max_abs': sum(all_max) / max(1, len(all_max)), 'max_max_abs': peak, 'max_abs_pre_norm': peak, 'mean_excess_kurtosis_pre_norm': mean_kurt}
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
    elif seq_len == 1:
        x = start
    else:
        # seq_len == 0: return the empty (batch, 0) sequence, not the
        # (batch, 1) start token, so x and y stay aligned.
        x = torch.empty((batch_size, 0), dtype=start.dtype, device=device)
    y = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long, device=device)
    y[:, :-1] = x[:, 1:]
    if outlier_id is not None:
        spike = torch.rand((batch_size, seq_len), generator=rng, device=device) < 0.03
        x[spike] = outlier_id
    return (x, y)

def _is_boolean_scalar(value) -> bool:
    """Python and NumPy booleans.

    ``isinstance(np.bool_(True), bool)`` is False; without this check a NumPy
    boolean is silently coerced to 1.0 by ``float()`` and enters the loss EMA
    as a real observation.
    """
    if isinstance(value, bool):
        return True
    cls = type(value)
    return cls.__module__ == 'numpy' and cls.__name__ in ('bool_', 'bool')

class LossSpikeCounter:

    def __init__(self, ema_beta: float=0.99, rel_threshold: float=0.2, abs_threshold: float=0.05):
        self.ema_beta = ema_beta
        self.rel_threshold = rel_threshold
        self.abs_threshold = abs_threshold
        self.ema = None
        self.n_spikes = 0
        # Recent (loss, ema, is_spike) triples for post-mortem inspection,
        # bounded to the last window.
        self.history: deque = deque(maxlen=4096)

    def update(self, loss_value: float) -> bool:
        # A NaN/Inf loss is not a jump relative to the EMA: folding it into
        # the EMA makes every later comparison false, permanently disabling
        # spike detection. Keep it visible in the history but leave the
        # baseline untouched so the next finite loss is still judged against
        # the last finite one.
        if (_is_boolean_scalar(loss_value)
                or isinstance(loss_value, (str, bytes, bytearray))):
            # A numeric string is not a measured loss: `float("10.0")` would
            # enter the EMA and could report a spike from a malformed log
            # record. Keep it visible in the history without changing the
            # baseline, like a non-finite loss.
            self.history.append((loss_value, self.ema, False))
            return False
        try:
            numeric_loss = float(loss_value)
        except (TypeError, ValueError, OverflowError):
            self.history.append((loss_value, self.ema, False))
            return False
        if not math.isfinite(numeric_loss):
            self.history.append((numeric_loss, self.ema, False))
            return False
        loss_value = numeric_loss
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
        self.history = deque(maxlen=self.history.maxlen)

    def state_dict(self) -> dict:
        return {'ema': self.ema, 'n_spikes': self.n_spikes}

    def load_state_dict(self, state: Optional[dict]) -> bool:
        if not isinstance(state, dict):
            return False
        # Validate the whole state before mutating either field: a malformed
        # count used to raise after `ema` had already been replaced, leaving
        # the counter half-loaded. A negative count is not a possible state
        # and must not be accepted as one.
        raw_n = state.get('n_spikes', 0)
        if raw_n is None:
            n_spikes = 0
        elif isinstance(raw_n, bool):
            return False
        elif isinstance(raw_n, int):
            n_spikes = raw_n
        elif (isinstance(raw_n, float) and math.isfinite(raw_n)
                and raw_n.is_integer()):
            n_spikes = int(raw_n)
        else:
            return False
        if n_spikes < 0:
            return False
        ema = state.get('ema')
        if ema is None:
            if n_spikes != 0:
                return False
            self.ema = None
            self.n_spikes = 0
            return True
        if not (isinstance(ema, (int, float)) and not isinstance(ema, bool)):
            return False
        try:
            ema_value = float(ema)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(ema_value):
            return False
        self.ema = ema_value
        self.n_spikes = n_spikes
        return True

@torch.no_grad()
def measure_activation_stats(model: nn.Module, batch: torch.Tensor) -> dict:
    stats = ActivationStats(model)
    stats.attach()
    try:
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]
        return stats.summary()
    finally:
        # A failed forward must not leave hooks bound to the model: later
        # forwards would keep feeding stale meters (and duplicate the hooks
        # on the next call).
        stats.detach()

def count_parameters(model: nn.Module, trainable_only: bool=False) -> int:
    return sum((p.numel() for p in model.parameters() if not trainable_only or p.requires_grad))
__all__ = ['ActivationStats', 'LossSpikeCounter', 'KurtosisMeter', 'MaxAbsMeter', 'measure_activation_stats', 'markov_batch', 'count_parameters', 'release_kurtosis_workspaces']
