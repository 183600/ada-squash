"""Shared scratch-buffer caching plumbing.

The fused kernels, the attention statistics path and the meters reuse
pre-allocated buffers through small LRU caches. Two invariants recur at
every cache site:

* a tensor allocated under ``torch.inference_mode()`` cannot serve as an
  ``out=`` target (nor be saved for backward) once that mode is left, so a
  stale inference buffer is dropped and re-allocated even though its key
  and values are unchanged;
* eviction is least-recently-used over an ``OrderedDict`` with a fixed
  entry budget.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch


def _inference_stale(t: Optional[torch.Tensor]) -> bool:
    """Whether ``t`` is an inference-mode buffer used outside that mode.

    Tensors allocated under ``torch.inference_mode()`` are immutable
    constants for autograd: reused as an ``out=`` target (or saved for
    backward) in a later grad-mode call they would raise. Such buffers are
    scratch only, so they are dropped and rebuilt; the key and the values
    are unchanged.
    """
    return t is not None and t.is_inference() and not torch.is_inference_mode_enabled()


def _lru_scratch_buffer(cache: OrderedDict, max_entries: int, key,
                        shape, dtype: torch.dtype, device) -> torch.Tensor:
    """Get or allocate an LRU-evicted scratch buffer keyed by ``key``.

    The buffer holds intermediate results only; it is never saved for
    backward. Inference-mode staleness is handled by
    :func:`_inference_stale`.
    """
    buf = cache.get(key)
    if _inference_stale(buf):
        buf = None
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        if len(cache) >= max_entries:
            cache.popitem(last=False)
        cache[key] = buf
    else:
        cache.move_to_end(key)
    return buf
