"""
AdaSquash / DyT++: Adaptive Squashing without Statistics for
Ultra-Deep and Low-Precision Transformers.

Submodules:
    adasquash.norms       - Normalization layers (LN, RMSNorm, DyT, DyT-C, DyT-D, DyT-CD)
    adasquash.attention   - Attention with optional QK-squashing (DyT-QK variant C)
    adasquash.model       - GPT-style transformer with configurable norm & depth schedule
    adasquash.utils       - Stability metrics (max|x|, kurtosis, loss-spike counter, etc.)
"""

from .norms import (
    LayerNorm,
    RMSNorm,
    DyT,
    DyTC,
    DyTD,
    DyTCD,
    build_norm,
    NORM_REGISTRY,
)
from .attention import CausalSelfAttention
from .model import GPTConfig, GPT, LayerConfig
from .utils import (
    ActivationStats,
    LossSpikeCounter,
    KurtosisMeter,
    MaxAbsMeter,
    measure_activation_stats,
    count_parameters,
)

__version__ = "0.1.0"
__all__ = [
    "LayerNorm",
    "RMSNorm",
    "DyT",
    "DyTC",
    "DyTD",
    "DyTCD",
    "build_norm",
    "NORM_REGISTRY",
    "CausalSelfAttention",
    "GPTConfig",
    "GPT",
    "LayerConfig",
    "ActivationStats",
    "LossSpikeCounter",
    "KurtosisMeter",
    "MaxAbsMeter",
    "measure_activation_stats",
    "count_parameters",
]
