import torch as _torch
try:
    _torch.backends.cuda.matmul.allow_tf32 = True
    _torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass
try:
    _torch.set_float32_matmul_precision('high')
except Exception:
    pass
from .norms import LayerNorm, RMSNorm, DyT, DyTC, DyTD, DyTCD, build_norm, generalized_softsign, NORM_REGISTRY
from .attention import CausalSelfAttention
from .model import GPTConfig, GPT, LayerConfig, build_gpt_124m, build_gpt_350m, build_llama_tiny_1b
from .utils import ActivationStats, LossSpikeCounter, KurtosisMeter, MaxAbsMeter, measure_activation_stats, markov_batch, count_parameters
__version__ = '0.1.0'
__all__ = ['LayerNorm', 'RMSNorm', 'DyT', 'DyTC', 'DyTD', 'DyTCD', 'build_norm', 'generalized_softsign', 'NORM_REGISTRY', 'CausalSelfAttention', 'GPTConfig', 'GPT', 'LayerConfig', 'build_gpt_124m', 'build_gpt_350m', 'build_llama_tiny_1b', 'ActivationStats', 'LossSpikeCounter', 'KurtosisMeter', 'MaxAbsMeter', 'measure_activation_stats', 'markov_batch', 'count_parameters']
