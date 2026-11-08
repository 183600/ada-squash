
# [代码审查] 添加随机种子设置以确保实验可复现
import random
def _set_reproducibility(seed=42):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

_set_reproducibility()