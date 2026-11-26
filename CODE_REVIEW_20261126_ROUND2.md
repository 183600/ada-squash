# Code Review — ada-squash (2026-11-26 第二轮独立审查)

## 审查范围
- `adasquash/norms.py` — DyT++ 归一化族（广义 softsign、深度条件化、tanh 变体）
- `adasquash/model.py` — GPT 模型、分块交叉熵、融合线性交叉熵、SwiGLU MLP
- `adasquash/attention.py` — SDPA、QK-norm、RoPE

## 审查方法
独立逐文件审查，重点检查：
1. 自定义 autograd.Function 的前向/反向正确性
2. 数值稳定性和精度处理
3. 分块交叉熵和融合线性交叉熵的正确性
4. 深度条件化归一化的实现

## 审查结果：未发现 bug 或科研方法问题

### DyT++ 归一化族 (`norms.py`)
- 广义 softsign `x / (alpha + |x|^p)` 的互反-最大值公式数值稳定
- `_MIN_P = 1e-3`, `_MIN_ALPHA = 1e-4` 正确防止退化到恒等映射
- 手写 `torch.autograd.Function`：前向闭式表达式，反向重算中间量——所有变体验证正确
- `_reduce_products` / `_chunked_reduce_product`：bf16/fp16 累加正确提升到 fp32
- 深度条件化 `alpha_l = alpha_0 * exp(-k * depth_ratio)` 带 `k_min` 下界防止负衰减
- `_effective_decay_rate` 正确 clamp k 到 k_min

### 模型架构 (`model.py`)
- `_ChunkedCrossEntropy` 分块 log_softmax + nll_loss 正确实现
- `_FusedLinearCrossEntropy` 融合 GEMM + softmax + NLL，正确尊重 ignore_index 掩码
- `_valid_target_mask` 正确处理 uint8 目标（避免负 ignore_index 在 unsigned dtype 上的静默错误）
- `_SwiGLU` 自定义 autograd.Function 的前向/反向实现正确，正确处理 autocast dtype 缓存
- `_LayerScaleResidual` 的梯度路由正确：`grad_residual = grad_out * scale`
- `build_norm_kwargs` 正确将配置分发到各归一化构造函数
- 共享深度调度 (`shared_depth_schedule`) 正确实现跨层共享 alpha_0/k 参数

### 注意力模块 (`attention.py`)
- SDPA 路径正确处理因果掩码和注意力 dropout
- QK-norm 可选启用，alpha 参数正确初始化

## 结论
DyT++ 归一化族数学正确，分块/融合交叉熵实现严谨，SwiGLU MLP 的自定义 autograd 无问题。
