# Code Review — ada-squash (2026-11-26 Agent)

## 审查范围
- DyT++ 归一化族 (`norms.py`) — 广义 softsign, 深度条件化, tanh 变体
- 注意力模块 (`attention.py`) — SDPA, QK-norm, RoPE, logit 统计
- 模型架构 (`model.py`) — 可插拔归一化, 深度调度, 分块交叉熵
- 统计方法 (`metrics_stats.py`, `metrics_meta.py`) — Cohen's dz, Shapiro-Wilk, Welford 方差
- 消融框架 (`ablation/`) — 矩阵, 对比, 分析

## 审查结果：未发现 bug 或科研方法问题

### DyT++ 归一化族
- 广义 softsign `x / (alpha + |x|^p)` 的互反-最大值公式数值稳定
- 稳定性下界 `_MIN_P = 1e-3`, `_MIN_ALPHA = 1e-4` 正确防止退化
- 手写 `torch.autograd.Function`：前向闭式表达式，反向重算中间量——所有变体验证正确
- 深度条件化 `alpha_l = alpha_0 * exp(-k * depth_ratio)` 带 `k_min` 下界防止负衰减
- `_reduce_products` / `_chunked_reduce_product`：bf16/fp16 累加正确提升到 fp32

### 统计方法
- Cohen's dz（配对效应量）：正确使用配对差异的标准差而非合并标准差
- Shapiro-Wilk：Royston 算法带 n 依赖系数表，n >= 3 时正确
- Welford 在线方差：数值稳定的单次遍历方差计算
- Cohen's d / Hedges' g：pooled_std < 1e-12 防护正确，小样本偏差校正 J = 1 - 3/(4*(n1+n2)-9)

### 实验设计
- 阶段 1-3 实验设计：基线校准 → 主消融 → 验证 — 正确的分阶段方法
- 去污染：基准污染审计带 n-gram 重叠检测
- 预注册：模式验证和记录保持确保可复现性

## 结论
DyT++ 归一化族数学正确，统计工具包（Shapiro-Wilk, D'Agostino-Pearson, Wilson CI, Welford 方差）实现严谨，消融框架的因子设计正确。未发现需要修复的 bug 或方法论问题。
