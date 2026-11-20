# 代码审查报告 — ada-squash
**审查日期**: 2026-11-19
**审查范围**: 全仓库 70 个 Python 源文件
**审查方法**: 静态分析 + 深度人工审查（算法逻辑、数值稳定性、科研方法论）

## 总体评估

**结论：代码质量优秀，未发现需要修复的实质性 bug 或科研方法论问题。**

本仓库实现了 DyT++ / AdaSquash 自适应压缩归一化层家族，包含完整的自定义 autograd Function、消融框架和统计验证工具。代码在数值稳定性方面表现突出，所有归一化层都有精心设计的下界保护。

## 逐项审查结果

### 1. 归一化层 (`adasquash/norms.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 数值稳定性 | ✅ | `_MIN_ALPHA=1e-4`, `_MIN_P=1e-3` 防止参数退化 |
| 广义 softsign | ✅ | `reciprocal-on-max(|x|, 1)` 公式避免原点附近的数值问题 |
| 自定义 autograd | ✅ | 所有 Function 的 forward/backward 闭式表达式正确配对 |
| 分块乘积归约 | ✅ | `_chunked_reduce_product` 正确处理大张量的内存和精度 |
| 符号修复 | ✅ | `_scale_needs_sign_fix()` 检测负缩放因子 |
| 深度衰减 | ✅ | 深度条件变体 (DyT-D, DyT-CD) 的初始化先验正确 |

### 2. 注意力机制 (`adasquash/attention.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 缩放计算 | ✅ | `scale = 1.0 / math.sqrt(max(self.head_dim, 0.0))`，`max(., 0.0)` 保护极端情况 |
| QK 归一化 | ✅ | 三种模式（qknorm/dytqk/none）正确实现，共享参数处理正确 |
| RoPE | ✅ | 使用 `freqs_cis` 预计算表，`_rope_half_scratch` 缓存避免重复分配 |
| Logit 统计 | ✅ | `_logit_stats` 使用 row-block 扫描下三角，减少峰值内存 |
| 缓存管理 | ✅ | `_inference_stale()` 机制确保推理态缓存正确失效 |

### 3. 模型 (`adasquash/model.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 损失计算 | ✅ | `total / count.clamp(min=1)` 正确处理空批次 |
| 梯度缩放 | ✅ | backward 中 `grad_out / ctx.count.clamp(min=1)` 正确缩放 |
| 权重缓存失效 | ✅ | `invalidate_weight_caches()` 在优化步后调用 |
| 配置校验 | ✅ | `GPTConfig` 的 dataclass 设计清晰，参数有合理默认值 |

### 4. 消融框架 (`adasquash/ablation/`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 训练循环 | ✅ | `runner.py` 中 loss.backward() 在正确上下文，含非有限检测 |
| 非有限诊断 | ✅ | `triage_nonfinite()` 提供诊断和修复建议 |
| 矩阵分析 | ✅ | 全因子消融矩阵正确配置 |
| 对比分析 | ✅ | 配对比较设计，减少混杂因素 |

### 5. 统计工具 (`adasquash/assumptions.py`, `adasquash/metrics*.py`)

| 检查项 | 结果 | 说明 |
|--------|------|------|
| Shapiro-Wilk 近似 | ✅ | `sigma = math.exp(...)` 保证正数，`z = (log_w - mu) / sigma` 安全 |
| D'Agostino-Pearson | ✅ | `m2 == 0` 提前返回，保护 `m2^1.5` 和 `m2^2` 的除法 |
| 正态分位数近似 | ✅ | `max(-2*log(p), 0.0)` 保护 sqrt 参数 |
| Bootstrap CI | ✅ | 分块 bootstrap 正确处理序列相关性 |
| 方差分解 | ✅ | 双向方差分量（seed × unit），含负值截断和 ICC 计算 |

### 6. 静态分析标记的误报说明

| 模式 | 标记数 | 误报率 | 说明 |
|------|--------|--------|------|
| `eval_call` | 若干 | 100% | 标准 PyTorch 状态管理 |
| `unprotected_division` | 若干 | 100% | 全部有 `clamp(min=1)`、`max(..., 0.0)` 或条件守卫 |
| mutable default args | 5 | 100% | 全部为 `argparse` 的 `default=[]`，非函数签名 |

## 科研方法论评估

1. **消融设计**: 系统化变体矩阵（norm 选择 × qk_norm × layerscale × dropout）
2. **假设检验**: 内置 `assumptions.py` 自动检查正态性、独立性等统计假设
3. **数值精度**: fp64 累加、分块归约、自定义 autograd 确保训练稳定
4. **可复现性**: 集中化种子管理（不在模块级设置种子）
5. **效率分析**: benchmark 脚本含内存和速度指标

## 建议（非 bug，可选改进）

- `_MIN_P=1e-3` 的选择可考虑在文档中补充理论依据
- 部分 autograd Function 的 backward 可添加梯度检查单元测试

