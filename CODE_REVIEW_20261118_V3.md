# 代码审查报告 (2026-11-18 第三轮)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | ada-squash |
| 审查日期 | 2026-11-18 (第三轮独立验证) |
| 审查方法 | 子 Agent 独立深度审查：自定义 autograd 数学验证 + 闭式 CE + 数值稳定性 + 去污染审计 |
| 审查范围 | 全部 70 个 Python 源文件 |

## 总体评价

代码质量**优秀**。经第三轮独立验证，确认前两轮审查结论，未发现新的缺陷。

## 本轮审查重点

### 1. 自定义 autograd Function 数学验证

`adasquash/norms.py` 中的 `DyTCDTanh` 系列：

- **前向**：`exp_kdr = exp(k * (-depth_ratio))`，`a = alpha_0 * exp_kdr`，`out = tanh_squash(x, a, gamma, beta)`
- **反向**：对 `exp_kdr` 的 `.detach()` 正确阻止二阶梯度流经指数函数
- **梯度公式**：`grad_k = -grad_a * a * depth_ratio` — 由链式法则验证正确
- **数值稳定性**：`_MIN_ALPHA = 1e-4`，`_MIN_P = 1e-3` 设计为固定常量，防止数值退化

### 2. 闭式交叉熵反向传播

`adasquash/model.py` 的 `_CEFunction`：

- `torch.softmax(out=buf)` 原地计算 + `scatter_add_` 实现梯度
- 分块处理 + dtype 提升确保混合精度下的数值稳定
- `grad_scale = (grad_out / ctx.count.clamp(min=1)).to(dt)` 正确处理归一化

### 3. 注意力机制

`adasquash/attention.py`：

- 使用 `F.scaled_dot_product_attention(q, k, v, is_causal=True)` — PyTorch 官方实现
- QK-Squash 归一化在 softmax 前对 Q、K 做联合归一化
- RoPE 半精度处理使用缓冲缓存机制
- RMSNorm 使用 `eps=1e-06` 保护

### 4. 去污染审计

`adasquash/decontamination.py`：

- 完整的 n-gram 索引去污染管线
- Wilson 置信上界 + SHA-256 确定性协议哈希
- `contamination_scan_bound` 给出扫描覆盖率与污染率上界的统计推断

### 5. 科研方法学

- **训练完整性**：`scripts/train_gpt2.py` 和 `scripts/train_moe_tiny.py` 均有完整的 `zero_grad → backward → clip → step` 循环
- **非有限值检测**：MoE 训练包含 triage 机制，工程实践优秀
- **统计推断**：metrics 模块提供 bootstrap 置信区间和效应量估计

## 修复清单

无新增修复。

## 审查结论

ada-squash 的自定义归一化层、闭式交叉熵和 MoE 训练管线经过三轮独立审查验证，数学实现正确，数值稳定性设计完善。代码体现了高水平的科研工程实践。
