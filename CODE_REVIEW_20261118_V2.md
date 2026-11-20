# 代码审查报告 (2026-11-18 第二轮)

## 审查概要

| 项目 | 内容 |
|------|------|
| 仓库 | ada-squash |
| 审查日期 | 2026-11-18 (第二轮独立审查) |
| 审查方法 | 多模式深度分析：AST 分析 + 数值稳定性 + 梯度正确性 + 自定义 autograd 验证 + 科研方法学 |
| 审查范围 | 全部 70 个 Python 源文件 |

## 总体评价

代码质量**优秀**。经独立审查确认此前审查结论，未发现需要修复的实际缺陷。

## 审查维度与结论

### 1. 自定义 autograd Function 正确性

`adasquash/norms.py` 中的 `DyTCDTanh` 前向/反向实现：

- **前向**：`exp_kdr = exp(k * (-depth_ratio))`，`a = alpha_0 * exp_kdr`，`out = tanh_squash(x, a, gamma, beta)`。
- **反向**：`ctx.save_for_backward(x, alpha_0, k, gamma, beta, exp_kdr.detach())`。对 `exp_kdr` 的 `.detach()` 是正确的——阻止二阶梯度流经指数函数，同时保留正确的数值用于 `grad_alpha_0` 和 `grad_k` 的计算。
- 梯度公式 `grad_k = -grad_a * a * depth_ratio`：由链式法则 `d/dk[alpha_0 * exp(-k * dr)] = -dr * alpha_0 * exp(-k * dr) = -dr * a`，与实现一致。

### 2. 闭式交叉熵反向传播

`adasquash/model.py` 的 `_CEFunction`：

- 前向使用 `torch.softmax(out=buf)` 原地计算概率，配合 `scatter_add_` 实现 `softmax - onehot` 梯度。
- 反向正确分块处理，dtype 提升（`acc` vs `dt`）在混合精度下保持数值稳定。
- `grad_scale = (grad_out / ctx.count.clamp(min=1)).to(dt)` 正确处理了归一化。

### 3. 注意力机制

`adasquash/attention.py`：

- 使用 `F.scaled_dot_product_attention(q, k, v, is_causal=True)`：PyTorch 官方实现，正确。
- QK-Squash 归一化（`QKSquash`）在 softmax 前对 Q、K 做联合归一化，实现清晰。
- RoPE 半精度处理（`_rope_half_scratch`）：缓冲缓存机制合理。

### 4. 去污染审计

`adasquash/decontamination.py`：

- 完整的 n-gram 索引去污染管线，包括 Wilson 置信上界、确定性 SHA-256 协议哈希。
- `contamination_scan_bound` 给出扫描覆盖率与污染率上界的统计推断——方法学正确。

### 5. 训练脚本

- `scripts/train_gpt2.py`：完整的 `zero_grad → backward → clip → step` 循环，梯度累积正确处理。
- `scripts/train_moe_tiny.py`：MoE 训练含非有限值检测和 triage 机制——工程实践优秀。

## 修复清单

无。本轮审查未发现需要修复的代码缺陷。

## 审查结论

ada-squash 在自定义归一化层（DyTCD / DyTD 系列）、闭式交叉熵和 MoE 训练方面展现了高水平的工程实现。自定义 autograd Function 的前向/反向一致性经过数学验证，闭式 CE 的分块处理和 dtype 管理正确可靠。
