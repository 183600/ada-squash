# 代码审查报告 - ada-squash
**审查日期**: 2026-11-15  
**审查范围**: 核心代码（adasquash/）和训练脚本（scripts/）  
**审查深度**: 人工逐行审查核心模块 + AST 静态分析  

## 审查结论

**代码质量评级：优秀**。未发现需要立即修复的关键代码 bug 或科研方法问题。

## 详细审查结果

### 1. 归一化层实现 (adasquash/norms.py)

**RMSNorm** ✅ 正确
- 前向：`rstd = 1/sqrt(mean(x²) + eps)` 正确实现
- 反向：手动推导验证正确
  - `grad_x = grad_out * w * rstd - x * rstd² * mean(grad_out * w * x * rstd)`
  - 代码实现 `u - x * corr * rstd²` 与公式一致
- 数值稳定：使用 `vector_norm` 避免平方和溢出

**DyT 家族** ✅ 正确
- `_tanh_squash_forward/backward` 手动推导验证正确
  - 前向：`out = gamma * tanh(alpha * x) + beta`
  - 反向：`grad_x = grad_out * gamma * alpha * (1 - tanh²(alpha*x))`
  - 代码实现 `w.mul_(alpha)` 正确得到 `grad_x`
- 广义 softsign 的反向重新计算中间值，正确实现 autograd
- 深度调度 `alpha_l = alpha_0 * exp(-k * depth_ratio)` 正确实现

**QK Squash** ✅ 正确
- 注意力 logit 的 tanh 压缩正确实现
- 反向正确处理 RoPE 与 QK squashing 的交互

### 2. 损失函数 (adasquash/model.py)

**分块交叉熵** ✅ 正确
- `_ChunkedCrossEntropy` 正确实现分块 log_softmax + NLL
- 反向正确计算 `(softmax - one_hot) * grad_scale`
- `ignore_index` 位置正确置零梯度
- 数值稳定：log_softmax 使用 PyTorch 内置实现

**融合线性交叉熵** ✅ 正确
- `_FusedLinearCrossEntropy` 正确融合线性层和交叉熵
- 减少内存占用（不需要存储完整 logits）

### 3. 注意力机制 (adasquash/attention.py)

**RoPE 实现** ✅ 正确
- `_ApplyRoPE` 正确实现旋转位置编码
- 反向正确计算旋转梯度

### 4. 数据处理 (scripts/train_gpt2.py)

**数据泄漏防护** ✅ 完善
- `shuffle_buffer` 仅应用于训练 split（`split == "train"`）
- 验证和测试 split 不进行 shuffle
- 支持 `exclude-hash-file` 等去污染机制

**Shuffle 警告澄清** ✅
- 训练数据的 shuffle 是在划分后进行的（训练批内 shuffle）
- 验证/测试数据明确禁用 shuffle buffer

### 5. 静态分析结果

- **eval() 调用**: 无 Python 内置 eval() 使用
- **数据 shuffle**: 均在划分后执行，无泄漏风险
- **除零风险**: `filedrawer.py` 中的除法均有保护

## 审查中发现的非关键观察

1. `_MIN_ALPHA = 1e-4` 和 `_MIN_P = 1e-3` 的设计边界合理
2. 模块级种子设置已移除（注释说明），改为实验入口点统一设置

## 结论

代码实现与论文方法一致，归一化层的 autograd 实现经过手动推导验证正确，无需修改。
