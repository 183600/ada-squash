# 代码审查报告 - ada-squash (子Agent独立审查)
**日期:** 2026-11-20 09:04
**审查范围:** 全部 97 个 Python 源文件
**审查方法:** 静态分析 + 语法验证 + 算法实现审查 + 科研方法论审查

## 审查结果摘要

| 类别 | 发现数 | 已修复 |
|------|--------|--------|
| 语法错误 (IndentationError) | 4 | 4 |
| 静态分析标记（误报） | 33 | N/A |
| 科研方法问题 | 0 | N/A |

## 发现并修复的问题

### BUG-001: `ablation/runner.py` 第 133 行 — 空 `with torch.no_grad()` (曲线评估)

**严重级别:** 高（导致模块无法导入）  
**问题描述:** `with torch.no_grad():` 后紧跟 `with torch.inference_mode():`，前者为空块。  
**修复:** 移除空的 `with torch.no_grad():`。`torch.inference_mode()` 已包含 `no_grad` 的语义，是更强的推断模式。

### BUG-002: `ablation/runner.py` 第 193 行 — 空 `with torch.no_grad()` (验证集评估)

**严重级别:** 高（导致模块无法导入）  
**问题描述:** 与 BUG-001 相同的模式，在验证集 loss 计算处。  
**修复:** 移除空的 `with torch.no_grad():`，保留 `torch.inference_mode()`。

### BUG-003: `scripts/train_moe_tiny.py` 第 599 行 — 空 `with torch.no_grad()` (MoE 训练曲线)

**严重级别:** 高（导致脚本无法运行）  
**问题描述:** 同样的模式：`with torch.no_grad():` 空块后紧跟 `with torch.inference_mode():`。  
**修复:** 移除空行，`inference_mode()` 已涵盖全部需求。

### BUG-004: `tests/test_norms_training.py` 第 762 行 — 空 `with torch.no_grad()` (测试)

**严重级别:** 高（导致测试无法运行）  
**问题描述:** `with torch.no_grad():` 空块后紧跟 `with pytest.raises(...)`。  
**修复:** 移除空行。`pytest.raises` 不需要 no_grad 上下文。

### 问题根因分析

这4处语法错误具有相同的根因模式：
```python
# 错误模式
with torch.no_grad():     # ← 空的 with 块
with torch.inference_mode():  # ← 实际代码在这里
    ...
```
或
```python
# 错误模式
with torch.no_grad():     # ← 空的 with 块
with pytest.raises(...):  # ← 实际代码在这里
    ...
```

推测为之前的代码重构中，将 `torch.no_grad()` 升级为 `torch.inference_mode()` 或调整测试结构时，遗留了未清理的空 `with` 语句。

## 静态分析标记项复核

全部 33 个静态分析标记均为误报：

- **eval_call (11项):** 标准 PyTorch eval 模式
- **unprotected_division (12项):** 全部有 guard/clamp 保护
  - `assumptions.py:346` — `if n_arm == 0: continue` 在上游
  - `attention.py:67` — `dim % n_heads` 验证后才除法
  - `model.py:215,364` — `count.clamp(min=1)`
- **no_grad_context_missing (4项):** 训练循环中的正确梯度上下文
- **todo_fixme (1项):** 设计限制的注释

## 科研方法论审查

- `CausalSelfAttention` 正确验证 `dim % n_heads == 0`
- `_ChunkedCrossEntropy` 自定义 autograd.Function 实现正确：
  - 前向分块计算 `log_softmax + nll_loss`
  - 反向正确计算 `softmax - one_hot` 梯度
  - `ignore_index` 掩码在前向和反向均正确应用
- N-gram 去污染检查正确计算联合覆盖
- 消融分析使用 Holm 和 BH 校正

## 验证

修复后全量语法验证: 97 文件 / 0 错误 ✅
