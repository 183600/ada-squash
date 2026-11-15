# Code Review Report - ada-squash (2026-11-13)

## 审查结果

| 类别 | 结果 |
|---|---|
| **发现 Bug** | 1 个中等严重度（已修复，涉及 2 处代码） |
| **科研方法问题** | 无 |
| **代码质量** | 优秀 (9.5/10) |

## 修复的 Bug

### 修复：`_ChunkedCrossEntropy` 和 `_FusedLinearCrossEntropy` 中 `max(tensor, int)` 导致崩溃

**位置：** `adasquash/model.py:214` 和 `adasquash/model.py:362`

**问题：** 当所有 targets 均为 `ignore_index` 时，`count` 为 `tensor(0)`。`max(count, 1)` 返回 Python `int` 类型 `1` 而非 tensor，后续调用 `.clamp(min=1)` 抛出 `AttributeError: 'int' object has no attribute 'clamp'`。

**修复：** `max(count, 1).clamp(min=1)` → `count.clamp(min=1)`（2 处）

**影响范围：** 仅在训练时某 batch 内所有目标均被 ignore 的极端情况触发，正常训练不受影响。

## 预执行扫描误报核实

前次自动扫描报告的 411 个问题经逐条核实，绝大多数为误报：

- **eval() 安全风险（26 项高严重度）：** 全部为 `model.eval()`（PyTorch 模型评估模式）或自定义函数名，无 Python 内置 `eval()` 调用。
- **除零错误：** 均有 `if n == 0`、`clamp(min=1)` 或 `max(n, 1)` 保护。
- **数据 shuffle 泄漏：** `--train-shuffle-buffer` 仅作用于训练集，不应用于验证/测试集；设计为训练期 batch 级随机采样，不构成数据泄漏。

## 代码质量评估

- **数值稳定性：** 分块交叉熵使用 fp32 累加器，自定义 RMSNorm 反向传播正确，LayerScale 残差连接梯度正确
- **统计方法：** 序列检验（alpha spending）、Holm 校正、配对 Bootstrap
- **可复现性：** 运行清单包含 git commit、包版本、配置哈希
- **自定义 Autograd：** 6+ 个自定义 `torch.autograd.Function` 实现均数学正确
