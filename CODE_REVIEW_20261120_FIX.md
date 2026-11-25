# Code Review Report — ada-squash (2026-11-20)

## 审查范围
70 个 Python 源文件，涵盖核心库、训练脚本、消融实验与测试。

## 发现的问题

### 1. [CRITICAL] 语法错误：空 `with torch.no_grad():` 块（3处）

#### 1a. `adasquash/ablation/runner.py:133`
- **问题**: `with torch.no_grad():` 后紧跟 `with torch.inference_mode():`，空块导致语法错误。
- **修复**: 删除空 `no_grad` 行，保留 `inference_mode()`。

#### 1b. `adasquash/ablation/runner.py:193`
- **问题**: 同上，训练循环中的验证评估代码无法执行。
- **修复**: 删除空 `no_grad` 行，保留 `inference_mode()`。

#### 1c. `scripts/train_moe_tiny.py:599`
- **问题**: MoE 训练脚本中的验证曲线计算处有同样的空块。
- **修复**: 删除空 `no_grad` 行，保留 `inference_mode()`。

#### 1d. `tests/test_norms_training.py:762`
- **问题**: `with torch.no_grad():` 后紧跟 `with pytest.raises(...)`，测试文件无法加载。
- **修复**: 删除空 `no_grad` 行。测试中的 `needle_in_haystack` 调用不需要显式 no_grad 包裹（pytest.raises 上下文已足够）。

### 静态分析误报（经验证为正确代码）
| 模式 | 数量 | 验证结论 |
|------|------|----------|
| `eval_call` | 12 | 正确的推理模式切换 |
| `unprotected_division` | 14 | 均有 `clamp(min=1)`、条件检查等保护 |
| `no_grad_context_missing` | 6 | 均在训练/诊断上下文中，backward 是设计意图 |
| `todo_fixme` | 1 | 注释提及已知限制，非功能缺陷 |

### 科研方法审查
- **消融实验**: runner.py 实现了完整的消融矩阵运行，包含学习曲线诊断（过拟合/欠拟合分类），方法论正确。
- **梯度累积**: `train_gpt2.py` 正确处理了 `loss / grad_accum` 缩放，累积后统一 backward。
- **非有限值处理**: 完整的 NaN/Inf 检测和三级诊断（数据→前向→梯度），包含自动 triage。
- **数据去污染**: 实现了 n-gram 重叠检测和 TVD 分布偏移测试。

## 修复摘要
| 文件 | 修复类型 | 影响 |
|------|----------|------|
| `adasquash/ablation/runner.py` | 2处语法错误修复 | 恢复消融实验功能 |
| `scripts/train_moe_tiny.py` | 语法错误修复 | 恢复 MoE 训练验证曲线 |
| `tests/test_norms_training.py` | 语法错误修复 | 恢复测试可执行性 |
