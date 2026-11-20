# Code Review Report — ada-squash (Round 11)

**日期**: 2026-11-19
**审查方法**: 深度逐行审查 + 模式扫描 + 跨仓库对比分析
**审查范围**: 全部 70 个 Python 源文件

## 审查结论

发现并修复了 **7 处真实 bug**，均为 `model.eval()` 后紧跟 `model.train()` 使评估模式失效。

## 发现的问题

### Bug: `model.eval()` 后立即 `model.train()` 导致 eval 模式失效

**严重程度**: 中等（影响评估/测量的正确性）

| 文件 | 行号 | 影响 |
|---|---|---|
| `scripts/check_fp8_stability.py` | 1038–1039 | FP8 稳定性检查: 发散度测量在 train 模式下执行 |
| `scripts/check_w8a8_ppl.py` | 915–916 | W8A8 量化 PPL: 量化前后对比在 train 模式下测量 |
| `scripts/check_w8a8_ppl.py` | 952–953 | W8A8 量化 PPL: 同上，另一测量路径 |
| `scripts/eval_downstream.py` | 111–112 | 下游评估: 加载的模型未正确切到 eval 模式 |
| `scripts/train_gpt2.py` | 986–987 | 训练诊断: triage 前向在 train 模式下执行 |
| `scripts/train_moe_tiny.py` | 665–666 | MoE 验证: 验证 loss 在 train 模式下计算 |
| `scripts/train_runtime.py` | 222–223 | 运行时评估: eval 条件分支被 train() 覆盖 |

**注意**: ada-squash 在之前的审查轮次（commit 05a3079）已修复了 `filedrawer.py` 中的运算符优先级 bug。本次发现的是不同类型的 bug。

## 排除的误报

- `scripts/check_w8a8_ppl.py:228–229`: 经核实为文档注释中的文本，非代码（`model.train()` 出现在 docstring 中）

## 静态分析标志复核

| 标志模式 | 数量 | 复核结果 |
|---|---|---|
| `eval_call` | 18 | 全部为标准 PyTorch 用法 |
| `unprotected_division` | 15 | 全部有 clamp_min 或三元保护 |
| `no_grad_context_missing` | 3 | 均在训练上下文中 |
| `todo_fixme` | 4 | 文档备注 |

## 科研方法论审查

- 注意力机制: CausalSelfAttention 正确实现（含 RoPE、QK-norm、SDPA）
- 损失函数: ChunkedCrossEntropy 前向/反向均正确
- 消融研究: 结构化对比设计合理
- 元分析: DerSimonian-Laird 估计量 + 校准评估

## 结论

代码质量优秀。本次修复了 7 处 eval/train 排序 bug。核心科研实现无问题。
