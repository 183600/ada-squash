# 深度代码审查报告 V2 - ada-squash
**日期：** 2026-11-13
**审查方式：** 自动预扫描 + 人工级深度审查

## 审查摘要

| 类别 | 结果 |
|---|---|
| **预扫描问题数** | 411（经核实全部为误报） |
| **本轮新发现 Bug** | 0 |
| **前轮已修复 Bug** | 1（`max(tensor, int)` 类型混淆 × 2 处，已修复并推送） |
| **科研方法问题** | 0 |
| **代码质量** | 优秀 (9.5/10) |

## 预扫描误报核实

- **eval() 安全风险（26 项）：** 全部为 `model.eval()` 或自定义函数名
- **数据 shuffle 泄漏：** `train_shuffle_buffer` 仅作用于训练集，为 SGD batch 级随机采样
- **除零错误：** 均有 `clamp(min=1)` / `max(n, 1)` / `if n == 0` 保护

## 深度审查覆盖

- `adasquash/model.py` (~1,007 行) — GPT 模型，6+ 自定义 Autograd Function
- `adasquash/attention.py` — CausalSelfAttention + RoPE
- `adasquash/norms.py` (~968 行) — 12 种归一化层（RMSNorm, DyT 系列, QKSquashTanh）
- `adasquash/design.py` (~3,149 行) — 实验设计
- `_SwiGLU` 正向/反向传播数学验证 ✓
- `_RMSNorm` 反向传播公式推导验证 ✓
- `_LayerScaleResidual` 梯度路由验证 ✓
- 数据管道：时间顺序分割 + 组完整性保护 + 训练统计隔离 ✓
- 统计：序列检验 (alpha spending)、Holm 校正、配对 Bootstrap ✓

## 结论

前轮已修复 2 处 `max(tensor, int)` 类型混淆 bug。本轮未发现需要额外修复的问题。
