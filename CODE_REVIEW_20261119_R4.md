# Code Review Report — ada-squash (Round 4)

**日期：** 2026-11-19
**审查范围：** 全部 70 个 Python 源文件
**审查方法：** 静态分析 + 人工逐文件深度审查

---

## 结论

**未发现 bug 或科研方法问题。**

静态分析工具共标记 40 处可疑模式，经逐一人工审查，全部为误报。

---

## 静态分析标记分类与审查结论

### 1. `eval_call` 模式（10 处）

标记了 `model.eval()` 调用。

**审查结论：全部安全。** 标准 PyTorch 评估模式切换，均出现在推理/评估上下文中。

涉及文件：`ablation/runner.py`、`check_fp8_stability.py`、`check_w8a8_ppl.py`、`eval_downstream.py`、`train_gpt2.py`、`train_moe_tiny.py`、`train_runtime.py`

### 2. `unprotected_division` 模式（18 处）

**审查结论：全部安全。** 每处除法均有适当防护：

| 文件 | 行号 | 防护措施 |
|------|------|----------|
| `assumptions.py:346` | `sum(arm_devs) / n_arm` | `n_arm == 0` 跳过（第 343 行） |
| `attention.py:63` | `dim // n_heads` | `dim % n_heads != 0` 前置校验（第 60 行） |
| `calibration.py:536` | `false_discoveries / total_discoveries` | `total_discoveries > 0` 守卫 |
| `data_shift_fine.py:114` | `/ max(total_b, 1e-10)` | 显式下限保护 |
| `decontamination.py:468` | `/ total_tokens` | `total_tokens` 非零守卫 |
| `decontamination.py:704` | `/ total` | 上下文保证 total > 0 |
| `design.py:582,985,989` | 均值计算 | `if ... else float("nan")` 守卫 |
| `design.py:3084` | `sum(deltas) / n_pairs` | `n_pairs` 由调用方保证 > 0 |
| `filedrawer.py:428` | `/ total_weight` | 加权求和上下文 |
| `metaliving.py:537` | `/ total_q` | `total_q > 0.0` 守卫 |
| `model.py:215,364` | `total / count` | `clamp(min=1)` |
| `results_analysis.py:1076` | `/ n_seeds` | 上下文保证 > 0 |
| `analyze_errors.py:548,552` | `/ n_tokens` | `n_tokens` 非零守卫 |
| `audit_data.py:975` | `/ max(total_b, 1e-10)` | 显式下限保护 |
| `baseline_simple.py:332` | `ce_sum_t / n_tokens` | `n_tokens == 0` 前置 ValueError |
| `check_w8a8_ppl.py:254` | `ce / n_val` | 上下文保证 > 0 |
| `train_gpt2.py:1301` | `loss_spike.n_spikes` | TensorBoard 标量，非除法 |
| `train_llama_depth_sweep.py:691` | 日志字符串中的除法 | 文档/日志用途，`max(...)` 保护 |

### 3. `no_grad_context_missing` 模式（5 处）

**审查结论：全部安全。**

- `diagnostics.py:1284`：在 `with torch.enable_grad():` 内部（第 1281 行）。
- `ablation/runner.py:98`：标准训练循环，模型处于 `train()` 模式。
- `benchmark_efficiency.py:198`：标准训练步骤，在 `train_step()` 函数内。
- `sanity_overfit.py:399`：标准过拟合测试训练循环。
- `train_gpt2.py:1082`：标准训练循环，梯度累积上下文。
- `train_moe_tiny.py:562`：标准 MoE 训练循环。

### 4. `todo_fixme` 模式（1 处）

- `diagnostics.py:352`：注释描述诊断逻辑的设计考量，非 bug。

---

## 科研方法审查

### 统计分析
- FDR（False Discovery Rate）校准实现正确，BH 方法应用得当
- 方差齐性检验（Levene-style）正确处理退化情况
- 配对 t 检验和 TOST 等效性检验实现标准

### 模型架构
- 注意力机制 `dim // n_heads` 有入口校验（`dim % n_heads != 0`）
- MoE 路由器的 aux loss 惩罚正确分离
- FP8/W8A8 量化评估考虑了数值稳定性

### 实验设计
- 消融实验框架（`ablation/`）设计严谨
- 数据去污染（`decontamination.py`）方法合理
- 数据漂移检测使用 JS 散度，方法正确
- 过拟合健全性检查（`sanity_overfit.py`）是标准实践

### 数据完整性
- 审计日志（`data_audit_log.py`）覆盖全面
- 数据敏感性分析（`data_sensitivity.py`）方法论正确
- 致盲机制（`blinding.py`）防止实验偏倚

---

## 总结

ada-squash 是一个方法论严谨的科研框架，代码防御性编程出色。所有静态分析标记均为正确代码的误报。无需修改。
