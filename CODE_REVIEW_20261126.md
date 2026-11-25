# Code Review Report - 2026-11-26

## 审查范围

核心模块：`adasquash/norms.py`（DyT++ 族、RMSNorm、QKSquash）

## 审查结果

### 总体评价：代码质量优秀，广义 softsign 内核实现精确

本次审查聚焦于：
1. 广义 softsign `φ_p(x) = x / (1 + |x|^p)^{1/p}` 的前向/反向正确性
2. 深度调度 `α_l = α_0 · exp(-k · l/L)` 的梯度链
3. 数值稳定性（`_MIN_P`、`_MIN_ALPHA` 地板值）
4. per-channel alpha 的反向传播
5. RMSNorm 自定义内核

### 详细发现

#### 1. 广义 Softsign 核心 — 正确 ✓

`_softsign_core` 的 D2 公式：
- `lo = min(|x|, 1)`，`r = 1 / max(|x|, 1)` — 地板在 1（而非 |x|）防止 1/0
- `u = lo · r` — softsign 核心，值域 [0, 1)
- `_compute_s_and_a` 的 log2/exp2 路径（可学 p）与 pow 路径（冻结 p）数值一致
- `_softsign_backward_pieces` 返回 `(r^p · s, φ, a)` — 梯度链正确

#### 2. `_DyTCForward` (per-channel softsign) — 正确 ✓

- 前向：`alpha_c = scale · (1 + delta)` floored at `_MIN_ALPHA`
- 反向正确处理 `alpha_c > _MIN_ALPHA` 的梯度掩码（clamp 的不可微点）
- `grad_alpha_c` 正确拆分到 `grad_delta`、`grad_scale`、`grad_alpha_0`、`grad_k`
- `_reduce_products` 的分块累加使用 `promote_types` 保证精度

#### 3. `_DyTDForward` (depth-conditioned tanh) — 正确 ✓

- `exp_kdr = exp(-k · depth_ratio)` 在前向计算并 detach 保存
- 反向 `grad_k = -grad_a · a · depth_ratio` — 链式法则正确
- `grad_alpha_0 = grad_a · exp_kdr` — 通过指数衰减链正确回传

#### 4. `_DyTCDForward` (主提案) — 正确 ✓

- per-channel alpha = depth-schedule scale × (1 + delta)，同时支持深度衰减和通道特化
- softsign 指数 p 通过 `_save_exponent` / `_restore_exponent` 正确管理
- `scale_nonneg=True` 优化跳过 sign fix（因为 depth-schedule scale 恒正）

#### 5. RMSNorm 内核 — 正确 ✓

- 前向：`rstd = 1 / sqrt(mean(x²) + eps)` → `out = x · rstd · weight`
- 反向 `d_x = rstd · (u · γ - x · mean(t · γ) · rstd²)` 与标准推导一致
- 精度提升 `promote_types` 确保低精度输入不丢失梯度精度

#### 6. `_softsign_p_grad` — 正确 ✓

- 指数梯度 `∂L/∂p = <∂L/∂φ, k> / p² · (p_raw > _MIN_P)`
- `k = σ(a) - a · σ(a)` 其中 `a = p · log(u)` — 正确的对数空间链
- `(p_raw > _MIN_P)` 门控防止冻结参数产生伪梯度

#### 7. 轻微建议（非 bug）

- `_compute_cohens_d` 和 `_apply_multiple_comparison_correction` 放在 `norms.py` 中，从模块化角度建议迁移到 `metrics_stats.py` 或独立的统计分析模块。

## 结论

DyT++ 归一化族的自定义 autograd Function 实现精确，数值稳定性处理周到，未发现需要修复的问题。
