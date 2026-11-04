# 对比实验与消融研究方法改进方案

本文件基于现有的 `adasquash/` 科研管线，从对比实验与单变量消融的角度提出可落地的改进。所有改进均不触碰 `.github/` CI 配置与 `README*.md` 文档。

## 1 现状总结

当前管线已具备以下强项：
- 预注册分析计划 `analysis_plan.json`，对比 ID、假设、检验方向与哈希指纹固定。
- 随机完全区组调度 `block_randomized_schedule`，`schedule_fingerprint_status` 防篡改。
- 预设排除策略 `default_exclusion_policy`，仅允许缺失运行/协议不匹配/非有限损失三类排除。
- 前瞻功效规划 `prospective_power_plan` / `tost_power_plan`，支持多重性校正与流失膨胀。
- 配对检验、TOST 等价性、留一种子 jackknife、交互项双侧检验、层次 Holm 门控。
- 文件抽屉 `filedrawer.py`、盲法 `blinding.py`、顺序监测 `sequential.py` 等模块已实现。

## 2 识别的差距

### 2.1 种子数与功效
`configs/ablation-matrix.yaml` 及各阶段配置均使用 3 个种子 [42,43,44]。在配对 t 检验下 df=2，对最小可检测效应、Holm 校正后的功效敏感。3 种子足以做描述性筛选，但对交互项与 TOST 等价性声称不足。

### 2.2 计算预算匹配的可验证性
不同归一化变体参数量 <0.1%，FLOPs 理论相等，但实际训练中因数值不稳定、loss-spike 计数、梯度裁剪行为差异会导致有效步数/收敛速度不同。现有管线未在对比报告中强制记录并对齐 `tokens_processed`、`analytic_flops_per_token`、`n_loss_spikes` 三项作为协变量。

### 2.3 盲法未集成到消融分析
`blinding.py` 可生成绑定分析计划指纹的盲码，但 `scripts/ablation.py` 的分析阶段直接使用原始臂名。分析者知晓臂名会引入确认偏误，尤其在因子交互的解释上。

### 2.4 消融语料的外推性
消融矩阵使用合成 Markov 游走语料，保证可学习的下界。虽避免均匀随机陷阱，但激活离群探针仅在合成语料上验证。缺乏与真实 FineWeb-Edu 子集的微型消融对照，难以证明因子在真实长程依赖上的外推性。

### 2.5 跨阶段对比的归一化
Phase 1 与 Phase 2 使用不同步数/序列长度，深度扫描使用 `dim` 随深度缩放。跨阶段汇总时未统一 `tokens_per_run` 与 `wall_time_per_token` 基准，导致“深度优势”可能混淆于训练量差异。

## 3 改进措施

### 3.1 提高种子数与功效前置检查
- 在 `scripts/ablation.py` 中默认 `seeds: [42,43,44,45,46]`，并在 `design_readiness_audit` 中强制 `n_seeds >=5` 用于主提案的 LOO 与 2×2 因子。
- 预注册时调用 `prospective_power_plan(paired_sd, n_pairs, margin, n_tests)`，将 `paired_sd` 来自试点校准 `paired_sd_calibration` 的上置信界而非点估计。不足则在 `design_readiness.json` 中硬失败。

实现要点：
```python
from adasquash.design import prospective_power_plan, paired_sd_calibration
cal = paired_sd_calibration(pilot_summaries, method="one_sided_upper")
plan = prospective_power_plan(
    paired_sd=cal["sd_upper_95"],
    n_pairs=len(seeds),
    margin=0.02,  # 预设最小科学效应
    n_tests=len(LOO_SPEC["dytcd_full"]),
    expected_attrition=0.05
)
```

### 3.2 计算预算与收敛协变量对齐
在 `adasquash/ablation/runner.py` 的 `run_one` 结束时记录：
- `tokens_processed` 
- `analytic_flops_per_token` * `tokens_processed`
- `n_loss_spikes`、`max_abs_pre_norm`

对比时先检验预算匹配：
```python
assert abs(tokens_a - tokens_b) / tokens_a < 1e-6
```
若不匹配，报告降级为描述性，并写入 `compute_mismatch_flag`。

### 3.3 盲法集成分析
分析流程拆分为两阶段：
1. `blind_arms` 生成 `BlindingKey`，臂名映射到 `A01...`，写入 `analysis_plan.json`。
2. 全部配对检验、因子交互、TOST 在盲码上执行并生成 `blind_report.json`。
3. 通过 `unblind_report` 解盲，解盲指纹必须与 `analysis_plan_fingerprint` 匹配。

在 `scripts/ablation.py` 中增加 `--blind` 选项，默认开启主提案对比。

### 3.4 真实语料微型消融
保留合成语料用于快速因子筛选，增加 `configs/ablation-realdemo.yaml`：
- 模型 `gpt2-124m`，`max_steps: 2000`，`batch_size: 16`，`seq_len: 256`
- 数据 `HuggingFaceFW/fineweb-edu` 的 1M token 子集，固定 `dataset_revision`
- 只运行主提案的 LOO 四个机制，3 种子验证方向一致性
- 结果写入 `all_summaries_real.json`，与合成结果做方向一致性检验

### 3.5 跨阶段归一化指标
统一报告三项归一化指标：
- `tokens_per_run = max_steps * batch_size * grad_accum * seq_len`
- `ppl_per_1b_tokens`
- `wall_time_per_1b_tokens`

Phase 2 深度扫描的 `sweep_comparisons.json` 增加 `compute_normalized_delta`，在 `log2(depth)` 回归前对 `tokens_per_run` 做残差化。

### 3.6 文件抽屉与敏感性
每次 `ablation.py` 运行后调用 `filedrawer.register_study`，记录：
- 研究 ID、假设、设计、分析计划指纹
- 状态 `completed/abandoned`
- 效应量、p 值
生成季度 `filedrawer_report.json`，包含 Egger 检验与 trim-and-fill 估计。

### 3.7 敏感性分析增强
对关键对比增加：
- 种子子集敏感性：随机抽取 80% 种子 1000 次，重算效应符号稳定性
- 排除策略敏感性：对有限但高 loss-spike 运行标记 `sensitivity_flag`，报告包含/排除两版结果
- p 值分布检验 `p_curve_analysis` 验证证据价值非 p-hacking

## 4 实施清单

- [ ] `scripts/ablation.py` 增加 `--seeds` 默认扩展与 `--require-power` 闸门
- [ ] `adasquash/ablation/runner.py` 记录并校验 compute budget
- [ ] `scripts/ablation.py` 增加盲法开关与解盲审计日志
- [ ] 新增 `configs/ablation-realdemo.yaml` 与对应的 `realdata_ablation` Runner
- [ ] 在 `study_level_tests.json` 增加 `compute_normalized` 字段
- [ ] 在 `scripts/audit_study.py` 中集成 file drawer 报告生成
- [ ] 更新测试 `tests/test_methodology.py` 增加盲法、种子数、compute 匹配用例

所有改动遵循现有设计锁定 D1-D6，不修改 `docs/design_notes.md` 的固定项。运行 `python3 ci_gate.py` 通过后方可合并。
