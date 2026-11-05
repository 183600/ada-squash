# 实验设计与准备改进建议

本文档基于当前 ada-squash 可复现性契约与 `adasquash.design` 预飞行检查，提出从实验设计与准备角度的强化措施。所有建议均可在不触碰 `.github/` 与 `README*.md` 的前提下实施。

## 现有基线

- 可复现性契约：种子固定 `set_all_seeds` + `seed_worker`、run_manifest、TrainStreamFingerprint、execution_identity_mismatches、resume 状态恢复。
- 设计准备：`block_randomized_schedule` / `randomized_complete_block_schedule`、`schedule_balance_report`、`schedule_fingerprint_status`、`order_drift_report`。
- 预先排除：`default_exclusion_policy`、`run_exclusion_audit`，仅允许缺失运行、协议不匹配、非有限损失。
- 功效计划：`prospective_power_plan`、`tost_power_plan`、`factorial_power_plan`，含多重性 Bonferroni 与 attrition `(1-a)^2` 放大。
- 试点方差校准：`paired_sd_calibration` 返回样本 SD、上四分位 MAD、单侧置信上界，强制试点/确认种子不相交。
- 设计就绪审计：`design_readiness_audit` 检查复制、假设、最小效应、功效、完整随机化排程、身份对照。

## 改进方向

### 1. 可复现性证据链闭环
**现状**：`run_manifest.json` 记录 git、命令行、环境指纹、种子、数据版本，但训练流指纹仅在内部使用。
**改进**：
- 在 `design_readiness_audit` 中新增 `reproducibility_artifacts` 检查：要求每个 arm 的 manifest 存在 `training_stream_fingerprint` 且 `schema` 为 `adasquash.run_manifest.v1`。
- 强制 `TrainStreamFingerprint` 的 `prefix` 与 `full` 指纹在配对分析前通过 `fingerprint_status` 校验；不一致时在 `analysis_plan.json` 中标记为 `stream_mismatch` 并禁止进入主要对比。
- 将 `execution_identity_mismatches` 纳入预飞行：对比两个 arm 的 manifest 的 `git_commit`、`torch`/`cuda`/`cudnn`/`TF32`/`deterministic_algorithms`/`thread_count`，差异即为结构性失败而非警告。
- 在 `save_run_manifest` 时写入 `data_audit_log` 的哈希与 `decontamination` 版本，确保数据版本与去污染步骤被锁定。

### 2. 数据版本与加载确定性
**现状**：数据版本由指纹隐式体现，未显式校验 DataLoader 参数。
**改进**：
- 在 `repro.py` 的 `collect_run_metadata` 中强制记录：`num_workers`、`prefetch_factor`、`persistent_workers`、`shuffle`、`worker_init_fn`、`PYTHONHASHSEED` 起始值、`DataLoader` 的 `generator` 种子。
- 在设计准备阶段新增 `data_loader_determinism_check`：若两个 arm 的上述字段不一致，视为 `protocol_mismatch`，由排除策略自动剔除。
- 对 `--resume` 场景，`resume` 恢复时必须校验 `training_stream_fingerprint.start_step` 与上次断点一致，否则拒绝配对。

### 3. 排程与漂移的更强保障
**现状**：RCBD 与 order drift 检测已存在。
**改进**：
- 将 `schedule_fingerprint` 与 `analysis_plan` 的 `run_schedule.fingerprint` 做双向绑定校验；`design_readiness_audit` 中 `schedule` 检查失败时直接阻止训练启动。
- 对 `order_drift_report` 增加基于 Spearman 的分层检验：按 `block` 分层后计算残差相关，检测块内漂移而非仅整体漂移。
- 要求每个矩阵至少包含一个身份对照对 `dytcd_full` / `dytcd_ls1e5`，并在预飞行中检查对照对的种子完全重叠且指纹前缀一致。

### 4. 功效计划的稳健性
**现状**：功效计划使用 Bonferroni 与 attrition 放大。
**改进**：
- 提供 `attrition_sensitivity` 输出：对 `expected_attrition` 在 `{0.0, 0.05, 0.10, 0.20}` 下分别计算所需计划种子，写入 `design_readiness.json`，防止单一假设掩盖风险。
- 对因子交互项，要求同时提供 `superiority` 与 `equivalence` 功效；`design_readiness_audit` 中若 `declares_interaction_claims=True` 但未提供 `interaction_equivalence_margin`，则失败而非警告。
- 将试点 SD 上界的置信水平从默认 0.90 提升为可配置参数，并在 `paired_sd_calibration` 中输出 0.90/0.95 双上界供决策者选择。

### 5. 排除策略与审计可追溯
**现状**：排除策略预先指定并审计。
**改进**：
- 在 `run_exclusion_audit` 中增加 `differential_attrition_test`：若两 arm 的排除比例差异 ≥0.5，则在 `study_level_tests.json` 中自动降级为敏感性分析。
- 要求 `exclusion_policy.json` 的指纹写入 `analysis_plan.json` 并在 CI `ci_gate.py` 中校验；指纹变更需同步移动 `design-pin` 标签。
- 对 `diverged_curve` 标记的 run，强制写入 `curve_classification` 的判定规则与阈值到 manifest，避免事后阈值选择。

### 6. 预注册与不可变记录
**现状**：`build_analysis_plan` 生成指纹。
**改进**：
- 在 `analysis_plan.json` 中加入 `preregistration_timestamp` 与 `git_tag`，并在 `design_readiness_audit` 中验证标签存在且指向当前提交。
- 对于 `sequential_looks > 0` 的组序贯设计，冻结 `alpha_spending` 函数的种子与边界计算脚本哈希，防止中期分析时更换花费函数。
- 将 `compute_budget` 冻结：臂数 × 种子数 的预算写进计划，超出预算的新臂视为探索性。

### 7. 文档与门禁
- 在 `scripts/ablation.py` 的预飞行阶段增加 `--require-repro-artifacts` 开关，强制所有 manifest 存在且通过 `execution_identity_mismatches` 检查。
- 在 `ci_gate.py` 中增加设计层检查：`design_readiness_audit` 必须通过 `ready=True` 且 `reproducibility_artifacts` 通过，否则 CI 返回非零。
- 所有改进均不修改 `.github/` 与 `README*.md`，符合项目宪法。

## 示例检查清单增强

```python
# 设计准备增强示例
audit = design_readiness_audit(
    seeds=[...],
    rows=rows,
    hypotheses=hypotheses,
    schedule=schedule,
    margin=0.02,
    equivalence_margin=0.01,
    assumed_paired_sd=None,               # 从试点校准获取
    paired_sd_evidence=calibration,       # 强制试点种子不相交
    expected_attrition=0.05,
    n_confirmatory_tests=...,
    require_variance_evidence=True,
)
assert audit["ready"]
assert audit["checks"]["reproducibility_artifacts"]["status"] == "pass"
```

以上改进在保持现有 D1–D6 设计锁定的前提下，强化实验准备阶段的证据链完整性、漂移检测与功效稳健性，使配对对比在执行身份、数据流、排程与排除策略上均可审计、可复现。
