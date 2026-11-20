# ada-squash 开发者 / 贡献者指南

本仓库是 DyT++ / AdaSquash（无统计自适应归一化层）的科研代码库。所有改动须先过可复现性与工程规范两道门禁，再进入科研结论。

## 绝对不动的文件（不得删除、移动、修改）

- `.github/` —— CI 配置（`.loop-phase`、`workflows/`），由 CI 循环所有阶段使用。
- `README.md`、`README_zh.md`（以及任何 `README*.md`）—— 项目说明文档，所有语言版本均不可删不可移。
- `docs/design_notes.md` —— "设计宪法"，D1–D6 固定决策。CI 每次提交前运行 `ci_gate.py` 校验；违者整段 `git reset --hard` 撤销。人工变更必须同步移动 `design-pin` 标签。

## 快速开始

```bash
# 装依赖
pip install -r requirements.txt
pip install ruff mypy pytest  # 可选：工程检查

# 运行全量测试（CI 第 3 层）
python -m pytest tests/ -q

# 运行 CI 门禁（语法 + 设计固定项 D1-D6 + 全量 pytest）
python3 ci_gate.py

# 可复现性自测
python3 -m adasquash.repro --self-test
```

## 可复现性契约（每次实验必须满足）

1. **种子固定**：`set_all_seeds(seed)` 固定 Python/NumPy/torch/CUDA/cuDNN；DataLoader 工作进程通过 `seed_workher` 继承。
2. **运行清单**：每次训练结束必须写 `run_manifest.json`（`collect_run_metadata` + `save_run_manifest`），记录 git 提交、命令行、完整环境指纹、种子、数据版本、训练曲线与最终指标。
3. **训练流指纹**：`TrainStreamFingerprint` 对两臂是否"同种子同顺序读同批 token"做端到端摘要，配对对比前先比对指纹。
4. **执行身份**：`execution_identity_mismatches` 比对两臂的代码提交与软/硬件栈（torch/CUDA/cuDNN/线程数/TF32/决定性算法等）；差异或脏工作树即不可比。
5. **恢复训练**：`--resume` 恢复 RNG 状态与数据位置，续跑看到新鲜数据；跨断点比较 PPL/验证曲线时勿用 `--resume-skip-data`。

## 实验设计（`design.py`）

- 区块随机化排程（`block_randomized_schedule` / `randomized_complete_block_schedule`）：避免时间/热/硬件漂移与 arm 混淆；排程指纹嵌入预登记清单。
- 预先指定的排除策略（`default_exclusion_policy`）：仅允许缺失运行、协议/出处不匹配、非有限损失三种原因排除；按结果剔除（如"丢最差种子"）属园中岔路，禁止。
- 功效计划：`prospective_power_plan` / `tost_power_plan` 给出多少*计划*种子对才能在多重性校正与预期流失后仍检出预设最小效应；`simulate_paired_design` 用合成零/效应校验分析流水线自身。
- 顺序漂移：`order_drift_report` 检验执行位置是否在 arm 内预测结局。

## 工程规范

| 工具 | 配置 | 作用 |
|-----|------|------|
| ruff | `pyproject.toml` `[tool.ruff]` | lint + format（`ruff check --fix`、`ruff format`） |
| mypy | `pyproject.toml` `[tool.mypy]` | 类型检查（`mypy --config-file pyproject.toml adasquash/`） |
| pre-commit | `.pre-commit-config.yaml` | 提交前 ruff + mypy + YAML/空行/尾空格 |
| coverage | `pyproject.toml` `[tool.coverage.report] fail_under = 80` | 覆盖率门禁 |

提交前请先运行：
```bash
pre-commit run --all-files
# 或手动
ruff check --fix adasquash scripts tests ci_gate.py
ruff format adasquash scripts tests ci_gate.py
mypy --config-file pyproject.toml adasquash/
```

## 代码约定

- 所有公共函数需类型注解与文档字符串（Google 风格）。
- 融合 autograd 内核（`norms.py`、`model.py`）必须保持"正向闭式 + 反向闭式重算"结构，由 `tests/test_norms.py` 逐参数对拍独立 autograd 重算；不得削弱对拍测试。
- 注意力主路径固定为 `F.scaled_dot_product_attention`（D1）；`generalized_softsign` 的数值稳定语义锁定（D2）；`QKSquash` 固定 `tanh`（D4）；注册表与超参分发锁定（D5）；缓存键语义锁定（D6）。
- YAML 配置放 `configs/`，每行须注明参照行与唯一变化轴；消融矩阵预登记 `analysis_plan.json`（哈希指纹，改动即标记 post-hoc）。

## 提交检查单

- [ ] `python3 ci_gate.py` 通过（语法 + 设计锁定 + pytest）
- [ ] `pre-commit run --all-files` 通过
- [ ] 未触碰 `.github/` 与任意 `README*.md`
- [ ] 新增训练脚本写 `run_manifest.json`，并可用 `execution_identity_mismatches` 与基线对比
- [ ] 配对对比前已比对 `TrainStreamFingerprint`
- [ ] 排除策略预先指定、结果盲
- [ ] 设计锁定 D1–D6 未变更（如变更同步移动 `design-pin` 标签）
