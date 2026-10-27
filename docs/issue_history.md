# 历史问题归档

本文件归档已经完成的临时清单条目。活跃条目应保留在 `.opencode_issues.md`；
截至 2026-10-27，仓库中没有活跃的 `.opencode_issues.md`（临时清单已在提交
`90caf97` 的清理中移除），因此这里不再复制逐条全文。

## 2026-10-27

- `GPT.forward` 在 `last_position_only=True` 且 `targets` 为 1-D 时按二维索引，
  会抛 `IndexError`；现已支持单样本/逐样本标签。
- `cross_entropy_mean` / `fused_cross_entropy_mean` 现在校验 targets 数量与
  logits 行数一致，并正确支持一维 targets。
- 各脚本解析记录/checkpoint 中的数值时统一使用 finite helper；超出 float 范围
  的整数按不可用处理，而不是抛 `OverflowError`。
- `wilson_upper_bound` / `contamination_scan_bound` 的超大 `confidence` /
  `max_rate` 现在统一抛 `ValueError`。
- 仓库卫生：`.coverage` 不再被版本库追踪，覆盖率产物加入 `.gitignore`。
- 大型脚本拆分：`scripts/ablation.py` 拆入 `adasquash/ablation/` 包；
  `adasquash/metrics.py`、`scripts/train_gpt2.py`、`scripts/audit_data.py`
  按职责拆分，入口与对外接口保持不变。

## 更早

- 2026-10 之前关于 metrics、FP8、W8A8、数据切分几何、RoPE 缓存和实验结果
  解析的修复记录，可通过 `git log -- .opencode_issues.md` 及各提交信息追溯。
