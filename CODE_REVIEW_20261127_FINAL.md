# Code Review — ada-squash (2026-11-27)

## 审查范围

核心源代码文件：
- `adasquash/model.py` — 模型配置与分块交叉熵
- `adasquash/attention.py` — 因果自注意力（QK-norm / DyTQK）
- `adasquash/norms.py` — 归一化层族（DyT 家族）
- `adasquash/metrics.py` — 不确定性感知的评测指标
- `adasquash/metrics_classification.py` — 分类指标（F1、AUC、NDCG）
- `adasquash/metrics_meta.py` — Meta-analysis、随机化检验、趋势检验
- `adasquash/metrics_stats.py` — 统计量计算
- `adasquash/ablation/` — 消融实验框架

## 审查结论：代码质量优秀，未发现实质性 bug 或科研方法问题

### 1. 核心算法正确性

**分块交叉熵** — `_ChunkedCrossEntropy` 正确实现了分块 log_softmax + nll_loss，`ignore_index` 掩码处理完备。`_valid_target_mask` 正确处理了 uint8 目标与负 ignore_index 的 unsigned 提升问题（将 uint8 先转 int64 再比较）。

**注意力实现** — `CausalSelfAttention` 支持 QK-norm（RMSNormHead）和 DyTQK 两种 QK 归一化方案，推理态使用缓存的 tril 掩码和 RoPE 表。

**权重缓存** — `_cast_weight_cached` 使用 (data_ptr, _version, dtype, device) 作为缓存键，正确处理了权重被替换或原地更新的情况。LRU 驱逐机制防止缓存无限增长。

### 2. 统计方法

- `mean_std_stderr` 报告均值标准误，使 delta 小于噪声时可见
- `paired_t_one_sided` 使用 Student-t 参考分布（n=3 时临界值 2.92），不用正态近似（z=1.64）
- `wilson_interval` 用于二项比例置信区间（保持在 [0,1] 内）
- `meta_analysis` 使用逆方差加权（固定效应）+ DerSimonian-Laird（随机效应），报告 Cochran's Q、I²、τ² 和预测区间
- `paired_trend_test` 对逐种子的配对差异斜率做单侧检验，避免伪重复
- `benjamini_hochberg_adjust` 提供 FDR 控制的 q 值

### 3. 消融框架

- 消融矩阵由 `ablation/matrix.py` 定义，`ablation/runner.py` 执行，`ablation/contrasts.py` 计算对比
- 主效应和交互效应分离明确，全因子设计与逐组件 LOO 设计可互相印证
- 脆弱性分析（`_fragility_summary`）计算了 fragility index（删除多少种子后失去显著性）和 sign-flip p 值

### 4. 代码质量

- DyT 家族归一化层（dyt/dyts/dytp/dytc/dytd/dytcd 等）参数化清晰，深度重参数化（`_DEPTH_REPARAM`）与软符号 p（`_SOFTSIGN_P`）分类明确
- `_CROSS_ENTROPY_CHUNK_ROWS_CACHE` 和 `_FUSED_LINEAR_CHUNK_ROWS_CACHE` 按词表大小自适应分块
- 模块级 seed 设置被正确移除，改用集中式 seed 入口

### 5. 注意事项（非 bug）

- `_CE_SCRATCH_MAX_ENTRIES = 2` 限制了交叉熵 scratch 缓冲数量，在大词表场景下可能需要调整
- `_CAST_WEIGHT_CACHE_MAX_BYTES = 1 << 28`（256MB）的缓存上限在超大模型下可能成为瓶颈

## 无需修改

代码无需修改，审查通过。
