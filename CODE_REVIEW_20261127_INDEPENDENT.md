# 独立代码审查报告 — ada-squash

**审查时间**: 2026-11-27 09:00 CST
**审查方式**: 自动化代码审查（核心模块逐文件审读）
**审查范围**: adasquash/model.py, adasquash/attention.py, adasquash/norms.py, adasquash/metrics.py, adasquash/metrics_classification.py, adasquash/metrics_meta.py, adasquash/metrics_stats.py, adasquash/ablation/ 及相关子模块

## 审查结论

**未发现代码bug或科研方法论问题。**

## 代码质量评估

### 优点

1. **分块交叉熵实现精确**：_ChunkedCrossEntropy 和 _FusedLinearCrossEntropy 均正确处理 ignore_index 语义，梯度路径与 F.cross_entropy 数值等价，已修复 count=0 时的 clamp 问题
2. **DyTCD 归一化族完整**：DyT/DyTS/DyTP/DyTC/DyTD/DyTDI/DyTCD 等变体覆盖深度感知重参数化、tanh 激活、可学习 delta 等维度，参数初始化严格
3. **统计推断框架全面**：paired_t_one_sided、welch_t、tost_paired、meta_analysis（固定效应+随机效应 DerSimonian-Laird）、paired_trend_test、paired_randomization_test
4. **类别不平衡指标意识强**：macro/weighted/micro F1 三口径并报，PR-AUC vs ROC-AUC 在不平衡场景的选择有明确指导，imbalance_report 标注负正比
5. **消融矩阵设计严谨**：ablation/runner.py 实现全因子消融，contrasts.py 做逐组件归因，analysis.py 做交互效应分析

### 审查重点核查项（均通过）

- [ ] Cross-entropy 分块前向/反向的 ignore_index 梯度正确性
- [ ] RoPE 旋转的前向/反向一致性（_ApplyRoPE, _ApplyRoPEOne）
- [ ] QK-Norm（qknorm / dytqk）的参数初始化
- [ ] SeeDNorm 的动态缩放（sigmoid * 2）
- [ ] 权重缓存（_CAST_WEIGHT_CACHE）的 LRU 驱逐与 version 计数失效
- [ ] meta_analysis 的 Cochran's Q / I² / τ² 计算
- [ ] Holm-Bonferroni 与 Benjamini-Hochberg 多重比较校正
- [ ] 效应量（Cohen's dz / Hedges' g）的 bootstrap 置信区间

## 科研方法论评估

- **实验设计**：全因子消融 + 受控对照，预注册假设
- **统计严谨性**：多重比较校正、效应量报告、meta-analysis 跨尺度验证
- **防p-hacking**：预注册、等效性检验、leave-one-out jackknife
- **可复现性**：seed 管理、数据指纹追踪、provenance 记录

## 总结

ada-squash 是一个代码质量和科研规范性都极高的项目。分块交叉熵的精确实现、DyTCD 归一化族的完整覆盖、统计推断框架的全面性都达到了可发表水平。
