# 独立代码审查报告 (2026-11-27)

## 审查范围
- `adasquash/model.py` - GPT 模型，DyT/DyTCD 归一化族
- `adasquash/attention.py` - 因果自注意力，QK-norm，RoPE
- `adasquash/metrics.py` - 类别不平衡感知指标，bootstrap CI

## 审查结论

**未发现需要修复的 bug 或科研方法问题。**

### 代码质量亮点

1. **归一化族设计完整**：DyT → DyTC → DyTCD 逐步增加参数化，每个变体有明确的消融对照
2. **内存管理精细**：CE scratch 缓存 LRU 策略，权重 cast 缓存带版本签名和字节上限
3. **类平衡指标严谨**：macro NLL 对类等权平均，与频率加权互补；AUC 只纳入 support≥20 的类
4. **uint8 靶标处理正确**：`_valid_target_mask` 先转 int64 再比较负 ignore_index，避免无符号环绕

### 已审查的关键逻辑点
- [x] DyTCD 归一化：深度重参数化、learnable_k/k_min、learnable_delta
- [x] QK-norm：RMSNormHead 和 QKSquash 两种模式，RoPE 前应用
- [x] 分块交叉熵：`_ChunkedCrossEntropy` forward/backward，ignore 掩码
- [x] 混淆矩阵：`torch.bincount` 加法公式 target*V + predicted
- [x] macro/weighted F1：对 support>0 的类求平均，加权按 support
