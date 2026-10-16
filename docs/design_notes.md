# DyT++ / AdaSquash 设计说明（固定版）

> **本文档是本仓库的"设计宪法"。** 其中标记【固定】的决策为最终决定，
> CI 循环（含 perf / bug / clean / apply 各阶段）**必须遵守，不得替换、
> 不得删除、不得修改本文件**。CI 会在每次阶段提交前运行 `ci_gate.py`
> 校验（语法 + 设计固定项 + 全量 pytest 套件）；违反固定设计的改动
> 会被整段撤销（`git reset --hard`）。
>
> 人工变更流程：直接修改本文件并提交后，**必须**同步移动固定标签：
> `git tag -f design-pin <新提交sha> && git push origin design-pin`。
> 未移动标签的修改会在下一次 CI 运行时被自动回滚。

## 0. 为什么有这份文档

自主 CI 循环由随机更换的免费模型驱动，不同模型对"最优实现"判断不一，
历史上已造成 88 处"加→删→再加"式震荡（核心数学 `generalized_softsign`
被改 5 轮、SDPA 调用被改 3 轮、QKSquash 被改 3 轮）。为终止布朗运动，
将关键设计固定如下。

## 1. 研究命题（背景，不可漂移）

**DyT++ / AdaSquash**：无统计量（statistic-free）的自适应压缩归一化
家族，沿三个正交轴推广 Dynamic Tanh（DyT, Zhu et al. 2025）：

| 变体 | 轴 | 增加什么 | 针对问题 |
|---|---|---|---|
| **DyT-C** | 逐通道 | per-channel alpha + 广义 softsign | 离群通道（LLaMA "super-active" dims） |
| **DyT-D** | 深度条件化 | 深度衰减 alpha + LayerScale 残差 | Pre-LN 的"深度诅咒" |
| **DyT-CD** | 两者结合（主提案） | DyT-C + DyT-D | 超深、MoE、FP8 |
| **DyT-QK** | 注意力 logits | Q/K 点积前的标量压缩 | 长上下文 logit 爆炸 |

核心假设：无统计量的逐元素自适应压缩可以替代基于统计的归一化，
代价 <0.1% 额外参数。对照基线：LayerNorm、RMSNorm、原版 DyT。

## 2.【固定】设计决策

### D1. 注意力固定为 SDPA 标准实现（基线设施，不是研究变量）

- **决策**：`CausalSelfAttention` 的主路径固定为
  `F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=...)`。
  本仓库的研究变量是**归一化层**；注意力刻意保持标准、简单、无自定义
  （见 attention.py 模块 docstring 的原始设计声明）。
- **禁止**：把手写注意力 math 路径、手写 softmax kernel、或任何自定义
  attention 实现替换进主路径——那会引入与研究变量无关的混淆因子。
  logit 监控（`return_stats=True` 分支）保持 no_grad 的独立旁路，不进入
  主路径。
- **理由**：SDPA ↔ 手写路径在历史上被来回切换 3 轮；标准 kernel 是
  所有变体公平共享的基线。

### D2. generalized_softsign 的数值稳定语义

- **决策**：`phi_p(x) = x / (1 + |x|^p)^{1/p}` 按双分支倒数形式实现：
  - `u = min(|x|, 1) / max(|x|, 1)`（恒 ≤1，pow 链永不爆炸）；
  - 倒数取在 `max(|x|, 1)` 上而**不是** `|x|`（精确零会导致
    `1/0 = inf`、未选中分支的 `-1/t^2` 导数 NaN，污染全部梯度）；
  - `p` 的下限 `_MIN_P = 1e-3`（p→0⁺ 层退化无梯度，p<0 在 |x|=0 处
    NaN）；
  - DyT-CD 深度条件 alpha 的下限 `_MIN_ALPHA = 1e-4`（非正 alpha 翻转
    通道符号，零 alpha 杀梯度）；
  - 符号通过 `copysign` 骑在符号位上（`|x|=inf` 仍得 ±1 而非 inf×0）。
- **禁止**：改动 `_MIN_P` / `_MIN_ALPHA` 量级；把倒数改成对 `|x|`
  直取；去掉 clamp；改回 `torch.where` 双分支实现（数值行为不同）。
- **理由**：这些 floor 只在极端运行点激活，是长训练 + p 可学习 +
  低精度（FP8 实验）下不 NaN 的保证；tests 的逐参数梯度验证依赖
  该精确语义。

### D3. 手写 fused autograd kernel 家族（研究核心，锁死）

- **决策**：以下 kernel 必须存在且保持"正向闭式 + 反向闭式重算"的
  结构，由 `tests/test_norms.py` 的**逐参数对拍**（与独立 autograd
  重算在随机参数点比较）保证正确性：
  - `norms.py`：`_GeneralizedSoftsign`、`_RMSNorm`、`_DyTForward`、
    `_DyTCForward`、`_DyTDForward`、`_DyTCDForward`；
  - `model.py`：`_ChunkedCrossEntropy`、`_FusedLinearCrossEntropy`、
    `_LayerScaleResidual`、`_SwiGLU`。
- **禁止**：删除 kernel 改用会物化中间量的普通 autograd 版本；
  绕过 `generalized_softsign()` 公共入口；把对拍测试改弱（如去掉
  随机参数化）。
- **理由**：fused kernel 省显存 + 保数值，是深度扫描（96 层）与
  FP8 实验能跑的前提；对拍测试是"kernel 正确"的最终裁判。

### D4. QKSquash 固定为 tanh 压缩

- **决策**：`Q' = tanh(alpha_q · Q)`，`K' = tanh(alpha_k · K)`，
  logits 随后乘 `1/sqrt(head_dim)`。`tanh` 把元素界在 [-1,1]，点积
  上界 ±head_dim，缩放后 ±sqrt(head_dim)。
- **上界澄清**：RMSNorm 有*相同*的天花板——归一化后向量 L2 范数恰为
  sqrt(head_dim)，柯西不等式给 |q'ᵀk'| ≤ head_dim，缩放后同样是
  ±sqrt(head_dim)。区别不在上界，而在有界向量的*形状*：tanh 非尺度
  不变（未饱和幅值透传，仅大值封顶），RMSNorm 严格尺度不变（整
  token 范数被精确移除）。与基线对照的是"逐元素饱和 vs 向量归一化"
  这一机制差异；公平比较还需对齐有效 logit 尺度（tanh 向量范数
  ≤ sqrt(head_dim)，通常更小）。
- **禁止**：把 tanh 换成 softsign / clip / 其它压缩函数（那是
  另一个研究问题）；改共享标量对（`k_squash = q_squash`）的设计。
- **理由**：与 RMSNorm 基线（重缩放不饱和）对照的正是"饱和截断"
  这一机制本身。

### D5. 归一化注册表与超参分发

- **决策**：`NORM_REGISTRY` 必须包含
  `ln / rmsnorm / dyt / dyts / dytp / dytc / dytd / dytdi / dytcd / dytcdi`
  十项（四格因子分解单元 + 深度重参数化与深度初始化对照）；超参
  （`alpha_init / alpha_0 / k_init / p_init`）经 `build_norm_kwargs`
  统一分发，不接收该超参的基线（LN/RMSNorm）自动忽略。
- **禁止**：改注册名（configs/ 与脚本依赖）；把分发逻辑内联到
  Block 构造里。

### D6. 注意力辅助缓存的键语义

- **决策**：tril 掩码按 `(T, device)` 缓存；RoPE 的 (cos, sin) 表按
  `(data_ptr, shape, device, dtype)` 缓存（调用方每次传的是同一 buffer
  的新 view，identity 测试会失效）。
- **禁止**：把缓存键改成 Python `id()`；去掉缓存逐调用重建
  （性能倒退且无害化震荡）。

## 3. 质量门禁（ci_gate.py）

CI 在每次阶段提交前运行，三层：

1. **语法**：全部 `.py` 可解析；
2. **设计固定项**：D1–D6 以源码检查强制（如"attention.py 必须含
   SDPA 调用"、"norms.py 必须含 `_MIN_P = 1e-3`"）；
3. **全量 pytest**：`python3 -m pytest tests/ -x -q`（82 项，含每个
   fused kernel 的逐参数梯度对拍）。torch/pytest 缺失时降级为前两层
   并告警，网络故障不会卡死循环。

任一层失败 → 该阶段改动被 `git reset --hard` 整段撤销。
`--syntax-only` 模式供 runner 关闭时的 trap 使用。

## 4. 目录

```
adasquash/norms.py      # D2–D5（归一化家族与核心数学）
adasquash/attention.py  # D1、D6（SDPA 基线 + QKSquash + RoPE）
adasquash/model.py      # D3（GPT + fused cross-entropy/SwGLU）
adasquash/utils.py       # 统计仪表（KurtosisMeter 等）
tests/test_norms.py     # 对拍测试套件（gate 第 3 层）
ci_gate.py              # 质量门禁
docs/design_notes.md    # 本文档
算力申请.md              # 算力申请材料（apply 阶段检查修复，不得删除）
```
