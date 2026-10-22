# DyT++ / AdaSquash

> 面向超深与低精度 Transformer 的免统计自适应压缩

[English](README.md) | **中文说明**

本仓库是 DyT++ 提案的实验代码库：一族*免统计*归一化层，沿三条正交轴推广
Dynamic Tanh（DyT, Zhu et al. 2025）：

| 变体 | 轴 | 新增内容 | 针对问题 |
|--------|------|---------------|---------|
| **DyT-C**  | 逐通道 | 逐通道 alpha + 广义 softsign | 离群值集中通道（LLaMA "super-active" 维度） |
| **DyT-D**  | 深度条件 | 深度衰减 alpha + LayerScale 残差门 | Pre-LN Transformer 的"深度诅咒" |
| **DyT-CD** | 两者结合（主提案） | DyT-C + DyT-D | 超深模型、MoE、FP8 |
| **DyT-QK** | 注意力 logits（变体 C） | Q/K 点积前的标量压缩 | 长上下文 logit 爆炸 |

核心假设：免统计的逐元素自适应压缩可以在仅增加 **<0.1% 参数** 的前提下
替代基于统计的归一化。

## 仓库结构

```
ada-squash/
├── adasquash/        # 可导入的包
│   ├── __init__.py
│   ├── norms.py        # LayerNorm, RMSNorm, DyT, DyT-C, DyT-D, DyT-CD, QKSquash
│   ├── attention.py    # 带可选 QK 压缩的因果自注意力
│   ├── model.py        # GPT 风格解码器，可选 RoPE，深度可配置
│   └── utils.py        # ActivationStats, KurtosisMeter, LossSpikeCounter, ...
├── scripts/           # CLI 入口
│   ├── train_gpt2.py             # 阶段 1 驱动（GPT-2 124M, FineWeb-Edu）
│   ├── train_llama_depth_sweep.py# 阶段 2 深度扫描（12/24/48/96）
│   ├── eval_downstream.py       # MMLU / GSM8K / HumanEval / DROP / needle
│   ├── ablation.py              # 一次性消融矩阵运行器
│   ├── train_moe_tiny.py        # 阶段 3a：tiny MoE 专家方差研究
│   ├── check_fp8_stability.py   # 阶段 3b：模拟 FP8 上溢/下溢
│   ├── check_w8a8_ppl.py       # 阶段 3c：训练后 W8A8 PPL 恶化
│   └── run_needle.py            # 阶段 3d：大海捞针扫描
├── tests/
│   └── test_norms.py            # 形状/dtype/梯度/调度健全性测试
├── configs/                    # YAML 配置（见 configs/）
├── requirements.txt
└── README.md
```

## 快速开始

### 安装

```bash
pip install -r requirements.txt
# （可选）下游评估：
pip install lm-eval
```

### 健全性测试

```bash
python -m pytest tests/ -v
```

覆盖形状/dtype 保持、softsign 有界性、深度调度单调性、QK 压缩 logit
上界、checkpoint 往返、指标实现，以及——由于本仓库每个归一化层都是手写
autograd kernel——把每个 kernel 与普通 autograd 的独立重组逐参数对拍，
且从*随机化*参数集出发，避免在 `gamma=1, beta=0` 初始化下对拍流于形式。

### 阶段 1：FineWeb-Edu 上的 GPT-2 124M

> 规模说明：仓库自带配置为 `max_steps × batch_size ×
> grad_accum × seq_len` = 10000 × 32 × 4 × 1024 ≈ **13 亿 token**，
> 不是完整的 100 亿 token FineWeb-Edu 样本。要跑完整样本请调大
> `--max-steps`（约 76 000 对应 10 B token），且对所有变体保持一致，
> 保证每个变体看到相同数据量。

每次训练由 `configs/` 下的 YAML 描述。`--config` 提供 CLI 默认值，
显式传入的 flag 优先于配置文件。

```bash
# Pre-LN 基线（按配置关闭 LayerScale）
python scripts/train_gpt2.py --config configs/gpt2-124m-ln.yaml

# RMSNorm 基线
python scripts/train_gpt2.py --config configs/gpt2-124m-rmsnorm.yaml

# 原版 DyT
python scripts/train_gpt2.py --config configs/gpt2-124m-dyt.yaml

# DyT++ 主提案
python scripts/train_gpt2.py --config configs/gpt2-124m-dytcd.yaml
```

不使用配置文件的等价显式写法：

```bash
python scripts/train_gpt2.py \
    --norm ln --no-layerscale \
    --max-steps 10000 --batch-size 32 --seq-len 1024 \
    --grad-accum 4 --lr 6e-4 --warmup 500 \
    --output-dir runs/gpt2-124m-ln
```

用 `--resume <ckpt>.pt` 恢复中断的训练：模型、优化器、调度器与步数计数
全部恢复，LR 曲线从断点继续而不是重放 warmup。定义调度的 flag
（`--max-steps`、`--warmup`、`--min-lr-ratio`、`--lr`、`--weight-decay`、
`--beta1` / `--beta2`、`--grad-clip`）也会从 checkpoint 读回——否则余弦
曲线会在训练中途被悄悄改变，因为 `load_state_dict` 只保留
`last_epoch`。命令行里保持默认值的项被静默替换；显式输入的项被保留并
报告，这正是延长训练的方式（`--resume <ckpt> --max-steps <更大值>` 续跑
余弦尾段而非截断）。loss-spike 计数器及其 EMA 基线也在 checkpoint 里，
`n_loss_spikes` 跨断点继续累计而非清零。

`<output-dir>/` 下的日志输出：
- `train.jsonl`     - 逐步 loss / lr
- `tb/`             - TensorBoard 标量
- `ckpt_NNNNNN.pt`  - 可续跑 checkpoint
- `stdout.log`      - 原始 stdout（深度扫描用）

### 阶段 2：LLaMA-tiny 1.1B 深度扫描

```bash
python scripts/train_llama_depth_sweep.py \
    --config configs/llama-1b-depth-sweep.yaml \
    --sequential
```

每个 `(depth, norm)` 组合在 `runs/depth-sweep/d{depth}-{norm}/` 下的独立
子进程运行。深度与宽度通过 `ADASQUASH_DEPTH_OVERRIDE`、
`ADASQUASH_DIM_OVERRIDE`、`ADASQUASH_HEADS_OVERRIDE` 环境变量传入
（head_dim 恒为 128；`dim` 随深度缩小使参数量近似恒定——否则"深度诅咒"
这条轴会与模型规模混淆）。`dim` 调度遵循 `dim ≈ base_dim ·
sqrt(base_depth / depth)`，在 L = 12 / 24 / 48 / 96 时约 1.16 / 1.07 /
0.98 / 1.01 B 参数，极差 1.19×；该表仅用于 LLaMA-tiny 基准形状，其他基
准模型按自身宽度重新缩放。同一训练脚本也可直接对单个配置调用：

```bash
ADASQUASH_DEPTH_OVERRIDE=48 \
ADASQUASH_DIM_OVERRIDE=1536 ADASQUASH_HEADS_OVERRIDE=12 \
    python scripts/train_gpt2.py \
    --model llama-1b --norm dytcd --max-steps 5000 ...
```

想让 `dim` 跨深度固定不变，给扫描启动器传 `--no-dim-scaling`。

### 阶段 3：特殊实验

```bash
# 阶段 3a：tiny MoE (8x0.3B) — 专家负载不均衡
python scripts/train_moe_tiny.py --norm dytcd --n-experts 8

# 阶段 3b：FP8 稳定性（CPU/GPU 模拟，无需真实 FP8 硬件）
# 把每次 GEMM 的两个操作数（含 SwiGLU MLP）都放到 FP8 量化值上跑真实前向，
# 报告 logit 偏差、逐归一化层的上/下溢、残差流离开 FP8 范围的首个块、
# 以及第一个出现非有限值的块。
# 加 `--ckpt <每个 --norms 一项>` 可用训练好的权重而非随机初始化探测。
python scripts/check_fp8_stability.py --norms ln rmsnorm dyt dytc dytd dytcd

# 阶段 3c：训练后 W8A8 PPL 恶化（同样量化 SwiGLU MLP）
# 探测语料很重要：均匀随机 token 下两个模型都在 ppl ~= vocab_size，
# 下降只是噪声，所以默认探测语料是可学习的合成语料（--probe dataset 改用真实文本）。
python scripts/check_w8a8_ppl.py \
    --ckpts runs/gpt2-124m-ln/ckpt_final.pt \
            runs/gpt2-124m-rmsnorm/ckpt_final.pt \
            runs/gpt2-124m-dytcd/ckpt_final.pt

# 阶段 3d：长上下文大海捞针
python scripts/run_needle.py --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
    --ctx-lens 512 1024 2048 4096
```

### 消融

消融运行器在合成马尔可夫链语料上扫变体矩阵（1000 步，GPT-2 124M）。
链是真正的*随机游走*（`s_{t+1} = (s_t + j_{t+1}) mod n_states`），目标
是当前状态的函数，loss 有远低于 `log(vocab_size)` 的可学习下界——均匀
随机 token 没有下一 token 信号，会把所有变体钉死在
`log(vocab_size)`，loss 列就排不出名次。刻意放大的 embedding 行提供了
压缩层要控制的激活离群通道。

```bash
python scripts/ablation.py --config configs/ablation-matrix.yaml
```

每个消融写一个 `summary.json`，外加聚合它们的 `all_summaries.json`
（与上次运行合并，部分重跑不会丢掉跳过的行）。每个 summary 含完整 loss
曲线、loss-spike 计数，以及全采样步汇聚的 pre-norm 激活统计。消融维度：

- 逐通道开/关
- 深度条件开/关
- LayerScale 初始化：1e-5 / 1e-3 / 关
- softsign 族：p ∈ {1, 2, 4, 可学习, →∞（=硬 tanh）}
- DyT alpha 初始化：0.3 / 0.5 / 1.0

训练前运行器会写 `analysis_plan.json`：确认性对比集合由已校验的矩阵导出，
每个对比有稳定 id 和明确的科学假设，整个注册表带哈希指纹。四份对比报告之后
再写 `study_level_tests.json`，对预注册对比做分层 Holm 门控（第一层：主提案
及其机制；第二层：其余单轴、对照与 QK 对比，只有第一层至少一项成立才检验）。
不在计划中的检验只作为探索性结果报告。每个对比还附带留一 seed 的 jackknife
（去掉任一 seed 后效应符号是否仍在），以及 `--equivalence-margin <delta>` 开启
的 TOST 等价性检验——"两个设置相同"因此是一个有检验的结论，而不是" superiority
检验不显著"。析因交互项按双侧报告，避免把拮抗交互误读成"协同不显著"。

## 方法速查

### 原版 DyT（基线）

```
y = γ · tanh(α · x) + β        # α 是单个标量
```

### DyT-C — 变体 A：逐通道自适应

```
x_hat = α_c · x                                # α_c : [C]
φ    = x_hat / (1 + |x_hat|^p)^{1/p}          # 广义 softsign
y    = γ · φ + β
```

* p=1：经典 softsign；p=2：平滑 softsign（默认）；p→∞：硬 tanh
  （即 clip 到 [-1, 1]——注意这*不是* tanh）。
* 全部严格逐元素 → 无跨 token 统计、无集合通信。对 kernel 友好：
  与周围 matmul 融合是平凡的。
* 每个归一化层多一个 `[C]` 向量，通常 <0.1% 模型参数。

### DyT-D — 变体 B：深度条件

```
α_l = α_0 · exp(−k · l / L)
y   = γ · tanh(α_l · x) + β
x_out = x + λ_l · Block(y)        # LayerScale 式残差门
```

* α_l 随深度衰减：浅层强压缩（限制最大的 pre-norm 激活），深层放松到
  近线性、梯度友好的响应。
* 与 LayerScale 残差门结合，针对 Pre-LN Transformer 的"深度诅咒"
  （Liu et al. 2020）。

### DyT-CD — 主提案（A + B 结合）

```
α_c,l = α_0 · exp(−k · l / L) · (1 + δ_c)
x_hat = α_c,l · x
φ    = x_hat / (1 + |x_hat|^p)^{1/p}
y    = γ · φ + β
```

* δ_c 初始化为**零**，因此逐通道因子从 `α_c,l = α_l` 起步（无逐通道
  偏差）。这使训练好的 DyT-D checkpoint 可以作为共享 `α_0`、`k`、
  `γ`、`β` 的*热启动*，但两层**并不**功能等价：DyT-D 用 `tanh` 压缩，
  DyT-CD 用广义 softsign 和可学习 `p`。因此该迁移不是无损的"免费升级"，
  消融矩阵的 `dytd → dytcd` 行把逐通道轴与激活族变更捆绑在一起（见
  `configs/ablation-matrix.yaml`）。

### DyT-QK — 变体 C：注意力 logit 压缩

```
Q' = tanh(α_q · Q);  K' = tanh(α_k · K)
logits = (Q' · K') · scale
```

* 用每层两个标量替代 QK-Norm（后者需要两个 `head_dim` 大小的 RMSNorm）。
* 把激活幅值从 logit 中彻底移除。`tanh` 把 `Q'` / `K'` 的每个元素限制
  在 ±1，`head_dim` 项点积在 scale 前上界 ±`head_dim`，经
  `1/sqrt(head_dim)` 后为 ±`sqrt(head_dim)`——一个与输入激活大小无关的
  硬上界。这正是消灭长上下文训练中 logit 爆炸失败模式的机制。
* 就此目的而言比 QK-Norm 更锋利：RMSNorm 只重缩放不饱和，单个主导通道
  仍能推高 logit，而 `tanh` 直接截断。

## 指标

所有训练脚本记录：

| 指标                | 含义                                              |
|---------------------|---------------------------------------------------|
| `loss`              | 训练交叉熵                                        |
| `n_loss_spikes`     | 整个训练中相对+绝对 loss 跳变次数                 |
| `max_abs_pre_norm`  | 进入每个归一化的最大 \|激活\|                     |
| `expert_cv`         | （仅 MoE）逐块 tokens-per-expert 的变异系数       |
| `logit_rel_divergence` | （FP8 模拟）相对 fp32 的最大绝对 logit 误差    |
| `overflow_rate_e4m3`| （FP8 模拟）激活超过 ±448 的比例                  |
| `underflow_rate_e4m3` | （FP8 模拟）激活低于 min-normal 的比例          |
| `first_overflow_layer` | （FP8 模拟）残差离开 ±448 的首个块             |
| `first_nonfinite_layer` | （FP8 模拟）残差出现非有限值的深度            |
| `ppl_drop_pct`      | （W8A8）训练后量化的 PPL 恶化幅度                 |

关于解读的提示：实验的好坏取决于探测语料。这里检查而非假设三种失败模式
——`check_w8a8_ppl.py` 拒绝 token（几乎）全不重复的探测语料，因为毫无
重复时任何模型都只能瞎猜；`check_fp8_stability.py` 只在训练好的权重上
测离群通道，所以必须传 `--ckpt`；消融/MoE 语料是真正的马尔可夫游走而
非均匀抽样。

## 引用

```
@misc{adasquash2026,
  title  = {DyT++: Adaptive Squashing without Statistics for Ultra-Deep and
            Low-Precision Transformers},
  author = {Zhong Wenjia},
  year   = {2026},
  url    = {https://gitee.com/qwe12345678/ada-squash}
}
```

## License

MIT
