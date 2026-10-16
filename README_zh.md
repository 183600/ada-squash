# DyT++ / AdaSquash

> 面向超深与低精度 Transformer 的免统计自适应压缩

[English](README.md) | **中文说明**

本仓库是 DyT++ 提案的实验代码库：一族*免统计*归一化层，沿三条正交轴推广
Dynamic Tanh（DyT, Zhu et al. 2025）：

| 变体 | 轴 | 新增内容 | 针对问题 |
|--------|------|---------------|---------|
| **DyT-S**  | 激活族 | 标量 α + 广义 softsign（{标量 × softsign} 因子格） | 饱和剖面 vs tanh |
| **DyT-P**  | 逐通道（tanh） | tanh 下的逐通道 α（{逐通道 × tanh} 因子格） | 把逐通道轴与激活族互换解耦 |
| **DyT-C**  | 逐通道 + softsign | 逐通道 alpha + 广义 softsign | 离群值集中通道（LLaMA "super-active" 维度） |
| **DyT-D**  | 深度条件 | 深度衰减 alpha + LayerScale 残差门 | Pre-LN Transformer 的"深度诅咒" |
| **DyT-CD** | 两者结合（主提案） | DyT-C + DyT-D | 超深模型、MoE、FP8 |
| **DyT-QK** | 注意力 logits（变体 C） | Q/K 点积前的标量压缩 | 长上下文 logit 爆炸 |

核心假设：免统计的逐元素自适应压缩可以在仅增加 **<0.1% 参数** 的前提下
替代基于统计的归一化。对 QK 变体，问题的诚实表述是：*逐元素饱和 vs 向量
归一化——在对齐有效 logit 尺度的前提下，谁更抗离群通道、谁更有注意力
表达力？*（两者给出的 logit 上界相同，见 DyT-QK 一节。）

## 仓库结构

```
ada-squash/
├── adasquash/        # 可导入的包
│   ├── __init__.py
│   ├── norms.py        # LayerNorm, RMSNorm, DyT(-S/-P/-C/-D/-CD/-DI), QKSquash
│   ├── attention.py    # 带可选 QK 压缩的因果自注意力
│   ├── model.py        # GPT 风格解码器，可选 RoPE，深度可配置
│   └── utils.py        # ActivationStats, KurtosisMeter, LossSpikeCounter, ...
├── scripts/           # CLI 入口
│   ├── train_gpt2.py             # 阶段 1 驱动（GPT-2 124M, FineWeb-Edu, 独立验证集）
│   ├── train_llama_depth_sweep.py# 阶段 2 深度扫描（12/24/48/96，含 dytdi 对照）
│   ├── eval_downstream.py       # lm-eval（自定义 LM 封装）+ needle
│   ├── ablation.py              # 单变量消融矩阵运行器，多种子
│   ├── train_moe_tiny.py        # 阶段 3a：tiny MoE 专家方差研究
│   ├── check_fp8_stability.py   # 阶段 3b：模拟 FP8 上溢/下溢（含适用范围声明）
│   ├── check_w8a8_ppl.py       # 阶段 3c：训练后 W8A8 PPL 恶化（独立保留集）
│   └── run_needle.py            # 阶段 3d：大海捞针扫描（按位置分解）
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

# 因子格单元：标量 softsign / 逐通道 tanh
python scripts/train_gpt2.py --config configs/gpt2-124m-dyts.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-dytp.yaml

# 深度对照：仅初始化变体与真正共享 (alpha_0, k)
python scripts/train_gpt2.py --config configs/gpt2-124m-dytdi.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-dytcdi.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-dytd-shared.yaml

# 基线下的 LayerScale 对照
python scripts/train_gpt2.py --config configs/gpt2-124m-ln-ls.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-rmsnorm-ls.yaml

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

用 `--resume <ckpt>.pt` 恢复中断的训练：模型、优化器、调度器、步数计数、
**随机数状态**（torch / CUDA / python，dropout 像未中断一样继续）与
**数据位置**（已消费的流式批数）全部恢复。默认会把 dataloader 快进跳过
已消费的批——流式顺序相同，因此续跑看到*新鲜*数据而非重放语料开头；
传 `--resume-skip-data` 可跳过快进（更快，但会重放训练数据——只有在
不跨断点比较 PPL/验证曲线时才可接受）。定义调度的 flag
（`--max-steps`、`--warmup`、`--min-lr-ratio`、`--lr`、`--weight-decay`、
`--beta1` / `--beta2`、`--grad-clip`）也会从 checkpoint 读回——否则余弦
曲线会在训练中途被悄悄改变，因为 `load_state_dict` 只保留
`last_epoch`。命令行里保持默认值的项被静默替换；显式输入的项被保留并
报告，这正是延长训练的方式（`--resume <ckpt> --max-steps <更大值>` 续跑
余弦尾段而非截断）。loss-spike 计数器及其 EMA 基线也在 checkpoint 里，
`n_loss_spikes` 跨断点继续累计而非清零。早于随机数/数据位置记录功能的
checkpoint 会在恢复时明确说明这一点，而不是假装是精确续跑。

`<output-dir>/` 下的日志输出：
- `train.jsonl`     - 逐步 loss / lr / 独立验证集 val_loss
- `tb/`             - TensorBoard 标量（train/ 与 val/）
- `ckpt_NNNNNN.pt`  - 可续跑 checkpoint（含随机数状态与数据位置）
- `stdout.log`      - 原始 stdout（深度扫描用）

自带的所有阶段 1 配置都启用了独立验证（`val_every: 500`，在
wikitext-103-raw-v1 *test* 上——与 FineWeb-Edu 训练流不相交），因此
验证曲线永远不会算在训练数据上。

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
# 报告 logit 偏差、逐归一化层的上溢/下溢/次正规/量化为零比例、残差流离开
# FP8 范围的首个块、以及第一个出现非有限值的块。
# 适用范围：仅前向（无 FP8 梯度/优化器状态），且 --scaling none（默认）
# 是*无缩放压力测试*——真实 FP8 方案会按张量 amax 重缩放，请同时跑
# --scaling dynamic-tensor 对照，再下训练稳定性结论。
# 加 `--ckpt <每个 --norms 一项>` 可用训练好的权重而非随机初始化探测。
python scripts/check_fp8_stability.py --norms ln rmsnorm dyt dyts dytp dytc dytd dytcd

# 阶段 3c：训练后 W8A8 PPL 恶化（同样量化 SwiGLU MLP）
# 探测语料很重要：均匀随机 token 下两个模型都在 ppl ~= vocab_size，
# 下降只是噪声，所以默认探测语料是可学习的合成语料。--probe dataset 读取
# *独立保留集*（默认 wikitext-103-raw-v1 test，与 FineWeb-Edu 训练流不相交）
# ——指向训练数据会使 ppl_fp 成为部分记忆的数字、恶化量基于被污染的基线。
# 报告含 ppl_fp（未量化基线）、ppl_w8a8、绝对/百分比恶化、评测集身份与
# token 数。
python scripts/check_w8a8_ppl.py \
    --ckpts runs/gpt2-124m-ln/ckpt_final.pt \
            runs/gpt2-124m-rmsnorm/ckpt_final.pt \
            runs/gpt2-124m-dytcd/ckpt_final.pt

# 阶段 3d：长上下文大海捞针
# 多个针（不同秘密与句式）× 上下文长度，并按针的*位置*三等分（early /
# middle / late）报告准确率——只记得住浅层针的模型在标量平均里与全位置
# 都记得的模型看起来一样。
python scripts/run_needle.py --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
    --ctx-lens 512 1024 2048 4096
```

### 消融

消融运行器在合成马尔可夫链语料上扫**单变量**矩阵（1000 步，GPT-2 124M）。
链是真正的*随机游走*（`s_{t+1} = (s_t + j_{t+1}) mod n_states`），目标
是当前状态的函数，loss 有远低于 `log(vocab_size)` 的可学习下界——均匀
随机 token 没有下一 token 信号，会把所有变体钉死在
`log(vocab_size)`，loss 列就排不出名次。

激活离群探针是**中间块之后对残差流单一通道的持续放大**（前向 hook 把
一个通道乘 25×；见 `install_residual_outlier`）。旧探针放大的是某个
token 的 embedding 行——但权重共享时该行同时是输出分类器的一行，于是
"处理输入侧离群通道"与"重新学会永不预测该 token"被混在一起，后者是
任何归一化层都不必处理的输出侧病灶。hook 不改任何权重，且对所有行
完全一致地施加，因此是比较的常量而非变量。

```bash
# 单种子
python scripts/ablation.py --config configs/ablation-matrix.yaml
# 每行 3 个种子 -> <out>/<name>/summary.json 里给出逐行 mean/std
python scripts/ablation.py --config configs/ablation-matrix.yaml --seeds 42 43 44
```

每个消融写一个 `summary.json`（多种子时逐种子在 `<out>/<name>/seed<N>/`），
外加聚合它们的 `all_summaries.json`（与上次运行合并，部分重跑不会丢掉
跳过的行）。每个 summary 含完整 loss 曲线、loss-spike 计数、离群探针
描述，以及全采样步汇聚的 pre-norm 激活统计。
`configs/ablation-matrix.yaml` 的每一行都注明参照行与它改变的唯一轴；维度：

- LN / RMSNorm / DyT-CD 下 ± LayerScale（旧方案 LN(无LS) → dytcd(开LS)
  的基线把该轴与归一化替换捆绑在一起）
- 激活族 × 逐通道因子格：`dyt` / `dyts` / `dytp` / `dytc`
  （每对相邻行只隔离一个轴），LS 关
- LS 对齐下的深度轴：`dyt_ls` → `dytd`（加重参数化）→ `dytdi`
  （同初始化、无重参数化）/ `dytd_shared`（全模型共享一组 `(α_0, k)`）/
  `dytd_k0`（关闭调度）
- softsign 指数 p：1 / 2 / 4 / 可学习
- DyT-CD 下的 LayerScale 初始化：1e-5 / 1e-3 / 关

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

**该参数化是什么（以及不是什么）。** 默认配置下每个归一化层持有独立的
`(α_0, k)` 对，而单层能*观察到*的只有乘积 `α_0 · exp(−k·l/L)`——一个
有效标量。因此深度衰减形状是每层 α 的**深度相关初始化先验**，而非跨层
涌现的规律：训练后各层 `α_l` 可以偏离任何共同曲线，学到的 `k` 的诚实
读法是"这层 α 起步时的衰减率"。两个机制让主张可检验而非修辞：

* `--k-min`（默认 `0.0`）给衰减率加下限，层的 `α_l` 不能经 `k < 0`
  随深度*增长*——调度在优化器到达的任何点都保持非递增
  （`--no-k-min` 恢复旧的 unconstrained k）。
* `--shared-depth-schedule` 用**全模型一组 `(α_0, k)`**（共 2 个标量）
  取代逐层参数对。此时该参数对是*可辨识的*：每层按自己的深度对同一组
  参数求 `α_l = α_0·exp(−k·l/L)`，学到的曲线是真正的跨层规律而非初始化。

消融矩阵的 `dytdi` / `dytcdi` 行是对应对照：普通逐层 α *初始化*在调度值
上然后自由学习。`dytd` vs `dytdi` 把"重参数化"与"初始化先验"分开；
若 `dytd` 只靠更好的起点取胜，`dytdi` 应与之打平。

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
  且 `dytd → dytcd` 一步跨*两*条轴（逐通道 α 与激活族）——矩阵把它分解为
  `dytd → dytcdi`（两轴同变、仅初始化）与 `dytcdi → dytcd`（加上
  重参数化），每一步重新成为单变量（见 `configs/ablation-matrix.yaml`）。

### DyT-QK — 变体 C：注意力 logit 压缩

```
Q' = tanh(α_q · Q);  K' = tanh(α_k · K)
logits = (Q' · K') · scale
```

* 用每层两个标量替代 QK-Norm（后者需要两个 `head_dim` 大小的 RMSNorm），
  且为逐元素操作（无集合通信）。
* **有界 logit——但 QK-Norm 同样有界。** `tanh` 把 `Q'` / `K'` 的每个
  元素限制在 ±1，`head_dim` 项点积在 scale 前上界 ±`head_dim`，经
  `1/sqrt(head_dim)` 后为 ±`sqrt(head_dim)`。但 RMSNorm 有*相同*上界：
  归一化后向量的 L2 范数恰为 `sqrt(head_dim)`，柯西不等式给
  `|q'ᵀk'| ≤ head_dim`（scale 前）——缩放后同样 ±`sqrt(head_dim)`。
  硬上界因此**不是**两者的区分点；真正不同的是有界向量的*形状*：

  | | tanh 压缩 | QK-RMSNorm |
  |---|---|---|
  | 尺度不变性 | 否——未饱和幅值透传（小 `z` 时 `tanh(z) ≈ z`），只有大值被封顶 | 是——token 向量大 10 倍，归一化结果*逐位相同* |
  | 单个主导通道 | 离群值饱和到 ±1，小通道保持相对大小 | 每个通道都被含离群值的 RMS 除，小通道相对被压垮 |
  | logit 能表达什么 | q·k 的幅值*和*方向（更有表达力，也更暴露） | 仅方向（每 token 范数被精确移除） |

* "把激活幅值从 logit 中彻底移除"是 QK-Norm 的主张，**不是** `tanh` 的：
  饱和封顶大幅值，中等激活照样透传并继续放大 logit。变体 C 的诚实表述
  是一个*权衡*：逐元素饱和（便宜、编译友好、保留部分幅值信息） vs 向量
  归一化（严格尺度不变、更贵）。哪个对长上下文训练更重要是本仓库去*测量*
  而非断言的开放问题。
* **公平比较需要对齐有效 logit 尺度。** RMS 归一化向量的 L2 范数是
  `sqrt(head_dim)`；tanh 压缩后的范数*至多* `sqrt(head_dim)`，通常小得
  多（只有完全饱和才达到）。相同 `α` 下 tanh 的 logit 系统性更小——任何
  tanh vs QKNorm 的比较都必须先校准 `α` 或在两侧扫温度，才能把差异归因
  于压缩形状本身。

## 指标

所有训练脚本记录：

| 指标                | 含义                                              |
|---------------------|---------------------------------------------------|
| `loss`              | 训练交叉熵                                        |
| `val_loss`          | 独立验证集交叉熵（自带配置用 wikitext-103 test）  |
| `n_loss_spikes`     | 整个训练中相对+绝对 loss 跳变次数                 |
| `max_abs_pre_norm`  | 进入每个归一化的最大 \|激活\|                     |
| `expert_cv`         | （仅 MoE）逐块 tokens-per-expert 的变异系数       |
| `logit_rel_divergence` | （FP8 模拟）相对 fp32 的最大绝对 logit 误差    |
| `overflow_rate_e4m3`| （FP8 模拟）激活超过 ±448 的比例                  |
| `underflow_rate_e4m3` | （FP8 模拟）激活低于 min-normal 的比例（小数值占比） |
| `subnormal_rate_e4m3` | （FP8 模拟）落在 (0, min-normal) 的比例——只能以次正规数表示 |
| `quantized_zero_rate_e4m3` | （FP8 模拟）非零激活被量化为精确 0 的比例   |
| `rel_quant_err_e4m3` | （FP8 模拟）平均相对量化误差                    |
| `first_overflow_layer` | （FP8 模拟）残差离开 ±448 的首个块             |
| `first_nonfinite_layer` | （FP8 模拟）残差出现非有限值的深度            |
| `ppl_fp`            | （W8A8）独立保留集上的未量化基线 PPL              |
| `ppl_w8a8`          | （W8A8）训练后量化后的 PPL                        |
| `ppl_drop_pct`      | （W8A8）训练后量化的 PPL 恶化幅度                 |

关于解读的提示：实验的好坏取决于探测语料。这里检查而非假设四种失败模式
——`check_w8a8_ppl.py` 拒绝 token（几乎）全不重复的探测语料，因为毫无
重复时任何模型都只能瞎猜，且其 `--probe dataset` 路径默认指向独立保留集
（wikitext-103 test）而非训练流；`check_fp8_stability.py` 只在训练好的
权重上测离群通道（所以必须传 `--ckpt`），并把自身输出标注为仅前向、
无缩放的压力测试；消融/MoE 语料是真正的马尔可夫游走而非均匀抽样；
大海捞针按插入位置报告准确率而不只是标量平均。

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
