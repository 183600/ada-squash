# DyT++ / AdaSquash

> Adaptive Squashing without Statistics for Ultra-Deep and Low-Precision Transformers

This repository is the experimental codebase for the DyT++ proposal: a
family of *statistic-free* normalization layers that generalize
Dynamic Tanh (DyT, Zhou et al. 2025) along three orthogonal axes:

| Variant | Axis | What it adds | Targets |
|--------|------|---------------|---------|
| **DyT-C**  | per-channel                | per-channel alpha + generalized softsign | outlier-heavy channels (LLaMA "super-active" dims) |
| **DyT-D**  | depth-conditioned         | depth-decaying alpha + LayerScale residual | the "curse of depth" in Pre-LN Transformers |
| **DyT-CD** | both (the main proposal)  | DyT-C + DyT-D combined                     | ultra-deep, MoE, FP8 |
| **DyT-QK** | attention logits (Variant C) | scalar squashing of Q/K before dot product | long-context logit explosion |

The core hypothesis: with **<0.1% additional parameters**, statistic-free
element-wise adaptive squashing matches or exceeds RMSNorm on stability
while saving the synchronization cost of `mean`/`var` reductions at
inference.

## Repository layout

```
ada-squash/
├── adasquash/        # importable package
│   ├── __init__.py
│   ├── norms.py        # LayerNorm, RMSNorm, DyT, DyT-C, DyT-D, DyT-CD, QKSquash
│   ├── attention.py    # CausalSelfAttention with optional QK-squashing
│   ├── model.py        # GPT-style decoder, RoPE optional, configurable depth
│   └── utils.py        # ActivationStats, KurtosisMeter, LossSpikeCounter, ...
├── scripts/           # CLI entrypoints
│   ├── train_gpt2.py             # Phase 1 driver (GPT-2 124M, FineWeb-Edu)
│   ├── train_llama_depth_sweep.py# Phase 2 depth sweep (12/24/48/96)
│   ├── eval_downstream.py       # MMLU / GSM8K / HumanEval / DROP / needle
│   ├── ablation.py              # one-shot ablation matrix runner
│   ├── train_moe_tiny.py        # Phase 3a: tiny MoE expert-variance study
│   ├── check_fp8_stability.py   # Phase 3b: simulated FP8 overflow / underflow
│   ├── check_w8a8_ppl.py       # Phase 3c: post-training W8A8 PPL drop
│   └── run_needle.py            # Phase 3d: needle-in-a-haystack sweep
├── tests/
│   └── test_norms.py            # shape/dtype/gradient/schedule sanity
├── configs/                    # YAML configs (see configs/)
├── requirements.txt
└── README.md
```

## Quick start

### Install

```bash
pip install -r requirements.txt
# (optional) for downstream eval:
pip install lm-eval
```

### Sanity tests

```bash
python -m pytest tests/ -v
```

These cover shape/dtype preservation, gradient flow, softsign boundedness,
depth schedule monotonicity, QK-squashing logit bounds, checkpoint
round-trip, and the metrics implementation.

### Phase 1: GPT-2 124M on FineWeb-Edu (10B tokens)

```bash
# Pre-LN baseline
python scripts/train_gpt2.py \
    --norm ln \
    --max-steps 10000 --batch-size 32 --seq-len 1024 \
    --grad-accum 4 --lr 6e-4 --warmup 500 \
    --output-dir runs/gpt2-124m-ln

# RMSNorm baseline
python scripts/train_gpt2.py --norm rmsnorm \
    --output-dir runs/gpt2-124m-rmsnorm ... # same as above

# Original DyT
python scripts/train_gpt2.py --norm dyt --alpha-init 0.6 \
    --output-dir runs/gpt2-124m-dyt ...

# DyT++ main proposal
python scripts/train_gpt2.py --norm dytcd \
    --alpha-0 0.9 --k-init 1.0 --p-init 2.0 \
    --output-dir runs/gpt2-124m-dytcd ...
```

Logged outputs in `<output-dir>/`:
- `train.jsonl`     - per-step loss / lr / throughput / spike count
- `tb/`             - TensorBoard scalars
- `ckpt_NNNNNN.pt`  - resumable checkpoints
- `stdout.log`      - raw stdout (for depth-sweep runs)

### Phase 2: LLaMA-tiny 1.1B with depth sweep

```bash
python scripts/train_llama_depth_sweep.py \
    --depths 12 24 48 96 \
    --norms rmsnorm dyt dytcd \
    --max-steps 5000 \
    --output-dir runs/depth-sweep
```

Each `(depth, norm)` combination runs in its own subprocess under
`runs/depth-sweep/d{depth}-{norm}/`. The depth override is signalled via
the `ADASQUASH_DEPTH_OVERRIDE` env var, so the same training script can be
invoked directly for a single config:

```bash
ADASQUASH_DEPTH_OVERRIDE=48 python scripts/train_gpt2.py \
    --model llama-1b --norm dytcd --max-steps 5000 ...
```

### Phase 3: special experiments

```bash
# Phase 3a: tiny MoE (8x0.3B) — expert variance drift
python scripts/train_moe_tiny.py --norm dytcd --n-experts 8

# Phase 3b: FP8 stability (CPU/GPU simulation, no real FP8 hardware needed)
python scripts/check_fp8_stability.py --norms ln rmsnorm dyt dytc dytd dytcd

# Phase 3c: post-training W8A8 PPL drop
python scripts/check_w8a8_ppl.py \
    --ckpts runs/gpt2-124m-ln/ckpt_final.pt \
            runs/gpt2-124m-rmsnorm/ckpt_final.pt \
            runs/gpt2-124m-dytcd/ckpt_final.pt

# Phase 3d: long-context needle-in-a-haystack
python scripts/run_needle.py --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
    --ctx-lens 512 1024 2048 4096
```

### Ablations

The ablation runner sweeps the variant matrix on synthetic data (1000
steps, GPT-2 124M) - cheap enough to fit on one H100 in <2 hours.

```bash
python scripts/ablation.py --output-dir runs/ablation
```

It writes a `summary.json` per ablation and an `all_summaries.json`
aggregating them. Ablations include:

- Per-channel on/off
- Depth-conditioning on/off
- LayerScale init: 1e-5 / 1e-3 / off
- Softsign family: p ∈ {1, 2, 4, learnable, →∞ (=tanh)}
- DyT alpha init: 0.3 / 0.5 / 1.0

## Method reference

### Original DyT (baseline)

```
y = γ · tanh(α · x) + β        # α is a single scalar
```

### DyT-C — Variant A: per-channel adaptive

```
x_hat = α_c · x                                # α_c : [C]
φ    = x_hat / (1 + |x_hat|^p)^{1/p}          # generalized softsign
y    = γ · φ + β
```

* p=1: classic softsign; p=2: smooth softsign (default); p→∞: tanh.
* All operations are strictly element-wise → no cross-token statistics
  and no collective ops. Kernel-friendly: trivially fuses with the
  surrounding matmul.
* Parameter cost: one extra `[C]` vector per norm, typically <0.1% of
  the model.

### DyT-D — Variant B: depth-conditioned

```
α_l = α_0 · exp(−k · l / L)
y   = γ · tanh(α_l · x) + β
x_out = x + λ_l · Block(y)        # LayerScale-style residual gate
```

* Shallow layers keep α large → mild squashing → preserves gradient flow.
* Deep layers compress aggressively → bounds the residual stream.
* Directly addresses the "curse of depth" (Liu et al. 2020).

### DyT-CD — main proposal (A + B combined)

```
α_c,l = α_0 · exp(−k · l / L) · (1 + δ_c)
x_hat = α_c,l · x
φ    = x_hat / (1 + |x_hat|^p)^{1/p}
y    = γ · φ + β
```

* δ_c is initialized to **zero**, so a trained DyT-D checkpoint can be
  warm-started into DyT-CD with no accuracy hit - this is the "free
  upgrade" path that makes ablations clean.

### DyT-QK — Variant C: attention logit squashing

```
Q' = tanh(α_q · Q);  K' = tanh(α_k · K)
logits = (Q' · K') · scale
```

* Replaces QK-Norm (which requires two RMSNorms of size `head_dim`) with
  two scalars per layer.
* Bounds logits to ±1 pre-scale → kills the logit-explosion failure mode
  that breaks long-context training.

## Metrics

All training scripts log:

| Metric              | What it captures                                   |
|---------------------|---------------------------------------------------|
| `loss`              | training cross-entropy                            |
| `n_spikes`          | `LossSpikeCounter` (EMA-relative jump)             |
| `max_abs_*`         | per-layer max abs pre-norm activation (outlier)    |
| `kurtosis_*`        | per-layer excess kurtosis pre-norm (heavy-tail)   |
| `tok_per_sec`       | throughput                                        |
| `expert_cv`         | (MoE only) CoV of tokens-per-expert               |
| `overflow_rate_e4m3`| (FP8 sim) fraction of activations above ±448      |
| `underflow_rate_e4m3` | (FP8 sim) fraction of activations below min-normal |

## Citation

```
@misc{adasquash2026,
  title  = {DyT++: Adaptive Squashing without Statistics for Ultra-Deep and
            Low-Precision Transformers},
  author = {AdaSquash Team},
  year   = {2026},
  url    = {https://gitee.com/qwe12345678/ada-squash}
}
```

## License

MIT.
