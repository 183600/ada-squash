# DyT++ / AdaSquash

> Adaptive Squashing without Statistics for Ultra-Deep and Low-Precision Transformers

This repository is the experimental codebase for the DyT++ proposal: a
family of *statistic-free* normalization layers that generalize
Dynamic Tanh (DyT, Zhu et al. 2025) along three orthogonal axes:

| Variant | Axis | What it adds | Targets |
|--------|------|---------------|---------|
| **DyT-C**  | per-channel                | per-channel alpha + generalized softsign | outlier-heavy channels (LLaMA "super-active" dims) |
| **DyT-D**  | depth-conditioned         | depth-decaying alpha + LayerScale residual | the "curse of depth" in Pre-LN Transformers |
| **DyT-CD** | both (the main proposal)  | DyT-C + DyT-D combined                     | ultra-deep, MoE, FP8 |
| **DyT-QK** | attention logits (Variant C) | scalar squashing of Q/K before dot product | long-context logit explosion |

The core hypothesis: statistic-free element-wise adaptive squashing can
stand in for statistic-based normalization while adding **<0.1% additional
parameters**.

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

### Phase 1: GPT-2 124M on FineWeb-Edu

> Note on scale: the shipped configs are `max_steps × batch_size ×
> grad_accum × seq_len` = 10000 × 32 × 4 × 1024 ≈ **1.3 B tokens**, not the
> full 10 B-token FineWeb-Edu sample. Raise `--max-steps` (to ≈ 76 000 for
> 10 B tokens) for the full sample, and raise it identically for every
> variant so that each variant sees the same amount of data.

Every run is described by a YAML file under `configs/`. `--config` seeds the
CLI defaults, and an explicitly passed flag still wins over the file.

```bash
# Pre-LN baseline (LayerScale off, per the config)
python scripts/train_gpt2.py --config configs/gpt2-124m-ln.yaml

# RMSNorm baseline
python scripts/train_gpt2.py --config configs/gpt2-124m-rmsnorm.yaml

# Original DyT
python scripts/train_gpt2.py --config configs/gpt2-124m-dyt.yaml

# DyT++ main proposal
python scripts/train_gpt2.py --config configs/gpt2-124m-dytcd.yaml
```

The equivalent explicit-flag form, if you prefer not to use the configs:

```bash
python scripts/train_gpt2.py \
    --norm ln --no-layerscale \
    --max-steps 10000 --batch-size 32 --seq-len 1024 \
    --grad-accum 4 --lr 6e-4 --warmup 500 \
    --output-dir runs/gpt2-124m-ln
```

Resume an interrupted run with `--resume <ckpt>.pt`; the model, optimizer,
scheduler and step counter are all restored, so the LR schedule continues
where it stopped instead of replaying warmup. The schedule-defining flags
(`--max-steps`, `--warmup`, `--min-lr-ratio`, `--lr`, `--weight-decay`,
`--betas`, `--grad-clip`) are read back from the checkpoint as well -
without them the cosine curve would be silently reshaped mid-run, since
only `last_epoch` survives `load_state_dict`. Any that are overridden are
printed on startup.

Logged outputs in `<output-dir>/`:
- `train.jsonl`     - per-step loss / lr
- `tb/`             - TensorBoard scalars
- `ckpt_NNNNNN.pt`  - resumable checkpoints
- `stdout.log`      - raw stdout (for depth-sweep runs)

### Phase 2: LLaMA-tiny 1.1B with depth sweep

```bash
python scripts/train_llama_depth_sweep.py \
    --config configs/llama-1b-depth-sweep.yaml \
    --sequential
```

Each `(depth, norm)` combination runs in its own subprocess under
`runs/depth-sweep/d{depth}-{norm}/`. Depth and width are signalled via the
`ADASQUASH_DEPTH_OVERRIDE`, `ADASQUASH_DIM_OVERRIDE` and
`ADASQUASH_HEADS_OVERRIDE` env vars (head_dim is held at 128 for every
depth, and `dim` shrinks with depth so the parameter count stays ~constant
— otherwise the "curse of depth" axis is confounded with model size). The
`dim` schedule follows `dim ≈ base_dim · sqrt(base_depth / depth)`, giving
≈ 1.16 / 1.07 / 0.98 / 1.01 B parameters at L = 12 / 24 / 48 / 96, a
1.19× spread; the table is only used for the LLaMA-tiny base shape and any
other base model is rescaled from its own width. The same training script
can also be invoked directly for a single config:

```bash
ADASQUASH_DEPTH_OVERRIDE=48 \
ADASQUASH_DIM_OVERRIDE=1536 ADASQUASH_HEADS_OVERRIDE=12 \
    python scripts/train_gpt2.py \
    --model llama-1b --norm dytcd --max-steps 5000 ...
```

Pass `--no-dim-scaling` to the sweep launcher to keep `dim` fixed across
depths instead.

### Phase 3: special experiments

```bash
# Phase 3a: tiny MoE (8x0.3B) — expert load imbalance
python scripts/train_moe_tiny.py --norm dytcd --n-experts 8

# Phase 3b: FP8 stability (CPU/GPU simulation, no real FP8 hardware needed)
# Runs the real forward pass with both operands of every GEMM (SwiGLU MLP
# included) on FP8-quantized values and reports the logit divergence,
# per-norm overflow/underflow, the first block whose residual stream leaves
# the FP8 range and the first block that goes non-finite.
# Add `--ckpt <one per --norms entry>` to probe trained weights instead of
# a fresh random init.
python scripts/check_fp8_stability.py --norms ln rmsnorm dyt dytc dytd dytcd

# Phase 3c: post-training W8A8 PPL drop (quantizes the SwiGLU MLPs too)
# The probe corpus matters: on uniform random tokens both models sit at
# ppl ~= vocab_size and the drop is noise, so the default probe is the
# learnable synthetic corpus (--probe dataset uses real text instead).
python scripts/check_w8a8_ppl.py \
    --ckpts runs/gpt2-124m-ln/ckpt_final.pt \
            runs/gpt2-124m-rmsnorm/ckpt_final.pt \
            runs/gpt2-124m-dytcd/ckpt_final.pt

# Phase 3d: long-context needle-in-a-haystack
python scripts/run_needle.py --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
    --ctx-lens 512 1024 2048 4096
```

### Ablations

The ablation runner sweeps the variant matrix (1000 steps, GPT-2 124M) over a
synthetic Markov-chain corpus. The chain is an actual *walk*
(`s_{t+1} = (s_t + j_{t+1}) mod n_states`), so the target is a function of
the current state and the loss has a learnable floor well below
`log(vocab_size)` — uniformly random tokens have no next-token signal and
pin every variant at `log(vocab_size)`, which would make the loss column
unable to rank anything. A deliberately inflated embedding row supplies the
activation-outlier channel the squashing layers are meant to control.

```bash
python scripts/ablation.py --config configs/ablation-matrix.yaml
```

It writes a `summary.json` per ablation and an `all_summaries.json`
aggregating them (merged with any previous run, so a partial re-run does
not discard rows it skipped). Each summary carries the full loss curve,
the loss-spike count, and the pre-norm activation statistics pooled over
every sampled step of the run. Ablations include:

- Per-channel on/off
- Depth-conditioning on/off
- LayerScale init: 1e-5 / 1e-3 / off
- Softsign family: p ∈ {1, 2, 4, learnable, →∞ (=hard tanh)}
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

* p=1: classic softsign; p=2: smooth softsign (default); p→∞: hard tanh
  (i.e. a clip to [-1, 1] — note this is *not* tanh).
* All operations are strictly element-wise → no cross-token statistics
  and no collective ops. Kernel-friendly: trivially fuses with the
  surrounding matmul.
* One extra `[C]` vector per norm, typically <0.1% of the model.

### DyT-D — Variant B: depth-conditioned

```
α_l = α_0 · exp(−k · l / L)
y   = γ · tanh(α_l · x) + β
x_out = x + λ_l · Block(y)        # LayerScale-style residual gate
```

* α_l decays with depth: shallow layers squash strongly (bounding the
  largest pre-norm activations), while deep layers relax toward a
  near-linear, gradient-friendly response.
* Combined with the LayerScale residual gate, this targets the "curse of
  depth" in Pre-LN Transformers (Liu et al. 2020).

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
* Removes the activation magnitude from the logit entirely. `tanh` bounds
  every element of `Q'` / `K'` to ±1, so the dot product over `head_dim`
  terms is bounded by ±`head_dim` pre-scale, i.e. ±`sqrt(head_dim)` after the
  `1/sqrt(head_dim)` scale — a hard ceiling independent of how large the
  incoming activations are. That is what kills the logit-explosion failure
  mode that breaks long-context training.
* Sharper than QK-Norm for this purpose: RMSNorm rescales but does not
  saturate, so a single dominant channel still drives the logit, whereas
  `tanh` clips it.

## Metrics

All training scripts log:

| Metric              | What it captures                                   |
|---------------------|---------------------------------------------------|
| `loss`              | training cross-entropy                            |
| `n_loss_spikes`     | relative+absolute loss jumps over the run         |
| `max_abs_pre_norm`  | max \|activation\| entering each normalization    |
| `expert_cv`         | (MoE only) CoV of tokens-per-expert, per block    |
| `logit_rel_divergence` | (FP8 sim) max abs logit error vs fp32         |
| `overflow_rate_e4m3`| (FP8 sim) fraction of activations above ±448      |
| `underflow_rate_e4m3` | (FP8 sim) fraction of activations below min-normal |
| `first_overflow_layer` | (FP8 sim) first block whose residual leaves ±448 |
| `first_nonfinite_layer` | (FP8 sim) depth at which the residual breaks  |
| `ppl_drop_pct`      | (W8A8) PPL increase after post-training quantization |

A note on interpreting these: an experiment is only as good as its probe
corpus. Three failure modes are checked rather than assumed —
`check_w8a8_ppl.py` rejects a probe whose tokens are (near-)all distinct,
since nothing repeats and no model can beat chance on it; `check_fp8_stability.py`
only measures outlier channels on trained weights, so pass `--ckpt`; and
the ablation / MoE corpora are genuine Markov walks rather than uniform
draws.

## Citation

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

MIT.
