# DyT++ / AdaSquash

**English** | [中文说明](README_zh.md) | [设计说明](docs/design_notes.md)

> Adaptive Squashing without Statistics for Ultra-Deep and Low-Precision Transformers

This repository is the experimental codebase for the DyT++ proposal: a
family of *statistic-free* normalization layers that generalize
Dynamic Tanh (DyT, Zhu et al. 2025) along three orthogonal axes:

| Variant | Axis | What it adds | Targets |
|--------|------|---------------|---------|
| **DyT-S**  | activation family         | scalar α + generalized softsign (the {scalar × softsign} factorial cell) | saturation profile vs tanh |
| **DyT-P**  | per-channel (tanh)        | per-channel α under tanh (the {per-channel × tanh} cell) | isolates the per-channel axis from the family swap |
| **DyT-C**  | per-channel + softsign    | per-channel α + generalized softsign | outlier-heavy channels (LLaMA "super-active" dims) |
| **DyT-D**  | depth-conditioned         | depth-decaying α + LayerScale residual | the "curse of depth" in Pre-LN Transformers |
| **DyT-CD** | both (the main proposal)  | DyT-C + DyT-D combined                     | ultra-deep, MoE, FP8 |
| **DyT-QK** | attention logits (Variant C) | scalar squashing of Q/K before dot product | long-context logit explosion |

The core hypothesis: statistic-free element-wise adaptive squashing can
stand in for statistic-based normalization while adding **<0.1% additional
parameters**. For the QK variant the sharper, honest form of the question
is: *element-wise saturation vs vector normalization — which handles
outlier channels and attention expressivity better, at matched effective
logit scale?* (Both bound the logit identically; see the DyT-QK section.)

The `norm_name` values used by configs and scripts are `ln`, `rmsnorm`,
`dyt`, `dyts`, `dytp`, `dytc`, `dytd`, `dytcd`, and their controlled
ablation variants. QK squashing is selected separately through
`qk_norm: dytqk`.

## Contents

- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Phase 1: GPT-2 124M on FineWeb-Edu](#phase-1-gpt-2-124m-on-fineweb-edu)
- [Phase 2: LLaMA-tiny 1.1B with depth sweep](#phase-2-llama-tiny-11b-with-depth-sweep)
- [Phase 3: special experiments](#phase-3-special-experiments)
- [Ablations](#ablations)
- [Method reference](#method-reference)
- [Metrics](#metrics)
- [Citation](#citation)
- [License](#license)

## Repository layout

```
ada-squash/
├── adasquash/        # importable package
│   ├── norms.py        # LayerNorm, RMSNorm, DyT(-S/-P/-C/-D/-CD/-DI), QKSquash
│   ├── attention.py    # CausalSelfAttention with optional QK-squashing and RoPE
│   ├── model.py        # GPT-style decoder, fused losses, configurable depth
│   ├── utils.py        # ActivationStats, KurtosisMeter, LossSpikeCounter, ...
│   ├── diagnostics.py  # train/val curve diagnosis and NaN/Inf triage
│   ├── efficiency.py   # parameter/FLOP accounting and efficiency verdicts
│   ├── repro.py        # seeds, run manifests, execution identity
│   ├── design.py       # run schedules, exclusion policy, prospective power
│   ├── decontamination.py  # benchmark n-gram scans and contamination bounds
│   ├── metrics*.py     # hypothesis tests, meta-analysis, classification metrics
│   └── ablation/       # matrix, runner, contrasts, analysis and CLI
├── scripts/           # CLI entrypoints
│   ├── train_gpt2.py             # Phase 1 driver (GPT-2 124M, FineWeb-Edu, held-out val)
│   ├── train_llama_depth_sweep.py# Phase 2 depth sweep (12/24/48/96, incl. the dytdi control)
│   ├── eval_downstream.py       # lm-eval harness (custom LM wrapper), needle
│   ├── ablation.py              # single-variable ablation matrix runner, multi-seed
│   ├── train_moe_tiny.py        # Phase 3a: tiny MoE expert-variance study
│   ├── check_fp8_stability.py   # Phase 3b: simulated FP8 overflow / underflow (scoped)
│   ├── check_w8a8_ppl.py       # Phase 3c: post-training W8A8 PPL drop (held-out)
│   └── run_needle.py            # Phase 3d: needle-in-a-haystack sweep (per-position)
├── tests/             # full pytest suite: kernels, methods, design, decontamination
├── configs/           # YAML configs (see configs/)
├── docs/design_notes.md  # fixed design decisions D1-D6 and quality gate
├── ci_gate.py         # syntax + design-pin + full-pytest gate
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

These cover shape/dtype preservation, softsign boundedness, depth schedule
monotonicity, QK-squashing logit bounds, checkpoint round-trip, the metrics
implementation, and — since every norm here is a hand-written autograd kernel
— a per-parameter comparison of each kernel against an independent
recomposition in plain autograd, run from a *randomized* parameter set so the
check is not vacuous at the `gamma=1, beta=0` initialization.

### Design gate

The fixed design decisions are listed in
[`docs/design_notes.md`](docs/design_notes.md). Before trusting a change to
the kernels or the model, run the same gate used by CI:

```bash
python3 ci_gate.py
```

It parses every Python file, checks the `【固定】` design pins D1–D6 against
the source, and then runs the full `tests/` suite. `--syntax-only` runs just
the parser layer.

### Use the package directly

```python
import torch
from adasquash import GPT, GPTConfig, build_norm, generalized_softsign

# A single normalization layer.
norm = build_norm("dytcd", dim=128)
x = torch.randn(2, 8, 128)
print(norm(x).shape)                    # torch.Size([2, 8, 128])

# The public generalized-softsign entry point (p=2 is the smooth default).
print(generalized_softsign(torch.randn(4), 2.0))

# A small decoder with the same block/norm wiring as the experiments.
cfg = GPTConfig(vocab_size=256, dim=128, n_layers=2, n_heads=4, max_seq_len=16)
model = GPT(cfg)
tokens = torch.randint(0, cfg.vocab_size, (2, 16))
targets = torch.randint(0, cfg.vocab_size, (2, 16))
logits, loss = model(tokens, targets=targets)
print(logits.shape, float(loss.detach()))
```

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

# Factorial cells: scalar softsign / per-channel tanh
python scripts/train_gpt2.py --config configs/gpt2-124m-dyts.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-dytp.yaml

# Depth controls: init-only variants and a truly shared (alpha_0, k)
python scripts/train_gpt2.py --config configs/gpt2-124m-dytdi.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-dytcdi.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-dytd-shared.yaml

# LayerScale controls under the baselines
python scripts/train_gpt2.py --config configs/gpt2-124m-ln-ls.yaml
python scripts/train_gpt2.py --config configs/gpt2-124m-rmsnorm-ls.yaml

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
scheduler, step counter, **RNG states** (torch / CUDA / python, so dropout
continues as if uninterrupted) and **data position** (number of stream
batches already consumed) are all restored. By default the dataloader is
fast-forwarded past the consumed batches — same streaming order, so the
resumed run trains on *fresh* data instead of replaying the beginning of
the corpus; pass `--resume-skip-data` to skip the fast-forward (faster, but
the run replays training data — only acceptable when you do not compare
PPL/val curves across the resume). The schedule-defining flags
(`--max-steps`, `--warmup`, `--min-lr-ratio`, `--lr`, `--weight-decay`,
`--beta1` / `--beta2`, `--grad-clip`) are read back from the checkpoint as
well -
without them the cosine curve would be silently reshaped mid-run, since
only `last_epoch` survives `load_state_dict`. Any that the command line
leaves at its default are replaced silently; one you typed explicitly is
kept and reported, which is how a finished run is extended
(`--resume <ckpt> --max-steps <larger>` continues the cosine tail instead
of truncating it). The loss-spike counter and its EMA baseline are part of
the checkpoint, so `n_loss_spikes` keeps counting across a resume rather
than restarting. Checkpoints written before RNG/data-position tracking
say so loudly on resume instead of pretending to be exact continuations.

Logged outputs in `<output-dir>/`:
- `train.jsonl`     - per-step loss / lr / held-out val_loss
- `tb/`             - TensorBoard scalars (train/ and val/)
- `ckpt_NNNNNN.pt`  - resumable checkpoints (incl. RNG + data position)
- `stdout.log`      - raw stdout (for depth-sweep runs)

Every shipped phase-1 config enables held-out validation
(`val_every: 500` on wikitext-103-raw-v1 *test*, disjoint from the
FineWeb-Edu training stream), so a run's val curve is never computed on
training data.

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

The per-depth comparisons answer "is the candidate better at this depth".
`sweep_comparisons.json` also carries a cross-depth dose-response block for
the depth-conditioning claim itself: each seed's paired delta (candidate -
reference) is regressed on `log2(depth)`, and the per-seed slopes are tested
one-sided (H1: mean slope < 0, the advantage grows per doubling of depth),
Holm-corrected across candidate norms. Incomplete or cost-mismatched cells
make the trend descriptive only, and a leave-one-seed-out jackknife plus a
leave-one-depth-out check report whether one seed or one depth carries it. A
flat advantage - significant at every depth but
not growing - is evidence against the depth-conditioning mechanism, which a
row of separately significant per-depth deltas cannot distinguish from a
genuine dose-response.

### Phase 3: special experiments

```bash
# Phase 3a: tiny MoE (8x0.3B) — expert load imbalance
python scripts/train_moe_tiny.py --norm dytcd --n-experts 8

# Phase 3b: FP8 stability (CPU/GPU simulation, no real FP8 hardware needed)
# Runs the real forward pass with both operands of every GEMM (SwiGLU MLP
# included) on FP8-quantized values and reports the logit divergence,
# per-norm overflow/underflow/subnormal/quantized-to-zero rates, the first
# block whose residual stream leaves the FP8 range and the first block that
# goes non-finite.
# SCOPE: forward-only (no FP8 gradients or optimizer states), and with
# --scaling none (default) it is a NO-RESCALING STRESS TEST: real FP8
# recipes rescale each tensor by its amax, so run --scaling dynamic-tensor
# as well and compare before reading any of this as a training-stability
# prediction.
# Add `--ckpt <one per --norms entry>` to probe trained weights instead of
# a fresh random init.
python scripts/check_fp8_stability.py --norms ln rmsnorm dyt dyts dytp dytc dytd dytcd

# Phase 3c: post-training W8A8 PPL drop (quantizes the SwiGLU MLPs too)
# The probe corpus matters: on uniform random tokens both models sit at
# ppl ~= vocab_size and the drop is noise, so the default probe is the
# learnable synthetic corpus. --probe dataset reads a HELD-OUT corpus
# (default wikitext-103-raw-v1 test split, disjoint from the FineWeb-Edu
# training stream) - pointing it at the training data would make ppl_fp a
# partially-memorized number and the drop a comparison against a
# contaminated baseline. The report carries ppl_fp (unquantized baseline),
# ppl_w8a8, the absolute/pct drop, the eval-dataset identity and the token
# count the numbers are computed over.
python scripts/check_w8a8_ppl.py \
    --ckpts runs/gpt2-124m-ln/ckpt_final.pt \
            runs/gpt2-124m-rmsnorm/ckpt_final.pt \
            runs/gpt2-124m-dytcd/ckpt_final.pt

# Phase 3d: long-context needle-in-a-haystack
# Sweeps several needles (different secrets and phrasings) x context
# lengths, and reports accuracy by needle POSITION tercile (early / middle /
# late) - a model that only recalls shallow needles looks identical to one
# that recalls everywhere in the scalar average.
python scripts/run_needle.py --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
    --ctx-lens 512 1024 2048 4096
```

### Ablations

The ablation runner sweeps a **single-variable** matrix (1000 steps, GPT-2
124M) over a synthetic Markov-chain corpus. The chain is an actual *walk*
(`s_{t+1} = (s_t + j_{t+1}) mod n_states`), so the target is a function of
the current state and the loss has a learnable floor well below
`log(vocab_size)` — uniformly random tokens have no next-token signal and
pin every variant at `log(vocab_size)`, which would make the loss column
unable to rank anything.

The activation-outlier probe is a **persistent residual-stream channel
amplification after the middle block** (a forward hook scaling one channel
by 25×; see `install_residual_outlier`). The old probe inflated a token
embedding row instead — but with tied embeddings that row is also a row of
the output classifier, so it confounded "handle an input-side outlier
channel" with "relearn to never predict a token", an output-side pathology
no normalization layer has to fix. The hook touches no weights and is
applied identically to every row, so it is a constant of the comparison.

```bash
# single seed
python scripts/ablation.py --config configs/ablation-matrix.yaml
# three seeds per row -> per-row mean/std in <out>/<name>/summary.json
python scripts/ablation.py --config configs/ablation-matrix.yaml --seeds 42 43 44
```

It writes a `summary.json` per ablation (per-seed under
`<out>/<name>/seed<N>/` when `--seeds` is given) and an `all_summaries.json`
aggregating them (merged with any previous run, so a partial re-run does
not discard rows it skipped). Each summary carries the full loss curve,
the loss-spike count, the outlier-probe description and the pre-norm
activation statistics pooled over every sampled step of the run. Every
row of `configs/ablation-matrix.yaml` documents its reference row and the
single axis it changes; the families:

- LayerScale ± under LN / RMSNorm / DyT-CD (the old plan's LN(no-LS) →
  dytcd(LS-on) baseline bundled this axis with the norm swap)
- Activation family × per-channel factorial: `dyt` / `dyts` / `dytp` /
  `dytc` (each adjacent pair isolates exactly one axis), LS off
- Depth axis at matched LS: `dyt_ls` → `dytd` (add reparameterization) →
  `dytdi` (same init, no reparameterization) / `dytd_shared` (one `(α_0, k)`
  for the whole model) / `dytd_k0` (schedule off)
- Softsign exponent p: 1 / 2 / 4 / learnable
- LayerScale init: 1e-5 / 1e-3 / off under DyT-CD

Before training, the runner writes `analysis_plan.json`: the set of
confirmatory contrasts is derived from the validated matrix, every contrast
gets a stable id and its stated hypothesis, and the whole registry is hashed.
After the four comparison reports it writes `study_level_tests.json`, which
applies hierarchical Holm gatekeeping across the pre-registered contrasts
(primary: the main proposal and its mechanisms; secondary: the remaining
single-axis, control and QK contrasts, tested only if a primary contrast
survives). Tests outside the plan are reported as exploratory. A rerun that
changes the plan fingerprint is recorded as post-hoc (`plan_changed`); pass
`--require-plan-lock` to reject such a rerun outright. Each contrast
also reports a leave-one-seed-out jackknife (does the sign survive dropping
any one seed?) and, with `--equivalence-margin <delta>`, a TOST equivalence
test, so "these two arms are the same" is a claim with a test rather than a
non-significant superiority result. Factorial interactions are reported
two-sided, so an antagonistic interaction is not mistaken for a failed
synergy claim.

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

**What the parameterization is (and is not).** With the default
configuration each norm owns an independent `(α_0, k)` pair, and a single
layer only ever *observes* the product `α_0 · exp(−k·l/L)` — one effective
scalar. The depth-decaying shape is therefore a **depth-dependent
initialization prior** for each layer's α, not an emergent cross-layer law:
after training, the per-layer `α_l` values are free to drift away from any
common curve, and the honest reading of a learned `k` is "the decay rate this
layer's α started from". Two mechanisms make the claim testable instead of
rhetorical:

* `--k-min` (default `0.0`) floors the decay rate, so a layer's `α_l` cannot
  *grow* with depth via `k < 0` — the schedule stays non-increasing wherever
  the optimizer takes it (`--no-k-min` restores the old unconstrained `k`).
* `--shared-depth-schedule` replaces the per-layer pairs with **one
  `(α_0, k)` for the whole model** (2 scalars total). Now the pair is
  *identifiable*: every layer evaluates `α_l = α_0·exp(−k·l/L)` at its own
  depth against the same parameters, and the learned curve is a genuine
  cross-layer law rather than an init.

The `dytdi` / `dytcdi` rows of the ablation matrix are the matching controls:
plain per-layer α *initialized* at the schedule value and then learned freely.
`dytd` vs `dytdi` isolates "the reparameterization" from "the init prior";
if `dytd` only wins through a better-conditioned start, `dytdi` matches it.

* Combined with the LayerScale residual gate, this targets the "curse of
  depth" in Pre-LN Transformers (Liu et al. 2020).

### DyT-CD — main proposal (A + B combined)

```
α_c,l = α_0 · exp(−k · l / L) · (1 + δ_c)
x_hat = α_c,l · x
φ    = x_hat / (1 + |x_hat|^p)^{1/p}
y    = γ · φ + β
```

* δ_c is initialized to **zero**, so the per-channel factor starts at
  `α_c,l = α_l` (no per-channel deviation). That makes a trained DyT-D
  checkpoint a usable *warm start* for the shared `α_0`, `k`, `γ`, `β`,
  but the two layers are **not** functionally identical: DyT-D squashes
  with `tanh`, while DyT-CD squashes with the generalized softsign and a
  learnable `p`. The transfer is therefore not a lossless "free upgrade",
  and the `dytd → dytcd` jump crosses *two* axes (per-channel α and the
  activation family) — the matrix decomposes it as
  `dytd → dytcdi` (both axes, init only) and `dytcdi → dytcd`
  (add the reparameterization), so each step is again single-variable.

### DyT-QK — Variant C: attention logit squashing

```
Q' = tanh(α_q · Q);  K' = tanh(α_k · K)
logits = (Q' · K') · scale
```

* Replaces QK-Norm (which requires two RMSNorms of size `head_dim`) with
  two scalars per layer, and the op is element-wise (no collectives).

* **Bounded logits — but so is QK-Norm.** `tanh` bounds every element of
  `Q'`/`K'` to ±1, so the dot product over `head_dim` terms is bounded by
  `±head_dim` pre-scale, i.e. `±sqrt(head_dim)` after the `1/sqrt(head_dim)`
  scale. But RMSNorm has the *same* ceiling: an RMS-normalized vector has
  L2 norm exactly `sqrt(head_dim)`, so by Cauchy-Schwarz
  `|q'ᵀk'| ≤ ||q'||·||k'|| = head_dim` pre-scale — identical
  `±sqrt(head_dim)` after scaling. The hard ceiling is therefore **not** a
  differentiator between the two; what actually differs is the *shape* of
  the bounded vectors:

  | | tanh squash | QK-RMSNorm |
  |---|---|---|
  | scale invariance | no — sub-saturated magnitudes pass through (`tanh(z) ≈ z` for small `z`), only large values are capped | yes — a 10× larger token vector yields a *bit-identical* normalized vector |
  | single dominant channel | the outlier saturates to ±1; the small channels keep their relative sizes | every channel is divided by the outlier-inflated RMS, so the small channels are crushed relative to it |
  | what the logit can express | magnitude *and* direction of q·k (more expressive, more exposed) | direction only (per-token norm removed exactly) |

* "Removes the activation magnitude from the logit" would be the claim for
  QK-Norm, **not** for `tanh`: saturation caps large magnitudes, but a
  medium activation passes through and still scales the logit. The honest
  framing of Variant C is a *trade-off*: element-wise saturation (cheap,
  compile-friendly, keeps some magnitude information) vs vector
  normalization (exactly scale-invariant, costlier). Which one matters more
  for long-context training is an open question this repo measures rather
  than asserts.

* **Fair comparison requires matched effective logit scale.** An
  RMS-normalized vector has L2 norm `sqrt(head_dim)`; a tanh-squashed one
  has L2 norm *at most* `sqrt(head_dim)` and typically much less (only
  fully-saturated vectors reach it). At the same `α`, tanh logits are
  systematically smaller — so any tanh-vs-QKNorm comparison must either
  calibrate `α` or sweep a temperature on both sides before attributing
  differences to the squashing shape itself.

## Metrics

All training scripts log:

| Metric              | What it captures                                   |
|---------------------|---------------------------------------------------|
| `loss`              | training cross-entropy                            |
| `val_loss`          | held-out cross-entropy (wikitext-103 test in the shipped configs) |
| `n_loss_spikes`     | relative+absolute loss jumps over the run         |
| `max_abs_pre_norm`  | max \|activation\| entering each normalization    |
| `expert_cv`         | (MoE only) CoV of tokens-per-expert, per block    |
| `logit_rel_divergence` | (FP8 sim) max abs logit error vs fp32         |
| `overflow_rate_e4m3`| (FP8 sim) fraction of activations above ±448      |
| `underflow_rate_e4m3` | (FP8 sim) fraction of activations below min-normal (small-value share) |
| `subnormal_rate_e4m3` | (FP8 sim) fraction in (0, min-normal) — representable only as subnormals |
| `quantized_zero_rate_e4m3` | (FP8 sim) fraction of nonzero activations that quantize to exactly 0 |
| `rel_quant_err_e4m3` | (FP8 sim) mean relative quantization error      |
| `first_overflow_layer` | (FP8 sim) first block whose residual leaves ±448 |
| `first_nonfinite_layer` | (FP8 sim) depth at which the residual breaks  |
| `ppl_fp`            | (W8A8) unquantized baseline PPL on the held-out probe |
| `ppl_w8a8`          | (W8A8) PPL after post-training quantization        |
| `ppl_drop_pct`      | (W8A8) PPL increase after post-training quantization |

A note on interpreting these: an experiment is only as good as its probe
corpus. Four failure modes are checked rather than assumed —
`check_w8a8_ppl.py` rejects a probe whose tokens are (near-)all distinct,
since nothing repeats and no model can beat chance on it, and defaults its
`--probe dataset` path to a held-out corpus (wikitext-103 test) rather than
the training stream; `check_fp8_stability.py` only measures outlier channels
on trained weights, so pass `--ckpt`, and labels its own output as a
forward-only, no-rescaling stress test; the ablation / MoE corpora are
genuine Markov walks rather than uniform draws; and the needle sweep
reports accuracy by insertion position, not just the scalar average.

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
