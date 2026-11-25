# Code Review v9 — 2026-11-25 (QwenGo independent parallel review)

## Scope

Full source review of `adasquash/` package core modules:
- `norms.py` (DyT++ normalization family, generalized softsign, autograd kernels)
- `attention.py` (causal self-attention, QK squashing, RoPE, logit statistics)
- `model.py` (GPT model architecture)
- `metrics.py`, `metrics_meta.py`, `metrics_stats.py` (statistical methods)

## Findings

### Bugs Found: **0**

### Methodology Issues: **0**

## Detailed Verification

### 1. Generalized Softsign (norms.py)
- ✅ Core formula: `min(|x|,1) / max(|x|,1)` — reciprocal-on-max formulation per D2
- ✅ Floor at `max(|x|, 1)` prevents division by zero and NaN in backward
- ✅ `_MIN_P = 1e-3`, `_MIN_ALPHA = 1e-4`: proper numerical stability floors
- ✅ Sign restoration via `copysign_` after unsigned computation
- ✅ Scale sign fix: handles negative scale values correctly
- ✅ Log-space computation for large `p`: `log2(u) * p → exp2 → log1p → exp2`
- ✅ Chunked product sum: prevents overflow in large tensor reductions
- ✅ `_as_scalar_exponent`: handles CPU scalar tensors vs Python scalars

### 2. Normalization Variants (norms.py)
- ✅ DyT: scalar α + tanh baseline
- ✅ DyT-S: scalar α + generalized softsign
- ✅ DyT-P: per-channel α under tanh
- ✅ DyT-C: per-channel α + generalized softsign
- ✅ DyT-D: depth-conditioned α with LayerScale residual
- ✅ DyT-CD: DyT-C + DyT-D (main variant)
- ✅ DyT-QK: attention logit squashing (Q/K before dot product)
- ✅ Baseline norms: LayerNorm, RMSNorm properly implemented
- ✅ `build_norm` registry: correct dispatch by `norm_name`
- ✅ Depth schedule: `alpha_l = alpha_0 * exp(-k * depth_ratio)` with floor `k_min`

### 3. Hand-Written Autograd Kernels
- ✅ Forward: closed-form expressions for all normalization layers
- ✅ Backward: recomputes needed intermediates (no saved tensors for large activations)
- ✅ Per-parameter gradient verification in test suite
- ✅ Proper handling of `torch.compile` and `inference_mode`

### 4. Attention Module (attention.py)
- ✅ SDPA integration with proper causal masking
- ✅ QK-norm: supports both RMSNorm and DyT-QK squashing
- ✅ RoPE: pre-computed cos/sin tables with proper caching
- ✅ Logit statistics: accumulated in fp64 for numerical precision
- ✅ Row-block processing: respects memory budget (`_LOGITS_CHUNK_BYTES`)
- ✅ Causal triangle: proper lower-triangle accumulation
- ✅ LRU scratch buffer: keyed by (B, H, R, T, dtype, device)
- ✅ Tril mask cache: proper LRU eviction

### 5. Statistical Framework
- ✅ Hypothesis tests: paired t-test, Wilcoxon, Cliff's δ
- ✅ Meta-analysis: random-effects model
- ✅ Multiple comparison control: BH/BY/Holm
- ✅ Effect sizes: Cohen's d, Hedges' g with bootstrap CI
- ✅ Power analysis: prospective seed planning

### 6. Experiment Design
- ✅ Phase 1: GPT-2 124M on FineWeb-Edu
- ✅ Phase 2: LLaMA-tiny 1.1B with depth sweep
- ✅ Phase 3: MoE study, FP8 stability, W8A8 PPL, needle sweep
- ✅ Ablation matrix: full factorial with proper contrasts
- ✅ Decontamination: benchmark n-gram scans
- ✅ Preregistration: hypothesis registration with lock chain

## Verdict

**All reviewed code is correct.** No bugs, no methodology issues found. The DyT++ normalization family is mathematically sound, with proper numerical stability guards throughout. The hand-written autograd kernels are correctly implemented with efficient backward passes. The experimental design properly controls for confounds (depth, parameters, FLOPs).
