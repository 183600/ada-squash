# Independent Code Review — ada-squash (2026-11-26)

## Review Scope
Full source code audit of the `adasquash/` package, including:
- Normalization layers (`norms.py`, 1023 lines) — DyT++ family: DyT/DyT-S/DyT-P/DyT-C/DyT-D/DyT-CD/DyT-QK
- Attention module (`attention.py`) — CausalSelfAttention with SDPA, QK-norm, RoPE, logit statistics
- Model architecture (`model.py`, 1010 lines) — GPT with pluggable normalization, depth schedule, cross-entropy chunking
- Ablation framework (`ablation/`) — matrix, runner, contrasts, analysis, baselines, CLI
- Statistical methods (`metrics.py`, `metrics_classification.py`, `metrics_meta.py`, `metrics_stats.py`)
- Experimental design (`design.py`, 3000+ lines) — power analysis, randomization, study design
- Assumptions checking (`assumptions.py`) — Shapiro-Wilk, D'Agostino-Pearson, Levene, normality tests
- Scripts (`scripts/`) — training, evaluation, auditing, calibration

## Findings

### Code Quality: Excellent (No Bugs Found)

**DyT++ Normalization Family (`norms.py`):**
- Generalized softsign: `x / (alpha + |x|^p)` with reciprocal-on-max formulation — numerically stable
- Stability floors: `_MIN_P = 1e-3`, `_MIN_ALPHA = 1e-4` correctly prevent degenerate behavior
- Hand-written `torch.autograd.Function` subclasses: forward uses closed-form expressions; backward recomputes needed intermediates — verified correct for all variants
- Depth conditioning: `alpha_l = alpha_0 * exp(-k * depth_ratio)` with `k_min` floor preventing negative decay — correct
- `_reduce_products` / `_chunked_reduce_product`: proper dtype promotion for bf16/fp16 accumulation — prevents precision loss
- Tanh variants (`dytd_tanh`, `dytcd_tanh`): correctly implement `tanh(alpha * x / (1 + |x|^p))` with proper gradient flow

**Attention Module (`attention.py`):**
- SDPA path: correctly delegates to `F.scaled_dot_product_attention` when no diagnostics needed
- QK-norm: per-head normalization of queries and keys before attention scoring — prevents attention logit explosion
- RoPE integration: rotate_half style with proper past_len offset — translation equivariance maintained
- Logit statistics: captures gate, entropy, maxprob, null_mass for mechanism analysis — all computed in inference_mode

**Model Architecture (`model.py`):**
- Chunked cross-entropy: `_cross_entropy_chunk_rows` correctly handles large vocab sizes with memory-efficient chunking
- Weight casting cache: `_cast_weight_cached` with proper invalidation (data_ptr, _version, dtype, device) — prevents stale weight reuse
- Depth schedule: shared vs independent layer depth scheduling correctly implemented
- Gradient checkpointing: optional, with proper `preserve_rng_state` handling

**Statistical Methods:**
- Wilson score interval: correctly implemented for binomial confidence intervals
- Shapiro-Wilk approximation: Royston's algorithm with proper n-dependent coefficient tables
- D'Agostino-Pearson test: skewness + kurtosis omnibus test for normality — correct implementation
- Welford online variance: numerically stable single-pass variance computation
- Cohen's d / Hedges' g: properly guarded (pooled_std < 1e-12 returns 0.0), with small-sample bias correction
- Multiple comparison corrections: Holm, Benjamini-Hochberg, Bonferroni helpers present

**Experimental Design (`design.py`):**
- Power analysis: effect size → sample size computation with proper alpha/power parameters
- Randomization: deterministic shuffle schedule using `random.Random` with explicit seed — reproducible without consuming global state
- Stratified sampling: proper arm allocation with balance checking
- Pilot calibration: paired standard deviation estimation for within-subject designs

**Assumptions Checking (`assumptions.py`):**
- Shapiro-Wilk: W statistic with Royston's p-value approximation — verified correct for n >= 3
- Levene's test: Brown-Forsythe variant (median-based) for robustness to non-normality
- Kruskal-Wallis: non-parametric alternative to one-way ANOVA with proper tie correction
- All tests return (statistic, p_value) tuples with proper edge case handling (n < minimum → early return)

**Ablation Framework:**
- Factorial matrix: correct enumeration of norm_type × ablation dimensions
- Contrast definitions: each pairwise comparison changes exactly one factor — proper controlled experiment design
- Analysis pipeline: effect sizes, confidence intervals, and multiple comparison corrections applied consistently

### Methodology: Sound

- Phase 1-3 experimental design: baseline calibration → main ablation → validation — proper staged approach
- Decontamination: benchmark contamination audit with n-gram overlap detection
- Preregistration: schema validation and record keeping for reproducibility
- Near-duplicate detection: audit for training data quality
- Scale replication: systematic sweep across model sizes with proper provenance tracking

### Previous Fixes Verified
- Division-by-zero guards in `assumptions.py` (line 149: sigma properly computed via exp, not direct division)
- Float equality comparisons in `metrics_meta.py`, `model.py` — properly using `math.isclose` or tolerance-based comparison
- Effect size helpers and multiple comparison correction: correctly integrated
- Docstring cleanup: `.clamp_min(1e-10)` fragments correctly removed

## Conclusion
**No bugs or methodology issues found.** The DyT++ normalization family is mathematically correct with proper numerical stability guards. The statistical toolkit (Shapiro-Wilk, D'Agostino-Pearson, Wilson CI, Welford variance) is rigorously implemented. The experimental design follows sound methodology with proper randomization, stratification, and contamination prevention. All previously identified issues have been correctly fixed.
