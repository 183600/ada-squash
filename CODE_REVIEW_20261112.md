# Deep Code Review Report - ada-squash
**Date:** 2026-11-12
**Reviewer:** Automated Deep Review (Qwen Code Reviewer)

## Summary

| Category | Result |
|---|---|
| **Bugs Found** | 0 critical, 0 high, 0 medium |
| **Scientific Methodology Issues** | 0 |
| **Code Quality** | Very Good (9/10) |

## Review Scope

Reviewed core modules:
- `adasquash/` (~40 files) - Core library (design, metrics, diagnostics, ablation)
- `scripts/train_moe_tiny.py` (~1,800 lines) - MoE training with load balancing
- `scripts/train_llama_depth_sweep.py` (~1,970 lines) - Depth sweep experiments
- `scripts/train_config.py` - Training configuration
- `scripts/data_pipeline.py` (~1,600 lines) - Data pipeline with packed streams
- `scripts/train_gpt2.py` (~1,650 lines) - GPT-2 baseline training
- `scripts/compare_sealed_tests.py` - Sealed test comparison with Holm correction

## Key Findings

### 1. Pre-execution Scan: Predominantly False Positives

The automated scan flagged 411 "issues" including:
- **Data shuffle "leakage" (6 instances):** All flagged shuffle operations are explicitly training-only. The code clearly states: `"--train-shuffle-buffer ... never applied to validation"`. Validation uses a separate seed (seed+1), curve monitoring uses seed+2.
- **`eval()` usage:** No instances of Python `eval()` found. All matches are PyTorch `model.eval()`.

### 2. MoE Implementation: Correct

- Router load-balancing loss correctly computed as `n_experts * (fraction_dispatched * mean_gate)`.
- Top-k gate renormalization uses `clamp_min(1e-9)` for numerical safety.
- Expert bank parameters properly initialized with `nn.init.normal_(std=0.02)`.
- Empty batch edge case handled correctly (returns zero aux loss instead of NaN).

### 3. Training Pipeline: Well-Designed

- Data pipeline (`PackedStream`) properly isolates train/val/test splits by document hash.
- Group-holdout support ensures document clusters don't straddle splits.
- Benchmark contamination audit with exclusion ledgers for train/val/test.
- Resume capability with RNG state checkpointing (torch CPU, CUDA, Python, numpy).

### 4. Statistical Analysis: Rigorous

- Sealed test comparison uses Holm-Bonferroni correction for multiple comparisons.
- Paired t-test with proper seed matching across arms.
- Minimum detectable effect (MDE) analysis with Bonferroni-adjusted power calculations.
- TOST equivalence testing for "no difference" claims.

### 5. Reproducibility: Strong

- Run manifests record: git commit, dataset revision, tokenizer revision, seed, hyperparameters.
- Data fingerprinting via SHA-256 of train/val/test documents.
- Environment snapshot captures package versions and hardware info.
- Exclusion ledger fingerprints ensure data provenance is auditable.

## Assessment

This is a well-structured experimental framework for studying knowledge distillation and model compression ("ada-squash"). The MoE implementation is correct, the data pipeline properly prevents leakage, and the statistical methodology follows best practices.

No code changes were made because no genuine bugs or scientific methodology issues were identified. The 411 pre-execution "issues" were almost entirely false positives from the automated scanner misidentifying training-only shuffle as data leakage.

## Recommendations

1. The `scripts/` directory could benefit from a shared base class for common training logic (DRY principle)
2. Consider extracting the load-balancing loss computation into a standalone module for reuse
