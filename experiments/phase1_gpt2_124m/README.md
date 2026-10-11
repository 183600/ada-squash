# Phase 1: GPT-2 124M on FineWeb-Edu (10B tokens)

**Goal:** establish the baseline PPL / stability / throughput comparison
between Pre-LN, RMSNorm, original DyT, and the three DyT++ variants
(DyT-C, DyT-D, DyT-CD).

## Configuration

| Setting           | Value                              |
|-------------------|------------------------------------|
| Model             | GPT-2 124M (12L, 768d, 12H, V=50304) |
| Data              | FineWeb-Edu sample-10BT           |
| Seq len           | 1024                               |
| Batch (effective) | 32 * 4 = 128 sequences            |
| Tokens / step     | 128 * 1024 = 131K                  |
| Steps             | 10000 (~1.3B tokens shown)         |
| Optimizer         | AdamW (β1=0.9, β2=0.95, wd=0.1)    |
| LR                | 6e-4 cosine, warmup 500            |
| Precision         | bf16                               |
| Hardware          | 1x A100 / H100                     |

## Runs

| Norm      | Output dir                    | Config file                       |
|-----------|-------------------------------|-----------------------------------|
| Pre-LN    | `runs/gpt2-124m-ln`           | `configs/gpt2-124m-ln.yaml`       |
| RMSNorm   | `runs/gpt2-124m-rmsnorm`      | `configs/gpt2-124m-rmsnorm.yaml`  |
| DyT       | `runs/gpt2-124m-dyt`           | `configs/gpt2-124m-dyt.yaml`      |
| DyT-C     | `runs/gpt2-124m-dytc`          | `configs/gpt2-124m-dytc.yaml`      |
| DyT-D     | `runs/gpt2-124m-dytd`          | `configs/gpt2-124m-dytd.yaml`      |
| DyT-CD    | `runs/gpt2-124m-dytcd`         | `configs/gpt2-124m-dytcd.yaml`    |

## Metrics

For each run we collect, every 500 steps:

* `loss`, `lr`, `tok_per_sec`, `n_spikes`
* per-norm `max_abs` and `kurtosis` (input distribution *before* the norm)
* aggregated `__summary__` block with mean / max

## Expected results (hypothesis)

* DyT-CD matches RMSNorm within ±0.02 PPL at 1.3B tokens.
* DyT-CD has 50%+ fewer loss spikes than RMSNorm.
* DyT-CD max |x| pre-norm is ≤ RMSNorm max |x| pre-norm.
* DyT-CD throughput ≥ RMSNorm throughput (no all-reduce on the norm
  reduction).

## Reproduce

```bash
for cfg in ln rmsnorm dyt dytc dytd dytcd; do
  python scripts/train_gpt2.py --config configs/gpt2-124m-${cfg}.yaml
done
```
