# Phase 1: GPT-2 124M on FineWeb-Edu (10B tokens)

**Goal:** establish the baseline PPL comparison
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


## Runs

| Norm      | Output dir                    | Config file                       |
|-----------|-------------------------------|-----------------------------------|
| Pre-LN    | `runs/gpt2-124m-ln`           | `configs/gpt2-124m-ln.yaml`       |
| RMSNorm   | `runs/gpt2-124m-rmsnorm`      | `configs/gpt2-124m-rmsnorm.yaml`  |
| DyT       | `runs/gpt2-124m-dyt`           | `configs/gpt2-124m-dyt.yaml`      |
| DyT-C     | `runs/gpt2-124m-dytc`          | `configs/gpt2-124m-dytc.yaml`      |
| DyT-D     | `runs/gpt2-124m-dytd`          | `configs/gpt2-124m-dytd.yaml`      |
| DyT-CD    | `runs/gpt2-124m-dytcd`         | `configs/gpt2-124m-dytcd.yaml`    |

## Expected results (hypothesis)

* DyT-CD matches RMSNorm within ±0.02 PPL at 1.3B tokens.

## Reproduce

`configs/gpt2-124m-<norm>.yaml` records the resolved hyper-parameters for
each run; pass them to the driver as flags:

```bash
for norm in ln rmsnorm dyt dytc dytd dytcd; do
  python scripts/train_gpt2.py --norm ${norm} \
      --max-steps 10000 --batch-size 32 --seq-len 1024 \
      --grad-accum 4 --lr 6e-4 --warmup 500 --min-lr-ratio 0.1 \
      --weight-decay 0.1 --beta1 0.9 --beta2 0.95 --grad-clip 1.0 \
      --output-dir runs/gpt2-124m-${norm}
done
```

`--no-layerscale` reproduces the `use_layerscale: false` rows (ln, rmsnorm,
dyt, dytc); omit it for the dytd / dytcd rows.
