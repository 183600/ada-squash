# Phase 2: LLaMA-tiny 1.1B with depth sweep + downstream eval

**Goal:** validate the DyT-CD story at 1B scale and demonstrate the
depth-scaling advantage over RMSNorm and original DyT.

## Depth sweep

We train each norm at four depths: 12, 24, 48, 96 layers, keeping the
parameter count approximately fixed by scaling `dim` to keep FLOPs
roughly constant.

| Depth | dim   | Heads | MLP ratio | ~Params |
|-------|-------|-------|----------|---------|
| 12    | 2560  | 20    | 8/3      | 0.95B   |
| 24    | 2048  | 16    | 8/3      | 1.07B   |
| 48    | 1536  | 12    | 8/3      | 1.06B   |
| 96    | 1280  | 10    | 8/3      | 1.10B   |

The exact dim choices are tunable; the key constraint is keeping
the embedding/head ratio consistent so RoPE / attention shapes don't
break. `scripts/train_llama_depth_sweep.py` encodes this table
(`DEPTH_SHAPE`) and exports it to the trainer as
`ADASQUASH_DEPTH_OVERRIDE` / `ADASQUASH_DIM_OVERRIDE` /
`ADASQUASH_HEADS_OVERRIDE`; head_dim is 128 at every depth so only depth
varies. Pass `--no-dim-scaling` to opt out (that makes the 96-layer run a
~4x larger model and the depth axis no longer isolated).

## Recipe

* Data: FineWeb-Edu (10B tokens) + 26B-token cooldown on FineWeb
* Steps: 5k warmup-only + 100k main + 26k cooldown
* LR: 3e-4 cosine, warmup 500, min_lr_ratio 0.1
* bf16, 4x H100 (DistributedDataParallel, no FSDP needed at 1B)
* Batch: 8 per device * 8 grad-accum * 4 devices = 256 sequences
* Tokens / step: 256 * 2048 = 524K

## Expected results (hypothesis)

* RMSNorm diverges (NaN) at 96 layers under the same LR / warmup that
  works at 12 layers - the "curse of depth".
* DyT-D and DyT-CD train stably to 96 layers *without* LR retuning,
  because the depth-conditioned α_l + LayerScale combination bounds the
  residual stream.
* Original DyT diverges somewhere between 48 and 96 layers because its
  single global α can't satisfy both shallow and deep layers.

## Downstream tasks

After each Phase 2 run, we evaluate:

| Task       | Metric      | Shots | Max tokens |
|------------|-------------|-------|-----------|
| MMLU       | acc         | 5     | 2k        |
| GSM8K       | acc (maj@1) | 0     | 2k        |
| HumanEval  | pass@1      | 0     | 1k        |
| DROP        | F1         | 3     | 4k        |
| HellaSwag  | acc        | 0     | 1k        |

## Reproduce

```bash
# depth sweep
python scripts/train_llama_depth_sweep.py \
    --model llama-1b \
    --depths 12 24 48 96 \
    --norms rmsnorm dyt dytcd \
    --seq-len 2048 --batch-size 8 --grad-accum 8 \
    --max-steps 5000 --lr 3e-4 --warmup 500 --weight-decay 0.1 \
    --sequential \
    --output-dir runs/depth-sweep

# downstream eval per run
for d in 12 24 48 96; do
  for norm in rmsnorm dyt dytcd; do
    python scripts/eval_downstream.py \
        --ckpt runs/depth-sweep/d${d}-${norm}/ckpt_final.pt \
        --tasks mmlu gsm8k hellaswag
  done
done
```
