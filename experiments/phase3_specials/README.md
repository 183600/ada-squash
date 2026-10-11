# Phase 3: Specials — MoE, FP8, W8A8, long-context

These experiments are the "differentiation" suite from the proposal:
*the same DyT-CD layer that wins on PPL must also win here* to
substantiate the "ultra-deep + low-precision + MoE" story.

## 3a. MoE-tiny (8x0.3B)

**Hypothesis:** because DyT-CD is purely element-wise (no
per-token mean / var), the *per-expert variance drift* problem that
breaks RMSNorm in MoE training should be much milder.

**Recipe:**

* 8 experts, top-2 routing, aux-loss coefficient 1e-3
* dim=512, n_layers=8, n_heads=8, seq_len=1024
* 1000 synthetic-data steps (cheap proxy)
* Track `expert_cv` (CoV of tokens-per-expert)

**Reproduce:**

```bash
for norm in rmsnorm dyt dytcd; do
  python scripts/train_moe_tiny.py --norm ${norm} \
      --output-dir runs/moe-tiny-${norm}
done
```

## 3b. FP8 stability check

**Hypothesis:** DyT-CD activations are bounded by construction
(|φ_p(x)| ≤ 1, scaled by γ), so they should have a *strictly lower*
FP8-E4M3 overflow rate than RMSNorm activations at the same
training step.

We run a CPU/GPU simulation of FP8-E4M3 (no real FP8 hardware
required).

**Reproduce:**

```bash
python scripts/check_fp8_stability.py \
    --norms ln rmsnorm dyt dytc dytd dytcd \
    --output runs/fp8_check/results.json
```

## 3c. Post-training W8A8 PPL drop

**Hypothesis:** naive W8A8 round-to-nearest quantization should produce
*less* PPL drop on DyT-CD checkpoints than on RMSNorm / LayerNorm ones.

**Reproduce:**

```bash
python scripts/check_w8a8_ppl.py \
    --ckpts runs/gpt2-124m-ln/ckpt_final.pt \
            runs/gpt2-124m-rmsnorm/ckpt_final.pt \
            runs/gpt2-124m-dytcd/ckpt_final.pt
```

## 3d. Long-context needle-in-a-haystack

**Hypothesis:** DyT-QK (Variant C, scalar squashing of Q and K) bounds
attention logits to ±1, which kills the logit-explosion failure mode
that breaks long-context training even when RoPE scaling is applied.

We sweep context lengths {512, 1024, 2048, 4096, 8192} and measure
greedy-recall accuracy of a planted secret.

**Reproduce:**

```bash
python scripts/run_needle.py --ckpt runs/gpt2-124m-dytcd/ckpt_final.pt \
    --ctx-lens 512 1024 2048 4096 8192
```

To enable DyT-QK, re-train with `--qk-norm dytqk`:

```bash
python scripts/train_gpt2.py --norm dytcd --qk-norm dytqk \
    --output-dir runs/gpt2-124m-dytcd-qk
```
