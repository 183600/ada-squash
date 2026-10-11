# Experiments

This directory holds configuration & result artifacts for the three
phases of the DyT++ study. Actual training output (checkpoints, logs,
TB events) should NOT be committed to git - see the `.gitignore`.

## Layout

```
experiments/
├── phase1_gpt2_124m/         # GPT-2 124M, FineWeb-Edu 10B
│   ├── README.md             # experiment card: what / why / when
│   └── results.jsonl         # aggregated final metrics (committed)
├── phase2_llama_1b_depth/    # LLaMA-tiny 1.1B, depth sweep
│   ├── README.md
│   └── results.jsonl
└── phase3_specials/          # MoE, FP8, W8A8, needle
    ├── README.md
    └── results.jsonl
```

Each experiment card should record:

* **Date run** - when the experiment was executed.
* **Config hash** - hash of the resolved config, for reproducibility.
* **Final loss / PPL** - mean over the last 100 steps.
* **n_spikes** - total loss-spike count over the run.
* **Throughput** - tokens/sec measured.
* **Stability stats** - mean/max kurtosis, max |x|, per-norm breakdown.
* **Hardware** - GPU type, count, peak memory.
* **Norm** - one of `ln` / `rmsnorm` / `dyt` / `dytc` / `dytd` / `dytcd`.
