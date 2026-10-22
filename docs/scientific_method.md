# Scientific-method checks (non-fixed companion to `design_notes.md`)

`docs/design_notes.md` fixes the model semantics. This note describes the
experimental-method checks layered around those fixed kernels; it does not
change D1–D6.

## 1. Controlled comparisons

The ablation runner holds the optimizer (AdamW, betas 0.9/0.95), learning
rate, schedule, step count, batch/sequence shape, synthetic data generator and
seed set constant for every row. A row that tries to set one of those keys is
rejected by `scripts/ablation.py` before training. Phase-1 `aggregate.json`
files are compared by `scripts/compare_arms.py`, which verifies the recorded
`run_manifest.json` training recipe (optimizer/schedule/token budget) as well
as model shape, final step and resolved data version. Only the normalization
may differ; a missing recipe is treated as unknown and cannot certify a win.

## 2. Ablation matrix

`scripts/ablation.py --check-matrix` validates:

* the main proposal's softsign 2×2 factorial:
  `neither = dytcd_no_perchannel_no_depth`, `C only =
  dytcd_k0_no_depth_schedule`, `D only = dytcd_no_perchannel`, `CD =
  dytcd_full`;
* the tanh control 2×2 factorial (`dytd_tanh_k0`,
  `dytc_perchannel_tanh_k0`, `dytd_tanh_full`, `dytcd_no_softsign`);
* the leave-one-out set for per-channel, depth, LayerScale, activation
  family, learnable `p`, and every declared QK contrast;
* the independent QK axis: QK off/on on LN, RMSNorm and DyT-CD, plus the
  QK-Norm comparator on each host. Only `qk_norm` may differ.

`factorial_comparison.json` reports main effects and the interaction;
`qk_axis_comparison.json` reports per-host QK effects and whether their sign
replicates across hosts.

## 3. Replication and statistics

`seeds` is the unit of replication. The default is three independent runs and
`MIN_SEEDS_FOR_SIGNIFICANCE = 3`; fewer than three shared seeds are reported
descriptively and can never certify a result. Every comparison reports
Mean ± Std, a one-sided paired t-test, a 95% CI, per-seed direction counts,
the minimum detectable delta at 80% power, and a Holm-adjusted p-value across
the family.

`scripts/compare_arms.py` adds the Phase-1 scale check: `delta/seed_std` and
`within_seed_noise` put the loss/PPL gap next to the run-to-run spread, and
`dPPL` is reported alongside the loss test. The ablation baseline comparison
now carries the same fields.

## 4. Learning-rate fairness

`scripts/tune.py` runs every arm over the same `(lr, weight_decay)` grid and
adds `lr_sensitivity`: the per-arm spread across complete cells relative to
its seed std, and the cross-arm winner at each shared cell. If the winner
changes with `lr`, a fixed-lr norm win is reported as an LR-bonus result, not
a mechanism effect.

## 5. Multi-scale replication

FP8 random-init probing accepts `--models` (GPT-2 124M/350M, LLaMA-tiny 1B);
W8A8 comparisons may include arm pairs from more than one checkpoint shape;
`scripts/moe_scale_sweep.py` runs the MoE study at several
`dim/depth/heads/experts` scales. Each comparison writes a
`scale_replication` block, and `scripts/check_scale_replication.py` combines
per-scale artifacts. A direction is *replicated* only when at least two
distinct scales agree in sign, and *certified across scales* only when at
least two scales individually pass their own test. One scale is a
scope-limited result, never a mechanism claim.
