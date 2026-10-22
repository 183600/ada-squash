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
replicates across hosts. Every factorial term also carries a **two-sided**
interaction test alongside the one-sided synergy test: an antagonistic
interaction (the combination is worse than the sum of its parts) is a
finding, not a failed synergy claim, so it is reported and is eligible for
the study-level correction as a two-sided hypothesis.

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

Two robustness layers sit next to the significance test and never replace it:

* **Leave-one-seed-out jackknife.** Every paired contrast reports what happens
  to the sign of the effect when each seed is dropped in turn. A sign that
  flips identifies the seed carrying the result (`SEED-SENSITIVE`), even if
  the full-sample t-test is significant. With a zero full-sample mean there is
  no direction to stabilise, so the block reports `sign_stable = null` rather
  than a misleading boolean.
* **Equivalence (TOST), not "not significant".** A non-significant superiority
  test is not evidence that two arms are the same. Pass
  `--equivalence-margin <val_loss>` to run two one-sided paired t-tests whose
  alternative is "the mean difference is inside ±margin"; the report records
  the margin, `p_tost`, the matching `1 - 2α` interval and `equivalent`. The
  margin is a scientific smallest effect of interest fixed before the run, not
  a knob to tune until the result looks equivalent.

## 4. Pre-registered analysis plan and study-level multiplicity

Each family (baseline, leave-one-out, factorial, QK) is corrected within
itself, but the study makes all of those claims at once and reuses the same
seeds. `scripts/ablation.py` therefore writes `analysis_plan.json` **before
training**: every confirmatory contrast is derived from the validated matrix,
carries a stable `id`, its stated hypothesis, the test direction and a
SHA-256 fingerprint of the whole registry (rows, hypotheses, seeds, α, margins).
Contrasts that the matrix does not isolate cannot appear in the plan.

After the four reports are written, `study_level_tests.json`:

* admits only ids present in the plan; a realized test absent from the plan is
  reported as unregistered/exploratory and cannot become a confirmed claim;
* applies **hierarchical Holm gatekeeping**: the primary tier (the main
  proposal vs the best baseline, its leave-one-out mechanisms, and the main
  softsign factorial) is corrected first; the secondary tier (single-axis
  variants, control factorials, QK axis, hyper-parameter sweeps) is tested only
  if at least one primary contrast survives. This controls the study-wide
  false-positive rate without making every exploratory sweep compete with the
  primary claim for the same alpha budget;
* keeps the family-local adjusted p next to the study-level p, and lists every
  contrast demoted by the correction (and, when the gate is closed, every
  secondary contrast that was therefore not tested);
* lists pre-registered contrasts that were not realized (missing rows/tests)
  instead of silently dropping them.

If a rerun in the same output directory changes the seeds, α, margins or
hypotheses, the plan fingerprint changes. That drift is recorded as
`plan_changed` and warned about; `--require-plan-lock` makes it a hard failure,
so a result cannot silently replace the plan it was originally collected
under. A deliberate re-registration needs a fresh output directory.

## 5. Learning-rate fairness

`scripts/tune.py` runs every arm over the same `(lr, weight_decay)` grid and
adds `lr_sensitivity`: the per-arm spread across complete cells relative to
its seed std, and the cross-arm winner at each shared cell. If the winner
changes with `lr`, a fixed-lr norm win is reported as an LR-bonus result, not
a mechanism effect.

## 6. Multi-scale replication

FP8 random-init probing accepts `--models` (GPT-2 124M/350M, LLaMA-tiny 1B);
W8A8 comparisons may include arm pairs from more than one checkpoint shape;
`scripts/moe_scale_sweep.py` runs the MoE study at several
`dim/depth/heads/experts` scales. Each comparison writes a
`scale_replication` block, and `scripts/check_scale_replication.py` combines
per-scale artifacts. A direction is *replicated* only when at least two
distinct scales agree in sign, and *certified across scales* only when at
least two scales individually pass their own test. One scale is a
scope-limited result, never a mechanism claim.

## 7. Held-out evaluation budget and batch reuse

A held-out split can be perfectly disjoint and still be too small for the
evaluation protocol that consumes it. `scripts/train_gpt2.py` validates every
`--val-every` optimizer steps over `--val-batches` batches; when the packed
held-out stream yields fewer distinct batches, `evaluate_heldout` cycles the
loader and scores the same windows more than once. The per-batch standard
error then treats repeated windows as independent draws, and taking the
minimum over many validation passes turns checkpoint selection into a maximum
over reused data (the winner's curse). The sealed test pass does not cycle:
it silently truncates, so its mean can rest on fewer batches than the
protocol advertises.

`scripts/audit_data.py` translates the effective held-out token budget into
the number of distinct packed batches:

```
packed_windows = floor((effective_tokens - 1) / seq_len)
unique_batches = floor(packed_windows / batch_size)
```

and reports, per held-out split, `unique_batches`, the
`requested_batches`, `distinct_batches_per_pass`, `cycles_per_pass` and
`distinct_fraction`. For validation it also computes
`validation_passes = floor(max_steps / val_every)`, `selection_batch_draws`
and `selection_reuse_factor` — how many times each distinct held-out batch is
evaluated while selecting a checkpoint. For the sealed test it reports
`truncated` / `shortfall_batches`. The capacity carries the token budget's
`basis`, so a number extrapolated from a sampled prefix is labelled
approximate rather than presented as exact.

This is a planning and audit check, not a replacement for the cross-split
hash/group ledgers: those establish that the split is disjoint, this
establishes that the evaluation can consume the split it promises.
`--fail-on-eval-reuse` fails an audit whose validation pass must cycle;
`--fail-on-eval-shortfall` fails one whose sealed-test pass would be scored on
fewer batches than requested. Pass the run's actual `--seq-len`,
`--eval-batch-size`, `--val-batches`, `--test-batches`, `--val-every` and
`--max-steps` to the audit.

The training run itself records the same contract: `evaluate_heldout` hashes
each held-out batch's input tokens (the held-out loaders are deterministic
and unshuffled, so a repeated packed window has identical content), counts
distinct windows in the pass, and stores `last_eval_unique_batches` /
`last_eval_cycled` / `last_eval_reused` / `last_eval_shortfall` in its state.
The results carry `final_val_loss_unique_batches`,
`final_val_loss_reused_batches` and `validation_batch_reuse_events`, and
`run_manifest.json` gets a `data.heldout_eval` block with the requested /
unique / reused batch counts (including the sealed-test shortfall). The first
validation pass whose losses are not all distinct prints a warning;
`--fail-on-eval-reuse` makes it a hard stop before more compute is spent on a
validation protocol that cannot support its own error bars.
