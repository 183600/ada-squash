from __future__ import annotations
"""Controlled ablation matrix for the DyT++ / AdaSquash proposal.

Comparison protocol (what the runner guarantees, and checks)
------------------------------------------------------------
* **Controlled variables.** Every row runs under one optimizer (AdamW,
  betas 0.9/0.95), one constant learning rate, one step count, one synthetic
  data protocol and one seed set. Rows may only set architectural /
  initialization knobs; a row that tries to set `lr`, `weight_decay`,
  `batch_size`, `max_steps`, `dropout`, ... is rejected by the loader, because
  a gain obtained by switching the optimizer is not a gain of the module.
* **Complete ablation coverage.** `validate_matrix` checks the leave-one-out
  set for the main proposal (`LOO_SPEC`): Full, Full-minus-per-channel,
  Full-minus-depth-decay, Full-minus-LayerScale, Full-minus-softsign,
  Full-minus-learnable-p and Full-minus-QK-squash must all be present and
  differ from Full in exactly the declared knob(s) - e.g. the per-channel
  contrast uses the frozen-delta row, not `dytd_full`, which also switches
  tanh -> softsign, and the activation family has its own `dytcd_tanh`
  comparator so no claimed mechanism is only ever removed together with
  another one. The attention-logits axis (DyT-QK, Variant C) is removed on
  both the main proposal and the strongest simple baseline, and gets the
  QK-Norm alternative as a comparator.
  `leave_one_out_comparison` then runs the paired Full-vs-Full-minus-X test
  (Holm-adjusted over mechanisms), so "each axis helps" is a tested claim
  rather than an eyeballed delta. `check_factorial` additionally verifies the
  four cells of each `FACTORIAL_SPEC` design, and `factorial_comparison`
  reports the two main effects and their interaction, so a synergy claim is
  tested rather than inferred from Full's point estimate.
* **Cross-axis (2x2x2) interactions and tested orthogonality.**
  `check_three_way_factorial` verifies the union of the softsign and tanh
  2x2 designs as one complete 2x2x2 factorial over per-channel, depth and
  activation. `three_way_factorial_comparison` then estimates the two
  cross-family two-way interactions and the three-way interaction that the
  separate 2x2 reports cannot see. Each pairwise interaction gets a TOST
  equivalence test, so the proposal's "three orthogonal axes" claim is
  certified only when every interaction is equivalent within a pre-specified
  loss margin - a non-significant interaction is reported as such, not
  relabelled as additivity.
* **Statistical reliability.** Every row is replicated over >= 3 seeds by
  default and reported as mean +/- std across seeds; the row-vs-baseline
  significance test is a one-sided paired Student-t test on the per-seed
  means when both rows share >= 3 seeds, because at 3 seeds the normal
  approximation is anti-conservative. Two shared seeds are tested for
  description only and are never certified (df = 1); a single shared run is
  tested at the batch level for description only (batch noise is not
  run-to-run noise). Alias rows that reuse a summary are counted once, and
  each delta carries a 95% confidence interval, per-seed direction counts and
  a power-based minimum detectable delta. The reference baseline is the best
  *replicated* baseline (with the point-estimate best recorded alongside), so
  a lucky single-seed row cannot make certification impossible.
* **Randomized complete-block execution order.** Every row runs once in each
  time block, with an independent per-row seed permutation and a keyed-SHA-256
  within-block order; the schedule (and its fingerprint) is frozen in
  `analysis_plan.json` before training. The pre-flight audit recomputes the
  fingerprint and verifies that every row appears exactly once in every block,
  so an edited or non-crossed allocation is not analysed as the planned
  design. A block-level drift then hits every row once and cancels from the
  paired arm contrast. `run_order_audit.json` tests
  whether the realized position still predicts the held-out loss after
  centering within arm, and reports when cached runs predate the schedule
  instead of inventing an order for them. `--no-randomize-order` is the
  recorded opt-out.
* **Pre-flight design readiness.** `design_readiness.json` checks replication,
  stated hypotheses, a pre-specified smallest effect, planned power (from
  `--assumed-paired-sd`), schedule completeness/balance and the presence of
  the identity `dytcd_full` / `dytcd_ls1e5` control pair. Power is planned
  over the pre-registered *primary* tier at the Bonferroni family alpha, not
  once per test at 0.05, and the seed count is sized for the expected
  complete pairs after `--expected-attrition` (a pair needs both arms, so
  run-level attrition `a` leaves `n*(1-a)^2` pairs). The attrition assumption
  is frozen in `analysis_plan.json`. `--require-design-ready` turns a
  structural failure into a pre-training error. When a margin and an assumed
  SD are supplied, the same file calibrates the actual paired-test code on
  synthetic data (`simulate_paired_design`) instead of trusting its formula.
* **Pre-specified attrition policy.** `exclusion_policy.json` fixes which runs
  may leave the primary contrast (missing run, protocol/provenance mismatch,
  non-finite loss) and which are retained with a sensitivity flag (a diverged
  but finite curve). The policy fingerprint is part of `analysis_plan.json`.
  After the runs, `exclusion_audit.json` keeps every planned run in a per-arm
  denominator, reports duplicate run/seed ids and differential attrition, and
  refuses to certify an arm left below three analyzable seeds;
  `--require-exclusion-audit` makes that a hard failure.
* **Pre-registered analysis plan.** `build_analysis_plan` freezes every
  confirmatory contrast (baseline, leave-one-out, 2x2 factorial, 2x2x2
  factorial interactions, QK) with a stable id, its stated hypothesis, test
  direction and a SHA-256 fingerprint before training starts, and writes
  `analysis_plan.json`. `study_level_tests.json`
  then applies hierarchical Holm gatekeeping: the primary tier (main
  proposal vs best baseline, its leave-one-out mechanisms, the main softsign
  factorial) is corrected first; the secondary tier is tested only if a
  primary contrast survives. A realized test absent from the plan is reported
  as exploratory, and a pre-registered test that was not run is reported as
  unresolved rather than silently dropped.
* **Equivalence and seed robustness.** `--equivalence-margin` enables TOST on
  every paired contrast, so "the two arms are the same" is a tested claim
  rather than a non-significant superiority result; each contrast also carries
  a leave-one-seed-out jackknife showing whether its sign survives dropping
  any single seed. Factorial interactions are reported two-sided (synergy or
  antagonism) alongside the one-sided synergy test.
* **Task-grounded calibration.** A transformer-vs-transformer ranking only
  says which model is least bad; the tuned count model and the exact Bayes
  limits from `scripts/baseline_simple.py` say whether any row learned the
  task at all. The runner therefore records the calibration status in
  `baseline_comparison.json` (`calibration_status`), can auto-run the cheap
  ladder (`--auto-simple-baselines`) and can make a missing calibration a
  hard failure (`--require-simple-baselines`); `--no-simple-baselines` is the
  explicit, recorded opt-out.
* **Tuned baseline certificate.** A win over a fixed-recipe baseline is not a
  win over a *tuned* baseline. `--baseline-tuning` consumes the
  `tuning_summary.json` from `scripts/tune.py` and records whether the
  baseline's selected (lr, weight_decay) cell is the matrix's fixed recipe,
  whether tune.py confirmed it on held-back seeds, and whether the task,
  step budget, batch/sequence shape and selection seeds match this matrix.
  `--require-baseline-tuning` makes a missing or non-certifying report a hard
  failure before training; otherwise the comparison is still written with an
  explicit `baseline_tuning.status`/`certified_against_tuned_reference=false`
  so a reader cannot mistake it for a tuned-baseline result.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.ablation import *  # noqa: F401,F403
from adasquash.ablation.cli import main  # noqa: F401

if __name__ == '__main__':
    main()
