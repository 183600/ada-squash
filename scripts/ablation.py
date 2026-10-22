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
  `analysis_plan.json` before training. A block-level drift then hits every row
  once and cancels from the paired arm contrast. `run_order_audit.json` tests
  whether the realized position still predicts the held-out loss after
  centering within arm, and reports when cached runs predate the schedule
  instead of inventing an order for them. `--no-randomize-order` is the
  recorded opt-out.
* **Pre-flight design readiness.** `design_readiness.json` checks replication,
  stated hypotheses, a pre-specified smallest effect, planned power (from
  `--assumed-paired-sd`), schedule completeness/balance and the presence of
  the identity `dytcd_full` / `dytcd_ls1e5` control pair. `--require-design-ready`
  turns a structural failure into a pre-training error. When a margin and an
  assumed SD are supplied, the same file calibrates the actual paired-test code
  on synthetic data (`simulate_paired_design`) instead of trusting its formula.
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
import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash import build_gpt_124m, count_parameters, ActivationStats, LossSpikeCounter, markov_batch
from adasquash.attention import clear_logit_scratch_cache
from adasquash.design import (classify_run_exclusions,
                               design_readiness_audit,
                               exclusion_policy_fingerprint, normalize_exclusion_policy,
                               order_drift_report,
                               randomized_complete_block_schedule,
                               run_exclusion_audit,
                               schedule_balance_report, simulate_paired_design)
from adasquash.diagnostics import diagnose_train_val_curve, triage_nonfinite
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust,
                               leave_one_out_jackknife, mean_std_stderr,
                               paired_design_summary, paired_t_one_sided,
                               student_t_ppf, tost_paired,
                               two_sided_p_from_t, welch_t_one_sided)
from adasquash.model import build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache, clear_cast_weight_cache
from adasquash.repro import artifact_record, collect_run_metadata, file_fingerprint, finish_metadata, git_revision, save_run_manifest, set_all_seeds
from scripts.train_gpt2 import (_explicit_arg_names, apply_config_to_parser,
                                build_optimizer, load_yaml_config)


def _finite_metric(value) -> bool:
    """A loss may be pooled/tested only when it is a finite number.

    `isinstance(value, (int, float))` alone admits NaN/Inf from a diverged
    run; NaN then propagates through the seed mean (and the paired tests),
    silently poisoning the row's pooled val_loss.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _safe_exp(value):
    """exp(loss) for PPL reporting; a diverged finite loss must stay visible."""
    try:
        return math.exp(float(value))
    except (OverflowError, ValueError):
        return math.inf


def _paired_effect_summary(row: dict, reference: dict, common_seeds, test: dict | None=None,
                           alpha: float=0.05, power: float=0.8,
                           margin: float=0.0) -> dict:
    """Seed-level direction and resolution for one comparison.

    A p-value alone hides two things a reader needs: how many independent
    seeds moved in the claimed direction (with three seeds a 2/3 split can
    still produce a small t p-value) and the smallest delta the test could
    have detected at the requested power. The latter is what separates "no
    evidence of a difference" from "the design could not have seen one".
    Both are descriptive: they never change `beats_*_significantly`.
    """
    out = {
        'n_seeds_favoring': None,
        'n_seeds_tied': None,
        'n_seeds_against': None,
        'per_seed_delta': None,
        'paired_delta_std': None,
        'minimum_detectable_delta_95_power80': None,
        'design_power': None,
    }
    common = list(common_seeds or [])
    if len(common) >= 2:
        row_seeds = _seed_means(row)
        ref_seeds = _seed_means(reference)
        deltas = {int(seed): float(row_seeds[seed] - ref_seeds[seed])
                  for seed in common if seed in row_seeds and seed in ref_seeds}
        if deltas:
            values = list(deltas.values())
            stats = mean_std_stderr(values)
            out.update({
                'n_seeds_favoring': sum(v < 0.0 for v in values),
                'n_seeds_tied': sum(v == 0.0 for v in values),
                'n_seeds_against': sum(v > 0.0 for v in values),
                'per_seed_delta': {str(seed): value for seed, value in deltas.items()},
                'paired_delta_std': None if stats is None else stats['std'],
            })
    if (test is not None and test.get('df') and len(common) >= 2):
        row_seeds = _seed_means(row)
        ref_seeds = _seed_means(reference)
        deltas = [float(row_seeds[seed] - ref_seeds[seed])
                  for seed in common if seed in row_seeds and seed in ref_seeds]
        stats = mean_std_stderr(deltas)
        if stats is not None and stats['stderr'] is not None:
            # Use the paired-delta stderr directly. Inverting the CI would
            # require the test to have been run at this `alpha`; `_paired_row_test`
            # uses its own default, so the inferred stderr would be wrong
            # whenever the caller overrides the significance level.
            # The test is H1 mean(delta) < -margin, so the raw effect must
            # clear both the practical margin and the sampling noise. Report
            # margin + t*stderr; omitting the margin understates the smallest
            # detectable effect whenever --baseline-margin is non-zero.
            out['minimum_detectable_delta_95_power80'] = (
                float(margin)
                + (student_t_ppf(1.0 - alpha, float(test['df']))
                   + student_t_ppf(power, float(test['df'])))
                * stats['stderr']
            )
            # Prospective sample-size check against the pre-specified margin.
            # It uses the observed paired SD (the best available planning
            # value) but the *scientific* margin, not the observed delta, so a
            # null result can be read as underpowered rather than as evidence
            # of no effect. With margin=0 the block explicitly says no target
            # was registered instead of inventing one post hoc.
            out['design_power'] = paired_design_summary(
                stats['std'], int(test['n']), margin=margin,
                alpha=alpha, power=power)
    return out


def _per_seed_deltas(row: dict, reference: dict, common_seeds) -> dict:
    """{seed: row - reference} for the shared seeds, or {} when not pairable.

    This is the single representation every robustness helper consumes, so
    the sign-flip-free jackknife, the equivalence test and the direction
    counts all agree on what "the effect for seed s" means.
    """
    row_seeds = _seed_means(row)
    ref_seeds = _seed_means(reference)
    out = {}
    for seed in common_seeds or []:
        if seed in row_seeds and seed in ref_seeds:
            out[int(seed)] = float(row_seeds[seed] - ref_seeds[seed])
    return out


def _robustness_for_deltas(deltas: dict, alpha: float = 0.05,
                           equivalence_margin: float = 0.0) -> dict:
    """Attach seed-jackknife stability and a TOST equivalence check.

    Two questions a significance test alone cannot answer are asked for every
    paired contrast:

    * **Is the sign carried by one seed?** `leave_one_out_jackknife` drops
      each seed in turn and reports whether the direction survives. At three
      seeds this is the difference between "the norm helps" and "one lucky
      run helped".
    * **Is the two arms' difference small enough to call them equivalent?**
      Only a pre-registered `--equivalence-margin` enables TOST; a
      non-significant superiority test is explicitly *not* relabelled as
      equivalence. The margin is stored in the artifact with the p-value.

    `deltas` maps seed -> paired difference (candidate - reference). Both
    blocks are descriptive for the certification decision; the pre-specified
    paired t-test remains the only certifying test.
    """
    values = [float(deltas[seed]) for seed in deltas]
    labels = [str(seed) for seed in deltas]
    jackknife = leave_one_out_jackknife(values, labels=labels)
    if jackknife is not None:
        jackknife["unit"] = "seed"
    equivalence = {
        "enabled": False,
        "test": "tost_paired",
        "margin": None,
        "equivalent": None,
        "p_tost": None,
        "note": ("pass --equivalence-margin <val_loss> to test whether the two "
                 "arms are equivalent; a non-significant difference is not "
                 "equivalence"),
    }
    if equivalence_margin and equivalence_margin > 0.0 and len(values) >= 2:
        tost = tost_paired(values, [0.0] * len(values),
                           margin=equivalence_margin, alpha=alpha)
        tost["enabled"] = True
        equivalence = tost
    return {"seed_jackknife": jackknife, "equivalence": equivalence}


def _robustness_note(entry: dict) -> str:
    """Compact human-readable suffix for the robustness/equivalence blocks."""
    parts = []
    jackknife = entry.get('seed_jackknife')
    if isinstance(jackknife, dict) and jackknife.get('meaningful'):
        if jackknife.get('sign_stable') is False:
            parts.append(f" [SEED-SENSITIVE: dropping "
                         f"{jackknife.get('influential_labels')} flips the sign]")
        elif jackknife.get('sign_stable') is True:
            parts.append(' [seed-robust sign]')
    equivalence = entry.get('equivalence')
    if isinstance(equivalence, dict) and equivalence.get('enabled'):
        margin = equivalence.get('margin')
        if equivalence.get('equivalent'):
            parts.append(f' [EQUIVALENT within {margin:g}]')
        else:
            parts.append(f' [NOT equivalent within {margin:g}]')
    return ''.join(parts)


ABLATIONS = {'ln_baseline': dict(norm_name='ln'), 'rmsnorm_baseline': dict(norm_name='rmsnorm'), 'dyt_no_perchannel': dict(norm_name='dyt'), 'dytc_perchannel_tanh_k0': dict(norm_name='dytcd_tanh', k_init=0.0, learnable_k=False), 'dytd_tanh_k0': dict(norm_name='dytd_tanh', k_init=0.0, learnable_k=False), 'dytcd_full': dict(norm_name='dytcd'), 'dytcd_no_perchannel': dict(norm_name='dytcd', learnable_delta=False), 'dytcd_k0_no_depth_schedule': dict(norm_name='dytcd', k_init=0.0, learnable_k=False), 'dytcd_no_perchannel_no_depth': dict(norm_name='dytcd', learnable_delta=False, k_init=0.0, learnable_k=False), 'dytd_k0_no_depth_schedule': dict(norm_name='dytd', k_init=0.0, learnable_k=False), 'dytd_full': dict(norm_name='dytd'), 'dytd_tanh_full': dict(norm_name='dytd_tanh'), 'dytcd_no_ls': dict(norm_name='dytcd', use_layerscale=False), 'dytcd_no_softsign': dict(norm_name='dytcd_tanh'), 'dytcd_fixed_p': dict(norm_name='dytcd', p_init=2.0, learnable_p=False), 'dytcd_ls1e5': dict(norm_name='dytcd', layerscale_init=1e-05), 'dytcd_ls1e3': dict(norm_name='dytcd', layerscale_init=0.001), 'dytc_p1_softsign': dict(norm_name='dytc', p_init=1.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p2_softsign': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p4_softsign': dict(norm_name='dytc', p_init=4.0, learnable_p=False, alpha_jitter=0.0), 'dytc_plearnable': dict(norm_name='dytc', p_init=2.0, learnable_p=True, alpha_jitter=0.0), 'dytc_hardtanh_eq': dict(norm_name='dytc', p_init=64.0, learnable_p=False, alpha_jitter=0.0), 'dyt_alpha_0.3': dict(norm_name='dyt', alpha_init=0.3), 'dyt_alpha_0.5': dict(norm_name='dyt', alpha_init=0.5), 'dyt_alpha_1.0': dict(norm_name='dyt', alpha_init=1.0), 'rmsnorm_qk': dict(norm_name='rmsnorm', qk_norm='dytqk'), 'rmsnorm_qknorm': dict(norm_name='rmsnorm', qk_norm='qknorm'), 'ln_qk': dict(norm_name='ln', qk_norm='dytqk'), 'ln_qknorm': dict(norm_name='ln', qk_norm='qknorm'), 'dytcd_qk': dict(norm_name='dytcd', qk_norm='dytqk'), 'dytcd_qknorm': dict(norm_name='dytcd', qk_norm='qknorm')}
# Every row must state *why* it is run (the scientific hypothesis it tests),
# not just what it configures. `load_ablations_from_config` enforces this for
# YAML rows; these are the hypotheses of the built-in fallback rows.
ABLATION_HYPOTHESES = {
    'ln_baseline': 'Simple statistical baseline: LayerNorm on the identical protocol. The DyT family must beat this untuned-but-fairly-run reference before any added machinery counts as progress.',
    'rmsnorm_baseline': 'Simple statistical baseline: RMSNorm, the strongest no-mean normalization used in modern LLMs. A DyT-family gain is only meaningful relative to this reference.',
    'dyt_no_perchannel': 'Original DyT (one scalar alpha, tanh) matches or beats the LayerNorm/RMSNorm references at this scale, reproducing the paper claim under our protocol.',
    'dytc_perchannel_tanh_k0': 'Cross-activation per-channel isolation at flat depth: against dytd_k0_no_depth_schedule this row changes exactly norm_name, keeping frozen k=0, tanh, gamma/beta and LayerScale identical, so a lower loss is attributable to the per-channel factor under tanh rather than to the simultaneous tanh -> softsign switch in dytc_p2_softsign vs dyt_no_perchannel.',
    'dytcd_full': 'Combining the per-channel and depth axes beats both single-axis variants under the same protocol. The leave-one-out contrasts, not this row alone, identify which component is load-bearing; no positive-interaction claim is made without a 2x2 factorial.',
    'dytcd_no_perchannel': 'Leave-one-out for the per-channel axis of the main proposal: with delta frozen at zero (a zero buffer, not a parameter) alpha_c == alpha_l exactly, while the depth schedule, the softsign family and the LayerScale residual stay identical to the full model. Unlike the dytd_full row, this contrast changes no other mechanism, so a loss increase here is attributable to the per-channel deviation alone.',
    'dytcd_k0_no_depth_schedule': 'Leave-one-out for the depth axis of the main proposal: with k pinned to 0 and frozen, DyT-CD keeps its per-channel factor, softsign and LayerScale, so any loss of the full model advantage is attributable to depth decay.',
    'dytd_k0_no_depth_schedule': 'Depth decay is the source of DyT-D gain: with k pinned to 0 and frozen, dytd_full advantage should vanish.',
    'dytd_full': 'Depth-conditioned alpha (+ LayerScale) beats scalar DyT here: shallow strong squashing + deep relaxed response helps this depth.',
    'dytcd_no_ls': 'The LayerScale residual gate contributes to DyT-CD; removing it (use_layerscale off) hurts held-out loss.',
    'dytcd_no_softsign': 'Leave-one-out for the activation family of the main proposal: the dytcd_tanh comparator keeps the per-channel x depth alpha, the LayerScale residual and gamma/beta identical to dytcd_full and swaps only the generalized softsign for tanh, so a loss increase here is attributable to the activation family alone. Unlike the dytd_full row, it does not also remove the per-channel factor, so the dytd -> dytcd bundle is no longer the only activation-family evidence.',
    'dytcd_fixed_p': 'Leave-one-out for the learnable saturation exponent of the main proposal: with p frozen at its init (2.0) the model keeps every mechanism of dytcd_full except the per-layer adaptation of the softsign steepness, so a loss increase here is attributable to learnable p alone.',
    'dytcd_ls1e5': 'LayerScale init 1e-5 (the DyT-CD default) is a competitive residual-gate scale, not an arbitrary strawman.',
    'dytcd_ls1e3': 'LayerScale init 1e-3 trades warmup stability for stronger early residual growth; the default should not be beaten by it.',
    'dytd_tanh_k0': 'Plain-autograd DyT-D at flat depth (k=0 frozen): the "neither" cell of the clean per-channel x depth tanh factorial, matching dytc_perchannel_tanh_k0 in implementation (both plain autograd).',
    'dytd_tanh_full': 'Plain-autograd DyT-D with depth schedule: the "depth" cell of the clean per-channel x depth tanh factorial, matching dytcd_no_softsign in implementation (both plain autograd).',
    'dytcd_no_perchannel_no_depth': 'DyT-CD with per-channel off (delta frozen) and depth off (k=0 frozen): the "neither" cell of the softsign arm in the 2x2x2 factorial.',
    'rmsnorm_qknorm': 'QK-Norm comparator on RMSNorm baseline: per-head RMSNorm on Q/K is the standard alternative; a dytqk win on RMSNorm is only credited if it also beats this comparator.',
    'ln_qknorm': 'QK-Norm comparator on LayerNorm baseline: mirrors rmsnorm_qknorm for the other simple baseline.',
    'dytc_p1_softsign': 'Softsign sharpness matters: p=1 (classic softsign) is worse than p=2 on this outlier-heavy corpus.',
    'dytc_p2_softsign': 'p=2 (smooth softsign) is the best fixed saturation exponent of {1, 2, 4} for outlier control with usable gradients.',
    'dytc_p4_softsign': 'p=4 saturates harder than p=2; the hypothesis is that this hurts gradient flow and loses to p=2.',
    'dytc_plearnable': 'A learnable p adapts the saturation steepness per layer and beats every fixed p in the sweep.',
    'dytc_hardtanh_eq': 'The p->infinity limit (hard tanh) starves gradients outside [-1,1] and loses to finite p.',
    'dyt_alpha_0.3': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_0.5': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_1.0': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'rmsnorm_qk': 'Attention-logits axis, standalone: adding the tanh Q/K scalar squashing to the strongest simple baseline (RMSNorm) changes exactly qk_norm, so a held-out-loss reduction is attributable to the QK squashing alone rather than to an interaction with the DyT-CD stack.',
    'dytcd_qk': 'Leave-one-out for the attention-logits axis of the proposal (DyT-QK, Variant C) in combination: with the norm stack, optimizer, data and qk_alpha_init (shared default 1.0) identical to dytcd_full, the only effective difference is qk_norm = dytqk, so the paired Full-minus-QK contrast attributes any held-out-loss change to the tanh Q/K squashing alone.',
    'dytcd_qknorm': 'QK-Norm comparator for the attention-logits axis: per-head RMSNorm on Q/K is the standard alternative the DyT-QK variant claims to sharpen, run on the identical protocol so a dytcd_qk win is only credited if it also beats this comparator, not just the no-QK row.',
    'ln_qk': 'Attention-logits axis, standalone on LayerNorm baseline: adding the tanh Q/K scalar squashing to LayerNorm changes exactly qk_norm, so a held-out-loss reduction is attributable to the QK squashing alone rather than to an interaction with the DyT-CD stack. Mirrors rmsnorm_qk for the other simple baseline.',
}
_MODEL_BUILDERS = {'gpt2-124m': build_gpt_124m, 'gpt2-350m': build_gpt_350m, 'llama-1b': build_llama_tiny_1b}
_ABLATION_METADATA_KEYS = ('hypothesis',)

# Knobs that the matrix runner holds fixed by construction: every row gets the
# same optimizer (AdamW, fixed betas), learning rate, weight decay, step count,
# batch size, sequence length, data generator and seed set. A row that sets one
# of these is not an ablation of the proposed module - it is a different
# optimization protocol whose gain cannot be attributed to the module - so the
# loader rejects it instead of silently running it.
_FORBIDDEN_ROW_KEYS = frozenset({
    'lr', 'learning_rate', 'weight_decay', 'beta1', 'beta2', 'betas',
    'optimizer', 'scheduler', 'schedule', 'warmup', 'min_lr_ratio',
    'max_steps', 'epochs', 'batch_size', 'grad_accum', 'seq_len',
    'grad_clip', 'dropout', 'bf16', 'seed', 'seeds', 'dataset',
    'dataset_config', 'tokenizer', 'augmentation',
})

# Default number of random seeds for the matrix. One seed cannot separate a
# method effect from run-to-run noise, so the key experiment is replicated
# over >= 3 seeds and reported as mean +/- std across seeds (see
# `aggregate_seed_summaries` and `baseline_comparison`).
DEFAULT_SEEDS = (42, 43, 44)

# The main proposal's row and its 2x2 factorial design. These define the
# primary tier of the study-level gatekeeping correction; a custom matrix with
# different row names must update them (the gate then falls back to
# "not applicable" rather than silently treating an arbitrary row as primary).
PRIMARY_MAIN_ROW = 'dytcd_full'
PRIMARY_FACTORIAL_DESIGN = 'per_channel_x_depth_softsign'

# Leave-one-out (LOO) coverage of the main proposal DyT-CD. Each entry names
# one claimed mechanism and the matrix row that removes exactly it. The claim
# "mechanism X helps" is supportable only when the Full model and its X-free
# reference ran on the identical protocol; `validate_matrix` refuses to run
# the matrix when a mechanism is never isolated. Each removal must change the
# declared knobs *and nothing else*: the `dytd_full` row also switches
# tanh -> generalized softsign, so it cannot isolate the per-channel axis (it
# stays in the matrix as the depth-axis reference); the per-channel contrast
# uses `dytcd_no_perchannel`, whose only effective difference from Full is the
# frozen delta buffer. The activation family gets the same treatment through
# `dytcd_tanh` (plain-autograd comparator, per-channel x depth alpha kept),
# so Full-minus-softsign changes exactly `norm_name`, and learnable p through
# the frozen-p row (`p_init` explicit at its default so only `learnable_p`
# differs). The attention-logits axis (DyT-QK, Variant C) is isolated once on
# the main proposal (`dytcd_qk`, only `qk_norm` differs from Full) and once on
# the strongest simple baseline (`rmsnorm_qk`), so the QK squashing is tested
# both in combination and standalone; `dytcd_qknorm` is the QK-Norm
# alternative comparator, not a removal.
LOO_SPEC = {
    'dytcd_full': {
        'per_channel_delta': {'row': 'dytcd_no_perchannel',
                              'diff': ('learnable_delta',)},
        'depth_decay': {'row': 'dytcd_k0_no_depth_schedule',
                        'diff': ('k_init', 'learnable_k')},
        'layerscale_residual': {'row': 'dytcd_no_ls',
                                'diff': ('use_layerscale',)},
        'activation_family': {'row': 'dytcd_no_softsign',
                              'diff': ('norm_name',)},
        'learnable_p': {'row': 'dytcd_fixed_p',
                        'diff': ('learnable_p',)},
    },
    'dytc_perchannel_tanh_k0': {
        'per_channel_axis_tanh': {'row': 'dytd_k0_no_depth_schedule',
                                  'diff': ('norm_name',)},
    },
    'rmsnorm_qk': {
        'qk_squash': {'row': 'rmsnorm_baseline',
                      'diff': ('qk_norm',)},
    },
    'rmsnorm_qknorm': {
        'qk_norm_comparator': {'row': 'rmsnorm_baseline',
                               'diff': ('qk_norm',)},
    },
    'ln_qk': {
        'qk_squash': {'row': 'ln_baseline',
                      'diff': ('qk_norm',)},
    },
    'ln_qknorm': {
        'qk_norm_comparator': {'row': 'ln_baseline',
                               'diff': ('qk_norm',)},
    },
    'dytcd_qk': {
        'qk_squash': {'row': 'dytcd_full',
                      'diff': ('qk_norm',)},
    },
'dytcd_qknorm': {
        'qk_norm_comparator': {'row': 'dytcd_full',
                               'diff': ('qk_norm',)},
    },
}

# The attention-logits axis is independent of the normalization family: it
# changes only the pre-dot-product Q/K compression and must be attributable
# on each host it is stacked on, not inferred from one row. Each entry is a
# clean triple: the same host norm with QK off, with the fixed tanh DyT-QK
# compression on, and with the standard per-head QK-Norm comparator. The
# checker below requires that both transitions change exactly `qk_norm`.
QK_AXIS_SPEC = {
    'ln': {'off': 'ln_baseline', 'on': 'ln_qk', 'comparator': 'ln_qknorm'},
    'rmsnorm': {'off': 'rmsnorm_baseline', 'on': 'rmsnorm_qk',
                'comparator': 'rmsnorm_qknorm'},
    'dytcd': {'off': 'dytcd_full', 'on': 'dytcd_qk',
              'comparator': 'dytcd_qknorm'},
}

# A complete 2x2 factorial for the two axes the main proposal combines. The
# four cells all use tanh, the same LayerScale residual and the same scalar
# initialization context, so the only moving parts are the per-channel factor
# and the depth schedule. Leave-one-out proves each axis helps from Full;
# this design additionally identifies a *positive interaction* (synergy)
# rather than assuming it from Full's point estimate. `check_factorial`
# verifies that switching a factor changes exactly its declared knobs in both
# arms of the other factor, so a confounded cell is rejected before training.
#
# This factorial now uses plain-autograd comparators for ALL cells (dytd_tanh
# and dytcd_tanh), so the per_channel factor cleanly isolates the per-channel
# axis without mixing fused vs plain implementations. The interaction term is
# now unconfounded and can be used to certify synergy.
FACTORIAL_SPEC = {
    # The main proposal's own activation family (generalized softsign). The
    # four cells already exist in the matrix; without this design the runner
    # only tested the tanh control factorial, so the softsign arm's
    # interaction was never validated or estimated.
    'per_channel_x_depth_softsign': {
        'context': ('generalized-softsign activation family (the main proposal), '
                    'LayerScale on, same gamma/beta and same alpha_0/k/p '
                    'initialization context across the four cells.'),
        'cells': {
            'neither': 'dytcd_no_perchannel_no_depth',
            'per_channel': 'dytcd_k0_no_depth_schedule',
            'depth': 'dytcd_no_perchannel',
            'both': 'dytcd_full',
        },
        'factors': {
            'per_channel': {
                'diff': ('learnable_delta',),
                'pairs': (('neither', 'per_channel'), ('depth', 'both')),
            },
            'depth': {
                'diff': ('k_init', 'learnable_k'),
                'pairs': (('neither', 'depth'), ('per_channel', 'both')),
            },
        },
    },
    'per_channel_x_depth_tanh': {
        'context': ('tanh activation family, LayerScale on, same gamma/beta and '
                    'same alpha/k initialization context across the four cells. '
                    'All cells use plain-autograd (dytd_tanh / dytcd_tanh) so the '
                    'per_channel and depth factors are implementation-clean.'),
        'cells': {
            'neither': 'dytd_tanh_k0',
            'per_channel': 'dytc_perchannel_tanh_k0',
            'depth': 'dytd_tanh_full',
            'both': 'dytcd_no_softsign',
        },
        'factors': {
            'per_channel': {
                'diff': ('norm_name',),
                'pairs': (('neither', 'per_channel'), ('depth', 'both')),
            },
            'depth': {
                'diff': ('k_init', 'learnable_k'),
                'pairs': (('neither', 'depth'), ('per_channel', 'both')),
            },
        },
    },
}

# A complete 2x2x2 factorial over the three axes the proposal calls
# orthogonal: per-channel adaptation, depth conditioning, and the activation
# family (generalized softsign vs tanh).  The two 2x2 designs above each
# estimate a per-channel x depth interaction *inside one activation family*;
# neither can tell whether that interaction is the same under the other
# family.  This design does, and it is the only place where the "three
# orthogonal axes" claim is actually tested rather than asserted.
#
# The eight cells already exist in the shipped matrix (they are the union of
# the softsign and tanh 2x2 designs), so the 2x2x2 analysis adds no training
# cost.  `check_three_way_factorial` verifies that switching one factor
# changes exactly the declared knobs in every context of the other two.  The
# per-channel "off" state is implemented as a frozen zero delta buffer in the
# softsign cells and as the delta-free dytd_tanh class in the tanh cells;
# those two representations are functionally the same (both contribute
# exactly alpha_l), and the declared diff sets below record the difference
# explicitly instead of hiding it.  Likewise the activation switch moves
# between a fused kernel and its plain-autograd comparator, which the
# per-parameter kernel tests already pin as numerically equivalent.  The
# activation factor is the whole squash family: the softsign cells carry the
# learnable `p`, the tanh cells do not (the `dytcd_fixed_p` leave-one-out row
# isolates learnable p inside the softsign family), so a cross-family
# interaction is a statement about the family as deployed, not about tanh
# with p held fixed.
#
# The interaction terms are tested two-sided (a synergy and an antagonism are
# both findings), and each pairwise interaction also gets a TOST equivalence
# test.  "The axes are orthogonal" is an equivalence claim: a non-significant
# interaction is not evidence of additivity, so the orthogonality verdict is
# only available when a positive `--equivalence-margin` was registered.
THREE_WAY_FACTORIAL_SPEC = {
    'per_channel_x_depth_x_activation': {
        'design_type': '2x2x2',
        'context': (
            'all eight cells of the shipped matrix: per-channel x depth under '
            'generalized softsign and under tanh, same gamma/beta, LayerScale, '
            'alpha_0/k/p initialization context and optimizer recipe. The '
            'per-channel-off tanh cells use the delta-free dytd_tanh class; '
            'the per-channel-off softsign cells use a frozen zero delta buffer. '
            'The activation factor is the whole squash family, so the softsign '
            'cells carry the learnable p while the tanh cells do not; the '
            'separate dytcd_fixed_p row isolates learnable p within softsign.'),
        'factors': ('per_channel', 'depth', 'activation'),
        'levels': {
            'per_channel': ('off', 'on'),
            'depth': ('off', 'on'),
            'activation': ('softsign', 'tanh'),
        },
        'cells': {
            'per_channel=off|depth=off|activation=softsign': 'dytcd_no_perchannel_no_depth',
            'per_channel=on|depth=off|activation=softsign': 'dytcd_k0_no_depth_schedule',
            'per_channel=off|depth=on|activation=softsign': 'dytcd_no_perchannel',
            'per_channel=on|depth=on|activation=softsign': 'dytcd_full',
            'per_channel=off|depth=off|activation=tanh': 'dytd_tanh_k0',
            'per_channel=on|depth=off|activation=tanh': 'dytc_perchannel_tanh_k0',
            'per_channel=off|depth=on|activation=tanh': 'dytd_tanh_full',
            'per_channel=on|depth=on|activation=tanh': 'dytcd_no_softsign',
        },
        # Allowed effective-config differences for one factor switch, keyed by
        # the context of the *other* factors. A context key is a
        # `|`-separated `factor=level` list; '*' matches every context, and a
        # partial key matches every context that agrees with it.
        'diffs': {
            'per_channel': {
                'activation=softsign': ('learnable_delta',),
                'activation=tanh': ('norm_name',),
            },
            'depth': {
                '*': ('k_init', 'learnable_k'),
            },
            'activation': {
                'per_channel=off|depth=off': ('norm_name', 'learnable_delta'),
                'per_channel=off|depth=on': ('norm_name', 'learnable_delta'),
                'per_channel=on|depth=off': ('norm_name',),
                'per_channel=on|depth=on': ('norm_name',),
            },
        },
    },
}

def _parse_three_way_cell_key(key: str) -> dict:
    """`'per_channel=on|depth=off|activation=tanh'` -> a plain assignment dict."""
    assignment = {}
    for part in str(key).split('|'):
        if not part:
            continue
        name, _, value = part.partition('=')
        if not name or not value:
            raise ValueError(f'malformed factorial cell key {key!r}')
        assignment[name] = value
    return assignment

def _three_way_cell_key(assignment: dict, factors) -> str:
    return '|'.join(f'{factor}={assignment[factor]}' for factor in factors)

def _context_matches(context: str, assignment: dict) -> bool:
    """Whether a partial `factor=level|...` context agrees with `assignment`."""
    if context == '*':
        return True
    for part in str(context).split('|'):
        if not part:
            continue
        name, _, value = part.partition('=')
        if not name or not value:
            return False
        if assignment.get(name) != value:
            return False
    return True

def check_three_way_factorial(rows: dict, model_name: str,
                              spec: dict | None=None) -> dict:
    """Verify the 2x2x2 design is complete and every factor switch is clean.

    A 2x2x2 interaction is only interpretable when each factor is switched
    against the same background in every context.  For each factor and each
    combination of the other two factors this checks that the effective model
    config changes in exactly one of the declared ways (the implementation of
    "per-channel off" legitimately differs between the softsign and tanh
    cells, so the allowed diff is context-dependent), and that all eight
    cells are present.  A missing or confounded cell is rejected before any
    training starts, exactly like the 2x2 coverage check.
    """
    spec = THREE_WAY_FACTORIAL_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for design_name, design in spec.items():
        factors = tuple(design.get('factors') or ())
        levels = design.get('levels') or {}
        cells = design.get('cells') or {}
        diffs = design.get('diffs') or {}
        if len(factors) != 3:
            problems.append(f"3-way factorial '{design_name}' must declare exactly "
                            f"three factors, got {list(factors)}")
            continue
        missing_levels = [factor for factor in factors
                          if len(levels.get(factor) or ()) != 2]
        if missing_levels:
            problems.append(f"3-way factorial '{design_name}' factor(s) without two "
                            f"levels: {missing_levels}")
            continue
        expected_cells = {
            _three_way_cell_key(dict(zip(factors, combo)), factors): combo
            for combo in itertools.product(*(levels[factor] for factor in factors))
        }
        missing = sorted(set(expected_cells) - set(cells))
        extra = sorted(set(cells) - set(expected_cells))
        if missing or extra:
            problems.append(
                f"3-way factorial '{design_name}' does not cover the full 2x2x2: "
                f"missing {missing}, extra {extra}")
            continue
        absent_rows = sorted({row for row in cells.values() if row not in rows})
        if absent_rows:
            problems.append(
                f"3-way factorial '{design_name}' references absent row(s) "
                f"{absent_rows}")
            continue
        cfg_cache: dict[str, dict] = {}

        def _cfg(row_name: str) -> dict:
            if row_name not in cfg_cache:
                cfg_cache[row_name] = _effective_cfg(model_name, rows[row_name])
            return cfg_cache[row_name]

        report[design_name] = {
            'design_type': design.get('design_type', '2x2x2'),
            'context': design.get('context'),
            'factors': list(factors),
            'levels': {factor: list(levels[factor]) for factor in factors},
            'cells': dict(cells),
            'switches': {},
        }
        clean = True
        for factor in factors:
            others = [name for name in factors if name != factor]
            allowed_by_context = diffs.get(factor) or {}
            if not allowed_by_context:
                problems.append(
                    f"3-way factorial '{design_name}', factor '{factor}': no "
                    f"declared effective-config diff")
                clean = False
                break
            switches = {}
            for combo in itertools.product(*(levels[name] for name in others)):
                background = dict(zip(others, combo))
                first, second = levels[factor]
                first_key = _three_way_cell_key(
                    dict(background, **{factor: first}), factors)
                second_key = _three_way_cell_key(
                    dict(background, **{factor: second}), factors)
                first_cfg, second_cfg = _cfg(cells[first_key]), _cfg(cells[second_key])
                actual = {key for key in set(first_cfg) | set(second_cfg)
                          if first_cfg.get(key) != second_cfg.get(key)}
                allowed = [set(value) for context, value in allowed_by_context.items()
                           if _context_matches(context, background)]
                if not allowed:
                    problems.append(
                        f"3-way factorial '{design_name}', factor '{factor}': "
                        f"context {background} has no declared diff")
                    clean = False
                    break
                if not any(actual == candidate for candidate in allowed):
                    problems.append(
                        f"3-way factorial '{design_name}', factor '{factor}' in "
                        f"{background}: {cells[second_key]} vs {cells[first_key]} "
                        f"differ in {sorted(actual)}, expected one of "
                        f"{[sorted(value) for value in allowed]}")
                    clean = False
                    break
                switches[_three_way_cell_key(background, others)] = {
                    'from': cells[first_key], 'to': cells[second_key],
                    'diff': sorted(actual),
                }
            if not clean:
                break
            report[design_name]['switches'][factor] = switches
    if problems:
        raise ValueError('3-way factorial coverage is incomplete: '
                         + '; '.join(problems))
    return report

def _effective_cfg(model_name: str, cfg_kwargs: dict) -> dict:
    """The full GPTConfig a matrix row builds, as a plain dict."""
    with torch.device('meta'):
        cfg = _MODEL_BUILDERS[model_name](init=False, **cfg_kwargs).cfg
    return dataclasses.asdict(cfg)

def effective_config_key(model_name: str, cfg_kwargs: dict) -> str:
    """Canonical identity of the model a matrix row trains.

    Rows may spell out builder/GPTConfig defaults explicitly (e.g.
    ``layerscale_init: 1.0e-5`` or ``alpha_init: 0.5``); two rows with equal
    keys build the exact same GPTConfig and, under the same seed, produce
    bit-identical runs. The runner trains the first and reuses its summary
    for the rest.
    """
    return json.dumps(_effective_cfg(model_name, cfg_kwargs), sort_keys=True)

def _check_row_keys(name: str, cfg: dict, source: str) -> None:
    bad = sorted(set(cfg) & _FORBIDDEN_ROW_KEYS)
    if bad:
        raise ValueError(
            f"{source}: ablation row '{name}' sets optimizer/schedule/data "
            f"knob(s) {bad}; those are held fixed for every row so a "
            f"comparison isolates the module under test instead of a "
            f"different training recipe"
        )

def check_leave_one_out(rows: dict, model_name: str, spec: dict | None=None) -> dict:
    """Verify every claimed mechanism is isolated by a Full-minus-X row.

    For each `(full, mechanism) -> {row, diff}` entry the reference row must
    exist and its *effective* model config must differ from the Full config in
    exactly the declared fields - no more (a second change would confound the
    contrast) and no fewer (a no-op row proves nothing). Raises ValueError
    listing every violation so one broken claim cannot hide behind another.
    """
    spec = LOO_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for full_name, removals in spec.items():
        if full_name not in rows:
            problems.append(f"leave-one-out full row '{full_name}' is missing")
            continue
        full_cfg = _effective_cfg(model_name, rows[full_name])
        report[full_name] = {}
        for mechanism, info in removals.items():
            ref_name = info['row']
            declared = set(info['diff'])
            if ref_name not in rows:
                problems.append(
                    f"'{mechanism}' has no Full-minus-X reference: row "
                    f"'{ref_name}' is missing")
                continue
            ref_cfg = _effective_cfg(model_name, rows[ref_name])
            actual = {k for k in full_cfg if full_cfg[k] != ref_cfg[k]}
            actual |= {k for k in ref_cfg if k not in full_cfg}
            if actual != declared:
                problems.append(
                    f"'{mechanism}': {ref_name} vs {full_name} differ in "
                    f"{sorted(actual)}, expected exactly {sorted(declared)}")
                continue
            report[full_name][mechanism] = ref_name
    if problems:
        raise ValueError('leave-one-out coverage is incomplete: ' + '; '.join(problems))
    return report

def check_factorial(rows: dict, model_name: str, spec: dict | None=None) -> dict:
    """Verify every declared 2x2 factorial is a clean, complete design.

    Each factor is switched on/off in *both* arms of the other factor (e.g.
    per-channel must be the only difference both with depth off and with
    depth on). A factor that changes different extra knobs in the two arms
    would make the interaction term a mixture of effects, so this is rejected
    before any run starts rather than discovered after the matrix finishes.
    """
    spec = FACTORIAL_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for design_name, design in spec.items():
        cells = design['cells']
        missing = [name for name in cells.values() if name not in rows]
        if missing:
            problems.append(
                f"factorial '{design_name}' is missing cell row(s) {sorted(missing)}")
            continue
        cfgs = {cell: _effective_cfg(model_name, rows[row])
                for cell, row in cells.items()}
        report[design_name] = {'cells': dict(cells), 'factors': {}}
        for factor, info in design['factors'].items():
            declared = set(info['diff'])
            clean = True
            for off_cell, on_cell in info['pairs']:
                off_cfg, on_cfg = cfgs[off_cell], cfgs[on_cell]
                actual = {key for key in off_cfg if off_cfg[key] != on_cfg[key]}
                actual |= {key for key in on_cfg if key not in off_cfg}
                if actual != declared:
                    problems.append(
                        f"factorial '{design_name}', factor '{factor}': "
                        f"{cells[on_cell]} vs {cells[off_cell]} differ in "
                        f"{sorted(actual)}, expected exactly {sorted(declared)}")
                    clean = False
                    break
            if clean:
                report[design_name]['factors'][factor] = {
                    'diff': sorted(declared),
                    'pairs': [list(pair) for pair in info['pairs']],
                }
    if problems:
        raise ValueError('factorial coverage is incomplete: ' + '; '.join(problems))
    return report

def check_qk_axis(rows: dict, model_name: str, spec: dict | None=None) -> dict:
    """Verify the QK-compression axis is isolated on every host norm.

    QK compression is a separate research axis from the normalization layer,
    so its effect must be measured by switching *only* `qk_norm` on a host
    whose weights/initialization are otherwise identical. This checks the
    off/on pair and the QK-Norm comparator for each host, rejecting a triple
    that also changes depth, initialization or the optimizer protocol.
    """
    spec = QK_AXIS_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for host, info in spec.items():
        off_name, on_name = info['off'], info['on']
        cmp_name = info['comparator']
        missing = [name for name in (off_name, on_name, cmp_name) if name not in rows]
        if missing:
            problems.append(f"QK axis on '{host}' is missing row(s) {missing}")
            continue
        off_cfg = _effective_cfg(model_name, rows[off_name])
        on_cfg = _effective_cfg(model_name, rows[on_name])
        cmp_cfg = _effective_cfg(model_name, rows[cmp_name])
        on_diff = {key for key in set(off_cfg) | set(on_cfg)
                   if off_cfg.get(key) != on_cfg.get(key)}
        cmp_diff = {key for key in set(on_cfg) | set(cmp_cfg)
                    if on_cfg.get(key) != cmp_cfg.get(key)}
        if on_diff != {'qk_norm'}:
            problems.append(
                f"QK axis on '{host}': {on_name} vs {off_name} differ in "
                f"{sorted(on_diff)}, expected exactly ['qk_norm']")
            continue
        if cmp_diff != {'qk_norm'}:
            problems.append(
                f"QK comparator on '{host}': {cmp_name} vs {on_name} differ in "
                f"{sorted(cmp_diff)}, expected exactly ['qk_norm']")
            continue
        if on_cfg.get('qk_norm') != 'dytqk':
            problems.append(
                f"QK axis on '{host}': {on_name} does not enable the fixed "
                f"tanh compression (qk_norm={on_cfg.get('qk_norm')!r})")
            continue
        report[host] = {'off': off_name, 'on': on_name,
                        'comparator': cmp_name,
                        'isolated_field': 'qk_norm'}
    if problems:
        raise ValueError('QK-axis coverage is incomplete: ' + '; '.join(problems))
    return report

def validate_matrix(rows: dict, hypotheses: dict, model_name: str) -> dict:
    """Every static check the matrix must pass before any training starts."""
    missing = []
    for name, cfg in rows.items():
        _check_row_keys(name, cfg, 'matrix')
        if not str(hypotheses.get(name) or '').strip():
            missing.append(name)
    if missing:
        raise ValueError(
            f"ablation rows without a stated scientific hypothesis: {sorted(missing)}")
    return {'rows': sorted(rows),
            'leave_one_out': check_leave_one_out(rows, model_name),
            'factorial': check_factorial(rows, model_name),
            'three_way_factorial': check_three_way_factorial(rows, model_name),
            'qk_axis': check_qk_axis(rows, model_name)}

def install_outlier_channel(model, scale: float=25.0) -> int:
    with torch.no_grad():
        oid = model.cfg.vocab_size - 1
        model.tok_emb.weight[oid] *= scale
    return oid

def install_residual_outlier(model, layer_idx: int | None=None,
                             channel: int | None=None, scale: float=25.0):
    """Amplify one residual-stream channel after a mid layer, for the whole run.

    This is the preferred probe: unlike inflating a token-embedding row, it
    does not touch any parameter (with tied embeddings that row is also an
    output-classifier row, so the old probe was confounded with an output-side
    pathology). A persistent forward hook scales the chosen channel of the
    residual stream after ``layer_idx``, giving every downstream layer the
    classic persistent-channel outlier without changing the weights.
    """
    n_layers = int(getattr(model.cfg, 'n_layers', 12))
    dim = int(getattr(model.cfg, 'dim', 768))
    layer_idx = n_layers // 2 if layer_idx is None else int(layer_idx)
    channel = dim // 2 if channel is None else int(channel)
    block = model.blocks[layer_idx]

    def hook(module, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        y[..., channel].mul_(scale)
        return out
    block.register_forward_hook(hook)
    return {'kind': 'residual_channel', 'layer_idx': layer_idx,
            'channel': channel, 'scale': scale}

def load_ablations_from_config(path: str, include_metadata: bool = False) -> dict:
    """Load the ablation matrix from YAML.

    Returns `{name: cfg_kwargs}` by default (the shape every existing caller
    expects). With `include_metadata=True` each value is
    `{'cfg': cfg_kwargs, 'hypothesis': str}` and every row is required to
    state a scientific hypothesis - an ablation without a stated hypothesis
    is a hyper-parameter lottery, not an experiment, so it is rejected rather
    than silently run.
    """
    cfg = load_yaml_config(path)
    rows = cfg.get('ablations')
    if not rows:
        return {}
    out = {}
    missing_hypothesis = []
    for row in rows:
        row = dict(row)
        name = row.pop('name')
        metadata = {key: row.pop(key) for key in _ABLATION_METADATA_KEYS if key in row}
        if 'norm' in row:
            row['norm_name'] = row.pop('norm')
        _check_row_keys(name, row, path)
        if name in out:
            raise ValueError(f"{path}: duplicate ablation name '{name}' - each row must be unique or an intended comparison is silently lost")
        if include_metadata:
            hypothesis = str(metadata.get('hypothesis') or '').strip()
            if not hypothesis:
                missing_hypothesis.append(name)
            out[name] = {'cfg': row, 'hypothesis': hypothesis or None}
        else:
            out[name] = row
    if missing_hypothesis:
        raise ValueError(
            f"{path}: every ablation row must declare its scientific hypothesis "
            f"(`hypothesis:`); missing for {sorted(missing_hypothesis)}"
        )
    return out

def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path, run_metadata: dict | None=None, hypothesis: str | None=None, seed: int | None=None, order_info: dict | None=None):
    if seed is None:
        seed = getattr(args, 'seed', None)
    if seed is None:
        seed = (getattr(args, 'seeds', None) or DEFAULT_SEEDS)[0]
    seed = int(seed)
    print(f'\n[abl] {name} (seed={seed}): {cfg_kwargs}')
    print(f'  [hypothesis] {hypothesis or "(none stated - this row cannot be interpreted as a test)"}')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seed)
    # Build with the constructor's init pass disabled:
    # `reset_parameters(seed)` below is the run's single, seeded
    # initialization.
    with torch.device(device):
        model = _MODEL_BUILDERS[args.model](init=False, **cfg_kwargs)
    model.reset_parameters(seed)
    model.to(device)
    n_params = count_parameters(model)
    param_breakdown = parameter_breakdown(model)
    flops = analytic_flops_per_token(model.cfg, seq_len=args.seq_len)
    opt = build_optimizer(model, args.lr, args.weight_decay, 0.9, 0.95)
    rng = torch.Generator(device=device).manual_seed(seed)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    outlier_id = install_outlier_channel(model)
    losses = []
    stats_windowed = False
    # In-training curve monitoring. The held-out stream for the curve is a
    # *separate* generator (seed + 2): the selection metric's generator
    # (seed + 1) is only consumed after training, so the per-batch val losses
    # that rows are paired on stay bit-identical whether or not the curve ran.
    # Without this a row is summarized by its final number alone, and a delta
    # produced while one arm was still underfit/overfit reads as a norm effect.
    curve_every = int(getattr(args, 'curve_every', 0) or 0)
    if curve_every == 0:
        curve_every = max(1, int(args.max_steps) // 5)
    elif curve_every < 0:
        curve_every = 0
    curve_batches = max(1, int(getattr(args, 'curve_batches', 5) or 5))
    curve_rng = torch.Generator(device=device).manual_seed(seed + 2)
    curve_train: list = []
    curve_val: list = []
    curve_val_stderr: list = []
    curve_steps: list = []
    curve_diagnoses: list = []
    curve_window_start = 0
    t0 = time.time()
    tokens_processed = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    peak_device_memory_mb = None
    for step in range(args.max_steps):
        x, y = markov_batch(rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device, outlier_id=outlier_id)
        tokens_processed += int(x.numel())
        measure = step % args.stats_every == 0
        if measure:
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        loss = model(x, y, return_logits=False)[1]
        # Backward is enqueued first; `loss` is not mutated by it, so the
        # host read below returns the same value.
        loss.backward()
        loss_val = float(loss.detach())
        if not math.isfinite(loss_val):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=args.lr, grad_clip=1.0)
            raise SystemExit(
                f"[abl][nan] '{name}' loss is non-finite at step {step} "
                f"(seed={seed}): triage stage='{triage['stage']}' "
                f"({triage['diagnosis']}); fixes: " + ' | '.join(triage['recommendations'])
            )
        if measure:
            act_stats.detach()
            clear_logit_scratch_cache()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(grad_norm)):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=args.lr, grad_clip=1.0)
            raise SystemExit(
                f"[abl][nan] '{name}' produced a non-finite gradient at step {step} "
                f"(pre-clip norm={float(grad_norm)}, seed={seed}): triage stage="
                f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
                + ' | '.join(triage['recommendations'])
            )
        opt.step()
        opt.zero_grad(set_to_none=True)
        clear_cast_weight_cache()
        loss_spike.update(loss_val)
        losses.append(loss_val)
        if step % args.log_every == 0:
            print(f'  [step {step:4d}] loss={loss_val:.4f}', flush=True)
        if curve_every > 0 and ((step + 1) % curve_every == 0 or step + 1 == args.max_steps):
            window = losses[curve_window_start:]
            curve_window_start = len(losses)
            train_point = sum(window) / max(1, len(window))
            model.eval()
            with torch.inference_mode():
                held_t = torch.stack([
                    model(*markov_batch(curve_rng, model.cfg.ignore_index, args.batch_size,
                                        args.seq_len, device, outlier_id=outlier_id),
                          return_logits=False)[1]
                    for _ in range(curve_batches)
                ])
            model.train()
            # `.tolist()` widens each fp32 value to the same Python double the
            # explicit fp64 cast produced.
            held = held_t.tolist()
            held_stats = mean_std_stderr(held)
            held_stderr = (held_stats['stderr']
                           if held_stats and held_stats['n'] >= 2 else None)
            curve_train.append(train_point)
            curve_val.append(held_stats['mean']
                           if held_stats and held_stats['n'] == len(held)
                           else float('nan'))
            curve_val_stderr.append(held_stderr)
            curve_steps.append(step + 1)
            diagnosis = diagnose_train_val_curve(
                curve_train, curve_val, train_steps=curve_steps, val_steps=curve_steps,
                log_vocab_size=math.log(model.cfg.vocab_size),
                val_stderr=curve_val_stderr,
            )
            curve_diagnoses.append({
                'step': step + 1,
                'classification': diagnosis['classification'],
                'train_trend': diagnosis['train_trend'],
                'val_trend': diagnosis['val_trend'],
                'train_window_loss': train_point,
                'val_loss': curve_val[-1],
                'val_stderr': held_stderr,
                'val_noise_threshold': diagnosis['evidence'].get('val_noise_threshold'),
                'val_change_within_noise': diagnosis['evidence'].get('val_change_within_noise'),
                'gap_change_within_noise': diagnosis['evidence'].get('gap_change_within_noise'),
                'generalization_gap': diagnosis['generalization_gap'],
                'recommendations': diagnosis['recommendations'],
            })
            if diagnosis['classification'] != 'insufficient_data':
                print(f"  [curve] step={step + 1} train_window={train_point:.4f} "
                      f"val={curve_val[-1]:.4f} -> {diagnosis['classification']} "
                      f"(train {diagnosis['train_trend']}, val {diagnosis['val_trend']})", flush=True)
                if diagnosis['classification'] not in ('healthy', 'insufficient_data'):
                    for line in diagnosis['recommendations']:
                        print(f'  [curve]   action: {line}')
    if not losses:
        raise ValueError(f"ablation '{name}' produced no steps")
    # Held-out validation loss: fresh synthetic batches from a generator the
    # optimizer has never seen (seed + 1), evaluated with no gradients. Model
    # comparisons must use this - the training loss above is fitted data.
    n_val_batches = max(1, getattr(args, 'val_batches', 20))
    val_rng = torch.Generator(device=device).manual_seed(seed + 1)
    # Batches are materialized so a non-finite held-out loss can be triaged on
    # the exact batch that produced it (the generator state advances, so
    # regenerating the batch later would not be the same input).
    val_batches = [markov_batch(val_rng, model.cfg.ignore_index, args.batch_size,
                                args.seq_len, device, outlier_id=outlier_id)
                   for _ in range(n_val_batches)]
    model.eval()
    with torch.inference_mode():
        val_losses = [model(x, y, return_logits=False)[1] for x, y in val_batches]
    bad_val = [(i, float(v)) for i, v in enumerate(val_losses) if not math.isfinite(float(v))]
    if bad_val:
        i_bad, v_bad = bad_val[0]
        triage = triage_nonfinite(model, val_batches[i_bad][0], val_batches[i_bad][1],
                                  device_type=device)
        model.train()
        raise SystemExit(
            f"[abl][nan] '{name}' held-out loss is non-finite on {len(bad_val)}/{n_val_batches} "
            f"val batch(es) (first index={i_bad}, value={v_bad}, seed={seed}): triage stage="
            f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
            + ' | '.join(triage['recommendations'])
        )
    val_losses = [float(v) for v in val_losses]
    model.train()
    # `val_loss` is a sample mean over held-out batches; keep the per-batch
    # losses (and their standard error) so comparisons can be *paired* across
    # rows (every row evaluates the identical batches, generated from the
    # fixed `seed + 1` generator) and report significance instead of a
    # bare point estimate.
    val_stats = mean_std_stderr(val_losses)
    val_loss = val_stats['mean']
    stats = act_stats.summary()
    # The knobs every matrix row shares. Recorded per summary so a reported
    # comparison can be audited for controlled variables after the fact.
    controls = {
        'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.95], 'lr': args.lr,
                      'weight_decay': args.weight_decay,
                      'grad_clip': 1.0},
        'schedule': {'type': 'constant', 'lr': args.lr},
        'data': {'generator': 'markov_batch', 'batch_size': args.batch_size,
                 'seq_len': args.seq_len, 'outlier_id': outlier_id,
                 'outlier_scale': 25.0,
                 'val_seed_offset': 1, 'val_batches': n_val_batches},
        'curve_monitoring': {
            'every': curve_every, 'batches': curve_batches,
            'val_seed_offset': 2,
            'note': ('curve points are held-out; their generator (seed+2) is disjoint '
                     'from the selection-val generator (seed+1), so enabling the curve '
                     'does not change the paired val batches'),
        },
        'train_steps': args.max_steps,
        'seed': seed,
    }
    elapsed_s = time.time() - t0
    if device == 'cuda':
        peak_device_memory_mb = float(
            torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
    summary = {'name': name, 'hypothesis': hypothesis, 'cfg': cfg_kwargs, 'model': args.model, 'n_params': n_params, 'seed': seed, 'seeds': [seed], 'final_loss': losses[-1], 'mean_last_100_loss': sum(losses[-100:]) / max(1, len(losses[-100:])), 'val_loss': val_loss, 'val_losses': val_losses, 'val_loss_std': val_stats['std'], 'val_loss_stderr': val_stats['stderr'], 'val_loss_std_basis': 'val_batches', 'n_val_batches': n_val_batches, 'selection_metric': 'val_loss', 'n_loss_spikes': loss_spike.n_spikes, 'loss_curve': losses, 'act_stats': stats, 'max_abs_pre_norm': stats.get('__summary__', {}).get('max_abs_pre_norm'), 'mean_max_abs_pre_norm': stats.get('__summary__', {}).get('mean_max_abs'), 'mean_excess_kurtosis_pre_norm': stats.get('__summary__', {}).get('mean_excess_kurtosis_pre_norm'), 'controls': controls, 'elapsed_s': elapsed_s, 'tokens_processed': tokens_processed, 'tokens_per_second': (tokens_processed / elapsed_s) if elapsed_s > 0 else None, 'peak_device_memory_mb': peak_device_memory_mb}
    # Per-row model accounting is stored next to the loss.
    summary['parameter_breakdown'] = param_breakdown
    summary['flops_per_token_forward'] = flops['forward_per_token']
    summary['flops_per_token_training_est'] = flops['training_per_token']
    # Execution order is run state, not a nuisance detail: a later run in the
    # same session may see a different thermal/clock regime. Only runs that
    # actually executed under the schedule carry it; a cached summary from an
    # earlier session keeps whatever (possibly absent) order it recorded.
    if order_info is not None:
        summary['run_order'] = dict(order_info)
    if run_metadata is None:
        run_metadata = collect_run_metadata(seed=seed)
    else:
        run_metadata = dict(run_metadata)
        run_metadata['seed'] = seed
    summary['run_metadata'] = run_metadata
    # The curve block: the per-point train/val pair and the regime verdict made
    # at each point, so a row is read as a training curve, not a final number.
    summary['curve_train_window_loss'] = curve_train
    summary['curve_val_loss'] = curve_val
    summary['curve_val_stderr'] = curve_val_stderr
    summary['curve_steps'] = curve_steps
    summary['curve_diagnosis'] = curve_diagnoses[-1] if curve_diagnoses else None
    summary['curve_diagnosis_history'] = curve_diagnoses
    summary['curve_classification'] = (curve_diagnoses[-1]['classification']
                                       if curve_diagnoses else None)
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    final_curve = ''
    if curve_diagnoses:
        last_curve = curve_diagnoses[-1]
        gap = last_curve['generalization_gap']
        gap_str = 'n/a' if gap is None else f'{gap:+.4f}'
        noise = last_curve.get('val_stderr')
        noise_str = '' if noise is None else f', val_se={noise:.4f}'
        final_curve = (f" curve={last_curve['classification']} "
                       f"(train {last_curve['train_trend']}, val {last_curve['val_trend']}, "
                       f"gap={gap_str}{noise_str})")
    print(f"  [done] final_loss={summary['final_loss']:.4f} val_loss={val_loss:.4f} "
          f"max_abs_pre_norm={summary['max_abs_pre_norm']}{final_curve}")
    # Drop the per-row scratch caches.
    clear_ce_scratch_cache()
    return summary

def aggregate_seed_summaries(name: str, summaries: list, cfg_kwargs: dict | None=None,
                             hypothesis: str | None=None,
                             exclusion_policy: dict | None=None) -> dict:
    """Pool per-seed runs into one Mean +/- Std summary.

    The seed is the unit of replication: two runs with different seeds are
    independent samples of the same protocol, while the held-out batches
    inside one run are not (they all come from one generator). The pooled
    `val_loss` is therefore the mean of the per-seed means, `val_loss_std` is
    the sample std across seeds whenever there is more than one seed (with
    `val_loss_std_basis` recording which variability it describes), and the
    per-batch losses are kept tagged by seed so the comparison can pair rows
    seed by seed.

    Every seed also carries its pre-specified exclusion status. A non-finite
    seed still contributes no value to the mean, but it is now visible as
    `excluded` in `per_seed` and `exclusion_status_by_seed` instead of only
    reducing `n_seeds` - the denominator is part of the result.
    """
    per_seed = []
    for s in summaries:
        seed = s.get('seed')
        if seed is None and isinstance(s.get('seeds'), list) and s['seeds']:
            seed = s['seeds'][0]
        per_seed.append({
            'seed': seed,
            'val_loss': s.get('val_loss'),
            'val_losses': s.get('val_losses'),
            'final_loss': s.get('final_loss'),
            'mean_last_100_loss': s.get('mean_last_100_loss'),
            'n_loss_spikes': s.get('n_loss_spikes'),
            'elapsed_s': s.get('elapsed_s'),
            'tokens_processed': s.get('tokens_processed'),
            'tokens_per_second': s.get('tokens_per_second'),
            'peak_device_memory_mb': s.get('peak_device_memory_mb'),
            # The final curve verdict of this run (None for summaries written
            # before curve monitoring existed, or for very short runs).
            'curve_classification': s.get('curve_classification'),
            'curve_diagnosis': s.get('curve_diagnosis'),
            # Present only for runs that executed under this session's
            # randomized schedule; the order audit must not invent one for a
            # cached summary from an earlier, differently ordered session.
            'run_order': s.get('run_order'),
        })
    seed_losses = [p['val_loss'] for p in per_seed if _finite_metric(p['val_loss'])]
    stats = mean_std_stderr(seed_losses) or {'mean': None, 'std': None, 'stderr': None, 'n': 0}
    multi = len(seed_losses) >= 2

    def _mean_of(key):
        vals = [p[key] for p in per_seed if _finite_metric(p.get(key))]
        return sum(vals) / len(vals) if vals else None

    tps_vals = [p['tokens_per_second'] for p in per_seed
                if _finite_metric(p.get('tokens_per_second'))]
    tps_stats = mean_std_stderr(tps_vals)

    # Attrition is part of the row, not an implicit side effect of filtering
    # non-finite values out of the mean. The policy is fixed before training
    # and every seed is classified against it; a diverged-but-finite seed is
    # retained and flagged, never dropped for being unlucky.
    policy = normalize_exclusion_policy(exclusion_policy)
    classified = classify_run_exclusions([
        {'run_id': f'{name}::seed={p.get("seed")}', 'arm': name,
         'seed': p.get('seed'), 'val_loss': p.get('val_loss'),
         'curve_classification': p.get('curve_classification')}
        for p in per_seed], policy=policy)
    classification_by_seed = {str(item.get('seed')): item for item in classified}
    for p in per_seed:
        status = classification_by_seed.get(str(p.get('seed')), {})
        p['exclusion_status'] = status.get('status', 'analyzed')
        p['exclusion_reasons'] = status.get('reasons', [])

    agg = dict(summaries[0]) if summaries else {}
    pooled: list = []
    by_seed: dict = {}
    for p in per_seed:
        if isinstance(p.get('val_losses'), list):
            pooled.extend(p['val_losses'])
            by_seed[str(p['seed'])] = p['val_losses']
    agg.update({
        'name': name,
        'cfg': cfg_kwargs if cfg_kwargs is not None else agg.get('cfg'),
        'hypothesis': hypothesis if hypothesis is not None else agg.get('hypothesis'),
        'seeds': [p['seed'] for p in per_seed],
        'n_seeds': len(seed_losses),
        'per_seed': per_seed,
        'exclusion_policy_fingerprint': exclusion_policy_fingerprint(policy),
        'exclusion_status_by_seed': {
            str(seed): item.get('status')
            for seed, item in classification_by_seed.items()},
        'exclusion_reasons_by_seed': {
            str(seed): item.get('reasons') or []
            for seed, item in classification_by_seed.items()},
        'excluded_seeds': sorted(
            (str(seed) for seed, item in classification_by_seed.items()
             if item.get('status') == 'excluded')),
        'n_seeds_excluded': sum(
            1 for item in classification_by_seed.values()
            if item.get('status') == 'excluded'),
        'val_loss': stats['mean'],
        'val_loss_seed_std': stats['std'] if multi else None,
        'val_loss_seed_stderr': stats['stderr'] if multi else None,
        'val_loss_std': stats['std'] if multi else agg.get('val_loss_std'),
        'val_loss_stderr': stats['stderr'] if multi else agg.get('val_loss_stderr'),
        'val_loss_std_basis': 'seeds' if multi else 'val_batches',
        'val_losses': pooled,
        'val_losses_by_seed': by_seed,
        'final_loss': _mean_of('final_loss'),
        'mean_last_100_loss': _mean_of('mean_last_100_loss'),
        'n_loss_spikes': sum(p['n_loss_spikes'] or 0 for p in per_seed
                             if isinstance(p.get('n_loss_spikes'), (int, float))),
        'elapsed_s': sum(p['elapsed_s'] or 0.0 for p in per_seed
                         if isinstance(p.get('elapsed_s'), (int, float))),
        # Measured cost of the arm, carried next to the loss delta.
        'tokens_processed': sum(p['tokens_processed'] for p in per_seed
                                if isinstance(p.get('tokens_processed'), int)),
        'tokens_per_second': None if tps_stats is None else tps_stats['mean'],
        'tokens_per_second_seed_std': (tps_stats['std'] if tps_stats and len(tps_vals) >= 2 else None),
        'peak_device_memory_mb': _mean_of('peak_device_memory_mb'),
        'cost_basis': 'tokens_per_second / peak_device_memory_mb are measured this run; memory is null on CPU',
        # Curve regimes across the row's seeds: a seed whose run was still
        # underfit/stalled or already overfitting/diverging makes its
        # val_loss a statement about that regime as much as about the norm,
        # so the aggregate keeps the verdicts visible instead of hiding them
        # behind one mean.
        'curve_classification_counts': dict(Counter(
            p['curve_classification'] for p in per_seed if p.get('curve_classification'))),
        'curve_classifications_by_seed': {
            str(p['seed']): p.get('curve_classification') for p in per_seed},
        'curve_last_generalization_gap_by_seed': {
            str(p['seed']): (p.get('curve_diagnosis') or {}).get('generalization_gap')
            for p in per_seed},
        # Per-seed execution position, so run_order_audit can test whether the
        # realized order predicts the outcome after removing the arm mean.
        'run_order_by_seed': {
            str(p['seed']): p.get('run_order') for p in per_seed
            if p.get('run_order') is not None},
    })
    return agg

def run_order_audit(summaries: list, schedule: dict | None=None) -> dict:
    """Test whether the realized execution order predicts the held-out loss.

    Only runs that actually executed under this session's schedule carry a
    `run_order`; cached summaries from an older session are reported as
    missing instead of having an order invented for them. Alias rows are
    excluded because their losses are copies of a canonical run, not
    independent observations. The per-arm centering inside
    `order_drift_report` is what keeps a real arm difference from being
    misread as a time trend.
    """
    observations: list = []
    rows_with_order = 0
    rows_without_order = 0
    for row in summaries or []:
        if not isinstance(row, dict) or row.get('reused_from'):
            continue
        per_seed = row.get('per_seed')
        if not isinstance(per_seed, list):
            continue
        found = False
        for item in per_seed:
            if not isinstance(item, dict) or not _finite_metric(item.get('val_loss')):
                continue
            order = item.get('run_order')
            if not isinstance(order, dict):
                continue
            position = order.get('position')
            if (isinstance(position, bool)
                    or not isinstance(position, (int, float))
                    or not math.isfinite(float(position))):
                continue
            observations.append({
                'unit': order.get('run_id') or f"{row.get('name')}::seed={item.get('seed')}",
                'arm': str(row.get('name')),
                'position': float(position),
                'value': float(item['val_loss']),
            })
            found = True
        if found:
            rows_with_order += 1
        else:
            rows_without_order += 1
    report = order_drift_report(observations)
    report['schedule_fingerprint'] = (schedule or {}).get('fingerprint')
    report['n_rows_with_order'] = rows_with_order
    report['n_rows_without_order'] = rows_without_order
    if schedule is not None:
        report['schedule_balance'] = schedule_balance_report(
            schedule,
            arm_of=lambda unit: unit.rsplit('::rep=', 1)[0]
            if '::rep=' in unit else unit)
    if report.get('status') == 'tested':
        print(f"[order] within-arm order drift: rho={report['spearman_rho']:+.3f}, "
              f"p={report['p_two_sided']:.3f}, direction={report['direction']} "
              f"({report['n_observations']} run(s) with recorded order)")
        if report['drift_detected']:
            print('[order] WARNING: execution order is confounded with the outcome; '
                  're-run the affected arms in a fresh randomized block before '
                  'interpreting their deltas')
    else:
        print(f"[order] not testable: {report.get('note')} "
              f"({rows_with_order} row(s) with recorded order, "
              f"{rows_without_order} without)")
    return report


def _load_summary(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None

def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

# The controlled recipe is more than (model, lr, steps): changing the
# optimizer type/betas, grad clip, schedule, data generator, outlier strength
# or the held-out generator/batch count also changes the number being
# compared. Every field below is recorded in each row's `controls` block and
# none of it is allowed to vary between rows in one comparison.
_PROTOCOL_KEYS = (
    'model', 'optimizer_name', 'optimizer_betas', 'optimizer_lr',
    'weight_decay', 'grad_clip', 'schedule_type', 'schedule_lr',
    'data_generator', 'batch_size', 'seq_len', 'outlier_id', 'outlier_scale',
    'val_seed_offset', 'val_batches', 'train_steps',
)
_PROTOCOL_MODEL_VOCAB = {'gpt2-124m': 50304, 'gpt2-350m': 50304, 'llama-1b': 32000}

def _protocol_outlier_id(model):
    vocab = _PROTOCOL_MODEL_VOCAB.get(model)
    return None if vocab is None else vocab - 1

def _protocol_payload(args) -> dict:
    """The controlled training recipe every row of this invocation shares."""
    return {
        'model': args.model,
        'optimizer_name': 'AdamW',
        'optimizer_betas': [0.9, 0.95],
        'optimizer_lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': 1.0,
        'schedule_type': 'constant',
        'schedule_lr': args.lr,
        'data_generator': 'markov_batch',
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'outlier_id': _protocol_outlier_id(args.model),
        'outlier_scale': 25.0,
        'val_seed_offset': 1,
        'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
        'train_steps': args.max_steps,
    }

def _protocol_signature(args) -> tuple:
    payload = _protocol_payload(args)
    return tuple(payload[key] for key in _PROTOCOL_KEYS)

def _row_protocol_values(row: dict, fallback_model=None):
    """Recorded controlled-recipe dict, or None when the row predates controls.

    Missing individual fields stay `None`; they compare unequal to this
    invocation's values, so an incomplete old record can never be treated as
    matching.
    """
    controls = row.get('controls')
    if not isinstance(controls, dict):
        return None
    optimizer = controls.get('optimizer') or {}
    schedule = controls.get('schedule') or {}
    data = controls.get('data') or {}
    return {
        # Do not fill a missing model from the current invocation: an old row
        # that never said which model it trained cannot be proven comparable,
        # and `None` is what makes the signature mismatch/fail closed.
        'model': row.get('model'),
        'optimizer_name': optimizer.get('name'),
        'optimizer_betas': optimizer.get('betas'),
        'optimizer_lr': optimizer.get('lr'),
        'weight_decay': optimizer.get('weight_decay'),
        'grad_clip': optimizer.get('grad_clip'),
        'schedule_type': schedule.get('type'),
        'schedule_lr': schedule.get('lr'),
        'data_generator': data.get('generator'),
        'batch_size': data.get('batch_size'),
        'seq_len': data.get('seq_len'),
        'outlier_id': data.get('outlier_id'),
        'outlier_scale': data.get('outlier_scale'),
        'val_seed_offset': data.get('val_seed_offset'),
        'val_batches': data.get('val_batches'),
        'train_steps': controls.get('train_steps'),
    }

def row_protocol_signature(row: dict, fallback_model=None):
    """Canonical recorded controlled-recipe tuple, or None when unknown."""
    values = _row_protocol_values(row, fallback_model)
    if values is None:
        return None
    return tuple(values[key] for key in _PROTOCOL_KEYS)

def partition_rows_by_protocol(rows: list, args) -> tuple[list, list]:
    """(rows recorded under this invocation's recipe, rows recorded under another).

    `all_summaries.json` deliberately accumulates rows across invocations, but
    a row trained with a different model, optimizer, schedule, data generator
    or validation protocol is not the same experiment: comparing it would
    credit the module with a change that came from the recipe. Mismatched rows
    stay in the file and in the report, but they are kept out of the
    statistical comparisons. Missing/partial controls count as unknown, never
    as matching.
    """
    expected_values = _protocol_payload(args)
    expected = _protocol_signature(args)
    keep, stale = [], []
    for row in rows:
        values = _row_protocol_values(row, args.model)
        signature = None if values is None else tuple(values[key] for key in _PROTOCOL_KEYS)
        if signature == expected:
            keep.append(row)
        else:
            mismatched = (list(_PROTOCOL_KEYS) if values is None else
                          [key for key in _PROTOCOL_KEYS
                           if values[key] != expected_values[key]])
            stale.append({
                'name': row.get('name'),
                'reason': ('no recorded controls' if values is None else
                           'trained under a different protocol than this invocation'),
                'mismatched_fields': mismatched,
                'recorded_protocol': values if values is not None else {},
                'expected_protocol': expected_values,
            })
    return keep, stale

def cached_seed_matches(summary: dict, name: str, cfg_kwargs: dict, args, seed: int) -> bool:
    """Whether a cached summary is the same seed of the same row under the same recipe.

    The path does not always encode the seed (a single-seed run writes its
    summary directly under the row directory), so the recorded seed is
    compared too. The held-out validation batch count is part of the recipe:
    a cached row evaluated on a different number of batches is not the same
    selection metric, so it is re-trained (a cached summary without the
    recorded count counts as unknown, never as matching).
    """
    if not (isinstance(summary, dict)
            and summary.get('name') == name
            and summary.get('model') == args.model
            and summary.get('cfg') == cfg_kwargs
            and row_protocol_signature(summary, args.model) == _protocol_signature(args)
            and summary.get('n_val_batches') == max(1, int(getattr(args, 'val_batches', 20)))):
        return False
    per_seed = summary.get('per_seed')
    if isinstance(per_seed, list) and len(per_seed) > 1:
        return False
    recorded = summary.get('seed')
    if recorded is None:
        seeds = summary.get('seeds')
        recorded = seeds[0] if isinstance(seeds, list) and seeds else None
    # A malformed cached seed is unknown, not another spelling of `seed`:
    # `int(True)`/`int(42.9)` used to reuse a run that was never seeded 1/42.
    if not _is_seed_id(recorded) or not _is_seed_id(seed):
        return False
    return recorded == seed

def merge_summaries(previous: list, fresh: list) -> list:
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in list(previous) + list(fresh):
        if not isinstance(row, dict) or 'name' not in row:
            continue
        name = row['name']
        if isinstance(name, bool) or not isinstance(name, (str, int)):
            continue
        if name not in merged:
            order.append(name)
        merged[name] = row
    return [merged[name] for name in order if name in merged]

def _is_seed_id(value) -> bool:
    """A seed id is an int; 42.9/True must not be paired with seed 42/1."""
    return isinstance(value, int) and not isinstance(value, bool)

def _seed_means(row) -> dict:
    """{seed: val_loss} for every row shape.

    Aggregates carry a `per_seed` list; a single-run summary (no `per_seed`)
    still identifies its seed, so it can pair with the same seed inside an
    aggregate.
    """
    per = row.get('per_seed')
    out = {}
    if isinstance(per, list):
        for item in per:
            if not isinstance(item, dict):
                return {}
            seed = item.get('seed')
            if not _is_seed_id(seed):
                # Unknown/malformed seed identity makes the whole row
                # unpairable: dropping just this entry would silently pair a
                # different number of runs than the row reports.
                return {}
            if _finite_metric(item.get('val_loss')):
                out[int(seed)] = float(item['val_loss'])
        return out
    seeds = row.get('seeds') if isinstance(row.get('seeds'), list) else []
    seeds = [s for s in seeds if s is not None] or ([row['seed']] if row.get('seed') is not None else [])
    # A single-run summary names exactly one seed, so its pooled val_loss is
    # that seed's mean. A multi-seed row without per_seed losses has no
    # per-seed observations: copying the pooled mean to every seed would
    # invent zero variance and can turn it into a "significant" paired win.
    if not all(_is_seed_id(s) for s in seeds):
        return {}
    unique_seeds = list(dict.fromkeys(int(s) for s in seeds))
    if len(unique_seeds) == 1 and _finite_metric(row.get('val_loss')):
        out[unique_seeds[0]] = float(row['val_loss'])
    return out

def _batches_of_seed(row, seed):
    """Held-out batch losses belonging to one run, or None.

    Never returns `val_losses` when the row pooled batches from several
    seeds: those concatenated lists must not be treated as one run's pairing
    basis (batch k of seed 42 would otherwise be zipped with batch k of
    seed 44).
    """
    by_seed = row.get('val_losses_by_seed')
    if isinstance(by_seed, dict) and isinstance(by_seed.get(str(seed)), list):
        return by_seed[str(seed)]
    # The legacy fallback below uses the row's pooled `val_losses`. That list
    # is one run's batches only when the row itself represents one run: a
    # `per_seed` record with several seeds (even if only one has a finite mean)
    # or a declared multi-seed row must never be zipped as if it were seed
    # `seed` alone. Batch k of another seed would otherwise be paired with
    # batch k of this one.
    per_seed = row.get('per_seed')
    if isinstance(per_seed, list) and len(per_seed) != 1:
        return None
    declared_seeds = row.get('seeds')
    if isinstance(declared_seeds, list) and len(declared_seeds) > 1:
        return None
    declared_n = row.get('n_seeds')
    if (isinstance(declared_n, (int, float)) and not isinstance(declared_n, bool)
            and float(declared_n) > 1.0):
        return None
    means = _seed_means(row)
    if len(means) == 1 and seed in means and isinstance(row.get('val_losses'), list):
        return row['val_losses']
    return None

def _row_std(row):
    """Across-seed std when replicated, within-run batch std otherwise.

    A replicated row normally carries `val_loss_seed_std`; fall back to the
    generic `val_loss_std` when an older/cached aggregate only wrote that
    name, so the delta-vs-seed-noise scale check is not silently disabled.
    """
    if row.get('n_seeds', 1) >= 2 and row.get('val_loss_seed_std') is not None:
        return row.get('val_loss_seed_std')
    return row.get('val_loss_std')

def _paired_row_test(row: dict, reference: dict, margin: float=0.0,
                     alpha: float=0.05):
    """One-sided paired Student-t test of `row` vs `reference`, seed-paired.

    Every comparison in this module must run through the *same* test logic,
    or `baseline_comparison` and `leave_one_out_comparison` could disagree on
    the same pair of rows. Seeds are the unit of replication, and only the
    paired per-seed means (`basis='seed_means'`, >= 3 shared seeds) may
    certify significance in the Holm family. With two shared seeds the paired
    test is reported for description only (df = 1), and when exactly one seed
    is shared the same run's held-out batches are tested *for description
    only* (`basis='val_batches_single_seed<s>'`): batch noise is not
    run-to-run noise, so a single run can never establish that one norm
    generalizes better across seeds. With no shared seed there is no test - in
    particular, batches pooled across different seeds are never zipped
    together. The t reference distribution - not the normal one - is the
    small-sample-correct choice for the 3-5 seeds these matrices use, and the
    returned interval makes the precision of the delta visible. Returns
    `(test, basis, common_seeds)`, with `test=None` when no basis exists.
    """
    row_seeds = _seed_means(row)
    ref_seeds = _seed_means(reference)
    common = sorted(set(row_seeds) & set(ref_seeds))
    if len(common) >= 2:
        test = paired_t_one_sided([row_seeds[s] for s in common],
                                  [ref_seeds[s] for s in common], margin=margin,
                                  alpha=alpha)
        return test, 'seed_means', common
    if len(common) == 1:
        seed = common[0]
        row_batches = _batches_of_seed(row, seed)
        ref_batches = _batches_of_seed(reference, seed)
        if (isinstance(row_batches, list) and isinstance(ref_batches, list)
                and len(row_batches) >= 2 and len(ref_batches) >= 2):
            if len(row_batches) == len(ref_batches):
                test = paired_t_one_sided(row_batches, ref_batches, margin=margin,
                                          alpha=alpha)
                return test, f'val_batches_single_seed{seed}', common
            test = welch_t_one_sided(row_batches, ref_batches, margin=margin,
                                     alpha=alpha)
            return test, f'val_batches_single_seed{seed}_unpaired', common
    basis = 'no_shared_seeds' if (row_seeds and ref_seeds) else 'insufficient_seeds'
    return None, basis, common

def _basis_counted_in_family(basis, n_shared: int=0) -> bool:
    """Only a >= 3-seed paired test is admissible to the Holm family."""
    return basis == 'seed_means' and n_shared >= MIN_SEEDS_FOR_SIGNIFICANCE

def _run_identity(row: dict, parent_of: dict) -> str:
    """Name of the run a row ultimately reuses, following `reused_from` chains.

    A row can be reused more than once (C reuses B, which reuses A), so a
    single hop would misread C and B as two independent runs; resolving to the
    root keeps "aliases are counted once" true for chains too.
    """
    name = row.get('name')
    seen = set()
    while isinstance(name, str) and name not in seen:
        seen.add(name)
        parent = parent_of.get(name)
        if not isinstance(parent, str):
            return name
        name = parent
    return row.get('name')

def baseline_comparison(summaries: list, baseline_names, margin: float=0.0, alpha: float=0.05,
                        equivalence_margin: float=0.0) -> dict | None:
    """Every row's held-out val_loss against the best reference baseline.

    The reference is the best (lowest val_loss) *replicated* row among
    `baseline_names` when one exists: a lucky single-seed baseline must not
    become the bar, because no comparison against it could ever be certified
    and its point estimate is the noisiest. The point-estimate best among all
    baselines is still recorded (`point_estimate_best_baseline`), so the
    selection cannot hide a lower unreplicated reference. Significance uses
    the one-sided paired
    Student-t test on the per-seed means whenever both rows share >= 2 seeds,
    but only >= 3 shared seeds can certify significance (2 seeds have df = 1).
    With exactly one shared seed the same run's held-out batches are tested
    for description only - batch noise is not run-to-run noise - and with no
    shared seed no test runs at all.
    Alias rows (`reused_from`) are the same underlying run and are counted
    once, both in the report and in the Holm family. The t reference distribution matters
    at this replication count: at n = 3, df = 2, the one-sided 5% critical
    value is 2.92, not the normal 1.64. `margin` is the smallest loss
    improvement treated as meaningful, and every entry carries a two-sided
    95% interval for the delta (`delta_ci_95`) so a win smaller than its own
    precision is visible as such. Because every row is tested against the
    *same* reference, the per-row p-values are a family of tests and are
    Holm-Bonferroni adjusted over the matrix before anything is called
    significant: `beats_reference_significantly` requires the adjusted
    p < alpha, and the raw one-sided p stays available as `p_one_sided`.
    Returns None when no baseline row has a usable val_loss.
    """
    baseline_names = set(baseline_names)
    usable = [r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))]
    baselines = [r for r in usable if r['name'] in baseline_names]
    if not baselines:
        return None
    parent_of = {r['name']: r['reused_from'] for r in summaries
                 if isinstance(r, dict) and 'name' in r and isinstance(r.get('reused_from'), str)}
    replicated_baselines = [r for r in baselines
                            if len(_seed_means(r)) >= MIN_SEEDS_FOR_SIGNIFICANCE]
    selection_pool = replicated_baselines or baselines
    reference = min(selection_pool, key=lambda r: r['val_loss'])
    point_best = min(baselines, key=lambda r: r['val_loss'])
    ref_seeds = _seed_means(reference)
    ref_stderr = reference.get('val_loss_seed_stderr')
    if ref_stderr is None:
        ref_stderr = reference.get('val_loss_stderr')
    out = {
        'reference': reference['name'],
        'reference_val_loss': reference['val_loss'],
        'reference_val_loss_std': _row_std(reference),
        'reference_val_loss_stderr': ref_stderr,
        'reference_n_seeds': reference.get('n_seeds', 1),
        'reference_seeds': sorted(ref_seeds),
        'reference_excluded_seeds': reference.get('excluded_seeds') or [],
        'reference_n_seeds_excluded': reference.get('n_seeds_excluded', 0),
        'reference_selection': {
            'rule': ('lowest val_loss among baselines with >= '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} seeds'
                     if replicated_baselines else
                     'lowest val_loss (no baseline reaches '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} seeds; certification unavailable)'),
            'replicated_baselines': sorted(r['name'] for r in replicated_baselines),
            'unreplicated_baselines': sorted(r['name'] for r in baselines
                                             if r not in replicated_baselines),
            'reference_can_certify': len(ref_seeds) >= MIN_SEEDS_FOR_SIGNIFICANCE,
            'reference_is_point_estimate_best': reference['name'] == point_best['name'],
        },
        'point_estimate_best_baseline': {
            'name': point_best['name'],
            'val_loss': point_best['val_loss'],
            'n_seeds': point_best.get('n_seeds', 1),
            'seeds': sorted(_seed_means(point_best)),
        },
        'baseline_rows': {r['name']: r['val_loss'] for r in baselines},
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'protocol': ('seeds are the unit of replication: mean +/- std across >= 3 seeds; '
                     'only the paired test on >= 3 shared per-seed means certifies significance; '
                     'two or one shared seed(s) are descriptive only, alias rows '
                     '(reused summaries) are counted once, and a seed excluded by the '
                     'pre-specified attrition policy is reported in excluded_seeds instead '
                     'of silently reducing n_seeds'),
        'rows': [],
    }
    for row in usable:
        if row['name'] in baseline_names:
            continue
        delta = row['val_loss'] - reference['val_loss']
        entry = {
            'name': row['name'],
            'hypothesis': row.get('hypothesis'),
            'run_identity': _run_identity(row, parent_of),
            'val_loss': row['val_loss'],
            'val_loss_std': _row_std(row),
            'n_seeds': row.get('n_seeds', 1),
            'excluded_seeds': row.get('excluded_seeds') or [],
            'n_seeds_excluded': row.get('n_seeds_excluded', 0),
            'exclusion_status_by_seed': row.get('exclusion_status_by_seed'),
            'delta_vs_reference': delta,
            # The paired test is the certifying statistic; its effect is the
            # shared-seed mean, which can differ from the difference of the
            # arms' own all-seed means when their seed sets differ.
            'delta_paired_mean': None,
            # The prompt's scale check: a loss/PPL gap is only readable next
            # to the run-to-run spread it is being compared against. These are
            # descriptive warnings; the paired t-test remains the certification.
            'delta_over_max_seed_std': None,
            'within_seed_noise': None,
            'ppl_reference': _safe_exp(reference['val_loss']),
            'ppl_row': None,
            'ppl_delta': None,
            'ppl_delta_pct': None,
            'beats_reference': bool(delta < -margin),
            'beats_reference_significantly': False,
            'test': None,
            'test_basis': None,
            'paired_seeds': [],
            'replicated_across_seeds': False,
            'significance_basis': None,
            'excluded_from_family': None,
            'p_one_sided': None,
            'p_one_sided_holm': None,
            't': None,
            'df': None,
            'delta_ci_95': None,
            'n_seeds_favoring': None,
            'n_seeds_tied': None,
            'n_seeds_against': None,
            'per_seed_delta': None,
            'paired_delta_std': None,
            'minimum_detectable_delta_95_power80': None,
            'seed_jackknife': None,
            'equivalence': None,
            # Per-seed held-out losses let the calibrated count-model
            # comparison test the same seeds (the count model is fitted on
            # seed s and scored on the seed s+1 stream, exactly like the row).
            'per_seed_val_loss': _seed_means(row),
            # Curve regimes of the row's seeds (None for pre-monitoring
            # summaries): a delta measured while one arm is underfit or
            # overfitting is annotated, so it cannot be read as pure norm
            # effect. It does not alter the significance test - the regime is
            # itself a property of the norm at the fixed step count.
            'curve_classification_counts': row.get('curve_classification_counts'),
        }
        row_std = _row_std(row)
        ref_std = _row_std(reference)
        seed_stds = [value for value in (row_std, ref_std)
                     if isinstance(value, (int, float)) and math.isfinite(float(value))]
        if seed_stds:
            denom = max(seed_stds)
            if denom > 0.0:
                entry['delta_over_max_seed_std'] = abs(delta) / denom
                entry['within_seed_noise'] = bool(abs(delta) < denom)
        entry['ppl_row'] = _safe_exp(row['val_loss'])
        entry['ppl_delta'] = entry['ppl_row'] - entry['ppl_reference']
        if math.isfinite(entry['ppl_reference']) and entry['ppl_reference'] > 0.0:
            entry['ppl_delta_pct'] = 100.0 * entry['ppl_delta'] / entry['ppl_reference']
        test, basis, common = _paired_row_test(row, reference, margin=margin,
                                               alpha=alpha)
        entry['test_basis'] = basis
        entry['paired_seeds'] = common
        entry['replicated_across_seeds'] = _basis_counted_in_family(basis, len(common))
        if test is not None:
            entry['test'] = test['test']
            entry['p_one_sided'] = test['p_one_sided']
            entry['t'] = test['t']
            entry['df'] = test['df']
            entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
            entry['delta_paired_mean'] = test['mean_delta']
            entry['n'] = test['n']
            # The direction must come from the same paired shared-seed
            # estimate as the p-value. The all-seed mean can have the
            # opposite sign when the arms have different seed sets, which
            # made a row both "does not beat" and "significantly beats".
            if test['mean_delta'] is not None:
                entry['beats_reference'] = bool(test['mean_delta'] < -margin)
        entry.update(_paired_effect_summary(row, reference, common, test, alpha=alpha,
                                           margin=margin))
        entry.update(_robustness_for_deltas(
            _per_seed_deltas(row, reference, common),
            alpha=alpha, equivalence_margin=equivalence_margin))
        if entry['test'] is not None and not entry['replicated_across_seeds']:
            entry['significance_basis'] = (
                f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the test cannot certify the norm effect; '
                'not counted in the Holm family')
        out['rows'].append(entry)
    # Alias rows (identical effective config, `reused_from`) are the *same*
    # run under a second name: they are reported, but a run must not be
    # counted twice as independent evidence nor inflate the Holm family size.
    baseline_identity: dict[str, str] = {}
    for row in baselines:
        baseline_identity.setdefault(_run_identity(row, parent_of), row['name'])
    rows_by_identity: dict[str, list] = {}
    for entry in out['rows']:
        identity = entry.get('run_identity') or entry['name']
        rows_by_identity.setdefault(identity, []).append(entry)
    for identity, aliases in rows_by_identity.items():
        canonical = baseline_identity.get(identity)
        if canonical is None:
            # Pick the canonical alias by evidence quality, not by summary
            # order: an ineligible one-seed alias must not shadow a later
            # replicated row that shares the same run identity.
            canonical = min(
                aliases,
                key=lambda entry: (
                    0 if entry['replicated_across_seeds'] else 1,
                    0 if entry['test'] is not None else 1,
                    -len(entry.get('paired_seeds') or []),
                    entry['name'],
                ),
            )['name']
        for entry in aliases:
            if entry['name'] != canonical:
                entry['excluded_from_family'] = (
                    f'duplicate run of {canonical} (reused summary, not '
                    'independent evidence)')
    # Every seed-replicated row above tests the same reference, so its raw
    # p-values form a family and the per-row alpha must be corrected
    # (Holm-Bonferroni step-down) before a row is reported as a significant
    # win. Rows tested on a single run, duplicates of another row, or rows
    # whose test could not run stay out of the family (and cannot be reported
    # significant).
    family = [entry for entry in out['rows']
              if entry['replicated_across_seeds'] and entry['excluded_from_family'] is None]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        # Match the effect estimate to the test that produced the p-value:
        # `delta_paired_mean` is the shared-seed mean, while
        # `delta_vs_reference` differences the arms' own all-seed means.
        paired_delta = entry.get('delta_paired_mean')
        entry['beats_reference_significantly'] = bool(
            paired_delta is not None and paired_delta < -margin
            and p_holm is not None and p_holm < alpha
        )
    described_only = [entry['name'] for entry in out['rows']
                      if entry['test'] is not None and not entry['replicated_across_seeds']]
    out['rows_tested_descriptively_only'] = described_only
    out['multiple_comparison'] = (
        f'Holm-Bonferroni over the {sum(p is not None for p in adjusted)} seed-replicated row(s) '
        f'against the same reference; `beats_reference_significantly` uses the adjusted p-value. '
        f'Single-run batch-level tests are descriptive only ('
        f'{described_only}) and duplicates/aliases are counted once.'
    )
    thin = {r['name'] for r in [reference, *baselines] if r.get('n_seeds', 1) < 3}
    out['rows_with_fewer_than_3_seeds'] = sorted(thin | {e['name'] for e in out['rows'] if e['n_seeds'] < 3})
    return out

def print_baseline_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[compare] no baseline row present; comparison skipped (include a baseline row to interpret the matrix)')
        return
    ref_std = comparison.get('reference_val_loss_std')
    ref_std_str = '' if ref_std is None else f" +/- {ref_std:.4f}"
    print(f"[compare] reference baseline: {comparison['reference']} "
          f"val_loss={comparison['reference_val_loss']:.4f}{ref_std_str} "
          f"(seeds={comparison.get('reference_seeds')}; "
          f"all baselines: { {k: round(v, 4) for k, v in comparison['baseline_rows'].items()} })")
    selection = comparison.get('reference_selection') or {}
    if selection:
        print(f"[compare] reference selection: {selection.get('rule')}; "
              f"point-estimate best={comparison.get('point_estimate_best_baseline')}; "
              f"reference can certify={selection.get('reference_can_certify')}")
    calibration_status = comparison.get('calibration_status')
    if calibration_status == 'missing':
        print('[compare] calibration: MISSING - rows are ranked only against each other; '
              'regenerate with scripts/baseline_simple.py to read them against the task '
              'Bayes limits (see simple_baseline.how_to_generate in the JSON)')
    elif calibration_status == 'explicitly_disabled':
        print('[compare] calibration: explicitly disabled (--no-simple-baselines)')
    tuning = comparison.get('baseline_tuning') or {}
    if tuning.get('certified_against_tuned_reference'):
        print(f"[compare] baseline tuning: CERTIFIED against {tuning.get('reference_row')!r} "
              f"at the matrix recipe (source: {tuning.get('source')})")
    elif tuning.get('status') == 'not_requested':
        print('[compare] baseline tuning: not requested - results are vs fixed-recipe '
              'baselines, not a certified tuned baseline (see baseline_tuning.how_to_generate)')
    else:
        why = tuning.get('why_not_certified') or [tuning.get('status') or 'missing']
        print('[compare] baseline tuning: NOT CERTIFIED - '
              f"{'; '.join(str(item) for item in why)}")
    for entry in comparison['rows']:
        p = entry['p_one_sided']
        p_holm = entry.get('p_one_sided_holm')
        p_str = 'n/a' if p is None else f'{p:.3g}'
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        std = entry.get('val_loss_std')
        std_str = '' if std is None else f" +/- {std:.4f}"
        basis = entry.get('test_basis') or '-'
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        note = ''
        if entry.get('excluded_from_family'):
            note = f" [{entry['excluded_from_family']}]"
        elif entry.get('significance_basis'):
            note = ' [descriptive only, not certified]'
        direction = ''
        if entry.get('n_seeds_favoring') is not None:
            direction = (f" seeds_favoring={entry['n_seeds_favoring']}/{len(entry.get('paired_seeds') or [])}")
            if entry.get('n_seeds_tied'):
                direction += f" tied={entry['n_seeds_tied']}"
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f" mde80={mde:.4f}"
        ratio = entry.get('delta_over_max_seed_std')
        ratio_str = '' if ratio is None else f" delta/seed_std={ratio:.2f}"
        ppl = entry.get('ppl_delta_pct')
        ppl_str = '' if ppl is None else f" dPPL={ppl:+.2f}%"
        power_warn = ''
        if entry.get('within_seed_noise'):
            power_warn += ' [DELTA WITHIN SEED NOISE]'
        if mde is not None and entry.get('delta_vs_reference') is not None:
            if abs(entry['delta_vs_reference']) < mde:
                power_warn += ' [UNDERPOWERED: |delta| < mde80]'
        regimes = entry.get('curve_classification_counts')
        regime_note = ''
        if regimes:
            off_regime = {k: v for k, v in regimes.items()
                          if k not in ('healthy', 'insufficient_data')}
            if off_regime:
                regime_note = (f" [curve regimes {off_regime}: the delta was not measured "
                               f"in a healthy falling-falling regime - read the curve block]")
        print(f"[compare] {entry['name']:>32s} val_loss={entry['val_loss']:.4f}{std_str} "
              f"delta={entry['delta_vs_reference']:+.4f}{ci_str}{ppl_str}{ratio_str} "
              f"p(one-sided)={p_str} p(holm)={p_holm_str} "
              f"test={entry.get('test') or '-'}@{basis}{direction}{mde_str} "
              f"beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}{note}{regime_note}{power_warn}"
              f"{_robustness_note(entry)}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[compare] NOTE: tests with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are "
              f"described but never certified: {comparison['rows_tested_descriptively_only']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[compare] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def _simple_baseline_protocol_mismatches(protocol, args) -> list:
    """Fields where a calibration report describes a different task/protocol."""
    if not isinstance(protocol, dict):
        return ['protocol']
    expected = {
        'corpus': 'markov_batch',
        'n_states': 64,
        'max_jump': 3,
        'outlier_channel': _protocol_outlier_id(args.model) is not None,
        'outlier_id': _protocol_outlier_id(args.model),
        'vocab_size': _PROTOCOL_MODEL_VOCAB.get(args.model),
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
    }
    mismatches = [f'{key}={protocol.get(key)!r} != {want!r}'
                  for key, want in expected.items() if protocol.get(key) != want]
    # Seeds are the pairing unit of the count-model test: a calibration fitted
    # on different seeds cannot be paired with these rows even when the task
    # and stream offsets are identical.
    recorded_seeds = protocol.get('seeds')
    current_seeds = getattr(args, 'seeds', None)
    if current_seeds is not None:
        recorded = set(recorded_seeds) if isinstance(recorded_seeds, list) else None
        current = set(int(seed) for seed in current_seeds)
        if recorded != current:
            mismatches.append(f'seeds={recorded_seeds!r} != {sorted(current)!r}')
    return mismatches

def load_simple_baselines(path, args=None) -> dict | None:
    """Load the Bayes-limit calibration written by scripts/baseline_simple.py.

    Returns None - with a warning for a malformed or stale file - when the
    path is absent or does not look like a calibration report, so a stale side
    file can never abort the matrix; it only removes the floor annotation.
    When `args` is given, the recorded corpus/vocabulary/batch/held-out
    protocol must match this invocation: annotating rows with limits measured
    on another task would silently misstate how much signal they captured.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[compare] WARNING: ignoring unreadable simple-baseline file {path}: {exc}")
        return None
    limits = data.get('bayes_limits') if isinstance(data, dict) else None
    tuned = data.get('tuned_count_baseline') if isinstance(data, dict) else None
    if not isinstance(limits, dict) or not isinstance(limits.get('memoryless_nats'), (int, float)):
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: missing bayes_limits.memoryless_nats")
        return None
    if not isinstance(tuned, dict) or not isinstance(tuned.get('mean_ce_nats'), (int, float)):
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: missing tuned_count_baseline.mean_ce_nats")
        return None
    if args is not None:
        mismatches = _simple_baseline_protocol_mismatches(data.get('protocol'), args)
        if mismatches:
            print(f"[compare] WARNING: ignoring simple-baseline file {path}: "
                  f"it describes a different protocol ({'; '.join(mismatches)})")
            return None
    return data

def run_simple_baseline_calibration(path, seeds, args) -> dict | None:
    """Run the cheap count-model/Bayes ladder for a missing calibration.

    The ladder fits on the same synthetic generator as the ablation rows
    (one stream per seed), tunes the add-k smoothing on a disjoint stream and
    scores on the held-out stream, so its numbers are directly comparable.
    The vocabulary, outlier id, batch shape and held-out batch count must be
    the invocation's protocol: scoring a different stream would attach a
    Bayes floor/ceiling that describes a different task.
    A failed subprocess is an error: continuing would silently drop the only
    reference that says whether a row learned the task.
    """
    import subprocess
    vocab_size = _PROTOCOL_MODEL_VOCAB.get(args.model)
    if vocab_size is None:
        raise SystemExit(f'[compare] ERROR: no calibration vocabulary is registered '
                         f'for model {args.model!r}')
    cmd = [sys.executable, str(ROOT / 'scripts' / 'baseline_simple.py'),
           '--output', str(path), '--seeds', *[str(int(seed)) for seed in seeds],
           '--vocab-size', str(int(vocab_size)),
           '--batch-size', str(int(args.batch_size)),
           '--seq-len', str(int(args.seq_len)),
           '--val-batches', str(max(1, int(getattr(args, 'val_batches', 20))))]
    print(f"[compare] calibration missing: running {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            sys.stderr.write(proc.stderr)
        raise SystemExit(f'[compare] ERROR: auto simple-baseline calibration failed '
                         f'(exit {proc.returncode}); fix the ladder run or pass '
                         f'--simple-baselines with an existing report')
    data = load_simple_baselines(path, args)
    if data is None:
        raise SystemExit(f'[compare] ERROR: simple-baseline calibration wrote {path} but it '
                         f'does not parse as a calibration report')
    return data


def _baseline_tuning_protocol_mismatches(protocol: dict, args, seeds) -> list:
    """Fields where a tune.py report describes a different matrix recipe."""
    expected = {
        'model': args.model,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
    }
    mismatches = [f'{key}={protocol.get(key)!r} != {want!r}'
                  for key, want in expected.items() if protocol.get(key) != want]
    recorded = protocol.get('selection_seeds')
    normalized = None
    if isinstance(recorded, list) and all(_is_seed_id(seed) for seed in recorded):
        normalized = [int(seed) for seed in recorded]
    expected = (sorted(int(seed) for seed in seeds)
                if all(_is_seed_id(seed) for seed in seeds) else None)
    if (normalized is None or expected is None
            or len(set(normalized)) != len(normalized)
            or sorted(normalized) != expected):
        mismatches.append(f'selection_seeds={recorded!r} != {expected!r} '
                          '(the baseline lr must be selected on the same independent runs)')
    return mismatches


def _tuning_cell(arm: dict, lr: float, weight_decay: float):
    """The recorded (lr, wd) cell of a tune.py arm, or None."""
    for cell in (arm.get('lr_grid') or {}).values():
        if (isinstance(cell, dict)
                and cell.get('lr') == float(lr)
                and cell.get('weight_decay') == float(weight_decay)):
            return cell
    return None


def load_baseline_tuning(path, args, rows: dict, seeds) -> dict | None:
    """Load the compute-matched tune.py report and certify the baseline recipe.

    A matrix at one fixed recipe can only claim a win over a *tuned* baseline
    if the baseline's selected ``(lr, weight_decay)`` cell is that same recipe
    and the selection was confirmed on held-back seeds. The report is also
    rejected when its task/protocol (model, steps, batch/sequence, held-out
    batch count or selection seeds) differs from this matrix: a baseline tuned
    on another task cannot certify this comparison.

    Returns ``None`` (with a warning) for a missing, unreadable or
    protocol-mismatched file. `main` turns that into a hard error under
    ``--require-baseline-tuning``; otherwise the comparison is still written
    but explicitly marked as not certified against a tuned reference.
    """
    path = Path(path)
    if not path.exists():
        print(f'[tune-cert] WARNING: baseline tuning report {path} does not exist')
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f'[tune-cert] WARNING: cannot read baseline tuning report {path}: {exc}')
        return None
    if not isinstance(data, dict):
        print(f'[tune-cert] WARNING: {path} is not a tuning summary object')
        return None
    protocol = data.get('protocol') or {}
    arms = data.get('arms') or {}
    if not isinstance(protocol, dict) or not isinstance(arms, dict):
        print(f'[tune-cert] WARNING: {path} has no protocol/arms block')
        return None
    mismatches = _baseline_tuning_protocol_mismatches(protocol, args, seeds)
    if mismatches:
        print(f'[tune-cert] WARNING: ignoring {path}: it describes a different matrix '
              f'recipe ({"; ".join(mismatches)})')
        return None
    weight_decays = protocol.get('weight_decays')
    if not isinstance(weight_decays, list):
        weight_decays = [protocol.get('weight_decay')]
    recipe_in_grid = (any(isinstance(wd, (int, float)) and float(wd) == args.weight_decay
                          for wd in weight_decays)
                      and any(isinstance(lr, (int, float)) and float(lr) == args.lr
                              for lr in (protocol.get('lrs') or [])))
    tuning_confirmed = bool((data.get('comparison') or {}).get('certified'))
    per_baseline = {}
    for row_name in args.baselines:
        norm_name = (rows.get(row_name) or {}).get('norm_name')
        arm = arms.get(norm_name) if norm_name else None
        best = (arm or {}).get('best') if isinstance(arm, dict) else None
        entry = {
            'row': row_name,
            'norm': norm_name,
            'found': isinstance(best, dict),
            'selected_lr': None if not isinstance(best, dict) else best.get('lr'),
            'selected_weight_decay': None if not isinstance(best, dict) else best.get('weight_decay'),
            'selected_val_loss': None if not isinstance(best, dict) else best.get('mean'),
            'selection_pool': None if not isinstance(best, dict) else best.get('selection_pool'),
            'selection_missing_seeds': (None if not isinstance(best, dict)
                                        else best.get('missing_seeds')),
            'selection_complete': bool(
                isinstance(best, dict)
                and best.get('selection_pool') == 'complete_coverage'
                and best.get('missing_seeds') == []),
            'confirm_complete': bool(((arm or {}).get('confirm') or {}).get('complete_coverage')),
            'confirm_val_loss': (((arm or {}).get('confirm') or {}).get('stats') or {}).get('mean'),
            'recipe_in_grid': recipe_in_grid,
            'fixed_recipe_cell_val_loss': None,
            'recipe_is_selected_cell': False,
        }
        if isinstance(best, dict):
            entry['recipe_is_selected_cell'] = (
                entry['selected_lr'] == args.lr
                and entry['selected_weight_decay'] == args.weight_decay)
            cell = _tuning_cell(arm, args.lr, args.weight_decay)
            if cell is not None:
                entry['fixed_recipe_cell_val_loss'] = cell.get('mean')
        per_baseline[row_name] = entry
    found = [entry for entry in per_baseline.values() if entry['found']]
    if not found:
        print(f'[tune-cert] WARNING: {path} has no tuned arm for any baseline '
              f'{args.baselines}')
        return None
    # Rank by the confirm split when every candidate has one, otherwise by the
    # selection grid mean (the same preference tune.py itself uses).
    if all(entry['confirm_complete'] and _finite_metric(entry['confirm_val_loss'])
           for entry in found):
        best_row = min(found, key=lambda entry: entry['confirm_val_loss'])['row']
    else:
        best_row = min(found, key=lambda entry: (entry['selected_val_loss']
                                                 if _finite_metric(entry['selected_val_loss'])
                                                 else float('inf')))['row']
    missing_baselines = sorted(entry['row'] for entry in per_baseline.values()
                               if not entry['found'])
    all_at_recipe = (not missing_baselines
                     and all(entry['recipe_is_selected_cell'] and entry['selection_complete']
                             for entry in found))
    block = {
        'source': str(path),
        'protocol_checked': True,
        'recipe': {'lr': args.lr, 'weight_decay': args.weight_decay,
                   'max_steps': args.max_steps, 'batch_size': args.batch_size,
                   'seq_len': args.seq_len,
                   'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
                   'seeds': list(seeds)},
        'recipe_in_tuning_grid': recipe_in_grid,
        'tuning_confirmed_on_held_back_seeds': tuning_confirmed,
        'per_baseline': per_baseline,
        'best_tuned_baseline_row': best_row,
        'missing_baseline_arms': missing_baselines,
        'all_baselines_selected_this_recipe': all_at_recipe,
        'certified': bool(recipe_in_grid and tuning_confirmed and all_at_recipe),
        'reading': ('a matrix at one fixed recipe may claim a win over a tuned baseline only '
                    'when every compared baseline selected that exact (lr, weight_decay) cell '
                    'and the selection was confirmed on seeds held back from the grid; '
                    'otherwise the comparison is against the fixed-recipe baseline, not its '
                    'tuned optimum'),
    }
    if not block['certified']:
        reasons = []
        if not recipe_in_grid:
            reasons.append('the matrix recipe is not in the tuning grid')
        if not tuning_confirmed:
            reasons.append('tune.py did not certify on held-back --confirm-seeds')
        if missing_baselines:
            reasons.append(f'no tuned arm for baseline(s) {missing_baselines}')
        if not all_at_recipe and not missing_baselines:
            incomplete = [entry['row'] for entry in found if not entry['selection_complete']]
            if incomplete:
                reasons.append('baseline selection used a partial-coverage (survivor-only) '
                               f'cell for {incomplete}')
            if any(not entry['recipe_is_selected_cell'] for entry in found):
                reasons.append('at least one baseline selected a different '
                               '(lr, weight_decay) cell')
        block['why_not_certified'] = reasons
        print(f'[tune-cert] WARNING: {path} does not certify the fixed recipe: '
              f'{"; ".join(reasons)}')
    else:
        print(f'[tune-cert] certified: baselines {sorted(per_baseline)} were tuned at '
              f'lr={args.lr:g} wd={args.weight_decay:g} and confirmed on held-back seeds '
              f'({path})')
    return block


def attach_baseline_tuning(comparison: dict, tuning: dict) -> dict:
    """Tie the tuning certificate to the reference baseline actually used."""
    block = dict(tuning)
    reference = comparison.get('reference')
    entry = (tuning.get('per_baseline') or {}).get(reference)
    block['reference_row'] = reference
    block['reference_baseline_tuned_at_this_recipe'] = bool(
        entry and entry.get('recipe_is_selected_cell'))
    block['certified_against_tuned_reference'] = bool(
        tuning.get('certified') and block['reference_baseline_tuned_at_this_recipe'])
    if not block['certified_against_tuned_reference']:
        print(f'[compare] WARNING: the reference baseline {reference!r} was not certified at '
              f'this matrix recipe; the significance rows below compare against a fixed-recipe '
              f'baseline, not necessarily its tuned optimum (see baseline_tuning)')
    else:
        print(f'[compare] tuned-baseline certificate: reference {reference!r} selected '
              f'lr={tuning["recipe"]["lr"]:g} wd={tuning["recipe"]["weight_decay"]:g} '
              f'and was confirmed on held-back seeds')
    return block


def _count_seed_ce(records: list, predictor: str | None) -> dict:
    """Per-seed CE of the tuned count model on the shared held-out stream."""
    out = {}
    if not predictor:
        return out
    for record in records or []:
        if not isinstance(record, dict) or record.get('seed') is None:
            continue
        value = (record.get('predictors') or {}).get(predictor)
        if _finite_metric(value):
            out[int(record['seed'])] = float(value)
    return out


def _mde_from_test(test: dict | None, alpha: float = 0.05, power: float = 0.8):
    if test is None or not test.get('df') or not _finite_metric(test.get('ci_low')):
        return None
    half = (test['ci_high'] - test['ci_low']) / 2.0
    critical = student_t_ppf(1.0 - alpha / 2.0, float(test['df']))
    if critical <= 0.0:
        return None
    stderr = half / critical
    return ((student_t_ppf(1.0 - alpha, float(test['df']))
             + student_t_ppf(power, float(test['df']))) * stderr)


def attach_simple_baseline_calibration(comparison: dict, floor_data: dict, source,
                                       alpha: float = 0.05) -> dict:
    """Annotate every comparison row with its distance to the task's Bayes limits.

    `baseline_comparison` answers "which row is best among these
    transformers"; this annotation answers "how much of the task's learnable
    signal did the row capture". Both are needed: a row can win the matrix
    while still sitting far above the ceiling, which means the comparison
    ranked under-trained models. The ceiling is the exact HMM-filter CE when
    the corpus carries the outlier channel (the transformer reads the full
    observed prefix), reported next to the closed-form memoryless walk floor.
    `excess_over_bayes_ceiling_nats` = val_loss - ceiling (0 = Bayes optimal),
    `excess_over_tuned_simple_nats` = distance to the tuned count model, and
    `signal_captured_fraction` rescales the remaining gap against the
    uniform-over-vocabulary strawman.

    The count model and the transformer consume the same generator stream for
    each seed (fit on s, scored on s+1), so their per-seed CEs are paired:
    `beats_tuned_simple_significantly` is set only by a one-sided paired
    t-test over shared seeds (>= 3) after a Holm correction across rows. A
    point-estimate win inside the seed noise remains visible as
    `beats_tuned_simple_baseline=True, beats_tuned_simple_significantly=False`.
    """
    limits = floor_data['bayes_limits']
    floor = float(limits['memoryless_nats'])
    ceiling = limits.get('full_context_nats')
    ceiling = float(ceiling) if isinstance(ceiling, (int, float)) else None
    tuned = floor_data['tuned_count_baseline']
    tuned_ce = float(tuned['mean_ce_nats'])
    tuned_predictor = tuned.get('predictor')
    seed_ce = _count_seed_ce(floor_data.get('per_seed') or [], tuned_predictor)
    protocol = floor_data.get('protocol') or {}
    vocab_size = protocol.get('vocab_size')
    log_vocab = math.log(vocab_size) if isinstance(vocab_size, (int, float)) and vocab_size > 0 else None
    annotated = []
    for entry in comparison['rows']:
        val_loss = entry.get('val_loss')
        if not _finite_metric(val_loss):
            # NaN/Inf/bool is not a measured held-out loss; annotating it
            # would turn an invalid row into a finite-looking Bayes gap.
            continue
        entry['excess_over_bayes_floor_nats'] = val_loss - floor
        if ceiling is not None:
            entry['excess_over_bayes_ceiling_nats'] = val_loss - ceiling
        entry['excess_over_tuned_simple_nats'] = val_loss - tuned_ce
        entry['beats_tuned_simple_baseline'] = bool(val_loss < tuned_ce)
        if log_vocab is not None and ceiling is not None and log_vocab > ceiling:
            entry['signal_captured_fraction'] = (log_vocab - val_loss) / (log_vocab - ceiling)
        row_seeds = {int(seed): float(value)
                     for seed, value in (entry.get('per_seed_val_loss') or {}).items()
                     if _finite_metric(value)}
        common = sorted(set(row_seeds) & set(seed_ce))
        test = None
        entry.update({
            'beats_tuned_simple_significantly': False,
            'n_paired_seeds_tuned_simple': len(common),
            'p_one_sided_tuned_simple': None,
            'p_one_sided_tuned_simple_holm': None,
            'delta_seed_mean_tuned_simple': None,
            'delta_ci_95_tuned_simple': None,
            'minimum_detectable_delta_tuned_simple': None,
            'simple_baseline_test_basis': None,
        })
        if len(common) >= 2:
            test = paired_t_one_sided([row_seeds[seed] for seed in common],
                                      [seed_ce[seed] for seed in common],
                                      alpha=alpha)
            entry.update({
                'p_one_sided_tuned_simple': test['p_one_sided'],
                'delta_seed_mean_tuned_simple': test['mean_delta'],
                'delta_ci_95_tuned_simple': [test['ci_low'], test['ci_high']],
                'minimum_detectable_delta_tuned_simple': _mde_from_test(test, alpha=alpha),
                'simple_baseline_test_basis': (
                    'seed_means' if len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
                    else f'descriptive only: {len(common)} shared seeds, fewer than the '
                         f'required {MIN_SEEDS_FOR_SIGNIFICANCE}'),
            })
        elif len(common) == 1:
            entry['simple_baseline_test_basis'] = (
                'descriptive only: one shared seed, so a paired count-model test '
                'cannot separate an arm effect from run-to-run noise')
        else:
            entry['simple_baseline_test_basis'] = 'no_shared_seeds'
        annotated.append(entry)
    family = [entry for entry in annotated
              if entry.get('n_paired_seeds_tuned_simple', 0) >= MIN_SEEDS_FOR_SIGNIFICANCE
              and entry.get('p_one_sided_tuned_simple') is not None
              and entry.get('excluded_from_family') is None]
    adjusted = holm_adjust([entry['p_one_sided_tuned_simple'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_tuned_simple_holm'] = p_holm
        # The certifying test is paired on the shared seeds, so its direction
        # must come from that paired delta, not the all-seed point estimate
        # (an extra seed can move the pooled mean across the tuned CE and
        # suppress a genuine paired win).
        paired_delta = entry.get('delta_seed_mean_tuned_simple')
        entry['beats_tuned_simple_significantly'] = bool(
            paired_delta is not None and paired_delta < 0.0
            and p_holm is not None and p_holm < alpha)
    block = {
        'source': str(source),
        'memoryless_floor_nats': floor,
        'full_context_ceiling_nats': ceiling,
        'tuned_simple_predictor': tuned_predictor,
        'tuned_simple_ce_nats': tuned_ce,
        'tuned_simple_std_ce_nats': tuned.get('std_ce_nats'),
        'uniform_vocab_strawman_nats': log_vocab,
        'calibration_seeds': protocol.get('seeds'),
        'hypothesis': floor_data.get('hypothesis'),
        'paired_test': ('one-sided paired t-test over the seeds shared by the row and the '
                        'tuned count model; only >= '
                        f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds enter the Holm family'),
        'reading': ('excess_over_bayes_ceiling_nats is the gap to the exact full-context Bayes predictor '
                    '(hmm_bayes) and the number that says whether a row learned the task; '
                    'beats_tuned_simple_baseline only says it beat a tuned count model on the point '
                    'estimate, beats_tuned_simple_significantly is the seed-paired test, and '
                    'signal_captured_fraction rescales the gap against uniform-over-vocab'),
        'rows': [{key: entry.get(key) for key in (
            'name', 'val_loss', 'excess_over_bayes_ceiling_nats', 'excess_over_bayes_floor_nats',
            'excess_over_tuned_simple_nats', 'beats_tuned_simple_baseline',
            'beats_tuned_simple_significantly', 'n_paired_seeds_tuned_simple',
            'p_one_sided_tuned_simple', 'p_one_sided_tuned_simple_holm',
            'delta_seed_mean_tuned_simple', 'delta_ci_95_tuned_simple',
            'minimum_detectable_delta_tuned_simple', 'simple_baseline_test_basis',
            'signal_captured_fraction',
        )} for entry in annotated],
    }
    if annotated:
        best = min(annotated, key=lambda entry: entry.get('excess_over_bayes_ceiling_nats', float('inf')))
        ceiling_str = 'n/a' if ceiling is None else f"{ceiling:.4f}"
        best_gap = best.get('excess_over_bayes_ceiling_nats')
        best_str = 'n/a' if best_gap is None else f"{best_gap:+.4f}"
        print(f"[compare] Bayes calibration ({source}): memoryless floor {floor:.4f} nats, full-context "
              f"ceiling {ceiling_str} nats, tuned simple baseline {tuned_ce:.4f} nats "
              f"({tuned_predictor}); closest row to the ceiling: {best['name']} "
              f"{best_str} nats; rows beating the tuned simple baseline: "
              f"{sum(1 for entry in annotated if entry.get('beats_tuned_simple_baseline'))}/{len(annotated)} "
              f"(significantly: "
              f"{sum(1 for entry in annotated if entry.get('beats_tuned_simple_significantly'))})")
    return block

def leave_one_out_comparison(summaries: list, spec: dict | None=None, margin: float=0.0,
                             alpha: float=0.05,
                             equivalence_margin: float=0.0) -> dict | None:
    """Full vs each Full-minus-X row: does removing mechanism X hurt?

    `check_leave_one_out` proves the *structure* of the leave-one-out matrix -
    the reference row differs from Full in exactly the declared knobs. This
    function adds the statistical half: for every declared mechanism it runs
    the same one-sided paired test as `baseline_comparison`, of H1 "Full has
    lower held-out val_loss than the X-free row". Because all mechanisms are
    tested against the same Full row, the per-mechanism p-values are
    Holm-Bonferroni adjusted before a mechanism is called significantly
    helpful; a point-estimate win inside the seed noise remains visible as
    `mechanism_helps=True, mechanism_helps_significantly=False`. This is what
    backs the claim that each axis of the main proposal is load-bearing on
    its own, not merely correlated with the others. Returns None when no
    declared contrast has both rows with a usable val_loss.
    """
    spec = LOO_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    parent_of = {r['name']: r['reused_from'] for r in summaries
                 if isinstance(r, dict) and 'name' in r and isinstance(r.get('reused_from'), str)}
    rows: list = []
    for full_name, removals in spec.items():
        full = usable.get(full_name)
        for mechanism, info in removals.items():
            ref_name = info['row']
            ref = usable.get(ref_name)
            entry = {
                'full': full_name,
                'mechanism': mechanism,
                'reference_row': ref_name,
                'declared_diff': sorted(info['diff']),
                'full_val_loss': None if full is None else full['val_loss'],
                'full_val_loss_std': None if full is None else _row_std(full),
                'reference_val_loss': None if ref is None else ref['val_loss'],
                'reference_val_loss_std': None if ref is None else _row_std(ref),
                'n_seeds_full': None if full is None else full.get('n_seeds', 1),
                'n_seeds_reference': None if ref is None else ref.get('n_seeds', 1),
                'full_curve_classification_counts': None if full is None else full.get('curve_classification_counts'),
                'reference_curve_classification_counts': None if ref is None else ref.get('curve_classification_counts'),
                'run_identity_full': None if full is None else _run_identity(full, parent_of),
                'run_identity_reference': None if ref is None else _run_identity(ref, parent_of),
                'delta_full_minus_reference': None,
                # Shared-seed effect of the paired test (the certifying
                # statistic); may differ from the all-seed mean difference.
                'delta_paired_mean': None,
                'delta_over_max_seed_std': None,
                'within_seed_noise': None,
                'ppl_full': None,
                'ppl_reference': None,
                'ppl_delta_pct': None,
                'mechanism_helps': False,
                'mechanism_helps_significantly': False,
                'test': None,
                'test_basis': None,
                'paired_seeds': [],
                'replicated_across_seeds': False,
                'significance_basis': None,
                'excluded_from_family': None,
                'p_one_sided': None,
                'p_one_sided_holm': None,
                't': None,
                'df': None,
                'delta_ci_95': None,
                'n_seeds_favoring': None,
                'n_seeds_tied': None,
                'n_seeds_against': None,
                'per_seed_delta': None,
                'paired_delta_std': None,
                'minimum_detectable_delta_95_power80': None,
                'seed_jackknife': None,
                'equivalence': None,
            }
            if full is None:
                entry['skipped'] = f"full row '{full_name}' has no usable val_loss"
                rows.append(entry)
                continue
            if ref is None:
                entry['skipped'] = f"Full-minus-X row '{ref_name}' has no usable val_loss"
                rows.append(entry)
                continue
            if entry['run_identity_full'] == entry['run_identity_reference']:
                entry['skipped'] = (f"'{full_name}' and '{ref_name}' are the same run "
                                    f"(reused summary); the contrast is vacuous")
                rows.append(entry)
                continue
            entry['delta_full_minus_reference'] = full['val_loss'] - ref['val_loss']
            entry['ppl_full'] = _safe_exp(full['val_loss'])
            entry['ppl_reference'] = _safe_exp(ref['val_loss'])
            if (math.isfinite(entry['ppl_full']) and math.isfinite(entry['ppl_reference'])
                    and entry['ppl_reference'] > 0.0):
                entry['ppl_delta_pct'] = (100.0 * (entry['ppl_full'] - entry['ppl_reference'])
                                          / entry['ppl_reference'])
            seed_stds = [value for value in (_row_std(full), _row_std(ref))
                         if isinstance(value, (int, float)) and math.isfinite(float(value))]
            if seed_stds:
                denom = max(seed_stds)
                if denom > 0.0:
                    entry['delta_over_max_seed_std'] = (
                        abs(entry['delta_full_minus_reference']) / denom)
                    entry['within_seed_noise'] = bool(
                        abs(entry['delta_full_minus_reference']) < denom)
            entry['mechanism_helps'] = bool(entry['delta_full_minus_reference'] < -margin)
            test, basis, common = _paired_row_test(full, ref, margin=margin,
                                                   alpha=alpha)
            entry['test_basis'] = basis
            entry['paired_seeds'] = common
            entry['replicated_across_seeds'] = _basis_counted_in_family(basis, len(common))
            if test is not None:
                entry['test'] = test['test']
                entry['p_one_sided'] = test['p_one_sided']
                entry['t'] = test['t']
                entry['df'] = test['df']
                entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
                entry['delta_paired_mean'] = test['mean_delta']
                entry['n'] = test['n']
                # Keep the mechanism direction on the same shared-seed
                # estimate the p-value tests, not the all-seed mean.
                if test['mean_delta'] is not None:
                    entry['mechanism_helps'] = bool(test['mean_delta'] < -margin)
            entry.update(_paired_effect_summary(full, ref, common, test, alpha=alpha,
                                               margin=margin))
            entry.update(_robustness_for_deltas(
                _per_seed_deltas(full, ref, common),
                alpha=alpha, equivalence_margin=equivalence_margin))
            if entry['test'] is not None and not entry['replicated_across_seeds']:
                entry['significance_basis'] = (
                    f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                    f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the test cannot certify the mechanism '
                    'effect; not counted in the Holm family')
            rows.append(entry)
    if not rows:
        return None
    # A contrast is identified by the underlying runs, not the row names: an
    # alias pair would otherwise appear as two independent mechanism tests,
    # and a run could even be contrasted with itself via aliases. Each
    # underlying contrast enters the Holm family exactly once.
    contrasts_by_key: dict[tuple, list] = {}
    for entry in rows:
        if entry.get('skipped') or entry.get('run_identity_full') is None:
            continue
        key = (entry['run_identity_full'], entry['run_identity_reference'])
        contrasts_by_key.setdefault(key, []).append(entry)
    for aliases in contrasts_by_key.values():
        # Same rule as the baseline aliases: pick the usable/testable alias,
        # not whichever one happened to be written first.
        canonical = min(
            aliases,
            key=lambda entry: (
                0 if entry['replicated_across_seeds'] else 1,
                0 if entry['test'] is not None else 1,
                -len(entry.get('paired_seeds') or []),
                entry['mechanism'],
            ),
        )
        for entry in aliases:
            if entry is not canonical:
                entry['excluded_from_family'] = (
                    f"duplicate of mechanism '{canonical['mechanism']}' "
                    "(same underlying runs, not independent evidence)")
    family = [entry for entry in rows
              if entry['replicated_across_seeds'] and entry['excluded_from_family'] is None]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        paired_delta = entry.get('delta_paired_mean')
        entry['mechanism_helps_significantly'] = bool(
            paired_delta is not None and paired_delta < -margin
            and p_holm is not None and p_holm < alpha
        )
    thin = sorted({name for entry in rows for name in (entry['full'], entry['reference_row'])
                   if 1 <= usable.get(name, {}).get('n_seeds', 1) < 3})
    tested_descriptively = [entry['mechanism'] for entry in rows
                            if entry['test'] is not None and not entry['replicated_across_seeds']]
    return {
        'protocol': ('Full vs Full-minus-X: one-sided paired test of H1 "Full has lower '
                     'held-out val_loss than the X-free row"; only the paired test on >= 3 '
                     f'shared per-seed means certifies significance (two or one shared seed(s) '
                     'are descriptive only), and each underlying contrast is counted once even '
                     'when rows are aliases'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'multiple_comparison': (f'Holm-Bonferroni over the '
                                f'{sum(e["p_one_sided"] is not None for e in family)} seed-replicated '
                                f'mechanism(s); `mechanism_helps_significantly` uses the adjusted '
                                f'p-value'),
        'rows': rows,
        'rows_tested_descriptively_only': tested_descriptively,
        'rows_with_fewer_than_3_seeds': thin,
    }

def qk_axis_comparison(summaries: list, spec: dict | None=None, margin: float=0.0,
                       alpha: float=0.05,
                       equivalence_margin: float=0.0) -> dict | None:
    """Attribute the independent QK-compression axis on every host norm.

    The matrix contains a clean QK-off/on pair and a QK-Norm comparator on
    LN, RMSNorm and the full DyT-CD stack. This function tests each pair with
    the same seed-paired one-sided test used everywhere else (`H1: on/off
    lowers held-out loss`) and then reports whether the *direction* of the
    QK effect replicates across hosts. A single host's point estimate is not
    evidence that the attention-logits axis generalizes; a host whose seeds
    do not share the required replication count stays descriptive.
    """
    spec = QK_AXIS_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    rows: list = []
    for host, info in spec.items():
        pair_specs = (
            ('qk_vs_off', info['on'], info['off'],
             'the fixed tanh QK compression lowers loss on this host'),
            ('qknorm_vs_qk', info['comparator'], info['on'],
             'the standard per-head QK-Norm comparator lowers loss relative to the fixed tanh QK compression'),
        )
        for contrast, cand_name, ref_name, hypothesis in pair_specs:
            cand, ref = usable.get(cand_name), usable.get(ref_name)
            entry = {
                'host': host, 'contrast': contrast,
                'candidate_row': cand_name, 'reference_row': ref_name,
                'hypothesis': hypothesis,
                'candidate_val_loss': None if cand is None else cand['val_loss'],
                'reference_val_loss': None if ref is None else ref['val_loss'],
                'delta_candidate_minus_reference': None,
                # Shared-seed effect of the paired test (the certifying
                # statistic); may differ from the all-seed mean difference.
                'delta_paired_mean': None,
                'effect_helps': False,
                'effect_helps_significantly': False,
                'test': None, 'test_basis': None, 'paired_seeds': [],
                'replicated_across_seeds': False, 'significance_basis': None,
                'p_one_sided': None, 'p_one_sided_holm': None,
                't': None, 'df': None, 'delta_ci_95': None,
                'n_seeds_favoring': None, 'n_seeds_tied': None,
                'n_seeds_against': None, 'per_seed_delta': None,
                'paired_delta_std': None,
                'minimum_detectable_delta_95_power80': None,
                'seed_jackknife': None,
                'equivalence': None,
            }
            if cand is None or ref is None:
                missing = [name for name, row in ((cand_name, cand), (ref_name, ref))
                           if row is None]
                entry['skipped'] = f"row(s) {missing} have no usable val_loss"
                rows.append(entry)
                continue
            entry['delta_candidate_minus_reference'] = cand['val_loss'] - ref['val_loss']
            entry['effect_helps'] = bool(entry['delta_candidate_minus_reference'] < -margin)
            test, basis, common = _paired_row_test(cand, ref, margin=margin,
                                                   alpha=alpha)
            entry['test_basis'] = basis
            entry['paired_seeds'] = common
            entry['replicated_across_seeds'] = _basis_counted_in_family(basis, len(common))
            if test is not None:
                entry.update({
                    'test': test['test'], 'p_one_sided': test['p_one_sided'],
                    't': test['t'], 'df': test['df'],
                    'delta_ci_95': [test['ci_low'], test['ci_high']],
                    'delta_paired_mean': test['mean_delta'],
                    'n': test['n'],
                })
                # The paired test is the certifying statistic: its effect is
                # the shared-seed mean. Keep the all-seed difference only as
                # descriptive context; otherwise `effect_helps` and the
                # cross-host direction can contradict a certified paired win
                # when the arms' seed sets differ.
                entry['effect_helps'] = bool(test['mean_delta'] < -margin)
            entry.update(_paired_effect_summary(cand, ref, common, test, alpha=alpha,
                                               margin=margin))
            entry.update(_robustness_for_deltas(
                _per_seed_deltas(cand, ref, common),
                alpha=alpha, equivalence_margin=equivalence_margin))
            if entry['test'] is not None and not entry['replicated_across_seeds']:
                entry['significance_basis'] = (
                    f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                    f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the QK effect cannot be certified; '
                    'not counted in the Holm family')
            rows.append(entry)
    if not rows:
        return None
    family = [entry for entry in rows
              if entry['replicated_across_seeds'] and not entry.get('skipped')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        paired_delta = entry.get('delta_paired_mean')
        entry['effect_helps_significantly'] = bool(
            paired_delta is not None and paired_delta < -margin
            and p_holm is not None and p_holm < alpha)
    # Cross-host replication of the QK-on vs QK-off direction. Only hosts
    # with a seed-replicated test count as independent replications; a host
    # tested on one run is reported but cannot make the axis "generalize".
    replicated_qk = [entry for entry in rows
                     if entry['contrast'] == 'qk_vs_off'
                     and entry['replicated_across_seeds']
                     and entry.get('delta_paired_mean') is not None]
    deltas = [entry['delta_paired_mean'] for entry in replicated_qk]
    signs = {('negative' if delta < 0 else 'positive' if delta > 0 else 'zero')
             for delta in deltas}
    # An all-zero set is a null result, not a replicated direction: no host
    # shows a QK effect, so there is no sign to agree on.
    direction_consistent = (len(signs) == 1 and bool(deltas)
                            and next(iter(signs)) in ('negative', 'positive'))
    certified_hosts = sorted(entry['host'] for entry in replicated_qk
                             if entry['effect_helps_significantly'])
    cross_host = {
        'hosts_with_replicated_test': sorted(entry['host'] for entry in replicated_qk),
        'host_deltas': {entry['host']: entry['delta_paired_mean']
                        for entry in replicated_qk},
        'host_delta_basis': 'shared-seed paired mean (same statistic as the test)',
        'direction_signs': sorted(signs),
        'direction_consistent_across_hosts': direction_consistent,
        'hosts_certified_individually': certified_hosts,
        'certified_at_multiple_hosts': (len(certified_hosts) >= 2
                                        and direction_consistent),
        'reading': ('the attention-logits axis is independent of the norm family: a QK effect '
                    'claimed as general must have the same direction on at least two hosts, and '
                    'is only certified at multiple hosts when >= 2 hosts pass the Holm-adjusted '
                    'paired test; a single host is a scope-limited result'),
    }
    return {
        'protocol': ('QK compression is isolated by switching only `qk_norm` on each host; '
                     'one-sided paired test H1 "on/comparator lowers held-out loss", '
                     f'certified only with >= {MIN_SEEDS_FOR_SIGNIFICANCE} shared per-seed means, '
                     'Holm-adjusted across all QK pairs'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'cross_host_replication': cross_host,
        'rows': rows,
        'rows_tested_descriptively_only': [
            f"{entry['host']}:{entry['contrast']}" for entry in rows
            if entry.get('test') is not None and not entry['replicated_across_seeds']],
    }

def print_qk_axis_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[qk] no QK-axis pair available; nothing to test')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[qk] {entry['host']}/{entry['contrast']}: SKIPPED ({entry['skipped']})")
            continue
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm = entry.get('p_one_sided_holm')
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        note = '' if entry['replicated_across_seeds'] else ' [descriptive only, not certified]'
        print(f"[qk] {entry['host']:>8s} {entry['contrast']:>14s}: "
              f"delta={entry['delta_candidate_minus_reference']:+.4f}{ci_str} "
              f"p={p_str} p(holm)={p_holm_str} helps={entry['effect_helps']} "
              f"significant={entry['effect_helps_significantly']}{note}"
              f"{_robustness_note(entry)}")
    cross = comparison.get('cross_host_replication') or {}
    print(f"[qk] cross-host: {cross.get('host_deltas')} signs={cross.get('direction_signs')} "
          f"consistent={cross.get('direction_consistent_across_hosts')} "
          f"certified_hosts={cross.get('hosts_certified_individually')} "
          f"multi_host={cross.get('certified_at_multiple_hosts')}")
    if not cross.get('certified_at_multiple_hosts'):
        print('[qk] NOTE: the QK axis is not certified at multiple hosts; report any effect '
              'with the host it was measured on (single-host scope), not as a general result')

def print_leave_one_out_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[loo] no leave-one-out contrast available; nothing to test')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[loo] {entry['mechanism']}: SKIPPED ({entry['skipped']})")
            continue
        full_std = entry.get('full_val_loss_std')
        ref_std = entry.get('reference_val_loss_std')
        full_std_str = '' if full_std is None else f' +/- {full_std:.4f}'
        ref_std_str = '' if ref_std is None else f' +/- {ref_std:.4f}'
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm_str = 'n/a' if entry.get('p_one_sided_holm') is None else f"{entry['p_one_sided_holm']:.3g}"
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        note = ''
        if entry.get('excluded_from_family'):
            note = f" [{entry['excluded_from_family']}]"
        elif entry.get('significance_basis'):
            note = ' [descriptive only, not certified]'
        direction = ''
        if entry.get('n_seeds_favoring') is not None:
            direction = (f" seeds_favoring={entry['n_seeds_favoring']}/{len(entry.get('paired_seeds') or [])}")
            if entry.get('n_seeds_tied'):
                direction += f" tied={entry['n_seeds_tied']}"
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f" mde80={mde:.4f}"
        ratio = entry.get('delta_over_max_seed_std')
        ratio_str = '' if ratio is None else f" delta/seed_std={ratio:.2f}"
        ppl = entry.get('ppl_delta_pct')
        ppl_str = '' if ppl is None else f" dPPL={ppl:+.2f}%"
        power_warn = ''
        if entry.get('within_seed_noise'):
            power_warn += ' [DELTA WITHIN SEED NOISE]'
        if mde is not None and entry.get('delta_full_minus_reference') is not None:
            if abs(entry['delta_full_minus_reference']) < mde:
                power_warn += ' [UNDERPOWERED: |delta| < mde80]'
        regime_note = ''
        for label, key in (('full', 'full_curve_classification_counts'),
                           ('ref', 'reference_curve_classification_counts')):
            regimes = entry.get(key)
            off_regime = {k: v for k, v in (regimes or {}).items()
                          if k not in ('healthy', 'insufficient_data')}
            if off_regime:
                regime_note += f" [{label} curve regimes {off_regime}]"
        print(f"[loo] {entry['mechanism']:>22s}: {entry['full']} "
              f"val_loss={entry['full_val_loss']:.4f}{full_std_str} vs "
              f"{entry['reference_row']} val_loss={entry['reference_val_loss']:.4f}{ref_std_str} "
              f"delta(full-ref)={entry['delta_full_minus_reference']:+.4f} {ci_str}"
              f"{ppl_str}{ratio_str} p(one-sided)={p_str} p(holm)={p_holm_str} "
              f"test={entry.get('test') or '-'}@{entry.get('test_basis') or '-'}{direction}{mde_str} "
              f"removing_hurts={entry['mechanism_helps']} "
              f"significant={entry['mechanism_helps_significantly']}{note}{regime_note}{power_warn}"
              f"{_robustness_note(entry)}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[loo] NOTE: tests with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are described "
              f"but never certified: {comparison['rows_tested_descriptively_only']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[loo] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def factorial_comparison(summaries: list, spec: dict | None=None, margin: float=0.0,
                         alpha: float=0.05,
                         equivalence_margin: float=0.0) -> dict | None:
    """Paired main effects and interaction for each declared 2x2 factorial.

    The interaction is the difference of the factor's effect in the two arms
    of the other factor, computed per seed:
    ``(both - depth) - (per_channel - neither)``. A negative interaction means
    the two mechanisms together lower loss by more than the sum of their
    separate effects (synergy); a positive one means they overlap or interfere.
    Testing the interaction is what lets the report distinguish "Full wins
    because either axis helps" from "the combination is special". Every term
    uses the same shared seed set across all four cells and the same
    small-sample paired t-test as the other comparisons; only >= 3 shared
    seeds enter the Holm family. Returns None when no design has all cells.
    """
    spec = FACTORIAL_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    rows: list = []
    for design_name, design in spec.items():
        cells = design['cells']
        factor_names = list(design['factors'])
        missing = [row for row in cells.values() if row not in usable]
        base_entry = {
            'design': design_name,
            'cells': dict(cells),
            'context': design.get('context'),
            'shared_seeds': [],
        }
        if missing:
            entry = dict(base_entry, effect=None, kind='design', skipped=(
                f"cell row(s) {sorted(missing)} have no usable val_loss"))
            rows.append(entry)
            continue
        seed_values = {cell: _seed_means(usable[row]) for cell, row in cells.items()}
        common = sorted(set.intersection(*(set(values) for values in seed_values.values())))
        if len(common) < 2:
            entry = dict(base_entry, effect=None, kind='design', shared_seeds=common,
                         skipped=(f"only {len(common)} seed(s) are shared by all four "
                                  f"cells; a paired effect needs at least 2"))
            rows.append(entry)
            continue
        effects = {}
        for factor in factor_names:
            arm_values = {}
            for off_cell, on_cell in design['factors'][factor]['pairs']:
                arm_values[f'{off_cell}->{on_cell}'] = [
                    seed_values[on_cell][seed] - seed_values[off_cell][seed]
                    for seed in common
                ]
            arm_means = [sum(values) / len(values) for values in arm_values.values()]
            main_values = [sum(vals) / len(vals) for vals in zip(*arm_values.values())]
            effects[factor] = {
                'kind': 'main_effect',
                'formula': 'mean over both arms of the other factor: on - off',
                'arm_effects': {name: {'mean_delta': sum(vals) / len(vals),
                                       'per_seed_delta': dict(zip(map(str, common), vals))}
                                for name, vals in arm_values.items()},
                'per_seed_values': dict(zip(map(str, common), main_values)),
                'mean_arm_deltas': arm_means,
                'delta_seed_mean': sum(main_values) / len(main_values),
                'per_seed_delta': dict(zip(map(str, common), main_values)),
            }
        if len(factor_names) == 2:
            first, second = factor_names
            first_arms = list(effects[first]['arm_effects'].items())
            interaction_values = [
                on - off for off, on in zip(
                    first_arms[0][1]['per_seed_delta'].values(),
                    first_arms[1][1]['per_seed_delta'].values())
            ]
            effects[f'{first}_x_{second}'] = {
                'kind': 'interaction',
                'formula': (f'effect of {first} with {second} on minus effect of {first} '
                            f'with {second} off; negative = synergy'),
                'per_seed_values': dict(zip(map(str, common), interaction_values)),
                'delta_seed_mean': sum(interaction_values) / len(interaction_values),
                'per_seed_delta': dict(zip(map(str, common), interaction_values)),
            }
        for effect_name, effect in effects.items():
            values = list(effect['per_seed_values'].values())
            test = paired_t_one_sided(values, [0.0] * len(values), margin=margin, alpha=alpha)
            # A two-sided test of "the effect is non-zero" is the right
            # question for an interaction: antagonism (positive interaction)
            # is as reportable as synergy, and the one-sided synergy test
            # alone would silently ignore it. The one-sided test stays the
            # certification for "helps", so the existing claim is unchanged.
            two_sided_test = paired_t_one_sided(
                values, [0.0] * len(values), margin=0.0, alpha=alpha)
            p_two_sided = two_sided_p_from_t(
                two_sided_test['t'], two_sided_test['df'])
            replicated = len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
            entry = dict(base_entry, effect=effect_name, kind=effect['kind'],
                         formula=effect['formula'], shared_seeds=common,
                         n_seeds=len(common),
                         delta_seed_mean=effect['delta_seed_mean'],
                         per_seed_delta=effect['per_seed_delta'],
                         n_seeds_favoring=sum(value < 0.0 for value in values),
                         n_seeds_tied=sum(value == 0.0 for value in values),
                         n_seeds_against=sum(value > 0.0 for value in values),
                         effect_helps=bool(effect['delta_seed_mean'] < -margin),
                         effect_helps_significantly=False,
                         test=test['test'],
                         test_basis=('seed_means' if replicated
                                     else f'seed_means_{len(common)}_descriptive'),
                         replicated_across_seeds=replicated,
                         p_one_sided=test['p_one_sided'],
                         p_one_sided_holm=None,
                         p_two_sided=p_two_sided,
                         p_two_sided_holm=None,
                         interaction_differs_significantly=None,
                         interaction_direction=None,
                         t=test['t'],
                         df=test['df'],
                         delta_ci_95=[test['ci_low'], test['ci_high']],
                         n=test['n'])
            if effect['kind'] == 'interaction':
                delta = effect['delta_seed_mean']
                if delta < -margin:
                    entry['interaction_direction'] = 'synergy'
                elif delta > margin:
                    entry['interaction_direction'] = 'antagonism'
                else:
                    entry['interaction_direction'] = 'additive_or_within_margin'
            effect_row = {'per_seed': [{'seed': int(seed), 'val_loss': value}
                                       for seed, value in effect['per_seed_values'].items()]}
            zero_row = {'per_seed': [{'seed': int(seed), 'val_loss': 0.0}
                                     for seed in effect['per_seed_values']]}
            entry.update(_paired_effect_summary(effect_row, zero_row, common, test,
                                              alpha=alpha, margin=margin))
            entry.update(_robustness_for_deltas(
                {str(seed): float(value)
                 for seed, value in effect['per_seed_values'].items()},
                alpha=alpha, equivalence_margin=equivalence_margin))
            paired_std = entry.get('paired_delta_std')
            if (isinstance(paired_std, (int, float)) and paired_std > 0.0
                    and entry.get('delta_seed_mean') is not None):
                entry['delta_over_paired_std'] = (abs(entry['delta_seed_mean'])
                                                 / paired_std)
                entry['within_paired_seed_noise'] = bool(
                    abs(entry['delta_seed_mean']) < paired_std)
            else:
                entry['delta_over_paired_std'] = None
                entry['within_paired_seed_noise'] = None
            if 'arm_effects' in effect:
                entry['arm_effects'] = effect['arm_effects']
            rows.append(entry)
    if not rows:
        return None
    family = [entry for entry in rows
              if entry.get('replicated_across_seeds') and not entry.get('skipped')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['effect_helps_significantly'] = bool(
            entry['effect_helps'] and p_holm is not None and p_holm < alpha)
    # Interactions answer a two-sided question ("is the combination more or
    # less than the sum of its parts?"), so they get their own Holm family
    # over the two-sided p-values. A significant positive interaction is an
    # antagonism finding, not a failed synergy claim.
    interaction_family = [entry for entry in family
                          if entry.get('kind') == 'interaction']
    interaction_adjusted = holm_adjust(
        [entry.get('p_two_sided') for entry in interaction_family])
    for entry, p_holm in zip(interaction_family, interaction_adjusted):
        entry['p_two_sided_holm'] = p_holm
        entry['interaction_differs_significantly'] = bool(
            p_holm is not None and p_holm < alpha)
    tested_descriptively = [entry['effect'] for entry in rows
                            if entry.get('p_one_sided') is not None
                            and not entry.get('replicated_across_seeds')]
    return {
        'protocol': ('2x2 factorial per_seed paired terms on the seeds shared by all '
                     'four cells; only tests with >= '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are certified, and all '
                     'main effects and interactions form one Holm family'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'multiple_comparison': (f'Holm-Bonferroni over {len(family)} seed-replicated '
                                'factorial term(s); `effect_helps_significantly` uses the '
                                'adjusted one-sided p-value'),
        'interaction_multiple_comparison': (
            f'Holm-Bonferroni over {len(interaction_family)} seed-replicated interaction '
            'term(s) using the two-sided p-value; `interaction_differs_significantly` '
            'covers synergy and antagonism'),
        'designs': spec,
        'rows': rows,
        'rows_tested_descriptively_only': tested_descriptively,
    }

def print_factorial_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[factorial] no 2x2 factorial design declared; interaction test skipped')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[factorial] {entry['design']}: SKIPPED ({entry['skipped']})")
            continue
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm_str = ('n/a' if entry['p_one_sided_holm'] is None
                      else f"{entry['p_one_sided_holm']:.3g}")
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f"mde80={mde:.4f} "
        ratio = entry.get('delta_over_paired_std')
        ratio_str = '' if ratio is None else f"delta/paired_std={ratio:.2f} "
        power_warn = ''
        if entry.get('within_paired_seed_noise'):
            power_warn += ' [DELTA WITHIN PAIRED SEED NOISE]'
        if mde is not None and entry.get('delta_seed_mean') is not None:
            if abs(entry['delta_seed_mean']) < mde:
                power_warn += ' [UNDERPOWERED: |delta| < mde80]'
        note = '' if entry['replicated_across_seeds'] else ' [descriptive only, not certified]'
        interaction = ''
        if entry.get('kind') == 'interaction':
            p2 = entry.get('p_two_sided')
            p2_holm = entry.get('p_two_sided_holm')
            interaction = (f" interaction={entry.get('interaction_direction')} "
                           f"p(two-sided)={'n/a' if p2 is None else f'{p2:.3g}'} "
                           f"p2(holm)={'n/a' if p2_holm is None else f'{p2_holm:.3g}'} "
                           f"differs={entry.get('interaction_differs_significantly')}")
        print(f"[factorial] {entry['design']} {entry['kind']} {entry['effect']}: "
              f"delta={entry['delta_seed_mean']:+.4f} {ci_str}"
              f"seeds_favoring={entry['n_seeds_favoring']}/{entry['n_seeds']} "
              f"p(one-sided)={p_str} p(holm)={p_holm_str} {mde_str}{ratio_str}"
              f"helps={entry['effect_helps']} "
              f"significant={entry['effect_helps_significantly']}{note}{power_warn}"
              f"{interaction}{_robustness_note(entry)}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[factorial] NOTE: terms with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared "
              f"seeds are described but never certified: "
              f"{comparison['rows_tested_descriptively_only']}")

def _factorial_term_coefficients(factors, levels: dict, cells: dict,
                                 term_factors) -> dict:
    """Orthogonal (sum-to-zero) coefficients for one factorial term.

    Levels are coded -1 / +1 in declaration order. A term over `t` of the
    `k` factors multiplies the cell's product code by ``2**-(k-t)``, which
    averages over the factors not in the term. For a 2x2x2 design the
    resulting coefficients are the usual ``+1/-1`` contrasts for interactions
    and ``+/-1/4`` for main effects, without hard-coding each formula.
    """
    term = set(term_factors)
    unknown = term - set(factors)
    if unknown:
        raise ValueError(f'factorial term names unknown factor(s) {sorted(unknown)}')
    if not term:
        raise ValueError('a factorial term must name at least one factor')
    scale = 1.0 / (2 ** (len(factors) - len(term)))
    out = {}
    for cell_key in cells:
        assignment = _parse_three_way_cell_key(cell_key)
        code = 1
        for factor in term:
            factor_levels = levels[factor]
            if assignment.get(factor) not in factor_levels:
                raise ValueError(
                    f'factorial cell {cell_key!r} has no valid level for {factor!r}')
            code *= 1 if assignment[factor] == factor_levels[1] else -1
        out[cell_key] = code * scale
    return out

def three_way_factorial_comparison(summaries: list, spec: dict | None=None,
                                   margin: float=0.0, alpha: float=0.05,
                                   equivalence_margin: float=0.0) -> dict | None:
    """Paired main effects and all interactions of a 2x2x2 ablation design.

    The two 2x2 reports estimate per-channel x depth interactions separately
    under generalized softsign and under tanh. This function pools the eight
    cells into one 2x2x2 design and reports:

    * the per-channel and depth main effects, averaged over both activation
      families (more efficient than either 2x2 estimate alone);
    * all three two-way interactions, including the per-channel x activation
      and depth x activation terms that no existing report estimates; and
    * the three-way interaction, i.e. whether the per-channel x depth
      interaction itself differs between the activation families.

    Every term is a per-seed linear contrast of the eight cell means (levels
    coded -1/+1), tested with the same paired t reference as the rest of the
    module; only >= 3 shared seeds can certify. Interactions are two-sided
    (synergy and antagonism are both findings), and each pairwise interaction
    also carries a TOST equivalence test. The orthogonality block therefore
    only returns a verdict when a positive `equivalence_margin` was
    pre-specified: "no significant interaction" is not evidence that the axes
    are additive, whereas an equivalence test within a stated loss margin is.
    """
    spec = THREE_WAY_FACTORIAL_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r
              and _finite_metric(r.get('val_loss'))}
    rows: list = []
    design_reports: dict = {}
    for design_name, design in spec.items():
        factors = tuple(design.get('factors') or ())
        levels = design.get('levels') or {}
        cells = design.get('cells') or {}
        base = {'design': design_name, 'factors': list(factors),
                'design_type': design.get('design_type', '2x2x2'),
                'context': design.get('context'), 'shared_seeds': []}
        if len(factors) != 3:
            rows.append(dict(base, effect=None, kind='design', skipped=(
                f'design must have exactly three factors, got {list(factors)}')))
            continue
        missing = [key for key, row in cells.items() if row not in usable]
        if missing:
            rows.append(dict(base, effect=None, kind='design', skipped=(
                f"cell row(s) {sorted({cells[key] for key in missing})} have no "
                f"usable val_loss")))
            continue
        seed_values = {key: _seed_means(usable[row]) for key, row in cells.items()}
        common = sorted(set.intersection(
            *(set(values) for values in seed_values.values())))
        if len(common) < 2:
            rows.append(dict(base, effect=None, kind='design',
                             shared_seeds=common, skipped=(
                                 f'only {len(common)} seed(s) are shared by all '
                                 f'eight cells; a paired contrast needs at least 2')))
            continue
        design_reports[design_name] = {
            'design_type': design.get('design_type', '2x2x2'),
            'factors': list(factors),
            'levels': {factor: list(levels[factor]) for factor in factors},
            'cells': dict(cells),
            'shared_seeds': common,
        }

        def _term_values(term_factors) -> dict:
            coefficients = _factorial_term_coefficients(
                factors, levels, cells, term_factors)
            return {
                seed: sum(coefficient * seed_values[key][seed]
                          for key, coefficient in coefficients.items())
                for seed in common
            }

        terms: list[tuple[tuple, str, str]] = []
        for factor in factors:
            if factor in ('per_channel', 'depth'):
                terms.append(((factor,), 'main_effect',
                              'mean over both levels of the other two factors: '
                              'on - off'))
        for first, second in itertools.combinations(factors, 2):
            others = [name for name in factors if name not in (first, second)]
            terms.append(((first, second), 'interaction',
                          f'{first} x {second} interaction, averaged over '
                          f'{others[0] if others else "the remaining factor"}'))
        if len(factors) == 3:
            terms.append((tuple(factors), 'interaction',
                          'three-way interaction: change in the '
                          f'{factors[0]} x {factors[1]} interaction between '
                          f'the two levels of {factors[2]}'))
        for term_factors, kind, formula in terms:
            per_seed = _term_values(term_factors)
            values = [per_seed[seed] for seed in common]
            test = paired_t_one_sided(values, [0.0] * len(values),
                                      margin=margin, alpha=alpha)
            two_sided_test = paired_t_one_sided(
                values, [0.0] * len(values), margin=0.0, alpha=alpha)
            p_two_sided = two_sided_p_from_t(
                two_sided_test['t'], two_sided_test['df'])
            replicated = len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
            effect = '_x_'.join(term_factors)
            entry = dict(
                base, effect=effect, kind=kind, term_factors=list(term_factors),
                # Only the interaction terms are added to the pre-registered
                # plan. The pooled main effects are supporting estimates: the
                # same factors are already pre-registered inside each 2x2
                # design, and registering a third, highly correlated estimate
                # would inflate the study family without adding a new claim.
                analysis_role=('supporting_pooled_main_effect'
                               if kind == 'main_effect'
                               else 'cross_axis_interaction'),
                formula=formula, shared_seeds=common, n_seeds=len(common),
                delta_seed_mean=sum(values) / len(values),
                per_seed_delta={str(seed): value
                                for seed, value in per_seed.items()},
                n_seeds_favoring=sum(value < 0.0 for value in values),
                n_seeds_tied=sum(value == 0.0 for value in values),
                n_seeds_against=sum(value > 0.0 for value in values),
                effect_helps=bool(sum(values) / len(values) < -margin),
                effect_helps_significantly=False,
                test=test['test'],
                test_basis=('seed_means' if replicated
                            else f'seed_means_{len(common)}_descriptive'),
                replicated_across_seeds=replicated,
                p_one_sided=test['p_one_sided'],
                p_one_sided_holm=None,
                p_two_sided=p_two_sided,
                p_two_sided_holm=None,
                interaction_direction=None,
                interaction_differs_significantly=None,
                orthogonal_within_margin=None,
                t=test['t'], df=test['df'],
                delta_ci_95=[test['ci_low'], test['ci_high']],
                n=test['n'])
            if kind == 'interaction':
                delta = entry['delta_seed_mean']
                if delta < -margin:
                    entry['interaction_direction'] = 'negative'
                elif delta > margin:
                    entry['interaction_direction'] = 'positive'
                else:
                    entry['interaction_direction'] = 'additive_or_within_margin'
            effect_row = {'per_seed': [
                {'seed': int(seed), 'val_loss': value}
                for seed, value in per_seed.items()]}
            zero_row = {'per_seed': [
                {'seed': int(seed), 'val_loss': 0.0} for seed in per_seed]}
            entry.update(_paired_effect_summary(
                effect_row, zero_row, common, test, alpha=alpha, margin=margin))
            entry.update(_robustness_for_deltas(
                {str(seed): float(value) for seed, value in per_seed.items()},
                alpha=alpha, equivalence_margin=equivalence_margin))
            paired_std = entry.get('paired_delta_std')
            if (isinstance(paired_std, (int, float)) and paired_std > 0.0
                    and entry.get('delta_seed_mean') is not None):
                entry['delta_over_paired_std'] = (
                    abs(entry['delta_seed_mean']) / paired_std)
                entry['within_paired_seed_noise'] = bool(
                    abs(entry['delta_seed_mean']) < paired_std)
            else:
                entry['delta_over_paired_std'] = None
                entry['within_paired_seed_noise'] = None
            if kind == 'interaction':
                equivalence = entry.get('equivalence') or {}
                if equivalence.get('enabled'):
                    entry['orthogonal_within_margin'] = bool(
                        equivalence.get('equivalent'))
            rows.append(entry)
    if not rows:
        return None
    family = [entry for entry in rows
              if entry.get('replicated_across_seeds') and not entry.get('skipped')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['effect_helps_significantly'] = bool(
            entry['effect_helps'] and p_holm is not None and p_holm < alpha)
    interaction_family = [entry for entry in family
                          if entry.get('kind') == 'interaction']
    interaction_adjusted = holm_adjust(
        [entry.get('p_two_sided') for entry in interaction_family])
    for entry, p_holm in zip(interaction_family, interaction_adjusted):
        entry['p_two_sided_holm'] = p_holm
        entry['interaction_differs_significantly'] = bool(
            p_holm is not None and p_holm < alpha)
    pairwise = [entry for entry in interaction_family
                if len(entry.get('term_factors') or ()) == 2]
    triple = [entry for entry in interaction_family
              if len(entry.get('term_factors') or ()) == 3]
    interactions_block = {
        entry['effect']: {
            'delta_seed_mean': entry['delta_seed_mean'],
            'interaction_direction': entry['interaction_direction'],
            'differs_significantly': entry['interaction_differs_significantly'],
            'equivalent': (entry.get('equivalence') or {}).get('equivalent'),
            'orthogonal_within_margin': entry.get('orthogonal_within_margin'),
        }
        for entry in pairwise + triple
    }
    if equivalence_margin and equivalence_margin > 0.0:
        if pairwise:
            pairwise_verdict = (
                all(entry.get('orthogonal_within_margin') is not None
                    for entry in pairwise)
                and all(entry.get('orthogonal_within_margin')
                        for entry in pairwise))
            orthogonality_note = (
                'pairwise axes are orthogonal within +/- '
                f'{equivalence_margin:g} only when every pairwise interaction '
                'TOST passes; a non-significant interaction alone is not '
                'equivalence')
        else:
            pairwise_verdict = None
            orthogonality_note = 'no pairwise interaction could be tested'
        if triple:
            triple_equiv = (triple[0].get('equivalence') or {}).get('equivalent')
        else:
            triple_equiv = None
    else:
        pairwise_verdict = None
        triple_equiv = None
        orthogonality_note = (
            'no positive --equivalence-margin was pre-specified, so the axes '
            'cannot be claimed orthogonal from non-significant interactions; '
            'the interaction estimates are reported descriptively')
    tested_descriptively = [entry['effect'] for entry in rows
                            if entry.get('p_one_sided') is not None
                            and not entry.get('replicated_across_seeds')]
    return {
        'schema': 'adasquash.three_way_factorial.v1',
        'protocol': (
            '2x2x2 per-seed orthogonal contrasts on the seeds shared by all '
            'eight cells; levels coded -1/+1, main effects averaged over the '
            'other two factors; interactions are two-sided and Holm-corrected '
            'as their own family. Only >= '
            f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds can certify. The four '
            'interaction terms are pre-registered; the pooled main effects '
            'carry analysis_role=supporting_pooled_main_effect and are not '
            'counted as new confirmatory claims.'),
        'margin': margin,
        'alpha': alpha,
        'equivalence_margin': equivalence_margin,
        'designs': design_reports,
        'rows': rows,
        'multiple_comparison': (
            f'Holm-Bonferroni over {len(family)} seed-replicated term(s) '
            'using the one-sided p-value for the directional `helps` claim'),
        'interaction_multiple_comparison': (
            f'Holm-Bonferroni over {len(interaction_family)} seed-replicated '
            'interaction(s) using the two-sided p-value; covers negative and '
            'positive interactions'),
        'orthogonality': {
            'margin': equivalence_margin,
            'interactions': interactions_block,
            'axes_orthogonal_within_margin': pairwise_verdict,
            'three_way_interaction_absent_within_margin': triple_equiv,
            'note': orthogonality_note,
        },
        'reading_guide': (
            'levels are coded -1/+1 in declaration order (per_channel off/on, '
            'depth off/on, activation softsign/tanh). A negative main effect '
            'means switching the factor on lowers val_loss. A negative '
            'two-way interaction means the combined effect is more negative '
            'than the sum of the separate effects (synergy); a positive one '
            'means antagonism. For an interaction involving activation, '
            'negative means the effect is stronger under tanh than softsign '
            '(coded +1); for the three-way term, negative means the '
            'per-channel x depth interaction is more negative under tanh than '
            'under softsign. `orthogonal_within_margin` is a TOST verdict and '
            'is None when no positive equivalence margin was registered.'),
        'rows_tested_descriptively_only': tested_descriptively,
    }

def print_three_way_factorial_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[factorial3] no 2x2x2 factorial design declared; '
              'cross-axis interaction test skipped')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[factorial3] {entry['design']}: SKIPPED ({entry['skipped']})")
            continue
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm = ('n/a' if entry['p_one_sided_holm'] is None
                  else f"{entry['p_one_sided_holm']:.3g}")
        p2 = entry.get('p_two_sided')
        p2_holm = entry.get('p_two_sided_holm')
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        note = ('' if entry['replicated_across_seeds']
                else ' [descriptive only, not certified]')
        interaction = ''
        if entry.get('kind') == 'interaction':
            interaction = (
                f" direction={entry.get('interaction_direction')} "
                f"p(two-sided)={'n/a' if p2 is None else f'{p2:.3g}'} "
                f"p2(holm)={'n/a' if p2_holm is None else f'{p2_holm:.3g}'} "
                f"differs={entry.get('interaction_differs_significantly')} "
                f"orthogonal_within_margin={entry.get('orthogonal_within_margin')}")
        print(f"[factorial3] {entry['effect']} ({entry['kind']}): "
              f"delta={entry['delta_seed_mean']:+.4f} {ci_str}"
              f"seeds_favoring={entry['n_seeds_favoring']}/{entry['n_seeds']} "
              f"p(one-sided)={p_str} p(holm)={p_holm} "
              f"helps={entry['effect_helps']} "
              f"significant={entry['effect_helps_significantly']}{note}"
              f"{interaction}{_robustness_note(entry)}")
    orthogonality = comparison.get('orthogonality') or {}
    print(f"[factorial3] orthogonality: axes_orthogonal_within_margin="
          f"{orthogonality.get('axes_orthogonal_within_margin')} "
          f"three_way_absent_within_margin="
          f"{orthogonality.get('three_way_interaction_absent_within_margin')} "
          f"margin={orthogonality.get('margin')}")
    if orthogonality.get('note'):
        print(f"[factorial3] note: {orthogonality['note']}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[factorial3] NOTE: terms with fewer than "
              f"{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are described but "
              f"never certified: {comparison['rows_tested_descriptively_only']}")

def build_analysis_plan(rows: dict, hypotheses: dict, matrix_report: dict,
                        baseline_names, model_name: str, seeds,
                        alpha: float=0.05, margin: float=0.0,
                        equivalence_margin: float=0.0,
                        run_schedule: dict | None=None,
                        exclusion_policy: dict | None=None) -> dict:
    """Freeze the confirmatory contrast registry *before* any training starts.

    A comparison experiment is only confirmatory if the set of tests is fixed
    in advance. Discovering an extra row after seeing the results and then
    correcting only the surviving family is a well-known way to manufacture a
    "significant" ablation. This function derives every mechanism / baseline /
    factorial / 2x2x2 interaction / QK contrast from the already-validated
    matrix (so it cannot name a contrast the matrix does not isolate), records
    the stated
    hypothesis and test direction, and hashes the whole registry. The hash is
    written next to the results; `study_level_correction` only admits tests
    whose id appears here, and everything else is reported as unregistered
    (exploratory) even when its p-value is small.

    The baseline reference is a *rule* ("best replicated baseline"), not a
    single row name, because which baseline is strongest is a result of the
    run; the realized reference is recorded in `baseline_comparison.json`.
    The exclusion policy is normalized and hashed into the same fingerprint:
    a rerun that quietly changes which runs may be dropped is a changed
    experiment, not a re-analysis.
    """
    baseline_set = set(baseline_names or [])
    contrasts: list = []
    seen_directional_pairs: set[tuple] = set()

    def _add(entry: dict) -> None:
        # The same candidate/reference contrast can be described by more than
        # one report family (the QK axis is both a leave-one-out removal and a
        # host-level QK contrast). It is one test, so the pre-registration and
        # the study-level Holm family must count it once; otherwise the
        # duplicated p-value doubles its weight and inflates the family size.
        if entry.get('kind') in ('superiority', 'mechanism_removal'):
            pair = (entry.get('candidate'), entry.get('reference'))
            if pair[0] is not None and pair[1] is not None:
                if pair in seen_directional_pairs:
                    return
                seen_directional_pairs.add(pair)
        contrasts.append(entry)

    for name in sorted(rows):
        if name in baseline_set:
            continue
        if not str(hypotheses.get(name) or '').strip():
            # `validate_matrix` already rejects this, but a direct call must
            # not silently drop a contrast from the pre-registration: a plan
            # that omits a row would let the missing hypothesis disappear and
            # later be reported as unregistered rather than as an error.
            raise ValueError(
                f"ablation row '{name}' has no stated scientific hypothesis; "
                "the analysis plan must not silently omit it")
        _add({
            'id': f'baseline:{name}',
            'family': 'baseline',
            'kind': 'superiority',
            'candidate': name,
            'reference': 'best_replicated_baseline',
            'metric': 'val_loss',
            'sided': 'one_sided',
            'hypothesis': hypotheses.get(name),
        })
    qk_axis_report = matrix_report.get('qk_axis') or {}
    qk_on_off_pairs = {
        (info.get('on'), info.get('off')) for info in qk_axis_report.values()
        if isinstance(info, dict)}
    for full_name, mechanisms in sorted(
            (matrix_report.get('leave_one_out') or {}).items()):
        for mechanism, reference in sorted((mechanisms or {}).items()):
            if (full_name, reference) in qk_on_off_pairs:
                # Canonicalised by the host-level QK-axis contrast below: the
                # LOO report still describes the same test, but the plan must
                # not pre-register it twice.
                continue
            _add({
                'id': f'loo:{full_name}:{mechanism}',
                'family': 'leave_one_out',
                'kind': 'mechanism_removal',
                'candidate': full_name,
                'reference': reference,
                'metric': 'val_loss',
                'sided': 'one_sided',
                'hypothesis': (f'removing {mechanism} from {full_name} raises '
                               f'held-out val_loss'),
            })
    for design, info in sorted((matrix_report.get('factorial') or {}).items()):
        factor_names = list((info or {}).get('factors') or {})
        for factor in factor_names:
            _add({
                'id': f'factorial:{design}:{factor}',
                'family': 'factorial',
                'kind': 'main_effect',
                'design': design,
                'factor': factor,
                'metric': 'val_loss',
                'sided': 'one_sided',
                'hypothesis': f'{factor} lowers held-out val_loss in {design}',
            })
        if len(factor_names) == 2:
            first, second = factor_names
            _add({
                'id': f'factorial:{design}:{first}_x_{second}',
                'family': 'factorial',
                'kind': 'interaction',
                'design': design,
                'factor': f'{first}_x_{second}',
                'metric': 'val_loss',
                'sided': 'two_sided',
                'hypothesis': (f'{first} and {second} interact (synergy or '
                               f'antagonism) in {design}'),
            })
    # The 2x2x2 design adds the two interactions that involve the activation
    # family and the three-way term. Main effects are deliberately not
    # re-registered: they are already pre-registered by the two 2x2 designs,
    # and registering the pooled estimate as well would count the same
    # mechanism twice in the study-level Holm family. The pooled 2x2x2
    # interaction terms are different estimands from the within-family 2x2
    # interactions, so they are new tests, not duplicates.
    for design, info in sorted(
            (matrix_report.get('three_way_factorial') or {}).items()):
        factor_names = list((info or {}).get('factors') or [])
        for first, second in itertools.combinations(factor_names, 2):
            _add({
                'id': f'factorial3:{design}:{first}_x_{second}',
                'family': 'factorial3',
                'kind': 'interaction',
                'design': design,
                'factor': f'{first}_x_{second}',
                'metric': 'val_loss',
                'sided': 'two_sided',
                'hypothesis': (f'{first} and {second} interact when the third '
                               f'axis is pooled over in {design}'),
            })
        if len(factor_names) == 3:
            first, second, third = factor_names
            _add({
                'id': (f'factorial3:{design}:{first}_x_{second}_x_{third}'),
                'family': 'factorial3',
                'kind': 'interaction',
                'design': design,
                'factor': f'{first}_x_{second}_x_{third}',
                'metric': 'val_loss',
                'sided': 'two_sided',
                'hypothesis': (f'the {first} x {second} interaction differs '
                               f'between the two levels of {third} in {design}'),
            })
    for host, info in sorted((matrix_report.get('qk_axis') or {}).items()):
        info = info or {}
        _add({
            'id': f'qk:{host}:qk_vs_off',
            'family': 'qk_axis',
            'kind': 'superiority',
            'candidate': info.get('on'),
            'reference': info.get('off'),
            'metric': 'val_loss',
            'sided': 'one_sided',
            'hypothesis': f'tanh QK compression lowers held-out val_loss on {host}',
        })
        _add({
            'id': f'qk:{host}:qknorm_vs_qk',
            'family': 'qk_axis',
            'kind': 'comparator',
            'candidate': info.get('comparator'),
            'reference': info.get('on'),
            'metric': 'val_loss',
            'sided': 'one_sided',
            'hypothesis': (f'per-head QK-Norm beats the tanh QK compression on '
                           f'{host}'),
        })
    # Hierarchical gatekeeping: the main proposal's own superiority and
    # mechanism contrasts are the primary family. Only if at least one of
    # them survives its Holm correction may the secondary contrasts (single
    # axis variants, control factorials, QK axis, hyper-parameter sweeps) be
    # tested at full alpha. Gatekeeping keeps the study-wide false-positive
    # rate at alpha without making every exploratory sweep compete with the
    # primary claim for the same alpha budget.
    for entry in contrasts:
        primary = (
            (entry['family'] == 'baseline'
             and entry.get('candidate') == PRIMARY_MAIN_ROW)
            or (entry['family'] == 'leave_one_out'
                and entry.get('candidate') == PRIMARY_MAIN_ROW)
            or (entry['family'] == 'factorial'
                and entry.get('design') == PRIMARY_FACTORIAL_DESIGN)
        )
        entry['tier'] = 1 if primary else 2
        entry['tier_role'] = 'primary' if primary else 'secondary'
    # The exclusion policy is a design decision too: changing which runs may
    # be dropped changes the analysed sample, so it belongs in the same
    # fingerprint as the contrasts and the randomized schedule.
    exclusion_policy = normalize_exclusion_policy(exclusion_policy)
    payload = {
        'model': model_name,
        'seeds': [int(seed) for seed in seeds],
        'alpha': float(alpha),
        'margin': float(margin),
        'equivalence_margin': float(equivalence_margin),
        'baseline_pool': sorted(baseline_set),
        'exclusion_policy': exclusion_policy,
        'exclusion_policy_fingerprint': exclusion_policy_fingerprint(exclusion_policy),
        # The contrast *ids* alone do not identify the experiment: two
        # matrices can reuse the same row names and hypotheses while changing
        # a model knob (e.g. alpha_init), or can map a factorial design to
        # different cells. Hash those definitions too, or a post-hoc model
        # change would not move the pre-registration fingerprint.
        'row_configs': {name: rows.get(name) for name in sorted(rows)},
        'matrix_cells': {
            key: matrix_report.get(key)
            for key in ('leave_one_out', 'factorial', 'three_way_factorial',
                        'qk_axis')
        },
        # The randomized execution order is part of the pre-registration: a
        # schedule chosen after seeing which arms look good is a design
        # degree of freedom exactly like choosing the contrasts afterwards.
        'run_schedule': run_schedule,
        'contrasts': contrasts,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
    return {
        'schema': 'adasquash.analysis_plan.v1',
        'fingerprint': fingerprint,
        'n_contrasts': len(contrasts),
        'contrast_ids': [entry['id'] for entry in contrasts],
        **payload,
        'protocol': (
            'confirmatory contrasts are fixed before training and identified by '
            '`id`; the study-level Holm correction admits only these ids, so a '
            'realized test absent from this plan is exploratory and cannot be '
            'counted as a confirmed claim'),
    }


def _finite_p(value):
    if (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value))):
        return float(value)
    return None


def study_level_correction(comparisons: dict, plan: dict | None,
                           alpha: float=0.05) -> dict | None:
    """Study-level gatekeeping over the pre-registered confirmatory contrasts.

    The per-family reports each correct within their own family (baseline
    rows, leave-one-out mechanisms, factorial terms, QK pairs). That is the
    right correction for a claim scoped to one family, but the study as a
    whole makes all of those claims at once - and they reuse the same seeds,
    so the tests are correlated. This function pools the pre-registered
    confirmatory p-values into a primary tier (the main proposal's superiority,
    its mechanisms and its main 2x2 factorial) and a secondary tier. The
    primary tier is Holm-corrected first; the secondary tier (including the
    cross-axis 2x2x2 interactions) is corrected and tested only if at least
    one primary contrast survives. Gatekeeping controls the study-wide
    false-positive rate while keeping exploratory single-axis / control / QK
    contrasts from competing with the primary claim for the same alpha budget.
    The family-local adjusted p stays next to the study-level one, and every
    demoted contrast (or secondary contrast not tested because the gate closed)
    is listed explicitly rather than silently dropped.

    Interactions enter with their two-sided p-value (either synergy or
    antagonism is a finding); main effects and superiority contrasts enter
    with the one-sided p-value that matches their directional claim. Tests
    that touch fewer than `MIN_SEEDS_FOR_SIGNIFICANCE` seeds, duplicate runs
    (aliases) and ids absent from the plan are excluded from the family and
    reported with the reason. A focused run with no primary contrast gets
    `primary_gate_status = not_applicable_no_primary_tests` (the secondary
    tier is still tested) rather than a misleading "gate closed". Returns
    `None` without a plan.
    """
    if not isinstance(plan, dict):
        return None
    planned = {entry['id']: entry for entry in plan.get('contrasts', [])
               if isinstance(entry, dict) and entry.get('id')}
    records: list = []

    def _record(cid, family, p, family_p, claim_direction, family_significant,
                *, hypothesis=None, replicated=False, test_basis=None,
                effect=None, excluded_reason=None):
        records.append({
            'id': cid,
            'family': family,
            'p_one_sided_or_two_sided': _finite_p(p),
            'p_family_holm': _finite_p(family_p),
            'claim_direction_observed': bool(claim_direction),
            'family_local_significant': bool(family_significant),
            'replicated_across_seeds': bool(replicated),
            'test_basis': test_basis,
            'effect': effect,
            'hypothesis': hypothesis,
            'excluded_from_study_family': excluded_reason,
            'p_study_holm': None,
            'confirmed_at_study_level': False,
            'downgraded_by_study_level': False,
        })

    baseline = comparisons.get('baseline') or {}
    for entry in baseline.get('rows', []) or []:
        name = entry.get('name')
        if not name:
            continue
        _record(
            f'baseline:{name}', 'baseline', entry.get('p_one_sided'),
            entry.get('p_one_sided_holm'), entry.get('beats_reference'),
            entry.get('beats_reference_significantly'),
            hypothesis=entry.get('hypothesis'),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            # The certifying p-value is the paired shared-seed test, so the
            # study-level effect must be that same paired delta. Falling back
            # to the all-seed difference can even reverse its sign.
            effect=(entry.get('delta_paired_mean')
                    if entry.get('delta_paired_mean') is not None
                    else entry.get('delta_vs_reference')),
            excluded_reason=entry.get('excluded_from_family'))
    loo = comparisons.get('leave_one_out') or {}
    for entry in loo.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('full'):
            continue
        _record(
            f"loo:{entry['full']}:{entry['mechanism']}", 'leave_one_out',
            entry.get('p_one_sided'), entry.get('p_one_sided_holm'),
            entry.get('mechanism_helps'),
            entry.get('mechanism_helps_significantly'),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            effect=(entry.get('delta_paired_mean')
                    if entry.get('delta_paired_mean') is not None
                    else entry.get('delta_full_minus_reference')),
            excluded_reason=entry.get('excluded_from_family'))
    factorial = comparisons.get('factorial') or {}
    for entry in factorial.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('design'):
            continue
        is_interaction = entry.get('kind') == 'interaction'
        _record(
            f"factorial:{entry['design']}:{entry['effect']}", 'factorial',
            entry.get('p_two_sided') if is_interaction else entry.get('p_one_sided'),
            (entry.get('p_two_sided_holm') if is_interaction
             else entry.get('p_one_sided_holm')),
            (entry.get('interaction_direction') not in
             (None, 'additive_or_within_margin'))
            if is_interaction else entry.get('effect_helps'),
            (entry.get('interaction_differs_significantly') if is_interaction
             else entry.get('effect_helps_significantly')),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            effect=entry.get('delta_seed_mean'))
    factorial3 = comparisons.get('factorial3') or {}
    for entry in factorial3.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('design'):
            continue
        is_interaction = entry.get('kind') == 'interaction'
        _record(
            f"factorial3:{entry['design']}:{entry['effect']}", 'factorial3',
            entry.get('p_two_sided') if is_interaction else entry.get('p_one_sided'),
            (entry.get('p_two_sided_holm') if is_interaction
             else entry.get('p_one_sided_holm')),
            (entry.get('interaction_direction') not in
             (None, 'additive_or_within_margin'))
            if is_interaction else entry.get('effect_helps'),
            (entry.get('interaction_differs_significantly') if is_interaction
             else entry.get('effect_helps_significantly')),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            effect=entry.get('delta_seed_mean'))
    qk = comparisons.get('qk_axis') or {}
    for entry in qk.get('rows', []) or []:
        if entry.get('skipped') or not entry.get('host'):
            continue
        _record(
            f"qk:{entry['host']}:{entry['contrast']}", 'qk_axis',
            entry.get('p_one_sided'), entry.get('p_one_sided_holm'),
            entry.get('effect_helps'), entry.get('effect_helps_significantly'),
            hypothesis=entry.get('hypothesis'),
            replicated=entry.get('replicated_across_seeds'),
            test_basis=entry.get('test_basis'),
            effect=entry.get('delta_paired_mean',
                             entry.get('delta_candidate_minus_reference')))

    registered: list = []
    unregistered: list = []
    for record in records:
        planned_entry = planned.get(record['id'])
        if planned_entry is None:
            record['excluded_from_study_family'] = (
                'not present in the pre-registered analysis plan '
                '(exploratory/unregistered)')
            unregistered.append(record['id'])
        else:
            record['plan_kind'] = planned_entry.get('kind')
            registered.append(record)
    resolved_ids = {record['id'] for record in registered}
    unresolved = sorted(set(planned) - resolved_ids)
    eligible = [record for record in registered
                if record.get('excluded_from_study_family') is None
                and record.get('replicated_across_seeds')
                and record.get('p_one_sided_or_two_sided') is not None]
    eligible_ids = {record['id'] for record in eligible}
    for record in registered:
        if record['id'] not in eligible_ids:
            if record.get('excluded_from_study_family') is None:
                if not record.get('replicated_across_seeds'):
                    record['excluded_from_study_family'] = (
                        f'fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds; '
                        'descriptive only')
                elif record.get('p_one_sided_or_two_sided') is None:
                    record['excluded_from_study_family'] = 'no testable p-value'
    primary = [record for record in eligible
               if (planned.get(record['id']) or {}).get('tier', 1) == 1]
    secondary = [record for record in eligible
                 if (planned.get(record['id']) or {}).get('tier', 1) == 2]
    primary_adjusted = holm_adjust(
        [record['p_one_sided_or_two_sided'] for record in primary])
    for record, p_holm in zip(primary, primary_adjusted):
        record['tier'] = 'primary'
        record['p_study_holm'] = p_holm
        record['confirmed_at_study_level'] = bool(
            record['claim_direction_observed'] and p_holm is not None
            and p_holm < alpha)
    primary_gate_open = any(
        record['confirmed_at_study_level'] for record in primary)
    # A focused run (e.g. only the QK rows) has no primary tier at all. That
    # is not a failed gate: the gate is only meaningful when the primary
    # family was part of the study. Distinguish "not applicable" from
    # "closed, so secondary claims are not licensed".
    if not primary:
        primary_gate_status = 'not_applicable_no_primary_tests'
    elif primary_gate_open:
        primary_gate_status = 'open'
    else:
        primary_gate_status = 'closed'
    secondary_allowed = primary_gate_status in ('open',
                                                'not_applicable_no_primary_tests')
    if secondary_allowed:
        secondary_adjusted = holm_adjust(
            [record['p_one_sided_or_two_sided'] for record in secondary])
    else:
        secondary_adjusted = [None] * len(secondary)
    for record, p_holm in zip(secondary, secondary_adjusted):
        record['tier'] = 'secondary'
        record['p_study_holm'] = p_holm
        record['confirmed_at_study_level'] = bool(
            secondary_allowed and record['claim_direction_observed']
            and p_holm is not None and p_holm < alpha)
        if not secondary_allowed:
            record['gate_status'] = (
                'secondary family not tested: no primary contrast survived '
                'its gatekeeping correction, so the study has no confirmed '
                'primary claim to license secondary claims')
    for record in records:
        if 'tier' not in record:
            planned_entry = planned.get(record['id'])
            if planned_entry is None:
                record['tier'] = 'unregistered'
            else:
                record['tier'] = ('primary' if planned_entry.get('tier', 1) == 1
                                  else 'secondary')
        record['downgraded_by_study_level'] = bool(
            record['family_local_significant']
            and not record['confirmed_at_study_level'])
    secondary_gate_closed = sorted(
        record['id'] for record in secondary if not secondary_allowed)
    local_significant = sorted(record['id'] for record in registered
                               if record.get('family_local_significant'))
    study_significant = sorted(record['id'] for record in primary + secondary
                               if record.get('confirmed_at_study_level'))
    downgraded = [
        {'id': record['id'],
         'family': record['family'],
         'tier': record.get('tier'),
         'p_family_holm': record['p_family_holm'],
         'p_study_holm': record['p_study_holm']}
        for record in primary + secondary if record['downgraded_by_study_level']
    ]
    records.sort(key=lambda record: record['id'])
    return {
        'schema': 'adasquash.study_level_tests.v1',
        'analysis_plan_fingerprint': plan.get('fingerprint'),
        'alpha': float(alpha),
        'n_planned_contrasts': len(planned),
        'n_registered_tests': len(registered),
        'n_in_study_family': len(primary) + len(secondary),
        'n_primary_tests': len(primary),
        'n_secondary_tests': len(secondary),
        'primary_gate_open': bool(primary_gate_open),
        'primary_gate_status': primary_gate_status,
        'n_confirmed_primary': sum(
            record.get('confirmed_at_study_level') for record in primary),
        'n_confirmed_secondary': sum(
            record.get('confirmed_at_study_level') for record in secondary),
        'n_confirmed_at_study_level': len(study_significant),
        'family_local_significant': local_significant,
        'study_level_significant': study_significant,
        'downgraded_by_study_level': downgraded,
        'secondary_not_tested_gate_closed': secondary_gate_closed,
        'unregistered_contrasts': sorted(unregistered),
        'unresolved_plan_contrasts': unresolved,
        'protocol': (
            'hierarchical Holm gatekeeping over the pre-registered confirmatory '
            'contrasts. Tier 1 (primary) = the main proposal vs the best baseline, '
            'its leave-one-out mechanisms, and the main softsign factorial. Only '
            'if >= 1 primary contrast survives does the secondary tier (single-axis '
            'variants, control factorials, QK axis, hyper-parameter sweeps) get '
            'tested at alpha; both tiers are Holm-corrected within themselves. '
            '`confirmed_at_study_level` additionally requires the claimed '
            'direction. Interactions use the two-sided p-value, everything else '
            'the one-sided p-value matching its directional hypothesis. '
            'Family-local adjusted p-values are retained, so a demotion by the '
            'study-level correction is visible rather than silent'),
        'rows': records,
    }


def print_study_level_correction(comparison: dict | None) -> None:
    if comparison is None:
        print('[study] no analysis plan; study-level multiplicity correction skipped')
        return
    print(f"[study] {comparison['n_in_study_family']}/{comparison['n_planned_contrasts']} "
          f"pre-registered contrast(s) tested "
          f"(primary={comparison['n_primary_tests']}, "
          f"secondary={comparison['n_secondary_tests']}); "
          f"family-local significant={len(comparison['family_local_significant'])}, "
          f"study-level confirmed={comparison['n_confirmed_at_study_level']} "
          f"(primary gate {comparison.get('primary_gate_status', 'open').upper()}; "
          f"plan {str(comparison.get('analysis_plan_fingerprint'))[:12]})")
    for entry in comparison.get('downgraded_by_study_level') or []:
        print(f"[study] DOWNGRADED {entry['id']}: family p_holm="
              f"{entry['p_family_holm']} -> study p_holm={entry['p_study_holm']} "
              f"(not confirmed once the whole study is corrected)")
    if comparison.get('secondary_not_tested_gate_closed'):
        print(f"[study] secondary contrasts not tested (primary gate closed): "
              f"{comparison['secondary_not_tested_gate_closed']}")
    if comparison.get('unregistered_contrasts'):
        ids = comparison['unregistered_contrasts']
        preview = f"{ids[:8]}{' ...' if len(ids) > 8 else ''}"
        print(f"[study] unregistered/exploratory ({len(ids)}, not in the plan, "
              f"not counted): {preview}")
    if comparison.get('unresolved_plan_contrasts'):
        ids = comparison['unresolved_plan_contrasts']
        preview = f"{ids[:8]}{' ...' if len(ids) > 8 else ''}"
        print(f"[study] pre-registered but not realized ({len(ids)}; full list in "
              f"study_level_tests.json): {preview}")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
    ablations = dict(ABLATIONS)
    hypotheses = dict(ABLATION_HYPOTHESES)
    if known.config:
        try:
            from_yaml = load_ablations_from_config(known.config, include_metadata=True)
        except ValueError as exc:
            print(f'[matrix] FAIL: {exc}')
            sys.exit(1)
        if from_yaml:
            ablations = {name: row['cfg'] for name, row in from_yaml.items()}
            hypotheses = {name: row['hypothesis'] for name, row in from_yaml.items()}
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='YAML run config; CLI flags override it')
    p.add_argument('--model', choices=sorted(_MODEL_BUILDERS), default='gpt2-124m')
    p.add_argument('--names', nargs='+', default=list(ablations.keys()))
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.0006)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help='one row is trained once per seed and reported as mean +/- std across seeds (>= 3 recommended)')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--stats-every', type=int, default=200)
    p.add_argument('--val-batches', type=int, default=20, help='held-out synthetic batches for the selection metric (val_loss); never used for gradient updates')
    p.add_argument('--curve-every', type=int, default=0, help='held-out curve-monitoring period in steps (0 = auto: max_steps//5; negative = disabled). Curve points come from a separate generator (seed+2) and never touch the selection batches, so a row reports its train/val regime, not only its final number')
    p.add_argument('--curve-batches', type=int, default=5, help='held-out batches per curve-monitoring point')
    p.add_argument('--baselines', nargs='+', default=['ln_baseline', 'rmsnorm_baseline'], help='rows treated as the simple reference baselines in the baseline-relative comparison')
    p.add_argument('--simple-baselines', default=None, help='Bayes-limit calibration JSON from scripts/baseline_simple.py; when omitted, auto-detected at <output-dir>/simple_baselines.json. Adds the tuned simple baseline and the memoryless-Bayes-floor / full-context-ceiling gaps to baseline_comparison.json')
    p.add_argument('--auto-simple-baselines', action='store_true', help='when the calibration is missing, run scripts/baseline_simple.py once for the selected seeds (same synthetic corpus protocol) instead of writing a comparison with no Bayes-limit reference')
    p.add_argument('--require-simple-baselines', action='store_true', help='fail before writing baseline_comparison.json when no calibration is available and --auto-simple-baselines did not produce one; a comparison without the tuned count model and Bayes limits cannot show that a row learned the task')
    p.add_argument('--no-simple-baselines', dest='simple_baselines_enabled', action='store_false', help='explicitly opt out of the simple-baseline calibration annotation; the opt-out is recorded in baseline_comparison.json')
    p.add_argument('--baseline-tuning', default=None, help='tuning_summary.json from scripts/tune.py; certifies whether the fixed matrix recipe is the baselines\' selected (lr, weight_decay) cell on the same seeds and records the result in baseline_comparison.json')
    p.add_argument('--require-baseline-tuning', action='store_true', help='fail before training unless --baseline-tuning certifies that every compared baseline selected this exact (lr, weight_decay) cell and tune.py confirmed it on held-back seeds; without this flag an uncertified comparison is written but marked as such')
    p.add_argument('--baseline-margin', type=float, default=0.0, help='minimum val_loss improvement over the best baseline to count as a win')
    p.add_argument('--equivalence-margin', type=float, default=0.0, help='TOST equivalence margin in val_loss for "the two arms are the same" claims (e.g. explicit-vs-default init rows); 0 disables equivalence testing, and a non-significant difference is never relabelled as equivalence')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the one-sided paired t-test vs the best baseline (Holm-adjusted over rows, then across the whole pre-registered study)')
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='run the small-batch overfit sanity check for every norm in the selected rows before training them (default: on; a row that cannot overfit one batch has a gradient/loss bug and must not be launched)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the pre-flight overfit sanity check (use only when the check itself is the problem, e.g. an offline host)')
    p.add_argument('--check-matrix', action='store_true', help='validate controlled knobs, stated hypotheses and leave-one-out coverage of the matrix, then exit without training')
    p.add_argument('--schedule-seed', type=int, default=0, help='seed of the randomized complete-block execution schedule; the schedule is frozen in analysis_plan.json before training, so changing it changes the pre-registration fingerprint')
    p.add_argument('--no-randomize-order', dest='randomize_order', action='store_false', help='keep the matrix row order instead of a randomized complete-block schedule (the order is then recorded as not randomized; order drift cannot be separated from the arm)')
    p.add_argument('--assumed-paired-sd', type=float, default=None, help='pre-run assumption for the paired across-seed SD (e.g. from a pilot); with --baseline-margin this makes the design-readiness audit check planned power before launching runs')
    p.add_argument('--require-design-ready', action='store_true', help='fail before training when the design-readiness audit has a structural failure (missing replication/hypothesis, incomplete schedule, absent control pair, or underpowered for the stated margin)')
    p.add_argument('--require-plan-lock', action='store_true', help='fail when this invocation would replace an existing analysis_plan.json with a different fingerprint (changed seeds/alpha/margins/hypotheses); without it the drift is recorded and warned about, but the run proceeds')
    p.add_argument('--require-exclusion-audit', action='store_true', help='fail after training when the pre-specified attrition audit finds a duplicate run/seed, an unaccounted exclusion, or an arm left below 3 analyzable seeds; the audit is always written to exclusion_audit.json')
    p.add_argument('--output-dir', default='runs/ablation')
    p.add_argument('--force', action='store_true')
    p.add_argument('--allow-uncertified', action='store_true', help='allow running with <3 seeds (certification will be disabled); by default <3 seeds is an error')
    apply_config_to_parser(p, known.config)
    args = p.parse_args()
    if not args.simple_baselines_enabled and (args.auto_simple_baselines
                                              or args.require_simple_baselines):
        p.error('--no-simple-baselines cannot be combined with '
                '--auto-simple-baselines or --require-simple-baselines: the '
                'calibration is explicitly disabled, so those requests cannot '
                'be honored')
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        p.error(f'--lr {args.lr} must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        p.error(f'--weight-decay {args.weight_decay} must be finite and >= 0')
    if not math.isfinite(args.baseline_margin) or args.baseline_margin < 0.0:
        p.error(f'--baseline-margin {args.baseline_margin} must be finite and >= 0')
    if not math.isfinite(args.equivalence_margin) or args.equivalence_margin < 0.0:
        p.error(f'--equivalence-margin {args.equivalence_margin} must be finite and >= 0 '
                f'(0 disables equivalence testing)')
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        p.error(f'--alpha {args.alpha} must be finite and strictly between 0 and 1')
    for flag, value in (('--max-steps', args.max_steps),
                        ('--batch-size', args.batch_size),
                        ('--seq-len', args.seq_len),
                        ('--val-batches', args.val_batches),
                        ('--curve-batches', args.curve_batches),
                        ('--log-every', args.log_every),
                        ('--stats-every', args.stats_every)):
        if value < 1:
            p.error(f'{flag} {value} must be >= 1')
    # A `seed:` supplied by --config is a parser default, not an explicit CLI
    # override: an explicit `--seeds 42 43 44` must not be replaced by one
    # config-seeded value (same precedence rule as train_gpt2.py).
    seed_names = _explicit_arg_names(sys.argv[1:], p, known.config)
    if args.seed is not None and ('seed' in seed_names or 'seeds' not in seed_names):
        raw_seeds = [args.seed]
    else:
        raw_seeds = list(args.seeds or DEFAULT_SEEDS)
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    # Keep `args.seeds` equal to the resolved set: protocol checks (e.g. the
    # simple-baseline calibration) must compare against the seeds actually
    # run, not the parser default when the deprecated --seed shorthand was used.
    args.seeds = list(seeds)
    if len(seeds) != len(raw_seeds):
        print(f'[warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed is the same run, not an independent replication)')
    if len(seeds) < 3 and not args.allow_uncertified:
        p.error(f'{len(seeds)} seed(s) selected: certification requires >= 3 independent seeds; use --allow-uncertified to run anyway (results will be descriptive only)')
    try:
        matrix_report = validate_matrix(ablations, hypotheses, args.model)
    except ValueError as exc:
        print(f'[matrix] FAIL: {exc}')
        sys.exit(1)
    print(f"[matrix] ok: {len(matrix_report['rows'])} rows, controlled optimizer/schedule/data, "
          f"every row has a hypothesis, leave-one-out coverage {matrix_report['leave_one_out']}")
    # Resolve the rows that will actually execute, and collapse alias rows
    # (identical effective model config) onto one canonical run. The schedule
    # is over canonical runs only, so an alias cannot consume a slot in a
    # block and unbalance the design.
    selected = list(dict.fromkeys(name for name in args.names if name in ablations))
    for unknown in sorted(set(args.names) - set(ablations)):
        print(f"[warn] unknown ablation '{unknown}', skipping")
    config_groups: dict[str, list[str]] = {}
    for name in selected:
        config_groups.setdefault(
            effective_config_key(args.model, ablations[name]), []).append(name)
    alias_of: dict[str, str] = {}
    canonical_rows: list[str] = []
    for group in config_groups.values():
        canonical = group[0]
        canonical_rows.append(canonical)
        for alias in group[1:]:
            alias_of[alias] = canonical
    if alias_of:
        print(f'[design] {len(alias_of)} alias row(s) reuse a canonical run and are '
              f'outside the execution schedule: {alias_of}')
    run_schedule = None
    if args.randomize_order and canonical_rows:
        run_schedule = randomized_complete_block_schedule(
            canonical_rows, [str(seed) for seed in seeds], seed=args.schedule_seed)
        print(f"[design] randomized complete-block schedule: {len(canonical_rows)} row(s) x "
              f"{len(seeds)} seed(s) = {run_schedule['n_units']} run(s) in "
              f"{run_schedule['n_blocks']} time block(s), fingerprint "
              f"{run_schedule['fingerprint'][:12]}; every row runs once per block")
    elif not canonical_rows:
        print('[design] no rows selected: no execution schedule was generated')
    else:
        print('[design] WARNING: --no-randomize-order keeps the input row order; '
              'a time drift is then confounded with the arm and cannot be '
              'separated from it after the fact')
    exclusion_policy = normalize_exclusion_policy()
    analysis_plan = build_analysis_plan(
        ablations, hypotheses, matrix_report, args.baselines, args.model, seeds,
        alpha=args.alpha, margin=args.baseline_margin,
        equivalence_margin=args.equivalence_margin, run_schedule=run_schedule,
        exclusion_policy=exclusion_policy)
    matrix_report['analysis_plan'] = analysis_plan
    print(f"[plan] {analysis_plan['n_contrasts']} pre-registered confirmatory contrast(s) "
          f"(fingerprint {analysis_plan['fingerprint'][:12]}); every test outside this "
          f"registry is reported as exploratory")
    if args.check_matrix:
        print(json.dumps(matrix_report, indent=2))
        return
    baseline_tuning = None
    if args.require_baseline_tuning and not args.baseline_tuning:
        p.error('--require-baseline-tuning needs --baseline-tuning <tuning_summary.json>: '
                'without the tune.py report there is no evidence that the baselines were '
                'tuned at this matrix recipe')
    if args.baseline_tuning:
        baseline_tuning = load_baseline_tuning(args.baseline_tuning, args, ablations, seeds)
        if baseline_tuning is None and args.require_baseline_tuning:
            raise SystemExit(
                f'[tune-cert] ERROR: --require-baseline-tuning is set but '
                f'{args.baseline_tuning} is missing, unreadable or describes a different '
                f'protocol; refusing to train/compare against baselines without a valid '
                f'tuning certificate.')
        if (baseline_tuning is not None and args.require_baseline_tuning
                and not baseline_tuning.get('certified')):
            raise SystemExit(
                f'[tune-cert] ERROR: --require-baseline-tuning is set but the report does not '
                f'certify this recipe: '
                f'{"; ".join(baseline_tuning.get("why_not_certified") or [])}. '
                f'Rerun the matrix at the baseline\'s selected (lr, weight_decay) cell, or '
                f'tune the baselines with this recipe and held-back --confirm-seeds.')
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    with open(base / 'exclusion_policy.json', 'w') as f:
        json.dump(exclusion_policy, f, indent=2)
    if run_schedule is not None:
        with open(base / 'run_schedule.json', 'w') as f:
            json.dump(run_schedule, f, indent=2)
    schedule_arm_of = (lambda unit: unit.rsplit('::rep=', 1)[0]
                       if '::rep=' in unit else unit)
    control_pairs = []
    if 'dytcd_full' in ablations and 'dytcd_ls1e5' in ablations:
        # Explicit-default LayerScale is an identity pair: the two rows build
        # the same effective model, so its contrast is a known null that tests
        # the analysis pipeline end to end.
        control_pairs.append(('dytcd_full', 'dytcd_ls1e5'))
    expected_schedule_units = (
        [f'{name}::rep={seed}' for name in canonical_rows for seed in seeds]
        if run_schedule is not None else None)
    design_readiness = design_readiness_audit(
        seeds, ablations, hypotheses, schedule=run_schedule,
        expected_units=expected_schedule_units,
        schedule_arm_of=schedule_arm_of, margin=args.baseline_margin,
        equivalence_margin=args.equivalence_margin,
        assumed_paired_sd=args.assumed_paired_sd, alpha=args.alpha,
        identity_controls=control_pairs, exclusion_policy=exclusion_policy)
    if run_schedule is not None:
        design_readiness['schedule_balance'] = schedule_balance_report(
            run_schedule, arm_of=schedule_arm_of)
    if (args.assumed_paired_sd is not None and args.baseline_margin > 0.0
            and len(seeds) >= 2):
        # Calibrate the actual paired-test code against the pre-run assumption:
        # under the alternative the planned effect is -margin (candidate lower
        # loss). This is a preparation check, not a result. A one-seed
        # --allow-uncertified run has no paired test to calibrate, and the
        # replication/power checks already record that failure.
        design_readiness['simulated_power_check'] = simulate_paired_design(
            n_pairs=len(seeds), effect=-float(args.baseline_margin),
            paired_sd=float(args.assumed_paired_sd), reps=1000,
            alpha=args.alpha, seed=args.schedule_seed)
    with open(base / 'design_readiness.json', 'w') as f:
        json.dump(design_readiness, f, indent=2)
    for check in design_readiness['checks']:
        if check['status'] != 'pass':
            print(f"[design] {check['status'].upper()}: {check['id']}: {check['detail']}")
    if not design_readiness['ready']:
        if args.require_design_ready:
            raise SystemExit(
                '[design] ERROR: --require-design-ready is set but the pre-flight '
                'design audit has structural failure(s): '
                + '; '.join(check['detail'] for check in design_readiness['checks']
                            if check['status'] == 'fail'))
        print('[design] WARNING: the design audit has a structural failure; the run '
              'will proceed, but the failure is recorded in design_readiness.json '
              'and the comparison cannot claim a clean confirmatory design')
    # Pre-registration drift: a rerun that changes the seeds, alpha, margins or
    # hypotheses changes the plan fingerprint. Silently replacing the old plan
    # would erase the fact that the confirmatory family moved after results
    # were seen, so the change is recorded (and can be made fatal).
    plan_path = base / 'analysis_plan.json'
    previous_plan = None
    if plan_path.exists():
        try:
            previous_plan = json.loads(plan_path.read_text())
        except (json.JSONDecodeError, OSError):
            previous_plan = None
    previous_fingerprint = (previous_plan or {}).get('fingerprint')
    plan_changed = bool(previous_fingerprint
                        and previous_fingerprint != analysis_plan['fingerprint'])
    analysis_plan['plan_changed'] = plan_changed
    analysis_plan['previous_fingerprint'] = previous_fingerprint
    analysis_plan['plan_change_policy'] = (
        'the fingerprint covers the confirmatory contrast registry, seeds, alpha, '
        'margins and the randomized execution schedule; a changed fingerprint '
        'means the analysis plan moved after it was first fixed')
    if plan_changed:
        message = (f'the analysis plan changed since the previous run in this '
                   f'output directory ({str(previous_fingerprint)[:12]} -> '
                   f'{analysis_plan["fingerprint"][:12]}; seeds, alpha, margins, '
                   f'hypotheses or the randomized schedule); the change is recorded '
                   f'as post-hoc unless this is a fresh pre-registration')
        if args.require_plan_lock:
            raise SystemExit(f'[plan] ERROR: {message}. Remove the old output '
                             f'directory for a deliberate re-registration, or drop '
                             f'--require-plan-lock to proceed with the drift recorded.')
        print(f'[plan] WARNING: {message}')
    with open(plan_path, 'w') as f:
        json.dump(analysis_plan, f, indent=2)
    run_metadata = collect_run_metadata(seed=seeds[0], extra={'seeds': seeds})
    if args.sanity_check:
        from scripts.sanity_overfit import run_preflight
        from scripts.train_gpt2 import _MODEL_SHAPES
        norms = sorted({ablations[name].get('norm_name', 'dytcd') for name in args.names if name in ablations})
        # The rows are trained on the selected --model, so the preflight must
        # exercise the same (n_layers, dim, n_heads) instead of
        # sanity_overfit.py's 2-layer dim=64 default; the small preflight
        # workload (steps/batch/context) is kept on purpose. Per-row knobs
        # (p/k/layerscale variants) are not forwarded: the preflight only
        # needs to catch wiring/NaN/gradient bugs, which are shape- and
        # norm-related.
        n_layers, dim, n_heads = _MODEL_SHAPES.get(args.model, _MODEL_SHAPES['gpt2-124m'])
        arch = ['--dim', str(dim), '--n-layers', str(n_layers), '--n-heads', str(n_heads)]

        run_preflight(norms or ['dytcd'], seeds[0], base, arch_args=arch)
        sanity_report_path = base / 'sanity_overfit.json'
    else:
        sanity_report_path = None
    multi = len(seeds) > 1
    all_summaries = []
    trained: dict[str, str] = {}

    def _report_curve_regimes(agg: dict) -> None:
        """Print the row's per-seed curve verdicts next to its aggregate."""
        regimes = agg.get('curve_classification_counts') or {}
        if not regimes:
            print(f"  [curve] {agg.get('name')}: no curve diagnosis in the cached "
                  f"summaries (pre-monitoring row); rerun with --force to record the "
                  f"train/val regimes")
            return
        print(f"  [curve] {agg['name']}: {regimes}")
        off = {k: v for k, v in regimes.items() if k not in ('healthy', 'insufficient_data')}
        if off:
            print(f"  [curve] WARNING: {agg['name']} has seed(s) in {off}: the val_loss "
                  f"delta may reflect that regime (underfit/stalled/overfit/diverging) as "
                  f"much as the module; read curve_val_loss and the per-seed verdicts "
                  f"before attributing it to the norm")

    def _prepare_row(name: str):
        """Create a row directory and locate its per-seed summary paths."""
        row_dir = base / name
        row_dir.mkdir(parents=True, exist_ok=True)
        summary_paths = {
            seed: (row_dir / f'seed{seed}' / 'summary.json' if multi
                   else row_dir / 'summary.json')
            for seed in seeds}
        legacy = row_dir / 'summary.json'
        if (multi and legacy.exists()
                and not (row_dir / 'summary.preseed.json').exists()
                and not any(path.exists() for path in summary_paths.values())):
            os.replace(legacy, row_dir / 'summary.preseed.json')
            print(f'[keep] {name}: moved pre-replication single-seed summary to '
                  f'summary.preseed.json')
        return row_dir, summary_paths

    row_layout = {name: _prepare_row(name) for name in canonical_rows}
    row_summaries: dict[str, list] = {name: [] for name in canonical_rows}
    row_done: set[str] = set()

    def _finish_row(name: str) -> None:
        cfg_kwargs = ablations[name]
        row_dir, _ = row_layout[name]
        seed_summaries = sorted(row_summaries[name], key=lambda item: item.get('seed'))
        agg = aggregate_seed_summaries(name, seed_summaries, cfg_kwargs,
                                       hypotheses.get(name),
                                       exclusion_policy=exclusion_policy)
        _write_summary(row_dir / 'summary.json', agg)
        _report_curve_regimes(agg)
        trained[effective_config_key(args.model, cfg_kwargs)] = name
        row_done.add(name)
        all_summaries.append(agg)

    if run_schedule is not None:
        schedule_entries = run_schedule['entries']
        schedule_fingerprint = run_schedule['fingerprint']
        randomized = True
    else:
        # Explicit opt-out: a deterministic row-major sequence, recorded as
        # not randomized so the order audit can say the design did not address
        # time drift rather than pretend it did.
        schedule_entries = []
        for block_index, seed in enumerate(seeds):
            for name in canonical_rows:
                schedule_entries.append({
                    'unit': name, 'replicate': str(seed),
                    'run_id': f'{name}::rep={seed}', 'block': block_index,
                    'position': len(schedule_entries)})
        schedule_fingerprint = None
        randomized = False

    for entry in schedule_entries:
        name = entry.get('unit')
        if name not in row_summaries or name in row_done:
            continue
        try:
            seed = int(entry.get('replicate'))
        except (TypeError, ValueError):
            raise SystemExit(
                f'[design] ERROR: schedule entry {entry!r} has no integer seed')
        if seed not in seeds:
            raise SystemExit(
                f'[design] ERROR: schedule entry {entry!r} names seed {seed} outside '
                f'the registered seed set {seeds}')
        cfg_kwargs = ablations[name]
        _, summary_paths = row_layout[name]
        path = summary_paths[seed]
        if path.exists() and (not args.force):
            summary = _load_summary(path)
            if isinstance(summary, dict):
                if cached_seed_matches(summary, name, cfg_kwargs, args, seed):
                    print(f'[skip] {name} seed={seed} already done, use --force to rerun')
                    row_summaries[name].append(summary)
                    if len(row_summaries[name]) == len(seeds):
                        _finish_row(name)
                    continue
                print(f'[stale] {name} seed={seed}: cached summary recorded a different '
                      f'model/cfg/protocol; retraining and overwriting')
        order_info = {
            'position': entry.get('position'),
            'block': entry.get('block'),
            'run_id': entry.get('run_id'),
            'randomized': randomized,
            'schedule_fingerprint': schedule_fingerprint,
            'schedule_seed': (run_schedule.get('seed') if run_schedule else None),
        }
        row_summaries[name].append(run_one(
            name, cfg_kwargs, args, path.parent, run_metadata=run_metadata,
            hypothesis=hypotheses.get(name), seed=seed, order_info=order_info))
        if len(row_summaries[name]) == len(seeds):
            _finish_row(name)

    unresolved_rows = [name for name in canonical_rows if name not in row_done]
    if unresolved_rows:
        raise SystemExit(
            f'[design] ERROR: the run schedule did not complete row(s) '
            f'{unresolved_rows}; no aggregate can be written for them. This is a '
            f'schedule/execution bug, not a scientific exclusion.')

    # Alias rows are deterministic copies of a canonical trained row. They are
    # written after the canonical run so they can never be scheduled as an
    # independent arm (which would double-count one run in the block design).
    for alias, canonical in alias_of.items():
        cfg_kwargs = ablations[alias]
        row_dir, summary_paths = _prepare_row(alias)
        source_row_dir, _ = row_layout[canonical]
        source_paths = {
            seed: (source_row_dir / f'seed{seed}' / 'summary.json' if multi
                   else source_row_dir / 'summary.json')
            for seed in seeds}
        reused = [_load_summary(path) for path in source_paths.values()]
        if not all(isinstance(summary, dict) for summary in reused):
            raise SystemExit(
                f'[design] ERROR: alias row {alias!r} reuses {canonical!r}, but its '
                f'per-seed summaries are incomplete; refusing to write a row that '
                f'is not actually a copy of a completed run')
        reused = [dict(summary, name=alias, cfg=cfg_kwargs,
                       reused_from=canonical, hypothesis=hypotheses.get(alias))
                  for summary in reused]
        for seed, summary in zip(seeds, reused):
            if multi:
                _write_summary(summary_paths[seed], summary)
        agg = aggregate_seed_summaries(alias, reused, cfg_kwargs,
                                       hypotheses.get(alias),
                                       exclusion_policy=exclusion_policy)
        _write_summary(row_dir / 'summary.json', agg)
        print(f'[alias] {alias}: identical model config to {canonical}, reusing its '
              f'summaries (no retrain)')
        _report_curve_regimes(agg)
        all_summaries.append(agg)
    agg_path = base / 'all_summaries.json'
    previous: list = []
    if agg_path.exists():
        try:
            previous = json.loads(agg_path.read_text())
            if not isinstance(previous, list):
                previous = []
        except json.JSONDecodeError:
            previous = []
    ordered = merge_summaries(previous, all_summaries)
    with open(agg_path, 'w') as f:
        json.dump(ordered, f, indent=2)
    # Accumulated rows whose recorded recipe differs from this invocation are
    # reported but excluded from both comparisons: a delta produced by a
    # different lr/step/batch/seq recipe is not a property of the module.
    comparable, stale_rows = partition_rows_by_protocol(ordered, args)
    if stale_rows:
        print(f"[compare] WARNING: {len(stale_rows)} accumulated row(s) excluded from the comparisons "
              f"(trained under a different model/cfg/protocol than this invocation): "
              f"{[r['name'] for r in stale_rows]}; rerun them under this recipe to include them")
    # Pre-specified attrition: the comparisons below silently drop a
    # non-finite seed from a pooled mean, so the denominator is audited
    # explicitly. Every planned (row, seed) is a record; missing or
    # non-finite runs are excluded only for the reasons frozen in
    # exclusion_policy.json, and an arm that falls below the certification
    # replication count is reported instead of being averaged over survivors.
    row_by_name = {row.get('name'): row for row in all_summaries
                   if isinstance(row, dict) and row.get('name')}
    stale_names = {row.get('name') for row in stale_rows
                   if isinstance(row, dict)}
    planned_runs = []
    for entry in schedule_entries:
        name = entry.get('unit')
        if name not in canonical_rows:
            continue
        try:
            seed = int(entry.get('replicate'))
        except (TypeError, ValueError):
            continue
        agg = row_by_name.get(name) or {}
        item = next((p for p in (agg.get('per_seed') or [])
                     if isinstance(p, dict) and p.get('seed') == seed), None)
        planned_runs.append({
            'run_id': entry.get('run_id') or f'{name}::rep={seed}',
            'arm': name,
            'seed': seed,
            'val_loss': item.get('val_loss') if item else None,
            'present': item is not None,
            'protocol_mismatch': name in stale_names,
            'curve_classification': (item or {}).get('curve_classification'),
        })
    exclusion_audit = run_exclusion_audit(planned_runs, exclusion_policy)
    exclusion_audit['schedule_fingerprint'] = (
        run_schedule.get('fingerprint') if run_schedule else None)
    with open(base / 'exclusion_audit.json', 'w') as f:
        json.dump(exclusion_audit, f, indent=2)
    if exclusion_audit['n_excluded'] or exclusion_audit['n_sensitivity']:
        print(f"[exclusion] {exclusion_audit['n_planned']} planned run(s): "
              f"{exclusion_audit['n_analyzed']} retained, "
              f"{exclusion_audit['n_sensitivity']} sensitivity-only, "
              f"{exclusion_audit['n_excluded']} excluded "
              f"(policy {exclusion_audit['policy_fingerprint'][:12]})")
    for warning in exclusion_audit['warnings']:
        print(f'[exclusion] WARNING: {warning}')
    if not exclusion_audit['ready']:
        message = ('the attrition audit found an unusable design: '
                   + '; '.join(exclusion_audit['violations']))
        if args.require_exclusion_audit:
            raise SystemExit('[exclusion] ERROR: ' + message)
        print('[exclusion] WARNING: ' + message
              + '; the comparisons below are reported with the attrition '
                'visible in exclusion_audit.json')
    comparison = baseline_comparison(comparable, args.baselines, margin=args.baseline_margin,
                                     alpha=args.alpha,
                                     equivalence_margin=args.equivalence_margin)
    simple_path = Path(args.simple_baselines) if args.simple_baselines else base / 'simple_baselines.json'
    floor_data = (load_simple_baselines(simple_path, args)
                  if args.simple_baselines_enabled else None)
    if (comparison is not None and floor_data is None
            and args.simple_baselines_enabled and args.auto_simple_baselines):
        floor_data = run_simple_baseline_calibration(simple_path, seeds, args)
    if comparison is not None and floor_data is not None:
        comparison['simple_baseline'] = attach_simple_baseline_calibration(
            comparison, floor_data, simple_path, alpha=args.alpha)
        comparison['calibration_status'] = 'attached'
    elif comparison is not None and not args.simple_baselines_enabled:
        # The opt-out must be visible in the artifact, not just on stdout: a
        # reader of baseline_comparison.json otherwise sees the same file as a
        # fully calibrated comparison.
        comparison['simple_baseline'] = {
            'enabled': False, 'status': 'explicitly_disabled',
            'reason': 'disabled by --no-simple-baselines',
            'bayes_limits': None, 'tuned_simple_ce_nats': None}
        comparison['calibration_status'] = 'explicitly_disabled'
        print('[compare] simple-baseline calibration disabled by --no-simple-baselines: rows are '
              'ranked only against each other, not against the task Bayes limits')
    elif comparison is not None:
        comparison['simple_baseline'] = {
            'enabled': False, 'status': 'missing', 'source': str(simple_path),
            'bayes_limits': None, 'tuned_simple_ce_nats': None,
            'how_to_generate': (f'python3 scripts/baseline_simple.py --output {simple_path} '
                                f'--seeds {" ".join(map(str, seeds))} '
                                f'--vocab-size {_PROTOCOL_MODEL_VOCAB[args.model]} '
                                f'--batch-size {args.batch_size} --seq-len {args.seq_len} '
                                f'--val-batches {max(1, int(getattr(args, "val_batches", 20)))}')}
        comparison['calibration_status'] = 'missing'
        print(f"[compare] WARNING: no tuned simple-baseline calibration at {simple_path}; the rows below are "
              f"only ranked against each other. Generate the task's Bayes limits and tuned count model with "
              f"'{comparison['simple_baseline']['how_to_generate']}' (or pass --auto-simple-baselines / "
              f"--require-simple-baselines)")
        if args.require_simple_baselines:
            raise SystemExit(
                f'[compare] ERROR: --require-simple-baselines is set but no calibration exists at '
                f'{simple_path}; refusing to write baseline_comparison.json without the tuned count-model '
                f'and Bayes-limit reference. Generate it with '
                f"'{comparison['simple_baseline']['how_to_generate']}'.")
    if comparison is not None:
        if baseline_tuning is not None:
            comparison['baseline_tuning'] = attach_baseline_tuning(comparison, baseline_tuning)
        else:
            baseline_norms = [ablations[name].get('norm_name', name)
                              for name in args.baselines if name in ablations]
            comparison['baseline_tuning'] = {
                'enabled': False,
                'status': ('missing' if args.baseline_tuning else 'not_requested'),
                'source': args.baseline_tuning,
                'certified': False,
                'certified_against_tuned_reference': False,
                'reading': ('the baseline comparison below is against the fixed matrix recipe, '
                            'not a certified tuned baseline; pass --baseline-tuning plus '
                            '--require-baseline-tuning to make this claim fail-closed'),
                'how_to_generate': (
                    f"python3 scripts/tune.py --model {args.model} "
                    f"--arms {' '.join(baseline_norms) or 'ln rmsnorm'} "
                    f"--max-steps {args.max_steps} --batch-size {args.batch_size} "
                    f"--seq-len {args.seq_len} "
                    f"--val-batches {max(1, int(getattr(args, 'val_batches', 20)))} "
                    f"--lrs {args.lr:g} 3e-4 1.2e-3 "
                    f"--weight-decays {args.weight_decay:g} "
                    f"--seeds {' '.join(map(str, seeds))} "
                    f"--confirm-seeds <held-back seeds> --output-dir {base / 'tune'}"),
            }
            print('[compare] WARNING: no baseline-tuning certificate; the significance test below '
                  'uses the fixed-recipe baselines. Generate one with '
                  f"'{comparison['baseline_tuning']['how_to_generate']}' and rerun with "
                  '--baseline-tuning (or --require-baseline-tuning to fail closed).')
        comparison['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'baseline_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    print_baseline_comparison(comparison)
    # The second half of the ablation contract: each claimed mechanism of the
    # main proposal must be shown to be load-bearing *on its own*, with a
    # paired Full-vs-Full-minus-X test over the same seeds.
    loo_comparison = leave_one_out_comparison(comparable, margin=args.baseline_margin,
                                              alpha=args.alpha,
                                              equivalence_margin=args.equivalence_margin)
    if loo_comparison is not None:
        loo_comparison['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'leave_one_out_comparison.json', 'w') as f:
            json.dump(loo_comparison, f, indent=2)
    print_leave_one_out_comparison(loo_comparison)
    # The third half of the contract: Full combines two axes, so report the
    # 2x2 main effects and their interaction rather than inferring synergy
    # from Full's point estimate alone.
    factorial = factorial_comparison(comparable, margin=args.baseline_margin,
                                     alpha=args.alpha,
                                     equivalence_margin=args.equivalence_margin)
    if factorial is not None:
        factorial['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'factorial_comparison.json', 'w') as f:
            json.dump(factorial, f, indent=2)
    print_factorial_comparison(factorial)
    # The fourth report closes the "three orthogonal axes" claim: the two 2x2
    # designs estimate per-channel x depth separately per activation family,
    # the 2x2x2 design estimates the cross-family interactions and tests
    # additivity with TOST instead of inferring it from a null result.
    three_way = three_way_factorial_comparison(
        comparable, margin=args.baseline_margin, alpha=args.alpha,
        equivalence_margin=args.equivalence_margin)
    if three_way is not None:
        three_way['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'three_way_factorial_comparison.json', 'w') as f:
            json.dump(three_way, f, indent=2)
    print_three_way_factorial_comparison(three_way)
    # The attention-logits axis (DyT-QK) is independent of the normalization
    # family, so it gets its own per-host attribution and cross-host
    # replication report instead of being folded into the baseline family.
    qk_axis = qk_axis_comparison(comparable, margin=args.baseline_margin,
                                 alpha=args.alpha,
                                 equivalence_margin=args.equivalence_margin)
    if qk_axis is not None:
        qk_axis['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'qk_axis_comparison.json', 'w') as f:
            json.dump(qk_axis, f, indent=2)
    print_qk_axis_comparison(qk_axis)
    # Study-level multiplicity: the four reports above each correct within
    # their own family, but the paper-level claim is made across all of them.
    # Pool only the pre-registered confirmatory contrasts and report what the
    # global Holm correction confirms (or demotes).
    study = study_level_correction(
        {'baseline': comparison, 'leave_one_out': loo_comparison,
         'factorial': factorial, 'factorial3': three_way,
         'qk_axis': qk_axis},
        analysis_plan, alpha=args.alpha)
    if study is not None:
        with open(base / 'study_level_tests.json', 'w') as f:
            json.dump(study, f, indent=2)
    print_study_level_correction(study)
    # Execution-order audit: a significant within-arm rank correlation is a
    # design warning, not an arm result. It is written next to the comparisons
    # so a reader can see whether the randomized schedule actually held.
    order_audit = run_order_audit(comparable, run_schedule)
    with open(base / 'run_order_audit.json', 'w') as f:
        json.dump(order_audit, f, indent=2)
    manifest = dict(run_metadata)
    manifest['git'] = git_revision()
    manifest['hyperparameters'] = vars(args)
    manifest['seeds'] = seeds
    manifest['matrix_validation'] = matrix_report
    manifest['rows'] = [row['name'] for row in ordered]
    manifest['rows_excluded_protocol_mismatch'] = stale_rows
    # Curve regimes per row (per-seed classification counts): the run manifest
    # keeps the train/val regime each comparison was measured in, so a future
    # reader can tell a healthy falling-falling delta from an early-stopped
    # underfit or an overfitting one without re-reading every summary.json.
    manifest['curve_regimes'] = {
        row.get('name'): row.get('curve_classification_counts')
        for row in ordered if isinstance(row, dict) and row.get('name')
    }
    manifest['baseline_comparison'] = comparison
    manifest['leave_one_out_comparison'] = loo_comparison
    manifest['factorial_comparison'] = factorial
    manifest['three_way_factorial_comparison'] = three_way
    manifest['qk_axis_comparison'] = qk_axis
    manifest['analysis_plan'] = analysis_plan
    manifest['study_level_tests'] = study
    manifest['run_schedule'] = run_schedule
    manifest['exclusion_policy'] = exclusion_policy
    manifest['exclusion_audit'] = exclusion_audit
    manifest['design_readiness'] = design_readiness
    manifest['run_order_audit'] = order_audit
    if sanity_report_path is not None:
        manifest['overfit_sanity_check'] = {'enabled': True, **file_fingerprint(sanity_report_path)}
    else:
        manifest['overfit_sanity_check'] = {'enabled': False}
    # Durable artifacts, so the manifest says which files the comparison above
    # was computed from. `all_summaries.json` can carry every per-step curve
    # and is recorded by size+mtime instead of a full re-hash.
    manifest = finish_metadata(manifest, artifacts={
        'all_summaries': artifact_record(base / 'all_summaries.json'),
        'analysis_plan': artifact_record(base / 'analysis_plan.json'),
        'baseline_comparison': artifact_record(base / 'baseline_comparison.json'),
        'leave_one_out_comparison': artifact_record(base / 'leave_one_out_comparison.json'),
        'factorial_comparison': artifact_record(base / 'factorial_comparison.json'),
        'three_way_factorial_comparison': artifact_record(
            base / 'three_way_factorial_comparison.json'),
        'qk_axis_comparison': artifact_record(base / 'qk_axis_comparison.json'),
        'study_level_tests': artifact_record(base / 'study_level_tests.json'),
        'run_schedule': (artifact_record(base / 'run_schedule.json')
                         if run_schedule is not None else {'enabled': False}),
        'exclusion_policy': artifact_record(base / 'exclusion_policy.json'),
        'exclusion_audit': artifact_record(base / 'exclusion_audit.json'),
        'design_readiness': artifact_record(base / 'design_readiness.json'),
        'run_order_audit': artifact_record(base / 'run_order_audit.json'),
        'sanity_report': file_fingerprint(sanity_report_path) if sanity_report_path else {'enabled': False},
    })
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[done] {len(ordered)} ablations saved to {base} '
          f'({len(seeds)} seed(s): {seeds}; row val_loss = mean +/- std across seeds)')
if __name__ == '__main__':
    main()
