from __future__ import annotations
import dataclasses
import hashlib
import itertools
import json
import math
import numbers
import os
import sys
import time
from collections import Counter
from pathlib import Path
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
                               leave_one_out_jackknife, mean_prediction_interval,
                               mean_std_stderr, max_p_combination,
                               paired_design_summary, paired_randomization_test,
                               paired_t_one_sided, student_t_ppf,
                               tost_design_summary, tost_paired,
                               two_sided_p_from_t, welch_t_one_sided)
from adasquash.model import build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache, clear_cast_weight_cache
from adasquash.repro import artifact_record, collect_run_metadata, file_fingerprint, finish_metadata, git_revision, save_run_manifest, set_all_seeds
from scripts.train_gpt2 import (_explicit_arg_names, apply_config_to_parser,
                                build_optimizer, load_yaml_config)

ROOT = Path(__file__).resolve().parents[2]

def _finite_metric(value) -> bool:
    """A loss may be pooled/tested only when it is a finite number.

    `isinstance(value, (int, float))` alone admits NaN/Inf from a diverged
    run; NaN then propagates through the seed mean (and the paired tests),
    silently poisoning the row's pooled val_loss.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


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
        'delta_prediction_interval_95': None,
        'delta_same_direction_probability': None,
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
            replication = mean_prediction_interval(values, alpha=alpha)
            if replication is not None:
                out['delta_prediction_interval_95'] = [
                    replication['prediction_low'],
                    replication['prediction_high']]
                out['delta_same_direction_probability'] = (
                    replication['same_direction_probability'])
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
                           equivalence_margin: float = 0.0,
                           margin: float = 0.0) -> dict:
    """Attach seed-jackknife, equivalence and distribution-free checks.

    Three questions a significance test alone cannot answer are asked for
    every paired contrast:

    * **Is the sign carried by one seed?** `leave_one_out_jackknife` drops
      each seed in turn and reports whether the direction survives. At three
      seeds this is the difference between "the norm helps" and "one lucky
      run helped".
    * **Is the two arms' difference small enough to call them equivalent?**
      Only a pre-registered `--equivalence-margin` enables TOST; a
      non-significant superiority test is explicitly *not* relabelled as
      equivalence. The margin is stored in the artifact with the p-value.
    * **Does the conclusion depend on the normal model?**
      `paired_randomization_test` runs the exact sign-flip test on the same
      deltas and records the smallest p-value attainable at this seed count.
      A t-test that rejects at three seeds while the distribution-free floor
      is 0.125 is reported as model-dependent rather than silently treated as
      a nonparametric result.

    `deltas` maps seed -> paired difference (candidate - reference). All
    blocks are descriptive for the certification decision; the pre-specified
    paired t-test remains the only certifying test.
    """
    if equivalence_margin and equivalence_margin > 0.0:
        if isinstance(alpha, bool):
            alpha_number = None
        else:
            try:
                alpha_number = float(alpha)
            except (TypeError, ValueError, OverflowError):
                alpha_number = None
        if (alpha_number is not None and math.isfinite(alpha_number)
                and alpha_number >= 0.5):
            raise ValueError(
                'a positive equivalence_margin requires alpha < 0.5 so the TOST '
                f'confidence interval 1 - 2*alpha is positive, got alpha={alpha!r}')
    values = [float(deltas[seed]) for seed in deltas]
    labels = [str(seed) for seed in deltas]
    jackknife = leave_one_out_jackknife(values, labels=labels)
    if jackknife is not None:
        jackknife["unit"] = "seed"
    randomization = paired_randomization_test(
        values, margin=margin, alpha=alpha)
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
        summary = mean_std_stderr(values)
        if summary is not None and summary.get("std") is not None:
            tost["design"] = tost_design_summary(
                summary["std"], int(tost["n"]), equivalence_margin,
                alpha=alpha)
        equivalence = tost
    return {"seed_jackknife": jackknife, "equivalence": equivalence,
            "randomization_test": randomization}


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
    randomization = entry.get('randomization_test')
    if isinstance(randomization, dict) and randomization.get('can_reach_alpha') is False:
        floor = randomization.get('minimum_attainable_p_one_sided')
        if floor is not None:
            parts.append(f' [distribution-free p floor {floor:.3g} at '
                         f"n={randomization.get('n')}: sign-flip test cannot "
                         'reject without the t model]')
    return ''.join(parts)


def _is_seed_id(value) -> bool:
    """A seed id is an integer scalar; 42.9/True must not pair with 42/1."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _seed_key_id(value):
    """Canonical int seed from an int or its exact JSON string key.

    JSON turns integer mapping keys into strings, so ``"42"`` names the same
    run as ``42``; ``"42.0"``, ``"042"``, booleans and fractional numbers do
    not. Returns None for anything that is not a canonical seed id.
    """
    if _is_seed_id(value):
        return int(value)
    if isinstance(value, str):
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if str(converted) == value:
            return converted
    return None


def _seed_means(row) -> dict:
    """{seed: val_loss} for every row shape.

    Aggregates carry a `per_seed` list; a single-run summary (no `per_seed`)
    still identifies its seed, so it can pair with the same seed inside an
    aggregate.
    """
    per = row.get('per_seed')
    out = {}
    seen: set[int] = set()
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
            seed = int(seed)
            if seed in seen:
                # Two records for one seed are the same run, not two
                # replications; collapsing them would inflate the paired
                # denominator and can certify an effect resting on a
                # duplicated observation. Track every occurrence, including a
                # non-finite first record that never enters `out`.
                return {}
            seen.add(seed)
            if _finite_metric(item.get('val_loss')):
                out[seed] = float(item['val_loss'])
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
    if len(unique_seeds) != len(seeds):
        return {}
    if len(unique_seeds) == 1 and _finite_metric(row.get('val_loss')):
        out[unique_seeds[0]] = float(row['val_loss'])
    return out

def _batches_of_seed(row, seed):
    """Held-out batch losses belonging to one run, or None.

    Never returns `val_losses` when the row pooled batches from several
    seeds: those concatenated lists must not be treated as one run's pairing
    basis (batch k of seed 42 would otherwise be zipped with batch k of
    seed 44). Non-finite batch losses are dropped before the test; they are
    corrupted measurements, not observations, and leaving one in the list
    made the paired/welch test raise deep inside the Student-t CDF. The
    number dropped is recorded on the row under
    ``val_losses_excluded_by_seed`` so the reduced pairing basis is visible.
    """
    def _finite_batches(values):
        finite = [value for value in values if _finite_metric(value)]
        dropped = len(values) - len(finite)
        if dropped and isinstance(row, dict):
            ledger = row.setdefault('val_losses_excluded_by_seed', {})
            if isinstance(ledger, dict):
                key = str(seed)
                ledger[key] = int(ledger.get(key, 0)) + dropped
        return finite

    by_seed = row.get('val_losses_by_seed')
    if isinstance(by_seed, dict):
        # Accept both JSON's string keys and in-memory integer seed keys.
        for key in (str(seed), seed):
            values = by_seed.get(key)
            if isinstance(values, list):
                return _finite_batches(values)
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
        return _finite_batches(row['val_losses'])
    return None

def _row_n_seeds(row) -> int:
    """Usable replication count of a row, deriving it from its per-seed means.

    `n_seeds` can be absent or malformed on a hand-built/legacy aggregate
    even though `per_seed` carries three pairable runs. Reporting that row as
    a one-seed study while the paired test certifies it on three means is an
    internal contradiction (and would show it in the "thin" warning).
    """
    if not isinstance(row, dict):
        return 1
    means = _seed_means(row)
    if means:
        return len(means)
    # If the row *has* per-seed/seed data but none of it is pairable (a
    # duplicate seed, malformed id, or no finite loss), the declared count
    # cannot turn it into replication; report 0 rather than trusting it.
    if (isinstance(row.get('per_seed'), list)
            or isinstance(row.get('seeds'), list)):
        return 0
    declared = row.get('n_seeds')
    if isinstance(declared, int) and not isinstance(declared, bool) and declared >= 0:
        return declared
    return 1


def _row_std(row):
    """Across-seed std when replicated, within-run batch std otherwise.

    A replicated row normally carries `val_loss_seed_std`; fall back to the
    generic `val_loss_std` when an older/cached aggregate only wrote that
    name, so the delta-vs-seed-noise scale check is not silently disabled.
    """
    if _row_n_seeds(row) >= 2 and row.get('val_loss_seed_std') is not None:
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
        return (None, f'single_seed_batches_insufficient{seed}', common)
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
