"""Experimental-design preparation for the training studies.


The comparison reports answer statistical questions *after* the runs exist:
paired t-tests, equivalence, multiplicity, power and jackknife checks. They do
not answer the design questions that have to be settled *before* a run starts:

* in what order will the runs execute, and is that order confounded with the
  arm?  A matrix that always trains LayerNorm first and the proposal last can
  turn a slow hardware/thermal drift into an apparent "norm effect";
* which runs are allowed to be excluded, and on what pre-specified rule?
  Excluding a diverged arm after seeing its loss is a garden of forking paths;
* is the planned number of seeds/pairs actually able to detect the smallest
  effect the study claims to care about?
* does the analysis code itself behave as advertised under a known null and a
  known effect?  A pipeline that reports a "significant" effect on a
  synthetic null is not evidence for anything.

This module supplies the reusable, pure-Python pieces:

``block_randomized_schedule``
    Deterministic within-block randomization of run ids.  Hash-based rather
    than ``random.shuffle`` so the same schedule is reproducible on every
    Python version and can be frozen in the pre-registration.
``schedule_balance_report``
    Per-arm position/block coverage, so an arm that only ever ran in one
    time block is visible before the results are interpreted. An RCBD is
    also checked for the stronger arm x block crossing contract: every arm
    must appear exactly once in every block, not merely the same number of
    times overall.
``schedule_fingerprint_status``
    Recomputes the schedule hash from its recorded seed/order, so an
    allocation edited after the pre-registration was frozen is detected
    before the run instead of being silently analysed as the planned design.
``order_drift_report``
    Within-arm rank correlation between execution position and outcome, with
    a within-arm permutation test.  Centering by arm is what distinguishes
    "time drift correlated with the arm" from a real arm difference.
``default_exclusion_policy`` / ``run_exclusion_audit``
    A pre-specified attrition rule: only missing runs, protocol/provenance
    mismatches and non-finite losses may be excluded, and every planned run
    is retained in a per-arm denominator instead of disappearing from a
    pooled mean.  A finite but unlucky seed is never an exclusion.
``design_readiness_audit``
    A pre-flight checklist: replication, stated hypotheses, a pre-specified
    smallest effect, adequate planned power, a complete randomized schedule,
    a locked exclusion policy and the presence of an identity/negative
    control pair. The power check uses the family-adjusted alpha for the
    planned primary claims and the expected complete pairs after attrition,
    not the nominal per-test alpha and seed count.
``prospective_power_plan``
    Family-level (Bonferroni) and attrition-inflated sample-size planning:
    how many *planned* seed pairs are needed before a run so that, after the
    study-wide multiplicity correction and the expected run-level attrition,
    the pre-specified smallest effect still has the target power.
``tost_power_plan``
    The equivalence counterpart. A superiority power formula evaluated at the
    equivalence margin answers the wrong question, so TOST claims get their
    own family-level and attrition-inflated sample-size plan.
``factorial_contrast_sd`` / ``factorial_power_plan``
    The planning counterpart of the factorial designs. An interaction is a
    different contrast with a different noise level than an arm-vs-arm
    comparison, so the seed count that powers the main claim does not power
    the orthogonality/synergy claims. The helper converts the pilot's paired
    SD into the SD of the factorial term itself and plans it with the same
    family-level and attrition corrections.
``paired_sd_calibration``
    Turns pilot runs into a *provenance-carrying* planning variance. A pilot
    point estimate is itself uncertain, so the helper reports the sample and
    robust SD, a one-sided upper confidence bound, and whether the pilot seeds
    overlap the confirmatory seeds (using the same seed twice would make the
    "independent" replication a re-analysis of one run).
``paired_sd_evidence_audit``
    Checks a frozen calibration against the SD actually used for planning:
    an optimistic point estimate below the pilot's upper bound is a design
    failure, while an SD at or above it is conservative and admissible.
``simulate_paired_design``
    A Monte-Carlo calibration of the actual paired-test code: empirical
    type-I error under a synthetic null and empirical power under a known
    effect.  This is a *preparation* test of the analysis pipeline, not a
    substitute for running the experiment.

Wiring in this repository
-------------------------
``scripts/ablation.py`` writes ``run_schedule.json`` (the RCBD),
``design_readiness.json`` (the pre-flight checklist, with
``--require-design-ready`` available to fail closed) and
``run_order_audit.json`` (the realized-order drift test); the schedule is
embedded in ``analysis_plan.json`` so changing it changes the pre-registration
fingerprint.  The same runner freezes ``exclusion_policy.json`` before
training, embeds its fingerprint in the plan, and writes
``exclusion_audit.json`` afterwards (``--require-exclusion-audit`` makes an
arm left below the replication minimum a hard failure).
``scripts/train_llama_depth_sweep.py`` writes the schedule and the order audit
for the depth/norm grid.  ``--no-randomize-order`` is the recorded opt-out in
both runners.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import random
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Optional

from .metrics import (
    MIN_SEEDS_FOR_SIGNIFICANCE,
    chi2_sf,
    mean_std_stderr,
    paired_design_summary,
    paired_required_n,
    paired_t_one_sided,
    tost_design_summary,
    tost_paired,
    tost_required_n,
)

__all__ = [
    "block_randomized_schedule",
    "classify_run_exclusions",
    "default_exclusion_policy",
    "design_readiness_audit",
    "exclusion_policy_fingerprint",
    "factorial_contrast_sd",
    "factorial_power_plan",
    "normalize_exclusion_policy",
    "order_drift_report",
    "paired_sd_calibration",
    "paired_sd_evidence_audit",
    "prospective_power_plan",
    "randomized_complete_block_schedule",
    "run_exclusion_audit",
    "schedule_balance_report",
    "schedule_fingerprint_status",
    "simulate_paired_design",
    "tost_power_plan",
]


def _validate_seed(seed) -> int:
    # Accept integral NumPy scalars too: they are exact integer seeds, while
    # a float (even an integral one) is not a seed identity.
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(f"schedule seed must be an integer, got {seed!r}")
    return int(seed)


def _validate_unit(unit) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"run id must be a non-empty string, got {unit!r}")
    return unit


def _finite_float(value) -> Optional[float]:
    """``float(value)`` when it is a finite non-boolean real, else None.

    An enormous Python integer is a real number but raises ``OverflowError``
    on conversion to float. Pre-flight reports must classify such a value as
    invalid instead of crashing.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _complete_pairs_after_attrition(n_pairs: int, survival: float) -> int:
    """Expected analyzable complete pairs, including for huge ``n_pairs``.

    ``n_pairs * survival`` raises ``OverflowError`` when ``n_pairs`` is larger
    than the float range even though the correct count is a perfectly ordinary
    integer. Reproduce the same tolerated flooring with an exact rational
    fallback rather than crashing the pre-registration plan.
    """
    try:
        return int(math.floor(n_pairs * survival + 1e-12))
    except OverflowError:
        product = Fraction(n_pairs) * Fraction.from_float(survival)
        return math.floor(product + Fraction(1, 10**12))


def _pairs_needed_after_attrition(required_pairs: int,
                                  survival: float) -> Optional[int]:
    """Inflate a required pair count for expected complete-pair survival.

    ``required_pairs / survival`` raises ``OverflowError`` when the required
    count is a huge integer, even though the inflated count is a perfectly
    ordinary integer. The exact rational fallback mirrors the tolerated
    ceiling used by the ordinary path.
    """
    if not math.isfinite(survival) or survival <= 0.0:
        return None
    try:
        return int(math.ceil(required_pairs / survival - 1e-12))
    except (OverflowError, ValueError):
        ratio = Fraction(required_pairs) / Fraction.from_float(survival)
        return math.ceil(ratio - Fraction(1, 10**12))


def block_randomized_schedule(units: Iterable[str], block_size: Optional[int] = None,
                              seed: int = 0) -> dict:
    """Deterministically randomize run ids inside consecutive time blocks.

    Every ``block_size`` consecutive run slots form one block and the ids in
    that block are ordered by a keyed SHA-256 digest.  Sorting by a
    cryptographic hash is a uniform random permutation whose result does not
    depend on the Python ``random`` implementation, so a schedule written
    before training can be re-computed and checked afterwards on any host.

    Blocking (rather than one global shuffle) keeps the schedule balanced if
    the study has to be stopped early: within every block each arm has the
    same expected number of early/late positions.  It also keeps a slow
    monotone drift from aligning with a single arm; the residual correlation
    is tested by :func:`order_drift_report`.

    ``block_size=None`` uses ``ceil(sqrt(n))`` for a grid with more than eight
    units (small grids are a single block).  The returned ``fingerprint``
    covers the seed, block size and final order, and is safe to embed in a
    pre-registration.
    """
    seed = _validate_seed(seed)
    if isinstance(units, (str, bytes)):
        raise ValueError("units must be an iterable of run ids, not a single string")
    ids = [_validate_unit(unit) for unit in units]
    if not ids:
        raise ValueError("a run schedule needs at least one run id")
    if len(set(ids)) != len(ids):
        duplicates = sorted({unit for unit in ids if ids.count(unit) > 1})
        raise ValueError(f"run ids must be unique; duplicated: {duplicates}")
    n = len(ids)
    if block_size is None:
        block_size = n if n <= 8 else max(1, math.ceil(math.sqrt(n)))
    if (isinstance(block_size, bool) or not isinstance(block_size, numbers.Integral)
            or block_size < 1):
        raise ValueError(f"block_size must be a positive integer, got {block_size!r}")
    block_size = int(block_size)

    order: list[str] = []
    entries: list[dict] = []
    block_sizes: list[int] = []
    for block_index, start in enumerate(range(0, n, block_size)):
        chunk = ids[start:start + block_size]
        block_sizes.append(len(chunk))
        digest_key = lambda unit: (
            hashlib.sha256(f"{seed}\x1f{block_index}\x1f{unit}".encode()).hexdigest(),
            unit,
        )
        for unit in sorted(chunk, key=digest_key):
            entries.append({"unit": unit, "block": block_index,
                            "position": len(order)})
            order.append(unit)

    payload = {"seed": seed, "block_size": block_size, "order": order}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "adasquash.run_schedule.v1",
        "seed": seed,
        "block_size": block_size,
        "n_units": n,
        "n_blocks": len(block_sizes),
        "block_sizes": block_sizes,
        "order": order,
        "entries": entries,
        "positions": {unit: position for position, unit in enumerate(order)},
        "fingerprint": fingerprint,
        "protocol": ("within-block order is a keyed SHA-256 sort; the same seed and "
                     "run-id set reproduce the same schedule on every host"),
    }


def randomized_complete_block_schedule(units: Iterable[str], replicates: Iterable[str],
                                       seed: int = 0) -> dict:
    """Randomized complete block design for replicated runs.

    Every time block contains *every* treatment/unit exactly once, and each
    unit receives every replicate exactly once across the blocks.  With
    ``units`` = matrix rows and ``replicates`` = training seeds this is the
    classic RCBD: a block-level drift (a slow machine, a thermal episode, a
    code change that only affects part of the session) hits every arm once and
    therefore cancels from the paired arm contrast instead of biasing the arm
    that happened to run last.

    The block order is time order; the within-block order is a keyed SHA-256
    sort, so it is reproducible and independent of Python's ``random``
    implementation.  Each row is still replicated over every seed.  A block
    uses one seed for *all* of its rows, and the keyed order decides which
    seed belongs to which block.  That binding is deliberate: the reports pair
    arms by seed, so if a seed were spread over several blocks the paired
    difference would also contain a block/time difference and block drift
    would not cancel.  Independent per-row seed permutations satisfy "each
    row sees each seed" but break the within-block pairing the analysis
    assumes.
    """
    seed = _validate_seed(seed)
    if isinstance(units, (str, bytes)) or isinstance(replicates, (str, bytes)):
        raise ValueError("units and replicates must be iterables, not single strings")
    unit_ids = [_validate_unit(unit) for unit in units]
    replicate_ids = [str(rep) for rep in replicates]
    if any(not rep.strip() for rep in replicate_ids):
        raise ValueError("replicate ids must be non-empty")
    if not unit_ids:
        raise ValueError("a block schedule needs at least one unit")
    if not replicate_ids:
        raise ValueError("a block schedule needs at least one replicate")
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("unit ids must be unique")
    if len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError("replicate ids must be unique")

    def _hash(*parts) -> str:
        return hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()

    block_replicates = sorted(
        replicate_ids, key=lambda rep: (_hash(seed, "base", rep), rep))
    entries: list[dict] = []
    order: list[str] = []
    for block_index, replicate in enumerate(block_replicates):
        for unit in sorted(unit_ids,
                           key=lambda unit, block=block_index: (
                               _hash(seed, "block", block, unit), unit)):
            run_id = f"{unit}::rep={replicate}"
            entries.append({"unit": unit, "replicate": replicate,
                            "run_id": run_id, "block": block_index,
                            "position": len(order)})
            order.append(run_id)
    run_ids = [entry["run_id"] for entry in entries]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("internal error: a unit x replicate run was assigned twice")
    expected = {(unit, rep) for unit in unit_ids for rep in replicate_ids}
    realized = {(entry["unit"], entry["replicate"]) for entry in entries}
    if realized != expected:
        raise ValueError("internal error: the block design does not cover every "
                         "unit x replicate pair exactly once")
    replicate_blocks: dict[str, set[int]] = {}
    for entry in entries:
        replicate_blocks.setdefault(entry["replicate"], set()).add(entry["block"])
    if any(len(blocks) != 1 for blocks in replicate_blocks.values()):
        raise ValueError("internal error: a replicate/seed is spread over "
                         "several time blocks, so seed-paired contrasts would "
                         "not be within-block")
    payload = {"seed": seed, "units": unit_ids, "replicates": replicate_ids,
               "order": order}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "adasquash.rcbd_schedule.v1",
        "seed": seed,
        "n_units": len(order),
        "n_treatments": len(unit_ids),
        "n_blocks": len(replicate_ids),
        "block_sizes": [len(unit_ids)] * len(replicate_ids),
        "units": unit_ids,
        "replicates": replicate_ids,
        "order": order,
        "entries": entries,
        "positions": {run_id: position for position, run_id in enumerate(order)},
        "fingerprint": fingerprint,
        "protocol": ("randomized complete block design: one time block per "
                     "replicate/seed, every treatment runs once in every block, "
                     "the seed-to-block assignment and the within-block order "
                     "are keyed SHA-256 sorts, so a seed-paired contrast is "
                     "always within-block"),
    }


def schedule_fingerprint_status(schedule: dict) -> dict:
    """Recompute a schedule's SHA-256 from its recorded fields.

    A schedule is only a pre-registration artefact if the order written to
    disk is the order the fingerprint was computed from. This helper rebuilds
    the canonical payload for either shipped schema and compares it with the
    stored ``fingerprint``; an edited order, seed or block size without a
    corresponding fingerprint update is visible before training starts.

    ``verified`` is ``True``/``False`` for a recognized schema and ``None``
    when the record is not a schedule this module can verify (for example a
    legacy or caller-defined dict). A mismatch is a hard failure in
    :func:`design_readiness_audit`; an unverifiable record is a warning.
    """
    if not isinstance(schedule, dict):
        raise ValueError("schedule must be a mapping")
    schema = schedule.get("schema")
    expected = schedule.get("fingerprint")
    if not isinstance(expected, str) or not expected:
        return {
            "schema": schema,
            "fingerprint": expected,
            "recomputed": None,
            "verified": None,
            "reason": "schedule has no stored fingerprint to verify",
        }
    if schema == "adasquash.run_schedule.v1":
        payload = {
            "seed": schedule.get("seed"),
            "block_size": schedule.get("block_size"),
            "order": schedule.get("order"),
        }
    elif schema == "adasquash.rcbd_schedule.v1":
        payload = {
            "seed": schedule.get("seed"),
            "units": schedule.get("units"),
            "replicates": schedule.get("replicates"),
            "order": schedule.get("order"),
        }
    else:
        return {
            "schema": schema,
            "fingerprint": expected,
            "recomputed": None,
            "verified": None,
            "reason": f"unrecognized schedule schema {schema!r}",
        }
    try:
        recomputed = hashlib.sha256(
            json.dumps(payload, sort_keys=True,
                       separators=(",", ":")).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        return {
            "schema": schema,
            "fingerprint": expected,
            "recomputed": None,
            "verified": None,
            "reason": "schedule fields are not JSON-serializable",
        }
    verified = recomputed == expected
    return {
        "schema": schema,
        "fingerprint": expected,
        "recomputed": recomputed,
        "verified": verified,
        "reason": ("fingerprint matches the recorded schedule" if verified else
                   "fingerprint does not match the recorded schedule fields"),
    }


def _arm_resolver(arm_of):
    if arm_of is None:
        return lambda unit: unit
    if callable(arm_of):
        return arm_of
    if isinstance(arm_of, Mapping):
        return lambda unit: arm_of.get(unit, unit)
    raise ValueError("arm_of must be None, a mapping or a callable")


def schedule_balance_report(schedule: dict, arm_of=None) -> dict:
    """Per-arm position and block coverage for a generated schedule.

    The report is descriptive: a random schedule can put one three-seed arm
    entirely in the first third by chance, and that is not by itself a flaw.
    The actionable warnings are structural (an arm that never appears in more
    than one block when it has several runs, a single-block schedule for a
    large grid, or missing/duplicate positions).  :func:`order_drift_report`
    tests whether the realized positions actually correlate with outcomes.
    """
    if not isinstance(schedule, dict) or not isinstance(schedule.get("entries"), list):
        raise ValueError("schedule must be the dict returned by block_randomized_schedule")
    resolve = _arm_resolver(arm_of)
    # An RCBD run id is `unit::rep=<replicate>`; its treatment/arm is the
    # `unit` field, not the replicated run id. Without an explicit resolver,
    # grouping by run id would give every arm one block and falsely reject a
    # perfectly crossed design. `run_id` is still the identity used for the
    # realized-order/duplicate checks below.
    rcbd = schedule.get("schema") == "adasquash.rcbd_schedule.v1"

    def _arm_of(entry: dict):
        if arm_of is None and rcbd:
            return entry.get("unit", entry.get("run_id"))
        if rcbd and isinstance(arm_of, Mapping):
            # An RCBD's treatment key is `unit`, not the replicated run id.
            # Honour a mapping keyed by that natural identifier; fall back to
            # the run id for callers that map the full run ids.
            unit = entry.get("unit")
            if unit in arm_of:
                return arm_of[unit]
        return resolve(entry.get("run_id", entry.get("unit")))

    entries = schedule["entries"]
    structural: list[str] = []
    if len(entries) != schedule.get("n_units"):
        structural.append("entry count does not match n_units")
    # Positions are the structural contract of the schedule. A malformed
    # record must produce a warning instead of a KeyError/TypeError: this
    # report is the pre-flight check that is supposed to expose a broken
    # schedule, not crash on it.
    positions: list[int] = []
    positions_valid = True
    for entry in entries:
        if not isinstance(entry, dict):
            positions_valid = False
            continue
        position = entry.get("position")
        if (isinstance(position, bool)
                or not isinstance(position, numbers.Integral)
                or position < 0):
            positions_valid = False
        else:
            positions.append(int(position))
    if not positions_valid:
        structural.append("positions are not a contiguous 0..n-1 range "
                          "(missing or non-integer position)")
    elif sorted(positions) != list(range(len(entries))):
        structural.append("positions are not a contiguous 0..n-1 range")
    units = []
    units_valid = True
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            units_valid = False
            units.append(("invalid-entry", index))
            continue
        unit = entry.get("run_id", entry.get("unit"))
        if not isinstance(unit, str) or not unit.strip():
            # A non-string run id cannot be compared/paired by the later
            # reports. It must produce a warning, not an unhashable-type crash
            # from the duplicate check.
            units_valid = False
            units.append(("invalid-unit", index, repr(unit)))
        else:
            units.append(("unit", unit))
    if not units_valid:
        structural.append("run ids are missing or not non-empty strings")
    if len(set(units)) != len(units):
        structural.append("duplicate run ids in the realized order")
    raw_order = schedule.get("order")
    if raw_order is not None and not isinstance(raw_order, list):
        structural.append("recorded order is not a list of run ids")
    elif isinstance(raw_order, list) and len(raw_order) != len(entries):
        # A truncated/extended frozen order cannot be compared position by
        # position. Silence here would certify a schedule whose recorded
        # allocation and realized entries disagree in size.
        structural.append(
            f"recorded order has {len(raw_order)} run id(s) but there are "
            f"{len(entries)} entries; the frozen schedule and the run records "
            "describe different allocations")
    elif positions_valid and isinstance(raw_order, list):
        by_position = sorted(entries, key=lambda entry: entry["position"])
        realized_order = [entry.get("run_id", entry.get("unit"))
                          for entry in by_position]
        if realized_order != raw_order:
            structural.append(
                "entries do not follow the recorded order; the frozen schedule "
                "and the run records describe different allocations")

    per_arm: dict[str, dict] = {}
    blocks_valid = True
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        arm = str(_arm_of(entry))
        bucket = per_arm.setdefault(arm, {
            "count": 0, "positions": [], "blocks": set(), "early": 0, "late": 0,
        })
        bucket["count"] += 1
        position = entry.get("position")
        if (not isinstance(position, bool)
                and isinstance(position, numbers.Integral)
                and position >= 0):
            bucket["positions"].append(int(position))
        block = entry.get("block")
        if (isinstance(block, bool)
                or not isinstance(block, numbers.Integral) or block < 0):
            blocks_valid = False
        else:
            bucket["blocks"].add(int(block))
    if not blocks_valid:
        structural.append("one or more entries have a missing or non-integer block")
    n = len(entries)
    overall_mean = (n - 1) / 2.0 if n else None
    extreme_counts: dict[str, int] = {}
    for arm, bucket in per_arm.items():
        bucket["mean_position"] = (
            (sum(bucket["positions"]) / len(bucket["positions"]) if bucket["positions"] else float("nan"))
            if bucket["positions"] else None)
        bucket["position_span"] = (max(bucket["positions"]) - min(bucket["positions"])
                                   if bucket["positions"] else None)
        bucket["blocks"] = sorted(bucket["blocks"])
        bucket["n_blocks"] = len(bucket["blocks"])
        if n:
            bucket["early"] = sum(1 for p in bucket["positions"] if p < n / 3.0)
            bucket["late"] = sum(1 for p in bucket["positions"] if p >= 2.0 * n / 3.0)
        if bucket["count"] >= 2 and bucket["n_blocks"] <= 1:
            structural.append(
                f"arm {arm!r} appears in only one time block "
                f"({bucket['count']} run(s)); its position is confounded with "
                f"any block-level drift")
        if bucket["count"] >= 2 and n and bucket["mean_position"] is not None:
            dev = abs(bucket["mean_position"] - overall_mean) / max(1.0, n - 1)
            if dev >= 1.0 / 3.0:
                extreme_counts[arm] = bucket["count"]
    if schedule.get("n_blocks") == 1 and n > 8:
        structural.append("a single block was used for a large grid; "
                          "blocking cannot spread a monotone drift")
    # A complete randomized block design is a stronger contract than "every
    # arm appears in several blocks": within *each* block every arm must
    # appear exactly once. The check only applies when the schedule looks
    # like an RCBD (each arm has one run per block), so a generic blocked
    # order with unequal arm counts is not misclassified.
    is_rcbd = schedule.get("schema") == "adasquash.rcbd_schedule.v1"
    has_replicates = is_rcbd or any(
        isinstance(entry, dict) and entry.get("replicate") is not None
        for entry in entries)
    block_arm_counts: dict[int, dict[str, int]] = {}
    block_replicate_counts: dict[int, dict[str, int]] = {}
    realized_replicates: set[str] = set()
    replicate_entries_valid = True
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        block = entry.get("block")
        if (isinstance(block, bool)
                or not isinstance(block, numbers.Integral) or block < 0):
            continue
        block = int(block)
        arm = str(_arm_of(entry))
        counts = block_arm_counts.setdefault(block, {})
        counts[arm] = counts.get(arm, 0) + 1
        if not has_replicates:
            continue
        replicate = entry.get("replicate")
        if not isinstance(replicate, str) or not replicate.strip():
            replicate_entries_valid = False
            continue
        realized_replicates.add(replicate)
        rep_counts = block_replicate_counts.setdefault(block, {})
        rep_counts[replicate] = rep_counts.get(replicate, 0) + 1
    declared_blocks = schedule.get("n_blocks")
    declared_blocks_valid = (
        isinstance(declared_blocks, numbers.Integral)
        and not isinstance(declared_blocks, bool)
        and declared_blocks > 0)
    if declared_blocks_valid:
        declared_blocks = int(declared_blocks)
    replicate_block_aligned = None
    replicate_blocks: dict[str, list[int]] = {}
    if has_replicates:
        if not replicate_entries_valid:
            block_replicate_counts = {}
            structural.append(
                "one or more entries have a missing or non-empty string "
                "replicate id; seed-by-block alignment cannot be checked")
        else:
            raw_replicates = schedule.get("replicates")
            declared_replicates = {
                str(rep) for rep in raw_replicates
                if str(rep).strip()
            } if isinstance(raw_replicates, list) else set()
            replicate_ids = sorted(declared_replicates | realized_replicates)
            blocks = sorted(block_replicate_counts)
            if replicate_ids and blocks:
                complete_counts = {}
                for block in blocks:
                    counts = dict.fromkeys(replicate_ids, 0)
                    for rep, value in block_replicate_counts[block].items():
                        counts[rep] = value
                    complete_counts[block] = counts
                block_replicate_counts = complete_counts
                replicate_blocks = {
                    rep: [block for block, counts in complete_counts.items()
                          if counts.get(rep, 0) > 0]
                    for rep in sorted(realized_replicates)
                }
                if replicate_blocks:
                    replicate_block_aligned = all(
                        len(rep_blocks) == 1
                        for rep_blocks in replicate_blocks.values())
                split = {rep: rep_blocks
                         for rep, rep_blocks in replicate_blocks.items()
                         if len(rep_blocks) > 1}
                if split:
                    structural.append(
                        f"replicate/seed(s) {sorted(split)} appear in multiple "
                        f"time blocks ({split}); a seed-paired arm contrast then "
                        f"also contains a block/time difference, so block drift "
                        f"does not cancel")
            else:
                block_replicate_counts = {}
    declared_missing = []
    declared_extra = []
    raw_units = schedule.get("units")
    if is_rcbd and isinstance(raw_units, list) and raw_units:
        # The frozen design declares every treatment; entries that retain only
        # a subset are not a complete block design even when the retained
        # subset is crossed. Compare the realized `unit` field, not the
        # (possibly remapped) arm labels, so this check also works with an
        # explicit `arm_of` mapping.
        declared = {str(unit) for unit in raw_units}
        realized = {str(entry.get("unit")) for entry in entries
                    if isinstance(entry, dict) and entry.get("unit") is not None}
        declared_missing = sorted(declared - realized)
        declared_extra = sorted(realized - declared)
        if declared_missing or declared_extra:
            structural.append(
                "the realized schedule does not cover the declared RCBD "
                f"treatment(s): missing {declared_missing}, extra {declared_extra}")
    looks_complete = is_rcbd or (
        declared_blocks_valid and bool(per_arm)
        and sorted(block_arm_counts) == list(range(declared_blocks))
        and all(bucket["count"] == declared_blocks
                and bucket["n_blocks"] == declared_blocks
                for bucket in per_arm.values()))
    complete_block_design = None
    if looks_complete:
        block_index_valid = declared_blocks_valid
        coverage_ok = (
            block_index_valid
            and sorted(block_arm_counts) == list(range(declared_blocks))
            and all(bucket["count"] == declared_blocks
                    and bucket["n_blocks"] == declared_blocks
                    for bucket in per_arm.values()))
        if not coverage_ok:
            complete_block_design = False
            structural.append(
                "the declared randomized complete block design does not cover "
                "every arm in every block exactly once (missing/extra block or "
                "arm x block cell)")
        else:
            complete_block_design = True
            for block, counts in sorted(block_arm_counts.items()):
                if (set(counts) != set(per_arm)
                        or any(value != 1 for value in counts.values())):
                    complete_block_design = False
                    structural.append(
                        f"block {block} does not contain every arm exactly once "
                        f"({counts}); the schedule is not a randomized complete "
                        f"block design")
    if complete_block_design and (declared_missing or declared_extra):
        complete_block_design = False
    return {
        "schema": "adasquash.schedule_balance.v1",
        "schedule_fingerprint": schedule.get("fingerprint"),
        "n_units": n,
        "n_blocks": declared_blocks,
        "block_sizes": schedule.get("block_sizes"),
        "overall_mean_position": overall_mean,
        "per_arm": per_arm,
        "block_arm_counts": {str(block): counts
                             for block, counts in sorted(block_arm_counts.items())},
        "block_replicate_counts": {str(block): counts
                                   for block, counts in sorted(
                                       block_replicate_counts.items())},
        "replicate_blocks": replicate_blocks,
        "replicate_block_aligned": replicate_block_aligned,
        "complete_block_design": complete_block_design,
        "arms_with_extreme_mean_position": sorted(extreme_counts),
        "balanced": not structural,
        "warnings": structural,
        "note": ("balance is descriptive; use order_drift_report to test whether "
                 "the realized run order predicts the outcome after removing "
                 "arm means"),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda index: values[index])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        average = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    try:
        mx = sum(x) / max(n, 1)
        my = sum(y) / max(n, 1)
        dx = [value - mx for value in x]
        dy = [value - my for value in y]
        sx = math.sqrt(sum(value * value for value in dx))
        sy = math.sqrt(sum(value * value for value in dy))
        if sx == 0.0 or sy == 0.0:
            return None
        rho = sum(a * b for a, b in zip(dx, dy)) / (sx * sy)
        if math.isfinite(rho):
            return rho
    except OverflowError:
        import logging as _logging; _logging.warning("Suppressed exception in design.py:795", exc_info=True)
    # Finite values near the float limit can overflow the raw sums even when
    # the correlation is ordinary. Scale both inputs by their largest
    # magnitude; the coefficient is invariant to positive rescaling.
    scale_x = max(abs(value) for value in x)
    scale_y = max(abs(value) for value in y)
    if scale_x == 0.0 or scale_y == 0.0:
        return None
    scaled_x = [value / scale_x for value in x]
    scaled_y = [value / scale_y for value in y]
    if not (all(math.isfinite(value) for value in scaled_x)
            and all(math.isfinite(value) for value in scaled_y)):
        return None
    mx = math.fsum(scaled_x) / max(n, 1)
    my = math.fsum(scaled_y) / max(n, 1)
    dx = [value - mx for value in scaled_x]
    dy = [value - my for value in scaled_y]
    sx = math.sqrt(math.fsum(value * value for value in dx))
    sy = math.sqrt(math.fsum(value * value for value in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    rho = math.fsum(a * b for a, b in zip(dx, dy)) / (sx * sy)
    if not math.isfinite(rho):
        return None
    return min(1.0, max(-1.0, rho))


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return _pearson(_average_ranks(x), _average_ranks(y))


def order_drift_report(observations: Sequence[dict], alpha: float = 0.05,
                       permutations: int = 2000, seed: int = 0) -> dict:
    """Test whether execution position predicts the outcome *within an arm*.

    ``observations`` is a list of ``{'unit', 'arm', 'position', 'value'}``
    records (one per completed run).  The outcome is centered by arm, so a
    genuine arm difference - including one that happens to be scheduled
    early - is removed before the rank correlation is computed.  What remains
    is the within-arm time trend: if later runs within the same arm score
    systematically worse, the schedule (hardware, thermal, code drift) is
    confounding the study.

    The p-value is a two-sided within-arm permutation test: positions are
    reshuffled inside each arm, keeping the arm sizes, outcome distribution
    and between-arm structure fixed.  Arms with fewer than two observations
    carry no within-arm information and are excluded rather than contributing
    a zero residual.  Fewer than six usable observations, or fewer than two
    distinct positions, returns ``status='insufficient_data'`` instead of a
    misleading p-value.
    """
    if (isinstance(permutations, bool)
            or not isinstance(permutations, numbers.Integral)
            or permutations < 1):
        raise ValueError(f"permutations must be a positive integer, got {permutations!r}")
    permutations = int(permutations)
    alpha_value = _finite_float(alpha)
    if alpha_value is None or not 0.0 < alpha_value < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    seed = _validate_seed(seed)
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError("observations must be a sequence of dict records")

    excluded = {"malformed": 0, "non_finite": 0, "duplicate_position": 0,
                "singleton_arm": 0}
    valid: list[dict] = []
    seen: set[tuple] = set()
    for record in observations:
        if not isinstance(record, dict):
            excluded["malformed"] += 1
            continue
        arm = record.get("arm")
        position = record.get("position")
        value = record.get("value")
        if (not isinstance(arm, str) or not arm
                or isinstance(position, bool) or not isinstance(position, numbers.Real)
                or not isinstance(value, numbers.Real) or isinstance(value, bool)):
            excluded["malformed"] += 1
            continue
        position_value = _finite_float(position)
        value_value = _finite_float(value)
        if position_value is None or value_value is None:
            excluded["non_finite"] += 1
            continue
        key = (arm, position_value)
        if key in seen:
            excluded["duplicate_position"] += 1
            continue
        seen.add(key)
        valid.append({"unit": record.get("unit"), "arm": arm,
                      "position": position_value, "value": value_value})

    groups: dict[str, list[dict]] = {}
    for record in valid:
        groups.setdefault(record["arm"], []).append(record)
    usable = {arm: records for arm, records in groups.items() if len(records) >= 2}
    excluded["singleton_arm"] = sum(len(records) for arm, records in groups.items()
                                    if len(records) < 2)
    flat = [record for arm in sorted(usable) for record in usable[arm]]
    distinct_positions = {record["position"] for record in flat}
    base = {
        "schema": "adasquash.order_drift.v1",
        "n_observations": len(flat),
        "n_arms": len(usable),
        "n_distinct_positions": len(distinct_positions),
        "excluded": excluded,
        "permutations": permutations,
        "alpha": float(alpha),
        "spearman_rho": None,
        "p_two_sided": None,
        "drift_detected": None,
        "status": "insufficient_data",
    }
    if len(flat) < 6 or len(distinct_positions) < 2:
        base["note"] = ("at least six observations from arms with >= 2 runs and two "
                        "distinct positions are needed to test order drift")
        return base

    residual_scale = None
    ordinary_ok = True
    try:
        for arm, records in usable.items():
            mean = (sum(record["value"] for record in records) / max(len(records), 1)) if records else float("nan")
            if not math.isfinite(mean):
                ordinary_ok = False
                break
            for record in records:
                record["residual"] = record["value"] - mean
                if not math.isfinite(record["residual"]):
                    ordinary_ok = False
                    break
            if not ordinary_ok:
                break
    except OverflowError:
        ordinary_ok = False
    if not ordinary_ok:
        # Finite values near the float limit overflow the per-arm sum, making
        # every residual ±inf even though the rank correlation is well
        # defined. Center in units of the largest observed magnitude; a
        # common positive scale preserves every rank, so the ordinary path
        # above (and its exact tie behaviour) remains unchanged.
        residual_scale = max(abs(record["value"]) for record in flat)
        if residual_scale == 0.0:
            for record in flat:
                record["residual"] = 0.0
        else:
            for arm, records in usable.items():
                scaled = [record["value"] / residual_scale for record in records]
                mean_scaled = (math.fsum(scaled) / max(len(scaled), 1)) if scaled else float("nan")
                for record in records:
                    record["residual"] = (
                        record["value"] / max(residual_scale, 1e-10) - mean_scaled)
    x = [record["position"] for record in flat]
    y = [record["residual"] for record in flat]
    rho = _spearman(x, y)
    if rho is None:
        base["note"] = "every within-arm residual is identical; no rank correlation is defined"
        return base

    rng = random.Random(seed)
    arm_positions = {arm: [record["position"] for record in records]
                     for arm, records in usable.items()}
    residuals = [record["residual"] for record in flat]
    rho_samples: list[float] = []
    for _ in range(permutations):
        shuffled_positions: list[float] = []
        for arm in sorted(usable):
            positions = list(arm_positions[arm])
            rng.shuffle(positions)
            shuffled_positions.extend(positions)
        permuted = _spearman(shuffled_positions, residuals)
        if permuted is not None:
            rho_samples.append(permuted)
    extreme = sum(1 for sample in rho_samples if abs(sample) >= abs(rho))
    p_two_sided = (1.0 + extreme) / (1.0 + len(rho_samples))
    # `drift_detected` orients a positive rho as "later runs worse": the
    # one-sided p for that alternative is the upper tail, not the lower tail.
    # The old lower-tail count returned ~1.0 for the very trend the report
    # labels later_runs_worse.
    decline = sum(1 for sample in rho_samples if sample >= rho)
    p_one_sided_decline = (1.0 + decline) / (1.0 + len(rho_samples))

    per_arm = []
    for arm in sorted(usable):
        records = usable[arm]
        positions = [record["position"] for record in records]
        per_arm.append({
            "arm": arm,
            "count": len(records),
            "mean_position": (sum(positions) / len(positions)) if positions else float("nan"),
            "position_span": max(positions) - min(positions),
            "mean_residual": (
                math.fsum(record["residual"] for record in records)
                / len(records) if records else float("nan")
                * (1.0 if residual_scale is None else residual_scale)),
        })
    base.update({
        "status": "tested",
        "spearman_rho": rho,
        "p_two_sided": p_two_sided,
        "p_one_sided_decline": p_one_sided_decline,
        "drift_detected": bool(p_two_sided < float(alpha) and rho != 0.0),
        "direction": ("later_runs_worse" if rho > 0 else
                      "later_runs_better" if rho < 0 else "flat"),
        "n_permutations_defined": len(rho_samples),
        "per_arm": per_arm,
        "note": ("a significant within-arm rank correlation means execution order "
                 "is confounded with the outcome; re-run the affected arms in a "
                 "fresh randomized block rather than interpreting the arm delta"),
    })
    return base


EXCLUSION_POLICY_SCHEMA = "adasquash.exclusion_policy.v1"
_EXCLUSION_ACTIONS = ("exclude", "sensitivity")
# A confirmatory design may not discard a run because its outcome is
# inconvenient. These are the only failure modes the default policy treats as
# exclusions; all of them are observable before/independently of the arm
# contrast.
_REQUIRED_EXCLUSION_RULES = ("missing_run", "protocol_mismatch", "non_finite_loss")
_DIVERGED_CLASSIFICATIONS = ("diverging", "diverged", "nonfinite_loss")


def default_exclusion_policy() -> dict:
    """The pre-specified attrition policy used by the confirmatory runners.

    An exclusion changes the analysed sample, so it is a design decision that
    has to be fixed before the outcomes are seen. The default admits only
    *protocol* failures - a planned run with no artifact, a run whose recorded
    recipe/provenance differs from the registered one, and a run whose
    held-out loss is NaN/Inf - plus a non-excluding ``sensitivity`` flag for a
    run whose loss curve already diverged. A finite but unlucky loss is never
    an exclusion reason: dropping the worst seed after looking at the results
    is exactly the garden of forking paths this policy is meant to stop.
    """
    return {
        "schema": EXCLUSION_POLICY_SCHEMA,
        "version": 1,
        "rules": [
            {"id": "missing_run", "action": "exclude", "outcome_blind": True,
             "description": ("planned run has no readable summary, checkpoint or "
                             "finished manifest")},
            {"id": "protocol_mismatch", "action": "exclude", "outcome_blind": True,
             "description": ("recorded model/data/optimizer recipe or pinned "
                             "provenance differs from the registered protocol")},
            {"id": "non_finite_loss", "action": "exclude", "outcome_blind": True,
             "description": ("held-out loss is NaN/Inf, so no finite paired "
                             "contrast can be computed")},
            {"id": "diverged_curve", "action": "sensitivity", "outcome_blind": False,
             "description": ("the training curve is already diverging; the run is "
                             "retained in the primary analysis and reported as a "
                             "sensitivity check instead of being dropped")},
        ],
        "forbidden_exclusion_criteria": [
            "val_loss_outlier", "best_of_seeds", "drop_worst_arm",
            "post_hoc_loss_threshold",
        ],
        "protocol": (
            "exclusions are fixed before outcomes are inspected; only missing "
            "runs, protocol/provenance mismatches and non-finite losses may be "
            "excluded, and every excluded or flagged run is counted per arm"),
    }


def normalize_exclusion_policy(policy: Optional[Mapping] = None) -> dict:
    """Validate and canonicalize an exclusion policy (``None`` -> default).

    The returned object has rules sorted by id and only the fields that carry
    meaning, so two semantically identical policies have the same
    fingerprint. A rule with ``action='exclude'`` must be declared
    ``outcome_blind=True``; an outcome-dependent exclusion is rejected rather
    than silently allowed. The three protocol rules are mandatory: a policy
    cannot quietly stop treating a missing or non-finite run as attrition.
    """
    if policy is None:
        policy = default_exclusion_policy()
    if not isinstance(policy, Mapping):
        raise ValueError(f"exclusion policy must be a mapping, got {type(policy).__name__}")
    if policy.get("schema") != EXCLUSION_POLICY_SCHEMA:
        raise ValueError(
            f"exclusion policy schema must be {EXCLUSION_POLICY_SCHEMA!r}, "
            f"got {policy.get('schema')!r}")
    version = policy.get("version", 1)
    if (isinstance(version, bool)
            or not isinstance(version, numbers.Integral) or version < 1):
        raise ValueError(f"exclusion policy version must be a positive integer, got {version!r}")
    raw_rules = policy.get("rules")
    if (not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes))
            or not raw_rules):
        raise ValueError("exclusion policy needs a non-empty list of rules")
    rules: list[dict] = []
    seen: set[str] = set()
    for raw in raw_rules:
        if not isinstance(raw, Mapping):
            raise ValueError(f"exclusion rule must be a mapping, got {type(raw).__name__}")
        rule_id = raw.get("id")
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise ValueError(f"exclusion rule id must be a non-empty string, got {rule_id!r}")
        if rule_id in seen:
            raise ValueError(f"duplicate exclusion rule id {rule_id!r}")
        seen.add(rule_id)
        action = raw.get("action")
        if action not in _EXCLUSION_ACTIONS:
            raise ValueError(
                f"exclusion rule {rule_id!r} action must be one of "
                f"{_EXCLUSION_ACTIONS}, got {action!r}")
        blind = raw.get("outcome_blind")
        if not isinstance(blind, bool):
            raise ValueError(
                f"exclusion rule {rule_id!r} must declare outcome_blind true/false")
        if action == "exclude" and not blind:
            raise ValueError(
                f"exclusion rule {rule_id!r} is outcome-dependent; an exclude "
                f"rule must be outcome_blind, otherwise a bad result can be "
                f"dropped after it is seen")
        rule = {"id": rule_id, "action": action, "outcome_blind": blind}
        description = raw.get("description")
        if description is not None:
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"exclusion rule {rule_id!r} description must be a non-empty string")
            rule["description"] = description
        rules.append(rule)
    missing = [rule_id for rule_id in _REQUIRED_EXCLUSION_RULES if rule_id not in seen]
    if missing:
        raise ValueError(
            f"exclusion policy is missing mandatory protocol rule(s): {missing}; "
            f"a run may not disappear without a recorded reason")
    # Keeping the rule id while changing its action to `sensitivity` would
    # silently stop treating a missing/non-finite run as attrition, so the
    # mandatory rules must remain `exclude` rules.
    downgraded = [rule["id"] for rule in rules
                  if rule["id"] in _REQUIRED_EXCLUSION_RULES
                  and rule["action"] != "exclude"]
    if downgraded:
        raise ValueError(
            f'mandatory protocol rule(s) {downgraded} must keep action="exclude"; '
            f'a missing, mismatched or non-finite run may not be retained as a '
            f'sensitivity case')
    forbidden = policy.get("forbidden_exclusion_criteria", [])
    if (not isinstance(forbidden, Sequence) or isinstance(forbidden, (str, bytes))):
        raise ValueError("forbidden_exclusion_criteria must be a sequence of strings")
    if any(not isinstance(item, str) or not item.strip() for item in forbidden):
        raise ValueError(
            "forbidden_exclusion_criteria must contain non-empty strings, got "
            f"{[item for item in forbidden if not isinstance(item, str) or not item.strip()]!r}")
    forbidden = list(forbidden)
    used_forbidden = sorted(seen.intersection(forbidden))
    if used_forbidden:
        raise ValueError(
            f"exclusion rule id(s) {used_forbidden} are listed as forbidden "
            f"outcome-based criteria; a policy cannot both forbid and use them")
    protocol = policy.get("protocol")
    if protocol is not None and not isinstance(protocol, str):
        raise ValueError("exclusion policy protocol must be a string")
    return {
        "schema": EXCLUSION_POLICY_SCHEMA,
        "version": int(version),
        "rules": sorted(rules, key=lambda rule: rule["id"]),
        "forbidden_exclusion_criteria": sorted(set(forbidden)),
        "protocol": protocol or (
            "only pre-specified, outcome-blind protocol failures may be excluded; "
            "the per-arm attrition is audited after the run"),
    }


def exclusion_policy_fingerprint(policy: Optional[Mapping] = None) -> str:
    """Stable SHA-256 of the canonical exclusion policy."""
    canonical = normalize_exclusion_policy(policy)
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def classify_run_exclusions(runs: Sequence[Mapping],
                            policy: Optional[Mapping] = None) -> list[dict]:
    """Apply the pre-specified policy to every planned run.

    ``runs`` records use ``run_id``, ``arm``, optional ``seed``,
    ``val_loss``, ``present``, ``protocol_mismatch`` and
    ``curve_classification``. The result keeps one record per planned run
    (including a missing one) with the reasons that fired and a status:

    * ``excluded``  - at least one ``exclude`` rule fired (removed from the
      primary contrast, but still counted in the audit);
    * ``sensitivity`` - only a non-excluding flag fired (kept in the primary
      analysis, reported as a sensitivity check);
    * ``analyzed`` - no rule fired.

    Facts are derived from protocol metadata and finiteness, never from the
    rank of the loss, so a surprisingly good/bad but finite run cannot be
    classified as an exclusion.
    """
    canonical = normalize_exclusion_policy(policy)
    rule_action = {rule["id"]: rule["action"] for rule in canonical["rules"]}
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
        raise ValueError("runs must be a sequence of run records")
    classified: list[dict] = []
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise ValueError(f"run record {index} must be a mapping, "
                             f"got {type(run).__name__}")
        run_id = run.get("run_id")
        arm = run.get("arm")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(f"run record {index} needs a non-empty string run_id")
        if not isinstance(arm, str) or not arm.strip():
            raise ValueError(f"run record {index} ({run_id!r}) needs a non-empty arm")
        present = run.get("present", True)
        if not isinstance(present, bool):
            raise ValueError(f"run record {run_id!r} present must be a boolean, "
                             f"got {present!r}")
        mismatch = run.get("protocol_mismatch", False)
        if not isinstance(mismatch, bool):
            raise ValueError(
                f"run record {run_id!r} protocol_mismatch must be a boolean, "
                f"got {mismatch!r}")
        facts: list[str] = []
        if not present:
            facts.append("missing_run")
        if mismatch:
            facts.append("protocol_mismatch")
        loss = run.get("val_loss")
        if present:
            if loss is None:
                facts.append("missing_run")
            else:
                try:
                    finite = (not isinstance(loss, bool)
                              and isinstance(loss, numbers.Real)
                              and math.isfinite(float(loss)))
                except (TypeError, ValueError, OverflowError):
                    # An enormous integer is a real number but has no finite
                    # float representation, so it cannot enter a paired
                    # contrast either.
                    finite = False
                if not finite:
                    # A bool, a string or a NaN/Inf all mean the same thing for
                    # the paired contrast: there is no finite loss to analyse.
                    facts.append("non_finite_loss")
        curve = run.get("curve_classification")
        if isinstance(curve, str) and curve in _DIVERGED_CLASSIFICATIONS:
            facts.append("diverged_curve")
        facts = list(dict.fromkeys(facts))
        unknown = [fact for fact in facts if fact not in rule_action]
        actions = [rule_action[fact] for fact in facts if fact in rule_action]
        if "exclude" in actions:
            status = "excluded"
        elif "sensitivity" in actions:
            status = "sensitivity"
        else:
            status = "analyzed"
        classified.append({
            "run_id": run_id,
            "arm": arm,
            "seed": run.get("seed"),
            "val_loss": loss if present else None,
            "curve_classification": curve,
            "reasons": facts,
            "unknown_reasons": unknown,
            "actions": actions,
            "status": status,
        })
    return classified


def run_exclusion_audit(runs: Sequence[Mapping],
                        policy: Optional[Mapping] = None,
                        min_seeds: int = MIN_SEEDS_FOR_SIGNIFICANCE) -> dict:
    """Pre-specified attrition audit over all planned runs.

    Unlike a pooled mean, this report keeps the denominator: every planned
    run is either retained, flagged as a sensitivity case, or excluded with a
    reason that was fixed in advance. It fails closed when an arm loses so
    many runs that it can no longer reach ``min_seeds`` independent
    replications, or when a run id or seed is duplicated. Differential
    attrition across arms is a warning: a random non-finite run can
    legitimately hit one arm, but an arm that lost most of its runs must not
    be read as a clean mean over the survivors.
    """
    if (isinstance(min_seeds, bool)
            or not isinstance(min_seeds, numbers.Integral) or min_seeds < 1):
        raise ValueError(f"min_seeds must be a positive integer, got {min_seeds!r}")
    min_seeds = int(min_seeds)
    canonical = normalize_exclusion_policy(policy)
    fingerprint = exclusion_policy_fingerprint(canonical)
    classified = classify_run_exclusions(runs, canonical)
    violations: list[str] = []
    warnings: list[str] = []
    attrition_balanced = True
    if not classified:
        violations.append("no planned runs were supplied to the attrition audit")

    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []
    seen_seed_keys: set[tuple] = set()
    duplicate_seeds: list[str] = []
    invalid_seed_ids: list[str] = []
    by_reason: dict[str, int] = {}
    by_status: dict[str, int] = {"analyzed": 0, "sensitivity": 0, "excluded": 0}
    per_arm: dict[str, dict] = {}
    excluded_runs: list[dict] = []
    sensitivity_runs: list[dict] = []
    unknown_reasons: set[str] = set()
    for record in classified:
        run_id = record["run_id"]
        if run_id in seen_ids:
            duplicate_ids.append(run_id)
        seen_ids.add(run_id)
        seed = record.get("seed")
        # A boolean/float/string is not an independent replication id. It
        # must not inflate the per-arm seed count just because it can be
        # stringified; treat it as a structural failure instead. Accept
        # integral NumPy scalars like `_validate_seed`: they are exact integer
        # seed identities, and normalizing through `int` makes int/np.int64
        # duplicates compare equal.
        valid_seed = (seed is None
                      or (isinstance(seed, numbers.Integral)
                          and not isinstance(seed, bool)))
        if seed is not None and not valid_seed:
            invalid_seed_ids.append(f'{record["arm"]}::seed={seed!r}')
        if seed is not None and valid_seed:
            seed = int(seed)
            key = (record["arm"], str(seed))
            if key in seen_seed_keys:
                duplicate_seeds.append(f'{record["arm"]}::seed={seed}')
            seen_seed_keys.add(key)
        status = record["status"]
        by_status[status] += 1
        for reason in record["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
        unknown_reasons.update(record["unknown_reasons"])
        bucket = per_arm.setdefault(record["arm"], {
            "planned": 0, "analyzed": 0, "clean": 0, "sensitivity": 0,
            "excluded": 0, "seeds": set(),
        })
        bucket["planned"] += 1
        if status != "excluded":
            bucket["analyzed"] += 1
            if valid_seed and seed is not None:
                bucket["seeds"].add(str(seed))
        if status == "analyzed":
            bucket["clean"] += 1
        elif status == "sensitivity":
            bucket["sensitivity"] += 1
            sensitivity_runs.append(record)
        else:
            bucket["excluded"] += 1
            excluded_runs.append(record)

    for arm, bucket in sorted(per_arm.items()):
        bucket["seeds"] = sorted(bucket["seeds"])
        bucket["n_seeds_analyzed"] = len(bucket["seeds"])
        bucket["excluded_fraction"] = (
            bucket["excluded"] / bucket["planned"] if bucket["planned"] else 0.0)
        if bucket["n_seeds_analyzed"] < min_seeds:
            violations.append(
                f'arm {arm!r}: {bucket["n_seeds_analyzed"]} analyzable seed(s) '
                f'after pre-specified exclusions, below the certification '
                f'minimum of {min_seeds}')
    if len(per_arm) >= 2:
        # Per-arm seed counts are not enough: the paired contrast uses the
        # seeds shared by every arm. Arms with disjoint seed sets can each look
        # replicated while contributing zero analyzable pairs.
        seed_sets = [set(bucket["seeds"]) for bucket in per_arm.values()]
        shared_seeds = set.intersection(*seed_sets) if seed_sets else set()
        if not shared_seeds:
            violations.append(
                "no seed is shared by every analyzed arm; the planned paired "
                "contrast has zero analyzable pairs (per-arm seed counts alone "
                "do not establish a paired replication)")
    if duplicate_ids:
        violations.append(f"duplicate run id(s): {sorted(set(duplicate_ids))}")
    if duplicate_seeds:
        violations.append(f"duplicate seed(s) within an arm: {sorted(set(duplicate_seeds))}")
    if invalid_seed_ids:
        violations.append(
            "invalid seed id(s) (seed must be an integer): "
            f"{sorted(set(invalid_seed_ids))}")
    if unknown_reasons:
        warnings.append(
            "run fact(s) with no policy rule, treated as no exclusion: "
            f"{sorted(unknown_reasons)}")
    if by_status["excluded"] and len(per_arm) > 1:
        fractions = [bucket["excluded_fraction"] for bucket in per_arm.values()]
        if max(fractions) - min(fractions) >= 0.5:
            attrition_balanced = False
            warnings.append(
                "differential attrition across arms (excluded fraction range "
                f"{min(fractions):.2f}-{max(fractions):.2f}); compare the per-arm "
                "counts before reading a pooled mean")
    return {
        "schema": "adasquash.exclusion_audit.v1",
        "policy_fingerprint": fingerprint,
        "policy": canonical,
        "min_seeds": min_seeds,
        "n_planned": len(classified),
        "n_analyzed": by_status["analyzed"] + by_status["sensitivity"],
        "n_clean": by_status["analyzed"],
        "n_sensitivity": by_status["sensitivity"],
        "n_excluded": by_status["excluded"],
        "by_status": by_status,
        "by_reason": by_reason,
        "per_arm": per_arm,
        "excluded_runs": excluded_runs,
        "sensitivity_runs": sensitivity_runs,
        "violations": violations,
        "warnings": warnings,
        "ready": not violations,
        "attrition_balanced": attrition_balanced,
        "protocol": (
            "every planned run is either retained, flagged for a sensitivity "
            "check, or excluded for a pre-specified outcome-blind protocol "
            "failure; an arm below the replication minimum cannot be certified"),
    }


def prospective_power_plan(paired_sd: float, n_pairs: int, margin: float,
                           *, alpha: float = 0.05, power: float = 0.8,
                           n_tests: int = 1, expected_attrition: float = 0.0,
                           max_pairs: int = 100000) -> dict:
    """Prospective sample-size plan with multiplicity and attrition.

    ``paired_required_n`` answers "how many pairs detect ``margin`` at
    ``alpha``". A confirmatory study asks two harder questions before launch:

    * **Multiplicity.** The study makes ``n_tests`` pre-registered claims, not
      one. Planning every comparison at the unadjusted ``alpha`` makes the
      planned power for the family optimistic: the test that eventually
      survives Holm must clear a smaller threshold. This helper plans at the
      Bonferroni family level ``alpha / max(n_tests, 1)``. Bonferroni is deliberately
      conservative and the analysis itself may use the more powerful Holm
      step-down; a design that is adequate under the conservative planning
      threshold is also adequate under Holm, whereas the reverse is not.
    * **Attrition.** Runs can be lost for pre-specified protocol reasons
      (missing artifact, non-finite loss, provenance mismatch). A *complete
      pair* requires both arms of the same seed, so if each planned run is
      lost independently with probability ``expected_attrition`` the expected
      number of analyzable pairs is ``floor(n_pairs * (1 - a)^2)``. Planning
      for the nominal seed count alone therefore overstates the final n.

    ``margin`` is the smallest effect of scientific interest fixed in advance,
    never the observed effect. The return value keeps the unadjusted and
    family-adjusted required n side by side, the planned n after attrition
    inflation, and the MDE at the expected analyzable n, so a null result
    cannot be attributed to a design that never had resolution.
    """
    sd = _finite_float(paired_sd)
    if sd is None or sd < 0.0:
        raise ValueError(f"paired_sd must be a finite number >= 0, got {paired_sd!r}")
    if (isinstance(n_pairs, bool)
            or not isinstance(n_pairs, numbers.Integral) or n_pairs < 2):
        raise ValueError(f"n_pairs must be an integer >= 2, got {n_pairs!r}")
    n_pairs = int(n_pairs)
    margin_value = _finite_float(margin)
    if margin_value is None or margin_value <= 0.0:
        raise ValueError(f"margin must be a finite number > 0, got {margin!r}")
    alpha_value = _finite_float(alpha)
    if alpha_value is None or not 0.0 < alpha_value < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    power_value = _finite_float(power)
    if power_value is None or not 0.0 < power_value < 1.0:
        raise ValueError(f"power must be strictly between 0 and 1, got {power!r}")
    if (isinstance(n_tests, bool)
            or not isinstance(n_tests, numbers.Integral) or n_tests < 1):
        raise ValueError(f"n_tests must be a positive integer, got {n_tests!r}")
    n_tests = int(n_tests)
    attrition = _finite_float(expected_attrition)
    if attrition is None or not 0.0 <= attrition < 1.0:
        raise ValueError(
            f"expected_attrition must be a probability in [0, 1), got "
            f"{expected_attrition!r}")
    if (isinstance(max_pairs, bool)
            or not isinstance(max_pairs, numbers.Integral) or max_pairs < 2):
        raise ValueError(f"max_pairs must be an integer >= 2, got {max_pairs!r}")
    max_pairs = int(max_pairs)

    try:
        family_alpha = alpha_value / max(n_tests, 1)
    except OverflowError:
        raise ValueError(
            f"n_tests={n_tests!r} is too large: alpha / max(n_tests, 1) exceeds the "
            f"float range") from None
    if family_alpha == 0.0:
        raise ValueError(
            f"n_tests={n_tests!r} is too large: alpha / max(n_tests, 1) underflows to "
            f"zero")
    required_unadjusted = paired_required_n(
        sd, margin_value, alpha=alpha_value, power=power_value, max_n=max_pairs)
    required_adjusted = paired_required_n(
        sd, margin_value, alpha=family_alpha, power=power_value, max_n=max_pairs)
    pair_survival = (1.0 - attrition) ** 2
    expected_pairs = _complete_pairs_after_attrition(n_pairs, pair_survival)
    required_planned = (
        None if required_adjusted is None else
        # The tiny subtraction matches the `+1e-12` tolerance used for
        # `expected_pairs`: without it a mathematically exact 200 planned
        # pairs can become 201 when `(1 - attrition) ** 2` rounds just below
        # its decimal value (e.g. attrition=0.8).
        _pairs_needed_after_attrition(required_adjusted, pair_survival))
    planned_unadjusted = paired_design_summary(
        sd, n_pairs, margin=margin_value, alpha=alpha_value, power=power_value)
    planned_summary = paired_design_summary(
        sd, n_pairs, margin=margin_value, alpha=family_alpha, power=power_value)
    effective_summary = (paired_design_summary(
        sd, expected_pairs, margin=margin_value, alpha=family_alpha,
        power=power_value) if expected_pairs >= 2 else None)
    adequate_unadjusted = (None if required_unadjusted is None
                           else bool(n_pairs >= required_unadjusted))
    adequate_adjusted = (None if required_adjusted is None
                         else bool(n_pairs >= required_adjusted))
    adequate_after_attrition = (
        None if required_adjusted is None else
        bool(expected_pairs >= 2 and expected_pairs >= required_adjusted))
    if required_adjusted is None:
        note = ("the pre-specified margin is not detectable within max_pairs at "
                "the family-level alpha even before attrition; increase the "
                "budget or reconsider the margin")
    elif required_planned is None:
        note = ("the attrition assumption leaves no positive complete-pair "
                "survival, so a finite planned pair count does not exist")
    elif adequate_after_attrition:
        note = ("the planned seed count still covers the family-adjusted "
                "requirement after the expected attrition")
    else:
        note = ("underpowered after multiplicity and attrition: plan "
                f"{required_planned} seed pair(s), not {n_pairs}")
    return {
        "schema": "adasquash.prospective_power.v1",
        "paired_sd": sd,
        "n_pairs": n_pairs,
        "margin": margin_value,
        "alpha": alpha_value,
        "power_target": power_value,
        "n_tests": n_tests,
        "multiplicity_method": "bonferroni",
        "family_alpha": family_alpha,
        "expected_attrition": attrition,
        "complete_pair_survival": pair_survival,
        "expected_analyzable_pairs": expected_pairs,
        "required_pairs_unadjusted": required_unadjusted,
        "required_pairs_family_adjusted": required_adjusted,
        "required_planned_pairs": required_planned,
        "shortfall_pairs": (None if required_planned is None
                            else max(0, required_planned - n_pairs)),
        "minimum_detectable_delta_unadjusted": planned_unadjusted[
            "minimum_detectable_delta"],
        "minimum_detectable_delta_family_adjusted": planned_summary[
            "minimum_detectable_delta"],
        "minimum_detectable_delta_after_attrition": (
            effective_summary["minimum_detectable_delta"]
            if effective_summary else None),
        "adequately_powered_unadjusted": adequate_unadjusted,
        "adequately_powered_family_adjusted": adequate_adjusted,
        "adequately_powered_after_attrition": adequate_after_attrition,
        "note": note,
    }


def tost_power_plan(paired_sd: float, n_pairs: int,
                    equivalence_margin: float, *, alpha: float = 0.05,
                    power: float = 0.8, n_tests: int = 1,
                    expected_attrition: float = 0.0,
                    max_pairs: int = 100000) -> dict:
    """Prospective sample-size plan for a TOST equivalence claim.

    This mirrors :func:`prospective_power_plan` but uses the equivalence
    power calculation. Substituting the equivalence bound into the
    superiority formula would answer "how often would a difference of this
    size be declared *superior*", which is not the TOST question and
    understates the pairs needed near the bound.

    ``n_tests`` applies the same conservative Bonferroni family correction
    to the TOST alpha, and ``expected_attrition`` inflates the planned pair
    count by the probability that both arms of a seed survive. The margin is
    the pre-specified equivalence bound; it must not be chosen from the
    observed interval.
    """
    sd = _finite_float(paired_sd)
    if sd is None or sd < 0.0:
        raise ValueError(
            f"paired_sd must be a finite number >= 0, got {paired_sd!r}")
    if (isinstance(n_pairs, bool)
            or not isinstance(n_pairs, numbers.Integral) or n_pairs < 2):
        raise ValueError(f"n_pairs must be an integer >= 2, got {n_pairs!r}")
    n_pairs = int(n_pairs)
    margin_value = _finite_float(equivalence_margin)
    if margin_value is None or margin_value <= 0.0:
        raise ValueError(
            f"equivalence_margin must be a finite number > 0, got "
            f"{equivalence_margin!r}")
    alpha_value = _finite_float(alpha)
    if alpha_value is None or not 0.0 < alpha_value < 0.5:
        raise ValueError(
            f"alpha must be strictly between 0 and 0.5 for a TOST, got "
            f"{alpha!r}")
    power_value = _finite_float(power)
    if power_value is None or not 0.0 < power_value < 1.0:
        raise ValueError(
            f"power must be strictly between 0 and 1, got {power!r}")
    if (isinstance(n_tests, bool)
            or not isinstance(n_tests, numbers.Integral) or n_tests < 1):
        raise ValueError(
            f"n_tests must be a positive integer, got {n_tests!r}")
    n_tests = int(n_tests)
    attrition = _finite_float(expected_attrition)
    if attrition is None or not 0.0 <= attrition < 1.0:
        raise ValueError(
            f"expected_attrition must be a probability in [0, 1), got "
            f"{expected_attrition!r}")
    if (isinstance(max_pairs, bool)
            or not isinstance(max_pairs, numbers.Integral) or max_pairs < 2):
        raise ValueError(
            f"max_pairs must be an integer >= 2, got {max_pairs!r}")
    max_pairs = int(max_pairs)

    try:
        family_alpha = alpha_value / max(n_tests, 1)
    except OverflowError:
        raise ValueError(
            f"n_tests={n_tests!r} is too large: alpha / max(n_tests, 1) exceeds the "
            f"float range") from None
    if family_alpha == 0.0:
        raise ValueError(
            f"n_tests={n_tests!r} is too large: alpha / max(n_tests, 1) underflows to "
            f"zero")
    required_unadjusted = tost_required_n(
        sd, margin_value, alpha=alpha_value, power=power_value,
        max_n=max_pairs)
    required_adjusted = tost_required_n(
        sd, margin_value, alpha=family_alpha, power=power_value,
        max_n=max_pairs)
    pair_survival = (1.0 - attrition) ** 2
    expected_pairs = _complete_pairs_after_attrition(n_pairs, pair_survival)
    required_planned = (
        None if required_adjusted is None else
        _pairs_needed_after_attrition(required_adjusted, pair_survival))
    planned_unadjusted = tost_design_summary(
        sd, n_pairs, margin_value, alpha=alpha_value, power=power_value)
    planned_summary = tost_design_summary(
        sd, n_pairs, margin_value, alpha=family_alpha, power=power_value)
    effective_summary = (
        tost_design_summary(
            sd, expected_pairs, margin_value, alpha=family_alpha,
            power=power_value) if expected_pairs >= 2 else None)
    adequate_unadjusted = (
        None if required_unadjusted is None
        else bool(n_pairs >= required_unadjusted))
    adequate_adjusted = (
        None if required_adjusted is None
        else bool(n_pairs >= required_adjusted))
    adequate_after_attrition = (
        None if required_adjusted is None else
        bool(expected_pairs >= 2 and expected_pairs >= required_adjusted))
    if required_adjusted is None:
        note = ("the equivalence margin is not reachable within max_pairs at "
                "the family-level alpha even before attrition; increase the "
                "budget or reconsider the margin")
    elif required_planned is None:
        note = ("the attrition assumption leaves no positive complete-pair "
                "survival, so a finite planned pair count does not exist")
    elif adequate_after_attrition:
        note = ("the planned seed count still covers the family-adjusted "
                "TOST requirement after the expected attrition")
    else:
        note = ("underpowered for equivalence after multiplicity and "
                f"attrition: plan {required_planned} seed pair(s), not "
                f"{n_pairs}")
    return {
        "schema": "adasquash.tost_power.v1",
        "paired_sd": sd,
        "n_pairs": n_pairs,
        "equivalence_margin": margin_value,
        "alpha": alpha_value,
        "power_target": power_value,
        "n_tests": n_tests,
        "multiplicity_method": "bonferroni",
        "family_alpha": family_alpha,
        "expected_attrition": attrition,
        "complete_pair_survival": pair_survival,
        "expected_analyzable_pairs": expected_pairs,
        "required_pairs_unadjusted": required_unadjusted,
        "required_pairs_family_adjusted": required_adjusted,
        "required_planned_pairs": required_planned,
        "shortfall_pairs": (None if required_planned is None
                            else max(0, required_planned - n_pairs)),
        "approximate_power_unadjusted": planned_unadjusted[
            "approximate_power_at_n"],
        "approximate_power_family_adjusted": planned_summary[
            "approximate_power_at_n"],
        "approximate_power_after_attrition": (
            effective_summary["approximate_power_at_n"]
            if effective_summary else None),
        "minimum_detectable_equivalence_margin_unadjusted":
            planned_unadjusted["minimum_detectable_equivalence_margin"],
        "minimum_detectable_equivalence_margin_family_adjusted":
            planned_summary["minimum_detectable_equivalence_margin"],
        "adequately_powered_unadjusted": adequate_unadjusted,
        "adequately_powered_family_adjusted": adequate_adjusted,
        "adequately_powered_after_attrition": adequate_after_attrition,
        "note": note,
    }


def factorial_contrast_sd(simple_paired_sd: float, n_factors: int,
                          term_size: int) -> float:
    """Per-seed SD of a factorial term, derived from a simple paired SD.

    A factorial design replaces one arm-vs-arm contrast with several
    simultaneous ones, and every extra term makes the *contrast itself*
    noisier. Planning the seed count of a factorial study with the SD of a
    simple paired difference therefore understates the required replication
    for the interaction claims and overstates it for a pooled main effect.

    Derivation (the same contrast construction the analysis code uses). Let
    the cells of a ``2**k`` factorial share the seed, let ``sigma`` be the
    across-seed SD of one cell's metric and ``rho`` the correlation between
    two cells of the same seed. A simple paired contrast between two cells
    has SD ``sigma * sqrt(2 * (1 - rho))``. An orthogonal sum-to-zero term
    over ``t`` of the ``k`` factors carries the coefficients ``+-2**-(k-t)``
    on the ``2**k`` cells, so its per-seed variance is
    ``sigma**2 * (1 - rho) * 2**(2*t - k)`` and its SD is the simple paired
    SD times ``2**(t - k/2 - 1/2)``:

    ==========  ==================  ==========================
    design      term                SD multiplier
    ==========  ==================  ==========================
    2x2         main effect         ``1 / sqrt(2)``
    2x2         interaction         ``sqrt(2)``
    2x2x2       main effect         ``1 / 2``
    2x2x2       two-way interaction ``1``
    2x2x2       three-way           ``2``
    ==========  ==================  ==========================

    The multipliers cancel the unknown ``sigma`` and ``rho``, which is why
    the pilot's plain paired SD is enough to plan a factorial study.
    """
    sd = _finite_float(simple_paired_sd)
    if sd is None or sd < 0.0:
        raise ValueError(
            f"simple_paired_sd must be a finite number >= 0, got {simple_paired_sd!r}")
    if (isinstance(n_factors, bool)
            or not isinstance(n_factors, numbers.Integral) or n_factors < 1):
        raise ValueError(f"n_factors must be a positive integer, got {n_factors!r}")
    n_factors = int(n_factors)
    if (isinstance(term_size, bool)
            or not isinstance(term_size, numbers.Integral) or term_size < 1):
        raise ValueError(f"term_size must be a positive integer, got {term_size!r}")
    term_size = int(term_size)
    if term_size > n_factors:
        raise ValueError(
            f"term_size {term_size} exceeds n_factors {n_factors}: a term "
            f"cannot involve more factors than the design has")
    multiplier = 2.0 ** (term_size - n_factors / 2.0 - 0.5)
    return sd * multiplier


def factorial_power_plan(simple_paired_sd: float, n_pairs: int, *,
                         n_factors: int = 2, term_size: int = 2,
                         margin: float = 0.0, alpha: float = 0.05,
                         power: float = 0.8, n_tests: int = 1,
                         expected_attrition: float = 0.0,
                         max_pairs: int = 100000,
                         equivalence_margin: float = 0.0) -> dict:
    """Prospective power plan for a factorial term (main effect or interaction).

    The repo's central proposition is that the per-channel, depth and
    activation axes are *orthogonal* and *synergistic*; those are interaction
    claims, and an interaction claim is a different test with a different
    noise level than the arm-vs-arm comparison the rest of the design is
    powered for. This helper plans the seed count of such a claim with the
    same machinery as :func:`prospective_power_plan` (Bonferroni family level
    over the pre-registered confirmatory tests, expected run attrition,
    minimum detectable effect) after converting the planning SD into the
    term's own contrast SD with :func:`factorial_contrast_sd`.

    ``margin`` is the smallest interaction of scientific interest: the
    deviation from perfect additivity that the study must be able to see. It
    is the *interaction* scale, so it is not comparable with the baseline
    margin of a simple contrast. ``equivalence_margin`` is the bound inside
    which the interaction is declared zero - the "the axes are orthogonal"
    claim - and is planned with the TOST power calculation, because a
    non-significant interaction test cannot certify orthogonality.

    The returned plan reports both readings side by side, so a study cannot
    quietly claim orthogonality with a design that was only powered to detect
    large deviations.
    """
    sd = _finite_float(simple_paired_sd)
    if sd is None or sd < 0.0:
        raise ValueError(
            f"simple_paired_sd must be a finite number >= 0, got {simple_paired_sd!r}")
    if (isinstance(n_pairs, bool)
            or not isinstance(n_pairs, numbers.Integral) or n_pairs < 2):
        raise ValueError(f"n_pairs must be an integer >= 2, got {n_pairs!r}")
    n_pairs = int(n_pairs)
    contrast_sd = factorial_contrast_sd(sd, n_factors, term_size)
    margin_value = 0.0 if margin is None else _finite_float(margin)
    if margin_value is None or margin_value < 0.0:
        raise ValueError(f"margin must be a finite number >= 0, got {margin!r}")
    equivalence_value = (0.0 if equivalence_margin is None
                         else _finite_float(equivalence_margin))
    if equivalence_value is None or equivalence_value < 0.0:
        raise ValueError(
            f"equivalence_margin must be a finite number >= 0, got "
            f"{equivalence_margin!r}")

    superiority = None
    if margin_value is not None and margin_value > 0.0:
        superiority = prospective_power_plan(
            contrast_sd, n_pairs, margin_value, alpha=alpha, power=power,
            n_tests=n_tests, expected_attrition=expected_attrition,
            max_pairs=max_pairs)
    equivalence = None
    if equivalence_value > 0.0:
        equivalence = tost_power_plan(
            contrast_sd, n_pairs, equivalence_value, alpha=alpha, power=power,
            n_tests=n_tests, expected_attrition=expected_attrition,
            max_pairs=max_pairs)

    multiplier = (contrast_sd / sd) if sd > 0.0 else None
    if superiority is not None and equivalence is not None:
        note = ("the design is planned for a superiority claim on the "
                "interaction and for the equivalence (orthogonality) claim at "
                "the same time; both readings must be satisfied")
    elif superiority is not None:
        note = ("the design is planned for a superiority claim on the "
                "interaction only; an equivalence (orthogonality) claim needs "
                "its own equivalence_margin")
    elif equivalence is not None:
        note = ("the design is planned for the equivalence (orthogonality) "
                "claim only; a superiority claim on the interaction needs its "
                "own margin")
    else:
        note = ("no interaction margin and no equivalence margin were given: "
                "the interaction claims of this study are descriptive only")
    return {
        "schema": "adasquash.factorial_power.v1",
        "simple_paired_sd": sd,
        "n_pairs": n_pairs,
        "n_factors": int(n_factors),
        "term_size": int(term_size),
        "contrast_sd": contrast_sd,
        "contrast_sd_multiplier": multiplier,
        "margin": margin_value if margin_value and margin_value > 0.0 else None,
        "equivalence_margin": (equivalence_value
                               if equivalence_value and equivalence_value > 0.0
                               else None),
        "alpha": float(alpha),
        "power_target": float(power),
        "n_tests": int(n_tests),
        "expected_attrition": float(expected_attrition),
        "superiority": superiority,
        "equivalence": equivalence,
        "note": note,
    }


PAIRED_SD_CALIBRATION_SCHEMA = "adasquash.paired_sd_calibration.v1"


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    middle = n // 2
    if n % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _chi2_lower_quantile(probability: float, df: float) -> float:
    """``x`` with ``P(chi2_df <= x) == probability``.

    The upper confidence bound for a variance needs the lower chi-square
    quantile, and the repository deliberately has no scipy dependency. The
    survival function is already implemented for the power tests, so invert
    it by bisection instead of adding a second approximation with its own
    error.
    """
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"chi-square probability must be strictly between 0 and 1, got "
            f"{probability!r}")
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError(f"chi-square df must be positive and finite, got {df!r}")
    target_tail = 1.0 - probability
    lo = 0.0
    hi = max(1.0, df)
    for _ in range(400):
        if chi2_sf(hi, df) <= target_tail:
            break
        hi *= 2.0
        if not math.isfinite(hi):
            raise ValueError(
                "chi-square quantile overflowed; the confidence level is too "
                "close to 1 for the available precision")
    else:
        raise ValueError("chi-square quantile did not converge")
    for _ in range(400):
        mid = (lo + hi) / 2.0
        if mid == lo or mid == hi:
            break
        if chi2_sf(mid, df) > target_tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _paired_sd_fingerprint(payload: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def paired_sd_calibration(differences, *,
                          pilot_seeds: Optional[Iterable] = None,
                          confirmatory_seeds: Optional[Iterable] = None,
                          confidence: float = 0.90,
                          min_pairs: int = 2) -> dict:
    """Estimate the planning paired SD from *pilot* runs, with provenance.

    Two design mistakes are easy to make before a confirmatory launch:

    * the planned SD is a single lucky number from a handful of pilot seeds.
      The sample SD of a small pilot is itself very uncertain, so planning at
      the point estimate is optimistic: the realised power is below target
      whenever the true SD is larger. This helper returns the sample SD, a
      robust MAD-based SD, and a one-sided ``confidence`` upper bound. The
      upper bound is the recommended planning value because it is the value
      the pilot data do not rule out.
    * the same seed is used in the pilot and in the confirmatory study. That
      is not independent replication: the two "runs" share the initialization
      and data order, and the pilot outcome can leak into the planning
      decision. ``confirmatory_seeds`` is compared with the pilot seeds and an
      overlap is reported (the readiness audit then fails closed).

    ``differences`` is an iterable of paired differences in the reports'
    sign convention (candidate minus reference; negative means the candidate
    has the lower loss). Each item may be a finite number or a mapping with
    ``difference``/``delta`` and an optional integer ``seed``. If the items
    carry no seed, pass ``pilot_seeds`` alongside; otherwise the seed
    provenance is recorded as unknown rather than invented.

    This is a preparation artifact, not a result: the pilot mean is reported
    for auditing but must not be used as the confirmatory effect size or to
    choose the scientifically important margin after seeing it.
    """
    if isinstance(differences, (str, bytes, bytearray)) or isinstance(differences, Mapping):
        raise ValueError("differences must be an iterable of paired differences")
    try:
        raw_items = list(differences)
    except TypeError:
        raise ValueError(
            "differences must be an iterable of paired differences") from None
    confidence_value = _finite_float(confidence)
    if confidence_value is None or not 0.5 < confidence_value < 1.0:
        raise ValueError(
            f"confidence must be strictly between 0.5 and 1, got {confidence!r}")
    if (isinstance(min_pairs, bool)
            or not isinstance(min_pairs, numbers.Integral) or min_pairs < 2):
        raise ValueError(f"min_pairs must be an integer >= 2, got {min_pairs!r}")
    min_pairs = int(min_pairs)

    seed_list = None
    if pilot_seeds is not None:
        if isinstance(pilot_seeds, (str, bytes)):
            raise ValueError("pilot_seeds must be an iterable of integer seeds")
        seed_list = [_validate_seed(seed) for seed in pilot_seeds]
    confirmatory = None
    if confirmatory_seeds is not None:
        if isinstance(confirmatory_seeds, (str, bytes)):
            raise ValueError(
                "confirmatory_seeds must be an iterable of integer seeds")
        confirmatory = [_validate_seed(seed) for seed in confirmatory_seeds]
        if len(set(confirmatory)) != len(confirmatory):
            raise ValueError("confirmatory_seeds must be unique")

    values: list[float] = []
    seeds: list[Optional[int]] = []
    for index, item in enumerate(raw_items):
        item_seed = None
        if isinstance(item, Mapping):
            if "difference" in item:
                raw_value = item.get("difference")
            elif "delta" in item:
                raw_value = item.get("delta")
            else:
                raise ValueError(
                    f'difference record {index} needs a "difference" (or '
                    f'"delta") field')
            item_seed = item.get("seed")
        else:
            raw_value = item
        value = _finite_float(raw_value)
        if value is None:
            raise ValueError(
                f"paired difference {index} must be a finite real number, got "
                f"{raw_value!r}")
        if item_seed is not None:
            item_seed = _validate_seed(item_seed)
        if seed_list is not None:
            if index >= len(seed_list):
                raise ValueError(
                    "pilot_seeds must have one entry per paired difference")
            if item_seed is not None and item_seed != seed_list[index]:
                raise ValueError(
                    f"difference record {index} carries seed {item_seed}, but "
                    f"pilot_seeds[{index}] is {seed_list[index]}")
            item_seed = seed_list[index]
        values.append(float(value))
        seeds.append(item_seed)
    if seed_list is not None and len(seed_list) != len(values):
        raise ValueError(
            f"pilot_seeds has {len(seed_list)} entries for {len(values)} "
            f"paired difference(s)")
    if len(values) < min_pairs:
        raise ValueError(
            f"a paired-SD calibration needs at least {min_pairs} finite pair(s), "
            f"got {len(values)}")
    present_seeds = [seed for seed in seeds if seed is not None]
    if len(set(present_seeds)) != len(present_seeds):
        duplicates = sorted({seed for seed in present_seeds
                             if present_seeds.count(seed) > 1})
        raise ValueError(
            f"pilot seeds must identify independent runs; duplicated: {duplicates}")
    if present_seeds and len(present_seeds) != len(values):
        raise ValueError(
            "either every pilot difference must carry its seed or none may; "
            "a partial seed table cannot prove seed-level independence")

    stats = mean_std_stderr(values)
    if stats is None:
        raise ValueError("the pilot differences have no finite statistics")
    n = int(stats["n"])
    mean = float(stats["mean"])
    sd = float(stats["std"])
    df = n - 1
    if sd == 0.0:
        upper = 0.0
        robust_sd = 0.0
    else:
        lower_quantile = _chi2_lower_quantile(1.0 - confidence_value, float(df))
        factor = math.sqrt(df / max(lower_quantile, 1e-10))
        candidate = sd * factor
        upper = candidate if math.isfinite(candidate) else None
        try:
            median = _median(values)
            mad = _median([abs(value - median) for value in values])
            robust = 1.4826 * mad
            robust_sd = robust if math.isfinite(robust) else None
        except (OverflowError, ValueError):
            robust_sd = None

    if sd == 0.0:
        # Identical pilot differences estimate zero variance, and the formal
        # upper bound is also zero. Planning at zero would make every sample
        # size look sufficient, so the calibration refuses to recommend it.
        recommended = None
        recommended_basis = "zero_variance_pilot_not_usable"
    elif upper is None:
        # A finite point estimate is still a better fallback than refusing to
        # produce any planning value, but the report says the bound overflowed.
        recommended = sd if math.isfinite(sd) else None
        recommended_basis = "point_estimate_after_bound_overflow"
    else:
        recommended = upper
        recommended_basis = "one_sided_upper_confidence_bound"

    pilot_seed_list = present_seeds if present_seeds else []
    overlap = None
    disjoint = None
    if pilot_seed_list and confirmatory is not None:
        overlap = sorted(set(pilot_seed_list).intersection(confirmatory))
        disjoint = not overlap
    payload = {
        "differences": values,
        "pilot_seeds": pilot_seed_list,
        "confirmatory_seeds": confirmatory,
        "confidence": confidence_value,
        "min_pairs": min_pairs,
        "n_pairs": n,
        "mean_delta": mean,
        "paired_sd": sd,
        "recommended_paired_sd": recommended,
    }
    fingerprint = _paired_sd_fingerprint(payload)
    return {
        "schema": PAIRED_SD_CALIBRATION_SCHEMA,
        "fingerprint": fingerprint,
        "source": "pilot",
        "differences": values,
        "pilot_seeds": pilot_seed_list,
        "confirmatory_seeds": confirmatory,
        "n_pairs": n,
        "min_pairs": min_pairs,
        "df": df,
        "mean_delta": mean,
        "paired_sd": sd,
        "robust_paired_sd": robust_sd,
        "confidence": confidence_value,
        "sd_upper_confidence_bound": upper,
        "recommended_paired_sd": recommended,
        "recommended_basis": recommended_basis,
        "zero_variance": bool(sd == 0.0),
        "seed_overlap": overlap,
        "seeds_disjoint": disjoint,
        "note": (
            "plan at the upper confidence bound, not the pilot point estimate; "
            "the pilot mean is descriptive and must not be used to choose the "
            "smallest effect of interest after seeing it"),
    }


def paired_sd_evidence_audit(evidence: Mapping,
                             assumed_paired_sd=None,
                             confirmatory_seeds: Optional[Iterable] = None,
                             *, tolerance: float = 0.05,
                             min_pilot_pairs: int = 3) -> dict:
    """Check the frozen pilot calibration against the SD used for planning.

    The evidence is recomputed from its recorded differences rather than
    trusting the stored summary fields. A mismatch between the stored
    fingerprint and the recomputation is a failure: an edited calibration is
    no longer the artifact the pre-registration named. The assumed planning
    SD is then compared with the pilot's upper confidence bound:

    * below the bound (outside ``tolerance``) is a failure, because the
      planned power is optimistic relative to what the pilot can support;
    * at or above the bound passes; being above it is conservative and is
      noted rather than punished.

    Seed overlap with the confirmatory seeds fails as well: reusing a pilot
    seed in the confirmatory analysis is not independent replication.
    """
    if not isinstance(evidence, Mapping):
        raise ValueError(
            f"paired-SD evidence must be a mapping, got "
            f"{type(evidence).__name__}")
    if evidence.get("schema") != PAIRED_SD_CALIBRATION_SCHEMA:
        raise ValueError(
            f'paired-SD evidence schema must be '
            f'{PAIRED_SD_CALIBRATION_SCHEMA!r}, got {evidence.get("schema")!r}')
    tolerance_value = _finite_float(tolerance)
    if tolerance_value is None or not 0.0 <= tolerance_value < 1.0:
        raise ValueError(
            f"tolerance must be a finite probability in [0, 1), got "
            f"{tolerance!r}")
    if (isinstance(min_pilot_pairs, bool)
            or not isinstance(min_pilot_pairs, numbers.Integral)
            or min_pilot_pairs < 2):
        raise ValueError(
            f"min_pilot_pairs must be an integer >= 2, got "
            f"{min_pilot_pairs!r}")
    min_pilot_pairs = int(min_pilot_pairs)

    stored_fingerprint = evidence.get("fingerprint")
    try:
        recomputed = paired_sd_calibration(
            evidence.get("differences"),
            pilot_seeds=evidence.get("pilot_seeds"),
            confirmatory_seeds=evidence.get("confirmatory_seeds"),
            confidence=evidence.get("confidence"),
            min_pairs=evidence.get("min_pairs"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"paired-SD evidence is not a valid calibration record: {exc}") from None
    fingerprint_verified = (
        isinstance(stored_fingerprint, str)
        and stored_fingerprint == recomputed["fingerprint"])
    checks: list[dict] = []

    def add(check_id: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail})

    if fingerprint_verified:
        add("fingerprint", "pass",
            "stored fingerprint matches the recorded pilot differences")
    else:
        add("fingerprint", "fail",
            "stored fingerprint is missing or does not match the recorded "
            "pilot differences; the calibration was edited after it was frozen")

    n_pairs = int(recomputed["n_pairs"])
    if n_pairs < min_pilot_pairs:
        add("pilot_replication", "warn",
            f"{n_pairs} pilot pair(s) is a very small variance pilot; the "
            f"upper bound is wide and the design should not be read as "
            f"precisely powered (recommended >= {min_pilot_pairs})")
    else:
        add("pilot_replication", "pass",
            f"{n_pairs} pilot pair(s) supply the variance calibration")

    confirm = None
    if confirmatory_seeds is not None:
        if isinstance(confirmatory_seeds, (str, bytes)):
            raise ValueError(
                "confirmatory_seeds must be an iterable of integer seeds")
        confirm = [_validate_seed(seed) for seed in confirmatory_seeds]
        if len(set(confirm)) != len(confirm):
            raise ValueError("confirmatory_seeds must be unique")
    stored_confirm = recomputed.get("confirmatory_seeds")
    overlap = recomputed.get("seed_overlap")
    disjoint = recomputed.get("seeds_disjoint")
    pilot_seeds = recomputed.get("pilot_seeds") or []
    if confirm is not None and pilot_seeds:
        overlap = sorted(set(pilot_seeds).intersection(confirm))
        disjoint = not overlap
    if not pilot_seeds:
        add("seed_disjointness", "warn",
            "the pilot differences carry no seed provenance, so independence "
            "from the confirmatory seeds cannot be verified")
    elif disjoint is None:
        add("seed_disjointness", "warn",
            "confirmatory seeds were not supplied, so pilot/confirmatory seed "
            "overlap cannot be checked")
    elif disjoint:
        add("seed_disjointness", "pass",
            f"pilot seeds {pilot_seeds} are disjoint from the confirmatory "
            f"seeds {confirm if confirm is not None else stored_confirm}")
    else:
        add("seed_disjointness", "fail",
            f"pilot seed(s) {overlap} are reused in the confirmatory design; "
            f"this is not independent replication and lets a pilot outcome "
            f"enter the confirmatory analysis")

    recommended = recomputed.get("recommended_paired_sd")
    if recomputed.get("zero_variance"):
        add("variance_estimate", "warn",
            "the pilot paired differences have exactly zero variance; a zero "
            "planning SD is not credible from a small pilot and makes any "
            "sample size look sufficient")
    if assumed_paired_sd is None:
        if recommended is None:
            add("planning_sd", "fail",
                "no planning SD was supplied and the pilot calibration has no "
                "finite upper confidence bound")
        else:
            add("planning_sd", "warn",
                f"no planning SD was supplied; the pilot recommends "
                f"{recommended:g}, but the design audit did not use it")
    else:
        assumed = _finite_float(assumed_paired_sd)
        if assumed is None or assumed < 0.0:
            add("planning_sd", "fail",
                f"assumed paired SD must be a finite number >= 0, got "
                f"{assumed_paired_sd!r}")
        elif recommended is None or not math.isfinite(float(recommended)):
            add("planning_sd", "fail",
                "the pilot calibration has no finite planning SD to compare "
                "against")
        else:
            lower = float(recommended) * (1.0 - tolerance_value)
            upper = float(recommended) * (1.0 + tolerance_value)
            if assumed < lower:
                add("planning_sd", "fail",
                    f"assumed paired SD={assumed:g} is below the pilot upper "
                    f"confidence bound {float(recommended):g} (tolerance "
                    f"{tolerance_value:.0%}); the planned power is optimistic")
            elif assumed > upper:
                add("planning_sd", "pass",
                    f"assumed paired SD={assumed:g} is above the pilot upper "
                    f"confidence bound {float(recommended):g}; the plan is "
                    f"conservative")
            else:
                add("planning_sd", "pass",
                    f"assumed paired SD={assumed:g} matches the pilot upper "
                    f"confidence bound {float(recommended):g} within "
                    f"{tolerance_value:.0%}")

    n_fail = sum(1 for check in checks if check["status"] == "fail")
    n_warn = sum(1 for check in checks if check["status"] == "warn")
    status = "fail" if n_fail else ("warn" if n_warn else "pass")
    return {
        "schema": "adasquash.paired_sd_evidence_audit.v1",
        "status": status,
        "checks": checks,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "evidence_fingerprint": recomputed["fingerprint"],
        "stored_fingerprint": stored_fingerprint,
        "fingerprint_verified": fingerprint_verified,
        "n_pilot_pairs": n_pairs,
        "paired_sd": recomputed["paired_sd"],
        "recommended_paired_sd": recommended,
        "assumed_paired_sd": assumed_paired_sd,
        "seed_overlap": overlap,
        "seeds_disjoint": disjoint,
        "note": (
            "a confirmatory plan should use the pilot upper confidence bound; "
            "the point estimate is reported for comparison only"),
    }


def design_readiness_audit(seeds: Sequence[int], rows: Iterable[str],
                           hypotheses: Mapping[str, str],
                           schedule: Optional[dict] = None,
                           expected_units: Optional[Iterable[str]] = None,
                           schedule_arm_of=None,
                           margin: float = 0.0,
                           equivalence_margin: float = 0.0,
                           assumed_paired_sd: Optional[float] = None,
                           target_power: float = 0.8,
                           alpha: float = 0.05,
                           min_seeds: int = MIN_SEEDS_FOR_SIGNIFICANCE,
                           identity_controls: Sequence[Sequence[str]] = (),
                           exclusion_policy: Optional[Mapping] = None,
                           n_confirmatory_tests: int = 1,
                           expected_attrition: float = 0.0,
                           paired_sd_evidence: Optional[Mapping] = None,
                           require_variance_evidence: bool = False,
                           interaction_margin: float = 0.0,
                           interaction_equivalence_margin: float = 0.0,
                           n_factors: int = 2,
                           interaction_term_size: int = 2,
                           declares_interaction_claims: bool = False) -> dict:
    """Pre-flight design checklist, written before any run starts.

    Failures are structural and should stop a confirmatory launch; warnings are
    judgement calls that are recorded rather than hidden.  In particular, the
    absence of a pre-specified smallest effect is a warning, not a failure: an
    exploratory matrix may legitimately run without one, but it cannot make an
    equivalence claim later.

    ``n_confirmatory_tests`` and ``expected_attrition`` make the power check
    prospective rather than optimistic: the required n is computed at the
    Bonferroni family level and inflated for the expected loss of complete
    pairs (see :func:`prospective_power_plan`). The default of one test and no
    attrition reproduces the single-comparison calculation. When an
    ``equivalence_margin`` is supplied the audit adds a separate
    ``equivalence_power`` check based on :func:`tost_power_plan`; the
    superiority formula is not reused for a TOST claim.

    ``paired_sd_evidence`` is a :func:`paired_sd_calibration` record from
    pilot runs. When supplied, the audit adopts its conservative upper bound
    if no ``assumed_paired_sd`` was given, checks that the planning SD is not
    below that bound, and fails on pilot/confirmatory seed overlap. Set
    ``require_variance_evidence`` to make a missing calibration a structural
    failure instead of leaving the planning variance unproven.

    A factorial study makes *interaction* claims ("the axes are orthogonal",
    "the combination is synergistic"), and an interaction is a different
    contrast with a different noise level than the arm-vs-arm comparison the
    rest of the audit is powered for. ``interaction_margin`` (the smallest
    interaction of interest) and ``interaction_equivalence_margin`` (the
    bound inside which the interaction is declared zero) therefore add their
    own power checks, planned on the interaction scale with
    :func:`factorial_power_plan`. ``declares_interaction_claims`` marks a
    study that contains factorial designs, so a matrix that can produce
    interaction claims but never states an interaction margin is recorded as
    an unplanned claim instead of silently producing descriptive numbers.
    """
    names = list(rows.keys()) if isinstance(rows, Mapping) else list(rows)
    checks: list[dict] = []
    planned_power = None
    planned_equivalence_power = None
    paired_sd_evidence_report = None
    effective_paired_sd = assumed_paired_sd

    def add(check_id: str, status: str, detail: str) -> None:
        checks.append({"id": check_id, "status": status, "detail": detail})

    # `seeds or []` asks a NumPy array for its truth value, which raises
    # "truth value is ambiguous" before any check can run. A None default is
    # the only case that means "no seeds".
    seeds = [] if seeds is None else list(seeds)
    min_seeds_valid = (isinstance(min_seeds, numbers.Integral)
                       and not isinstance(min_seeds, bool)
                       and min_seeds >= 1)
    if not min_seeds_valid:
        # A floor of 0 (or a bool) would switch off both the replication check
        # and the attrition floor; the audit must not certify such a design.
        add("replication", "fail",
            f"invalid min_seeds {min_seeds!r}: must be a positive integer; "
            f"using {MIN_SEEDS_FOR_SIGNIFICANCE}")
        min_seeds = MIN_SEEDS_FOR_SIGNIFICANCE
    else:
        min_seeds = int(min_seeds)
    # The power check must use the number of *distinct* observations the
    # paired test will actually have, not the raw list length (duplicate seeds
    # are the same run and the replication check above rejects them). NumPy
    # integer scalars are exact seed identities too, matching `_validate_seed`;
    # normalizing through `int` also makes int/np.int64 duplicates compare
    # equal in the distinctness set.
    integral_seeds = [seed for seed in seeds
                      if isinstance(seed, numbers.Integral)
                      and not isinstance(seed, bool)]
    n_pairs = len({int(seed) for seed in integral_seeds})
    if min_seeds_valid and (
            len(seeds) < min_seeds
            or len(integral_seeds) != len(seeds)
            or n_pairs != len(seeds)):
        add("replication", "fail",
            f"{len(seeds)} seed(s) {seeds}: need >= {min_seeds} distinct integer seeds")
    elif min_seeds_valid:
        add("replication", "pass", f"{len(seeds)} distinct seeds: {seeds}")

    # The planning variance is a design assumption, so it needs provenance too.
    # A pilot point estimate is optimistic and cannot be checked after the
    # fact; a frozen calibration with disjoint seeds can.
    confirm_for_evidence = (integral_seeds
                            if len(integral_seeds) == len(seeds) else None)
    if paired_sd_evidence is None:
        if require_variance_evidence:
            add("variance_evidence", "fail",
                "no paired-SD calibration was supplied; the planning variance "
                "has no pilot provenance and --require-variance-evidence is set")
    else:
        try:
            adoption = paired_sd_evidence_audit(
                paired_sd_evidence, None, confirm_for_evidence)
        except ValueError as exc:
            add("variance_evidence", "fail",
                f"invalid paired-SD calibration: {exc}")
        else:
            if effective_paired_sd is None:
                recommended = adoption.get("recommended_paired_sd")
                if (recommended is not None
                        and isinstance(recommended, numbers.Real)
                        and not isinstance(recommended, bool)
                        and math.isfinite(float(recommended))):
                    effective_paired_sd = float(recommended)
            try:
                paired_sd_evidence_report = paired_sd_evidence_audit(
                    paired_sd_evidence, effective_paired_sd,
                    confirm_for_evidence)
            except ValueError as exc:
                add("variance_evidence", "fail",
                    f"invalid paired-SD calibration: {exc}")
            else:
                status = paired_sd_evidence_report["status"]
                detail = "; ".join(
                    f'{check["id"]}: {check["detail"]}'
                    for check in paired_sd_evidence_report["checks"]
                    if check["status"] != "pass")
                if not detail:
                    detail = ("pilot calibration and planning SD agree, and "
                              "pilot seeds are disjoint from the confirmatory "
                              "design")
                add("variance_evidence", status, detail)

    missing = sorted(name for name in names if not str(hypotheses.get(name) or "").strip())
    if missing:
        add("hypotheses", "fail", f"rows without a stated hypothesis: {missing}")
    else:
        add("hypotheses", "pass", f"all {len(names)} rows state a hypothesis")

    # `margin or 0.0` silently turned malformed falsy values (False, "",
    # []) into the legitimate "no margin specified" case. Only an actual
    # None means unspecified; every other value goes through validation.
    margin_value = 0.0 if margin is None else _finite_float(margin)
    equivalence_value = (0.0 if equivalence_margin is None
                         else _finite_float(equivalence_margin))
    if margin_value is None or equivalence_value is None:
        margin = equivalence_margin = float("nan")
    else:
        margin, equivalence_margin = margin_value, equivalence_value
    if not math.isfinite(margin) or margin < 0.0:
        add("effect_size", "fail", f"invalid superiority margin {margin!r}")
    elif not math.isfinite(equivalence_margin) or equivalence_margin < 0.0:
        add("effect_size", "fail", f"invalid equivalence margin {equivalence_margin!r}")
    elif margin <= 0.0 and equivalence_margin <= 0.0:
        add("effect_size", "warn",
            "no positive smallest effect of interest (--baseline-margin or "
            "--equivalence-margin); the study can rank arms but cannot claim "
            "practical significance or equivalence")
    else:
        add("effect_size", "pass",
            f"superiority margin={margin}, equivalence margin={equivalence_margin}")

    alpha_value = _finite_float(alpha)
    target_power_value = _finite_float(target_power)
    n_tests_valid = (isinstance(n_confirmatory_tests, numbers.Integral)
                     and not isinstance(n_confirmatory_tests, bool)
                     and n_confirmatory_tests >= 1)
    if n_tests_valid:
        n_confirmatory_tests = int(n_confirmatory_tests)
    if alpha_value is None or not 0.0 < alpha_value < 1.0:
        add("power", "fail", f"invalid alpha {alpha!r}")
    elif equivalence_margin > 0.0 and alpha_value >= 0.5:
        add("power", "fail",
            f"alpha={alpha_value:g} cannot support the requested TOST "
            "equivalence test: its confidence interval 1 - 2*alpha would be "
            "non-positive, so alpha must be < 0.5")
    elif target_power_value is None or not 0.0 < target_power_value < 1.0:
        add("power", "fail", f"invalid target power {target_power!r}")
    elif not n_tests_valid:
        add("power", "fail",
            f"invalid n_confirmatory_tests {n_confirmatory_tests!r}: must be a "
            "positive integer")
    elif effective_paired_sd is None:
        add("power", "warn",
            "no assumed paired SD supplied; planned power for the smallest effect "
            "cannot be checked before the run (the comparison report still "
            "reports the observed MDE afterwards)")
    else:
        sd = _finite_float(effective_paired_sd)
        target = margin
        if sd is None or sd < 0.0:
            add("power", "fail", f"invalid assumed paired SD {effective_paired_sd!r}")
        elif target <= 0.0:
            if equivalence_margin <= 0.0:
                add("power", "warn",
                    "a paired SD was supplied but no positive margin defines the "
                    "effect to power for")
        elif n_pairs < 2:
            # A paired design needs at least two distinct observations before
            # its power can be estimated. Reporting a crash here would turn a
            # failed pre-flight check into an unusable audit.
            add("power", "fail",
                f"{n_pairs} distinct seed(s): at least 2 paired observations are "
                f"needed to estimate power for margin={target:g}")
        else:
            try:
                plan = prospective_power_plan(
                    sd, n_pairs, target, alpha=float(alpha),
                    power=float(target_power),
                    n_tests=n_confirmatory_tests,
                    expected_attrition=expected_attrition)
            except ValueError as exc:
                add("power", "fail", f"invalid power-planning input: {exc}")
            else:
                planned_power = plan
                family_note = (
                    f'{plan["n_tests"]} test(s) at family alpha '
                    f'{plan["family_alpha"]:.4g}' if plan["n_tests"] > 1 else
                    "one test at alpha")
                if plan["adequately_powered_after_attrition"] is True:
                    add("power", "pass",
                        f'{n_pairs} pairs at paired SD={sd:g} reach the '
                        f'{target:g} target ({family_note}); '
                        f'{plan["required_planned_pairs"]} planned pairs needed '
                        f'after {plan["expected_attrition"]:.0%} attrition')
                elif plan["adequately_powered_after_attrition"] is False:
                    add("power", "fail",
                        f'{n_pairs} pairs at paired SD={sd:g} are underpowered for '
                        f'margin={target:g} ({family_note}): '
                        f'{plan["required_pairs_family_adjusted"]} analyzable pairs '
                        f'are required, so plan '
                        f'{plan["required_planned_pairs"]} pairs after '
                        f'{plan["expected_attrition"]:.0%} attrition')
                else:
                    add("power", "warn",
                        f'margin={target:g} is not reachable within the search cap '
                        f'at paired SD={sd:g} and family alpha '
                        f'{plan["family_alpha"]:.4g}')

    if equivalence_margin > 0.0 and effective_paired_sd is not None:
        eq_sd = _finite_float(effective_paired_sd)
        if eq_sd is None or eq_sd < 0.0:
            add("equivalence_power", "fail",
                f"invalid assumed paired SD {effective_paired_sd!r}")
        elif n_pairs < 2:
            add("equivalence_power", "fail",
                f"{n_pairs} distinct seed(s): at least 2 paired observations "
                f"are needed to plan a TOST at margin="
                f"{equivalence_margin:g}")
        else:
            try:
                eq_plan = tost_power_plan(
                    eq_sd, n_pairs, equivalence_margin,
                    alpha=alpha_value, power=target_power_value,
                    n_tests=n_confirmatory_tests,
                    expected_attrition=expected_attrition)
            except ValueError as exc:
                add("equivalence_power", "fail",
                    f"invalid equivalence power-planning input: {exc}")
            else:
                planned_equivalence_power = eq_plan
                family_note = (
                    f'{eq_plan["n_tests"]} test(s) at family alpha '
                    f'{eq_plan["family_alpha"]:.4g}'
                    if eq_plan["n_tests"] > 1 else "one test at alpha")
                if eq_plan["adequately_powered_after_attrition"] is True:
                    add("equivalence_power", "pass",
                        f'{n_pairs} pairs at paired SD={eq_sd:g} reach the '
                        f'{target_power:g} TOST target for margin='
                        f'{equivalence_margin:g} ({family_note}); '
                        f'{eq_plan["required_planned_pairs"]} planned pairs '
                        f'needed after {eq_plan["expected_attrition"]:.0%} '
                        f'attrition')
                elif eq_plan["adequately_powered_after_attrition"] is False:
                    add("equivalence_power", "fail",
                        f'{n_pairs} pairs at paired SD={eq_sd:g} are '
                        f'underpowered for TOST equivalence at margin='
                        f'{equivalence_margin:g} ({family_note}): '
                        f'{eq_plan["required_pairs_family_adjusted"]} '
                        f'analyzable pairs are required, so plan '
                        f'{eq_plan["required_planned_pairs"]} pairs after '
                        f'{eq_plan["expected_attrition"]:.0%} attrition')
                else:
                    add("equivalence_power", "warn",
                        f'equivalence margin={equivalence_margin:g} is not '
                        f'reachable within the search cap at paired SD='
                        f'{eq_sd:g} and family alpha '
                        f'{eq_plan["family_alpha"]:.4g}')

    interaction_value = _finite_float(interaction_margin)
    interaction_equivalence_value = _finite_float(interaction_equivalence_margin)
    if interaction_value is None or interaction_value < 0.0:
        add("interaction_effect_size", "fail",
            f"invalid interaction margin {interaction_margin!r}")
        interaction_value = 0.0
    if interaction_equivalence_value is None or interaction_equivalence_value < 0.0:
        add("interaction_effect_size", "fail",
            f"invalid interaction equivalence margin {interaction_equivalence_margin!r}")
        interaction_equivalence_value = 0.0
    factors_valid = (isinstance(n_factors, numbers.Integral)
                     and not isinstance(n_factors, bool) and n_factors >= 1)
    term_valid = (isinstance(interaction_term_size, numbers.Integral)
                  and not isinstance(interaction_term_size, bool)
                  and 1 <= interaction_term_size <= max(1, int(n_factors or 1)))
    if not factors_valid or not term_valid:
        add("interaction_design", "fail",
            f"invalid factorial geometry: n_factors={n_factors!r}, "
            f"interaction_term_size={interaction_term_size!r} (the term may "
            f"not involve more factors than the design has)")
        n_factors = 2
        interaction_term_size = 2
    if (declares_interaction_claims and interaction_value <= 0.0
            and interaction_equivalence_value <= 0.0):
        add("interaction_effect_size", "warn",
            "the study contains factorial designs but pre-specifies no "
            "interaction margin: the orthogonality and synergy claims will "
            "be descriptive and cannot be certified")
    elif interaction_value > 0.0 or interaction_equivalence_value > 0.0:
        add("interaction_effect_size", "pass",
            f"interaction margin={interaction_value}, interaction equivalence "
            f"margin={interaction_equivalence_value} on a "
            f"2**{n_factors} design")

    planned_interaction_power = None
    if interaction_value > 0.0 or interaction_equivalence_value > 0.0:
        if alpha_value is None or target_power_value is None or not n_tests_valid:
            add("interaction_power", "fail",
                "the interaction claims cannot be planned because alpha, the "
                "target power or the family size is invalid")
        elif effective_paired_sd is None:
            add("interaction_power", "warn",
                "no assumed paired SD supplied: the planned power for the "
                "interaction claims cannot be checked before the run")
        else:
            interaction_sd = _finite_float(effective_paired_sd)
            if interaction_sd is None or interaction_sd < 0.0:
                add("interaction_power", "fail",
                    f"invalid assumed paired SD {effective_paired_sd!r}")
            elif n_pairs < 2:
                add("interaction_power", "fail",
                    f"{n_pairs} distinct seed(s): at least 2 paired observations "
                    f"are needed to plan an interaction contrast")
            else:
                try:
                    planned_interaction_power = factorial_power_plan(
                        interaction_sd, n_pairs, n_factors=int(n_factors),
                        term_size=int(interaction_term_size),
                        margin=float(interaction_value),
                        alpha=float(alpha_value),
                        power=float(target_power_value),
                        n_tests=int(n_confirmatory_tests),
                        expected_attrition=float(expected_attrition),
                        equivalence_margin=float(interaction_equivalence_value))
                except (ValueError, TypeError) as exc:
                    add("interaction_power", "fail",
                        f"invalid interaction power-planning input: {exc}")
                else:
                    superiority = planned_interaction_power.get("superiority")
                    if superiority is not None:
                        if superiority["adequately_powered_after_attrition"] is True:
                            add("interaction_power", "pass",
                                f"{n_pairs} pairs reach the target for an "
                                f"interaction of {interaction_value:g} at "
                                f"contrast SD={planned_interaction_power['contrast_sd']:g} "
                                f"({superiority['required_planned_pairs']} planned "
                                f"pairs needed after "
                                f"{superiority['expected_attrition']:.0%} attrition)")
                        elif superiority["adequately_powered_after_attrition"] is False:
                            add("interaction_power", "fail",
                                f"{n_pairs} pairs are underpowered for an "
                                f"interaction of {interaction_value:g} at contrast "
                                f"SD={planned_interaction_power['contrast_sd']:g}: "
                                f"{superiority['required_pairs_family_adjusted']} "
                                f"analyzable pairs are required, so plan "
                                f"{superiority['required_planned_pairs']} pairs")
                        else:
                            add("interaction_power", "warn",
                                f"an interaction of {interaction_value:g} is not "
                                f"reachable within the search cap at contrast "
                                f"SD={planned_interaction_power['contrast_sd']:g}")
                    equivalence = planned_interaction_power.get("equivalence")
                    if equivalence is not None:
                        if equivalence["adequately_powered_after_attrition"] is True:
                            add("interaction_equivalence_power", "pass",
                                f"{n_pairs} pairs reach the target for declaring "
                                f"the interaction equivalent to zero within "
                                f"{interaction_equivalence_value:g}")
                        elif equivalence["adequately_powered_after_attrition"] is False:
                            add("interaction_equivalence_power", "fail",
                                f"{n_pairs} pairs cannot certify orthogonality "
                                f"within {interaction_equivalence_value:g}: "
                                f"{equivalence['required_pairs_family_adjusted']} "
                                f"analyzable pairs are required, so plan "
                                f"{equivalence['required_planned_pairs']} pairs")
                        else:
                            add("interaction_equivalence_power", "warn",
                                f"the orthogonality bound {interaction_equivalence_value:g} "
                                f"is not reachable within the search cap")

    attrition_value = _finite_float(expected_attrition)
    if attrition_value is None or not 0.0 <= attrition_value < 1.0:
        add("attrition", "fail",
            f"invalid expected attrition {expected_attrition!r}: must be a "
            f"probability in [0, 1)")
    else:
        expected_pairs = int(math.floor(
            n_pairs * (1.0 - attrition_value) ** 2 + 1e-12))
        if expected_pairs < min_seeds:
            add("attrition", "fail",
                f"{expected_pairs} complete pair(s) are expected to survive "
                f"{attrition_value:.0%} run attrition, below the replication "
                f"minimum of {min_seeds}; plan more seeds or lower the "
                f"attrition assumption")
        else:
            add("attrition", "pass",
                f"{expected_pairs} of {n_pairs} planned pair(s) expected to "
                f"survive {attrition_value:.0%} run attrition")

    if schedule is None:
        add("schedule", "warn",
            "no randomized run schedule: execution order may be confounded with "
            "the arm and with time drift")
    else:
        try:
            integrity = schedule_fingerprint_status(schedule)
        except ValueError as exc:
            add("schedule_integrity", "fail",
                f"malformed run schedule: {exc}")
        else:
            if integrity["verified"] is True:
                add("schedule_integrity", "pass",
                    "schedule fingerprint matches the recorded order")
            elif integrity["verified"] is False:
                add("schedule_integrity", "fail",
                    "schedule fingerprint does not match the recorded order; "
                    "the pre-registered allocation was edited after freezing")
            else:
                add("schedule_integrity", "warn",
                    f'schedule cannot be verified: {integrity["reason"]}')
        try:
            balance = schedule_balance_report(schedule, arm_of=schedule_arm_of)
        except ValueError as exc:
            add("schedule", "fail", f"malformed run schedule: {exc}")
        else:
            alignment = balance.get("replicate_block_aligned")
            if alignment is True:
                add("schedule_replicate_block_alignment", "pass",
                    "every replicate/seed stays inside one time block, so "
                    "seed-paired arm contrasts are within-block")
            elif alignment is False:
                split = {rep: blocks for rep, blocks in
                         (balance.get("replicate_blocks") or {}).items()
                         if len(blocks) != 1}
                add("schedule_replicate_block_alignment", "fail",
                    f"replicate/seed(s) {sorted(split)} are not bound to one "
                    f"time block ({split}); seed-paired arm contrasts then "
                    f"also contain a block/time difference, so regenerate the "
                    f"frozen schedule with one block per seed")
            raw_order = schedule.get("order")
            if not isinstance(raw_order, list):
                add("schedule", "fail",
                    "malformed run schedule: order is not a list of run ids")
            elif expected_units is not None:
                expected = {str(unit) for unit in expected_units}
                realized = {str(unit) for unit in raw_order}
                if expected != realized:
                    add("schedule", "fail",
                        f"schedule does not cover the planned runs: missing "
                        f"{sorted(expected - realized)}, extra {sorted(realized - expected)}")
                elif balance["warnings"]:
                    add("schedule", "warn", "; ".join(balance["warnings"]))
                else:
                    add("schedule", "pass",
                        f'{schedule.get("n_units")} run(s) in '
                        f'{schedule.get("n_blocks")} block(s), fingerprint '
                        f'{str(schedule.get("fingerprint"))[:12]}')
            elif balance["warnings"]:
                add("schedule", "warn", "; ".join(balance["warnings"]))
            else:
                add("schedule", "pass",
                    f'{schedule.get("n_units")} run(s) in '
                    f'{schedule.get("n_blocks")} block(s)')

    pairs = []
    malformed_controls = []
    if identity_controls is None:
        controls = ()
    else:
        try:
            controls = list(identity_controls)
        except TypeError:
            controls = None
            malformed_controls.append(identity_controls)
    if controls is not None:
        for pair in controls:
            if not pair:
                continue
            # A string/dict/scalar is not a pair of row names. `tuple("ab")`
            # silently becomes ('a', 'b') and could certify a control the
            # caller never specified; an int used to crash the whole audit.
            if not isinstance(pair, (list, tuple)):
                malformed_controls.append(pair)
                continue
            pairs.append(tuple(pair))
    if malformed_controls:
        add("controls", "fail",
            f"control pair(s) must be two-element lists/tuples of row names: "
            f"{malformed_controls!r}")
    elif not pairs:
        if equivalence_margin > 0.0:
            add("controls", "fail",
                "no identity/negative-control pair declared: equivalence testing requires a known-null contrast to check calibration")
        else:
            add("controls", "warn",
                "no identity/negative-control pair declared: the analysis has no "
                "known-null contrast to check its own calibration")
    else:
        missing_pairs = [pair for pair in pairs
                         if len(pair) != 2 or any(str(name) not in names for name in pair)]
        if missing_pairs:
            add("controls", "fail",
                f"control pair(s) reference absent rows: {missing_pairs}")
        else:
            add("controls", "pass",
                f"{len(pairs)} identity/negative-control pair(s) present: "
                f"{[list(pair) for pair in pairs]}")

    canonical_policy = None
    policy_fingerprint = None
    try:
        canonical_policy = normalize_exclusion_policy(exclusion_policy)
        policy_fingerprint = exclusion_policy_fingerprint(canonical_policy)
    except ValueError as exc:
        add("exclusions", "fail", f"invalid exclusion policy: {exc}")
    else:
        rules = {rule["id"]: rule for rule in canonical_policy["rules"]}
        add("exclusions", "pass",
            f"pre-specified attrition policy with {len(rules)} rule(s), "
            f"fingerprint {policy_fingerprint[:12]}")

    n_fail = sum(1 for check in checks if check["status"] == "fail")
    n_warn = sum(1 for check in checks if check["status"] == "warn")
    return {
        "schema": "adasquash.design_readiness.v1",
        "ready": n_fail == 0,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "checks": checks,
        "planned_power": planned_power,
        "planned_equivalence_power": planned_equivalence_power,
        "planned_interaction_power": planned_interaction_power,
        "n_confirmatory_tests": n_confirmatory_tests,
        "expected_attrition": expected_attrition,
        "assumed_paired_sd": effective_paired_sd,
        "paired_sd_evidence_audit": paired_sd_evidence_report,
        "exclusion_policy": canonical_policy,
        "exclusion_policy_fingerprint": policy_fingerprint,
        "note": ("failures block a confirmatory launch; warnings are recorded so a "
                 "reader can see which design decisions were left open"),
    }


def design_analysis_alignment_report(
        design_readiness: dict,
        comparison: dict,
        *,
        min_seeds: int = MIN_SEEDS_FOR_SIGNIFICANCE) -> dict:
    """Check whether the actual analysis aligns with the pre-registered design.

    Compares the design_readiness audit (pre-flight checklist) with the actual
    comparison results (arm_comparison.json) to surface deviations that affect
    interpretability and certifiability.

    Checks include:
     - Schedule fingerprint: was the frozen schedule respected (order drift
       tested, block alignment verified)?
     - Exclusion policy: does the per-arm attrition match the pre-specified
       policy, or were runs excluded post-hoc?
     - Seed coverage: do the shared seeds used in the comparison match the
       planned seed set?
     - Metric consistency: is the compared metric the one planned (final_val_loss
       is the only certifiable endpoint)?
     - Reference selection: was the reference pre-specified or data-selected?

    Returns a dict with ``"aligned"`` (bool), ``"deviations`` (list of strings)
    and ``"warnings`` (list of strings).
    """
    # execution_identity_mismatches import removed: unused

    deviations: list[str] = []
    warnings: list[str] = []

    # Extract design readiness components
    # dr_checks removed: unused variable (extracted but never consumed)
    dr_exclusion_fp = design_readiness.get("exclusion_policy_fingerprint")
    dr_schedule = design_readiness.get("schedule_integrity")
    dr_balance = design_readiness.get("schedule_replicate_block_alignment")
    # dr_min_seeds removed: unused variable (extracted but never consumed)

    # Extract comparison components
    cmp_rows = comparison.get("rows", [])
    # cmp_protocol removed: unused variable (extracted but never consumed)
    cmp_reference = comparison.get("reference")
    cmp_reference_selection = comparison.get("reference_selection", {})
    cmp_metric = comparison.get("metric", "final_val_loss")
    cmp_certifiable = comparison.get("certifiable", False)
    cmp_n_fail = design_readiness.get("n_fail", 0)

    # 1. Schedule integrity
    if dr_schedule is not None:
        sp_status = dr_schedule.get("status")
        if sp_status == "fail":
            deviations.append(
                "schedule fingerprint FAIL: the frozen allocation was edited after "
                "pre-registration; analysis against the planned design is invalid")
        elif sp_status == "warn":
            warnings.append(
                f"schedule fingerprint WARNING: {dr_schedule.get('reason', 'unknown')}")
        else:
            warnings.append("schedule fingerprint pass: frozen schedule respected")

    # 2. Block replicate alignment
    if dr_balance is not None:
        ba_status = dr_balance.get("status")
        if ba_status == "fail":
            deviations.append(
                "block replicate alignment FAIL: replicate(s)/seed(s) spread across "
                "time blocks; seed-paired contrasts contain block/time confounding")
        elif ba_status == "pass":
            warnings.append(
                "block replicate alignment pass: every seed confined to one block")
        else:
            warnings.append(
                "block replicate alignment warn: check replicate-block mapping")

    # 3. Exclusion policy deviation
    if dr_exclusion_fp is not None:
        # Check if the comparison reports any post-hoc exclusions
        for row in cmp_rows:
            proto_problems = row.get("protocol_problems", [])
            if proto_problems:
                deviations.append(
                    f"protocol problems detected in comparison: {'; '.join(proto_problems)}")

    # 4. Seed coverage vs planned minimum
    shared_seed_counts = {row.get("name"): row.get("n_shared_seeds")
                         for row in cmp_rows if row.get("n_shared_seeds") is not None}
    for name, n_shared in shared_seed_counts.items():
        if n_shared is not None and n_shared < min_seeds:
            deviations.append(
                f"arm {name!r}: only {n_shared} shared seed(s) < {min_seeds} "
                f"planned minimum; result is descriptive only")

    # 5. Reference selection alignment
    if cmp_reference_selection.get("data_dependent", False):
        pool = cmp_reference_selection.get("pool", [])
        if cmp_reference not in pool:
            deviations.append(
                f"reference {cmp_reference!r} not in declared pool {pool}; "
                "data-selected reference may not match pre-registration")
        pool_size = cmp_reference_selection.get("pool_size", 0)
        if pool_size > 1 and not cmp_reference_selection.get("selection_adjustment_complete", False):
            deviations.append(
                f"data-selected reference from pool of {pool_size} but "
                "selection adjustment is incomplete; certification is conditional")
        if pool_size == 1:
            warnings.append(
                "single-arm reference pool: data selection had no alternative; "
                "no selection adjustment needed")
    elif not cmp_reference_selection.get("data_dependent", True):
        # Explicit reference was pre-specified
        warnings.append(
            "reference pre-specified: no data-dependent selection, no adjustment needed")

    # 6. Metric certifiability alignment
    if cmp_metric != "final_val_loss":
        warnings.append(
            f"comparing metric '{cmp_metric}': only 'final_val_loss' is certifiable "
            f"from aggregate.json per the pre-registered endpoint policy")

    # 7. Overall certifiability vs design readiness
    if not cmp_certifiable and cmp_n_fail > 0:
        deviations.append(
            "comparison is not certifiable (design readiness had failures) but "
            "results are being interpreted as confirmatory")

    # 8. Per-row protocol problems
    for row in cmp_rows:
        proto_probs = row.get("protocol_problems", [])
        if proto_probs:
            deviations.append(
                f"row {row.get('name')}: protocol problems - {'; '.join(proto_probs)}")

    aligned = len(deviations) == 0

    return {
        "schema": "adasquash.design_analysis_alignment.v1",
        "aligned": aligned,
        "n_deviations": len(deviations),
        "n_warnings": len(warnings),
        "deviations": deviations,
        "warnings": warnings,
        "reading": (
            "no deviations found: the analysis is consistent with the "
            "pre-registered design" if aligned
            else "deviations from pre-registered design found; see deviations list"
        ),
    }


def _normal_sampler(rng: random.Random, mean: float, sd: float):
    if sd == 0.0:
        return lambda: mean
    return lambda: rng.gauss(mean, sd)


def simulate_paired_design(n_pairs: int = 3, effect: float = 0.0,
                           paired_sd: float = 0.05, reps: int = 2000,
                           alpha: float = 0.05, equivalence_margin: float = 0.0,
                           seed: int = 0) -> dict:
    """Monte-Carlo calibration of the paired-test pipeline.

    Draws synthetic paired differences from ``N(effect, paired_sd)`` and runs
    the *actual* ``paired_t_one_sided`` code used by the reports:

    * ``effect=0`` estimates the empirical type-I error of the one-sided
      superiority claim (the pipeline should be at or below ``alpha``);
    * ``effect<0`` estimates power for that effect size (the candidate has the
      lower loss, matching the reports' sign convention);
    * a positive ``equivalence_margin`` additionally reports how often the
      actual TOST code declares equivalence.

    Monte-Carlo error is reported so a "0.049" is not over-read: with
    ``reps=2000`` the standard error of a 0.05 rate is about 0.005.
    """
    if (isinstance(n_pairs, bool)
            or not isinstance(n_pairs, numbers.Integral) or n_pairs < 2):
        raise ValueError(f"n_pairs must be an integer >= 2, got {n_pairs!r}")
    n_pairs = int(n_pairs)
    if (isinstance(reps, bool)
            or not isinstance(reps, numbers.Integral) or reps < 100):
        raise ValueError(f"reps must be an integer >= 100, got {reps!r}")
    reps = int(reps)
    effect_value = _finite_float(effect)
    paired_sd_value = _finite_float(paired_sd)
    if effect_value is None:
        raise ValueError(f"effect must be a finite number, got {effect!r}")
    if paired_sd_value is None:
        raise ValueError(f"paired_sd must be a finite number, got {paired_sd!r}")
    if paired_sd_value < 0.0:
        raise ValueError(f"paired_sd must be >= 0, got {paired_sd!r}")
    alpha_value = _finite_float(alpha)
    if alpha_value is None or not 0.0 < alpha_value < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    equivalence_value = _finite_float(equivalence_margin)
    if equivalence_value is None or equivalence_value < 0.0:
        raise ValueError(f"equivalence_margin must be a finite number >= 0, "
                         f"got {equivalence_margin!r}")
    if equivalence_value > 0.0 and alpha_value >= 0.5:
        raise ValueError(f"a positive equivalence_margin requires alpha < 0.5 so "
                         f"the TOST confidence interval 1 - 2*alpha is positive, "
                         f"got alpha={alpha!r}")
    seed = _validate_seed(seed)

    rng = random.Random(seed)
    # Above this scale the direct draws and their squares can overflow even
    # when the reported mean and standard error are representable. The paired
    # tests are invariant to a positive common scale, so draw in normalized
    # units and map only the descriptive estimates back afterwards.
    simulation_scale = max(abs(effect_value), paired_sd_value)
    scaled_run = simulation_scale > 1e150
    scale = simulation_scale if scaled_run else 1.0
    draw = _normal_sampler(rng, effect_value / scale, paired_sd_value / scale)
    test_margin = equivalence_value / scale
    detection = 0
    equivalence = 0
    undefined = 0
    means: list[float] = []
    for _ in range(reps):
        deltas = [draw() for _ in range(n_pairs)]
        means.append(sum(deltas) / n_pairs)
        test = paired_t_one_sided(deltas, [0.0] * n_pairs, margin=0.0,
                                  alpha=alpha_value)
        p_value = test.get("p_one_sided")
        if p_value is None or not math.isfinite(float(p_value)):
            undefined += 1
        elif float(p_value) < alpha_value:
            detection += 1
        if equivalence_value > 0.0:
            equiv = tost_paired(deltas, [0.0] * n_pairs,
                                margin=test_margin, alpha=alpha_value)
            if equiv.get("equivalent") is True:
                equivalence += 1
    mean_scaled = (sum(means) / max(len(means), 1)) if means else float("nan")
    mean_estimate = mean_scaled * scale
    if len(means) > 1:
        variance_scaled = (
            sum((value - mean_scaled) ** 2 for value in means)
            / (len(means) - 1))
        if math.isfinite(variance_scaled) and variance_scaled > 0.0:
            estimate_sd = math.sqrt(variance_scaled) * scale
        else:
            # A zero variance can be a genuine constant sample or the
            # underflow of tiny squared deviations. Recompute around the
            # largest spread in normalized units; the constant case still
            # falls through with spread == 0 and reports a zero SD.
            spread = max(abs(value - mean_scaled) for value in means)
            if spread == 0.0:
                estimate_sd = 0.0
            else:
                variance_scaled = (
                    sum(((value - mean_scaled) / spread) ** 2 for value in means)
                    / (len(means) - 1))
                estimate_sd = math.sqrt(variance_scaled) * spread * scale
    else:
        estimate_sd = 0.0
    mean_bias = ((mean_scaled - effect_value / scale) * scale
                 if scaled_run else mean_estimate - effect_value)
    detection_rate = detection / reps
    return {
        "schema": "adasquash.design_simulation.v1",
        "n_pairs": n_pairs,
        "effect": float(effect),
        "paired_sd": float(paired_sd),
        "reps": reps,
        "alpha": float(alpha),
        "equivalence_margin": (float(equivalence_margin)
                               if float(equivalence_margin) > 0.0 else None),
        "empirical_detection_rate": detection_rate,
        "empirical_type_i_error": (detection_rate if float(effect) == 0.0 else None),
        "empirical_power": (detection_rate if float(effect) < 0.0 else None),
        "empirical_equivalence_rate": (equivalence / reps
                                       if float(equivalence_margin) > 0.0 else None),
        "wrong_direction_rate": (sum(1 for value in means if value > 0.0) / reps
                                 if float(effect) <= 0.0 else None),
        "mean_estimate": mean_estimate,
        "mean_bias": mean_bias,
        "estimate_sd": estimate_sd,
        "monte_carlo_se_detection_rate": math.sqrt(
            max(0.0, detection_rate * (1.0 - detection_rate)) / reps),
        "undefined_p_rate": undefined / reps,
        "note": ("a well-calibrated superiority pipeline has "
                 "empirical_type_i_error <= alpha (within Monte-Carlo error); "
                 "power below the target means the planned seed count cannot "
                 "detect the stated effect"),
    }
