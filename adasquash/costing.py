"""Prospective compute-cost planning for a training study.

A confirmatory study is a *schedule of runs*, and every run costs wall-clock
time on hardware that is usually shared, quota-bound or rented. Two design
questions therefore have to be answered before the first run starts, not
after the results are in:

* does the planned design fit the compute budget at all? A study that is
  stopped half-way is worse than a smaller study that was finished, because
  the pre-registered schedule, the seed set and the exclusion policy all
  assume that every planned run happened;
* if it does not fit, what should be traded away - breadth (fewer arms) or
  replication (fewer seeds per arm)? Those are different experiments. Dropping
  an arm removes a hypothesis; dropping a seed removes the resolution that
  makes the remaining hypothesis testable at all. The trade-off has to be
  made explicitly, with the power of the primary claim attached to it.

This module answers both questions without running anything:

``estimate_run_cost``
    The wall-clock cost of one run from its token count and the measured
    throughput of the target device. The throughput is an *assumption* and is
    carried through the report, never hidden inside a number.
``study_budget_plan``
    The study-level plan: the cost of every arm, whether the planned design
    fits the budget, the largest uniform replication that does fit, and a
    ranked set of allocations (as planned, minimum replication, primary arms
    only) with the prospective power of the primary claim attached to each
    one. The recommended allocation is the first candidate that fits; when
    nothing fits, the report says so instead of silently shrinking the
    design.

The cost model is deliberately simple - tokens divided by a measured
throughput - because an elaborate but unverifiable FLOPs model would be a
worse planning tool than an explicit, checkable assumption.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional

from .design import prospective_power_plan
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

__all__ = [
    "STUDY_BUDGET_SCHEMA",
    "estimate_run_cost",
    "plan_runs_from_matrix",
    "study_budget_plan",
]

STUDY_BUDGET_SCHEMA = "adasquash.study_budget.v1"


def _finite_float(value: Any) -> Optional[float]:
    """``float(value)`` when it is a finite non-boolean real, else None."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _positive_int(value: Any, name: str) -> int:
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or int(value) < 1):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def estimate_run_cost(tokens: int, *, tokens_per_second: Optional[float] = None,
                      overhead_seconds: float = 0.0) -> dict:
    """Wall-clock cost of a single run.

    Args:
        tokens: Tokens the run processes (the measured ``tokens_processed``
            of a finished run, or ``steps * batch_size * seq_len`` for a
            plan).
        tokens_per_second: Measured throughput of the target device. Without
            it the cost in hours is unknown and only the token count is
            reported; a missing throughput must never become an invented
            duration.
        overhead_seconds: Fixed per-run overhead (process start, data
            generation, evaluation), added on top of the training time.

    Returns:
        A mapping with the token count, the throughput assumption, the
        estimated seconds and hours, and the basis of the estimate.
    """
    token_count = _positive_int(tokens, "tokens")
    throughput = _finite_float(tokens_per_second)
    if throughput is not None and throughput <= 0.0:
        raise ValueError(
            f"tokens_per_second must be a positive number, got "
            f"{tokens_per_second!r}")
    overhead = _finite_float(overhead_seconds)
    if overhead is None or overhead < 0.0:
        raise ValueError(
            f"overhead_seconds must be a finite number >= 0, got "
            f"{overhead_seconds!r}")
    if throughput is None:
        seconds = None
        hours = None
        basis = "tokens only: no measured throughput was supplied"
    else:
        seconds = token_count / throughput + overhead
        hours = seconds / 3600.0
        basis = ("tokens / measured tokens_per_second + fixed overhead; the "
                 "throughput is an explicit assumption")
    return {
        "tokens": token_count,
        "tokens_per_second": throughput,
        "overhead_seconds": overhead,
        "seconds": seconds,
        "hours": hours,
        "basis": basis,
    }


def plan_runs_from_matrix(arms: Iterable[str], n_seeds: int,
                           tokens_per_run: int, *,
                           primary_arms: Sequence[str] = (),
                           tokens_per_second: Optional[float] = None) -> list[dict]:
    """Expand a matrix into the run list a budget plan consumes.

    Args:
        arms: Row names of the design matrix.
        n_seeds: Seeds each arm is replicated over.
        tokens_per_run: Tokens a single run of an arm processes.
        primary_arms: Rows that carry the primary claim. They are the last
            to be dropped when the budget is short.
        tokens_per_second: Measured throughput of the target device.

    Returns:
        One mapping per arm with ``arm``, ``n_seeds``, ``tokens``,
        ``primary`` and ``tokens_per_second``.
    """
    seed_count = _positive_int(n_seeds, "n_seeds")
    token_count = _positive_int(tokens_per_run, "tokens_per_run")
    primary = {str(name) for name in (primary_arms or ())}
    runs: list[dict] = []
    for arm in arms:
        name = str(arm)
        if not name.strip():
            raise ValueError(f"arm names must be non-empty strings, got {arm!r}")
        runs.append({
            "arm": name,
            "n_seeds": seed_count,
            "tokens": token_count,
            "primary": name in primary,
            "tokens_per_second": tokens_per_second,
        })
    return runs


def _canonical_runs(runs: Iterable[Mapping]) -> list[dict]:
    canonical: list[dict] = []
    seen: set[str] = set()
    for entry in runs:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"every run must be a mapping, got {type(entry).__name__}")
        arm = str(entry.get("arm") or "").strip()
        if not arm:
            raise ValueError(f"every run needs a non-empty arm name, got {entry!r}")
        if arm in seen:
            raise ValueError(f"duplicate arm in the run list: {arm!r}")
        seen.add(arm)
        n_seeds = _positive_int(entry.get("n_seeds"), f"n_seeds of {arm!r}")
        tokens = _positive_int(entry.get("tokens"), f"tokens of {arm!r}")
        throughput = _finite_float(entry.get("tokens_per_second"))
        if throughput is not None and throughput <= 0.0:
            raise ValueError(
                f"tokens_per_second of {arm!r} must be positive, got "
                f"{entry.get('tokens_per_second')!r}")
        canonical.append({
            "arm": arm,
            "n_seeds": n_seeds,
            "tokens": tokens,
            "primary": bool(entry.get("primary")),
            "tokens_per_second": throughput,
        })
    if not canonical:
        raise ValueError("the run list is empty: there is nothing to plan")
    return canonical


def _allocation_cost(arms: Sequence[dict], seeds_by_arm: Mapping[str, int],
                     overhead_seconds: float) -> dict:
    total_tokens = 0
    total_hours = 0.0
    hours_known = True
    per_arm: list[dict] = []
    for arm in arms:
        seeds = int(seeds_by_arm[arm["arm"]])
        cost = estimate_run_cost(
            arm["tokens"] * seeds,
            tokens_per_second=arm["tokens_per_second"],
            overhead_seconds=overhead_seconds * seeds)
        total_tokens += arm["tokens"] * seeds
        if cost["hours"] is None:
            hours_known = False
        else:
            total_hours += float(cost["hours"])
        per_arm.append({
            "arm": arm["arm"],
            "n_seeds": seeds,
            "tokens_per_run": arm["tokens"],
            "tokens": arm["tokens"] * seeds,
            "primary": arm["primary"],
            "hours": cost["hours"],
        })
    return {
        "per_arm": per_arm,
        "total_tokens": total_tokens,
        "total_hours": total_hours if hours_known else None,
        "hours_known": hours_known,
    }


def study_budget_plan(runs: Iterable[Mapping], *, budget_hours: Optional[float] = None,
                      min_seeds: int = 3, overhead_seconds: float = 0.0,
                      paired_sd: Optional[float] = None, margin: Optional[float] = None,
                      alpha: float = 0.05, power: float = 0.8, n_tests: int = 1,
                      expected_attrition: float = 0.0) -> dict:
    """Plan a study against a compute budget and report the trade-offs.

    Args:
        runs: The planned runs, one mapping per arm with ``arm``,
            ``n_seeds``, ``tokens`` (per run) and optionally ``primary`` and
            ``tokens_per_second``.
        budget_hours: The wall-clock budget available for the study. Without
            it the report is a cost estimate only.
        min_seeds: The replication floor below which an arm may not be run.
        overhead_seconds: Fixed per-run overhead (start-up, evaluation).
        paired_sd: The planning SD of the primary contrast, used to attach
            the prospective power to every candidate allocation.
        margin: The pre-specified smallest effect of interest for the
            primary contrast.
        alpha: Family-wise alpha of the primary claim.
        power: Target power of the primary claim.
        n_tests: Number of pre-registered confirmatory tests, used for the
            conservative Bonferroni family level.
        expected_attrition: Expected fraction of runs lost to pre-specified
            protocol failures.

    Returns:
        A mapping with the per-arm cost, the totals, the largest uniform
        replication that fits the budget, a ranked list of candidate
        allocations and the recommended one.
    """
    canonical = _canonical_runs(runs)
    budget = _finite_float(budget_hours)
    if budget is not None and budget <= 0.0:
        raise ValueError(
            f"budget_hours must be a positive number, got {budget_hours!r}")
    floor = _positive_int(min_seeds, "min_seeds")
    sd = _finite_float(paired_sd)
    if sd is not None and sd < 0.0:
        raise ValueError(f"paired_sd must be a finite number >= 0, got {paired_sd!r}")
    margin_value = _finite_float(margin)
    if margin_value is not None and margin_value < 0.0:
        raise ValueError(f"margin must be a finite number >= 0, got {margin!r}")
    if (isinstance(n_tests, bool) or not isinstance(n_tests, numbers.Integral)
            or int(n_tests) < 1):
        raise ValueError(f"n_tests must be a positive integer, got {n_tests!r}")

    planned_seeds = {arm["arm"]: arm["n_seeds"] for arm in canonical}
    primary_arms = [arm["arm"] for arm in canonical if arm["primary"]]
    if not primary_arms:
        primary_arms = [arm["arm"] for arm in canonical]

    def _power_for(seeds: int) -> Optional[dict]:
        if sd is None or margin_value is None or margin_value <= 0.0:
            return None
        if seeds < 2:
            return None
        try:
            plan = prospective_power_plan(
                sd, seeds, margin_value, alpha=alpha, power=power,
                n_tests=int(n_tests), expected_attrition=expected_attrition)
        except ValueError:
            return None
        return {
            "required_planned_pairs": plan["required_planned_pairs"],
            "expected_analyzable_pairs": plan["expected_analyzable_pairs"],
            "adequately_powered_after_attrition":
                plan["adequately_powered_after_attrition"],
            "minimum_detectable_delta_after_attrition":
                plan["minimum_detectable_delta_after_attrition"],
        }

    def _candidate(name: str, arms: Sequence[dict],
                   seeds_by_arm: Mapping[str, int], rationale: str) -> dict:
        cost = _allocation_cost(arms, seeds_by_arm, overhead_seconds)
        hours = cost["total_hours"]
        fits = None
        if budget is not None and hours is not None:
            fits = bool(hours <= budget + 1e-9)
        seed_counts = sorted({int(value) for value in seeds_by_arm.values()})
        primary_seed_count = min(
            (int(seeds_by_arm[arm]) for arm in primary_arms
             if arm in seeds_by_arm), default=0)
        return {
            "name": name,
            "arms": [arm["arm"] for arm in arms],
            "seeds": {arm["arm"]: int(seeds_by_arm[arm["arm"]]) for arm in arms},
            "n_seeds": seed_counts[0] if len(seed_counts) == 1 else None,
            "seed_counts": seed_counts,
            "primary_n_seeds": primary_seed_count,
            "per_arm": cost["per_arm"],
            "total_tokens": cost["total_tokens"],
            "total_hours": hours,
            "fits_budget": fits,
            "primary_power": _power_for(primary_seed_count),
            "rationale": rationale,
        }

    candidates: list[dict] = []
    candidates.append(_candidate(
        "as_planned", canonical, planned_seeds,
        "the design exactly as pre-registered"))
    floor_seeds = {name: min(count, floor) for name, count in planned_seeds.items()}
    candidates.append(_candidate(
        "minimum_replication", canonical, floor_seeds,
        f"every arm kept, replication reduced to the floor of {floor} seed(s)"))
    primary_runs = [arm for arm in canonical if arm["arm"] in primary_arms]
    candidates.append(_candidate(
        "primary_only_as_planned", primary_runs,
        {name: planned_seeds[name] for name in primary_arms},
        "exploratory arms dropped, the primary claim keeps its replication"))
    candidates.append(_candidate(
        "primary_only_minimum", primary_runs,
        {name: min(planned_seeds[name], floor) for name in primary_arms},
        f"exploratory arms dropped and the primary claim reduced to the floor "
        f"of {floor} seed(s)"))

    max_uniform = None
    if budget is not None and canonical[0]["tokens_per_second"] is not None:
        planned_max = max(planned_seeds.values())
        for candidate_seeds in range(planned_max, floor - 1, -1):
            seeds_by_arm = {name: min(count, candidate_seeds)
                            for name, count in planned_seeds.items()}
            cost = _allocation_cost(canonical, seeds_by_arm, overhead_seconds)
            if cost["total_hours"] is not None and cost["total_hours"] <= budget + 1e-9:
                max_uniform = candidate_seeds
                break

    recommendation = None
    status = "unconstrained"
    for candidate in candidates:
        if candidate["fits_budget"] is True:
            recommendation = candidate
            status = "fits_budget"
            break
    if recommendation is None:
        status = "infeasible"
        recommendation = candidates[-1]
        if budget is None:
            status = "unconstrained"

    planned_cost = candidates[0]
    checks: list[dict] = []

    def add(check_id: str, check_status: str, detail: str) -> None:
        checks.append({"id": check_id, "status": check_status, "detail": detail})

    if budget is None:
        add("budget", "warn",
            "no compute budget was supplied: the cost is reported, but "
            "whether the study fits is unknown")
    elif planned_cost["total_hours"] is None:
        add("budget", "warn",
            "the run cost in hours is unknown (no measured throughput): the "
            "token count is reported, but the study cannot be checked against "
            "the budget")
    elif planned_cost["fits_budget"]:
        add("budget", "pass",
            f"the planned design needs {planned_cost['total_hours']:.4g} h of "
            f"the {budget:.4g} h budget")
    else:
        add("budget", "fail",
            f"the planned design needs {planned_cost['total_hours']:.4g} h, "
            f"which exceeds the {budget:.4g} h budget; the recommended "
            f"allocation is '{recommendation['name']}'")
    if any(int(seeds_by_arm) < floor for seeds_by_arm in planned_seeds.values()):
        add("replication", "fail",
            f"at least one arm is planned with fewer than {floor} seed(s)")
    else:
        add("replication", "pass",
            f"every arm is planned with at least {floor} seed(s)")
    if recommendation["fits_budget"] is False:
        add("feasibility", "fail",
            f"even the smallest candidate allocation ('{recommendation['name']}') "
            f"does not fit the budget")
    elif recommendation["fits_budget"] is True:
        add("feasibility", "pass",
            f"the recommended allocation '{recommendation['name']}' fits the "
            f"budget")
    else:
        add("feasibility", "warn",
            "the feasibility of the allocations is unknown without a measured "
            "throughput")

    return {
        "schema": STUDY_BUDGET_SCHEMA,
        "n_arms": len(canonical),
        "budget_hours": budget,
        "min_seeds": floor,
        "planned": planned_cost,
        "per_arm": planned_cost["per_arm"],
        "total_tokens": planned_cost["total_tokens"],
        "total_hours": planned_cost["total_hours"],
        "fits_budget": planned_cost["fits_budget"],
        "max_uniform_seeds_within_budget": max_uniform,
        "candidates": candidates,
        "recommended": recommendation,
        "status": status,
        "checks": checks,
        "note": ("the recommended allocation is the first candidate that fits "
                 "the budget, in the order: as planned, minimum replication, "
                 "primary arms only; dropping an arm removes a hypothesis and "
                 "dropping a seed removes the resolution of the remaining one"),
    }
