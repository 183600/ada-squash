"""Compute-budget planning for a training study.

A study that is stopped half-way is worse than a smaller study that was
finished: the pre-registered schedule, the seed set and the exclusion policy
all assume that every planned run happened. The budget plan therefore has to
answer two questions before the first run starts - does the design fit, and
if not, what is traded away - and it has to attach the prospective power of
the primary claim to every candidate allocation, because "drop an arm" and
"drop a seed" are different experiments with different scientific content.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.costing import (
    estimate_run_cost,
    plan_runs_from_matrix,
    study_budget_plan,
)
from scripts.ablation import build_analysis_plan


# ---------------------------------------------------------------------------
# The cost of one run
# ---------------------------------------------------------------------------
def test_run_cost_is_tokens_divided_by_the_measured_throughput():
    cost = estimate_run_cost(3_600_000, tokens_per_second=1000.0)
    assert cost["tokens"] == 3_600_000
    assert cost["seconds"] == pytest.approx(3600.0)
    assert cost["hours"] == pytest.approx(1.0)
    assert cost["tokens_per_second"] == pytest.approx(1000.0)


def test_run_cost_adds_the_fixed_overhead_per_run():
    cost = estimate_run_cost(1000, tokens_per_second=100.0,
                             overhead_seconds=30.0)
    assert cost["seconds"] == pytest.approx(40.0)


def test_run_cost_without_a_throughput_reports_tokens_only():
    cost = estimate_run_cost(1000)
    assert cost["hours"] is None
    assert cost["seconds"] is None
    assert cost["tokens"] == 1000
    assert "no measured throughput" in cost["basis"]


def test_invalid_cost_inputs_are_rejected():
    with pytest.raises(ValueError):
        estimate_run_cost(0)
    with pytest.raises(ValueError):
        estimate_run_cost(1000, tokens_per_second=0.0)
    with pytest.raises(ValueError):
        estimate_run_cost(1000, overhead_seconds=-1.0)


# ---------------------------------------------------------------------------
# The study-level plan
# ---------------------------------------------------------------------------
def _runs(n_seeds=5, tokens=10_000_000, throughput=20_000.0,
          primary=("a", "b")):
    return plan_runs_from_matrix(
        ["a", "b", "c", "d"], n_seeds, tokens,
        primary_arms=primary, tokens_per_second=throughput)


def test_a_plan_that_fits_is_recommended_unchanged():
    plan = study_budget_plan(_runs(), budget_hours=10.0, min_seeds=3)
    assert plan["fits_budget"] is True
    assert plan["status"] == "fits_budget"
    assert plan["recommended"]["name"] == "as_planned"
    assert plan["recommended"]["seeds"]["a"] == 5
    assert all(check["status"] == "pass" for check in plan["checks"])


def test_a_plan_that_does_not_fit_falls_back_to_the_next_allocation():
    # 4 arms x 5 seeds x 10M tokens at 20k tokens/s is 2.78 h in total;
    # every arm at the replication floor still needs 1.67 h.
    plan = study_budget_plan(_runs(), budget_hours=2.0, min_seeds=3)
    assert plan["fits_budget"] is False
    assert plan["recommended"]["name"] == "minimum_replication"
    assert plan["recommended"]["seeds"]["a"] == 3
    assert plan["recommended"]["fits_budget"] is True
    budget_check = next(check for check in plan["checks"]
                        if check["id"] == "budget")
    assert budget_check["status"] == "fail"


def test_the_primary_arms_are_the_last_to_be_dropped():
    plan = study_budget_plan(_runs(), budget_hours=1.0, min_seeds=3)
    assert plan["recommended"]["name"] in (
        "primary_only_as_planned", "primary_only_minimum")
    assert set(plan["recommended"]["arms"]) <= {"a", "b"}


def test_an_infeasible_budget_is_reported_instead_of_shrinking_the_design():
    plan = study_budget_plan(_runs(), budget_hours=0.01, min_seeds=3)
    assert plan["status"] == "infeasible"
    assert plan["recommended"]["fits_budget"] is False
    feasibility = next(check for check in plan["checks"]
                       if check["id"] == "feasibility")
    assert feasibility["status"] == "fail"


def test_every_candidate_carries_the_power_of_the_primary_claim():
    plan = study_budget_plan(_runs(), budget_hours=1.0, min_seeds=3,
                              paired_sd=0.1, margin=0.05, alpha=0.05,
                              power=0.8, n_tests=4)
    for candidate in plan["candidates"]:
        power = candidate["primary_power"]
        assert power is not None
        assert power["required_planned_pairs"] >= 1
    # Fewer seeds means less resolution, so the recommended allocation can
    # never claim a smaller required n than the full plan.
    planned = plan["candidates"][0]["primary_power"]["required_planned_pairs"]
    recommended = plan["recommended"]["primary_power"]["required_planned_pairs"]
    assert recommended >= planned


def test_the_largest_uniform_replication_within_the_budget_is_reported():
    plan = study_budget_plan(_runs(n_seeds=8), budget_hours=2.0, min_seeds=3)
    assert plan["max_uniform_seeds_within_budget"] is not None
    assert plan["max_uniform_seeds_within_budget"] < 8
    assert plan["max_uniform_seeds_within_budget"] >= 3


def test_a_budget_without_a_throughput_cannot_be_checked():
    runs = plan_runs_from_matrix(["a", "b"], 3, 1000)
    plan = study_budget_plan(runs, budget_hours=1.0)
    assert plan["total_hours"] is None
    assert plan["fits_budget"] is None
    budget_check = next(check for check in plan["checks"]
                        if check["id"] == "budget")
    assert budget_check["status"] == "warn"


def test_no_budget_at_all_reports_the_cost_only():
    plan = study_budget_plan(_runs())
    assert plan["budget_hours"] is None
    assert plan["status"] == "unconstrained"
    assert plan["total_hours"] is not None
    assert plan["total_tokens"] == 4 * 5 * 10_000_000


def test_invalid_run_lists_are_rejected():
    with pytest.raises(ValueError):
        study_budget_plan([])
    with pytest.raises(ValueError):
        study_budget_plan([{"arm": "a", "n_seeds": 0, "tokens": 10}])
    with pytest.raises(ValueError):
        study_budget_plan([{"arm": "a", "n_seeds": 1, "tokens": 10},
                           {"arm": "a", "n_seeds": 1, "tokens": 10}])
    with pytest.raises(ValueError):
        study_budget_plan([{"arm": "  ", "n_seeds": 1, "tokens": 10}])
    with pytest.raises(ValueError):
        study_budget_plan(_runs(), budget_hours=0.0)
    with pytest.raises(ValueError):
        study_budget_plan(_runs(), min_seeds=0)


# ---------------------------------------------------------------------------
# The pre-registration
# ---------------------------------------------------------------------------
def test_the_recommended_allocation_is_frozen_in_the_pre_registration():
    rows = {"ln_baseline": {"norm_name": "ln"},
            "dytcd_full": {"norm_name": "dytcd"}}
    hypotheses = dict.fromkeys(rows, "h")
    matrix_report = {"leave_one_out": {}, "factorial": {}, "qk_axis": {},
                     "three_way_factorial": {}, "dose_response": {}}
    plain = build_analysis_plan(rows, hypotheses, matrix_report,
                                ["ln_baseline"], "gpt2-124m", [1, 2, 3])
    payload = {"budget_hours": 4.0, "tokens_per_run": 1000,
               "planned_seeds": {"ln_baseline": 3, "dytcd_full": 3},
               "recommended_allocation": {"name": "as_planned",
                                          "arms": ["ln_baseline", "dytcd_full"],
                                          "seeds": {"ln_baseline": 3,
                                                    "dytcd_full": 3}},
               "fits_budget": True}
    frozen = build_analysis_plan(rows, hypotheses, matrix_report,
                                 ["ln_baseline"], "gpt2-124m", [1, 2, 3],
                                 compute_budget=payload)
    assert "compute_budget" not in plain
    assert frozen["compute_budget"] == payload
    assert plain["fingerprint"] != frozen["fingerprint"]
