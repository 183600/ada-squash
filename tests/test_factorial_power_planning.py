"""Prospective power planning for the factorial (interaction) claims.


The ablation matrix makes two kinds of claim. The arm-vs-arm claim ("the
proposal lowers the held-out loss") is a simple paired contrast, and the
existing power planner sizes it. The factorial claim ("the per-channel and
depth axes are orthogonal", "the combination is synergistic") is a *different
contrast*: it is built from four or eight cells at once, so its own noise
level is larger than the noise of a single arm-vs-arm difference. A design
that is powered for the main claim can still be unable to see an interaction
of the size it claims to care about.

These tests pin the conversion between the two scales, check it against the
coefficients the analysis code actually uses, and check that the pre-flight
audit refuses a factorial study whose interaction claims were never planned.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.ablation.contrasts import _factorial_term_coefficients
from adasquash.ablation.matrix import THREE_WAY_FACTORIAL_SPEC
from adasquash.design import (
    design_readiness_audit,
    factorial_contrast_sd,
    factorial_power_plan,
    prospective_power_plan,
)
from scripts.ablation import build_analysis_plan


# ---------------------------------------------------------------------------
# The contrast scales
# ---------------------------------------------------------------------------
def test_contrast_sd_multipliers_follow_the_design_geometry():
    sd = 0.1
    assert factorial_contrast_sd(sd, 2, 1) == pytest.approx(sd / math.sqrt(2.0))
    assert factorial_contrast_sd(sd, 2, 2) == pytest.approx(sd * math.sqrt(2.0))
    assert factorial_contrast_sd(sd, 3, 1) == pytest.approx(sd / 2.0)
    assert factorial_contrast_sd(sd, 3, 2) == pytest.approx(sd)
    assert factorial_contrast_sd(sd, 3, 3) == pytest.approx(sd * 2.0)


def test_contrast_sd_matches_the_coefficients_of_the_analysis_code():
    design = THREE_WAY_FACTORIAL_SPEC["per_channel_x_depth_x_activation"]
    factors = list(design["factors"])
    levels = design["levels"]
    cells = design["cells"]
    for term in [("per_channel",), ("depth",), ("activation",),
                 ("per_channel", "depth"), ("per_channel", "activation"),
                 ("depth", "activation"),
                 ("per_channel", "depth", "activation")]:
        coefficients = _factorial_term_coefficients(
            factors, levels, cells, term)
        sum_squares = sum(value ** 2 for value in coefficients.values())
        # A simple paired contrast between two cells has sum(w^2) = 2; the
        # multiplier of the term is sqrt(sum(w^2) / 2).
        multiplier = math.sqrt(sum_squares / 2.0)
        assert factorial_contrast_sd(1.0, 3, len(term)) == pytest.approx(
            multiplier)


def test_empirical_interaction_noise_matches_the_planned_scale():
    rng = random.Random(20240501)
    n_seeds = 20000
    sigma = 0.2
    rho = 0.5
    shared = [rng.gauss(0.0, sigma * math.sqrt(rho)) for _ in range(n_seeds)]
    cells = {}
    for name in ("neither", "per_channel", "depth", "both"):
        cells[name] = [shared[index]
                       + rng.gauss(0.0, sigma * math.sqrt(1.0 - rho))
                       for index in range(n_seeds)]
    simple = [cells["per_channel"][index] - cells["neither"][index]
              for index in range(n_seeds)]
    interaction = [(cells["both"][index] - cells["depth"][index])
                   - (cells["per_channel"][index] - cells["neither"][index])
                   for index in range(n_seeds)]

    def _sd(values):
        mean = sum(values) / max(len(values), 1)
        return math.sqrt(sum((value - mean) ** 2 for value in values)
                         / (len(values) - 1))

    simple_sd = _sd(simple)
    interaction_sd = _sd(interaction)
    assert interaction_sd / simple_sd == pytest.approx(math.sqrt(2.0), rel=0.03)
    assert factorial_contrast_sd(simple_sd, 2, 2) == pytest.approx(
        interaction_sd, rel=0.03)


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------
def test_interaction_claim_needs_more_pairs_than_the_simple_contrast():
    simple = prospective_power_plan(0.05, 20, 0.02)
    interaction = factorial_power_plan(
        0.05, 20, n_factors=2, term_size=2, margin=0.02)
    assert interaction["contrast_sd"] > 0.05
    assert (interaction["superiority"]["required_pairs_family_adjusted"]
            > simple["required_pairs_family_adjusted"])


def test_factorial_power_plan_carries_multiplicity_and_attrition():
    single = factorial_power_plan(0.05, 10, n_factors=2, term_size=2,
                                  margin=0.01)
    family = factorial_power_plan(0.05, 10, n_factors=2, term_size=2,
                                  margin=0.01, n_tests=6)
    assert family["superiority"]["family_alpha"] == pytest.approx(
        0.05 / 6.0)
    assert (family["superiority"]["required_pairs_family_adjusted"]
            > single["superiority"]["required_pairs_family_adjusted"])
    with_attrition = factorial_power_plan(
        0.05, 10, n_factors=2, term_size=2, margin=0.01, n_tests=6,
        expected_attrition=0.2)
    assert (with_attrition["superiority"]["required_planned_pairs"]
            > family["superiority"]["required_planned_pairs"])


def test_equivalence_branch_plans_the_orthogonality_claim():
    plan = factorial_power_plan(
        0.05, 5, n_factors=2, term_size=2, equivalence_margin=0.05)
    assert plan["superiority"] is None
    assert plan["equivalence"] is not None
    assert plan["equivalence"]["equivalence_margin"] == pytest.approx(0.05)
    assert plan["equivalence"]["adequately_powered_after_attrition"] is False
    assert plan["note"].startswith("the design is planned for the equivalence")


def test_plan_without_any_margin_is_descriptive_only():
    plan = factorial_power_plan(0.05, 5, n_factors=2, term_size=2)
    assert plan["superiority"] is None
    assert plan["equivalence"] is None
    assert "descriptive only" in plan["note"]


def test_invalid_geometry_and_inputs_are_rejected():
    with pytest.raises(ValueError):
        factorial_contrast_sd(0.1, 2, 3)
    with pytest.raises(ValueError):
        factorial_contrast_sd(-0.1, 2, 2)
    with pytest.raises(ValueError):
        factorial_contrast_sd(0.1, 0, 1)
    with pytest.raises(ValueError):
        factorial_power_plan(0.1, 1, n_factors=2, term_size=2, margin=0.01)
    with pytest.raises(ValueError):
        factorial_power_plan(0.1, 5, n_factors=2, term_size=2, margin=-0.01)
    with pytest.raises(ValueError):
        factorial_power_plan(0.1, 5, n_factors=2, term_size=2,
                             equivalence_margin=-0.01)


# ---------------------------------------------------------------------------
# The pre-flight audit
# ---------------------------------------------------------------------------
def _audit(**kwargs):
    defaults = {
        "seeds": [1, 2, 3, 4, 5],
        "rows": ["proposal", "baseline"],
        "hypotheses": {"proposal": "h", "baseline": "h"},
        "identity_controls": [("proposal", "baseline")],
    }
    defaults.update(kwargs)
    return design_readiness_audit(
        defaults["seeds"], defaults["rows"], defaults["hypotheses"],
        **{key: value for key, value in defaults.items()
           if key not in ("seeds", "rows", "hypotheses")})


def test_audit_warns_when_a_factorial_study_plans_no_interaction_claim():
    report = _audit(assumed_paired_sd=0.05, declares_interaction_claims=True)
    check = next(item for item in report["checks"]
                 if item["id"] == "interaction_effect_size")
    assert check["status"] == "warn"
    assert "descriptive" in check["detail"]


def test_audit_fails_an_unpowered_interaction_claim():
    report = _audit(assumed_paired_sd=0.05, interaction_margin=0.001)
    check = next(item for item in report["checks"]
                 if item["id"] == "interaction_power")
    assert check["status"] == "fail"
    assert report["ready"] is False
    plan = report["planned_interaction_power"]
    assert plan["superiority"]["adequately_powered_after_attrition"] is False


def test_audit_passes_a_powered_interaction_claim():
    report = _audit(seeds=list(range(20)), assumed_paired_sd=0.01,
                    interaction_margin=0.01)
    check = next(item for item in report["checks"]
                 if item["id"] == "interaction_power")
    assert check["status"] == "pass"
    assert report["ready"] is True


def test_audit_checks_the_orthogonality_claim_separately():
    report = _audit(seeds=list(range(20)), assumed_paired_sd=0.01,
                    interaction_equivalence_margin=0.05)
    check = next(item for item in report["checks"]
                 if item["id"] == "interaction_equivalence_power")
    assert check["status"] in ("pass", "fail")
    equivalence = report["planned_interaction_power"]["equivalence"]
    assert equivalence["equivalence_margin"] == pytest.approx(0.05)
    if check["status"] == "fail":
        assert equivalence["adequately_powered_after_attrition"] is False


def test_audit_without_a_planning_sd_leaves_the_interaction_unplanned():
    report = _audit(interaction_margin=0.01)
    check = next(item for item in report["checks"]
                 if item["id"] == "interaction_power")
    assert check["status"] == "warn"


# ---------------------------------------------------------------------------
# The pre-registration
# ---------------------------------------------------------------------------
def _plan_rows():
    rows = {"ln_baseline": {"norm_name": "ln"},
            "dytcd_full": {"norm_name": "dytcd"}}
    hypotheses = dict.fromkeys(rows, "h")
    matrix_report = {"leave_one_out": {}, "factorial": {}, "qk_axis": {},
                     "three_way_factorial": {}, "dose_response": {}}
    return rows, hypotheses, matrix_report


def test_the_interaction_plan_is_frozen_in_the_pre_registration():
    rows, hypotheses, matrix_report = _plan_rows()
    plain = build_analysis_plan(rows, hypotheses, matrix_report,
                                ["ln_baseline"], "gpt2-124m", [1, 2, 3])
    payload = {"worst_geometry": 2, "required_planned_pairs": 42,
               "geometries": {}}
    frozen = build_analysis_plan(rows, hypotheses, matrix_report,
                                 ["ln_baseline"], "gpt2-124m", [1, 2, 3],
                                 interaction_power=payload)
    assert "interaction_power" not in plain
    assert frozen["interaction_power"] == payload
    assert plain["fingerprint"] != frozen["fingerprint"]


def test_the_pre_registration_rejects_a_non_serializable_plan():
    rows, hypotheses, matrix_report = _plan_rows()
    with pytest.raises(ValueError):
        build_analysis_plan(rows, hypotheses, matrix_report,
                            ["ln_baseline"], "gpt2-124m", [1, 2, 3],
                            interaction_power={"value": object()})
    with pytest.raises(ValueError):
        build_analysis_plan(rows, hypotheses, matrix_report,
                            ["ln_baseline"], "gpt2-124m", [1, 2, 3],
                            compute_budget=["not", "a", "mapping"])
