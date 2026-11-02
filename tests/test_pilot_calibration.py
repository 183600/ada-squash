"""Pilot variance calibration and its pre-registration audit.

A confirmatory power calculation is only as honest as the SD it is given. A
small pilot's point estimate is optimistic, and a pilot seed reused in the
confirmatory run is not independent replication. These tests pin:

* the conservative one-sided upper bound and its stable fingerprint;
* seed provenance and pilot/confirmatory overlap detection;
* rejection of edited or optimistic evidence;
* readiness adoption of the recommended SD (and fail-closed opt-in);
* inclusion of the evidence in the analysis-plan fingerprint;
* the pilot-summary CLI helper that produces the evidence record.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.design import (
    design_readiness_audit,
    paired_sd_calibration,
    paired_sd_evidence_audit,
    randomized_complete_block_schedule,
)
from scripts.ablation import build_analysis_plan
from scripts.calibrate_paired_sd import main as calibrate_main

DIFFERENCES = [-0.030, -0.010, -0.020, -0.015, -0.025]
PILOT_SEEDS = [11, 12, 13, 14, 15]
CONFIRMATORY_SEEDS = [42, 43, 44]


def _evidence(confidence=0.90):
    return paired_sd_calibration(
        DIFFERENCES, pilot_seeds=PILOT_SEEDS,
        confirmatory_seeds=CONFIRMATORY_SEEDS, confidence=confidence)


def test_upper_bound_is_conservative_and_fingerprint_is_deterministic():
    first = _evidence()
    second = _evidence()
    assert first["n_pairs"] == len(DIFFERENCES)
    assert first["paired_sd"] > 0.0
    assert first["sd_upper_confidence_bound"] > first["paired_sd"]
    assert first["recommended_paired_sd"] == first["sd_upper_confidence_bound"]
    assert first["recommended_basis"] == "one_sided_upper_confidence_bound"
    assert first["fingerprint"] == second["fingerprint"]
    assert first["seed_overlap"] == []
    assert first["seeds_disjoint"] is True


def test_evidence_audit_accepts_frozen_disjoint_evidence():
    evidence = _evidence()
    report = paired_sd_evidence_audit(
        evidence, evidence["recommended_paired_sd"], CONFIRMATORY_SEEDS)
    assert report["status"] == "pass"
    assert report["fingerprint_verified"] is True
    assert report["n_fail"] == 0


def test_evidence_audit_fails_optimistic_or_edited_evidence():
    evidence = _evidence()
    optimistic = paired_sd_evidence_audit(
        evidence, evidence["recommended_paired_sd"] * 0.5,
        CONFIRMATORY_SEEDS)
    assert optimistic["status"] == "fail"
    assert {check["id"] for check in optimistic["checks"]
            if check["status"] == "fail"} == {"planning_sd"}

    edited = dict(evidence)
    edited["differences"] = list(evidence["differences"])
    edited["differences"][0] -= 0.05
    tampered = paired_sd_evidence_audit(edited, None, CONFIRMATORY_SEEDS)
    assert tampered["status"] == "fail"
    assert tampered["fingerprint_verified"] is False
    assert "fingerprint" in {check["id"] for check in tampered["checks"]
                             if check["status"] == "fail"}


def test_evidence_audit_fails_pilot_seed_reuse():
    evidence = _evidence()
    reused = paired_sd_evidence_audit(
        evidence, evidence["recommended_paired_sd"], [11, 42, 43])
    assert reused["status"] == "fail"
    assert reused["seed_overlap"] == [11]
    assert reused["seeds_disjoint"] is False


@pytest.mark.parametrize("differences, kwargs", [
    ([0.01], {}),
    ([0.01, 0.02], {"confidence": 0.5}),
    ([0.01, 0.02], {"confidence": 1.0}),
    ([0.01, 0.02], {"min_pairs": 1}),
    ([0.01, float("nan")], {}),
    ([{"difference": 0.01, "seed": 1},
      {"difference": 0.02, "seed": 1}], {}),
])
def test_calibration_rejects_invalid_inputs(differences, kwargs):
    with pytest.raises(ValueError):
        paired_sd_calibration(differences, **kwargs)


def _readiness(evidence, *, assumed=None, require=False, n_seeds=20):
    rows = {"candidate": {}, "reference": {}}
    hypotheses = dict.fromkeys(rows, "h")
    seeds = list(range(100, 100 + n_seeds))
    schedule = randomized_complete_block_schedule(
        list(rows), [str(seed) for seed in seeds], seed=3)
    report = design_readiness_audit(
        seeds, rows, hypotheses, schedule=schedule,
        expected_units=schedule["order"], margin=0.05,
        assumed_paired_sd=assumed,
        schedule_arm_of=lambda unit: unit.rsplit("::rep=", 1)[0],
        identity_controls=[("candidate", "reference")],
        paired_sd_evidence=evidence,
        require_variance_evidence=require)
    return report


def test_readiness_adopts_the_upper_bound_when_no_sd_was_supplied():
    evidence = _evidence()
    report = _readiness(evidence)
    by_id = {check["id"]: check for check in report["checks"]}
    assert by_id["variance_evidence"]["status"] == "pass"
    assert by_id["power"]["status"] == "pass"
    assert report["assumed_paired_sd"] == pytest.approx(
        evidence["recommended_paired_sd"])
    assert report["paired_sd_evidence_audit"]["status"] in ("pass", "warn")


def test_readiness_fails_when_variance_evidence_is_required_but_absent():
    report = _readiness(None, assumed=0.01, require=True)
    by_id = {check["id"]: check for check in report["checks"]}
    assert by_id["variance_evidence"]["status"] == "fail"
    assert report["ready"] is False


def test_analysis_plan_freezes_the_pilot_evidence():
    rows = {"ln_baseline": {"norm_name": "ln"},
            "dytcd_full": {"norm_name": "dytcd"}}
    hypotheses = dict.fromkeys(rows, "h")
    matrix_report = {"leave_one_out": {}, "factorial": {}, "qk_axis": {}}
    kwargs = dict(alpha=0.05, margin=0.0, equivalence_margin=0.0)
    without = build_analysis_plan(
        rows, hypotheses, matrix_report, ["ln_baseline"], "gpt2-124m",
        [42, 43, 44], **kwargs)
    evidence = _evidence()
    with_evidence = build_analysis_plan(
        rows, hypotheses, matrix_report, ["ln_baseline"], "gpt2-124m",
        [42, 43, 44], paired_sd_evidence=evidence, **kwargs)
    assert "paired_sd_evidence" not in without
    assert with_evidence["paired_sd_evidence"] == evidence
    assert with_evidence["fingerprint"] != without["fingerprint"]
    with pytest.raises(ValueError):
        build_analysis_plan(
            rows, hypotheses, matrix_report, ["ln_baseline"], "gpt2-124m",
            [42, 43, 44], paired_sd_evidence=["not", "a", "mapping"], **kwargs)


def test_calibration_script_reads_pilot_summaries(tmp_path):
    summaries = [
        {"name": "candidate", "per_seed": [
            {"seed": 42, "val_loss": 0.90},
            {"seed": 43, "val_loss": 0.93},
            {"seed": 44, "val_loss": 0.91},
        ]},
        {"name": "reference", "per_seed": [
            {"seed": 42, "val_loss": 0.94},
            {"seed": 43, "val_loss": 0.95},
            {"seed": 44, "val_loss": 0.96},
        ]},
    ]
    summaries_path = tmp_path / "all_summaries.json"
    summaries_path.write_text(json.dumps(summaries))
    output = tmp_path / "paired_sd.json"
    rc = calibrate_main([
        str(summaries_path), "--candidate", "candidate",
        "--reference", "reference", "--confirmatory-seeds", "50,51,52",
        "--output", str(output)])
    assert rc == 0
    evidence = json.loads(output.read_text())
    assert evidence["schema"] == "adasquash.paired_sd_calibration.v1"
    assert evidence["n_pairs"] == 3
    assert evidence["pilot_seeds"] == [42, 43, 44]
    assert evidence["confirmatory_seeds"] == [50, 51, 52]
    assert evidence["seeds_disjoint"] is True
    assert evidence["recommended_paired_sd"] > 0.0
    assert paired_sd_evidence_audit(
        evidence, evidence["recommended_paired_sd"], [50, 51, 52])["status"] == "pass"


def test_zero_variance_pilot_never_recommends_a_zero_planning_sd():
    evidence = paired_sd_calibration([0.01, 0.01, 0.01],
                                     pilot_seeds=[1, 2, 3])
    assert evidence["zero_variance"] is True
    assert evidence["sd_upper_confidence_bound"] == 0.0
    assert evidence["recommended_paired_sd"] is None
    assert evidence["recommended_basis"] == "zero_variance_pilot_not_usable"
    report = paired_sd_evidence_audit(evidence)
    assert report["status"] == "fail"
    assert {check["id"] for check in report["checks"]
            if check["status"] == "fail"} == {"planning_sd"}
