"""Tests for data-processing parameter sensitivity analysis."""
# ruff: noqa: PLR2004

from __future__ import annotations

import pytest

from adasquash.data_sensitivity import (
    flag_sensitive_statistics,
    parameter_sweep,
    split_position_robustness,
)


def _clean_and_split(options: dict) -> dict:
    """Toy protocol: keep documents above min_doc_chars, optionally dedup."""
    min_chars = int(options.get("min_doc_chars", 0))
    dedup = bool(options.get("dedup", True))
    val_frac = float(options.get("val_frac", 0.0))
    test_frac = float(options.get("test_frac", 0.0))
    docs = ["a" * 5, "b" * 50, "c" * 200, "d" * 5, "e" * 50]
    kept = [d for d in docs if len(d) >= min_chars]
    if dedup:
        kept = list(dict.fromkeys(kept))
    n = len(kept)
    val_n = round(n * val_frac)
    test_n = round(n * test_frac)
    return {
        "documents_kept": n,
        "val_documents": val_n,
        "test_documents": test_n,
        "train_documents": n - val_n - test_n,
        "dedup_rate": 1.0 if dedup else 0.0,
    }


def test_parameter_sweep_runs_every_combination():
    report = parameter_sweep(
        base_options={"seed": 42},
        parameter_grids={
            "min_doc_chars": [0, 10],
            "dedup": [True, False],
        },
        clean_and_split=_clean_and_split,
    )
    assert report["schema"] == "adasquash.data_sensitivity.v1"
    assert report["n_combinations"] == 4
    assert report["n_errored"] == 0
    assert len(report["combinations"]) == 4
    run_ids = {entry["run_id"] for entry in report["combinations"]}
    assert len(run_ids) == 4


def test_parameter_sweep_records_errors_without_crashing():
    def failing(options: dict) -> dict:
        if options.get("min_doc_chars") == 99:
            msg = "boom"
            raise ValueError(msg)
        return {"documents_kept": 1}

    report = parameter_sweep(
        base_options={},
        parameter_grids={"min_doc_chars": [0, 99]},
        clean_and_split=failing,
    )
    assert report["n_errored"] == 1
    errored = [e for e in report["combinations"] if e["error"] is not None]
    assert len(errored) == 1
    assert "boom" in errored[0]["error"]
    ok = [e for e in report["combinations"] if e["error"] is None]
    assert len(ok) == 1


def test_parameter_sweep_per_statistic_sensitivity():
    report = parameter_sweep(
        base_options={},
        parameter_grids={"min_doc_chars": [0, 10, 30]},
        clean_and_split=_clean_and_split,
    )
    sens = report["per_statistic"]
    assert "documents_kept" in sens
    assert sens["documents_kept"]["min"] <= sens["documents_kept"]["max"]
    assert sens["documents_kept"]["n"] == 3
    assert "values_by_run" in sens["documents_kept"]


def test_split_position_robustness_baseline_and_perturbation():
    report = split_position_robustness(
        base_options={"val_frac": 0.1, "test_frac": 0.1, "min_doc_chars": 0},
        perturbations=[0.0, 0.05],
        clean_and_split=_clean_and_split,
    )
    assert report["schema"] == "adasquash.split_position_robustness.v1"
    assert report["perturbations"] == [0.0, 0.05]
    assert len(report["entries"]) == 2
    baseline = report["entries"][0]
    assert baseline["offset"] == 0.0
    assert baseline["error"] is None
    assert "delta_from_baseline" in baseline
    assert all(delta["delta"] == 0.0 for delta in baseline["delta_from_baseline"].values())


def test_split_position_robustness_records_delta():
    report = split_position_robustness(
        base_options={"val_frac": 0.1, "test_frac": 0.1, "min_doc_chars": 0},
        perturbations=[0.2],
        clean_and_split=_clean_and_split,
    )
    entry = report["entries"][0]
    assert entry["error"] is None
    delta = entry["delta_from_baseline"]
    assert "val_documents" in delta
    assert "delta" in delta["val_documents"]


def test_split_position_robustness_rejects_invalid_perturbation():
    with pytest.raises(TypeError):
        split_position_robustness(
            base_options={},
            perturbations=["not-a-number"],
            clean_and_split=_clean_and_split,
        )
    with pytest.raises(ValueError):
        split_position_robustness(
            base_options={},
            perturbations=[float("inf")],
            clean_and_split=_clean_and_split,
        )


def test_split_position_robustness_clip_leaves_training():
    report = split_position_robustness(
        base_options={"val_frac": 0.4, "test_frac": 0.4, "min_doc_chars": 0},
        perturbations=[0.3],
        clean_and_split=_clean_and_split,
    )
    entry = report["entries"][0]
    assert entry["error"] is not None
    assert "leave no training documents" in entry["error"]


def test_flag_sensitive_statistics_flags_high_cv():
    report = parameter_sweep(
        base_options={},
        parameter_grids={"min_doc_chars": [0, 6, 11]},
        clean_and_split=_clean_and_split,
    )
    flags = flag_sensitive_statistics(report, relative_range_threshold=0.01)
    assert "documents_kept" in flags["flags"]
    assert flags["flags"]["documents_kept"]["source"] == "parameter_grid"
    assert flags["flags"]["documents_kept"]["triggered"]


def test_flag_sensitive_statistics_no_flags_when_stable():
    def stable(options: dict) -> dict:
        return {"constant": 1.0}

    report = parameter_sweep(
        base_options={},
        parameter_grids={"min_doc_chars": [0, 0]},
        clean_and_split=stable,
    )
    flags = flag_sensitive_statistics(report, relative_range_threshold=0.5)
    assert flags["n_flagged"] == 0
    assert flags["flags"] == {}


def test_flag_sensitive_statistics_rejects_bad_threshold():
    report = {"per_statistic": {}}
    with pytest.raises(ValueError):
        flag_sensitive_statistics(report, relative_range_threshold=1.5)
    with pytest.raises(TypeError):
        flag_sensitive_statistics(report, relative_range_threshold="nope")
