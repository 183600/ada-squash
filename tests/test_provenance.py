"""Study-level provenance checks.

The training and ablation scripts already record manifests, artifact
fingerprints and a pre-registration hash. These tests pin the audit that
verifies those records against the files on disk, and the atomic JSON writer
used by the report producers. A tampered or stale study must fail closed; a
consistent study must not produce a false alarm.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.ablation.analysis import analysis_plan_fingerprint
from adasquash.jsonio import canonical_json_dumps, save_json_atomic
from adasquash.provenance import ANALYSIS_PLAN_SCHEMA, audit_study, verify_artifact_record
from adasquash.repro import (
    RUN_MANIFEST_SCHEMA,
    artifact_record,
    file_fingerprint,
    save_run_manifest,
)


def _manifest(seed: int, log_path: Path, checkpoint_path: Path) -> dict:
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "status": "finished",
        "seed": seed,
        "git": {"commit": "a" * 40, "dirty": False},
        "results": {
            "final_val_loss": 1.0 + seed / 100.0,
            "best_val_loss": 1.0 + seed / 100.0,
            "final_val_step": 10,
            "best_val_step": 10,
            "n_params": 100,
            "n_layers": 2,
            "dim": 16,
            "n_heads": 4,
            "n_loss_spikes": 0,
            "n_nonfinite_losses": 0,
            "n_nonfinite_gradients": 0,
            "n_nonfinite_val_losses": 0,
        },
        "data": {
            "split_reports": {
                "train": {
                    "dataset_revision_resolved": "d" * 40,
                    "tokenizer_revision_resolved": "t" * 40,
                },
            },
        },
        "artifacts": {
            "train_jsonl": file_fingerprint(log_path),
            "checkpoint_final": artifact_record(checkpoint_path),
        },
    }


def _study(tmp_path: Path, seeds=(42, 43, 44)) -> dict:
    seed_records = []
    for seed in seeds:
        run_dir = tmp_path / f"seed{seed}"
        run_dir.mkdir(parents=True)
        log_path = run_dir / "train.jsonl"
        log_path.write_text(f'{{"seed": {seed}, "loss": 1.0}}\n', encoding="utf-8")
        checkpoint_path = run_dir / "ckpt_final.pt"
        checkpoint_path.write_bytes(f"checkpoint-{seed}".encode("ascii"))
        manifest = _manifest(seed, log_path, checkpoint_path)
        save_run_manifest(run_dir / "run_manifest.json", manifest)
        seed_records.append({
            "seed": seed,
            **{key: manifest["results"][key] for key in (
                "final_val_loss", "best_val_loss", "final_val_step",
                "best_val_step", "n_params", "n_layers", "dim", "n_heads",
                "n_loss_spikes", "n_nonfinite_losses", "n_nonfinite_gradients",
                "n_nonfinite_val_losses")},
            "data_version": {
                "dataset_revision_resolved": "d" * 40,
                "tokenizer_revision_resolved": "t" * 40,
            },
        })

    aggregate = {
        "model": "tiny",
        "norm": "dyt",
        "seeds": list(seeds),
        "n_seeds": len(seeds),
        "n_seeds_requested": len(seeds),
        "complete": True,
        "certified": True,
        "missing_seeds": [],
        "invalid_seeds": [],
        "manifest_rejections": [],
        "data_versions_complete": True,
        "data_versions_consistent": True,
        "shapes_complete": True,
        "shapes_consistent": True,
        "data_ledgers_complete": True,
        "per_seed": seed_records,
    }
    aggregate_path = tmp_path / "aggregate.json"
    save_json_atomic(aggregate_path, aggregate)
    replication = {
        "schema": RUN_MANIFEST_SCHEMA,
        "status": "finished",
        "seed": int(seeds[0]),
        "git": {"commit": "a" * 40, "dirty": False},
        "seeds": list(seeds),
        "aggregate": aggregate,
        "artifacts": {
            "aggregate": artifact_record(aggregate_path),
            "per_seed_manifests": {
                str(seed): artifact_record(tmp_path / f"seed{seed}" / "run_manifest.json")
                for seed in seeds
            },
        },
    }
    save_run_manifest(tmp_path / "run_manifest.json", replication)
    return aggregate


def _analysis_plan() -> dict:
    payload = {
        "model": "tiny",
        "seeds": [42, 43, 44],
        "alpha": 0.05,
        "margin": 0.0,
        "equivalence_margin": 0.0,
        "baseline_pool": [],
        "expected_attrition": 0.0,
        "exclusion_policy": {},
        "exclusion_policy_fingerprint": "policy",
        "row_configs": {},
        "matrix_cells": {},
        "run_schedule": None,
        "contrasts": [],
    }
    return {
        "schema": ANALYSIS_PLAN_SCHEMA,
        **payload,
        "fingerprint": analysis_plan_fingerprint(payload),
        "n_contrasts": 0,
        "plan_changed": False,
    }


# ---------------------------------------------------------------------------
# Atomic canonical JSON
# ---------------------------------------------------------------------------
def test_canonical_json_is_key_sorted_and_round_trips():
    text = canonical_json_dumps({"b": 1, "a": {"d": 2, "c": 3}})
    assert text == '{"a":{"c":3,"d":2},"b":1}'
    assert json.loads(text) == {"a": {"c": 3, "d": 2}, "b": 1}


def test_save_json_atomic_replaces_and_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "report.json"
    save_json_atomic(path, {"old": True})
    save_json_atomic(path, {"new": True})
    assert json.loads(path.read_text()) == {"new": True}
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_atomic_write_preserves_the_previous_report(tmp_path):
    path = tmp_path / "report.json"
    save_json_atomic(path, {"ok": True})
    with pytest.raises(ValueError):
        save_json_atomic(path, {"bad": float("nan")}, allow_nan=False)
    assert json.loads(path.read_text()) == {"ok": True}


# ---------------------------------------------------------------------------
# Artifact records
# ---------------------------------------------------------------------------
def test_artifact_record_verification_detects_tampering(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"original")
    record = file_fingerprint(path)
    assert verify_artifact_record(record)["ok"] is True
    path.write_bytes(b"tampered")
    result = verify_artifact_record(record)
    assert result["ok"] is False
    assert any("sha256 mismatch" in problem for problem in result["problems"])


def test_artifact_record_verification_reports_a_missing_file(tmp_path):
    path = tmp_path / "missing.bin"
    record = artifact_record(path)
    record["exists"] = True
    result = verify_artifact_record(record)
    assert result["ok"] is False
    assert any("missing" in problem for problem in result["problems"])


# ---------------------------------------------------------------------------
# Study audit
# ---------------------------------------------------------------------------
def test_audit_accepts_a_consistent_study(tmp_path):
    _study(tmp_path)
    report = audit_study(tmp_path)
    assert report["ok"] is True, report["problems"]
    assert report["summary"]["n_manifests"] == 4
    assert report["summary"]["n_artifacts"] >= 6
    assert report["summary"]["n_aggregates"] == 1


def test_audit_fails_on_a_tampered_artifact(tmp_path):
    _study(tmp_path)
    (tmp_path / "seed42" / "train.jsonl").write_text("tampered\n")
    report = audit_study(tmp_path)
    assert report["ok"] is False
    assert any("sha256 mismatch" in problem for problem in report["problems"])


def test_audit_fails_when_an_aggregate_disagrees_with_a_seed_manifest(tmp_path):
    aggregate = _study(tmp_path)
    aggregate["per_seed"][0]["final_val_loss"] = 99.0
    save_json_atomic(tmp_path / "aggregate.json", aggregate)
    report = audit_study(tmp_path)
    assert report["ok"] is False
    assert any("final_val_loss differs" in problem for problem in report["problems"])


def test_audit_fails_on_a_dirty_run(tmp_path):
    _study(tmp_path)
    path = tmp_path / "seed42" / "run_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["git"]["dirty"] = True
    save_run_manifest(path, manifest)
    report = audit_study(tmp_path)
    assert report["ok"] is False
    assert any("dirty working tree" in problem for problem in report["problems"])


def test_audit_verifies_the_analysis_plan_fingerprint(tmp_path):
    _study(tmp_path)
    plan = _analysis_plan()
    save_json_atomic(tmp_path / "analysis_plan.json", plan)
    assert audit_study(tmp_path)["ok"] is True

    plan["contrasts"] = [{"id": "post-hoc"}]
    save_json_atomic(tmp_path / "analysis_plan.json", plan)
    report = audit_study(tmp_path)
    assert report["ok"] is False
    assert any("fingerprint does not match" in problem
               for problem in report["problems"])


def test_audit_fails_on_a_post_hoc_plan_change(tmp_path):
    _study(tmp_path)
    plan = _analysis_plan()
    plan["plan_changed"] = True
    save_json_atomic(tmp_path / "analysis_plan.json", plan)
    report = audit_study(tmp_path)
    assert report["ok"] is False
    assert any("changed after its first registration" in problem
               for problem in report["problems"])


def test_audit_cli_exit_status_follows_the_problem_list(tmp_path, capsys):
    from scripts.audit_study import main

    _study(tmp_path)
    assert main([str(tmp_path)]) == 0
    (tmp_path / "seed42" / "train.jsonl").write_text("tampered\n")
    assert main([str(tmp_path)]) == 1
    capsys.readouterr()
