"""Tests for the deterministic data-processing audit log."""
# ruff: noqa: PLR2004

from __future__ import annotations

import json

import pytest

from adasquash.data_audit_log import (
    DATA_AUDIT_LOG_SCHEMA,
    DataAuditEntry,
    DataAuditLog,
    audit_step,
)


def test_entry_records_counts_and_drop_reasons():
    entry = DataAuditEntry(
        step="clean",
        input_count=100,
        output_count=80,
        drop_reasons={"dropped_empty": 10, "dropped_duplicate": 10},
        parameters={"min_doc_chars": 1, "dedup": True},
    )
    assert entry.step == "clean"
    assert entry.input_count == 100
    assert entry.output_count == 80
    assert entry.drop_reasons == {
        "dropped_duplicate": 10,
        "dropped_empty": 10,
    }
    assert entry.to_dict()["dropped"] == 20
    assert entry.to_dict()["drop_rate"] == pytest.approx(0.2)


def test_entry_to_dict_and_from_dict_roundtrip():
    entry = DataAuditEntry(
        step="pack",
        input_count=50,
        output_count=50,
        input_hash="aa" * 32,
        output_hash="bb" * 32,
        parameters={"seq_len": 1024, "seed": 42},
    )
    data = entry.to_dict()
    restored = DataAuditEntry.from_dict(data)
    assert restored == entry
    assert restored.parameters == {"seq_len": 1024, "seed": 42}


def test_entry_rejects_unknown_step():
    with pytest.raises(ValueError):
        DataAuditEntry(step="bogus", input_count=1, output_count=1)


def test_entry_rejects_more_output_than_input():
    with pytest.raises(ValueError):
        DataAuditEntry(step="clean", input_count=5, output_count=10)


def test_log_append_and_summary():
    log = DataAuditLog(pipeline_id="pipe-1")
    log.add_step("load", 1000, 1000, parameters={"dataset": "x"})
    log.add_step("clean", 1000, 950, drop_reasons={"dropped_empty": 50})
    log.add_step("pack", 950, 950, parameters={"seq_len": 1024})
    summary = log.summary()
    assert summary["pipeline_id"] == "pipe-1"
    assert summary["n_entries"] == 3
    assert summary["steps"] == ["load", "clean", "pack"]
    assert summary["total_input_documents"] == 2950
    assert summary["total_output_documents"] == 2900
    assert summary["total_dropped"] == 50
    assert summary["aggregate_drop_reasons"] == {"dropped_empty": 50}


def test_log_to_dict_and_from_dict_roundtrip():
    log = DataAuditLog(pipeline_id="pipe-2")
    log.add_step("clean", 10, 8, drop_reasons={"dropped_duplicate": 2})
    data = log.to_dict()
    assert data["schema"] == DATA_AUDIT_LOG_SCHEMA
    assert data["pipeline_id"] == "pipe-2"
    restored = DataAuditLog.from_dict(data)
    assert restored.pipeline_id == "pipe-2"
    assert len(restored) == 1
    assert restored._entries[0].drop_reasons == {"dropped_duplicate": 2}


def test_log_to_json_and_from_json_roundtrip():
    log = DataAuditLog(pipeline_id="pipe-3")
    log.add_step("split", 1000, 1000, parameters={"val_frac": 0.005})
    text = log.to_json()
    restored = DataAuditLog.from_json(text)
    assert restored.pipeline_id == "pipe-3"
    assert restored.steps() == ["split"]


def test_audit_step_convenience():
    log = DataAuditLog()
    entry = audit_step(
        log,
        "dedup",
        500,
        480,
        drop_reasons={"dropped_duplicate": 20},
        parameters={"dedup_max": 1000},
    )
    assert len(log) == 1
    assert entry is log._entries[0]
    assert entry.input_count == 500
    assert entry.output_count == 480


def test_verify_determinism_matches_identical_logs():
    a = DataAuditLog(pipeline_id="same")
    a.add_step(
        "clean", 100, 90, drop_reasons={"dropped_empty": 10}, parameters={"min_doc_chars": 1}
    )
    b = DataAuditLog(pipeline_id="same")
    b.add_step(
        "clean", 100, 90, drop_reasons={"dropped_empty": 10}, parameters={"min_doc_chars": 1}
    )
    result = a.verify_determinism(b)
    assert result["deterministic"] is True
    assert result["problems"] == []


def test_verify_determinism_detects_differences():
    a = DataAuditLog(pipeline_id="same")
    a.add_step(
        "clean", 100, 90, drop_reasons={"dropped_empty": 10}, parameters={"min_doc_chars": 1}
    )
    b = DataAuditLog(pipeline_id="same")
    b.add_step(
        "clean", 100, 85, drop_reasons={"dropped_empty": 15}, parameters={"min_doc_chars": 2}
    )
    result = a.verify_determinism(b)
    assert result["deterministic"] is False
    assert any("output_count" in p for p in result["problems"])
    assert any("drop_reasons" in p for p in result["problems"])
    assert any("parameters" in p for p in result["problems"])


def test_verify_determinism_rejects_different_step_order():
    a = DataAuditLog(pipeline_id="same")
    a.add_step("clean", 10, 10)
    a.add_step("pack", 10, 10)
    b = DataAuditLog(pipeline_id="same")
    b.add_step("pack", 10, 10)
    b.add_step("clean", 10, 10)
    result = a.verify_determinism(b)
    assert result["deterministic"] is False
    assert any("step order" in p for p in result["problems"])


def test_verify_determinism_rejects_different_pipeline_id():
    a = DataAuditLog(pipeline_id="a")
    a.add_step("clean", 10, 10)
    b = DataAuditLog(pipeline_id="b")
    b.add_step("clean", 10, 10)
    result = a.verify_determinism(b)
    assert result["deterministic"] is False
    assert any("pipeline id" in p for p in result["problems"])


def test_entry_json_serialisable():
    entry = DataAuditEntry(
        step="exclude",
        input_count=30,
        output_count=25,
        drop_reasons={"dropped_cross_split_duplicate": 5},
        parameters={"exclude_hash_file": "ledger.txt"},
    )
    payload = json.dumps(entry.to_dict(), sort_keys=True)
    assert "exclude" in payload
    assert "dropped_cross_split_duplicate" in payload
