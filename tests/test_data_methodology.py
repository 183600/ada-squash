"""Data-processing checks for the leakage-safe research pipeline.

The training pipeline already refuses an unvalidated split; these tests pin
the newer data-quality guarantees:

* the cleaning options are validated before they can silently drop every
  document or disable deduplication;
* the audit's held-out token budget separates the observed effective token
  count from a prefix extrapolation, so a small validation slice cannot look
  precise just because it is disjoint;
* token-shingle MinHash near-duplicate detection catches cross-split
  contamination that exact/canonical document hashes miss, while still
  reporting a sampled scan as a lower bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_data import (split_token_budget, minhash_signature,
                                near_duplicate_leakage, near_duplicate_record,
                                token_shingles, train_wrap_status)
from scripts.data_pipeline import (configure_data_options, get_data_options,
                                   split_document_boundaries,
                                   validate_data_options)


@pytest.fixture(autouse=True)
def _restore_global_data_options():
    before = get_data_options()
    yield
    from scripts import data_pipeline
    data_pipeline._active_data_options.clear()
    data_pipeline._active_data_options.update(before)


def _record(tokens, doc_index):
    record = near_duplicate_record(tokens, shingle_size=8)
    record["doc_index"] = doc_index
    return record


# ---------------------------------------------------------------------------
# Shingles and MinHash
# ---------------------------------------------------------------------------
def test_token_shingles_are_deterministic_and_order_sensitive():
    tokens = list(range(40))
    first = token_shingles(tokens, 8)
    assert first == token_shingles(tokens, 8)
    assert len(first) == len(tokens) - 8 + 1
    reversed_hashes = token_shingles(list(reversed(tokens)), 8)
    assert first != reversed_hashes
    # A document shorter than the shingle size still yields one shingle,
    # rather than silently disappearing from the near-duplicate scan.
    assert len(token_shingles([1, 2, 3], 8)) == 1
    assert token_shingles([], 8) == set()


def test_minhash_signature_is_stable_and_bounded():
    shingles = token_shingles(list(range(50)), 8)
    sig_a = minhash_signature(shingles)
    sig_b = minhash_signature(set(shingles))
    assert sig_a == sig_b
    assert len(sig_a) == 240
    assert all(0 <= value <= (1 << 64) - 1 for value in sig_a)


# ---------------------------------------------------------------------------
# Cross-split near-duplicate contamination
# ---------------------------------------------------------------------------
def test_near_duplicate_leakage_flags_a_reworded_training_document():
    base = list(range(200))
    # Keep most of the token stream, change a few tokens: exact/canonical
    # hashes see two different documents, MinHash sees the overlap.
    reworded = base[:95] + [9000, 9001, 9002, 9003, 9004] + base[100:]
    train = _record(base, 0)
    validation = _record(reworded, 1)
    report = near_duplicate_leakage(
        {"train": [train], "validation": [validation]}, threshold=0.8)
    pair = report["pairs"]["train->validation"]
    assert pair["flagged_later_documents"] == 1
    assert pair["flagged_pairs"] == 1
    assert pair["examples"][0]["estimated_jaccard"] >= 0.8
    assert report["problems"]
    assert "near-duplicate documents cross train->validation" in report["problems"][0]


def test_near_duplicate_leakage_does_not_flag_unrelated_or_distant_documents():
    train = _record(list(range(300)), 0)
    unrelated = _record(list(range(1000, 1300)), 1)
    report = near_duplicate_leakage(
        {"train": [train], "validation": [unrelated]}, threshold=0.8)
    pair = report["pairs"]["train->validation"]
    assert pair["flagged_pairs"] == 0
    assert pair["flagged_later_documents"] == 0
    assert report["problems"] == []


def test_near_duplicate_leakage_checks_every_earlier_later_boundary():
    base = list(range(120))
    report = near_duplicate_leakage({
        "train": [_record(base, 0)],
        "validation": [_record(base, 1)],
        "test": [_record(base, 2)],
    }, threshold=0.8)
    assert set(report["pairs"]) == {
        "train->validation", "train->test", "validation->test"}
    assert all(pair["flagged_pairs"] == 1 for pair in report["pairs"].values())


def test_near_duplicate_sketch_caps_very_long_documents():
    record = near_duplicate_record(list(range(5000)), shingle_size=8,
                                   max_shingles=64)
    assert record["n_shingles"] == 5000 - 8 + 1
    assert record["n_shingles_sketched"] == 64
    assert len(record["signature"]) == 240


def test_near_duplicate_threshold_must_be_a_probability():
    train = _record(list(range(100)), 0)
    with pytest.raises(ValueError):
        near_duplicate_leakage({"train": [train]}, threshold=0.0)
    with pytest.raises(ValueError):
        near_duplicate_leakage({"train": [train]}, threshold=1.5)


def test_sampled_near_duplicate_scan_reports_a_lower_bound_not_a_clean_bill():
    train = _record(list(range(100)), 0)
    unrelated = _record(list(range(1000, 1100)), 1)
    report = near_duplicate_leakage(
        {"train": [train], "validation": [unrelated]}, threshold=0.8,
        exhaustive_by_split={"train": False, "validation": False},
    )
    assert report["exhaustive"] is False
    assert report["lower_bound"] is True
    assert report["notes"]
    assert "lower bound" in report["notes"][0]


# ---------------------------------------------------------------------------
# Cleaning configuration and split geometry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"max_doc_chars": -1},
    {"min_doc_chars": 10, "max_doc_chars": 5},
    {"dedup_max": 0},
    {"num_documents": 0},
    {"num_workers": -1},
    {"prefetch_factor": 0},
    {"val_frac": 0.6, "test_frac": 0.5},
    {"split": "dev"},
    {"group_key": "  "},
])
def test_validate_data_options_rejects_silently_destructive_values(bad):
    with pytest.raises(ValueError):
        validate_data_options(bad)


def test_validate_data_options_accepts_a_normal_heldout_configuration(tmp_path):
    ledger = tmp_path / "train.hash"
    ledger.write_text("00\n", encoding="utf-8")
    opts = validate_data_options({
        "split": "test",
        "val_frac": 0.0,
        "test_frac": 0.01,
        "num_documents": 1000,
        "dedup": True,
        "dedup_max": 100,
        "min_doc_chars": 1,
        "max_doc_chars": 0,
        "exclude_hash_file": ledger,
        "group_key": "meta.url",
    })
    assert opts["split"] == "test"
    assert opts["num_documents"] == 1000
    assert opts["exclude_hash_file"] == ledger


def test_invalid_configure_call_does_not_commit_a_partial_configuration():
    before = get_data_options()
    with pytest.raises(ValueError):
        configure_data_options(split="validation", val_frac=0.9, test_frac=0.9)
    assert get_data_options() == before
    # A valid call, by contrast, is committed atomically.
    configure_data_options(split="validation", val_frac=0.01, test_frac=0.01)
    assert get_data_options()["split"] == "validation"


def test_chronological_split_is_contiguous_and_disjoint():
    boundaries = split_document_boundaries(1000, 0.01, 0.02)
    train = boundaries["train"]
    validation = boundaries["validation"]
    test = boundaries["test"]
    assert train[0] == 0
    assert train[0] + train[1] == validation[0]
    assert validation[0] + validation[1] == test[0]
    assert test[0] + test[1] == 1000
    assert train[1] > 0 and validation[1] > 0 and test[1] > 0


# ---------------------------------------------------------------------------
# Held-out token budget
# ---------------------------------------------------------------------------
def test_split_token_budget_is_exact_when_the_split_was_fully_read():
    stats = {
        "documents_sampled": 1000,
        "effective_documents_kept": 800,
        "tokens_total_effective": 800_000,
        "exhaustive": True,
    }
    budget = split_token_budget(stats, {"count": 1000})
    assert budget["basis"] == "observed"
    assert budget["estimated_effective_documents"] == 800
    assert budget["estimated_effective_tokens"] == 800_000


def test_split_token_budget_extrapolates_a_sampled_prefix():
    stats = {
        "documents_sampled": 100,
        "effective_documents_kept": 80,
        "tokens_total_effective": 80_000,
        "exhaustive": False,
    }
    budget = split_token_budget(stats, {"count": 1000})
    assert budget["basis"] == "extrapolated_from_sampled_prefix"
    assert budget["estimated_effective_documents"] == 800
    # 1000 effective tokens per kept document * 800 effective documents.
    assert budget["estimated_effective_tokens"] == 800_000
    assert budget["sampled_effective_tokens"] == 80_000


def test_train_wrap_status_exposes_a_recycled_stream():
    budget = {"estimated_effective_tokens": 100_000.0, "basis": "observed",
              "exhaustive": True}
    wrap = train_wrap_status(budget, 1_000_000)
    assert wrap["wrap_expected"] is True
    assert wrap["coverage_ratio"] == pytest.approx(0.1)
    no_wrap = train_wrap_status(budget, 50_000)
    assert no_wrap["wrap_expected"] is False
    assert no_wrap["coverage_ratio"] == pytest.approx(2.0)
    unknown = train_wrap_status(None, 0)
    assert unknown["wrap_expected"] is None
    assert unknown["scheduled_tokens"] is None


def test_split_token_budget_refuses_to_invent_tokens_from_an_empty_split():
    budget = split_token_budget(
        {"documents_sampled": 10, "effective_documents_kept": 0,
         "tokens_total_effective": 0, "exhaustive": True},
        {"count": 10})
    assert budget["estimated_effective_tokens"] is None
    assert budget["basis"] == "no_usable_documents"
