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
  reporting a sampled scan as a lower bound;
* the held-out *evaluation* budget is expressed in distinct packed batches,
  so a validation pass that would cycle its loader (and count repeated
  windows as independent) is visible before a run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.repro import compare_train_streams
from scripts.audit_data import (eval_batch_capacity, evaluation_reuse_status,
                                packed_window_count, split_token_budget,
                                minhash_signature, near_duplicate_leakage,
                                near_duplicate_ledger_additions,
                                near_duplicate_record,
                                near_duplicate_scan_bound, token_shingles,
                                train_wrap_status)
from scripts.data_pipeline import (canonical_document_hash,
                                   configure_data_options, get_data_options,
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


# ---------------------------------------------------------------------------
# Held-out evaluation budget and batch reuse
# ---------------------------------------------------------------------------
def test_packed_window_count_matches_the_packed_stream_geometry():
    # PackedStream needs seq_len + 1 tokens and advances by seq_len, so T
    # tokens yield floor((T - 1) / seq_len) unique windows.
    assert packed_window_count(None, 1024) is None
    assert packed_window_count(0, 1024) == 0
    assert packed_window_count(1024, 1024) == 0
    assert packed_window_count(1025, 1024) == 1
    assert packed_window_count(2048, 1024) == 1
    assert packed_window_count(2049, 1024) == 2
    with pytest.raises(ValueError):
        packed_window_count(10, 0)


def test_eval_batch_capacity_converts_tokens_to_distinct_batches():
    capacity = eval_batch_capacity(
        {"estimated_effective_tokens": 10_000, "basis": "observed"}, 1024, 2)
    assert capacity["packed_windows"] == (10_000 - 1) // 1024
    assert capacity["unique_batches"] == capacity["packed_windows"] // 2
    assert capacity["tokens_per_batch"] == 2048
    assert capacity["approximate"] is False
    approximate = eval_batch_capacity(
        {"estimated_effective_tokens": 10_000,
         "basis": "extrapolated_from_sampled_prefix"}, 1024, 2)
    assert approximate["approximate"] is True
    with pytest.raises(ValueError):
        eval_batch_capacity({"estimated_effective_tokens": 10}, 0, 1)
    with pytest.raises(ValueError):
        eval_batch_capacity({"estimated_effective_tokens": 10}, 1, 0)


def test_validation_pass_reuse_is_reported_with_cycles_and_selection_cost():
    capacity = {
        "unique_batches": 3, "estimated_effective_tokens": 10_000,
        "tokens_per_batch": 2048, "basis": "observed", "approximate": False,
    }
    status = evaluation_reuse_status(
        capacity, 10, split="validation", val_every=100, max_steps=1000)
    assert status["reuses_batches"] is True
    assert status["distinct_batches_per_pass"] == 3
    assert status["cycled_batches"] == 7
    assert status["cycles_per_pass"] == 4
    assert status["distinct_fraction"] == pytest.approx(0.3)
    # 1000 steps / 100 = 10 passes, each asking for 10 batches, over only 3
    # distinct held-out batches.
    assert status["validation_passes"] == 10
    assert status["selection_batch_draws"] == 100
    assert status["selection_reuse_factor"] == pytest.approx(100 / 3)
    assert status["selection_reuses_batches"] is True
    assert status["truncated"] is False
    # A partial final interval gets no validation pass: floor(1050/100) = 10.
    partial = evaluation_reuse_status(
        capacity, 10, split="validation", val_every=100, max_steps=1050)
    assert partial["validation_passes"] == 10


def test_sealed_test_shortfall_is_truncation_not_cycling():
    capacity = {"unique_batches": 3, "estimated_effective_tokens": 10_000,
                "tokens_per_batch": 2048, "basis": "observed"}
    status = evaluation_reuse_status(capacity, 10, split="test")
    assert status["reuses_batches"] is False
    assert status["truncated"] is True
    assert status["shortfall_batches"] == 7
    assert status["distinct_batches_per_pass"] == 3
    assert status["cycles_per_pass"] is None


def test_eval_reuse_status_does_not_invent_a_verdict_when_capacity_is_unknown():
    status = evaluation_reuse_status({"unique_batches": None}, 10,
                                     split="validation", val_every=100,
                                     max_steps=1000)
    assert status["reuses_batches"] is None
    assert status["truncated"] is False
    assert status["distinct_batches_per_pass"] is None
    assert status["selection_reuse_factor"] is None
    assert status["selection_reuses_batches"] is None


def test_eval_reuse_status_rejects_invalid_requests():
    capacity = {"unique_batches": 3}
    for kwargs in ({"requested_batches": 0, "split": "validation"},
                   {"requested_batches": 10, "split": "dev"}):
        with pytest.raises(ValueError):
            evaluation_reuse_status(capacity, **kwargs)
    with pytest.raises(ValueError):
        evaluation_reuse_status(capacity, 10, split="validation", val_every=-1)


def test_evaluate_heldout_reports_cycle_and_shortfall_without_changing_the_mean():
    from contextlib import nullcontext

    import torch

    from scripts.train_gpt2 import evaluate_heldout

    class _Cfg:
        dropout = 0.0
        gradient_checkpointing = False

    class _Model:
        cfg = _Cfg()
        training = False

        def __call__(self, x, y, return_logits=False):
            return None, torch.zeros((), dtype=torch.float32)

    batch = (torch.zeros((1, 2), dtype=torch.long),
             torch.zeros((1, 2), dtype=torch.long))
    loader = [(torch.full((1, 2), i, dtype=torch.long), batch[1])
              for i in range(3)]

    # Validation cycles: 7 requested losses describe only 3 distinct batches.
    state = {"iter": None, "batches_consumed": 0}
    collected: list = []
    mean = evaluate_heldout(_Model(), loader, state, 7, nullcontext(), "cpu",
                            collect=collected)
    assert mean == 0.0 and len(collected) == 7
    assert state["last_eval_cycled"] is True
    assert state["last_eval_reused"] is True
    assert state["last_eval_unique_batches"] == 3
    assert state["last_eval_shortfall"] == 0

    # The sealed pass never cycles: it truncates and records the shortfall.
    state = {"iter": None, "batches_consumed": 0}
    collected = []
    mean = evaluate_heldout(_Model(), loader, state, 7, nullcontext(), "cpu",
                            collect=collected, cycle=False)
    assert mean == 0.0 and len(collected) == 3
    assert state["last_eval_cycled"] is False
    assert state["last_eval_reused"] is False
    assert state["last_eval_unique_batches"] == 3
    assert state["last_eval_shortfall"] == 4


# ---------------------------------------------------------------------------
# Near-duplicate uncertainty and remediation
# ---------------------------------------------------------------------------
def test_near_duplicate_record_carries_the_canonical_digest():
    digest = canonical_document_hash("a document whose wording changes")
    record = near_duplicate_record(list(range(40)), canonical_digest=digest)
    assert record["canonical_digest"] == digest.hex()
    # The digest is optional: a token stream that did not come from a text
    # document still gets a usable sketch.
    assert "canonical_digest" not in near_duplicate_record(list(range(40)))
    with pytest.raises(ValueError):
        near_duplicate_record([1, 2, 3], canonical_digest="not-hex")
    with pytest.raises(ValueError):
        near_duplicate_record([1, 2, 3], canonical_digest=b"")


def _reworded_pair():
    base = list(range(200))
    reworded = base[:95] + [9000, 9001, 9002, 9003, 9004] + base[100:]
    return base, reworded


def test_near_duplicate_leakage_unions_flagged_documents_per_later_split():
    base, reworded = _reworded_pair()
    train = near_duplicate_record(base, canonical_digest=canonical_document_hash("train"))
    train["doc_index"] = 0
    validation = near_duplicate_record(
        reworded, canonical_digest=canonical_document_hash("validation"))
    validation["doc_index"] = 1
    # The same test document is similar to both train and validation. It is
    # one contaminated held-out document, not two, so the per-later-split
    # denominator for the contamination bound must count it once.
    test = near_duplicate_record(
        reworded, canonical_digest=canonical_document_hash("test"))
    test["doc_index"] = 2
    report = near_duplicate_leakage(
        {"train": [train], "validation": [validation], "test": [test]},
        threshold=0.8)
    later = report["by_later_split"]
    assert later["validation"]["flagged_documents"] == 1
    assert later["test"]["flagged_documents"] == 1
    assert later["test"]["flagged_document_indices"] == [2]
    assert later["test"]["flagged_digests"] == [canonical_document_hash("test").hex()]
    # A pair still records its own hits; the union is an additional view.
    assert report["pairs"]["train->test"]["flagged_later_documents"] == 1
    assert report["pairs"]["validation->test"]["flagged_later_documents"] == 1


def test_near_duplicate_scan_bound_quantifies_a_zero_hit_scan():
    report = {
        "enabled": True,
        "exhaustive": True,
        "by_later_split": {
            "validation": {"documents": 4000, "flagged_documents": 0},
        },
        "pairs": {},
    }
    bound = near_duplicate_scan_bound(report)
    row = bound["by_later_split"]["validation"]
    assert row["observed_rate"] == 0.0
    assert 0.0 < row["upper_bound"] < 0.001
    assert row["scope"] == "audited_split"
    assert bound["method"] == "wilson_score_upper_bound"
    assert bound["certifies_population"] is True
    # A clean sample of 4000 documents still cannot rule out a rate above
    # 0.0005 (the Wilson upper bound is roughly 0.00068).
    strict = near_duplicate_scan_bound(report, max_rate=0.0005)
    assert strict["by_later_split"]["validation"]["passes_max_rate"] is False
    assert strict["problems"]
    # It can rule out the much larger rate that a 20-document sample could not.
    loose = near_duplicate_scan_bound(report, max_rate=0.01)
    assert loose["by_later_split"]["validation"]["passes_max_rate"] is True
    assert loose["problems"] == []


def test_near_duplicate_scan_bound_marks_a_sampled_prefix_as_narrower():
    report = {
        "enabled": True,
        "exhaustive": False,
        "by_later_split": {
            "test": {"documents": 100, "flagged_documents": 1},
        },
        "pairs": {},
    }
    bound = near_duplicate_scan_bound(report, max_rate=0.5)
    row = bound["by_later_split"]["test"]
    assert row["scope"] == "scanned_prefix"
    assert bound["certifies_population"] is False
    assert row["observed_rate"] == pytest.approx(0.01)
    assert row["upper_bound"] < 0.5
    assert row["passes_max_rate"] is True
    assert any("scanned prefix only" in note for note in bound["notes"])


def test_near_duplicate_scan_bound_rejects_malformed_inputs():
    good = {"enabled": True, "exhaustive": True,
            "by_later_split": {"validation": {"documents": 10,
                                              "flagged_documents": 1}}}
    with pytest.raises(ValueError):
        near_duplicate_scan_bound(good, confidence=0.0)
    with pytest.raises(ValueError):
        near_duplicate_scan_bound(good, confidence=True)
    with pytest.raises(ValueError):
        near_duplicate_scan_bound(good, max_rate=1.5)
    with pytest.raises(ValueError):
        near_duplicate_scan_bound({"by_later_split": {
            "validation": {"documents": True, "flagged_documents": 0}}})
    with pytest.raises(ValueError):
        near_duplicate_scan_bound({"by_later_split": {
            "validation": {"documents": 2, "flagged_documents": 3}}})
    with pytest.raises(ValueError):
        near_duplicate_scan_bound(None)
    # A max-rate claim on a report with no scan counts is unknown, not clean.
    empty = near_duplicate_scan_bound({"by_later_split": {}}, max_rate=0.1)
    assert empty["problems"]


def test_near_duplicate_ledger_additions_only_include_later_documents():
    def digest(text):
        return canonical_document_hash(text).hex()

    report = {
        "pairs": {
            "train->validation": {
                "earlier_split": "train", "later_split": "validation",
                "flagged_later_digests": [digest("val-of-train")],
                "flagged_later_missing_digest": 0,
            },
            "train->test": {
                "earlier_split": "train", "later_split": "test",
                "flagged_later_digests": [digest("test-of-train")],
                "flagged_later_missing_digest": 0,
            },
            "validation->test": {
                "earlier_split": "validation", "later_split": "test",
                "flagged_later_digests": [digest("test-of-val")],
                "flagged_later_missing_digest": 0,
            },
        },
    }
    additions = near_duplicate_ledger_additions(report)
    assert additions["hashes"]["train"] == {
        canonical_document_hash("val-of-train"),
        canonical_document_hash("test-of-train"),
    }
    assert additions["hashes"]["validation"] == {
        canonical_document_hash("test-of-val"),
    }
    # A training document is never an exclusion candidate.
    assert canonical_document_hash("train-of-val") not in additions["hashes"]["train"]


def test_near_duplicate_ledger_additions_report_missing_digests():
    report = {
        "by_later_split": {
            "validation": {"documents": 10, "flagged_documents": 2,
                           "missing_digest_documents": 2},
        },
        "pairs": {
            "train->validation": {
                "earlier_split": "train", "later_split": "validation",
                "flagged_later_digests": [],
                "flagged_later_missing_digest": 2,
            },
        },
    }
    additions = near_duplicate_ledger_additions(report)
    assert additions["hashes"] == {"train": set(), "validation": set()}
    assert additions["missing_digest_documents"] == 2


# ---------------------------------------------------------------------------
# Paired data identity: the training stream an arm actually consumed
# ---------------------------------------------------------------------------
def _stream_record(*batches, start_step=0, prefix_batches=2):
    from adasquash.repro import TrainStreamFingerprint

    fingerprint = TrainStreamFingerprint(
        start_step=start_step, prefix_batches=prefix_batches)
    for batch in batches:
        fingerprint.update(batch)
    return fingerprint.record()


def test_train_stream_fingerprint_is_order_and_shape_sensitive():
    import torch

    a = _stream_record(torch.tensor([[1, 2, 3], [4, 5, 6]]),
                       torch.tensor([[7, 8, 9], [10, 11, 12]]))
    b = _stream_record(torch.tensor([[1, 2, 3], [4, 5, 6]]),
                       torch.tensor([[7, 8, 9], [10, 11, 12]]))
    assert a == b
    assert a["batches"] == 2 and a["tokens"] == 12
    assert a["prefix_batches"] == 2

    # Same token multiset, different order: the optimizer saw a different
    # sequence, so the paired-design evidence must differ.
    reordered = _stream_record(torch.tensor([[4, 5, 6], [1, 2, 3]]),
                               torch.tensor([[7, 8, 9], [10, 11, 12]]))
    assert reordered["prefix_fingerprint"] != a["prefix_fingerprint"]
    assert reordered["stream_fingerprint"] != a["stream_fingerprint"]

    # Same flattened tokens, different window shape: a different batch
    # boundary is a different training stream too.
    reshaped = _stream_record(torch.tensor([[1, 2, 3, 4, 5, 6]]),
                              torch.tensor([[7, 8, 9], [10, 11, 12]]))
    assert reshaped["prefix_fingerprint"] != a["prefix_fingerprint"]


def test_train_stream_fingerprint_includes_the_target_tokens():
    import torch
    from adasquash.repro import TrainStreamFingerprint

    x = torch.tensor([[1, 2], [3, 4]])
    first = TrainStreamFingerprint()
    first.update(x, torch.tensor([[2, 3], [4, 5]]))
    second = TrainStreamFingerprint()
    # The final label of a packed window is not part of its input; a stream
    # that differs only in that label is still a different training stream.
    second.update(x, torch.tensor([[2, 3], [4, 9]]))
    assert first.record()["prefix_fingerprint"] != second.record()["prefix_fingerprint"]


def test_train_stream_prefix_stays_comparable_when_one_run_stops_earlier():
    import torch

    first = torch.tensor([[1, 2], [3, 4]])
    second = torch.tensor([[5, 6], [7, 8]])
    third = torch.tensor([[9, 10], [11, 12]])
    short = _stream_record(first, second)
    long = _stream_record(first, second, third)
    check = compare_train_streams(short, long)
    assert check["comparable"] is True
    assert check["identical"] is True
    assert check["scope"] == "shared_prefix"
    assert check["full_stream_compared"] is False

    diverged = _stream_record(first, second, torch.tensor([[0, 0], [0, 0]]))
    check = compare_train_streams(long, diverged)
    assert check["comparable"] is True
    assert check["identical"] is False
    assert check["full_stream_compared"] is True
    assert check["problems"]


def test_compare_train_streams_treats_missing_records_as_unknown_not_equal():
    import torch

    record = _stream_record(torch.tensor([[1, 2], [3, 4]]))
    missing = compare_train_streams(None, None)
    assert missing["recorded"] is False
    assert missing["comparable"] is False
    assert missing["identical"] is None
    one_side = compare_train_streams(record, None)
    assert one_side["recorded"] is True
    assert one_side["comparable"] is False
    resumed = compare_train_streams(
        _stream_record(torch.tensor([[1, 2]]), start_step=10),
        _stream_record(torch.tensor([[1, 2]]), start_step=0))
    assert resumed["comparable"] is False
    assert any("different steps" in problem for problem in resumed["problems"])


def test_train_stream_lookup_accepts_json_string_seed_keys():
    from scripts.compare_arms import _train_stream_for_seed

    record = {"prefix_fingerprint": "x", "batches": 1}
    assert _train_stream_for_seed({42: record}, 42) is record
    # `json.dump` turns integer aggregate keys into strings.
    assert _train_stream_for_seed({"42": record}, 42) is record
    assert _train_stream_for_seed({}, 42) is None
    assert _train_stream_for_seed(None, 42) is None


def test_compare_arms_reports_a_train_stream_mismatch_without_certifying():
    from scripts.compare_arms import compare_arms

    import torch

    matching = _stream_record(torch.tensor([[1, 2], [3, 4]]))
    different = _stream_record(torch.tensor([[9, 9], [9, 9]]))

    def aggregate(values, stream):
        return {
            'model': 'gpt2-124m',
            'per_seed': [{'seed': seed, 'final_val_loss': value,
                          'final_val_step': 100}
                         for seed, value in sorted(values.items())],
            'shapes_by_seed': {str(seed): {'n_layers': 12, 'dim': 768,
                                           'n_heads': 12}
                               for seed in values},
            'data_versions_by_seed': {str(seed): {'name': 'corpus',
                                                  'revision': 'v1'}
                                      for seed in values},
            'train_stream_by_seed': {seed: stream for seed in values},
        }

    values = {42: 2.0, 43: 2.0, 44: 2.0}
    clean = {
        'reference': aggregate(values, matching),
        'candidate': aggregate({42: 1.5, 43: 1.52, 44: 1.49}, matching),
    }
    out = compare_arms(clean, reference='reference', require_recipe=False)
    row = out['rows'][0]
    assert not any('train stream' in problem.lower()
                   for problem in row['protocol_problems'])
    assert row['train_stream_identity']['checked'] is True

    dirty = {
        'reference': aggregate(values, matching),
        'candidate': aggregate({42: 1.5, 43: 1.52, 44: 1.49}, different),
    }
    out = compare_arms(dirty, reference='reference', require_recipe=False)
    row = out['rows'][0]
    assert any('train stream differs' in problem
               for problem in row['protocol_problems'])
    assert row['certifiable'] is False

    # A legacy aggregate without a stream record is unknown, not equal. It is
    # tolerated by default but fails closed when the comparison is required to
    # prove data identity.
    legacy = {
        'reference': {**aggregate(values, matching),
                      'train_stream_by_seed': {}},
        'candidate': {**aggregate({42: 1.5, 43: 1.52, 44: 1.49}, matching),
                      'train_stream_by_seed': {}},
    }
    out = compare_arms(legacy, reference='reference', require_recipe=False,
                       require_train_stream_identity=True)
    row = out['rows'][0]
    assert any('identity was not established' in problem
               for problem in row['protocol_problems'])
