"""Tests for fine-grained distribution-shift analysis."""
# ruff: noqa: PLR2004

from __future__ import annotations

from collections import Counter

import pytest

from adasquash.data_shift_fine import (
    document_length_buckets,
    document_length_tvd,
    fine_grained_shift_report,
    token_frequency_buckets,
    token_frequency_tvd,
)


def test_token_frequency_buckets_equal_frequency():
    counter = Counter({1: 100, 2: 50, 3: 25, 4: 25, 5: 1})
    buckets = token_frequency_buckets(counter, n_buckets=2)
    assert len(buckets) == 2
    assert 1 in buckets[0]
    assert 5 in buckets[1]
    # Every token appears in exactly one bucket.
    all_tokens = set().union(*buckets)
    assert all_tokens == {1, 2, 3, 4, 5}


def test_token_frequency_buckets_single_bucket():
    counter = Counter({1: 10, 2: 20})
    buckets = token_frequency_buckets(counter, n_buckets=1)
    assert len(buckets) == 1
    assert buckets[0] == {1, 2}


def test_token_frequency_buckets_rejects_bad_n():
    counter = Counter({1: 1})
    with pytest.raises(ValueError):
        token_frequency_buckets(counter, n_buckets=0)
    with pytest.raises(TypeError):
        token_frequency_buckets(counter, n_buckets="nope")


def test_token_frequency_tvd_identical_is_zero():
    counter = Counter({1: 10, 2: 20, 3: 30})
    report = token_frequency_tvd(counter, counter, n_buckets=2)
    assert report["overall_tvd"] == 0.0
    for bucket in report["per_bucket"]:
        assert bucket["tvd"] == 0.0


def test_token_frequency_tvd_disjoint_is_one():
    train = Counter({1: 100})
    heldout = Counter({2: 100})
    report = token_frequency_tvd(train, heldout, n_buckets=2)
    assert report["overall_tvd"] == 1.0


def test_token_frequency_tvd_per_bucket_mass():
    train = Counter({1: 100, 2: 1, 3: 1, 4: 1})
    heldout = Counter({1: 1, 2: 100, 3: 1, 4: 1})
    report = token_frequency_tvd(train, heldout, n_buckets=2)
    assert len(report["per_bucket"]) == 2
    assert report["n_buckets"] == 2


def test_document_length_buckets_equal_count():
    lengths = [1, 2, 3, 4, 5, 6, 7, 8]
    buckets = document_length_buckets(lengths, n_buckets=2)
    assert len(buckets) == 2
    assert buckets[0] == (1, 4)
    assert buckets[1] == (5, 8)


def test_document_length_buckets_fewer_than_n():
    lengths = [1, 2, 3]
    buckets = document_length_buckets(lengths, n_buckets=10)
    assert len(buckets) == 3


def test_document_length_buckets_empty():
    assert document_length_buckets([], n_buckets=4) == []


def test_document_length_tvd_identical_is_zero():
    lengths = [10, 20, 30, 40, 50]
    report = document_length_tvd(lengths, lengths, n_buckets=2)
    assert report["overall_tvd"] == 0.0
    for bucket in report["per_bucket"]:
        assert bucket["tvd"] == 0.0


def test_document_length_tvd_divergent():
    train = [10, 10, 10, 10]
    heldout = [100, 100, 100, 100]
    report = document_length_tvd(train, heldout, n_buckets=2)
    assert report["overall_tvd"] == 1.0


def test_document_length_tvd_rejects_empty_train():
    with pytest.raises(ValueError):
        document_length_tvd([], [1, 2, 3], n_buckets=2)


def test_fine_grained_shift_report_structure():
    train_counter = Counter({1: 100, 2: 50, 3: 25})
    heldout_counter = Counter({1: 100, 2: 50, 3: 25})
    train_lengths = [10, 20, 30, 40, 50]
    heldout_lengths = [10, 20, 30, 40, 50]
    report = fine_grained_shift_report(
        train_counter,
        heldout_counter,
        train_lengths,
        heldout_lengths,
        n_buckets=2,
        top_k=3,
    )
    assert report["schema"] == "adasquash.fine_grained_shift.v1"
    assert report["n_buckets"] == 2
    assert report["top_k"] == 3
    assert report["overall_token_tvd"] == 0.0
    assert report["overall_length_tvd"] == 0.0
    assert len(report["token_frequency_buckets"]) == 2
    assert len(report["document_length_buckets"]) == 2
    assert len(report["top_shifted_tokens"]) <= 3
    assert report["train_total_tokens"] == 175
    assert report["heldout_total_tokens"] == 175
    assert report["n_train_documents"] == 5
    assert report["n_heldout_documents"] == 5


def test_fine_grained_shift_report_top_shifted():
    train_counter = Counter({1: 100, 2: 1, 3: 1})
    heldout_counter = Counter({1: 1, 2: 100, 3: 1})
    report = fine_grained_shift_report(
        train_counter,
        heldout_counter,
        [10, 20, 30],
        [10, 20, 30],
        n_buckets=2,
        top_k=2,
    )
    top = report["top_shifted_tokens"]
    assert len(top) == 2
    # Token 1 shifts from train-dominant to heldout-rare.
    token_ids = {entry["token_id"] for entry in top}
    assert token_ids == {1, 2}
    for entry in top:
        assert "abs_shift" in entry
        assert "train_probability" in entry
        assert "heldout_probability" in entry


def test_fine_grained_shift_report_rejects_bad_args():
    with pytest.raises(ValueError):
        fine_grained_shift_report(
            Counter({1: 1}),
            Counter({1: 1}),
            [1],
            [1],
            n_buckets=0,
            top_k=1,
        )
    with pytest.raises(ValueError):
        fine_grained_shift_report(
            Counter({1: 1}),
            Counter({1: 1}),
            [1],
            [1],
            n_buckets=2,
            top_k=0,
        )
