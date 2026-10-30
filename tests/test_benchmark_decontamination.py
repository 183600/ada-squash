"""Data-processing checks for downstream-benchmark decontamination.

The corpus's own split audit cannot see a benchmark example that sits on the
*train* side of the split: that text is not leakage between train and
validation, it is a pretraining input whose downstream evaluation will later
reward memorization.  These tests pin the primitive used to find and
remediate that contamination:

* exact token n-gram matching is deterministic, order-sensitive and reports
  which benchmark a document overlaps;
* a clean scan is reported as a one-sided bound with a scope, not as proof of
  a clean corpus;
* a finding can be converted to the same canonical digest the training
  pipeline's exclusion ledger consumes, and the pipeline drops that document
  from train (and only from train).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.decontamination import (
    NgramIndex,
    benchmark_fingerprint,
    benchmark_text,
    build_ngram_index,
    canonical_document_hash,
    canonical_ledger_text,
    contamination_ledger_hashes,
    contamination_protocol,
    contamination_scan_bound,
    find_ngram_matches,
    normalize_match_text,
    protocol_digest,
    scan_token_documents,
    token_ngrams,
    wilson_upper_bound,
)
from scripts.data_pipeline import (
    DocumentCleaner,
    canonical_document_hash as pipeline_canonical_document_hash,
    document_hash,
    load_hash_file,
    validate_data_options,
)


# ---------------------------------------------------------------------------
# Text extraction and fingerprints
# ---------------------------------------------------------------------------
def test_benchmark_text_combines_question_choices_and_answer():
    record = {
        "question": "What is 2 + 2?",
        "choices": ["3", "4", "5"],
        "answer": "4",
        "id": "not-evaluated-metadata",
    }
    text = benchmark_text(record)
    assert "What is 2 + 2?" in text
    assert all(choice in text for choice in ("3", "4", "5"))
    # Metadata is deliberately excluded: a shared id/licence string is not
    # contamination of the evaluated content.
    assert "not-evaluated-metadata" not in text
    # A prebuilt text field wins over field inference.
    assert benchmark_text({"text": "already built", "question": "ignored"}) == "already built"


def test_benchmark_text_rejects_records_without_evaluated_content():
    with pytest.raises(ValueError):
        benchmark_text({"id": "only-metadata"})
    with pytest.raises(ValueError):
        benchmark_text("not a mapping")


def test_benchmark_fingerprint_is_order_insensitive_but_content_sensitive():
    first = [{"question": "q1", "answer": "a1"},
             {"question": "q2", "answer": "a2"}]
    second = list(reversed(first))
    a = benchmark_fingerprint({"mmlu": first, "gsm8k": [{"text": "solve it"}]})
    b = benchmark_fingerprint({"gsm8k": [{"text": "solve it"}], "mmlu": second})
    assert a["mmlu"]["content_sha256"] == b["mmlu"]["content_sha256"]
    assert a["combined_sha256"] == b["combined_sha256"]
    changed = benchmark_fingerprint(
        {"mmlu": first[:-1] + [{"question": "q2", "answer": "DIFFERENT"}]})
    assert changed["mmlu"]["content_sha256"] != a["mmlu"]["content_sha256"]


def test_normalization_is_case_and_whitespace_insensitive():
    assert normalize_match_text("  A\t B\nC ") == normalize_match_text("a b c")
    assert canonical_ledger_text("A  B") == "A B"


def test_ledger_digest_matches_the_training_pipeline_exactly():
    text = "  A benchmark question with  odd\twhitespace.  "
    assert canonical_document_hash(text) == pipeline_canonical_document_hash(text)
    assert len(canonical_document_hash(text)) == 16


# ---------------------------------------------------------------------------
# N-gram index
# ---------------------------------------------------------------------------
def test_token_ngrams_are_exact_and_handle_short_documents():
    assert token_ngrams([1, 2, 3, 4], 3) == {(1, 2, 3), (2, 3, 4)}
    assert token_ngrams([1, 2], 3) == set()
    assert token_ngrams([], 1) == set()
    with pytest.raises(ValueError):
        token_ngrams([1, 2], 0)
    with pytest.raises(ValueError):
        token_ngrams("not-token-ids", 2)


def test_ngram_index_reports_matches_and_positions():
    base = list(range(40))
    index = build_ngram_index({"bench": [base]}, ngram_size=8)
    assert isinstance(index, NgramIndex)
    assert index.summary()["benchmarks"]["bench"]["documents"] == 1
    # A short example contributes no n-gram; report it instead of letting the
    # scan look complete.
    short_index = build_ngram_index({"short": [[1, 2], list(range(20))]},
                                    ngram_size=8)
    assert short_index.summary()["benchmarks"]["short"]["documents_without_ngrams"] == 1
    copied = [999] + base[5:25] + [888]
    match = find_ngram_matches(copied, index)
    assert match["matched"] is True
    assert match["benchmarks"] == ["bench"]
    assert match["first_position"] == 1
    # Tokens 5..19 of the benchmark appear at positions 1..15.
    assert match["by_benchmark"]["bench"]["matched_ngrams"] == 13


def test_ngram_index_does_not_flag_unrelated_tokens():
    index = build_ngram_index({"bench": [list(range(100))]}, ngram_size=8)
    match = find_ngram_matches(list(range(1000, 1100)), index)
    assert match["matched"] is False
    assert match["benchmarks"] == []


def test_ngram_index_can_attribute_a_document_to_multiple_benchmarks():
    shared = list(range(30))
    index = build_ngram_index({
        "a": [shared],
        "b": [list(range(100, 130)) + shared],
    }, ngram_size=8)
    match = find_ngram_matches(shared, index)
    assert match["benchmarks"] == ["a", "b"]


def test_ngram_match_extent_distinguishes_a_copy_from_an_incidental_hit():
    base = list(range(100))
    index = build_ngram_index({"bench": [base]}, ngram_size=8)

    copied = find_ngram_matches(base, index)
    assert copied["longest_match_tokens"] == 100
    assert copied["matched_token_coverage"] == pytest.approx(1.0)
    assert copied["by_benchmark"]["bench"]["benchmark_ngram_coverage"] == pytest.approx(1.0)

    excerpt = find_ngram_matches([999] + base[10:30] + [888], index)
    assert excerpt["longest_match_tokens"] == 20
    assert excerpt["matched_token_coverage"] == pytest.approx(20 / 22)
    assert excerpt["by_benchmark"]["bench"]["benchmark_ngram_coverage"] == pytest.approx(
        13 / 93)

    # A single shared 8-gram in a 1000-token document is a much weaker signal
    # than a copied 100-token document, even though both are "matched".
    incidental = find_ngram_matches(list(range(500, 600)) + base[20:28]
                                    + list(range(700, 1100)), index)
    assert incidental["longest_match_tokens"] == 8
    assert incidental["matched_token_coverage"] < 0.02


def test_ngram_match_coverage_unions_overlapping_token_spans():
    index = build_ngram_index({
        "bench": [list(range(8)), list(range(2, 10))],
    }, ngram_size=8)
    match = find_ngram_matches(list(range(10)), index)
    # The matching windows start at 0 and 2: their token spans overlap even
    # though the window at start 1 does not match. Coverage must count the
    # union (10 tokens), not the sum of the two 8-token spans.
    assert match["longest_match_tokens"] == 8
    assert match["matched_token_coverage"] == pytest.approx(1.0)
    assert match["by_benchmark"]["bench"]["matched_token_coverage"] == pytest.approx(1.0)
    assert match["by_benchmark"]["bench"]["occurrences"] == 2


def test_scan_extent_rule_keeps_below_threshold_hits_visible():
    base = list(range(100))
    index = build_ngram_index({"bench": [base]}, ngram_size=8)
    document = {"tokens": base[10:50], "text": "excerpt"}
    strict = scan_token_documents(
        [document], index, min_match_tokens=41,
        min_match_fraction=0.0)
    assert strict["contaminated_documents"] == 0
    assert strict["below_threshold_documents"] == 1
    assert strict["contaminated_hashes"] == []
    assert strict["by_benchmark"]["bench"]["below_threshold_documents"] == 1
    assert strict["by_benchmark"]["bench"]["max_longest_match_tokens"] == 40
    assert strict["by_benchmark"]["bench"][
        "max_benchmark_ngram_coverage"] == pytest.approx(33 / 93)
    assert strict["examples"] == []

    looser = scan_token_documents(
        [document], index, min_match_tokens=40,
        min_match_fraction=0.0)
    assert looser["contaminated_documents"] == 1
    assert looser["below_threshold_documents"] == 0
    assert looser["examples"][0]["longest_match_tokens"] == 40
    assert looser["examples"][0]["matched_token_coverage"] == pytest.approx(1.0)


def test_scan_rejects_malformed_extent_rule():
    index = build_ngram_index({"bench": [list(range(20))]}, ngram_size=8)
    for kwargs in ({"min_match_tokens": -1},
                   {"min_match_tokens": True},
                   {"min_match_fraction": -0.1},
                   {"min_match_fraction": 1.5},
                   {"min_match_fraction": True}):
        with pytest.raises(ValueError):
            scan_token_documents([], index, **kwargs)


def test_text_normalization_is_applied_to_corpus_and_benchmark_alike():
    from types import SimpleNamespace

    import scripts.audit_benchmark_contamination as audit
    from scripts.data_pipeline import DocumentCleaner

    class Tokenizer:
        def __call__(self, texts, add_special_tokens=False):
            single = isinstance(texts, str)
            rows = [[ord(char) % 97 for char in text if not char.isspace()]
                    for text in ([texts] if single else list(texts))]
            return {"input_ids": rows[0] if single else rows}

    # The benchmark text is upper-case while the training document is
    # lower-case with different spacing.  Casefold+whitespace normalization
    # must make them the same token stream.
    benchmark_tokens = [[ord(char) % 97 for char in "abcdefgh"]]
    index = build_ngram_index({"bench": benchmark_tokens}, ngram_size=4)
    stream = [{"text": "xx   ABCDEFGH   yy"}]

    def fake_iter(_args):
        return iter(stream), {"split": "train"}, DocumentCleaner()

    original = audit._iter_training_documents
    audit._iter_training_documents = fake_iter
    try:
        args = SimpleNamespace(
            n_docs=0, tokenize_batch_size=4, max_examples=10,
            val_frac=0.005, test_frac=0.005, num_documents=None, dedup=True,
            min_doc_chars=0, max_doc_chars=0, dataset="fake", dataset_config=None,
            dataset_revision=None, tokenizer_revision=None,
            text_normalization="nfkc_casefold", min_match_tokens=0,
            min_match_fraction=0.0,
        )
        scan, _ = audit._scan_training(args, Tokenizer(), index)
    finally:
        audit._iter_training_documents = original
    assert scan["documents_scanned"] == 1
    assert scan["contaminated_documents"] == 1
    assert scan["contaminated_hashes"] == [
        canonical_document_hash("xx   ABCDEFGH   yy").hex()]
    # The raw protocol would tokenize two different strings and miss it.
    args.text_normalization = "raw"
    audit._iter_training_documents = fake_iter
    try:
        raw_scan, _ = audit._scan_training(args, Tokenizer(), index)
    finally:
        audit._iter_training_documents = original
    assert raw_scan["contaminated_documents"] == 0


def test_contamination_protocol_digest_covers_normalization_and_extent_rule():
    records = {"bench": [{"text": "a benchmark question"}]}
    base = contamination_protocol(
        ngram_size=13, benchmarks=records, tokenizer="gpt2")
    raw = contamination_protocol(
        ngram_size=13, benchmarks=records, tokenizer="gpt2",
        normalization="raw")
    strict = contamination_protocol(
        ngram_size=13, benchmarks=records, tokenizer="gpt2",
        min_match_tokens=26, min_match_fraction=0.5)
    assert protocol_digest(base) != protocol_digest(raw)
    assert protocol_digest(base) != protocol_digest(strict)
    assert strict["extent_rule"] == {
        "min_match_tokens": 26, "min_match_fraction": 0.5}


def test_ngram_index_rejects_empty_or_malformed_benchmarks():
    with pytest.raises(ValueError):
        build_ngram_index({})
    with pytest.raises(ValueError):
        build_ngram_index({"empty": []})
    with pytest.raises(ValueError):
        build_ngram_index({"bad": [[1, 2, True]]})
    with pytest.raises(ValueError):
        build_ngram_index({"bad": ["a string, not ids"]})


# ---------------------------------------------------------------------------
# Scan and uncertainty
# ---------------------------------------------------------------------------
def test_scan_counts_contaminated_documents_and_keeps_hashes_complete():
    base = list(range(100))
    index = build_ngram_index({"bench": [base]}, ngram_size=8)
    clean = {"tokens": list(range(500, 600)), "text": "clean document"}
    contaminated = {"tokens": base[10:50], "text": "a contaminated document"}
    report = scan_token_documents([clean, contaminated, contaminated],
                                  index, max_examples=1)
    assert report["documents_scanned"] == 3
    assert report["contaminated_documents"] == 2
    assert report["any"]["observed_rate"] == pytest.approx(2 / 3)
    assert report["by_benchmark"]["bench"]["contaminated_documents"] == 2
    # The examples list is capped, but remediation hashes are not.
    assert len(report["examples"]) == 1
    assert report["examples_truncated"] is True
    assert report["contaminated_hashes"] == sorted(
        [canonical_document_hash("a contaminated document").hex()] * 1)
    assert len(contamination_ledger_hashes(report)) == 1


def test_scan_reports_a_prefix_as_a_lower_bound():
    index = build_ngram_index({"bench": [list(range(50))]}, ngram_size=8)
    report = scan_token_documents(
        [{"tokens": list(range(100, 150))}], index, exhaustive=False)
    assert report["exhaustive"] is False
    assert report["lower_bound"] is True
    bound = contamination_scan_bound(report)
    assert bound["scope"] == "scanned_prefix"
    assert bound["certifies_population"] is False
    assert any("prefix" in note for note in bound["notes"])


def test_scan_bound_quantifies_a_zero_hit_sample():
    index = build_ngram_index({"bench": [list(range(50))]}, ngram_size=8)
    report = scan_token_documents(
        [{"tokens": list(range(1000 + i, 1100 + i))} for i in range(400)],
        index)
    bound = contamination_scan_bound(report, max_rate=0.01)
    assert bound["contaminated_documents"] == 0
    assert 0.0 < bound["upper_bound"] < 0.01
    # A clean sample of 400 is compatible with a rate below 1%, but a
    # stricter claim is not certified.
    assert bound["passes_max_rate"] is True
    assert contamination_scan_bound(report, max_rate=0.001)["passes_max_rate"] is False


def test_scan_bound_can_make_per_benchmark_bounds_simultaneous():
    index = build_ngram_index({
        "a": [list(range(50))],
        "b": [list(range(200, 250))],
    }, ngram_size=8)
    report = scan_token_documents(
        [{"tokens": list(range(1000 + i, 1100 + i))} for i in range(400)],
        index)
    marginal = contamination_scan_bound(report)
    simultaneous = contamination_scan_bound(
        report, familywise_benchmark_bounds=True)
    assert marginal["multiplicity"]["method"] == "none"
    assert simultaneous["multiplicity"]["method"] == "bonferroni"
    assert simultaneous["multiplicity"]["benchmarks"] == 2
    assert simultaneous["multiplicity"]["per_benchmark_confidence"] == pytest.approx(0.975)
    for name in ("a", "b"):
        assert (simultaneous["by_benchmark"][name]["upper_bound"]
                == wilson_upper_bound(0, 400, 0.975))
        assert (simultaneous["by_benchmark"][name]["upper_bound"]
                > marginal["by_benchmark"][name]["upper_bound"])
    # The any-benchmark bound stays at the requested confidence.
    assert simultaneous["upper_bound"] == marginal["upper_bound"]
    with pytest.raises(ValueError):
        contamination_scan_bound(report, familywise_benchmark_bounds="yes")


def test_scan_bound_marks_an_empty_scan_as_unknown_not_clean():
    index = build_ngram_index({"bench": [list(range(50))]}, ngram_size=8)
    report = scan_token_documents([], index)
    bound = contamination_scan_bound(report, max_rate=0.01)
    assert bound["upper_bound"] is None
    assert bound["passes_max_rate"] is False
    assert bound["problems"]


def test_scan_bound_rejects_malformed_reports_and_rates():
    with pytest.raises(ValueError):
        contamination_scan_bound(None)
    with pytest.raises(ValueError):
        contamination_scan_bound({"documents_scanned": 10,
                                  "contaminated_documents": 11})
    with pytest.raises(ValueError):
        contamination_scan_bound({"documents_scanned": 10,
                                  "contaminated_documents": 1}, confidence=1.0)
    with pytest.raises(ValueError):
        contamination_scan_bound({"documents_scanned": 10,
                                  "contaminated_documents": 1}, max_rate=1.5)


def test_wilson_bound_is_none_for_an_empty_sample_and_monotone_in_k():
    assert wilson_upper_bound(0, 0) is None
    assert wilson_upper_bound(0, 100) < wilson_upper_bound(1, 100) < wilson_upper_bound(10, 100)
    with pytest.raises(ValueError):
        wilson_upper_bound(2, 1)


# ---------------------------------------------------------------------------
# Protocol identity
# ---------------------------------------------------------------------------
def test_protocol_digest_is_stable_and_covers_ngram_geometry():
    records = {"bench": [{"text": "a benchmark question"}]}
    first = contamination_protocol(
        ngram_size=13, benchmarks=records, tokenizer="gpt2",
        tokenizer_revision="a" * 40, dataset="corpus", dataset_revision="b" * 40)
    second = contamination_protocol(
        ngram_size=13, benchmarks=records, tokenizer="gpt2",
        tokenizer_revision="a" * 40, dataset="corpus", dataset_revision="b" * 40)
    assert protocol_digest(first) == protocol_digest(second)
    changed = contamination_protocol(
        ngram_size=8, benchmarks=records, tokenizer="gpt2",
        tokenizer_revision="a" * 40, dataset="corpus", dataset_revision="b" * 40)
    assert protocol_digest(changed) != protocol_digest(first)
    assert first["benchmarks"]["bench"]["documents"] == 1


# ---------------------------------------------------------------------------
# Training-side remediation
# ---------------------------------------------------------------------------
def test_document_cleaner_excludes_a_benchmark_contaminated_training_document():
    contaminated = "What is the capital of France? Paris."
    ledger = {canonical_document_hash(contaminated)}
    cleaner = DocumentCleaner(exclude_train_hashes=ledger)
    assert cleaner({"text": contaminated}) is None
    assert cleaner.report()["dropped_training_exclusion"] == 1
    # Exact and canonical formatting variants are both matched.
    assert cleaner({"text": "  What is the capital of France?   Paris.  "}) is None
    assert cleaner({"text": "An unrelated training document."}) is not None
    assert cleaner.report()["documents_kept"] == 1


def test_training_exclusion_does_not_consume_the_heldout_ledger_role():
    text = "same text"
    cleaner = DocumentCleaner(
        exclude_hashes={pipeline_canonical_document_hash(text)},
        exclude_train_hashes=set())
    assert cleaner({"text": text}) is None
    assert cleaner.report()["dropped_cross_split_duplicate"] == 1
    assert cleaner.report().get("dropped_training_exclusion") is None


def test_validate_data_options_accepts_a_training_exclusion_ledger(tmp_path):
    ledger = tmp_path / "benchmark_train.hash"
    ledger.write_text(document_hash("x").hex() + "\n", encoding="utf-8")
    opts = validate_data_options({
        "split": "train",
        "exclude_train_hash_file": ledger,
    })
    assert opts["exclude_train_hash_file"] == ledger
    with pytest.raises(ValueError):
        validate_data_options({"exclude_train_hash_file": True})


def test_ledger_writer_output_is_loadable_by_the_pipeline(tmp_path):
    from scripts.audit_benchmark_contamination import _write_ledger

    path = tmp_path / "exclude.hash"
    hashes = {canonical_document_hash("one"), canonical_document_hash("two")}
    info = _write_ledger(path, hashes, exhaustive=True, n_docs=0)
    assert info["n_hashes"] == 2
    assert load_hash_file(path) == hashes


# ---------------------------------------------------------------------------
# Benchmark file loading
# ---------------------------------------------------------------------------
def test_benchmark_file_loader_accepts_jsonl_and_json_lists(tmp_path):
    from scripts.audit_benchmark_contamination import _load_benchmark_file

    jsonl = tmp_path / "bench.jsonl"
    jsonl.write_text(
        json.dumps({"question": "q1", "answer": "a1"}) + "\n"
        + json.dumps({"question": "q2", "answer": "a2"}) + "\n",
        encoding="utf-8")
    assert len(_load_benchmark_file(jsonl)) == 2

    array = tmp_path / "bench.json"
    array.write_text(json.dumps([{"text": "one"}, {"text": "two"}]), encoding="utf-8")
    assert len(_load_benchmark_file(array)) == 2

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_benchmark_file(empty)


def test_benchmark_spec_parser_distinguishes_presets_and_files():
    from scripts.audit_benchmark_contamination import _parse_benchmark_spec

    assert _parse_benchmark_spec("mmlu") == ("mmlu", None)
    assert _parse_benchmark_spec("mybench=data/bench.jsonl") == (
        "mybench", "data/bench.jsonl")
    with pytest.raises(ValueError):
        _parse_benchmark_spec("not-a-preset")


def test_scan_training_streams_aligned_documents_and_honours_n_docs(monkeypatch):
    """The CLI's streaming path tokenizes batches but reports aligned docs."""
    from types import SimpleNamespace

    import scripts.audit_benchmark_contamination as audit
    from scripts.data_pipeline import DocumentCleaner

    class Tokenizer:
        def __call__(self, texts, add_special_tokens=False):
            if isinstance(texts, str):
                texts = [texts]
            rows = []
            for text in texts:
                # A tiny deterministic vocabulary: one id per character code.
                rows.append([ord(char) % 97 for char in text if not char.isspace()])
            return {"input_ids": rows}

    benchmark_tokens = [[ord(char) % 97 for char in "12345678"]]
    index = build_ngram_index({"bench": benchmark_tokens}, ngram_size=4)
    stream = [
        {"text": "xx 12345678 yy"},   # contaminated
        {"text": "zz 87654321 ww"},   # clean
        {"text": "12345678 again"},   # contaminated but beyond --n-docs
    ]

    def fake_iter(_args):
        return iter(stream), {"split": "train"}, DocumentCleaner()

    monkeypatch.setattr(audit, "_iter_training_documents", fake_iter)
    args = SimpleNamespace(
        n_docs=2, tokenize_batch_size=1, max_examples=10,
        val_frac=0.005, test_frac=0.005, num_documents=None, dedup=True,
        min_doc_chars=0, max_doc_chars=0, dataset="fake", dataset_config=None,
        dataset_revision=None, tokenizer_revision=None,
    )
    scan, split_report = audit._scan_training(args, Tokenizer(), index)
    assert split_report["split"] == "train"
    assert scan["documents_scanned"] == 2
    assert scan["documents_read"] == 2
    assert scan["contaminated_documents"] == 1
    assert scan["contaminated_hashes"] == [
        canonical_document_hash("xx 12345678 yy").hex()]
    # The contaminated document's text is retained for the report preview,
    # while the clean second document was still tokenized.
    assert scan["examples"][0]["text_preview"] == "xx 12345678 yy"


def test_audit_cli_end_to_end_writes_a_consumable_training_ledger(tmp_path, monkeypatch):
    """The documented flow: scan -> ledger -> pipeline-loadable exclusion file.

    `transformers` and `datasets` are stubbed so the CPU-only CI environment
    can exercise the CLI's orchestration (loading, scanning, reporting,
    ledger writing) without downloading a corpus or a model.
    """
    import types

    import scripts.audit_benchmark_contamination as audit

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, name, revision=None):
            return cls()

        def __call__(self, texts, add_special_tokens=False):
            single = isinstance(texts, str)
            rows = [[ord(char) % 97 for char in text if not char.isspace()]
                    for text in ([texts] if single else list(texts))]
            return {"input_ids": rows[0] if single else rows}

    class Stream:
        def __init__(self, documents):
            self.documents = list(documents)

        def skip(self, n):
            return Stream(self.documents[n:])

        def take(self, n):
            return Stream(self.documents[:n])

        def __iter__(self):
            return iter(self.documents)

    documents = [
        {"text": "xx 12345678 yy"},
        {"text": "a clean validation document"},
        {"text": "another clean document"},
    ]
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = Tokenizer
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda *args, **kwargs: Stream(documents)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(json.dumps({"text": "12345678"}) + "\n", encoding="utf-8")
    output = tmp_path / "audit.json"
    ledger = tmp_path / "exclude.hash"
    code = audit.main([
        "--benchmark", f"demo={benchmark}",
        "--dataset", "fake", "--dataset-revision", "a" * 40,
        "--num-documents", "3", "--val-frac", "0.1", "--test-frac", "0.1",
        "--n-docs", "0", "--ngram-size", "4", "--tokenizer", "fake",
        "--write-train-exclusion-file", str(ledger),
        "--output", str(output),
    ])
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "finished"
    assert report["scan"]["documents_scanned"] == 1
    assert report["scan"]["contaminated_documents"] == 1
    assert report["scan"]["contaminated_hashes"] == [
        canonical_document_hash("xx 12345678 yy").hex()]
    assert report["train_exclusion_ledger"]["n_hashes"] == 1
    loaded = load_hash_file(ledger)
    assert loaded == {canonical_document_hash("xx 12345678 yy")}
    manifest = json.loads((tmp_path / "benchmark_contamination_manifest.json").read_text())
    assert manifest["status"] == "finished"
    # A pre-registered zero-tolerance gate must fail the same audit.
    assert audit.main([
        "--benchmark", f"demo={benchmark}",
        "--dataset", "fake", "--dataset-revision", "a" * 40,
        "--num-documents", "3", "--val-frac", "0.1", "--test-frac", "0.1",
        "--n-docs", "0", "--ngram-size", "4", "--tokenizer", "fake",
        "--fail-on-any-contamination", "--output", str(tmp_path / "failed.json"),
    ]) == 1


def test_training_manifest_ledger_identity_includes_the_train_exclusion(tmp_path):
    """The new ledger is part of the run identity, not an untracked side input."""
    from types import SimpleNamespace

    from scripts.train_gpt2 import _data_ledger_fingerprints

    ledger = tmp_path / "benchmark_train.hash"
    ledger.write_text(canonical_document_hash("x").hex() + "\n", encoding="utf-8")
    args = SimpleNamespace(
        exclude_hash_file=None, exclude_test_hash_file=None,
        exclude_train_hash_file=ledger, exclude_group_file=None,
        exclude_test_group_file=None)
    identity = _data_ledger_fingerprints(args)
    assert identity["exclude_train_hash_file"]["exists"] is True
    assert len(identity["exclude_train_hash_file"]["sha256"]) == 64
    assert identity["exclude_hash_file"] is None

