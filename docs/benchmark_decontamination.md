# Downstream-benchmark decontamination

`docs/design_notes.md` fixes the model mathematics; `scripts/data_pipeline.py`
and `scripts/audit_data.py` enforce the corpus's internal split protocol. This
note covers a different data-quality threat that neither one addresses:

> The corpus's train/validation/test split can be perfectly disjoint and the
> model can still have memorized an MMLU question, a GSM8K solution or a
> HumanEval prompt from its **training** documents. The downstream benchmark
> is outside the corpus split, so no internal split audit can see that
> overlap.

The project evaluates on downstream benchmarks (`scripts/eval_downstream.py`)
after pretraining on FineWeb-Edu. Public pretraining corpora and public
benchmark sets are both scraped from the web; overlap is expected, not
hypothetical. Treating a downstream score as capability without checking the
overlap would be an unverified claim.

## What is measured

`scripts/audit_benchmark_contamination.py` loads one or more benchmark sets,
tokenizes the evaluated text with the **same tokenizer and revision as the
run**, and streams the training split through the **same split selector and
cleaner as the run**. For every cleaned training document it looks up all
token n-grams (default 13) in an exact inverted index of the benchmark
n-grams.

The unit of a finding is a **training document**, not a benchmark example.
For each benchmark the report gives:

* the number of scanned training documents that overlap it;
* the number of overlapping n-gram occurrences;
* a first-match token position and a short text preview for bounded examples;
* a one-sided Wilson upper bound on the contamination rate over the scanned
  documents.

The canonical digest of every contaminated training document is retained
even when the report's example list is capped, so remediation is complete
for the scanned portion. Digests use the same 16-byte blake2b canonical-text
format as `scripts/data_pipeline.py`; this is tested explicitly.

## A clean scan is a bound, not a proof

Two scopes are reported separately:

* `audited_corpus` - `--n-docs 0` read the whole training split; the bound
  describes that split.
* `scanned_prefix` - a bounded `--n-docs` sample was read; the bound applies
  only to the prefix. It is not extrapolated to the rest of the corpus. The
  prefix is the earliest part of the upstream training split, not a random
  sample of the batches the optimizer consumed; use `--n-docs 0` when the
  claim must cover the whole training split.

At zero hits a sample of 400 documents still permits a contamination rate of
roughly 1% at 95% confidence. `--max-contamination-rate` turns the bound into
a pre-registered verdict: the audit fails when the upper bound exceeds the
tolerated rate, and it also fails when nothing was scanned. Choose the
tolerated rate before looking at the result.

## Remediation, not just reporting

If the scan finds overlap, pass:

```bash
--write-train-exclusion-file runs/ledgers/benchmark_train_exclude.hash
```

The file contains the canonical digests of the contaminated training
documents. Every training arm consumes it with:

```bash
--exclude-train-hash-file runs/ledgers/benchmark_train_exclude.hash
```

`scripts/data_pipeline.py` applies that ledger to the **training split only**
(`dropped_training_exclusion` in the split report). It is deliberately not
applied to validation or test: the purpose is to stop the model fitting
benchmark text, not to alter the evaluation slice. The ledger path and its
content fingerprint are part of the run manifest, and a resume refuses to
swap the ledger under a checkpoint.

Generate the ledger with `--n-docs 0` when a full guarantee is needed; a
prefix-only ledger removes contamination only in the prefix and the audit
says so.

## Reproducible benchmark identity

The audit records:

* the requested benchmark revision and the local file fingerprint;
* a content digest of every benchmark set (order-insensitive; a changed
  question, choice, answer or solution changes it);
* the tokenizer name/revision and the n-gram size;
* the training split boundary and cleaning options;
* a deterministic protocol digest over all of the above.

A result produced against an unpinned benchmark snapshot is still useful as
a measurement, but it cannot be replayed; the report does not pretend
otherwise. Pin `--benchmark-revision` (and, for presets,
`--dataset-revision`/`--tokenizer-revision`) for a reproducible claim.

## Reading the limits of an n-gram scan

* **Short benchmark examples.** A benchmark whose every example is shorter
  than the n-gram size contributes no n-gram and cannot be detected. The
  audit reports those benchmarks as a problem rather than as clean. Lower
  `--ngram-size` for short-answer tasks, and inspect them separately.
* **Paraphrase.** Exact n-grams catch copied spans, not a fully rewritten
  question. A zero count is evidence about copied text, not about
  paraphrase and not about semantic leakage.
* **Bounded sketch.** The index is exact over the loaded examples; nothing is
  approximated by MinHash here. The bounded resource is the scanned training
  prefix (`--n-docs`), not the match decision.
* **Contamination found in held-out corpus splits** is a different question,
  handled by `scripts/audit_data.py` (exact/canonical/group/near-duplicate
  ledgers). Use both audits; neither replaces the other.

## Pre-run checklist

1. Pin the tokenizer revision and the dataset revision used for training.
2. Pin each benchmark revision and record its content digest.
3. Run the audit with the same `--val-frac`, `--test-frac`, cleaning flags
   and `--n-docs` policy the training arms use.
4. Pre-register `--max-contamination-rate`.
5. If overlap is found, write the training-exclusion ledger and pass it to
   **every** arm with the same path, so the data protocol remains identical
   across arms.
6. Keep `benchmark_contamination.json` and its manifest next to the run
   outputs; the downstream score is reported against the audit that preceded
   it.
