# Data-processing methodology (companion to `design_notes.md`)

`docs/design_notes.md` fixes the model mathematics. This note records the
data-side protocol that makes the normalization comparisons interpretable.
It does not change D1–D6 and it is deliberately narrower than a generic
"data cleaning" checklist: every rule below exists because a run without it
would answer a different question than the one it claims to answer.

## 1. What the data protocol must guarantee

A comparison of two normalization layers is only a comparison of those
layers if every arm:

1. reads the same **corpus snapshot** (dataset and tokenizer commit), not a
   moving branch;
2. trains on the same chronological **past** and is scored on the same
   **future** documents, with no document or group on two sides;
3. applies the same **cleaning** policy, in the same order, after the split;
4. sees held-out documents that are not exact, formatting-variant, grouped,
   or near-duplicate copies of a training document; and
5. evaluates on a held-out set large enough that the reported loss is not
   just a confidence interval over a handful of reused batches.

Items 1–3 and 5 are enforced by `scripts/data_pipeline.py` and audited by
`scripts/audit_data.py`. Item 4 has two layers: exact/canonical/group
ledgers, which the trainers apply, and the token-shingle near-duplicate
scan, which historically only *reported* contamination. Section 3 closes
that detect-to-remediate gap.

## 2. Split, clean, deduplicate — in that order

The stream is split **before** any text is touched. `train` is the oldest
contiguous document range, then `validation`, then `test`. The held-out
splits are never shuffled. A bounded shuffle may reorder packed *training*
batches only.

Cleaning is then applied identically to every split (missing/non-string
text, empty documents, length fences, tokenizer failures). Within-split
dedup runs on the exact **and** canonical (Unicode NFKC + collapsed
whitespace) text. Cross-split ledgers are applied *before* the local dedup
set, because a ledger drop must not make the next copy of a document look
unique.

Two consequences matter for the statistics:

* the effective held-out set is a **non-random subset** of the chronological
  future, because documents duplicated in train are removed from the
  held-out side; the audit reports the distribution shift before and after
  the ledgers (`*_tvd_effective`), and the token budget of the effective
  split, so the number is not read as an estimate over the raw future; and
* a partial ledger is worse than no ledger if it is mistaken for a
  guarantee. The audit therefore marks a prefix-only exact/canonical/group
  ledger `exhaustive=false` and fails the audit rather than blessing it. A
  near-duplicate ledger written from a prefix is reported separately under
  `near_duplicate_coverage` (section 3.2), because it is partial remediation
  rather than a claim of complete coverage.

## 3. Near-duplicates: quantify the scan, then remediate

Exact/canonical hashes cannot see a held-out document that repeats most of a
training document with a changed sentence, header or template. The audit
therefore compares token 8-shingles with a deterministic MinHash + LSH
sketch (`near_duplicate_leakage`). Three additions turn that sketch into a
usable methodological control.

### 3.1 A clean scan is a bound, not a proof

With `N` sketched held-out documents and `k` flagged ones, the observed rate
`k/N` is not the whole answer: at `k = 0` a scan of 4,000 documents is still
compatible with a contamination rate of roughly 0.07% at 95% confidence.
`near_duplicate_scan_bound` reports the one-sided Wilson upper bound for
each later split, using the **union** of documents flagged against any
earlier split (one contaminated document is one document, not one per
pair).

The denominator is the set of **scanned** documents. A prefix scan is
reported as `scope = "scanned_prefix"`: the bound applies to that prefix,
not to the unsampled remainder. The audit never extrapolates a prefix bound
to the whole split.

Pass `--max-near-dup-rate <p>` to pre-register the largest contamination
rate the study tolerates. The audit fails when the one-sided upper bound on
the scanned documents exceeds `p` (and when nothing was scanned, because an
empty scan cannot certify anything). Choose `p` before looking at the
audit; changing it after seeing the result is the same garden of forking
paths as changing a p-value threshold.

### 3.2 Make the bound actionable

`--append-near-dup-to-ledgers` writes the canonical digests of flagged
held-out documents into the existing ledger files:

* documents flagged in `validation` against `train`, and in `test` against
  `train`, are appended to `--write-train-hash-file`;
* documents flagged in `test` against `validation` are appended to
  `--write-val-hash-file`.

The trainers already consume those files through `--exclude-hash-file` and
`--exclude-test-hash-file`; no new training flag is needed. Only the later
(held-out) document is excluded — a training document is never removed by a
near-duplicate finding. The ledger info records
`near_duplicate_hashes_added`, `near_duplicate_coverage` and
`near_duplicate_missing_digest_documents`, so a reader can tell whether the
remediation covered the whole split or only the scanned prefix.

This is deliberately opt-in: appending held-out hashes changes the
meaning of the ledger, and the audit report must show that it was requested.

### 3.3 What the remediation does not fix

The audit's effective token budget and `*_tvd_effective` block are computed
before the near-duplicate ledger is applied. They are therefore a
*pre-remediation* view: the run that consumes the augmented ledger will drop
a few more documents, so its effective held-out count is slightly smaller.
Use `flagged_tokens` (reported per later split) as the size of that
correction, and re-run the audit after regenerating the ledger if an exact
post-remediation budget is needed.

A prefix-only scan can only find prefix-only contamination. The scan's
`lower_bound` flag and `scan_bound.scope` say so; no output claims the whole
corpus is near-duplicate free when only a prefix was sketched.

## 4. Paired data identity: the stream the optimizer actually saw

Two arms can record the same dataset revision, split fractions, cleaning
flags and seed and still consume different token windows — a worker-sharding
change, a loader rewrite, a shuffle seed that is not part of the recipe, or a
different resume point can all do it. `TrainStreamFingerprint` in
`adasquash/repro.py` hashes every consumed input/target batch before the
device transfer and writes `data.train_stream` to `run_manifest.json`. The
target is included because a packed window's final label is not part of its
input, so an input-only digest could miss a label difference:

* `start_step` identifies a resumed prefix;
* `prefix_batches` / `prefix_fingerprint` cover the first 64 consumed batches,
  so arms that stopped at different steps can still prove their shared prefix
  is identical;
* `batches` / `stream_fingerprint` cover the whole run and are compared when
  both arms consumed the same number of batches.

`compare_arms.py` pairs the fingerprint per shared training seed. A recorded
difference is a pairing failure (the row is reported but cannot certify); a
missing record is treated as *unknown*, not equal. Pass
`--require-train-stream-identity` to make an unknown record fatal for the
comparison. A resumed run starts a new fingerprint at its resume step, so it
is comparable only to another run resumed at the same step — which is the
correct rule for a data-prefix claim.

## 5. Evaluation budget: batches are not independent documents

`eval_batch_capacity` converts the effective held-out token count into the
number of distinct packed batches, and `evaluation_reuse_status` compares
that with the requested `--val-batches` / `--test-batches`. A validation
loader with fewer distinct batches than requested cycles: repeated windows
are then treated as independent observations by a per-batch standard error,
and a checkpoint-selection minimum becomes a maximum over reused data. The
training loop records the same information (`last_eval_unique_batches`,
`last_eval_cycled`, `last_eval_reused`) in the run manifest.

The token budget itself is a *document-fraction* estimate converted through
observed tokens per kept document, not a promise that `val_frac` equals a
token fraction. When the split was only read as a prefix, the audit labels
the estimate `extrapolated_from_sampled_prefix`.

## 6. Pre-run checklist

Before a comparison is launched:

1. pin `--dataset-revision` and `--tokenizer-revision` to full commit
   hashes (or let the audit resolve and record them);
2. run `scripts/audit_data.py` with the **same** split fractions, cleaning
   flags, group key and ledger paths the run will use;
3. pre-register `--max-near-dup-rate`, `--max-tvd`, `--min-eval-tokens`,
   `--min-eval-docs` and the evaluation batch geometry;
4. generate the exact/canonical/group ledgers with `--n-docs 0` when a full
   guarantee is required, and append near-duplicate exclusions if the scan
   found any;
5. pass the resulting ledgers to every arm with the same paths and
   `--exclude-hash-file` / `--exclude-test-hash-file` flags;
6. compare arms with `--require-train-stream-identity` once every arm records
   a stream fingerprint, so a configuration-only match is not mistaken for a
   data-paired comparison;
7. keep `analysis_plan.json` (and its data-protocol fields) with the run
   outputs so a reviewer can see the protocol that was registered before the
   loss was observed.

## 7. Reproducing the audit addition

```bash
python scripts/audit_data.py \
    --dataset HuggingFaceFW/fineweb-edu --dataset-config sample-10BT \
    --dataset-revision <commit> --tokenizer gpt2 --tokenizer-revision <commit> \
    --val-frac 0.005 --test-frac 0.005 --num-documents <N> \
    --near-dup --max-near-dup-rate 0.001 \
    --append-near-dup-to-ledgers \
    --write-train-hash-file runs/ledgers/train.hash \
    --write-val-hash-file runs/ledgers/val.hash \
    --output runs/data_audit.json
```

Then train with the same ledgers:

```bash
python scripts/train_gpt2.py ... \
    --exclude-hash-file runs/ledgers/train.hash \
    --exclude-test-hash-file runs/ledgers/val.hash
```

The audit is a read-only pass over the corpus; it never trains or writes to
a split.
