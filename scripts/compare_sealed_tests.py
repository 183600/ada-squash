from __future__ import annotations
"""Paired statistical comparison of sealed-test results.

`train_gpt2.py --final-test-eval` opens the sealed test split exactly once per
run and stores the per-batch losses it averaged (plus the data-protocol block
that decides which documents form those batches). Two runs whose `pairing`
blocks match were scored on the *same* sealed segments: the held-out loader
never shuffles, so the per-batch losses are a paired sample and the noise
separating two arms can be read off the per-batch deltas instead of guessed
from two bare means.

This tool performs that comparison honestly:

* pairing metadata must match exactly. A mismatch means the runs were not
  scored on the same held-out batches, so the comparison is not paired; the
  script reports the row as not comparable (and exits non-zero) instead of
  silently substituting a weaker unpaired test.
* every candidate is compared one-sided against the same reference with the
  paired Student-t test, and all raw p-values enter one `holm_adjust` family,
  because a matrix of candidates against one baseline is a family of tests.
* each row reports the minimum delta detectable at the requested power, so a
  non-significant result is qualified by the resolution the comparison had
  rather than read as evidence of equality.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import holm_adjust, mean_ci, paired_t_one_sided, student_t_ppf
from adasquash.repro import collect_run_metadata, save_run_manifest

PAIRING_KEYS = (
    "dataset", "dataset_config", "dataset_revision",
    "tokenizer", "tokenizer_revision",
    "num_documents", "val_frac", "test_frac",
    "dedup", "min_doc_chars", "max_doc_chars", "exclude_hash_file",
    "batch_size", "seq_len",
)


def load_result(path) -> dict:
    """Load one `test_results.json` and validate the per-batch record."""
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[compare] {p} does not exist")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[compare] {p} is not valid JSON: {exc}")
    losses = data.get("test_batch_losses")
    if not isinstance(losses, list) or not losses:
        raise SystemExit(
            f"[compare] {p} has no 'test_batch_losses': it was written before the sealed "
            "evaluation stored per-batch losses (or the record is incomplete), so a paired "
            "comparison is impossible. Re-run the sealed evaluation with the current "
            "scripts/train_gpt2.py (--final-test-eval).")
    losses = [float(v) for v in losses]
    if not all(math.isfinite(v) for v in losses):
        raise SystemExit(f"[compare] {p} contains non-finite per-batch test losses; fix the run first")
    reported = data.get("test_loss")
    mean = sum(losses) / len(losses)
    if not isinstance(reported, (int, float)) or not math.isfinite(reported):
        raise SystemExit(f"[compare] {p} has no finite 'test_loss' to cross-check")
    if abs(mean - float(reported)) > 1e-9 * max(1.0, abs(float(reported))):
        raise SystemExit(
            f"[compare] {p}: stored test_loss={reported} disagrees with the mean of "
            f"test_batch_losses={mean}; the record is internally inconsistent.")
    return data


def pairing_mismatches(ref: dict, other: dict) -> dict:
    """Keys whose values differ between the two runs' data-protocol blocks."""
    a, b = ref.get("pairing"), other.get("pairing")
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {"pairing": ["missing in one result file", "missing in the other"]}
    return {k: [a.get(k), b.get(k)] for k in PAIRING_KEYS if a.get(k) != b.get(k)}


def _mde(stderr: float, df: float, alpha: float, power: float) -> float:
    """Minimum detectable delta of a one-sided t-test at `alpha` and `power`."""
    if stderr <= 0.0 or df <= 0.0:
        return 0.0
    return (student_t_ppf(1.0 - alpha, df) + student_t_ppf(power, df)) * stderr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", nargs="+", help="test_results.json files to compare against the reference")
    p.add_argument("--reference", required=True, help="test_results.json of the baseline arm")
    p.add_argument("--alpha", type=float, default=0.05, help="one-sided significance level (Holm-adjusted)")
    p.add_argument("--margin", type=float, default=0.0, help="practical margin: the candidate must be better by more than this")
    p.add_argument("--power", type=float, default=0.8, help="target power for the minimum-detectable-delta column")
    p.add_argument("--hypothesis", default=None, help="the claim this comparison tests, recorded in the output manifest")
    p.add_argument("--output", default=None, help="write the full comparison (rows + manifests) to this JSON file")
    args = p.parse_args()
    reference_path = Path(args.reference)
    ref = load_result(reference_path)
    ref_batches = [float(v) for v in ref["test_batch_losses"]]
    ref_stats = mean_ci(ref_batches, alpha=args.alpha)
    print(f"[compare] reference {reference_path}: test_loss={ref_stats['mean']:.4f} "
          f"+/- {ref_stats['stderr']:.4f} (stderr, n={ref_stats['n']} batches)")

    rows = []
    for cand_path in args.candidates:
        cand_path = Path(cand_path)
        if cand_path.resolve() == reference_path.resolve():
            raise SystemExit("[compare] a candidate is the reference itself")
        cand = load_result(cand_path)
        mismatches = pairing_mismatches(ref, cand)
        if mismatches:
            rows.append({
                "candidate": str(cand_path),
                "comparable": False,
                "reason": "pairing metadata differs: " + json.dumps(mismatches, sort_keys=True),
            })
            continue
        cand_batches = [float(v) for v in cand["test_batch_losses"]]
        if len(cand_batches) != len(ref_batches):
            rows.append({
                "candidate": str(cand_path),
                "comparable": False,
                "reason": (f"scored on {len(cand_batches)} sealed batches vs the reference's "
                           f"{len(ref_batches)}; the runs did not share the same test pass"),
            })
            continue
        stats = mean_ci(cand_batches, alpha=args.alpha)
        test = paired_t_one_sided(cand_batches, ref_batches, margin=args.margin, alpha=args.alpha)
        rows.append({
            "candidate": str(cand_path),
            "comparable": True,
            "n_paired_batches": len(cand_batches),
            "mean": stats["mean"],
            "std": stats["std"],
            "stderr": stats["stderr"],
            "ci_level": stats["ci_level"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "mean_delta": test["mean_delta"],
            "delta_ci_low": test["ci_low"],
            "delta_ci_high": test["ci_high"],
            "t": test["t"],
            "df": test["df"],
            "p_one_sided": test["p_one_sided"],
            "beats_reference": bool(test["mean_delta"] is not None and test["mean_delta"] < -args.margin),
            "min_detectable_delta": _mde(stats["stderr"], test["df"], args.alpha, args.power),
        })

    comparable = [r for r in rows if r["comparable"]]
    adjusted = holm_adjust([r["p_one_sided"] for r in comparable])
    for row, p_holm in zip(comparable, adjusted):
        row["p_one_sided_holm"] = p_holm
        row["beats_reference_significantly"] = bool(
            row["beats_reference"] and p_holm is not None and p_holm < args.alpha)

    print(f"[compare] paired one-sided t-test, margin={args.margin:g}, alpha={args.alpha:g}, "
          f"Holm over {len(comparable)} comparable row(s)")
    for r in rows:
        if not r["comparable"]:
            print(f"[compare] {r['candidate']}: NOT COMPARABLE - {r['reason']}")
            continue
        p_str = "n/a" if r["p_one_sided_holm"] is None else f"{r['p_one_sided_holm']:.4g}"
        verdict = ("significantly better" if r["beats_reference_significantly"]
                   else ("better but not significant" if r["beats_reference"] else "not better"))
        print(f"[compare] {r['candidate']}: mean={r['mean']:.4f} +/- {r['stderr']:.4f} "
              f"delta={r['mean_delta']:+.4f} (95% CI [{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]) "
              f"t={r['t']:.3f} df={r['df']:.0f} p={r['p_one_sided']:.4g} p_holm={p_str} "
              f"MDE80={r['min_detectable_delta']:.4f} -> {verdict}")

    if args.hypothesis:
        print(f"[hypothesis] {args.hypothesis}")
    not_comparable = [r for r in rows if not r["comparable"]]
    if args.output:
        payload = {
            "reference": str(reference_path),
            "reference_mean": ref_stats["mean"],
            "reference_stderr": ref_stats["stderr"],
            "reference_n_batches": ref_stats["n"],
            "alpha": args.alpha,
            "margin": args.margin,
            "power": args.power,
            "hypothesis": args.hypothesis,
            "holm_family_size": len(comparable),
            "rows": rows,
        }
        save_run_manifest(args.output, {**collect_run_metadata(), "comparison": payload})
        print(f"[compare] full comparison -> {args.output}")
    if not_comparable:
        print(f"[compare] ERROR: {len(not_comparable)} row(s) were not comparable; "
              "no paired test was fabricated for them", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
