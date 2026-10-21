from __future__ import annotations
"""Paired statistical comparison of sealed-test results.

`train_gpt2.py --final-test-eval` opens the sealed test split exactly once per
run and stores the per-batch losses it averaged (plus the data-protocol block
that decides which documents form those batches). A directory holding per-seed
runs (`seed{N}/test_results.json`, the layout the Phase 1 multi-seed launcher
writes) is one *arm* replicated over independent training seeds.

This tool performs that comparison honestly:

* pairing metadata must match exactly. A mismatch means the runs were not
  scored on the same held-out batches, so the comparison is not paired; the
  script reports the row as not comparable (and exits non-zero) instead of
  silently substituting a weaker unpaired test.
* the sealed result must belong to the finished run it sits next to. A
  leftover `test_results.json` from an earlier run is rejected unless the
  manifest records the same `results.test_loss` and, when present, the
  artifact fingerprint matches its content.
* **training seeds are the unit of replication.** The certification test is
  the one-sided paired Student-t on the per-seed sealed losses of the arms'
  *shared* training seeds (>= 3 required), and the complete seed-paired rows
  enter one `holm_adjust` family. Two shared seeds are tested for description
  only (df = 1). A single training run per arm - or a comparison in which only
  one seed is shared - carries no run-to-run variance: the per-batch paired
  test is reported for description only and can never set
  `beats_reference_significantly`, because batch noise is not run-to-run
  noise (the same rule `scripts/ablation.py` and
  `scripts/train_llama_depth_sweep.py` enforce on validation losses).
* each row reports the number of paired seeds and the minimum delta
  detectable at the requested power, so a non-significant result is qualified
  by the resolution the comparison had rather than read as evidence of
  equality.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.metrics import MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust, mean_ci, mean_std_stderr, paired_t_one_sided, student_t_ppf
from adasquash.repro import (collect_run_metadata, finish_metadata, save_run_manifest,
                             training_recipe_mismatches)

PAIRING_KEYS = (
    "dataset", "dataset_config", "dataset_revision",
    "dataset_revision_resolved", "num_workers",
    "tokenizer", "tokenizer_revision", "tokenizer_revision_resolved",
    "num_documents", "val_frac", "test_frac",
    "dedup", "min_doc_chars", "max_doc_chars", "exclude_hash_file", "exclude_test_hash_file",
    "group_key", "exclude_group_file", "exclude_test_group_file",
    "ledger_fingerprints", "batch_size", "seq_len",
)

_LEDGER_MANIFEST_KEYS = (
    "exclude_hash_file", "exclude_test_hash_file",
    "exclude_group_file", "exclude_test_group_file",
)

_SEED_DIR_RE = re.compile(r"seed(\d+)$")

# A comparison is certifiable only when at least this many training seeds are
# shared by both arms: with one seed the test would measure the noise of a
# single run, and with two seeds df = 1 is too weak for a decision rule.
MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE = MIN_SEEDS_FOR_SIGNIFICANCE


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
            f"[compare] {p} has no 'test_batch_losses': the record is incomplete, so a "
            "paired comparison is impossible. Re-run the sealed evaluation with the "
            "current scripts/train_gpt2.py (--final-test-eval).")
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


def load_run_manifest(result_path) -> dict | None:
    """Training manifest stored next to a sealed-test result, if readable."""
    path = Path(result_path).with_name("run_manifest.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None

def _sealed_result_problem(run: dict) -> str | None:
    """Why this sealed-test file cannot be tied to its finished run, or None."""
    manifest = run.get("manifest")
    results = manifest.get("results") if isinstance(manifest, dict) else None
    if not isinstance(results, dict):
        return "manifest has no results block"
    recorded = results.get("test_loss")
    if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
            or not math.isfinite(float(recorded))):
        return ("manifest has no finite results.test_loss: this run did not record a "
                "sealed test evaluation, so a test_results.json in its directory cannot "
                "be attributed to it")
    actual = run["data"].get("test_loss")
    if (isinstance(actual, bool) or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))):
        return "test_results.json has no finite test_loss"
    tolerance = 1e-9 * max(1.0, abs(float(recorded)))
    if abs(float(actual) - float(recorded)) > tolerance:
        return (f"test_results.json test_loss={actual} disagrees with manifest "
                f"results.test_loss={recorded}; the file does not belong to this run")
    artifacts = manifest.get("artifacts")
    artifact = artifacts.get("test_results") if isinstance(artifacts, dict) else None
    if isinstance(artifact, dict):
        sha = artifact.get("sha256")
        if artifact.get("exists") is False or not isinstance(sha, str) or not sha:
            return "manifest artifacts.test_results is missing or has no sha256"
        path = Path(run["path"])
        hasher = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    hasher.update(chunk)
            size = path.stat().st_size
        except OSError as exc:
            return f"cannot read test_results.json for fingerprint verification: {exc}"
        if hasher.hexdigest() != sha:
            return "test_results.json content hash disagrees with the manifest fingerprint"
        expected_bytes = artifact.get("bytes")
        if expected_bytes is not None:
            try:
                expected_size = int(expected_bytes)
            except (TypeError, ValueError):
                return "manifest artifacts.test_results has a malformed byte count"
            if size != expected_size:
                return "test_results.json size disagrees with the manifest fingerprint"
    return None

def _shape_mismatches(ref_manifest: dict, other_manifest: dict) -> dict:
    a = ref_manifest.get("results") or {}
    b = other_manifest.get("results") or {}
    missing = "<missing>"
    return {f"results.{key}": [a.get(key, missing), b.get(key, missing)]
            for key in ("n_layers", "dim", "n_heads")
            if key not in a or key not in b or a.get(key) != b.get(key)}

def _manifest_ledger_content(manifest: dict):
    missing = "<missing>"
    data = manifest.get("data") if isinstance(manifest, dict) else None
    if not isinstance(data, dict):
        return None
    out = {}
    for key in _LEDGER_MANIFEST_KEYS:
        if key not in data:
            out[key] = missing
            continue
        fingerprint = data.get(key)
        if isinstance(fingerprint, dict):
            out[key] = {"exists": fingerprint.get("exists"),
                        "sha256": fingerprint.get("sha256")}
        else:
            out[key] = fingerprint
    return out

def _ledger_mismatches(ref_manifest: dict, other_manifest: dict) -> dict:
    a = _manifest_ledger_content(ref_manifest)
    b = _manifest_ledger_content(other_manifest)
    # Missing ledger identity is unknown, never "the same": fail closed instead
    # of certifying that two runs were scored on identical held-out documents.
    if a is None or b is None:
        return {"data": ["missing in one run_manifest" if a is None else "recorded",
                         "missing in the other" if b is None else "recorded"]}
    missing = "<missing>"
    return {f"data.{key}": [a[key], b[key]]
            for key in _LEDGER_MANIFEST_KEYS
            if a[key] == missing or b[key] == missing or a[key] != b[key]}

def pairing_mismatches(ref: dict, other: dict) -> dict:
    """Keys whose values differ between the two runs' data-protocol blocks."""
    a, b = ref.get("pairing"), other.get("pairing")
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {"pairing": ["missing in one result file", "missing in the other"]}
    missing = "<missing>"
    return {k: [a.get(k, missing), b.get(k, missing)] for k in PAIRING_KEYS
            if k not in a or k not in b or a.get(k) != b.get(k)}


def _mde(stderr: float, df: float, alpha: float, power: float) -> float:
    """Minimum detectable delta of a one-sided t-test at `alpha` and `power`."""
    if stderr <= 0.0 or df <= 0.0:
        return 0.0
    return (student_t_ppf(1.0 - alpha, df) + student_t_ppf(power, df)) * stderr


def _discover_results(path: Path) -> list[Path]:
    """One run per training seed under an arm directory, or the file itself."""
    if not path.is_dir():
        return [path]
    # Only `seed<digits>` directories name a training seed (`seed_backup`,
    # `seed0_old`, ... are skipped, exactly like `_seed_of` does).
    seed_files = sorted(
        (p for p in path.glob("seed*/test_results.json")
         if _SEED_DIR_RE.search(p.parent.name)),
        key=lambda p: int(_SEED_DIR_RE.search(p.parent.name).group(1)))
    if seed_files:
        return seed_files
    direct = path / "test_results.json"
    if direct.exists():
        return [direct]
    raise SystemExit(
        f"[compare] {path} is a directory with neither seed*/test_results.json nor "
        "test_results.json; point --reference / the candidate at the run directory or "
        "at one test_results.json file")


def _seed_of(run_path: Path, data: dict):
    """Training seed recorded by the run (or encoded in its seedN/ directory)."""
    value = data.get("seed")
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    match = _SEED_DIR_RE.search(run_path.parent.name)
    if match:
        return int(match.group(1))
    return None


def load_arm(path) -> dict:
    """All runs of one arm: `{runs: [{seed, path, data, batch_losses}]}`."""
    arm_path = Path(path)
    runs = []
    for run_path in _discover_results(arm_path):
        data = load_result(run_path)
        runs.append({
            "path": str(run_path),
            "seed": _seed_of(run_path, data),
            "data": data,
            "manifest": load_run_manifest(run_path),
            "batch_losses": [float(v) for v in data["test_batch_losses"]],
        })
    seeds = [r["seed"] for r in runs if r["seed"] is not None]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(
            f"[compare] {arm_path}: two runs report the same training seed; a seed must "
            "identify exactly one run, otherwise the pairing is ambiguous")
    return {
        "path": str(arm_path),
        "runs": runs,
        "n_runs": len(runs),
        "seeds": seeds,
        "unseeded_runs": [r["path"] for r in runs if r["seed"] is None],
    }


def _seed_means(arm: dict) -> dict:
    """{training seed: sealed test loss} for the seeded runs of an arm."""
    out = {}
    for run in arm["runs"]:
        if run["seed"] is None:
            continue
        out[int(run["seed"])] = sum(run["batch_losses"]) / len(run["batch_losses"])
    return out


def _arm_seed_stats(arm: dict) -> dict:
    """Mean +/- Std of the per-seed sealed losses (None std for a single run)."""
    vals = list(_seed_means(arm).values())
    stats = mean_std_stderr(vals)
    if stats is None:
        return {"n_seeds": 0, "mean": None, "std": None, "stderr": None}
    return {
        "n_seeds": stats["n"],
        "mean": stats["mean"],
        "std": stats["std"] if stats["n"] >= 2 else None,
        "stderr": stats["stderr"] if stats["n"] >= 2 else None,
    }


def _pairing_problem(ref_arm: dict, cand_arm: dict) -> str | None:
    """Reason the two arms cannot be treated as paired, or None."""
    problems = []
    for run in cand_arm["runs"]:
        for ref_run in ref_arm["runs"]:
            mismatches = pairing_mismatches(ref_run["data"], run["data"])
            if mismatches:
                problems.append(
                    f"{ref_run['path']} vs {run['path']}: "
                    + json.dumps(mismatches, sort_keys=True))
    if problems:
        return "pairing metadata differs: " + "; ".join(problems[:3])
    counts = {len(r["batch_losses"]) for r in ref_arm["runs"] + cand_arm["runs"]}
    if len(counts) != 1:
        return (f"runs were scored on different numbers of sealed batches "
                f"({sorted(counts)}); the arms did not share the same test pass")
    # Identical test batches are not enough: the training recipe itself must
    # be controlled, otherwise the delta can be an lr/step/data effect rather
    # than a property of the compared norm. A missing or unfinished manifest
    # leaves the recipe unknown, which is never treated as matching.
    unverified = [r["path"] for r in ref_arm["runs"] + cand_arm["runs"]
                  if not isinstance(r.get("manifest"), dict)
                  or r["manifest"].get("status") != "finished"]
    if unverified:
        return ("training recipe cannot be verified: missing/unfinished run_manifest.json "
                "for " + "; ".join(unverified[:3]))
    association_problems = []
    for run in ref_arm["runs"] + cand_arm["runs"]:
        problem = _sealed_result_problem(run)
        if problem:
            association_problems.append(f"{run['path']}: {problem}")
    if association_problems:
        return ("sealed test result cannot be tied to its run: "
                + "; ".join(association_problems[:3]))
    recipe_problems = []
    ledger_problems = []
    for run in cand_arm["runs"]:
        for ref_run in ref_arm["runs"]:
            ref_manifest = ref_run["manifest"]
            manifest = run["manifest"]
            mismatches = training_recipe_mismatches(
                ref_manifest.get("hyperparameters"), manifest.get("hyperparameters"))
            mismatches.update(_shape_mismatches(ref_manifest, manifest))
            if mismatches:
                recipe_problems.append(
                    f"{ref_run['path']} vs {run['path']}: "
                    + json.dumps(mismatches, sort_keys=True))
            ledger_mismatches = _ledger_mismatches(ref_manifest, manifest)
            if ledger_mismatches:
                ledger_problems.append(
                    f"{ref_run['path']} vs {run['path']}: "
                    + json.dumps(ledger_mismatches, sort_keys=True))
    if ledger_problems:
        return ("cross-split ledger contents differ, so the arms were not scored on the "
                "same held-out documents: " + "; ".join(ledger_problems[:3]))
    if recipe_problems:
        return ("training recipe/model shape differs, so the arms are not a controlled "
                "comparison: " + "; ".join(recipe_problems[:3]))
    return None


def _describe_arm(label: str, arm: dict) -> str:
    stats = _arm_seed_stats(arm)
    if stats["n_seeds"] >= 2:
        return (f"{label}: test_loss={stats['mean']:.4f} +/- {stats['std']:.4f} "
                f"(std across {stats['n_seeds']} training seeds)")
    if arm["runs"]:
        batches = arm["runs"][0]["batch_losses"]
        ci = mean_ci(batches)
        return (f"{label}: test_loss={ci['mean']:.4f} +/- {ci['stderr']:.4f} "
                f"(stderr over {ci['n']} batches of a single training run; "
                f"no run-to-run spread available)")
    return f"{label}: no runs"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("candidates", nargs="+", help="arm directories (holding seed*/test_results.json) or single test_results.json files to compare against the reference")
    p.add_argument("--reference", required=True, help="arm directory (holding seed*/test_results.json) or test_results.json of the baseline arm")
    p.add_argument("--alpha", type=float, default=0.05, help="one-sided significance level (Holm-adjusted)")
    p.add_argument("--margin", type=float, default=0.0, help="practical margin: the candidate must be better by more than this")
    p.add_argument("--power", type=float, default=0.8, help="target power for the minimum-detectable-delta column")
    p.add_argument("--hypothesis", default=None, help="the claim this comparison tests, recorded in the output manifest")
    p.add_argument("--output", default=None, help="write the full comparison (rows + manifests) to this JSON file")
    args = p.parse_args()
    reference_path = Path(args.reference)
    ref_arm = load_arm(reference_path)
    ref_seed_means = _seed_means(ref_arm)
    print(f"[compare] reference {_describe_arm(str(reference_path), ref_arm)}")
    if ref_arm["n_runs"] < MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE:
        print(f"[compare] reference arm has fewer than {MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE} "
              "training runs: any test below is descriptive (batch noise is not run-to-run "
              "noise) and can never certify a significance claim. Provide seed-replicated "
              f"runs (>= {MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE}, recommend 3-5) to certify.")
    if ref_arm["unseeded_runs"]:
        print(f"[compare] reference runs without a recorded training seed: {ref_arm['unseeded_runs']} "
              "(they can only enter batch-level description)")

    rows = []
    for cand_path in args.candidates:
        cand_path = Path(cand_path)
        if cand_path.resolve() == reference_path.resolve():
            raise SystemExit("[compare] a candidate is the reference itself")
        cand_arm = load_arm(cand_path)
        problem = _pairing_problem(ref_arm, cand_arm)
        if problem:
            rows.append({
                "candidate": str(cand_path),
                "comparable": False,
                "reason": problem,
                "n_runs_candidate": cand_arm["n_runs"],
                "n_runs_reference": ref_arm["n_runs"],
            })
            continue

        cand_seed_means = _seed_means(cand_arm)
        common = sorted(set(cand_seed_means) & set(ref_seed_means))
        cand_stats = _arm_seed_stats(cand_arm)
        row = {
            "candidate": str(cand_path),
            "comparable": True,
            "n_runs_candidate": cand_arm["n_runs"],
            "n_runs_reference": ref_arm["n_runs"],
            "n_seeds_candidate": len(cand_seed_means),
            "n_seeds_reference": len(ref_seed_means),
            "paired_seeds": common,
            "n_paired_seeds": len(common),
            "seed_means_candidate": {str(s): v for s, v in cand_seed_means.items()},
            "seed_means_reference": {str(s): v for s, v in ref_seed_means.items()},
            "mean": cand_stats["mean"],
            "std": cand_stats["std"],
            "stderr": cand_stats["stderr"],
            "mean_delta": None,
            "delta_ci_low": None,
            "delta_ci_high": None,
            "t": None,
            "df": None,
            "p_one_sided": None,
            "p_one_sided_holm": None,
            "beats_reference": False,
            "beats_reference_significantly": False,
            "certifiable": False,
            "significance_basis": None,
            "basis": None,
            "min_detectable_delta": None,
        }
        test = None
        if len(common) >= 2:
            a = [cand_seed_means[s] for s in common]
            b = [ref_seed_means[s] for s in common]
            test = paired_t_one_sided(a, b, margin=args.margin, alpha=args.alpha)
            row["basis"] = "seed_means"
            row["certifiable"] = len(common) >= MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE
            # Headline mean/std over the seeds used by the paired test.
            paired_stats = mean_std_stderr(a)
            row["mean"] = paired_stats["mean"]
            row["std"] = paired_stats["std"] if paired_stats["n"] >= 2 else None
            row["stderr"] = paired_stats["stderr"] if paired_stats["n"] >= 2 else None
            row["mean_all_candidate_seeds"] = cand_stats["mean"]
            row["std_all_candidate_seeds"] = cand_stats["std"]
            row["stderr_all_candidate_seeds"] = cand_stats["stderr"]
            deltas = [x - y for x, y in zip(a, b)]
            delta_stats = mean_std_stderr(deltas)
            row["min_detectable_delta"] = _mde(
                delta_stats["stderr"], test["df"], args.alpha, args.power)
        else:
            # One shared training run (or only unseeded runs): a paired test
            # over that run's batches describes the measurement noise on the
            # shared sealed segments, nothing more.
            pairs = None
            if len(common) == 1:
                seed = common[0]
                cand_run = next(r for r in cand_arm["runs"] if r["seed"] == seed)
                ref_run = next(r for r in ref_arm["runs"] if r["seed"] == seed)
                pairs = (cand_run, ref_run)
                row["basis"] = f"batch_single_seed{seed}"
            elif cand_arm["n_runs"] == 1 and ref_arm["n_runs"] == 1:
                pairs = (cand_arm["runs"][0], ref_arm["runs"][0])
                row["basis"] = "batch_single_run"
            if pairs is not None:
                a, b = pairs[0]["batch_losses"], pairs[1]["batch_losses"]
                if len(a) == len(b):
                    test = paired_t_one_sided(a, b, margin=args.margin, alpha=args.alpha)
                    stats = mean_ci(a, alpha=args.alpha)
                    row["mean"], row["std"], row["stderr"] = stats["mean"], stats["std"], stats["stderr"]
                    deltas = [x - y for x, y in zip(a, b)]
                    delta_stats = mean_std_stderr(deltas)
                    if test["df"] is not None:
                        row["min_detectable_delta"] = _mde(
                            delta_stats["stderr"], test["df"], args.alpha, args.power)
        if test is not None and test["mean_delta"] is not None:
            row["mean_delta"] = test["mean_delta"]
            row["delta_ci_low"] = test["ci_low"]
            row["delta_ci_high"] = test["ci_high"]
            row["t"] = test["t"]
            row["df"] = test["df"]
            row["p_one_sided"] = test["p_one_sided"]
            row["beats_reference"] = bool(test["mean_delta"] < -args.margin)
        if row.get("basis") == "seed_means" and not row.get("certifiable"):
            row["significance_basis"] = (
                f"descriptive only: {len(common)} shared training seed(s), fewer than the "
                f"required {MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE}; the delta is reported but "
                "not certified")
        elif row.get("basis") and row.get("basis") != "seed_means":
            row["significance_basis"] = (
                "descriptive only: batch-level test on a single training run; batch noise "
                "is not run-to-run noise")
        rows.append(row)

    # Only complete >= 3-seed paired tests are admissible to the Holm family;
    # a 2-seed or batch-level description is not a test of the run-to-run
    # claim and must not influence the family-wise correction.
    certifiable = [r for r in rows
                   if r.get("basis") == "seed_means" and r.get("certifiable")]
    adjusted = holm_adjust([r["p_one_sided"] for r in certifiable])
    for row, p_holm in zip(certifiable, adjusted):
        row["p_one_sided_holm"] = p_holm
        row["beats_reference_significantly"] = bool(
            row["beats_reference"] and p_holm is not None and p_holm < args.alpha)

    ci_label = f"{100 * (1 - args.alpha):g}% CI"
    mde_label = f"MDE{100 * args.power:g}"
    print(f"[compare] paired one-sided t-test, margin={args.margin:g}, alpha={args.alpha:g}, "
          f"Holm over {len(certifiable)} seed-paired row(s)")
    for r in rows:
        if not r["comparable"]:
            print(f"[compare] {r['candidate']}: NOT COMPARABLE - {r['reason']}")
            continue
        p_str = "n/a" if r["p_one_sided_holm"] is None else f"{r['p_one_sided_holm']:.4g}"
        base = r.get("basis")
        if r["mean_delta"] is None:
            print(f"[compare] {r['candidate']}: no testable pairing "
                  f"(basis={base}, {r['n_runs_candidate']} vs {r['n_runs_reference']} runs)")
            continue
        if base == "seed_means":
            if r["beats_reference_significantly"]:
                verdict = "significantly better"
            elif r["beats_reference"] and not r.get("certifiable", False):
                verdict = (f"better but descriptive only (only {r['n_paired_seeds']} shared "
                           "training seeds)")
            elif r["beats_reference"]:
                verdict = "better but not significant"
            else:
                verdict = "not better"
            print(f"[compare] {r['candidate']}: mean={r['mean']:.4f} +/- {r['std']:.4f} "
                  f"(seed std, n_seeds={r['n_paired_seeds']}) "
                  f"delta={r['mean_delta']:+.4f} ({ci_label} [{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]) "
                  f"t={r['t']:.3f} df={r['df']:.0f} p={r['p_one_sided']:.4g} p_holm={p_str} "
                  f"{mde_label}={r['min_detectable_delta']:.4f} -> {verdict}")
        else:
            note = ("descriptive only: single training run, batch noise is not run-to-run noise - "
                    "never certified as significant")
            if r["t"] is None:
                print(f"[compare] {r['candidate']}: mean={r['mean']:.4f} +/- {r['stderr']:.4f} "
                      f"(batch stderr) delta={r['mean_delta']:+.4f} "
                      f"(single sealed batch: no interval or test) "
                      f"basis={base} -> {note}")
            else:
                print(f"[compare] {r['candidate']}: mean={r['mean']:.4f} +/- {r['stderr']:.4f} "
                      f"(batch stderr) delta={r['mean_delta']:+.4f} "
                      f"({ci_label} [{r['delta_ci_low']:+.4f}, {r['delta_ci_high']:+.4f}]) "
                      f"t={r['t']:.3f} df={r['df']:.0f} p={r['p_one_sided']:.4g} "
                      f"basis={base} -> {note}")

    if args.hypothesis:
        print(f"[hypothesis] {args.hypothesis}")
    not_comparable = [r for r in rows if not r["comparable"]]
    if args.output:
        payload = {
            "reference": str(reference_path),
            "reference_n_runs": ref_arm["n_runs"],
            "reference_n_seeds": len(ref_seed_means),
            "reference_seed_means": {str(s): v for s, v in ref_seed_means.items()},
            "alpha": args.alpha,
            "margin": args.margin,
            "power": args.power,
            "hypothesis": args.hypothesis,
            "holm_family_size": len(certifiable),
            "min_paired_seeds_for_significance": MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE,
            "certification_basis": (
                "only the one-sided paired Student-t on per-seed sealed losses over the arms' "
                f"shared training seeds (>= {MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE}) can set "
                "beats_reference_significantly; two-seed and single-run batch-level tests are "
                "reported as description and never certified"),
            "limitations": (
                "batch-level intervals describe the measurement noise on the shared sealed "
                "segments of one training run; they do not cover initialization / data-order "
                f"/ batch-sampling variance. Run each arm with >= {MIN_PAIRED_SEEDS_FOR_SIGNIFICANCE} "
                "(recommend 3-5) seeds and compare same-seed runs to make the delta a "
                "run-to-run statement."),
            "rows": rows,
        }
        manifest = {**collect_run_metadata(hyperparameters=vars(args)), "comparison": payload}
        if not_comparable:
            # The SystemExit below reports rows that produced no paired test;
            # the manifest must not claim a complete comparison.
            manifest['status'] = 'failed'
            manifest['not_comparable_rows'] = list(not_comparable)
        else:
            manifest = finish_metadata(manifest)
        save_run_manifest(args.output, manifest)
        print(f"[compare] full comparison -> {args.output}")
    if not_comparable:
        print(f"[compare] ERROR: {len(not_comparable)} row(s) were not comparable; "
              "no paired test was fabricated for them", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
