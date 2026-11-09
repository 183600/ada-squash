"""Result-analysis enhancement report.

This script reads a compare_arms output and produces an analysis-reproducibility
report focused on result interpretation: standardized effect sizes, robustness
checks, and analysis provenance.

It does NOT modify .github or README files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from adasquash.jsonio import save_json_atomic
from adasquash.metrics import cohens_dz, hedges_g_paired, mean_std_stderr


def _fingerprint(obj):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def generate_report(comparison_path: Path, out_path: Path | None = None):
    comp = json.loads(comparison_path.read_text())
    report = {
        "schema": "adasquash.result_analysis_report.v1",
        "input": str(comparison_path),
        "input_fingerprint": _fingerprint(comp),
        "metric": comp.get("metric"),
        "reference": comp.get("reference"),
        "alpha": comp.get("alpha"),
        "arms_summary": [],
    }

    rows = comp.get("rows", [])
    for row in rows:
        name = row.get("name")
        n_shared = row.get("n_shared_seeds", len(row.get("shared_seeds", [])))
        deltas = []
        per_seed = row.get("per_seed_delta") or {}
        for v in per_seed.values():
            try:
                deltas.append(float(v))
            except Exception:
                pass

        stats = mean_std_stderr(deltas) if deltas else None
        mean_delta = stats["mean"] if stats else None
        sd_delta = stats["std"] if stats else None

        standardized = None
        hedges = None
        if sd_delta and sd_delta > 0 and deltas:
            # `deltas` already are the paired differences; the paired effect
            # sizes take two arms, so pair them against zeros to standardise
            # the differences themselves rather than a single arm's values.
            zeros = [0.0] * len(deltas)
            dz = cohens_dz(deltas, zeros)
            standardized = dz
            hedges = hedges_g_paired(deltas, zeros)

        # Robustness: leave-one-out sign stability already in compare_arms
        jackknife = row.get("seed_jackknife") or {}
        sign_stable = jackknife.get("sign_stable")

        report["arms_summary"].append({
            "name": name,
            "n_shared_seeds": n_shared,
            "delta_paired_mean": row.get("delta_paired_mean"),
            "paired_delta_std": sd_delta,
            "cohens_dz": standardized,
            "hedges_g": hedges,
            "p_one_sided": row.get("p_one_sided"),
            "p_one_sided_holm": row.get("p_one_sided_holm"),
            "sign_stable_jackknife": sign_stable,
            "certified": row.get("certified"),
            "certification_basis": row.get("certification_basis"),
        })

    report["analysis_provenance"] = {
        "metrics_module": "adasquash.metrics",
        "compare_arms_version": "v1",
        "note": "Standardized effect sizes added for interpretability across scales."
    }

    out = out_path or comparison_path.parent / "result_analysis_report.json"
    save_json_atomic(out, report)
    print(f"[report] wrote {out}")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("comparison_json", type=Path, help="arm_comparison.json produced by scripts/compare_arms.py")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    generate_report(args.comparison_json, args.output)


if __name__ == "__main__":
    main()
