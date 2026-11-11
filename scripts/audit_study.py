"""Audit a completed study directory against its provenance records.

The audit is read-only.  It verifies that run manifests are finished and
clean, that every artifact fingerprint still matches its file, that an
aggregate is the same object embedded in its replication manifest and agrees
with each per-seed manifest, and that an ablation analysis plan's
pre-registration fingerprint still matches its payload.

Exit status is 0 only when no problems are found; warnings are printed but
do not fail the audit.  Use ``--allow-*`` flags only for a deliberate audit
of an unfinished or post-hoc study, not for a confirmatory result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.jsonio import canonical_json_dumps, save_json_atomic
from adasquash.provenance import audit_study


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a study directory against its run manifests, "
                    "artifact fingerprints, aggregates and analysis plan")
    parser.add_argument("study_dir", help="directory containing the study")
    parser.add_argument("--output", default=None,
                        help="write the JSON audit report here instead of stdout")
    parser.add_argument("--allow-unfinished", action="store_true",
                        help="do not fail on manifests whose status is not finished")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="do not fail on runs from a dirty working tree")
    parser.add_argument("--allow-commit-drift", action="store_true",
                        help="do not fail when seed runs used different commits")
    parser.add_argument("--allow-plan-drift", action="store_true",
                        help="do not fail when analysis_plan.json records a post-hoc change")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = audit_study(
        args.study_dir,
        require_finished=not args.allow_unfinished,
        require_clean_tree=not args.allow_dirty,
        require_same_commit=not args.allow_commit_drift,
        require_plan_lock=not args.allow_plan_drift,
    )
    if args.output:
        save_json_atomic(args.output, report)
        print(f"[audit] wrote {args.output}")
    else:
        print(canonical_json_dumps(report, indent=2))
    summary = report["summary"]
    print(
        f"[audit] {summary.get('n_manifests', 0)} manifest(s), "
        f"{summary.get('n_artifacts', 0)} artifact record(s), "
        f"{summary.get('n_problems', 0)} problem(s), "
        f"{summary.get('n_warnings', 0)} warning(s)",
        file=sys.stderr,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
