"""Audit a study directory against its own provenance records.

A finished experiment is not just a set of JSON reports.  Every run manifest
claims a code revision, a completed status and a set of artifacts; the
replication manifest claims that its aggregate describes those seed runs; and
an ablation study additionally freezes an analysis-plan fingerprint.  Those
claims can drift independently: a checkpoint can be replaced, an aggregate
can be edited, a seed manifest can be rerun at another commit, or a plan can
be rewritten after the results were seen.

This module walks a study directory and checks the claims it can verify from
the files themselves.  It is deliberately read-only.  It reports problems
rather than raising so an audit can show every failure in one pass; the CLI
in ``scripts/audit_study.py`` turns a non-empty problem list into a non-zero
exit status.

The audit does not claim that a result is scientifically correct.  It proves
that the recorded inputs, artifacts and pre-registration are internally
consistent, which is the minimum condition for a result to be auditable.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

from .ablation.analysis import analysis_plan_fingerprint
from .repro import RUN_MANIFEST_SCHEMA

STUDY_AUDIT_SCHEMA = "adasquash.study_audit.v1"
ANALYSIS_PLAN_SCHEMA = "adasquash.analysis_plan.v1"

_MISSING = object()


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _looks_like_artifact_record(node) -> bool:
    return (
        isinstance(node, Mapping)
        and isinstance(node.get("path"), str)
        and bool(node.get("path"))
        and any(key in node for key in ("exists", "sha256", "bytes", "mtime"))
    )


def _iter_artifact_records(node, prefix: str = ""):
    """Yield ``(label, record)`` for every file-fingerprint-shaped mapping."""
    if _looks_like_artifact_record(node):
        yield prefix or "artifact", node
        return
    if isinstance(node, Mapping):
        for key, value in node.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_artifact_records(value, label)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            label = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _iter_artifact_records(value, label)


def _resolve_artifact_path(raw: str, search_dirs) -> Path:
    raw_path = Path(raw)
    if raw_path.is_absolute():
        return raw_path.resolve()
    candidates = []
    for base in search_dirs:
        if base is None:
            continue
        candidates.append(Path(base) / raw_path)
    # A moved study directory keeps relative paths relative to the original
    # working directory.  The basename in the manifest's own directory is a
    # safe last resort: it is only used when the exact candidates are absent.
    if search_dirs:
        candidates.append(Path(search_dirs[0]) / raw_path.name)
    seen = set()
    fallback = None
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if fallback is None:
            fallback = candidate
        if candidate.exists():
            return candidate.resolve()
    return (fallback or raw_path).resolve()


def verify_artifact_record(record, *, search_dirs=(), label: str | None = None) -> dict:
    """Verify one artifact record against the file it points at.

    ``sha256`` is authoritative when present; ``bytes`` catches replacement
    with a same-sized file only when no hash is recorded.  An ``exists: False``
    record is not a failure by itself (some optional checkpoints are absent),
    but it is reported so a caller can distinguish an intentional absence from
    a missing file the manifest claimed existed.
    """
    label = label or "artifact"
    result = {
        "label": label,
        "path": None,
        "resolved_path": None,
        "recorded_exists": None,
        "actual_exists": None,
        "ok": False,
        "problems": [],
        "warnings": [],
    }
    if not isinstance(record, Mapping):
        result["problems"].append(
            f"{label}: artifact record is not a mapping")
        return result
    raw = record.get("path")
    if not isinstance(raw, str) or not raw.strip():
        result["problems"].append(f"{label}: artifact record has no path")
        return result
    result["path"] = raw
    resolved = _resolve_artifact_path(raw, search_dirs)
    result["resolved_path"] = str(resolved)
    recorded_exists = record.get("exists", _MISSING)
    if recorded_exists is not _MISSING:
        result["recorded_exists"] = recorded_exists
        if not isinstance(recorded_exists, bool):
            result["problems"].append(
                f"{label}: exists must be a boolean, got {recorded_exists!r}")
            return result
    actual_exists = resolved.exists()
    result["actual_exists"] = actual_exists
    if recorded_exists is False:
        if actual_exists:
            result["warnings"].append(
                f"{label}: recorded as absent but the file exists")
        else:
            result["warnings"].append(f"{label}: recorded as absent")
        result["ok"] = not result["problems"]
        return result
    if not actual_exists:
        result["problems"].append(f"{label}: artifact is missing ({resolved})")
        return result
    if not resolved.is_file():
        result["problems"].append(
            f"{label}: artifact path is not a regular file ({resolved})")
        return result

    stat = resolved.stat()
    recorded_bytes = record.get("bytes")
    if recorded_bytes is not None:
        if not _is_int(recorded_bytes) or recorded_bytes < 0:
            result["problems"].append(
                f"{label}: bytes must be a non-negative integer, "
                f"got {recorded_bytes!r}")
        elif recorded_bytes != stat.st_size:
            result["problems"].append(
                f"{label}: size mismatch: recorded {recorded_bytes} bytes, "
                f"actual {stat.st_size} bytes")

    recorded_sha = record.get("sha256")
    if recorded_sha is not None:
        if (not isinstance(recorded_sha, str)
                or len(recorded_sha) != 64
                or any(char not in "0123456789abcdefABCDEF"
                       for char in recorded_sha)):
            result["problems"].append(
                f"{label}: sha256 is not a 64-character hex digest")
        else:
            digest = hashlib.sha256()
            try:
                with open(resolved, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
            except OSError as exc:
                result["problems"].append(
                    f"{label}: cannot read artifact: {exc}")
            else:
                actual_sha = digest.hexdigest()
                if actual_sha.lower() != recorded_sha.lower():
                    result["problems"].append(
                        f"{label}: sha256 mismatch: recorded "
                        f"{recorded_sha[:12]}..., actual {actual_sha[:12]}...")

    recorded_mtime = record.get("mtime")
    if recorded_mtime is not None:
        if not _is_finite_number(recorded_mtime):
            result["warnings"].append(
                f"{label}: mtime is not a finite number; skipped")
        elif abs(float(recorded_mtime) - stat.st_mtime) > 1e-6:
            result["warnings"].append(
                f"{label}: mtime differs from the recorded value "
                "(content fingerprint is authoritative)")

    if recorded_sha is None and recorded_bytes is None:
        result["warnings"].append(
            f"{label}: no content fingerprint (sha256/bytes); integrity "
            "cannot be verified")
    result["ok"] = not result["problems"]
    return result


def audit_manifest_artifacts(manifest, *, manifest_path, root) -> dict:
    """Audit every file record embedded in one run manifest."""
    manifest_path = Path(manifest_path)
    search_dirs = [manifest_path.parent, Path(root), Path.cwd()]
    records = []
    problems = []
    warnings = []
    for label, record in _iter_artifact_records(manifest):
        verified = verify_artifact_record(
            record, search_dirs=search_dirs, label=label)
        records.append(verified)
        for problem in verified["problems"]:
            problems.append(f"{manifest_path.name}:{problem}")
        for warning in verified["warnings"]:
            warnings.append(f"{manifest_path.name}:{warning}")
    return {
        "n_records": len(records),
        "n_ok": sum(1 for record in records if record["ok"]),
        "records": records,
        "problems": problems,
        "warnings": warnings,
    }


def _manifest_seed(manifest) -> int | None:
    if not isinstance(manifest, Mapping):
        return None
    seed = manifest.get("seed")
    return int(seed) if _is_int(seed) else None


def _manifest_results(manifest) -> dict:
    results = manifest.get("results") if isinstance(manifest, Mapping) else None
    return results if isinstance(results, dict) else {}


def _manifest_data_version(manifest) -> dict | None:
    data = manifest.get("data") if isinstance(manifest, Mapping) else None
    if not isinstance(data, dict):
        return None
    reports = data.get("split_reports")
    train = reports.get("train") if isinstance(reports, dict) else None
    train = train if isinstance(train, dict) else {}
    dataset = (train.get("dataset_revision_resolved")
               or data.get("dataset_revision_resolved"))
    tokenizer = (train.get("tokenizer_revision_resolved")
                 or data.get("tokenizer_revision_resolved"))
    if not isinstance(dataset, str) or not dataset.strip():
        return None
    if not isinstance(tokenizer, str) or not tokenizer.strip():
        return None
    return {
        "dataset_revision_resolved": dataset,
        "tokenizer_revision_resolved": tokenizer,
    }


def _audit_manifest(path: Path, root: Path, *, require_finished: bool,
                    require_clean_tree: bool) -> tuple[dict, dict | None]:
    data, error = _load_json(path)
    entry = {
        "path": str(path),
        "relative_path": _relative(path, root),
        "seed": None,
        "status": None,
        "git_commit": None,
        "git_dirty": None,
        "artifact_count": 0,
        "problems": [],
        "warnings": [],
    }
    if error is not None:
        entry["problems"].append(f"cannot read manifest: {error}")
        return entry, None
    if not isinstance(data, dict):
        entry["problems"].append("manifest is not a JSON object")
        return entry, None

    schema = data.get("schema")
    if schema is None:
        entry["warnings"].append("manifest has no schema field")
    elif schema != RUN_MANIFEST_SCHEMA:
        entry["warnings"].append(
            f"unexpected manifest schema {schema!r}")

    status = data.get("status")
    entry["status"] = status
    if require_finished and status != "finished":
        entry["problems"].append(
            f"status is {status!r}, not 'finished'")

    git = data.get("git")
    if isinstance(git, dict):
        commit = git.get("commit")
        dirty = git.get("dirty")
        entry["git_commit"] = commit
        entry["git_dirty"] = dirty
        if not isinstance(commit, str) or not commit.strip():
            entry["warnings"].append("git.commit is missing or empty")
        if not isinstance(dirty, bool):
            entry["warnings"].append("git.dirty is not a boolean")
        elif require_clean_tree and dirty:
            entry["problems"].append(
                "run was produced from a dirty working tree; the commit "
                "does not pin the executed code")
    else:
        entry["warnings"].append("manifest has no git block")

    seed = _manifest_seed(data)
    entry["seed"] = seed
    artifacts = data.get("artifacts")
    if artifacts is not None and not isinstance(artifacts, dict):
        entry["problems"].append("artifacts block is not a mapping")
    artifact_audit = audit_manifest_artifacts(
        data, manifest_path=path, root=root)
    entry["artifact_count"] = artifact_audit["n_records"]
    entry["problems"].extend(artifact_audit["problems"])
    entry["warnings"].extend(artifact_audit["warnings"])
    return entry, data


def _audit_aggregate(path: Path, root: Path, manifests_by_path: dict,
                     manifests_by_seed: dict) -> tuple[dict, dict | None]:
    data, error = _load_json(path)
    entry = {
        "path": str(path),
        "relative_path": _relative(path, root),
        "n_seeds": None,
        "problems": [],
        "warnings": [],
    }
    if error is not None:
        entry["problems"].append(f"cannot read aggregate: {error}")
        return entry, None
    if not isinstance(data, dict):
        entry["problems"].append("aggregate is not a JSON object")
        return entry, None

    per_seed = data.get("per_seed")
    if not isinstance(per_seed, list):
        entry["problems"].append("aggregate has no per_seed list")
        per_seed = []
    entry["n_seeds"] = len(per_seed)
    if _is_int(data.get("n_seeds")) and data["n_seeds"] != len(per_seed):
        entry["problems"].append(
            f"aggregate n_seeds={data['n_seeds']} does not match "
            f"{len(per_seed)} per_seed record(s)")

    declared = data.get("seeds")
    declared_set = set()
    if isinstance(declared, list):
        for seed in declared:
            if not _is_int(seed):
                entry["problems"].append(
                    f"declared seed {seed!r} is not an integer")
            else:
                declared_set.add(int(seed))
        if len(declared_set) != len(declared):
            entry["problems"].append("aggregate has duplicate declared seeds")
    elif declared is not None:
        entry["problems"].append("aggregate seeds field is not a list")

    if data.get("complete") is False:
        entry["problems"].append(
            "aggregate is explicitly marked incomplete")
    elif data.get("complete") is not True:
        entry["warnings"].append(
            "aggregate has no explicit complete=True flag")
    if data.get("certified") is False:
        entry["warnings"].append("aggregate is not certified")
    for key in ("missing_seeds", "invalid_seeds", "manifest_rejections"):
        value = data.get(key)
        if isinstance(value, list) and value:
            entry["problems"].append(
                f"aggregate reports non-empty {key}: {value!r}")
    for key in ("data_versions_complete", "data_versions_consistent",
                "shapes_complete", "shapes_consistent",
                "data_ledgers_complete"):
        if data.get(key) is False:
            entry["problems"].append(f"aggregate {key} is false")

    sibling_path = (path.parent / "run_manifest.json").resolve()
    sibling = manifests_by_path.get(str(sibling_path))
    if isinstance(sibling, dict):
        recorded = sibling.get("aggregate")
        if recorded is None:
            entry["warnings"].append(
                "sibling run_manifest.json does not embed the aggregate")
        elif recorded != data:
            entry["problems"].append(
                "sibling run_manifest.json embeds an aggregate that differs "
                "from aggregate.json")

    seen = set()
    per_seed_ids = set()
    for record in per_seed:
        if not isinstance(record, dict):
            entry["problems"].append(
                "aggregate contains a non-mapping per_seed record")
            continue
        seed = record.get("seed")
        if not _is_int(seed):
            entry["problems"].append(
                f"per_seed record has a malformed seed {seed!r}")
            continue
        seed = int(seed)
        if seed in seen:
            entry["problems"].append(
                f"aggregate has duplicate per_seed record for seed {seed}")
        seen.add(seed)
        per_seed_ids.add(seed)
        if record.get("rejected"):
            entry["problems"].append(
                f"seed {seed} is marked rejected: {record.get('rejected')}")
        manifest = manifests_by_seed.get(seed)
        if not isinstance(manifest, dict):
            entry["problems"].append(
                f"no per-seed run manifest found for seed {seed}")
            continue
        results = _manifest_results(manifest)
        for key in ("final_val_loss", "best_val_loss", "final_val_step",
                    "best_val_step", "test_loss", "n_params", "n_layers",
                    "dim", "n_heads", "n_loss_spikes",
                    "n_nonfinite_losses", "n_nonfinite_gradients",
                    "n_nonfinite_val_losses"):
            aggregate_value = record.get(key)
            if aggregate_value is None:
                continue
            manifest_value = results.get(key)
            if manifest_value != aggregate_value:
                entry["problems"].append(
                    f"seed {seed} {key} differs: aggregate "
                    f"{aggregate_value!r}, manifest {manifest_value!r}")
        aggregate_version = record.get("data_version")
        manifest_version = _manifest_data_version(manifest)
        if (aggregate_version is not None and manifest_version is not None
                and aggregate_version != manifest_version):
            entry["problems"].append(
                f"seed {seed} data_version differs: aggregate "
                f"{aggregate_version!r}, manifest {manifest_version!r}")

    if declared_set and per_seed_ids != declared_set:
        entry["problems"].append(
            "aggregate declared seeds do not match its per_seed records: "
            f"missing {sorted(declared_set - per_seed_ids)}, "
            f"extra {sorted(per_seed_ids - declared_set)}")
    return entry, data


def _audit_analysis_plan(path: Path, root: Path, *,
                         require_plan_lock: bool) -> tuple[dict, dict | None]:
    data, error = _load_json(path)
    entry = {
        "path": str(path),
        "relative_path": _relative(path, root),
        "fingerprint": None,
        "problems": [],
        "warnings": [],
    }
    if error is not None:
        entry["problems"].append(f"cannot read analysis plan: {error}")
        return entry, None
    if not isinstance(data, dict):
        entry["problems"].append("analysis plan is not a JSON object")
        return entry, None
    fingerprint = data.get("fingerprint")
    entry["fingerprint"] = fingerprint
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdefABCDEF"
                   for char in fingerprint)):
        entry["problems"].append(
            "analysis plan fingerprint is not a 64-character hex digest")
    else:
        try:
            recomputed = analysis_plan_fingerprint(data)
        except ValueError as exc:
            entry["problems"].append(
                f"analysis plan cannot be fingerprinted: {exc}")
        else:
            if recomputed != fingerprint:
                entry["problems"].append(
                    "analysis plan fingerprint does not match its payload: "
                    f"recorded {fingerprint[:12]}..., recomputed "
                    f"{recomputed[:12]}...")
    if data.get("schema") != ANALYSIS_PLAN_SCHEMA:
        entry["warnings"].append(
            f"unexpected analysis-plan schema {data.get('schema')!r}")
    if data.get("plan_changed") is True:
        message = (
            "analysis plan was changed after its first registration; "
            "the confirmatory family moved after results could be seen")
        if require_plan_lock:
            entry["problems"].append(message)
        else:
            entry["warnings"].append(message)

    study_path = path.parent / "study_level_tests.json"
    if study_path.exists():
        study, study_error = _load_json(study_path)
        if study_error is not None:
            entry["warnings"].append(
                f"cannot read study_level_tests.json: {study_error}")
        elif isinstance(study, dict):
            recorded = study.get("analysis_plan_fingerprint")
            if recorded is None:
                entry["warnings"].append(
                    "study_level_tests.json does not name the analysis plan")
            elif recorded != fingerprint:
                entry["problems"].append(
                    "study_level_tests.json was computed against a different "
                    f"analysis plan ({recorded!r})")
    return entry, data


def audit_study(root, *, require_finished: bool = True,
                require_clean_tree: bool = True,
                require_same_commit: bool = True,
                require_plan_lock: bool = True) -> dict:
    """Audit a study directory and return a structured report.

    The function never mutates the study.  ``ok`` is true only when every
    manifest, artifact, aggregate/seed link and analysis plan passes the
    selected checks.  Warnings describe unverifiable-but-not-failing claims
    (for example an artifact record that carries only a size, not a hash).
    """
    root = Path(root)
    if root.is_file():
        root = root.parent
    root = root.resolve()
    report = {
        "schema": STUDY_AUDIT_SCHEMA,
        "root": str(root),
        "ok": False,
        "options": {
            "require_finished": bool(require_finished),
            "require_clean_tree": bool(require_clean_tree),
            "require_same_commit": bool(require_same_commit),
            "require_plan_lock": bool(require_plan_lock),
        },
        "summary": {},
        "problems": [],
        "warnings": [],
        "manifests": [],
        "aggregates": [],
        "analysis_plans": [],
    }
    if not root.is_dir():
        report["problems"].append(f"study directory does not exist: {root}")
        report["summary"] = {
            "n_manifests": 0, "n_finished": 0, "n_artifacts": 0,
            "n_aggregates": 0, "n_analysis_plans": 0,
            "n_problems": len(report["problems"]),
            "n_warnings": 0,
        }
        return report

    manifest_paths = sorted(root.rglob("run_manifest.json"))
    if not manifest_paths:
        report["problems"].append(
            f"no run_manifest.json found under {root}")
    manifests_by_path = {}
    manifests_by_seed = {}
    for path in manifest_paths:
        entry, data = _audit_manifest(
            path, root, require_finished=require_finished,
            require_clean_tree=require_clean_tree)
        report["manifests"].append(entry)
        for problem in entry["problems"]:
            report["problems"].append(
                f"{entry['relative_path']}: {problem}")
        for warning in entry["warnings"]:
            report["warnings"].append(
                f"{entry['relative_path']}: {warning}")
        if isinstance(data, dict):
            manifests_by_path[str(path.resolve())] = data
            # A replication manifest records the seed of its first child, so
            # it must not be mistaken for that seed's own run.
            if "aggregate" not in data:
                seed = _manifest_seed(data)
                if seed is not None:
                    if seed in manifests_by_seed:
                        report["problems"].append(
                            f"multiple per-seed manifests claim seed {seed}: "
                            f"{manifests_by_seed[seed].get('__path__')} and "
                            f"{path}")
                    else:
                        stored = dict(data)
                        stored["__path__"] = str(path)
                        manifests_by_seed[seed] = stored

    commits = {entry["git_commit"] for entry in report["manifests"]
               if isinstance(entry["git_commit"], str)
               and entry["git_commit"].strip()}
    if require_same_commit and len(commits) > 1:
        report["problems"].append(
            "study manifests were produced at different commits: "
            f"{sorted(commits)}")
    if any(entry["git_commit"] is None for entry in report["manifests"]):
        report["warnings"].append(
            "at least one manifest does not record a git commit")

    for path in sorted(root.rglob("aggregate.json")):
        entry, _ = _audit_aggregate(
            path, root, manifests_by_path, manifests_by_seed)
        report["aggregates"].append(entry)
        for problem in entry["problems"]:
            report["problems"].append(
                f"{entry['relative_path']}: {problem}")
        for warning in entry["warnings"]:
            report["warnings"].append(
                f"{entry['relative_path']}: {warning}")

    for path in sorted(root.rglob("analysis_plan.json")):
        entry, _ = _audit_analysis_plan(
            path, root, require_plan_lock=require_plan_lock)
        report["analysis_plans"].append(entry)
        for problem in entry["problems"]:
            report["problems"].append(
                f"{entry['relative_path']}: {problem}")
        for warning in entry["warnings"]:
            report["warnings"].append(
                f"{entry['relative_path']}: {warning}")

    report["summary"] = {
        "n_manifests": len(report["manifests"]),
        "n_finished": sum(
            1 for entry in report["manifests"]
            if entry["status"] == "finished"),
        "n_artifacts": sum(
            entry["artifact_count"] for entry in report["manifests"]),
        "n_aggregates": len(report["aggregates"]),
        "n_analysis_plans": len(report["analysis_plans"]),
        "n_problems": len(report["problems"]),
        "n_warnings": len(report["warnings"]),
    }
    report["ok"] = not report["problems"]
    return report


__all__ = [
    "ANALYSIS_PLAN_SCHEMA",
    "STUDY_AUDIT_SCHEMA",
    "audit_manifest_artifacts",
    "audit_study",
    "verify_artifact_record",
]
