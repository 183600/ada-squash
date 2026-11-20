"""Pre-registration and timestamped commitment infrastructure.

A pre-registration is only credible if it cannot be modified after the fact.
This module provides tools for creating tamper-evident, timestamped commitments
to analysis plans, hypotheses, and design choices.

Key concepts:
- **Commitment**: A cryptographic hash of the pre-registration content, stored
  in an append-only log (git, blockchain, trusted timestamping service).
- **Verification**: Anyone can verify that a pre-registration existed at a
  specific time and has not been modified.
- **Integration**: Works with the existing `analysis_plan.json` fingerprint.

This module provides:
``create_preregistration``
    Create a pre-registration document with hypotheses, analysis plan, and design.
``commit_preregistration``
    Create a tamper-evident commitment (git tag, file, or external service).
``verify_preregistration``
    Verify that a pre-registration matches its commitment and timestamp.
``PreregistrationRegistry``
    A registry for managing multiple pre-registrations within a study.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).

__all__ = [
    "Preregistration",
    "PreregistrationRegistry",
    "commit_preregistration_file",
    "commit_preregistration_git",
    "create_preregistration",
    "timestamp_now",
    "verify_preregistration",
]


@dataclass(frozen=True)
class Preregistration:
    """A pre-registration document with integrity verification.

    Attributes
    ----------
    version: Schema version.
    title: Human-readable title.
    hypotheses: Dict mapping contrast IDs to hypothesis statements.
    analysis_plan: The analysis plan (contrasts, tests, corrections).
    design: Experimental design (schedule, exclusion policy, power plan).
    created_at: Unix timestamp of creation.
    author: Optional author identifier.
    fingerprint: SHA-256 of the canonical content (excluding timestamp/commit fields).
    commit_info: Optional commit/timestamp information added after commitment.
    """
    version: int = 1
    title: str = ""
    hypotheses: dict[str, str] = field(default_factory=dict)
    analysis_plan: dict = field(default_factory=dict)
    design: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    author: str = ""
    fingerprint: str = ""
    commit_info: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "version": self.version,
            "title": self.title,
            "hypotheses": self.hypotheses,
            "analysis_plan": self.analysis_plan,
            "design": self.design,
            "created_at": self.created_at,
            "author": self.author,
            "fingerprint": self.fingerprint,
            "commit_info": self.commit_info,
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, data: str) -> Preregistration:
        obj = json.loads(data)
        return cls(
            version=obj.get("version", 1),
            title=obj.get("title", ""),
            hypotheses=obj.get("hypotheses", {}),
            analysis_plan=obj.get("analysis_plan", {}),
            design=obj.get("design", {}),
            created_at=obj.get("created_at", 0.0),
            author=obj.get("author", ""),
            fingerprint=obj.get("fingerprint", ""),
            commit_info=obj.get("commit_info", {}),
        )

    def _compute_fingerprint(self) -> str:
        """Compute SHA-256 of the immutable content."""
        payload = json.dumps({
            "version": self.version,
            "title": self.title,
            "hypotheses": self.hypotheses,
            "analysis_plan": self.analysis_plan,
            "design": self.design,
            "created_at": self.created_at,
            "author": self.author,
        }, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def verify_fingerprint(self) -> bool:
        """Verify the fingerprint matches the content."""
        return self.fingerprint == self._compute_fingerprint()


def create_preregistration(
    title: str,
    hypotheses: Mapping[str, str],
    analysis_plan: Mapping[str, Any],
    design: Mapping[str, Any],
    author: str = "",
) -> Preregistration:
    """Create a pre-registration document.

    Parameters
    ----------
    title: Human-readable title (e.g., "DyT++ Ablation Study - Phase 1").
    hypotheses: Mapping from contrast ID to hypothesis statement.
    analysis_plan: The analysis plan (from build_analysis_plan or similar).
    design: Experimental design including schedule, exclusion policy, power plan.
    author: Optional author identifier.

    Returns
    -------
    Preregistration with computed fingerprint.
    """
    prereg = Preregistration(
        title=title,
        hypotheses=dict(hypotheses),
        analysis_plan=dict(analysis_plan),
        design=dict(design),
        author=author,
    )
    fingerprint = prereg._compute_fingerprint()
    return Preregistration(
        version=prereg.version,
        title=prereg.title,
        hypotheses=prereg.hypotheses,
        analysis_plan=prereg.analysis_plan,
        design=prereg.design,
        created_at=prereg.created_at,
        author=prereg.author,
        fingerprint=fingerprint,
    )


def timestamp_now() -> dict:
    """Get current timestamp with multiple formats."""
    now = time.time()
    return {
        "unix": now,
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "iso_local": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
    }


def commit_preregistration_git(
    prereg: Preregistration,
    repo_path: str = ".",
    tag_prefix: str = "prereg-",
    sign: bool = False,
) -> Preregistration:
    """Commit a pre-registration to git as a signed tag.

    This creates an immutable, timestamped record in the git history.
    The tag name includes the fingerprint for easy lookup.

    Parameters
    ----------
    prereg: The pre-registration to commit.
    repo_path: Path to git repository.
    tag_prefix: Prefix for the git tag.
    sign: Whether to create a signed tag (requires GPG).

    Returns
    -------
    Updated Preregistration with commit_info populated.
    """
    # Write pre-registration to a file
    prereg_dir = os.path.join(repo_path, ".preregistrations")
    os.makedirs(prereg_dir, exist_ok=True)

    filename = f"prereg_{prereg.fingerprint[:16]}.json"
    filepath = os.path.join(prereg_dir, filename)

    with open(filepath, "w") as f:
        f.write(prereg.to_json())

    # Git add and commit
    try:
        subprocess.run(["git", "add", filepath], cwd=repo_path, check=True, capture_output=True)
        commit_msg = f"Pre-registration: {prereg.title} ({prereg.fingerprint[:12]})"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_path, check=True, capture_output=True)

        # Create tag
        tag_name = f"{tag_prefix}{prereg.fingerprint[:16]}"
        tag_msg = f"Pre-registration: {prereg.title}\nFingerprint: {prereg.fingerprint}\nAuthor: {prereg.author}"
        tag_args = ["git", "tag"]
        if sign:
            tag_args.append("-s")
        tag_args.extend(["-m", tag_msg, tag_name])
        subprocess.run(tag_args, cwd=repo_path, check=True, capture_output=True)

        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, check=True, capture_output=True, text=True
        )
        commit_hash = result.stdout.strip()

        commit_info = {
            "method": "git",
            "commit_hash": commit_hash,
            "tag": tag_name,
            "filepath": filepath,
            "timestamp": timestamp_now(),
            "signed": sign,
        }

        return Preregistration(
            version=prereg.version,
            title=prereg.title,
            hypotheses=prereg.hypotheses,
            analysis_plan=prereg.analysis_plan,
            design=prereg.design,
            created_at=prereg.created_at,
            author=prereg.author,
            fingerprint=prereg.fingerprint,
            commit_info=commit_info,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git commit failed: {e.stderr.decode() if e.stderr else str(e)}")


def commit_preregistration_file(
    prereg: Preregistration,
    output_dir: str = ".preregistrations",
) -> Preregistration:
    """Save pre-registration to a file with timestamp (no git).

    This is a fallback for environments without git. The file system
    modification time serves as a weak timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"prereg_{prereg.fingerprint[:16]}_{int(prereg.created_at)}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        f.write(prereg.to_json())

    commit_info = {
        "method": "file",
        "filepath": filepath,
        "timestamp": timestamp_now(),
        "signed": False,
    }

    return Preregistration(
        version=prereg.version,
        title=prereg.title,
        hypotheses=prereg.hypotheses,
        analysis_plan=prereg.analysis_plan,
        design=prereg.design,
        created_at=prereg.created_at,
        author=prereg.author,
        fingerprint=prereg.fingerprint,
        commit_info=commit_info,
    )


def verify_preregistration(
    prereg: Preregistration,
    expected_fingerprint: Optional[str] = None,
    check_git: bool = False,
    repo_path: str = ".",
) -> dict:
    """Verify a pre-registration's integrity and optionally its git commitment.

    Returns a verification report with:
    - fingerprint_valid: content matches fingerprint
    - commit_verified: git commit/tag exists and matches
    - timestamp: recorded creation time
    """
    result = {
        "fingerprint_valid": prereg.verify_fingerprint(),
        "commit_verified": False,
        "timestamp": prereg.created_at,
        "fingerprint": prereg.fingerprint,
    }

    if expected_fingerprint:
        result["fingerprint_matches_expected"] = prereg.fingerprint == expected_fingerprint

    if check_git and prereg.commit_info.get("method") == "git":
        try:
            tag = prereg.commit_info.get("tag")
            commit_hash = prereg.commit_info.get("commit_hash")
            if tag:
                # Verify tag exists
                subprocess.run(
                    ["git", "rev-parse", tag], cwd=repo_path, check=True, capture_output=True
                )
            if commit_hash:
                # Verify commit exists
                subprocess.run(
                    ["git", "cat-file", "-t", commit_hash], cwd=repo_path, check=True, capture_output=True
                )
            result["commit_verified"] = True
        except subprocess.CalledProcessError:
            result["commit_verified"] = False
            result["commit_error"] = "Git tag or commit not found"

    result["all_checks_pass"] = result["fingerprint_valid"] and (
        not check_git or result["commit_verified"]
    )
    return result


class PreregistrationRegistry:
    """Registry for managing multiple pre-registrations in a study.

    Each pre-registration gets a unique ID and can be verified independently.
    """

    def __init__(self, storage_dir: str = ".preregistrations"):
        self.storage_dir = storage_dir
        self._index: dict[str, Preregistration] = {}
        self._load_index()

    def _load_index(self) -> None:
        if not os.path.exists(self.storage_dir):
            return
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath) as f:
                        prereg = Preregistration.from_json(f.read())
                    self._index[prereg.fingerprint] = prereg
                except (json.JSONDecodeError, KeyError):
                    pass  # Skip malformed files

    def add(self, prereg: Preregistration) -> str:
        """Add a pre-registration to the registry."""
        self._index[prereg.fingerprint] = prereg
        self._save(prereg)
        return prereg.fingerprint

    def _save(self, prereg: Preregistration) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
        filename = f"prereg_{prereg.fingerprint[:16]}.json"
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "w") as f:
            f.write(prereg.to_json())

    def get(self, fingerprint: str) -> Optional[Preregistration]:
        return self._index.get(fingerprint)

    def list_all(self) -> list[Preregistration]:
        return list(self._index.values())

    def verify_all(self, check_git: bool = False, repo_path: str = ".") -> list[dict]:
        return [
            {**verify_preregistration(pr, check_git=check_git, repo_path=repo_path), "fingerprint": fp}
            for fp, pr in self._index.items()
        ]