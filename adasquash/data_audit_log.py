"""Deterministic audit log for the data-processing pipeline.

The reproducibility contract covers the *training* stream (seed, RNG,
torch version, TrainStreamFingerprint) but not the *data-processing*
decisions that precede it: which documents the cleaner dropped, which
duplicates the dedup set removed, which cross-split ledgers excluded a
held-out document, and which boundary the chronological split placed.  Two
runs that record the same seed and the same data revision can still train on
different documents if the cleaning parameters or the ledger contents differ
silently.

This module records one audit entry per pipeline step
(load, split, clean, dedup, exclude, pack).  Each entry carries the input and
output document counts, the drop-reason counter, content digests of the
documents that entered and left the step, and a frozen parameter snapshot.
The log can be serialised to JSON and embedded in a run manifest, and
:func:`verify_determinism` compares two logs so an auditor can prove that the
same options produced the same processing decisions.

All functions are pure with respect to the log itself: they never read from or
write to a dataset, so the log is cheap to produce and can be validated
without re-running the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from ._validation import is_boolean_scalar as _is_boolean_scalar
# Reproducibility: Set random seeds
import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

DATA_AUDIT_LOG_SCHEMA = "adasquash.data_audit_log.v1"

_PIPELINE_STEPS = (
    "load",
    "split",
    "clean",
    "dedup",
    "exclude",
    "pack",
)


def _digest_count(counter: Counter) -> str:
    """Stable digest of a drop-reason counter."""
    payload = json.dumps(
        {key: int(value) for key, value in sorted(counter.items())},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_texts(texts: Iterable[str]) -> str:
    """Stable digest of a sequence of document texts."""
    hasher = hashlib.sha256()
    for text in texts:
        hasher.update(text.encode("utf-8", "replace"))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def _as_int(value, minimum: int = 0) -> int:
    """Validate a non-negative integer count without silent coercion."""
    if _is_boolean_scalar(value) or not isinstance(value, int):
        msg = f"count must be an integer, got {value!r}"
        raise ValueError(msg)
    if value < minimum:
        msg = f"count must be >= {minimum}, got {value}"
        raise ValueError(msg)
    return value


@dataclass(frozen=True)
class DataAuditEntry:
    """One deterministic audit entry for a single pipeline step.

    Attributes
    ----------
    step:
        Pipeline step name: one of ``load``, ``split``, ``clean``, ``dedup``,
        ``exclude``, ``pack``.
    input_count:
        Number of documents that entered the step.
    output_count:
        Number of documents that left the step.
    drop_reasons:
        Mapping from drop reason to count (empty when the step does not drop
        documents).
    input_hash:
        SHA-256 of the concatenated input document texts (empty string when
        not recorded).
    output_hash:
        SHA-256 of the concatenated output document texts (empty string when
        not recorded).
    parameters:
        Frozen parameter snapshot used by this step.
    timestamp:
        Unix timestamp when the entry was created.
    """

    step: str
    input_count: int
    output_count: int
    drop_reasons: dict[str, int] = field(default_factory=dict)
    input_hash: str = ""
    output_hash: str = ""
    parameters: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.step not in _PIPELINE_STEPS:
            msg = f"unknown pipeline step {self.step!r}"
            raise ValueError(msg)
        _as_int(self.input_count)
        _as_int(self.output_count)
        for key, value in self.drop_reasons.items():
            if not isinstance(key, str) or not key.strip():
                msg = f"drop reason must be a non-empty string, got {key!r}"
                raise ValueError(msg)
            _as_int(value, 0)
        object.__setattr__(self, "drop_reasons", dict(sorted(self.drop_reasons.items())))
        if self.output_count > self.input_count:
            msg = (
                f"step {self.step!r} produced {self.output_count} documents "
                f"from {self.input_count} input documents"
            )
            raise ValueError(msg)
        object.__setattr__(self, "timestamp", float(self.timestamp))

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe mapping."""
        return {
            "step": self.step,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "drop_reasons": self.drop_reasons,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "dropped": self.input_count - self.output_count,
            "drop_rate": (
                (self.input_count - self.output_count) / self.input_count
                if self.input_count
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DataAuditEntry:
        """Reconstruct an entry from a JSON-safe mapping."""
        if not isinstance(data, dict):
            msg = "audit entry must be a mapping"
            raise TypeError(msg)
        return cls(
            step=str(data["step"]),
            input_count=int(data["input_count"]),
            output_count=int(data["output_count"]),
            drop_reasons={str(k): int(v) for k, v in data.get("drop_reasons", {}).items()},
            input_hash=str(data.get("input_hash", "")),
            output_hash=str(data.get("output_hash", "")),
            parameters=data.get("parameters", {}) or {},
            timestamp=float(data.get("timestamp", time.time())),
        )


class DataAuditLog:
    """Ordered container of pipeline-step audit entries.

    Entries are appended in pipeline order.  The log can be serialised to a
    JSON-safe mapping (and embedded in a run manifest) and compared against
    another log with :meth:`verify_determinism`.
    """

    def __init__(self, pipeline_id: str | None = None) -> None:
        self.pipeline_id = (
            pipeline_id or hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()[:16]
        )
        self._entries: list[DataAuditEntry] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterable[DataAuditEntry]:
        return iter(self._entries)

    def append(self, entry: DataAuditEntry) -> None:
        """Append one audit entry, preserving pipeline order."""
        if not isinstance(entry, DataAuditEntry):
            msg = f"audit log entry must be a DataAuditEntry, got {type(entry).__name__}"
            raise TypeError(msg)
        self._entries.append(entry)

    def add_step(  # noqa: PLR0913
        self,
        step: str,
        input_count: int,
        output_count: int,
        *,
        drop_reasons: dict[str, int] | None = None,
        input_texts: Iterable[str] | None = None,
        output_texts: Iterable[str] | None = None,
        parameters: dict | None = None,
    ) -> DataAuditEntry:
        """Create and append one audit entry for a pipeline step."""
        entry = DataAuditEntry(
            step=step,
            input_count=input_count,
            output_count=output_count,
            drop_reasons=dict(drop_reasons or {}),
            input_hash=_digest_texts(input_texts or ()),
            output_hash=_digest_texts(output_texts or ()),
            parameters=dict(parameters or {}),
        )
        self.append(entry)
        return entry

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe mapping embeddable in a run manifest."""
        return {
            "schema": DATA_AUDIT_LOG_SCHEMA,
            "pipeline_id": self.pipeline_id,
            "n_entries": len(self._entries),
            "entries": [entry.to_dict() for entry in self._entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> DataAuditLog:
        """Reconstruct a log from a JSON-safe mapping."""
        if not isinstance(data, dict):
            msg = "audit log must be a mapping"
            raise TypeError(msg)
        log = cls(pipeline_id=str(data.get("pipeline_id", "")))
        for item in data.get("entries", []):
            log.append(DataAuditEntry.from_dict(item))
        return log

    def to_json(self, indent: int = 2, sort_keys: bool = True) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys, default=str)

    @classmethod
    def from_json(cls, text: str) -> DataAuditLog:
        """Reconstruct a log from a JSON string."""
        return cls.from_dict(json.loads(text))

    def steps(self) -> list[str]:
        """Return the step names in order."""
        return [entry.step for entry in self._entries]

    def summary(self) -> dict:
        """Aggregate drop reasons and counts across all steps."""
        total_input = sum(entry.input_count for entry in self._entries)
        total_output = sum(entry.output_count for entry in self._entries)
        aggregate: dict[str, int] = {}
        for entry in self._entries:
            for reason, count in entry.drop_reasons.items():
                aggregate[reason] = aggregate.get(reason, 0) + int(count)
        return {
            "pipeline_id": self.pipeline_id,
            "n_entries": len(self._entries),
            "steps": self.steps(),
            "total_input_documents": total_input,
            "total_output_documents": total_output,
            "total_dropped": total_input - total_output,
            "aggregate_drop_reasons": dict(sorted(aggregate.items())),
            "first_step": self._entries[0].step if self._entries else None,
            "last_step": self._entries[-1].step if self._entries else None,
        }

    def verify_determinism(self, other: DataAuditLog) -> dict:
        """Compare this log with another for deterministic equality.

        Two logs are deterministic-identical when they have the same pipeline
        id, the same steps in the same order, and every entry matches on
        input/output counts, drop reasons, input/output hashes and parameters.
        Timestamps are deliberately excluded: the same processing decisions
        made at different wall-clock times are still deterministic.
        """
        problems: list[str] = []
        if self.pipeline_id != other.pipeline_id:
            problems.append(f"pipeline id differs: {self.pipeline_id} vs {other.pipeline_id}")
        if self.steps() != other.steps():
            problems.append(f"pipeline step order differs: {self.steps()} vs {other.steps()}")
        if len(self._entries) != len(other._entries):
            problems.append(f"entry count differs: {len(self._entries)} vs {len(other._entries)}")
        for index, (a, b) in enumerate(zip(self._entries, other._entries, strict=False)):
            label = f"entry {index} ({a.step})"
            if a.input_count != b.input_count:
                problems.append(f"{label}: input_count {a.input_count} vs {b.input_count}")
            if a.output_count != b.output_count:
                problems.append(f"{label}: output_count {a.output_count} vs {b.output_count}")
            if a.drop_reasons != b.drop_reasons:
                problems.append(f"{label}: drop_reasons {a.drop_reasons} vs {b.drop_reasons}")
            if a.input_hash != b.input_hash:
                problems.append(
                    f"{label}: input_hash {a.input_hash[:16]}... vs {b.input_hash[:16]}..."
                )
            if a.output_hash != b.output_hash:
                problems.append(
                    f"{label}: output_hash {a.output_hash[:16]}... vs {b.output_hash[:16]}..."
                )
            if _parameters_canonical(a.parameters) != _parameters_canonical(b.parameters):
                problems.append(f"{label}: parameters differ")
        return {
            "deterministic": not problems,
            "problems": problems,
            "n_entries_a": len(self._entries),
            "n_entries_b": len(other._entries),
        }


def _parameters_canonical(parameters: dict) -> str:
    """Stable canonical form of a parameter snapshot."""
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)


def audit_step(  # noqa: PLR0913
    log: DataAuditLog,
    step: str,
    input_count: int,
    output_count: int,
    *,
    drop_reasons: dict[str, int] | None = None,
    input_texts: Iterable[str] | None = None,
    output_texts: Iterable[str] | None = None,
    parameters: dict | None = None,
) -> DataAuditEntry:
    """Convenience: append one audit entry to ``log`` for a pipeline step."""
    return log.add_step(
        step=step,
        input_count=input_count,
        output_count=output_count,
        drop_reasons=drop_reasons,
        input_texts=input_texts,
        output_texts=output_texts,
        parameters=parameters,
    )


__all__ = [
    "DATA_AUDIT_LOG_SCHEMA",
    "DataAuditEntry",
    "DataAuditLog",
    "audit_step",
]
