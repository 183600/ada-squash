"""Deterministic, crash-safe JSON artifact writing.

Experiment reports are read by later audit and analysis steps, so a reader
must never observe a half-written JSON document.  ``save_json_atomic`` writes
to a temporary file in the destination directory and renames it into place
with ``os.replace``; a serialization error therefore leaves the previous
report intact instead of truncating it.

``canonical_json_dumps`` fixes key ordering and separators.  Two equivalent
Python values then produce byte-identical JSON, which makes fingerprints and
diff-based provenance checks meaningful.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def canonical_json_dumps(value, *, indent: int | None = None,
                         sort_keys: bool = True,
                         allow_nan: bool = True) -> str:
    """Serialize ``value`` deterministically.

    ``allow_nan`` defaults to True to match Python's ``json.dump`` and keep
    existing diagnostic reports writable.  Fingerprint callers should pass
    ``allow_nan=False`` so a non-finite value cannot silently produce a
    non-standard JSON document.
    """
    separators = (",", ":") if indent is None else None
    return json.dumps(
        value,
        sort_keys=sort_keys,
        indent=indent,
        separators=separators,
        ensure_ascii=False,
        allow_nan=allow_nan,
        default=str,
    )


def save_json_atomic(path, value, *, indent: int = 2,
                     sort_keys: bool = True,
                     allow_nan: bool = True) -> Path:
    """Write ``value`` to ``path`` atomically and return the destination.

    The temporary file is created in the same directory (a rename across
    filesystems is not atomic), flushed and fsynced before replacement, and
    removed on failure.  The destination's parent directories are created if
    needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json_dumps(value, indent=indent, sort_keys=sort_keys,
                                allow_nan=allow_nan)
    if not text.endswith("\n"):
        text += "\n"

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        finally:
            raise
    return path


__all__ = ["canonical_json_dumps", "save_json_atomic"]
