from __future__ import annotations
import argparse
import json
import math
import numbers
import sys
from array import array
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash._validation import is_boolean_scalar as _is_boolean_scalar
from adasquash.metrics import wilson_interval
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest
from scripts.data_pipeline import (DocumentCleaner, _is_token_row,
                                   canonical_document_hash, document_hash,
                                   extract_group_value, group_hash, load_document_stream)


def _strict_bool_option(value, name: str, *, allow_none: bool = False):
    """A boolean flag, rejecting truthy/falsy stand-ins.

    ``bool("false")`` is True: a malformed JSON/YAML spelling must not turn a
    sampled scan into an exhaustive one (or vice versa). ``None`` is kept only
    for callers whose field distinguishes "not recorded" from False.
    """
    if value is None and allow_none:
        return None
    if not _is_boolean_scalar(value):
        raise ValueError(f"{name} must be a bool, got {value!r}")
    return bool(value)


def _strict_int_option(value, name: str, minimum: int) -> int:
    """An integer-valued measurement, rejecting bools and fractional values.

    ``int(2.5)`` silently becomes 2; for a sequence length, batch count or
    evaluation cadence that changes the geometry being audited while the
    report still looks like it described the requested protocol. A bool is
    not a count either.
    """
    if _is_boolean_scalar(value):
        raise ValueError(f"{name} must be an integer, got bool {value!r}")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if converted != value:
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {converted}")
    return converted


