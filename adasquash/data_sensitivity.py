"""Data-processing sensitivity analysis for the leakage-safe research pipeline.

A confirmatory comparison between normalization layers is only as stable as
the data-processing choices behind it.  Changing ``min_doc_chars``,
``max_doc_chars``, ``dedup``, ``val_frac`` or ``test_frac`` can change the
effective token budget, the cross-split overlap rate and the train/held-out
distribution shift - any of which can flip a paired arm contrast without the
researcher noticing.

This module sweeps the data-processing parameter grid, runs the same cleaning
and split protocol under every combination, and reports how much each reported
statistic moves.  A statistic whose range across the grid is large relative to
the arm-level effect size is a warning that the conclusion is sensitive to a
data-processing choice rather than to the module under study.

The pure functions here take a ``clean_and_split`` callable supplied by the
caller (so this module never pulls in the HuggingFace dataset stream itself)
and return JSON-safe mappings that can be embedded in a run manifest or an
analysis plan.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Callable

from ._validation import is_boolean_scalar as _is_boolean_scalar
# NOTE: Module-level seed setting removed to prevent silent RNG state
# reset on import. Use the centralized seed function at the experiment
# entry point instead (e.g., repro.set_all_seeds / seed_everything).


def _as_float(value) -> float | None:
    """Coerce a recorded statistic to a finite float, or return None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _option_key(options: dict, name: str) -> str:
    """Stable JSON key for one parameter setting."""
    value = options.get(name)
    if isinstance(value, bool):
        return f"{name}={value!s}"
    if value is None:
        return f"{name}=None"
    return f"{name}={json.dumps(value, sort_keys=True, default=str)}"


def parameter_sweep(
    base_options: dict,
    parameter_grids: dict[str, list],
    clean_and_split: Callable[[dict], dict],
) -> dict:
    """Run the cleaning/split protocol under every parameter combination.

    ``parameter_grids`` maps a data-option name to the list of values to try.
    The Cartesian product is taken, each combination is merged over
    ``base_options``, validated for key presence (not for values - the caller's
    ``clean_and_split`` is responsible for rejecting bad values) and run once.
    Every combination receives a deterministic run id derived from its sorted
    parameter string, so the sweep is reproducible and its results can be
    embedded in a pre-registration.

    Returns a mapping with one entry per combination plus an aggregate
    ``per_statistic`` block that summarises the range each reported statistic
    travels across the grid.
    """
    if not isinstance(base_options, dict):
        msg = "base_options must be a mapping"
        raise TypeError(msg)
    if not isinstance(parameter_grids, dict) or not parameter_grids:
        msg = "parameter_grids must be a non-empty mapping"
        raise TypeError(msg)
    if not callable(clean_and_split):
        msg = "clean_and_split must be callable"
        raise TypeError(msg)

    grid_names = sorted(parameter_grids)
    for name in grid_names:
        values = parameter_grids[name]
        if not isinstance(values, (list, tuple)) or not values:
            msg = f"parameter grid for {name!r} must be a non-empty sequence, got {values!r}"
            raise ValueError(msg)

    combinations: list[dict] = []
    per_combination: list[dict] = []
    for values_tuple in itertools.product(
        *(parameter_quiver(name, parameter_grids[name]) for name in grid_names)
    ):
        options = dict(base_options)
        for name, value in zip(grid_names, values_tuple, strict=False):
            options[name] = value
        run_id = _run_id(options, grid_names)
        try:
            stats = clean_and_split(options)
        except Exception as exc:
            combinations.append(
                {
                    "run_id": run_id,
                    "options": {name: options[name] for name in grid_names},
                    "error": f"{type(exc).__name__}: {exc}",
                    "stats": None,
                }
            )
            continue
        if not isinstance(stats, dict):
            msg = f"clean_and_split returned {type(stats).__name__}, expected a mapping"
            raise TypeError(msg)
        entry = {
            "run_id": run_id,
            "options": {name: options[name] for name in grid_names},
            "error": None,
            "stats": stats,
        }
        per_combination.append(entry)
        combinations.append(entry)

    return {
        "schema": "adasquash.data_sensitivity.v1",
        "grid": {name: list(parameter_grids[name]) for name in grid_names},
        "base_options": dict(base_options),
        "n_combinations": len(combinations),
        "n_errored": sum(1 for entry in combinations if entry["error"] is not None),
        "combinations": combinations,
        "per_statistic": _statistic_sensitivity(per_combination),
    }


def parameter_quiver(name: str, values: list) -> list:
    """Return the value list for one grid dimension, unchanged."""
    return list(values)


def _run_id(options: dict, grid_names: list[str]) -> str:
    """Deterministic run id for one parameter combination."""
    payload = json.dumps(
        {name: options[name] for name in grid_names},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _statistic_sensitivity(combinations: list[dict]) -> dict:
    """Summarise how far each statistic moves across the grid."""
    successful = [
        entry for entry in combinations if entry["error"] is None and entry["stats"] is not None
    ]
    if not successful:
        return {}
    stat_names: set[str] = set()
    for entry in successful:
        for key in entry["stats"]:
            if _as_float(entry["stats"].get(key)) is not None:
                stat_names.add(key)
    out: dict[str, dict] = {}
    for name in sorted(stat_names):
        values = []
        for entry in successful:
            value = _as_float(entry["stats"].get(name))
            if value is not None:
                values.append(value)
        if not values:
            continue
        mean = sum(values) / max(len(values), 1) if values else float("nan")
        population_variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1) if values else float("nan")
        std = math.sqrt(population_variance)
        cv = (std / abs(mean)) if mean else None
        data_range = max(values) - min(values)
        relative_range = (data_range / abs(mean)) if mean else None
        out[name] = {
            "n": len(values),
            "min": min(values),
            "max": max(values),
            "mean": mean,
            "std": std,
            "coefficient_of_variation": cv,
            "range": data_range,
            "relative_range": relative_range,
            "values_by_run": {
                entry["run_id"]: _as_float(entry["stats"].get(name)) for entry in successful
            },
        }
    return out


def _validate_split_position_args(
    base_options: dict,
    perturbations: list[float],
    clean_and_split: Callable[[dict], dict],
) -> None:
    """Validate arguments for ``split_position_robustness``."""
    if not isinstance(base_options, dict):
        msg = "base_options must be a mapping"
        raise TypeError(msg)
    if not isinstance(perturbations, (list, tuple)) or not perturbations:
        msg = "perturbations must be a non-empty sequence"
        raise TypeError(msg)
    if not callable(clean_and_split):
        msg = "clean_and_split must be callable"
        raise TypeError(msg)
    for offset in perturbations:
        if _is_boolean_scalar(offset) or not isinstance(offset, (int, float)):
            msg = f"perturbation must be a number, got {offset!r}"
            raise TypeError(msg)
        if not math.isfinite(float(offset)):
            msg = f"perturbation must be finite, got {offset!r}"
            raise ValueError(msg)


def _run_single_perturbation(
    offset: float,
    base_options: dict,
    clean_and_split: Callable[[dict], dict],
    baseline_stats: dict,
) -> dict:
    """Run one split-position perturbation and return its entry."""
    options = dict(base_options)
    val_frac = options.get("val_frac", 0.0)
    test_frac = options.get("test_frac", 0.0)
    new_val = _clip_fraction(val_frac + offset)
    new_test = _clip_fraction(test_frac + offset)
    if new_val + new_test >= 1.0:
        return {
            "offset": offset,
            "error": "perturbed fractions leave no training documents",
            "stats": None,
        }
    options["val_frac"] = new_val
    options["test_frac"] = new_test
    run_id = hashlib.sha256(
        json.dumps(options, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    try:
        stats = clean_and_split(options)
    except Exception as exc:
        return {
            "run_id": run_id,
            "offset": offset,
            "options": {"val_frac": new_val, "test_frac": new_test},
            "error": f"{type(exc).__name__}: {exc}",
            "stats": None,
        }
    if not isinstance(stats, dict):
        msg = f"clean_and_split returned {type(stats).__name__}, expected a mapping"
        raise TypeError(msg)
    return {
        "run_id": run_id,
        "offset": offset,
        "error": None,
        "options": {"val_frac": new_val, "test_frac": new_test},
        "stats": stats,
        "delta_from_baseline": _statistic_deltas(baseline_stats, stats),
    }


def _split_position_sensitivity(
    entries: list[dict],
    baseline_stats: dict,
) -> dict[str, dict]:
    """Build the per-statistic sensitivity block from perturbation entries."""
    successful = [
        entry for entry in entries if entry["error"] is None and entry["stats"] is not None
    ]
    sensitivity: dict[str, dict] = {}
    if not successful or not baseline_stats:
        return sensitivity
    stat_names = {key for key in baseline_stats if _as_float(baseline_stats.get(key)) is not None}
    for name in sorted(stat_names):
        base_value = _as_float(baseline_stats.get(name))
        if base_value is None:
            continue
        deltas = []
        for entry in successful:
            value = _as_float(entry["stats"].get(name))
            if value is not None:
                deltas.append(value - base_value)
        if deltas:
            sensitivity[name] = {
                "baseline": base_value,
                "min_delta": min(deltas),
                "max_delta": max(deltas),
                "max_abs_delta": max(abs(delta) for delta in deltas),
                "mean_delta": (sum(deltas) / max(len(deltas), 1)) if deltas else float("nan"),
            }
    return sensitivity


def split_position_robustness(
    base_options: dict,
    perturbations: list[float],
    clean_and_split: Callable[[dict], dict],
) -> dict:
    """Test whether conclusions hold when the held-out split boundary moves.

    ``perturbations`` are additive offsets applied to both ``val_frac`` and
    ``test_frac`` (clipped to keep each fraction in [0, 1) and their sum below
    1).  For each offset the protocol is re-run and the held-out statistics are
    compared against the unperturbed baseline, so a researcher can see whether
    an arm-level advantage survives a different chronology cut.

    Returns one entry per perturbation plus a ``sensitivity`` block with the
    maximum absolute change per statistic.
    """
    _validate_split_position_args(base_options, perturbations, clean_and_split)

    baseline = dict(base_options)
    baseline_stats = clean_and_split(baseline)
    if not isinstance(baseline_stats, dict):
        msg = f"clean_and_split returned {type(baseline_stats).__name__}, expected a mapping"
        raise TypeError(msg)

    entries: list[dict] = [
        _run_single_perturbation(offset, base_options, clean_and_split, baseline_stats)
        for offset in perturbations
    ]

    sensitivity = _split_position_sensitivity(entries, baseline_stats)

    return {
        "schema": "adasquash.split_position_robustness.v1",
        "baseline_options": dict(base_options),
        "baseline_stats": baseline_stats,
        "perturbations": [float(offset) for offset in perturbations],
        "entries": entries,
        "sensitivity": sensitivity,
    }


def _clip_fraction(value: float) -> float:
    """Clip a split fraction into [0, 1)."""
    if value < 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0 - 1e-9
    return value


def _statistic_deltas(baseline: dict, other: dict) -> dict:
    """Absolute and relative change of every shared numeric statistic."""
    deltas: dict[str, dict] = {}
    for key in sorted(set(baseline) & set(other)):
        base = _as_float(baseline.get(key))
        value = _as_float(other.get(key))
        if base is None or value is None:
            continue
        delta = value - base
        relative = (delta / base) if base else None
        deltas[key] = {
            "baseline": base,
            "perturbed": value,
            "delta": delta,
            "relative": relative,
        }
    return deltas


def _flag_per_statistic(per_statistic: dict, relative_range_threshold: float) -> dict[str, dict]:
    """Flag statistics whose grid-level CV or relative range exceeds threshold."""
    flags: dict[str, dict] = {}
    if not isinstance(per_statistic, dict):
        return flags
    for name, summary in per_statistic.items():
        if not isinstance(summary, dict):
            continue
        cv = summary.get("coefficient_of_variation")
        relative_range = summary.get("relative_range")
        triggered: list[str] = []
        if cv is not None and cv > relative_range_threshold:
            triggered.append(f"coefficient_of_variation={cv:.4f} > {relative_range_threshold:.4f}")
        if relative_range is not None and relative_range > relative_range_threshold:
            triggered.append(
                f"relative_range={relative_range:.4f} > {relative_range_threshold:.4f}"
            )
        if triggered:
            flags[name] = {
                "source": "parameter_grid",
                "coefficient_of_variation": cv,
                "relative_range": relative_range,
                "triggered": triggered,
            }
    return flags


def _flag_sensitivity(sensitivity: dict, max_abs_delta_threshold: float) -> dict[str, dict]:
    """Flag statistics whose split-position max absolute delta exceeds threshold."""
    flags: dict[str, dict] = {}
    if not isinstance(sensitivity, dict):
        return flags
    for name, summary in sensitivity.items():
        if not isinstance(summary, dict):
            continue
        max_abs_delta = summary.get("max_abs_delta")
        if max_abs_delta_threshold is None or max_abs_delta is None:
            continue
        if not (max_abs_delta > max_abs_delta_threshold):
            continue
        triggered = [f"max_abs_delta={max_abs_delta:.6f} > {max_abs_delta_threshold:.6f}"]
        flags[name] = {
            "source": "split_position",
            "coefficient_of_variation": None,
            "relative_range": None,
            "triggered": triggered,
            "split_position": {
                "baseline": summary.get("baseline"),
                "min_delta": summary.get("min_delta"),
                "max_delta": summary.get("max_delta"),
                "max_abs_delta": max_abs_delta,
                "mean_delta": summary.get("mean_delta"),
                "triggered": triggered,
            },
        }
    return flags


def _merge_split_position_flag(
    flags: dict[str, dict], name: str, summary: dict, triggered: list[str]
) -> None:
    """Merge a split-position flag into an existing parameter-grid flag."""
    if name in flags:
        flags[name]["split_position"] = {
            "baseline": summary.get("baseline"),
            "min_delta": summary.get("min_delta"),
            "max_delta": summary.get("max_delta"),
            "max_abs_delta": summary.get("max_abs_delta"),
            "mean_delta": summary.get("mean_delta"),
            "triggered": triggered,
        }


def flag_sensitive_statistics(
    report: dict,
    *,
    relative_range_threshold: float = 0.05,
    max_abs_delta_threshold: float | None = None,
) -> dict:
    """Identify statistics whose values are sensitive to data-processing choices.

    A statistic is flagged when its coefficient of variation or relative range
    across the parameter grid exceeds ``relative_range_threshold``, or when its
    maximum absolute change across split-position perturbations exceeds
    ``max_abs_delta_threshold``.  Flagged statistics should be reported
    alongside the arm-level contrast so reviewers can judge whether the
    conclusion depends on a particular data-processing setting.

    Returns the subset of statistics that exceed either threshold, with the
    evidence that triggered each flag.
    """
    if not isinstance(report, dict):
        msg = "report must be a mapping"
        raise TypeError(msg)
    if _is_boolean_scalar(relative_range_threshold) or not isinstance(
        relative_range_threshold, (int, float)
    ):
        msg = f"relative_range_threshold must be a number, got {relative_range_threshold!r}"
        raise TypeError(msg)
    relative_range_threshold = float(relative_range_threshold)
    if not 0.0 <= relative_range_threshold <= 1.0:
        msg = f"relative_range_threshold must be in [0, 1], got {relative_range_threshold}"
        raise ValueError(msg)
    if max_abs_delta_threshold is not None:
        if _is_boolean_scalar(max_abs_delta_threshold) or not isinstance(
            max_abs_delta_threshold, (int, float)
        ):
            msg = f"max_abs_delta_threshold must be a number, got {max_abs_delta_threshold!r}"
            raise TypeError(msg)
        max_abs_delta_threshold = float(max_abs_delta_threshold)
        if not math.isfinite(max_abs_delta_threshold) or max_abs_delta_threshold < 0.0:
            msg = f"max_abs_delta_threshold must be finite and >= 0, got {max_abs_delta_threshold}"
            raise ValueError(msg)

    flags = _flag_per_statistic(report.get("per_statistic", {}), relative_range_threshold)
    if max_abs_delta_threshold is not None:
        sensitivity_flags = _flag_sensitivity(
            report.get("sensitivity", {}), max_abs_delta_threshold
        )
        for name, flag in sensitivity_flags.items():
            if name in flags:
                flags[name].update(flag)
                flags[name]["source"] = "parameter_grid_and_split_position"
            else:
                flags[name] = flag

    return {
        "schema": "adasquash.data_sensitivity_flags.v1",
        "relative_range_threshold": relative_range_threshold,
        "max_abs_delta_threshold": max_abs_delta_threshold,
        "n_flagged": len(flags),
        "flags": dict(sorted(flags.items())),
    }


__all__ = [
    "flag_sensitive_statistics",
    "parameter_sweep",
    "split_position_robustness",
]
