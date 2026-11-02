from __future__ import annotations

"""Calibrate the confirmatory planning variance from pilot ablation runs.

The readiness audit can plan power only if the paired across-seed SD is known
before the confirmatory launch. Using the pilot point estimate is optimistic:
a small pilot's sample SD is itself uncertain, so the realised SD is often
larger and the planned power is not reached. This script reads the pilot
``all_summaries.json``, forms candidate-minus-reference paired differences on
shared seeds, and freezes the one-sided upper confidence bound produced by
:func:`adasquash.design.paired_sd_calibration`.

The output is the evidence record consumed by
``scripts/ablation.py --paired-sd-evidence``. Seeds must be listed explicitly
(or be inferred from the summaries) and should be disjoint from the
confirmatory seeds; the readiness audit fails when they overlap. This is a
design artifact, not a result: do not choose the effect margin from the pilot
mean after reading it.
"""
import argparse
import json
import numbers
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adasquash.design import paired_sd_calibration
from adasquash.jsonio import save_json_atomic


def _seed_list(value):
    if value is None:
        return None
    seeds = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            raise ValueError("seed lists cannot contain empty entries")
        try:
            seed = int(item)
        except ValueError:
            raise ValueError(f"seed {item!r} is not an integer") from None
        if seed in seeds:
            raise ValueError(f"duplicate seed {seed} in the seed list")
        seeds.append(seed)
    return seeds


def _load_rows(path):
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from None
    if isinstance(raw, dict):
        raw = raw.get("rows")
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list of row summaries")
    rows = {}
    for row in raw:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            rows[row["name"]] = row
    return raw, rows


def _loss_by_seed(row, name):
    per_seed = row.get("per_seed")
    if not isinstance(per_seed, list):
        raise ValueError(f"row {name!r} has no per_seed table")
    losses = {}
    unusable = {}
    seen = set()
    for entry in per_seed:
        if not isinstance(entry, dict):
            continue
        seed = entry.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            continue
        seed = int(seed)
        if seed in seen:
            raise ValueError(f"row {name!r} lists seed {seed} more than once")
        seen.add(seed)
        if entry.get("exclusion_status") == "excluded":
            unusable[seed] = "excluded by the pre-specified policy"
            continue
        loss = entry.get("val_loss")
        if isinstance(loss, bool) or not isinstance(loss, numbers.Real):
            unusable[seed] = "missing val_loss"
            continue
        loss = float(loss)
        if loss != loss or loss in (float("inf"), float("-inf")):
            unusable[seed] = "non-finite val_loss"
            continue
        losses[seed] = loss
    if not losses:
        raise ValueError(f"row {name!r} has no finite per-seed val_loss values")
    return losses, unusable


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Freeze a pilot paired-SD calibration for the ablation "
                    "readiness audit")
    p.add_argument("summaries", help="pilot all_summaries.json")
    p.add_argument("--candidate", required=True,
                   help="candidate row name in the pilot summaries")
    p.add_argument("--reference", required=True,
                   help="reference row name in the pilot summaries")
    p.add_argument("--pilot-seeds", default=None,
                   help="comma-separated pilot seeds; default: every seed "
                        "shared by the two rows")
    p.add_argument("--confirmatory-seeds", default=None,
                   help="comma-separated seeds planned for the confirmatory "
                        "run; recorded so the audit can prove independence")
    p.add_argument("--confidence", type=float, default=0.90,
                   help="one-sided confidence for the SD upper bound")
    p.add_argument("--min-pairs", type=int, default=2,
                   help="minimum complete pilot pairs to accept")
    p.add_argument("--output", default="paired_sd_calibration.json",
                   help="where to write the frozen calibration JSON")
    args = p.parse_args(argv)

    try:
        _, rows = _load_rows(args.summaries)
        for name in (args.candidate, args.reference):
            if name not in rows:
                raise ValueError(
                    f"row {name!r} is absent from {args.summaries}; available "
                    f"rows: {sorted(rows)}")
        candidate_losses, candidate_unusable = _loss_by_seed(
            rows[args.candidate], args.candidate)
        reference_losses, reference_unusable = _loss_by_seed(
            rows[args.reference], args.reference)
        requested = _seed_list(args.pilot_seeds)
        shared = sorted(set(candidate_losses).intersection(reference_losses))
        selected = shared if requested is None else requested
        if not selected:
            raise ValueError("no shared pilot seeds between the two rows")
        missing = [seed for seed in selected
                   if seed not in candidate_losses or seed not in reference_losses]
        if missing:
            raise ValueError(
                f"pilot seed(s) {missing} are missing a finite val_loss in one "
                f"of the two rows; complete pairs are required")
        differences = [candidate_losses[seed] - reference_losses[seed]
                       for seed in selected]
        confirmatory = _seed_list(args.confirmatory_seeds)
        calibration = paired_sd_calibration(
            differences, pilot_seeds=selected,
            confirmatory_seeds=confirmatory,
            confidence=args.confidence, min_pairs=args.min_pairs)
    except ValueError as exc:
        print(f"[calibration] ERROR: {exc}", file=sys.stderr)
        return 2

    calibration["pilot_rows"] = {
        "summaries": str(args.summaries),
        "candidate": args.candidate,
        "reference": args.reference,
        "candidate_seeds": sorted(candidate_losses),
        "reference_seeds": sorted(reference_losses),
        "candidate_unusable_seeds": candidate_unusable,
        "reference_unusable_seeds": reference_unusable,
    }
    output = Path(args.output)
    save_json_atomic(output, calibration)
    recommended = calibration["recommended_paired_sd"]
    recommended_text = (
        f"{recommended:.6g}" if recommended is not None else "unavailable")
    print(f"[calibration] {len(selected)} pair(s) on seeds {selected}: "
          f"paired SD={calibration['paired_sd']:.6g}, "
          f"recommended planning SD={recommended_text} "
          f"({calibration['recommended_basis']})")
    if candidate_unusable or reference_unusable:
        print(f"[calibration] WARNING: unusable pilot seed(s) were not "
              f"paired: candidate={candidate_unusable}, "
              f"reference={reference_unusable}", file=sys.stderr)
    if calibration["seeds_disjoint"] is False:
        print("[calibration] WARNING: pilot seeds overlap the confirmatory "
              "seeds; the readiness audit will reject this evidence",
              file=sys.stderr)
    elif calibration["seeds_disjoint"] is None:
        print("[calibration] WARNING: --confirmatory-seeds was not supplied, "
              "so seed disjointness cannot be proven", file=sys.stderr)
    print(f'[calibration] wrote {output} (fingerprint '
          f'{calibration["fingerprint"][:12]})')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
