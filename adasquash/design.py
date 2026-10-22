from __future__ import annotations
"""Experimental-design preparation for the training studies.

The comparison reports answer statistical questions *after* the runs exist:
paired t-tests, equivalence, multiplicity, power and jackknife checks. They do
not answer the design questions that have to be settled *before* a run starts:

* in what order will the runs execute, and is that order confounded with the
  arm?  A matrix that always trains LayerNorm first and the proposal last can
  turn a slow hardware/thermal drift into an apparent "norm effect";
* which runs are allowed to be excluded, and on what pre-specified rule?
  Excluding a diverged arm after seeing its loss is a garden of forking paths;
* is the planned number of seeds/pairs actually able to detect the smallest
  effect the study claims to care about?
* does the analysis code itself behave as advertised under a known null and a
  known effect?  A pipeline that reports a "significant" effect on a
  synthetic null is not evidence for anything.

This module supplies the reusable, pure-Python pieces:

``block_randomized_schedule``
    Deterministic within-block randomization of run ids.  Hash-based rather
    than ``random.shuffle`` so the same schedule is reproducible on every
    Python version and can be frozen in the pre-registration.
``schedule_balance_report``
    Per-arm position/block coverage, so an arm that only ever ran in one
    time block is visible before the results are interpreted.
``order_drift_report``
    Within-arm rank correlation between execution position and outcome, with
    a within-arm permutation test.  Centering by arm is what distinguishes
    "time drift correlated with the arm" from a real arm difference.
``design_readiness_audit``
    A pre-flight checklist: replication, stated hypotheses, a pre-specified
    smallest effect, adequate planned power, a complete randomized schedule
    and the presence of an identity/negative control pair.
``simulate_paired_design``
    A Monte-Carlo calibration of the actual paired-test code: empirical
    type-I error under a synthetic null and empirical power under a known
    effect.  This is a *preparation* test of the analysis pipeline, not a
    substitute for running the experiment.

Wiring in this repository
-------------------------
``scripts/ablation.py`` writes ``run_schedule.json`` (the RCBD),
``design_readiness.json`` (the pre-flight checklist, with
``--require-design-ready`` available to fail closed) and
``run_order_audit.json`` (the realized-order drift test); the schedule is
embedded in ``analysis_plan.json`` so changing it changes the pre-registration
fingerprint.  ``scripts/train_llama_depth_sweep.py`` writes the schedule and
the order audit for the depth/norm grid.  ``--no-randomize-order`` is the
recorded opt-out in both runners.
"""
import hashlib
import json
import math
import random
from typing import Callable, Iterable, Mapping, Optional, Sequence

from .metrics import MIN_SEEDS_FOR_SIGNIFICANCE, paired_design_summary, paired_t_one_sided, tost_paired

__all__ = [
    'block_randomized_schedule',
    'randomized_complete_block_schedule',
    'schedule_balance_report',
    'order_drift_report',
    'design_readiness_audit',
    'simulate_paired_design',
]


def _validate_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f'schedule seed must be an integer, got {seed!r}')
    return int(seed)


def _validate_unit(unit) -> str:
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f'run id must be a non-empty string, got {unit!r}')
    return unit


def block_randomized_schedule(units: Iterable[str], block_size: Optional[int] = None,
                              seed: int = 0) -> dict:
    """Deterministically randomize run ids inside consecutive time blocks.

    Every ``block_size`` consecutive run slots form one block and the ids in
    that block are ordered by a keyed SHA-256 digest.  Sorting by a
    cryptographic hash is a uniform random permutation whose result does not
    depend on the Python ``random`` implementation, so a schedule written
    before training can be re-computed and checked afterwards on any host.

    Blocking (rather than one global shuffle) keeps the schedule balanced if
    the study has to be stopped early: within every block each arm has the
    same expected number of early/late positions.  It also keeps a slow
    monotone drift from aligning with a single arm; the residual correlation
    is tested by :func:`order_drift_report`.

    ``block_size=None`` uses ``ceil(sqrt(n))`` for a grid with more than eight
    units (small grids are a single block).  The returned ``fingerprint``
    covers the seed, block size and final order, and is safe to embed in a
    pre-registration.
    """
    seed = _validate_seed(seed)
    if isinstance(units, (str, bytes)):
        raise ValueError('units must be an iterable of run ids, not a single string')
    ids = [_validate_unit(unit) for unit in units]
    if not ids:
        raise ValueError('a run schedule needs at least one run id')
    if len(set(ids)) != len(ids):
        duplicates = sorted({unit for unit in ids if ids.count(unit) > 1})
        raise ValueError(f'run ids must be unique; duplicated: {duplicates}')
    n = len(ids)
    if block_size is None:
        block_size = n if n <= 8 else max(1, math.ceil(math.sqrt(n)))
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError(f'block_size must be a positive integer, got {block_size!r}')

    order: list[str] = []
    entries: list[dict] = []
    block_sizes: list[int] = []
    for block_index, start in enumerate(range(0, n, block_size)):
        chunk = ids[start:start + block_size]
        block_sizes.append(len(chunk))
        digest_key = lambda unit: (
            hashlib.sha256(f'{seed}\x1f{block_index}\x1f{unit}'.encode('utf-8')).hexdigest(),
            unit,
        )
        for unit in sorted(chunk, key=digest_key):
            entries.append({'unit': unit, 'block': block_index,
                            'position': len(order)})
            order.append(unit)

    payload = {'seed': seed, 'block_size': block_size, 'order': order}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return {
        'schema': 'adasquash.run_schedule.v1',
        'seed': seed,
        'block_size': block_size,
        'n_units': n,
        'n_blocks': len(block_sizes),
        'block_sizes': block_sizes,
        'order': order,
        'entries': entries,
        'positions': {unit: position for position, unit in enumerate(order)},
        'fingerprint': fingerprint,
        'protocol': ('within-block order is a keyed SHA-256 sort; the same seed and '
                     'run-id set reproduce the same schedule on every host'),
    }


def randomized_complete_block_schedule(units: Iterable[str], replicates: Iterable[str],
                                       seed: int = 0) -> dict:
    """Randomized complete block design for replicated runs.

    Every time block contains *every* treatment/unit exactly once, and each
    unit receives every replicate exactly once across the blocks.  With
    ``units`` = matrix rows and ``replicates`` = training seeds this is the
    classic RCBD: a block-level drift (a slow machine, a thermal episode, a
    code change that only affects part of the session) hits every arm once and
    therefore cancels from the paired arm contrast instead of biasing the arm
    that happened to run last.

    The block order is time order; the within-block order is a keyed SHA-256
    sort, so it is reproducible and independent of Python's ``random``
    implementation.  Each row is still replicated over every seed, but the
    seeds are assigned to blocks through an independent per-unit permutation,
    so a row is not always the first/last run of a block.
    """
    seed = _validate_seed(seed)
    if isinstance(units, (str, bytes)) or isinstance(replicates, (str, bytes)):
        raise ValueError('units and replicates must be iterables, not single strings')
    unit_ids = [_validate_unit(unit) for unit in units]
    replicate_ids = [str(rep) for rep in replicates]
    if any(not rep.strip() for rep in replicate_ids):
        raise ValueError('replicate ids must be non-empty')
    if not unit_ids:
        raise ValueError('a block schedule needs at least one unit')
    if not replicate_ids:
        raise ValueError('a block schedule needs at least one replicate')
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError('unit ids must be unique')
    if len(set(replicate_ids)) != len(replicate_ids):
        raise ValueError('replicate ids must be unique')

    def _hash(*parts) -> str:
        return hashlib.sha256('\x1f'.join(str(part) for part in parts).encode('utf-8')).hexdigest()

    replicate_order = {
        unit: sorted(replicate_ids,
                     key=lambda rep, unit=unit: (_hash(seed, 'rep', unit, rep), rep))
        for unit in unit_ids
    }
    entries: list[dict] = []
    order: list[str] = []
    for block_index in range(len(replicate_ids)):
        for unit in sorted(unit_ids,
                           key=lambda unit, block=block_index: (
                               _hash(seed, 'block', block, unit), unit)):
            replicate = replicate_order[unit][block_index]
            run_id = f'{unit}::rep={replicate}'
            entries.append({'unit': unit, 'replicate': replicate,
                            'run_id': run_id, 'block': block_index,
                            'position': len(order)})
            order.append(run_id)
    run_ids = [entry['run_id'] for entry in entries]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError('internal error: a unit x replicate run was assigned twice')
    expected = {(unit, rep) for unit in unit_ids for rep in replicate_ids}
    realized = {(entry['unit'], entry['replicate']) for entry in entries}
    if realized != expected:
        raise ValueError('internal error: the block design does not cover every '
                         'unit x replicate pair exactly once')
    payload = {'seed': seed, 'units': unit_ids, 'replicates': replicate_ids,
               'order': order}
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    return {
        'schema': 'adasquash.rcbd_schedule.v1',
        'seed': seed,
        'n_units': len(order),
        'n_treatments': len(unit_ids),
        'n_blocks': len(replicate_ids),
        'block_sizes': [len(unit_ids)] * len(replicate_ids),
        'units': unit_ids,
        'replicates': replicate_ids,
        'order': order,
        'entries': entries,
        'positions': {run_id: position for position, run_id in enumerate(order)},
        'fingerprint': fingerprint,
        'protocol': ('randomized complete block design: one run of every treatment '
                     'per time block, each treatment gets every replicate once '
                     'across blocks, within-block order is a keyed SHA-256 sort'),
    }


def _arm_resolver(arm_of):
    if arm_of is None:
        return lambda unit: unit
    if callable(arm_of):
        return arm_of
    if isinstance(arm_of, Mapping):
        return lambda unit: arm_of.get(unit, unit)
    raise ValueError('arm_of must be None, a mapping or a callable')


def schedule_balance_report(schedule: dict, arm_of=None) -> dict:
    """Per-arm position and block coverage for a generated schedule.

    The report is descriptive: a random schedule can put one three-seed arm
    entirely in the first third by chance, and that is not by itself a flaw.
    The actionable warnings are structural (an arm that never appears in more
    than one block when it has several runs, a single-block schedule for a
    large grid, or missing/duplicate positions).  :func:`order_drift_report`
    tests whether the realized positions actually correlate with outcomes.
    """
    if not isinstance(schedule, dict) or not isinstance(schedule.get('entries'), list):
        raise ValueError('schedule must be the dict returned by block_randomized_schedule')
    resolve = _arm_resolver(arm_of)
    entries = schedule['entries']
    structural: list[str] = []
    if len(entries) != schedule.get('n_units'):
        structural.append('entry count does not match n_units')
    # Positions are the structural contract of the schedule. A malformed
    # record must produce a warning instead of a KeyError/TypeError: this
    # report is the pre-flight check that is supposed to expose a broken
    # schedule, not crash on it.
    positions: list[int] = []
    positions_valid = True
    for entry in entries:
        if not isinstance(entry, dict):
            positions_valid = False
            continue
        position = entry.get('position')
        if (isinstance(position, bool) or not isinstance(position, int)
                or position < 0):
            positions_valid = False
        else:
            positions.append(position)
    if not positions_valid:
        structural.append('positions are not a contiguous 0..n-1 range '
                          '(missing or non-integer position)')
    elif sorted(positions) != list(range(len(entries))):
        structural.append('positions are not a contiguous 0..n-1 range')
    units = [entry.get('run_id', entry.get('unit'))
             if isinstance(entry, dict) else None for entry in entries]
    if len(set(units)) != len(units):
        structural.append('duplicate run ids in the realized order')

    per_arm: dict[str, dict] = {}
    blocks_valid = True
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        arm = str(resolve(entry.get('run_id', entry.get('unit'))))
        bucket = per_arm.setdefault(arm, {
            'count': 0, 'positions': [], 'blocks': set(), 'early': 0, 'late': 0,
        })
        bucket['count'] += 1
        position = entry.get('position')
        if (not isinstance(position, bool) and isinstance(position, int)
                and position >= 0):
            bucket['positions'].append(position)
        block = entry.get('block')
        if isinstance(block, bool) or not isinstance(block, int) or block < 0:
            blocks_valid = False
        else:
            bucket['blocks'].add(block)
    if not blocks_valid:
        structural.append('one or more entries have a missing or non-integer block')
    n = len(entries)
    overall_mean = (n - 1) / 2.0 if n else None
    extreme_counts: dict[str, int] = {}
    for arm, bucket in per_arm.items():
        bucket['mean_position'] = (sum(bucket['positions']) / bucket['count']
                                   if bucket['count'] else None)
        bucket['position_span'] = (max(bucket['positions']) - min(bucket['positions'])
                                   if bucket['positions'] else None)
        bucket['blocks'] = sorted(bucket['blocks'])
        bucket['n_blocks'] = len(bucket['blocks'])
        if n:
            bucket['early'] = sum(1 for p in bucket['positions'] if p < n / 3.0)
            bucket['late'] = sum(1 for p in bucket['positions'] if p >= 2.0 * n / 3.0)
        if bucket['count'] >= 2 and bucket['n_blocks'] <= 1:
            structural.append(
                f"arm {arm!r} appears in only one time block "
                f"({bucket['count']} run(s)); its position is confounded with "
                f"any block-level drift")
        if bucket['count'] >= 2 and n:
            dev = abs(bucket['mean_position'] - overall_mean) / max(1.0, n - 1)
            if dev >= 1.0 / 3.0:
                extreme_counts[arm] = bucket['count']
    if schedule.get('n_blocks') == 1 and n > 8:
        structural.append('a single block was used for a large grid; '
                          'blocking cannot spread a monotone drift')
    return {
        'schema': 'adasquash.schedule_balance.v1',
        'schedule_fingerprint': schedule.get('fingerprint'),
        'n_units': n,
        'n_blocks': schedule.get('n_blocks'),
        'block_sizes': schedule.get('block_sizes'),
        'overall_mean_position': overall_mean,
        'per_arm': per_arm,
        'arms_with_extreme_mean_position': sorted(extreme_counts),
        'balanced': not structural,
        'warnings': structural,
        'note': ('balance is descriptive; use order_drift_report to test whether '
                 'the realized run order predicts the outcome after removing '
                 'arm means'),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda index: values[index])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        average = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = average
        i = j
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 2 or len(y) != n:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx = math.sqrt(sum(value * value for value in dx))
    sy = math.sqrt(sum(value * value for value in dy))
    if sx == 0.0 or sy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / (sx * sy)


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    return _pearson(_average_ranks(x), _average_ranks(y))


def order_drift_report(observations: Sequence[dict], alpha: float = 0.05,
                       permutations: int = 2000, seed: int = 0) -> dict:
    """Test whether execution position predicts the outcome *within an arm*.

    ``observations`` is a list of ``{'unit', 'arm', 'position', 'value'}``
    records (one per completed run).  The outcome is centered by arm, so a
    genuine arm difference - including one that happens to be scheduled
    early - is removed before the rank correlation is computed.  What remains
    is the within-arm time trend: if later runs within the same arm score
    systematically worse, the schedule (hardware, thermal, code drift) is
    confounding the study.

    The p-value is a two-sided within-arm permutation test: positions are
    reshuffled inside each arm, keeping the arm sizes, outcome distribution
    and between-arm structure fixed.  Arms with fewer than two observations
    carry no within-arm information and are excluded rather than contributing
    a zero residual.  Fewer than six usable observations, or fewer than two
    distinct positions, returns ``status='insufficient_data'`` instead of a
    misleading p-value.
    """
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise ValueError(f'permutations must be a positive integer, got {permutations!r}')
    if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
            and math.isfinite(float(alpha)) and 0.0 < float(alpha) < 1.0):
        raise ValueError(f'alpha must be strictly between 0 and 1, got {alpha!r}')
    seed = _validate_seed(seed)
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise ValueError('observations must be a sequence of dict records')

    excluded = {'malformed': 0, 'non_finite': 0, 'duplicate_position': 0,
                'singleton_arm': 0}
    valid: list[dict] = []
    seen: set[tuple] = set()
    for record in observations:
        if not isinstance(record, dict):
            excluded['malformed'] += 1
            continue
        arm = record.get('arm')
        position = record.get('position')
        value = record.get('value')
        if (not isinstance(arm, str) or not arm
                or isinstance(position, bool) or not isinstance(position, (int, float))
                or not isinstance(value, (int, float)) or isinstance(value, bool)):
            excluded['malformed'] += 1
            continue
        if not (math.isfinite(float(position)) and math.isfinite(float(value))):
            excluded['non_finite'] += 1
            continue
        key = (arm, float(position))
        if key in seen:
            excluded['duplicate_position'] += 1
            continue
        seen.add(key)
        valid.append({'unit': record.get('unit'), 'arm': arm,
                      'position': float(position), 'value': float(value)})

    groups: dict[str, list[dict]] = {}
    for record in valid:
        groups.setdefault(record['arm'], []).append(record)
    usable = {arm: records for arm, records in groups.items() if len(records) >= 2}
    excluded['singleton_arm'] = sum(len(records) for arm, records in groups.items()
                                    if len(records) < 2)
    flat = [record for arm in sorted(usable) for record in usable[arm]]
    distinct_positions = {record['position'] for record in flat}
    base = {
        'schema': 'adasquash.order_drift.v1',
        'n_observations': len(flat),
        'n_arms': len(usable),
        'n_distinct_positions': len(distinct_positions),
        'excluded': excluded,
        'permutations': permutations,
        'alpha': float(alpha),
        'spearman_rho': None,
        'p_two_sided': None,
        'drift_detected': None,
        'status': 'insufficient_data',
    }
    if len(flat) < 6 or len(distinct_positions) < 2:
        base['note'] = ('at least six observations from arms with >= 2 runs and two '
                        'distinct positions are needed to test order drift')
        return base

    for arm, records in usable.items():
        mean = sum(record['value'] for record in records) / len(records)
        for record in records:
            record['residual'] = record['value'] - mean
    x = [record['position'] for record in flat]
    y = [record['residual'] for record in flat]
    rho = _spearman(x, y)
    if rho is None:
        base['note'] = 'every within-arm residual is identical; no rank correlation is defined'
        return base

    rng = random.Random(seed)
    arm_positions = {arm: [record['position'] for record in records]
                     for arm, records in usable.items()}
    residuals = [record['residual'] for record in flat]
    rho_samples: list[float] = []
    for _ in range(permutations):
        shuffled_positions: list[float] = []
        for arm in sorted(usable):
            positions = list(arm_positions[arm])
            rng.shuffle(positions)
            shuffled_positions.extend(positions)
        permuted = _spearman(shuffled_positions, residuals)
        if permuted is not None:
            rho_samples.append(permuted)
    extreme = sum(1 for sample in rho_samples if abs(sample) >= abs(rho))
    p_two_sided = (1.0 + extreme) / (1.0 + len(rho_samples))
    # `drift_detected` orients a positive rho as "later runs worse": the
    # one-sided p for that alternative is the upper tail, not the lower tail.
    # The old lower-tail count returned ~1.0 for the very trend the report
    # labels later_runs_worse.
    decline = sum(1 for sample in rho_samples if sample >= rho)
    p_one_sided_decline = (1.0 + decline) / (1.0 + len(rho_samples))

    per_arm = []
    for arm in sorted(usable):
        records = usable[arm]
        positions = [record['position'] for record in records]
        per_arm.append({
            'arm': arm,
            'count': len(records),
            'mean_position': sum(positions) / len(positions),
            'position_span': max(positions) - min(positions),
            'mean_residual': sum(record['residual'] for record in records) / len(records),
        })
    base.update({
        'status': 'tested',
        'spearman_rho': rho,
        'p_two_sided': p_two_sided,
        'p_one_sided_decline': p_one_sided_decline,
        'drift_detected': bool(p_two_sided < float(alpha) and rho != 0.0),
        'direction': ('later_runs_worse' if rho > 0 else
                      'later_runs_better' if rho < 0 else 'flat'),
        'n_permutations_defined': len(rho_samples),
        'per_arm': per_arm,
        'note': ('a significant within-arm rank correlation means execution order '
                 'is confounded with the outcome; re-run the affected arms in a '
                 'fresh randomized block rather than interpreting the arm delta'),
    })
    return base


def design_readiness_audit(seeds: Sequence[int], rows: Iterable[str],
                           hypotheses: Mapping[str, str],
                           schedule: Optional[dict] = None,
                           expected_units: Optional[Iterable[str]] = None,
                           schedule_arm_of=None,
                           margin: float = 0.0,
                           equivalence_margin: float = 0.0,
                           assumed_paired_sd: Optional[float] = None,
                           target_power: float = 0.8,
                           alpha: float = 0.05,
                           min_seeds: int = MIN_SEEDS_FOR_SIGNIFICANCE,
                           identity_controls: Sequence[Sequence[str]] = ()) -> dict:
    """Pre-flight design checklist, written before any run starts.

    Failures are structural and should stop a confirmatory launch; warnings are
    judgement calls that are recorded rather than hidden.  In particular, the
    absence of a pre-specified smallest effect is a warning, not a failure: an
    exploratory matrix may legitimately run without one, but it cannot make an
    equivalence claim later.
    """
    names = list(rows.keys()) if isinstance(rows, Mapping) else list(rows)
    checks: list[dict] = []

    def add(check_id: str, status: str, detail: str) -> None:
        checks.append({'id': check_id, 'status': status, 'detail': detail})

    seeds = list(seeds or [])
    # The power check must use the number of *distinct* observations the
    # paired test will actually have, not the raw list length (duplicate seeds
    # are the same run and the replication check above rejects them).
    n_pairs = len({seed for seed in seeds
                   if isinstance(seed, int) and not isinstance(seed, bool)})
    if (len(seeds) < min_seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)):
        add('replication', 'fail',
            f'{len(seeds)} seed(s) {seeds}: need >= {min_seeds} distinct integer seeds')
    else:
        add('replication', 'pass', f'{len(seeds)} distinct seeds: {seeds}')

    missing = sorted(name for name in names if not str(hypotheses.get(name) or '').strip())
    if missing:
        add('hypotheses', 'fail', f'rows without a stated hypothesis: {missing}')
    else:
        add('hypotheses', 'pass', f'all {len(names)} rows state a hypothesis')

    try:
        margin = float(margin or 0.0)
        equivalence_margin = float(equivalence_margin or 0.0)
    except (TypeError, ValueError):
        margin = equivalence_margin = float('nan')
    if not math.isfinite(margin) or margin < 0.0:
        add('effect_size', 'fail', f'invalid superiority margin {margin!r}')
    elif not math.isfinite(equivalence_margin) or equivalence_margin < 0.0:
        add('effect_size', 'fail', f'invalid equivalence margin {equivalence_margin!r}')
    elif margin <= 0.0 and equivalence_margin <= 0.0:
        add('effect_size', 'warn',
            'no positive smallest effect of interest (--baseline-margin or '
            '--equivalence-margin); the study can rank arms but cannot claim '
            'practical significance or equivalence')
    else:
        add('effect_size', 'pass',
            f'superiority margin={margin}, equivalence margin={equivalence_margin}')

    if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
            and math.isfinite(float(alpha)) and 0.0 < float(alpha) < 1.0):
        add('power', 'fail', f'invalid alpha {alpha!r}')
    elif not (isinstance(target_power, (int, float)) and not isinstance(target_power, bool)
              and math.isfinite(float(target_power)) and 0.0 < float(target_power) < 1.0):
        add('power', 'fail', f'invalid target power {target_power!r}')
    elif assumed_paired_sd is None:
        add('power', 'warn',
            'no assumed paired SD supplied; planned power for the smallest effect '
            'cannot be checked before the run (the comparison report still '
            'reports the observed MDE afterwards)')
    else:
        try:
            sd = float(assumed_paired_sd)
        except (TypeError, ValueError):
            sd = float('nan')
        target = margin if margin > 0.0 else equivalence_margin
        if not math.isfinite(sd) or sd < 0.0:
            add('power', 'fail', f'invalid assumed paired SD {assumed_paired_sd!r}')
        elif target <= 0.0:
            add('power', 'warn',
                'a paired SD was supplied but no positive margin defines the effect '
                'to power for')
        elif n_pairs < 2:
            # A paired design needs at least two distinct observations before
            # its power can be estimated. Reporting a crash here would turn a
            # failed pre-flight check into an unusable audit.
            add('power', 'fail',
                f'{n_pairs} distinct seed(s): at least 2 paired observations are '
                f'needed to estimate power for margin={target:g}')
        else:
            design = paired_design_summary(sd, n_pairs, margin=target,
                                           alpha=float(alpha), power=float(target_power))
            if design.get('adequately_powered_for_margin') is True:
                add('power', 'pass',
                    f'{n_pairs} pairs at paired SD={sd:g} reach the '
                    f'{target:g} target (required '
                    f'{design.get("required_pairs_for_margin")})')
            elif design.get('adequately_powered_for_margin') is False:
                add('power', 'fail',
                    f'{n_pairs} pairs at paired SD={sd:g} are underpowered for '
                    f'margin={target:g}; need {design.get("required_pairs_for_margin")}')
            else:
                add('power', 'warn',
                    f'margin={target:g} is not reachable within the search cap at '
                    f'paired SD={sd:g}')

    if schedule is None:
        add('schedule', 'warn',
            'no randomized run schedule: execution order may be confounded with '
            'the arm and with time drift')
    else:
        balance = schedule_balance_report(schedule, arm_of=schedule_arm_of)
        if expected_units is not None:
            expected = {str(unit) for unit in expected_units}
            realized = {str(unit) for unit in schedule.get('order') or []}
            if expected != realized:
                add('schedule', 'fail',
                    f'schedule does not cover the planned runs: missing '
                    f'{sorted(expected - realized)}, extra {sorted(realized - expected)}')
            elif balance['warnings']:
                add('schedule', 'warn', '; '.join(balance['warnings']))
            else:
                add('schedule', 'pass',
                    f'{schedule.get("n_units")} run(s) in '
                    f'{schedule.get("n_blocks")} block(s), fingerprint '
                    f'{str(schedule.get("fingerprint"))[:12]}')
        elif balance['warnings']:
            add('schedule', 'warn', '; '.join(balance['warnings']))
        else:
            add('schedule', 'pass',
                f'{schedule.get("n_units")} run(s) in '
                f'{schedule.get("n_blocks")} block(s)')

    pairs = [tuple(pair) for pair in identity_controls if pair]
    if not pairs:
        add('controls', 'warn',
            'no identity/negative-control pair declared: the analysis has no '
            'known-null contrast to check its own calibration')
    else:
        missing_pairs = [pair for pair in pairs
                         if len(pair) != 2 or any(str(name) not in names for name in pair)]
        if missing_pairs:
            add('controls', 'fail',
                f'control pair(s) reference absent rows: {missing_pairs}')
        else:
            add('controls', 'pass',
                f'{len(pairs)} identity/negative-control pair(s) present: '
                f'{[list(pair) for pair in pairs]}')

    n_fail = sum(1 for check in checks if check['status'] == 'fail')
    n_warn = sum(1 for check in checks if check['status'] == 'warn')
    return {
        'schema': 'adasquash.design_readiness.v1',
        'ready': n_fail == 0,
        'n_fail': n_fail,
        'n_warn': n_warn,
        'checks': checks,
        'note': ('failures block a confirmatory launch; warnings are recorded so a '
                 'reader can see which design decisions were left open'),
    }


def _normal_sampler(rng: random.Random, mean: float, sd: float):
    if sd == 0.0:
        return lambda: mean
    return lambda: rng.gauss(mean, sd)


def simulate_paired_design(n_pairs: int = 3, effect: float = 0.0,
                           paired_sd: float = 0.05, reps: int = 2000,
                           alpha: float = 0.05, equivalence_margin: float = 0.0,
                           seed: int = 0) -> dict:
    """Monte-Carlo calibration of the paired-test pipeline.

    Draws synthetic paired differences from ``N(effect, paired_sd)`` and runs
    the *actual* ``paired_t_one_sided`` code used by the reports:

    * ``effect=0`` estimates the empirical type-I error of the one-sided
      superiority claim (the pipeline should be at or below ``alpha``);
    * ``effect<0`` estimates power for that effect size (the candidate has the
      lower loss, matching the reports' sign convention);
    * a positive ``equivalence_margin`` additionally reports how often the
      actual TOST code declares equivalence.

    Monte-Carlo error is reported so a "0.049" is not over-read: with
    ``reps=2000`` the standard error of a 0.05 rate is about 0.005.
    """
    if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs < 2:
        raise ValueError(f'n_pairs must be an integer >= 2, got {n_pairs!r}')
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 100:
        raise ValueError(f'reps must be an integer >= 100, got {reps!r}')
    for name, value in (('effect', effect), ('paired_sd', paired_sd)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            raise ValueError(f'{name} must be a finite number, got {value!r}')
    if float(paired_sd) < 0.0:
        raise ValueError(f'paired_sd must be >= 0, got {paired_sd!r}')
    if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool)
            and math.isfinite(float(alpha)) and 0.0 < float(alpha) < 1.0):
        raise ValueError(f'alpha must be strictly between 0 and 1, got {alpha!r}')
    if (isinstance(equivalence_margin, bool)
            or not isinstance(equivalence_margin, (int, float))
            or not math.isfinite(float(equivalence_margin))
            or float(equivalence_margin) < 0.0):
        raise ValueError(f'equivalence_margin must be a finite number >= 0, '
                         f'got {equivalence_margin!r}')
    seed = _validate_seed(seed)

    rng = random.Random(seed)
    draw = _normal_sampler(rng, float(effect), float(paired_sd))
    detection = 0
    equivalence = 0
    undefined = 0
    means: list[float] = []
    for _ in range(reps):
        deltas = [draw() for _ in range(n_pairs)]
        means.append(sum(deltas) / n_pairs)
        test = paired_t_one_sided(deltas, [0.0] * n_pairs, margin=0.0, alpha=float(alpha))
        p_value = test.get('p_one_sided')
        if p_value is None or not math.isfinite(float(p_value)):
            undefined += 1
        elif float(p_value) < float(alpha):
            detection += 1
        if float(equivalence_margin) > 0.0:
            equiv = tost_paired(deltas, [0.0] * n_pairs,
                                margin=float(equivalence_margin), alpha=float(alpha))
            if equiv.get('equivalent') is True:
                equivalence += 1
    mean_estimate = sum(means) / len(means)
    variance = (sum((value - mean_estimate) ** 2 for value in means) / (len(means) - 1)
                if len(means) > 1 else 0.0)
    detection_rate = detection / reps
    return {
        'schema': 'adasquash.design_simulation.v1',
        'n_pairs': n_pairs,
        'effect': float(effect),
        'paired_sd': float(paired_sd),
        'reps': reps,
        'alpha': float(alpha),
        'equivalence_margin': (float(equivalence_margin)
                               if float(equivalence_margin) > 0.0 else None),
        'empirical_detection_rate': detection_rate,
        'empirical_type_i_error': (detection_rate if float(effect) == 0.0 else None),
        'empirical_power': (detection_rate if float(effect) < 0.0 else None),
        'empirical_equivalence_rate': (equivalence / reps
                                       if float(equivalence_margin) > 0.0 else None),
        'wrong_direction_rate': (sum(1 for value in means if value > 0.0) / reps
                                 if float(effect) <= 0.0 else None),
        'mean_estimate': mean_estimate,
        'mean_bias': mean_estimate - float(effect),
        'estimate_sd': math.sqrt(variance),
        'monte_carlo_se_detection_rate': math.sqrt(
            max(0.0, detection_rate * (1.0 - detection_rate)) / reps),
        'undefined_p_rate': undefined / reps,
        'note': ('a well-calibrated superiority pipeline has '
                 'empirical_type_i_error <= alpha (within Monte-Carlo error); '
                 'power below the target means the planned seed count cannot '
                 'detect the stated effect'),
    }
