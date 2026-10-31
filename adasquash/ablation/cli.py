from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

from adasquash.design import (design_readiness_audit, normalize_exclusion_policy,
                              randomized_complete_block_schedule,
                              run_exclusion_audit, schedule_balance_report,
                              simulate_paired_design)
from adasquash.repro import (artifact_record, collect_run_metadata,
                             file_fingerprint, finish_metadata, git_revision,
                             save_run_manifest)
from scripts.train_gpt2 import _explicit_arg_names, apply_config_to_parser

from .analysis import (build_analysis_plan, print_study_level_correction,
                    study_level_correction)
from .baselines import (attach_baseline_tuning,
                        attach_simple_baseline_calibration, baseline_comparison,
                        load_baseline_tuning, load_simple_baselines,
                        print_baseline_comparison, run_simple_baseline_calibration)
from .common import _is_seed_id
from .contrasts import (factorial_comparison, leave_one_out_comparison,
                        print_factorial_comparison,
                        print_leave_one_out_comparison,
                        print_qk_axis_comparison,
                        print_three_way_factorial_comparison,
                        qk_axis_comparison, three_way_factorial_comparison)
from .matrix import (ABLATIONS, ABLATION_HYPOTHESES, DEFAULT_SEEDS,
                     _MODEL_BUILDERS, effective_config_key,
                     load_ablations_from_config, validate_matrix)
from .runner import (aggregate_seed_summaries, cached_seed_matches, _load_summary,
                     _write_summary, merge_summaries, partition_rows_by_protocol,
                     run_one, run_order_audit)
from .runner import _PROTOCOL_MODEL_VOCAB

def build_arg_parser(argv=None):
    """Build the ablation CLI, its config-derived rows, and YAML defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args(argv)
    ablations = dict(ABLATIONS)
    hypotheses = dict(ABLATION_HYPOTHESES)
    if known.config:
        try:
            from_yaml = load_ablations_from_config(known.config, include_metadata=True)
        except ValueError as exc:
            print(f'[matrix] FAIL: {exc}')
            sys.exit(1)
        if from_yaml:
            ablations = {name: row['cfg'] for name, row in from_yaml.items()}
            hypotheses = {name: row['hypothesis'] for name, row in from_yaml.items()}
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='YAML run config; CLI flags override it')
    p.add_argument('--model', choices=sorted(_MODEL_BUILDERS), default='gpt2-124m')
    p.add_argument('--names', nargs='+', default=list(ablations.keys()))
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.0006)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--outlier-layer', type=int, default=None, help='block after which the residual-channel outlier is injected (default: the middle block)')
    p.add_argument('--outlier-channel', type=int, default=None, help='residual-stream channel to amplify (default: the middle channel)')
    p.add_argument('--outlier-scale', type=float, default=25.0, help='amplification applied to the outlier channel (0 disables the outlier)')
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help='one row is trained once per seed and reported as mean +/- std across seeds (>= 3 recommended)')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--stats-every', type=int, default=200)
    p.add_argument('--val-batches', type=int, default=20, help='held-out synthetic batches for the selection metric (val_loss); never used for gradient updates')
    p.add_argument('--curve-every', type=int, default=0, help='held-out curve-monitoring period in steps (0 = auto: max_steps//5; negative = disabled). Curve points come from a separate generator (seed+2) and never touch the selection batches, so a row reports its train/val regime, not only its final number')
    p.add_argument('--curve-batches', type=int, default=5, help='held-out batches per curve-monitoring point')
    p.add_argument('--baselines', nargs='+', default=['ln_baseline', 'rmsnorm_baseline'], help='rows treated as the simple reference baselines in the baseline-relative comparison')
    p.add_argument('--simple-baselines', default=None, help='Bayes-limit calibration JSON from scripts/baseline_simple.py; when omitted, auto-detected at <output-dir>/simple_baselines.json. Adds the tuned simple baseline and the memoryless-Bayes-floor / full-context-ceiling gaps to baseline_comparison.json')
    p.add_argument('--auto-simple-baselines', action='store_true', help='when the calibration is missing, run scripts/baseline_simple.py once for the selected seeds (same synthetic corpus protocol) instead of writing a comparison with no Bayes-limit reference')
    p.add_argument('--require-simple-baselines', action='store_true', help='fail before writing baseline_comparison.json when no calibration is available and --auto-simple-baselines did not produce one; a comparison without the tuned count model and Bayes limits cannot show that a row learned the task')
    p.add_argument('--no-simple-baselines', dest='simple_baselines_enabled', action='store_false', help='explicitly opt out of the simple-baseline calibration annotation; the opt-out is recorded in baseline_comparison.json')
    p.add_argument('--baseline-tuning', default=None, help='tuning_summary.json from scripts/tune.py; certifies whether the fixed matrix recipe is the baselines\' selected (lr, weight_decay) cell on the same seeds and records the result in baseline_comparison.json')
    p.add_argument('--require-baseline-tuning', action='store_true', help='fail before training unless --baseline-tuning certifies that every compared baseline selected this exact (lr, weight_decay) cell and tune.py confirmed it on held-back seeds; without this flag an uncertified comparison is written but marked as such')
    p.add_argument('--baseline-margin', type=float, default=0.0, help='minimum val_loss improvement over the best baseline to count as a win')
    p.add_argument('--equivalence-margin', type=float, default=0.0, help='TOST equivalence margin in val_loss for "the two arms are the same" claims (e.g. explicit-vs-default init rows); 0 disables equivalence testing, and a non-significant difference is never relabelled as equivalence')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the one-sided paired t-test vs the best baseline (Holm-adjusted over rows, then across the whole pre-registered study)')
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='run the small-batch overfit sanity check for every norm in the selected rows before training them (default: on; a row that cannot overfit one batch has a gradient/loss bug and must not be launched)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the pre-flight overfit sanity check (use only when the check itself is the problem, e.g. an offline host)')
    p.add_argument('--check-matrix', action='store_true', help='validate controlled knobs, stated hypotheses and leave-one-out coverage of the matrix, then exit without training')
    p.add_argument('--schedule-seed', type=int, default=0, help='seed of the randomized complete-block execution schedule; the schedule is frozen in analysis_plan.json before training, so changing it changes the pre-registration fingerprint')
    p.add_argument('--no-randomize-order', dest='randomize_order', action='store_false', help='keep the matrix row order instead of a randomized complete-block schedule (the order is then recorded as not randomized; order drift cannot be separated from the arm)')
    p.add_argument('--assumed-paired-sd', type=float, default=None, help='pre-run assumption for the paired across-seed SD (e.g. from a pilot); with --baseline-margin this makes the design-readiness audit check planned power before launching runs')
    p.add_argument('--expected-attrition', type=float, default=0.0, help='pre-run expected fraction of planned runs lost to pre-specified protocol failures; a complete pair needs both arms, so the readiness audit plans for n*(1-attrition)^2 analyzable pairs and freezes this assumption in analysis_plan.json')
    p.add_argument('--require-design-ready', action='store_true', help='fail before training when the design-readiness audit has a structural failure (missing replication/hypothesis, incomplete schedule, absent control pair, or underpowered for the stated margin)')
    p.add_argument('--require-plan-lock', action='store_true', help='fail when this invocation would replace an existing analysis_plan.json with a different fingerprint (changed seeds/alpha/margins/hypotheses); without it the drift is recorded and warned about, but the run proceeds')
    p.add_argument('--require-exclusion-audit', action='store_true', help='fail after training when the pre-specified attrition audit finds a duplicate run/seed, an unaccounted exclusion, or an arm left below 3 analyzable seeds; the audit is always written to exclusion_audit.json')
    p.add_argument('--output-dir', default='runs/ablation')
    p.add_argument('--force', action='store_true')
    p.add_argument('--allow-uncertified', action='store_true', help='allow running with <3 seeds (certification will be disabled); by default <3 seeds is an error')
    try:
        apply_config_to_parser(p, known.config, strict=True,
                               allowed_unknown=('ablations',))
    except ValueError as exc:
        p.error(str(exc))
    return p, known.config, ablations, hypotheses

def main():
    p, config_path, ablations, hypotheses = build_arg_parser()
    args = p.parse_args()
    if not args.simple_baselines_enabled and (args.auto_simple_baselines
                                              or args.require_simple_baselines):
        p.error('--no-simple-baselines cannot be combined with '
                '--auto-simple-baselines or --require-simple-baselines: the '
                'calibration is explicitly disabled, so those requests cannot '
                'be honored')
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        p.error(f'--lr {args.lr} must be finite and positive')
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        p.error(f'--weight-decay {args.weight_decay} must be finite and >= 0')
    if not math.isfinite(args.baseline_margin) or args.baseline_margin < 0.0:
        p.error(f'--baseline-margin {args.baseline_margin} must be finite and >= 0')
    if not math.isfinite(args.equivalence_margin) or args.equivalence_margin < 0.0:
        p.error(f'--equivalence-margin {args.equivalence_margin} must be finite and >= 0 '
                f'(0 disables equivalence testing)')
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        p.error(f'--alpha {args.alpha} must be finite and strictly between 0 and 1')
    if args.equivalence_margin > 0.0 and args.alpha >= 0.5:
        p.error('--equivalence-margin requires --alpha < 0.5 so the TOST '
                'confidence interval 1 - 2*alpha is positive')
    if not math.isfinite(args.expected_attrition) or not 0.0 <= args.expected_attrition < 1.0:
        p.error(f'--expected-attrition {args.expected_attrition} must be finite and in [0, 1)')
    for flag, value in (('--max-steps', args.max_steps),
                        ('--batch-size', args.batch_size),
                        ('--seq-len', args.seq_len),
                        ('--val-batches', args.val_batches),
                        ('--curve-batches', args.curve_batches),
                        ('--log-every', args.log_every),
                        ('--stats-every', args.stats_every)):
        if value < 1:
            p.error(f'{flag} {value} must be >= 1')
    if args.seq_len < 2:
        # The synthetic Markov corpus labels y[:, :-1] = x[:, 1:], so a
        # one-token window leaves every target at ignore_index and no row can
        # be trained or checked.
        p.error(f'--seq-len {args.seq_len} must be >= 2 for next-token '
                'supervision')
    # A `seed:` supplied by --config is a parser default, not an explicit CLI
    # override: an explicit `--seeds 42 43 44` must not be replaced by one
    # config-seeded value (same precedence rule as train_gpt2.py).
    seed_names = _explicit_arg_names(sys.argv[1:], p, config_path)
    if args.seed is not None and ('seed' in seed_names or 'seeds' not in seed_names):
        raw_seeds = [args.seed]
    else:
        raw_seeds = list(args.seeds or DEFAULT_SEEDS)
    # A `--config` value is a parser default and bypasses argparse's `type=int`
    # conversion. Reject `42.9`/`true` instead of truncating them to a
    # different seed than the configuration names.
    malformed_seeds = [seed for seed in raw_seeds if not _is_seed_id(seed)]
    if malformed_seeds:
        p.error(f'seeds must be integer ids, got {malformed_seeds!r}')
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    # Keep `args.seeds` equal to the resolved set: protocol checks (e.g. the
    # simple-baseline calibration) must compare against the seeds actually
    # run, not the parser default when the deprecated --seed shorthand was used.
    args.seeds = list(seeds)
    if len(seeds) != len(raw_seeds):
        print(f'[warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed is the same run, not an independent replication)')
    if len(seeds) < 3 and not args.allow_uncertified:
        p.error(f'{len(seeds)} seed(s) selected: certification requires >= 3 independent seeds; use --allow-uncertified to run anyway (results will be descriptive only)')
    try:
        matrix_report = validate_matrix(ablations, hypotheses, args.model)
    except ValueError as exc:
        print(f'[matrix] FAIL: {exc}')
        sys.exit(1)
    print(f"[matrix] ok: {len(matrix_report['rows'])} rows, controlled optimizer/schedule/data, "
          f"every row has a hypothesis, leave-one-out coverage {matrix_report['leave_one_out']}")
    # Resolve the rows that will actually execute, and collapse alias rows
    # (identical effective model config) onto one canonical run. The schedule
    # is over canonical runs only, so an alias cannot consume a slot in a
    # block and unbalance the design.
    selected = list(dict.fromkeys(name for name in args.names if name in ablations))
    for unknown in sorted(set(args.names) - set(ablations)):
        print(f"[warn] unknown ablation '{unknown}', skipping")
    config_groups: dict[str, list[str]] = {}
    for name in selected:
        config_groups.setdefault(
            effective_config_key(args.model, ablations[name]), []).append(name)
    alias_of: dict[str, str] = {}
    canonical_rows: list[str] = []
    for group in config_groups.values():
        canonical = group[0]
        canonical_rows.append(canonical)
        for alias in group[1:]:
            alias_of[alias] = canonical
    if alias_of:
        print(f'[design] {len(alias_of)} alias row(s) reuse a canonical run and are '
              f'outside the execution schedule: {alias_of}')
    run_schedule = None
    if args.randomize_order and canonical_rows:
        run_schedule = randomized_complete_block_schedule(
            canonical_rows, [str(seed) for seed in seeds], seed=args.schedule_seed)
        print(f"[design] randomized complete-block schedule: {len(canonical_rows)} row(s) x "
              f"{len(seeds)} seed(s) = {run_schedule['n_units']} run(s) in "
              f"{run_schedule['n_blocks']} time block(s), fingerprint "
              f"{run_schedule['fingerprint'][:12]}; every row runs once per block")
    elif not canonical_rows:
        print('[design] no rows selected: no execution schedule was generated')
    else:
        print('[design] WARNING: --no-randomize-order keeps the input row order; '
              'a time drift is then confounded with the arm and cannot be '
              'separated from it after the fact')
    exclusion_policy = normalize_exclusion_policy()
    analysis_plan = build_analysis_plan(
        ablations, hypotheses, matrix_report, args.baselines, args.model, seeds,
        alpha=args.alpha, margin=args.baseline_margin,
        equivalence_margin=args.equivalence_margin, run_schedule=run_schedule,
        exclusion_policy=exclusion_policy,
        expected_attrition=args.expected_attrition)
    matrix_report['analysis_plan'] = analysis_plan
    print(f"[plan] {analysis_plan['n_contrasts']} pre-registered confirmatory contrast(s) "
          f"(fingerprint {analysis_plan['fingerprint'][:12]}); every test outside this "
          f"registry is reported as exploratory")
    if args.check_matrix:
        print(json.dumps(matrix_report, indent=2))
        return
    baseline_tuning = None
    if args.require_baseline_tuning and not args.baseline_tuning:
        p.error('--require-baseline-tuning needs --baseline-tuning <tuning_summary.json>: '
                'without the tune.py report there is no evidence that the baselines were '
                'tuned at this matrix recipe')
    if args.baseline_tuning:
        baseline_tuning = load_baseline_tuning(args.baseline_tuning, args, ablations, seeds)
        if baseline_tuning is None and args.require_baseline_tuning:
            raise SystemExit(
                f'[tune-cert] ERROR: --require-baseline-tuning is set but '
                f'{args.baseline_tuning} is missing, unreadable or describes a different '
                f'protocol; refusing to train/compare against baselines without a valid '
                f'tuning certificate.')
        if (baseline_tuning is not None and args.require_baseline_tuning
                and not baseline_tuning.get('certified')):
            raise SystemExit(
                f'[tune-cert] ERROR: --require-baseline-tuning is set but the report does not '
                f'certify this recipe: '
                f'{"; ".join(baseline_tuning.get("why_not_certified") or [])}. '
                f'Rerun the matrix at the baseline\'s selected (lr, weight_decay) cell, or '
                f'tune the baselines with this recipe and held-back --confirm-seeds.')
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    with open(base / 'exclusion_policy.json', 'w') as f:
        json.dump(exclusion_policy, f, indent=2)
    if run_schedule is not None:
        with open(base / 'run_schedule.json', 'w') as f:
            json.dump(run_schedule, f, indent=2)
    schedule_arm_of = (lambda unit: unit.rsplit('::rep=', 1)[0]
                       if '::rep=' in unit else unit)
    control_pairs = []
    if 'dytcd_full' in ablations and 'dytcd_ls1e5' in ablations:
        # Explicit-default LayerScale is an identity pair: the two rows build
        # the same effective model, so its contrast is a known null that tests
        # the analysis pipeline end to end.
        control_pairs.append(('dytcd_full', 'dytcd_ls1e5'))
    expected_schedule_units = (
        [f'{name}::rep={seed}' for name in canonical_rows for seed in seeds]
        if run_schedule is not None else None)
    # Prospective power is planned over the *primary* confirmatory tier, not
    # every secondary sweep: the hierarchical gatekeeping test only pays the
    # family-wise price on the contrasts that carry the main claim. A focused
    # run with no primary contrast is planned as a single test.
    n_confirmatory_tests = max(
        1, sum(1 for entry in analysis_plan['contrasts']
               if entry.get('tier') == 1))
    design_readiness = design_readiness_audit(
        seeds, ablations, hypotheses, schedule=run_schedule,
        expected_units=expected_schedule_units,
        schedule_arm_of=schedule_arm_of, margin=args.baseline_margin,
        equivalence_margin=args.equivalence_margin,
        assumed_paired_sd=args.assumed_paired_sd, alpha=args.alpha,
        identity_controls=control_pairs, exclusion_policy=exclusion_policy,
        n_confirmatory_tests=n_confirmatory_tests,
        expected_attrition=args.expected_attrition)
    if run_schedule is not None:
        design_readiness['schedule_balance'] = schedule_balance_report(
            run_schedule, arm_of=schedule_arm_of)
    if (args.assumed_paired_sd is not None and args.baseline_margin > 0.0
            and len(seeds) >= 2):
        # Calibrate the actual paired-test code against the pre-run assumption:
        # under the alternative the planned effect is -margin (candidate lower
        # loss). This is a preparation check, not a result. A one-seed
        # --allow-uncertified run has no paired test to calibrate, and the
        # replication/power checks already record that failure.
        expected_pairs = int(math.floor(
            len(seeds) * (1.0 - args.expected_attrition) ** 2 + 1e-12))
        if expected_pairs >= 2:
            design_readiness['simulated_power_check'] = simulate_paired_design(
                n_pairs=expected_pairs, effect=-float(args.baseline_margin),
                paired_sd=float(args.assumed_paired_sd), reps=1000,
                alpha=args.alpha / n_confirmatory_tests,
                seed=args.schedule_seed)
            design_readiness['simulated_power_check']['n_confirmatory_tests'] = (
                n_confirmatory_tests)
            design_readiness['simulated_power_check']['expected_attrition'] = (
                args.expected_attrition)
    with open(base / 'design_readiness.json', 'w') as f:
        json.dump(design_readiness, f, indent=2)
    for check in design_readiness['checks']:
        if check['status'] != 'pass':
            print(f"[design] {check['status'].upper()}: {check['id']}: {check['detail']}")
    if not design_readiness['ready']:
        if args.require_design_ready:
            raise SystemExit(
                '[design] ERROR: --require-design-ready is set but the pre-flight '
                'design audit has structural failure(s): '
                + '; '.join(check['detail'] for check in design_readiness['checks']
                            if check['status'] == 'fail'))
        print('[design] WARNING: the design audit has a structural failure; the run '
              'will proceed, but the failure is recorded in design_readiness.json '
              'and the comparison cannot claim a clean confirmatory design')
    # Pre-registration drift: a rerun that changes the seeds, alpha, margins or
    # hypotheses changes the plan fingerprint. Silently replacing the old plan
    # would erase the fact that the confirmatory family moved after results
    # were seen, so the change is recorded (and can be made fatal).
    plan_path = base / 'analysis_plan.json'
    previous_plan = None
    if plan_path.exists():
        try:
            previous_plan = json.loads(plan_path.read_text())
        except (json.JSONDecodeError, OSError):
            previous_plan = None
    previous_fingerprint = (previous_plan or {}).get('fingerprint')
    plan_changed = bool(previous_fingerprint
                        and previous_fingerprint != analysis_plan['fingerprint'])
    analysis_plan['plan_changed'] = plan_changed
    analysis_plan['previous_fingerprint'] = previous_fingerprint
    analysis_plan['plan_change_policy'] = (
        'the fingerprint covers the confirmatory contrast registry, seeds, alpha, '
        'margins and the randomized execution schedule; a changed fingerprint '
        'means the analysis plan moved after it was first fixed')
    if plan_changed:
        message = (f'the analysis plan changed since the previous run in this '
                   f'output directory ({str(previous_fingerprint)[:12]} -> '
                   f'{analysis_plan["fingerprint"][:12]}; seeds, alpha, margins, '
                   f'hypotheses or the randomized schedule); the change is recorded '
                   f'as post-hoc unless this is a fresh pre-registration')
        if args.require_plan_lock:
            raise SystemExit(f'[plan] ERROR: {message}. Remove the old output '
                             f'directory for a deliberate re-registration, or drop '
                             f'--require-plan-lock to proceed with the drift recorded.')
        print(f'[plan] WARNING: {message}')
    with open(plan_path, 'w') as f:
        json.dump(analysis_plan, f, indent=2)
    run_metadata = collect_run_metadata(seed=seeds[0], extra={'seeds': seeds})
    if args.sanity_check:
        from scripts.sanity_overfit import run_preflight
        from scripts.train_runtime import _MODEL_SHAPES
        norms = sorted({ablations[name].get('norm_name', 'dytcd') for name in args.names if name in ablations})
        # The rows are trained on the selected --model, so the preflight must
        # exercise the same (n_layers, dim, n_heads) instead of
        # sanity_overfit.py's 2-layer dim=64 default; the small preflight
        # workload (steps/batch/context) is kept on purpose. Per-row knobs
        # (p/k/layerscale variants) are not forwarded: the preflight only
        # needs to catch wiring/NaN/gradient bugs, which are shape- and
        # norm-related.
        n_layers, dim, n_heads = _MODEL_SHAPES.get(args.model, _MODEL_SHAPES['gpt2-124m'])
        arch = ['--dim', str(dim), '--n-layers', str(n_layers), '--n-heads', str(n_heads)]

        run_preflight(norms or ['dytcd'], seeds[0], base, arch_args=arch)
        sanity_report_path = base / 'sanity_overfit.json'
    else:
        sanity_report_path = None
    multi = len(seeds) > 1
    all_summaries = []
    trained: dict[str, str] = {}

    def _report_curve_regimes(agg: dict) -> None:
        """Print the row's per-seed curve verdicts next to its aggregate."""
        regimes = agg.get('curve_classification_counts') or {}
        if not regimes:
            print(f"  [curve] {agg.get('name')}: no curve diagnosis in the cached "
                  f"summaries (pre-monitoring row); rerun with --force to record the "
                  f"train/val regimes")
            return
        print(f"  [curve] {agg['name']}: {regimes}")
        off = {k: v for k, v in regimes.items() if k not in ('healthy', 'insufficient_data')}
        if off:
            print(f"  [curve] WARNING: {agg['name']} has seed(s) in {off}: the val_loss "
                  f"delta may reflect that regime (underfit/stalled/overfit/diverging) as "
                  f"much as the module; read curve_val_loss and the per-seed verdicts "
                  f"before attributing it to the norm")

    def _prepare_row(name: str):
        """Create a row directory and locate its per-seed summary paths."""
        row_dir = base / name
        row_dir.mkdir(parents=True, exist_ok=True)
        summary_paths = {
            seed: (row_dir / f'seed{seed}' / 'summary.json' if multi
                   else row_dir / 'summary.json')
            for seed in seeds}
        legacy = row_dir / 'summary.json'
        if (multi and legacy.exists()
                and not (row_dir / 'summary.preseed.json').exists()
                and not any(path.exists() for path in summary_paths.values())):
            os.replace(legacy, row_dir / 'summary.preseed.json')
            print(f'[keep] {name}: moved pre-replication single-seed summary to '
                  f'summary.preseed.json')
        return row_dir, summary_paths

    row_layout = {name: _prepare_row(name) for name in canonical_rows}
    row_summaries: dict[str, list] = {name: [] for name in canonical_rows}
    row_done: set[str] = set()

    def _finish_row(name: str) -> None:
        cfg_kwargs = ablations[name]
        row_dir, _ = row_layout[name]
        seed_summaries = sorted(row_summaries[name], key=lambda item: item.get('seed'))
        agg = aggregate_seed_summaries(name, seed_summaries, cfg_kwargs,
                                       hypotheses.get(name),
                                       exclusion_policy=exclusion_policy)
        _write_summary(row_dir / 'summary.json', agg)
        _report_curve_regimes(agg)
        trained[effective_config_key(args.model, cfg_kwargs)] = name
        row_done.add(name)
        all_summaries.append(agg)

    if run_schedule is not None:
        schedule_entries = run_schedule['entries']
        schedule_fingerprint = run_schedule['fingerprint']
        randomized = True
    else:
        # Explicit opt-out: a deterministic row-major sequence, recorded as
        # not randomized so the order audit can say the design did not address
        # time drift rather than pretend it did.
        schedule_entries = []
        for block_index, seed in enumerate(seeds):
            for name in canonical_rows:
                schedule_entries.append({
                    'unit': name, 'replicate': str(seed),
                    'run_id': f'{name}::rep={seed}', 'block': block_index,
                    'position': len(schedule_entries)})
        schedule_fingerprint = None
        randomized = False

    for entry in schedule_entries:
        name = entry.get('unit')
        if name not in row_summaries or name in row_done:
            continue
        try:
            seed = int(entry.get('replicate'))
        except (TypeError, ValueError):
            raise SystemExit(
                f'[design] ERROR: schedule entry {entry!r} has no integer seed')
        if seed not in seeds:
            raise SystemExit(
                f'[design] ERROR: schedule entry {entry!r} names seed {seed} outside '
                f'the registered seed set {seeds}')
        cfg_kwargs = ablations[name]
        _, summary_paths = row_layout[name]
        path = summary_paths[seed]
        if path.exists() and (not args.force):
            summary = _load_summary(path)
            if isinstance(summary, dict):
                if cached_seed_matches(summary, name, cfg_kwargs, args, seed):
                    print(f'[skip] {name} seed={seed} already done, use --force to rerun')
                    row_summaries[name].append(summary)
                    if len(row_summaries[name]) == len(seeds):
                        _finish_row(name)
                    continue
                print(f'[stale] {name} seed={seed}: cached summary recorded a different '
                      f'model/cfg/protocol; retraining and overwriting')
        order_info = {
            'position': entry.get('position'),
            'block': entry.get('block'),
            'run_id': entry.get('run_id'),
            'randomized': randomized,
            'schedule_fingerprint': schedule_fingerprint,
            'schedule_seed': (run_schedule.get('seed') if run_schedule else None),
        }
        row_summaries[name].append(run_one(
            name, cfg_kwargs, args, path.parent, run_metadata=run_metadata,
            hypothesis=hypotheses.get(name), seed=seed, order_info=order_info))
        if len(row_summaries[name]) == len(seeds):
            _finish_row(name)

    unresolved_rows = [name for name in canonical_rows if name not in row_done]
    if unresolved_rows:
        raise SystemExit(
            f'[design] ERROR: the run schedule did not complete row(s) '
            f'{unresolved_rows}; no aggregate can be written for them. This is a '
            f'schedule/execution bug, not a scientific exclusion.')

    # Alias rows are deterministic copies of a canonical trained row. They are
    # written after the canonical run so they can never be scheduled as an
    # independent arm (which would double-count one run in the block design).
    for alias, canonical in alias_of.items():
        cfg_kwargs = ablations[alias]
        row_dir, summary_paths = _prepare_row(alias)
        source_row_dir, _ = row_layout[canonical]
        source_paths = {
            seed: (source_row_dir / f'seed{seed}' / 'summary.json' if multi
                   else source_row_dir / 'summary.json')
            for seed in seeds}
        reused = [_load_summary(path) for path in source_paths.values()]
        if not all(isinstance(summary, dict) for summary in reused):
            raise SystemExit(
                f'[design] ERROR: alias row {alias!r} reuses {canonical!r}, but its '
                f'per-seed summaries are incomplete; refusing to write a row that '
                f'is not actually a copy of a completed run')
        reused = [dict(summary, name=alias, cfg=cfg_kwargs,
                       reused_from=canonical, hypothesis=hypotheses.get(alias))
                  for summary in reused]
        for seed, summary in zip(seeds, reused):
            if multi:
                _write_summary(summary_paths[seed], summary)
        agg = aggregate_seed_summaries(alias, reused, cfg_kwargs,
                                       hypotheses.get(alias),
                                       exclusion_policy=exclusion_policy)
        _write_summary(row_dir / 'summary.json', agg)
        print(f'[alias] {alias}: identical model config to {canonical}, reusing its '
              f'summaries (no retrain)')
        _report_curve_regimes(agg)
        all_summaries.append(agg)
    agg_path = base / 'all_summaries.json'
    previous: list = []
    if agg_path.exists():
        try:
            previous = json.loads(agg_path.read_text())
            if not isinstance(previous, list):
                previous = []
        except json.JSONDecodeError:
            previous = []
    ordered = merge_summaries(previous, all_summaries)
    with open(agg_path, 'w') as f:
        json.dump(ordered, f, indent=2)
    # Accumulated rows whose recorded recipe differs from this invocation are
    # reported but excluded from both comparisons: a delta produced by a
    # different lr/step/batch/seq recipe is not a property of the module.
    comparable, stale_rows = partition_rows_by_protocol(ordered, args)
    if stale_rows:
        print(f"[compare] WARNING: {len(stale_rows)} accumulated row(s) excluded from the comparisons "
              f"(trained under a different model/cfg/protocol than this invocation): "
              f"{[r['name'] for r in stale_rows]}; rerun them under this recipe to include them")
    # Pre-specified attrition: the comparisons below silently drop a
    # non-finite seed from a pooled mean, so the denominator is audited
    # explicitly. Every planned (row, seed) is a record; missing or
    # non-finite runs are excluded only for the reasons frozen in
    # exclusion_policy.json, and an arm that falls below the certification
    # replication count is reported instead of being averaged over survivors.
    row_by_name = {row.get('name'): row for row in all_summaries
                   if isinstance(row, dict) and row.get('name')}
    stale_names = {row.get('name') for row in stale_rows
                   if isinstance(row, dict)}
    planned_runs = []
    for entry in schedule_entries:
        name = entry.get('unit')
        if name not in canonical_rows:
            continue
        try:
            seed = int(entry.get('replicate'))
        except (TypeError, ValueError):
            continue
        agg = row_by_name.get(name) or {}
        item = next((p for p in (agg.get('per_seed') or [])
                     if isinstance(p, dict) and p.get('seed') == seed), None)
        planned_runs.append({
            'run_id': entry.get('run_id') or f'{name}::rep={seed}',
            'arm': name,
            'seed': seed,
            'val_loss': item.get('val_loss') if item else None,
            'present': item is not None,
            'protocol_mismatch': name in stale_names,
            'curve_classification': (item or {}).get('curve_classification'),
        })
    exclusion_audit = run_exclusion_audit(planned_runs, exclusion_policy)
    exclusion_audit['schedule_fingerprint'] = (
        run_schedule.get('fingerprint') if run_schedule else None)
    with open(base / 'exclusion_audit.json', 'w') as f:
        json.dump(exclusion_audit, f, indent=2)
    if exclusion_audit['n_excluded'] or exclusion_audit['n_sensitivity']:
        print(f"[exclusion] {exclusion_audit['n_planned']} planned run(s): "
              f"{exclusion_audit['n_analyzed']} retained, "
              f"{exclusion_audit['n_sensitivity']} sensitivity-only, "
              f"{exclusion_audit['n_excluded']} excluded "
              f"(policy {exclusion_audit['policy_fingerprint'][:12]})")
    for warning in exclusion_audit['warnings']:
        print(f'[exclusion] WARNING: {warning}')
    if not exclusion_audit['ready']:
        message = ('the attrition audit found an unusable design: '
                   + '; '.join(exclusion_audit['violations']))
        if args.require_exclusion_audit:
            raise SystemExit('[exclusion] ERROR: ' + message)
        print('[exclusion] WARNING: ' + message
              + '; the comparisons below are reported with the attrition '
                'visible in exclusion_audit.json')
    comparison = baseline_comparison(comparable, args.baselines, margin=args.baseline_margin,
                                     alpha=args.alpha,
                                     equivalence_margin=args.equivalence_margin)
    simple_path = Path(args.simple_baselines) if args.simple_baselines else base / 'simple_baselines.json'
    floor_data = (load_simple_baselines(simple_path, args)
                  if args.simple_baselines_enabled else None)
    if (comparison is not None and floor_data is None
            and args.simple_baselines_enabled and args.auto_simple_baselines):
        floor_data = run_simple_baseline_calibration(simple_path, seeds, args)
    if comparison is not None and floor_data is not None:
        comparison['simple_baseline'] = attach_simple_baseline_calibration(
            comparison, floor_data, simple_path, alpha=args.alpha)
        comparison['calibration_status'] = 'attached'
    elif comparison is not None and not args.simple_baselines_enabled:
        # The opt-out must be visible in the artifact, not just on stdout: a
        # reader of baseline_comparison.json otherwise sees the same file as a
        # fully calibrated comparison.
        comparison['simple_baseline'] = {
            'enabled': False, 'status': 'explicitly_disabled',
            'reason': 'disabled by --no-simple-baselines',
            'bayes_limits': None, 'tuned_simple_ce_nats': None}
        comparison['calibration_status'] = 'explicitly_disabled'
        print('[compare] simple-baseline calibration disabled by --no-simple-baselines: rows are '
              'ranked only against each other, not against the task Bayes limits')
    elif comparison is not None:
        comparison['simple_baseline'] = {
            'enabled': False, 'status': 'missing', 'source': str(simple_path),
            'bayes_limits': None, 'tuned_simple_ce_nats': None,
            'how_to_generate': (f'python3 scripts/baseline_simple.py --output {simple_path} '
                                f'--seeds {" ".join(map(str, seeds))} '
                                f'--vocab-size {_PROTOCOL_MODEL_VOCAB[args.model]} '
                                f'--batch-size {args.batch_size} --seq-len {args.seq_len} '
                                f'--val-batches {max(1, int(getattr(args, "val_batches", 20)))}')}
        comparison['calibration_status'] = 'missing'
        print(f"[compare] WARNING: no tuned simple-baseline calibration at {simple_path}; the rows below are "
              f"only ranked against each other. Generate the task's Bayes limits and tuned count model with "
              f"'{comparison['simple_baseline']['how_to_generate']}' (or pass --auto-simple-baselines / "
              f"--require-simple-baselines)")
        if args.require_simple_baselines:
            raise SystemExit(
                f'[compare] ERROR: --require-simple-baselines is set but no calibration exists at '
                f'{simple_path}; refusing to write baseline_comparison.json without the tuned count-model '
                f'and Bayes-limit reference. Generate it with '
                f"'{comparison['simple_baseline']['how_to_generate']}'.")
    if comparison is not None:
        if baseline_tuning is not None:
            comparison['baseline_tuning'] = attach_baseline_tuning(comparison, baseline_tuning)
        else:
            baseline_norms = [ablations[name].get('norm_name', name)
                              for name in args.baselines if name in ablations]
            comparison['baseline_tuning'] = {
                'enabled': False,
                'status': ('missing' if args.baseline_tuning else 'not_requested'),
                'source': args.baseline_tuning,
                'certified': False,
                'certified_against_tuned_reference': False,
                'reading': ('the baseline comparison below is against the fixed matrix recipe, '
                            'not a certified tuned baseline; pass --baseline-tuning plus '
                            '--require-baseline-tuning to make this claim fail-closed'),
                'how_to_generate': (
                    f"python3 scripts/tune.py --model {args.model} "
                    f"--arms {' '.join(baseline_norms) or 'ln rmsnorm'} "
                    f"--max-steps {args.max_steps} --batch-size {args.batch_size} "
                    f"--seq-len {args.seq_len} "
                    f"--val-batches {max(1, int(getattr(args, 'val_batches', 20)))} "
                    f"--lrs {args.lr:g} 3e-4 1.2e-3 "
                    f"--weight-decays {args.weight_decay:g} "
                    f"--seeds {' '.join(map(str, seeds))} "
                    f"--confirm-seeds <held-back seeds> --output-dir {base / 'tune'}"),
            }
            print('[compare] WARNING: no baseline-tuning certificate; the significance test below '
                  'uses the fixed-recipe baselines. Generate one with '
                  f"'{comparison['baseline_tuning']['how_to_generate']}' and rerun with "
                  '--baseline-tuning (or --require-baseline-tuning to fail closed).')
        comparison['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'baseline_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    print_baseline_comparison(comparison)
    # The second half of the ablation contract: each claimed mechanism of the
    # main proposal must be shown to be load-bearing *on its own*, with a
    # paired Full-vs-Full-minus-X test over the same seeds.
    loo_comparison = leave_one_out_comparison(comparable, margin=args.baseline_margin,
                                              alpha=args.alpha,
                                              equivalence_margin=args.equivalence_margin)
    if loo_comparison is not None:
        loo_comparison['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'leave_one_out_comparison.json', 'w') as f:
            json.dump(loo_comparison, f, indent=2)
    print_leave_one_out_comparison(loo_comparison)
    # The third half of the contract: Full combines two axes, so report the
    # 2x2 main effects and their interaction rather than inferring synergy
    # from Full's point estimate alone.
    factorial = factorial_comparison(comparable, margin=args.baseline_margin,
                                     alpha=args.alpha,
                                     equivalence_margin=args.equivalence_margin)
    if factorial is not None:
        factorial['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'factorial_comparison.json', 'w') as f:
            json.dump(factorial, f, indent=2)
    print_factorial_comparison(factorial)
    # The fourth report closes the "three orthogonal axes" claim: the two 2x2
    # designs estimate per-channel x depth separately per activation family,
    # the 2x2x2 design estimates the cross-family interactions and tests
    # additivity with TOST instead of inferring it from a null result.
    three_way = three_way_factorial_comparison(
        comparable, margin=args.baseline_margin, alpha=args.alpha,
        equivalence_margin=args.equivalence_margin)
    if three_way is not None:
        three_way['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'three_way_factorial_comparison.json', 'w') as f:
            json.dump(three_way, f, indent=2)
    print_three_way_factorial_comparison(three_way)
    # The attention-logits axis (DyT-QK) is independent of the normalization
    # family, so it gets its own per-host attribution and cross-host
    # replication report instead of being folded into the baseline family.
    qk_axis = qk_axis_comparison(comparable, margin=args.baseline_margin,
                                 alpha=args.alpha,
                                 equivalence_margin=args.equivalence_margin)
    if qk_axis is not None:
        qk_axis['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'qk_axis_comparison.json', 'w') as f:
            json.dump(qk_axis, f, indent=2)
    print_qk_axis_comparison(qk_axis)
    # Study-level multiplicity: the four reports above each correct within
    # their own family, but the paper-level claim is made across all of them.
    # Pool only the pre-registered confirmatory contrasts and report what the
    # global Holm correction confirms (or demotes).
    study = study_level_correction(
        {'baseline': comparison, 'leave_one_out': loo_comparison,
         'factorial': factorial, 'factorial3': three_way,
         'qk_axis': qk_axis},
        analysis_plan, alpha=args.alpha)
    if study is not None:
        with open(base / 'study_level_tests.json', 'w') as f:
            json.dump(study, f, indent=2)
    print_study_level_correction(study)
    # Execution-order audit: a significant within-arm rank correlation is a
    # design warning, not an arm result. It is written next to the comparisons
    # so a reader can see whether the randomized schedule actually held.
    order_audit = run_order_audit(comparable, run_schedule)
    with open(base / 'run_order_audit.json', 'w') as f:
        json.dump(order_audit, f, indent=2)
    manifest = dict(run_metadata)
    manifest['git'] = git_revision()
    manifest['hyperparameters'] = vars(args)
    manifest['config_file'] = (file_fingerprint(args.config)
                               if args.config else None)
    manifest['seeds'] = seeds
    manifest['matrix_validation'] = matrix_report
    manifest['rows'] = [row['name'] for row in ordered]
    manifest['rows_excluded_protocol_mismatch'] = stale_rows
    # Curve regimes per row (per-seed classification counts): the run manifest
    # keeps the train/val regime each comparison was measured in, so a future
    # reader can tell a healthy falling-falling delta from an early-stopped
    # underfit or an overfitting one without re-reading every summary.json.
    manifest['curve_regimes'] = {
        row.get('name'): row.get('curve_classification_counts')
        for row in ordered if isinstance(row, dict) and row.get('name')
    }
    manifest['baseline_comparison'] = comparison
    manifest['leave_one_out_comparison'] = loo_comparison
    manifest['factorial_comparison'] = factorial
    manifest['three_way_factorial_comparison'] = three_way
    manifest['qk_axis_comparison'] = qk_axis
    manifest['analysis_plan'] = analysis_plan
    manifest['study_level_tests'] = study
    manifest['run_schedule'] = run_schedule
    manifest['exclusion_policy'] = exclusion_policy
    manifest['exclusion_audit'] = exclusion_audit
    manifest['design_readiness'] = design_readiness
    manifest['run_order_audit'] = order_audit
    if sanity_report_path is not None:
        manifest['overfit_sanity_check'] = {'enabled': True, **file_fingerprint(sanity_report_path)}
    else:
        manifest['overfit_sanity_check'] = {'enabled': False}
    # Durable artifacts, so the manifest says which files the comparison above
    # was computed from. `all_summaries.json` can carry every per-step curve
    # and is recorded by size+mtime instead of a full re-hash.
    manifest = finish_metadata(manifest, artifacts={
        'all_summaries': artifact_record(base / 'all_summaries.json'),
        'analysis_plan': artifact_record(base / 'analysis_plan.json'),
        'baseline_comparison': artifact_record(base / 'baseline_comparison.json'),
        'leave_one_out_comparison': artifact_record(base / 'leave_one_out_comparison.json'),
        'factorial_comparison': artifact_record(base / 'factorial_comparison.json'),
        'three_way_factorial_comparison': artifact_record(
            base / 'three_way_factorial_comparison.json'),
        'qk_axis_comparison': artifact_record(base / 'qk_axis_comparison.json'),
        'study_level_tests': artifact_record(base / 'study_level_tests.json'),
        'run_schedule': (artifact_record(base / 'run_schedule.json')
                         if run_schedule is not None else {'enabled': False}),
        'exclusion_policy': artifact_record(base / 'exclusion_policy.json'),
        'exclusion_audit': artifact_record(base / 'exclusion_audit.json'),
        'design_readiness': artifact_record(base / 'design_readiness.json'),
        'run_order_audit': artifact_record(base / 'run_order_audit.json'),
        'sanity_report': file_fingerprint(sanity_report_path) if sanity_report_path else {'enabled': False},
    })
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[done] {len(ordered)} ablations saved to {base} '
          f'({len(seeds)} seed(s): {seeds}; row val_loss = mean +/- std across seeds)')
