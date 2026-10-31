from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path

import torch

from adasquash import ActivationStats, LossSpikeCounter, count_parameters, markov_batch
from adasquash.attention import clear_logit_scratch_cache
from adasquash.design import (classify_run_exclusions,
                              exclusion_policy_fingerprint,
                              normalize_exclusion_policy, order_drift_report,
                              schedule_balance_report)
from adasquash.diagnostics import diagnose_train_val_curve, triage_nonfinite
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.metrics import mean_std_stderr
from adasquash.model import clear_cast_weight_cache, clear_ce_scratch_cache
from adasquash.repro import collect_run_metadata, set_all_seeds
from scripts.train_gpt2 import build_optimizer

from .common import _finite_metric, _is_seed_id
from .matrix import (DEFAULT_SEEDS, _MODEL_BUILDERS, install_residual_outlier)

def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path, run_metadata: dict | None=None, hypothesis: str | None=None, seed: int | None=None, order_info: dict | None=None):
    if seed is None:
        seed = getattr(args, 'seed', None)
    if seed is None:
        seed = (getattr(args, 'seeds', None) or DEFAULT_SEEDS)[0]
    seed = int(seed)
    print(f'\n[abl] {name} (seed={seed}): {cfg_kwargs}')
    print(f'  [hypothesis] {hypothesis or "(none stated - this row cannot be interpreted as a test)"}')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seed)
    # Build with the constructor's init pass disabled:
    # `reset_parameters(seed)` below is the run's single, seeded
    # initialization.
    with torch.device(device):
        model = _MODEL_BUILDERS[args.model](init=False, **cfg_kwargs)
    model.reset_parameters(seed)
    model.to(device)
    n_params = count_parameters(model)
    param_breakdown = parameter_breakdown(model)
    flops = analytic_flops_per_token(model.cfg, seq_len=args.seq_len)
    opt = build_optimizer(model, args.lr, args.weight_decay, 0.9, 0.95)
    rng = torch.Generator(device=device).manual_seed(seed)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    install_residual_outlier(model, layer_idx=args.outlier_layer, channel=args.outlier_channel, scale=args.outlier_scale)
    # The synthetic corpus carries the protocol's outlier token id (the same
    # id the residual-channel probe is paired with); it is independent of the
    # forward hook and must be recorded in every training/validation batch.
    outlier_id = _protocol_outlier_id(args.model)
    losses = []
    stats_windowed = False
    # In-training curve monitoring. The held-out stream for the curve is a
    # *separate* generator (seed + 2): the selection metric's generator
    # (seed + 1) is only consumed after training, so the per-batch val losses
    # that rows are paired on stay bit-identical whether or not the curve ran.
    # Without this a row is summarized by its final number alone, and a delta
    # produced while one arm was still underfit/overfit reads as a norm effect.
    curve_every = int(getattr(args, 'curve_every', 0) or 0)
    if curve_every == 0:
        curve_every = max(1, int(args.max_steps) // 5)
    elif curve_every < 0:
        curve_every = 0
    curve_batches = max(1, int(getattr(args, 'curve_batches', 5) or 5))
    curve_rng = torch.Generator(device=device).manual_seed(seed + 2)
    curve_train: list = []
    curve_val: list = []
    curve_val_stderr: list = []
    curve_steps: list = []
    curve_diagnoses: list = []
    curve_window_start = 0
    t0 = time.time()
    tokens_processed = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    peak_device_memory_mb = None
    for step in range(args.max_steps):
        x, y = markov_batch(rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device, outlier_id=outlier_id)
        tokens_processed += int(x.numel())
        measure = step % args.stats_every == 0
        if measure:
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        loss = model(x, y, return_logits=False)[1]
        # Backward is enqueued first; `loss` is not mutated by it, so the
        # host read below returns the same value.
        loss.backward()
        loss_val = float(loss.detach())
        if not math.isfinite(loss_val):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=args.lr, grad_clip=1.0)
            raise SystemExit(
                f"[abl][nan] '{name}' loss is non-finite at step {step} "
                f"(seed={seed}): triage stage='{triage['stage']}' "
                f"({triage['diagnosis']}); fixes: " + ' | '.join(triage['recommendations'])
            )
        if measure:
            act_stats.detach()
            clear_logit_scratch_cache()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(grad_norm)):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=args.lr, grad_clip=1.0)
            raise SystemExit(
                f"[abl][nan] '{name}' produced a non-finite gradient at step {step} "
                f"(pre-clip norm={float(grad_norm)}, seed={seed}): triage stage="
                f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
                + ' | '.join(triage['recommendations'])
            )
        opt.step()
        opt.zero_grad(set_to_none=True)
        clear_cast_weight_cache()
        loss_spike.update(loss_val)
        losses.append(loss_val)
        if step % args.log_every == 0:
            print(f'  [step {step:4d}] loss={loss_val:.4f}', flush=True)
        if curve_every > 0 and ((step + 1) % curve_every == 0 or step + 1 == args.max_steps):
            window = losses[curve_window_start:]
            curve_window_start = len(losses)
            train_point = sum(window) / max(1, len(window))
            model.eval()
            with torch.inference_mode():
                held_t = torch.stack([
                    model(*markov_batch(curve_rng, model.cfg.ignore_index, args.batch_size,
                                        args.seq_len, device, outlier_id=outlier_id),
                          return_logits=False)[1]
                    for _ in range(curve_batches)
                ])
            model.train()
            # `.tolist()` widens each fp32 value to the same Python double the
            # explicit fp64 cast produced.
            held = held_t.tolist()
            held_stats = mean_std_stderr(held)
            held_stderr = (held_stats['stderr']
                           if held_stats and held_stats['n'] >= 2 else None)
            curve_train.append(train_point)
            curve_val.append(held_stats['mean']
                           if held_stats and held_stats['n'] == len(held)
                           else float('nan'))
            curve_val_stderr.append(held_stderr)
            curve_steps.append(step + 1)
            diagnosis = diagnose_train_val_curve(
                curve_train, curve_val, train_steps=curve_steps, val_steps=curve_steps,
                log_vocab_size=math.log(model.cfg.vocab_size),
                val_stderr=curve_val_stderr,
            )
            curve_diagnoses.append({
                'step': step + 1,
                'classification': diagnosis['classification'],
                'train_trend': diagnosis['train_trend'],
                'val_trend': diagnosis['val_trend'],
                'train_window_loss': train_point,
                'val_loss': curve_val[-1],
                'val_stderr': held_stderr,
                'val_noise_threshold': diagnosis['evidence'].get('val_noise_threshold'),
                'val_change_within_noise': diagnosis['evidence'].get('val_change_within_noise'),
                'gap_change_within_noise': diagnosis['evidence'].get('gap_change_within_noise'),
                'generalization_gap': diagnosis['generalization_gap'],
                'recommendations': diagnosis['recommendations'],
            })
            if diagnosis['classification'] != 'insufficient_data':
                print(f"  [curve] step={step + 1} train_window={train_point:.4f} "
                      f"val={curve_val[-1]:.4f} -> {diagnosis['classification']} "
                      f"(train {diagnosis['train_trend']}, val {diagnosis['val_trend']})", flush=True)
                if diagnosis['classification'] not in ('healthy', 'insufficient_data'):
                    for line in diagnosis['recommendations']:
                        print(f'  [curve]   action: {line}')
    if not losses:
        raise ValueError(f"ablation '{name}' produced no steps")
    # Held-out validation loss: fresh synthetic batches from a generator the
    # optimizer has never seen (seed + 1), evaluated with no gradients. Model
    # comparisons must use this - the training loss above is fitted data.
    n_val_batches = max(1, getattr(args, 'val_batches', 20))
    val_rng = torch.Generator(device=device).manual_seed(seed + 1)
    # Batches are materialized so a non-finite held-out loss can be triaged on
    # the exact batch that produced it (the generator state advances, so
    # regenerating the batch later would not be the same input).
    val_batches = [markov_batch(val_rng, model.cfg.ignore_index, args.batch_size,
                                args.seq_len, device, outlier_id=outlier_id)
                   for _ in range(n_val_batches)]
    model.eval()
    with torch.inference_mode():
        val_losses = [model(x, y, return_logits=False)[1] for x, y in val_batches]
    bad_val = [(i, float(v)) for i, v in enumerate(val_losses) if not math.isfinite(float(v))]
    if bad_val:
        i_bad, v_bad = bad_val[0]
        triage = triage_nonfinite(model, val_batches[i_bad][0], val_batches[i_bad][1],
                                  device_type=device)
        model.train()
        raise SystemExit(
            f"[abl][nan] '{name}' held-out loss is non-finite on {len(bad_val)}/{n_val_batches} "
            f"val batch(es) (first index={i_bad}, value={v_bad}, seed={seed}): triage stage="
            f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
            + ' | '.join(triage['recommendations'])
        )
    val_losses = [float(v) for v in val_losses]
    model.train()
    # `val_loss` is a sample mean over held-out batches; keep the per-batch
    # losses (and their standard error) so comparisons can be *paired* across
    # rows (every row evaluates the identical batches, generated from the
    # fixed `seed + 1` generator) and report significance instead of a
    # bare point estimate.
    val_stats = mean_std_stderr(val_losses)
    val_loss = val_stats['mean']
    stats = act_stats.summary()
    # The knobs every matrix row shares. Recorded per summary so a reported
    # comparison can be audited for controlled variables after the fact.
    controls = {
        'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.95], 'lr': args.lr,
                      'weight_decay': args.weight_decay,
                      'grad_clip': 1.0},
        'schedule': {'type': 'constant', 'lr': args.lr},
        'data': {'generator': 'markov_batch', 'batch_size': args.batch_size,
                 'seq_len': args.seq_len, 'outlier_id': outlier_id,
                 'outlier_scale': args.outlier_scale,
                 'val_seed_offset': 1, 'val_batches': n_val_batches},
        'curve_monitoring': {
            'every': curve_every, 'batches': curve_batches,
            'val_seed_offset': 2,
            'note': ('curve points are held-out; their generator (seed+2) is disjoint '
                     'from the selection-val generator (seed+1), so enabling the curve '
                     'does not change the paired val batches'),
        },
        'train_steps': args.max_steps,
        'seed': seed,
    }
    elapsed_s = time.time() - t0
    if device == 'cuda':
        peak_device_memory_mb = float(
            torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
    summary = {'name': name, 'hypothesis': hypothesis, 'cfg': cfg_kwargs, 'model': args.model, 'n_params': n_params, 'seed': seed, 'seeds': [seed], 'final_loss': losses[-1], 'mean_last_100_loss': sum(losses[-100:]) / max(1, len(losses[-100:])), 'val_loss': val_loss, 'val_losses': val_losses, 'val_loss_std': val_stats['std'], 'val_loss_stderr': val_stats['stderr'], 'val_loss_std_basis': 'val_batches', 'n_val_batches': n_val_batches, 'selection_metric': 'val_loss', 'n_loss_spikes': loss_spike.n_spikes, 'loss_curve': losses, 'act_stats': stats, 'max_abs_pre_norm': stats.get('__summary__', {}).get('max_abs_pre_norm'), 'mean_max_abs_pre_norm': stats.get('__summary__', {}).get('mean_max_abs'), 'mean_excess_kurtosis_pre_norm': stats.get('__summary__', {}).get('mean_excess_kurtosis_pre_norm'), 'controls': controls, 'elapsed_s': elapsed_s, 'tokens_processed': tokens_processed, 'tokens_per_second': (tokens_processed / elapsed_s) if elapsed_s > 0 else None, 'peak_device_memory_mb': peak_device_memory_mb}
    # Per-row model accounting is stored next to the loss.
    summary['parameter_breakdown'] = param_breakdown
    summary['flops_per_token_forward'] = flops['forward_per_token']
    summary['flops_per_token_training_est'] = flops['training_per_token']
    # Execution order is run state, not a nuisance detail: a later run in the
    # same session may see a different thermal/clock regime. Only runs that
    # actually executed under the schedule carry it; a cached summary from an
    # earlier session keeps whatever (possibly absent) order it recorded.
    if order_info is not None:
        summary['run_order'] = dict(order_info)
    if run_metadata is None:
        run_metadata = collect_run_metadata(seed=seed)
    else:
        run_metadata = dict(run_metadata)
        run_metadata['seed'] = seed
    summary['run_metadata'] = run_metadata
    # The curve block: the per-point train/val pair and the regime verdict made
    # at each point, so a row is read as a training curve, not a final number.
    summary['curve_train_window_loss'] = curve_train
    summary['curve_val_loss'] = curve_val
    summary['curve_val_stderr'] = curve_val_stderr
    summary['curve_steps'] = curve_steps
    summary['curve_diagnosis'] = curve_diagnoses[-1] if curve_diagnoses else None
    summary['curve_diagnosis_history'] = curve_diagnoses
    summary['curve_classification'] = (curve_diagnoses[-1]['classification']
                                       if curve_diagnoses else None)
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    final_curve = ''
    if curve_diagnoses:
        last_curve = curve_diagnoses[-1]
        gap = last_curve['generalization_gap']
        gap_str = 'n/a' if gap is None else f'{gap:+.4f}'
        noise = last_curve.get('val_stderr')
        noise_str = '' if noise is None else f', val_se={noise:.4f}'
        final_curve = (f" curve={last_curve['classification']} "
                       f"(train {last_curve['train_trend']}, val {last_curve['val_trend']}, "
                       f"gap={gap_str}{noise_str})")
    print(f"  [done] final_loss={summary['final_loss']:.4f} val_loss={val_loss:.4f} "
          f"max_abs_pre_norm={summary['max_abs_pre_norm']}{final_curve}")
    # Drop the per-row scratch caches.
    clear_ce_scratch_cache()
    return summary

def aggregate_seed_summaries(name: str, summaries: list, cfg_kwargs: dict | None=None,
                             hypothesis: str | None=None,
                             exclusion_policy: dict | None=None) -> dict:
    """Pool per-seed runs into one Mean +/- Std summary.

    The seed is the unit of replication: two runs with different seeds are
    independent samples of the same protocol, while the held-out batches
    inside one run are not (they all come from one generator). The pooled
    `val_loss` is therefore the mean of the per-seed means, `val_loss_std` is
    the sample std across seeds whenever there is more than one seed (with
    `val_loss_std_basis` recording which variability it describes), and the
    per-batch losses are kept tagged by seed so the comparison can pair rows
    seed by seed.

    Every seed also carries its pre-specified exclusion status. A non-finite
    seed still contributes no value to the mean, but it is now visible as
    `excluded` in `per_seed` and `exclusion_status_by_seed` instead of only
    reducing `n_seeds` - the denominator is part of the result.
    """
    per_seed = []
    for s in summaries:
        seed = s.get('seed')
        if seed is None and isinstance(s.get('seeds'), list) and s['seeds']:
            seed = s['seeds'][0]
        per_seed.append({
            'seed': seed,
            'val_loss': s.get('val_loss'),
            'val_losses': s.get('val_losses'),
            'final_loss': s.get('final_loss'),
            'mean_last_100_loss': s.get('mean_last_100_loss'),
            'n_loss_spikes': s.get('n_loss_spikes'),
            'elapsed_s': s.get('elapsed_s'),
            'tokens_processed': s.get('tokens_processed'),
            'tokens_per_second': s.get('tokens_per_second'),
            'peak_device_memory_mb': s.get('peak_device_memory_mb'),
            # The final curve verdict of this run (None for summaries written
            # before curve monitoring existed, or for very short runs).
            'curve_classification': s.get('curve_classification'),
            'curve_diagnosis': s.get('curve_diagnosis'),
            # Present only for runs that executed under this session's
            # randomized schedule; the order audit must not invent one for a
            # cached summary from an earlier, differently ordered session.
            'run_order': s.get('run_order'),
        })
    seed_losses = [p['val_loss'] for p in per_seed if _finite_metric(p['val_loss'])]
    stats = mean_std_stderr(seed_losses) or {'mean': None, 'std': None, 'stderr': None, 'n': 0}
    multi = len(seed_losses) >= 2

    def _mean_of(key):
        vals = [p[key] for p in per_seed if _finite_metric(p.get(key))]
        return sum(vals) / len(vals) if vals else None

    tps_vals = [p['tokens_per_second'] for p in per_seed
                if _finite_metric(p.get('tokens_per_second'))]
    tps_stats = mean_std_stderr(tps_vals)

    # Attrition is part of the row, not an implicit side effect of filtering
    # non-finite values out of the mean. The policy is fixed before training
    # and every seed is classified against it; a diverged-but-finite seed is
    # retained and flagged, never dropped for being unlucky.
    policy = normalize_exclusion_policy(exclusion_policy)
    classified = classify_run_exclusions([
        {'run_id': f'{name}::seed={p.get("seed")}', 'arm': name,
         'seed': p.get('seed'), 'val_loss': p.get('val_loss'),
         'curve_classification': p.get('curve_classification')}
        for p in per_seed], policy=policy)
    classification_by_seed = {str(item.get('seed')): item for item in classified}
    for p in per_seed:
        status = classification_by_seed.get(str(p.get('seed')), {})
        p['exclusion_status'] = status.get('status', 'analyzed')
        p['exclusion_reasons'] = status.get('reasons', [])

    # Carry a single-run summary's batch-level std/stderr only from a run
    # whose loss is actually finite. Copying the first raw summary blindly
    # leaked an excluded (NaN) seed's std/stderr into a row whose pooled
    # val_loss came from a different seed, or was None entirely.
    base_summary = next(
        (s for s in summaries if _finite_metric(s.get('val_loss'))), None)
    agg = dict(base_summary if base_summary is not None
               else (summaries[0] if summaries else {}))
    pooled: list = []
    by_seed: dict = {}
    for p in per_seed:
        if isinstance(p.get('val_losses'), list):
            pooled.extend(p['val_losses'])
            by_seed[str(p['seed'])] = p['val_losses']
    agg.update({
        'name': name,
        'cfg': cfg_kwargs if cfg_kwargs is not None else agg.get('cfg'),
        'hypothesis': hypothesis if hypothesis is not None else agg.get('hypothesis'),
        'seeds': [p['seed'] for p in per_seed],
        'n_seeds': len(seed_losses),
        'per_seed': per_seed,
        'exclusion_policy_fingerprint': exclusion_policy_fingerprint(policy),
        'exclusion_status_by_seed': {
            str(seed): item.get('status')
            for seed, item in classification_by_seed.items()},
        'exclusion_reasons_by_seed': {
            str(seed): item.get('reasons') or []
            for seed, item in classification_by_seed.items()},
        'excluded_seeds': sorted(
            (str(seed) for seed, item in classification_by_seed.items()
             if item.get('status') == 'excluded')),
        'n_seeds_excluded': sum(
            1 for item in classification_by_seed.values()
            if item.get('status') == 'excluded'),
        'val_loss': stats['mean'],
        'val_loss_seed_std': stats['std'] if multi else None,
        'val_loss_seed_stderr': stats['stderr'] if multi else None,
        # A single usable run may contribute its batch-level std/stderr, but
        # only from the summary that supplied the finite val_loss; an
        # excluded/NaN seed's uncertainty must not be attached to another
        # seed's mean (or to a None mean).
        'val_loss_std': (
            stats['std'] if multi else
            (None if base_summary is None
             or not _finite_metric(base_summary.get('val_loss_std'))
             else float(base_summary['val_loss_std']))),
        'val_loss_stderr': (
            stats['stderr'] if multi else
            (None if base_summary is None
             or not _finite_metric(base_summary.get('val_loss_stderr'))
             else float(base_summary['val_loss_stderr']))),
        'val_loss_std_basis': (
            'seeds' if multi else
            (None if base_summary is None
             else base_summary.get('val_loss_std_basis', 'val_batches'))),
        'val_losses': pooled,
        'val_losses_by_seed': by_seed,
        'final_loss': _mean_of('final_loss'),
        'mean_last_100_loss': _mean_of('mean_last_100_loss'),
        # Sum only measured finite values: one NaN seed used to poison the
        # row's elapsed time (and a bool/Inf spike count was admitted as a
        # real event) even though the loss mean was filtered correctly.
        'n_loss_spikes': sum(p['n_loss_spikes'] for p in per_seed
                             if _finite_metric(p.get('n_loss_spikes'))),
        'elapsed_s': sum(float(p['elapsed_s']) for p in per_seed
                         if _finite_metric(p.get('elapsed_s'))),
        # Measured cost of the arm, carried next to the loss delta.
        'tokens_processed': sum(p['tokens_processed'] for p in per_seed
                                if isinstance(p.get('tokens_processed'), int)
                                and not isinstance(p.get('tokens_processed'), bool)),
        'tokens_per_second': None if tps_stats is None else tps_stats['mean'],
        'tokens_per_second_seed_std': (tps_stats['std'] if tps_stats and len(tps_vals) >= 2 else None),
        'peak_device_memory_mb': _mean_of('peak_device_memory_mb'),
        'cost_basis': 'tokens_per_second / peak_device_memory_mb are measured this run; memory is null on CPU',
        # Curve regimes across the row's seeds: a seed whose run was still
        # underfit/stalled or already overfitting/diverging makes its
        # val_loss a statement about that regime as much as about the norm,
        # so the aggregate keeps the verdicts visible instead of hiding them
        # behind one mean.
        'curve_classification_counts': dict(Counter(
            p['curve_classification'] for p in per_seed if p.get('curve_classification'))),
        'curve_classifications_by_seed': {
            str(p['seed']): p.get('curve_classification') for p in per_seed},
        'curve_last_generalization_gap_by_seed': {
            str(p['seed']): (p.get('curve_diagnosis') or {}).get('generalization_gap')
            for p in per_seed},
        # Per-seed execution position, so run_order_audit can test whether the
        # realized order predicts the outcome after removing the arm mean.
        'run_order_by_seed': {
            str(p['seed']): p.get('run_order') for p in per_seed
            if p.get('run_order') is not None},
    })
    return agg

def run_order_audit(summaries: list, schedule: dict | None=None) -> dict:
    """Test whether the realized execution order predicts the held-out loss.

    Only runs that actually executed under this session's schedule carry a
    `run_order`; cached summaries from an older session are reported as
    missing instead of having an order invented for them. Alias rows are
    excluded because their losses are copies of a canonical run, not
    independent observations. The per-arm centering inside
    `order_drift_report` is what keeps a real arm difference from being
    misread as a time trend.
    """
    observations: list = []
    rows_with_order = 0
    rows_without_order = 0
    for row in summaries or []:
        if not isinstance(row, dict) or row.get('reused_from'):
            continue
        per_seed = row.get('per_seed')
        if not isinstance(per_seed, list):
            continue
        found = False
        for item in per_seed:
            if not isinstance(item, dict) or not _finite_metric(item.get('val_loss')):
                continue
            order = item.get('run_order')
            if not isinstance(order, dict):
                continue
            position = order.get('position')
            if not _finite_metric(position):
                continue
            observations.append({
                'unit': order.get('run_id') or f"{row.get('name')}::seed={item.get('seed')}",
                'arm': str(row.get('name')),
                'position': float(position),
                'value': float(item['val_loss']),
            })
            found = True
        if found:
            rows_with_order += 1
        else:
            rows_without_order += 1
    report = order_drift_report(observations)
    report['schedule_fingerprint'] = (schedule or {}).get('fingerprint')
    report['n_rows_with_order'] = rows_with_order
    report['n_rows_without_order'] = rows_without_order
    if schedule is not None:
        report['schedule_balance'] = schedule_balance_report(
            schedule,
            arm_of=lambda unit: unit.rsplit('::rep=', 1)[0]
            if '::rep=' in unit else unit)
    if report.get('status') == 'tested':
        print(f"[order] within-arm order drift: rho={report['spearman_rho']:+.3f}, "
              f"p={report['p_two_sided']:.3f}, direction={report['direction']} "
              f"({report['n_observations']} run(s) with recorded order)")
        if report['drift_detected']:
            print('[order] WARNING: execution order is confounded with the outcome; '
                  're-run the affected arms in a fresh randomized block before '
                  'interpreting their deltas')
    else:
        print(f"[order] not testable: {report.get('note')} "
              f"({rows_with_order} row(s) with recorded order, "
              f"{rows_without_order} without)")
    return report


def _load_summary(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None

def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

# The controlled recipe is more than (model, lr, steps): changing the
# optimizer type/betas, grad clip, schedule, data generator, outlier strength
# or the held-out generator/batch count also changes the number being
# compared. Every field below is recorded in each row's `controls` block and
# none of it is allowed to vary between rows in one comparison.
_PROTOCOL_KEYS = (
    'model', 'optimizer_name', 'optimizer_betas', 'optimizer_lr',
    'weight_decay', 'grad_clip', 'schedule_type', 'schedule_lr',
    'data_generator', 'batch_size', 'seq_len', 'outlier_id', 'outlier_scale',
    'val_seed_offset', 'val_batches', 'train_steps',
)
_PROTOCOL_MODEL_VOCAB = {'gpt2-124m': 50304, 'gpt2-350m': 50304, 'llama-1b': 32000}

def _protocol_outlier_id(model):
    vocab = _PROTOCOL_MODEL_VOCAB.get(model)
    return None if vocab is None else vocab - 1

def _protocol_payload(args) -> dict:
    """The controlled training recipe every row of this invocation shares."""
    return {
        'model': args.model,
        'optimizer_name': 'AdamW',
        'optimizer_betas': [0.9, 0.95],
        'optimizer_lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': 1.0,
        'schedule_type': 'constant',
        'schedule_lr': args.lr,
        'data_generator': 'markov_batch',
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'outlier_id': _protocol_outlier_id(args.model),
        'outlier_scale': args.outlier_scale,
        'val_seed_offset': 1,
        'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
        'train_steps': args.max_steps,
    }

def _protocol_signature(args) -> tuple:
    payload = _protocol_payload(args)
    return tuple(payload[key] for key in _PROTOCOL_KEYS)

def _row_protocol_values(row: dict, fallback_model=None):
    """Recorded controlled-recipe dict, or None when the row predates controls.

    Missing individual fields stay `None`; they compare unequal to this
    invocation's values, so an incomplete old record can never be treated as
    matching.
    """
    controls = row.get('controls')
    if not isinstance(controls, dict):
        return None
    optimizer = controls.get('optimizer') or {}
    schedule = controls.get('schedule') or {}
    data = controls.get('data') or {}
    return {
        # Do not fill a missing model from the current invocation: an old row
        # that never said which model it trained cannot be proven comparable,
        # and `None` is what makes the signature mismatch/fail closed.
        'model': row.get('model'),
        'optimizer_name': optimizer.get('name'),
        'optimizer_betas': optimizer.get('betas'),
        'optimizer_lr': optimizer.get('lr'),
        'weight_decay': optimizer.get('weight_decay'),
        'grad_clip': optimizer.get('grad_clip'),
        'schedule_type': schedule.get('type'),
        'schedule_lr': schedule.get('lr'),
        'data_generator': data.get('generator'),
        'batch_size': data.get('batch_size'),
        'seq_len': data.get('seq_len'),
        'outlier_id': data.get('outlier_id'),
        'outlier_scale': data.get('outlier_scale'),
        'val_seed_offset': data.get('val_seed_offset'),
        'val_batches': data.get('val_batches'),
        'train_steps': controls.get('train_steps'),
    }

def row_protocol_signature(row: dict, fallback_model=None):
    """Canonical recorded controlled-recipe tuple, or None when unknown."""
    values = _row_protocol_values(row, fallback_model)
    if values is None:
        return None
    return tuple(values[key] for key in _PROTOCOL_KEYS)

def partition_rows_by_protocol(rows: list, args) -> tuple[list, list]:
    """(rows recorded under this invocation's recipe, rows recorded under another).

    `all_summaries.json` deliberately accumulates rows across invocations, but
    a row trained with a different model, optimizer, schedule, data generator
    or validation protocol is not the same experiment: comparing it would
    credit the module with a change that came from the recipe. Mismatched rows
    stay in the file and in the report, but they are kept out of the
    statistical comparisons. Missing/partial controls count as unknown, never
    as matching.
    """
    expected_values = _protocol_payload(args)
    expected = _protocol_signature(args)
    keep, stale = [], []
    for row in rows:
        values = _row_protocol_values(row, args.model)
        signature = None if values is None else tuple(values[key] for key in _PROTOCOL_KEYS)
        if signature == expected:
            keep.append(row)
        else:
            mismatched = (list(_PROTOCOL_KEYS) if values is None else
                          [key for key in _PROTOCOL_KEYS
                           if values[key] != expected_values[key]])
            stale.append({
                'name': row.get('name'),
                'reason': ('no recorded controls' if values is None else
                           'trained under a different protocol than this invocation'),
                'mismatched_fields': mismatched,
                'recorded_protocol': values if values is not None else {},
                'expected_protocol': expected_values,
            })
    return keep, stale

def cached_seed_matches(summary: dict, name: str, cfg_kwargs: dict, args, seed: int) -> bool:
    """Whether a cached summary is the same seed of the same row under the same recipe.

    The path does not always encode the seed (a single-seed run writes its
    summary directly under the row directory), so the recorded seed is
    compared too. The held-out validation batch count is part of the recipe:
    a cached row evaluated on a different number of batches is not the same
    selection metric, so it is re-trained (a cached summary without the
    recorded count counts as unknown, never as matching).
    """
    if not (isinstance(summary, dict)
            and summary.get('name') == name
            and summary.get('model') == args.model
            and summary.get('cfg') == cfg_kwargs
            and row_protocol_signature(summary, args.model) == _protocol_signature(args)
            and summary.get('n_val_batches') == max(1, int(getattr(args, 'val_batches', 20)))):
        return False
    # The body of the cached summary must describe exactly the requested run,
    # not merely carry a matching top-level `seed`. A one-seed aggregate whose
    # per-seed record names a different seed, or a multi-seed summary with no
    # per-seed records, is a different cache entry.
    if not _is_seed_id(seed):
        return False
    per_seed = summary.get('per_seed')
    if per_seed is not None:
        if not isinstance(per_seed, list) or len(per_seed) != 1:
            return False
        record = per_seed[0]
        if not isinstance(record, dict) or not _is_seed_id(record.get('seed')):
            return False
        if int(record['seed']) != int(seed):
            return False
    seeds = summary.get('seeds')
    if seeds is not None:
        if not isinstance(seeds, list) or len(seeds) != 1:
            return False
        if not _is_seed_id(seeds[0]) or int(seeds[0]) != int(seed):
            return False
    recorded = summary.get('seed')
    if recorded is None:
        recorded = seeds[0] if isinstance(seeds, list) and seeds else None
    # A malformed cached seed is unknown, not another spelling of `seed`:
    # `int(True)`/`int(42.9)` used to reuse a run that was never seeded 1/42.
    if not _is_seed_id(recorded) or not _is_seed_id(seed):
        return False
    return int(recorded) == int(seed)

def merge_summaries(previous: list, fresh: list) -> list:
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in list(previous) + list(fresh):
        if not isinstance(row, dict) or 'name' not in row:
            continue
        name = row['name']
        if isinstance(name, bool) or not isinstance(name, (str, int)):
            continue
        if name not in merged:
            order.append(name)
        merged[name] = row
    return [merged[name] for name in order if name in merged]

