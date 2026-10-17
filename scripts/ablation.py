from __future__ import annotations
"""Controlled ablation matrix for the DyT++ / AdaSquash proposal.

Comparison protocol (what the runner guarantees, and checks)
------------------------------------------------------------
* **Controlled variables.** Every row runs under one optimizer (AdamW,
  betas 0.9/0.95), one constant learning rate, one step count, one synthetic
  data protocol and one seed set. Rows may only set architectural /
  initialization knobs; a row that tries to set `lr`, `weight_decay`,
  `batch_size`, `max_steps`, `dropout`, ... is rejected by the loader, because
  a gain obtained by switching the optimizer is not a gain of the module.
* **Complete ablation coverage.** `validate_matrix` checks the leave-one-out
  set for the main proposal (`LOO_SPEC`): Full, Full-minus-per-channel,
  Full-minus-depth-decay and Full-minus-LayerScale must all be present and
  differ from Full in exactly the declared knob(s) - e.g. the per-channel
  contrast uses the frozen-delta row, not `dytd_full`, which also switches
  tanh -> softsign. `leave_one_out_comparison` then runs the paired
  Full-vs-Full-minus-X test (Holm-adjusted over mechanisms), so "each axis
  helps" is a tested claim rather than an eyeballed delta.
* **Statistical reliability.** Every row is replicated over >= 3 seeds by
  default and reported as mean +/- std across seeds; the row-vs-baseline
  significance test runs on the paired per-seed means when both rows share
  >= 2 seeds (falling back to the shared held-out batches otherwise).
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash import GPTConfig, GPT, build_gpt_124m, count_parameters, ActivationStats, LossSpikeCounter, markov_batch
from adasquash.diagnostics import triage_nonfinite
from adasquash.metrics import holm_adjust, mean_std_stderr, paired_z_one_sided, welch_z_one_sided
from adasquash.model import build_gpt_350m, build_llama_tiny_1b
from adasquash.repro import collect_run_metadata, git_revision, save_run_manifest, set_all_seeds
from scripts.train_gpt2 import apply_config_to_parser, build_optimizer, load_yaml_config
ABLATIONS = {'ln_baseline': dict(norm_name='ln'), 'rmsnorm_baseline': dict(norm_name='rmsnorm'), 'dyt_no_perchannel': dict(norm_name='dyt'), 'dytc_perchannel': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytcd_full': dict(norm_name='dytcd'), 'dytcd_no_perchannel': dict(norm_name='dytcd', learnable_delta=False), 'dytcd_k0_no_depth_schedule': dict(norm_name='dytcd', k_init=0.0, learnable_k=False), 'dytd_k0_no_depth_schedule': dict(norm_name='dytd', k_init=0.0, learnable_k=False), 'dytd_full': dict(norm_name='dytd'), 'dytcd_no_ls': dict(norm_name='dytcd', use_layerscale=False), 'dytcd_ls1e5': dict(norm_name='dytcd', layerscale_init=1e-05), 'dytcd_ls1e3': dict(norm_name='dytcd', layerscale_init=0.001), 'dytc_p1_softsign': dict(norm_name='dytc', p_init=1.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p2_softsign': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p4_softsign': dict(norm_name='dytc', p_init=4.0, learnable_p=False, alpha_jitter=0.0), 'dytc_plearnable': dict(norm_name='dytc', p_init=2.0, learnable_p=True, alpha_jitter=0.0), 'dytc_hardtanh_eq': dict(norm_name='dytc', p_init=64.0, learnable_p=False, alpha_jitter=0.0), 'dyt_alpha_0.3': dict(norm_name='dyt', alpha_init=0.3), 'dyt_alpha_0.5': dict(norm_name='dyt', alpha_init=0.5), 'dyt_alpha_1.0': dict(norm_name='dyt', alpha_init=1.0)}
# Every row must state *why* it is run (the scientific hypothesis it tests),
# not just what it configures. `load_ablations_from_config` enforces this for
# YAML rows; these are the hypotheses of the built-in fallback rows.
ABLATION_HYPOTHESES = {
    'ln_baseline': 'Simple statistical baseline: LayerNorm on the identical protocol. The DyT family must beat this untuned-but-fairly-run reference before any added machinery counts as progress.',
    'rmsnorm_baseline': 'Simple statistical baseline: RMSNorm, the strongest no-mean normalization used in modern LLMs. A DyT-family gain is only meaningful relative to this reference.',
    'dyt_no_perchannel': 'Original DyT (one scalar alpha, tanh) matches or beats the LayerNorm/RMSNorm references at this scale, reproducing the paper claim under our protocol.',
    'dytc_perchannel': 'Adding the per-channel axis (p fixed at 2, jitter off, so only the axis changes) improves held-out loss over scalar DyT.',
    'dytcd_full': 'Combining the per-channel and depth axes beats both single-axis variants: the two mechanisms interact positively.',
    'dytcd_no_perchannel': 'Leave-one-out for the per-channel axis of the main proposal: with delta frozen at zero (a zero buffer, not a parameter) alpha_c == alpha_l exactly, while the depth schedule, the softsign family and the LayerScale residual stay identical to the full model. Unlike the dytd_full row, this contrast changes no other mechanism, so a loss increase here is attributable to the per-channel deviation alone.',
    'dytcd_k0_no_depth_schedule': 'Leave-one-out for the depth axis of the main proposal: with k pinned to 0 and frozen, DyT-CD keeps its per-channel factor, softsign and LayerScale, so any loss of the full model advantage is attributable to depth decay.',
    'dytd_k0_no_depth_schedule': 'Depth decay is the source of DyT-D gain: with k pinned to 0 and frozen, dytd_full advantage should vanish.',
    'dytd_full': 'Depth-conditioned alpha (+ LayerScale) beats scalar DyT here: shallow strong squashing + deep relaxed response helps this depth.',
    'dytcd_no_ls': 'The LayerScale residual gate contributes to DyT-CD; removing it (use_layerscale off) hurts held-out loss.',
    'dytcd_ls1e5': 'LayerScale init 1e-5 (the DyT-CD default) is a competitive residual-gate scale, not an arbitrary strawman.',
    'dytcd_ls1e3': 'LayerScale init 1e-3 trades warmup stability for stronger early residual growth; the default should not be beaten by it.',
    'dytc_p1_softsign': 'Softsign sharpness matters: p=1 (classic softsign) is worse than p=2 on this outlier-heavy corpus.',
    'dytc_p2_softsign': 'p=2 (smooth softsign) is the best fixed saturation exponent of {1, 2, 4} for outlier control with usable gradients.',
    'dytc_p4_softsign': 'p=4 saturates harder than p=2; the hypothesis is that this hurts gradient flow and loses to p=2.',
    'dytc_plearnable': 'A learnable p adapts the saturation steepness per layer and beats every fixed p in the sweep.',
    'dytc_hardtanh_eq': 'The p->infinity limit (hard tanh) starves gradients outside [-1,1] and loses to finite p.',
    'dyt_alpha_0.3': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_0.5': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_1.0': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
}
_MODEL_BUILDERS = {'gpt2-124m': build_gpt_124m, 'gpt2-350m': build_gpt_350m, 'llama-1b': build_llama_tiny_1b}
_ABLATION_METADATA_KEYS = ('hypothesis',)

# Knobs that the matrix runner holds fixed by construction: every row gets the
# same optimizer (AdamW, fixed betas), learning rate, weight decay, step count,
# batch size, sequence length, data generator and seed set. A row that sets one
# of these is not an ablation of the proposed module - it is a different
# optimization protocol whose gain cannot be attributed to the module - so the
# loader rejects it instead of silently running it.
_FORBIDDEN_ROW_KEYS = frozenset({
    'lr', 'learning_rate', 'weight_decay', 'beta1', 'beta2', 'betas',
    'optimizer', 'scheduler', 'schedule', 'warmup', 'min_lr_ratio',
    'max_steps', 'epochs', 'batch_size', 'grad_accum', 'seq_len',
    'grad_clip', 'dropout', 'bf16', 'seed', 'seeds', 'dataset',
    'dataset_config', 'tokenizer', 'augmentation',
})

# Default number of random seeds for the matrix. One seed cannot separate a
# method effect from run-to-run noise, so the key experiment is replicated
# over >= 3 seeds and reported as mean +/- std across seeds (see
# `aggregate_seed_summaries` and `baseline_comparison`).
DEFAULT_SEEDS = (42, 43, 44)

# Leave-one-out (LOO) coverage of the main proposal DyT-CD. Each entry names
# one claimed mechanism and the matrix row that removes exactly it. The claim
# "mechanism X helps" is supportable only when the Full model and its X-free
# reference ran on the identical protocol; `validate_matrix` refuses to run
# the matrix when a mechanism is never isolated. Each removal must change the
# declared knobs *and nothing else*: the `dytd_full` row also switches
# tanh -> generalized softsign, so it cannot isolate the per-channel axis (it
# stays in the matrix as the activation-family reference); the per-channel
# contrast uses `dytcd_no_perchannel`, whose only effective difference from
# Full is the frozen delta buffer.
LOO_SPEC = {
    'dytcd_full': {
        'per_channel_delta': {'row': 'dytcd_no_perchannel',
                              'diff': ('learnable_delta',)},
        'depth_decay': {'row': 'dytcd_k0_no_depth_schedule',
                        'diff': ('k_init', 'learnable_k')},
        'layerscale_residual': {'row': 'dytcd_no_ls',
                                'diff': ('use_layerscale',)},
    },
}

def _effective_cfg(model_name: str, cfg_kwargs: dict) -> dict:
    """The full GPTConfig a matrix row builds, as a plain dict."""
    with torch.device('meta'):
        cfg = _MODEL_BUILDERS[model_name](**cfg_kwargs).cfg
    return dataclasses.asdict(cfg)

def effective_config_key(model_name: str, cfg_kwargs: dict) -> str:
    """Canonical identity of the model a matrix row trains.

    Rows may spell out builder/GPTConfig defaults explicitly (e.g.
    ``layerscale_init: 1.0e-5`` or ``alpha_init: 0.5``); two rows with equal
    keys build the exact same GPTConfig and, under the same seed, produce
    bit-identical runs. The runner trains the first and reuses its summary
    for the rest.
    """
    return json.dumps(_effective_cfg(model_name, cfg_kwargs), sort_keys=True)

def _check_row_keys(name: str, cfg: dict, source: str) -> None:
    bad = sorted(set(cfg) & _FORBIDDEN_ROW_KEYS)
    if bad:
        raise ValueError(
            f"{source}: ablation row '{name}' sets optimizer/schedule/data "
            f"knob(s) {bad}; those are held fixed for every row so a "
            f"comparison isolates the module under test instead of a "
            f"different training recipe"
        )

def check_leave_one_out(rows: dict, model_name: str, spec: dict | None=None) -> dict:
    """Verify every claimed mechanism is isolated by a Full-minus-X row.

    For each `(full, mechanism) -> {row, diff}` entry the reference row must
    exist and its *effective* model config must differ from the Full config in
    exactly the declared fields - no more (a second change would confound the
    contrast) and no fewer (a no-op row proves nothing). Raises ValueError
    listing every violation so one broken claim cannot hide behind another.
    """
    spec = LOO_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for full_name, removals in spec.items():
        if full_name not in rows:
            problems.append(f"leave-one-out full row '{full_name}' is missing")
            continue
        full_cfg = _effective_cfg(model_name, rows[full_name])
        report[full_name] = {}
        for mechanism, info in removals.items():
            ref_name = info['row']
            declared = set(info['diff'])
            if ref_name not in rows:
                problems.append(
                    f"'{mechanism}' has no Full-minus-X reference: row "
                    f"'{ref_name}' is missing")
                continue
            ref_cfg = _effective_cfg(model_name, rows[ref_name])
            actual = {k for k in full_cfg if full_cfg[k] != ref_cfg[k]}
            actual |= {k for k in ref_cfg if k not in full_cfg}
            if actual != declared:
                problems.append(
                    f"'{mechanism}': {ref_name} vs {full_name} differ in "
                    f"{sorted(actual)}, expected exactly {sorted(declared)}")
                continue
            report[full_name][mechanism] = ref_name
    if problems:
        raise ValueError('leave-one-out coverage is incomplete: ' + '; '.join(problems))
    return report

def validate_matrix(rows: dict, hypotheses: dict, model_name: str) -> dict:
    """Every static check the matrix must pass before any training starts."""
    missing = []
    for name, cfg in rows.items():
        _check_row_keys(name, cfg, 'matrix')
        if not str(hypotheses.get(name) or '').strip():
            missing.append(name)
    if missing:
        raise ValueError(
            f"ablation rows without a stated scientific hypothesis: {sorted(missing)}")
    return {'rows': sorted(rows),
            'leave_one_out': check_leave_one_out(rows, model_name)}

def install_outlier_channel(model, scale: float=25.0) -> int:
    with torch.no_grad():
        oid = model.cfg.vocab_size - 1
        model.tok_emb.weight[oid] *= scale
    return oid

def load_ablations_from_config(path: str, include_metadata: bool = False) -> dict:
    """Load the ablation matrix from YAML.

    Returns `{name: cfg_kwargs}` by default (the shape every existing caller
    expects). With `include_metadata=True` each value is
    `{'cfg': cfg_kwargs, 'hypothesis': str}` and every row is required to
    state a scientific hypothesis - an ablation without a stated hypothesis
    is a hyper-parameter lottery, not an experiment, so it is rejected rather
    than silently run.
    """
    cfg = load_yaml_config(path)
    rows = cfg.get('ablations')
    if not rows:
        return {}
    out = {}
    missing_hypothesis = []
    for row in rows:
        row = dict(row)
        name = row.pop('name')
        metadata = {key: row.pop(key) for key in _ABLATION_METADATA_KEYS if key in row}
        if 'norm' in row:
            row['norm_name'] = row.pop('norm')
        _check_row_keys(name, row, path)
        if name in out:
            raise ValueError(f"{path}: duplicate ablation name '{name}' - each row must be unique or an intended comparison is silently lost")
        if include_metadata:
            hypothesis = str(metadata.get('hypothesis') or '').strip()
            if not hypothesis:
                missing_hypothesis.append(name)
            out[name] = {'cfg': row, 'hypothesis': hypothesis or None}
        else:
            out[name] = row
    if missing_hypothesis:
        raise ValueError(
            f"{path}: every ablation row must declare its scientific hypothesis "
            f"(`hypothesis:`); missing for {sorted(missing_hypothesis)}"
        )
    return out

def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path, run_metadata: dict | None=None, hypothesis: str | None=None, seed: int | None=None):
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
    model = _MODEL_BUILDERS[args.model](**cfg_kwargs)
    model.reset_parameters(seed)
    model.to(device)
    n_params = count_parameters(model)
    opt = build_optimizer(model, args.lr, args.weight_decay, 0.9, 0.95)
    rng = torch.Generator(device=device).manual_seed(seed)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    outlier_id = install_outlier_channel(model)
    losses = []
    stats_windowed = False
    t0 = time.time()
    for step in range(args.max_steps):
        x, y = markov_batch(rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device, outlier_id=outlier_id)
        measure = step % args.stats_every == 0
        if measure:
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        loss = model(x, y, return_logits=False)[1]
        if not math.isfinite(float(loss.detach().item())):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=args.lr, grad_clip=1.0)
            raise SystemExit(
                f"[abl][nan] '{name}' loss is non-finite at step {step} "
                f"(seed={seed}): triage stage='{triage['stage']}' "
                f"({triage['diagnosis']}); fixes: " + ' | '.join(triage['recommendations'])
            )
        loss.backward()
        if measure:
            act_stats.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val = loss.item()
        loss_spike.update(loss_val)
        losses.append(loss_val)
        if step % args.log_every == 0:
            print(f'  [step {step:4d}] loss={loss_val:.4f}', flush=True)
    if not losses:
        raise ValueError(f"ablation '{name}' produced no steps")
    # Held-out validation loss: fresh synthetic batches from a generator the
    # optimizer has never seen (seed + 1), evaluated with no gradients. Model
    # comparisons must use this - the training loss above is fitted data.
    n_val_batches = max(1, getattr(args, 'val_batches', 20))
    val_rng = torch.Generator(device=device).manual_seed(seed + 1)
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(n_val_batches):
            vx, vy = markov_batch(val_rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device, outlier_id=outlier_id)
            val_losses.append(model(vx, vy, return_logits=False)[1].item())
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
                 'val_seed_offset': 1},
        'train_steps': args.max_steps,
        'seed': seed,
    }
    summary = {'name': name, 'hypothesis': hypothesis, 'cfg': cfg_kwargs, 'model': args.model, 'n_params': n_params, 'seed': seed, 'seeds': [seed], 'final_loss': losses[-1], 'mean_last_100_loss': sum(losses[-100:]) / max(1, len(losses[-100:])), 'val_loss': val_loss, 'val_losses': val_losses, 'val_loss_std': val_stats['std'], 'val_loss_stderr': val_stats['stderr'], 'val_loss_std_basis': 'val_batches', 'n_val_batches': n_val_batches, 'selection_metric': 'val_loss', 'n_loss_spikes': loss_spike.n_spikes, 'loss_curve': losses, 'act_stats': stats, 'max_abs_pre_norm': stats.get('__summary__', {}).get('max_abs_pre_norm'), 'mean_max_abs_pre_norm': stats.get('__summary__', {}).get('mean_max_abs'), 'mean_excess_kurtosis_pre_norm': stats.get('__summary__', {}).get('mean_excess_kurtosis_pre_norm'), 'controls': controls, 'elapsed_s': time.time() - t0}
    if run_metadata is None:
        run_metadata = collect_run_metadata(seed=seed)
    summary['run_metadata'] = run_metadata
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [done] final_loss={summary['final_loss']:.4f} val_loss={val_loss:.4f} max_abs_pre_norm={summary['max_abs_pre_norm']}")
    return summary

def aggregate_seed_summaries(name: str, summaries: list, cfg_kwargs: dict | None=None, hypothesis: str | None=None) -> dict:
    """Pool per-seed runs into one Mean +/- Std summary.

    The seed is the unit of replication: two runs with different seeds are
    independent samples of the same protocol, while the held-out batches
    inside one run are not (they all come from one generator). The pooled
    `val_loss` is therefore the mean of the per-seed means, `val_loss_std` is
    the sample std across seeds whenever there is more than one seed (with
    `val_loss_std_basis` recording which variability it describes), and the
    per-batch losses are kept tagged by seed so the comparison can pair rows
    seed by seed.
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
        })
    seed_losses = [p['val_loss'] for p in per_seed if isinstance(p['val_loss'], (int, float))]
    stats = mean_std_stderr(seed_losses) or {'mean': None, 'std': None, 'stderr': None, 'n': 0}
    multi = len(seed_losses) >= 2

    def _mean_of(key):
        vals = [p[key] for p in per_seed if isinstance(p.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    agg = dict(summaries[0]) if summaries else {}
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
        'val_loss': stats['mean'],
        'val_loss_seed_std': stats['std'] if multi else None,
        'val_loss_seed_stderr': stats['stderr'] if multi else None,
        'val_loss_std': stats['std'] if multi else agg.get('val_loss_std'),
        'val_loss_stderr': stats['stderr'] if multi else agg.get('val_loss_stderr'),
        'val_loss_std_basis': 'seeds' if multi else 'val_batches',
        'val_losses': pooled,
        'val_losses_by_seed': by_seed,
        'final_loss': _mean_of('final_loss'),
        'mean_last_100_loss': _mean_of('mean_last_100_loss'),
        'n_loss_spikes': sum(p['n_loss_spikes'] or 0 for p in per_seed
                             if isinstance(p.get('n_loss_spikes'), (int, float))),
        'elapsed_s': sum(p['elapsed_s'] or 0.0 for p in per_seed
                         if isinstance(p.get('elapsed_s'), (int, float))),
    })
    return agg

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

def merge_summaries(previous: list, fresh: list) -> list:
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in list(previous) + list(fresh):
        if not isinstance(row, dict) or 'name' not in row:
            continue
        if row['name'] not in merged:
            order.append(row['name'])
        merged[row['name']] = row
    return [merged[name] for name in order if name in merged]

def _seed_means(row) -> dict:
    """{seed: val_loss} of a multi-seed aggregate; {} for single-seed rows."""
    per = row.get('per_seed')
    out = {}
    if isinstance(per, list):
        for item in per:
            if not isinstance(item, dict) or item.get('seed') is None:
                continue
            if isinstance(item.get('val_loss'), (int, float)):
                out[int(item['seed'])] = float(item['val_loss'])
    return out

def _row_std(row):
    """Across-seed std when replicated, within-run batch std otherwise."""
    if row.get('n_seeds', 1) >= 2:
        return row.get('val_loss_seed_std')
    return row.get('val_loss_std')

def _paired_row_test(row: dict, reference: dict, margin: float=0.0):
    """One-sided paired test of `row` against `reference`, seed-paired first.

    Every comparison in this module must run through the *same* test logic,
    or `baseline_comparison` and `leave_one_out_comparison` could disagree on
    the same pair of rows. Seeds are the unit of replication: when both rows
    share >= 2 seeds the test runs on the paired per-seed means; otherwise it
    falls back to the shared held-out batches of one run (paired when the
    batch counts match, Welch when they do not). Returns
    `(test, basis, common_seeds)`, with `test=None` when no basis exists.
    """
    row_seeds = _seed_means(row)
    ref_seeds = _seed_means(reference)
    common = sorted(set(row_seeds) & set(ref_seeds))
    if len(common) >= 2:
        test = paired_z_one_sided([row_seeds[s] for s in common],
                                  [ref_seeds[s] for s in common], margin=margin)
        return test, 'seed_means', common
    row_batches = row.get('val_losses')
    ref_batches = reference.get('val_losses')
    if (isinstance(row_batches, list) and isinstance(ref_batches, list)
            and len(row_batches) == len(ref_batches) and len(row_batches) >= 2):
        return paired_z_one_sided(row_batches, ref_batches, margin=margin), 'val_batches', common
    if (isinstance(row_batches, list) and isinstance(ref_batches, list)
            and len(row_batches) >= 2 and len(ref_batches) >= 2):
        return welch_z_one_sided(row_batches, ref_batches, margin=margin), 'val_batches_unpaired', common
    return None, None, common

def baseline_comparison(summaries: list, baseline_names, margin: float=0.0, alpha: float=0.05) -> dict | None:
    """Every row's held-out val_loss against the best reference baseline.

    The reference is the *best* (lowest val_loss) row among `baseline_names`,
    so a variant is only credited with a win if it beats the strongest simple
    baseline on the identical protocol. Significance uses the paired per-seed
    means whenever both rows share >= 2 seeds (seeds are the independent
    replications), and falls back to the shared held-out batches of a single
    run (paired z when the batch counts match, otherwise Welch) when they do
    not. `margin` is the smallest loss improvement treated as meaningful.
    Because every row is tested against the *same* reference, the per-row
    p-values are a family of tests and are Holm-Bonferroni adjusted over the
    matrix before anything is called significant: `beats_reference_significantly`
    requires the adjusted p < alpha, and the raw one-sided p stays available as
    `p_one_sided`. Returns None when no baseline row has a usable val_loss.
    """
    baseline_names = set(baseline_names)
    usable = [r for r in summaries
              if isinstance(r, dict) and 'name' in r and isinstance(r.get('val_loss'), (int, float))]
    baselines = [r for r in usable if r['name'] in baseline_names]
    if not baselines:
        return None
    reference = min(baselines, key=lambda r: r['val_loss'])
    ref_seeds = _seed_means(reference)
    out = {
        'reference': reference['name'],
        'reference_val_loss': reference['val_loss'],
        'reference_val_loss_std': _row_std(reference),
        'reference_val_loss_stderr': reference.get('val_loss_seed_stderr', reference.get('val_loss_stderr')),
        'reference_n_seeds': reference.get('n_seeds', 1),
        'reference_seeds': sorted(ref_seeds),
        'baseline_rows': {r['name']: r['val_loss'] for r in baselines},
        'margin': margin,
        'alpha': alpha,
        'protocol': ('seeds are the unit of replication: mean +/- std across >= 3 seeds; '
                     'the paired test runs on per-seed means when both rows share >= 2 seeds, '
                     'otherwise on the shared held-out batches of one run'),
        'rows': [],
    }
    for row in usable:
        if row['name'] in baseline_names:
            continue
        delta = row['val_loss'] - reference['val_loss']
        entry = {
            'name': row['name'],
            'hypothesis': row.get('hypothesis'),
            'val_loss': row['val_loss'],
            'val_loss_std': _row_std(row),
            'n_seeds': row.get('n_seeds', 1),
            'delta_vs_reference': delta,
            'beats_reference': bool(delta < -margin),
            'beats_reference_significantly': False,
            'test': None,
            'test_basis': None,
            'paired_seeds': [],
            'p_one_sided': None,
            'p_one_sided_holm': None,
        }
        test, basis, common = _paired_row_test(row, reference, margin=margin)
        entry['test_basis'] = basis
        entry['paired_seeds'] = common
        if test is not None:
            entry['test'] = 'paired_z_one_sided' if test['paired'] else 'welch_z_one_sided'
            entry['p_one_sided'] = test['p_one_sided']
            entry['z'] = test['z']
            entry['n'] = test['n']
        out['rows'].append(entry)
    # Every row above tests the same reference, so the raw p-values form a
    # family and the per-row alpha must be corrected (Holm-Bonferroni step-down)
    # before a row is reported as a significant win. Rows whose test could not
    # run stay None and are excluded from the family.
    adjusted = holm_adjust([entry['p_one_sided'] for entry in out['rows']])
    for entry, p_holm in zip(out['rows'], adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['beats_reference_significantly'] = bool(
            entry['beats_reference'] and p_holm is not None and p_holm < alpha
        )
    out['multiple_comparison'] = (
        f'Holm-Bonferroni over the {sum(p is not None for p in adjusted)} tested row(s) '
        f'against the same reference; `beats_reference_significantly` uses the adjusted p-value'
    )
    thin = {r['name'] for r in [reference, *baselines] if r.get('n_seeds', 1) < 3}
    out['rows_with_fewer_than_3_seeds'] = sorted(thin | {e['name'] for e in out['rows'] if e['n_seeds'] < 3})
    return out

def print_baseline_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[compare] no baseline row present; comparison skipped (include a baseline row to interpret the matrix)')
        return
    ref_std = comparison.get('reference_val_loss_std')
    ref_std_str = '' if ref_std is None else f" +/- {ref_std:.4f}"
    print(f"[compare] reference baseline: {comparison['reference']} "
          f"val_loss={comparison['reference_val_loss']:.4f}{ref_std_str} "
          f"(seeds={comparison.get('reference_seeds')}; "
          f"all baselines: { {k: round(v, 4) for k, v in comparison['baseline_rows'].items()} })")
    for entry in comparison['rows']:
        p = entry['p_one_sided']
        p_holm = entry.get('p_one_sided_holm')
        p_str = 'n/a' if p is None else f'{p:.3g}'
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        std = entry.get('val_loss_std')
        std_str = '' if std is None else f" +/- {std:.4f}"
        basis = entry.get('test_basis') or '-'
        print(f"[compare] {entry['name']:>32s} val_loss={entry['val_loss']:.4f}{std_str} "
              f"delta={entry['delta_vs_reference']:+.4f} p(one-sided)={p_str} p(holm)={p_holm_str} "
              f"test={entry.get('test') or '-'}@{basis} "
              f"beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[compare] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def leave_one_out_comparison(summaries: list, spec: dict | None=None, margin: float=0.0, alpha: float=0.05) -> dict | None:
    """Full vs each Full-minus-X row: does removing mechanism X hurt?

    `check_leave_one_out` proves the *structure* of the leave-one-out matrix -
    the reference row differs from Full in exactly the declared knobs. This
    function adds the statistical half: for every declared mechanism it runs
    the same one-sided paired test as `baseline_comparison`, of H1 "Full has
    lower held-out val_loss than the X-free row". Because all mechanisms are
    tested against the same Full row, the per-mechanism p-values are
    Holm-Bonferroni adjusted before a mechanism is called significantly
    helpful; a point-estimate win inside the seed noise remains visible as
    `mechanism_helps=True, mechanism_helps_significantly=False`. This is what
    backs the claim that each axis of the main proposal is load-bearing on
    its own, not merely correlated with the others. Returns None when no
    declared contrast has both rows with a usable val_loss.
    """
    spec = LOO_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and isinstance(r.get('val_loss'), (int, float))}
    rows: list = []
    for full_name, removals in spec.items():
        full = usable.get(full_name)
        for mechanism, info in removals.items():
            ref_name = info['row']
            ref = usable.get(ref_name)
            entry = {
                'full': full_name,
                'mechanism': mechanism,
                'reference_row': ref_name,
                'declared_diff': sorted(info['diff']),
                'full_val_loss': None if full is None else full['val_loss'],
                'full_val_loss_std': None if full is None else _row_std(full),
                'reference_val_loss': None if ref is None else ref['val_loss'],
                'reference_val_loss_std': None if ref is None else _row_std(ref),
                'n_seeds_full': None if full is None else full.get('n_seeds', 1),
                'n_seeds_reference': None if ref is None else ref.get('n_seeds', 1),
                'delta_full_minus_reference': None,
                'mechanism_helps': False,
                'mechanism_helps_significantly': False,
                'test': None,
                'test_basis': None,
                'paired_seeds': [],
                'p_one_sided': None,
                'p_one_sided_holm': None,
            }
            if full is None:
                entry['skipped'] = f"full row '{full_name}' has no usable val_loss"
                rows.append(entry)
                continue
            if ref is None:
                entry['skipped'] = f"Full-minus-X row '{ref_name}' has no usable val_loss"
                rows.append(entry)
                continue
            entry['delta_full_minus_reference'] = full['val_loss'] - ref['val_loss']
            entry['mechanism_helps'] = bool(entry['delta_full_minus_reference'] < -margin)
            test, basis, common = _paired_row_test(full, ref, margin=margin)
            entry['test_basis'] = basis
            entry['paired_seeds'] = common
            if test is not None:
                entry['test'] = 'paired_z_one_sided' if test['paired'] else 'welch_z_one_sided'
                entry['p_one_sided'] = test['p_one_sided']
                entry['z'] = test['z']
                entry['n'] = test['n']
            rows.append(entry)
    if not rows:
        return None
    adjusted = holm_adjust([entry['p_one_sided'] for entry in rows])
    for entry, p_holm in zip(rows, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['mechanism_helps_significantly'] = bool(
            entry['mechanism_helps'] and p_holm is not None and p_holm < alpha
        )
    thin = sorted({name for entry in rows for name in (entry['full'], entry['reference_row'])
                   if 1 <= usable.get(name, {}).get('n_seeds', 1) < 3})
    return {
        'protocol': ('Full vs Full-minus-X: one-sided paired test of H1 "Full has lower '
                     'held-out val_loss than the X-free row"; seeds are the unit of '
                     'replication (paired per-seed means when both rows share >= 2 seeds, '
                     'else the shared held-out batches of one run)'),
        'margin': margin,
        'alpha': alpha,
        'multiple_comparison': (f'Holm-Bonferroni over the '
                                f'{sum(e["p_one_sided"] is not None for e in rows)} tested mechanism(s); '
                                f'`mechanism_helps_significantly` uses the adjusted p-value'),
        'rows': rows,
        'rows_with_fewer_than_3_seeds': thin,
    }

def print_leave_one_out_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[loo] no leave-one-out contrast available; nothing to test')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[loo] {entry['mechanism']}: SKIPPED ({entry['skipped']})")
            continue
        full_std = entry.get('full_val_loss_std')
        ref_std = entry.get('reference_val_loss_std')
        full_std_str = '' if full_std is None else f' +/- {full_std:.4f}'
        ref_std_str = '' if ref_std is None else f' +/- {ref_std:.4f}'
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm_str = 'n/a' if entry.get('p_one_sided_holm') is None else f"{entry['p_one_sided_holm']:.3g}"
        print(f"[loo] {entry['mechanism']:>22s}: {entry['full']} "
              f"val_loss={entry['full_val_loss']:.4f}{full_std_str} vs "
              f"{entry['reference_row']} val_loss={entry['reference_val_loss']:.4f}{ref_std_str} "
              f"delta(full-ref)={entry['delta_full_minus_reference']:+.4f} "
              f"p(one-sided)={p_str} p(holm)={p_holm_str} "
              f"test={entry.get('test') or '-'}@{entry.get('test_basis') or '-'} "
              f"removing_hurts={entry['mechanism_helps']} "
              f"significant={entry['mechanism_helps_significantly']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[loo] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
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
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help='one row is trained once per seed and reported as mean +/- std across seeds (>= 3 recommended)')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--stats-every', type=int, default=200)
    p.add_argument('--val-batches', type=int, default=20, help='held-out synthetic batches for the selection metric (val_loss); never used for gradient updates')
    p.add_argument('--baselines', nargs='+', default=['ln_baseline', 'rmsnorm_baseline'], help='rows treated as the simple reference baselines in the baseline-relative comparison')
    p.add_argument('--baseline-margin', type=float, default=0.0, help='minimum val_loss improvement over the best baseline to count as a win')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the one-sided paired z-test vs the best baseline')
    p.add_argument('--sanity-check', action='store_true', help='run the small-batch overfit sanity check for every norm in the selected rows before training them')
    p.add_argument('--check-matrix', action='store_true', help='validate controlled knobs, stated hypotheses and leave-one-out coverage of the matrix, then exit without training')
    p.add_argument('--output-dir', default='runs/ablation')
    p.add_argument('--force', action='store_true')
    apply_config_to_parser(p, known.config)
    args = p.parse_args()
    seeds = [args.seed] if args.seed is not None else list(args.seeds or DEFAULT_SEEDS)
    if len(seeds) < 3:
        print(f'[warn] {len(seeds)} seed(s) selected: with <3 independent runs a difference cannot be separated from run-to-run noise; report error bars and treat significance as provisional')
    try:
        matrix_report = validate_matrix(ablations, hypotheses, args.model)
    except ValueError as exc:
        print(f'[matrix] FAIL: {exc}')
        sys.exit(1)
    print(f"[matrix] ok: {len(matrix_report['rows'])} rows, controlled optimizer/schedule/data, "
          f"every row has a hypothesis, leave-one-out coverage {matrix_report['leave_one_out']}")
    if args.check_matrix:
        print(json.dumps(matrix_report, indent=2))
        return
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_metadata = collect_run_metadata(seed=seeds[0], extra={'seeds': seeds})
    if args.sanity_check:
        from scripts.sanity_overfit import run_preflight
        norms = sorted({ablations[name].get('norm_name', 'dytcd') for name in args.names if name in ablations})
        run_preflight(norms or ['dytcd'], seeds[0], base)
    multi = len(seeds) > 1
    all_summaries = []
    trained: dict[str, str] = {}
    for name in args.names:
        if name not in ablations:
            print(f"[warn] unknown ablation '{name}', skipping")
            continue
        cfg_kwargs = ablations[name]
        row_dir = base / name
        summary_paths = {seed: (row_dir / f'seed{seed}' / 'summary.json' if multi else row_dir / 'summary.json') for seed in seeds}
        legacy = row_dir / 'summary.json'
        if multi and legacy.exists() and (not (row_dir / 'summary.preseed.json').exists()) and (not any(p.exists() for p in summary_paths.values())):
            os.replace(legacy, row_dir / 'summary.preseed.json')
            print(f'[keep] {name}: moved pre-replication single-seed summary to summary.preseed.json')
        config_key = effective_config_key(args.model, cfg_kwargs)
        source = trained.get(config_key)
        if source is not None and (not args.force):
            source_paths = {seed: (base / source / f'seed{seed}' / 'summary.json' if multi else base / source / 'summary.json') for seed in seeds}
            if all(p.exists() for p in source_paths.values()):
                reused = [_load_summary(p) for p in source_paths.values()]
                if all(isinstance(s, dict) for s in reused):
                    reused = [dict(s, name=name, cfg=cfg_kwargs, reused_from=source, hypothesis=hypotheses.get(name)) for s in reused]
                    for seed, summary in zip(seeds, reused):
                        if multi:
                            _write_summary(summary_paths[seed], summary)
                    agg = aggregate_seed_summaries(name, reused, cfg_kwargs, hypotheses.get(name))
                    _write_summary(row_dir / 'summary.json', agg)
                    print(f'[alias] {name}: identical model config to {source}, reusing its summaries (no retrain)')
                    trained[config_key] = name
                    all_summaries.append(agg)
                    continue
        seed_summaries = []
        for seed in seeds:
            path = summary_paths[seed]
            if path.exists() and (not args.force):
                summary = _load_summary(path)
                if isinstance(summary, dict):
                    print(f'[skip] {name} seed={seed} already done, use --force to rerun')
                    seed_summaries.append(summary)
                    continue
            seed_summaries.append(run_one(name, cfg_kwargs, args, path.parent, run_metadata=run_metadata, hypothesis=hypotheses.get(name), seed=seed))
        agg = aggregate_seed_summaries(name, seed_summaries, cfg_kwargs, hypotheses.get(name))
        _write_summary(row_dir / 'summary.json', agg)
        trained[config_key] = name
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
    comparison = baseline_comparison(ordered, args.baselines, margin=args.baseline_margin, alpha=args.alpha)
    if comparison is not None:
        with open(base / 'baseline_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    print_baseline_comparison(comparison)
    # The second half of the ablation contract: each claimed mechanism of the
    # main proposal must be shown to be load-bearing *on its own*, with a
    # paired Full-vs-Full-minus-X test over the same seeds.
    loo_comparison = leave_one_out_comparison(ordered, margin=args.baseline_margin, alpha=args.alpha)
    if loo_comparison is not None:
        with open(base / 'leave_one_out_comparison.json', 'w') as f:
            json.dump(loo_comparison, f, indent=2)
    print_leave_one_out_comparison(loo_comparison)
    manifest = dict(run_metadata)
    manifest['git'] = git_revision()
    manifest['hyperparameters'] = vars(args)
    manifest['seeds'] = seeds
    manifest['matrix_validation'] = matrix_report
    manifest['rows'] = [row['name'] for row in ordered]
    manifest['baseline_comparison'] = comparison
    manifest['leave_one_out_comparison'] = loo_comparison
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[done] {len(ordered)} ablations saved to {base} '
          f'({len(seeds)} seed(s): {seeds}; row val_loss = mean +/- std across seeds)')
if __name__ == '__main__':
    main()
