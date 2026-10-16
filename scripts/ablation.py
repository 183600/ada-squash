from __future__ import annotations
import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash import GPTConfig, GPT, build_gpt_124m, count_parameters, ActivationStats, LossSpikeCounter, markov_batch
from adasquash.metrics import mean_std_stderr, paired_z_one_sided, welch_z_one_sided
from adasquash.model import build_gpt_350m, build_llama_tiny_1b
from adasquash.repro import collect_run_metadata, git_revision, save_run_manifest, set_all_seeds
from scripts.train_gpt2 import apply_config_to_parser, build_optimizer, load_yaml_config
ABLATIONS = {'ln_baseline': dict(norm_name='ln'), 'rmsnorm_baseline': dict(norm_name='rmsnorm'), 'dyt_no_perchannel': dict(norm_name='dyt'), 'dytc_perchannel': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytcd_full': dict(norm_name='dytcd'), 'dytd_k0_no_depth_schedule': dict(norm_name='dytd', k_init=0.0, learnable_k=False), 'dytd_full': dict(norm_name='dytd'), 'dytcd_no_ls': dict(norm_name='dytcd', use_layerscale=False), 'dytcd_ls1e5': dict(norm_name='dytcd', layerscale_init=1e-05), 'dytcd_ls1e3': dict(norm_name='dytcd', layerscale_init=0.001), 'dytc_p1_softsign': dict(norm_name='dytc', p_init=1.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p2_softsign': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p4_softsign': dict(norm_name='dytc', p_init=4.0, learnable_p=False, alpha_jitter=0.0), 'dytc_plearnable': dict(norm_name='dytc', p_init=2.0, learnable_p=True, alpha_jitter=0.0), 'dytc_hardtanh_eq': dict(norm_name='dytc', p_init=64.0, learnable_p=False, alpha_jitter=0.0), 'dyt_alpha_0.3': dict(norm_name='dyt', alpha_init=0.3), 'dyt_alpha_0.5': dict(norm_name='dyt', alpha_init=0.5), 'dyt_alpha_1.0': dict(norm_name='dyt', alpha_init=1.0)}
# Every row must state *why* it is run (the scientific hypothesis it tests),
# not just what it configures. `load_ablations_from_config` enforces this for
# YAML rows; these are the hypotheses of the built-in fallback rows.
ABLATION_HYPOTHESES = {
    'ln_baseline': 'Simple statistical baseline: LayerNorm on the identical protocol. The DyT family must beat this untuned-but-fairly-run reference before any added machinery counts as progress.',
    'rmsnorm_baseline': 'Simple statistical baseline: RMSNorm, the strongest no-mean normalization used in modern LLMs. A DyT-family gain is only meaningful relative to this reference.',
    'dyt_no_perchannel': 'Original DyT (one scalar alpha, tanh) matches or beats the LayerNorm/RMSNorm references at this scale, reproducing the paper claim under our protocol.',
    'dytc_perchannel': 'Adding the per-channel axis (p fixed at 2, jitter off, so only the axis changes) improves held-out loss over scalar DyT.',
    'dytcd_full': 'Combining the per-channel and depth axes beats both single-axis variants: the two mechanisms interact positively.',
    'dytd_k0_no_depth_schedule': 'Depth decay is the source of DyT-D gain: with k pinned to 0 and frozen, dytd_full advantage should vanish.',
    'dytd_full': 'Depth-conditioned alpha (+ LayerScale) beats scalar DyT here: shallow strong squashing + deep relaxed response helps this depth.',
    'dytcd_no_ls': 'The LayerScale residual gate contributes to DyT-CD; removing it (use_layerscale off) costs held-out loss.',
    'dytcd_ls1e5': 'LayerScale init 1e-5 (the DyT-CD default) is a competitive residual-gate scale, not an arbitrary strawman.',
    'dytcd_ls1e3': 'LayerScale init 1e-3 trades warmup stability for faster residual growth; the default should not be beaten by it.',
    'dytc_p1_softsign': 'Softsign sharpness matters: p=1 (classic softsign) is worse than p=2 on this outlier-heavy corpus.',
    'dytc_p2_softsign': 'p=2 (smooth softsign) is the best fixed saturation exponent of {1, 2, 4} for outlier control with usable gradients.',
    'dytc_p4_softsign': 'p=4 saturates harder than p=2; the hypothesis is that this costs gradient flow and loses to p=2.',
    'dytc_plearnable': 'A learnable p adapts the saturation steepness per layer and beats every fixed p in the sweep.',
    'dytc_hardtanh_eq': 'The p->infinity limit (hard tanh) starves gradients outside [-1,1] and loses to finite p.',
    'dyt_alpha_0.3': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_0.5': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_1.0': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
}
_MODEL_BUILDERS = {'gpt2-124m': build_gpt_124m, 'gpt2-350m': build_gpt_350m, 'llama-1b': build_llama_tiny_1b}
_ABLATION_METADATA_KEYS = ('hypothesis',)

def effective_config_key(model_name: str, cfg_kwargs: dict) -> str:
    """Canonical identity of the model a matrix row trains.

    Rows may spell out builder/GPTConfig defaults explicitly (e.g.
    ``layerscale_init: 1.0e-5`` or ``alpha_init: 0.5``); two rows with equal
    keys build the exact same GPTConfig and, under the same seed, produce
    bit-identical runs. The runner trains the first and reuses its summary
    for the rest.
    """
    with torch.device('meta'):
        cfg = _MODEL_BUILDERS[model_name](**cfg_kwargs).cfg
    return json.dumps(dataclasses.asdict(cfg), sort_keys=True)

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

def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path, run_metadata: dict | None=None, hypothesis: str | None=None):
    print(f'\n[abl] {name}: {cfg_kwargs}')
    print(f'  [hypothesis] {hypothesis or "(none stated - this row cannot be interpreted as a test)"}')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(args.seed)
    model = _MODEL_BUILDERS[args.model](**cfg_kwargs)
    model.reset_parameters(args.seed)
    model.to(device)
    n_params = count_parameters(model)
    opt = build_optimizer(model, args.lr, args.weight_decay, 0.9, 0.95)
    rng = torch.Generator(device=device).manual_seed(args.seed)
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
    val_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
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
    # fixed `args.seed + 1` generator) and report significance instead of a
    # bare point estimate.
    val_stats = mean_std_stderr(val_losses)
    val_loss = val_stats['mean']
    stats = act_stats.summary()
    summary = {'name': name, 'hypothesis': hypothesis, 'cfg': cfg_kwargs, 'model': args.model, 'n_params': n_params, 'final_loss': losses[-1], 'mean_last_100_loss': sum(losses[-100:]) / max(1, len(losses[-100:])), 'val_loss': val_loss, 'val_losses': val_losses, 'val_loss_std': val_stats['std'], 'val_loss_stderr': val_stats['stderr'], 'n_val_batches': n_val_batches, 'selection_metric': 'val_loss', 'n_loss_spikes': loss_spike.n_spikes, 'loss_curve': losses, 'act_stats': stats, 'max_abs_pre_norm': stats.get('__summary__', {}).get('max_abs_pre_norm'), 'mean_max_abs_pre_norm': stats.get('__summary__', {}).get('mean_max_abs'), 'mean_excess_kurtosis_pre_norm': stats.get('__summary__', {}).get('mean_excess_kurtosis_pre_norm'), 'elapsed_s': time.time() - t0}
    if run_metadata is None:
        run_metadata = collect_run_metadata(seed=args.seed)
    summary['run_metadata'] = run_metadata
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [done] final_loss={summary['final_loss']:.4f} val_loss={val_loss:.4f} max_abs_pre_norm={summary['max_abs_pre_norm']}")
    return summary

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

def baseline_comparison(summaries: list, baseline_names, margin: float=0.0, alpha: float=0.05) -> dict | None:
    """Every row's held-out val_loss against the best reference baseline.

    The reference is the *best* (lowest val_loss) row among `baseline_names`,
    so a variant is only credited with a win if it beats the strongest simple
    baseline on the identical protocol. Because every row evaluates the same
    held-out batches (the runner seeds the validation generator with
    `args.seed + 1`), per-batch losses are paired and the one-sided paired
    z-test is used; the unpaired Welch test is the fallback when the batch
    counts differ. `margin` is the smallest loss improvement treated as
    meaningful; `beats_reference_significantly` additionally requires
    p < alpha. Returns None when no baseline row has a usable val_loss.
    """
    baseline_names = set(baseline_names)
    usable = [r for r in summaries
              if isinstance(r, dict) and 'name' in r and isinstance(r.get('val_loss'), (int, float))]
    baselines = [r for r in usable if r['name'] in baseline_names]
    if not baselines:
        return None
    reference = min(baselines, key=lambda r: r['val_loss'])
    out = {
        'reference': reference['name'],
        'reference_val_loss': reference['val_loss'],
        'reference_val_loss_stderr': reference.get('val_loss_stderr'),
        'baseline_rows': {r['name']: r['val_loss'] for r in baselines},
        'margin': margin,
        'alpha': alpha,
        'rows': [],
    }
    ref_batches = reference.get('val_losses')
    for row in usable:
        if row['name'] in baseline_names:
            continue
        delta = row['val_loss'] - reference['val_loss']
        entry = {
            'name': row['name'],
            'hypothesis': row.get('hypothesis'),
            'val_loss': row['val_loss'],
            'delta_vs_reference': delta,
            'beats_reference': bool(delta < -margin),
            'beats_reference_significantly': False,
            'test': None,
            'p_one_sided': None,
        }
        row_batches = row.get('val_losses')
        test = None
        if (isinstance(row_batches, list) and isinstance(ref_batches, list)
                and len(row_batches) == len(ref_batches) and len(row_batches) >= 2):
            test = paired_z_one_sided(row_batches, ref_batches, margin=margin)
        elif (isinstance(row_batches, list) and isinstance(ref_batches, list)
              and len(row_batches) >= 2 and len(ref_batches) >= 2):
            test = welch_z_one_sided(row_batches, ref_batches, margin=margin)
        if test is not None:
            entry['test'] = 'paired_z_one_sided' if test['paired'] else 'welch_z_one_sided'
            entry['p_one_sided'] = test['p_one_sided']
            entry['z'] = test['z']
            entry['n'] = test['n']
            entry['beats_reference_significantly'] = bool(
                entry['beats_reference'] and test['p_one_sided'] is not None
                and test['p_one_sided'] < alpha
            )
        out['rows'].append(entry)
    return out

def print_baseline_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[compare] no baseline row present; comparison skipped (include a baseline row to interpret the matrix)')
        return
    print(f"[compare] reference baseline: {comparison['reference']} val_loss={comparison['reference_val_loss']:.4f} "
          f"(all baselines: { {k: round(v, 4) for k, v in comparison['baseline_rows'].items()} })")
    for entry in comparison['rows']:
        p = entry['p_one_sided']
        p_str = 'n/a' if p is None else f'{p:.3g}'
        print(f"[compare] {entry['name']:>32s} delta={entry['delta_vs_reference']:+.4f} "
              f"p(one-sided)={p_str} beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}")

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
    ablations = dict(ABLATIONS)
    hypotheses = dict(ABLATION_HYPOTHESES)
    if known.config:
        from_yaml = load_ablations_from_config(known.config, include_metadata=True)
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
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--stats-every', type=int, default=200)
    p.add_argument('--val-batches', type=int, default=20, help='held-out synthetic batches for the selection metric (val_loss); never used for gradient updates')
    p.add_argument('--baselines', nargs='+', default=['ln_baseline', 'rmsnorm_baseline'], help='rows treated as the simple reference baselines in the baseline-relative comparison')
    p.add_argument('--baseline-margin', type=float, default=0.0, help='minimum val_loss improvement over the best baseline to count as a win')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the one-sided paired z-test vs the best baseline')
    p.add_argument('--sanity-check', action='store_true', help='run the small-batch overfit sanity check for every norm in the selected rows before training them')
    p.add_argument('--output-dir', default='runs/ablation')
    p.add_argument('--force', action='store_true')
    apply_config_to_parser(p, known.config)
    args = p.parse_args()
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_metadata = collect_run_metadata(seed=args.seed)
    if args.sanity_check:
        from scripts.sanity_overfit import run_preflight
        norms = sorted({ablations[name].get('norm_name', 'dytcd') for name in args.names if name in ablations})
        run_preflight(norms or ['dytcd'], args.seed, base)
    all_summaries = []
    trained: dict[str, str] = {}
    for name in args.names:
        if name not in ablations:
            print(f"[warn] unknown ablation '{name}', skipping")
            continue
        cfg_kwargs = ablations[name]
        run_dir = base / name
        config_key = effective_config_key(args.model, cfg_kwargs)
        if (run_dir / 'summary.json').exists() and (not args.force):
            print(f'[skip] {name} already done, use --force to rerun')
            trained.setdefault(config_key, name)
            try:
                summary = json.loads((run_dir / 'summary.json').read_text())
            except (OSError, json.JSONDecodeError):
                summary = None
            if isinstance(summary, dict):
                all_summaries.append(summary)
            continue
        source = trained.get(config_key)
        if source is not None and (base / source / 'summary.json').exists():
            try:
                summary = json.loads((base / source / 'summary.json').read_text())
            except (OSError, json.JSONDecodeError):
                summary = None
            if isinstance(summary, dict):
                summary = dict(summary, name=name, cfg=cfg_kwargs, reused_from=source, hypothesis=hypotheses.get(name))
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(run_dir / 'summary.json', 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f'[alias] {name}: identical model config to {source}, reusing its summary (no retrain)')
                all_summaries.append(summary)
                continue
        s = run_one(name, cfg_kwargs, args, run_dir, run_metadata=run_metadata, hypothesis=hypotheses.get(name))
        trained[config_key] = name
        all_summaries.append(s)
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
    manifest = dict(run_metadata)
    manifest['git'] = git_revision()
    manifest['hyperparameters'] = vars(args)
    manifest['rows'] = [row['name'] for row in ordered]
    manifest['baseline_comparison'] = comparison
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[done] {len(ordered)} ablations saved to {base}')
if __name__ == '__main__':
    main()
