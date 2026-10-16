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
from adasquash.model import build_gpt_350m, build_llama_tiny_1b
from adasquash.repro import collect_run_metadata, git_revision, save_run_manifest, set_all_seeds
from scripts.train_gpt2 import apply_config_to_parser, build_optimizer, load_yaml_config
ABLATIONS = {'dyt_no_perchannel': dict(norm_name='dyt'), 'dytc_perchannel': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytcd_full': dict(norm_name='dytcd'), 'dytd_k0_no_depth_schedule': dict(norm_name='dytd', k_init=0.0, learnable_k=False), 'dytd_full': dict(norm_name='dytd'), 'dytcd_no_ls': dict(norm_name='dytcd', use_layerscale=False), 'dytcd_ls1e5': dict(norm_name='dytcd', layerscale_init=1e-05), 'dytcd_ls1e3': dict(norm_name='dytcd', layerscale_init=0.001), 'dytc_p1_softsign': dict(norm_name='dytc', p_init=1.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p2_softsign': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p4_softsign': dict(norm_name='dytc', p_init=4.0, learnable_p=False, alpha_jitter=0.0), 'dytc_plearnable': dict(norm_name='dytc', p_init=2.0, learnable_p=True, alpha_jitter=0.0), 'dytc_hardtanh_eq': dict(norm_name='dytc', p_init=64.0, learnable_p=False, alpha_jitter=0.0), 'dyt_alpha_0.3': dict(norm_name='dyt', alpha_init=0.3), 'dyt_alpha_0.5': dict(norm_name='dyt', alpha_init=0.5), 'dyt_alpha_1.0': dict(norm_name='dyt', alpha_init=1.0)}
_MODEL_BUILDERS = {'gpt2-124m': build_gpt_124m, 'gpt2-350m': build_gpt_350m, 'llama-1b': build_llama_tiny_1b}

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

def load_ablations_from_config(path: str) -> dict:
    cfg = load_yaml_config(path)
    rows = cfg.get('ablations')
    if not rows:
        return {}
    out = {}
    for row in rows:
        row = dict(row)
        name = row.pop('name')
        if 'norm' in row:
            row['norm_name'] = row.pop('norm')
        if name in out:
            raise ValueError(f"{path}: duplicate ablation name '{name}' - each row must be unique or an intended comparison is silently lost")
        out[name] = row
    return out

def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path, run_metadata: dict | None=None):
    print(f'\n[abl] {name}: {cfg_kwargs}')
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
    val_loss = sum(val_losses) / len(val_losses)
    stats = act_stats.summary()
    summary = {'name': name, 'cfg': cfg_kwargs, 'model': args.model, 'n_params': n_params, 'final_loss': losses[-1], 'mean_last_100_loss': sum(losses[-100:]) / max(1, len(losses[-100:])), 'val_loss': val_loss, 'n_val_batches': n_val_batches, 'selection_metric': 'val_loss', 'n_loss_spikes': loss_spike.n_spikes, 'loss_curve': losses, 'act_stats': stats, 'max_abs_pre_norm': stats.get('__summary__', {}).get('max_abs_pre_norm'), 'mean_max_abs_pre_norm': stats.get('__summary__', {}).get('mean_max_abs'), 'mean_excess_kurtosis_pre_norm': stats.get('__summary__', {}).get('mean_excess_kurtosis_pre_norm'), 'elapsed_s': time.time() - t0}
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

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
    ablations = dict(ABLATIONS)
    if known.config:
        from_yaml = load_ablations_from_config(known.config)
        if from_yaml:
            ablations = from_yaml
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
                summary = dict(summary, name=name, cfg=cfg_kwargs, reused_from=source)
                run_dir.mkdir(parents=True, exist_ok=True)
                with open(run_dir / 'summary.json', 'w') as f:
                    json.dump(summary, f, indent=2)
                print(f'[alias] {name}: identical model config to {source}, reusing its summary (no retrain)')
                all_summaries.append(summary)
                continue
        s = run_one(name, cfg_kwargs, args, run_dir, run_metadata=run_metadata)
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
    manifest = dict(run_metadata)
    manifest['git'] = git_revision()
    manifest['hyperparameters'] = vars(args)
    manifest['rows'] = [row['name'] for row in ordered]
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[done] {len(ordered)} ablations saved to {base}')
if __name__ == '__main__':
    main()
