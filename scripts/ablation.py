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
from scripts.train_gpt2 import apply_config_to_parser, build_optimizer, load_yaml_config
# Single-variable contrasts (see configs/ablation-matrix.yaml for the derivation
# of every adjacent pair). The runner reads the YAML when --config is given; this
# dict is the no-arg fallback and mirrors it.
ABLATIONS = {'ln': dict(norm_name='ln', use_layerscale=False), 'ln_ls': dict(norm_name='ln'), 'rmsnorm': dict(norm_name='rmsnorm', use_layerscale=False), 'rmsnorm_ls': dict(norm_name='rmsnorm'), 'dyt': dict(norm_name='dyt', alpha_init=0.5, use_layerscale=False), 'dyts': dict(norm_name='dyts', p_init=2.0, learnable_p=False, alpha_init=0.5, use_layerscale=False), 'dytp': dict(norm_name='dytp', alpha_init=0.5, alpha_jitter=0.0, use_layerscale=False), 'dytc': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_init=0.5, alpha_jitter=0.0, use_layerscale=False), 'dyt_ls': dict(norm_name='dyt', alpha_init=0.5, use_layerscale=True, layerscale_init=1e-05), 'dytd': dict(norm_name='dytd', alpha_0=0.9, k_init=1.0, k_min=0.0, use_layerscale=True, layerscale_init=1e-05), 'dytd_k0_no_depth_schedule': dict(norm_name='dytd', alpha_0=0.9, k_init=0.0, learnable_k=False, k_min=0.0, use_layerscale=True, layerscale_init=1e-05), 'dytdi': dict(norm_name='dytdi', alpha_0=0.9, k_init=1.0, use_layerscale=True, layerscale_init=1e-05), 'dytd_shared': dict(norm_name='dytd', alpha_0=0.9, k_init=1.0, k_min=0.0, shared_depth_schedule=True, use_layerscale=True, layerscale_init=1e-05), 'dytc_ls': dict(norm_name='dytc', alpha_init=0.5, alpha_jitter=0.0, p_init=2.0, learnable_p=True, use_layerscale=True, layerscale_init=1e-05), 'dytcdi': dict(norm_name='dytcdi', alpha_0=0.9, k_init=1.0, p_init=2.0, learnable_p=True, alpha_jitter=0.0, use_layerscale=True, layerscale_init=1e-05), 'dytcd': dict(norm_name='dytcd', alpha_0=0.9, k_init=1.0, k_min=0.0, p_init=2.0, learnable_p=True, alpha_jitter=0.0, use_layerscale=True, layerscale_init=1e-05), 'dytcd_no_ls': dict(norm_name='dytcd', alpha_0=0.9, k_init=1.0, k_min=0.0, p_init=2.0, learnable_p=True, alpha_jitter=0.0, use_layerscale=False), 'dytcd_ls1e3': dict(norm_name='dytcd', alpha_0=0.9, k_init=1.0, k_min=0.0, p_init=2.0, learnable_p=True, alpha_jitter=0.0, use_layerscale=True, layerscale_init=0.001), 'dytc_p1_softsign': dict(norm_name='dytc', p_init=1.0, learnable_p=False, alpha_init=0.5, alpha_jitter=0.0, use_layerscale=False), 'dytc_p4_softsign': dict(norm_name='dytc', p_init=4.0, learnable_p=False, alpha_init=0.5, alpha_jitter=0.0, use_layerscale=False), 'dytc_plearnable': dict(norm_name='dytc', p_init=2.0, learnable_p=True, alpha_init=0.5, alpha_jitter=0.0, use_layerscale=False)}
_MODEL_BUILDERS = {'gpt2-124m': build_gpt_124m, 'gpt2-350m': build_gpt_350m, 'llama-1b': build_llama_tiny_1b}

def effective_config_key(model_name: str, cfg_kwargs: dict, seed: int | None=None) -> str:
    """Canonical identity of the model a matrix row trains.

    Rows may spell out builder/GPTConfig defaults explicitly (e.g.
    ``layerscale_init: 1.0e-5`` or ``alpha_init: 0.5``); two rows with equal
    keys build the exact same GPTConfig and, under the same seed, produce
    bit-identical runs. The runner trains the first and reuses its summary
    for the rest. The seed is part of the key: identical configs under
    different seeds are *different* runs and must both be trained.
    """
    with torch.device('meta'):
        cfg = _MODEL_BUILDERS[model_name](**cfg_kwargs).cfg
    key = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    if seed is not None:
        key = f'{key}#seed={seed}'
    return key

def install_residual_outlier(model, layer_idx: int | None=None, channel: int | None=None, scale: float=25.0):
    """Amplify one residual-stream channel after a mid layer, for the whole run.

    This replaces the old probe that inflated a token-embedding row: with tied
    embeddings that row IS a row of the output classifier, so the "outlier
    channel" claim was confounded with "the model must relearn to never
    predict that token" - an output-side pathology no input-side norm has to
    handle. A persistent forward hook that scales channel ``channel`` of the
    residual stream after block ``layer_idx`` injects the LLaMA-style
    persistent-channel outlier without touching a single weight: whatever the
    block wrote into that channel is amplified for every position and every
    downstream layer, and only the normalization layers' element-wise squash
    can bound it. Defaults are deterministic: middle block, middle channel.
    """
    n_layers = int(getattr(model.cfg, 'n_layers', 12))
    dim = int(getattr(model.cfg, 'dim', 768))
    layer_idx = n_layers // 2 if layer_idx is None else int(layer_idx)
    channel = dim // 2 if channel is None else int(channel)
    block = model.blocks[layer_idx]

    def hook(module, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        y[..., channel].mul_(scale)
        return out
    block.register_forward_hook(hook)
    return {'kind': 'residual_channel', 'layer_idx': layer_idx, 'channel': channel, 'scale': scale}

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

def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path, seed: int | None=None):
    seed = args.seed if seed is None else seed
    print(f'\n[abl] {name} (seed {seed}): {cfg_kwargs}')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    model = _MODEL_BUILDERS[args.model](**cfg_kwargs)
    model.reset_parameters(seed)
    model.to(device)
    n_params = count_parameters(model)
    opt = build_optimizer(model, args.lr, args.weight_decay, 0.9, 0.95)
    rng = torch.Generator(device=device).manual_seed(seed)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    outlier = install_residual_outlier(model, layer_idx=args.outlier_layer, channel=args.outlier_channel, scale=args.outlier_scale)
    losses = []
    stats_windowed = False
    t0 = time.time()
    for step in range(args.max_steps):
        x, y = markov_batch(rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device)
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
    stats = act_stats.summary()
    summary = {'name': name, 'seed': seed, 'cfg': cfg_kwargs, 'model': args.model, 'n_params': n_params, 'outlier': outlier, 'final_loss': losses[-1], 'mean_last_100_loss': sum(losses[-100:]) / max(1, len(losses[-100:])), 'n_loss_spikes': loss_spike.n_spikes, 'loss_curve': losses, 'act_stats': stats, 'max_abs_pre_norm': stats.get('__summary__', {}).get('max_abs_pre_norm'), 'mean_max_abs_pre_norm': stats.get('__summary__', {}).get('mean_max_abs'), 'mean_excess_kurtosis_pre_norm': stats.get('__summary__', {}).get('mean_excess_kurtosis_pre_norm'), 'elapsed_s': time.time() - t0}
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [done] {summary['final_loss']:.4f} max_abs_pre_norm={summary['max_abs_pre_norm']}")
    return summary

def aggregate_seeds(summaries: list) -> dict:
    """Mean / std / min / max of every numeric headline metric across seeds."""
    keys = ['final_loss', 'mean_last_100_loss', 'n_loss_spikes', 'max_abs_pre_norm', 'mean_max_abs_pre_norm', 'mean_excess_kurtosis_pre_norm']
    agg = {'name': summaries[0]['name'], 'cfg': summaries[0].get('cfg'), 'model': summaries[0].get('model'), 'outlier': summaries[0].get('outlier'), 'seeds': [s.get('seed') for s in summaries], 'n_seeds': len(summaries)}
    for k in keys:
        vals = [s[k] for s in summaries if isinstance(s.get(k), (int, float))]
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum(((v - mean) ** 2 for v in vals)) / max(1, len(vals) - 1) if len(vals) > 1 else 0.0
        agg[f'{k}_mean'] = mean
        agg[f'{k}_std'] = var ** 0.5
        agg[f'{k}_min'] = min(vals)
        agg[f'{k}_max'] = max(vals)
    return agg

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
    p.add_argument('--seeds', nargs='+', type=int, default=None, help='run every ablation under each of these seeds and report mean/std (single-seed runs live in <out>/<name>/seed<N>/, the aggregate in <out>/<name>/summary.json). Default: just --seed.')
    p.add_argument('--outlier-layer', type=int, default=None, help='block after which the residual-channel outlier is injected (default: the middle block)')
    p.add_argument('--outlier-channel', type=int, default=None, help='residual-stream channel to amplify (default: the middle channel)')
    p.add_argument('--outlier-scale', type=float, default=25.0, help='amplification applied to the outlier channel (0 disables the outlier)')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--stats-every', type=int, default=200)
    p.add_argument('--output-dir', default='runs/ablation')
    p.add_argument('--force', action='store_true')
    apply_config_to_parser(p, known.config)
    args = p.parse_args()
    seeds = list(args.seeds) if args.seeds else [args.seed]
    multi_seed = len(seeds) > 1
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    trained: dict[str, str] = {}
    for name in args.names:
        if name not in ablations:
            print(f"[warn] unknown ablation '{name}', skipping")
            continue
        cfg_kwargs = ablations[name]
        run_dir = base / name
        seed_summaries = []
        for seed in seeds:
            seed_dir = run_dir / f'seed{seed}' if multi_seed else run_dir
            config_key = effective_config_key(args.model, cfg_kwargs, seed)
            if (seed_dir / 'summary.json').exists() and (not args.force):
                print(f'[skip] {name} (seed {seed}) already done, use --force to rerun')
                trained.setdefault(config_key, str(seed_dir.relative_to(base)))
                try:
                    summary = json.loads((seed_dir / 'summary.json').read_text())
                except (OSError, json.JSONDecodeError):
                    summary = None
                if isinstance(summary, dict):
                    seed_summaries.append(summary)
                continue
            source = trained.get(config_key)
            if source is not None and (base / source / 'summary.json').exists():
                try:
                    summary = json.loads((base / source / 'summary.json').read_text())
                except (OSError, json.JSONDecodeError):
                    summary = None
                if isinstance(summary, dict):
                    summary = dict(summary, name=name, cfg=cfg_kwargs, reused_from=source)
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    with open(seed_dir / 'summary.json', 'w') as f:
                        json.dump(summary, f, indent=2)
                    print(f'[alias] {name} (seed {seed}): identical model config to {source}, reusing its summary (no retrain)')
                    seed_summaries.append(summary)
                    continue
            s = run_one(name, cfg_kwargs, args, seed_dir, seed=seed)
            trained[config_key] = str(seed_dir.relative_to(base))
            seed_summaries.append(s)
        if not seed_summaries:
            continue
        if multi_seed:
            agg = aggregate_seeds(seed_summaries)
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / 'summary.json', 'w') as f:
                json.dump(agg, f, indent=2)
            print(f'[agg] {name}: final_loss {agg.get("final_loss_mean"):.4f} +/- {agg.get("final_loss_std"):.4f} over {agg["n_seeds"]} seeds')
            all_summaries.append(agg)
        else:
            all_summaries.extend(seed_summaries)
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
    print(f'\n[done] {len(ordered)} ablations saved to {base}')
if __name__ == '__main__':
    main()
