from __future__ import annotations
"""Compute-matched depth sweep with per-cell seed replication.

Every (depth, norm) cell runs the *identical* recipe (same data slices, same
optimizer/schedule, same token budget); depth-only comparisons must not mix in
different training recipes. Each cell is replicated over `--seeds` (>= 3
recommended) and the launcher aggregates `results.best_val_loss` into
`Mean +/- Std` across those independent runs, because a single run cannot
separate a depth/norm effect from run-to-run noise.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adasquash.metrics import mean_std_stderr
from adasquash.repro import collect_run_metadata, save_run_manifest

def _cosine_lr(step, max_steps, warmup, min_lr_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
_BASE_DEPTH = 24
_BASE_DIM = 2048
_BASE_HEADS = 16
_BASE_SHAPE = (_BASE_DIM, _BASE_HEADS)
_BASE_HEAD_DIM = _BASE_DIM // _BASE_HEADS
DEPTH_SHAPE = {12: (2944, 23), 24: (2048, 16), 48: (1408, 11), 96: (1024, 8)}
_BASE_SHAPES = {'llama-1b': (2048, 16), 'gpt2-124m': (768, 12), 'gpt2-350m': (1024, 16)}

def base_shape_for_model(model: str) -> tuple:
    return _BASE_SHAPES.get(model, _BASE_SHAPE)
_SWEEP_CONFIG_ALIASES = {'use_layerscale': ('no_layerscale', True), 'learnable_p': ('no_learnable_p', True), 'learnable_k': ('no_learnable_k', True), 'bf16': ('no_bf16', True), 'tie_word_embeddings': ('tie_embeddings', False), 'beta': ('beta', False), 'alpha_jitter': ('alpha_jitter', False)}

def shape_for_depth(depth: int, base_dim: int, base_heads: int):
    if (base_dim, base_heads) == _BASE_SHAPE and depth in DEPTH_SHAPE:
        return DEPTH_SHAPE[depth]
    base_head_dim = max(1, base_dim // base_heads)
    dim = int(round(base_dim * math.sqrt(_BASE_DEPTH / max(1, depth))))
    heads = max(1, int(round(dim / base_head_dim)))
    dim = heads * base_head_dim
    return (dim, heads)

def seed_dir(cell_dir: Path, seed: int, multi: bool) -> Path:
    """Per-seed run directory: `seed{N}/` once replicated, the cell dir itself
    for the legacy single-seed layout."""
    return cell_dir / f'seed{seed}' if multi else cell_dir

def aggregate_sweep_cell(cell_dir: Path, seeds, multi: bool) -> dict:
    """Pool each seed's `results.best_val_loss` into a Mean +/- Std record."""
    per_seed = []
    for seed in seeds:
        path = seed_dir(cell_dir, seed, multi) / 'run_manifest.json'
        best = None
        try:
            manifest = json.loads(path.read_text())
            if isinstance(manifest, dict):
                best = (manifest.get('results') or {}).get('best_val_loss')
        except (OSError, json.JSONDecodeError):
            pass
        per_seed.append({'seed': int(seed), 'best_val_loss': best})
    vals = [p['best_val_loss'] for p in per_seed if isinstance(p['best_val_loss'], (int, float))]
    stats = mean_std_stderr(vals)
    return {
        'cell': cell_dir.name,
        'seeds': [int(s) for s in seeds],
        'per_seed': per_seed,
        'n_seeds': len(vals),
        'best_val_loss': None if stats is None else stats['mean'],
        'best_val_loss_std': None if stats is None or len(vals) < 2 else stats['std'],
        'best_val_loss_std_basis': 'seeds' if len(vals) >= 2 else 'single_run',
        'aggregation': 'Mean +/- Std across seeds; each seed is an independent run of the identical recipe',
    }

def launch_one(args, depth: int, norm: str, output_dir: Path, seed: int):
    cmd = [sys.executable, str(Path(__file__).parent / 'train_gpt2.py'), '--model', args.model, '--norm', norm, '--dataset', args.dataset, '--dataset-config', args.dataset_config, '--tokenizer', args.tokenizer, '--seq-len', str(args.seq_len), '--batch-size', str(args.batch_size), '--grad-accum', str(args.grad_accum), '--max-steps', str(args.max_steps), '--lr', str(args.lr), '--min-lr-ratio', str(args.min_lr_ratio), '--warmup', str(args.warmup), '--weight-decay', str(args.weight_decay), '--beta1', str(args.beta1), '--beta2', str(args.beta2), '--grad-clip', str(args.grad_clip), '--seed', str(seed), '--alpha-0', str(args.alpha_0), '--alpha-init', str(args.alpha_init), '--k-init', str(args.k_init), '--p-init', str(args.p_init), '--qk-alpha-init', str(args.qk_alpha_init), '--output-dir', str(output_dir), '--log-every', str(args.log_every), '--stats-every', str(args.stats_every), '--save-every', str(args.save_every)]
    if args.qk_norm:
        cmd += ['--qk-norm', args.qk_norm]
    if args.layerscale_init is not None:
        cmd += ['--layerscale-init', str(args.layerscale_init)]
    if args.no_bf16:
        cmd.append('--no-bf16')
    if args.tie_embeddings is not None:
        cmd.append('--tie-embeddings' if args.tie_embeddings else '--no-tie-embeddings')
    if args.no_learnable_p:
        cmd.append('--no-learnable-p')
    if args.no_learnable_k:
        cmd.append('--no-learnable-k')
    if args.no_layerscale:
        cmd.append('--no-layerscale')
    if args.beta is False:
        cmd.append('--no-beta')
    if args.alpha_jitter != 0.05:
        cmd += ['--alpha-jitter', str(args.alpha_jitter)]
    # The split / evaluation protocol is part of the controlled variables:
    # every depth-norm row must see the same train slice and the same held-out
    # validation and (sealed) test slices.
    cmd += ['--val-frac', str(args.val_frac), '--test-frac', str(args.test_frac),
            '--val-every', str(args.val_every), '--val-batches', str(args.val_batches),
            '--min-doc-chars', str(args.min_doc_chars), '--max-doc-chars', str(args.max_doc_chars),
            '--train-shuffle-buffer', str(args.train_shuffle_buffer)]
    if args.num_documents is not None:
        cmd += ['--num-documents', str(args.num_documents)]
    if not args.dedup:
        cmd.append('--no-dedup')
    if args.exclude_hash_file:
        cmd += ['--exclude-hash-file', str(args.exclude_hash_file)]
    if args.gradient_checkpointing:
        cmd.append('--gradient-checkpointing')
    if args.sanity_check:
        cmd.append('--sanity-check')
    if args.compile:
        cmd.append('--compile')
        cmd += ['--compile-mode', args.compile_mode]
    env = dict(os.environ)
    env['ADASQUASH_DEPTH_OVERRIDE'] = str(depth)
    shape = ''
    if not args.no_dim_scaling:
        base_dim, base_heads = base_shape_for_model(args.model)
        dim, heads = shape_for_depth(depth, base_dim, base_heads)
        env['ADASQUASH_DIM_OVERRIDE'] = str(dim)
        env['ADASQUASH_HEADS_OVERRIDE'] = str(heads)
        shape = f' (dim={dim}, heads={heads})'
    print(f'[sweep] launching depth={depth} norm={norm} -> {output_dir}{shape}')
    log = open(output_dir / 'stdout.log', 'w')
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    except Exception:
        log.close()
        raise
    log.close()
    return proc

def main():
    from scripts.train_gpt2 import apply_config_to_parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='YAML run config; CLI flags override it')
    p.add_argument('--model', default='llama-1b')
    p.add_argument('--depths', nargs='+', type=int, default=[12, 24, 48, 96])
    p.add_argument('--norms', nargs='+', default=['rmsnorm', 'dyt', 'dytcd'])
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--val-frac', type=float, default=0.005, help='chronological held-out validation fraction (identical for every row)')
    p.add_argument('--test-frac', type=float, default=0.005, help='sealed test fraction (identical for every row)')
    p.add_argument('--num-documents', type=int, default=None, help='total documents; required when the dataset reports no size')
    p.add_argument('--val-every', type=int, default=500, help='held-out validation period in optimizer steps')
    p.add_argument('--val-batches', type=int, default=10, help='batches per validation pass')
    p.add_argument('--min-doc-chars', type=int, default=0)
    p.add_argument('--max-doc-chars', type=int, default=0)
    p.add_argument('--no-dedup', dest='dedup', action='store_false', help='disable exact-duplicate document removal')
    p.add_argument('--exclude-hash-file', default=None, help='training-document hash ledger shared by every depth/norm row (build it once with scripts/audit_data.py --write-train-hash-file)')
    p.add_argument('--train-shuffle-buffer', type=int, default=0, help='packed training-batch shuffle window (0 = off)')
    p.add_argument('--tokenizer', default='HuggingFaceFW/fineweb-edu/llama-tokenizer')
    p.add_argument('--seq-len', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=5000)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--min-lr-ratio', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', nargs='+', type=int, default=[42], help='each (depth, norm) cell is replicated over these seeds and reported as Mean +/- Std (>= 3 recommended)')
    p.add_argument('--qk-norm', default=None, choices=[None, 'qknorm', 'dytqk'])
    p.add_argument('--layerscale-init', type=float, default=None)
    p.add_argument('--no-bf16', action='store_true')
    p.add_argument('--tie-embeddings', dest='tie_embeddings', action='store_true', default=None)
    p.add_argument('--no-tie-embeddings', dest='tie_embeddings', action='store_false')
    p.add_argument('--alpha-0', type=float, default=0.9)
    p.add_argument('--alpha-init', type=float, default=0.5)
    p.add_argument('--k-init', type=float, default=1.0)
    p.add_argument('--p-init', type=float, default=2.0)
    p.add_argument('--qk-alpha-init', type=float, default=1.0, help='initial alpha_q / alpha_k for --qk-norm dytqk')
    p.add_argument('--no-learnable-p', action='store_true')
    p.add_argument('--no-learnable-k', action='store_true')
    p.add_argument('--no-layerscale', action='store_true')
    p.add_argument('--no-beta', dest='beta', action='store_false')
    p.add_argument('--alpha-jitter', type=float, default=0.05, help='per-channel alpha init jitter for DyTC (0 for deterministic)')
    p.add_argument('--gradient-checkpointing', action='store_true', help='enable gradient checkpointing')
    p.add_argument('--sanity-check', action='store_true', help='run the small-batch overfit sanity check in every row before training it')
    p.add_argument('--compile', action='store_true', help='enable torch.compile')
    p.add_argument('--compile-mode', default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--no-dim-scaling', action='store_true', help='keep dim/n_heads fixed across depths')
    p.add_argument('--output-dir', default='runs/depth-sweep')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stats-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--sequential', action='store_true', help='run configs sequentially')
    apply_config_to_parser(p, known.config, _SWEEP_CONFIG_ALIASES)
    args = p.parse_args()
    seeds = [args.seed] if args.seed is not None else list(args.seeds or [42])
    if len(seeds) < 3:
        print(f'[sweep][warn] {len(seeds)} seed(s): a single depth/norm cell cannot be '
              f'separated from run-to-run noise; use >= 3 seeds before interpreting deltas')
    multi = len(seeds) > 1
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    sweep_manifest = collect_run_metadata(seed=seeds[0])
    sweep_manifest['hyperparameters'] = vars(args)
    sweep_manifest['seeds'] = seeds
    sweep_manifest['grid'] = {
        'depths': list(args.depths),
        'norms': list(args.norms),
        'seeds': seeds,
        'runs': [f'd{depth}-{norm}-s{seed}'
                 for depth in args.depths for norm in args.norms for seed in seeds],
    }
    save_run_manifest(base / 'run_manifest.json', sweep_manifest)
    procs = []
    cells = []
    for depth in args.depths:
        for norm in args.norms:
            cell_dir = base / f'd{depth}-{norm}'
            cell_dir.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                out_dir = seed_dir(cell_dir, seed, multi)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / 'depth.txt').write_text(str(depth))
                proc = launch_one(args, depth, norm, out_dir, seed)
                if args.sequential:
                    rc = proc.wait()
                    print(f'[sweep] finished d{depth}-{norm} seed={seed} rc={rc}')
                else:
                    procs.append(proc)
            cells.append((depth, norm, cell_dir))
    if not args.sequential:
        for proc in procs:
            proc.wait()
    for depth, norm, cell_dir in cells:
        agg = aggregate_sweep_cell(cell_dir, seeds, multi)
        with open(cell_dir / 'aggregate.json', 'w') as f:
            json.dump(agg, f, indent=2)
        mean = agg['best_val_loss']
        std = agg.get('best_val_loss_std')
        if mean is not None:
            std_str = '' if std is None else f' +/- {std:.4f}'
            print(f"[sweep] d{depth}-{norm}: best_val_loss={mean:.4f}{std_str} "
                  f"over {agg['n_seeds']} seed(s)")
        else:
            print(f"[sweep] d{depth}-{norm}: no usable best_val_loss in the run manifests")
    print('[sweep] done')
if __name__ == '__main__':
    main()
