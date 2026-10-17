from __future__ import annotations
"""Compute-matched depth sweep with per-cell seed replication.

Every (depth, norm) cell runs the *identical* recipe (same data slices, same
optimizer/schedule, same token count); depth-only comparisons must not mix in
different training recipes. Each cell is replicated over `--seeds` (>= 3
recommended) and the launcher aggregates the runs into `Mean +/- Std` across
those independent runs, because a single run cannot separate a depth/norm
effect from run-to-run noise.

Two numbers travel side by side, and the sweep reports both rather than
picking whichever looks better:

* `best_val_loss` (min over the validation passes) is the checkpoint-selection
  statistic; reporting it as the cell's loss is optimistically biased;
* `final_val_loss` (last validation pass at the fixed step budget) is the
  comparable fixed-step estimate used for depth/norm comparisons.

Because every cell runs the same seeds, two cells at a depth are *paired*
samples. The launcher therefore also tests each norm against `--reference-norm`
at each depth on the per-seed `final_val_loss` differences (paired one-sided
test, Holm-Bonferroni over the sweep's comparison family) and reports the
per-seed sign counts next to the p-value, since with the default 3 seeds the
normal approximation alone is thin evidence.
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
from adasquash.metrics import holm_adjust, mean_std_stderr, paired_z_one_sided
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
_SWEEP_CONFIG_ALIASES = {'use_layerscale': ('no_layerscale', True), 'learnable_p': ('no_learnable_p', True), 'learnable_k': ('no_learnable_k', True), 'learnable_delta': ('no_learnable_delta', True), 'bf16': ('no_bf16', True), 'tie_word_embeddings': ('tie_embeddings', False), 'beta': ('beta', False), 'alpha_jitter': ('alpha_jitter', False)}

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
    """Pool each seed's reported losses into Mean +/- Std records.

    The two validation statistics a run reports are not the same number:

    * `best_val_loss` is the minimum over the validation passes - the
      checkpoint-selection statistic. As a *reported* estimate it is
      optimistically biased (the minimum of noisy draws), so it must not be
      quoted as the cell's generalization loss;
    * `final_val_loss` is the last held-out pass of the run. With early
      stopping off (the sweep default) every cell validates at the same
      training-step budget, so it is the fixed-step estimate that makes
      depth/norm cells comparable.

    Both are pooled over the per-seed independent runs of the identical
    recipe. Legacy manifests without `final_val_loss` leave the final-pass
    fields at `None` with `n_seeds_final = 0` instead of silently mixing the
    two statistics.
    """
    per_seed = []
    for seed in seeds:
        path = seed_dir(cell_dir, seed, multi) / 'run_manifest.json'
        best = None
        final = None
        try:
            manifest = json.loads(path.read_text())
            if isinstance(manifest, dict):
                results = manifest.get('results') or {}
                best = results.get('best_val_loss')
                final = results.get('final_val_loss')
        except (OSError, json.JSONDecodeError):
            pass
        per_seed.append({'seed': int(seed), 'best_val_loss': best, 'final_val_loss': final})

    def _pool(key: str):
        vals = [p[key] for p in per_seed if isinstance(p[key], (int, float))]
        return vals, mean_std_stderr(vals)

    best_vals, best_stats = _pool('best_val_loss')
    final_vals, final_stats = _pool('final_val_loss')
    return {
        'cell': cell_dir.name,
        'seeds': [int(s) for s in seeds],
        'per_seed': per_seed,
        'n_seeds': len(best_vals),
        'best_val_loss': None if best_stats is None else best_stats['mean'],
        'best_val_loss_std': None if best_stats is None or len(best_vals) < 2 else best_stats['std'],
        'best_val_loss_std_basis': 'seeds' if len(best_vals) >= 2 else 'single_run',
        'n_seeds_final': len(final_vals),
        'final_val_loss': None if final_stats is None else final_stats['mean'],
        'final_val_loss_std': None if final_stats is None or len(final_vals) < 2 else final_stats['std'],
        'final_val_loss_std_basis': 'seeds' if len(final_vals) >= 2 else 'single_run',
        'aggregation': ('Mean +/- Std across seeds (each seed is an independent run of the identical recipe). '
                        'best_val_loss = min over validation passes (checkpoint-selection statistic, optimistically biased); '
                        'final_val_loss = last validation pass (fixed-step estimate, comparable across cells when early stopping is off).'),
    }

def _cell_seed_metric(cell: dict, metric: str) -> dict:
    """{seed: metric value} of an aggregate cell; seeds without the metric dropped."""
    out = {}
    for item in cell.get('per_seed') or []:
        if not isinstance(item, dict) or item.get('seed') is None:
            continue
        value = item.get(metric)
        if isinstance(value, (int, float)):
            out[int(item['seed'])] = float(value)
    return out

def compare_sweep_cells(cells_by_depth: dict, reference_norm: str, metric: str='final_val_loss', margin: float=0.0, alpha: float=0.05) -> dict:
    """One-sided paired comparison of every norm against `reference_norm` per depth.

    Seeds are the unit of replication and every cell of the sweep shares the
    same seed set, so candidates and reference are *paired*: the test runs on
    the per-seed `metric` differences (default `final_val_loss`, the fixed-step
    estimate - the selection-biased `best_val_loss` is deliberately not the
    basis). Because the normal approximation on 3 seeds is thin evidence, the
    per-seed sign counts (`n_seeds_favoring`, `n_seeds_tied`) are reported next
    to the p-value as the assumption-free reading. Holm-Bonferroni adjusts the
    whole family of sweep comparisons before any row is called significant.
    """
    rows = []
    for depth in sorted(cells_by_depth):
        by_norm = cells_by_depth[depth]
        ref = by_norm.get(reference_norm)
        if ref is None:
            continue
        ref_seeds = _cell_seed_metric(ref, metric)
        for norm in sorted(by_norm):
            if norm == reference_norm:
                continue
            cand_seeds = _cell_seed_metric(by_norm[norm], metric)
            common = sorted(set(ref_seeds) & set(cand_seeds))
            entry = {
                'depth': depth,
                'norm': norm,
                'reference_norm': reference_norm,
                'metric': metric,
                'n_paired_seeds': len(common),
                'paired_seeds': common,
                'mean_delta': None,
                'n_seeds_favoring': None,
                'n_seeds_tied': None,
                'p_one_sided': None,
                'p_one_sided_holm': None,
                'beats_reference_significantly': False,
            }
            if len(common) >= 2:
                a = [cand_seeds[s] for s in common]
                b = [ref_seeds[s] for s in common]
                test = paired_z_one_sided(a, b, margin=margin)
                entry['mean_delta'] = test['mean_delta']
                entry['z'] = test['z']
                entry['p_one_sided'] = test['p_one_sided']
                entry['n_seeds_favoring'] = sum(1 for x, y in zip(a, b) if x < y)
                entry['n_seeds_tied'] = sum(1 for x, y in zip(a, b) if x == y)
            elif common:
                entry['mean_delta'] = cand_seeds[common[0]] - ref_seeds[common[0]]
            rows.append(entry)
    adjusted = holm_adjust([r['p_one_sided'] for r in rows])
    for entry, p_holm in zip(rows, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['beats_reference_significantly'] = bool(
            entry['mean_delta'] is not None and entry['mean_delta'] < -margin
            and p_holm is not None and p_holm < alpha)
    return {
        'reference_norm': reference_norm,
        'metric': metric,
        'margin': margin,
        'alpha': alpha,
        'rows': rows,
        'multiple_comparison': (f'Holm-Bonferroni over the {sum(p is not None for p in adjusted)} '
                                f'tested sweep comparison(s) against the same per-depth reference'),
        'reading_guide': ('paired on the shared seeds; `mean_delta` < 0 favors the candidate. '
                          'With 3 seeds read `n_seeds_favoring`/3 first: the z p-value is a normal '
                          'approximation and is not trustworthy on its own at this replication count.'),
    }

def print_sweep_comparisons(comparison: dict) -> None:
    if not comparison['rows']:
        print(f"[compare] no (depth, candidate) cell for reference '{comparison['reference_norm']}'; comparison skipped")
        return
    print(f"[compare] reference norm '{comparison['reference_norm']}' on {comparison['metric']} (paired per seed):")
    for r in comparison['rows']:
        if r['mean_delta'] is None:
            print(f"[compare] d{r['depth']}-{r['norm']}: no paired seeds - not testable")
            continue
        sign = ''
        if r['n_seeds_favoring'] is not None:
            sign = f" seeds_favoring={r['n_seeds_favoring']}/{r['n_paired_seeds']}"
            if r['n_seeds_tied']:
                sign += f" tied={r['n_seeds_tied']}"
        p_holm = r['p_one_sided_holm']
        p_str = 'n/a' if p_holm is None else f'{p_holm:.4g}'
        print(f"[compare] d{r['depth']}-{r['norm']}: delta={r['mean_delta']:+.4f}{sign} "
              f"p_holm={p_str} significant={r['beats_reference_significantly']}")

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
    if args.no_learnable_delta:
        cmd.append('--no-learnable-delta')
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
    p.add_argument('--reference-norm', default='rmsnorm', help='baseline norm each other norm is compared against at every depth (paired per seed, Holm-adjusted)')
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
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44], help='each (depth, norm) cell is replicated over these seeds and reported as Mean +/- Std across runs; the default is 3 independent seeds because fewer cannot separate a depth/norm effect from run-to-run noise')
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
    p.add_argument('--no-learnable-delta', action='store_true', help='freeze the DyT-CD per-channel deviation at zero (controlled per-channel off)')
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
    aggs_by_depth: dict = {}
    for depth, norm, cell_dir in cells:
        agg = aggregate_sweep_cell(cell_dir, seeds, multi)
        with open(cell_dir / 'aggregate.json', 'w') as f:
            json.dump(agg, f, indent=2)
        aggs_by_depth.setdefault(depth, {})[norm] = agg
        mean = agg['best_val_loss']
        std = agg.get('best_val_loss_std')
        final_mean = agg.get('final_val_loss')
        final_std = agg.get('final_val_loss_std')
        if mean is not None:
            std_str = '' if std is None else f' +/- {std:.4f}'
            line = (f"[sweep] d{depth}-{norm}: best_val_loss={mean:.4f}{std_str} "
                    f"over {agg['n_seeds']} seed(s) [selection statistic]")
            if final_mean is not None:
                final_std_str = '' if final_std is None else f' +/- {final_std:.4f}'
                line += (f"; final_val_loss={final_mean:.4f}{final_std_str} "
                         f"over {agg['n_seeds_final']} seed(s) [fixed-step estimate]")
            else:
                line += '; final_val_loss missing (legacy manifest) - cannot compare cells on the fixed-step estimate'
            print(line)
        else:
            print(f"[sweep] d{depth}-{norm}: no usable best_val_loss in the run manifests")
    comparison = compare_sweep_cells(aggs_by_depth, args.reference_norm)
    print_sweep_comparisons(comparison)
    comp_path = base / 'sweep_comparisons.json'
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'[compare] comparisons -> {comp_path}')
    print('[sweep] done')
if __name__ == '__main__':
    main()
