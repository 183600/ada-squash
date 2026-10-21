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
* `final_val_loss` (last validation pass at the fixed step count) is the
  comparable fixed-step estimate used for depth/norm comparisons.

Because every cell runs the same seeds, two cells at a depth are *paired*
samples. The launcher therefore also tests each norm against `--reference-norm`
at each depth on the per-seed `final_val_loss` differences (one-sided paired
Student-t test, Holm-Bonferroni over the sweep's comparison family) and reports
the per-seed sign counts plus a 95% confidence interval for the delta next to
the p-value. The t reference distribution is what makes the default 3 seeds
readable: at df = 2 the one-sided 5% critical value is 2.92, where the normal
approximation would use 1.64 and turn noise into "wins".
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust, mean_std_stderr,
                               paired_t_one_sided, student_t_ppf)
from adasquash.repro import artifact_record, collect_run_metadata, finish_metadata, save_run_manifest

def _cosine_lr(step, max_steps, warmup, min_lr_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

def _finite(value) -> bool:
    """A usable numeric statistic: not None/NaN/Inf (bool is not a number here).

    A diverged run can report `Inf`/`NaN` from a readable manifest; pooling it
    would poison the cell mean while looking like a complete seed.
    """
    return isinstance(value, (int, float)) and (not isinstance(value, bool)) and math.isfinite(value)

# Both ratios are reported next to every comparison row; a cell pair outside
# this band is not compute-matched, so its loss delta is not a depth/norm-only
# effect and must be read against the compute difference.
_COMPUTE_MATCH_TOL = 0.25
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
    for a single-seed layout."""
    return cell_dir / f'seed{seed}' if multi else cell_dir

# Hyperparameters that must be identical in every child run of the sweep. The
# launcher passes them explicitly; comparing the recorded manifest against
# this list is what prevents an old run (or a rerun with a different recipe)
# from being aggregated as if it belonged to the current grid.
_CHILD_PROTOCOL_KEYS = (
    'model', 'dataset', 'dataset_config', 'dataset_revision',
    'tokenizer', 'tokenizer_revision', 'seq_len', 'batch_size', 'grad_accum',
    'max_steps', 'lr', 'min_lr_ratio', 'warmup', 'weight_decay', 'beta1',
    'beta2', 'grad_clip', 'qk_norm', 'layerscale_init', 'alpha_0',
    'alpha_init', 'k_init', 'p_init', 'qk_alpha_init', 'no_learnable_p',
    'no_learnable_k', 'no_learnable_delta', 'no_layerscale', 'beta',
    'alpha_jitter', 'gradient_checkpointing', 'tie_embeddings', 'no_bf16',
    'compile', 'compile_mode', 'val_frac', 'test_frac', 'num_documents',
    'val_every', 'val_batches', 'min_doc_chars', 'max_doc_chars', 'dedup',
    'train_shuffle_buffer', 'exclude_hash_file', 'exclude_test_hash_file',
    'group_key', 'exclude_group_file', 'exclude_test_group_file',
)

def expected_child_protocol(args, depth: int, norm: str, seed: int) -> dict:
    """The controlled recipe and shape a child manifest must record."""
    hyperparameters = {key: getattr(args, key) for key in _CHILD_PROTOCOL_KEYS}
    hyperparameters.update({'norm': norm, 'seed': int(seed)})
    base_dim, base_heads = base_shape_for_model(args.model)
    if args.no_dim_scaling:
        dim, heads = base_dim, base_heads
    else:
        dim, heads = shape_for_depth(depth, base_dim, base_heads)
    return {
        'hyperparameters': hyperparameters,
        'shape': {'n_layers': int(depth), 'dim': int(dim), 'n_heads': int(heads)},
    }

# Revisions are recorded in the child manifest as the concrete commit the
# loader resolved, not necessarily the string the sweep requested (`None` means
# "latest on the default branch"). The launch spec already pins the request;
# the manifest check must accept the resolved hash instead of rejecting every
# unpinned sweep.
_REVISION_PROTOCOL_KEYS = ('dataset_revision', 'tokenizer_revision')


def _is_commit_revision(value) -> bool:
    """Whether `value` looks like a full Hub commit hash (not a branch/tag)."""
    if not isinstance(value, str) or len(value) != 40:
        return False
    return all(c in '0123456789abcdefABCDEF' for c in value)


def manifest_protocol_mismatches(manifest: dict, expected: dict) -> dict:
    """Fields where a child manifest disagrees with this sweep invocation."""
    recorded = manifest.get('hyperparameters') or {}
    results = manifest.get('results') or {}
    mismatches = {}
    for key, wanted in expected['hyperparameters'].items():
        actual = recorded.get(key)
        if key in _REVISION_PROTOCOL_KEYS and actual is not None and actual != wanted:
            # `train_gpt2` replaces the requested revision (a branch, tag or
            # "latest") with the concrete commit it resolved before writing
            # the manifest; that is the stronger pin, not a recipe mismatch.
            # Only a *recorded* full commit may stand in for a non-commit
            # request: a different branch/tag means the child did not train on
            # the requested snapshot, and a requested full commit must be the
            # commit recorded. Either is a real protocol mismatch.
            if _is_commit_revision(actual) and not _is_commit_revision(wanted):
                continue
        if actual != wanted:
            mismatches[key] = [wanted, actual]
    for key, wanted in expected['shape'].items():
        if results.get(key) != wanted:
            mismatches[f'results.{key}'] = [wanted, results.get(key)]
    return mismatches


def _manifest_data_version(manifest: dict) -> dict | None:
    """Concrete data version a child actually trained on, when recorded."""
    data = manifest.get('data') if isinstance(manifest, dict) else None
    if not isinstance(data, dict):
        return None
    reports = data.get('split_reports') or {}
    train = reports.get('train') if isinstance(reports, dict) else None
    train = train if isinstance(train, dict) else {}
    # Only the explicit `*_resolved` fields prove what the loader used. The
    # manifest's `dataset_revision`/`tokenizer_revision` may still be the
    # requested branch or tag when the Hub was unreachable; treating that as
    # the resolved version would silently certify a mutable snapshot as pinned.
    dataset_resolved = (train.get('dataset_revision_resolved')
                        or data.get('dataset_revision_resolved'))
    tokenizer_resolved = (train.get('tokenizer_revision_resolved')
                          or data.get('tokenizer_revision_resolved'))
    if dataset_resolved is None and tokenizer_resolved is None:
        return None
    return {
        'dataset_revision_resolved': dataset_resolved,
        'tokenizer_revision_resolved': tokenizer_resolved,
    }

def _load_launch_spec(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None

def aggregate_sweep_cell(cell_dir: Path, seeds, multi: bool, protocol_for_seed=None, launch_id=None) -> dict:
    """Pool each seed's reported losses into Mean +/- Std records.

    The two validation statistics a run reports are not the same number:

    * `best_val_loss` is the minimum over the validation passes - the
      checkpoint-selection statistic. As a *reported* estimate it is
      optimistically biased (the minimum of noisy draws), so it must not be
      quoted as the cell's generalization loss;
    * `final_val_loss` is the last held-out pass of the run. With early
      stopping off (the sweep default) every cell validates at the same
      training-step count, so it is the fixed-step estimate that makes
      depth/norm cells comparable.

    Both are pooled over the per-seed independent runs of the identical
    recipe. Legacy manifests without `final_val_loss` leave the final-pass
    fields at `None` with `n_seeds_final = 0` instead of silently mixing the
    two statistics.
    """
    seeds = [int(s) for s in seeds]
    per_seed = []
    rejected = []
    for seed in seeds:
        path = seed_dir(cell_dir, seed, multi) / 'run_manifest.json'
        best = None
        final = None
        curve = None
        n_params = None
        breakdown = None
        flops_fwd = None
        flops_train = None
        spikes = None
        nonfinite = None
        nonfinite_grad = None
        nonfinite_val = None
        tokens_per_second = None
        peak_device_memory_mb = None
        data_version = None
        n_layers = None
        dim = None
        n_heads = None
        expected = protocol_for_seed(seed) if protocol_for_seed is not None else None
        reject_reason = None
        if launch_id is not None:
            spec = _load_launch_spec(path.with_name('launch_spec.json'))
            if spec is None:
                reject_reason = ('no launch spec from this sweep invocation: the manifest '
                                 'may predate this run or the cell was never launched')
            elif spec.get('launch_id') != launch_id:
                reject_reason = ('launched by a different sweep invocation; its recipe may '
                                 'differ from the current grid')
            elif expected is not None and spec.get('protocol') != expected:
                reject_reason = 'launch spec does not match the current sweep recipe'
        manifest = None
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        if reject_reason is None and launch_id is not None:
            if not isinstance(manifest, dict):
                reject_reason = 'run_manifest.json is missing or unreadable'
            elif manifest.get('status') != 'finished':
                reject_reason = (f"run status is {manifest.get('status')!r}, not 'finished'; "
                                 'the run did not complete')
            elif expected is not None:
                mismatches = manifest_protocol_mismatches(manifest, expected)
                if mismatches:
                    reject_reason = ('recorded recipe/shape differs from this sweep: '
                                     + json.dumps(mismatches, sort_keys=True))
        if reject_reason is not None:
            rejected.append({'seed': int(seed), 'path': str(path),
                             'reason': reject_reason})
        elif isinstance(manifest, dict):
            data_version = _manifest_data_version(manifest)
            results = manifest.get('results') or {}
            best = results.get('best_val_loss')
            final = results.get('final_val_loss')
            # Curve, parameter and FLOPs accounting already travel in
            # every run manifest; the sweep must pool them instead of
            # dropping them, otherwise a depth cell is read from one final
            # number and a wider/deeper cell's shape accounting is
            # invisible.
            curve = results.get('curve_diagnosis')
            n_params = results.get('n_params')
            n_layers = results.get('n_layers')
            dim = results.get('dim')
            n_heads = results.get('n_heads')
            breakdown = results.get('parameter_breakdown')
            flops_fwd = results.get('flops_per_token_forward')
            flops_train = results.get('flops_per_token_training_est')
            spikes = results.get('n_loss_spikes')
            nonfinite = results.get('n_nonfinite_losses')
            nonfinite_grad = results.get('n_nonfinite_gradients')
            nonfinite_val = results.get('n_nonfinite_val_losses')
            tokens_per_second = results.get('tokens_per_second')
            peak_device_memory_mb = results.get('peak_device_memory_mb')
        per_seed.append({
            'seed': int(seed), 'best_val_loss': best, 'final_val_loss': final,
            'curve_classification': (curve or {}).get('classification'),
            'curve_generalization_gap': (curve or {}).get('generalization_gap'),
            'n_params': n_params,
            'n_layers': n_layers,
            'dim': dim,
            'n_heads': n_heads,
            'norm_share_of_params': (breakdown or {}).get('norm_share_of_total'),
            'flops_per_token_forward': flops_fwd,
            'flops_per_token_training_est': flops_train,
            'tokens_per_second': tokens_per_second,
            'peak_device_memory_mb': peak_device_memory_mb,
            'n_loss_spikes': spikes,
            'n_nonfinite_losses': nonfinite,
            'n_nonfinite_gradients': nonfinite_grad,
            'n_nonfinite_val_losses': nonfinite_val,
            'data_version': data_version,
        })

    versions_by_seed = {
        p['seed']: p['data_version'] for p in per_seed}
    known_versions = {
        json.dumps(v, sort_keys=True) for v in versions_by_seed.values()
        if v is not None}
    unknown_version_seeds = [p['seed'] for p in per_seed
                             if p.get('data_version') is None]
    data_version_mismatch = len(known_versions) > 1
    data_versions_complete = bool(versions_by_seed) and not unknown_version_seeds

    def _pool(key: str):
        vals = [p[key] for p in per_seed if _finite(p[key])]
        return vals, mean_std_stderr(vals)

    best_vals, best_stats = _pool('best_val_loss')
    final_vals, final_stats = _pool('final_val_loss')
    seed_list = seeds
    curve_counts: dict = {}
    for p in per_seed:
        if p.get('curve_classification'):
            curve_counts[p['curve_classification']] = curve_counts.get(p['curve_classification'], 0) + 1
    gaps = [p['curve_generalization_gap'] for p in per_seed
            if _finite(p.get('curve_generalization_gap'))]
    # Model accounting pooled next to the losses: the shared recipe
    # (same dim/heads per depth) makes these quantities identical across a
    # cell's seeds, so any cross-seed spread here is a red flag.
    params = [p['n_params'] for p in per_seed if _finite(p.get('n_params'))]
    layers = [p['n_layers'] for p in per_seed if _finite(p.get('n_layers'))]
    dims = [p['dim'] for p in per_seed if _finite(p.get('dim'))]
    heads = [p['n_heads'] for p in per_seed if _finite(p.get('n_heads'))]
    norm_shares = [p['norm_share_of_params'] for p in per_seed
                   if _finite(p.get('norm_share_of_params'))]
    flops_fwd = [p['flops_per_token_forward'] for p in per_seed
                 if _finite(p.get('flops_per_token_forward'))]
    flops_train = [p['flops_per_token_training_est'] for p in per_seed
                   if _finite(p.get('flops_per_token_training_est'))]
    throughput_vals = [p['tokens_per_second'] for p in per_seed
                       if _finite(p.get('tokens_per_second'))]
    throughput_stats = mean_std_stderr(throughput_vals)
    peak_mem_vals = [p['peak_device_memory_mb'] for p in per_seed
                     if _finite(p.get('peak_device_memory_mb'))]
    peak_mem_stats = mean_std_stderr(peak_mem_vals)
    # A run that crashed, diverged or never finished leaves no readable
    # manifest. Pooling only the survivors would silently turn "this norm
    # destabilized 2 of 3 seeds" into a clean mean over the lucky seed, so the
    # missing seeds are recorded and the comparison refuses to certify an
    # incomplete cell as significantly better. A manifest that exists but
    # reports NaN/Inf is *not* a usable replication either: it is excluded from
    # the mean like a missing seed and listed separately.
    missing_seeds = [p['seed'] for p in per_seed
                     if not _finite(p['best_val_loss'])
                     and not _finite(p['final_val_loss'])]
    invalid_seeds = [p['seed'] for p in per_seed
                     if (isinstance(p['best_val_loss'], float) and not math.isfinite(p['best_val_loss']))
                     or (isinstance(p['final_val_loss'], float) and not math.isfinite(p['final_val_loss']))]
    seeds_without_final = [p['seed'] for p in per_seed
                           if not _finite(p['final_val_loss'])]
    return {
        'cell': cell_dir.name,
        'launch_id': launch_id,
        'seeds': seed_list,
        'n_seeds_requested': len(seed_list),
        'missing_seeds': missing_seeds,
        'invalid_seeds': invalid_seeds,
        'seeds_without_final_val_loss': seeds_without_final,
        'complete': (len(missing_seeds) == 0 and len(invalid_seeds) == 0
                     and not seeds_without_final
                     and not rejected and data_versions_complete
                     and not data_version_mismatch),
        'manifest_rejections': rejected,
        'protocol_verified_by_launch_id': launch_id is not None,
        # A cell is only one data version: if the resolved upstream commits
        # differ across its seeds, the pooled delta mixes snapshots and cannot
        # be certified as a depth/norm effect. A seed without resolved
        # provenance is not evidence of consistency either: the cell cannot be
        # certified until every seed records the commit it trained on.
        'data_versions_by_seed': versions_by_seed,
        'unknown_data_version_seeds': unknown_version_seeds,
        'data_versions_complete': data_versions_complete,
        'data_version': (json.loads(next(iter(known_versions)))
                         if len(known_versions) == 1 and not unknown_version_seeds
                         else None),
        'data_version_mismatch': data_version_mismatch,
        'per_seed': per_seed,
        'n_seeds': len(best_vals),
        'best_val_loss': None if best_stats is None else best_stats['mean'],
        'best_val_loss_std': None if best_stats is None or len(best_vals) < 2 else best_stats['std'],
        'best_val_loss_std_basis': 'seeds' if len(best_vals) >= 2 else 'single_run',
        'n_seeds_final': len(final_vals),
        'final_val_loss': None if final_stats is None else final_stats['mean'],
        'final_val_loss_std': None if final_stats is None or len(final_vals) < 2 else final_stats['std'],
        'final_val_loss_std_basis': 'seeds' if len(final_vals) >= 2 else 'single_run',
        'curve_classification_counts': curve_counts,
        'curve_classifications_by_seed': {p['seed']: p.get('curve_classification') for p in per_seed},
        'curve_generalization_gap': (sum(gaps) / len(gaps)) if gaps else None,
        'n_params': (sum(params) / len(params)) if params else None,
        'n_params_by_seed': {p['seed']: p.get('n_params') for p in per_seed},
        # Shape covariates: under the equal-parameter scaling (shape_for_depth)
        # a deeper cell is also a narrower one, so depth, dim and n_heads move
        # together and a cross-depth delta is never a pure depth effect. They
        # are pooled and flagged instead of left to be guessed from the cell
        # name.
        'n_layers': layers[0] if layers else None,
        'dim': (sum(dims) / len(dims)) if dims else None,
        'n_heads': heads[0] if heads else None,
        'shape_by_seed': {p['seed']: {'n_layers': p.get('n_layers'), 'dim': p.get('dim'),
                                      'n_heads': p.get('n_heads')} for p in per_seed},
        'shape_consistent_across_seeds': len({(p.get('n_layers'), p.get('dim'), p.get('n_heads'))
                                              for p in per_seed
                                              if any(_finite(p.get(k)) for k in ('n_layers', 'dim', 'n_heads'))}) <= 1,
        'norm_share_of_params': (sum(norm_shares) / len(norm_shares)) if norm_shares else None,
        'flops_per_token_forward': (sum(flops_fwd) / len(flops_fwd)) if flops_fwd else None,
        'flops_per_token_training_est': (sum(flops_train) / len(flops_train)) if flops_train else None,
        # Measured cost from the child manifests. Unlike the analytic FLOPs,
        # throughput and peak memory include the actual kernels and allocator,
        # so a depth/norm delta is read next to what it cost on this device.
        'tokens_per_second': None if throughput_stats is None else throughput_stats['mean'],
        'tokens_per_second_std': (throughput_stats['std']
                                  if throughput_stats and len(throughput_vals) >= 2 else None),
        'peak_device_memory_mb': None if peak_mem_stats is None else peak_mem_stats['mean'],
        'peak_device_memory_mb_std': (peak_mem_stats['std']
                                      if peak_mem_stats and len(peak_mem_vals) >= 2 else None),
        'cost_basis': ('tokens_per_second / peak_device_memory_mb are measured by the '
                       'child run; memory is null on CPU and throughput is device-dependent'),
        'n_loss_spikes': sum(p['n_loss_spikes'] or 0 for p in per_seed
                             if isinstance(p.get('n_loss_spikes'), (int, float))),
        'n_nonfinite_losses': sum(p['n_nonfinite_losses'] or 0 for p in per_seed
                                  if isinstance(p.get('n_nonfinite_losses'), (int, float))),
        'n_nonfinite_gradients': sum(p['n_nonfinite_gradients'] or 0 for p in per_seed
                                     if isinstance(p.get('n_nonfinite_gradients'), (int, float))),
        'n_nonfinite_val_losses': sum(p['n_nonfinite_val_losses'] or 0 for p in per_seed
                                      if isinstance(p.get('n_nonfinite_val_losses'), (int, float))),
        'aggregation': ('Mean +/- Std across seeds (each seed is an independent run of the identical recipe). '
                        'best_val_loss = min over validation passes (checkpoint-selection statistic, optimistically biased); '
                        'final_val_loss = last validation pass (fixed-step estimate, comparable across cells when early stopping is off). '
                        'curve_classification_counts = per-seed final train/val regime: a cell whose mean was produced '
                        'while some seed was underfit/stalled or overfitting/diverging must not be read as a clean '
                        'depth/norm effect. n_params/flops_per_token_* are analytic accounting (identical across the '
                        'cell seeds by construction); tokens_per_second/peak_device_memory_mb are measured by the '
                        'child runs, so the accuracy delta can be weighed against its actual cost. '
                        'n_layers/dim/n_heads = shape covariates: under the equal-parameter scaling a deeper cell '
                        'is narrower (dim and n_heads change with depth), so a cross-depth delta confounds depth '
                        'with width/head count and must not be read as a pure depth effect; a cross-norm delta at '
                        'the same depth is not affected. '
                        'missing_seeds = failed/unfinished runs, invalid_seeds = runs whose manifest reports '
                        'NaN/Inf; both are excluded from the mean but never from the report, because a cell with '
                        'missing/invalid seeds is not a clean replication. '
                        'n_nonfinite_losses / n_nonfinite_gradients / n_nonfinite_val_losses count triaged '
                        'numerical events across the cell: > 0 means the pooled mean mixes explicitly handled '
                        'failures into the arm and the triage evidence must be read before comparing.'),
    }

def _cell_seed_metric(cell: dict, metric: str) -> dict:
    """{seed: metric value} of an aggregate cell; seeds without the metric dropped."""
    out = {}
    for item in cell.get('per_seed') or []:
        if not isinstance(item, dict) or item.get('seed') is None:
            continue
        value = item.get(metric)
        if _finite(value):
            out[int(item['seed'])] = float(value)
    return out

def compare_sweep_cells(cells_by_depth: dict, reference_norm: str, metric: str='final_val_loss', margin: float=0.0, alpha: float=0.05) -> dict:
    """One-sided paired comparison of every norm against `reference_norm` per depth.

    Seeds are the unit of replication and every cell of the sweep shares the
    same seed set, so candidates and reference are *paired*: the test is a
    one-sided paired Student-t test on the per-seed `metric` differences
    (default `final_val_loss`, the fixed-step estimate - the selection-biased
    `best_val_loss` is deliberately not the basis). The t distribution, not
    the normal one, is the small-sample-correct reference at the default 3
    seeds (df = 2), and each row also reports a two-sided 95% confidence
    interval for the delta. The assumption-free per-seed sign counts
    (`n_seeds_favoring`, `n_seeds_tied`) are reported alongside. Only complete
    cells with >= 3 shared seeds enter the Holm-Bonferroni family and can be
    certified; two shared seeds (df = 1) are descriptive only.
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
                'certifiable': False,
                'significance_basis': None,
                't': None,
                'df': None,
                'delta_ci_95': None,
                'minimum_detectable_delta_95_power80': None,
                'candidate_missing_seeds': None,
                'reference_missing_seeds': None,
                'complete': True,
                # Each side's per-seed curve regimes, parameter count and
                # FLOPs, so a delta is read next to the regime it was
                # measured in.
                'candidate_curve_classification_counts': by_norm[norm].get('curve_classification_counts'),
                'reference_curve_classification_counts': ref.get('curve_classification_counts'),
                'candidate_n_params': by_norm[norm].get('n_params'),
                'reference_n_params': ref.get('n_params'),
                # Shape covariates of both sides: depth is what the sweep
                # varies, but under the equal-parameter scaling dim/n_heads
                # move with it, so the pair also fixes how much width differs.
                'candidate_shape': {'n_layers': by_norm[norm].get('n_layers'),
                                    'dim': by_norm[norm].get('dim'),
                                    'n_heads': by_norm[norm].get('n_heads')},
                'reference_shape': {'n_layers': ref.get('n_layers'),
                                    'dim': ref.get('dim'),
                                    'n_heads': ref.get('n_heads')},
                'candidate_flops_per_token_forward': by_norm[norm].get('flops_per_token_forward'),
                'reference_flops_per_token_forward': ref.get('flops_per_token_forward'),
                # The sweep's premise is a compute-matched depth/norm
                # comparison; these two ratios make that premise checkable
                # instead of assumed (candidate/reference, 1.0 = matched).
                'params_ratio_vs_reference': None,
                'flops_ratio_vs_reference': None,
                'cost_matched_within_25pct': None,
            }
            c_params = by_norm[norm].get('n_params')
            r_params = ref.get('n_params')
            c_flops = by_norm[norm].get('flops_per_token_forward')
            r_flops = ref.get('flops_per_token_forward')
            if _finite(c_params) and _finite(r_params) and r_params:
                entry['params_ratio_vs_reference'] = c_params / r_params
            if _finite(c_flops) and _finite(r_flops) and r_flops:
                entry['flops_ratio_vs_reference'] = c_flops / r_flops
            ratios = [entry['params_ratio_vs_reference'],
                      entry['flops_ratio_vs_reference']]
            if all(r is not None for r in ratios):
                entry['cost_matched_within_25pct'] = all(
                    abs(r - 1.0) <= _COMPUTE_MATCH_TOL for r in ratios)
            requested = sorted(set(ref.get('seeds') or []) | set(by_norm[norm].get('seeds') or []))
            entry['candidate_missing_seeds'] = [s for s in requested if s not in cand_seeds]
            entry['reference_missing_seeds'] = [s for s in requested if s not in ref_seeds]
            entry['complete'] = (
                not entry['candidate_missing_seeds']
                and not entry['reference_missing_seeds']
                and ref.get('complete', True)
                and by_norm[norm].get('complete', True))
            if len(common) >= 2:
                a = [cand_seeds[s] for s in common]
                b = [ref_seeds[s] for s in common]
                test = paired_t_one_sided(a, b, margin=margin)
                entry['mean_delta'] = test['mean_delta']
                entry['t'] = test['t']
                entry['df'] = test['df']
                entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
                entry['p_one_sided'] = test['p_one_sided']
                entry['n_seeds_favoring'] = sum(1 for x, y in zip(a, b) if x < y)
                entry['n_seeds_tied'] = sum(1 for x, y in zip(a, b) if x == y)
                half = (test['ci_high'] - test['ci_low']) / 2.0
                critical = student_t_ppf(1.0 - alpha / 2.0, float(test['df']))
                if critical > 0.0:
                    stderr = half / critical
                    entry['minimum_detectable_delta_95_power80'] = (
                        student_t_ppf(1.0 - alpha, float(test['df']))
                        + student_t_ppf(0.8, float(test['df']))
                    ) * stderr
            elif common:
                entry['mean_delta'] = cand_seeds[common[0]] - ref_seeds[common[0]]
            entry['certifiable'] = bool(
                entry['complete'] and len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
                and entry['cost_matched_within_25pct'] is True)
            if entry['p_one_sided'] is not None and not entry['certifiable']:
                if entry['cost_matched_within_25pct'] is False:
                    reason = ('the params/FLOPs ratio exceeds the 25% match tolerance, so the '
                              'delta is confounded with cost')
                elif entry['cost_matched_within_25pct'] is None:
                    reason = ('the params/FLOPs ratios are unavailable, so the compute-matched '
                              'premise cannot be verified')
                else:
                    reason = (f'{len(common)} shared seed(s), fewer than the required '
                              f'{MIN_SEEDS_FOR_SIGNIFICANCE}, or an incomplete cell')
                entry['significance_basis'] = f'descriptive only: {reason}; not certified'
            rows.append(entry)
    family = [r for r in rows if r['certifiable'] and r['p_one_sided'] is not None]
    adjusted = holm_adjust([r['p_one_sided'] for r in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['beats_reference_significantly'] = bool(
            entry['mean_delta'] is not None and entry['mean_delta'] < -margin
            and p_holm is not None and p_holm < alpha)
    incomplete = [
        {'depth': r['depth'], 'norm': r['norm'],
         'candidate_missing_seeds': r['candidate_missing_seeds'],
         'reference_missing_seeds': r['reference_missing_seeds']}
        for r in rows if not r['complete']
    ]
    return {
        'reference_norm': reference_norm,
        'metric': metric,
        'margin': margin,
        'alpha': alpha,
        'multiple_comparison': (f'Holm-Bonferroni over the {len(family)} complete, cost-matched, '
                                f'>= {MIN_SEEDS_FOR_SIGNIFICANCE}-seed cell comparison(s); '
                                f'incomplete, < {MIN_SEEDS_FOR_SIGNIFICANCE}-seed, or cost-mismatched '
                                f'cells are excluded from the family and are never certified'),
        'rows': rows,
        'incomplete_cells': incomplete,
        'reading_guide': ('paired Student-t on the shared seeds; `mean_delta` < 0 favors the candidate. '
                          'At 3 seeds (df = 2) the one-sided 5% critical value is t = 2.92: read '
                          '`n_seeds_favoring`/3 and the `delta_ci_95` range next to the p-value before '
                          'treating a delta as real. `minimum_detectable_delta_95_power80` is the '
                          'smallest true delta this seed count could detect at 80% power; a '
                          'non-significant row whose MDE exceeds the expected effect is underpowered, '
                          'not evidence of no effect. Only cells with >= 3 shared seeds and complete '
                          'coverage enter the Holm family and can be certified; a cell with `missing_seeds` '
                          '(`incomplete_cells`) is never certified as significantly better because which '
                          'runs survived is itself a selection effect; a row with '
                          '`cost_matched_within_25pct = false` is also withheld because its delta is '
                          'confounded with a params/FLOPs difference. Every row also carries the two sides\' '
                          '`*_curve_classification_counts` and `*_n_params` / '
                          '`*_flops_per_token_forward` (with `params_ratio_vs_reference` / '
                          '`flops_ratio_vs_reference` / `cost_matched_within_25pct`): an off-healthy-regime '
                          'side, an uncounted params/FLOPs increase, or a `cost_matched_within_25pct = false` '
                          'row means the delta is not yet a clean depth/norm claim - it is confounded with cost. '
                          'Rows also carry both sides\' `*_shape` (n_layers/dim/n_heads): with the equal-parameter '
                          'scaling a deeper cell is also narrower, so `candidate_shape.dim` (and n_heads) differ '
                          'across depths by construction - a same-depth cross-norm delta is a norm claim, a '
                          'cross-depth delta is depth *and* width/head count, never depth alone.'),
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
        ci = r.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        mde = r.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f" mde80={mde:.4f}"
        miss = ''
        if not r.get('complete', True):
            miss = (f" INCOMPLETE(candidate missing {r['candidate_missing_seeds']}, "
                    f"reference missing {r['reference_missing_seeds']})")
        regime = ''
        for label, key in (('cand', 'candidate_curve_classification_counts'),
                           ('ref', 'reference_curve_classification_counts')):
            off = {k: v for k, v in (r.get(key) or {}).items()
                   if k not in ('healthy', 'insufficient_data')}
            if off:
                regime += f" [{label} curve regimes {off}: delta measured off a healthy regime]"
        cost = ''
        c_params, r_params = r.get('candidate_n_params'), r.get('reference_n_params')
        if c_params is not None and r_params is not None:
            cost = f" params={c_params / 1e6:.2f}M vs {r_params / 1e6:.2f}M"
        ratios = []
        if r.get('params_ratio_vs_reference') is not None:
            ratios.append(f"params_ratio={r['params_ratio_vs_reference']:.3f}")
        if r.get('flops_ratio_vs_reference') is not None:
            ratios.append(f"flops_ratio={r['flops_ratio_vs_reference']:.3f}")
        if ratios:
            cost += ' (' + ', '.join(ratios) + ')'
        if r.get('cost_matched_within_25pct') is False:
            cost += ' [NOT compute-matched: delta confounded with cost]'
        shape = ''
        c_shape = r.get('candidate_shape') or {}
        r_shape = r.get('reference_shape') or {}
        if any(v is not None for v in c_shape.values()) or any(v is not None for v in r_shape.values()):
            shape = (f" shape cand L={c_shape.get('n_layers')},dim={c_shape.get('dim')},h={c_shape.get('n_heads')} vs "
                     f"ref L={r_shape.get('n_layers')},dim={r_shape.get('dim')},h={r_shape.get('n_heads')}")
        note = ''
        if r.get('significance_basis'):
            note = f" [{r['significance_basis']}]"
        print(f"[compare] d{r['depth']}-{r['norm']}: delta={r['mean_delta']:+.4f}{ci_str}{sign}{mde_str} "
              f"p_holm={p_str} significant={r['beats_reference_significantly']}{miss}{cost}{shape}{regime}{note}")
    if comparison.get('incomplete_cells'):
        print(f"[compare] WARNING: {len(comparison['incomplete_cells'])} comparison(s) involve a cell "
              f"with missing seed runs; significance is withheld there: "
              f"{comparison['incomplete_cells']}")

def _child_env(args, gpu=None, seed=None) -> dict:
    """Environment for one training child.

    Each cell is pinned to one GPU (round-robin over --gpus) and its BLAS
    thread count is capped: without this every concurrent child lands on the
    same visible device and the host oversubscribes its cores, which makes
    the sweep unreproducible.

    `PYTHONHASHSEED` is set per child from its seed: the child's own
    `set_all_seeds` runs long after its interpreter picked the string-hash
    seed, so the env var is the only point where the run seed can reach it
    (the multi-seed launcher in train_gpt2.py does the same).
    """
    env = dict(os.environ)
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    if seed is not None:
        env['PYTHONHASHSEED'] = str(int(seed) % (2 ** 32))
    threads = int(getattr(args, 'threads_per_run', 0) or 0)
    if threads > 0:
        for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            env[var] = str(threads)
    return env

def launch_one(args, depth: int, norm: str, output_dir: Path, seed: int, gpu=None, launch_id=None):
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
    if args.exclude_test_hash_file:
        cmd += ['--exclude-test-hash-file', str(args.exclude_test_hash_file)]
    if args.group_key:
        cmd += ['--group-key', str(args.group_key)]
    if args.exclude_group_file:
        cmd += ['--exclude-group-file', str(args.exclude_group_file)]
    if args.exclude_test_group_file:
        cmd += ['--exclude-test-group-file', str(args.exclude_test_group_file)]
    if args.dataset_revision:
        cmd += ['--dataset-revision', str(args.dataset_revision)]
    if args.tokenizer_revision:
        cmd += ['--tokenizer-revision', str(args.tokenizer_revision)]
    if args.gradient_checkpointing:
        cmd.append('--gradient-checkpointing')
    # The child training script runs the small-batch overfit sanity check by
    # default; forward the opt-out so a sweep can intentionally skip it.
    if not args.sanity_check:
        cmd.append('--no-sanity-check')
    if args.compile:
        cmd.append('--compile')
        cmd += ['--compile-mode', args.compile_mode]
    # Every cell carries the hypothesis the sweep tests, recorded in the run
    # manifest: the reference cell establishes the bar, a candidate cell
    # claims to clear it at the same depth and step count.
    if norm == args.reference_norm:
        hypothesis = (f'd{depth} reference cell: {norm} fixes the held-out final_val_loss '
                      f'every candidate norm must beat under the identical recipe (paired over '
                      f'the shared seeds).')
    else:
        hypothesis = (f'd{depth}: {norm} reaches a lower final_val_loss than the '
                      f'{args.reference_norm} reference at the same depth, recipe and step '
                      f'count, paired over the shared seeds.')
    cmd += ['--hypothesis', hypothesis]
    env = _child_env(args, gpu, seed=seed)
    env['ADASQUASH_DEPTH_OVERRIDE'] = str(depth)
    shape = ''
    if not args.no_dim_scaling:
        base_dim, base_heads = base_shape_for_model(args.model)
        dim, heads = shape_for_depth(depth, base_dim, base_heads)
        env['ADASQUASH_DIM_OVERRIDE'] = str(dim)
        env['ADASQUASH_HEADS_OVERRIDE'] = str(heads)
        shape = f' (dim={dim}, heads={heads})'
    print(f'[sweep] launching depth={depth} norm={norm} -> {output_dir}{shape}')
    # Invalidate the previous result *before* spawning. A child that dies
    # before train_gpt2.py writes its own manifest must leave no readable
    # result, otherwise the aggregator would silently reuse the old run.
    manifest_path = output_dir / 'run_manifest.json'
    try:
        manifest_path.unlink()
    except FileNotFoundError:
        pass
    spec = {
        'launch_id': launch_id,
        'depth': int(depth),
        'norm': norm,
        'seed': int(seed),
        'protocol': expected_child_protocol(args, depth, norm, seed),
    }
    (output_dir / 'launch_spec.json').write_text(json.dumps(spec, indent=2))
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
    p.add_argument('--dataset-revision', default=None, help='dataset revision (commit hash) pinned for every row; recorded in each run manifest')
    p.add_argument('--tokenizer-revision', default=None, help='tokenizer revision (commit hash) pinned for every row; recorded in each run manifest')
    p.add_argument('--val-frac', type=float, default=0.005, help='chronological held-out validation fraction (identical for every row)')
    p.add_argument('--test-frac', type=float, default=0.005, help='sealed test fraction (identical for every row)')
    p.add_argument('--num-documents', type=int, default=None, help='total documents; required when the dataset reports no size')
    p.add_argument('--val-every', type=int, default=500, help='held-out validation period in optimizer steps')
    p.add_argument('--val-batches', type=int, default=10, help='batches per validation pass')
    p.add_argument('--min-doc-chars', type=int, default=0)
    p.add_argument('--max-doc-chars', type=int, default=0)
    p.add_argument('--no-dedup', dest='dedup', action='store_false', help='disable exact-duplicate document removal')
    p.add_argument('--exclude-hash-file', default=None, help='training-document hash ledger shared by every depth/norm row (build it once with scripts/audit_data.py --write-train-hash-file)')
    p.add_argument('--exclude-test-hash-file', default=None, help='validation-document hash ledger shared by every depth/norm row (build it once with scripts/audit_data.py --write-val-hash-file); keeps the sealed test disjoint from the validation used for checkpoint selection')
    p.add_argument('--group-key', default=None, help='grouping-unit field shared by every row (see train_gpt2.py --group-key)')
    p.add_argument('--exclude-group-file', default=None, help='training-group hash ledger shared by every row (scripts/audit_data.py --write-train-group-file --group-key K); no group straddles train<->held-out')
    p.add_argument('--exclude-test-group-file', default=None, help='validation-group hash ledger shared by every row (scripts/audit_data.py --write-val-group-file --group-key K); no group straddles validation<->test')
    p.add_argument('--train-shuffle-buffer', type=int, default=0, help='packed training-batch shuffle window (0 = off)')
    p.add_argument('--tokenizer', default='huggyllama/llama-7b')
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
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='run the small-batch overfit sanity check in every row before training it (default: on, forwarded to the child trainers)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the per-row overfit sanity check')
    p.add_argument('--compile', action='store_true', help='enable torch.compile')
    p.add_argument('--compile-mode', default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--no-dim-scaling', action='store_true', help='keep dim/n_heads fixed across depths')
    p.add_argument('--output-dir', default='runs/depth-sweep')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stats-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--sequential', action='store_true', help='run configs strictly one at a time (equivalent to --max-parallel 1, which is the default)')
    p.add_argument('--max-parallel', type=int, default=1, help='maximum number of concurrently training cells; the default 1 keeps a 4x3x3 depth/norm/seed grid from launching 36 1B runs at once. Only raise it together with --gpus so each child has its own device')
    p.add_argument('--gpus', nargs='+', default=None, metavar='ID', help='GPU ids the cells are pinned to, round-robin in launch order (written to CUDA_VISIBLE_DEVICES of each child); without it all children share the ambient CUDA_VISIBLE_DEVICES')
    p.add_argument('--threads-per-run', type=int, default=1, help='OMP/MKL/OpenBLAS/etc thread cap for each child (0 = inherit the parent environment); 1 avoids oversubscription when cells share a host, raise it for a single serial run')
    apply_config_to_parser(p, known.config, _SWEEP_CONFIG_ALIASES)
    args = p.parse_args()
    raw_seeds = [args.seed] if args.seed is not None else list(args.seeds or [42])
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[sweep][warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed is the same run, not an independent replication)')
    if len(seeds) < 3:
        print(f'[sweep][warn] {len(seeds)} seed(s): a single depth/norm cell cannot be '
              f'separated from run-to-run noise; use >= 3 seeds before interpreting deltas')
    multi = len(seeds) > 1
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    # Every child written by this invocation carries this id in its launch
    # spec; aggregation refuses manifests from another sweep even when the
    # cell directory still exists from an earlier run.
    sweep_launch_id = f'{int(time.time())}-{os.getpid()}'
    sweep_manifest = collect_run_metadata(seed=seeds[0])
    sweep_manifest['hyperparameters'] = vars(args)
    sweep_manifest['seeds'] = seeds
    sweep_manifest['launch_id'] = sweep_launch_id
    sweep_manifest['grid'] = {
        'depths': list(args.depths),
        'norms': list(args.norms),
        'seeds': seeds,
        'runs': [f'd{depth}-{norm}-s{seed}'
                 for depth in args.depths for norm in args.norms for seed in seeds],
    }
    save_run_manifest(base / 'run_manifest.json', sweep_manifest)
    cells = []
    jobs = []
    for depth in args.depths:
        for norm in args.norms:
            cell_dir = base / f'd{depth}-{norm}'
            cell_dir.mkdir(parents=True, exist_ok=True)
            # This aggregate belongs to the previous sweep invocation until
            # the new cells are re-aggregated below. Remove it up front so a
            # launcher that dies mid-run cannot leave old means that look like
            # the current grid's results.
            (cell_dir / 'aggregate.json').unlink(missing_ok=True)
            for seed in seeds:
                out_dir = seed_dir(cell_dir, seed, multi)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / 'depth.txt').write_text(str(depth))
                jobs.append((depth, norm, out_dir, seed))
            cells.append((depth, norm, cell_dir))
    gpus = list(dict.fromkeys(str(g) for g in args.gpus)) if args.gpus else []
    failed_jobs = []

    def _gpu_for(idx: int):
        return gpus[idx % len(gpus)] if gpus else None

    def _reap(proc, depth, norm, seed):
        rc = proc.wait()
        if rc != 0:
            failed_jobs.append((depth, norm, seed, rc))
            print(f'[sweep] WARNING: d{depth}-{norm} seed={seed} exited rc={rc}; '
                  f'its manifest may be missing, which flags the cell incomplete')
        return rc

    max_parallel = max(1, int(args.max_parallel))
    if max_parallel > 1 and not args.sequential:
        if not gpus:
            raise SystemExit(
                '[sweep] --max-parallel > 1 requires --gpus: without it every '
                'child inherits the ambient CUDA_VISIBLE_DEVICES and would train '
                'on the same (default cuda:0) device at once. Pass --gpus 0 1 ... '
                'or keep --max-parallel 1.'
            )
        if max_parallel > len(gpus):
            print(f'[sweep][warn] --max-parallel {max_parallel} exceeds the '
                  f'{len(gpus)} GPU id(s); clamping to {len(gpus)} so no two '
                  f'cells share a device')
            max_parallel = len(gpus)
    if args.sequential or max_parallel == 1:
        if args.sequential and int(args.max_parallel) > 1:
            print('[sweep][warn] --sequential overrides --max-parallel')
        for idx, (depth, norm, out_dir, seed) in enumerate(jobs):
            proc = launch_one(args, depth, norm, out_dir, seed, gpu=_gpu_for(idx), launch_id=sweep_launch_id)
            rc = proc.wait()
            if rc != 0:
                failed_jobs.append((depth, norm, seed, rc))
            print(f'[sweep] finished d{depth}-{norm} seed={seed} rc={rc}')
    else:
        # A bounded launch queue: at most `max_parallel` training children
        # alive at once, each pinned to a currently free GPU.
        running: list = []
        busy_gpus: set = set()
        next_job = 0
        while next_job < len(jobs) or running:
            while next_job < len(jobs) and len(running) < max_parallel:
                free = [gpu for gpu in gpus if gpu not in busy_gpus]
                if not free:
                    break
                gpu = free[0]
                depth, norm, out_dir, seed = jobs[next_job]
                proc = launch_one(args, depth, norm, out_dir, seed, gpu=gpu, launch_id=sweep_launch_id)
                next_job += 1
                busy_gpus.add(gpu)
                running.append((proc, depth, norm, seed, gpu))
            done = [entry for entry in running if entry[0].poll() is not None]
            if not done:
                time.sleep(0.5)
                continue
            for entry in done:
                running.remove(entry)
                busy_gpus.discard(entry[4])
                _reap(*entry[:4])
    aggs_by_depth: dict = {}
    for depth, norm, cell_dir in cells:
        agg = aggregate_sweep_cell(
            cell_dir, seeds, multi,
            protocol_for_seed=lambda seed, d=depth, n=norm: expected_child_protocol(args, d, n, seed),
            launch_id=sweep_launch_id)
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
                line += '; final_val_loss missing from the manifest - cannot compare cells on the fixed-step estimate'
            print(line)
        else:
            print(f"[sweep] d{depth}-{norm}: no usable best_val_loss in the run manifests")
        regimes = agg.get('curve_classification_counts') or {}
        if regimes:
            line = f"[sweep] d{depth}-{norm}: curve regimes {regimes}"
            gap = agg.get('curve_generalization_gap')
            if gap is not None:
                line += f" (mean gap={gap:+.4f})"
            print(line)
            off = {k: v for k, v in regimes.items()
                   if k not in ('healthy', 'insufficient_data')}
            if off:
                print(f"[sweep] WARNING: d{depth}-{norm}: {sum(off.values())} seed(s) ended in "
                      f"{off}: the pooled val_loss describes that regime as much as the norm; "
                      f"read the per-seed curves before attributing depth/norm effects")
        else:
            print(f"[sweep] d{depth}-{norm}: no curve diagnosis in the run manifests")
        if agg.get('n_params') is not None:
            eff = f"[sweep] d{depth}-{norm}: n_params={agg['n_params'] / 1e6:.2f}M"
            if agg.get('dim') is not None:
                eff += (f" shape L={agg.get('n_layers')},dim={agg.get('dim')},"
                        f"h={agg.get('n_heads')}")
            share = agg.get('norm_share_of_params')
            if share is not None:
                eff += f" (norm share={share * 100:.2f}%)"
            fwd = agg.get('flops_per_token_forward')
            train = agg.get('flops_per_token_training_est')
            if fwd is not None:
                eff += f" flops/token fwd={fwd / 1e6:.2f}M"
            if train is not None:
                eff += f" train_est={train / 1e6:.2f}M"
            tps = agg.get('tokens_per_second')
            if tps is not None:
                tps_std = agg.get('tokens_per_second_std')
                tps_str = '' if tps_std is None else f" +/- {tps_std:.1f}"
                eff += f" measured={tps:.1f}{tps_str} tok/s"
            peak = agg.get('peak_device_memory_mb')
            if peak is not None:
                eff += f" peak={peak:.1f}MB"
            print(eff)
        bad_events = {label: agg.get(key) or 0 for label, key in (
            ('loss step(s)', 'n_nonfinite_losses'),
            ('gradient event(s)', 'n_nonfinite_gradients'),
            ('validation loss(es)', 'n_nonfinite_val_losses'))}
        if any(bad_events.values()):
            summary = ', '.join(f"{v} {k}" for k, v in bad_events.items() if v)
            print(f"[sweep] WARNING: d{depth}-{norm}: {summary} across seeds were handled by "
                  f"the ordered triage; read the per-seed triage evidence before comparing "
                  f"this cell (a cell with explicitly handled numerical failures is not a "
                  f"clean replication)")
        for rejection in agg.get('manifest_rejections') or []:
            print(f"[sweep] WARNING: d{depth}-{norm} seed={rejection['seed']}: rejected "
                  f"manifest ({rejection['reason']})")
        if agg.get('missing_seeds'):
            print(f"[sweep] WARNING: d{depth}-{norm}: missing/unfinished runs for seeds "
                  f"{agg['missing_seeds']}; the pooled mean covers {agg['n_seeds']} of "
                  f"{agg['n_seeds_requested']} requested seeds and comparisons will treat the "
                  f"cell as incomplete")
        if agg.get('invalid_seeds'):
            print(f"[sweep] WARNING: d{depth}-{norm}: seeds {agg['invalid_seeds']} report "
                  f"NaN/Inf losses; they are excluded from the pooled mean exactly like "
                  f"missing runs (a diverged run is not a clean replication)")
        if agg.get('data_version_mismatch'):
            print(f"[sweep] WARNING: d{depth}-{norm}: seeds were trained on different resolved "
                  f"data versions {agg['data_versions_by_seed']}; the cell is incomplete and "
                  f"its delta cannot be certified")
        if agg.get('unknown_data_version_seeds'):
            print(f"[sweep] WARNING: d{depth}-{norm}: seeds "
                  f"{agg['unknown_data_version_seeds']} do not record the resolved data "
                  f"version; the cell is incomplete because its snapshot cannot be audited")
    # Cross-cell version audit: a cell can be internally consistent while a
    # later cell resolved a moved upstream branch. Comparisons across such
    # cells would be confounded with the data snapshot, so mark every cell
    # incomplete rather than certifying them against each other.
    cell_data_versions = {
        f'd{depth}-{norm}': agg.get('data_version')
        for depth, norm, _ in cells
        for agg in [aggs_by_depth.get(depth, {}).get(norm, {})]
    }
    unique_cell_versions = {
        json.dumps(v, sort_keys=True) for v in cell_data_versions.values()
        if v is not None}
    unknown_cells = [name for name, version in cell_data_versions.items()
                     if version is None]
    data_versions_consistent = (bool(cell_data_versions)
                                and not unknown_cells
                                and len(unique_cell_versions) <= 1)
    if not data_versions_consistent:
        print('[sweep] WARNING: cells resolved different or unknown data revisions '
              f'{cell_data_versions}; every cell is marked incomplete and no '
              'cross-cell delta is certifiable')
        for by_norm in aggs_by_depth.values():
            for agg in by_norm.values():
                agg['complete'] = False
                agg['data_version_mismatch_across_cells'] = True
    print('[sweep][note] shape covariates: under the equal-parameter scaling dim and n_heads move with '
          'depth by construction, so a cross-depth loss delta confounds depth with width/head count and '
          'must not be read as a pure depth effect (same-depth cross-norm deltas are unaffected); every '
          'comparison row records both sides\' shape.')
    comparison = compare_sweep_cells(aggs_by_depth, args.reference_norm)
    print_sweep_comparisons(comparison)
    comp_path = base / 'sweep_comparisons.json'
    with open(comp_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f'[compare] comparisons -> {comp_path}')
    # The manifest was written before the children launched; finalize it now
    # with the outcome, otherwise a completed sweep is indistinguishable from a
    # launcher that died silently after writing the (pre-run) manifest.
    sweep_manifest['data_versions_by_cell'] = cell_data_versions
    sweep_manifest['data_versions_consistent'] = data_versions_consistent
    sweep_manifest['cells'] = {
        f'd{depth}-{norm}': {
            'n_seeds': agg.get('n_seeds'),
            'n_seeds_final': agg.get('n_seeds_final'),
            'best_val_loss': agg.get('best_val_loss'),
            'best_val_loss_std': agg.get('best_val_loss_std'),
            'final_val_loss': agg.get('final_val_loss'),
            'final_val_loss_std': agg.get('final_val_loss_std'),
            'missing_seeds': agg.get('missing_seeds'),
            'invalid_seeds': agg.get('invalid_seeds'),
            'manifest_rejections': agg.get('manifest_rejections'),
            'curve_classification_counts': agg.get('curve_classification_counts'),
        }
        for depth, norm, cell_dir in cells
        for agg in [aggs_by_depth.get(depth, {}).get(norm, {})]
    }
    sweep_manifest = finish_metadata(
        sweep_manifest,
        results={
            'n_cells': len(cells),
            'n_jobs': len(jobs),
            'incomplete_cells': comparison.get('incomplete_cells') if isinstance(comparison, dict) else None,
        },
        artifacts={'sweep_comparisons': artifact_record(comp_path)},
    )
    save_run_manifest(base / 'run_manifest.json', sweep_manifest)
    if jobs and len(failed_jobs) == len(jobs):
        print(f'[sweep] ERROR: every one of the {len(jobs)} child run(s) failed; '
              'the sweep produced no usable cell. See the per-seed logs above.',
              file=sys.stderr)
        raise SystemExit(1)
    print('[sweep] done')
if __name__ == '__main__':
    main()
