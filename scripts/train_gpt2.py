from __future__ import annotations
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from adasquash import GPTConfig, GPT, count_parameters, ActivationStats, LossSpikeCounter
from adasquash.attention import clear_logit_scratch_cache
from adasquash.diagnostics import diagnose_train_val_curve, triage_nonfinite
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.metrics import mean_ci, mean_std_stderr
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache, clear_cast_weight_cache
from adasquash.norms import NORM_REGISTRY
from adasquash.repro import artifact_record, collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds
from scripts.data_pipeline import build_dataloader, configure_data_options

def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config '{path}' must be a mapping, got {type(data)}")
    return data

def apply_config_to_parser(parser, path: str | None, aliases: dict | None=None):
    if not path:
        return
    cfg = load_yaml_config(path)
    aliases = aliases or {}
    known = {a.dest for a in parser._actions}
    unknown = []
    for key, value in cfg.items():
        dest, invert = aliases.get(key, (key, False))
        if dest not in known:
            unknown.append(key)
            continue
        if invert and isinstance(value, bool):
            value = not value
        parser.set_defaults(**{dest: value})
    if unknown:
        print(f'[config] {path}: ignoring unknown keys {sorted(unknown)}')
_TRAIN_CONFIG_ALIASES = {'use_layerscale': ('no_layerscale', True), 'learnable_p': ('no_learnable_p', True), 'learnable_k': ('no_learnable_k', True), 'learnable_delta': ('no_learnable_delta', True), 'tie_word_embeddings': ('tie_embeddings', False), 'bf16': ('no_bf16', True), 'beta': ('beta', False), 'alpha_jitter': ('alpha_jitter', False)}

def _snapshot_rng_state() -> dict:
    """Every RNG whose stream a resumed run must continue.

    torch CPU covers the model/init sampling, the CUDA generators cover
    dropout and any device-side sampling, and Python's `random` covers the
    data-shuffle buffer (`data_pipeline._shuffle`). Saving them with the
    weights makes a resume continue the same stochastic run instead of
    restarting the streams from `--seed`.
    """
    state = {'torch_cpu': torch.get_rng_state(), 'python': random.getstate()}
    try:
        import numpy as _np
    except ImportError:
        pass
    else:
        state['numpy'] = _np.random.get_state()
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def _atomic_torch_save(obj, path: Path) -> None:
    """Serialize to a sibling `.tmp` file, then atomically replace `path`.

    Writing straight to the target means a crash or full disk leaves a
    truncated file where a usable checkpoint used to be.
    """
    tmp = path.with_name(path.name + '.tmp')
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

def _load_checkpoint_cpu(path) -> dict:
    """Load a checkpoint on CPU (memory-mapped when the format allows).

    The caller drops the optimizer/scheduler entries once the weights have
    been copied.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (RuntimeError, ValueError):
        return torch.load(path, map_location='cpu', weights_only=False)

_ADAMW_FUSED_CPU_SUPPORTED: bool | None = None

def _adamw_fused_supported(device: torch.device) -> bool:
    """One-time probe for a usable fused AdamW kernel on `device`.

    CUDA always has one; CPU support depends on the build, so a one-element
    probe decides portability without special-casing torch versions.
    """
    if device.type == 'cuda':
        return True
    global _ADAMW_FUSED_CPU_SUPPORTED
    if _ADAMW_FUSED_CPU_SUPPORTED is None:
        probe = nn.Parameter(torch.zeros(1, device=device))
        probe.grad = torch.zeros(1, device=device)
        try:
            AdamW([probe], lr=0.0, fused=True).step()
            _ADAMW_FUSED_CPU_SUPPORTED = True
        except Exception:
            _ADAMW_FUSED_CPU_SUPPORTED = False
    return _ADAMW_FUSED_CPU_SUPPORTED

def build_optimizer(model: nn.Module, lr: float, weight_decay: float, beta1: float, beta2: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or 'norm' in name or 'delta' in name or ('ls1' in name) or ('ls2' in name) or name.endswith('.bias'):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    devices = {p.device for p in decay + no_decay}
    fused = len(devices) == 1 and _adamw_fused_supported(next(iter(devices)))
    return AdamW(groups, lr=lr, betas=(beta1, beta2), fused=fused)

def cosine_schedule(step: int, max_steps: int, warmup: int, min_lr_ratio: float):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

def build_lr_lambda(args):
    return lambda s: cosine_schedule(s + 1, args.max_steps, args.warmup, args.min_lr_ratio)
_SCHEDULE_ARGS = ('max_steps', 'warmup', 'min_lr_ratio', 'lr', 'weight_decay', 'beta1', 'beta2', 'grad_clip')
_ARCH_ARGS = ('model', 'norm', 'qk_norm', 'qk_alpha_init', 'no_layerscale', 'layerscale_init', 'tie_embeddings', 'no_learnable_p', 'no_learnable_k', 'no_learnable_delta', 'beta', 'alpha_0', 'k_init', 'alpha_init', 'p_init', 'alpha_jitter')
_NUMERIC_ARGS = ('compile', 'compile_mode', 'no_bf16')
_RESUME_RUN_ARGS = _ARCH_ARGS + _NUMERIC_ARGS

def restore_schedule_args(args, ckpt_args: dict, explicit: frozenset=frozenset()) -> list:
    changed = []
    for name in _SCHEDULE_ARGS:
        if name not in ckpt_args:
            continue
        if getattr(args, name, None) != ckpt_args[name]:
            changed.append(name)
        if name in explicit:
            continue
        setattr(args, name, ckpt_args[name])
    return changed
# The split-defining args are part of a run's identity: resuming with a
# different boundary could move documents the model already trained on into
# the validation split (and vice versa), so they are restored from the
# checkpoint exactly like the schedule flags. The cleaning knobs and the
# cross-split duplicate ledger are part of the same identity: changing them
# changes which documents are trained on and which are held out. The dataset /
# tokenizer revisions are the data *version*: resuming a run against a moved
# upstream branch would continue on different data under the same manifest.
_DATA_ARGS = ('dataset', 'dataset_config', 'dataset_revision', 'tokenizer_revision',
              'val_frac', 'test_frac', 'num_documents',
              'dedup', 'min_doc_chars', 'max_doc_chars', 'exclude_hash_file')

def restore_data_args(args, ckpt_args: dict, explicit: frozenset=frozenset()) -> list:
    changed = []
    for name in _DATA_ARGS:
        if name not in ckpt_args:
            continue
        if getattr(args, name, None) != ckpt_args[name]:
            changed.append(name)
        if name in explicit:
            continue
        setattr(args, name, ckpt_args[name])
    return changed

def open_data_split(split: str, args, seq_len: int | None = None):
    """Open one logical split through the shared leakage-safe pipeline."""
    configure_data_options(
        split=split,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        num_documents=args.num_documents,
        dedup=args.dedup,
        min_doc_chars=args.min_doc_chars,
        max_doc_chars=args.max_doc_chars,
        exclude_hash_file=getattr(args, 'exclude_hash_file', None),
        dataset_revision=getattr(args, 'dataset_revision', None),
        tokenizer_revision=getattr(args, 'tokenizer_revision', None),
        shuffle_buffer=args.train_shuffle_buffer if split == 'train' else 0,
        num_workers=getattr(args, 'num_workers', 2),
        seed=args.seed,
    )
    return build_dataloader(args.dataset, args.dataset_config, seq_len or args.seq_len, args.batch_size, tokenizer=args.tokenizer)

def evaluate_heldout(model, loader, state: dict, n_batches: int, ctx, device, collect: list | None = None) -> float | None:
    """Mean loss over held-out batches; `state['iter']` carries the position.

    Runs under `inference_mode` and eval mode, so no gradient ever flows from
    a validation or test batch into the parameters. When `collect` is given,
    every batch's loss is also appended to it: the mean alone cannot say how
    noisy the held-out estimate is, nor whether two arms were scored on the
    same held-out segments (the pairing a paired test needs). The accumulator
    is untouched by `collect`, so the returned mean is bit-identical either
    way.
    """
    # With dropout at its default 0 the module-mode switch changes no
    # numerics, so skip it; inference_mode already blocks grads. `training`
    # does more than dropout, though: gradient checkpointing keys its
    # recompute branch on it, so the switch must stay when that is on.
    cfg = getattr(model, 'cfg', None)
    toggle_mode = float(getattr(cfg, 'dropout', 0.0)) > 0.0 or bool(getattr(cfg, 'gradient_checkpointing', False))
    if toggle_mode:
        model.eval()
    it = state.get('iter')
    total = None
    n = 0
    terms: list = [] if collect is not None else None
    with torch.inference_mode():
        for _ in range(max(1, int(n_batches))):
            if it is None:
                it = iter(loader)
                state['iter'] = it
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                state['iter'] = it
                try:
                    batch = next(it)
                except StopIteration:
                    break
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with ctx:
                loss = model(x, y, return_logits=False)[1]
            # Accumulate in a device fp64 scalar; fp64 keeps the sequential
            # summation bit-identical to host float accumulation.
            term = loss.detach().double()
            if terms is not None:
                terms.append(term)
            total = term.clone() if total is None else total.add_(term)
            n += 1
    if toggle_mode:
        model.train()
    if collect is not None and terms:
        # `.tolist()` yields the same Python doubles in the same batch order,
        # so the returned mean and the collected series are unchanged.
        collect.extend(torch.stack(terms).cpu().tolist())
    return float(total) / n if n else None

def load_model_state_dict(model, state_dict) -> None:
    target = getattr(model, '_orig_mod', model)
    prefix = '_orig_mod.'
    if any(k.startswith(prefix) for k in state_dict):
        state_dict = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
    target.load_state_dict(state_dict)

def _is_no_decay_group(group) -> bool:
    # build_optimizer routes every parameter with ndim < 2 (norm gains, biases,
    # LayerScale and the scalar squashing parameters) to the no-decay group and
    # every >= 2-D weight to the decay group. Re-deriving the split from the
    # parameters keeps it correct after load_state_dict replaces each group's
    # dict (weight_decay included) with the checkpoint's copy.
    return all(p.ndim < 2 for p in group['params'])

def apply_explicit_optimizer_args(opt, sched, args, explicit: frozenset=frozenset()) -> None:
    if 'lr' in explicit and sched.base_lrs:
        base = float(args.lr)
        ratio = base / sched.base_lrs[0]
        if ratio != 1.0:
            sched.base_lrs = [lr * ratio for lr in sched.base_lrs]
            for group in opt.param_groups:
                group['lr'] = group['lr'] * ratio
    for group in opt.param_groups:
        if 'weight_decay' in explicit:
            group['weight_decay'] = 0.0 if _is_no_decay_group(group) else args.weight_decay
        if 'beta1' in explicit or 'beta2' in explicit:
            b0, b1 = group['betas']
            group['betas'] = (args.beta1 if 'beta1' in explicit else b0, args.beta2 if 'beta2' in explicit else b1)

DEFAULT_SEED = 42

# Default (n_layers, dim, n_heads) of the model variants. The preflight uses
# them - adjusted by the sweep's ADASQUASH_*_OVERRIDE env vars - so the
# small-batch overfit check exercises the architecture this run will train
# instead of sanity_overfit.py's 2-layer dim=64 default.
_MODEL_SHAPES = {
    'gpt2-124m': (12, 768, 12),
    'gpt2-350m': (24, 1024, 16),
    'llama-1b': (24, 2048, 16),
}

def preflight_arch_args(args, depth_override=None, dim_override=None, heads_override=None) -> list:
    """CLI flags making the overfit preflight exercise this run's architecture.

    The preflight keeps sanity_overfit.py's small batch/step workload (a
    wiring bug is visible at any sequence length), but the shape and every
    architecture / numeric-format flag match the run: norm, quantization of
    Q/K, LayerScale, learnable parameters, gradient checkpointing, tied
    embeddings, autocast format and RoPE - the paths a full run uses.
    """
    n_layers, dim, n_heads = _MODEL_SHAPES.get(args.model, _MODEL_SHAPES['gpt2-124m'])
    if depth_override:
        n_layers = int(depth_override)
    if dim_override:
        dim = int(dim_override)
    if heads_override:
        n_heads = int(heads_override)
    arch = ['--dim', str(dim), '--n-layers', str(n_layers), '--n-heads', str(n_heads)]
    if not args.no_layerscale:
        arch.append('--use-layerscale')
    arch += ['--layerscale-init',
             str(args.layerscale_init if args.layerscale_init is not None else 1e-5)]
    if args.qk_norm:
        arch += ['--qk-norm', args.qk_norm, '--qk-alpha-init', str(args.qk_alpha_init)]
    if args.no_learnable_p:
        arch.append('--no-learnable-p')
    if args.no_learnable_k:
        arch.append('--no-learnable-k')
    if args.no_learnable_delta:
        arch.append('--no-learnable-delta')
    if not args.beta:
        arch.append('--no-beta')
    arch += ['--alpha-0', str(args.alpha_0), '--alpha-init', str(args.alpha_init),
             '--k-init', str(args.k_init), '--p-init', str(args.p_init),
             '--alpha-jitter', str(args.alpha_jitter)]
    if args.gradient_checkpointing:
        arch.append('--gradient-checkpointing')
    if args.tie_embeddings is not None:
        arch.append('--tie-embeddings' if args.tie_embeddings else '--no-tie-embeddings')
    elif args.model == 'llama-1b':
        # `build_llama_tiny_1b` is untied by default while sanity_overfit.py's
        # config defaults to tied, so without an explicit flag the preflight
        # would exercise a different model than this run trains.
        arch.append('--no-tie-embeddings')
    if args.no_bf16:
        arch.append('--no-bf16')
    if args.model == 'llama-1b':
        arch.append('--rotary')
    return arch

def seed_run_dir(base: Path, seed: int) -> Path:
    """Per-seed run directory: `<output-dir>/seed{N}/` in multi-seed mode."""
    return base / f'seed{int(seed)}'

def aggregate_seed_runs(base_dir: Path, seeds, model: str, norm: str) -> dict:
    """Pool per-seed runs into one Mean +/- Std record (Phase 1 statistics).

    The seed is the unit of replication: seed runs are independent draws of
    the whole protocol (initialization, data order, batch sampling), while the
    held-out batches inside one run are not. `final_val_loss` (the last
    validation pass at the fixed step count) is the number the variants are
    compared on; `best_val_loss` travels next to it because it is the
    checkpoint-selection statistic and is optimistically biased. Both are
    pooled across seeds, and the sealed `test_loss` of every seed (when the
    run evaluated it) is kept per seed so `scripts/compare_sealed_tests.py`
    can pair two arms by their shared training seeds instead of by test
    batches of a single run. Missing/invalid seeds are reported, never
    silently dropped.
    """
    per_seed = []
    for seed in seeds:
        run_dir = seed_run_dir(base_dir, seed)
        record = {
            'seed': int(seed), 'run_dir': str(run_dir),
            'final_val_loss': None, 'best_val_loss': None,
            'final_val_step': None, 'best_val_step': None,
            'n_params': None, 'n_layers': None, 'dim': None, 'n_heads': None,
            'n_loss_spikes': None, 'n_nonfinite_losses': None,
            'curve_classification': None, 'test_loss': None,
        }
        try:
            manifest = json.loads((run_dir / 'run_manifest.json').read_text())
            results = manifest.get('results') if isinstance(manifest, dict) else None
            if isinstance(results, dict):
                record['final_val_loss'] = results.get('final_val_loss')
                record['best_val_loss'] = results.get('best_val_loss')
                record['final_val_step'] = results.get('final_val_step')
                record['best_val_step'] = results.get('best_val_step')
                record['n_params'] = results.get('n_params')
                record['n_layers'] = results.get('n_layers')
                record['dim'] = results.get('dim')
                record['n_heads'] = results.get('n_heads')
                record['n_loss_spikes'] = results.get('n_loss_spikes')
                record['n_nonfinite_losses'] = results.get('n_nonfinite_losses')
                curve = results.get('curve_diagnosis') or {}
                record['curve_classification'] = curve.get('classification')
        except (OSError, json.JSONDecodeError):
            pass
        try:
            test = json.loads((run_dir / 'test_results.json').read_text())
            value = test.get('test_loss') if isinstance(test, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                record['test_loss'] = float(value)
        except (OSError, json.JSONDecodeError):
            pass
        per_seed.append(record)

    def _finite(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    def _pool(key: str):
        vals = [float(r[key]) for r in per_seed if _finite(r.get(key))]
        return vals, (mean_std_stderr(vals) if vals else None)

    final_vals, final_stats = _pool('final_val_loss')
    best_vals, best_stats = _pool('best_val_loss')
    test_vals, test_stats = _pool('test_loss')
    missing = [r['seed'] for r in per_seed if not _finite(r['final_val_loss']) and not _finite(r['best_val_loss'])]
    invalid = [r['seed'] for r in per_seed
               if any(isinstance(r[k], float) and not math.isfinite(r[k]) for k in ('final_val_loss', 'best_val_loss'))]
    return {
        'model': model, 'norm': norm,
        'seeds': [int(s) for s in seeds],
        'n_seeds_requested': len(list(seeds)),
        'n_seeds': len(final_vals),
        'missing_seeds': missing,
        'invalid_seeds': invalid,
        'complete': not missing and not invalid,
        'per_seed': per_seed,
        'final_val_loss': None if final_stats is None else final_stats['mean'],
        'final_val_loss_std': None if final_stats is None or len(final_vals) < 2 else final_stats['std'],
        'final_val_loss_stderr': None if final_stats is None or len(final_vals) < 2 else final_stats['stderr'],
        'final_val_loss_std_basis': 'seeds' if len(final_vals) >= 2 else 'single_run',
        'best_val_loss': None if best_stats is None else best_stats['mean'],
        'best_val_loss_std': None if best_stats is None or len(best_vals) < 2 else best_stats['std'],
        'best_val_loss_std_basis': 'seeds' if len(best_vals) >= 2 else 'single_run',
        'n_seeds_test': len(test_vals),
        'test_loss': None if test_stats is None else test_stats['mean'],
        'test_loss_std': None if test_stats is None or len(test_vals) < 2 else test_stats['std'],
        'test_loss_by_seed': {r['seed']: r['test_loss'] for r in per_seed},
        'aggregation': (
            'Mean +/- Std across independent seed runs of the identical recipe (same optimizer, '
            'schedule, data protocol and step count). final_val_loss = last validation pass at the '
            'fixed step count (the comparable estimate); best_val_loss = min over validation passes '
            '(checkpoint-selection statistic, optimistically biased as an estimate). A single-seed '
            'std is not a run-to-run spread, so std is None there. missing_seeds/invalid_seeds are '
            'reported but excluded from the mean; a comparison over an incomplete set is not clean.'),
    }

def launch_multi_seed(args, seeds) -> None:
    """Run one full training child per seed, then aggregate Mean +/- Std.

    Each child is the same command line with `--seeds <one>` and its own
    `--output-dir <base>/seed{N}`: identical optimizer, schedule, data and
    architecture, only the seed differs. Children run sequentially so the
    replication does not change resource pressure per run; `PYTHONHASHSEED`
    is pinned per child so interpreter-level hash randomization cannot leak
    into the run.
    """
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    if len(seeds) < 3:
        print(f'[seeds][warn] {len(seeds)} seed(s): a norm effect cannot be separated from '
              f'run-to-run noise; use >= 3 independent seeds before interpreting deltas')
    child_argv = list(sys.argv[1:])
    script = str(Path(__file__).resolve())
    failed_seeds: list = []
    for seed in seeds:
        out_dir = seed_run_dir(base, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'[seeds] launching seed={seed} -> {out_dir}', flush=True)
        env = dict(os.environ)
        env['PYTHONHASHSEED'] = str(seed)
        cmd = [sys.executable, script, *child_argv, '--seeds', str(seed), '--output-dir', str(out_dir)]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            failed_seeds.append(seed)
            print(f'[seeds][warn] seed={seed} exited rc={rc}; its manifest may be missing and the '
                  f'aggregate will flag it', flush=True)
    agg = aggregate_seed_runs(base, seeds, args.model, args.norm)
    (base / 'aggregate.json').write_text(json.dumps(agg, indent=2))
    for label, mean_key, std_key in (('final_val_loss', 'final_val_loss', 'final_val_loss_std'),
                                     ('best_val_loss', 'best_val_loss', 'best_val_loss_std'),
                                     ('test_loss', 'test_loss', 'test_loss_std')):
        mean = agg.get(mean_key)
        if mean is None:
            continue
        std = agg.get(std_key)
        std_str = '' if std is None else f' +/- {std:.4f}'
        print(f'[seeds] {args.norm}: {label}={mean:.4f}{std_str} over {agg["n_seeds"]} seed(s)')
    if agg['missing_seeds']:
        print(f'[seeds][warn] missing/unfinished seed runs: {agg["missing_seeds"]}; the pooled mean '
              f'covers {agg["n_seeds"]} of {agg["n_seeds_requested"]} requested seeds')
    print(f'[seeds] per-seed aggregate -> {base / "aggregate.json"}')
    manifest = collect_run_metadata(seed=seeds[0])
    manifest['hyperparameters'] = vars(args)
    manifest['seeds'] = [int(s) for s in seeds]
    manifest['aggregate'] = agg
    # The replication manifest points at the pooled result and at every child
    # manifest, so a Phase-1 statistic can be audited down to the per-seed run
    # that produced it.
    manifest = finish_metadata(
        manifest,
        artifacts={
            'aggregate': artifact_record(base / 'aggregate.json'),
            'per_seed_manifests': {
                str(int(seed)): artifact_record(seed_run_dir(base, seed) / 'run_manifest.json')
                for seed in seeds
            },
        },
    )
    if failed_seeds:
        manifest['status'] = 'failed'
        manifest['failed_seeds'] = [int(s) for s in failed_seeds]
    save_run_manifest(base / 'run_manifest.json', manifest)
    if failed_seeds:
        # A multi-seed invocation is a replication study: a failed child is
        # not a valid draw, and exiting 0 would let CI/orchestration treat
        # the pooled mean as a clean result while it silently covers fewer
        # seeds than requested. The artifacts above are already on disk.
        raise SystemExit(
            f'[seeds] {len(failed_seeds)}/{len(seeds)} seed run(s) failed {failed_seeds}: the '
            f'aggregate covers {agg["n_seeds"]} of {agg["n_seeds_requested"]} requested seeds, '
            'so this invocation is not a clean replication. Fix the failures (each seed writes '
            'its own run_manifest.json) and re-run before quoting the pooled numbers.')

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='YAML run config; CLI flags override it')
    p.add_argument('--model', choices=['gpt2-124m', 'gpt2-350m', 'llama-1b'], default='gpt2-124m')
    p.add_argument('--norm', default='dytcd', choices=sorted(NORM_REGISTRY))
    p.add_argument('--qk-norm', default=None, choices=[None, 'qknorm', 'dytqk'])
    p.add_argument('--qk-alpha-init', type=float, default=1.0, help='initial alpha_q / alpha_k for --qk-norm dytqk')
    p.add_argument('--no-layerscale', action='store_true')
    p.add_argument('--layerscale-init', type=float, default=None)
    p.add_argument('--alpha-0', type=float, default=0.9)
    p.add_argument('--alpha-init', type=float, default=0.5)
    p.add_argument('--k-init', type=float, default=1.0)
    p.add_argument('--p-init', type=float, default=2.0)
    p.add_argument('--no-learnable-p', action='store_true')
    p.add_argument('--no-learnable-k', action='store_true')
    p.add_argument('--no-learnable-delta', action='store_true', help='freeze the DyT-CD per-channel deviation at zero (controlled per-channel off)')
    p.add_argument('--no-beta', dest='beta', action='store_false', help='disable output affine bias in DyT-family norms')
    p.add_argument('--alpha-jitter', type=float, default=0.05, help='per-channel alpha init jitter for DyTC (0 for deterministic)')
    p.add_argument('--gradient-checkpointing', action='store_true', help='enable gradient checkpointing')
    p.add_argument('--tie-embeddings', dest='tie_embeddings', action='store_true', default=None, help='tie input/output embeddings (default: GPT-2 yes, LLaMA no)')
    p.add_argument('--no-tie-embeddings', dest='tie_embeddings', action='store_false')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--dataset-revision', default=None, help='dataset revision (commit hash) to pin; recorded in the run manifest so the trained-on data version is auditable (None = upstream default branch, which can move)')
    p.add_argument('--tokenizer-revision', default=None, help='tokenizer revision (commit hash) to pin; recorded in the run manifest (None = upstream default branch)')
    p.add_argument('--val-frac', type=float, default=0.005, help='chronological held-out validation fraction of the document stream (split order is train=oldest, then val, then test; never shuffled)')
    p.add_argument('--test-frac', type=float, default=0.005, help='sealed test fraction of the document stream, evaluated once at the end only with --final-test-eval')
    p.add_argument('--num-documents', type=int, default=None, help='total documents in the stream; required when the dataset reports no size')
    p.add_argument('--val-every', type=int, default=500, help='run a held-out validation pass every N optimizer steps (0 disables; 0 is not recommended)')
    p.add_argument('--val-batches', type=int, default=10, help='batches per held-out validation pass')
    p.add_argument('--early-stop-patience', type=int, default=0, help='stop after this many validation passes without improvement (0 = off)')
    p.add_argument('--nonfinite-policy', choices=['abort', 'skip'], default='abort', help='what to do when a training loss is NaN/Inf: abort (default) runs the ordered data->math->precision->gradient triage and stops; skip runs the triage and discards only that step (5 consecutive non-finite steps still abort)')
    p.add_argument('--final-test-eval', action='store_true', help='after training, evaluate the sealed test split exactly once (restores the best-validation checkpoint first); per-batch losses and the data-pairing block are stored in test_results.json so scripts/compare_sealed_tests.py can test two arms on the shared sealed batches')
    p.add_argument('--allow-test-rerun', action='store_true', help='permit a second one-shot evaluation of the sealed test split in the same output dir (breaks the seal: the test set becomes an implicit validation set)')
    p.add_argument('--test-batches', type=int, default=20, help='batches for the one-shot final test evaluation')
    p.add_argument('--min-doc-chars', type=int, default=0, help='drop documents shorter than this many characters (0 = keep all non-empty)')
    p.add_argument('--max-doc-chars', type=int, default=0, help='drop documents longer than this many characters (0 = no length-outlier cap)')
    p.add_argument('--no-dedup', dest='dedup', action='store_false', help='disable exact-duplicate document removal')
    p.add_argument('--exclude-hash-file', default=None, help='training-document hash ledger (scripts/audit_data.py --write-train-hash-file); held-out documents whose exact text also occurs in train are dropped from validation/test, never from train')
    p.add_argument('--train-shuffle-buffer', type=int, default=0, help='shuffle window over packed TRAINING batches (0 = off); never applied to validation or test')
    p.add_argument('--num-workers', type=int, default=2, help='DataLoader worker processes (0 = everything in the parent process); with dedup enabled each worker scans and hashes the whole stream, so raising this multiplies CPU work rather than throughput')
    p.add_argument('--seq-len', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=32, help='per-device micro batch')
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--max-steps', type=int, default=10000)
    p.add_argument('--lr', type=float, default=0.0006)
    p.add_argument('--min-lr-ratio', type=float, default=0.1)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.95)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--no-bf16', action='store_true', help='disable bf16 autocast')
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given (single-seed default: 42)')
    p.add_argument('--seeds', nargs='+', type=int, default=None, help='independent replication seeds for Phase 1: each seed is a full training run under <output-dir>/seed{N}/ and Mean +/- Std across seeds is written to <output-dir>/aggregate.json (seeds are the unit of replication; >= 3 recommended, because a single run cannot separate a norm effect from run-to-run noise). One seed runs in place, exactly as before. Phase 1 configs ship seeds: [42, 43, 44].')
    p.add_argument('--compile', action='store_true', help='enable torch.compile')
    p.add_argument('--compile-mode', default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='run the small-batch overfit sanity check (scripts/sanity_overfit.py) before training and abort if it fails (default: on; a model that cannot overfit one batch has a gradient/loss bug and must not start a full run)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the pre-flight overfit sanity check (use only when the check itself is the problem, e.g. offline runs where the preflight cannot be spawned)')
    p.add_argument('--output-dir', default='runs/gpt2-124m-dytcd')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stats-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--wandb-project', default=None)
    p.add_argument('--run-name', default=None)
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this run tests; recorded in the run manifest (and checkpoint args) so a result can be audited against the claim it was meant to support, the same contract the ablation matrix enforces per row')
    p.add_argument('--resume', default=None, help='checkpoint to resume from')
    p.add_argument('--resume-skip-data', action='store_true', help='do NOT fast-forward the training dataloader past the batches the checkpoint already consumed (the resumed run then replays training data)')
    apply_config_to_parser(p, known.config, _TRAIN_CONFIG_ALIASES)
    args = p.parse_args()
    raw_seeds = [args.seed] if args.seed is not None else list(args.seeds or [DEFAULT_SEED])
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[seeds][warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed is the same run, not an independent replication)')
    if len(seeds) > 1:
        if args.resume:
            raise SystemExit(
                '[seeds] --resume continues one run; combining it with multiple --seeds would '
                'either resume the same step count into every seed directory or silently ignore '
                'the requested replication. Resume a single seed with --seed N --resume <ckpt>.')
        launch_multi_seed(args, seeds)
        return
    args.seed = seeds[0]
    explicit = frozenset((name for name in _SCHEDULE_ARGS + _RESUME_RUN_ARGS + _DATA_ARGS if getattr(args, name, None) != p.get_default(name)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resume_ckpt = None
    run_args: dict = {}
    if args.resume:
        resume_ckpt = _load_checkpoint_cpu(args.resume)
        run_args = resume_ckpt.get('args') or {}
        resume_kept = []
        for name in _RESUME_RUN_ARGS:
            if name not in run_args:
                continue
            if name in explicit:
                if getattr(args, name, None) != run_args[name]:
                    resume_kept.append(name)
                continue
            setattr(args, name, run_args[name])
        if resume_kept:
            print(f'[resume] keeping your explicit override(s) {sorted(resume_kept)} for the architecture / numeric-format flags as well - the resumed run therefore differs from the one it continues')
        for env_name, arg_name in (('ADASQUASH_DEPTH_OVERRIDE', '_depth_override'), ('ADASQUASH_DIM_OVERRIDE', '_dim_override'), ('ADASQUASH_HEADS_OVERRIDE', '_heads_override')):
            if run_args.get(arg_name) is not None:
                os.environ[env_name] = str(run_args[arg_name])
    args.bf16 = not args.no_bf16
    set_all_seeds(args.seed)
    # The depth sweep feeds its per-cell shape through these env vars (see
    # ADASQUASH_*_OVERRIDE below). Read them *before* the preflight so the
    # check runs on the cell's real architecture rather than the default.
    depth_override = os.environ.get('ADASQUASH_DEPTH_OVERRIDE')
    dim_override = os.environ.get('ADASQUASH_DIM_OVERRIDE')
    heads_override = os.environ.get('ADASQUASH_HEADS_OVERRIDE')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    sanity_report_path = None
    if args.sanity_check:
        # Fail fast: a model that cannot overfit one small batch has a
        # gradient/loss plumbing bug and must not start the full run. The
        # preflight subprocess writes its report into the output dir; the
        # manifest below fingerprints it. Seeding first also exports
        # PYTHONHASHSEED so the child interpreter starts with a fixed hash
        # seed. The architecture flags make the check exercise this run's
        # model shape and numeric path, not the checker's tiny default.
        from scripts.sanity_overfit import run_preflight
        run_preflight(args.norm, args.seed, args.output_dir,
                      arch_args=preflight_arch_args(args, depth_override,
                                                    dim_override, heads_override))
        sanity_report_path = Path(args.output_dir) / 'sanity_overfit.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.bf16 and device == 'cuda' else torch.float32
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if dtype != torch.float32 else nullcontext()
    model_cfg = dict(norm_name=args.norm, qk_norm=args.qk_norm, qk_alpha_init=args.qk_alpha_init, use_layerscale=not args.no_layerscale, alpha_0=args.alpha_0, k_init=args.k_init, alpha_init=args.alpha_init, p_init=args.p_init, learnable_p=not args.no_learnable_p, learnable_k=not args.no_learnable_k, learnable_delta=not args.no_learnable_delta, beta=args.beta, alpha_jitter=args.alpha_jitter, gradient_checkpointing=args.gradient_checkpointing)
    if args.layerscale_init is not None:
        model_cfg['layerscale_init'] = args.layerscale_init
    if args.tie_embeddings is not None:
        model_cfg['tie_word_embeddings'] = args.tie_embeddings
    # Construct on the target device (no full-model H2D) with the
    # constructor's init pass disabled: `reset_parameters(args.seed)` below is
    # the run's single, seeded initialization.
    with torch.device(device):
        if args.model == 'gpt2-124m':
            model = build_gpt_124m(init=False, **model_cfg)
        elif args.model == 'gpt2-350m':
            model = build_gpt_350m(init=False, **model_cfg)
        else:
            model = build_llama_tiny_1b(init=False, **model_cfg)
    if depth_override or dim_override or heads_override:
        cfg_dict = {k: getattr(model.cfg, k) for k in vars(model.cfg)}
        if depth_override:
            cfg_dict['n_layers'] = int(depth_override)
            args._depth_override = int(depth_override)
        if dim_override:
            cfg_dict['dim'] = int(dim_override)
            args._dim_override = int(dim_override)
        if heads_override:
            cfg_dict['n_heads'] = int(heads_override)
            args._heads_override = int(heads_override)
        print(f"[shape-override] L={cfg_dict['n_layers']} dim={cfg_dict['dim']} n_heads={cfg_dict['n_heads']} (was L={model.cfg.n_layers} dim={model.cfg.dim} n_heads={model.cfg.n_heads})")
        with torch.device(device):
            model = GPT(GPTConfig(**cfg_dict), init=False)
    model.reset_parameters(args.seed)
    model.to(device)
    if args.compile:
        print(f'[compile] enabling torch.compile with mode={args.compile_mode}')
        model = torch.compile(model, mode=args.compile_mode)
    n_params = count_parameters(model)
    n_trainable = count_parameters(model, trainable_only=True)
    print(f'[model] {args.model} | total={n_params / 1000000.0:.2f}M | trainable={n_trainable / 1000000.0:.2f}M | norm={args.norm}')
    print(f'[model] norm param breakdown: {model.norm_param_count()}')
    if resume_ckpt is not None:
        data_changed = restore_data_args(args, run_args, explicit=explicit)
        if data_changed:
            kept = [n for n in data_changed if n in explicit]
            adopted = [n for n in data_changed if n not in explicit]
            if adopted:
                print(f'[resume] data-split args restored from checkpoint: {adopted}')
            if kept:
                print(f'[resume] keeping your explicit data-split override(s) {kept} - the resumed run therefore evaluates on a different split than the one it continues')
    data_reports: dict = {}
    loader = open_data_split('train', args)
    data_iter = iter(loader)
    batches_consumed = 0
    report = getattr(loader, 'split_report', None)
    if report:
        data_reports['train'] = report
        print(f'[data] train split: {report}')
    val_loader = None
    val_state: dict = {'iter': None}
    best_val_loss = float('inf')
    best_val_step = 0
    val_checks_without_improvement = 0
    last_val_loss = None
    last_val_step = None
    last_val_batches: list = []
    last_train_loss = None
    start_step = 0
    if resume_ckpt is not None:
        changed = restore_schedule_args(args, run_args, explicit=explicit)
        if changed:
            kept = [n for n in changed if n in explicit]
            adopted = [n for n in changed if n not in explicit]
            if adopted:
                print(f'[resume] schedule args restored from checkpoint: {adopted}')
            if kept:
                print(f'[resume] keeping your explicit override(s) {kept} - the resumed run therefore follows a different schedule than the one it continues')
        start_step = int(resume_ckpt.get('step', 0))
        print(f'[resume] {args.resume} -> step {start_step}')
        # Restore the stochastic state before the first post-resume batch, so
        # dropout and the Python-side shuffle buffer continue where the saved
        # run stopped instead of restarting from --seed.
        rng_state = resume_ckpt.get('rng_state')
        if rng_state is not None:
            torch.set_rng_state(rng_state.get('torch_cpu', torch.get_rng_state()))
            if rng_state.get('torch_cuda') is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
            if rng_state.get('python') is not None:
                random.setstate(rng_state['python'])
            if rng_state.get('numpy') is not None:
                try:
                    import numpy as _np
                except ImportError:
                    pass
                else:
                    _np.random.set_state(rng_state['numpy'])
            print('[resume] restored RNG states (torch / cuda / python / numpy) - dropout and any stochastic step continues as if uninterrupted')
        else:
            print('[resume] checkpoint predates RNG-state saving: RNGs restart from --seed, so a run with dropout>0 will not bitwise-continue the original stream')
        # The loader is a forward-only stream and does not persist its cursor:
        # without skipping, a resumed run silently retrains on the corpus from
        # the start. `data_position` counts every stream batch consumed (one
        # per gradient-accumulation micro-step), including non-finite steps.
        saved_position = resume_ckpt.get('data_position')
        if saved_position is None:
            print('[resume] checkpoint predates data-position tracking: the stream restarts from the beginning and REPLAYS training data - finish the run with this caveat, or restart from a newer checkpoint')
        elif args.resume_skip_data:
            print(f'[resume] --resume-skip-data: NOT skipping the {saved_position} batches the checkpoint already consumed - the resumed run replays training data (accepted)')
        else:
            print(f'[resume] fast-forwarding the dataloader past {saved_position} already-consumed batches (same streaming order, so the resumed run sees fresh data)...')
            t_skip = time.time()
            for i in range(int(saved_position)):
                try:
                    next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    next(data_iter)
                if (i + 1) % 1000 == 0:
                    print(f'  [skip] {i + 1}/{saved_position} batches ({time.time() - t_skip:.0f}s)', flush=True)
            batches_consumed = int(saved_position)
            print(f'[resume] data position restored: the next batch is #{int(saved_position) + 1} in the stream')
        # The best-validation tracker is run state like the step counter and
        # the loss-spike baseline: restarting it at `inf` would let the first
        # post-resume validation unconditionally overwrite a possibly better
        # `ckpt_best.pt`, which is the checkpoint the final test evaluation
        # restores. Seed it from the historical best instead.
        best_path = Path(args.output_dir) / 'ckpt_best.pt'
        best_candidates = []
        if resume_ckpt.get('val_loss') is not None:
            best_candidates.append({'val_loss': float(resume_ckpt['val_loss']),
                                    'step': int(resume_ckpt.get('step', 0))})
        if best_path.exists() and Path(args.resume).resolve() != best_path.resolve():
            # Only the two scalars drive the tracker; loading the whole best
            # checkpoint (model + AdamW state) onto the GPU for them would
            # keep another full training state alive for the rest of the run.
            prior = _load_checkpoint_cpu(best_path)
            best_candidates.append({'val_loss': float(prior['val_loss']),
                                    'step': int(prior.get('step', 0))})
            del prior
        if best_candidates:
            prior_best = min(best_candidates, key=lambda c: c['val_loss'])
            best_val_loss = float(prior_best['val_loss'])
            best_val_step = int(prior_best.get('step', 0))
            print(f'[resume] best-validation tracker seeded with val_loss={best_val_loss:.4f}@{best_val_step} from the previous best checkpoint')
    opt = build_optimizer(model, args.lr, args.weight_decay, args.beta1, args.beta2)
    sched = LambdaLR(opt, build_lr_lambda(args))
    if resume_ckpt is not None:
        load_model_state_dict(model, resume_ckpt['model'])
        if 'optimizer' in resume_ckpt:
            opt.load_state_dict(resume_ckpt['optimizer'])
        if 'scheduler' in resume_ckpt:
            sched.load_state_dict(resume_ckpt['scheduler'])
        apply_explicit_optimizer_args(opt, sched, args, explicit)
        # Everything heavy has been copied into the live objects; drop the
        # checkpoint's copies. `val_loss` / `step` stay: they are scalars the
        # tracker reads.
        resume_ckpt.pop('model', None)
        resume_ckpt.pop('optimizer', None)
        resume_ckpt.pop('scheduler', None)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter(rel_threshold=0.2, abs_threshold=0.05)
    if resume_ckpt is not None:
        if loss_spike.load_state_dict(resume_ckpt.get('loss_spike')):
            print(f'[resume] restored loss-spike counter (n_spikes={loss_spike.n_spikes})')
        elif 'n_loss_spikes' in resume_ckpt:
            loss_spike.n_spikes = int(resume_ckpt['n_loss_spikes'] or 0)
            print(f'[resume] restored loss-spike counter from n_loss_spikes (n_spikes={loss_spike.n_spikes}, EMA warm restarted)')
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = Path(args.output_dir) / 'train.jsonl'
    log_path.touch()
    # One line-buffered handle for the whole run: each written row stays
    # visible to tail -f / crash inspection.
    log_file = open(log_path, 'a', buffering=1)
    manifest_path = Path(args.output_dir) / 'run_manifest.json'
    run_manifest: dict = {}
    run_finished = False
    step = 0
    try:
        run_manifest = collect_run_metadata(seed=args.seed)
        run_manifest['hyperparameters'] = vars(args)
        run_manifest['status'] = 'running'
        if args.hypothesis:
            run_manifest['hypothesis'] = args.hypothesis
            print(f'[hypothesis] {args.hypothesis}')
        run_manifest['data'] = {
            'dataset': args.dataset,
            'dataset_config': args.dataset_config,
            'dataset_revision': args.dataset_revision,
            'tokenizer': args.tokenizer,
            'tokenizer_revision': args.tokenizer_revision,
            'split_reports': data_reports,
            'exclude_hash_file': file_fingerprint(args.exclude_hash_file) if args.exclude_hash_file else None,
            'config_file': file_fingerprint(args.config) if args.config else None,
        }
        if sanity_report_path is not None:
            run_manifest['overfit_sanity_check'] = {'enabled': True, **file_fingerprint(sanity_report_path)}
        else:
            run_manifest['overfit_sanity_check'] = {'enabled': False}
        if resume_ckpt is not None:
            run_manifest['resumed_from'] = str(args.resume)
        save_run_manifest(manifest_path, run_manifest)
        provenance = {
            'git_commit': run_manifest['git']['commit'],
            'git_dirty': run_manifest['git']['dirty'],
            'seed': args.seed,
            'manifest': manifest_path.name,
        }
        print(f'[track] run manifest -> {manifest_path}')
        tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=str(Path(args.output_dir) / 'tb'))
        except Exception:
            pass
        if args.wandb_project:
            import wandb
            wandb.init(project=args.wandb_project, name=args.run_name or args.norm, config=vars(args))
        step = start_step
        t0 = time.time()
        stats_windowed = False
        # Curve-monitoring state: `train_window` is the mean training loss over the
        # step interval since the last validation pass; together with the held-out
        # points it forms the (train, val) curves the diagnosis reads. The final
        # number alone is not a verdict.
        train_window: list = []
        curve_train: list = []
        curve_val: list = []
        curve_steps: list = []
        diagnosis_history: list = []
        last_diagnosis = None
        n_nonfinite_losses = 0
        nonfinite_streak = 0

        def _handle_nonfinite(step: int, x_bad, y_bad, bad_value: float) -> None:
            """Ordered NaN triage: data -> fp32 math -> precision -> gradients."""
            nonlocal n_nonfinite_losses, nonfinite_streak
            n_nonfinite_losses += 1
            nonfinite_streak += 1
            lr_now = opt.param_groups[0]['lr']
            triage = triage_nonfinite(
                model, x_bad, y_bad,
                autocast_dtype=dtype if (args.bf16 and device == 'cuda') else None,
                device_type=device,
                lr=lr_now, grad_clip=args.grad_clip,
            )
            print(f'[nan] step={step + 1} loss={bad_value} -> triage stage={triage["stage"]}: {triage["diagnosis"]}', flush=True)
            for line in triage['recommendations']:
                print(f'[nan]   fix: {line}')
            opt.zero_grad(set_to_none=True)
            log_file.write(json.dumps({
                'step': step + 1, 'event': 'nonfinite_loss', 'loss': bad_value,
                'lr': lr_now, 'triage_stage': triage['stage'],
                'triage_diagnosis': triage['diagnosis'],
                'triage_evidence': triage['evidence'],
                'n_nonfinite_losses': n_nonfinite_losses,
            }, default=str) + '\n')
            log_file.flush()
            if args.nonfinite_policy == 'abort' or nonfinite_streak >= 5:
                raise SystemExit(
                    f'[nan] aborting at step {step + 1}: loss is non-finite and the triage '
                    f'localized the first failure to stage "{triage["stage"]}". '
                    f'Fix it (or raise --grad-clip / lower --lr) before resuming from the '
                    f'last checkpoint; see the [nan] recommendations above.'
                )
            print(f'[nan] policy=skip: discarding step {step + 1} without an optimizer step', flush=True)

        while step < args.max_steps:
            opt.zero_grad(set_to_none=True)
            attach_stats = (step + 1) % args.stats_every == 0
            if attach_stats:
                act_stats.attach(reset=not stats_windowed)
                stats_windowed = True
            micro_losses: list = []
            micro_xy: list = []
            nonfinite_batch = None
            for _ in range(args.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)
                # Count every consumed stream batch (including the ones of a
                # step later discarded for a non-finite loss): resume must
                # skip exactly these.
                batches_consumed += 1
                x, y = batch
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with ctx:
                    loss = model(x, y, return_logits=False)[1]
                # Keep the device scalar (no clone): the scaling below is
                # out-of-place, so the recorded per-micro value is never mutated.
                micro_losses.append(loss.detach())
                micro_xy.append((x, y))
                if args.grad_accum > 1:
                    # Out-of-place: `loss` is still needed for the recorded
                    # per-micro value above. With accumulation 1 the op is
                    # skipped, so the reported loss is the raw CE value.
                    loss = loss / args.grad_accum
                loss.backward()
            # Device-side group reduction: `mean` propagates NaN/Inf, so a
            # finite group mean certifies every micro. A non-finite mean is the
            # rare path and uses the `.tolist()` to locate the bad micro for
            # triage. A non-finite loss cannot be detected before its
            # own backward without inspecting each micro separately, so a bad
            # micro is backpropped and the step is discarded wholesale by
            # _handle_nonfinite (which zeroes the gradients); the optimizer
            # trajectory is unchanged.
            loss_vec = torch.stack(micro_losses)
            loss_val = float(loss_vec.mean()) if args.grad_accum > 1 else float(loss_vec[0])
            if not math.isfinite(loss_val):
                micro_vals = loss_vec.tolist()
                bad_idx = next((i for i, v in enumerate(micro_vals) if not math.isfinite(v)), None)
                if bad_idx is None:
                    # The group mean is non-finite although every micro is finite
                    # (an overflowed sum): treat the first micro as the bad case.
                    bad_idx = 0
                    micro_vals[0] = loss_val
                nonfinite_batch = (micro_xy[bad_idx][0], micro_xy[bad_idx][1], micro_vals[bad_idx])
            if attach_stats:
                act_stats.detach()
                clear_logit_scratch_cache()
            if nonfinite_batch is not None:
                _handle_nonfinite(step, *nonfinite_batch)
                continue
            nonfinite_streak = 0
            last_train_loss = loss_val
            # Grad-norm clipping; the value is read only when logging.
            grad_norm_t = None
            if args.grad_clip > 0:
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            clear_cast_weight_cache()
            loss_spike.update(loss_val)
            if args.val_every > 0:
                # The window is only consumed by the validation block below; with
                # --val-every 0 nothing ever resets it, so appending would grow it
                # for the entire run.
                train_window.append(loss_val)
            step += 1
            if args.val_every > 0 and step % args.val_every == 0:
                if val_loader is None:
                    val_loader = open_data_split('validation', args)
                    rval = getattr(val_loader, 'split_report', None)
                    if rval:
                        data_reports['validation'] = rval
                        print(f'[data] validation split: {rval}')
                val_batch_losses: list = []
                val_loss = evaluate_heldout(model, val_loader, val_state, args.val_batches, ctx, device, collect=val_batch_losses)
                if val_loss is None:
                    # The loader yielded zero batches: the held-out slice is empty.
                    # Silently continuing would train with no selection signal (and
                    # save no best checkpoint), so fail before continuing a run
                    # whose model-selection protocol is already void.
                    raise SystemExit(
                        f'[val] the validation split produced no batches, so checkpoint selection is '
                        f'impossible; refusing to continue without a held-out signal. Likely causes: '
                        f'--num-documents={args.num_documents} is larger than the actual stream (the '
                        f'chronological boundary points past its end), the dataset/tokenizer revision '
                        f'changed, or the split fractions leave validation empty. Verify the split '
                        f'layout with scripts/audit_data.py before re-running.')
                if val_loss is not None:
                    last_val_loss = val_loss
                    last_val_step = step
                    last_val_batches = val_batch_losses
                    improved = val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss
                        best_val_step = step
                        val_checks_without_improvement = 0
                    else:
                        val_checks_without_improvement += 1
                    print(f'[val] step={step} val_loss={val_loss:.4f} best={best_val_loss:.4f}@{best_val_step} (no improvement x{val_checks_without_improvement})', flush=True)
                    if tb:
                        tb.add_scalar('val/loss', val_loss, step)
                    if improved:
                        ckpt_best = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'loss_spike': loss_spike.state_dict(), 'val_loss': val_loss, 'data_position': batches_consumed, 'rng_state': _snapshot_rng_state(), 'provenance': provenance}
                        path = Path(args.output_dir) / 'ckpt_best.pt'
                        _atomic_torch_save(ckpt_best, path)
                        print(f'[ckpt] saved {path} (val_loss={val_loss:.4f})')
                    # Curve diagnosis: pair the held-out point with the mean
                    # training loss over the interval it covers, then classify the
                    # regime (overfit / underfit / diverging / healthy) and print
                    # the concrete remedy.
                    train_point = (sum(train_window) / len(train_window)) if train_window else loss_val
                    train_window = []
                    curve_train.append(train_point)
                    curve_val.append(val_loss)
                    curve_steps.append(step)
                    last_diagnosis = diagnose_train_val_curve(
                        curve_train, curve_val, train_steps=curve_steps, val_steps=curve_steps,
                        log_vocab_size=math.log(model.cfg.vocab_size),
                    )
                    diagnosis_history.append({
                        'step': step,
                        'classification': last_diagnosis['classification'],
                        'train_trend': last_diagnosis['train_trend'],
                        'val_trend': last_diagnosis['val_trend'],
                        'train_relative_change': last_diagnosis['train_relative_change'],
                        'val_relative_change': last_diagnosis['val_relative_change'],
                        'generalization_gap': last_diagnosis['generalization_gap'],
                        'recommendations': last_diagnosis['recommendations'],
                    })
                    print(f"[curve] step={step} train_window={train_point:.4f} val={val_loss:.4f} "
                          f"-> {last_diagnosis['classification']} "
                          f"(train {last_diagnosis['train_trend']}, val {last_diagnosis['val_trend']})", flush=True)
                    if last_diagnosis['classification'] not in ('healthy', 'insufficient_data'):
                        for line in last_diagnosis['recommendations']:
                            print(f'[curve]   action: {line}')
                    if tb:
                        tb.add_scalar('curve/train_window_loss', train_point, step)
                        tb.add_scalar('curve/generalization_gap', last_diagnosis['generalization_gap'] or 0.0, step)
            if step % args.log_every == 0:
                lr = opt.param_groups[0]['lr']
                grad_norm = float(grad_norm_t) if grad_norm_t is not None else None
                line = f'step={step:>6d} loss={loss_val:8.4f} lr={lr:.2e}'
                if grad_norm is not None:
                    line += f' grad_norm={grad_norm:.3f}'
                print(line, flush=True)
                row = {'step': step, 'loss': loss_val, 'lr': lr, 'n_loss_spikes': loss_spike.n_spikes}
                if grad_norm is not None:
                    row['grad_norm'] = grad_norm
                if last_val_loss is not None:
                    row['val_loss'] = last_val_loss
                if last_diagnosis is not None:
                    row['curve_classification'] = last_diagnosis['classification']
                    row['curve_train_trend'] = last_diagnosis['train_trend']
                    row['curve_val_trend'] = last_diagnosis['val_trend']
                log_file.write(json.dumps(row) + '\n')
                if tb:
                    tb.add_scalar('train/loss', loss_val, step)
                    tb.add_scalar('train/lr', lr, step)
                    tb.add_scalar('train/n_loss_spikes', loss_spike.n_spikes, step)
                    if grad_norm is not None:
                        tb.add_scalar('train/grad_norm_pre_clip', grad_norm, step)
                if args.wandb_project:
                    import wandb
                    wandb.log(row, step=step)
            if step % args.stats_every == 0 and step > 0:
                stats = act_stats.summary()
                print(f"[stats] {stats.get('__summary__', {})}")
                if tb:
                    s = stats.get('__summary__', {})
                    for k, v in s.items():
                        tb.add_scalar(f'stats/{k}', v, step)
            if step % args.save_every == 0 and step > 0:
                ckpt = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'loss_spike': loss_spike.state_dict(), 'data_position': batches_consumed, 'rng_state': _snapshot_rng_state(), 'provenance': provenance}
                path = Path(args.output_dir) / f'ckpt_{step:06d}.pt'
                _atomic_torch_save(ckpt, path)
                print(f'[ckpt] saved {path}')
            if args.early_stop_patience > 0 and val_loader is not None and val_checks_without_improvement >= args.early_stop_patience:
                print(f'[early-stop] validation did not improve for {val_checks_without_improvement} checks (best val_loss={best_val_loss:.4f}@{best_val_step}); stopping at step {step}')
                break
        ckpt = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'n_loss_spikes': loss_spike.n_spikes, 'loss_spike': loss_spike.state_dict(), 'data_position': batches_consumed, 'rng_state': _snapshot_rng_state(), 'provenance': provenance}
        path = Path(args.output_dir) / 'ckpt_final.pt'
        _atomic_torch_save(ckpt, path)
        print(f'[done] saved final ckpt to {path}, time = {time.time() - t0:.1f}s')
        run_manifest['data']['split_reports'] = data_reports
        run_manifest['status'] = 'finished'
        run_manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        flops = analytic_flops_per_token(model.cfg, seq_len=args.seq_len)
        final_val_stats = mean_ci(last_val_batches) if last_val_batches else None
        run_manifest['results'] = {
            'final_step': step,
            'last_train_loss': last_train_loss,
            # `best_val_loss` is the minimum over the validation passes: it is the
            # checkpoint-selection statistic, so as a reported generalization
            # estimate it is optimistically biased (the minimum of noisy draws).
            # `final_val_loss` is the last held-out pass at the fixed training-step
            # count and is the number that makes two arms comparable.
            'best_val_loss': best_val_loss if val_loader is not None else None,
            'best_val_step': best_val_step if val_loader is not None else None,
            'final_val_loss': last_val_loss if val_loader is not None else None,
            'final_val_step': last_val_step if val_loader is not None else None,
            # The final pass is a sample mean over batches: its standard error and
            # interval are the resolution of the number quoted above. A delta
            # smaller than the stderr is not evidence of a difference.
            'final_val_loss_n_batches': len(last_val_batches),
            'final_val_loss_std': final_val_stats['std'] if final_val_stats else None,
            'final_val_loss_stderr': final_val_stats['stderr'] if final_val_stats else None,
            'final_val_loss_ci_level': final_val_stats['ci_level'] if final_val_stats else None,
            'final_val_loss_ci_low': final_val_stats['ci_low'] if final_val_stats else None,
            'final_val_loss_ci_high': final_val_stats['ci_high'] if final_val_stats else None,
            'n_loss_spikes': loss_spike.n_spikes,
            'n_nonfinite_losses': n_nonfinite_losses,
            'elapsed_s': time.time() - t0,
            # Curve monitoring: the full (train window, val) pairs and the latest
            # classification, so a reviewer can read the regime, not just the end.
            'curve_train_window_loss': curve_train,
            'curve_val_loss': curve_val,
            'curve_steps': curve_steps,
            'curve_diagnosis': last_diagnosis,
            'curve_diagnosis_history': diagnosis_history,
            # Efficiency accounting travels with the accuracy numbers: parameters
            # by role and the analytic matmul FLOPs per token.
            'n_params': n_params,
            'n_trainable': n_trainable,
            # Shape accounting: the depth sweep scales dim/heads with depth to hold
            # parameters ~constant, so "depth" is really a depth+width trade-off
            # there. Recording the shape next to the loss keeps that covariance
            # visible in every aggregate instead of inferable only from launcher
            # env vars.
            'n_layers': int(model.cfg.n_layers),
            'dim': int(model.cfg.dim),
            'n_heads': int(model.cfg.n_heads),
            'parameter_breakdown': parameter_breakdown(model),
            'flops_per_token_forward': flops['forward_per_token'],
            'flops_per_token_training_est': flops['training_per_token'],
        }
        # Point the manifest at the files this run actually produced: the log
        # (content hash, it is small) plus the checkpoints (size+mtime). A
        # result is then traceable to the exact curve/weights behind it.
        run_manifest['artifacts'] = {
            'train_jsonl': file_fingerprint(log_path),
            'checkpoint_final': artifact_record(Path(args.output_dir) / 'ckpt_final.pt'),
        }
        best_ckpt_path = Path(args.output_dir) / 'ckpt_best.pt'
        if best_ckpt_path.exists():
            run_manifest['artifacts']['checkpoint_best'] = artifact_record(best_ckpt_path)
        if n_nonfinite_losses:
            print(f'[warn] {n_nonfinite_losses} non-finite loss step(s) were handled by '
                  f'--nonfinite-policy {args.nonfinite_policy}; see train.jsonl event rows')
        save_run_manifest(manifest_path, run_manifest)
        # Training (and its manifest) is complete from here on: an optional
        # sealed-test evaluation or a tensorboard close failure below must not
        # make the manifest look like an interrupted training run.
        run_finished = True
        if args.final_test_eval:
            # The test split is opened exactly once, after training is over, and
            # only the checkpoint selected on the validation split is scored.
            # Re-running the evaluation in the same output dir would let the test
            # score steer the next attempt, i.e. turn the sealed set into an
            # implicit validation set, so it is refused unless explicitly allowed.
            out_path = Path(args.output_dir) / 'test_results.json'
            if out_path.exists() and (not args.allow_test_rerun):
                raise SystemExit(
                    f'[test] {out_path} already exists: the sealed test split has already been '
                    'evaluated once for this run, and scoring it again would turn it into an '
                    'implicit validation set. Pass --allow-test-rerun only if you accept that '
                    'the seal is broken.')
            if args.val_every <= 0:
                print('[test] WARNING: --val-every 0 disables validation-based selection; '
                      'the sealed test split reports the final model only and must never be '
                      'used to pick hyperparameters.', file=sys.stderr)
            test_loader = open_data_split('test', args)
            rtest = getattr(test_loader, 'split_report', None)
            if rtest:
                data_reports['test'] = rtest
                print(f'[data] test split: {rtest}')
            best_path = Path(args.output_dir) / 'ckpt_best.pt'
            if val_loader is not None and best_path.exists():
                best = _load_checkpoint_cpu(best_path)
                load_model_state_dict(model, best['model'])
                best_step = int(best.get('step', 0))
                best_val = float(best.get('val_loss', float('nan')))
                # The weights are copied; the optimizer state in the file is
                # not needed for the test pass.
                del best
                print(f"[test] restored best-validation checkpoint (step {best_step}, val_loss={best_val:.4f}) before the one-shot test evaluation")
            test_state = {'iter': iter(test_loader)}
            test_batch_losses: list = []
            test_loss = evaluate_heldout(model, test_loader, test_state, args.test_batches, ctx, device, collect=test_batch_losses)
            if test_loss is None:
                # Writing test_results.json now would consume the one-shot seal
                # without any measurement: the next run would refuse to evaluate
                # the test split again (without --allow-test-rerun) even though no
                # number was ever produced. Refuse instead, and leave the seal
                # intact so the fixed data configuration can be re-evaluated once.
                raise SystemExit(
                    '[test] the sealed test split produced no batches; refusing to write '
                    'test_results.json so the one-shot seal is not consumed by a void '
                    'evaluation. Check that --num-documents matches the dataset and that the '
                    'dataset/tokenizer revisions still resolve, then re-run.')
            if len(test_batch_losses) < args.test_batches:
                print(f'[test] WARNING: only {len(test_batch_losses)} of the requested '
                      f'{args.test_batches} test batches were available; the mean and interval '
                      f'rest on {len(test_batch_losses)} batch(es).', file=sys.stderr)
            test_stats = mean_ci(test_batch_losses)
            # The per-batch losses are kept, not just their mean: they are the
            # resolution of the headline number and, because the held-out loader
            # never shuffles, two runs sharing `pairing` below were scored on the
            # same sealed segments - which is what makes their difference a paired
            # sample (`scripts/compare_sealed_tests.py` enforces exactly this).
            pairing = {
                'dataset': args.dataset, 'dataset_config': args.dataset_config,
                'dataset_revision': args.dataset_revision,
                'tokenizer': args.tokenizer, 'tokenizer_revision': args.tokenizer_revision,
                'num_documents': args.num_documents,
                'val_frac': args.val_frac, 'test_frac': args.test_frac,
                'dedup': args.dedup,
                'min_doc_chars': args.min_doc_chars, 'max_doc_chars': args.max_doc_chars,
                'exclude_hash_file': args.exclude_hash_file,
                'batch_size': args.batch_size, 'seq_len': args.seq_len,
            }
            results = {'step': step, 'seed': args.seed, 'model': args.model, 'norm': args.norm, 'best_val_step': best_val_step if val_loader is not None else None, 'best_val_loss': best_val_loss if val_loader is not None else None, 'final_val_loss': last_val_loss if val_loader is not None else None, 'final_val_step': last_val_step if val_loader is not None else None, 'test_loss': test_loss, 'test_batches': args.test_batches, 'test_batches_seen': len(test_batch_losses), 'test_batch_losses': test_batch_losses, 'test_loss_std': test_stats['std'], 'test_loss_stderr': test_stats['stderr'], 'test_loss_ci_level': test_stats['ci_level'], 'test_loss_ci_low': test_stats['ci_low'], 'test_loss_ci_high': test_stats['ci_high'], 'pairing': pairing, 'num_documents': args.num_documents, 'val_frac': args.val_frac, 'test_frac': args.test_frac, 'exclude_hash_file': args.exclude_hash_file}
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            run_manifest.setdefault('artifacts', {})['test_results'] = file_fingerprint(out_path)
            run_manifest['data']['split_reports'] = data_reports
            run_manifest['results']['test_loss'] = test_loss
            run_manifest['results']['test_loss_n_batches'] = len(test_batch_losses)
            run_manifest['results']['test_loss_stderr'] = test_stats['stderr']
            run_manifest['results']['test_loss_ci_level'] = test_stats['ci_level']
            run_manifest['results']['test_loss_ci_low'] = test_stats['ci_low']
            run_manifest['results']['test_loss_ci_high'] = test_stats['ci_high']
            save_run_manifest(manifest_path, run_manifest)
            ci = ('no interval (a single batch has no degrees of freedom)' if test_stats['ci_low'] is None
                  else f"{100 * test_stats['ci_level']:.0f}% CI [{test_stats['ci_low']:.4f}, {test_stats['ci_high']:.4f}]")
            print(f'[test] sealed test split evaluated once: test_loss={test_loss:.4f} '
                  f'(n={len(test_batch_losses)} batches, stderr={test_stats["stderr"]:.4f}, {ci}) -> {out_path}')
        if tb:
            tb.close()
        if args.wandb_project:
            import wandb
            wandb.finish()
        run_finished = True
    finally:
        log_file.close()
        if not run_finished and run_manifest:
            # A crash / NaN abort / external kill must not leave a manifest
            # that looks like a completed run: mark it and persist whatever
            # was known at the time (the step-by-step curve is already in
            # train.jsonl; these fields make the manifest point at it).
            run_manifest['status'] = 'interrupted'
            run_manifest['interrupted_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            run_manifest['last_step'] = int(step)
            run_manifest['last_train_loss'] = last_train_loss
            run_manifest['last_val_loss'] = last_val_loss
            # The curve is the artifact a reader needs for an interrupted run;
            # fingerprint whatever made it to disk.
            run_manifest.setdefault('artifacts', {})['train_jsonl'] = file_fingerprint(log_path)
            try:
                save_run_manifest(manifest_path, run_manifest)
                print(f'[track] run did not finish: manifest marked interrupted at step {step}', flush=True)
            except OSError:
                pass
    # Drop the per-run scratch caches.
    clear_ce_scratch_cache()
if __name__ == '__main__':
    main()
