from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
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
from adasquash.diagnostics import diagnose_train_val_curve, parameter_finite_report, tensor_finite_report, triage_nonfinite
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.metrics import mean_ci, mean_std_stderr
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache, clear_cast_weight_cache
from adasquash.norms import NORM_REGISTRY
from adasquash.repro import (TRAINING_RECIPE_KEYS, TrainStreamFingerprint,
                             artifact_record, collect_run_metadata,
                             file_fingerprint, finish_metadata, save_run_manifest,
                             set_all_seeds, training_recipe_mismatches)
from scripts.data_pipeline import (build_dataloader, configure_data_options, _is_pinned_revision,
                                   resolve_dataset_revision, resolve_tokenizer_revision)
from scripts.train_config import (_DATA_ARGS, _LEDGER_ARGS,
                                        _MONITOR_ARGS, _RESUME_RUN_ARGS,
                                        _SCHEDULE_ARGS,
                                        _WITHIN_ARM_PROTOCOL_KEYS)

def _ledger_identity(path):
    if not path:
        return None
    fingerprint = file_fingerprint(path)
    return {key: fingerprint.get(key) for key in ('path', 'exists', 'sha256', 'bytes')}

def _data_ledger_fingerprints(args) -> dict:
    """Content fingerprints of the cross-split ledgers a run depends on.

    The checkpoint stores these so a resume can prove it is continuing on the
    same held-out documents: a ledger file replaced *at the same path* between
    runs would otherwise silently change which held-out documents are dropped
    (and therefore which documents the validation/test numbers describe)
    while every recorded argument still looks identical.
    """
    return {name: _ledger_identity(getattr(args, name, None)) for name in _LEDGER_ARGS}

def restore_data_args(args, ckpt_args: dict, explicit: frozenset=frozenset()) -> list:
    changed = []
    for name in _DATA_ARGS:
        if name not in ckpt_args:
            continue
        current = getattr(args, name, None)
        if (name in explicit and current != ckpt_args[name]
                and name in ('dataset_revision', 'tokenizer_revision')):
            # The checkpoint records the concrete commit, while the user may
            # legitimately repeat the mutable branch/tag they originally
            # requested. Resolve the request first; if it still names the
            # checkpointed commit this is not a data change.
            try:
                resolved = (resolve_dataset_revision(args.dataset, current)
                            if name == 'dataset_revision'
                            else resolve_tokenizer_revision(args.tokenizer, current))
            except Exception:
                resolved = None
            if resolved == ckpt_args[name]:
                setattr(args, name, resolved)
                current = resolved
        if current != ckpt_args[name]:
            changed.append(name)
        if name in explicit:
            continue
        setattr(args, name, ckpt_args[name])
    return changed

def run_error_analysis(args, ckpt_path: Path, output_dir: Path) -> dict:
    """Run the bad-case analysis as a reproducible post-training step.

    A final loss says *how often* the model is wrong; the error taxonomy and
    printed examples say *why*, which is where the next hypothesis comes from.
    The data flags (including pinned revisions/ledgers) are forwarded exactly
    as the run used them, so the analyzed held-out documents are the ones the
    checkpoint was selected on. Failure is reported in the manifest, not
    raised: the training result is already complete and must remain readable.
    """
    out_path = output_dir / 'error_analysis_validation.json'
    if not args.dataset or not args.tokenizer:
        reason = ('the run has no --dataset/--tokenizer to reproduce the held-out '
                  'documents from; run scripts/analyze_errors.py manually with the '
                  'matching data flags if this was intentional')
        print(f'[errors][warn] skipping bad-case analysis: {reason}', file=sys.stderr)
        return {'ran': False, 'returncode': None, 'checkpoint': str(ckpt_path),
                'split': 'validation', 'artifact': None, 'reason': reason}
    if not Path(ckpt_path).exists():
        reason = f'checkpoint {ckpt_path} does not exist'
        print(f'[errors][warn] skipping bad-case analysis: {reason}', file=sys.stderr)
        return {'ran': False, 'returncode': None, 'checkpoint': str(ckpt_path),
                'split': 'validation', 'artifact': None, 'reason': reason}
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'analyze_errors.py'),
        '--ckpt', str(ckpt_path), '--corpus', 'dataset', '--split', 'validation',
        '--dataset', str(args.dataset),
        '--tokenizer', str(args.tokenizer),
        '--val-frac', str(args.val_frac), '--test-frac', str(args.test_frac),
        '--min-doc-chars', str(args.min_doc_chars),
        '--max-doc-chars', str(args.max_doc_chars),
        '--num-workers', str(args.num_workers),
        '--seq-len', str(args.seq_len),
        '--batch-size', str(args.batch_size),
        '--n-batches', str(args.error_analysis_batches),
        '--max-examples', str(args.error_analysis_examples),
        '--confidence', str(args.error_analysis_confidence),
        '--seed', str(args.seed), '--output', str(out_path),
    ]
    for flag, value in (
        ('--dataset-config', args.dataset_config),
        ('--dataset-revision', args.dataset_revision),
        ('--tokenizer-revision', args.tokenizer_revision),
        ('--num-documents', args.num_documents),
        ('--exclude-hash-file', args.exclude_hash_file),
        ('--exclude-test-hash-file', args.exclude_test_hash_file),
        ('--group-key', args.group_key),
        ('--exclude-group-file', args.exclude_group_file),
        ('--exclude-test-group-file', args.exclude_test_group_file),
    ):
        if value is not None:
            cmd += [flag, str(value)]
    if not args.dedup:
        cmd.append('--no-dedup')
    print(f'[errors] bad-case analysis on {ckpt_path.name} over the validation split '
          f'({args.error_analysis_batches} batches) -> {out_path}', flush=True)
    # Inherit the caller's working directory: cmd contains paths the parent
    # resolved relative to it (checkpoint/output/ledgers). Running the child
    # from ROOT made a relative --output-dir point at a different place and
    # the artifact fingerprint then reported a missing file.
    # A failed child must not leave the previous run's JSON to be fingerprinted
    # as this run's artifact. Remove it before the spawn and require a clean
    # exit before accepting whatever the child wrote.
    try:
        out_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        print(f'[errors][warn] could not remove stale {out_path}: {exc}',
              file=sys.stderr)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f'[errors][warn] analyze_errors.py exited rc={rc}; no error_analysis '
              f'artifact was recorded', file=sys.stderr)
    artifact = (file_fingerprint(out_path)
                if rc == 0 and out_path.exists() else None)
    return {
        'ran': rc == 0,
        'returncode': rc,
        'checkpoint': str(ckpt_path),
        'split': 'validation',
        'artifact': artifact,
        'command': cmd,
    }

def _heldout_batch_fingerprint(x, y=None) -> bytes:
    """Content digest of one held-out batch's tokens.

    The held-out loaders are deterministic and never shuffle, so a repeated
    packed window produces exactly the same tokens. The target is hashed too:
    a window's final label is not part of its input, so two different held-out
    windows can share `x` and differ only in that last target. Fingerprinting
    the full batch therefore lets `evaluate_heldout` count *distinct* held-out
    windows instead of trusting the requested batch count; hashing the small
    host tensors is negligible next to a model forward pass.
    """
    hasher = hashlib.blake2b(digest_size=16)
    for name, tensor in (('x', x), ('y', y)):
        if tensor is None:
            continue
        hasher.update(name.encode('ascii'))
        hasher.update(str(tuple(tensor.shape)).encode('ascii'))
        hasher.update(str(tensor.dtype).encode('ascii'))
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.digest()

def evaluate_heldout(model, loader, state: dict, n_batches: int, ctx, device, collect: list | None = None, cycle: bool = True) -> float | None:
    """Mean loss over held-out batches; `state['iter']` carries the position.

    Runs under `inference_mode` and eval mode, so no gradient ever flows from
    a validation or test batch into the parameters. When `collect` is given,
    every batch's loss is also appended to it: the mean alone cannot say how
    noisy the held-out estimate is, nor whether two arms were scored on the
    same held-out segments (the pairing a paired test needs). The accumulator
    is untouched by `collect`, so the returned mean is bit-identical either
    way. `cycle` restarts a validation loader that has fewer batches than
    requested; the one-shot sealed-test pass disables it so the same batch is
    never counted twice. `state['batches_consumed']` counts every fetched
    batch across passes, so a checkpoint can restore this cursor instead of
    silently scoring an earlier held-out slice after a resume.

    The pass also records its own data budget in `state`:
    `last_eval_requested_batches`, `last_eval_unique_batches`,
    `last_eval_cycled`, `last_eval_reused` and `last_eval_shortfall`. A
    validation loader with fewer distinct batches than requested is cycled,
    so its collected losses are not independent draws; the training loop uses
    these fields to expose that pseudo-replication instead of averaging it
    away.
    """
    # With dropout at its default 0 the module-mode switch changes no
    # numerics, so skip it; inference_mode already blocks grads. `training`
    # does more than dropout, though: gradient checkpointing keys its
    # recompute branch on it, so the switch must stay when that is on.
    cfg = getattr(model, 'cfg', None)
    toggle_mode = float(getattr(cfg, 'dropout', 0.0)) > 0.0 or bool(getattr(cfg, 'gradient_checkpointing', False))
    was_training = model.training
    if toggle_mode:
        model.eval()
    it = state.get('iter')
    total = None
    n = 0
    requested = max(0, int(n_batches))
    seen = 0
    seen_batches: set = set()
    repeated_batches = 0
    state['last_eval_requested_batches'] = requested
    state['last_eval_unique_batches'] = None
    state['last_eval_cycled'] = False
    state['last_eval_reused'] = False
    state['last_eval_shortfall'] = 0
    terms: list = [] if collect is not None else None
    # A previous non-finite hold must not be attributed to this pass.
    state.pop('nonfinite_batch', None)
    with torch.inference_mode():
        for _ in range(requested):
            if it is None:
                it = iter(loader)
                state['iter'] = it
            try:
                batch = next(it)
            except StopIteration:
                if not cycle:
                    state['last_eval_shortfall'] = requested - seen
                    break
                state['last_eval_cycled'] = True
                it = iter(loader)
                state['iter'] = it
                try:
                    batch = next(it)
                except StopIteration:
                    break
            seen += 1
            state['batches_consumed'] = int(state.get('batches_consumed', 0)) + 1
            x, y = batch
            # Fingerprint on the loader's (host) tensor before the device
            # transfer: this avoids a per-batch device->host copy and keeps the
            # validation data budget independent of the compute device.
            fingerprint = _heldout_batch_fingerprint(x, y)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if fingerprint in seen_batches:
                repeated_batches += 1
            else:
                seen_batches.add(fingerprint)
            with ctx:
                loss = model(x, y, return_logits=False)[1]
            # Accumulate in a device fp64 scalar; fp64 keeps the sequential
            # summation bit-identical to host float accumulation.
            term = loss.detach().double()
            if not bool(torch.isfinite(term).item()) and 'nonfinite_batch' not in state:
                # Keep the first offending batch/point so the caller can run
                # the ordered data -> math -> precision -> gradients triage
                # on the *held-out* input instead of only seeing a NaN mean.
                state['nonfinite_batch'] = (x, y, float(term))
            if terms is not None:
                terms.append(term)
            total = term.clone() if total is None else total.add_(term)
            n += 1
    if toggle_mode:
        model.train(was_training)
    state['last_eval_unique_batches'] = len(seen_batches)
    state['last_eval_reused'] = repeated_batches > 0
    if collect is not None and terms:
        # `.tolist()` yields the same Python doubles in the same batch order,
        # so the returned mean and the collected series are unchanged.
        collect.extend(torch.stack(terms).cpu().tolist())
    return float(total) / n if n else None

def _fast_forward_validation_stream(loader, state: dict) -> None:
    """Restore a resumed run's validation cursor.

    The pipeline exposes a forward-only iterator, so a checkpoint stores the
    number of held-out batches already scored and this helper replays the
    deterministic split up to that point. Without it the post-resume passes
    read earlier documents than an uninterrupted run, which breaks paired
    cross-arm comparisons even though the model weights are restored exactly.
    """
    count = int(state.get('resume_skip') or 0)
    if count <= 0:
        state['resume_skip'] = 0
        return
    print(f'[resume] fast-forwarding the validation stream past {count} already-scored batch(es)...', flush=True)
    it = iter(loader)
    skipped = 0
    while skipped < count:
        try:
            next(it)
            skipped += 1
        except StopIteration:
            it = iter(loader)
            try:
                next(it)
                skipped += 1
            except StopIteration:
                print('[resume][warn] validation split produced no batches while restoring the cursor; '
                      'the next validation pass will fail rather than score an empty slice', flush=True)
                state['iter'] = None
                state['resume_skip'] = 0
                return
    state['iter'] = it
    state['resume_skip'] = 0
    print('[resume] validation stream position restored: the next pass continues at the '
          'same held-out batch an uninterrupted run would have scored', flush=True)


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


def _is_seed_id(value) -> bool:
    """A seed id is an integer; bool/fractional/string values are not."""
    return isinstance(value, int) and not isinstance(value, bool)


def _normalize_seed_ids(seeds) -> list[int]:
    """Materialize unique integer seed ids without coercing malformed values."""
    values = list(seeds)
    if not all(_is_seed_id(seed) for seed in values):
        raise ValueError(f'seeds must be integer ids, got {values!r}')
    return list(dict.fromkeys(int(seed) for seed in values))


def _is_event_count(value) -> bool:
    """A numerical-health counter is a non-negative integer, not NaN/bool/1.5."""
    return (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _is_finite_number(value) -> bool:
    """A real scalar with a finite float image (huge JSON ints are not)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


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

def _manifest_data_version(manifest: dict):
    """Concrete dataset/tokenizer commits a seed actually trained on, if pinned."""
    data = manifest.get('data') if isinstance(manifest, dict) else None
    if not isinstance(data, dict):
        return None
    reports = data.get('split_reports') or {}
    train = reports.get('train') if isinstance(reports, dict) else None
    train = train if isinstance(train, dict) else {}
    dataset_resolved = (train.get('dataset_revision_resolved')
                        or data.get('dataset_revision_resolved'))
    tokenizer_resolved = (train.get('tokenizer_revision_resolved')
                          or data.get('tokenizer_revision_resolved'))
    if not _is_pinned_revision(dataset_resolved):
        return None
    if not _is_pinned_revision(tokenizer_resolved):
        return None
    return {
        'dataset_revision_resolved': dataset_resolved,
        'tokenizer_revision_resolved': tokenizer_resolved,
    }

def _parse_manifest_ledger_fingerprints(manifest: dict):
    """``(fingerprints, problem)`` for a seed manifest's ledger identities.

    A path alone cannot prove that two seeds excluded the same held-out
    documents: a ledger file can be regenerated at the same path between runs.
    ``fingerprints`` is ``None`` when the manifest has no ledger fields at all
    (a legacy record), and ``problem`` names the first malformed record so the
    caller can fail closed instead of treating it as an unknown legacy value.
    """
    data = manifest.get('data') if isinstance(manifest, dict) else None
    if not isinstance(data, dict):
        if isinstance(manifest, dict) and manifest.get('data') is not None:
            return None, 'run_manifest.json data block is not a mapping'
        return None, None
    out = {}
    present = False
    for name in _LEDGER_ARGS:
        if name not in data:
            # A partial legacy record is still comparable as long as every seed
            # omits the same role; an explicit value on one side versus this
            # sentinel on the other is a mismatch.
            out[name] = '<missing>'
            continue
        present = True
        value = data.get(name)
        if value is None:
            out[name] = None
        elif isinstance(value, dict):
            # A partial fingerprint cannot prove content identity. `exists:
            # False` is complete without hashes (there is no file), but an
            # existing file must carry a non-empty sha256 and a byte count.
            path = value.get('path')
            exists = value.get('exists')
            if not isinstance(path, str) or not path:
                return None, f'data.{name} has no valid path'
            if exists is False:
                out[name] = {'path': path, 'exists': False}
            elif exists is True:
                sha = value.get('sha256')
                nbytes = value.get('bytes')
                if (not isinstance(sha, str) or not sha
                        or not isinstance(nbytes, int)
                        or isinstance(nbytes, bool) or nbytes < 0):
                    return None, f'data.{name} has an incomplete file fingerprint'
                out[name] = {'path': path, 'exists': True,
                             'sha256': sha, 'bytes': nbytes}
            else:
                return None, f'data.{name} has no boolean exists flag'
        else:
            return None, f'data.{name} is not null or a fingerprint mapping'
    return (out if present else None), None


def _manifest_ledger_fingerprints(manifest: dict):
    """Content identity of the cross-split ledgers a seed's run manifest records.

    Missing/malformed records return ``None`` so they fail closed rather than
    comparing two unknowns as equal; use `_parse_manifest_ledger_fingerprints`
    when the caller must distinguish a legacy absence from malformed data.
    """
    ledgers, problem = _parse_manifest_ledger_fingerprints(manifest)
    return None if problem is not None else ledgers

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
    batches of a single run. Missing, invalid, and rejected (unfinished or
    mismatched) seed manifests are reported, never silently dropped.
    """
    # A seed names one underlying run. Keep one record per seed even when a
    # direct caller passes a duplicate: reading the same directory twice would
    # otherwise count the same run as independent replication (and could lift
    # a two-seed set over the three-seed certification threshold).
    seeds = _normalize_seed_ids(seeds)
    per_seed = []
    rejections = []
    reference_recipe = None
    reference_context = None
    reference_ledgers = None
    ledger_identity_complete = True
    ledger_missing_seeds = []
    ledger_present_seeds = []
    for seed in seeds:
        run_dir = seed_run_dir(base_dir, seed)
        record = {
            'seed': int(seed), 'run_dir': str(run_dir),
            'final_val_loss': None, 'best_val_loss': None,
            'final_val_step': None, 'best_val_step': None,
            'n_params': None, 'n_layers': None, 'dim': None, 'n_heads': None,
            'n_loss_spikes': None, 'n_nonfinite_losses': None,
            'n_nonfinite_gradients': None, 'n_nonfinite_val_losses': None,
            'curve_classification': None, 'curve_generalization_gap': None,
            'test_loss': None, 'data_version': None, 'data_ledgers': None,
            'train_stream': None,
            'tokens_per_second': None, 'peak_device_memory_mb': None,
        }
        reject_reason = None
        manifest = None
        try:
            manifest = json.loads((run_dir / 'run_manifest.json').read_text())
        except (OSError, json.JSONDecodeError):
            reject_reason = 'run_manifest.json is missing or unreadable'
        results = manifest.get('results') if isinstance(manifest, dict) else None
        if reject_reason is None:
            if not isinstance(manifest, dict):
                reject_reason = 'run_manifest.json is missing or unreadable'
            elif manifest.get('status') != 'finished':
                reject_reason = (f"run status is {manifest.get('status')!r}, not 'finished'; "
                                 'the run did not complete')
            else:
                recorded = manifest.get('hyperparameters')
                if not isinstance(recorded, dict):
                    reject_reason = ('run_manifest.json has no hyperparameters block; a '
                                     'finished run must record the recipe it trained with')
                else:
                    expected = {'model': model, 'norm': norm, 'seed': int(seed)}
                    # A missing key is unknown, not the expected value: a
                    # manifest that cannot say which model/norm/seed it ran
                    # cannot certify this aggregate.
                    def _matches(key, wanted):
                        actual = recorded.get(key)
                        if key == 'seed':
                            # `True == 1` and `1.0 == 1` must not let a
                            # malformed manifest stand in for seed 1.
                            return (_is_seed_id(actual)
                                    and int(actual) == int(wanted))
                        return actual == wanted

                    mismatches = {
                        key: [wanted, recorded.get(key)]
                        for key, wanted in expected.items()
                        if not _matches(key, wanted)
                    }
                    if mismatches:
                        reject_reason = ('recorded model/norm/seed differs from this aggregate: '
                                         + json.dumps(mismatches, sort_keys=True))
                if reject_reason is None and (not isinstance(results, dict) or not results):
                    reject_reason = ('run_manifest.json has no results block; a finished '
                                     'run must record its results')
                if reject_reason is None and reference_recipe is not None:
                    # Compare every run-identity field that at least one side
                    # records. Keys absent from both are unknown (older
                    # manifests); a key present on only one side is a mismatch.
                    within_keys = tuple(
                        key for key in _WITHIN_ARM_PROTOCOL_KEYS
                        if key in reference_recipe or key in recorded)
                    recipe_mismatches = training_recipe_mismatches(
                        reference_recipe, recorded, keys=within_keys)
                    if recipe_mismatches:
                        reject_reason = ('recorded training recipe differs from seed '
                                         f'{reference_recipe.get("seed")}: '
                                         + json.dumps(recipe_mismatches, sort_keys=True))
                if reject_reason is None:
                    ledgers, ledger_problem = _parse_manifest_ledger_fingerprints(manifest)
                    if ledger_problem is not None:
                        # A malformed record is not a legacy absence: pooling it
                        # would let unverifiable exclusions certify the arm.
                        reject_reason = (
                            'run_manifest.json has a malformed ledger identity: '
                            + ledger_problem)
                    elif ledgers is None:
                        # Older/minimal manifests may predate ledger fingerprinting.
                        # Leave them unknown when *no* seed records one, but a mix of
                        # known and unknown identities cannot be pooled; that is
                        # settled after the loop.
                        ledger_missing_seeds.append(int(seed))
                    else:
                        ledger_present_seeds.append(int(seed))
                        if reference_ledgers is None:
                            reference_ledgers = ledgers
                        elif ledgers != reference_ledgers:
                            reject_reason = ('recorded cross-split ledger content differs from '
                                             'the other seed runs; a different exclusion set '
                                             'means the pooled losses are not one protocol: '
                                             + json.dumps({'this': ledgers,
                                                           'reference': reference_ledgers},
                                                          sort_keys=True))
                if reject_reason is None:
                    # A missing step is unknown, not evidence that every seed
                    # stopped at the same fixed step. Without both numbers the
                    # aggregate cannot certify the fixed-step comparison it is
                    # about to pool. `final_val_step` only exists when the run
                    # validated: with `--val-every 0` there is no final held-out
                    # pass, so its absence is not a malformed manifest (the
                    # sweep aggregator applies the same guard).
                    for step_key in ('final_step', 'final_val_step'):
                        if (step_key == 'final_val_step'
                                and results.get('final_val_loss') is None):
                            continue
                        step_value = results.get(step_key)
                        if (not isinstance(step_value, int)
                                or isinstance(step_value, bool) or step_value <= 0):
                            reject_reason = (
                                f'run_manifest.json results.{step_key} is '
                                f'{step_value!r}, not a positive integer step; '
                                'the fixed-step comparison cannot be certified')
                            break
                if (reject_reason is None
                        and results.get('final_val_loss') is not None):
                    # The comparable number is the held-out pass *at the
                    # fixed final step*. If validation stopped earlier
                    # (e.g. --max-steps 1200 with --val-every 500 ends at
                    # final_val_step=1000), pooling final_val_loss would
                    # compare an earlier model under the final step's name.
                    if results.get('final_val_step') != results.get('final_step'):
                        reject_reason = (
                            f"run_manifest.json results.final_val_step is "
                            f"{results.get('final_val_step')!r} but results.final_step is "
                            f"{results.get('final_step')!r}; the final held-out pass was not "
                            "taken at the fixed final step, so the fixed-step comparison "
                            "cannot be certified")
                if reject_reason is None:
                    context = {'final_step': results.get('final_step'),
                               'final_val_step': results.get('final_val_step')}
                    if reference_context is not None:
                        context_mismatches = {
                            key: [reference_context[key], value]
                            for key, value in context.items()
                            if reference_context[key] != value}
                        if context_mismatches:
                            reject_reason = ('run ended at a different training step than '
                                             f'seed {reference_recipe.get("seed")}: '
                                             + json.dumps(context_mismatches, sort_keys=True))
                    if reject_reason is None and reference_recipe is None:
                        reference_recipe = recorded
                        reference_context = context
        if reject_reason is not None:
            record['rejected'] = reject_reason
            rejections.append({'seed': int(seed), 'path': str(run_dir / 'run_manifest.json'),
                               'reason': reject_reason})
        else:
            record['final_val_loss'] = results.get('final_val_loss')
            record['best_val_loss'] = results.get('best_val_loss')
            record['final_val_step'] = results.get('final_val_step')
            record['best_val_step'] = results.get('best_val_step')
            record['n_params'] = results.get('n_params')
            record['n_layers'] = results.get('n_layers')
            record['dim'] = results.get('dim')
            record['n_heads'] = results.get('n_heads')
            record['data_version'] = _manifest_data_version(manifest)
            record['data_ledgers'] = ledgers
            stream = (manifest.get('data') or {}).get('train_stream')
            record['train_stream'] = stream if isinstance(stream, dict) else None
            record['n_loss_spikes'] = results.get('n_loss_spikes')
            record['n_nonfinite_losses'] = results.get('n_nonfinite_losses')
            record['n_nonfinite_gradients'] = results.get('n_nonfinite_gradients')
            record['n_nonfinite_val_losses'] = results.get('n_nonfinite_val_losses')
            record['tokens_per_second'] = results.get('tokens_per_second')
            record['peak_device_memory_mb'] = results.get('peak_device_memory_mb')
            curve = results.get('curve_diagnosis') or {}
            record['curve_classification'] = curve.get('classification')
            record['curve_generalization_gap'] = curve.get('generalization_gap')
            value = results.get('test_loss')
            if _is_finite_number(value):
                record['test_loss'] = float(value)
        per_seed.append(record)

    def _finite(value) -> bool:
        return _is_finite_number(value)

    # A seed whose manifest reports NaN/Inf is not a usable replication even
    # when its other loss is finite: it must not enter any pooled mean (the
    # same exclusion missing seeds get).
    invalid_seed_set = {
        r['seed'] for r in per_seed
        if any(r.get(key) is not None and not _is_finite_number(r.get(key))
               for key in ('final_val_loss', 'best_val_loss'))}

    def _pool(key: str):
        vals = [float(r[key]) for r in per_seed
                if r['seed'] not in invalid_seed_set and _finite(r.get(key))]
        return vals, (mean_std_stderr(vals) if vals else None)

    if ledger_missing_seeds and ledger_present_seeds:
        # Some seeds recorded ledger content and others did not: the held-out
        # exclusion set cannot be proven identical across the arm.
        ledger_identity_complete = False
        for missing_seed in ledger_missing_seeds:
            rejections.append({
                'seed': int(missing_seed),
                'path': str(seed_run_dir(base_dir, missing_seed) / 'run_manifest.json'),
                'reason': ('run_manifest.json has no complete ledger identity while '
                           'other seed runs record one; the held-out document set '
                           'cannot be compared'),
            })
    # A pooled loss is only one arm's number if every seed trained on the
    # same upstream snapshot. A missing resolved revision is unknown, not a
    # match (the Hub may have been unreachable and left a branch name).
    data_versions_by_seed = {
        int(r['seed']): r.get('data_version') for r in per_seed}
    known_versions = {
        json.dumps(v, sort_keys=True) for v in data_versions_by_seed.values()
        if v is not None}
    unknown_data_version_seeds = sorted(
        seed for seed, v in data_versions_by_seed.items() if v is None)
    data_versions_complete = bool(data_versions_by_seed) and not unknown_data_version_seeds
    data_versions_consistent = (data_versions_complete and len(known_versions) <= 1)
    # The same recipe also means the same architecture: a stale checkpoint or
    # a per-seed ADASQUASH_*_OVERRIDE leaves a seed with a different shape,
    # whose loss cannot be averaged with the others.
    shape_keys = ('n_params', 'n_layers', 'dim', 'n_heads')
    shapes_by_seed = {}
    unknown_shape_seeds = []
    for r in per_seed:
        shape = {key: r.get(key) for key in shape_keys}
        if all(isinstance(v, int) and not isinstance(v, bool) and v > 0
               for v in shape.values()):
            shapes_by_seed[int(r['seed'])] = shape
        else:
            unknown_shape_seeds.append(int(r['seed']))
    shape_signatures = {
        json.dumps(shape, sort_keys=True) for shape in shapes_by_seed.values()}
    shapes_complete = bool(shapes_by_seed) and not unknown_shape_seeds
    shapes_consistent = shapes_complete and len(shape_signatures) <= 1
    final_vals, final_stats = _pool('final_val_loss')
    best_vals, best_stats = _pool('best_val_loss')
    test_vals, test_stats = _pool('test_loss')
    tps_vals, tps_stats = _pool('tokens_per_second')
    mem_vals, mem_stats = _pool('peak_device_memory_mb')
    gap_vals, gap_stats = _pool('curve_generalization_gap')
    curve_counts = dict(Counter(
        r['curve_classification'] for r in per_seed if r.get('curve_classification')))
    # `n_seeds` and the pooled final estimate are built from final validation
    # passes, so completeness means every requested seed has a finite final
    # value. A finite `best_val_loss` alone is not a substitute: it is the
    # selection statistic (the minimum over passes), not the fixed-step
    # estimate the cross-arm comparison uses.
    missing = [r['seed'] for r in per_seed if not _finite(r['final_val_loss'])]
    invalid = [r['seed'] for r in per_seed if r['seed'] in invalid_seed_set]
    pool_ok = (not missing and not invalid
               and ledger_identity_complete
               and data_versions_complete and data_versions_consistent
               and shapes_complete and shapes_consistent)
    return {
        'model': model, 'norm': norm,
        'seeds': seeds,
        'n_seeds_requested': len(seeds),
        'n_seeds': len(final_vals),
        'missing_seeds': missing,
        'invalid_seeds': invalid,
        'manifest_rejections': rejections,
        'data_versions_by_seed': data_versions_by_seed,
        'data_ledgers_by_seed': {int(r['seed']): r.get('data_ledgers')
                                 for r in per_seed},
        'data_ledgers_complete': ledger_identity_complete,
        'train_stream_by_seed': {int(r['seed']): r.get('train_stream')
                                 for r in per_seed},
        'train_stream_complete': bool(per_seed) and all(
            isinstance(r.get('train_stream'), dict) for r in per_seed),
        'unknown_data_version_seeds': unknown_data_version_seeds,
        'data_versions_complete': data_versions_complete,
        'data_versions_consistent': data_versions_consistent,
        'shapes_by_seed': shapes_by_seed,
        'unknown_shape_seeds': sorted(unknown_shape_seeds),
        'shapes_complete': shapes_complete,
        'shapes_consistent': shapes_consistent,
        'complete': pool_ok,
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
        # Measured cost of the arm (not an analytic estimate).
        'tokens_per_second': None if tps_stats is None else tps_stats['mean'],
        'tokens_per_second_std': None if tps_stats is None or len(tps_vals) < 2 else tps_stats['std'],
        'peak_device_memory_mb': None if mem_stats is None else mem_stats['mean'],
        'peak_device_memory_mb_std': None if mem_stats is None or len(mem_vals) < 2 else mem_stats['std'],
        'cost_basis': 'tokens_per_second / peak_device_memory_mb are measured this run; memory is null on CPU',
        # The pooled loss can hide that one seed ended in a different regime.
        # Keep the per-seed verdicts next to the mean, as the ablation and MoE
        # studies do, so a delta from an underfit/overfit seed is visible.
        'curve_classification_counts': curve_counts,
        'curve_classifications_by_seed': {
            int(r['seed']): r.get('curve_classification') for r in per_seed},
        'curve_generalization_gap': None if gap_stats is None else gap_stats['mean'],
        # Numerical-health counters pooled across seeds: an arm whose mean
        # covers triaged NaN gradient/validation events is not a clean
        # replication of an arm with none.
        'n_nonfinite_losses': sum(r['n_nonfinite_losses'] for r in per_seed
                                  if _is_event_count(r.get('n_nonfinite_losses'))),
        'n_nonfinite_gradients': sum(r['n_nonfinite_gradients'] for r in per_seed
                                     if _is_event_count(r.get('n_nonfinite_gradients'))),
        'n_nonfinite_val_losses': sum(r['n_nonfinite_val_losses'] for r in per_seed
                                      if _is_event_count(r.get('n_nonfinite_val_losses'))),
        'certified': len(final_vals) >= 3 and pool_ok,
        'certification_basis': ('certified: >= 3 seeds with finite final_val_loss, no missing/invalid seeds, one cross-split ledger identity, and one resolved data version'
                                if len(final_vals) >= 3 and pool_ok
                                else 'not certified: ' + (
                                    f'{len(final_vals)} seeds (< 3)' if len(final_vals) < 3 else
                                    f'missing seeds: {missing}' if missing else
                                    f'invalid seeds: {invalid}' if invalid else
                                    'seeds do not record one cross-split ledger identity'
                                    if not ledger_identity_complete else
                                    f'unknown data version for seeds: {unknown_data_version_seeds}'
                                    if not data_versions_complete else
                                    'seeds trained on different resolved data versions'
                                    if not data_versions_consistent else
                                    f'unknown model shape for seeds: {sorted(unknown_shape_seeds)}'
                                    if not shapes_complete else
                                    'seeds trained with different model shapes')),
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
        print(f'[seeds][error] {len(seeds)} seed(s): certification requires >= 3 independent seeds to separate norm effect from run-to-run noise; use --seeds with >= 3 values')
        raise SystemExit('[seeds] insufficient seeds for certified replication')
    (base / 'aggregate.json').unlink(missing_ok=True)
    (base / 'run_manifest.json').unlink(missing_ok=True)
    child_argv = list(sys.argv[1:])
    # This helper lives in train_runtime, but the trainable entry point is
    # train_gpt2: relaunching `__file__` would run an import-only module that
    # exits successfully without training or writing a seed manifest.
    script = str(ROOT / 'scripts' / 'train_gpt2.py')
    failed_seeds: list = []
    for seed in seeds:
        out_dir = seed_run_dir(base, seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in ('run_manifest.json', 'test_results.json'):
            (out_dir / stale).unlink(missing_ok=True)
        print(f'[seeds] launching seed={seed} -> {out_dir}', flush=True)
        env = dict(os.environ)
        # PYTHONHASHSEED only accepts [0, 2**32); a negative or larger seed
        # would kill the child before Python starts. Match `set_all_seeds`,
        # which folds the same seed into that range for this process.
        env['PYTHONHASHSEED'] = str(int(seed) % (2 ** 32))
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
    if agg.get('tokens_per_second') is not None:
        tps_std = agg.get('tokens_per_second_std')
        tps_str = '' if tps_std is None else f' +/- {tps_std:.1f}'
        print(f'[seeds] {args.norm}: measured cost={agg["tokens_per_second"]:.1f}{tps_str} tok/s'
              + ('' if agg.get('peak_device_memory_mb') is None
                 else f', peak={agg["peak_device_memory_mb"]:.1f} MB'))
    regimes = agg.get('curve_classification_counts') or {}
    if regimes:
        print(f'[seeds] {args.norm}: per-seed curve regimes={regimes}')
        off = {k: v for k, v in regimes.items()
               if k not in ('healthy', 'insufficient_data')}
        if off:
            print(f'[seeds][warn] {sum(off.values())} seed(s) ended in {off}: the pooled '
                  'loss describes that regime as much as the norm; read the per-seed '
                  'curves before attributing the mean delta to the change')
    for rejection in agg.get('manifest_rejections') or []:
        print(f'[seeds][warn] seed={rejection["seed"]}: rejected manifest '
              f'({rejection["reason"]})')
    if agg['missing_seeds']:
        print(f'[seeds][warn] missing/unfinished seed runs: {agg["missing_seeds"]}; the pooled mean '
              f'covers {agg["n_seeds"]} of {agg["n_seeds_requested"]} requested seeds')
    for label, key in (('training losses', 'n_nonfinite_losses'),
                       ('gradient events', 'n_nonfinite_gradients'),
                       ('validation losses', 'n_nonfinite_val_losses')):
        n_bad = agg.get(key) or 0
        if n_bad:
            per_seed_hits = {r['seed']: r.get(key) for r in agg['per_seed'] if r.get(key)}
            print(f'[seeds][warn] {n_bad} non-finite {label} across seeds {per_seed_hits}: '
                  f'the pooled mean mixes triaged numerical failures into the arm; '
                  f'read each seed manifest\'s triage evidence before comparing')
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
    incomplete_seeds = sorted(set(int(s) for s in failed_seeds)
                              | set(agg.get('missing_seeds') or [])
                              | set(agg.get('invalid_seeds') or []))
    if incomplete_seeds:
        manifest['status'] = 'failed'
        manifest['failed_seeds'] = [int(s) for s in failed_seeds]
        manifest['incomplete_seeds'] = incomplete_seeds
    save_run_manifest(base / 'run_manifest.json', manifest)
    if incomplete_seeds:
        # A multi-seed invocation is a replication study: a failed child is
        # not a valid draw, and exiting 0 would let CI/orchestration treat
        # the pooled mean as a clean result while it silently covers fewer
        # seeds than requested. The artifacts above are already on disk.
        raise SystemExit(
            f'[seeds] {len(incomplete_seeds)}/{len(seeds)} seed run(s) are incomplete '
            f'{incomplete_seeds}: the aggregate covers {agg["n_seeds"]} of '
            f'{agg["n_seeds_requested"]} requested seeds, so this invocation is not a clean '
            'replication. Fix the failures (each seed writes its own run_manifest.json) and '
            're-run before quoting the pooled numbers.')

