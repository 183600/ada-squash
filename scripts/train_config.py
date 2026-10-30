from __future__ import annotations
import argparse
import hashlib
import json
import math
import numbers
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
def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config '{path}' must be a mapping, got {type(data)}")
    return data

def _config_bool(value):
    """Parse a YAML boolean spelling, or return None when it is not one."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in (
            'true', 'yes', 'on', '1', 'false', 'no', 'off', '0'):
        return value.strip().lower() in ('true', 'yes', 'on', '1')
    return None

def _coerce_config_scalar(action, value, path: str, key: str):
    """Apply argparse's `type`/`choices` contract to one YAML value.

    `apply_config_to_parser` writes YAML values straight into parser defaults,
    which bypasses argparse's conversion and validation. A typo such as
    `norm: dytcc` or `batch_size: "32"` would otherwise survive until the
    model or training loop fails, after the run has already started.
    """
    if action.const is True or action.const is False:
        parsed = _config_bool(value)
        if parsed is None:
            raise ValueError(
                f"config '{path}': {key}={value!r} is not a boolean")
        return parsed
    if action.type is int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(
                f"config '{path}': {key}={value!r} must be an integer")
        converted = int(value)
    elif action.type is float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"config '{path}': {key}={value!r} must be a real number")
        converted = float(value)
    elif action.type is str:
        if not isinstance(value, str):
            raise ValueError(
                f"config '{path}': {key}={value!r} must be a string")
        converted = value
    elif action.type is not None:
        try:
            converted = action.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"config '{path}': {key}={value!r} is not valid for "
                f"{action.dest}") from exc
    else:
        converted = value
    if action.choices is not None and converted not in action.choices:
        raise ValueError(
            f"config '{path}': {key}={value!r} is not one of "
            f"{list(action.choices)}")
    return converted

def _coerce_config_value(action, value, path: str, key: str):
    """Validate one config value against its parser action."""
    if value is None:
        nullable = action.default is None or (
            action.choices is not None and None in action.choices)
        if nullable:
            return None
        raise ValueError(
            f"config '{path}': {key} is null but the option is not nullable")
    if action.nargs in ('+', '*') or isinstance(action.nargs, int) and action.nargs > 1:
        if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
            raise ValueError(
                f"config '{path}': {key}={value!r} must be a list")
        converted = [_coerce_config_scalar(action, item, path, key)
                     for item in value]
        if isinstance(action.nargs, int) and action.nargs > 1 and len(converted) != action.nargs:
            raise ValueError(
                f"config '{path}': {key} expects {action.nargs} values, "
                f"got {len(converted)}")
        return converted
    return _coerce_config_scalar(action, value, path, key)

def apply_config_to_parser(parser, path: str | None, aliases: dict | None=None,
                           strict: bool = False,
                           allowed_unknown: tuple[str, ...] = ()):
    """Fold a YAML config into `parser` defaults.

    `strict=True` turns a key the parser does not know into an error instead
    of a printed warning. A typo such as `learning_rate:` in a config is then
    rejected before training rather than silently leaving the default in
    place: the run would otherwise execute a protocol nobody wrote while the
    manifest records the default as if it were the intended value.
    `allowed_unknown` names keys a caller consumes outside argparse (for
    example the ablation runner's `ablations` block).
    """
    if not path:
        return
    cfg = load_yaml_config(path)
    aliases = aliases or {}
    allowed_unknown = frozenset(allowed_unknown)
    actions_by_dest: dict[str, list] = {}
    for action in parser._actions:
        actions_by_dest.setdefault(action.dest, []).append(action)
    known = set(actions_by_dest)
    unknown = []
    for key, value in cfg.items():
        dest, invert = aliases.get(key, (key, False))
        if dest not in known:
            if key not in allowed_unknown:
                unknown.append(key)
            continue
        if invert:
            # `use_layerscale: 0` / `bf16: "false"` must not remain a truthy
            # default that `not args.no_*` later reads as enabled.
            bool_value = _config_bool(value)
            if bool_value is None:
                raise ValueError(
                    f"config '{path}': {key}={value!r} is not a boolean; "
                    f"cannot map it to {dest}")
            value = not bool_value
        elif dest in ('tie_embeddings', 'beta'):
            # Positive-form boolean options have the same YAML aliases; a raw
            # `"false"` string is truthy and would silently enable the option.
            bool_value = _config_bool(value)
            if bool_value is None:
                raise ValueError(
                    f"config '{path}': {key}={value!r} is not a boolean")
            value = bool_value
        else:
            actions = actions_by_dest[dest]
            action = next(
                (candidate for candidate in actions
                 if candidate.type is not None or candidate.choices is not None),
                actions[0])
            value = _coerce_config_value(action, value, path, key)
        parser.set_defaults(**{dest: value})
    if unknown:
        message = (f"config '{path}' contains key(s) the parser does not "
                   f"understand: {sorted(unknown)}")
        if strict:
            raise ValueError(
                message + '; fix or remove them (an unrecognized key would '
                          'otherwise be silently ignored and the run would use '
                          'the default instead of the intended value)')
        print(f'[config] {path}: ignoring unknown keys {sorted(unknown)}')
_TRAIN_CONFIG_ALIASES = {'use_layerscale': ('no_layerscale', True), 'learnable_p': ('no_learnable_p', True), 'learnable_k': ('no_learnable_k', True), 'learnable_delta': ('no_learnable_delta', True), 'tie_word_embeddings': ('tie_embeddings', False), 'bf16': ('no_bf16', True), 'beta': ('beta', False), 'alpha_jitter': ('alpha_jitter', False), 'k_min': ('k_min', False), 'shared_depth_schedule': ('shared_depth_schedule', False)}

def _optional_dest(parser, option: str):
    """`dest` of an option spelling, including argparse's unambiguous prefixes.

    `_parse_optional` returns a list of `(action, option_string, sep,
    explicit_arg)` candidates (a single tuple on older Pythons); an unknown or
    ambiguous spelling has no single destination and is ignored.
    """
    try:
        parsed = parser._parse_optional(option)
    except Exception:
        return None
    if not parsed:
        return None
    entries = parsed if isinstance(parsed, list) else [parsed]
    actions = [entry[0] for entry in entries if entry and entry[0] is not None]
    dests = {action.dest for action in actions}
    return dests.pop() if len(dests) == 1 else None

def _explicit_arg_names(argv, parser, config_path=None) -> frozenset:
    """Names of options the user typed on the command line.

    `--config` seeds the parser defaults (see `apply_config_to_parser`), so a
    key in the YAML file is a default, not an explicit override. Counting it
    here would let a config value silently reshape the schedule of the run it
    resumes. The `config_path` argument is retained only for callers that
    still pass it; config keys are never explicit. Option strings are mapped
    through the parser so negated aliases such as --no-dedup (dest 'dedup')
    and --no-beta (dest 'beta') are attributed correctly.
    """
    names = set()
    for token in argv:
        if token == '--':
            break
        if token.startswith('--'):
            # Resolve through argparse, not the exact option-string table:
            # argparse accepts unambiguous prefixes (`--val-f` for
            # `--val-frac`) and that spelling is just as explicit.
            dest = _optional_dest(parser, token.split('=', 1)[0])
            if dest is not None:
                names.add(dest)
    return frozenset(names)

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

def _checkpoint_nonfinite(model, opt) -> list:
    """Non-finite weights or optimizer moments that must never be persisted.

    A finite loss and finite gradients do not guarantee a finite optimizer
    state: an overflow in `exp_avg_sq` can leave the weights finite at save
    time and surface as a NaN on the next step. Writing such a checkpoint
    silently destroys the last restorable state, so both the model and every
    floating optimizer tensor are inspected before the atomic replace.
    """
    failures: list = []
    report = parameter_finite_report(model)
    if not report['finite']:
        for entry in report['nonfinite_parameters']:
            failures.append(f"parameter {entry['name']} (nan={entry['n_nan']}, inf={entry['n_inf']})")
        for entry in report['nonfinite_buffers']:
            failures.append(f"buffer {entry['name']} (nan={entry['n_nan']}, inf={entry['n_inf']})")
    names = {id(p): name for name, p in model.named_parameters()}
    for param, state in opt.state.items():
        name = names.get(id(param), f"parameter#{id(param)}")
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if not (value.is_floating_point() or value.is_complex()):
                continue
            finite = tensor_finite_report(value.detach())
            if not finite['finite']:
                failures.append(
                    f"optimizer {name}.{key} (nan={finite['n_nan']}, inf={finite['n_inf']})")
    return failures


def _assert_checkpoint_finite(model, opt, path: Path, step: int) -> None:
    failures = _checkpoint_nonfinite(model, opt)
    if not failures:
        return
    shown = '; '.join(failures[:10])
    more = '' if len(failures) <= 10 else f' (+{len(failures) - 10} more)'
    raise SystemExit(
        f'[ckpt] refusing to write {path} at step {step}: non-finite training state '
        f'({shown}{more}). The previous checkpoint is kept. Restore it and run the '
        f'NaN triage on the step that produced the overflow (data -> fp32 math -> '
        f'mixed precision -> gradients) instead of persisting a checkpoint that is '
        f'already corrupted.')


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
    if args.warmup == 0:
        # LambdaLR evaluates step 0 for the first update. With no warmup the
        # first update must use the peak, and the last update (s =
        # max_steps-1) must reach the cosine floor.
        return lambda s: cosine_schedule(
            s, max(1, args.max_steps - 1), 0, args.min_lr_ratio)
    return lambda s: cosine_schedule(s + 1, args.max_steps, args.warmup, args.min_lr_ratio)
_SCHEDULE_ARGS = ('max_steps', 'warmup', 'min_lr_ratio', 'lr', 'weight_decay', 'beta1', 'beta2', 'grad_clip', 'grad_accum')
_MONITOR_ARGS = ('val_every', 'val_batches', 'early_stop_patience')
_ARCH_ARGS = ('model', 'norm', 'qk_norm', 'qk_alpha_init', 'no_layerscale', 'layerscale_init', 'tie_embeddings', 'no_learnable_p', 'no_learnable_k', 'no_learnable_delta', 'beta', 'alpha_0', 'k_init', 'k_min', 'shared_depth_schedule', 'alpha_init', 'p_init', 'alpha_jitter', 'gradient_checkpointing')
_NUMERIC_ARGS = ('compile', 'compile_mode', 'no_bf16')
_RESUME_RUN_ARGS = _ARCH_ARGS + _NUMERIC_ARGS
# A within-arm aggregation must compare the whole run identity, not only the
# optimizer/data recipe: two seeds that trained the same shape with different
# alpha/k/p init or a different compile path are not replications of one arm.
_WITHIN_ARM_PROTOCOL_KEYS = tuple(dict.fromkeys(
    TRAINING_RECIPE_KEYS + _RESUME_RUN_ARGS))

def _restore_run_args(args, ckpt_args: dict, names, explicit: frozenset=frozenset()) -> list:
    changed = []
    for name in names:
        if name not in ckpt_args:
            continue
        if getattr(args, name, None) != ckpt_args[name]:
            changed.append(name)
        if name in explicit:
            continue
        setattr(args, name, ckpt_args[name])
    return changed

def restore_schedule_args(args, ckpt_args: dict, explicit: frozenset=frozenset()) -> list:
    return _restore_run_args(args, ckpt_args, _SCHEDULE_ARGS, explicit)

def restore_monitor_args(args, ckpt_args: dict, explicit: frozenset=frozenset()) -> list:
    """Restore the cadence that defines the validation/selection protocol.

    `val_every` and `val_batches` decide which held-out slices produce curve
    points and how noisy each point is; `early_stop_patience` decides when a
    run ends. A resume that silently resets them to CLI defaults would mix two
    selection protocols in one history (the same reason schedule and data flags
    are restored). `grad_accum` travels with the schedule because it changes
    the effective batch size and the meaning of the saved data position.
    """
    return _restore_run_args(args, ckpt_args, _MONITOR_ARGS, explicit)
# The split-defining args are part of a run's identity: resuming with a
# different boundary could move documents the model already trained on into
# the validation split (and vice versa), so they are restored from the
# checkpoint exactly like the schedule flags. The cleaning knobs and the
# cross-split duplicate ledger are part of the same identity: changing them
# changes which documents are trained on and which are held out. The dataset /
# tokenizer revisions are the data *version*: resuming a run against a moved
# upstream branch would continue on different data under the same manifest.
_DATA_ARGS = ('dataset', 'dataset_config', 'dataset_revision', 'tokenizer', 'tokenizer_revision',
              'val_frac', 'test_frac', 'num_documents',
              'dedup', 'min_doc_chars', 'max_doc_chars', 'exclude_hash_file',
              'exclude_test_hash_file', 'exclude_train_hash_file',
              'group_key', 'exclude_group_file',
              'exclude_test_group_file', 'train_shuffle_buffer',
              # These define the stream geometry: the resumed run must read the
              # same document windows with the same batch/sequence shape, or
              # `data_position` no longer points at the same tokens.
              'seq_len', 'batch_size', 'num_workers')

_LEDGER_ARGS = ('exclude_hash_file', 'exclude_test_hash_file',
                'exclude_train_hash_file',
                'exclude_group_file', 'exclude_test_group_file')

def unpinned_data_revisions(dataset_revision, tokenizer_revision) -> list:
    """Requested data revisions that are not immutable Hub commit hashes.

    A branch or tag can move between two supposedly identical runs, so a
    confirmatory protocol must pin both the dataset and the tokenizer to a
    concrete commit *before* the run starts. The resolved value recorded in
    the manifest proves what this run read; it does not let a later rerun
    reconstruct the same command-line protocol.
    """
    problems = []
    for flag, value in (('--dataset-revision', dataset_revision),
                        ('--tokenizer-revision', tokenizer_revision)):
        if not _is_pinned_revision(value):
            problems.append(f'{flag} is not a full 40-hex commit hash: {value!r}')
    return problems
