from __future__ import annotations
import argparse
import json
import math
import os
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
from adasquash import GPTConfig, GPT, build_norm, count_parameters, ActivationStats, LossSpikeCounter
from adasquash.diagnostics import diagnose_train_val_curve, triage_nonfinite
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b
from adasquash.repro import collect_run_metadata, file_fingerprint, save_run_manifest, set_all_seeds
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
    fused = len(devices) == 1 and next(iter(devices)).type == 'cuda'
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
# changes which documents are trained on and which are held out.
_DATA_ARGS = ('dataset', 'dataset_config', 'val_frac', 'test_frac', 'num_documents',
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
        shuffle_buffer=args.train_shuffle_buffer if split == 'train' else 0,
        seed=args.seed,
    )
    return build_dataloader(args.dataset, args.dataset_config, seq_len or args.seq_len, args.batch_size, tokenizer=args.tokenizer)

def evaluate_heldout(model, loader, state: dict, n_batches: int, ctx, device) -> float | None:
    """Mean loss over held-out batches; `state['iter']` carries the position.

    Runs under `no_grad` and eval mode: no gradient ever flows from a
    validation or test batch into the parameters.
    """
    model.eval()
    it = state.get('iter')
    total = 0.0
    n = 0
    with torch.no_grad():
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
            total += float(loss.detach().item())
            n += 1
    model.train()
    return total / n if n else None

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

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='YAML run config; CLI flags override it')
    p.add_argument('--model', choices=['gpt2-124m', 'gpt2-350m', 'llama-1b'], default='gpt2-124m')
    p.add_argument('--norm', default='dytcd', choices=list(['ln', 'rmsnorm', 'dyt', 'dytc', 'dytd', 'dytcd']))
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
    p.add_argument('--val-frac', type=float, default=0.005, help='chronological held-out validation fraction of the document stream (split order is train=oldest, then val, then test; never shuffled)')
    p.add_argument('--test-frac', type=float, default=0.005, help='sealed test fraction of the document stream, evaluated once at the end only with --final-test-eval')
    p.add_argument('--num-documents', type=int, default=None, help='total documents in the stream; required when the dataset reports no size')
    p.add_argument('--val-every', type=int, default=500, help='run a held-out validation pass every N optimizer steps (0 disables; 0 is not recommended)')
    p.add_argument('--val-batches', type=int, default=10, help='batches per held-out validation pass')
    p.add_argument('--early-stop-patience', type=int, default=0, help='stop after this many validation passes without improvement (0 = off)')
    p.add_argument('--nonfinite-policy', choices=['abort', 'skip'], default='abort', help='what to do when a training loss is NaN/Inf: abort (default) runs the ordered data->math->precision->gradient triage and stops; skip runs the triage and discards only that step (5 consecutive non-finite steps still abort)')
    p.add_argument('--final-test-eval', action='store_true', help='after training, evaluate the sealed test split exactly once (restores the best-validation checkpoint first)')
    p.add_argument('--allow-test-rerun', action='store_true', help='permit a second one-shot evaluation of the sealed test split in the same output dir (breaks the seal: the test set becomes an implicit validation set)')
    p.add_argument('--test-batches', type=int, default=20, help='batches for the one-shot final test evaluation')
    p.add_argument('--min-doc-chars', type=int, default=0, help='drop documents shorter than this many characters (0 = keep all non-empty)')
    p.add_argument('--max-doc-chars', type=int, default=0, help='drop documents longer than this many characters (0 = no length-outlier cap)')
    p.add_argument('--no-dedup', dest='dedup', action='store_false', help='disable exact-duplicate document removal')
    p.add_argument('--exclude-hash-file', default=None, help='training-document hash ledger (scripts/audit_data.py --write-train-hash-file); held-out documents whose exact text also occurs in train are dropped from validation/test, never from train')
    p.add_argument('--train-shuffle-buffer', type=int, default=0, help='shuffle window over packed TRAINING batches (0 = off); never applied to validation or test')
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
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--compile', action='store_true', help='enable torch.compile')
    p.add_argument('--compile-mode', default='reduce-overhead', choices=['default', 'reduce-overhead', 'max-autotune'])
    p.add_argument('--sanity-check', action='store_true', help='run the small-batch overfit sanity check (scripts/sanity_overfit.py) before training and abort if it fails')
    p.add_argument('--output-dir', default='runs/gpt2-124m-dytcd')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stats-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--wandb-project', default=None)
    p.add_argument('--run-name', default=None)
    p.add_argument('--resume', default=None, help='checkpoint to resume from')
    apply_config_to_parser(p, known.config, _TRAIN_CONFIG_ALIASES)
    args = p.parse_args()
    explicit = frozenset((name for name in _SCHEDULE_ARGS + _RESUME_RUN_ARGS + _DATA_ARGS if getattr(args, name, None) != p.get_default(name)))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resume_ckpt = None
    run_args: dict = {}
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
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
    if args.sanity_check:
        from scripts.sanity_overfit import run_preflight
        run_preflight(args.norm, args.seed, args.output_dir)
    set_all_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.bf16 and device == 'cuda' else torch.float32
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if dtype != torch.float32 else nullcontext()
    model_cfg = dict(norm_name=args.norm, qk_norm=args.qk_norm, qk_alpha_init=args.qk_alpha_init, use_layerscale=not args.no_layerscale, alpha_0=args.alpha_0, k_init=args.k_init, alpha_init=args.alpha_init, p_init=args.p_init, learnable_p=not args.no_learnable_p, learnable_k=not args.no_learnable_k, learnable_delta=not args.no_learnable_delta, beta=args.beta, alpha_jitter=args.alpha_jitter, gradient_checkpointing=args.gradient_checkpointing)
    if args.layerscale_init is not None:
        model_cfg['layerscale_init'] = args.layerscale_init
    if args.tie_embeddings is not None:
        model_cfg['tie_word_embeddings'] = args.tie_embeddings
    if args.model == 'gpt2-124m':
        model = build_gpt_124m(**model_cfg)
    elif args.model == 'gpt2-350m':
        model = build_gpt_350m(**model_cfg)
    else:
        model = build_llama_tiny_1b(**model_cfg)
    depth_override = os.environ.get('ADASQUASH_DEPTH_OVERRIDE')
    dim_override = os.environ.get('ADASQUASH_DIM_OVERRIDE')
    heads_override = os.environ.get('ADASQUASH_HEADS_OVERRIDE')
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
        model = GPT(GPTConfig(**cfg_dict))
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
        # The best-validation tracker is run state like the step counter and
        # the loss-spike baseline: restarting it at `inf` would let the first
        # post-resume validation unconditionally overwrite a possibly better
        # `ckpt_best.pt`, which is the checkpoint the final test evaluation
        # restores. Seed it from the historical best instead.
        best_path = Path(args.output_dir) / 'ckpt_best.pt'
        best_candidates = []
        if resume_ckpt.get('val_loss') is not None:
            best_candidates.append(resume_ckpt)
        if best_path.exists() and Path(args.resume).resolve() != best_path.resolve():
            best_candidates.append(torch.load(best_path, map_location=device, weights_only=False))
        if best_candidates:
            prior_best = min(best_candidates, key=lambda c: float(c['val_loss']))
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
    manifest_path = Path(args.output_dir) / 'run_manifest.json'
    run_manifest = collect_run_metadata(seed=args.seed)
    run_manifest['hyperparameters'] = vars(args)
    run_manifest['data'] = {
        'dataset': args.dataset,
        'dataset_config': args.dataset_config,
        'tokenizer': args.tokenizer,
        'split_reports': data_reports,
        'exclude_hash_file': file_fingerprint(args.exclude_hash_file) if args.exclude_hash_file else None,
        'config_file': file_fingerprint(args.config) if args.config else None,
    }
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
        with open(log_path, 'a') as f:
            f.write(json.dumps({
                'step': step + 1, 'event': 'nonfinite_loss', 'loss': bad_value,
                'lr': lr_now, 'triage_stage': triage['stage'],
                'triage_diagnosis': triage['diagnosis'],
                'triage_evidence': triage['evidence'],
                'n_nonfinite_losses': n_nonfinite_losses,
            }, default=str) + '\n')
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
        loss_acc = None
        nonfinite_batch = None
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with ctx:
                loss = model(x, y, return_logits=False)[1]
            micro_loss = float(loss.detach().item())
            if not math.isfinite(micro_loss):
                # Do not backward a non-finite loss: it would poison every
                # accumulated gradient before the triage ever ran.
                nonfinite_batch = (x, y, micro_loss)
                break
            loss = loss / args.grad_accum
            loss.backward()
            if loss_acc is None:
                loss_acc = loss.detach()
            else:
                loss_acc = loss_acc + loss.detach()
        if attach_stats:
            act_stats.detach()
        if nonfinite_batch is not None:
            _handle_nonfinite(step, *nonfinite_batch)
            continue
        nonfinite_streak = 0
        loss_val = loss_acc.item()
        last_train_loss = loss_val
        grad_norm = None
        if args.grad_clip > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
        opt.step()
        sched.step()
        loss_spike.update(loss_val)
        train_window.append(loss_val)
        step += 1
        if args.val_every > 0 and step % args.val_every == 0:
            if val_loader is None:
                val_loader = open_data_split('validation', args)
                rval = getattr(val_loader, 'split_report', None)
                if rval:
                    data_reports['validation'] = rval
                    print(f'[data] validation split: {rval}')
            val_loss = evaluate_heldout(model, val_loader, val_state, args.val_batches, ctx, device)
            if val_loss is not None:
                last_val_loss = val_loss
                last_val_step = step
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
                    ckpt_best = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'loss_spike': loss_spike.state_dict(), 'val_loss': val_loss, 'provenance': provenance}
                    path = Path(args.output_dir) / 'ckpt_best.pt'
                    torch.save(ckpt_best, path)
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
            with open(log_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
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
            ckpt = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'loss_spike': loss_spike.state_dict(), 'provenance': provenance}
            path = Path(args.output_dir) / f'ckpt_{step:06d}.pt'
            torch.save(ckpt, path)
            print(f'[ckpt] saved {path}')
        if args.early_stop_patience > 0 and val_loader is not None and val_checks_without_improvement >= args.early_stop_patience:
            print(f'[early-stop] validation did not improve for {val_checks_without_improvement} checks (best val_loss={best_val_loss:.4f}@{best_val_step}); stopping at step {step}')
            break
    ckpt = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'n_loss_spikes': loss_spike.n_spikes, 'loss_spike': loss_spike.state_dict(), 'provenance': provenance}
    path = Path(args.output_dir) / 'ckpt_final.pt'
    torch.save(ckpt, path)
    print(f'[done] saved final ckpt to {path}, time = {time.time() - t0:.1f}s')
    run_manifest['data']['split_reports'] = data_reports
    run_manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    flops = analytic_flops_per_token(model.cfg)
    run_manifest['results'] = {
        'final_step': step,
        'last_train_loss': last_train_loss,
        # `best_val_loss` is the minimum over the validation passes: it is the
        # checkpoint-selection statistic, so as a reported generalization
        # estimate it is optimistically biased (the minimum of noisy draws).
        # `final_val_loss` is the last held-out pass at the fixed training-step
        # budget and is the number that makes two arms comparable.
        'best_val_loss': best_val_loss if val_loader is not None else None,
        'best_val_step': best_val_step if val_loader is not None else None,
        'final_val_loss': last_val_loss if val_loader is not None else None,
        'final_val_step': last_val_step if val_loader is not None else None,
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
        'parameter_breakdown': parameter_breakdown(model),
        'flops_per_token_forward': flops['forward_per_token'],
        'flops_per_token_training_est': flops['training_per_token'],
    }
    if n_nonfinite_losses:
        print(f'[warn] {n_nonfinite_losses} non-finite loss step(s) were handled by '
              f'--nonfinite-policy {args.nonfinite_policy}; see train.jsonl event rows')
    save_run_manifest(manifest_path, run_manifest)
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
            best = torch.load(best_path, map_location=device, weights_only=False)
            load_model_state_dict(model, best['model'])
            print(f"[test] restored best-validation checkpoint (step {int(best.get('step', 0))}, val_loss={float(best.get('val_loss', float('nan'))):.4f}) before the one-shot test evaluation")
        test_state = {'iter': iter(test_loader)}
        test_loss = evaluate_heldout(model, test_loader, test_state, args.test_batches, ctx, device)
        results = {'step': step, 'best_val_step': best_val_step if val_loader is not None else None, 'best_val_loss': best_val_loss if val_loader is not None else None, 'final_val_loss': last_val_loss if val_loader is not None else None, 'final_val_step': last_val_step if val_loader is not None else None, 'test_loss': test_loss, 'test_batches': args.test_batches, 'num_documents': args.num_documents, 'val_frac': args.val_frac, 'test_frac': args.test_frac, 'exclude_hash_file': args.exclude_hash_file}
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        run_manifest['data']['split_reports'] = data_reports
        run_manifest['results']['test_loss'] = test_loss
        save_run_manifest(manifest_path, run_manifest)
        print(f'[test] sealed test split evaluated once: test_loss={test_loss} -> {out_path}')
    if tb:
        tb.close()
    if args.wandb_project:
        import wandb
        wandb.finish()
if __name__ == '__main__':
    main()
