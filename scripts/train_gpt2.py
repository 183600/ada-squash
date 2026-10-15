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
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b

def build_dataloader(dataset_name: str, dataset_config: str | None, seq_len: int, batch_size: int, tokenizer: str='gpt2'):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, IterableDataset
    tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    class PackedStream(IterableDataset):
        tokenize_batch = 64

        def __init__(self, ds, tok, seq_len):
            self.ds = ds
            self.tok = tok
            self.seq_len = seq_len
            self.buf: list = []
            self.pos = 0

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            n_workers = worker_info.num_workers if worker_info is not None else 1
            buf = self.buf
            pos = self.pos
            need = self.seq_len + 1
            texts: list = []

            def drain():
                nonlocal pos
                while len(buf) - pos >= need:
                    x = torch.tensor(buf[pos:pos + self.seq_len], dtype=torch.long)
                    y = torch.tensor(buf[pos + 1:pos + need], dtype=torch.long)
                    pos += self.seq_len
                    yield (x, y)
                del buf[:pos]
                pos = 0

            def extend(batch):
                for ids in self.tok(batch, add_special_tokens=False)['input_ids']:
                    buf.extend(ids)
                    buf.append(self.tok.eos_token_id)
            i = 0
            for ex in self.ds:
                if i % n_workers != worker_id:
                    i += 1
                    continue
                i += 1
                texts.append(ex.get('text', ''))
                if len(texts) < self.tokenize_batch and len(buf) - pos < need:
                    continue
                extend(texts)
                texts = []
                yield from drain()
            if texts:
                extend(texts)
                yield from drain()
            self.pos = pos
    ds = load_dataset(dataset_name, dataset_config, split='train', streaming=True)
    stream = PackedStream(ds, tok, seq_len)
    return DataLoader(stream, batch_size=batch_size, num_workers=2, pin_memory=torch.cuda.is_available())

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
_TRAIN_CONFIG_ALIASES = {'use_layerscale': ('no_layerscale', True), 'learnable_p': ('no_learnable_p', True), 'learnable_k': ('no_learnable_k', True), 'tie_word_embeddings': ('tie_embeddings', False), 'bf16': ('no_bf16', True), 'beta': ('beta', False), 'alpha_jitter': ('alpha_jitter', False)}

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
_ARCH_ARGS = ('model', 'norm', 'qk_norm', 'qk_alpha_init', 'no_layerscale', 'layerscale_init', 'tie_embeddings', 'no_learnable_p', 'no_learnable_k', 'beta', 'alpha_0', 'k_init', 'alpha_init', 'p_init', 'alpha_jitter')
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

def _is_no_decay_group(group) -> bool:
    # build_optimizer routes every parameter with ndim < 2 (norm gains, biases,
    # LayerScale and the scalar squashing parameters) to the no-decay group and
    # only nondescript >= 2-D weights to the decay group. Re-deriving the split
    # from the parameters themselves keeps it available after load_state_dict,
    # which replaces each group's dict - weight_decay included - with the
    # checkpoint's copy, so the groups can no longer be told apart by value.
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
    p.add_argument('--no-beta', dest='beta', action='store_false', help='disable output affine bias in DyT-family norms')
    p.add_argument('--alpha-jitter', type=float, default=0.05, help='per-channel alpha init jitter for DyTC (0 for deterministic)')
    p.add_argument('--gradient-checkpointing', action='store_true', help='enable gradient checkpointing')
    p.add_argument('--tie-embeddings', dest='tie_embeddings', action='store_true', default=None, help='tie input/output embeddings (default: GPT-2 yes, LLaMA no)')
    p.add_argument('--no-tie-embeddings', dest='tie_embeddings', action='store_false')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
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
    p.add_argument('--output-dir', default='runs/gpt2-124m-dytcd')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stats-every', type=int, default=500)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--wandb-project', default=None)
    p.add_argument('--run-name', default=None)
    p.add_argument('--resume', default=None, help='checkpoint to resume from')
    apply_config_to_parser(p, known.config, _TRAIN_CONFIG_ALIASES)
    args = p.parse_args()
    explicit = frozenset((name for name in _SCHEDULE_ARGS + _RESUME_RUN_ARGS if getattr(args, name, None) != p.get_default(name)))
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
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if args.bf16 and device == 'cuda' else torch.float32
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if dtype != torch.float32 else nullcontext()
    model_cfg = dict(norm_name=args.norm, qk_norm=args.qk_norm, qk_alpha_init=args.qk_alpha_init, use_layerscale=not args.no_layerscale, alpha_0=args.alpha_0, k_init=args.k_init, alpha_init=args.alpha_init, p_init=args.p_init, learnable_p=not args.no_learnable_p, learnable_k=not args.no_learnable_k, beta=args.beta, alpha_jitter=args.alpha_jitter, gradient_checkpointing=args.gradient_checkpointing)
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
    loader = build_dataloader(args.dataset, args.dataset_config, args.seq_len, args.batch_size, tokenizer=args.tokenizer)
    data_iter = iter(loader)
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
    opt = build_optimizer(model, args.lr, args.weight_decay, args.beta1, args.beta2)
    sched = LambdaLR(opt, build_lr_lambda(args))
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model'])
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
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        attach_stats = (step + 1) % args.stats_every == 0
        if attach_stats:
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        loss_acc = None
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
            loss = loss / args.grad_accum
            loss.backward()
            if loss_acc is None:
                loss_acc = loss.detach()
            else:
                loss_acc = loss_acc + loss.detach()
        if attach_stats:
            act_stats.detach()
        loss_val = loss_acc.item()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        loss_spike.update(loss_val)
        step += 1
        if step % args.log_every == 0:
            lr = opt.param_groups[0]['lr']
            line = f'step={step:>6d} loss={loss_val:8.4f} lr={lr:.2e}'
            print(line, flush=True)
            row = {'step': step, 'loss': loss_val, 'lr': lr, 'n_loss_spikes': loss_spike.n_spikes}
            with open(log_path, 'a') as f:
                f.write(json.dumps(row) + '\n')
            if tb:
                tb.add_scalar('train/loss', loss_val, step)
                tb.add_scalar('train/lr', lr, step)
                tb.add_scalar('train/n_loss_spikes', loss_spike.n_spikes, step)
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
            ckpt = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'loss_spike': loss_spike.state_dict()}
            path = Path(args.output_dir) / f'ckpt_{step:06d}.pt'
            torch.save(ckpt, path)
            print(f'[ckpt] saved {path}')
    ckpt = {'step': step, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(), 'args': vars(args), 'n_loss_spikes': loss_spike.n_spikes, 'loss_spike': loss_spike.state_dict()}
    path = Path(args.output_dir) / 'ckpt_final.pt'
    torch.save(ckpt, path)
    print(f'[done] saved final ckpt to {path}, time = {time.time() - t0:.1f}s')
    if tb:
        tb.close()
    if args.wandb_project:
        import wandb
        wandb.finish()
if __name__ == '__main__':
    main()
