"""
Phase 1 training driver: GPT-2 124M on FineWeb-Edu (or any HF dataset).

Supports:
  - Pre-LN, RMSNorm, DyT, DyT-C, DyT-D, DyT-CD via --norm
  - QK-Norm / DyT-QK via --qk-norm
  - LayerScale toggle via --no-layerscale

  - Wandb / TensorBoard logging
  - bf16 mixed precision
  - Gradient accumulation
  - Checkpoint resume

Example:
    python scripts/train_gpt2.py \
        --norm dytcd \
        --dataset HuggingFWC/fineweb-edu \
        --dataset-config sample-10BT \
        --max-steps 10000 \
        --batch-size 32 --seq-len 1024 \
        --grad-accum 4 \
        --lr 6e-4 --warmup 500 --min-lr 6e-5 \
        --output-dir runs/gpt2-124m-dytcd
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

# Make the project importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from adasquash import (
    GPTConfig, GPT, build_norm, count_parameters,
    ActivationStats, LossSpikeCounter,
)
from adasquash.model import build_gpt_124m, build_gpt_350m, build_llama_tiny_1b


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataloader(
    dataset_name: str,
    dataset_config: str | None,
    seq_len: int,
    batch_size: int,
    tokenizer: str = "gpt2",
):
    """Build an infinite, tokenized, packed-stream DataLoader from HF datasets."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader, IterableDataset

    tok = AutoTokenizer.from_pretrained(tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    class PackedStream(IterableDataset):
        def __init__(self, ds, tok, seq_len):
            self.ds = ds
            self.tok = tok
            self.seq_len = seq_len
            self.buf: list = []

        def __iter__(self):
            # Shard the document stream across DataLoader workers: worker i
            # consumes every i-th document, so the shards stay disjoint.
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            n_workers = worker_info.num_workers if worker_info is not None else 1
            # Operate on `self.buf` *in place* so the unconsumed tail is
            # carried over to the next call.
            buf = self.buf
            for i, ex in enumerate(self.ds):
                if i % n_workers != worker_id:
                    continue
                ids = self.tok(ex.get("text", ""), add_special_tokens=False)["input_ids"]
                buf.extend(ids)
                buf.append(self.tok.eos_token_id)
                while len(buf) >= self.seq_len + 1:
                    chunk = buf[: self.seq_len + 1]
                    del buf[: self.seq_len]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y

    ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
    stream = PackedStream(ds, tok, seq_len)
    return DataLoader(
        stream,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Config plumbing
# ---------------------------------------------------------------------------
def load_yaml_config(path: str) -> dict:
    """Read a YAML run config. Keys are returned as-is (snake_case)."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config '{path}' must be a mapping, got {type(data)}")
    return data


def apply_config_to_parser(parser, path: str | None, aliases: dict | None = None):
    """Seed `parser` defaults from a YAML config.

    `aliases` maps a config key onto `(argparse_dest, invert)`. `invert` is
    for YAML keys that are positive (`use_layerscale: false`) while the flag
    is negative (`--no-layerscale`).

    Explicit command-line flags still win, because argparse only overrides a
    default that the user actually supplied; every key the config names is
    taken from the file. Note that `use_layerscale` is positive in YAML while
    the flag is negative (`--no-layerscale`), which is what `invert` is for.
    """
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
        print(f"[config] {path}: ignoring unknown keys {sorted(unknown)}")


# YAML key -> (argparse dest, invert) for the negative-form flags.
_TRAIN_CONFIG_ALIASES = {
    "use_layerscale": ("no_layerscale", True),
    "learnable_p": ("no_learnable_p", True),
    "learnable_k": ("no_learnable_k", True),
    "tie_word_embeddings": ("tie_embeddings", False),
    "bf16": ("no_bf16", True),
}


# ---------------------------------------------------------------------------
# Optimizer & schedule
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, lr: float, weight_decay: float, beta1: float, beta2: float):
    decay = []
    no_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "norm" in name or "delta" in name or "ls1" in name or "ls2" in name or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=(beta1, beta2))


def cosine_schedule(step: int, max_steps: int, warmup: int, min_lr_ratio: float):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Main train loop
# ---------------------------------------------------------------------------
def main():
    # Pre-parse `--config` so the YAML can seed the parser defaults; the
    # real parse below then lets explicit CLI flags override them.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML run config; CLI flags override it")
    # Model
    p.add_argument("--model", choices=["gpt2-124m", "gpt2-350m", "llama-1b"], default="gpt2-124m")
    p.add_argument("--norm", default="dytcd", choices=list(["ln", "rmsnorm", "dyt", "dytc", "dytd", "dytcd"]))
    p.add_argument("--qk-norm", default=None, choices=[None, "qknorm", "dytqk"])
    p.add_argument("--no-layerscale", action="store_true")
    p.add_argument("--layerscale-init", type=float, default=None)
    p.add_argument("--alpha-0", type=float, default=0.9)
    p.add_argument("--alpha-init", type=float, default=0.5)
    p.add_argument("--k-init", type=float, default=1.0)
    p.add_argument("--p-init", type=float, default=2.0)
    p.add_argument("--no-learnable-p", action="store_true")
    p.add_argument("--no-learnable-k", action="store_true")
    # Embedding tying is per-model: GPT-2 ties, LLaMA does not. The builders
    # already encode the right default, so the flag is tri-state - `None`
    # means "leave the builder's default alone". The previous
    # `store_true, default=True` could never turn anything off.
    p.add_argument("--tie-embeddings", dest="tie_embeddings",
                   action="store_true", default=None,
                   help="tie input/output embeddings (default: GPT-2 yes, LLaMA no)")
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings",
                   action="store_false")
    # Data
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32, help="per-device micro batch")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--tokenizer", default="gpt2")
    # Opt
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast")
    p.add_argument("--seed", type=int, default=42)
    # Logging / IO
    p.add_argument("--output-dir", default="runs/gpt2-124m-dytcd")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--stats-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    apply_config_to_parser(p, known.config, _TRAIN_CONFIG_ALIASES)
    args = p.parse_args()
    args.bf16 = not args.no_bf16

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device == "cuda" else torch.float32
    ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if dtype != torch.float32 else nullcontext()
    )

    # Build model
    model_cfg = dict(
        norm_name=args.norm,
        qk_norm=args.qk_norm,
        use_layerscale=not args.no_layerscale,
        alpha_0=args.alpha_0,
        k_init=args.k_init,
        alpha_init=args.alpha_init,
        p_init=args.p_init,
        learnable_p=not args.no_learnable_p,
        learnable_k=not args.no_learnable_k,
    )
    if args.layerscale_init is not None:
        model_cfg["layerscale_init"] = args.layerscale_init
    if args.tie_embeddings is not None:
        model_cfg["tie_word_embeddings"] = args.tie_embeddings
    if args.model == "gpt2-124m":
        model = build_gpt_124m(**model_cfg)
    elif args.model == "gpt2-350m":
        model = build_gpt_350m(**model_cfg)
    else:
        model = build_llama_tiny_1b(**model_cfg)

    # Depth override (used by the depth-scaling sweep launcher). This lets
    # us run 12/24/48/96-layer variants without writing a separate script.
    depth_override = os.environ.get("ADASQUASH_DEPTH_OVERRIDE")
    dim_override = os.environ.get("ADASQUASH_DIM_OVERRIDE")
    heads_override = os.environ.get("ADASQUASH_HEADS_OVERRIDE")
    if depth_override or dim_override or heads_override:
        cfg_dict = {k: getattr(model.cfg, k) for k in vars(model.cfg)}
        if depth_override:
            cfg_dict["n_layers"] = int(depth_override)
            args._depth_override = int(depth_override)
        if dim_override:
            cfg_dict["dim"] = int(dim_override)
            args._dim_override = int(dim_override)
        if heads_override:
            cfg_dict["n_heads"] = int(heads_override)
            args._heads_override = int(heads_override)
        print(f"[shape-override] L={cfg_dict['n_layers']} dim={cfg_dict['dim']} "
              f"n_heads={cfg_dict['n_heads']} (was L={model.cfg.n_layers} "
              f"dim={model.cfg.dim} n_heads={model.cfg.n_heads})")
        model = GPT(GPTConfig(**cfg_dict))
    model.reset_parameters(args.seed)
    model.to(device)
    n_params = count_parameters(model)
    n_trainable = count_parameters(model, trainable_only=True)
    print(f"[model] {args.model} | total={n_params/1e6:.2f}M | trainable={n_trainable/1e6:.2f}M | norm={args.norm}")
    print(f"[model] norm param breakdown: {model.norm_param_count()}")

    # Data
    loader = build_dataloader(
        args.dataset, args.dataset_config, args.seq_len,
        args.batch_size, tokenizer=args.tokenizer,
    )
    data_iter = iter(loader)

    # Optimizer
    opt = build_optimizer(model, args.lr, args.weight_decay, args.beta1, args.beta2)
    sched = LambdaLR(
        opt,
        lambda s: cosine_schedule(s, args.max_steps, args.warmup, args.min_lr_ratio),
    )

    # Resume: restore model / optimizer / scheduler and the step counter so
    # the LR schedule continues where it stopped instead of replaying warmup.
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] {args.resume} -> step {start_step}")

    # Stability meters
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter(rel_threshold=0.20, abs_threshold=0.05)
    _ = loss_spike  # placeholder - keep minimal logging

    # Logging setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = Path(args.output_dir) / "train.jsonl"
    log_path.touch()
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(Path(args.output_dir) / "tb"))
    except Exception:
        pass
    if args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or args.norm,
            config=vars(args),
        )

    # Training loop
    step = start_step
    t0 = time.time()
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        # Activation stats over one macro-step is enough. The window is keyed
        # on the *post-increment* step so that the hooks are attached for
        # exactly the step whose summary is reported below; keying on the
        # pre-increment step would attach at step 0/500/1000 but report at
        # 500/1000/1500, labelling every number with a step it was not
        # measured on.
        attach_stats = ((step + 1) % args.stats_every == 0)
        if attach_stats:
            act_stats.attach()
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
                logits, loss = model(x, y)
            del logits
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
            lr = opt.param_groups[0]["lr"]
            line = (
                f"step={step:>6d} loss={loss_val:8.4f} lr={lr:.2e}"
            )
            print(line, flush=True)
            row = {"step": step, "loss": loss_val, "lr": lr}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if tb:
                tb.add_scalar("train/loss", loss_val, step)
                tb.add_scalar("train/lr", lr, step)
            if args.wandb_project:
                import wandb
                wandb.log(row, step=step)

        if step % args.stats_every == 0 and step > 0:
            stats = act_stats.summary()
            print(f"[stats] {stats.get('__summary__', {})}")
            if tb:
                s = stats.get("__summary__", {})
                for k, v in s.items():
                    tb.add_scalar(f"stats/{k}", v, step)


        if step % args.save_every == 0 and step > 0:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "args": vars(args),
            }
            path = Path(args.output_dir) / f"ckpt_{step:06d}.pt"
            torch.save(ckpt, path)
            print(f"[ckpt] saved {path}")

    # final save
    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "args": vars(args),
    }
    path = Path(args.output_dir) / "ckpt_final.pt"
    torch.save(ckpt, path)
    print(f"[done] saved final ckpt to {path}, time = {time.time()-t0:.1f}s")
    if tb:
        tb.close()
    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
