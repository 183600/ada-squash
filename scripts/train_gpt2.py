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
        --dataset HuggingFaceFW/fineweb-edu \
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
        # Documents are tokenized a batch at a time: the fast tokenizer's
        # Rust path parallelizes over a list of inputs, which a
        # one-document-at-a-time call never gets to use.
        tokenize_batch = 64

        def __init__(self, ds, tok, seq_len):
            self.ds = ds
            self.tok = tok
            self.seq_len = seq_len
            self.buf: list = []
            self.pos = 0

        def __iter__(self):
            # Shard the document stream across DataLoader workers: worker i
            # consumes every i-th document, so the shards stay disjoint.
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            n_workers = worker_info.num_workers if worker_info is not None else 1
            # Operate on `self.buf` *in place* so the unconsumed tail is
            # carried over to the next call. `pos` walks the buffer instead
            # of deleting its prefix after every chunk: with packed streams
            # the tail is much longer than one chunk, so front-deleting it
            # once per chunk re-shifts the whole tail and makes tokenizing a
            # document quadratic in its length.
            buf = self.buf
            pos = self.pos
            need = self.seq_len + 1
            texts: list = []

            def drain():
                nonlocal pos
                while len(buf) - pos >= need:
                    x = torch.tensor(
                        buf[pos:pos + self.seq_len], dtype=torch.long
                    )
                    y = torch.tensor(
                        buf[pos + 1:pos + need], dtype=torch.long
                    )
                    pos += self.seq_len
                    yield x, y
                # Safe to compact unconditionally: the loop above stops as
                # soon as a partial chunk is left, so the tail is at most
                # `need` tokens regardless of how long the last document was.
                del buf[:pos]
                pos = 0

            def extend(batch):
                for ids in self.tok(batch, add_special_tokens=False)["input_ids"]:
                    buf.extend(ids)
                    buf.append(self.tok.eos_token_id)

            i = 0
            for ex in self.ds:
                if i % n_workers != worker_id:
                    i += 1
                    continue
                i += 1
                texts.append(ex.get("text", ""))
                if len(texts) < self.tokenize_batch and len(buf) - pos < need:
                    continue
                extend(texts)
                texts = []
                yield from drain()
            if texts:
                extend(texts)
                yield from drain()
            self.pos = pos

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
    # The fused multi-tensor step issues one kernel launch per device and
    # needs no per-parameter temporary, so it is both faster and leaner than
    # the default foreach path. It is CUDA-only and needs a single device.
    devices = {p.device for p in decay + no_decay}
    fused = len(devices) == 1 and next(iter(devices)).type == "cuda"
    return AdamW(groups, lr=lr, betas=(beta1, beta2), fused=fused)


def cosine_schedule(step: int, max_steps: int, warmup: int, min_lr_ratio: float):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_lr_lambda(args):
    """LR multiplier as a function of `LambdaLR.last_epoch`.

    The `+ 1` is load-bearing: `LambdaLR` evaluates the lambda for step 0 at
    construction, so with a bare `step` the very first `opt.step()` runs at
    lr == 0 (a wasted update) and the whole warmup is shifted one step late.
    """
    return lambda s: cosine_schedule(
        s + 1, args.max_steps, args.warmup, args.min_lr_ratio
    )


# Hyper-parameters that define the LR trajectory. A resumed run has to keep
# every one of them, otherwise the schedule silently changes shape mid-run
# even though `LambdaLR.load_state_dict` restores `last_epoch` faithfully.
_SCHEDULE_ARGS = (
    "max_steps",
    "warmup",
    "min_lr_ratio",
    "lr",
    "weight_decay",
    "beta1",
    "beta2",
    "grad_clip",
)


def restore_schedule_args(
    args, ckpt_args: dict, explicit: frozenset = frozenset()
) -> list:
    """Adopt the original run's schedule / optimizer hyper-parameters.

    Returns the names where the command line and the checkpoint disagree.
    `max_steps` is included on purpose: extending a finished run (resume with
    a larger `--max-steps` to continue the cosine tail) changes the schedule
    by design, so such a divergence is recorded rather than rejected -
    silently, it would look like an uninterrupted run.

    A name the caller flags as *explicit* (passed on the command line, so
    not just a default that would otherwise be adopted) is left alone: the
    point of the restore is to keep a resumed run from silently reshaping the
    curve it claims to continue, not to make an explicit extension
    impossible. The disagreement is still returned so the caller can log it.
    """
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
    p.add_argument("--qk-alpha-init", type=float, default=1.0,
                   help="initial alpha_q / alpha_k for --qk-norm dytqk")
    p.add_argument("--no-layerscale", action="store_true")
    p.add_argument("--layerscale-init", type=float, default=None)
    p.add_argument("--alpha-0", type=float, default=0.9)
    p.add_argument("--alpha-init", type=float, default=0.5)
    p.add_argument("--k-init", type=float, default=1.0)
    p.add_argument("--p-init", type=float, default=2.0)
    p.add_argument("--no-learnable-p", action="store_true")
    p.add_argument("--no-learnable-k", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true", help="enable gradient checkpointing")
    # Embedding tying is per-model: GPT-2 ties, LLaMA does not. The builders
    # already encode the right default, so the flag is tri-state - `None`
    # means "leave the builder's default alone".
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
    # Compile
    p.add_argument("--compile", action="store_true", help="enable torch.compile")
    p.add_argument("--compile-mode", default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
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
    # Which schedule flags the operator actually typed, as opposed to the
    # defaults left over after the config seeded them (`set_defaults` above
    # already folded the config into `get_default`). Only an explicit flag is
    # allowed to diverge from the checkpoint's schedule on a resume - that is
    # how a run is extended; a default that merely happens to differ gets
    # replaced, because it is what would silently reshape the curve.
    explicit = frozenset(
        name for name in _SCHEDULE_ARGS
        if getattr(args, name, None) != p.get_default(name)
    )
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
        qk_alpha_init=args.qk_alpha_init,
        use_layerscale=not args.no_layerscale,
        alpha_0=args.alpha_0,
        k_init=args.k_init,
        alpha_init=args.alpha_init,
        p_init=args.p_init,
        learnable_p=not args.no_learnable_p,
        learnable_k=not args.no_learnable_k,
        gradient_checkpointing=args.gradient_checkpointing,
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

    # torch.compile for faster training (requires PyTorch 2.0+)
    if args.compile:
        print(f"[compile] enabling torch.compile with mode={args.compile_mode}")
        model = torch.compile(model, mode=args.compile_mode)

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

    # Resume: restore model / optimizer / scheduler and the step counter so
    # the LR schedule continues where it stopped instead of replaying warmup.
    # The lambda closes over `max_steps` / `warmup` / `min_lr_ratio`, and
    # `load_state_dict` only restores `last_epoch`, so those values have to
    # come back from the checkpoint too - otherwise a resumed run silently
    # follows a different curve than the run it claims to continue.
    start_step = 0
    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        changed = restore_schedule_args(
            args, resume_ckpt.get("args") or {}, explicit=explicit
        )
        if changed:
            kept = [n for n in changed if n in explicit]
            adopted = [n for n in changed if n not in explicit]
            if adopted:
                print(f"[resume] schedule args restored from checkpoint: {adopted}")
            if kept:
                print(f"[resume] keeping your explicit override(s) {kept} - the "
                      f"resumed run therefore follows a different schedule "
                      f"than the one it continues")
        start_step = int(resume_ckpt.get("step", 0))
        print(f"[resume] {args.resume} -> step {start_step}")

    # Optimizer
    opt = build_optimizer(model, args.lr, args.weight_decay, args.beta1, args.beta2)
    sched = LambdaLR(opt, build_lr_lambda(args))

    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model"])
        if "optimizer" in resume_ckpt:
            opt.load_state_dict(resume_ckpt["optimizer"])
        if "scheduler" in resume_ckpt:
            sched.load_state_dict(resume_ckpt["scheduler"])

    # Stability meters
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter(rel_threshold=0.20, abs_threshold=0.05)
    if resume_ckpt is not None:
        # The counter is logged once per `log_every` steps as a running count
        # over the run, so a resumed run has to continue from the saved count
        # (and the saved EMA baseline) instead of starting a new series that
        # then looks like the whole history.
        if loss_spike.load_state_dict(resume_ckpt.get("loss_spike")):
            print(f"[resume] restored loss-spike counter "
                  f"(n_spikes={loss_spike.n_spikes})")
        elif "n_loss_spikes" in resume_ckpt:
            # Checkpoints written before the full state was saved carry the
            # count only; the EMA still has to rebuild, which the next update
            # does from the first resumed loss.
            loss_spike.n_spikes = int(resume_ckpt["n_loss_spikes"] or 0)
            print(f"[resume] restored loss-spike counter from n_loss_spikes "
                  f"(n_spikes={loss_spike.n_spikes}, EMA warm restarted)")

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
        # exactly the step whose summary is reported below.
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
                # The logit block is fused into the loss, so the projection's
                # [B, T, V] output - the largest tensor in the model - is
                # never materialized.
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
            lr = opt.param_groups[0]["lr"]
            line = (
                f"step={step:>6d} loss={loss_val:8.4f} lr={lr:.2e}"
            )
            print(line, flush=True)
            row = {
                "step": step,
                "loss": loss_val,
                "lr": lr,
                # Running counter over this run: resumed runs continue from
                # the checkpoint's saved count (see the `loss_spike` entry in
                # every checkpoint), so it covers the whole run and not just
                # the post-resume window.
                "n_loss_spikes": loss_spike.n_spikes,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            if tb:
                tb.add_scalar("train/loss", loss_val, step)
                tb.add_scalar("train/lr", lr, step)
                tb.add_scalar("train/n_loss_spikes", loss_spike.n_spikes, step)
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
                # Kept in *every* checkpoint, not only the final one: a run
                # that is resumed from a periodic checkpoint would otherwise
                # drop the spike history written before it.
                "loss_spike": loss_spike.state_dict(),
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
        "n_loss_spikes": loss_spike.n_spikes,
        "loss_spike": loss_spike.state_dict(),
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
