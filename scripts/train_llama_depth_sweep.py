"""
Phase 2 training driver: LLaMA-tiny 1.1B with cosine cooldown, plus depth
scaling sweep launcher (12/24/48/96 layers).

The depth-scaling sweep is the headline experiment for the DyT-D / DyT-CD
story: shallow vs deep stability and the "curse of depth" curve.

Example depth sweep:
    python scripts/train_llama_depth_sweep.py \
        --depths 12 24 48 96 \
        --max-steps 5000 \
        --output-dir runs/depth-sweep
"""

from __future__ import annotations

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


def _cosine_lr(step, max_steps, warmup, min_lr_ratio):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


# Per-depth (dim, n_heads) taken from the Phase-2 experiment card
# (experiments/phase2_llama_1b_depth/README.md). The point of the sweep is
# the *depth* axis, so `dim` has to shrink as depth grows or the 96-layer
# run is simply a 4x larger model and "RMSNorm diverges at depth" is
# unfalsifiable. head_dim is 128 for every row, so RoPE / attention shapes
# stay identical across the sweep.
DEPTH_SHAPE = {
    12: (2560, 20),
    24: (2048, 16),
    48: (1536, 12),
    96: (1280, 10),
}
_BASE_DEPTH = 24
_BASE_SHAPE = (2048, 16)  # LLaMA-tiny 1b defaults

# YAML key -> (argparse dest, invert) for the negative-form flags.
_SWEEP_CONFIG_ALIASES = {
    "use_layerscale": ("no_layerscale", True),
    "learnable_p": ("no_learnable_p", True),
    "learnable_k": ("no_learnable_k", True),
    "bf16": ("no_bf16", True),
    "tie_word_embeddings": ("tie_embeddings", False),
}


def shape_for_depth(depth: int, base_dim: int, base_heads: int):
    """(dim, n_heads) for a given depth, keeping the parameter count ~fixed.

    Falls back to a 1/sqrt(depth) dim schedule (with head_dim held fixed)
    for depths that are not in the documented table.
    """
    if depth in DEPTH_SHAPE:
        return DEPTH_SHAPE[depth]
    base_head_dim = max(1, base_dim // base_heads)
    dim = int(round(base_dim * math.sqrt(_BASE_DEPTH / max(1, depth))))
    heads = max(1, int(round(dim / base_head_dim)))
    dim = heads * base_head_dim
    return dim, heads


def launch_one(args, depth: int, norm: str, output_dir: Path, additional: list | None = None):
    """Launch a single training run via subprocess."""
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_gpt2.py"),
        "--model", args.model,
        "--norm", norm,
        "--dataset", args.dataset,
        "--dataset-config", args.dataset_config,
        "--tokenizer", args.tokenizer,
        "--seq-len", str(args.seq_len),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--max-steps", str(args.max_steps),
        "--lr", str(args.lr),
        "--min-lr-ratio", str(args.min_lr_ratio),
        "--warmup", str(args.warmup),
        "--weight-decay", str(args.weight_decay),
        "--beta1", str(args.beta1),
        "--beta2", str(args.beta2),
        "--grad-clip", str(args.grad_clip),
        "--seed", str(args.seed),
        "--alpha-0", str(args.alpha_0),
        "--alpha-init", str(args.alpha_init),
        "--k-init", str(args.k_init),
        "--p-init", str(args.p_init),
        "--output-dir", str(output_dir),
        "--log-every", str(args.log_every),
        "--stats-every", str(args.stats_every),
        "--save-every", str(args.save_every),
    ]
    # Forward the remaining knobs. Omitting any of these made every sweep
    # arm silently fall back to the training script's defaults, so e.g. a
    # sweep asking for `--qk-norm dytqk` or a non-unit `--grad-clip` would
    # have run with neither.
    if args.qk_norm:
        cmd += ["--qk-norm", args.qk_norm]
    if args.layerscale_init is not None:
        cmd += ["--layerscale-init", str(args.layerscale_init)]
    if args.no_bf16:
        cmd.append("--no-bf16")
    if args.tie_embeddings is not None:
        cmd.append("--tie-embeddings" if args.tie_embeddings
                   else "--no-tie-embeddings")
    if args.no_learnable_p:
        cmd.append("--no-learnable-p")
    if args.no_learnable_k:
        cmd.append("--no-learnable-k")
    if args.no_layerscale:
        cmd.append("--no-layerscale")
    if additional:
        cmd.extend(additional)
    # depth / width override - train_gpt2.py picks these up from the env.
    env = dict(os.environ)
    env["ADASQUASH_DEPTH_OVERRIDE"] = str(depth)
    shape = ""
    if not args.no_dim_scaling:
        base_dim, base_heads = _BASE_SHAPE if args.model == "llama-1b" else (768, 12)
        dim, heads = shape_for_depth(depth, base_dim, base_heads)
        env["ADASQUASH_DIM_OVERRIDE"] = str(dim)
        env["ADASQUASH_HEADS_OVERRIDE"] = str(heads)
        shape = f" (dim={dim}, heads={heads})"
    print(f"[sweep] launching depth={depth} norm={norm} -> {output_dir}{shape}")
    log = open(output_dir / "stdout.log", "w")
    try:
        return subprocess.Popen(cmd, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
    except Exception:
        log.close()
        raise


def main():
    from scripts.train_gpt2 import apply_config_to_parser

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML run config; CLI flags override it")
    p.add_argument("--model", default="llama-1b")
    p.add_argument("--depths", nargs="+", type=int, default=[12, 24, 48, 96])
    p.add_argument("--norms", nargs="+", default=["rmsnorm", "dyt", "dytcd"])
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset-config", default="sample-10BT")
    p.add_argument("--tokenizer", default="HuggingFaceFW/fineweb-edu/llama-tokenizer")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qk-norm", default=None, choices=[None, "qknorm", "dytqk"])
    p.add_argument("--layerscale-init", type=float, default=None)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--tie-embeddings", dest="tie_embeddings",
                   action="store_true", default=None)
    p.add_argument("--no-tie-embeddings", dest="tie_embeddings",
                   action="store_false")
    p.add_argument("--alpha-0", type=float, default=0.9)
    p.add_argument("--alpha-init", type=float, default=0.5)
    p.add_argument("--k-init", type=float, default=1.0)
    p.add_argument("--p-init", type=float, default=2.0)
    p.add_argument("--no-learnable-p", action="store_true")
    p.add_argument("--no-learnable-k", action="store_true")
    p.add_argument("--no-layerscale", action="store_true")
    p.add_argument(
        "--no-dim-scaling",
        action="store_true",
        help="keep dim/n_heads fixed across depths (param count then grows "
             "linearly with depth, which confounds the depth axis)",
    )
    p.add_argument("--output-dir", default="runs/depth-sweep")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--stats-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--sequential", action="store_true", help="run configs sequentially")
    apply_config_to_parser(p, known.config, _SWEEP_CONFIG_ALIASES)
    args = p.parse_args()

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    procs = []
    for depth in args.depths:
        for norm in args.norms:
            out_dir = base / f"d{depth}-{norm}"
            out_dir.mkdir(parents=True, exist_ok=True)
            # Save the depth-override scriptlet in the run dir so the
            # subprocess can read it. We pass via env var, so this is purely
            # a documentation artifact.
            (out_dir / "depth.txt").write_text(str(depth))
            proc = launch_one(args, depth, norm, out_dir)
            if args.sequential:
                rc = proc.wait()
                print(f"[sweep] finished d{depth}-{norm} rc={rc}")
            else:
                procs.append(proc)
    if not args.sequential:
        for proc in procs:
            proc.wait()
    print("[sweep] done")


if __name__ == "__main__":
    main()
