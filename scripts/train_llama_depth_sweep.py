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


def launch_one(args, depth: int, norm: str, output_dir: Path, additional: list | None = None):
    """Launch a single training run via subprocess (so that we can use
    separate CUDA contexts and avoid OOM fragmentation between configs)."""
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
        "--warmup", str(args.warmup),
        "--weight-decay", str(args.weight_decay),
        "--beta1", str(args.beta1),
        "--beta2", str(args.beta2),
        "--alpha-0", str(args.alpha_0),
        "--alpha-init", str(args.alpha_init),
        "--k-init", str(args.k_init),
        "--p-init", str(args.p_init),
        "--output-dir", str(output_dir),
        "--log-every", str(args.log_every),
        "--stats-every", str(args.stats_every),
        "--save-every", str(args.save_every),
    ]
    if args.no_learnable_p:
        cmd.append("--no-learnable-p")
    if args.no_learnable_k:
        cmd.append("--no-learnable-k")
    if args.no_layerscale:
        cmd.append("--no-layerscale")
    if additional:
        cmd.extend(additional)
    # depth override - we monkeypatch by setting an env var that the
    # train_gpt2.py script picks up via GPT_DEPTH env-var.
    env = dict(os.environ)
    env["ADASQUASH_DEPTH_OVERRIDE"] = str(depth)
    print(f"[sweep] launching depth={depth} norm={norm} -> {output_dir}")
    return subprocess.Popen(cmd, env=env, stdout=open(output_dir / "stdout.log", "w"),
                            stderr=subprocess.STDOUT)


def main():
    p = argparse.ArgumentParser()
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
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--alpha-0", type=float, default=0.9)
    p.add_argument("--alpha-init", type=float, default=0.5)
    p.add_argument("--k-init", type=float, default=1.0)
    p.add_argument("--p-init", type=float, default=2.0)
    p.add_argument("--no-learnable-p", action="store_true")
    p.add_argument("--no-learnable-k", action="store_true")
    p.add_argument("--no-layerscale", action="store_true")
    p.add_argument("--output-dir", default="runs/depth-sweep")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--stats-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--sequential", action="store_true", help="run configs sequentially")
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
