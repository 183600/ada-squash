"""
Ablation runner: per-channel, depth-conditioned, LayerScale, tanh vs
softsign family, alpha initialization grid.

Each ablation is a small training run (default: GPT-2 124M, 1000 steps)
plus a single stability measurement at the end. The results land in a
JSON file under runs/ablation/<name>/summary.json.

Designed to be runnable on a single H100 in <2 hours total.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.optim import AdamW

from adasquash import (
    GPTConfig, GPT, build_gpt_124m, count_parameters,
    ActivationStats, LossSpikeCounter,
)


ABLATIONS = {
    # ---- Per-channel ----
    "dyt_no_perchannel": dict(norm="dyt"),
    "dytc_no_perchannel": dict(norm="dytc", learnable_p=False),
    "dytcd_full": dict(norm="dytcd"),
    # ---- Depth-cond ----
    "dyt_no_depth": dict(norm="dyt"),
    "dytd_k0": dict(norm="dytd", k_init=0.0),
    "dytd_full": dict(norm="dytd"),
    "dytcd_full": dict(norm="dytcd"),
    # ---- LayerScale ----
    "dytcd_no_ls": dict(norm="dytcd", use_layerscale=False),
    "dytcd_ls1e5": dict(norm="dytcd", layerscale_init=1e-5),
    "dytcd_ls1e3": dict(norm="dytcd", layerscale_init=1e-3),
    # ---- tanh vs softsign family ----
    "dytc_p1_softsign": dict(norm="dytc", p_init=1.0, learnable_p=False),
    "dytc_p2_softsign": dict(norm="dytc", p_init=2.0, learnable_p=False),
    "dytc_p4_softsign": dict(norm="dytc", p_init=4.0, learnable_p=False),
    "dytc_plearnable": dict(norm="dytc", p_init=2.0, learnable_p=True),
    "dytc_tanh_eq": dict(norm="dytc", p_init=64.0, learnable_p=False),  # p->inf == tanh
    # ---- alpha init grid ----
    "dyt_alpha_0.3": dict(norm="dyt", alpha_init=0.3),
    "dyt_alpha_0.5": dict(norm="dyt", alpha_init=0.5),
    "dyt_alpha_1.0": dict(norm="dyt", alpha_init=1.0),
}


def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path):
    print(f"\n[abl] {name}: {cfg_kwargs}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_gpt_124m(**cfg_kwargs)
    model.to(device)
    n_params = count_parameters(model)

    # Synthetic data: random token ids are good enough for ablations that
    # are about stability, not quality.
    rng = torch.Generator(device=device).manual_seed(args.seed)
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()

    losses = []
    t0 = time.time()
    for step in range(args.max_steps):
        x = torch.randint(0, model.cfg.vocab_size, (args.batch_size, args.seq_len), generator=rng, device=device)
        y = x.clone()  # auto-regressive target: predict the next token (here just for stability)
        y = torch.cat([x[:, 1:], torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)], dim=1)

        if step % args.stats_every == 0:
            act_stats.attach()
        out = model(x, y)
        loss = out[1] if isinstance(out, tuple) else out[1]
        loss.backward()
        if step % args.stats_every == 0:
            act_stats.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_spike.update(loss.item())
        losses.append(loss.item())
        if step % args.log_every == 0:
            print(f"  [step {step:4d}] loss={loss.item():.4f} spikes={loss_spike.n_spikes}", flush=True)

    stats = act_stats.summary()
    summary = {
        "name": name,
        "cfg": cfg_kwargs,
        "n_params": n_params,
        "n_spikes": loss_spike.n_spikes,
        "final_loss": losses[-1],
        "mean_last_100_loss": sum(losses[-100:]) / 100,
        "max_abs_summary": stats.get("__summary__", {}).get("max_max_abs", 0.0),
        "mean_kurtosis": stats.get("__summary__", {}).get("mean_kurtosis", 0.0),
        "elapsed_s": time.time() - t0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [done] {summary}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", default=list(ABLATIONS.keys()))
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--stats-every", type=int, default=200)
    p.add_argument("--output-dir", default="runs/ablation")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for name in args.names:
        if name not in ABLATIONS:
            print(f"[warn] unknown ablation '{name}', skipping")
            continue
        run_dir = base / name
        if (run_dir / "summary.json").exists() and not args.force:
            print(f"[skip] {name} already done, use --force to rerun")
            continue
        s = run_one(name, ABLATIONS[name], args, run_dir)
        all_summaries.append(s)

    with open(base / "all_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[done] {len(all_summaries)} ablations saved to {base}")


if __name__ == "__main__":
    main()
