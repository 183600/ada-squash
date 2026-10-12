"""
Ablation runner: per-channel, depth-conditioned, LayerScale, tanh vs
softsign family, alpha initialization grid.

Each ablation is a small training run (default: GPT-2 124M, 1000 steps)
on a synthetic Markov corpus plus a stability measurement at the end. The
results land in a JSON file under runs/ablation/<name>/summary.json.

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

from adasquash import (
    GPTConfig, GPT, build_gpt_124m, count_parameters,
    ActivationStats, LossSpikeCounter,
)
from scripts.train_gpt2 import apply_config_to_parser, build_optimizer, load_yaml_config


# Component ablations. `dyt` / `dytd` are the "axis removed" references, so
# each row must differ from `dytcd_full` in exactly one component.
ABLATIONS = {
    # ---- Per-channel ----
    "dyt_no_perchannel": dict(norm_name="dyt"),
    "dytc_perchannel": dict(norm_name="dytc", p_init=2.0, learnable_p=False),
    "dytcd_full": dict(norm_name="dytcd"),
    # ---- Depth-cond ----
    "dytd_k0_no_depth_schedule": dict(norm_name="dytd", k_init=0.0),
    "dytd_full": dict(norm_name="dytd"),
    # ---- LayerScale ----
    "dytcd_no_ls": dict(norm_name="dytcd", use_layerscale=False),
    "dytcd_ls1e5": dict(norm_name="dytcd", layerscale_init=1e-5),
    "dytcd_ls1e3": dict(norm_name="dytcd", layerscale_init=1e-3),
    # ---- tanh vs softsign family ----
    "dytc_p1_softsign": dict(norm_name="dytc", p_init=1.0, learnable_p=False),
    "dytc_p2_softsign": dict(norm_name="dytc", p_init=2.0, learnable_p=False),
    "dytc_p4_softsign": dict(norm_name="dytc", p_init=4.0, learnable_p=False),
    "dytc_plearnable": dict(norm_name="dytc", p_init=2.0, learnable_p=True),
    "dytc_tanh_eq": dict(norm_name="dytc", p_init=64.0, learnable_p=False),  # p->inf == tanh
    # ---- alpha init grid ----
    "dyt_alpha_0.3": dict(norm_name="dyt", alpha_init=0.3),
    "dyt_alpha_0.5": dict(norm_name="dyt", alpha_init=0.5),
    "dyt_alpha_1.0": dict(norm_name="dyt", alpha_init=1.0),
}


def make_batch(rng, ignore_index: int, batch_size: int, seq_len: int,
               device: str, n_states: int = 64, outlier_id: int | None = None):
    """Synthetic corpus with *learnable* structure.

    Uniformly random token ids carry no next-token signal: the optimal loss
    is log(vocab_size) no matter what the model does, so every ablation would
    report the same flat number and the loss column could not rank anything.
    Here the sequence is a Markov chain over a small hidden state set, which
    caps the attainable loss at log(n_states) instead.

    `outlier_id` designates one token whose embedding row has been scaled
    up (see `install_outlier_channel`). Positions are drawn from it on a few
    percent of steps, which reproduces the "super-active dimension" regime
    the DyT-C axis is meant to handle: an i.i.d. corpus never produces such a
    channel, so the ablations could not exercise the component under test.
    """
    n = batch_size * seq_len
    states = torch.randint(0, n_states, (n,), generator=rng, device=device)
    # Transition: the next state is the current one advanced by a small jump.
    jump = torch.randint(0, 3, (n,), generator=rng, device=device)
    nxt = (states + jump) % n_states
    x = nxt.view(batch_size, seq_len).clone()
    # Target is the shifted successor; the final position has no successor
    # and carries `ignore_index` so it is masked out of the loss.
    y = torch.full(
        (batch_size, seq_len), ignore_index, dtype=torch.long, device=device
    )
    y[:, :-1] = nxt.view(batch_size, seq_len)[:, 1:]
    if outlier_id is not None:
        spike = torch.rand((batch_size, seq_len), generator=rng, device=device) < 0.03
        x[spike] = outlier_id
    return x, y


def install_outlier_channel(model, scale: float = 25.0) -> int:
    """Scale up one embedding row to act as an activation-outlier channel.

    Returns the token id of the outlier row. Run *after* `reset_parameters`
    so the scaling is not overwritten by the initialization pass.
    """
    with torch.no_grad():
        oid = model.cfg.vocab_size - 1
        model.tok_emb.weight[oid] *= scale
    return oid


def load_ablations_from_config(path: str) -> dict:
    """Read the ablation table out of a YAML run config.

    The in-script `ABLATIONS` dict and `configs/ablation-matrix.yaml` used to
    list *different* rows under *different* names, so a run driven from the
    YAML silently skipped names the script did not know about and vice versa.
    Making the file authoritative removes that drift.
    """
    cfg = load_yaml_config(path)
    rows = cfg.get("ablations")
    if not rows:
        return {}
    out = {}
    for row in rows:
        row = dict(row)
        name = row.pop("name")
        # The YAML spells the norm selector `norm`; GPTConfig calls it
        # `norm_name`.
        if "norm" in row:
            row["norm_name"] = row.pop("norm")
        out[name] = row
    return out


def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path):
    print(f"\n[abl] {name}: {cfg_kwargs}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Seed so every ablation starts from a reproducible initialization.
    torch.manual_seed(args.seed)
    model = build_gpt_124m(**cfg_kwargs)
    model.reset_parameters(args.seed)
    model.to(device)
    n_params = count_parameters(model)

    # Same parameter grouping as the Phase-1 driver: weight decay is applied
    # to matmul weights only. Using a single group here would decay the norm
    # / LayerScale parameters too, so the ablation would not be comparable
    # with the training runs it is meant to explain.
    opt = build_optimizer(model, args.lr, args.weight_decay, 0.9, 0.95)

    rng = torch.Generator(device=device).manual_seed(args.seed)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    outlier_id = install_outlier_channel(model)

    losses = []
    t0 = time.time()
    for step in range(args.max_steps):
        x, y = make_batch(
            rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device,
            outlier_id=outlier_id,
        )
        measure = (step % args.stats_every == 0)
        if measure:
            act_stats.attach()
        logits, loss = model(x, y)
        del logits
        loss.backward()
        if measure:
            act_stats.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_spike.update(loss.item())
        losses.append(loss.item())
        if step % args.log_every == 0:
            print(f"  [step {step:4d}] loss={loss.item():.4f}", flush=True)

    if not losses:
        raise ValueError(f"ablation '{name}' produced no steps")

    stats = act_stats.summary()
    summary = {
        "name": name,
        "cfg": cfg_kwargs,
        "n_params": n_params,
        "final_loss": losses[-1],
        "mean_last_100_loss": sum(losses[-100:]) / max(1, len(losses[-100:])),
        "n_loss_spikes": loss_spike.n_spikes,
        # The whole point of the ablation is stability, so the activation
        # statistics have to reach the summary - previously they were
        # measured and then thrown away, leaving only a loss curve that
        # cannot separate the variants.
        "act_stats": stats,
        "max_abs_pre_norm": stats.get("__summary__", {}).get("max_max_abs"),
        "mean_max_abs_pre_norm": stats.get("__summary__", {}).get("mean_max_abs"),
        "elapsed_s": time.time() - t0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [done] {summary}")
    return summary


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args()

    ablations = dict(ABLATIONS)
    if known.config:
        from_yaml = load_ablations_from_config(known.config)
        if from_yaml:
            ablations = from_yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="YAML run config; CLI flags override it")
    p.add_argument("--names", nargs="+", default=list(ablations.keys()))
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--stats-every", type=int, default=200)
    p.add_argument("--output-dir", default="runs/ablation")
    p.add_argument("--force", action="store_true")
    apply_config_to_parser(p, known.config)
    args = p.parse_args()

    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for name in args.names:
        if name not in ablations:
            print(f"[warn] unknown ablation '{name}', skipping")
            continue
        run_dir = base / name
        if (run_dir / "summary.json").exists() and not args.force:
            print(f"[skip] {name} already done, use --force to rerun")
            continue
        s = run_one(name, ablations[name], args, run_dir)
        all_summaries.append(s)

    with open(base / "all_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[done] {len(all_summaries)} ablations saved to {base}")


if __name__ == "__main__":
    main()
