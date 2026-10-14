"""
Ablation runner: per-channel, depth-conditioned, LayerScale, tanh vs
softsign family, alpha initialization grid.

Each ablation is a small training run (default: GPT-2 124M, 1000 steps)
on a synthetic Markov corpus plus a stability measurement at the end. The
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
    ActivationStats, LossSpikeCounter, markov_batch,
)
from adasquash.model import build_gpt_350m, build_llama_tiny_1b
from scripts.train_gpt2 import apply_config_to_parser, build_optimizer, load_yaml_config


# Component ablations. The `dyt*` rows are the single-axis references
# (`dyt` = plain Dynamic Tanh, `dytc` = the per-channel axis,
# `dytd` = the depth axis) and `dytcd_full` is the combined layer, so the
# contrasts are *between references* rather than one-component diffs off
# `dytcd_full`: the per-channel axis is isolated by `dytc_*` vs `dyt`, the
# depth schedule by `dytd_full` vs `dytd_k0_no_depth_schedule`, and the
# activation family by the `dytc_p*` sweep. `dytcd_full` against
# `dytd_full` / `dytc_perchannel` does *not* isolate an axis, because the
# combined layer also switches tanh for the generalized softsign and adds
# the `delta` vector - `configs/ablation-matrix.yaml` spells this out row
# by row and is the authoritative table.
ABLATIONS = {
    # ---- Per-channel ----
    "dyt_no_perchannel": dict(norm_name="dyt"),
    "dytc_perchannel": dict(norm_name="dytc", p_init=2.0, learnable_p=False, alpha_jitter=0.0),
    "dytcd_full": dict(norm_name="dytcd"),
    # ---- Depth-cond ----
    # `learnable_k=False` is load-bearing: `k_init=0.0` alone only sets the
    # *initial* decay rate, the optimizer would immediately start learning k
    # back off zero and the row would stop isolating the schedule.
    "dytd_k0_no_depth_schedule": dict(norm_name="dytd", k_init=0.0, learnable_k=False),
    "dytd_full": dict(norm_name="dytd"),
    # ---- LayerScale ----
    "dytcd_no_ls": dict(norm_name="dytcd", use_layerscale=False),
    "dytcd_ls1e5": dict(norm_name="dytcd", layerscale_init=1e-5),
    "dytcd_ls1e3": dict(norm_name="dytcd", layerscale_init=1e-3),
    # ---- softsign family vs its hard-tanh limit ----
    "dytc_p1_softsign": dict(norm_name="dytc", p_init=1.0, learnable_p=False, alpha_jitter=0.0),
    "dytc_p2_softsign": dict(norm_name="dytc", p_init=2.0, learnable_p=False, alpha_jitter=0.0),
    "dytc_p4_softsign": dict(norm_name="dytc", p_init=4.0, learnable_p=False, alpha_jitter=0.0),
    "dytc_plearnable": dict(norm_name="dytc", p_init=2.0, learnable_p=True, alpha_jitter=0.0),
    "dytc_hardtanh_eq": dict(norm_name="dytc", p_init=64.0, learnable_p=False, alpha_jitter=0.0),  # p->inf == hard tanh
    # ---- alpha init grid ----
    "dyt_alpha_0.3": dict(norm_name="dyt", alpha_init=0.3),
    "dyt_alpha_0.5": dict(norm_name="dyt", alpha_init=0.5),
    "dyt_alpha_1.0": dict(norm_name="dyt", alpha_init=1.0),
}

_MODEL_BUILDERS = {
    "gpt2-124m": build_gpt_124m,
    "gpt2-350m": build_gpt_350m,
    "llama-1b": build_llama_tiny_1b,
}


def install_outlier_channel(model, scale: float = 25.0) -> int:
    """Scale up one embedding row to act as an activation-outlier channel.

    Returns the token id of the outlier row. Run *after* `reset_parameters`
    """
    with torch.no_grad():
        oid = model.cfg.vocab_size - 1
        model.tok_emb.weight[oid] *= scale
    return oid


def load_ablations_from_config(path: str) -> dict:
    """Read the ablation table out of a YAML run config.

    The config file is authoritative: it is the single place the
    ablation table is listed, keyed by each row's `name`.
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
        if name in out:
            # `out[name] = row` would silently keep the last definition, so a
            # duplicated row would quietly drop a condition of the matrix.
            raise ValueError(
                f"{path}: duplicate ablation name '{name}' - each row must be "
                "unique or an intended comparison is silently lost"
            )
        out[name] = row
    return out


def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path):
    print(f"\n[abl] {name}: {cfg_kwargs}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Seed so every ablation starts from a reproducible initialization.
    torch.manual_seed(args.seed)
    model = _MODEL_BUILDERS[args.model](**cfg_kwargs)
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

    # `markov_batch` is a genuine walk, so the corpus has next-token signal
    # and the loss column can actually rank the variants (see its docstring).
    losses = []
    stats_windowed = False
    t0 = time.time()
    for step in range(args.max_steps):
        x, y = markov_batch(
            rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device,
            outlier_id=outlier_id,
        )
        measure = (step % args.stats_every == 0)
        if measure:
            # `reset=False` after the first window so the reported activation
            # statistics pool every sampled step of the run instead of
            # describing only the last sampling window.
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        loss = model(x, y, return_logits=False)[1]
        loss.backward()
        if measure:
            act_stats.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val = loss.item()
        loss_spike.update(loss_val)
        losses.append(loss_val)
        if step % args.log_every == 0:
            print(f"  [step {step:4d}] loss={loss_val:.4f}", flush=True)

    if not losses:
        raise ValueError(f"ablation '{name}' produced no steps")

    stats = act_stats.summary()
    summary = {
        "name": name,
        "cfg": cfg_kwargs,
        "model": args.model,
        "n_params": n_params,
        "final_loss": losses[-1],
        "mean_last_100_loss": sum(losses[-100:]) / max(1, len(losses[-100:])),
        "n_loss_spikes": loss_spike.n_spikes,
        # The full curve, not just its endpoints: a flat-at-log(vocab) run and
        # a run that halved its loss have the same `final_loss` scale story.
        "loss_curve": losses,
        # Activation statistics belong in the summary alongside the loss curve.
        "act_stats": stats,
        "max_abs_pre_norm": stats.get("__summary__", {}).get("max_abs_pre_norm"),
        "mean_max_abs_pre_norm": stats.get("__summary__", {}).get("mean_max_abs"),
        "mean_excess_kurtosis_pre_norm": stats.get("__summary__", {}).get(
            "mean_excess_kurtosis_pre_norm"
        ),
        "elapsed_s": time.time() - t0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [done] {summary['final_loss']:.4f} "
          f"max_abs_pre_norm={summary['max_abs_pre_norm']}")
    return summary


def merge_summaries(previous: list, fresh: list) -> list:
    """Fold freshly-computed rows into the existing aggregate.

    `previous` is what `all_summaries.json` held, `fresh` what this run
    produced. Every row survives: a partial re-run (`--names <one> --force`)
    must not reorder the aggregate into a single-row table, and rows written
    by earlier runs that are not in `args.names` must not be dropped from it.
    Order follows first appearance, so the aggregate is stable across re-runs;
    a repeated name is overwritten by the newer row.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in list(previous) + list(fresh):
        if not isinstance(row, dict) or "name" not in row:
            continue
        if row["name"] not in merged:
            order.append(row["name"])
        merged[row["name"]] = row
    return [merged[name] for name in order if name in merged]


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
    p.add_argument("--model", choices=sorted(_MODEL_BUILDERS), default="gpt2-124m")
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
    # `ablations` is consumed by `load_ablations_from_config` above, not by
    # the parser; everything else in the YAML maps straight onto a dest.
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

    # Merge with whatever is already on disk: skipped rows still belong in the
    # aggregate, and overwriting with only the freshly-run ones would drop
    # every previously completed ablation on a partial re-run.
    agg_path = base / "all_summaries.json"
    previous: list = []
    if agg_path.exists():
        try:
            previous = json.loads(agg_path.read_text())
            if not isinstance(previous, list):
                previous = []
        except json.JSONDecodeError:
            previous = []
    ordered = merge_summaries(previous, all_summaries)
    with open(agg_path, "w") as f:
        json.dump(ordered, f, indent=2)
    print(f"\n[done] {len(ordered)} ablations saved to {base}")


if __name__ == "__main__":
    main()
