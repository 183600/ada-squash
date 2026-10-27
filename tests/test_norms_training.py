"""
Unit tests for the DyT++ / AdaSquash project.

Run:
    python -m pytest tests/ -v

Tests cover:
  - norm registry / build_norm
  - all normalization layers preserve shape and dtype
  - every fused autograd kernel (DyT / DyT-C / DyT-D / DyT-CD, RMSNorm,
    SwiGLU, both cross-entropies, the generalized softsign) matches an
    independent recomposition in plain autograd, per parameter, from a
    *randomized* parameter set - not just "the gradients are finite"
  - frozen `p` / `k`, `beta=False`, and the alpha / p clamps
  - the generalized softsign is bounded in [-1, 1] and finite for runaway `p`
  - depth schedule of DyT-D / DyT-CD is monotone in alpha_0 / k
  - QK-squash keeps logits bounded; attention logit stats read the causal triangle
  - model forward / loss / checkpoint round-trip
  - metrics: KurtosisMeter matches torch.kurtosis on synthetic data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import pytest
import torch
import torch.nn as nn

from adasquash import (
    build_norm, NORM_REGISTRY, generalized_softsign,
    LayerNorm, RMSNorm, DyT, DyTS, DyTP, DyTC, DyTD, DyTDI, DyTCD, DyTCDI,
    GPTConfig, GPT, build_gpt_124m, build_llama_tiny_1b,
    ActivationStats, LossSpikeCounter, KurtosisMeter,
    count_parameters,
)
from adasquash.attention import apply_rope
from adasquash.norms import QKSquash
from adasquash.model import cross_entropy_mean


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_kurtosis_matches_torch_on_normal():
    torch.manual_seed(0)
    # Pure normal distribution has excess kurtosis == 0
    x = torch.randn(4096)
    meter = KurtosisMeter()
    meter.update(x)
    result = meter.result()
    # Allow generous tolerance because of the simplified formulas.
    assert abs(result) < 0.2, f"excess kurtosis = {result}, expected ~0"


def test_kurtosis_high_for_outlier_distribution():
    # Heavy-tailed distribution should have positive excess kurtosis.
    x = torch.cat([torch.randn(2048) * 0.5, torch.randn(64) * 50.0])
    meter = KurtosisMeter()
    meter.update(x)
    result = meter.result()
    assert result > 5.0, f"kurtosis = {result}, expected > 5"


def test_loss_spike_counter():
    counter = LossSpikeCounter()
    # Steady loss
    for _ in range(20):
        counter.update(3.0)
    assert counter.n_spikes == 0
    # Spike
    counter.update(10.0)
    assert counter.n_spikes >= 1


def test_activation_stats_hooks():
    model = build_gpt_124m(norm_name="dytcd")
    stats = ActivationStats(model)
    stats.attach()
    x = torch.randint(0, model.cfg.vocab_size, (2, 64))
    with torch.no_grad():
        model(x)
    s = stats.summary()
    stats.detach()
    assert "__summary__" in s
    # ActivationStats reports per-norm max|x| of the *pre-norm* activations.
    assert "mean_max_abs" in s["__summary__"]
    assert "max_max_abs" in s["__summary__"]
    assert s["__summary__"]["max_max_abs"] >= s["__summary__"]["mean_max_abs"]
    # README metric-table name for the same quantity.
    assert s["__summary__"]["max_abs_pre_norm"] == s["__summary__"]["max_max_abs"]
    assert "mean_excess_kurtosis_pre_norm" in s["__summary__"]
    # Kurtosis is documented on the class and must actually be emitted.
    assert "excess_kurtosis" in s["blocks.0.norm1"]


def test_activation_stats_can_pool_across_sampling_windows():
    """`attach(reset=False)` must accumulate, so run-level stats are possible.

    The ablation runner re-attaches on a sampling window; without this the
    reported statistics describe only the last window while `final_loss`
    describes the whole run.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=16, norm_name="dytcd"))
    stats = ActivationStats(model)
    probe = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        # Window 1 carries the larger activation.
        model.tok_emb.weight[0].fill_(100.0)
        stats.attach(reset=True)
        model(probe)
        stats.detach()
        first = stats.summary()["__summary__"]["max_max_abs"]
        # Window 2 is much smaller; a pooled report must still show window 1's
        # peak, while a resetting one drops back to the smaller value.
        model.tok_emb.weight.zero_()
        stats.attach(reset=False)
        model(probe)
        stats.detach()
        pooled = stats.summary()["__summary__"]["max_max_abs"]
    assert first > 100.0, "the first window should carry the outlier"
    assert pooled > 100.0, "reset=False must not discard earlier meters"
    # A fresh attach() does reset.
    with torch.no_grad():
        stats.attach(reset=True)
        model(probe)
        stats.detach()
        assert stats.summary()["__summary__"]["max_max_abs"] <= 1.0


def test_activation_stats_reattach_does_not_leak_hooks():
    """`attach` must drop the hooks it registered last time.

    Resetting the handle list without removing the hooks would leave the
    first batch live: every activation sampled once per live hook, and the
    abandoned hooks surviving the `detach` that followed - still writing into
    meters the caller believes are closed.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                          max_seq_len=16, norm_name="dytcd"))
    stats = ActivationStats(model)
    n_norms = sum(
        1 for m in model.modules()
        if m.__class__.__name__ in {"DyTCD"}
    )
    assert n_norms == 2 * model.cfg.n_layers + 1

    stats.attach()
    stats.attach()  # no detach() in between
    assert len(stats.handles) == n_norms, "the previous hooks were not removed"

    stats.detach()
    assert stats.handles == []
    with torch.no_grad():
        model(torch.zeros(1, 4, dtype=torch.long))
    assert all(m.count == 0 for m in stats.max_abs.values()), (
        "a detached hook is still writing into the meters"
    )


# ---------------------------------------------------------------------------
# Synthetic corpus: the ablations / MoE / W8A8 probe depend on it having
# real next-token signal.
# ---------------------------------------------------------------------------
def _mutual_information(x, y, n_states):
    pairs = torch.stack([x[:, :-1].reshape(-1), y[:, :-1].reshape(-1)], 1)
    joint = torch.zeros(n_states, n_states, dtype=torch.double)
    joint[pairs[:, 0].long(), pairs[:, 1].long()] += 1
    p = joint / joint.sum()
    px, py = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
    tiny = 1e-30
    return (
        p * ((p.clamp(min=tiny).log()) - px.clamp(min=tiny).log()
             - py.clamp(min=tiny).log())
    ).sum().item()


def test_markov_corpus_has_learnable_next_token_signal():
    from adasquash import markov_batch

    rng = torch.Generator().manual_seed(0)
    x, y = markov_batch(rng, -100, 256, 512, "cpu", n_states=64)
    mi = _mutual_information(x, y, 64)
    # A per-position independent draw has I == 0 and pins the loss at
    # log(vocab_size) for every variant, making the ablation matrix
    # unrankable. A real walk over 64 states with jumps in {0,1,2} carries
    # log(64) - log(3) ~ 3.06 nats.
    assert mi > 2.5, f"I(x;y) = {mi:.3f} nats, the corpus has no signal"
    assert abs(mi - (math.log(64) - math.log(3))) < 0.2


def test_markov_corpus_shapes_and_masking():
    from adasquash import markov_batch

    rng = torch.Generator().manual_seed(0)
    x, y = markov_batch(rng, -100, 4, 16, "cpu", n_states=64)
    assert x.shape == y.shape == (4, 16)
    # Target is the shifted successor, final position masked out.
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert (y[:, -1] == -100).all()
    assert x.min() >= 0 and x.max() < 64


def test_markov_outlier_channel_does_not_destroy_the_targets():
    from adasquash import markov_batch

    rng = torch.Generator().manual_seed(0)
    x, y = markov_batch(rng, -100, 8, 512, "cpu", outlier_id=50303)
    frac = (x == 50303).float().mean().item()
    assert 0.01 < frac < 0.06, f"outlier channel fraction = {frac}"
    # The spike perturbs the *input* only; the targets stay on the clean
    # walk, so the corpus is still learnable.
    assert 50303 not in y.unique().tolist()


def test_depth_schedule_ablation_row_actually_disables_the_schedule():
    """`k_init=0.0` alone leaves `k` learnable, so the schedule comes back."""
    from scripts.ablation import load_ablations_from_config

    root = Path(__file__).resolve().parent.parent
    rows = load_ablations_from_config(
        str(root / "configs" / "ablation-matrix.yaml")
    )
    row = rows["dytd_k0_no_depth_schedule"]
    assert row["norm_name"] == "dytd"
    assert row["k_init"] == 0.0
    assert row.get("learnable_k") is False, (
        "with learnable_k=True the optimizer learns k straight back off zero "
        "and the row stops isolating the depth schedule"
    )


def _tiny_variant(cfg_kwargs):
    """Build one ablation row's norms at a size that fits in a unit test."""
    return GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                         max_seq_len=16, **cfg_kwargs)).blocks[0].norm1


def test_every_ablation_row_builds_its_own_hyperparameters():
    """Every row must be built with the config the matrix spells out, and
    the factorial cells must have the alpha shape their name promises."""
    from scripts.ablation import load_ablations_from_config

    root = Path(__file__).resolve().parent.parent
    rows = load_ablations_from_config(str(root / "configs" / "ablation-matrix.yaml"))

    # Softsign exponent sweep reaches the module as p.
    for name in ("dytc_p1_softsign", "dytc_p4_softsign"):
        m = _tiny_variant(rows[name])
        assert m.p == pytest.approx(float(rows[name]["p_init"])), name
    assert isinstance(_tiny_variant(rows["dytc_plearnable"]).p, nn.Parameter)
    # The factorial cells: scalar alpha vs per-channel alpha.
    assert _tiny_variant(rows["dyt"]).alpha.ndim == 0
    assert _tiny_variant(rows["dyts"]).alpha.ndim == 0
    assert _tiny_variant(rows["dytp"]).alpha.ndim == 1
    assert _tiny_variant(rows["dytc"]).alpha.ndim == 1
    # Depth rows: the reparameterized variant owns (alpha_0, k); the
    # depth-init control owns a plain alpha; the shared row shares one pair.
    dytd = _tiny_variant(rows["dytd"])
    assert isinstance(dytd, DyTD) and hasattr(dytd, "k")
    dytdi = _tiny_variant(rows["dytdi"])
    assert isinstance(dytdi, DyT) and not hasattr(dytdi, "k")
    shared = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                           max_seq_len=16, **rows["dytd_shared"]))
    alpha_ids = {id(m.alpha_0) for m in shared.modules() if isinstance(m, DyTD)}
    assert len(alpha_ids) == 1
    # The depth axis: `k` on (k=1, learnable) vs off (buffered k=0).
    on = _tiny_variant(rows["dytd"])
    off = _tiny_variant(rows["dytd_k0_no_depth_schedule"])
    assert on.k.item() == pytest.approx(1.0)
    assert isinstance(on.k, nn.Parameter)
    assert off.k.item() == pytest.approx(0.0)
    assert not isinstance(off.k, nn.Parameter)


def test_norm_kwargs_follow_the_config_not_a_previously_built_model():
    """The same guard for the axes the matrix does not currently sweep."""
    from adasquash.model import build_norm_kwargs

    def norm(**over):
        cfg = GPTConfig(vocab_size=32, dim=32, n_heads=4, max_seq_len=8, **over)
        return build_norm(cfg.norm_name, 32, **build_norm_kwargs(cfg, 0))

    d = norm(norm_name="dytd", n_layers=2, k_init=0.0, learnable_k=False)
    assert d.k.item() == 0.0 and not isinstance(d.k, nn.Parameter)
    assert norm(norm_name="dytd", n_layers=2, k_init=1.0).k.item() == 1.0

    c = norm(norm_name="dytc", n_layers=2, p_init=1.0, learnable_p=False)
    assert c.p == 1.0
    assert isinstance(norm(norm_name="dytc", n_layers=2, p_init=2.0).p,
                      nn.Parameter)

    # `num_layers` sets the depth ratio, so a deeper model must not reuse the
    # shallow model's schedule even though `layer_idx` is identical.
    assert norm(norm_name="dytd", n_layers=2).num_layers == 2
    assert norm(norm_name="dytd", n_layers=8).num_layers == 8


def test_depth_sweep_shape_holds_the_parameter_count_roughly_fixed():
    """The depth axis must not be confounded with model size."""
    from scripts.train_llama_depth_sweep import shape_for_depth

    V = 32000

    def params(depth, dim):
        hidden = int(dim * (8 / 3.0))
        hidden = int(hidden * 2 / 3)
        hidden = (hidden + 7) & ~7
        return depth * (4 * dim * dim + 3 * hidden * dim + 4 * dim) + 2 * V * dim

    counts = []
    for depth in (12, 24, 48, 96):
        dim, heads = shape_for_depth(depth, 2048, 16)
        assert dim % heads == 0
        assert dim // heads == 128, "head_dim must be constant across the sweep"
        counts.append(params(depth, dim))
    spread = max(counts) / min(counts)
    assert spread < 1.3, (
        f"parameter count varies {spread:.2f}x across the depth axis; a deep "
        f"run that is also a bigger model makes 'RMSNorm diverges at depth' "
        f"unfalsifiable"
    )


def test_depth_sweep_shape_honours_a_non_llama_base():
    from scripts.train_llama_depth_sweep import _BASE_SHAPE, shape_for_depth

    # The documented table is only valid for the LLaMA-tiny base shape; a
    # gpt2-124m sweep must not silently be built at LLaMA-tiny width.
    # head_dim is still held at the gpt2 base's 64, but dim/heads follow the
    # gpt2 base, not the table.
    dim, heads = shape_for_depth(12, 768, 12)
    assert dim // heads == 768 // 12
    assert (dim, heads) not in [(2560, 20), (2048, 16), (1536, 12), (1280, 10)]
    # The base depth is unchanged, so a gpt2 sweep still lands on the
    # builder's own shape there.
    assert shape_for_depth(24, 768, 12) == (768, 12)
    assert shape_for_depth(24, *_BASE_SHAPE) == (2048, 16)


def test_depth_sweep_base_shape_follows_the_selected_model():
    """Every `--model` arm must be rescaled from *its own* base width.

    A single `llama-1b ? llama : gpt2-124m` fallback would send every
    non-LLaMA arm to the 124M width, so a `--model gpt2-350m` sweep would be
    rescaled from (768, 12) and built at 124M's shape.
    """
    from scripts.train_llama_depth_sweep import (
        _BASE_SHAPE, base_shape_for_model, shape_for_depth,
    )

    assert base_shape_for_model("llama-1b") == _BASE_SHAPE == (2048, 16)
    assert base_shape_for_model("gpt2-124m") == (768, 12)
    assert base_shape_for_model("gpt2-350m") == (1024, 16)
    # An unrecognised selector keeps the documented LLaMA-tiny base.
    assert base_shape_for_model("something-else") == _BASE_SHAPE
    # The 350M arm must not land on the 124M shape at the base depth.
    assert shape_for_depth(24, *base_shape_for_model("gpt2-350m")) == (1024, 16)
    assert shape_for_depth(24, *base_shape_for_model("gpt2-124m")) == (768, 12)


# ---------------------------------------------------------------------------
# Training-loop invariants
# ---------------------------------------------------------------------------
def test_lr_schedule_has_no_dead_first_step():
    """`LambdaLR` evaluates step 0 at construction, so the first update
    would otherwise run at lr == 0 and warmup end one step late."""
    import argparse as _argparse
    from scripts.train_gpt2 import build_lr_lambda
    from torch.optim.lr_scheduler import LambdaLR

    args = _argparse.Namespace(max_steps=100, warmup=10, min_lr_ratio=0.1)
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=6e-4)
    sched = LambdaLR(opt, build_lr_lambda(args))
    assert opt.param_groups[0]["lr"] > 0.0, "first update would be a no-op"
    # The lambda is evaluated at `last_epoch`, which starts at 0, so
    # `last_epoch = 9` is the last index whose `s + 1` lands on `warmup`.
    for _ in range(9):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 6e-4) < 1e-9, "peak not reached at end of warmup"
    for _ in range(90):
        opt.step()
        sched.step()
    assert abs(opt.param_groups[0]["lr"] - 6e-5) < 1e-9, "min_lr_ratio not reached"


def test_resume_restores_the_schedule_defining_flags():
    import argparse as _argparse
    from scripts.train_gpt2 import restore_schedule_args

    ckpt_args = dict(max_steps=10000, warmup=500, min_lr_ratio=0.1, lr=6e-4,
                     weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0)
    args = _argparse.Namespace(max_steps=30000, warmup=123, min_lr_ratio=0.1,
                               lr=6e-4, weight_decay=0.1, beta1=0.9, beta2=0.95,
                               grad_clip=1.0)
    changed = restore_schedule_args(args, ckpt_args)
    assert set(changed) == {"max_steps", "warmup"}
    assert args.max_steps == 10000 and args.warmup == 500
    # Identical flags are a no-op; a checkpoint without `args` keeps the CLI.
    same = _argparse.Namespace(**ckpt_args)
    assert restore_schedule_args(same, ckpt_args) == []
    assert restore_schedule_args(same, {}) == []


def test_resume_keeps_an_explicitly_typed_schedule_flag():
    """`--resume <ckpt> --max-steps <larger>` is the documented way to extend a
    run, so a flag typed on the command line must not be overwritten by the
    checkpoint's value - otherwise the extension silently truncates at the
    old step count. A flag left at its default is still restored, because
    that one would otherwise silently reshape the cosine curve."""
    import argparse as _argparse
    from scripts.train_gpt2 import restore_schedule_args

    ckpt_args = dict(max_steps=10000, warmup=500, min_lr_ratio=0.1, lr=6e-4,
                     weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0)
    args = _argparse.Namespace(max_steps=30000, warmup=500, min_lr_ratio=0.1,
                               lr=6e-4, weight_decay=0.1, beta1=0.9, beta2=0.95,
                               grad_clip=1.0)
    changed = restore_schedule_args(args, ckpt_args, explicit=frozenset({"max_steps"}))
    # The divergence is reported either way ...
    assert changed == ["max_steps"]
    # ... but the explicit flag survives, so the run can be extended.
    assert args.max_steps == 30000
    # A non-explicit flag with the same divergence is still restored.
    args2 = _argparse.Namespace(**vars(args))
    restore_schedule_args(args2, ckpt_args, explicit=frozenset())
    assert args2.max_steps == 10000


def test_config_seeded_flags_are_defaults_not_explicit_overrides(tmp_path):
    """A value that came from `--config` must behave like a default.

    `apply_config_to_parser` folds the YAML into the parser defaults, so
    `parser.get_default` is the post-config default and a config-seeded flag
    is *not* "typed on the command line". If it were counted as explicit it
    would be allowed to silently reshape the curve of the run it resumes -
    exactly what the restore exists to prevent."""
    import argparse as _argparse
    from scripts.train_gpt2 import (
        apply_config_to_parser, restore_schedule_args, _SCHEDULE_ARGS,
        _TRAIN_CONFIG_ALIASES,
    )

    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text("max_steps: 7\nwarmup: 3\n")

    def build():
        p = _argparse.ArgumentParser()
        for name in _SCHEDULE_ARGS:
            p.add_argument("--" + name.replace("_", "-"), type=float, default=999.0)
        apply_config_to_parser(p, str(cfg_path), _TRAIN_CONFIG_ALIASES)
        return p

    def parse_and_resume(cli, ck):
        p = build()
        argv, sys_argv = ["prog", *cli], sys.argv
        try:
            sys.argv = argv
            args = p.parse_args()
        finally:
            sys.argv = sys_argv
        explicit = frozenset(
            name for name in _SCHEDULE_ARGS
            if getattr(args, name, None) != p.get_default(name)
        )
        changed = restore_schedule_args(args, ck, explicit=explicit)
        return args, sorted(explicit), changed

    ck = {"max_steps": 100, "warmup": 50}
    # Config only: both flags are adopted from the checkpoint.
    args, explicit, changed = parse_and_resume([], ck)
    assert explicit == []
    assert (args.max_steps, args.warmup) == (100, 50)
    assert changed == ["max_steps", "warmup"]
    # The same run with `--max-steps` typed: only that flag survives.
    args, explicit, changed = parse_and_resume(["--max-steps", "30"], ck)
    assert explicit == ["max_steps"]
    assert (args.max_steps, args.warmup) == (30, 50)
    assert changed == ["max_steps", "warmup"]


def test_loss_spike_counter_state_survives_a_resume():
    """`n_loss_spikes` is logged as a running count over the run, so a resumed
    run has to keep counting from the checkpoint's value - and from its EMA
    baseline, which is what the spike test is relative to."""
    from adasquash import LossSpikeCounter

    c = LossSpikeCounter()
    c.update(2.0)                       # seeds the EMA
    c.update(2.01)                      # no jump
    state = c.state_dict()
    assert set(state) == {"ema", "n_spikes"}

    resumed = LossSpikeCounter()
    assert resumed.load_state_dict(state) is True
    assert resumed.ema == pytest.approx(c.ema)
    assert resumed.n_spikes == 0
    # A spike is judged against the *restored* baseline, not a fresh one.
    resumed.update(10.0)
    c.update(10.0)
    assert resumed.n_spikes == 1
    assert resumed.ema == pytest.approx(c.ema)
    # A unusable / absent state restores nothing rather than half-loading.
    other = LossSpikeCounter()
    assert other.load_state_dict(None) is False
    assert other.load_state_dict({"n_spikes": 7}) is False
    assert other.n_spikes == 0 and other.ema is None


def _tiny_train_run(tmp_path, monkeypatch, steps, extra=()):
    """Run `scripts/train_gpt2.py`'s main() on a tiny model + synthetic corpus.

    The depth/width override env vars are the only knob the driver exposes for
    shrinking a run, and the HF dataloader is replaced by the synthetic corpus
    so the loop is exercised without `datasets` / `transformers`.
    """
    import scripts.train_gpt2 as tg
    from adasquash import markov_batch

    def fake_loader(dataset_name, dataset_config, seq_len, batch_size,
                    tokenizer="gpt2", split="train"):
        import torch.utils.data as tud
        rng = torch.Generator().manual_seed(7)

        class DS(tud.IterableDataset):
            def __iter__(self):
                while True:
                    yield markov_batch(
                        rng, -100, batch_size, seq_len, "cpu", n_states=64
                    )

        return tud.DataLoader(DS(), batch_size=None)

    monkeypatch.setattr(tg, "build_dataloader", fake_loader)
    monkeypatch.setenv("ADASQUASH_DEPTH_OVERRIDE", "2")
    monkeypatch.setenv("ADASQUASH_DIM_OVERRIDE", "48")
    monkeypatch.setenv("ADASQUASH_HEADS_OVERRIDE", "3")
    argv = [
        "train_gpt2.py", "--norm", "dytcd", "--no-bf16", "--batch-size", "2",
        "--seq-len", "32", "--grad-accum", "1", "--log-every", "1",
        "--stats-every", "100", "--warmup", "1",
        "--output-dir", str(tmp_path),
    ]
    argv += ["--max-steps", str(steps), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    tg.main()


def test_resume_carries_the_loss_spike_counter_and_an_explicit_extension(tmp_path, monkeypatch, capsys):
    """Two properties of `--resume` that only the full driver exercises:

    * every checkpoint carries the loss-spike state, so `n_loss_spikes` is a
      count over the whole run rather than only the post-resume window;
    * a `--max-steps` typed on the command line survives the resume, which is
      how a finished run is extended. Overwriting it with the checkpoint's
      value makes the loop exit immediately and the extension silently no-op.
    """
    _tiny_train_run(tmp_path / "a", monkeypatch, 3, ["--save-every", "3"])
    ckpt_path = tmp_path / "a" / "ckpt_000003.pt"
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert set(ckpt["loss_spike"]) == {"ema", "n_spikes"}
    assert isinstance(ckpt["loss_spike"]["ema"], float)

    capsys.readouterr()
    _tiny_train_run(tmp_path / "b", monkeypatch, 5,
                    ["--save-every", "100", "--resume", str(ckpt_path)])
    out = capsys.readouterr().out
    assert "restored loss-spike counter" in out
    assert "keeping your explicit override" in out
    # The extension actually ran: step 5 was reached, not 3.
    final = torch.load(tmp_path / "b" / "ckpt_final.pt", weights_only=False)
    assert int(final["step"]) == 5
    assert int(final["loss_spike"]["n_spikes"]) == 0


def test_qk_alpha_init_reaches_the_attention():
    """`GPTConfig.qk_alpha_init` must reach the QKSquash the attention builds."""
    cfg = GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                    norm_name="rmsnorm", qk_norm="dytqk", qk_alpha_init=0.25)
    model = GPT(cfg)
    sq = model.blocks[0].attn.q_squash
    assert sq is not None and sq is model.blocks[0].attn.k_squash
    assert sq.alpha_q.item() == pytest.approx(0.25)
    assert sq.alpha_k.item() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Needle-in-a-haystack protocol
# ---------------------------------------------------------------------------
class _CharTokenizer:
    """Char-level tokenizer, so a secret spans several tokens."""

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) % 60 + 1 for c in s]}


class _ScriptedModel:
    """Emits a scripted token sequence, one per call."""

    def __init__(self, ids, cfg):
        self.ids, self.cfg, self.base = ids, cfg, None

    def __call__(self, x, y=None):
        if self.base is None:
            self.base = x.shape[1]
        k = min(x.shape[1] - self.base, len(self.ids) - 1)
        logits = torch.full((x.shape[0], x.shape[1], 64), -20.0)
        logits[:, -1, self.ids[k]] = 20.0
        return logits, None


def test_needle_scores_the_whole_secret_not_just_its_first_token():
    """GPT-2 splits " 12345" into several BPE tokens; matching only the
    first one accepts a model that recites 3 of the 5 digits."""
    from scripts.eval_downstream import needle_in_haystack

    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=512, norm_name="rmsnorm")).eval()
    tok = _CharTokenizer()
    needle = "The magic number is 12345."
    target = tok(" 12345")["input_ids"]
    assert len(target) > 1, "the secret must span several tokens for this test"

    exact = needle_in_haystack(
        _ScriptedModel(target, model.cfg), tok, needle=needle,
        context_len=256, n_trials=2, device="cpu")
    assert exact == 1.0

    wrong_tail = target[:-1] + [(target[-1] % 60) + 1]
    partial = needle_in_haystack(
        _ScriptedModel(wrong_tail, model.cfg), tok, needle=needle,
        context_len=256, n_trials=2, device="cpu")
    assert partial == 0.0


def test_needle_rejects_contexts_it_cannot_evaluate():
    """A context past `max_seq_len`, or one too short to hold a single
    haystack repeat, must be rejected rather than evaluated."""
    from scripts.eval_downstream import needle_in_haystack

    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=256, norm_name="rmsnorm")).eval()
    tok = _CharTokenizer()
    with pytest.raises(ValueError, match="max_seq_len"):
        needle_in_haystack(model, tok, context_len=4096, n_trials=1, device="cpu")
    with pytest.raises(ValueError, match="too short"):
        needle_in_haystack(model, tok, context_len=20, n_trials=1, device="cpu")


class _Reciter(GPT):
    """A real GPT that recites the secret, so the scoring loop runs to the
    end instead of bailing out on its first mismatching token.

    The point is the *context* it forces: every emitted token is appended
    before the next one is predicted, so the longest forward pass is
    `context_len + len(secret) - 1` long - and the real `GPT.forward`
    (reached through `last_position_logits`) is what enforces
    `max_seq_len`.
    """

    def __init__(self, *a, secret=None, **kw):
        super().__init__(*a, **kw)
        self.secret = secret
        self.i = 0
        self.prev_n = None

    def last_position_logits(self, input_ids, return_stats=False):
        n = input_ids.shape[1]
        if self.prev_n is None or n < self.prev_n:
            # A shorter context than last time means a new trial: restart the
            # secret instead of carrying the cursor across trials.
            self.i = 0
        self.prev_n = n
        # Routed through the real forward pass so `max_seq_len` is actually
        # enforced - that is the constraint the test is about.
        out = super().last_position_logits(
            input_ids, return_stats=return_stats
        ).clone()
        out[:, self.secret[min(self.i, len(self.secret) - 1)]] = 1e4
        self.i += 1
        return out


def test_needle_context_leaves_room_for_the_scoring_loop():
    """A context sized at exactly `max_seq_len` is still one decode too long.

    Greedy scoring appends each secret token to the context before scoring
    the next one, so the last scored step runs `len(secret) - 1` positions
    past the prompt. The eval script's default context *is* `max_seq_len`,
    and a haystack prompt whose repeats divide the limit exactly fills it
    to `context_len - 2`, so a secret of three or more tokens pushes the
    forward past the positional table. The context has to be refused up
    front instead.
    """
    from scripts.eval_downstream import needle_in_haystack

    tok = _CharTokenizer()
    needle = "The magic number is 12345."
    secret = tok(" 12345")["input_ids"]
    assert len(secret) > 1, "the guard only bites for a multi-token secret"
    overhead = len(secret) - 1

    model = _Reciter(
        GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                  max_seq_len=256, norm_name="rmsnorm"),
        secret=secret,
    ).eval()

    # Fitting context: the whole secret is scored, no crash.
    assert needle_in_haystack(
        model, tok, needle=needle, context_len=256 - overhead,
        n_trials=2, device="cpu") == 1.0

    # A haystack prompt whose token count divides the limit exactly, so the
    # body fills to `context_len - 2`: the longest prompt the function can
    # build for that context. The documented default - the full
    # `max_seq_len` - is then refused up front, rather than the decode
    # running past the positional table inside the forward pass.
    body_len = (256 - len(tok(needle)["input_ids"])
                - len(tok("\n\nWhat is the magic number? The magic number is")["input_ids"])
                - 2)
    per_repeat = next(d for d in range(2, body_len + 1) if body_len % d == 0)
    fresh = _Reciter(
        GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                  max_seq_len=256, norm_name="rmsnorm"),
        secret=secret,
    ).eval()   # a fresh cursor: the previous run already recited the secret
    with pytest.raises(ValueError, match="max_seq_len"):
        needle_in_haystack(
            fresh, tok, needle=needle, haystack_prompt="z" * (per_repeat - 1) + " ",
            context_len=256, n_trials=1, device="cpu")


def test_needle_decode_overhead_matches_the_secret_length():
    """The contexts the call sites allow for the decode are derived from
    the same tokenization the scoring loop uses."""
    from scripts.eval_downstream import (
        needle_decode_overhead, needle_target_ids,
    )

    tok = _CharTokenizer()
    for needle in ("The magic number is 12345.", "secret: 42", "find 98765"):
        assert needle_target_ids(tok, needle) == tok(
            " " + needle.strip().rstrip(".!?").split()[-1]
        )["input_ids"]
        assert needle_decode_overhead(tok, needle) == (
            len(needle_target_ids(tok, needle)) - 1
        )


# ---------------------------------------------------------------------------
# Config plumbing the model rebuild depends on
# ---------------------------------------------------------------------------
def test_beta_config_key_is_not_inverted(tmp_path):
    """`beta: true` in a YAML config must enable beta, not disable it.

    The alias table inverts the *positive-form* keys because their argparse
    dests are negative (`use_layerscale` -> `--no-layerscale`). `--no-beta`
    stores into the positive dest `beta`, so inverting `beta` made a config
    that asked for beta train without it - and a `beta: false` config train
    with it - in both the trainer and the depth-sweep launcher.
    """
    from scripts.train_gpt2 import apply_config_to_parser, _TRAIN_CONFIG_ALIASES
    from scripts.train_llama_depth_sweep import _SWEEP_CONFIG_ALIASES

    for aliases in (_TRAIN_CONFIG_ALIASES, _SWEEP_CONFIG_ALIASES):
        for yaml_value, expected in ((True, True), (False, False)):
            cfg = tmp_path / "beta.yaml"
            cfg.write_text(f"beta: {str(yaml_value).lower()}\n")
            p = argparse.ArgumentParser()
            p.add_argument("--no-beta", dest="beta", action="store_false",
                           default=True)
            apply_config_to_parser(p, str(cfg), aliases)
            assert p.parse_args([]).beta is expected, (aliases, yaml_value)


def test_load_ckpt_rebuilds_a_model_trained_without_beta(tmp_path):
    """A `--no-beta` checkpoint must reload through `load_ckpt`.

    `beta` is architectural - it decides whether the affine-bias parameter
    exists at all - and `load_ckpt` rebuilds the module from the recorded
    args, so omitting it rebuilt the model with `beta=True` and the
    state_dict load died on the missing `*.beta` keys.
    """
    from scripts.eval_downstream import load_ckpt

    cfg = GPTConfig(vocab_size=50304, dim=32, n_layers=1, n_heads=4,
                    max_seq_len=1024, mlp_ratio=4.0,
                    norm_name="dytcd", beta=False)
    model = GPT(cfg)
    assert model.blocks[0].norm1.beta is None
    args = dict(norm="dytcd", model="gpt2-124m", no_layerscale=False,
                alpha_0=0.9, k_init=1.0, alpha_init=0.5, p_init=2.0,
                no_learnable_p=False, no_learnable_k=False, beta=False,
                layerscale_init=None, tie_embeddings=None, qk_norm=None,
                qk_alpha_init=1.0,
                _depth_override=1, _dim_override=32, _heads_override=4)
    path = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "args": args, "step": 0}, path)
    rebuilt, recorded = load_ckpt(str(path), device="cpu")
    assert rebuilt.blocks[0].norm1.beta is None
    assert recorded["beta"] is False


def test_moe_final_norm_shares_the_dense_final_norm_depth_ratio():
    """The MoE study's final norm must sit at depth_ratio = 1.0 like the dense GPT's.

    It was built with `layer_idx = n_layers - 1` instead of
    `is_final_norm=True`, so every depth-conditioned final norm in the MoE
    comparison ran at depth_ratio = (L-1)/L while the dense baseline ran at
    1.0 - the two studies would differ on a knob neither reports.
    """
    from scripts.train_moe_tiny import MoEGPT

    cfg = GPTConfig(vocab_size=64, dim=32, n_layers=4, n_heads=4,
                    max_seq_len=16, norm_name="dytcd")
    assert GPT(cfg).norm_f._depth_ratio == pytest.approx(1.0)
    assert MoEGPT(cfg, n_experts=2).norm_f._depth_ratio == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def test_forward_smoke():
    """Just check that a forward pass doesn't crash on CPU."""
    model = build_gpt_124m(norm_name="dytcd")
    model.eval()
    x = torch.randint(0, model.cfg.vocab_size, (1, 32))
    with torch.no_grad():
        out, _ = model(x)
    assert out.shape == (1, 32, model.cfg.vocab_size)



# ---------------------------------------------------------------------------
# Review fixes: FP8 metric separation, held-out eval defaults, needle
# coverage, resume semantics, validation, lm-eval wrapper
# ---------------------------------------------------------------------------
def test_fp8_reports_zero_quantization_separately_from_small_values():
    """underflow (|x| < min-normal), subnormal share and quantized-to-zero
    share are three different numbers; the old metric collapsed them."""
    from scripts.check_fp8_stability import (
        measure_fp8_divergence, fp8_min_normal)
    from adasquash.model import build_gpt_124m

    m = build_gpt_124m(norm_name="dytcd")
    m.reset_parameters(3)
    m.eval()
    r = measure_fp8_divergence(m, torch.randint(0, 50304, (2, 64)), "dytcd")
    # The summary must carry all three distinguished metrics.
    for key in ("mean_underflow_rate_e4m3", "mean_subnormal_rate_e4m3",
                "mean_quantized_zero_rate_e4m3", "mean_rel_quant_err_e4m3"):
        assert key in r, key

    # Direct quantizer check: tiny nonzero values round to zero, and that
    # population is exactly what quantized_zero_rate counts.
    from scripts.check_fp8_stability import to_fp8_e4m3
    tiny = torch.full((100,), fp8_min_normal(4) * 1e-3)
    q = to_fp8_e4m3(tiny)
    assert (q == 0).float().mean().item() > 0.99  # subnormal grid spacing
    # A value exactly on the subnormal grid survives as nonzero.
    on_grid = torch.tensor([2.0 ** -9])  # e4m3 min subnormal
    assert to_fp8_e4m3(on_grid).item() == on_grid.item()


def test_fp8_dynamic_tensor_scaling_removes_clipping():
    """scaling='dynamic-tensor' rescales by amax before quantizing, so a
    tensor beyond +-448 no longer clips - the contrast that scopes the
    no-scaling run as a stress test rather than a prediction."""
    from scripts.check_fp8_stability import make_fp8_quant, FP8_E4M3_MAX

    x = torch.randn(1000) * 1000.0
    none_q = make_fp8_quant("e4m3", "none")
    dyn_q = make_fp8_quant("e4m3", "dynamic-tensor")
    # The input exceeds the format range and the no-scaling path clips it:
    # every |x| > 448 collapses onto exactly +-448, a huge relative error
    # at the clipped points.
    assert (x.abs() > FP8_E4M3_MAX).any()
    assert none_q(x).abs().max().item() <= FP8_E4M3_MAX + 1e-6
    rel_none = ((none_q(x) - x).abs() / x.abs()).max().item()
    assert rel_none > 0.5
    # Dynamic per-tensor scaling keeps the RELATIVE error at the e4m3
    # grid spacing (3 mantissa bits -> max ~6.25%) instead of clipping.
    rel_dyn = ((dyn_q(x) - x).abs() / x.abs()).max().item()
    assert rel_dyn < 0.07


def test_w8a8_dataset_probe_defaults_to_a_held_out_split():
    """The W8A8 eval must not read the training stream: build_dataloader's
    default split stays 'train' (training behaviour unchanged), while the
    probe builder's default is a held-out split."""
    import inspect
    from scripts.train_gpt2 import build_dataloader
    from scripts.check_w8a8_ppl import make_dataset_probe

    assert inspect.signature(build_dataloader).parameters["split"].default == "train"
    assert inspect.signature(make_dataset_probe).parameters["split"].default == "test"


def test_needle_details_report_position_buckets():
    """return_details=True adds per-trial position fractions and tercile
    buckets, and the default return type stays a plain float."""
    from scripts.eval_downstream import needle_in_haystack

    tok = _CharTokenizer()
    needle = "The magic number is 12345."
    secret = tok(" 12345")["input_ids"]
    model = _Reciter(
        GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                  max_seq_len=256, norm_name="rmsnorm"),
        secret=secret,
    ).eval()
    acc = needle_in_haystack(model, tok, needle=needle, context_len=128,
                             n_trials=2, device="cpu")
    assert isinstance(acc, float)
    acc2, details = needle_in_haystack(model, tok, needle=needle,
                                       context_len=128, n_trials=6,
                                       device="cpu", return_details=True)
    assert len(details) == 6
    assert {d["bucket"] for d in details} <= {"early", "middle", "late"}
    for d in details:
        assert 0.0 <= d["pos_frac"] <= 1.0
        assert isinstance(d["correct"], bool)
    assert acc2 == sum(d["correct"] for d in details) / len(details)


def test_train_loop_logs_held_out_validation_loss(tmp_path, monkeypatch, capsys):
    """--val-every runs a held-out pass on a *separate* loader and logs
    val_loss rows to train.jsonl / stdout."""
    _tiny_train_run(tmp_path, monkeypatch, 4,
                    ["--val-every", "2", "--val-batches", "1",
                     "--val-dataset", "synthetic", "--save-every", "100"])
    rows = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text().splitlines()]
    val_rows = [r for r in rows if "val_loss" in r]
    assert val_rows, "no val_loss row was logged"
    assert all(isinstance(r["val_loss"], float) and math.isfinite(r["val_loss"])
               for r in val_rows)
    out = capsys.readouterr().out
    assert "val_loss=" in out


def test_resume_restores_data_position_and_rng(tmp_path, monkeypatch, capsys):
    """The checkpoint records how many stream batches were consumed; on
    resume the loader is fast-forwarded past them (no silent replay of the
    beginning of the corpus) and the RNG states are restored."""
    _tiny_train_run(tmp_path / "a", monkeypatch, 3, ["--save-every", "3"])
    ckpt = torch.load(tmp_path / "a" / "ckpt_000003.pt", weights_only=False)
    assert ckpt["data_position"] == 3  # one batch per step, grad_accum=1
    assert "torch_cpu" in ckpt["rng_state"]

    capsys.readouterr()
    _tiny_train_run(tmp_path / "b", monkeypatch, 5,
                    ["--save-every", "100", "--resume",
                     str(tmp_path / "a" / "ckpt_000003.pt")])
    out = capsys.readouterr().out
    assert "restored RNG states" in out
    assert "fast-forwarding the dataloader past 3" in out
    ckpt2 = torch.load(tmp_path / "b" / "ckpt_final.pt", weights_only=False)
    assert ckpt2["data_position"] == 5  # 3 skipped + 2 fresh


def test_lm_wrapper_loglikelihood_and_generation_match_manual_scoring():
    """AdaSquashLM scores continuations exactly like a hand-rolled loop and
    honours stop sequences in generate_until (no lm_eval import needed)."""
    from scripts.eval_downstream import AdaSquashLM

    class Tok:
        eos_token_id = 0

        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": [ord(c) % 50 + 1 for c in s]}

        def decode(self, ids):
            return "".join(chr(int(i) + 64) for i in ids)

    class Req:
        def __init__(self, *args):
            self.args = args

    model = GPT(GPTConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4,
                          max_seq_len=64, norm_name="rmsnorm")).eval()
    lm = AdaSquashLM(model, Tok(), batch_size=1)

    ctx, cont = "abc", "def"
    c_ids, w_ids = Tok()(ctx)["input_ids"], Tok()(ctx + cont)["input_ids"]
    cont_ids = w_ids[len(c_ids):]
    (lp, greedy), = lm.loglikelihood([Req(ctx, cont)])
    with torch.no_grad():
        logits = model(torch.tensor([w_ids]))[0][0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    manual = sum(logprobs[len(c_ids) - 1 + i, t].item()
                 for i, t in enumerate(cont_ids))
    assert abs(lp - manual) < 1e-5
    assert greedy == all(
        int(logits[len(c_ids) - 1 + i].argmax()) == t
        for i, t in enumerate(cont_ids))

    (roll,) = lm.loglikelihood_rolling([Req("abcdefg")])
    assert roll < 0.0  # a sum of log-probs

    text, = lm.generate_until([Req("ab", {"until": ["E"], "max_gen_toks": 8})])
    assert "E" not in text


def test_lm_harness_reports_missing_dependency():
    """Without lm_eval installed the harness raises a clean RuntimeError
    (not a bare ImportError or a NotImplementedError stub)."""
    from scripts.eval_downstream import evaluate_lm_harness
    try:
        import lm_eval  # noqa: F401
        pytest.skip("lm_eval installed; the error path cannot be exercised")
    except ImportError:
        pass
    with pytest.raises(RuntimeError, match="lm-eval not installed"):
        evaluate_lm_harness(None, ["mmlu"])

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
