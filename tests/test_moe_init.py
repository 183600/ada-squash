"""Initialization-control checks for the paired MoE comparisons.

A seed is only a valid block when the same seed fixes every *non-treatment*
parameter. In the MoE study the treatment is the normalization layer; the
transformer, router and expert banks must therefore be identical across norms
for a given seed. This is easy to break because both the norm constructors and
the expert banks draw from the global RNG at build time: DyT-C's alpha jitter
consumes ``randn`` while LayerNorm does not, so building the arms in sequence
would give the later linear/expert initializations different draws.

The tests pin the two properties the runner relies on:

* building with ``init=False`` and calling ``reset_parameters(seed)`` makes all
  non-norm parameters identical across norm arms;
* that reset really re-initializes the raw expert banks (they are not
  ``nn.Linear`` modules, so the generic ``_init_weights`` branches miss them).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.model import GPTConfig
from adasquash.repro import set_all_seeds
from scripts.train_moe_tiny import MoEGPT


def _cfg(norm: str) -> GPTConfig:
    return GPTConfig(vocab_size=64, dim=24, n_layers=2, n_heads=4,
                     max_seq_len=12, mlp_ratio=2.0, norm_name=norm)


def _build(norm: str, seed: int = 42) -> MoEGPT:
    set_all_seeds(seed)
    with torch.device('cpu'):
        model = MoEGPT(_cfg(norm), n_experts=3, init=False)
    model.reset_parameters(seed)
    return model


def _build_with_constructor_init(norm: str, seed: int) -> MoEGPT:
    """Build with the constructor pass, leaving a non-reproducible pre-reset state."""
    set_all_seeds(seed)
    with torch.device('cpu'):
        return MoEGPT(_cfg(norm), n_experts=3, init=True)


def _non_norm_parameters(model: MoEGPT) -> dict[str, torch.Tensor]:
    out = {}
    for name, value in model.named_parameters():
        # Norm parameters are the treatment (and have a different shape per
        # norm family), so they are deliberately excluded from the pairing.
        if any(marker in name for marker in ('.norm1.', '.norm2.', 'norm_f.')):
            continue
        out[name] = value.detach().clone()
    return out


def test_seed_controls_every_non_norm_parameter_across_norm_arms():
    norms = ('ln', 'rmsnorm', 'dyt', 'dytc', 'dytd', 'dytcd')
    reference = _non_norm_parameters(_build(norms[0]))
    assert reference  # a vacuous comparison would pass without any weight

    for norm in norms[1:]:
        candidate = _non_norm_parameters(_build(norm))
        assert set(candidate) == set(reference), (
            f'{norm} changed the non-norm parameter set; the arms are not '
            'comparable parameter-for-parameter')
        for name, expected in reference.items():
            assert torch.equal(candidate[name], expected), (
                f'{name} differs between ln and {norm} at the same seed: the '
                'norm constructor perturbed the shared initialization stream')


def test_reset_reinitializes_the_raw_expert_banks():
    # Construct with the constructor pass enabled so each model starts from a
    # different expert-bank state, then scramble the global RNG again before
    # the controlled reset. If the reset skipped the expert banks, their values
    # would still reflect these different build/RNG histories.
    first = _build_with_constructor_init('dytc', 1)
    first_experts_before = first.blocks[0].moe.experts_w1.detach().clone()
    torch.manual_seed(1234)
    first.reset_parameters(7)

    second = _build_with_constructor_init('dytc', 2)
    second_experts_before = second.blocks[0].moe.experts_w1.detach().clone()
    torch.manual_seed(9876)
    second.reset_parameters(7)

    # The premise: the two builds really did start from different experts.
    assert not torch.equal(first_experts_before, second_experts_before)
    a = _non_norm_parameters(first)
    b = _non_norm_parameters(second)
    assert set(a) == set(b)
    for name in a:
        assert torch.equal(a[name], b[name]), (
            f'{name} was not re-initialized by reset_parameters; an expert '
            'bank or another raw Parameter escaped the controlled pass')
    # The expert banks themselves are part of the controlled state.
    assert torch.equal(first.blocks[0].moe.experts_w1,
                       second.blocks[0].moe.experts_w1)
