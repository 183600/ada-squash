from __future__ import annotations
"""Controlled ablation matrix for the DyT++ / AdaSquash proposal.

Comparison protocol (what the runner guarantees, and checks)
------------------------------------------------------------
* **Controlled variables.** Every row runs under one optimizer (AdamW,
  betas 0.9/0.95), one constant learning rate, one step count, one synthetic
  data protocol and one seed set. Rows may only set architectural /
  initialization knobs; a row that tries to set `lr`, `weight_decay`,
  `batch_size`, `max_steps`, `dropout`, ... is rejected by the loader, because
  a gain obtained by switching the optimizer is not a gain of the module.
* **Complete ablation coverage.** `validate_matrix` checks the leave-one-out
  set for the main proposal (`LOO_SPEC`): Full, Full-minus-per-channel,
  Full-minus-depth-decay, Full-minus-LayerScale, Full-minus-softsign,
  Full-minus-learnable-p and Full-minus-QK-squash must all be present and
  differ from Full in exactly the declared knob(s) - e.g. the per-channel
  contrast uses the frozen-delta row, not `dytd_full`, which also switches
  tanh -> softsign, and the activation family has its own `dytcd_tanh`
  comparator so no claimed mechanism is only ever removed together with
  another one. The attention-logits axis (DyT-QK, Variant C) is removed on
  both the main proposal and the strongest simple baseline, and gets the
  QK-Norm alternative as a comparator.
  `leave_one_out_comparison` then runs the paired Full-vs-Full-minus-X test
  (Holm-adjusted over mechanisms), so "each axis helps" is a tested claim
  rather than an eyeballed delta. `check_factorial` additionally verifies the
  four tanh cells of `FACTORIAL_SPEC`, and `factorial_comparison` reports the
  two main effects and their interaction, so a synergy claim is tested rather
  than inferred from Full's point estimate.
* **Statistical reliability.** Every row is replicated over >= 3 seeds by
  default and reported as mean +/- std across seeds; the row-vs-baseline
  significance test is a one-sided paired Student-t test on the per-seed
  means when both rows share >= 3 seeds, because at 3 seeds the normal
  approximation is anti-conservative. Two shared seeds are tested for
  description only and are never certified (df = 1); a single shared run is
  tested at the batch level for description only (batch noise is not
  run-to-run noise). Alias rows that reuse a summary are counted once, and
  each delta carries a 95% confidence interval, per-seed direction counts and
  a power-based minimum detectable delta. The reference baseline is the best
  *replicated* baseline (with the point-estimate best recorded alongside), so
  a lucky single-seed row cannot make certification impossible.
"""
import argparse
import dataclasses
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
from adasquash import build_gpt_124m, count_parameters, ActivationStats, LossSpikeCounter, markov_batch
from adasquash.attention import clear_logit_scratch_cache
from adasquash.diagnostics import diagnose_train_val_curve, triage_nonfinite
from adasquash.efficiency import analytic_flops_per_token, parameter_breakdown
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust, mean_std_stderr,
                               paired_t_one_sided, student_t_ppf, welch_t_one_sided)
from adasquash.model import build_gpt_350m, build_llama_tiny_1b, clear_ce_scratch_cache, clear_cast_weight_cache
from adasquash.repro import artifact_record, collect_run_metadata, file_fingerprint, finish_metadata, git_revision, save_run_manifest, set_all_seeds
from scripts.train_gpt2 import apply_config_to_parser, build_optimizer, load_yaml_config


def _finite_metric(value) -> bool:
    """A loss may be pooled/tested only when it is a finite number.

    `isinstance(value, (int, float))` alone admits NaN/Inf from a diverged
    run; NaN then propagates through the seed mean (and the paired tests),
    silently poisoning the row's pooled val_loss.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _paired_effect_summary(row: dict, reference: dict, common_seeds, test: dict | None=None,
                           alpha: float=0.05, power: float=0.8) -> dict:
    """Seed-level direction and resolution for one comparison.

    A p-value alone hides two things a reader needs: how many independent
    seeds moved in the claimed direction (with three seeds a 2/3 split can
    still produce a small t p-value) and the smallest delta the test could
    have detected at the requested power. The latter is what separates "no
    evidence of a difference" from "the design could not have seen one".
    Both are descriptive: they never change `beats_*_significantly`.
    """
    out = {
        'n_seeds_favoring': None,
        'n_seeds_tied': None,
        'n_seeds_against': None,
        'per_seed_delta': None,
        'paired_delta_std': None,
        'minimum_detectable_delta_95_power80': None,
    }
    common = list(common_seeds or [])
    if len(common) >= 2:
        row_seeds = _seed_means(row)
        ref_seeds = _seed_means(reference)
        deltas = {int(seed): float(row_seeds[seed] - ref_seeds[seed])
                  for seed in common if seed in row_seeds and seed in ref_seeds}
        if deltas:
            values = list(deltas.values())
            stats = mean_std_stderr(values)
            out.update({
                'n_seeds_favoring': sum(v < 0.0 for v in values),
                'n_seeds_tied': sum(v == 0.0 for v in values),
                'n_seeds_against': sum(v > 0.0 for v in values),
                'per_seed_delta': {str(seed): value for seed, value in deltas.items()},
                'paired_delta_std': None if stats is None else stats['std'],
            })
    if test is not None and test.get('df') and isinstance(test.get('ci_low'), (int, float)):
        half = (test['ci_high'] - test['ci_low']) / 2.0
        critical = student_t_ppf(1.0 - alpha / 2.0, float(test['df']))
        if critical > 0.0:
            stderr = half / critical
            out['minimum_detectable_delta_95_power80'] = (
                student_t_ppf(1.0 - alpha, float(test['df']))
                + student_t_ppf(power, float(test['df']))
            ) * stderr
    return out


ABLATIONS = {'ln_baseline': dict(norm_name='ln'), 'rmsnorm_baseline': dict(norm_name='rmsnorm'), 'dyt_no_perchannel': dict(norm_name='dyt'), 'dytc_perchannel_tanh_k0': dict(norm_name='dytcd_tanh', k_init=0.0, learnable_k=False), 'dytcd_full': dict(norm_name='dytcd'), 'dytcd_no_perchannel': dict(norm_name='dytcd', learnable_delta=False), 'dytcd_k0_no_depth_schedule': dict(norm_name='dytcd', k_init=0.0, learnable_k=False), 'dytd_k0_no_depth_schedule': dict(norm_name='dytd', k_init=0.0, learnable_k=False), 'dytd_full': dict(norm_name='dytd'), 'dytcd_no_ls': dict(norm_name='dytcd', use_layerscale=False), 'dytcd_no_softsign': dict(norm_name='dytcd_tanh'), 'dytcd_fixed_p': dict(norm_name='dytcd', p_init=2.0, learnable_p=False), 'dytcd_ls1e5': dict(norm_name='dytcd', layerscale_init=1e-05), 'dytcd_ls1e3': dict(norm_name='dytcd', layerscale_init=0.001), 'dytc_p1_softsign': dict(norm_name='dytc', p_init=1.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p2_softsign': dict(norm_name='dytc', p_init=2.0, learnable_p=False, alpha_jitter=0.0), 'dytc_p4_softsign': dict(norm_name='dytc', p_init=4.0, learnable_p=False, alpha_jitter=0.0), 'dytc_plearnable': dict(norm_name='dytc', p_init=2.0, learnable_p=True, alpha_jitter=0.0), 'dytc_hardtanh_eq': dict(norm_name='dytc', p_init=64.0, learnable_p=False, alpha_jitter=0.0), 'dyt_alpha_0.3': dict(norm_name='dyt', alpha_init=0.3), 'dyt_alpha_0.5': dict(norm_name='dyt', alpha_init=0.5), 'dyt_alpha_1.0': dict(norm_name='dyt', alpha_init=1.0), 'rmsnorm_qk': dict(norm_name='rmsnorm', qk_norm='dytqk'), 'dytcd_qk': dict(norm_name='dytcd', qk_norm='dytqk'), 'dytcd_qknorm': dict(norm_name='dytcd', qk_norm='qknorm')}
# Every row must state *why* it is run (the scientific hypothesis it tests),
# not just what it configures. `load_ablations_from_config` enforces this for
# YAML rows; these are the hypotheses of the built-in fallback rows.
ABLATION_HYPOTHESES = {
    'ln_baseline': 'Simple statistical baseline: LayerNorm on the identical protocol. The DyT family must beat this untuned-but-fairly-run reference before any added machinery counts as progress.',
    'rmsnorm_baseline': 'Simple statistical baseline: RMSNorm, the strongest no-mean normalization used in modern LLMs. A DyT-family gain is only meaningful relative to this reference.',
    'dyt_no_perchannel': 'Original DyT (one scalar alpha, tanh) matches or beats the LayerNorm/RMSNorm references at this scale, reproducing the paper claim under our protocol.',
    'dytc_perchannel_tanh_k0': 'Cross-activation per-channel isolation at flat depth: against dytd_k0_no_depth_schedule this row changes exactly norm_name, keeping frozen k=0, tanh, gamma/beta and LayerScale identical, so a lower loss is attributable to the per-channel factor under tanh rather than to the simultaneous tanh -> softsign switch in dytc_p2_softsign vs dyt_no_perchannel.',
    'dytcd_full': 'Combining the per-channel and depth axes beats both single-axis variants under the same protocol. The leave-one-out contrasts, not this row alone, identify which component is load-bearing; no positive-interaction claim is made without a 2x2 factorial.',
    'dytcd_no_perchannel': 'Leave-one-out for the per-channel axis of the main proposal: with delta frozen at zero (a zero buffer, not a parameter) alpha_c == alpha_l exactly, while the depth schedule, the softsign family and the LayerScale residual stay identical to the full model. Unlike the dytd_full row, this contrast changes no other mechanism, so a loss increase here is attributable to the per-channel deviation alone.',
    'dytcd_k0_no_depth_schedule': 'Leave-one-out for the depth axis of the main proposal: with k pinned to 0 and frozen, DyT-CD keeps its per-channel factor, softsign and LayerScale, so any loss of the full model advantage is attributable to depth decay.',
    'dytd_k0_no_depth_schedule': 'Depth decay is the source of DyT-D gain: with k pinned to 0 and frozen, dytd_full advantage should vanish.',
    'dytd_full': 'Depth-conditioned alpha (+ LayerScale) beats scalar DyT here: shallow strong squashing + deep relaxed response helps this depth.',
    'dytcd_no_ls': 'The LayerScale residual gate contributes to DyT-CD; removing it (use_layerscale off) hurts held-out loss.',
    'dytcd_no_softsign': 'Leave-one-out for the activation family of the main proposal: the dytcd_tanh comparator keeps the per-channel x depth alpha, the LayerScale residual and gamma/beta identical to dytcd_full and swaps only the generalized softsign for tanh, so a loss increase here is attributable to the activation family alone. Unlike the dytd_full row, it does not also remove the per-channel factor, so the dytd -> dytcd bundle is no longer the only activation-family evidence.',
    'dytcd_fixed_p': 'Leave-one-out for the learnable saturation exponent of the main proposal: with p frozen at its init (2.0) the model keeps every mechanism of dytcd_full except the per-layer adaptation of the softsign steepness, so a loss increase here is attributable to learnable p alone.',
    'dytcd_ls1e5': 'LayerScale init 1e-5 (the DyT-CD default) is a competitive residual-gate scale, not an arbitrary strawman.',
    'dytcd_ls1e3': 'LayerScale init 1e-3 trades warmup stability for stronger early residual growth; the default should not be beaten by it.',
    'dytc_p1_softsign': 'Softsign sharpness matters: p=1 (classic softsign) is worse than p=2 on this outlier-heavy corpus.',
    'dytc_p2_softsign': 'p=2 (smooth softsign) is the best fixed saturation exponent of {1, 2, 4} for outlier control with usable gradients.',
    'dytc_p4_softsign': 'p=4 saturates harder than p=2; the hypothesis is that this hurts gradient flow and loses to p=2.',
    'dytc_plearnable': 'A learnable p adapts the saturation steepness per layer and beats every fixed p in the sweep.',
    'dytc_hardtanh_eq': 'The p->infinity limit (hard tanh) starves gradients outside [-1,1] and loses to finite p.',
    'dyt_alpha_0.3': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_0.5': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'dyt_alpha_1.0': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.',
    'rmsnorm_qk': 'Attention-logits axis, standalone: adding the tanh Q/K scalar squashing to the strongest simple baseline (RMSNorm) changes exactly qk_norm, so a held-out-loss reduction is attributable to the QK squashing alone rather than to an interaction with the DyT-CD stack.',
    'dytcd_qk': 'Leave-one-out for the attention-logits axis of the proposal (DyT-QK, Variant C) in combination: with the norm stack, optimizer, data and qk_alpha_init (shared default 1.0) identical to dytcd_full, the only effective difference is qk_norm = dytqk, so the paired Full-minus-QK contrast attributes any held-out-loss change to the tanh Q/K squashing alone.',
    'dytcd_qknorm': 'QK-Norm comparator for the attention-logits axis: per-head RMSNorm on Q/K is the standard alternative the DyT-QK variant claims to sharpen, run on the identical protocol so a dytcd_qk win is only credited if it also beats this comparator, not just the no-QK row.',
}
_MODEL_BUILDERS = {'gpt2-124m': build_gpt_124m, 'gpt2-350m': build_gpt_350m, 'llama-1b': build_llama_tiny_1b}
_ABLATION_METADATA_KEYS = ('hypothesis',)

# Knobs that the matrix runner holds fixed by construction: every row gets the
# same optimizer (AdamW, fixed betas), learning rate, weight decay, step count,
# batch size, sequence length, data generator and seed set. A row that sets one
# of these is not an ablation of the proposed module - it is a different
# optimization protocol whose gain cannot be attributed to the module - so the
# loader rejects it instead of silently running it.
_FORBIDDEN_ROW_KEYS = frozenset({
    'lr', 'learning_rate', 'weight_decay', 'beta1', 'beta2', 'betas',
    'optimizer', 'scheduler', 'schedule', 'warmup', 'min_lr_ratio',
    'max_steps', 'epochs', 'batch_size', 'grad_accum', 'seq_len',
    'grad_clip', 'dropout', 'bf16', 'seed', 'seeds', 'dataset',
    'dataset_config', 'tokenizer', 'augmentation',
})

# Default number of random seeds for the matrix. One seed cannot separate a
# method effect from run-to-run noise, so the key experiment is replicated
# over >= 3 seeds and reported as mean +/- std across seeds (see
# `aggregate_seed_summaries` and `baseline_comparison`).
DEFAULT_SEEDS = (42, 43, 44)

# Leave-one-out (LOO) coverage of the main proposal DyT-CD. Each entry names
# one claimed mechanism and the matrix row that removes exactly it. The claim
# "mechanism X helps" is supportable only when the Full model and its X-free
# reference ran on the identical protocol; `validate_matrix` refuses to run
# the matrix when a mechanism is never isolated. Each removal must change the
# declared knobs *and nothing else*: the `dytd_full` row also switches
# tanh -> generalized softsign, so it cannot isolate the per-channel axis (it
# stays in the matrix as the depth-axis reference); the per-channel contrast
# uses `dytcd_no_perchannel`, whose only effective difference from Full is the
# frozen delta buffer. The activation family gets the same treatment through
# `dytcd_tanh` (plain-autograd comparator, per-channel x depth alpha kept),
# so Full-minus-softsign changes exactly `norm_name`, and learnable p through
# the frozen-p row (`p_init` explicit at its default so only `learnable_p`
# differs). The attention-logits axis (DyT-QK, Variant C) is isolated once on
# the main proposal (`dytcd_qk`, only `qk_norm` differs from Full) and once on
# the strongest simple baseline (`rmsnorm_qk`), so the QK squashing is tested
# both in combination and standalone; `dytcd_qknorm` is the QK-Norm
# alternative comparator, not a removal.
LOO_SPEC = {
    'dytcd_full': {
        'per_channel_delta': {'row': 'dytcd_no_perchannel',
                              'diff': ('learnable_delta',)},
        'depth_decay': {'row': 'dytcd_k0_no_depth_schedule',
                        'diff': ('k_init', 'learnable_k')},
        'layerscale_residual': {'row': 'dytcd_no_ls',
                                'diff': ('use_layerscale',)},
        'activation_family': {'row': 'dytcd_no_softsign',
                              'diff': ('norm_name',)},
        'learnable_p': {'row': 'dytcd_fixed_p',
                        'diff': ('learnable_p',)},
    },
    'dytc_perchannel_tanh_k0': {
        'per_channel_axis_tanh': {'row': 'dytd_k0_no_depth_schedule',
                                  'diff': ('norm_name',)},
    },
    'rmsnorm_qk': {
        'qk_squash': {'row': 'rmsnorm_baseline',
                      'diff': ('qk_norm',)},
    },
    'dytcd_qk': {
        'qk_squash': {'row': 'dytcd_full',
                      'diff': ('qk_norm',)},
    },
}

# A complete 2x2 factorial for the two axes the main proposal combines. The
# four cells all use tanh, the same LayerScale residual and the same scalar
# initialization context, so the only moving parts are the per-channel factor
# and the depth schedule. Leave-one-out proves each axis helps from Full;
# this design additionally identifies a *positive interaction* (synergy)
# rather than assuming it from Full's point estimate. `check_factorial`
# verifies that switching a factor changes exactly its declared knobs in both
# arms of the other factor, so a confounded cell is rejected before training.
FACTORIAL_SPEC = {
    'per_channel_x_depth_tanh': {
        'context': ('tanh activation family, LayerScale on, same gamma/beta and '
                    'same alpha/k initialization context across the four cells'),
        'cells': {
            'neither': 'dytd_k0_no_depth_schedule',
            'per_channel': 'dytc_perchannel_tanh_k0',
            'depth': 'dytd_full',
            'both': 'dytcd_no_softsign',
        },
        'factors': {
            'per_channel': {
                'diff': ('norm_name',),
                'pairs': (('neither', 'per_channel'), ('depth', 'both')),
            },
            'depth': {
                'diff': ('k_init', 'learnable_k'),
                'pairs': (('neither', 'depth'), ('per_channel', 'both')),
            },
        },
    },
}

def _effective_cfg(model_name: str, cfg_kwargs: dict) -> dict:
    """The full GPTConfig a matrix row builds, as a plain dict."""
    with torch.device('meta'):
        cfg = _MODEL_BUILDERS[model_name](init=False, **cfg_kwargs).cfg
    return dataclasses.asdict(cfg)

def effective_config_key(model_name: str, cfg_kwargs: dict) -> str:
    """Canonical identity of the model a matrix row trains.

    Rows may spell out builder/GPTConfig defaults explicitly (e.g.
    ``layerscale_init: 1.0e-5`` or ``alpha_init: 0.5``); two rows with equal
    keys build the exact same GPTConfig and, under the same seed, produce
    bit-identical runs. The runner trains the first and reuses its summary
    for the rest.
    """
    return json.dumps(_effective_cfg(model_name, cfg_kwargs), sort_keys=True)

def _check_row_keys(name: str, cfg: dict, source: str) -> None:
    bad = sorted(set(cfg) & _FORBIDDEN_ROW_KEYS)
    if bad:
        raise ValueError(
            f"{source}: ablation row '{name}' sets optimizer/schedule/data "
            f"knob(s) {bad}; those are held fixed for every row so a "
            f"comparison isolates the module under test instead of a "
            f"different training recipe"
        )

def check_leave_one_out(rows: dict, model_name: str, spec: dict | None=None) -> dict:
    """Verify every claimed mechanism is isolated by a Full-minus-X row.

    For each `(full, mechanism) -> {row, diff}` entry the reference row must
    exist and its *effective* model config must differ from the Full config in
    exactly the declared fields - no more (a second change would confound the
    contrast) and no fewer (a no-op row proves nothing). Raises ValueError
    listing every violation so one broken claim cannot hide behind another.
    """
    spec = LOO_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for full_name, removals in spec.items():
        if full_name not in rows:
            problems.append(f"leave-one-out full row '{full_name}' is missing")
            continue
        full_cfg = _effective_cfg(model_name, rows[full_name])
        report[full_name] = {}
        for mechanism, info in removals.items():
            ref_name = info['row']
            declared = set(info['diff'])
            if ref_name not in rows:
                problems.append(
                    f"'{mechanism}' has no Full-minus-X reference: row "
                    f"'{ref_name}' is missing")
                continue
            ref_cfg = _effective_cfg(model_name, rows[ref_name])
            actual = {k for k in full_cfg if full_cfg[k] != ref_cfg[k]}
            actual |= {k for k in ref_cfg if k not in full_cfg}
            if actual != declared:
                problems.append(
                    f"'{mechanism}': {ref_name} vs {full_name} differ in "
                    f"{sorted(actual)}, expected exactly {sorted(declared)}")
                continue
            report[full_name][mechanism] = ref_name
    if problems:
        raise ValueError('leave-one-out coverage is incomplete: ' + '; '.join(problems))
    return report

def check_factorial(rows: dict, model_name: str, spec: dict | None=None) -> dict:
    """Verify every declared 2x2 factorial is a clean, complete design.

    Each factor is switched on/off in *both* arms of the other factor (e.g.
    per-channel must be the only difference both with depth off and with
    depth on). A factor that changes different extra knobs in the two arms
    would make the interaction term a mixture of effects, so this is rejected
    before any run starts rather than discovered after the matrix finishes.
    """
    spec = FACTORIAL_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for design_name, design in spec.items():
        cells = design['cells']
        missing = [name for name in cells.values() if name not in rows]
        if missing:
            problems.append(
                f"factorial '{design_name}' is missing cell row(s) {sorted(missing)}")
            continue
        cfgs = {cell: _effective_cfg(model_name, rows[row])
                for cell, row in cells.items()}
        report[design_name] = {'cells': dict(cells), 'factors': {}}
        for factor, info in design['factors'].items():
            declared = set(info['diff'])
            clean = True
            for off_cell, on_cell in info['pairs']:
                off_cfg, on_cfg = cfgs[off_cell], cfgs[on_cell]
                actual = {key for key in off_cfg if off_cfg[key] != on_cfg[key]}
                actual |= {key for key in on_cfg if key not in off_cfg}
                if actual != declared:
                    problems.append(
                        f"factorial '{design_name}', factor '{factor}': "
                        f"{cells[on_cell]} vs {cells[off_cell]} differ in "
                        f"{sorted(actual)}, expected exactly {sorted(declared)}")
                    clean = False
                    break
            if clean:
                report[design_name]['factors'][factor] = {
                    'diff': sorted(declared),
                    'pairs': [list(pair) for pair in info['pairs']],
                }
    if problems:
        raise ValueError('factorial coverage is incomplete: ' + '; '.join(problems))
    return report

def validate_matrix(rows: dict, hypotheses: dict, model_name: str) -> dict:
    """Every static check the matrix must pass before any training starts."""
    missing = []
    for name, cfg in rows.items():
        _check_row_keys(name, cfg, 'matrix')
        if not str(hypotheses.get(name) or '').strip():
            missing.append(name)
    if missing:
        raise ValueError(
            f"ablation rows without a stated scientific hypothesis: {sorted(missing)}")
    return {'rows': sorted(rows),
            'leave_one_out': check_leave_one_out(rows, model_name),
            'factorial': check_factorial(rows, model_name)}

def install_outlier_channel(model, scale: float=25.0) -> int:
    with torch.no_grad():
        oid = model.cfg.vocab_size - 1
        model.tok_emb.weight[oid] *= scale
    return oid

def load_ablations_from_config(path: str, include_metadata: bool = False) -> dict:
    """Load the ablation matrix from YAML.

    Returns `{name: cfg_kwargs}` by default (the shape every existing caller
    expects). With `include_metadata=True` each value is
    `{'cfg': cfg_kwargs, 'hypothesis': str}` and every row is required to
    state a scientific hypothesis - an ablation without a stated hypothesis
    is a hyper-parameter lottery, not an experiment, so it is rejected rather
    than silently run.
    """
    cfg = load_yaml_config(path)
    rows = cfg.get('ablations')
    if not rows:
        return {}
    out = {}
    missing_hypothesis = []
    for row in rows:
        row = dict(row)
        name = row.pop('name')
        metadata = {key: row.pop(key) for key in _ABLATION_METADATA_KEYS if key in row}
        if 'norm' in row:
            row['norm_name'] = row.pop('norm')
        _check_row_keys(name, row, path)
        if name in out:
            raise ValueError(f"{path}: duplicate ablation name '{name}' - each row must be unique or an intended comparison is silently lost")
        if include_metadata:
            hypothesis = str(metadata.get('hypothesis') or '').strip()
            if not hypothesis:
                missing_hypothesis.append(name)
            out[name] = {'cfg': row, 'hypothesis': hypothesis or None}
        else:
            out[name] = row
    if missing_hypothesis:
        raise ValueError(
            f"{path}: every ablation row must declare its scientific hypothesis "
            f"(`hypothesis:`); missing for {sorted(missing_hypothesis)}"
        )
    return out

def run_one(name: str, cfg_kwargs: dict, args, output_dir: Path, run_metadata: dict | None=None, hypothesis: str | None=None, seed: int | None=None):
    if seed is None:
        seed = getattr(args, 'seed', None)
    if seed is None:
        seed = (getattr(args, 'seeds', None) or DEFAULT_SEEDS)[0]
    seed = int(seed)
    print(f'\n[abl] {name} (seed={seed}): {cfg_kwargs}')
    print(f'  [hypothesis] {hypothesis or "(none stated - this row cannot be interpreted as a test)"}')
    output_dir.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seed)
    # Build with the constructor's init pass disabled:
    # `reset_parameters(seed)` below is the run's single, seeded
    # initialization.
    with torch.device(device):
        model = _MODEL_BUILDERS[args.model](init=False, **cfg_kwargs)
    model.reset_parameters(seed)
    model.to(device)
    n_params = count_parameters(model)
    param_breakdown = parameter_breakdown(model)
    flops = analytic_flops_per_token(model.cfg, seq_len=args.seq_len)
    opt = build_optimizer(model, args.lr, args.weight_decay, 0.9, 0.95)
    rng = torch.Generator(device=device).manual_seed(seed)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    outlier_id = install_outlier_channel(model)
    losses = []
    stats_windowed = False
    # In-training curve monitoring. The held-out stream for the curve is a
    # *separate* generator (seed + 2): the selection metric's generator
    # (seed + 1) is only consumed after training, so the per-batch val losses
    # that rows are paired on stay bit-identical whether or not the curve ran.
    # Without this a row is summarized by its final number alone, and a delta
    # produced while one arm was still underfit/overfit reads as a norm effect.
    curve_every = int(getattr(args, 'curve_every', 0) or 0)
    if curve_every == 0:
        curve_every = max(1, int(args.max_steps) // 5)
    elif curve_every < 0:
        curve_every = 0
    curve_batches = max(1, int(getattr(args, 'curve_batches', 5) or 5))
    curve_rng = torch.Generator(device=device).manual_seed(seed + 2)
    curve_train: list = []
    curve_val: list = []
    curve_val_stderr: list = []
    curve_steps: list = []
    curve_diagnoses: list = []
    curve_window_start = 0
    t0 = time.time()
    for step in range(args.max_steps):
        x, y = markov_batch(rng, model.cfg.ignore_index, args.batch_size, args.seq_len, device, outlier_id=outlier_id)
        measure = step % args.stats_every == 0
        if measure:
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        loss = model(x, y, return_logits=False)[1]
        # Backward is enqueued first; `loss` is not mutated by it, so the
        # host read below returns the same value.
        loss.backward()
        loss_val = float(loss.detach())
        if not math.isfinite(loss_val):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=args.lr, grad_clip=1.0)
            raise SystemExit(
                f"[abl][nan] '{name}' loss is non-finite at step {step} "
                f"(seed={seed}): triage stage='{triage['stage']}' "
                f"({triage['diagnosis']}); fixes: " + ' | '.join(triage['recommendations'])
            )
        if measure:
            act_stats.detach()
            clear_logit_scratch_cache()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not math.isfinite(float(grad_norm)):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=args.lr, grad_clip=1.0)
            raise SystemExit(
                f"[abl][nan] '{name}' produced a non-finite gradient at step {step} "
                f"(pre-clip norm={float(grad_norm)}, seed={seed}): triage stage="
                f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
                + ' | '.join(triage['recommendations'])
            )
        opt.step()
        opt.zero_grad(set_to_none=True)
        clear_cast_weight_cache()
        loss_spike.update(loss_val)
        losses.append(loss_val)
        if step % args.log_every == 0:
            print(f'  [step {step:4d}] loss={loss_val:.4f}', flush=True)
        if curve_every > 0 and (step + 1) % curve_every == 0:
            window = losses[curve_window_start:]
            curve_window_start = len(losses)
            train_point = sum(window) / max(1, len(window))
            model.eval()
            with torch.inference_mode():
                held_t = torch.stack([
                    model(*markov_batch(curve_rng, model.cfg.ignore_index, args.batch_size,
                                        args.seq_len, device, outlier_id=outlier_id),
                          return_logits=False)[1]
                    for _ in range(curve_batches)
                ])
            model.train()
            # `.tolist()` widens each fp32 value to the same Python double the
            # explicit fp64 cast produced.
            held = held_t.tolist()
            held_stats = mean_std_stderr(held)
            held_stderr = (held_stats['stderr']
                           if held_stats and held_stats['n'] >= 2 else None)
            curve_train.append(train_point)
            curve_val.append(held_stats['mean'])
            curve_val_stderr.append(held_stderr)
            curve_steps.append(step + 1)
            diagnosis = diagnose_train_val_curve(
                curve_train, curve_val, train_steps=curve_steps, val_steps=curve_steps,
                log_vocab_size=math.log(model.cfg.vocab_size),
                val_stderr=curve_val_stderr,
            )
            curve_diagnoses.append({
                'step': step + 1,
                'classification': diagnosis['classification'],
                'train_trend': diagnosis['train_trend'],
                'val_trend': diagnosis['val_trend'],
                'train_window_loss': train_point,
                'val_loss': curve_val[-1],
                'val_stderr': held_stderr,
                'val_noise_threshold': diagnosis['evidence'].get('val_noise_threshold'),
                'val_change_within_noise': diagnosis['evidence'].get('val_change_within_noise'),
                'generalization_gap': diagnosis['generalization_gap'],
                'recommendations': diagnosis['recommendations'],
            })
            if diagnosis['classification'] != 'insufficient_data':
                print(f"  [curve] step={step + 1} train_window={train_point:.4f} "
                      f"val={curve_val[-1]:.4f} -> {diagnosis['classification']} "
                      f"(train {diagnosis['train_trend']}, val {diagnosis['val_trend']})", flush=True)
                if diagnosis['classification'] not in ('healthy', 'insufficient_data'):
                    for line in diagnosis['recommendations']:
                        print(f'  [curve]   action: {line}')
    if not losses:
        raise ValueError(f"ablation '{name}' produced no steps")
    # Held-out validation loss: fresh synthetic batches from a generator the
    # optimizer has never seen (seed + 1), evaluated with no gradients. Model
    # comparisons must use this - the training loss above is fitted data.
    n_val_batches = max(1, getattr(args, 'val_batches', 20))
    val_rng = torch.Generator(device=device).manual_seed(seed + 1)
    # Batches are materialized so a non-finite held-out loss can be triaged on
    # the exact batch that produced it (the generator state advances, so
    # regenerating the batch later would not be the same input).
    val_batches = [markov_batch(val_rng, model.cfg.ignore_index, args.batch_size,
                                args.seq_len, device, outlier_id=outlier_id)
                   for _ in range(n_val_batches)]
    model.eval()
    with torch.inference_mode():
        val_losses = [model(x, y, return_logits=False)[1] for x, y in val_batches]
    bad_val = [(i, float(v)) for i, v in enumerate(val_losses) if not math.isfinite(float(v))]
    if bad_val:
        i_bad, v_bad = bad_val[0]
        triage = triage_nonfinite(model, val_batches[i_bad][0], val_batches[i_bad][1],
                                  device_type=device)
        model.train()
        raise SystemExit(
            f"[abl][nan] '{name}' held-out loss is non-finite on {len(bad_val)}/{n_val_batches} "
            f"val batch(es) (first index={i_bad}, value={v_bad}, seed={seed}): triage stage="
            f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
            + ' | '.join(triage['recommendations'])
        )
    val_losses = [float(v) for v in val_losses]
    model.train()
    # `val_loss` is a sample mean over held-out batches; keep the per-batch
    # losses (and their standard error) so comparisons can be *paired* across
    # rows (every row evaluates the identical batches, generated from the
    # fixed `seed + 1` generator) and report significance instead of a
    # bare point estimate.
    val_stats = mean_std_stderr(val_losses)
    val_loss = val_stats['mean']
    stats = act_stats.summary()
    # The knobs every matrix row shares. Recorded per summary so a reported
    # comparison can be audited for controlled variables after the fact.
    controls = {
        'optimizer': {'name': 'AdamW', 'betas': [0.9, 0.95], 'lr': args.lr,
                      'weight_decay': args.weight_decay,
                      'grad_clip': 1.0},
        'schedule': {'type': 'constant', 'lr': args.lr},
        'data': {'generator': 'markov_batch', 'batch_size': args.batch_size,
                 'seq_len': args.seq_len, 'outlier_id': outlier_id,
                 'outlier_scale': 25.0,
                 'val_seed_offset': 1, 'val_batches': n_val_batches},
        'curve_monitoring': {
            'every': curve_every, 'batches': curve_batches,
            'val_seed_offset': 2,
            'note': ('curve points are held-out; their generator (seed+2) is disjoint '
                     'from the selection-val generator (seed+1), so enabling the curve '
                     'does not change the paired val batches'),
        },
        'train_steps': args.max_steps,
        'seed': seed,
    }
    summary = {'name': name, 'hypothesis': hypothesis, 'cfg': cfg_kwargs, 'model': args.model, 'n_params': n_params, 'seed': seed, 'seeds': [seed], 'final_loss': losses[-1], 'mean_last_100_loss': sum(losses[-100:]) / max(1, len(losses[-100:])), 'val_loss': val_loss, 'val_losses': val_losses, 'val_loss_std': val_stats['std'], 'val_loss_stderr': val_stats['stderr'], 'val_loss_std_basis': 'val_batches', 'n_val_batches': n_val_batches, 'selection_metric': 'val_loss', 'n_loss_spikes': loss_spike.n_spikes, 'loss_curve': losses, 'act_stats': stats, 'max_abs_pre_norm': stats.get('__summary__', {}).get('max_abs_pre_norm'), 'mean_max_abs_pre_norm': stats.get('__summary__', {}).get('mean_max_abs'), 'mean_excess_kurtosis_pre_norm': stats.get('__summary__', {}).get('mean_excess_kurtosis_pre_norm'), 'controls': controls, 'elapsed_s': time.time() - t0}
    # Per-row model accounting is stored next to the loss.
    summary['parameter_breakdown'] = param_breakdown
    summary['flops_per_token_forward'] = flops['forward_per_token']
    summary['flops_per_token_training_est'] = flops['training_per_token']
    if run_metadata is None:
        run_metadata = collect_run_metadata(seed=seed)
    else:
        run_metadata = dict(run_metadata)
        run_metadata['seed'] = seed
    summary['run_metadata'] = run_metadata
    # The curve block: the per-point train/val pair and the regime verdict made
    # at each point, so a row is read as a training curve, not a final number.
    summary['curve_train_window_loss'] = curve_train
    summary['curve_val_loss'] = curve_val
    summary['curve_val_stderr'] = curve_val_stderr
    summary['curve_steps'] = curve_steps
    summary['curve_diagnosis'] = curve_diagnoses[-1] if curve_diagnoses else None
    summary['curve_diagnosis_history'] = curve_diagnoses
    summary['curve_classification'] = (curve_diagnoses[-1]['classification']
                                       if curve_diagnoses else None)
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    final_curve = ''
    if curve_diagnoses:
        last_curve = curve_diagnoses[-1]
        gap = last_curve['generalization_gap']
        gap_str = 'n/a' if gap is None else f'{gap:+.4f}'
        noise = last_curve.get('val_stderr')
        noise_str = '' if noise is None else f', val_se={noise:.4f}'
        final_curve = (f" curve={last_curve['classification']} "
                       f"(train {last_curve['train_trend']}, val {last_curve['val_trend']}, "
                       f"gap={gap_str}{noise_str})")
    print(f"  [done] final_loss={summary['final_loss']:.4f} val_loss={val_loss:.4f} "
          f"max_abs_pre_norm={summary['max_abs_pre_norm']}{final_curve}")
    # Drop the per-row scratch caches.
    clear_ce_scratch_cache()
    return summary

def aggregate_seed_summaries(name: str, summaries: list, cfg_kwargs: dict | None=None, hypothesis: str | None=None) -> dict:
    """Pool per-seed runs into one Mean +/- Std summary.

    The seed is the unit of replication: two runs with different seeds are
    independent samples of the same protocol, while the held-out batches
    inside one run are not (they all come from one generator). The pooled
    `val_loss` is therefore the mean of the per-seed means, `val_loss_std` is
    the sample std across seeds whenever there is more than one seed (with
    `val_loss_std_basis` recording which variability it describes), and the
    per-batch losses are kept tagged by seed so the comparison can pair rows
    seed by seed.
    """
    per_seed = []
    for s in summaries:
        seed = s.get('seed')
        if seed is None and isinstance(s.get('seeds'), list) and s['seeds']:
            seed = s['seeds'][0]
        per_seed.append({
            'seed': seed,
            'val_loss': s.get('val_loss'),
            'val_losses': s.get('val_losses'),
            'final_loss': s.get('final_loss'),
            'mean_last_100_loss': s.get('mean_last_100_loss'),
            'n_loss_spikes': s.get('n_loss_spikes'),
            'elapsed_s': s.get('elapsed_s'),
            # The final curve verdict of this run (None for summaries written
            # before curve monitoring existed, or for very short runs).
            'curve_classification': s.get('curve_classification'),
            'curve_diagnosis': s.get('curve_diagnosis'),
        })
    seed_losses = [p['val_loss'] for p in per_seed if _finite_metric(p['val_loss'])]
    stats = mean_std_stderr(seed_losses) or {'mean': None, 'std': None, 'stderr': None, 'n': 0}
    multi = len(seed_losses) >= 2

    def _mean_of(key):
        vals = [p[key] for p in per_seed if _finite_metric(p.get(key))]
        return sum(vals) / len(vals) if vals else None

    agg = dict(summaries[0]) if summaries else {}
    pooled: list = []
    by_seed: dict = {}
    for p in per_seed:
        if isinstance(p.get('val_losses'), list):
            pooled.extend(p['val_losses'])
            by_seed[str(p['seed'])] = p['val_losses']
    agg.update({
        'name': name,
        'cfg': cfg_kwargs if cfg_kwargs is not None else agg.get('cfg'),
        'hypothesis': hypothesis if hypothesis is not None else agg.get('hypothesis'),
        'seeds': [p['seed'] for p in per_seed],
        'n_seeds': len(seed_losses),
        'per_seed': per_seed,
        'val_loss': stats['mean'],
        'val_loss_seed_std': stats['std'] if multi else None,
        'val_loss_seed_stderr': stats['stderr'] if multi else None,
        'val_loss_std': stats['std'] if multi else agg.get('val_loss_std'),
        'val_loss_stderr': stats['stderr'] if multi else agg.get('val_loss_stderr'),
        'val_loss_std_basis': 'seeds' if multi else 'val_batches',
        'val_losses': pooled,
        'val_losses_by_seed': by_seed,
        'final_loss': _mean_of('final_loss'),
        'mean_last_100_loss': _mean_of('mean_last_100_loss'),
        'n_loss_spikes': sum(p['n_loss_spikes'] or 0 for p in per_seed
                             if isinstance(p.get('n_loss_spikes'), (int, float))),
        'elapsed_s': sum(p['elapsed_s'] or 0.0 for p in per_seed
                         if isinstance(p.get('elapsed_s'), (int, float))),
        # Curve regimes across the row's seeds: a seed whose run was still
        # underfit/stalled or already overfitting/diverging makes its
        # val_loss a statement about that regime as much as about the norm,
        # so the aggregate keeps the verdicts visible instead of hiding them
        # behind one mean.
        'curve_classification_counts': dict(Counter(
            p['curve_classification'] for p in per_seed if p.get('curve_classification'))),
        'curve_classifications_by_seed': {
            str(p['seed']): p.get('curve_classification') for p in per_seed},
        'curve_last_generalization_gap_by_seed': {
            str(p['seed']): (p.get('curve_diagnosis') or {}).get('generalization_gap')
            for p in per_seed},
    })
    return agg

def _load_summary(path: Path):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None

def _write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)

# The controlled recipe is more than (model, lr, steps): changing the
# optimizer type/betas, grad clip, schedule, data generator, outlier strength
# or the held-out generator/batch count also changes the number being
# compared. Every field below is recorded in each row's `controls` block and
# none of it is allowed to vary between rows in one comparison.
_PROTOCOL_KEYS = (
    'model', 'optimizer_name', 'optimizer_betas', 'optimizer_lr',
    'weight_decay', 'grad_clip', 'schedule_type', 'schedule_lr',
    'data_generator', 'batch_size', 'seq_len', 'outlier_id', 'outlier_scale',
    'val_seed_offset', 'val_batches', 'train_steps',
)
_PROTOCOL_MODEL_VOCAB = {'gpt2-124m': 50304, 'gpt2-350m': 50304, 'llama-1b': 32000}

def _protocol_outlier_id(model):
    vocab = _PROTOCOL_MODEL_VOCAB.get(model)
    return None if vocab is None else vocab - 1

def _protocol_payload(args) -> dict:
    """The controlled training recipe every row of this invocation shares."""
    return {
        'model': args.model,
        'optimizer_name': 'AdamW',
        'optimizer_betas': [0.9, 0.95],
        'optimizer_lr': args.lr,
        'weight_decay': args.weight_decay,
        'grad_clip': 1.0,
        'schedule_type': 'constant',
        'schedule_lr': args.lr,
        'data_generator': 'markov_batch',
        'batch_size': args.batch_size,
        'seq_len': args.seq_len,
        'outlier_id': _protocol_outlier_id(args.model),
        'outlier_scale': 25.0,
        'val_seed_offset': 1,
        'val_batches': max(1, int(getattr(args, 'val_batches', 20))),
        'train_steps': args.max_steps,
    }

def _protocol_signature(args) -> tuple:
    payload = _protocol_payload(args)
    return tuple(payload[key] for key in _PROTOCOL_KEYS)

def _row_protocol_values(row: dict, fallback_model=None):
    """Recorded controlled-recipe dict, or None when the row predates controls.

    Missing individual fields stay `None`; they compare unequal to this
    invocation's values, so an incomplete old record can never be treated as
    matching.
    """
    controls = row.get('controls')
    if not isinstance(controls, dict):
        return None
    optimizer = controls.get('optimizer') or {}
    schedule = controls.get('schedule') or {}
    data = controls.get('data') or {}
    return {
        'model': row.get('model', fallback_model),
        'optimizer_name': optimizer.get('name'),
        'optimizer_betas': optimizer.get('betas'),
        'optimizer_lr': optimizer.get('lr'),
        'weight_decay': optimizer.get('weight_decay'),
        'grad_clip': optimizer.get('grad_clip'),
        'schedule_type': schedule.get('type'),
        'schedule_lr': schedule.get('lr'),
        'data_generator': data.get('generator'),
        'batch_size': data.get('batch_size'),
        'seq_len': data.get('seq_len'),
        'outlier_id': data.get('outlier_id'),
        'outlier_scale': data.get('outlier_scale'),
        'val_seed_offset': data.get('val_seed_offset'),
        'val_batches': data.get('val_batches'),
        'train_steps': controls.get('train_steps'),
    }

def row_protocol_signature(row: dict, fallback_model=None):
    """Canonical recorded controlled-recipe tuple, or None when unknown."""
    values = _row_protocol_values(row, fallback_model)
    if values is None:
        return None
    return tuple(values[key] for key in _PROTOCOL_KEYS)

def partition_rows_by_protocol(rows: list, args) -> tuple[list, list]:
    """(rows recorded under this invocation's recipe, rows recorded under another).

    `all_summaries.json` deliberately accumulates rows across invocations, but
    a row trained with a different model, optimizer, schedule, data generator
    or validation protocol is not the same experiment: comparing it would
    credit the module with a change that came from the recipe. Mismatched rows
    stay in the file and in the report, but they are kept out of the
    statistical comparisons. Missing/partial controls count as unknown, never
    as matching.
    """
    expected_values = _protocol_payload(args)
    expected = _protocol_signature(args)
    keep, stale = [], []
    for row in rows:
        values = _row_protocol_values(row, args.model)
        signature = None if values is None else tuple(values[key] for key in _PROTOCOL_KEYS)
        if signature == expected:
            keep.append(row)
        else:
            mismatched = (list(_PROTOCOL_KEYS) if values is None else
                          [key for key in _PROTOCOL_KEYS
                           if values[key] != expected_values[key]])
            stale.append({
                'name': row.get('name'),
                'reason': ('no recorded controls' if values is None else
                           'trained under a different protocol than this invocation'),
                'mismatched_fields': mismatched,
                'recorded_protocol': values if values is not None else {},
                'expected_protocol': expected_values,
            })
    return keep, stale

def cached_seed_matches(summary: dict, name: str, cfg_kwargs: dict, args, seed: int) -> bool:
    """Whether a cached summary is the same seed of the same row under the same recipe.

    The path does not always encode the seed (a single-seed run writes its
    summary directly under the row directory), so the recorded seed is
    compared too. The held-out validation batch count is part of the recipe:
    a cached row evaluated on a different number of batches is not the same
    selection metric, so it is re-trained (a cached summary without the
    recorded count counts as unknown, never as matching).
    """
    if not (isinstance(summary, dict)
            and summary.get('name') == name
            and summary.get('model') == args.model
            and summary.get('cfg') == cfg_kwargs
            and row_protocol_signature(summary, args.model) == _protocol_signature(args)
            and summary.get('n_val_batches') == max(1, int(getattr(args, 'val_batches', 20)))):
        return False
    per_seed = summary.get('per_seed')
    if isinstance(per_seed, list) and len(per_seed) > 1:
        return False
    recorded = summary.get('seed')
    if recorded is None:
        seeds = summary.get('seeds')
        recorded = seeds[0] if isinstance(seeds, list) and seeds else None
    return recorded is not None and int(recorded) == int(seed)

def merge_summaries(previous: list, fresh: list) -> list:
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in list(previous) + list(fresh):
        if not isinstance(row, dict) or 'name' not in row:
            continue
        if row['name'] not in merged:
            order.append(row['name'])
        merged[row['name']] = row
    return [merged[name] for name in order if name in merged]

def _seed_means(row) -> dict:
    """{seed: val_loss} for every row shape.

    Aggregates carry a `per_seed` list; a single-run summary (no `per_seed`)
    still identifies its seed, so it can pair with the same seed inside an
    aggregate.
    """
    per = row.get('per_seed')
    out = {}
    if isinstance(per, list):
        for item in per:
            if not isinstance(item, dict) or item.get('seed') is None:
                continue
            if _finite_metric(item.get('val_loss')):
                out[int(item['seed'])] = float(item['val_loss'])
        return out
    seeds = row.get('seeds') if isinstance(row.get('seeds'), list) else []
    seeds = [s for s in seeds if s is not None] or ([row['seed']] if row.get('seed') is not None else [])
    # A single-run summary names exactly one seed, so its pooled val_loss is
    # that seed's mean. A multi-seed row without per_seed losses has no
    # per-seed observations: copying the pooled mean to every seed would
    # invent zero variance and can turn it into a "significant" paired win.
    unique_seeds = []
    for s in seeds:
        if int(s) not in unique_seeds:
            unique_seeds.append(int(s))
    if len(unique_seeds) == 1 and _finite_metric(row.get('val_loss')):
        out[unique_seeds[0]] = float(row['val_loss'])
    return out

def _batches_of_seed(row, seed):
    """Held-out batch losses belonging to one run, or None.

    Never returns `val_losses` when the row pooled batches from several
    seeds: those concatenated lists must not be treated as one run's pairing
    basis (batch k of seed 42 would otherwise be zipped with batch k of
    seed 44).
    """
    by_seed = row.get('val_losses_by_seed')
    if isinstance(by_seed, dict) and isinstance(by_seed.get(str(seed)), list):
        return by_seed[str(seed)]
    means = _seed_means(row)
    if len(means) == 1 and seed in means and isinstance(row.get('val_losses'), list):
        return row['val_losses']
    return None

def _row_std(row):
    """Across-seed std when replicated, within-run batch std otherwise."""
    if row.get('n_seeds', 1) >= 2:
        return row.get('val_loss_seed_std')
    return row.get('val_loss_std')

def _paired_row_test(row: dict, reference: dict, margin: float=0.0):
    """One-sided paired Student-t test of `row` vs `reference`, seed-paired.

    Every comparison in this module must run through the *same* test logic,
    or `baseline_comparison` and `leave_one_out_comparison` could disagree on
    the same pair of rows. Seeds are the unit of replication, and only the
    paired per-seed means (`basis='seed_means'`, >= 3 shared seeds) may
    certify significance in the Holm family. With two shared seeds the paired
    test is reported for description only (df = 1), and when exactly one seed
    is shared the same run's held-out batches are tested *for description
    only* (`basis='val_batches_single_seed<s>'`): batch noise is not
    run-to-run noise, so a single run can never establish that one norm
    generalizes better across seeds. With no shared seed there is no test - in
    particular, batches pooled across different seeds are never zipped
    together. The t reference distribution - not the normal one - is the
    small-sample-correct choice for the 3-5 seeds these matrices use, and the
    returned interval makes the precision of the delta visible. Returns
    `(test, basis, common_seeds)`, with `test=None` when no basis exists.
    """
    row_seeds = _seed_means(row)
    ref_seeds = _seed_means(reference)
    common = sorted(set(row_seeds) & set(ref_seeds))
    if len(common) >= 2:
        test = paired_t_one_sided([row_seeds[s] for s in common],
                                  [ref_seeds[s] for s in common], margin=margin)
        return test, 'seed_means', common
    if len(common) == 1:
        seed = common[0]
        row_batches = _batches_of_seed(row, seed)
        ref_batches = _batches_of_seed(reference, seed)
        if (isinstance(row_batches, list) and isinstance(ref_batches, list)
                and len(row_batches) >= 2 and len(ref_batches) >= 2):
            if len(row_batches) == len(ref_batches):
                test = paired_t_one_sided(row_batches, ref_batches, margin=margin)
                return test, f'val_batches_single_seed{seed}', common
            test = welch_t_one_sided(row_batches, ref_batches, margin=margin)
            return test, f'val_batches_single_seed{seed}_unpaired', common
    basis = 'no_shared_seeds' if (row_seeds and ref_seeds) else 'insufficient_seeds'
    return None, basis, common

def _basis_counted_in_family(basis, n_shared: int=0) -> bool:
    """Only a >= 3-seed paired test is admissible to the Holm family."""
    return basis == 'seed_means' and n_shared >= MIN_SEEDS_FOR_SIGNIFICANCE

def _run_identity(row: dict, parent_of: dict) -> str:
    """Name of the run a row ultimately reuses, following `reused_from` chains.

    A row can be reused more than once (C reuses B, which reuses A), so a
    single hop would misread C and B as two independent runs; resolving to the
    root keeps "aliases are counted once" true for chains too.
    """
    name = row.get('name')
    seen = set()
    while isinstance(name, str) and name not in seen:
        seen.add(name)
        parent = parent_of.get(name)
        if not isinstance(parent, str):
            return name
        name = parent
    return row.get('name')

def baseline_comparison(summaries: list, baseline_names, margin: float=0.0, alpha: float=0.05) -> dict | None:
    """Every row's held-out val_loss against the best reference baseline.

    The reference is the best (lowest val_loss) *replicated* row among
    `baseline_names` when one exists: a lucky single-seed baseline must not
    become the bar, because no comparison against it could ever be certified
    and its point estimate is the noisiest. The point-estimate best among all
    baselines is still recorded (`point_estimate_best_baseline`), so the
    selection cannot hide a lower unreplicated reference. Significance uses
    the one-sided paired
    Student-t test on the per-seed means whenever both rows share >= 2 seeds,
    but only >= 3 shared seeds can certify significance (2 seeds have df = 1).
    With exactly one shared seed the same run's held-out batches are tested
    for description only - batch noise is not run-to-run noise - and with no
    shared seed no test runs at all.
    Alias rows (`reused_from`) are the same underlying run and are counted
    once, both in the report and in the Holm family. The t reference distribution matters
    at this replication count: at n = 3, df = 2, the one-sided 5% critical
    value is 2.92, not the normal 1.64. `margin` is the smallest loss
    improvement treated as meaningful, and every entry carries a two-sided
    95% interval for the delta (`delta_ci_95`) so a win smaller than its own
    precision is visible as such. Because every row is tested against the
    *same* reference, the per-row p-values are a family of tests and are
    Holm-Bonferroni adjusted over the matrix before anything is called
    significant: `beats_reference_significantly` requires the adjusted
    p < alpha, and the raw one-sided p stays available as `p_one_sided`.
    Returns None when no baseline row has a usable val_loss.
    """
    baseline_names = set(baseline_names)
    usable = [r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))]
    baselines = [r for r in usable if r['name'] in baseline_names]
    if not baselines:
        return None
    parent_of = {r['name']: r['reused_from'] for r in summaries
                 if isinstance(r, dict) and 'name' in r and isinstance(r.get('reused_from'), str)}
    replicated_baselines = [r for r in baselines
                            if len(_seed_means(r)) >= MIN_SEEDS_FOR_SIGNIFICANCE]
    selection_pool = replicated_baselines or baselines
    reference = min(selection_pool, key=lambda r: r['val_loss'])
    point_best = min(baselines, key=lambda r: r['val_loss'])
    ref_seeds = _seed_means(reference)
    ref_stderr = reference.get('val_loss_seed_stderr')
    if ref_stderr is None:
        ref_stderr = reference.get('val_loss_stderr')
    out = {
        'reference': reference['name'],
        'reference_val_loss': reference['val_loss'],
        'reference_val_loss_std': _row_std(reference),
        'reference_val_loss_stderr': ref_stderr,
        'reference_n_seeds': reference.get('n_seeds', 1),
        'reference_seeds': sorted(ref_seeds),
        'reference_selection': {
            'rule': ('lowest val_loss among baselines with >= '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} seeds'
                     if replicated_baselines else
                     'lowest val_loss (no baseline reaches '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} seeds; certification unavailable)'),
            'replicated_baselines': sorted(r['name'] for r in replicated_baselines),
            'unreplicated_baselines': sorted(r['name'] for r in baselines
                                             if r not in replicated_baselines),
            'reference_can_certify': len(ref_seeds) >= MIN_SEEDS_FOR_SIGNIFICANCE,
            'reference_is_point_estimate_best': reference['name'] == point_best['name'],
        },
        'point_estimate_best_baseline': {
            'name': point_best['name'],
            'val_loss': point_best['val_loss'],
            'n_seeds': point_best.get('n_seeds', 1),
            'seeds': sorted(_seed_means(point_best)),
        },
        'baseline_rows': {r['name']: r['val_loss'] for r in baselines},
        'margin': margin,
        'alpha': alpha,
        'protocol': ('seeds are the unit of replication: mean +/- std across >= 3 seeds; '
                     'only the paired test on >= 3 shared per-seed means certifies significance; '
                     'two or one shared seed(s) are descriptive only, and alias rows '
                     '(reused summaries) are counted once'),
        'rows': [],
    }
    for row in usable:
        if row['name'] in baseline_names:
            continue
        delta = row['val_loss'] - reference['val_loss']
        entry = {
            'name': row['name'],
            'hypothesis': row.get('hypothesis'),
            'run_identity': _run_identity(row, parent_of),
            'val_loss': row['val_loss'],
            'val_loss_std': _row_std(row),
            'n_seeds': row.get('n_seeds', 1),
            'delta_vs_reference': delta,
            'beats_reference': bool(delta < -margin),
            'beats_reference_significantly': False,
            'test': None,
            'test_basis': None,
            'paired_seeds': [],
            'replicated_across_seeds': False,
            'significance_basis': None,
            'excluded_from_family': None,
            'p_one_sided': None,
            'p_one_sided_holm': None,
            't': None,
            'df': None,
            'delta_ci_95': None,
            'n_seeds_favoring': None,
            'n_seeds_tied': None,
            'n_seeds_against': None,
            'per_seed_delta': None,
            'paired_delta_std': None,
            'minimum_detectable_delta_95_power80': None,
            # Curve regimes of the row's seeds (None for pre-monitoring
            # summaries): a delta measured while one arm is underfit or
            # overfitting is annotated, so it cannot be read as pure norm
            # effect. It does not alter the significance test - the regime is
            # itself a property of the norm at the fixed step count.
            'curve_classification_counts': row.get('curve_classification_counts'),
        }
        test, basis, common = _paired_row_test(row, reference, margin=margin)
        entry['test_basis'] = basis
        entry['paired_seeds'] = common
        entry['replicated_across_seeds'] = _basis_counted_in_family(basis, len(common))
        if test is not None:
            entry['test'] = test['test']
            entry['p_one_sided'] = test['p_one_sided']
            entry['t'] = test['t']
            entry['df'] = test['df']
            entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
            entry['n'] = test['n']
        entry.update(_paired_effect_summary(row, reference, common, test, alpha=alpha))
        if entry['test'] is not None and not entry['replicated_across_seeds']:
            entry['significance_basis'] = (
                f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the test cannot certify the norm effect; '
                'not counted in the Holm family')
        out['rows'].append(entry)
    # Alias rows (identical effective config, `reused_from`) are the *same*
    # run under a second name: they are reported, but a run must not be
    # counted twice as independent evidence nor inflate the Holm family size.
    by_identity: dict[str, str] = {
        _run_identity(row, parent_of): row['name'] for row in baselines}
    for entry in out['rows']:
        identity = entry.get('run_identity') or entry['name']
        canonical = by_identity.get(identity)
        if canonical is None:
            by_identity[identity] = entry['name']
        else:
            entry['excluded_from_family'] = f'duplicate run of {canonical} (reused summary, not independent evidence)'
    # Every seed-replicated row above tests the same reference, so its raw
    # p-values form a family and the per-row alpha must be corrected
    # (Holm-Bonferroni step-down) before a row is reported as a significant
    # win. Rows tested on a single run, duplicates of another row, or rows
    # whose test could not run stay out of the family (and cannot be reported
    # significant).
    family = [entry for entry in out['rows']
              if entry['replicated_across_seeds'] and entry['excluded_from_family'] is None]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['beats_reference_significantly'] = bool(
            entry['beats_reference'] and p_holm is not None and p_holm < alpha
        )
    described_only = [entry['name'] for entry in out['rows']
                      if entry['test'] is not None and not entry['replicated_across_seeds']]
    out['rows_tested_descriptively_only'] = described_only
    out['multiple_comparison'] = (
        f'Holm-Bonferroni over the {sum(p is not None for p in adjusted)} seed-replicated row(s) '
        f'against the same reference; `beats_reference_significantly` uses the adjusted p-value. '
        f'Single-run batch-level tests are descriptive only ('
        f'{described_only}) and duplicates/aliases are counted once.'
    )
    thin = {r['name'] for r in [reference, *baselines] if r.get('n_seeds', 1) < 3}
    out['rows_with_fewer_than_3_seeds'] = sorted(thin | {e['name'] for e in out['rows'] if e['n_seeds'] < 3})
    return out

def print_baseline_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[compare] no baseline row present; comparison skipped (include a baseline row to interpret the matrix)')
        return
    ref_std = comparison.get('reference_val_loss_std')
    ref_std_str = '' if ref_std is None else f" +/- {ref_std:.4f}"
    print(f"[compare] reference baseline: {comparison['reference']} "
          f"val_loss={comparison['reference_val_loss']:.4f}{ref_std_str} "
          f"(seeds={comparison.get('reference_seeds')}; "
          f"all baselines: { {k: round(v, 4) for k, v in comparison['baseline_rows'].items()} })")
    selection = comparison.get('reference_selection') or {}
    if selection:
        print(f"[compare] reference selection: {selection.get('rule')}; "
              f"point-estimate best={comparison.get('point_estimate_best_baseline')}; "
              f"reference can certify={selection.get('reference_can_certify')}")
    for entry in comparison['rows']:
        p = entry['p_one_sided']
        p_holm = entry.get('p_one_sided_holm')
        p_str = 'n/a' if p is None else f'{p:.3g}'
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        std = entry.get('val_loss_std')
        std_str = '' if std is None else f" +/- {std:.4f}"
        basis = entry.get('test_basis') or '-'
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f" ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]"
        note = ''
        if entry.get('excluded_from_family'):
            note = f" [{entry['excluded_from_family']}]"
        elif entry.get('significance_basis'):
            note = ' [descriptive only, not certified]'
        direction = ''
        if entry.get('n_seeds_favoring') is not None:
            direction = (f" seeds_favoring={entry['n_seeds_favoring']}/{len(entry.get('paired_seeds') or [])}")
            if entry.get('n_seeds_tied'):
                direction += f" tied={entry['n_seeds_tied']}"
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f" mde80={mde:.4f}"
        regimes = entry.get('curve_classification_counts')
        regime_note = ''
        if regimes:
            off_regime = {k: v for k, v in regimes.items()
                          if k not in ('healthy', 'insufficient_data')}
            if off_regime:
                regime_note = (f" [curve regimes {off_regime}: the delta was not measured "
                               f"in a healthy falling-falling regime - read the curve block]")
        print(f"[compare] {entry['name']:>32s} val_loss={entry['val_loss']:.4f}{std_str} "
              f"delta={entry['delta_vs_reference']:+.4f}{ci_str} p(one-sided)={p_str} p(holm)={p_holm_str} "
              f"test={entry.get('test') or '-'}@{basis}{direction}{mde_str} "
              f"beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}{note}{regime_note}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[compare] NOTE: tests with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are "
              f"described but never certified: {comparison['rows_tested_descriptively_only']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[compare] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def load_simple_baselines(path) -> dict | None:
    """Load the Bayes-limit calibration written by scripts/baseline_simple.py.

    Returns None - with a warning for a malformed file - when the path is
    absent or does not look like a calibration report, so a stale side file
    can never abort the matrix; it only removes the floor annotation.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[compare] WARNING: ignoring unreadable simple-baseline file {path}: {exc}")
        return None
    limits = data.get('bayes_limits') if isinstance(data, dict) else None
    tuned = data.get('tuned_count_baseline') if isinstance(data, dict) else None
    if not isinstance(limits, dict) or not isinstance(limits.get('memoryless_nats'), (int, float)):
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: missing bayes_limits.memoryless_nats")
        return None
    if not isinstance(tuned, dict) or not isinstance(tuned.get('mean_ce_nats'), (int, float)):
        print(f"[compare] WARNING: ignoring simple-baseline file {path}: missing tuned_count_baseline.mean_ce_nats")
        return None
    return data

def attach_simple_baseline_calibration(comparison: dict, floor_data: dict, source) -> dict:
    """Annotate every comparison row with its distance to the task's Bayes limits.

    `baseline_comparison` answers "which row is best among these
    transformers"; this annotation answers "how much of the task's learnable
    signal did the row capture". Both are needed: a row can win the matrix
    while still sitting far above the ceiling, which means the comparison
    ranked under-trained models. The ceiling is the exact HMM-filter CE when
    the corpus carries the outlier channel (the transformer reads the full
    observed prefix), reported next to the closed-form memoryless walk floor.
    `excess_over_bayes_ceiling_nats` = val_loss - ceiling (0 = Bayes optimal),
    `excess_over_tuned_simple_nats` = distance to the tuned count model, and
    `signal_captured_fraction` rescales the remaining gap against the
    uniform-over-vocabulary strawman.
    """
    limits = floor_data['bayes_limits']
    floor = float(limits['memoryless_nats'])
    ceiling = limits.get('full_context_nats')
    ceiling = float(ceiling) if isinstance(ceiling, (int, float)) else None
    tuned = floor_data['tuned_count_baseline']
    tuned_ce = float(tuned['mean_ce_nats'])
    protocol = floor_data.get('protocol') or {}
    vocab_size = protocol.get('vocab_size')
    log_vocab = math.log(vocab_size) if isinstance(vocab_size, (int, float)) and vocab_size > 0 else None
    for entry in comparison['rows']:
        val_loss = entry.get('val_loss')
        if not isinstance(val_loss, (int, float)):
            continue
        entry['excess_over_bayes_floor_nats'] = val_loss - floor
        if ceiling is not None:
            entry['excess_over_bayes_ceiling_nats'] = val_loss - ceiling
        entry['excess_over_tuned_simple_nats'] = val_loss - tuned_ce
        entry['beats_tuned_simple_baseline'] = bool(val_loss < tuned_ce)
        if log_vocab is not None and ceiling is not None and log_vocab > ceiling:
            entry['signal_captured_fraction'] = (log_vocab - val_loss) / (log_vocab - ceiling)
    annotated = [entry for entry in comparison['rows'] if isinstance(entry.get('val_loss'), (int, float))]
    block = {
        'source': str(source),
        'memoryless_floor_nats': floor,
        'full_context_ceiling_nats': ceiling,
        'tuned_simple_predictor': tuned.get('predictor'),
        'tuned_simple_ce_nats': tuned_ce,
        'tuned_simple_std_ce_nats': tuned.get('std_ce_nats'),
        'uniform_vocab_strawman_nats': log_vocab,
        'calibration_seeds': protocol.get('seeds'),
        'hypothesis': floor_data.get('hypothesis'),
        'reading': ('excess_over_bayes_ceiling_nats is the gap to the exact full-context Bayes predictor '
                    '(hmm_bayes) and the number that says whether a row learned the task; '
                    'beats_tuned_simple_baseline only says it beat a tuned count model, and '
                    'signal_captured_fraction rescales the gap against uniform-over-vocab'),
        'rows': [{key: entry.get(key) for key in (
            'name', 'val_loss', 'excess_over_bayes_ceiling_nats', 'excess_over_bayes_floor_nats',
            'excess_over_tuned_simple_nats', 'beats_tuned_simple_baseline', 'signal_captured_fraction',
        )} for entry in annotated],
    }
    if annotated:
        best = min(annotated, key=lambda entry: entry.get('excess_over_bayes_ceiling_nats', float('inf')))
        ceiling_str = 'n/a' if ceiling is None else f"{ceiling:.4f}"
        best_gap = best.get('excess_over_bayes_ceiling_nats')
        best_str = 'n/a' if best_gap is None else f"{best_gap:+.4f}"
        print(f"[compare] Bayes calibration ({source}): memoryless floor {floor:.4f} nats, full-context "
              f"ceiling {ceiling_str} nats, tuned simple baseline {tuned_ce:.4f} nats "
              f"({tuned.get('predictor')}); closest row to the ceiling: {best['name']} "
              f"{best_str} nats; rows beating the tuned simple baseline: "
              f"{sum(1 for entry in annotated if entry.get('beats_tuned_simple_baseline'))}/{len(annotated)}")
    return block

def leave_one_out_comparison(summaries: list, spec: dict | None=None, margin: float=0.0, alpha: float=0.05) -> dict | None:
    """Full vs each Full-minus-X row: does removing mechanism X hurt?

    `check_leave_one_out` proves the *structure* of the leave-one-out matrix -
    the reference row differs from Full in exactly the declared knobs. This
    function adds the statistical half: for every declared mechanism it runs
    the same one-sided paired test as `baseline_comparison`, of H1 "Full has
    lower held-out val_loss than the X-free row". Because all mechanisms are
    tested against the same Full row, the per-mechanism p-values are
    Holm-Bonferroni adjusted before a mechanism is called significantly
    helpful; a point-estimate win inside the seed noise remains visible as
    `mechanism_helps=True, mechanism_helps_significantly=False`. This is what
    backs the claim that each axis of the main proposal is load-bearing on
    its own, not merely correlated with the others. Returns None when no
    declared contrast has both rows with a usable val_loss.
    """
    spec = LOO_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    parent_of = {r['name']: r['reused_from'] for r in summaries
                 if isinstance(r, dict) and 'name' in r and isinstance(r.get('reused_from'), str)}
    rows: list = []
    for full_name, removals in spec.items():
        full = usable.get(full_name)
        for mechanism, info in removals.items():
            ref_name = info['row']
            ref = usable.get(ref_name)
            entry = {
                'full': full_name,
                'mechanism': mechanism,
                'reference_row': ref_name,
                'declared_diff': sorted(info['diff']),
                'full_val_loss': None if full is None else full['val_loss'],
                'full_val_loss_std': None if full is None else _row_std(full),
                'reference_val_loss': None if ref is None else ref['val_loss'],
                'reference_val_loss_std': None if ref is None else _row_std(ref),
                'n_seeds_full': None if full is None else full.get('n_seeds', 1),
                'n_seeds_reference': None if ref is None else ref.get('n_seeds', 1),
                'full_curve_classification_counts': None if full is None else full.get('curve_classification_counts'),
                'reference_curve_classification_counts': None if ref is None else ref.get('curve_classification_counts'),
                'run_identity_full': None if full is None else _run_identity(full, parent_of),
                'run_identity_reference': None if ref is None else _run_identity(ref, parent_of),
                'delta_full_minus_reference': None,
                'mechanism_helps': False,
                'mechanism_helps_significantly': False,
                'test': None,
                'test_basis': None,
                'paired_seeds': [],
                'replicated_across_seeds': False,
                'significance_basis': None,
                'excluded_from_family': None,
                'p_one_sided': None,
                'p_one_sided_holm': None,
                't': None,
                'df': None,
                'delta_ci_95': None,
                'n_seeds_favoring': None,
                'n_seeds_tied': None,
                'n_seeds_against': None,
                'per_seed_delta': None,
                'paired_delta_std': None,
                'minimum_detectable_delta_95_power80': None,
            }
            if full is None:
                entry['skipped'] = f"full row '{full_name}' has no usable val_loss"
                rows.append(entry)
                continue
            if ref is None:
                entry['skipped'] = f"Full-minus-X row '{ref_name}' has no usable val_loss"
                rows.append(entry)
                continue
            if entry['run_identity_full'] == entry['run_identity_reference']:
                entry['skipped'] = (f"'{full_name}' and '{ref_name}' are the same run "
                                    f"(reused summary); the contrast is vacuous")
                rows.append(entry)
                continue
            entry['delta_full_minus_reference'] = full['val_loss'] - ref['val_loss']
            entry['mechanism_helps'] = bool(entry['delta_full_minus_reference'] < -margin)
            test, basis, common = _paired_row_test(full, ref, margin=margin)
            entry['test_basis'] = basis
            entry['paired_seeds'] = common
            entry['replicated_across_seeds'] = _basis_counted_in_family(basis, len(common))
            if test is not None:
                entry['test'] = test['test']
                entry['p_one_sided'] = test['p_one_sided']
                entry['t'] = test['t']
                entry['df'] = test['df']
                entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
                entry['n'] = test['n']
            entry.update(_paired_effect_summary(full, ref, common, test, alpha=alpha))
            if entry['test'] is not None and not entry['replicated_across_seeds']:
                entry['significance_basis'] = (
                    f'descriptive only: {len(common)} shared seed(s), fewer than the required '
                    f'{MIN_SEEDS_FOR_SIGNIFICANCE}, so the test cannot certify the mechanism '
                    'effect; not counted in the Holm family')
            rows.append(entry)
    if not rows:
        return None
    # A contrast is identified by the underlying runs, not the row names: an
    # alias pair would otherwise appear as two independent mechanism tests,
    # and a run could even be contrasted with itself via aliases. Each
    # underlying contrast enters the Holm family exactly once.
    seen_contrasts: dict[tuple, str] = {}
    for entry in rows:
        if entry.get('skipped') or entry.get('run_identity_full') is None:
            continue
        key = (entry['run_identity_full'], entry['run_identity_reference'])
        canonical = seen_contrasts.get(key)
        if canonical is None:
            seen_contrasts[key] = entry['mechanism']
        else:
            entry['excluded_from_family'] = (f"duplicate of mechanism '{canonical}' "
                                             f"(same underlying runs, not independent evidence)")
    family = [entry for entry in rows
              if entry['replicated_across_seeds'] and entry['excluded_from_family'] is None]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['mechanism_helps_significantly'] = bool(
            entry['mechanism_helps'] and p_holm is not None and p_holm < alpha
        )
    thin = sorted({name for entry in rows for name in (entry['full'], entry['reference_row'])
                   if 1 <= usable.get(name, {}).get('n_seeds', 1) < 3})
    tested_descriptively = [entry['mechanism'] for entry in rows
                            if entry['test'] is not None and not entry['replicated_across_seeds']]
    return {
        'protocol': ('Full vs Full-minus-X: one-sided paired test of H1 "Full has lower '
                     'held-out val_loss than the X-free row"; only the paired test on >= 3 '
                     f'shared per-seed means certifies significance (two or one shared seed(s) '
                     'are descriptive only), and each underlying contrast is counted once even '
                     'when rows are aliases'),
        'margin': margin,
        'alpha': alpha,
        'multiple_comparison': (f'Holm-Bonferroni over the '
                                f'{sum(e["p_one_sided"] is not None for e in family)} seed-replicated '
                                f'mechanism(s); `mechanism_helps_significantly` uses the adjusted '
                                f'p-value'),
        'rows': rows,
        'rows_tested_descriptively_only': tested_descriptively,
        'rows_with_fewer_than_3_seeds': thin,
    }

def print_leave_one_out_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[loo] no leave-one-out contrast available; nothing to test')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[loo] {entry['mechanism']}: SKIPPED ({entry['skipped']})")
            continue
        full_std = entry.get('full_val_loss_std')
        ref_std = entry.get('reference_val_loss_std')
        full_std_str = '' if full_std is None else f' +/- {full_std:.4f}'
        ref_std_str = '' if ref_std is None else f' +/- {ref_std:.4f}'
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm_str = 'n/a' if entry.get('p_one_sided_holm') is None else f"{entry['p_one_sided_holm']:.3g}"
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        note = ''
        if entry.get('excluded_from_family'):
            note = f" [{entry['excluded_from_family']}]"
        elif entry.get('significance_basis'):
            note = ' [descriptive only, not certified]'
        direction = ''
        if entry.get('n_seeds_favoring') is not None:
            direction = (f" seeds_favoring={entry['n_seeds_favoring']}/{len(entry.get('paired_seeds') or [])}")
            if entry.get('n_seeds_tied'):
                direction += f" tied={entry['n_seeds_tied']}"
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f" mde80={mde:.4f}"
        regime_note = ''
        for label, key in (('full', 'full_curve_classification_counts'),
                           ('ref', 'reference_curve_classification_counts')):
            regimes = entry.get(key)
            off_regime = {k: v for k, v in (regimes or {}).items()
                          if k not in ('healthy', 'insufficient_data')}
            if off_regime:
                regime_note += f" [{label} curve regimes {off_regime}]"
        print(f"[loo] {entry['mechanism']:>22s}: {entry['full']} "
              f"val_loss={entry['full_val_loss']:.4f}{full_std_str} vs "
              f"{entry['reference_row']} val_loss={entry['reference_val_loss']:.4f}{ref_std_str} "
              f"delta(full-ref)={entry['delta_full_minus_reference']:+.4f} {ci_str}"
              f"p(one-sided)={p_str} p(holm)={p_holm_str} "
              f"test={entry.get('test') or '-'}@{entry.get('test_basis') or '-'}{direction}{mde_str} "
              f"removing_hurts={entry['mechanism_helps']} "
              f"significant={entry['mechanism_helps_significantly']}{note}{regime_note}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[loo] NOTE: tests with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are described "
              f"but never certified: {comparison['rows_tested_descriptively_only']}")
    if comparison.get('rows_with_fewer_than_3_seeds'):
        print(f"[loo] WARNING: rows with <3 seeds cannot separate effect from noise on their own: "
              f"{comparison['rows_with_fewer_than_3_seeds']}")

def factorial_comparison(summaries: list, spec: dict | None=None, margin: float=0.0,
                         alpha: float=0.05) -> dict | None:
    """Paired main effects and interaction for each declared 2x2 factorial.

    The interaction is the difference of the factor's effect in the two arms
    of the other factor, computed per seed:
    ``(both - depth) - (per_channel - neither)``. A negative interaction means
    the two mechanisms together lower loss by more than the sum of their
    separate effects (synergy); a positive one means they overlap or interfere.
    Testing the interaction is what lets the report distinguish "Full wins
    because either axis helps" from "the combination is special". Every term
    uses the same shared seed set across all four cells and the same
    small-sample paired t-test as the other comparisons; only >= 3 shared
    seeds enter the Holm family. Returns None when no design has all cells.
    """
    spec = FACTORIAL_SPEC if spec is None else spec
    usable = {r['name']: r for r in summaries
              if isinstance(r, dict) and 'name' in r and _finite_metric(r.get('val_loss'))}
    rows: list = []
    for design_name, design in spec.items():
        cells = design['cells']
        factor_names = list(design['factors'])
        missing = [row for row in cells.values() if row not in usable]
        base_entry = {
            'design': design_name,
            'cells': dict(cells),
            'context': design.get('context'),
            'shared_seeds': [],
        }
        if missing:
            entry = dict(base_entry, effect=None, kind='design', skipped=(
                f"cell row(s) {sorted(missing)} have no usable val_loss"))
            rows.append(entry)
            continue
        seed_values = {cell: _seed_means(usable[row]) for cell, row in cells.items()}
        common = sorted(set.intersection(*(set(values) for values in seed_values.values())))
        if len(common) < 2:
            entry = dict(base_entry, effect=None, kind='design', shared_seeds=common,
                         skipped=(f"only {len(common)} seed(s) are shared by all four "
                                  f"cells; a paired effect needs at least 2"))
            rows.append(entry)
            continue
        effects = {}
        for factor in factor_names:
            arm_values = {}
            for off_cell, on_cell in design['factors'][factor]['pairs']:
                arm_values[f'{off_cell}->{on_cell}'] = [
                    seed_values[on_cell][seed] - seed_values[off_cell][seed]
                    for seed in common
                ]
            arm_means = [sum(values) / len(values) for values in arm_values.values()]
            main_values = [sum(vals) / len(vals) for vals in zip(*arm_values.values())]
            effects[factor] = {
                'kind': 'main_effect',
                'formula': 'mean over both arms of the other factor: on - off',
                'arm_effects': {name: {'mean_delta': sum(vals) / len(vals),
                                       'per_seed_delta': dict(zip(map(str, common), vals))}
                                for name, vals in arm_values.items()},
                'per_seed_values': dict(zip(map(str, common), main_values)),
                'mean_arm_deltas': arm_means,
                'delta_seed_mean': sum(main_values) / len(main_values),
                'per_seed_delta': dict(zip(map(str, common), main_values)),
            }
        if len(factor_names) == 2:
            first, second = factor_names
            first_arms = list(effects[first]['arm_effects'].items())
            interaction_values = [
                on - off for off, on in zip(
                    first_arms[0][1]['per_seed_delta'].values(),
                    first_arms[1][1]['per_seed_delta'].values())
            ]
            effects[f'{first}_x_{second}'] = {
                'kind': 'interaction',
                'formula': (f'effect of {first} with {second} on minus effect of {first} '
                            f'with {second} off; negative = synergy'),
                'per_seed_values': dict(zip(map(str, common), interaction_values)),
                'delta_seed_mean': sum(interaction_values) / len(interaction_values),
                'per_seed_delta': dict(zip(map(str, common), interaction_values)),
            }
        for effect_name, effect in effects.items():
            values = list(effect['per_seed_values'].values())
            test = paired_t_one_sided(values, [0.0] * len(values), margin=margin, alpha=alpha)
            replicated = len(common) >= MIN_SEEDS_FOR_SIGNIFICANCE
            entry = dict(base_entry, effect=effect_name, kind=effect['kind'],
                         formula=effect['formula'], shared_seeds=common,
                         n_seeds=len(common),
                         delta_seed_mean=effect['delta_seed_mean'],
                         per_seed_delta=effect['per_seed_delta'],
                         n_seeds_favoring=sum(value < 0.0 for value in values),
                         n_seeds_tied=sum(value == 0.0 for value in values),
                         n_seeds_against=sum(value > 0.0 for value in values),
                         effect_helps=bool(effect['delta_seed_mean'] < -margin),
                         effect_helps_significantly=False,
                         test=test['test'],
                         test_basis=('seed_means' if replicated
                                     else f'seed_means_{len(common)}_descriptive'),
                         replicated_across_seeds=replicated,
                         p_one_sided=test['p_one_sided'],
                         p_one_sided_holm=None,
                         t=test['t'],
                         df=test['df'],
                         delta_ci_95=[test['ci_low'], test['ci_high']],
                         n=test['n'])
            effect_row = {'per_seed': [{'seed': int(seed), 'val_loss': value}
                                       for seed, value in effect['per_seed_values'].items()]}
            zero_row = {'per_seed': [{'seed': int(seed), 'val_loss': 0.0}
                                     for seed in effect['per_seed_values']]}
            entry.update(_paired_effect_summary(effect_row, zero_row, common, test, alpha=alpha))
            if 'arm_effects' in effect:
                entry['arm_effects'] = effect['arm_effects']
            rows.append(entry)
    if not rows:
        return None
    family = [entry for entry in rows
              if entry.get('replicated_across_seeds') and not entry.get('skipped')]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['effect_helps_significantly'] = bool(
            entry['effect_helps'] and p_holm is not None and p_holm < alpha)
    tested_descriptively = [entry['effect'] for entry in rows
                            if entry.get('p_one_sided') is not None
                            and not entry.get('replicated_across_seeds')]
    return {
        'protocol': ('2x2 factorial per_seed paired terms on the seeds shared by all '
                     'four cells; only tests with >= '
                     f'{MIN_SEEDS_FOR_SIGNIFICANCE} shared seeds are certified, and all '
                     'main effects and interactions form one Holm family'),
        'margin': margin,
        'alpha': alpha,
        'multiple_comparison': (f'Holm-Bonferroni over {len(family)} seed-replicated '
                                'factorial term(s); `effect_helps_significantly` uses the '
                                'adjusted p-value'),
        'designs': spec,
        'rows': rows,
        'rows_tested_descriptively_only': tested_descriptively,
    }

def print_factorial_comparison(comparison: dict | None) -> None:
    if comparison is None:
        print('[factorial] no 2x2 factorial design declared; interaction test skipped')
        return
    for entry in comparison['rows']:
        if entry.get('skipped'):
            print(f"[factorial] {entry['design']}: SKIPPED ({entry['skipped']})")
            continue
        p_str = 'n/a' if entry['p_one_sided'] is None else f"{entry['p_one_sided']:.3g}"
        p_holm_str = ('n/a' if entry['p_one_sided_holm'] is None
                      else f"{entry['p_one_sided_holm']:.3g}")
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f"ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}] "
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f"mde80={mde:.4f} "
        note = '' if entry['replicated_across_seeds'] else ' [descriptive only, not certified]'
        print(f"[factorial] {entry['design']} {entry['kind']} {entry['effect']}: "
              f"delta={entry['delta_seed_mean']:+.4f} {ci_str}"
              f"seeds_favoring={entry['n_seeds_favoring']}/{entry['n_seeds']} "
              f"p(one-sided)={p_str} p(holm)={p_holm_str} {mde_str}"
              f"helps={entry['effect_helps']} "
              f"significant={entry['effect_helps_significantly']}{note}")
    if comparison.get('rows_tested_descriptively_only'):
        print(f"[factorial] NOTE: terms with fewer than {MIN_SEEDS_FOR_SIGNIFICANCE} shared "
              f"seeds are described but never certified: "
              f"{comparison['rows_tested_descriptively_only']}")

def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default=None)
    known, _ = pre.parse_known_args()
    ablations = dict(ABLATIONS)
    hypotheses = dict(ABLATION_HYPOTHESES)
    if known.config:
        try:
            from_yaml = load_ablations_from_config(known.config, include_metadata=True)
        except ValueError as exc:
            print(f'[matrix] FAIL: {exc}')
            sys.exit(1)
        if from_yaml:
            ablations = {name: row['cfg'] for name, row in from_yaml.items()}
            hypotheses = {name: row['hypothesis'] for name, row in from_yaml.items()}
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=None, help='YAML run config; CLI flags override it')
    p.add_argument('--model', choices=sorted(_MODEL_BUILDERS), default='gpt2-124m')
    p.add_argument('--names', nargs='+', default=list(ablations.keys()))
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--lr', type=float, default=0.0006)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS), help='one row is trained once per seed and reported as mean +/- std across seeds (>= 3 recommended)')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--stats-every', type=int, default=200)
    p.add_argument('--val-batches', type=int, default=20, help='held-out synthetic batches for the selection metric (val_loss); never used for gradient updates')
    p.add_argument('--curve-every', type=int, default=0, help='held-out curve-monitoring period in steps (0 = auto: max_steps//5; negative = disabled). Curve points come from a separate generator (seed+2) and never touch the selection batches, so a row reports its train/val regime, not only its final number')
    p.add_argument('--curve-batches', type=int, default=5, help='held-out batches per curve-monitoring point')
    p.add_argument('--baselines', nargs='+', default=['ln_baseline', 'rmsnorm_baseline'], help='rows treated as the simple reference baselines in the baseline-relative comparison')
    p.add_argument('--simple-baselines', default=None, help='Bayes-limit calibration JSON from scripts/baseline_simple.py; when omitted, auto-detected at <output-dir>/simple_baselines.json. Adds the tuned simple baseline and the memoryless-Bayes-floor / full-context-ceiling gaps to baseline_comparison.json')
    p.add_argument('--baseline-margin', type=float, default=0.0, help='minimum val_loss improvement over the best baseline to count as a win')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the one-sided paired t-test vs the best baseline (Holm-adjusted over rows)')
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='run the small-batch overfit sanity check for every norm in the selected rows before training them (default: on; a row that cannot overfit one batch has a gradient/loss bug and must not be launched)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the pre-flight overfit sanity check (use only when the check itself is the problem, e.g. an offline host)')
    p.add_argument('--check-matrix', action='store_true', help='validate controlled knobs, stated hypotheses and leave-one-out coverage of the matrix, then exit without training')
    p.add_argument('--output-dir', default='runs/ablation')
    p.add_argument('--force', action='store_true')
    apply_config_to_parser(p, known.config)
    args = p.parse_args()
    raw_seeds = [args.seed] if args.seed is not None else list(args.seeds or DEFAULT_SEEDS)
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed is the same run, not an independent replication)')
    if len(seeds) < 3:
        print(f'[warn] {len(seeds)} seed(s) selected: with <3 independent runs a difference cannot be separated from run-to-run noise; report error bars and treat significance as provisional')
    try:
        matrix_report = validate_matrix(ablations, hypotheses, args.model)
    except ValueError as exc:
        print(f'[matrix] FAIL: {exc}')
        sys.exit(1)
    print(f"[matrix] ok: {len(matrix_report['rows'])} rows, controlled optimizer/schedule/data, "
          f"every row has a hypothesis, leave-one-out coverage {matrix_report['leave_one_out']}")
    if args.check_matrix:
        print(json.dumps(matrix_report, indent=2))
        return
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_metadata = collect_run_metadata(seed=seeds[0], extra={'seeds': seeds})
    if args.sanity_check:
        from scripts.sanity_overfit import run_preflight
        from scripts.train_gpt2 import _MODEL_SHAPES
        norms = sorted({ablations[name].get('norm_name', 'dytcd') for name in args.names if name in ablations})
        # The rows are trained on the selected --model, so the preflight must
        # exercise the same (n_layers, dim, n_heads) instead of
        # sanity_overfit.py's 2-layer dim=64 default; the small preflight
        # workload (steps/batch/context) is kept on purpose. Per-row knobs
        # (p/k/layerscale variants) are not forwarded: the preflight only
        # needs to catch wiring/NaN/gradient bugs, which are shape- and
        # norm-related.
        n_layers, dim, n_heads = _MODEL_SHAPES.get(args.model, _MODEL_SHAPES['gpt2-124m'])
        arch = ['--dim', str(dim), '--n-layers', str(n_layers), '--n-heads', str(n_heads)]

        run_preflight(norms or ['dytcd'], seeds[0], base, arch_args=arch)
        sanity_report_path = base / 'sanity_overfit.json'
    else:
        sanity_report_path = None
    multi = len(seeds) > 1
    all_summaries = []
    trained: dict[str, str] = {}

    def _report_curve_regimes(agg: dict) -> None:
        """Print the row's per-seed curve verdicts next to its aggregate."""
        regimes = agg.get('curve_classification_counts') or {}
        if not regimes:
            print(f"  [curve] {agg.get('name')}: no curve diagnosis in the cached "
                  f"summaries (pre-monitoring row); rerun with --force to record the "
                  f"train/val regimes")
            return
        print(f"  [curve] {agg['name']}: {regimes}")
        off = {k: v for k, v in regimes.items() if k not in ('healthy', 'insufficient_data')}
        if off:
            print(f"  [curve] WARNING: {agg['name']} has seed(s) in {off}: the val_loss "
                  f"delta may reflect that regime (underfit/stalled/overfit/diverging) as "
                  f"much as the module; read curve_val_loss and the per-seed verdicts "
                  f"before attributing it to the norm")

    for name in args.names:
        if name not in ablations:
            print(f"[warn] unknown ablation '{name}', skipping")
            continue
        cfg_kwargs = ablations[name]
        row_dir = base / name
        summary_paths = {seed: (row_dir / f'seed{seed}' / 'summary.json' if multi else row_dir / 'summary.json') for seed in seeds}
        legacy = row_dir / 'summary.json'
        if multi and legacy.exists() and (not (row_dir / 'summary.preseed.json').exists()) and (not any(p.exists() for p in summary_paths.values())):
            os.replace(legacy, row_dir / 'summary.preseed.json')
            print(f'[keep] {name}: moved pre-replication single-seed summary to summary.preseed.json')
        config_key = effective_config_key(args.model, cfg_kwargs)
        source = trained.get(config_key)
        if source is not None and (not args.force):
            source_paths = {seed: (base / source / f'seed{seed}' / 'summary.json' if multi else base / source / 'summary.json') for seed in seeds}
            if all(p.exists() for p in source_paths.values()):
                reused = [_load_summary(p) for p in source_paths.values()]
                if all(isinstance(s, dict) for s in reused):
                    reused = [dict(s, name=name, cfg=cfg_kwargs, reused_from=source, hypothesis=hypotheses.get(name)) for s in reused]
                    for seed, summary in zip(seeds, reused):
                        if multi:
                            _write_summary(summary_paths[seed], summary)
                    agg = aggregate_seed_summaries(name, reused, cfg_kwargs, hypotheses.get(name))
                    _write_summary(row_dir / 'summary.json', agg)
                    print(f'[alias] {name}: identical model config to {source}, reusing its summaries (no retrain)')
                    _report_curve_regimes(agg)
                    trained[config_key] = name
                    all_summaries.append(agg)
                    continue
        seed_summaries = []
        for seed in seeds:
            path = summary_paths[seed]
            if path.exists() and (not args.force):
                summary = _load_summary(path)
                if isinstance(summary, dict):
                    if cached_seed_matches(summary, name, cfg_kwargs, args, seed):
                        print(f'[skip] {name} seed={seed} already done, use --force to rerun')
                        seed_summaries.append(summary)
                        continue
                    print(f'[stale] {name} seed={seed}: cached summary recorded a different '
                          f'model/cfg/protocol; retraining and overwriting')
            seed_summaries.append(run_one(name, cfg_kwargs, args, path.parent, run_metadata=run_metadata, hypothesis=hypotheses.get(name), seed=seed))
        agg = aggregate_seed_summaries(name, seed_summaries, cfg_kwargs, hypotheses.get(name))
        _write_summary(row_dir / 'summary.json', agg)
        _report_curve_regimes(agg)
        trained[config_key] = name
        all_summaries.append(agg)
    agg_path = base / 'all_summaries.json'
    previous: list = []
    if agg_path.exists():
        try:
            previous = json.loads(agg_path.read_text())
            if not isinstance(previous, list):
                previous = []
        except json.JSONDecodeError:
            previous = []
    ordered = merge_summaries(previous, all_summaries)
    with open(agg_path, 'w') as f:
        json.dump(ordered, f, indent=2)
    # Accumulated rows whose recorded recipe differs from this invocation are
    # reported but excluded from both comparisons: a delta produced by a
    # different lr/step/batch/seq recipe is not a property of the module.
    comparable, stale_rows = partition_rows_by_protocol(ordered, args)
    if stale_rows:
        print(f"[compare] WARNING: {len(stale_rows)} accumulated row(s) excluded from the comparisons "
              f"(trained under a different model/cfg/protocol than this invocation): "
              f"{[r['name'] for r in stale_rows]}; rerun them under this recipe to include them")
    comparison = baseline_comparison(comparable, args.baselines, margin=args.baseline_margin, alpha=args.alpha)
    simple_path = Path(args.simple_baselines) if args.simple_baselines else base / 'simple_baselines.json'
    floor_data = load_simple_baselines(simple_path)
    if comparison is not None and floor_data is not None:
        comparison['simple_baseline'] = attach_simple_baseline_calibration(comparison, floor_data, simple_path)
    elif comparison is not None:
        print(f"[compare] note: no tuned simple-baseline calibration at {simple_path}; run "
              f"'python3 scripts/baseline_simple.py --output {simple_path}' so every row is read against the "
              f"task's Bayes limits and a tuned count model, not only against the other transformer rows")
    if comparison is not None:
        comparison['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'baseline_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    print_baseline_comparison(comparison)
    # The second half of the ablation contract: each claimed mechanism of the
    # main proposal must be shown to be load-bearing *on its own*, with a
    # paired Full-vs-Full-minus-X test over the same seeds.
    loo_comparison = leave_one_out_comparison(comparable, margin=args.baseline_margin, alpha=args.alpha)
    if loo_comparison is not None:
        loo_comparison['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'leave_one_out_comparison.json', 'w') as f:
            json.dump(loo_comparison, f, indent=2)
    print_leave_one_out_comparison(loo_comparison)
    # The third half of the contract: Full combines two axes, so report the
    # 2x2 main effects and their interaction rather than inferring synergy
    # from Full's point estimate alone.
    factorial = factorial_comparison(comparable, margin=args.baseline_margin, alpha=args.alpha)
    if factorial is not None:
        factorial['rows_excluded_protocol_mismatch'] = stale_rows
        with open(base / 'factorial_comparison.json', 'w') as f:
            json.dump(factorial, f, indent=2)
    print_factorial_comparison(factorial)
    manifest = dict(run_metadata)
    manifest['git'] = git_revision()
    manifest['hyperparameters'] = vars(args)
    manifest['seeds'] = seeds
    manifest['matrix_validation'] = matrix_report
    manifest['rows'] = [row['name'] for row in ordered]
    manifest['rows_excluded_protocol_mismatch'] = stale_rows
    # Curve regimes per row (per-seed classification counts): the run manifest
    # keeps the train/val regime each comparison was measured in, so a future
    # reader can tell a healthy falling-falling delta from an early-stopped
    # underfit or an overfitting one without re-reading every summary.json.
    manifest['curve_regimes'] = {
        row.get('name'): row.get('curve_classification_counts')
        for row in ordered if isinstance(row, dict) and row.get('name')
    }
    manifest['baseline_comparison'] = comparison
    manifest['leave_one_out_comparison'] = loo_comparison
    manifest['factorial_comparison'] = factorial
    if sanity_report_path is not None:
        manifest['overfit_sanity_check'] = {'enabled': True, **file_fingerprint(sanity_report_path)}
    else:
        manifest['overfit_sanity_check'] = {'enabled': False}
    # Durable artifacts, so the manifest says which files the comparison above
    # was computed from. `all_summaries.json` can carry every per-step curve
    # and is recorded by size+mtime instead of a full re-hash.
    manifest = finish_metadata(manifest, artifacts={
        'all_summaries': artifact_record(base / 'all_summaries.json'),
        'baseline_comparison': artifact_record(base / 'baseline_comparison.json'),
        'leave_one_out_comparison': artifact_record(base / 'leave_one_out_comparison.json'),
        'factorial_comparison': artifact_record(base / 'factorial_comparison.json'),
        'sanity_report': file_fingerprint(sanity_report_path) if sanity_report_path else {'enabled': False},
    })
    save_run_manifest(base / 'run_manifest.json', manifest)
    print(f'\n[done] {len(ordered)} ablations saved to {base} '
          f'({len(seeds)} seed(s): {seeds}; row val_loss = mean +/- std across seeds)')
if __name__ == '__main__':
    main()
