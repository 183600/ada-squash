from __future__ import annotations

import dataclasses
import itertools
import json

import torch

from adasquash import build_gpt_124m
from adasquash.model import build_gpt_350m, build_llama_tiny_1b
from scripts.train_config import load_yaml_config

ABLATIONS = {'ln_baseline': {'norm_name': 'ln'}, 'rmsnorm_baseline': {'norm_name': 'rmsnorm'}, 'dyt_no_perchannel': {'norm_name': 'dyt'}, 'dytc_perchannel_tanh_k0': {'norm_name': 'dytcd_tanh', 'k_init': 0.0, 'learnable_k': False}, 'dytd_tanh_k0': {'norm_name': 'dytd_tanh', 'k_init': 0.0, 'learnable_k': False}, 'dytcd_full': {'norm_name': 'dytcd'}, 'dytcd_no_perchannel': {'norm_name': 'dytcd', 'learnable_delta': False}, 'dytcd_k0_no_depth_schedule': {'norm_name': 'dytcd', 'k_init': 0.0, 'learnable_k': False}, 'dytcd_no_perchannel_no_depth': {'norm_name': 'dytcd', 'learnable_delta': False, 'k_init': 0.0, 'learnable_k': False}, 'dytd_k0_no_depth_schedule': {'norm_name': 'dytd', 'k_init': 0.0, 'learnable_k': False}, 'dytd_full': {'norm_name': 'dytd'}, 'dytd_tanh_full': {'norm_name': 'dytd_tanh'}, 'dytcd_no_ls': {'norm_name': 'dytcd', 'use_layerscale': False}, 'dytcd_no_softsign': {'norm_name': 'dytcd_tanh'}, 'dytcd_fixed_p': {'norm_name': 'dytcd', 'p_init': 2.0, 'learnable_p': False}, 'dytcd_ls1e5': {'norm_name': 'dytcd', 'layerscale_init': 1e-05}, 'dytcd_ls1e3': {'norm_name': 'dytcd', 'layerscale_init': 0.001}, 'dytc_p1_softsign': {'norm_name': 'dytc', 'p_init': 1.0, 'learnable_p': False, 'alpha_jitter': 0.0}, 'dytc_p2_softsign': {'norm_name': 'dytc', 'p_init': 2.0, 'learnable_p': False, 'alpha_jitter': 0.0}, 'dytc_p4_softsign': {'norm_name': 'dytc', 'p_init': 4.0, 'learnable_p': False, 'alpha_jitter': 0.0}, 'dytc_plearnable': {'norm_name': 'dytc', 'p_init': 2.0, 'learnable_p': True, 'alpha_jitter': 0.0}, 'dytc_hardtanh_eq': {'norm_name': 'dytc', 'p_init': 64.0, 'learnable_p': False, 'alpha_jitter': 0.0}, 'dyt_alpha_0.3': {'norm_name': 'dyt', 'alpha_init': 0.3}, 'dyt_alpha_0.5': {'norm_name': 'dyt', 'alpha_init': 0.5}, 'dyt_alpha_1.0': {'norm_name': 'dyt', 'alpha_init': 1.0}, 'rmsnorm_qk': {'norm_name': 'rmsnorm', 'qk_norm': 'dytqk'}, 'rmsnorm_qknorm': {'norm_name': 'rmsnorm', 'qk_norm': 'qknorm'}, 'ln_qk': {'norm_name': 'ln', 'qk_norm': 'dytqk'}, 'ln_qknorm': {'norm_name': 'ln', 'qk_norm': 'qknorm'}, 'dytcd_qk': {'norm_name': 'dytcd', 'qk_norm': 'dytqk'}, 'dytcd_qknorm': {'norm_name': 'dytcd', 'qk_norm': 'qknorm'}, 'ln': {'norm_name': 'ln', 'use_layerscale': False}, 'ln_ls': {'norm_name': 'ln'}, 'rmsnorm': {'norm_name': 'rmsnorm', 'use_layerscale': False}, 'rmsnorm_ls': {'norm_name': 'rmsnorm'}, 'dyt': {'norm_name': 'dyt', 'alpha_init': 0.5, 'use_layerscale': False}, 'dyts': {'norm_name': 'dyts', 'p_init': 2.0, 'learnable_p': False, 'alpha_init': 0.5, 'use_layerscale': False}, 'dytp': {'norm_name': 'dytp', 'alpha_init': 0.5, 'alpha_jitter': 0.0, 'use_layerscale': False}, 'dytc': {'norm_name': 'dytc', 'p_init': 2.0, 'learnable_p': False, 'alpha_init': 0.5, 'alpha_jitter': 0.0, 'use_layerscale': False}, 'dyt_ls': {'norm_name': 'dyt', 'alpha_init': 0.5, 'use_layerscale': True, 'layerscale_init': 1e-05}, 'dytd': {'norm_name': 'dytd', 'alpha_0': 0.9, 'k_init': 1.0, 'k_min': 0.0, 'use_layerscale': True, 'layerscale_init': 1e-05}, 'dytdi': {'norm_name': 'dytdi', 'alpha_0': 0.9, 'k_init': 1.0, 'use_layerscale': True, 'layerscale_init': 1e-05}, 'dytd_shared': {'norm_name': 'dytd', 'alpha_0': 0.9, 'k_init': 1.0, 'k_min': 0.0, 'shared_depth_schedule': True, 'use_layerscale': True, 'layerscale_init': 1e-05}, 'dytc_ls': {'norm_name': 'dytc', 'alpha_init': 0.5, 'alpha_jitter': 0.0, 'p_init': 2.0, 'learnable_p': True, 'use_layerscale': True, 'layerscale_init': 1e-05}, 'dytcdi': {'norm_name': 'dytcdi', 'alpha_0': 0.9, 'k_init': 1.0, 'p_init': 2.0, 'learnable_p': True, 'alpha_jitter': 0.0, 'use_layerscale': True, 'layerscale_init': 1e-05}, 'dytcd': {'norm_name': 'dytcd', 'alpha_0': 0.9, 'k_init': 1.0, 'k_min': 0.0, 'p_init': 2.0, 'learnable_p': True, 'alpha_jitter': 0.0, 'use_layerscale': True, 'layerscale_init': 1e-05}}
# Every row must state *why* it is run (the scientific hypothesis it tests),
# not just what it configures. `load_ablations_from_config` enforces this for
# YAML rows; these are the hypotheses of the built-in fallback rows.
ABLATION_HYPOTHESES = {'ln_baseline': 'Simple statistical baseline: LayerNorm on the identical protocol. The DyT family must beat this untuned-but-fairly-run reference before any added machinery counts as progress.', 'rmsnorm_baseline': 'Simple statistical baseline: RMSNorm, the strongest no-mean normalization used in modern LLMs. A DyT-family gain is only meaningful relative to this reference.', 'dyt_no_perchannel': 'Original DyT (one scalar alpha, tanh) matches or beats the LayerNorm/RMSNorm references at this scale, reproducing the paper claim under our protocol.', 'dytc_perchannel_tanh_k0': 'Cross-activation per-channel isolation at flat depth: against dytd_k0_no_depth_schedule this row changes exactly norm_name, keeping frozen k=0, tanh, gamma/beta and LayerScale identical, so a lower loss is attributable to the per-channel factor under tanh rather than to the simultaneous tanh -> softsign switch in dytc_p2_softsign vs dyt_no_perchannel.', 'dytcd_full': 'Combining the per-channel and depth axes beats both single-axis variants under the same protocol. The leave-one-out contrasts, not this row alone, identify which component is load-bearing; no positive-interaction claim is made without a 2x2 factorial.', 'dytcd_no_perchannel': 'Leave-one-out for the per-channel axis of the main proposal: with delta frozen at zero (a zero buffer, not a parameter) alpha_c == alpha_l exactly, while the depth schedule, the softsign family and the LayerScale residual stay identical to the full model. Unlike the dytd_full row, this contrast changes no other mechanism, so a loss increase here is attributable to the per-channel deviation alone.', 'dytcd_k0_no_depth_schedule': 'Leave-one-out for the depth axis of the main proposal: with k pinned to 0 and frozen, DyT-CD keeps its per-channel factor, softsign and LayerScale, so any loss of the full model advantage is attributable to depth decay.', 'dytd_k0_no_depth_schedule': 'Depth decay is the source of DyT-D gain: with k pinned to 0 and frozen, dytd_full advantage should vanish.', 'dytd_full': 'Depth-conditioned alpha (+ LayerScale) beats scalar DyT here: shallow strong squashing + deep relaxed response helps this depth.', 'dytcd_no_ls': 'The LayerScale residual gate contributes to DyT-CD; removing it (use_layerscale off) hurts held-out loss.', 'dytcd_no_softsign': 'Leave-one-out for the activation family of the main proposal: the dytcd_tanh comparator keeps the per-channel x depth alpha, the LayerScale residual and gamma/beta identical to dytcd_full and swaps only the generalized softsign for tanh, so a loss increase here is attributable to the activation family alone. Unlike the dytd_full row, it does not also remove the per-channel factor, so the dytd -> dytcd bundle is no longer the only activation-family evidence.', 'dytcd_fixed_p': 'Leave-one-out for the learnable saturation exponent of the main proposal: with p frozen at its init (2.0) the model keeps every mechanism of dytcd_full except the per-layer adaptation of the softsign steepness, so a loss increase here is attributable to learnable p alone.', 'dytcd_ls1e5': 'LayerScale init 1e-5 (the DyT-CD default) is a competitive residual-gate scale, not an arbitrary strawman.', 'dytcd_ls1e3': 'LayerScale init 1e-3 trades warmup stability for stronger early residual growth; the default should not be beaten by it.', 'dytd_tanh_k0': 'Plain-autograd DyT-D at flat depth (k=0 frozen): the "neither" cell of the clean per-channel x depth tanh factorial, matching dytc_perchannel_tanh_k0 in implementation (both plain autograd).', 'dytd_tanh_full': 'Plain-autograd DyT-D with depth schedule: the "depth" cell of the clean per-channel x depth tanh factorial, matching dytcd_no_softsign in implementation (both plain autograd).', 'dytcd_no_perchannel_no_depth': 'DyT-CD with per-channel off (delta frozen) and depth off (k=0 frozen): the "neither" cell of the softsign arm in the 2x2x2 factorial.', 'rmsnorm_qknorm': 'QK-Norm comparator on RMSNorm baseline: per-head RMSNorm on Q/K is the standard alternative; a dytqk win on RMSNorm is only credited if it also beats this comparator.', 'ln_qknorm': 'QK-Norm comparator on LayerNorm baseline: mirrors rmsnorm_qknorm for the other simple baseline.', 'dytc_p1_softsign': 'Softsign sharpness matters: p=1 (classic softsign) is worse than p=2 on this outlier-heavy corpus.', 'dytc_p2_softsign': 'p=2 (smooth softsign) is the best fixed saturation exponent of {1, 2, 4} for outlier control with usable gradients.', 'dytc_p4_softsign': 'p=4 saturates harder than p=2; the hypothesis is that this hurts gradient flow and loses to p=2.', 'dytc_plearnable': 'A learnable p adapts the saturation steepness per layer and beats every fixed p in the sweep.', 'dytc_hardtanh_eq': 'The p->infinity limit (hard tanh) starves gradients outside [-1,1] and loses to finite p.', 'dyt_alpha_0.3': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.', 'dyt_alpha_0.5': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.', 'dyt_alpha_1.0': 'DyT alpha init in {0.3, 0.5, 1.0} materially changes the run; the best must be reported so the family is not compared at a strawman init.', 'rmsnorm_qk': 'Attention-logits axis, standalone: adding the tanh Q/K scalar squashing to the strongest simple baseline (RMSNorm) changes exactly qk_norm, so a held-out-loss reduction is attributable to the QK squashing alone rather than to an interaction with the DyT-CD stack.', 'dytcd_qk': 'Leave-one-out for the attention-logits axis of the proposal (DyT-QK, Variant C) in combination: with the norm stack, optimizer, data and qk_alpha_init (shared default 1.0) identical to dytcd_full, the only effective difference is qk_norm = dytqk, so the paired Full-minus-QK contrast attributes any held-out-loss change to the tanh Q/K squashing alone.', 'dytcd_qknorm': 'QK-Norm comparator for the attention-logits axis: per-head RMSNorm on Q/K is the standard alternative the DyT-QK variant claims to sharpen, run on the identical protocol so a dytcd_qk win is only credited if it also beats this comparator, not just the no-QK row.', 'ln_qk': 'Attention-logits axis, standalone on LayerNorm baseline: adding the tanh Q/K scalar squashing to LayerNorm changes exactly qk_norm, so a held-out-loss reduction is attributable to the QK squashing alone rather than to an interaction with the DyT-CD stack. Mirrors rmsnorm_qk for the other simple baseline.', 'ln': 'Simple LayerNorm baseline without LayerScale; the untreated reference.', 'ln_ls': 'LayerNorm plus LayerScale; isolates the residual gate from the normalization.', 'rmsnorm': 'RMSNorm baseline without LayerScale, the strongest no-mean reference.', 'rmsnorm_ls': 'RMSNorm plus LayerScale; isolates the residual gate from RMSNorm.', 'dyt': 'Original scalar DyT with tanh and no LayerScale matches or beats the simple baselines.', 'dyts': 'Scalar softsign DyT is the activation-family counterpart of scalar tanh DyT.', 'dytp': 'Per-channel tanh isolates the per-channel parameter axis under tanh.', 'dytc': 'Per-channel softsign isolates the per-channel parameter axis under softsign.', 'dyt_ls': 'Adding LayerScale to scalar DyT isolates the residual gate contribution.', 'dytd': 'Depth-conditioned alpha improves on scalar DyT at this depth.', 'dytdi': 'Depth-conditioned alpha with depth-aware initialization isolates the initialization axis.', 'dytd_shared': 'Sharing one alpha/k pair across depth tests whether per-layer schedules are necessary.', 'dytc_ls': 'Per-channel softsign plus LayerScale isolates the residual gate under the strong activation.', 'dytcdi': 'The full per-channel depth proposal with depth-aware initialization.', 'dytcd': 'The full per-channel x depth softsign proposal beats both single-axis variants.'}
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

# The main proposal's row and its 2x2 factorial design. These define the
# primary tier of the study-level gatekeeping correction; a custom matrix with
# different row names must update them (the gate then falls back to
# "not applicable" rather than silently treating an arbitrary row as primary).
PRIMARY_MAIN_ROW = 'dytcd_full'
PRIMARY_FACTORIAL_DESIGN = 'per_channel_x_depth_softsign'

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
    'rmsnorm_qknorm': {
        'qk_norm_comparator': {'row': 'rmsnorm_baseline',
                               'diff': ('qk_norm',)},
    },
    'ln_qk': {
        'qk_squash': {'row': 'ln_baseline',
                      'diff': ('qk_norm',)},
    },
    'ln_qknorm': {
        'qk_norm_comparator': {'row': 'ln_baseline',
                               'diff': ('qk_norm',)},
    },
    'dytcd_qk': {
        'qk_squash': {'row': 'dytcd_full',
                      'diff': ('qk_norm',)},
    },
'dytcd_qknorm': {
        'qk_norm_comparator': {'row': 'dytcd_full',
                               'diff': ('qk_norm',)},
    },
}

# The attention-logits axis is independent of the normalization family: it
# changes only the pre-dot-product Q/K compression and must be attributable
# on each host it is stacked on, not inferred from one row. Each entry is a
# clean triple: the same host norm with QK off, with the fixed tanh DyT-QK
# compression on, and with the standard per-head QK-Norm comparator. The
# checker below requires that both transitions change exactly `qk_norm`.
QK_AXIS_SPEC = {
    'ln': {'off': 'ln_baseline', 'on': 'ln_qk', 'comparator': 'ln_qknorm'},
    'rmsnorm': {'off': 'rmsnorm_baseline', 'on': 'rmsnorm_qk',
                'comparator': 'rmsnorm_qknorm'},
    'dytcd': {'off': 'dytcd_full', 'on': 'dytcd_qk',
              'comparator': 'dytcd_qknorm'},
}

# A complete 2x2 factorial for the two axes the main proposal combines. The
# four cells all use tanh, the same LayerScale residual and the same scalar
# initialization context, so the only moving parts are the per-channel factor
# and the depth schedule. Leave-one-out proves each axis helps from Full;
# this design additionally identifies a *positive interaction* (synergy)
# rather than assuming it from Full's point estimate. `check_factorial`
# verifies that switching a factor changes exactly its declared knobs in both
# arms of the other factor, so a confounded cell is rejected before training.
#
# This factorial now uses plain-autograd comparators for ALL cells (dytd_tanh
# and dytcd_tanh), so the per_channel factor cleanly isolates the per-channel
# axis without mixing fused vs plain implementations. The interaction term is
# now unconfounded and can be used to certify synergy.
FACTORIAL_SPEC = {
    # The main proposal's own activation family (generalized softsign). The
    # four cells already exist in the matrix; without this design the runner
    # only tested the tanh control factorial, so the softsign arm's
    # interaction was never validated or estimated.
    'per_channel_x_depth_softsign': {
        'context': ('generalized-softsign activation family (the main proposal), '
                    'LayerScale on, same gamma/beta and same alpha_0/k/p '
                    'initialization context across the four cells.'),
        'cells': {
            'neither': 'dytcd_no_perchannel_no_depth',
            'per_channel': 'dytcd_k0_no_depth_schedule',
            'depth': 'dytcd_no_perchannel',
            'both': 'dytcd_full',
        },
        'factors': {
            'per_channel': {
                'diff': ('learnable_delta',),
                'pairs': (('neither', 'per_channel'), ('depth', 'both')),
            },
            'depth': {
                'diff': ('k_init', 'learnable_k'),
                'pairs': (('neither', 'depth'), ('per_channel', 'both')),
            },
        },
    },
    'per_channel_x_depth_tanh': {
        'context': ('tanh activation family, LayerScale on, same gamma/beta and '
                    'same alpha/k initialization context across the four cells. '
                    'All cells use plain-autograd (dytd_tanh / dytcd_tanh) so the '
                    'per_channel and depth factors are implementation-clean.'),
        'cells': {
            'neither': 'dytd_tanh_k0',
            'per_channel': 'dytc_perchannel_tanh_k0',
            'depth': 'dytd_tanh_full',
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

# A complete 2x2x2 factorial over the three axes the proposal calls
# orthogonal: per-channel adaptation, depth conditioning, and the activation
# family (generalized softsign vs tanh).  The two 2x2 designs above each
# estimate a per-channel x depth interaction *inside one activation family*;
# neither can tell whether that interaction is the same under the other
# family.  This design does, and it is the only place where the "three
# orthogonal axes" claim is actually tested rather than asserted.
#
# The eight cells already exist in the shipped matrix (they are the union of
# the softsign and tanh 2x2 designs), so the 2x2x2 analysis adds no training
# cost.  `check_three_way_factorial` verifies that switching one factor
# changes exactly the declared knobs in every context of the other two.  The
# per-channel "off" state is implemented as a frozen zero delta buffer in the
# softsign cells and as the delta-free dytd_tanh class in the tanh cells;
# those two representations are functionally the same (both contribute
# exactly alpha_l), and the declared diff sets below record the difference
# explicitly instead of hiding it.  Likewise the activation switch moves
# between a fused kernel and its plain-autograd comparator, which the
# per-parameter kernel tests already pin as numerically equivalent.  The
# activation factor is the whole squash family: the softsign cells carry the
# learnable `p`, the tanh cells do not (the `dytcd_fixed_p` leave-one-out row
# isolates learnable p inside the softsign family), so a cross-family
# interaction is a statement about the family as deployed, not about tanh
# with p held fixed.
#
# The interaction terms are tested two-sided (a synergy and an antagonism are
# both findings), and each pairwise interaction also gets a TOST equivalence
# test.  "The axes are orthogonal" is an equivalence claim: a non-significant
# interaction is not evidence of additivity, so the orthogonality verdict is
# only available when a positive `--equivalence-margin` was registered.
THREE_WAY_FACTORIAL_SPEC = {
    'per_channel_x_depth_x_activation': {
        'design_type': '2x2x2',
        'context': (
            'all eight cells of the shipped matrix: per-channel x depth under '
            'generalized softsign and under tanh, same gamma/beta, LayerScale, '
            'alpha_0/k/p initialization context and optimizer recipe. The '
            'per-channel-off tanh cells use the delta-free dytd_tanh class; '
            'the per-channel-off softsign cells use a frozen zero delta buffer. '
            'The activation factor is the whole squash family, so the softsign '
            'cells carry the learnable p while the tanh cells do not; the '
            'separate dytcd_fixed_p row isolates learnable p within softsign.'),
        'factors': ('per_channel', 'depth', 'activation'),
        'levels': {
            'per_channel': ('off', 'on'),
            'depth': ('off', 'on'),
            'activation': ('softsign', 'tanh'),
        },
        'cells': {
            'per_channel=off|depth=off|activation=softsign': 'dytcd_no_perchannel_no_depth',
            'per_channel=on|depth=off|activation=softsign': 'dytcd_k0_no_depth_schedule',
            'per_channel=off|depth=on|activation=softsign': 'dytcd_no_perchannel',
            'per_channel=on|depth=on|activation=softsign': 'dytcd_full',
            'per_channel=off|depth=off|activation=tanh': 'dytd_tanh_k0',
            'per_channel=on|depth=off|activation=tanh': 'dytc_perchannel_tanh_k0',
            'per_channel=off|depth=on|activation=tanh': 'dytd_tanh_full',
            'per_channel=on|depth=on|activation=tanh': 'dytcd_no_softsign',
        },
        # Allowed effective-config differences for one factor switch, keyed by
        # the context of the *other* factors. A context key is a
        # `|`-separated `factor=level` list; '*' matches every context, and a
        # partial key matches every context that agrees with it.
        'diffs': {
            'per_channel': {
                'activation=softsign': ('learnable_delta',),
                'activation=tanh': ('norm_name',),
            },
            'depth': {
                '*': ('k_init', 'learnable_k'),
            },
            'activation': {
                'per_channel=off|depth=off': ('norm_name', 'learnable_delta'),
                'per_channel=off|depth=on': ('norm_name', 'learnable_delta'),
                'per_channel=on|depth=off': ('norm_name',),
                'per_channel=on|depth=on': ('norm_name',),
            },
        },
    },
}

def _parse_three_way_cell_key(key: str) -> dict:
    """`'per_channel=on|depth=off|activation=tanh'` -> a plain assignment dict."""
    assignment = {}
    for part in str(key).split('|'):
        if not part:
            continue
        name, _, value = part.partition('=')
        if not name or not value:
            raise ValueError(f'malformed factorial cell key {key!r}')
        assignment[name] = value
    return assignment

def _three_way_cell_key(assignment: dict, factors) -> str:
    return '|'.join(f'{factor}={assignment[factor]}' for factor in factors)

def _context_matches(context: str, assignment: dict) -> bool:
    """Whether a partial `factor=level|...` context agrees with `assignment`."""
    if context == '*':
        return True
    for part in str(context).split('|'):
        if not part:
            continue
        name, _, value = part.partition('=')
        if not name or not value:
            return False
        if assignment.get(name) != value:
            return False
    return True

def check_three_way_factorial(rows: dict, model_name: str,
                              spec: dict | None=None) -> dict:
    """Verify the 2x2x2 design is complete and every factor switch is clean.

    A 2x2x2 interaction is only interpretable when each factor is switched
    against the same background in every context.  For each factor and each
    combination of the other two factors this checks that the effective model
    config changes in exactly one of the declared ways (the implementation of
    "per-channel off" legitimately differs between the softsign and tanh
    cells, so the allowed diff is context-dependent), and that all eight
    cells are present.  A missing or confounded cell is rejected before any
    training starts, exactly like the 2x2 coverage check.
    """
    spec = THREE_WAY_FACTORIAL_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for design_name, design in spec.items():
        factors = tuple(design.get('factors') or ())
        levels = design.get('levels') or {}
        cells = design.get('cells') or {}
        diffs = design.get('diffs') or {}
        if len(factors) != 3:
            problems.append(f"3-way factorial '{design_name}' must declare exactly "
                            f"three factors, got {list(factors)}")
            continue
        missing_levels = [factor for factor in factors
                          if len(levels.get(factor) or ()) != 2]
        if missing_levels:
            problems.append(f"3-way factorial '{design_name}' factor(s) without two "
                            f"levels: {missing_levels}")
            continue
        expected_cells = {
            _three_way_cell_key(dict(zip(factors, combo)), factors): combo
            for combo in itertools.product(*(levels[factor] for factor in factors))
        }
        missing = sorted(set(expected_cells) - set(cells))
        extra = sorted(set(cells) - set(expected_cells))
        if missing or extra:
            problems.append(
                f"3-way factorial '{design_name}' does not cover the full 2x2x2: "
                f"missing {missing}, extra {extra}")
            continue
        absent_rows = sorted({row for row in cells.values() if row not in rows})
        if absent_rows:
            problems.append(
                f"3-way factorial '{design_name}' references absent row(s) "
                f"{absent_rows}")
            continue
        cfg_cache: dict[str, dict] = {}

        def _cfg(row_name: str) -> dict:
            if row_name not in cfg_cache:
                cfg_cache[row_name] = _effective_cfg(model_name, rows[row_name])
            return cfg_cache[row_name]

        report[design_name] = {
            'design_type': design.get('design_type', '2x2x2'),
            'context': design.get('context'),
            'factors': list(factors),
            'levels': {factor: list(levels[factor]) for factor in factors},
            'cells': dict(cells),
            'switches': {},
        }
        clean = True
        for factor in factors:
            others = [name for name in factors if name != factor]
            allowed_by_context = diffs.get(factor) or {}
            if not allowed_by_context:
                problems.append(
                    f"3-way factorial '{design_name}', factor '{factor}': no "
                    f"declared effective-config diff")
                clean = False
                break
            switches = {}
            for combo in itertools.product(*(levels[name] for name in others)):
                background = dict(zip(others, combo))
                first, second = levels[factor]
                first_key = _three_way_cell_key(
                    dict(background, **{factor: first}), factors)
                second_key = _three_way_cell_key(
                    dict(background, **{factor: second}), factors)
                first_cfg, second_cfg = _cfg(cells[first_key]), _cfg(cells[second_key])
                actual = {key for key in set(first_cfg) | set(second_cfg)
                          if first_cfg.get(key) != second_cfg.get(key)}
                allowed = [set(value) for context, value in allowed_by_context.items()
                           if _context_matches(context, background)]
                if not allowed:
                    problems.append(
                        f"3-way factorial '{design_name}', factor '{factor}': "
                        f"context {background} has no declared diff")
                    clean = False
                    break
                if not any(actual == candidate for candidate in allowed):
                    problems.append(
                        f"3-way factorial '{design_name}', factor '{factor}' in "
                        f"{background}: {cells[second_key]} vs {cells[first_key]} "
                        f"differ in {sorted(actual)}, expected one of "
                        f"{[sorted(value) for value in allowed]}")
                    clean = False
                    break
                switches[_three_way_cell_key(background, others)] = {
                    'from': cells[first_key], 'to': cells[second_key],
                    'diff': sorted(actual),
                }
            if not clean:
                break
            report[design_name]['switches'][factor] = switches
    if problems:
        raise ValueError('3-way factorial coverage is incomplete: '
                         + '; '.join(problems))
    return report

def _effective_cfg(model_name: str, cfg_kwargs: dict) -> dict:
    """The full GPTConfig a matrix row builds, as a plain dict."""
    with torch.device('meta'):
        cfg = _MODEL_BUILDERS[model_name](init=False, **cfg_kwargs).cfg
    return dataclasses.asdict(cfg)

def effective_config_key(model_name: str, cfg_kwargs: dict, seed: int | None=None) -> str:
    """Canonical identity of the model a matrix row trains.

    Rows may spell out builder/GPTConfig defaults explicitly (e.g.
    ``layerscale_init: 1.0e-5`` or ``alpha_init: 0.5``); two rows with equal
    keys build the exact same GPTConfig and, under the same seed, produce
    bit-identical runs. The runner trains the first and reuses its summary
    for the rest. The seed is part of the key: identical configs under
    different seeds are *different* runs and must both be trained.
    """
    key = json.dumps(_effective_cfg(model_name, cfg_kwargs), sort_keys=True)
    if seed is not None:
        key = f'{key}#seed={seed}'
    return key

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
            # A 2x2 factor must be switched in both arms of the other factor;
            # one/no pair is not a factorial contrast (and leaves the
            # comparison's main-effect mean over an empty list).
            if len(info['pairs']) != 2:
                problems.append(
                    f"factorial '{design_name}', factor '{factor}': expected "
                    f"exactly 2 on/off pairs, got {len(info['pairs'])}")
                clean = False
                continue
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

def check_qk_axis(rows: dict, model_name: str, spec: dict | None=None) -> dict:
    """Verify the QK-compression axis is isolated on every host norm.

    QK compression is a separate research axis from the normalization layer,
    so its effect must be measured by switching *only* `qk_norm` on a host
    whose weights/initialization are otherwise identical. This checks the
    off/on pair and the QK-Norm comparator for each host, rejecting a triple
    that also changes depth, initialization or the optimizer protocol.
    """
    spec = QK_AXIS_SPEC if spec is None else spec
    problems: list[str] = []
    report: dict = {}
    for host, info in spec.items():
        off_name, on_name = info['off'], info['on']
        cmp_name = info['comparator']
        missing = [name for name in (off_name, on_name, cmp_name) if name not in rows]
        if missing:
            problems.append(f"QK axis on '{host}' is missing row(s) {missing}")
            continue
        off_cfg = _effective_cfg(model_name, rows[off_name])
        on_cfg = _effective_cfg(model_name, rows[on_name])
        cmp_cfg = _effective_cfg(model_name, rows[cmp_name])
        on_diff = {key for key in set(off_cfg) | set(on_cfg)
                   if off_cfg.get(key) != on_cfg.get(key)}
        cmp_diff = {key for key in set(on_cfg) | set(cmp_cfg)
                    if on_cfg.get(key) != cmp_cfg.get(key)}
        if on_diff != {'qk_norm'}:
            problems.append(
                f"QK axis on '{host}': {on_name} vs {off_name} differ in "
                f"{sorted(on_diff)}, expected exactly ['qk_norm']")
            continue
        if cmp_diff != {'qk_norm'}:
            problems.append(
                f"QK comparator on '{host}': {cmp_name} vs {on_name} differ in "
                f"{sorted(cmp_diff)}, expected exactly ['qk_norm']")
            continue
        if on_cfg.get('qk_norm') != 'dytqk':
            problems.append(
                f"QK axis on '{host}': {on_name} does not enable the fixed "
                f"tanh compression (qk_norm={on_cfg.get('qk_norm')!r})")
            continue
        report[host] = {'off': off_name, 'on': on_name,
                        'comparator': cmp_name,
                        'isolated_field': 'qk_norm'}
    if problems:
        raise ValueError('QK-axis coverage is incomplete: ' + '; '.join(problems))
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
            'factorial': check_factorial(rows, model_name),
            'three_way_factorial': check_three_way_factorial(rows, model_name),
            'qk_axis': check_qk_axis(rows, model_name)}

def install_residual_outlier(model, layer_idx: int | None=None,
                             channel: int | None=None, scale: float=25.0):
    """Amplify one residual-stream channel after a mid layer, for the whole run.

    This is the preferred probe: unlike inflating a token-embedding row, it
    does not touch any parameter (with tied embeddings that row is also an
    output-classifier row, so the old probe was confounded with an output-side
    pathology). A persistent forward hook scales the chosen channel of the
    residual stream after ``layer_idx``, giving every downstream layer the
    classic persistent-channel outlier without changing the weights.
    """
    n_layers = int(getattr(model.cfg, 'n_layers', 12))
    dim = int(getattr(model.cfg, 'dim', 768))
    layer_idx = n_layers // 2 if layer_idx is None else int(layer_idx)
    channel = dim // 2 if channel is None else int(channel)
    block = model.blocks[layer_idx]

    def hook(module, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        y[..., channel].mul_(scale)
        return out
    block.register_forward_hook(hook)
    return {'kind': 'residual_channel', 'layer_idx': layer_idx,
            'channel': channel, 'scale': scale}

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

