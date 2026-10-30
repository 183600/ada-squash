"""Execution-identity checks: code + numeric stack are part of the protocol.

A hyperparameter recipe can match while two arms still ran under different
commits, torch/CUDA builds, thread counts or TF32/cuDNN settings. These tests
pin the fail-closed contract used by the cross-arm certificate:

* a recorded difference is incomparable, even when every recipe field matches;
* a dirty working tree is incomparable because the commit does not pin code;
* missing identity fields are *unknown*, not equal, and become fatal only when
  the caller asks for a confirmatory comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adasquash.repro import (EXECUTION_IDENTITY_KEYS, RUN_MANIFEST_SCHEMA,
                             TRAINING_RECIPE_KEYS, collect_run_metadata,
                             execution_identity_mismatches,
                             training_recipe_mismatches)

SEEDS = (42, 43, 44)


def _environment(**overrides):
    environment = {
        'python': '3.12.0',
        'platform': 'Linux-6.1-x86_64',
        'torch': '2.14.1',
        'numpy': '2.5.4',
        'transformers': '4.44.0',
        'datasets': '2.20.0',
        'tokenizers': '0.19.1',
        'pyyaml': '6.0.1',
        'cuda_available': True,
        'cuda_version': '12.4',
        'cudnn_version': 90100,
        'device_names': ['NVIDIA A100-SXM4-80GB'],
        'rng_config': {
            'torch_num_threads': 8,
            'torch_num_interop_threads': 1,
            'cudnn_deterministic': True,
            'cudnn_benchmark': False,
            'tf32_matmul_allowed': False,
            'tf32_cudnn_allowed': False,
            'deterministic_algorithms': False,
            'deterministic_algorithms_warn_only': False,
            'cublas_workspace_config': ':4096:8',
            'hash_randomization': True,
            'pythonhashseed_at_process_start': None,
        },
    }
    environment.update(overrides)
    return environment


def _manifest(commit='abc123', dirty=False, environment=None):
    return {
        'schema': RUN_MANIFEST_SCHEMA,
        'git': {'commit': commit, 'branch': 'master', 'dirty': dirty},
        'environment': environment or _environment(),
    }


def _aggregate(values=None):
    values = values or {seed: 2.0 for seed in SEEDS}
    return {
        'norm': 'synthetic',
        'model': 'gpt2-124m',
        'seeds': sorted(values),
        'complete': True,
        'per_seed': [
            {'seed': seed, 'final_val_loss': value, 'final_val_step': 1000}
            for seed, value in sorted(values.items())
        ],
        'shapes_by_seed': {
            str(seed): {'n_layers': 12, 'dim': 768, 'n_heads': 12}
            for seed in values
        },
        'data_versions_by_seed': {
            str(seed): {'name': 'corpus', 'revision': 'v1'} for seed in values
        },
        'data_ledgers_by_seed': {
            str(seed): {'sha256': 'ledger', 'bytes': 10} for seed in values
        },
    }


def _run_manifest(aggregate, identity=None):
    return {
        'status': 'finished',
        'seeds': list(aggregate['seeds']),
        'aggregate': aggregate,
        'hyperparameters': {key: 'synthetic' for key in TRAINING_RECIPE_KEYS},
        **(identity or {}),
    }


def test_execution_identity_accepts_identical_manifests():
    report = execution_identity_mismatches(_manifest(), _manifest())
    assert report['recorded'] is True
    assert report['comparable'] is True
    assert report['identical'] is True
    assert report['problems'] == []
    assert report['unknown'] == []
    assert report['mismatches'] == {}


def test_execution_identity_detects_commit_stack_and_thread_mismatches():
    other = _manifest(commit='def456')
    other['environment']['torch'] = '2.15.0'
    other['environment']['rng_config']['torch_num_threads'] = 4
    report = execution_identity_mismatches(_manifest(), other)
    assert report['comparable'] is False
    assert set(report['mismatches']) == {
        'git.commit', 'environment.torch',
        'environment.rng_config.torch_num_threads'}
    assert any('git.commit' in problem for problem in report['problems'])


def test_execution_identity_dirty_tree_blocks_even_with_same_commit():
    report = execution_identity_mismatches(_manifest(), _manifest(dirty=True))
    assert report['mismatches'] == {}
    assert report['comparable'] is False
    assert report['dirty_tree'] == {'reference': False, 'other': True}
    assert any('dirty working tree' in problem for problem in report['problems'])


def test_execution_identity_missing_fields_are_unknown_not_equal():
    partial = _manifest()
    del partial['environment']['numpy']
    report = execution_identity_mismatches(_manifest(), partial)
    assert report['comparable'] is False
    assert 'environment.numpy' in report['unknown']
    assert 'environment.numpy' not in report['mismatches']
    assert report['problems'] == []


def test_execution_identity_null_commit_is_unknown_not_matching():
    report = execution_identity_mismatches(
        _manifest(commit=None), _manifest(commit=None))
    assert report['comparable'] is False
    assert 'git.commit' in report['unknown']
    assert 'git.commit' not in report['mismatches']


def test_execution_identity_unreadable_manifest_is_unknown():
    report = execution_identity_mismatches(None, _manifest())
    assert report['recorded'] is False
    assert report['comparable'] is False
    assert set(report['unknown']) == set(EXECUTION_IDENTITY_KEYS) | {'git.dirty'}
    assert any('missing or unreadable' in problem for problem in report['problems'])


def test_execution_identity_rejects_a_non_boolean_dirty_flag():
    other = _manifest()
    other['git']['dirty'] = 'false'
    report = execution_identity_mismatches(_manifest(), other)
    assert report['comparable'] is False
    assert any('git.dirty is not a boolean' in problem
               for problem in report['problems'])


def test_collected_manifest_carries_a_schema_and_complete_identity():
    metadata = collect_run_metadata(seed=7)
    assert metadata['schema'] == RUN_MANIFEST_SCHEMA
    assert metadata['environment']['device_names'] == []
    copied = json.loads(json.dumps(metadata))
    # The test checkout itself is dirty; a dirty tree is deliberately an
    # incomparability, so compare the recorded fields with the dirt flag
    # cleared rather than depending on the test runner's git status.
    metadata['git']['dirty'] = False
    copied['git']['dirty'] = False
    report = execution_identity_mismatches(metadata, copied)
    assert report['comparable'] is True, report


def _arms_and_manifests(candidate_identity=None, reference_identity=None):
    reference = _aggregate()
    candidate = _aggregate({seed: 1.5 for seed in SEEDS})
    arms = {'reference': reference, 'candidate': candidate}
    manifests = {
        'reference': _run_manifest(reference, reference_identity or _manifest()),
        'candidate': _run_manifest(candidate, candidate_identity or _manifest()),
    }
    return arms, manifests


def test_compare_arms_blocks_a_recorded_execution_mismatch():
    from scripts.compare_arms import compare_arms

    arms, manifests = _arms_and_manifests(
        candidate_identity=_manifest(commit='different'))
    out = compare_arms(arms, reference='reference', manifests=manifests,
                       require_recipe=True)
    row = out['rows'][0]
    assert row['execution_identity']['mismatches']
    assert row['certified'] is False
    assert any('execution identity' in problem
               for problem in row['protocol_problems'])
    assert out['protocol']['require_execution_identity'] is False


def test_compare_arms_missing_identity_only_blocks_when_required():
    from scripts.compare_arms import compare_arms

    arms, manifests = _arms_and_manifests()
    for manifest in manifests.values():
        manifest.pop('environment')
        manifest.pop('git')
    descriptive = compare_arms(arms, reference='reference', manifests=manifests,
                               require_recipe=True)
    row = descriptive['rows'][0]
    assert row['execution_identity']['unknown']
    assert row['certified'] is True

    strict = compare_arms(arms, reference='reference', manifests=manifests,
                          require_recipe=True,
                          require_execution_identity=True)
    strict_row = strict['rows'][0]
    assert strict_row['certified'] is False
    assert any('execution identity' in problem
               for problem in strict_row['protocol_problems'])
    assert strict['protocol']['require_execution_identity'] is True


def test_execution_identity_is_not_replaced_by_recipe_equality():
    # The same recipe under two identities is exactly the case the recipe
    # comparison cannot see, so the two reports must disagree.
    recipe = {key: 'same' for key in TRAINING_RECIPE_KEYS}
    assert training_recipe_mismatches(recipe, recipe) == {}
    assert execution_identity_mismatches(
        _manifest(), _manifest(commit='changed'))['comparable'] is False
