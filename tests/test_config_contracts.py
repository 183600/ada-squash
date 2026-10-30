from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'configs'
PHASE1_CONFIGS = sorted(CONFIG_DIR.glob('gpt2-124m-*.yaml'))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def _builder_for(path: Path):
    if path.name == 'llama-1b-depth-sweep.yaml':
        from scripts.train_llama_depth_sweep import build_arg_parser
    elif path.name == 'ablation-matrix.yaml':
        from adasquash.ablation.cli import build_arg_parser
    else:
        from scripts.train_gpt2 import build_arg_parser
    return build_arg_parser


@pytest.mark.parametrize('path', sorted(CONFIG_DIR.glob('*.yaml')),
                         ids=lambda path: path.name)
def test_every_shipped_config_loads_through_its_entrypoint(path):
    """The shipped YAML must match the real parser, not a parallel schema.

    `build_arg_parser` applies the config with `strict=True`, so this covers
    unknown keys, missing options, argparse types, choices and booleans.
    """
    _builder_for(path)(['--config', str(path)])


def test_phase1_configs_declare_replication_validation_and_hypothesis():
    output_dirs = set()
    for path in PHASE1_CONFIGS:
        cfg = _load(path)
        seeds = cfg.get('seeds')
        assert isinstance(seeds, list), f'{path.name}: seeds must be an explicit list'
        assert len(seeds) >= 3, f'{path.name}: Phase 1 needs >= 3 independent seeds'
        assert all(isinstance(seed, int) and not isinstance(seed, bool)
                   for seed in seeds), f'{path.name}: seeds must be integers'
        assert len(set(seeds)) == len(seeds), f'{path.name}: duplicate seeds'

        assert isinstance(cfg.get('hypothesis'), str) and cfg['hypothesis'].strip(), (
            f'{path.name}: state the hypothesis the run tests')
        assert isinstance(cfg.get('val_every'), int) and cfg['val_every'] >= 1, (
            f'{path.name}: val_every must be explicit and positive')
        assert isinstance(cfg.get('val_batches'), int) and cfg['val_batches'] >= 1, (
            f'{path.name}: val_batches must be explicit and positive')
        assert isinstance(cfg.get('val_dataset'), str) and cfg['val_dataset'].strip(), (
            f'{path.name}: val_dataset must be explicit')
        assert isinstance(cfg.get('val_dataset_config'), str) and cfg['val_dataset_config'].strip(), (
            f'{path.name}: val_dataset_config must be explicit')
        assert isinstance(cfg.get('val_split'), str) and cfg['val_split'].strip(), (
            f'{path.name}: val_split must be explicit')

        output_dir = cfg.get('output_dir')
        assert isinstance(output_dir, str) and output_dir.strip(), (
            f'{path.name}: output_dir must be explicit')
        assert output_dir not in output_dirs, (
            f'{path.name}: output_dir {output_dir!r} collides with another config')
        output_dirs.add(output_dir)


def test_sweep_and_ablation_configs_declare_replication_and_protocol():
    sweep = _load(CONFIG_DIR / 'llama-1b-depth-sweep.yaml')
    assert len(sweep.get('seeds', [])) >= 3
    assert sweep.get('val_every', 0) >= 1
    assert sweep.get('val_batches', 0) >= 1

    matrix = _load(CONFIG_DIR / 'ablation-matrix.yaml')
    assert len(matrix.get('seeds', [])) >= 3
    rows = matrix.get('ablations') or []
    assert rows
    assert all(isinstance(row.get('hypothesis'), str) and row['hypothesis'].strip()
               for row in rows)


def test_config_values_are_checked_against_the_argparse_contract(tmp_path):
    from scripts.train_config import _TRAIN_CONFIG_ALIASES, apply_config_to_parser

    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', choices=['dyt', 'dytcd'], default='dyt')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--qk-norm', choices=[None, 'qknorm'], default=None)
    parser.add_argument('--layerscale-init', type=float, default=None)
    parser.add_argument('--no-layerscale', action='store_true')
    parser.add_argument('--shared-depth-schedule', action='store_true')

    def apply(text: str):
        path = tmp_path / 'run.yaml'
        path.write_text(text, encoding='utf-8')
        apply_config_to_parser(parser, str(path), _TRAIN_CONFIG_ALIASES,
                               strict=True)
        return parser.parse_args([])

    with pytest.raises(ValueError, match='not one of'):
        apply('norm: dytcc\n')
    with pytest.raises(ValueError, match='must be an integer'):
        apply('batch_size: "32"\n')
    with pytest.raises(ValueError, match='must be a list'):
        apply('seeds: 42\n')
    with pytest.raises(ValueError, match='not nullable'):
        apply('max_steps: null\n')

    args = apply(
        'batch_size: 8\n'
        'lr: 1\n'
        'seeds: [1, 2, 3]\n'
        'qk_norm: null\n'
        'layerscale_init: null\n'
        'use_layerscale: false\n'
        'shared_depth_schedule: "yes"\n'
    )
    assert args.batch_size == 8
    assert args.lr == 1.0
    assert args.seeds == [1, 2, 3]
    assert args.qk_norm is None
    assert args.layerscale_init is None
    assert args.no_layerscale is True
    assert args.shared_depth_schedule is True
