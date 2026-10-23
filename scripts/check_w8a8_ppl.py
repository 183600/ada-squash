from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from adasquash import markov_batch
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust, mean_std_stderr,
                               paired_t_one_sided, scale_replication_summary)
from adasquash.model import MLP
from adasquash.repro import (TRAINING_RECIPE_KEYS, collect_run_metadata, file_fingerprint,
                             finish_metadata, save_run_manifest, set_all_seeds,
                             training_recipe_mismatches)
from scripts.eval_downstream import load_ckpt
from scripts.data_pipeline import _is_pinned_revision
DEFAULT_SEEDS = (42, 43, 44)
# One probe draw is one sample: the PPL drop is aggregated as Mean +/- Std
# over the `--seeds` probe replications, with the per-seed records kept so a
# quantization result resting on one lucky probe cannot hide.
#
# Significance needs the training-run dimension replicated too: a `--ckpts`
# entry may be a comma-separated list of checkpoints (ideally distinct training
# seeds of the same arm), which are all probed and averaged per probe seed.
# `compare_w8a8_checkpoints` certifies a cross-arm Delta only when both arms
# supply >= 3 checkpoints with shared training seeds; two are descriptive only; with a single trained
# checkpoint per arm the probe-paired test is descriptive only, because one run
# per arm cannot separate the quantization effect from run-to-run noise. The
# same applies to the per-arm degradation test, so "0.2% drop under a 0.5% seed
# spread" is not a result; the comparison is saved next to the results as
# `ppl_comparison.json`.
_W8A8_SEED_METRIC_KEYS = ('ppl_fp', 'ppl_w8a8', 'ppl_drop', 'ppl_drop_pct',
                          'probe_token_coverage')

def _explicit_cli_names(argv, parser=None) -> set:
    """Option names typed on the command line (`--val-frac=...` or `--val-frac ...`).

    argparse accepts unambiguous prefixes (`--val-f` for `--val-frac`); when a
    parser is supplied those spellings are resolved to the destination name,
    otherwise the literal normalized spelling is returned.
    """
    names = set()
    for token in argv:
        if token == '--':
            # Everything after the separator is a positional value, never an
            # explicit option spelling.
            break
        if token.startswith('--') and len(token) > 2:
            option = token[2:].split('=', 1)[0]
            if parser is not None:
                try:
                    parsed = parser._parse_optional('--' + option)
                except Exception:
                    parsed = None
                if parsed:
                    entries = parsed if isinstance(parsed, list) else [parsed]
                    actions = [entry[0] for entry in entries
                               if entry and entry[0] is not None]
                    dests = {action.dest for action in actions}
                    if len(dests) == 1:
                        names.add(dests.pop())
                        continue
            names.add(option.replace('-', '_'))
    return names

def _checkpoint_defaulted(args, parser, ckpt_args: dict, name: str,
                          explicit: set | None = None):
    """The CLI value when explicitly set, else the checkpoint's recorded value.

    Comparing the parsed value to the parser default cannot tell an explicit
    `--val-frac 0.005` from an omitted flag when the checkpoint used a
    different fraction, so callers pass the names actually typed.
    """
    value = getattr(args, name)
    if explicit is not None and name in explicit:
        return value
    # Key presence, not truthiness, records the checkpoint's value: a
    # legitimate `dataset_config=None` (datasets without a config) must stay
    # None instead of being replaced by the CLI default and probing a
    # different corpus configuration than the checkpoint was trained on.
    if isinstance(ckpt_args, dict) and name in ckpt_args:
        return ckpt_args[name]
    return value

def quantize_weight_per_channel(w: torch.Tensor, n_bits: int=8):
    # `ord=inf` vector norm is max(|w|).
    mx = torch.linalg.vector_norm(w, ord=float('inf'), dim=-1, keepdim=True).clamp_(min=1e-06)
    q_max = 2 ** (n_bits - 1) - 1
    scale = mx / q_max
    q = w / scale
    q.round_()
    q.clamp_(-q_max - 1, q_max)
    q.mul_(scale)
    return (q, scale)

def quantize_activation_per_token(x: torch.Tensor, n_bits: int=8):
    mx = torch.linalg.vector_norm(x, ord=float('inf'), dim=-1, keepdim=True).clamp_(min=1e-06)
    q_max = 2 ** (n_bits - 1) - 1
    scale = mx / q_max
    q = x / scale
    q.round_()
    q.clamp_(-q_max - 1, q_max)
    q.mul_(scale)
    return q

def _install_quantized_weight(linear: nn.Module, qw: torch.Tensor):
    """Quantize `linear.weight` in place.

    The forward closures and the fused CE both read `weight`, so the parameter
    storage itself holds the quantized values; `_q_weight` is a detached view
    of that same storage for callers that want the quantized tensor.
    """
    with torch.no_grad():
        linear.weight.copy_(qw)
    linear._q_weight = linear.weight.detach()
    linear._w8a8_patched = True

def _unwrap_w8a8(model: nn.Module) -> None:
    """Undo the instance-level patches so the next pass starts from fp32.

    `wrap_model_w8a8` quantizes weights in place and rebinds `forward` on the
    module *instances*; reloading the fp32 state dict alone would leave the
    quantized forward closures (and a duplicated norm hook) behind.
    """
    for m in model.modules():
        if getattr(m, '_w8a8_patched', False):
            m.forward = type(m).forward.__get__(m, type(m))
            m._w8a8_patched = False
        if '_q_weight' in vars(m):
            delattr(m, '_q_weight')
    handle = getattr(model, '_w8a8_act_hook', None)
    if handle is not None:
        handle.remove()
        model._w8a8_act_hook = None

def wrap_linear_w8a8(linear: nn.Linear):
    qw, _ = quantize_weight_per_channel(linear.weight.data)
    _install_quantized_weight(linear, qw)

    def quantized_forward(x):
        x_q = quantize_activation_per_token(x)
        return torch.nn.functional.linear(x_q, linear.weight, linear.bias)
    linear.forward = quantized_forward
    return linear

def wrap_swiglu_w8a8(mlp):
    weights = []
    for lin in (mlp.w1, mlp.w2, mlp.w3):
        qw, _ = quantize_weight_per_channel(lin.weight.data)
        _install_quantized_weight(lin, qw)
        # The wrapped forward reads the parameter storage.
        weights.append(lin.weight)
    qa = quantize_activation_per_token

    def quantized_forward(x):
        x_q = qa(x)
        a = qa(F.linear(x_q, weights[0]))
        b = qa(F.linear(x_q, weights[2]))
        h = F.silu(a)
        h.mul_(b)
        return mlp.drop(F.linear(qa(h), weights[1]))
    mlp.forward = quantized_forward
    mlp._w8a8_patched = True
    return mlp

def wrap_embedding_w8a8(emb: nn.Embedding):
    # The embedding is tied to `lm_head.weight`; quantizing the parameter in
    # place keeps that tie intact and leaves a single quantized storage.
    qw, _ = quantize_weight_per_channel(emb.weight.data)

    def quantized_forward(idx):
        return F.embedding(idx, emb.weight)
    with torch.no_grad():
        emb.weight.copy_(qw)
    emb._q_weight = emb.weight.detach()
    emb.forward = quantized_forward
    emb._w8a8_patched = True
    return emb

def wrap_model_w8a8(model: nn.Module):
    handled: set[int] = set()
    for m in model.modules():
        if isinstance(m, MLP):
            wrap_swiglu_w8a8(m)
            handled.update((id(child) for child in (m.w1, m.w2, m.w3)))
            continue
        if isinstance(m, nn.Embedding):
            wrap_embedding_w8a8(m)
            continue
        if isinstance(m, nn.Linear):
            if id(m) in handled:
                continue
            wrap_linear_w8a8(m)
    norm_f = getattr(model, 'norm_f', None)
    if isinstance(norm_f, nn.Module) and isinstance(getattr(model, 'lm_head', None), nn.Module):
        # Register once: a repeated wrap would stack another quantizing hook
        # and double-quantize the final norm output.
        if getattr(model, '_w8a8_act_hook', None) is None:
            model._w8a8_act_hook = norm_f.register_forward_hook(
                lambda m, inp, out: quantize_activation_per_token(out))
    return model

@torch.no_grad()
def compute_ce_sum(model, batch, target, device='cuda'):
    """`(sum of loss*N, N)` as device scalars; callers accumulate on device.

    The pair stays on the device; the loss is widened to fp64 before the
    multiply so the accumulated total is fp64. `model.eval()` is the caller's
    responsibility (once per model, not once per batch).
    """
    batch = batch.to(device)
    target = target.to(device)
    n = (target != model.cfg.ignore_index).sum()
    loss = model(batch, target, return_logits=False)[1]
    return (loss.detach().double() * n.double(), n)

def _exp_or_inf(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        # A finite but huge CE is a diverged model, not a reason to abort the
        # probe: record the PPL as non-finite and let the seed-level filter
        # report it instead of crashing mid-run.
        return math.inf

def compute_ppl(model, batch, target, device='cuda'):
    ce, n = compute_ce_sum(model, batch, target, device=device)
    n_val = int(n.item())
    if n_val == 0:
        raise ValueError('probe batch contains no valid target tokens')
    return _exp_or_inf(float(ce) / n_val)

def probe_token_coverage(batches) -> float:
    n_pos = 0
    ids: set = set()
    for x in batches:
        flat = x.reshape(-1)
        ids.update(flat.tolist())
        n_pos += flat.numel()
    return len(ids) / max(1, n_pos)

def _probe_random(rng, vocab_size, batch_size, seq_len, device):
    return torch.randint(0, vocab_size, (batch_size, seq_len), generator=rng, device=device)

def _probe_markov(rng, batch_size, seq_len, device, n_states=64):
    x, _ = markov_batch(rng, -100, batch_size, seq_len, device, n_states=n_states)
    return x

def _next_token_targets(x: torch.Tensor, ignore_index: int) -> torch.Tensor:
    y = torch.full_like(x, ignore_index)
    y[:, :-1] = x[:, 1:]
    return y

def make_dataset_probe(dataset, dataset_config, tokenizer, batch_size, seq_len, split='validation', val_frac=0.005, test_frac=0.005, num_documents=None, dedup=True, min_doc_chars=0, max_doc_chars=0, exclude_hash_file=None, exclude_test_hash_file=None, group_key=None, exclude_group_file=None, exclude_test_group_file=None, dataset_revision=None, tokenizer_revision=None, seed=42, num_workers=2):
    # Scoring a checkpoint on the documents it trained on understates the
    # quantization damage; the probe must come from a held-out split. The
    # default is the chronological validation slice, and the cleaning policy
    # (dedup / length caps / cross-split duplicate and group ledgers) is taken
    # from the checkpoint so the probe is the same held-out distribution the
    # model was validated on. The dataset/tokenizer revisions are the data
    # *version*: a probe on a moved upstream branch would not be the same
    # held-out slice.
    from scripts.train_gpt2 import build_dataloader
    from scripts.data_pipeline import configure_data_options
    configure_data_options(split=split, val_frac=val_frac, test_frac=test_frac,
                           num_documents=num_documents, dedup=dedup,
                           min_doc_chars=min_doc_chars, max_doc_chars=max_doc_chars,
                           exclude_hash_file=exclude_hash_file,
                           exclude_test_hash_file=exclude_test_hash_file,
                           group_key=group_key,
                           exclude_group_file=exclude_group_file,
                           exclude_test_group_file=exclude_test_group_file,
                           dataset_revision=dataset_revision,
                           tokenizer_revision=tokenizer_revision,
                           seed=seed, num_workers=num_workers)
    loader = build_dataloader(dataset, dataset_config, seq_len, batch_size, tokenizer=tokenizer)
    report = getattr(loader, 'split_report', None)
    if split != 'train' and report:
        print(f'[w8a8] dataset probe split: {report}', file=sys.stderr)
    it = iter(loader)

    def _next():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)
    # Keep the split report on the callable so the manifest can pin the data
    # version (boundary, resolved dataset/tokenizer commits, cleaning policy)
    # of the exact probe that produced the reported PPL.
    _next.split_report = report
    return _next

def aggregate_w8a8_seed_rows(ckpt_path: str, per_seed: list) -> dict:
    """Mean +/- Std of one checkpoint's W8A8 metrics over independent probes.

    Each seed is one independent probe draw (a fresh generator seeded with
    that seed). `ppl_fp`, `ppl_w8a8`, `ppl_drop`, `ppl_drop_pct` and the probe
    coverage are averaged; each gets a `<key>_std` sibling. The per-seed
    records stay in `per_seed`, and `n_seeds` records how many independent
    draws the mean rests on.
    """
    if not per_seed:
        raise ValueError('aggregate_w8a8_seed_rows needs at least one seed record')
    agg = dict(per_seed[0])
    agg.pop('seed', None)
    agg['ckpt'] = ckpt_path
    agg['seeds'] = [r['seed'] for r in per_seed]
    agg['n_seeds'] = len(per_seed)
    for key in _W8A8_SEED_METRIC_KEYS:
        # Keep the numeric draws (including non-finite ones) for the census;
        # `mean_std_stderr` filters non-finite values itself. Filtering them
        # out here first made `_n_nonfinite` and `nonfinite_metric_counts`
        # always zero, silently hiding the diverged probes the aggregate is
        # supposed to expose.
        raw = [r.get(key) for r in per_seed
               if isinstance(r.get(key), (int, float))
               and not isinstance(r.get(key), bool)]
        stats = mean_std_stderr(raw)
        if stats is None:
            # Every probe draw was non-finite: report "unmeasured", never keep
            # the first seed's NaN as if it were the aggregate. A probe that
            # overflowed is a numerical failure to triage, not a PPL result.
            agg[key] = None
            agg[key + '_std'] = None
            agg[key + '_n_finite'] = 0
            agg[key + '_n_nonfinite'] = len(raw)
            if raw:
                agg.setdefault('nonfinite_metric_counts', {})[key] = len(raw)
            continue
        agg[key] = stats['mean']
        agg[key + '_std'] = stats['std'] if stats['n'] >= 2 else None
        agg[key + '_n_finite'] = stats['n']
        agg[key + '_n_nonfinite'] = stats['n_nonfinite']
        if stats['n_nonfinite']:
            agg.setdefault('nonfinite_metric_counts', {})[key] = stats['n_nonfinite']
    agg['per_seed'] = per_seed
    return agg

def _finite_metric(value) -> bool:
    """Whether a probe metric may enter a paired test.

    A diverged probe can leave NaN/Inf in the per-seed record. The aggregate
    reports those draws as excluded, but the paired tests must exclude them
    too: a single NaN makes the t statistic NaN and aborts the whole
    comparison instead of testing the finite draws.
    """
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))

def _seed_id(value):
    """A seed key as an int, or None for a boolean/fractional/foreign key.

    JSON turns an integer mapping key into its string form, so ``"42"`` is the
    same seed as ``42``; ``"42.0"``, ``"042"`` and booleans are not.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        try:
            converted = int(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if str(converted) == value:
            return converted
    return None

def _seed_metric_values(agg: dict, key: str) -> list:
    return [float(r[key]) for r in agg.get('per_seed', [])
            if _finite_metric(r.get(key))]

def _train_seed_metric_values(agg: dict, key: str) -> dict:
    """Per-training-seed mean of `key` (averaged over that seed's checkpoints)."""
    out = {}
    for train_seed, metrics in (agg.get('per_train_seed') or {}).items():
        seed = _seed_id(train_seed)
        value = metrics.get(key) if isinstance(metrics, dict) else None
        if seed is not None and _finite_metric(value):
            out[seed] = float(value)
    return out

def _shared_seed_samples(agg_a: dict, agg_b: dict, key: str) -> tuple[list, list, list]:
    """Per-seed values of `key` for the seeds both runs measured, paired by seed.

    The probe batches of a given seed are identical across checkpoints (one
    generator seeded with that seed), so the two arms' records are paired
    samples - the same reason the ablation runner pairs its rows by seed. A
    seed with a non-finite metric on either side is not a usable pair and is
    excluded rather than poisoning the paired t statistic. A malformed seed
    field is unknown and is skipped, never coerced.
    """
    a_by_seed = {_seed_id(r.get('seed')): r.get(key)
                 for r in agg_a.get('per_seed', [])
                 if _seed_id(r.get('seed')) is not None}
    b_by_seed = {_seed_id(r.get('seed')): r.get(key)
                 for r in agg_b.get('per_seed', [])
                 if _seed_id(r.get('seed')) is not None}
    shared = sorted(s for s in set(a_by_seed) & set(b_by_seed)
                    if _finite_metric(a_by_seed[s])
                    and _finite_metric(b_by_seed[s]))
    return shared, [float(a_by_seed[s]) for s in shared], [float(b_by_seed[s]) for s in shared]

_TREATMENT_KEYS = (
    'norm', 'qk_norm', 'qk_alpha_init', 'alpha_0', 'k_init', 'alpha_init',
    'p_init', 'learnable_p', 'learnable_k', 'learnable_delta', 'beta',
    'use_layerscale', 'layerscale_init', 'alpha_jitter',
)


def _treatment_record(ckpt_args, model) -> dict:
    """The normalization treatment recorded by one checkpoint.

    This is deliberately separate from `_protocol_record`: the treatment is
    the one thing that *should* differ between comparison arms, but every
    checkpoint inside one arm must share it. A comma-separated `--ckpts` arm
    that mixes norms would otherwise average different treatments and then
    label the arm from the last checkpoint.
    """
    args = ckpt_args if isinstance(ckpt_args, dict) else {}
    cfg = getattr(model, 'cfg', None)

    def recorded(name, cfg_name=None, default=None):
        if name in args:
            return args[name]
        return getattr(cfg, cfg_name or name, default)

    def negated(name, cfg_name, default):
        if name in args:
            return not bool(args[name])
        return getattr(cfg, cfg_name, default)

    return {
        'norm': recorded('norm', 'norm_name'),
        'qk_norm': recorded('qk_norm'),
        'qk_alpha_init': recorded('qk_alpha_init'),
        'alpha_0': recorded('alpha_0'),
        'k_init': recorded('k_init'),
        'alpha_init': recorded('alpha_init'),
        'p_init': recorded('p_init'),
        'learnable_p': negated('no_learnable_p', 'learnable_p', True),
        'learnable_k': negated('no_learnable_k', 'learnable_k', True),
        'learnable_delta': negated('no_learnable_delta', 'learnable_delta', True),
        'beta': recorded('beta', default=True),
        'use_layerscale': negated('no_layerscale', 'use_layerscale', True),
        'layerscale_init': recorded('layerscale_init'),
        'alpha_jitter': recorded('alpha_jitter'),
    }


def _treatment_mismatches(records) -> dict:
    """Per-key differing treatment values inside one checkpoint group."""
    values = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for key in _TREATMENT_KEYS:
            values.setdefault(key, set()).add(repr(record.get(key)))
    return {key: sorted(seen) for key, seen in values.items() if len(seen) > 1}


def _protocol_record(ckpt_args: dict, model) -> dict:
    """Training recipe and shape recorded by one checkpoint."""
    return {
        # Keep absent keys absent: `training_recipe_mismatches` treats a
        # missing key as unknown, but materializing it as None made two
        # unrecorded recipes compare equal.
        'hyperparameters': {key: ckpt_args[key]
                            for key in TRAINING_RECIPE_KEYS
                            if key in ckpt_args},
        # The requested dataset/tokenizer revision may be None/branch while
        # the loader actually resolved a concrete commit. Each checkpoint's
        # probe is built from its own resolved revision, so the protocol guard
        # must compare these too: otherwise probes from different corpora are
        # paired as if they were identical.
        'resolved_revisions': {
            'dataset_revision_resolved': ckpt_args.get('dataset_revision_resolved'),
            'tokenizer_revision_resolved': ckpt_args.get('tokenizer_revision_resolved'),
        },
        'shape': {'model': ckpt_args.get('model'),
                  'n_layers': int(getattr(model.cfg, 'n_layers', 0)),
                  'dim': int(getattr(model.cfg, 'dim', 0)),
                  'n_heads': int(getattr(model.cfg, 'n_heads', 0))},
        'checkpoint': {'step': ckpt_args.get('_checkpoint_step')},
    }

def _protocol_pair_mismatches(a: dict, b: dict) -> dict:
    """Differences in the training recipe or model shape of two checkpoints."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {'protocol': ['<missing>', '<missing>']}
    out = training_recipe_mismatches(a.get('hyperparameters'), b.get('hyperparameters'))
    revisions_a = a.get('resolved_revisions') or {}
    revisions_b = b.get('resolved_revisions') or {}
    for key in ('dataset_revision_resolved', 'tokenizer_revision_resolved'):
        value_a = revisions_a.get(key, a.get(key))
        value_b = revisions_b.get(key, b.get(key))
        # Two unknown revisions are not two equal revisions. Require concrete
        # pinned commits so a pair probed after the mutable default branch
        # moved (or while the Hub was unreachable) cannot certify as one
        # data/tokenizer snapshot.
        if (not _is_pinned_revision(value_a)
                or not _is_pinned_revision(value_b)
                or value_a != value_b):
            out[f'data.{key}'] = [value_a, value_b]
    shape_a = a.get('shape') or {}
    shape_b = b.get('shape') or {}
    for key in ('model', 'n_layers', 'dim', 'n_heads'):
        value_a = shape_a.get(key)
        value_b = shape_b.get(key)
        if key == 'model':
            valid_a = isinstance(value_a, str) and bool(value_a.strip())
            valid_b = isinstance(value_b, str) and bool(value_b.strip())
        else:
            valid_a = (isinstance(value_a, int) and not isinstance(value_a, bool)
                       and value_a > 0)
            valid_b = (isinstance(value_b, int) and not isinstance(value_b, bool)
                       and value_b > 0)
        # Unknown is not equal: two arms that both omit a shape field have
        # not proven that they share an architecture.
        if not valid_a or not valid_b or value_a != value_b:
            out[f'shape.{key}'] = [value_a, value_b]
    step_a = (a.get('checkpoint') or {}).get('step')
    step_b = (b.get('checkpoint') or {}).get('step')
    valid_step_a = (isinstance(step_a, int) and not isinstance(step_a, bool)
                    and step_a >= 0)
    valid_step_b = (isinstance(step_b, int) and not isinstance(step_b, bool)
                    and step_b >= 0)
    if not valid_step_a or not valid_step_b or step_a != step_b:
        out['checkpoint.step'] = [step_a, step_b]
    return out

def _result_scale(result: dict, index: int) -> str:
    """Architecture label for a W8A8 arm from its recorded checkpoint shape."""
    protocol = result.get('protocol') if isinstance(result, dict) else None
    shape = protocol.get('shape') if isinstance(protocol, dict) else None
    if isinstance(shape, dict) and shape:
        return json.dumps(shape, sort_keys=True)
    return 'unknown'

def compare_w8a8_checkpoints(results: list, metric: str='ppl_drop_pct', alpha: float=0.05) -> dict:
    """Paired, Holm-adjusted significance of one checkpoint's PPL drop vs another's.

    The test family is the one-sided paired Student-t `H1: mean(A - B) < 0` on
    `ppl_drop_pct`, so a negative `mean_delta` means A loses less quality to
    W8A8; pairs with fewer than 2 shared samples are reported without a
    p-value rather than silently tested unpaired. p-values are Holm-adjusted
    over the whole family of pairwise tests, so "the best of many pairs" is not
    read as significant by chance. `main` writes this next to the results as
    `ppl_comparison.json`, mirroring the ablation runner's
    `leave_one_out_comparison.json`.

    Every arm is a trained checkpoint, so the run-to-run unit is the *training*
    seed: a pairwise test is certifiable (`basis='train_seed_means'`) only when
    both arms supply >= 3 checkpoints with shared training seeds (comma-separated
    `--ckpts` entries for the same arm); two shared seeds are descriptive only
    (df = 1). A single trained checkpoint per arm
    leaves just probe noise in the denominator: the probe-paired test is still
    computed, but labeled `basis='probe_paired_single_train_run'`,
    `certifiable=False`, and `significant` stays False. The per-arm degradation
    test (`H1: mean(ppl_drop_pct) > 0`) is likewise certifiable only across
    >= 3 training seeds; with fewer it is a descriptive probe-only test, so a
    0.2% drop under a 0.5% seed spread is never sold as a real degradation.
    """
    labels = [str(r.get('ckpt') or f'run{i}') for i, r in enumerate(results)]
    # `ckpt`/label includes the per-scale path, so it cannot identify the same
    # arm pair at another model scale. Keep a scale-independent arm id for the
    # cross-scale replication buckets (falling back to the display label when
    # a caller supplies no arm metadata).
    arms = [str(r.get('arm') or r.get('norm') or labels[i])
            for i, r in enumerate(results)]
    scales = [_result_scale(r, i) for i, r in enumerate(results)]
    tests = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a_train = _train_seed_metric_values(results[i], metric)
            b_train = _train_seed_metric_values(results[j], metric)
            shared_train = sorted(set(a_train) & set(b_train))
            shared, av, bv = _shared_seed_samples(results[i], results[j], metric)
            t = {'a': labels[i], 'b': labels[j],
                 'a_arm': arms[i], 'b_arm': arms[j], 'metric': metric,
                 'a_scale': scales[i], 'b_scale': scales[j],
                 'same_scale': scales[i] == scales[j],
                 'direction': f'negative mean_delta means A has lower {metric} (more W8A8-robust)',
                 'a_train_seeds': sorted(a_train), 'b_train_seeds': sorted(b_train),
                 'n_shared_train_seeds': len(shared_train), 'shared_train_seeds': shared_train,
                 'n_shared_seeds': len(shared), 'shared_seeds': shared}
            arm_problems = []
            for idx, arm in ((i, results[i]), (j, results[j])):
                if arm.get('protocol_consistent') is False:
                    arm_problems.append(f'{labels[idx]} mixes training recipes across its checkpoints')
            if arm_problems:
                t.update({'basis': 'protocol_mismatch', 'certifiable': False,
                          'mean_delta': None, 'p_one_sided': None,
                          'reason': '; '.join(arm_problems)})
                tests.append(t)
                continue
            protocol_mismatches = _protocol_pair_mismatches(
                results[i].get('protocol'), results[j].get('protocol'))
            if protocol_mismatches:
                only_shape = all(key.startswith('shape.') for key in protocol_mismatches)
                t.update({'basis': ('cross_scale' if only_shape else 'protocol_mismatch'),
                          'certifiable': False,
                          'mean_delta': None, 'p_one_sided': None,
                          'protocol_mismatches': protocol_mismatches,
                          'reason': (('the arms are different model shapes; a cross-scale delta '
                                      'is not a same-scale norm/quantization effect (add the '
                                      'same arm pair at each scale and use the '
                                      'scale_replication block)') if only_shape else
                                     ('training recipe/model shape differs between arms, so the '
                                      'delta is not attributable to the norm/quantization: '
                                      + json.dumps(protocol_mismatches, sort_keys=True)))})
                tests.append(t)
                continue
            if len(shared_train) >= 2:
                t['basis'] = 'train_seed_means'
                t['certifiable'] = len(shared_train) >= MIN_SEEDS_FOR_SIGNIFICANCE
                t.update(paired_t_one_sided([a_train[s] for s in shared_train],
                                            [b_train[s] for s in shared_train], alpha=alpha))
            else:
                t['basis'] = 'probe_paired_single_train_run'
                t['certifiable'] = False
                if len(shared) < 2:
                    t['mean_delta'] = (sum(av) - sum(bv)) / len(shared) if shared else None
                    t['p_one_sided'] = None
                    t['reason'] = 'need >= 2 shared samples to pair a test'
                else:
                    t.update(paired_t_one_sided(av, bv, alpha=alpha))
                # Append, never replace: when there are too few shared probe
                # samples *and* one training run per arm, both facts matter.
                t['reason'] = '; '.join(filter(None, (
                    t.get('reason'),
                    f'only {len(shared_train)} shared training seed(s): fewer than the required '
                    '3 cannot certify the arm against run-to-run noise, so the test is '
                    'descriptive only; pass comma-separated checkpoints with distinct training '
                    'seeds per arm to certify')))
            tests.append(t)
    for t, p_adj in zip(tests, holm_adjust([t.get('p_one_sided') for t in tests])):
        t['p_holm'] = p_adj
        t['significant'] = bool(t.get('certifiable') and p_adj is not None and p_adj < alpha)
    scale_obs = []
    for t in tests:
        if t.get('same_scale') and t.get('mean_delta') is not None:
            # A Holm p-value can be explicitly None for a row excluded from
            # its family; `dict.get(key, fallback)` only falls back on a
            # missing key, so choose the raw p explicitly.
            p_holm = t.get('p_holm')
            scale_obs.append({'scale': t.get('a_scale'), 'metric': t.get('metric'),
                              'contrast': (f"{t.get('a_arm')} vs {t.get('b_arm')}"),
                              'delta': t.get('mean_delta'),
                              'certified': t.get('significant') is True,
                              'p_one_sided': (p_holm if p_holm is not None
                                              else t.get('p_one_sided')),
                              'n_seeds': t.get('n_shared_train_seeds')
                              or t.get('n_shared_seeds'),
                              # Forward the paired test's interval so the
                              # multi-scale report can weight each scale by
                              # its precision, not count signs.
                              'ci_low': t.get('ci_low'),
                              'ci_high': t.get('ci_high'),
                              'df': t.get('df'),
                              'ci_level': t.get('ci_level')})
    scale_replication = scale_replication_summary(scale_obs)
    degradation = []
    for label, r in zip(labels, results):
        vals = _seed_metric_values(r, metric)
        stats = mean_std_stderr(vals)
        train_vals = list(_train_seed_metric_values(r, metric).values())
        train_stats = mean_std_stderr(train_vals)
        probe_mean = stats['mean'] if stats else None
        train_mean = train_stats['mean'] if train_stats else None
        # The headline number must be the sample the p-value/CI is computed
        # from: with >= 2 training seeds that is the training-seed mean, not
        # the probe-seed mean (which weights each checkpoint equally).
        headline_mean = (train_mean if len(train_vals) >= 2 and train_mean is not None
                         else probe_mean)
        d = {'ckpt': label, 'metric': metric, 'n_seeds': len(vals),
             'n_train_seeds': len(train_vals),
             'mean_drop_pct': headline_mean,
             'mean_drop_pct_across_probe_seeds': probe_mean,
             'mean_drop_pct_across_train_seeds': train_mean,
             'direction': f'H1: mean({metric}) > 0 (quantization measurably hurts)'}
        if r.get('protocol_consistent') is False:
            d.update({'basis': 'protocol_mismatch', 'certifiable': False,
                      'p_one_sided': None,
                      'reason': ('this arm mixes training recipes/model shapes across its '
                                 'checkpoints, so its pooled degradation is not one experiment')})
        elif len(train_vals) >= 2:
            d['basis'] = 'train_seed_means'
            d['certifiable'] = len(train_vals) >= MIN_SEEDS_FOR_SIGNIFICANCE
            # paired_t_one_sided(zeros, vals) tests mean(0 - vals) < 0, i.e.
            # mean(vals) > 0; the interval is for -mean(vals), so flip it back.
            t = paired_t_one_sided([0.0] * len(train_vals), train_vals, alpha=alpha)
            d.update({'p_one_sided': t['p_one_sided'], 'ci_low': -t['ci_high'],
                      'ci_high': -t['ci_low']})
        else:
            d['basis'] = 'probe_only_single_train_run'
            d['certifiable'] = False
            if len(vals) >= 2:
                t = paired_t_one_sided([0.0] * len(vals), vals, alpha=alpha)
                d.update({'p_one_sided': t['p_one_sided'], 'ci_low': -t['ci_high'],
                          'ci_high': -t['ci_low'],
                          'reason': (f'only {len(train_vals)} training seed(s): the drop is measured on one '
                                     'trained run, so the probe-paired p-value is descriptive only')})
            else:
                d.update({'p_one_sided': None, 'reason': 'need >= 2 probe seeds'})
        degradation.append(d)
    for d, p_adj in zip(degradation, holm_adjust([d.get('p_one_sided') for d in degradation])):
        d['p_holm'] = p_adj
        d['significant'] = bool(d.get('certifiable') and p_adj is not None and p_adj < alpha)
    ranking = []
    for label, r in zip(labels, results):
        vals = _seed_metric_values(r, metric)
        stats = mean_std_stderr(vals)
        train_vals = list(_train_seed_metric_values(r, metric).values())
        train_stats = mean_std_stderr(train_vals)
        ranking.append({'ckpt': label, 'n_seeds': len(vals), 'n_train_seeds': len(train_vals),
                        'mean': stats['mean'] if stats else None,
                        'std': stats['std'] if stats and stats['n'] >= 2 else None,
                        'mean_across_train_seeds': train_stats['mean'] if train_stats else None,
                        'std_across_train_seeds': (train_stats['std']
                                                   if train_stats and train_stats['n'] >= 2 else None)})
    ranking.sort(key=lambda row: (row['mean'] is None, row['mean']))
    return {'metric': metric, 'alpha': alpha,
            'scale_replication': scale_replication,
            'pairing': ('probe batches are paired across arms by probe seed; certification additionally '
                        'requires >= 3 shared training seeds per arm and otherwise stays descriptive'),
            'ranking': ranking, 'pairwise_tests': tests, 'degradation_tests': degradation}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True, help='one entry per arm; a plain path is one trained checkpoint. An entry may be a comma-separated list of checkpoints (ideally distinct training seeds of the same arm), which are all probed and averaged per probe seed; cross-arm certification in ppl_comparison.json pairs training seeds, so >= 3 shared training seeds across the compared arms are required to certify a winner')
    p.add_argument('--probe', default='markov', choices=['markov', 'dataset', 'random'], help="probe corpus. 'random' is uniform tokens: it has no next-token signal, so ppl_fp ~= vocab_size and the PPL drop is unmeasurable (kept only for parity)")
    p.add_argument('--probe-split', default='validation', choices=['validation', 'test', 'train'], help="held-out split used by --probe dataset. Keep the default: 'test' is sealed for one final evaluation and 'train' reports an optimistically biased PPL")
    p.add_argument('--allow-sealed-test', action='store_true', help="permit the single final check on the sealed test split; comparing checkpoints or quantization settings on 'test' would turn it into an implicit validation set")
    p.add_argument('--val-frac', type=float, default=0.005, help='chronological validation fraction (must match the trained run)')
    p.add_argument('--test-frac', type=float, default=0.005, help='chronological test fraction (must match the trained run)')
    p.add_argument('--num-documents', type=int, default=None, help='total documents in the stream; required when the dataset reports no size')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--dataset-revision', default=None, help='dataset revision (commit hash) for --probe dataset; a checkpoint that recorded the one it was trained with wins (None = upstream default branch)')
    p.add_argument('--tokenizer', default='gpt2', help='tokenizer for --probe dataset; a checkpoint that recorded the one it was trained with wins')
    p.add_argument('--tokenizer-revision', default=None, help='tokenizer revision (commit hash) for --probe dataset; same precedence as --tokenizer')
    p.add_argument('--n-states', type=int, default=64, help="state count of the synthetic 'markov' probe")
    p.add_argument('--max-probe-coverage', type=float, default=0.5, help='reject a probe whose tokens are more than this fraction distinct (uniform noise is ~1.0)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--n-batches', type=int, default=20)
    p.add_argument('--num-workers', type=int, default=2, help='DataLoader workers for --probe dataset; recorded in the manifest')
    p.add_argument('--seed', type=int, default=None, help='deprecated single-probe shorthand; overrides --seeds when given')
    p.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS), help='one independent probe draw per seed; every metric is reported as Mean +/- Std across seeds, and >= 3 seeds keeps a single lucky probe from deciding a comparison')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the paired cross-checkpoint tests in ppl_comparison.json (Holm-adjusted over the pairwise family)')
    p.add_argument('--output', default='runs/w8a8/results.json')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this check tests (e.g. "DyT-CD loses less perplexity under W8A8 than RMSNorm at the same checkpoint step count"); recorded in the run manifest so the result can be audited against the claim it was meant to support')
    args = p.parse_args()
    if args.probe == 'dataset' and args.probe_split == 'test' and (not args.allow_sealed_test):
        p.error("--probe-split test is sealed: comparing checkpoints or quantization "
                "settings on it would tune on the test set. Use 'validation' (default), "
                "or pass --allow-sealed-test for the single final check.")
    raw_seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if not raw_seeds:
        p.error('--seeds must not be empty')
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[w8a8][warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed reruns the identical probe, not an independent sample)')
    if len(seeds) < 3:
        print(f'[w8a8][warn] {len(seeds)} seed(s): one probe draw is one sample; use >= 3 independent probes before treating a PPL-drop difference as real')
    for flag, value in (('--batch-size', args.batch_size),
                        ('--seq-len', args.seq_len),
                        ('--n-batches', args.n_batches)):
        if value < 1:
            p.error(f'{flag} {value} must be >= 1')
    if args.num_workers < 0:
        p.error(f'--num-workers {args.num_workers} must be >= 0')
    if args.n_states < 2:
        p.error(f'--n-states {args.n_states} must be >= 2: a one-state probe has a '
                'deterministic successor and measures no next-token structure')
    if (not math.isfinite(args.max_probe_coverage)
            or not 0.0 < args.max_probe_coverage <= 1.0):
        p.error(f'--max-probe-coverage {args.max_probe_coverage} must be finite and in (0, 1]')
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        p.error(f'--alpha {args.alpha} must be finite and strictly between 0 and 1')
    for name, value in (('--val-frac', args.val_frac), ('--test-frac', args.test_frac)):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            p.error(f'{name} {value} must be finite and in [0, 1)')
    if args.val_frac + args.test_frac >= 1.0:
        p.error(f'--val-frac {args.val_frac} + --test-frac {args.test_frac} leaves no '
                'training stream; the split fractions must sum to less than 1')
    if args.num_documents is not None and args.num_documents < 1:
        p.error(f'--num-documents {args.num_documents} must be >= 1')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seeds[0])

    def make_probe(rng, cfg, dataset_probe):
        if args.probe == 'dataset':
            return dataset_probe()
        if args.probe == 'random':
            x = _probe_random(rng, cfg.vocab_size, args.batch_size, args.seq_len, device)
        else:
            if args.n_states > int(cfg.vocab_size):
                raise SystemExit(
                    f'[w8a8] --n-states {args.n_states} exceeds the model vocabulary '
                    f'({int(cfg.vocab_size)}); the Markov probe would emit out-of-range ids')
            x = _probe_markov(rng, args.batch_size, args.seq_len, device, n_states=args.n_states)
        return x, _next_token_targets(x, cfg.ignore_index)
    explicit_cli = _explicit_cli_names(sys.argv[1:], p)
    ckpt_groups = [[path for path in str(entry).split(',') if path] for entry in args.ckpts]
    if any(not group for group in ckpt_groups):
        p.error('--ckpts entries must not be empty')
    for group in ckpt_groups:
        if len(set(group)) != len(group):
            p.error(f'--ckpts arm {group} repeats a checkpoint; a repeated path is the '
                    'same trained run and would be weighted twice instead of adding evidence')
    normalized_groups = [tuple(sorted(group)) for group in ckpt_groups]
    if len(set(normalized_groups)) != len(normalized_groups):
        p.error('--ckpts contains the same checkpoint set as two arms; a self-comparison '
                'has no treatment difference to certify')
    # Partial overlap is just as invalid as an identical set: the shared
    # checkpoint is the same trained run on both sides, so its paired delta is
    # forced to zero and the treatment contrast is contaminated. Compare
    # resolved paths so ./a and a are not treated as different runs.
    seen_paths: dict[str, int] = {}
    for group_index, group in enumerate(ckpt_groups):
        for path in group:
            key = str(Path(path).resolve())
            previous = seen_paths.get(key)
            if previous is not None:
                where = ('within one arm' if previous == group_index
                         else f'in arms {previous + 1} and {group_index + 1}')
                p.error(f'--ckpts checkpoint {path} appears more than once ({where}); '
                        'each training run may contribute to exactly one arm')
            seen_paths[key] = group_index
    results = []
    dataset_probe_reports = []
    for group in ckpt_groups:
        label = group[0] if len(group) == 1 else f'{group[0]} (+{len(group) - 1} more)'
        per_ckpt = []
        for ckpt_path in group:
            model, ckpt_args = load_ckpt(ckpt_path, device=device)
            # Prefer the dataset and split the checkpoint was trained with, so
            # the probe is the same held-out slice; a flag typed on the command
            # line wins.
            def split_arg(name):
                return _checkpoint_defaulted(args, p, ckpt_args, name, explicit_cli)
            # Same precedence for the data version: an explicit flag wins, else
            # the revision the checkpoint recorded, else upstream default branch.
            dataset_revision = (
                args.dataset_revision if args.dataset_revision is not None
                else ckpt_args.get('dataset_revision_resolved')
                or ckpt_args.get('dataset_revision'))
            tokenizer_revision = (
                args.tokenizer_revision if args.tokenizer_revision is not None
                else ckpt_args.get('tokenizer_revision_resolved')
                or ckpt_args.get('tokenizer_revision'))
            dataset_probe = make_dataset_probe(split_arg('dataset'), split_arg('dataset_config'), ckpt_args.get('tokenizer') or args.tokenizer, args.batch_size, args.seq_len, split=args.probe_split, val_frac=split_arg('val_frac'), test_frac=split_arg('test_frac'), num_documents=split_arg('num_documents'), dedup=ckpt_args.get('dedup', True), min_doc_chars=ckpt_args.get('min_doc_chars', 0), max_doc_chars=ckpt_args.get('max_doc_chars', 0), exclude_hash_file=ckpt_args.get('exclude_hash_file'), exclude_test_hash_file=ckpt_args.get('exclude_test_hash_file'), group_key=ckpt_args.get('group_key'), exclude_group_file=ckpt_args.get('exclude_group_file'), exclude_test_group_file=ckpt_args.get('exclude_test_group_file'), dataset_revision=dataset_revision, tokenizer_revision=tokenizer_revision, seed=seeds[0], num_workers=args.num_workers) if args.probe == 'dataset' else None
            if dataset_probe is not None:
                dataset_probe_reports.append({
                    'ckpt': ckpt_path,
                    'dataset_revision': dataset_revision,
                    'tokenizer_revision': tokenizer_revision,
                    'split_report': getattr(dataset_probe, 'split_report', None),
                })
            raw_seed = ckpt_args.get('seed')
            train_seed = _seed_id(raw_seed)
            # Snapshot the fp32 weights on the host before the in-place
            # quantizer runs; each seed needs them back.
            fp_state = {k: t.detach().to('cpu', copy=True) for k, t in model.state_dict().items()}
            dataset_probe_batches = None
            if dataset_probe is not None:
                dataset_probe_batches = [
                    dataset_probe() for _ in range(args.n_batches * len(seeds))]
            per_seed = []
            for seed_index, seed in enumerate(seeds):
                _unwrap_w8a8(model)
                model.load_state_dict(fp_state)
                model.eval()
                rng = torch.Generator(device=device).manual_seed(seed)
                if dataset_probe_batches is not None:
                    start = seed_index * args.n_batches
                    batches = list(dataset_probe_batches[start:start + args.n_batches])
                else:
                    batches = [make_probe(rng, model.cfg, dataset_probe)
                               for _ in range(args.n_batches)]
                # The dataset probe's loader already supplies the packed
                # window's real target (including the last valid position);
                # synthesizing it here would silently drop that token from
                # every batch. Keep the pair for both the fp32 and W8A8 pass.
                pairs = batches
                coverage = probe_token_coverage([x for x, _ in pairs])
                if coverage > args.max_probe_coverage:
                    raise ValueError(f"probe='{args.probe}' carries no next-token signal: {coverage:.0%} of its tokens are distinct, so nothing repeats and no model can beat chance. Use --probe markov|dataset (--max-probe-coverage overrides).")
                ce_fp = torch.zeros((), dtype=torch.float64, device=device)
                n_tok = torch.zeros((), dtype=torch.long, device=device)
                for x, y in pairs:
                    a, n = compute_ce_sum(model, x, y, device=device)
                    ce_fp += a
                    n_tok += n
                n_tok_val = int(n_tok.item())
                if n_tok_val == 0:
                    raise ValueError('probe batch contains no valid target tokens')
                wrap_model_w8a8(model)
                model.eval()
                ce_q = torch.zeros((), dtype=torch.float64, device=device)
                for x, y in pairs:
                    b, _ = compute_ce_sum(model, x, y, device=device)
                    ce_q += b
                ppl_fp = _exp_or_inf(float(ce_fp) / n_tok_val)
                ppl_q = _exp_or_inf(float(ce_q) / n_tok_val)
                per_seed.append({'ckpt': ckpt_path, 'probe': args.probe, 'probe_split': args.probe_split if args.probe == 'dataset' else None, 'seed': seed, 'ppl_fp': ppl_fp, 'ppl_w8a8': ppl_q, 'ppl_drop': ppl_q - ppl_fp, 'ppl_drop_pct': 100.0 * (ppl_q - ppl_fp) / ppl_fp, 'probe_token_coverage': coverage})
            r_ckpt = aggregate_w8a8_seed_rows(ckpt_path, per_seed)
            if args.probe == 'dataset' and args.probe_split != 'validation':
                print(f"[w8a8] WARNING: --probe dataset scored on the '{args.probe_split}' split; use 'validation' for model comparison ('test' is reserved for the single final evaluation, 'train' is optimistically biased)", file=sys.stderr)
            nonfinite_counts = r_ckpt.get('nonfinite_metric_counts') or {}
            if nonfinite_counts:
                print(f"[w8a8] WARNING: {ckpt_path}: non-finite probe metric(s) "
                      f"excluded from the reported means: {nonfinite_counts} "
                      f"(per-seed records kept); triage the overflow before quoting "
                      f"the PPL drop", file=sys.stderr)
            if r_ckpt.get('ppl_fp') is not None and r_ckpt['ppl_fp'] >= 0.5 * model.cfg.vocab_size:
                print(f"[w8a8] WARNING: {ckpt_path} ppl_fp={r_ckpt['ppl_fp']:.1f} (mean over {r_ckpt['n_seeds']} seeds) is at the vocab_size={model.cfg.vocab_size} ceiling - the probe is fine (coverage={r_ckpt['probe_token_coverage']:.1%}) but this checkpoint appears untrained, so ppl_drop_pct is not meaningful yet", file=sys.stderr)
            per_ckpt.append({'path': ckpt_path, 'train_seed': train_seed,
                             'protocol': _protocol_record(ckpt_args, model),
                             'treatment': _treatment_record(ckpt_args, model),
                             'agg': {k: v for k, v in r_ckpt.items() if k != 'per_seed'},
                             'per_seed': per_seed})
        # A comma-separated group is one arm only when every checkpoint in it
        # is the same treatment; averaging different norms would manufacture a
        # per-seed mean for a treatment that was never run.
        treatment_mismatches = _treatment_mismatches(
            c.get('treatment') for c in per_ckpt)
        if treatment_mismatches:
            p.error(
                f'--ckpts group {label!r} mixes treatment values '
                f'{json.dumps(treatment_mismatches, sort_keys=True)}; one '
                'checkpoint group must contain a single normalization treatment')
        # A multi-checkpoint arm is one treatment only if its checkpoints
        # share the training recipe and model shape; record any disagreement
        # instead of averaging it away.
        base_protocol = per_ckpt[0].get('protocol') if per_ckpt else None
        arm_protocol_mismatches = []
        for c in per_ckpt[1:]:
            mismatches = _protocol_pair_mismatches(base_protocol, c.get('protocol'))
            if mismatches:
                arm_protocol_mismatches.append({'ckpt': c.get('path'),
                                                'mismatches': mismatches})
        # Arm-level probe records: each probe seed keeps one row whose metrics
        # are the mean over this arm's checkpoints (the probe seed is the
        # shared draw; checkpoints replicate the training run).
        per_seed = []
        for j, seed in enumerate(seeds):
            record = dict(per_ckpt[0]['per_seed'][j])
            for key in _W8A8_SEED_METRIC_KEYS:
                vals = [c['per_seed'][j][key] for c in per_ckpt
                        if _finite_metric(c['per_seed'][j].get(key))]
                if vals:
                    record[key] = sum(vals) / len(vals)
            record['seed'] = seed
            record['n_checkpoints'] = len(per_ckpt)
            per_seed.append(record)
        r = aggregate_w8a8_seed_rows(label, per_seed)
        # Scale-independent treatment identity for cross-scale replication;
        # `label` may include the model name to keep within-run arms distinct,
        # but the arm itself is the norm recorded by the checkpoint (the path
        # differs at every scale and cannot identify the same treatment).
        arm_name = (ckpt_args.get('norm') if isinstance(ckpt_args, dict) else None)
        if not isinstance(arm_name, str) or not arm_name.strip():
            arm_name = getattr(getattr(model, 'cfg', None), 'norm_name', None)
        r['arm'] = str(arm_name).strip() if isinstance(arm_name, str) and arm_name.strip() else label
        train_seeds = sorted({c['train_seed'] for c in per_ckpt if c['train_seed'] is not None})
        r['train_seeds'] = train_seeds
        r['n_train_seeds'] = len(train_seeds)
        r['protocol'] = base_protocol
        r['protocol_consistent'] = not arm_protocol_mismatches
        r['protocol_mismatches'] = arm_protocol_mismatches
        r['ckpts'] = list(group)
        r['per_checkpoint'] = per_ckpt
        per_train_seed = {}
        for c in per_ckpt:
            if c['train_seed'] is None:
                continue
            bucket = per_train_seed.setdefault(c['train_seed'], {key: [] for key in _W8A8_SEED_METRIC_KEYS})
            for rec in c['per_seed']:
                for key in _W8A8_SEED_METRIC_KEYS:
                    value = rec.get(key)
                    if _finite_metric(value):
                        bucket[key].append(float(value))
        r['per_train_seed'] = {ts: {key: sum(vals) / len(vals) for key, vals in bucket.items() if vals}
                               for ts, bucket in per_train_seed.items()}
        if arm_protocol_mismatches:
            print(f'[w8a8][warn] {label}: checkpoints disagree on the training recipe/model '
                  f'shape; cross-arm tests for this arm are descriptive only: '
                  f'{json.dumps(arm_protocol_mismatches, sort_keys=True)}', file=sys.stderr)
        if len(group) > 1 and len(train_seeds) < MIN_SEEDS_FOR_SIGNIFICANCE:
            print(f'[w8a8][warn] {label}: {len(group)} checkpoints but {len(train_seeds)} distinct training '
                  'seed(s); cross-arm certification needs >= 3 independent training runs per arm', file=sys.stderr)
        print(f'[w8a8] {label}: {r}')
        results.append(r)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    comparison = compare_w8a8_checkpoints(results, alpha=args.alpha)
    comparison_path = out_path.with_name('ppl_comparison.json')
    comparison_path.write_text(json.dumps(comparison, indent=2))
    scale = comparison.get('scale_replication') or {}
    if scale.get('single_scale_only'):
        print(f"[w8a8][scale] WARNING: this comparison covers {scale.get('n_scales', 0)} "
              f"model scale(s) ({scale.get('scales')}); a certified same-scale result is not a "
              'multi-scale result. Add an arm pair from another checkpoint shape and rerun, '
              'or combine the per-scale comparison files.')
    elif scale.get('multi_scale_certified_metrics'):
        print(f"[w8a8][scale] replicated across scales: "
              f"{scale.get('multi_scale_certified_metrics')} over {scale.get('scales')}")
    else:
        print(f"[w8a8][scale] scales={scale.get('scales')} "
              f"replicated_metrics={scale.get('replicated_metrics')} "
              f"(no metric certified at >= 2 scales yet)")
    for t in comparison['pairwise_tests']:
        if t.get('p_one_sided') is None:
            print(f"[w8a8][compare] {t['a']} vs {t['b']} on {t['metric']}: {t['reason']}")
            continue
        verdict = 'A more robust' if t['significant'] and t['mean_delta'] < 0 else ('B more robust' if t['significant'] else 'inside the noise')
        if not t.get('certifiable'):
            verdict = (f'descriptive only ({t.get("n_shared_train_seeds", 0)} shared '
                       'training seed(s); >= 3 required)')
        print(f"[w8a8][compare] {t['a']} vs {t['b']} on {t['metric']}: delta={t['mean_delta']:+.4f} pts, 95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}], Holm p={t['p_holm']:.4f} (n={t['n_shared_seeds']}, basis={t['basis']}) -> {verdict}")
    for d in comparison['degradation_tests']:
        if d.get('p_one_sided') is None:
            print(f"[w8a8][compare] {d['ckpt']}: degradation not testable ({d['reason']})")
            continue
        verdict = 'real degradation' if d['significant'] else 'inside the noise'
        if not d.get('certifiable'):
            verdict = (f'descriptive only ({d.get("n_train_seeds", 0)} training '
                       'seed(s); >= 3 required)')
        n_used = (d.get('n_train_seeds') if d.get('basis') == 'train_seed_means'
                  else d.get('n_seeds'))
        print(f"[w8a8][compare] {d['ckpt']}: mean drop {d['mean_drop_pct']:.4f}% pts, 95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}], Holm p={d['p_holm']:.4f} (n={n_used}, basis={d['basis']}) -> {verdict}")
    manifest = collect_run_metadata(seed=seeds[0])
    manifest['hyperparameters'] = vars(args)
    manifest['seeds'] = seeds
    manifest['data'] = {
        'probe': args.probe,
        'probe_split': args.probe_split if args.probe == 'dataset' else None,
        'probe_loader_seed': seeds[0],
        'probe_slice_batches': (args.n_batches if args.probe == 'dataset' else None),
        'probe_slice_seeds': (list(seeds) if args.probe == 'dataset' else None),
        'num_workers': args.num_workers,
        'split_reports': dataset_probe_reports,
    }
    if args.hypothesis:
        manifest['hypothesis'] = args.hypothesis
        print(f'[hypothesis] {args.hypothesis}')
    else:
        print('[hypothesis] none stated: pass --hypothesis "..." so this check is auditable as a test of a claim, not a sweep')
    manifest['results_file'] = str(out_path)
    manifest['comparison_file'] = str(comparison_path)
    manifest = finish_metadata(manifest, artifacts={
        'results': file_fingerprint(out_path),
        'comparison': file_fingerprint(comparison_path),
    })
    save_run_manifest(out_path.with_name('run_manifest.json'), manifest)
    print(f'[done] saved to {out_path}')
if __name__ == '__main__':
    main()
