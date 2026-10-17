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
from adasquash.metrics import holm_adjust, mean_std_stderr, paired_t_one_sided
from adasquash.model import MLP
from adasquash.repro import collect_run_metadata, save_run_manifest, set_all_seeds
from scripts.eval_downstream import load_ckpt
DEFAULT_SEEDS = (42, 43, 44)
# One probe draw is one sample: the PPL drop is aggregated as Mean +/- Std
# over the `--seeds` probe replications, with the per-seed records kept so a
# quantization result resting on one lucky probe cannot hide. Checkpoint
# differences are additionally tested for significance in
# `compare_w8a8_checkpoints` (paired across the shared seeds, Holm-adjusted),
# because "0.2% drop under a 0.5% seed spread" is not a result; the comparison
# is saved next to the results as `ppl_comparison.json`.
_W8A8_SEED_METRIC_KEYS = ('ppl_fp', 'ppl_w8a8', 'ppl_drop', 'ppl_drop_pct',
                          'probe_token_coverage')

def quantize_weight_per_channel(w: torch.Tensor, n_bits: int=8):
    mx = w.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-06)
    q_max = 2 ** (n_bits - 1) - 1
    scale = mx / q_max
    q = w / scale
    q.round_()
    q.clamp_(-q_max - 1, q_max)
    q.mul_(scale)
    return (q, scale)

def quantize_activation_per_token(x: torch.Tensor, n_bits: int=8):
    mx = x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-06)
    q_max = 2 ** (n_bits - 1) - 1
    scale = mx / q_max
    q = x / scale
    q.round_()
    q.clamp_(-q_max - 1, q_max)
    q.mul_(scale)
    return q

def _install_quantized_weight(linear: nn.Module, qw: torch.Tensor):
    with torch.no_grad():
        linear.weight.copy_(qw)
    linear._q_weight = qw

def wrap_linear_w8a8(linear: nn.Linear):
    qw, _ = quantize_weight_per_channel(linear.weight.data)
    _install_quantized_weight(linear, qw)

    def quantized_forward(x):
        x_q = quantize_activation_per_token(x)
        return torch.nn.functional.linear(x_q, qw, linear.bias)
    linear.forward = quantized_forward
    return linear

def wrap_swiglu_w8a8(mlp):
    weights = []
    for lin in (mlp.w1, mlp.w2, mlp.w3):
        qw, _ = quantize_weight_per_channel(lin.weight.data)
        _install_quantized_weight(lin, qw)
        weights.append(qw)
    qa = quantize_activation_per_token

    def quantized_forward(x):
        x_q = qa(x)
        a = qa(F.linear(x_q, weights[0]))
        b = qa(F.linear(x_q, weights[2]))
        h = F.silu(a)
        h.mul_(b)
        return mlp.drop(F.linear(qa(h), weights[1]))
    mlp.forward = quantized_forward
    return mlp

def wrap_embedding_w8a8(emb: nn.Embedding):
    qw, _ = quantize_weight_per_channel(emb.weight.data)

    def quantized_forward(idx):
        return F.embedding(idx, qw)
    with torch.no_grad():
        emb.weight.copy_(qw)
    emb._q_weight = qw
    emb.forward = quantized_forward
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
        norm_f.register_forward_hook(lambda m, inp, out: quantize_activation_per_token(out))
    return model

@torch.no_grad()
def compute_ce_sum(model, batch, target, device='cuda'):
    model.eval()
    batch = batch.to(device)
    target = target.to(device)
    n = (target != model.cfg.ignore_index).sum().item()
    if n == 0:
        raise ValueError('probe batch contains no valid target tokens')
    loss = model(batch, target, return_logits=False)[1]
    return (loss.item() * n, n)

def compute_ppl(model, batch, target, device='cuda'):
    ce, n = compute_ce_sum(model, batch, target, device=device)
    return math.exp(ce / n)

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

def make_dataset_probe(dataset, dataset_config, tokenizer, batch_size, seq_len, split='validation', val_frac=0.005, test_frac=0.005, num_documents=None, dedup=True, min_doc_chars=0, max_doc_chars=0, exclude_hash_file=None):
    # Scoring a checkpoint on the documents it trained on understates the
    # quantization damage; the probe must come from a held-out split. The
    # default is the chronological validation slice, and the cleaning policy
    # (dedup / length caps / cross-split duplicate ledger) is taken from the
    # checkpoint so the probe is the same held-out distribution the model was
    # validated on.
    from scripts.train_gpt2 import build_dataloader
    from scripts.data_pipeline import configure_data_options
    configure_data_options(split=split, val_frac=val_frac, test_frac=test_frac,
                           num_documents=num_documents, dedup=dedup,
                           min_doc_chars=min_doc_chars, max_doc_chars=max_doc_chars,
                           exclude_hash_file=exclude_hash_file)
    loader = build_dataloader(dataset, dataset_config, seq_len, batch_size, tokenizer=tokenizer)
    if split != 'train':
        report = getattr(loader, 'split_report', None)
        if report:
            print(f'[w8a8] dataset probe split: {report}', file=sys.stderr)
    it = iter(loader)

    def _next():
        nonlocal it
        try:
            return next(it)[0]
        except StopIteration:
            it = iter(loader)
            return next(it)[0]
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
        vals = [r[key] for r in per_seed if isinstance(r.get(key), (int, float))]
        stats = mean_std_stderr(vals)
        if stats is None:
            continue
        agg[key] = stats['mean']
        agg[key + '_std'] = stats['std'] if stats['n'] >= 2 else None
    agg['per_seed'] = per_seed
    return agg

def _seed_metric_values(agg: dict, key: str) -> list:
    return [float(r[key]) for r in agg.get('per_seed', [])
            if isinstance(r.get(key), (int, float))]

def _shared_seed_samples(agg_a: dict, agg_b: dict, key: str) -> tuple[list, list, list]:
    """Per-seed values of `key` for the seeds both runs measured, paired by seed.

    The probe batches of a given seed are identical across checkpoints (one
    generator seeded with that seed), so the two arms' records are paired
    samples - the same reason the ablation runner pairs its rows by seed.
    """
    a_by_seed = {r.get('seed'): r.get(key) for r in agg_a.get('per_seed', [])}
    b_by_seed = {r.get('seed'): r.get(key) for r in agg_b.get('per_seed', [])}
    shared = sorted(s for s in set(a_by_seed) & set(b_by_seed)
                    if isinstance(a_by_seed[s], (int, float))
                    and isinstance(b_by_seed[s], (int, float)))
    return shared, [float(a_by_seed[s]) for s in shared], [float(b_by_seed[s]) for s in shared]

def compare_w8a8_checkpoints(results: list, metric: str='ppl_drop_pct', alpha: float=0.05) -> dict:
    """Paired, Holm-adjusted significance of one checkpoint's PPL drop vs another's.

    For every unordered checkpoint pair the test is the one-sided paired
    Student-t `H1: mean(A - B) < 0` on the per-seed `ppl_drop_pct` values, so
    a negative `mean_delta` means A loses less quality to W8A8. A pair with
    fewer than 2 shared seeds is reported without a p-value rather than
    silently tested unpaired. p-values are Holm-adjusted over the whole family
    of pairwise tests, so "the best of many pairs" is not read as significant
    by chance. The same report tests per checkpoint whether the degradation is
    real (`H1: mean(ppl_drop_pct) > 0`): a drop of 0.2% under a 0.5% seed
    spread is inside the noise. `main` writes this next to the results as
    `ppl_comparison.json`, mirroring the ablation runner's
    `leave_one_out_comparison.json`.
    """
    labels = [str(r.get('ckpt') or f'run{i}') for i, r in enumerate(results)]
    tests = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            shared, av, bv = _shared_seed_samples(results[i], results[j], metric)
            t = {'a': labels[i], 'b': labels[j], 'metric': metric,
                 'n_shared_seeds': len(shared), 'shared_seeds': shared,
                 'direction': f'negative mean_delta means A has lower {metric} (more W8A8-robust)'}
            if len(shared) < 2:
                t['mean_delta'] = (sum(av) - sum(bv)) / len(shared) if shared else None
                t['p_one_sided'] = None
                t['reason'] = 'need >= 2 seeds measured for both checkpoints to pair a test'
            else:
                t.update(paired_t_one_sided(av, bv, alpha=alpha))
            tests.append(t)
    for t, p_adj in zip(tests, holm_adjust([t.get('p_one_sided') for t in tests])):
        t['p_holm'] = p_adj
        t['significant'] = bool(p_adj is not None and p_adj < alpha)
    degradation = []
    for label, r in zip(labels, results):
        vals = _seed_metric_values(r, metric)
        stats = mean_std_stderr(vals)
        d = {'ckpt': label, 'metric': metric, 'n_seeds': len(vals),
             'mean_drop_pct': stats['mean'] if stats else None,
             'direction': f'H1: mean({metric}) > 0 (quantization measurably hurts)'}
        if len(vals) >= 2:
            # paired_t_one_sided(zeros, vals) tests mean(0 - vals) < 0, i.e.
            # mean(vals) > 0; the interval is for -mean(vals), so flip it back.
            t = paired_t_one_sided([0.0] * len(vals), vals, alpha=alpha)
            d.update({'p_one_sided': t['p_one_sided'], 'ci_low': -t['ci_high'],
                      'ci_high': -t['ci_low']})
        else:
            d.update({'p_one_sided': None, 'reason': 'need >= 2 seeds'})
        degradation.append(d)
    for d, p_adj in zip(degradation, holm_adjust([d.get('p_one_sided') for d in degradation])):
        d['p_holm'] = p_adj
        d['significant'] = bool(p_adj is not None and p_adj < alpha)
    ranking = []
    for label, r in zip(labels, results):
        vals = _seed_metric_values(r, metric)
        stats = mean_std_stderr(vals)
        ranking.append({'ckpt': label, 'n_seeds': len(vals),
                        'mean': stats['mean'] if stats else None,
                        'std': stats['std'] if stats and stats['n'] >= 2 else None})
    ranking.sort(key=lambda row: (row['mean'] is None, row['mean']))
    return {'metric': metric, 'alpha': alpha,
            'pairing': 'per-seed probe batches are identical across checkpoints (same generator seed)',
            'ranking': ranking, 'pairwise_tests': tests, 'degradation_tests': degradation}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--probe', default='markov', choices=['markov', 'dataset', 'random'], help="probe corpus. 'random' is uniform tokens: it has no next-token signal, so ppl_fp ~= vocab_size and the PPL drop is unmeasurable (kept only for parity)")
    p.add_argument('--probe-split', default='validation', choices=['validation', 'test', 'train'], help="held-out split used by --probe dataset. Keep the default: 'test' is sealed for one final evaluation and 'train' reports an optimistically biased PPL")
    p.add_argument('--allow-sealed-test', action='store_true', help="permit the single final check on the sealed test split; comparing checkpoints or quantization settings on 'test' would turn it into an implicit validation set")
    p.add_argument('--val-frac', type=float, default=0.005, help='chronological validation fraction (must match the trained run)')
    p.add_argument('--test-frac', type=float, default=0.005, help='chronological test fraction (must match the trained run)')
    p.add_argument('--num-documents', type=int, default=None, help='total documents in the stream; required when the dataset reports no size')
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--tokenizer', default='gpt2', help='tokenizer for --probe dataset; a checkpoint that recorded the one it was trained with wins')
    p.add_argument('--n-states', type=int, default=64, help="state count of the synthetic 'markov' probe")
    p.add_argument('--max-probe-coverage', type=float, default=0.5, help='reject a probe whose tokens are more than this fraction distinct (uniform noise is ~1.0)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--n-batches', type=int, default=20)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-probe shorthand; overrides --seeds when given')
    p.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS), help='one independent probe draw per seed; every metric is reported as Mean +/- Std across seeds, and >= 3 seeds keeps a single lucky probe from deciding a comparison')
    p.add_argument('--alpha', type=float, default=0.05, help='significance level of the paired cross-checkpoint tests in ppl_comparison.json (Holm-adjusted over the pairwise family)')
    p.add_argument('--output', default='runs/w8a8/results.json')
    args = p.parse_args()
    if args.probe == 'dataset' and args.probe_split == 'test' and (not args.allow_sealed_test):
        p.error("--probe-split test is sealed: comparing checkpoints or quantization "
                "settings on it would tune on the test set. Use 'validation' (default), "
                "or pass --allow-sealed-test for the single final check.")
    seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if not seeds:
        p.error('--seeds must not be empty')
    if len(seeds) < 3:
        print(f'[w8a8][warn] {len(seeds)} seed(s): one probe draw is one sample; use >= 3 independent probes before treating a PPL-drop difference as real')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seeds[0])

    def make_probe(rng, cfg, dataset_probe):
        if args.probe == 'random':
            return _probe_random(rng, cfg.vocab_size, args.batch_size, args.seq_len, device)
        if args.probe == 'markov':
            return _probe_markov(rng, args.batch_size, args.seq_len, device, n_states=args.n_states)
        return dataset_probe()
    results = []
    for ckpt_path in args.ckpts:
        model, ckpt_args = load_ckpt(ckpt_path, device=device)
        # Prefer the split the checkpoint was trained with, so the probe is the
        # same held-out slice; a flag typed on the command line wins.
        def split_arg(name):
            value = getattr(args, name)
            default = p.get_default(name)
            if value == default and name in ckpt_args and ckpt_args.get(name) is not None:
                return ckpt_args[name]
            return value
        dataset_probe = make_dataset_probe(args.dataset, args.dataset_config, ckpt_args.get('tokenizer') or args.tokenizer, args.batch_size, args.seq_len, split=args.probe_split, val_frac=split_arg('val_frac'), test_frac=split_arg('test_frac'), num_documents=split_arg('num_documents'), dedup=ckpt_args.get('dedup', True), min_doc_chars=ckpt_args.get('min_doc_chars', 0), max_doc_chars=ckpt_args.get('max_doc_chars', 0), exclude_hash_file=ckpt_args.get('exclude_hash_file')) if args.probe == 'dataset' else None
        per_seed = []
        for seed in seeds:
            # A fresh copy per seed: `wrap_model_w8a8` quantizes the weights in
            # place, so the next seed's fp32 reference needs its own unmodified
            # model, and each seed is an independent probe draw.
            model, _ = load_ckpt(ckpt_path, device=device)
            rng = torch.Generator(device=device).manual_seed(seed)
            batches = []
            for _ in range(max(1, args.n_batches)):
                x = make_probe(rng, model.cfg, dataset_probe)
                batches.append(x)
            coverage = probe_token_coverage(batches)
            if coverage > args.max_probe_coverage:
                raise ValueError(f"probe='{args.probe}' carries no next-token signal: {coverage:.0%} of its tokens are distinct, so nothing repeats and no model can beat chance. Use --probe markov|dataset (--max-probe-coverage overrides).")
            ce_fp = 0.0
            n_tok = 0
            for x in batches:
                y = torch.full((args.batch_size, args.seq_len), model.cfg.ignore_index, dtype=torch.long, device=device)
                y[:, :-1] = x[:, 1:]
                a, n = compute_ce_sum(model, x, y, device=device)
                ce_fp += a
                n_tok += n
            wrap_model_w8a8(model)
            ce_q = 0.0
            for x in batches:
                y = torch.full((args.batch_size, args.seq_len), model.cfg.ignore_index, dtype=torch.long, device=device)
                y[:, :-1] = x[:, 1:]
                b, _ = compute_ce_sum(model, x, y, device=device)
                ce_q += b
            ppl_fp = math.exp(ce_fp / n_tok)
            ppl_q = math.exp(ce_q / n_tok)
            per_seed.append({'ckpt': ckpt_path, 'probe': args.probe, 'probe_split': args.probe_split if args.probe == 'dataset' else None, 'seed': seed, 'ppl_fp': ppl_fp, 'ppl_w8a8': ppl_q, 'ppl_drop': ppl_q - ppl_fp, 'ppl_drop_pct': 100.0 * (ppl_q - ppl_fp) / ppl_fp, 'probe_token_coverage': coverage})
        r = aggregate_w8a8_seed_rows(ckpt_path, per_seed)
        if args.probe == 'dataset' and args.probe_split != 'validation':
            print(f"[w8a8] WARNING: --probe dataset scored on the '{args.probe_split}' split; use 'validation' for model comparison ('test' is reserved for the single final evaluation, 'train' is optimistically biased)", file=sys.stderr)
        if r['ppl_fp'] >= 0.5 * model.cfg.vocab_size:
            print(f"[w8a8] WARNING: {ckpt_path} ppl_fp={r['ppl_fp']:.1f} (mean over {r['n_seeds']} seeds) is at the vocab_size={model.cfg.vocab_size} ceiling - the probe is fine (coverage={r['probe_token_coverage']:.1%}) but this checkpoint appears untrained, so ppl_drop_pct is not meaningful yet", file=sys.stderr)
        print(f'[w8a8] {ckpt_path}: {r}')
        results.append(r)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    comparison = compare_w8a8_checkpoints(results, alpha=args.alpha)
    comparison_path = out_path.with_name('ppl_comparison.json')
    comparison_path.write_text(json.dumps(comparison, indent=2))
    for t in comparison['pairwise_tests']:
        if t.get('p_one_sided') is None:
            print(f"[w8a8][compare] {t['a']} vs {t['b']} on {t['metric']}: {t['reason']}")
            continue
        verdict = 'A more robust' if t['significant'] and t['mean_delta'] < 0 else ('B more robust' if t['significant'] else 'inside the noise')
        print(f"[w8a8][compare] {t['a']} vs {t['b']} on {t['metric']}: delta={t['mean_delta']:+.4f} pts, 95% CI [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}], Holm p={t['p_holm']:.4f} (n={t['n_shared_seeds']}) -> {verdict}")
    for d in comparison['degradation_tests']:
        if d.get('p_one_sided') is None:
            print(f"[w8a8][compare] {d['ckpt']}: degradation not testable ({d['reason']})")
            continue
        verdict = 'real degradation' if d['significant'] else 'inside the noise'
        print(f"[w8a8][compare] {d['ckpt']}: mean drop {d['mean_drop_pct']:.4f}% pts, 95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}], Holm p={d['p_holm']:.4f} (n={d['n_seeds']}) -> {verdict}")
    manifest = collect_run_metadata(seed=seeds[0])
    manifest['hyperparameters'] = vars(args)
    manifest['seeds'] = seeds
    manifest['results_file'] = str(out_path)
    manifest['comparison_file'] = str(comparison_path)
    save_run_manifest(out_path.with_name('run_manifest.json'), manifest)
    print(f'[done] saved to {out_path}')
if __name__ == '__main__':
    main()
