from __future__ import annotations
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from adasquash import build_norm, markov_batch
from adasquash.diagnostics import triage_nonfinite
from adasquash.attention import CausalSelfAttention, clear_logit_scratch_cache
from adasquash.metrics import mean_std_stderr
from adasquash.model import GPTConfig, build_norm_kwargs, fused_cross_entropy_mean, _LayerScaleResidual
from adasquash.repro import collect_run_metadata, file_fingerprint, save_run_manifest, set_all_seeds
from adasquash.utils import ActivationStats, LossSpikeCounter, count_parameters
_FLAT_TOKEN_CACHE: dict = {}

class MoELayer(nn.Module):

    def __init__(self, dim: int, hidden: int, n_experts: int=8, top_k: int=2, dropout: float=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        hidden = int(hidden * 2 / 3)
        hidden = hidden + 7 & ~7
        self.experts_w1 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.experts_w2 = nn.Parameter(torch.empty(n_experts, hidden, dim))
        self.experts_w3 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.drop = nn.Dropout(dropout)
        for p in (self.experts_w1, self.experts_w2, self.experts_w3):
            nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        logits = self.router(x_flat)
        probs = logits.softmax(-1)
        if self.top_k == 1:
            scores, indices = probs.max(dim=-1, keepdim=True)
        else:
            scores, indices = probs.topk(self.top_k, dim=-1)
        # Renormalize the top-k gates to sum to 1; `scores` is a fresh
        # topk/max output, so the division can write into it.
        scores.div_(scores.sum(dim=-1, keepdim=True).clamp_min(1e-09))
        n_tokens = x_flat.shape[0]
        n_experts = self.n_experts
        k = self.top_k
        flat_expert = indices.reshape(-1)
        key = (n_tokens, k, x.device)
        flat_token = _FLAT_TOKEN_CACHE.get(key)
        if flat_token is None:
            flat_token = torch.arange(n_tokens, device=x.device).repeat_interleave(k)
            _FLAT_TOKEN_CACHE[key] = flat_token
        flat_gate = scores.reshape(-1)
        order = flat_expert.argsort()
        token_of = flat_token[order]
        gate_of = flat_gate[order]
        counts = flat_expert.bincount(minlength=n_experts)
        ends = counts.cumsum(0).tolist()
        y = x_flat.new_zeros(n_tokens, C)
        start = 0
        for e in range(n_experts):
            end = ends[e]
            n_e = end - start
            if n_e:
                tok = token_of[start:end]
                gate = gate_of[start:end].unsqueeze(-1)
                xe = x_flat.index_select(0, tok)
                h = F.silu(xe @ self.experts_w1[e])
                h.mul_(xe @ self.experts_w3[e])
                y.index_add_(0, tok, (h @ self.experts_w2[e]).mul_(gate))
            start = end
        y = self.drop(y)
        y = y.view(B, T, C)
        tokens_per_expert = counts.to(probs.dtype).detach()
        f_i = tokens_per_expert / n_tokens
        p_i = probs.mean(dim=0)
        aux = self.n_experts * (f_i * p_i).sum()
        return (y, aux, tokens_per_expert)

class MoEBlock(nn.Module):

    def __init__(self, cfg: GPTConfig, n_experts: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        norm_kwargs = build_norm_kwargs(cfg, layer_idx)
        self.norm1 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.attn = CausalSelfAttention(cfg.dim, cfg.n_heads, qk_norm=cfg.qk_norm, alpha_init=cfg.qk_alpha_init, dropout=cfg.dropout)
        self.norm2 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.moe = MoELayer(cfg.dim, hidden, n_experts=n_experts, top_k=2, dropout=cfg.dropout)
        self.use_layerscale = cfg.use_layerscale
        if self.use_layerscale:
            self.ls1 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale_init))
            self.ls2 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale_init))

    def forward(self, x, freqs_cis=None):
        h = self.norm1(x)
        a, _ = self.attn(h, freqs_cis=freqs_cis)
        if self.use_layerscale:
            x = _LayerScaleResidual.apply(x, a, self.ls1)
        else:
            x = x + a
        h = self.norm2(x)
        m, aux, tpe = self.moe(h)
        if self.use_layerscale:
            x = _LayerScaleResidual.apply(x, m, self.ls2)
        else:
            x = x + m
        return (x, aux, tpe)

class MoEGPT(nn.Module):
    # Coefficient of the switch load-balancing penalty. The training loss
    # combines it with the cross-entropy; held-out reports keep the two apart
    # (`val_ce` vs `val_aux`) because the aux term is feature of the *router*,
    # not of language modelling.
    AUX_LOSS_COEF = 0.001

    def __init__(self, cfg: GPTConfig, n_experts: int=8):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([MoEBlock(cfg, n_experts, i) for i in range(cfg.n_layers)])
        self.norm_f = build_norm(cfg.norm_name, cfg.dim, **build_norm_kwargs(cfg, cfg.n_layers, is_final_norm=True))
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.register_buffer('_pos_ids', torch.arange(cfg.max_seq_len), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.weight is self.tok_emb.weight:
                # lm_head shares the input embedding's storage (tied); `apply`
                # already initialized it through `tok_emb`.
                return
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None, return_aux=False, return_logits=True):
        B, T = x.shape
        z = self.tok_emb(x)
        pe = self.pos_emb.weight[:T]
        z.add_(pe if pe.dtype == z.dtype else pe.to(z.dtype))
        z = self.drop(z)
        aux_total = 0.0
        tpe_list = []
        for blk in self.blocks:
            z, aux, tpe = blk(z)
            aux_total = aux_total + aux
            tpe_list.append(tpe)
        z = self.norm_f(z)
        loss = None
        if y is not None and (not return_logits):
            loss = fused_cross_entropy_mean(z.reshape(-1, z.size(-1)), self.lm_head.weight, y.reshape(-1), ignore_index=self.cfg.ignore_index) + self.AUX_LOSS_COEF * aux_total
            logits = None
        else:
            logits = self.lm_head(z)
            if y is not None:
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.cfg.ignore_index)
                loss = ce + self.AUX_LOSS_COEF * aux_total
        if return_aux:
            return (logits, loss, aux_total, tpe_list)
        return (logits, loss)

DEFAULT_SEEDS = (42, 43, 44)
_MOE_SEED_METRIC_KEYS = ('val_ce', 'val_loss', 'final_loss', 'final_expert_cv')

def run_one(args, seed: int, output_dir: Path) -> dict:
    """One complete MoE training run on `seed`, writing its own artifacts."""
    # The data pipeline needs an integer seed for the split/loader RNG, and
    # every run must use its own: in the default multi-seed mode (`--seeds`)
    # `args.seed` is None. The copy mirrors `tune.run_trial`, so the caller's
    # namespace is not mutated.
    args = copy.copy(args)
    args.seed = int(seed)
    output_dir = Path(output_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(seed)
    sanity_report_path = None
    if getattr(args, 'sanity_check', False):
        # Fail fast on the exact model class being trained: the dense-GPT
        # preflight would not catch a broken router / aux-loss path. Seeding
        # first exports PYTHONHASHSEED for the child interpreter.
        from scripts.sanity_overfit import run_preflight
        # The preflight must exercise the *actual* architecture (same depth /
        # width / heads / context), otherwise it validates plumbing on a model
        # this run does not use.
        run_preflight(args.norm, seed, output_dir, model='moe', n_experts=args.n_experts,
                      arch_args=['--dim', str(args.dim), '--n-layers', str(args.n_layers),
                                 '--n-heads', str(args.n_heads), '--seq-len', str(args.seq_len),
                                 '--batch-size', str(args.batch_size)])
        sanity_report_path = output_dir / 'sanity_overfit.json'
    cfg = GPTConfig(vocab_size=50304, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads, max_seq_len=args.seq_len, mlp_ratio=4.0, norm_name=args.norm)
    model = MoEGPT(cfg, n_experts=args.n_experts).to(device)
    print(f'[moe] total params = {count_parameters(model) / 1000000.0:.2f}M')
    from scripts.train_gpt2 import build_optimizer, open_data_split
    opt = build_optimizer(model, args.lr, 0.1, 0.9, 0.95)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'run_manifest.json'
    run_manifest = collect_run_metadata(seed=seed)
    run_manifest['hyperparameters'] = vars(args)
    if args.hypothesis:
        run_manifest['hypothesis'] = args.hypothesis
        print(f'[hypothesis] {args.hypothesis}')
    else:
        print('[hypothesis] none stated: pass --hypothesis "..." so this run is auditable as a test of a claim, not a sweep')
    data_reports: dict = {}
    run_manifest['data'] = {
        'dataset': args.dataset,
        'dataset_config': args.dataset_config,
        'dataset_revision': getattr(args, 'dataset_revision', None),
        'tokenizer': getattr(args, 'tokenizer', None),
        'tokenizer_revision': getattr(args, 'tokenizer_revision', None),
        'split_reports': data_reports,
    }
    if sanity_report_path is not None:
        run_manifest['overfit_sanity_check'] = {'enabled': True, **file_fingerprint(sanity_report_path)}
    else:
        run_manifest['overfit_sanity_check'] = {'enabled': False}
    save_run_manifest(manifest_path, run_manifest)
    print(f'[moe][track] run manifest -> {manifest_path}')
    rng = torch.Generator(device=device).manual_seed(seed)
    data_iter = None
    if args.dataset:
        # Both loaders go through the same split configuration: the training
        # boundary must reserve exactly the held-out validation range with the
        # same cleaning policy, otherwise validation documents could overlap
        # the documents the model trained on.
        loader = open_data_split('train', args)
        report = getattr(loader, 'split_report', None)
        if report:
            data_reports['train'] = report
            print(f'[moe][data] train split: {report}')
        data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        if data_iter is None:
            if args.probe == 'random':
                x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), generator=rng, device=device)
                y = torch.full_like(x, cfg.ignore_index)
                y[:, :-1] = x[:, 1:]
                return (x, y)
            return markov_batch(rng, cfg.ignore_index, args.batch_size, args.seq_len, device)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        return (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
    history = []
    stats_windowed = False
    for step in range(args.max_steps):
        x, y = next_batch()
        # `next_batch` hands back a freshly built pair (probe generators or a
        # freshly collated loader batch), so the next-token column can be
        # masked in place.
        y[:, -1] = cfg.ignore_index
        if step % args.stats_every == 0:
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        logits, loss, aux, tpe_list = model(x, y, return_aux=True, return_logits=False)
        del logits
        # `loss` is not touched by backward, so the pre-backward value is the
        # value to record.
        loss_val = float(loss.detach())
        if not math.isfinite(loss_val):
            triage = triage_nonfinite(model, x, y, device_type=device,
                                      lr=opt.param_groups[0]['lr'], grad_clip=1.0)
            raise SystemExit(
                f"[moe][nan] non-finite loss at step {step}: triage stage="
                f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
                + ' | '.join(triage['recommendations'])
            )
        loss.backward()
        if step % args.stats_every == 0:
            act_stats.detach()
            # The `_logit_stats` scratch only serves the sweep of the forward
            # that just finished; between stats steps it would otherwise pin
            # a full-sweep buffer on the device for nothing.
            clear_logit_scratch_cache()
        # Per-block routed-load CVs stay on the device; the history
        # serialization materializes them once at the end of the run.
        expert_cv = _expert_cv_per_block(tpe_list)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_spike.update(loss_val)
        history.append({'step': step, 'loss': loss_val, 'aux_t': aux.detach(), 'expert_cv_t': expert_cv, 'n_loss_spikes': loss_spike.n_spikes})
        if step % args.log_every == 0:
            exp_cv = float(expert_cv[-1])
            print(f'  [moe] step={step:4d} loss={loss_val:.4f} aux={float(aux.detach()):.4f} expert_cv={exp_cv:.3f}', flush=True)
    for row in history:
        row['aux'] = float(row.pop('aux_t'))
        cv = row.pop('expert_cv_t').tolist()
        row['expert_cv_per_block'] = cv
        row['expert_cv'] = cv[-1]
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f)
    # Held-out validation loss: real data comes from the chronological
    # validation split; the synthetic corpus uses a generator (seed + 1) the
    # optimizer never saw. No gradients flow from either.
    val_loss = val_ce = val_aux = None
    if args.val_batches > 0:
        model.eval()
        val_losses = []
        val_ce_losses = []
        val_auxes = []
        with torch.inference_mode():
            if args.dataset:
                val_loader = open_data_split('validation', args)
                rval = getattr(val_loader, 'split_report', None)
                if rval:
                    data_reports['validation'] = rval
                    print(f'[moe][data] validation split: {rval}')
                val_iter = iter(val_loader)
            else:
                val_iter = None
            val_rng = torch.Generator(device=device).manual_seed(seed + 1)
            for _ in range(args.val_batches):
                if val_iter is not None:
                    try:
                        vx, vy = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_loader)
                        vx, vy = next(val_iter)
                    vx = vx.to(device, non_blocking=True)
                    vy = vy.to(device, non_blocking=True)
                else:
                    vx, vy = markov_batch(val_rng, cfg.ignore_index, args.batch_size, args.seq_len, device)
                _, total, aux, _ = model(vx, vy, return_aux=True, return_logits=False)
                total_val = float(total)
                vaux = float(aux)
                val_losses.append(total_val)
                # The total loss the router regularizer rides on is not a
                # language-modelling number: strip the aux term back out so
                # `val_ce` compares only predictive quality.
                val_ce_losses.append(total_val - model.AUX_LOSS_COEF * vaux)
                val_auxes.append(vaux)
        model.train()
        val_loss = sum(val_losses) / len(val_losses)
        val_ce = sum(val_ce_losses) / len(val_ce_losses)
        val_aux = sum(val_auxes) / len(val_auxes)
        print(f'[moe][val] held-out cross-entropy={val_ce:.4f} '
              f'(combined loss={val_loss:.4f}, aux={val_aux:.4f} - the combined '
              f'number is confounded with routing and is not the comparison metric)')
    s = act_stats.summary()
    if val_loss is not None:
        s['val_loss'] = val_loss
        s['val_ce'] = val_ce
        s['val_aux'] = val_aux
    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    run_manifest['data']['split_reports'] = data_reports
    run_manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    run_manifest['results'] = {
        'steps': args.max_steps,
        'final_loss': history[-1]['loss'] if history else None,
        'final_expert_cv': history[-1]['expert_cv'] if history else None,
        'val_loss': val_loss,
        'val_ce': val_ce,
        'val_aux': val_aux,
        'n_loss_spikes': loss_spike.n_spikes,
    }
    save_run_manifest(manifest_path, run_manifest)
    print(f'[done] saved to {output_dir} | n_loss_spikes={loss_spike.n_spikes}')
    return dict(run_manifest['results'], seed=seed, output_dir=str(output_dir))

def aggregate_moe_seed_results(results: list, seeds) -> dict:
    """Mean +/- Std of the MoE study over independent seeds.

    `val_ce` (held-out cross-entropy) is the language-modelling selection
    metric: it excludes the router's load-balancing penalty, which depends on
    how the norm changes routing and would otherwise be scored as if it were
    predictive quality. The combined `val_loss` (= val_ce + 0.001 * aux) and
    its `val_aux` component are still reported (and aggregated) next to it for
    reference, as are `final_loss` and `final_expert_cv`. The per-seed records
    stay in `per_seed` so a load-imbalance result resting on one seed cannot
    hide behind the mean.
    """
    def _usable(value) -> bool:
        return (isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value))

    agg = {'seeds': [int(s) for s in seeds], 'n_seeds': len(results), 'per_seed': results}
    for key in _MOE_SEED_METRIC_KEYS:
        vals = [r[key] for r in results if _usable(r.get(key))]
        stats = mean_std_stderr(vals)
        if stats is None:
            agg[key] = None
            agg[key + '_std'] = None
            continue
        agg[key] = stats['mean']
        agg[key + '_std'] = stats['std'] if stats['n'] >= 2 else None
    # A readable manifest can still report NaN/Inf (a diverged validation
    # pass); it is not a usable replication and is listed rather than silently
    # pooled into the mean.
    agg['invalid_seeds'] = sorted({
        int(r['seed']) for r in results
        if isinstance(r.get('seed'), int)
        and any(isinstance(r.get(k), float) and not math.isfinite(r[k])
                for k in _MOE_SEED_METRIC_KEYS)})
    agg['n_loss_spikes_total'] = sum(r['n_loss_spikes'] or 0 for r in results
                                     if isinstance(r.get('n_loss_spikes'), (int, float)))
    agg['aggregation'] = ('Mean +/- Std across independent seeds; the unit of '
                          'replication is the seed, not the step. NaN/Inf seed '
                          'values are excluded and listed in invalid_seeds.')
    return agg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--norm', default='dytcd', choices=['ln', 'rmsnorm', 'dyt', 'dytc', 'dytd', 'dytcd'])
    p.add_argument('--n-experts', type=int, default=8)
    p.add_argument('--dim', type=int, default=512)
    p.add_argument('--n-layers', type=int, default=8)
    p.add_argument('--n-heads', type=int, default=8)
    p.add_argument('--seq-len', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=0.0006)
    p.add_argument('--seed', type=int, default=None, help='deprecated single-seed shorthand; overrides --seeds when given')
    p.add_argument('--seeds', nargs='+', type=int, default=list(DEFAULT_SEEDS), help='independent replication seeds; each seed reruns the identical recipe and the study is reported as Mean +/- Std across seeds (>= 3 recommended)')
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--stats-every', type=int, default=200)
    p.add_argument('--output-dir', default='runs/moe-tiny')
    p.add_argument('--probe', default='markov', choices=['markov', 'random'], help="corpus. 'random' is uniform token ids: it has no next-token signal, so the LM loss is pinned at log(vocab_size) and the run learns nothing")
    p.add_argument('--dataset', default=None, help='optional HF dataset name; when given, replaces the synthetic corpus with packed real text')
    p.add_argument('--dataset-config', default=None)
    p.add_argument('--val-frac', type=float, default=0.005, help='chronological validation fraction (real-data mode)')
    p.add_argument('--test-frac', type=float, default=0.005, help='sealed test fraction (real-data mode)')
    p.add_argument('--num-documents', type=int, default=None, help='total documents; required when the dataset reports no size')
    p.add_argument('--min-doc-chars', type=int, default=0, help='drop documents shorter than this many characters (0 = keep all non-empty); applied identically to train and validation')
    p.add_argument('--max-doc-chars', type=int, default=0, help='drop documents longer than this many characters (0 = no length-outlier cap); applied identically to train and validation')
    p.add_argument('--no-dedup', dest='dedup', action='store_false', help='disable exact-duplicate document removal')
    p.add_argument('--exclude-hash-file', default=None, help='training-document hash ledger (scripts/audit_data.py --write-train-hash-file); validation documents duplicated in train are dropped, never training documents')
    p.add_argument('--train-shuffle-buffer', type=int, default=0, help='shuffle window over packed TRAINING batches (0 = off); never applied to validation')
    p.add_argument('--val-batches', type=int, default=10, help='held-out validation batches at the end of training (0 disables)')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--dataset-revision', default=None, help='dataset revision (commit hash) to pin; recorded in the run manifest so the trained-on data version is auditable (None = upstream default branch, which can move)')
    p.add_argument('--tokenizer-revision', default=None, help='tokenizer revision (commit hash) to pin; recorded in the run manifest (None = upstream default branch)')
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='per-seed small-batch overfit sanity check on the MoE model (router + aux loss) before training; aborts the run if it fails (default: on)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the pre-flight overfit sanity check (use only when the check itself is the problem, e.g. offline runs where the preflight cannot be spawned)')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this study tests (e.g. "DyT-CD keeps expert load balanced and matches ln/RMSNorm val_ce at this scale"); recorded in the per-seed and aggregate run manifests so the result can be audited against the claim it was meant to support')
    args = p.parse_args()
    raw_seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if not raw_seeds:
        p.error('--seeds must not be empty')
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[moe][warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed reruns the identical run, it is not an independent replication)')
    if len(seeds) < 3:
        print(f'[moe][warn] {len(seeds)} seed(s): expert load and held-out loss vary run to run; use >= 3 independent seeds before interpreting a difference')
    multi = len(seeds) > 1
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        out_dir = base / f'seed{seed}' if multi else base
        print(f'[moe] seed={seed} -> {out_dir}')
        results.append(run_one(args, seed, out_dir))
    if multi:
        agg = aggregate_moe_seed_results(results, seeds)
        summary_path = base / 'seeds_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(agg, f, indent=2, default=str)
        manifest = collect_run_metadata(seed=seeds[0])
        manifest['hyperparameters'] = vars(args)
        manifest['seeds'] = seeds
        if args.hypothesis:
            manifest['hypothesis'] = args.hypothesis
        manifest['aggregate'] = agg
        save_run_manifest(base / 'run_manifest.json', manifest)
        val = agg.get('val_ce')
        val_std = agg.get('val_ce_std')
        if val is not None:
            std_str = '' if val_std is None else f' +/- {val_std:.4f}'
            loss = agg.get('val_loss')
            loss_str = '' if loss is None else f' (combined incl. aux={loss:.4f})'
            print(f"[moe][done] {args.norm}: val_ce={val:.4f}{std_str}{loss_str} "
                  f"over {agg['n_seeds']} seeds -> {summary_path}")
        else:
            print(f'[moe][done] no usable val_ce across seeds -> {summary_path}')
        if agg.get('invalid_seeds'):
            print(f"[moe][warn] seeds {agg['invalid_seeds']} produced NaN/Inf metrics; "
                  f"they are excluded from the pooled mean, not treated as clean replications")

def _expert_cv(tpe) -> float:
    if not isinstance(tpe, torch.Tensor):
        return 0.0
    return (tpe.std() / (tpe.mean() + 1e-06)).item()

def _expert_cv_per_block(tpe_list) -> torch.Tensor:
    """Per-block routed-load CVs as a device tensor.

    The host materialization happens when the run history is written.
    """
    n = len(tpe_list)
    present = [t.detach().reshape(1, -1) if isinstance(t, torch.Tensor) else None for t in tpe_list]
    if not any(t is not None for t in present):
        return torch.zeros(n)
    stacked = torch.cat([t for t in present if t is not None], dim=0)
    vals = stacked.std(dim=-1) / (stacked.mean(dim=-1) + 1e-06)
    if n == vals.numel():
        return vals
    out = torch.zeros(n, dtype=vals.dtype, device=vals.device)
    idx = torch.tensor([i for i, t in enumerate(present) if t is not None], device=vals.device)
    return out.index_copy_(0, idx, vals)
if __name__ == '__main__':
    main()
