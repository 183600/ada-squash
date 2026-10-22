"""Tiny MoE study for the DyT++ / AdaSquash normalization comparison.

The scientific unit is a seed-paired contrast between normalization layers.
That design is only valid if the seed fixes every non-treatment parameter:
the transformer weights, router and expert banks must be identical across
norm arms, so a loss difference can be attributed to the norm rather than to
a different initialization stream. Norm constructors may draw from the global
RNG (DyT-C's alpha jitter does), and the expert banks are raw Parameters that
the generic Linear/Embedding initialization does not cover. The runner
therefore builds the model with ``init=False`` and calls
``MoEGPT.reset_parameters(seed)`` once, after all norm modules exist; the
reset also covers the expert banks. ``tests/test_moe_init.py`` pins this
invariant.

The comparison metric is held-out cross-entropy (``val_ce``), not the combined
training objective: the router's load-balancing penalty is a property of the
routing mechanism, and scoring it as predictive quality would confound the
norm effect with expert usage. ``val_loss`` / ``val_aux`` are reported next to
it for transparency. Training seeds are the replication unit, and a cell is
only certified when it is seed-complete, shares the reference's recipe and
data snapshot, and passes the paired Student-t with Holm correction.
"""
from __future__ import annotations
import argparse
import copy
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
import torch.nn as nn
import torch.nn.functional as F
from adasquash import build_norm, markov_batch
from adasquash.diagnostics import diagnose_train_val_curve, triage_nonfinite
from adasquash.attention import CausalSelfAttention, clear_logit_scratch_cache
from adasquash.metrics import (MIN_SEEDS_FOR_SIGNIFICANCE, holm_adjust,
                               mean_std_stderr, paired_design_summary,
                               paired_t_one_sided, student_t_ppf,
                               scale_replication_summary)
from adasquash.model import GPTConfig, build_norm_kwargs, fused_cross_entropy_mean, clear_ce_scratch_cache, clear_cast_weight_cache, _LayerScaleResidual
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds
from adasquash.utils import ActivationStats, LossSpikeCounter, count_parameters
from scripts.data_pipeline import _is_pinned_revision
_FLAT_TOKEN_CACHE: dict = {}
_FLAT_TOKEN_CACHE_MAX_ENTRIES = 8

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
            if len(_FLAT_TOKEN_CACHE) >= _FLAT_TOKEN_CACHE_MAX_ENTRIES:
                _FLAT_TOKEN_CACHE.clear()
            _FLAT_TOKEN_CACHE[key] = flat_token
        flat_gate = scores.reshape(-1)
        order = flat_expert.argsort()
        token_of = flat_token[order]
        gate_of = flat_gate[order]
        counts = flat_expert.bincount(minlength=n_experts)
        y = x_flat.new_zeros(n_tokens, C)
        n_slots = int(token_of.numel())
        max_n = int(counts.max().item()) if n_experts and n_slots else 0
        if max_n and n_experts * max_n <= 4 * max(1, n_slots):
            # One batched GEMM per weight: each expert's sorted bucket is
            # padded to `max_n` rows, padding slots reference the bucket's
            # first token with a gate of exactly zero, so the sum is the same
            # as the per-expert loop but without n_experts tiny serial GEMMs.
            starts = counts.cumsum(0) - counts
            arange = torch.arange(max_n, device=x.device)
            slot_ok = arange.unsqueeze(0) < counts.unsqueeze(1)
            gather = starts.unsqueeze(1) + arange.unsqueeze(0)
            # Padding slots borrow a real slot (the last one) rather than
            # `starts`, which is past the end of the sorted list for an empty
            # expert; their gate is exactly zero, so they contribute nothing.
            pad = starts.clamp(max=n_slots - 1).unsqueeze(1)
            gather = torch.where(slot_ok, gather, pad)
            tok_e = token_of[gather]
            gate_e = (gate_of[gather] * slot_ok.to(gate_of.dtype)).unsqueeze(-1)
            xe = x_flat[tok_e]
            h = F.silu(torch.bmm(xe, self.experts_w1))
            h.mul_(torch.bmm(xe, self.experts_w3))
            ye = torch.bmm(h, self.experts_w2).mul_(gate_e)
            y.index_add_(0, tok_e.reshape(-1), ye.reshape(-1, C))
        else:
            ends = counts.cumsum(0).tolist()
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
        self.moe = MoELayer(cfg.dim, hidden, n_experts=n_experts,
                            top_k=min(2, n_experts), dropout=cfg.dropout)
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

    def __init__(self, cfg: GPTConfig, n_experts: int=8, init: bool=True):
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
        if init:
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
        elif isinstance(m, MoELayer):
            # Expert banks are raw Parameters, not Linear modules, so the
            # branches above never reach them. Re-initialize them here too:
            # otherwise a post-construction reset leaves the expert banks at
            # whatever RNG state the norm constructors happened to leave. A
            # jittered DyT-C norm draws randn at build time, so without this
            # the "same seed" pair no longer shares its non-treatment
            # initialization across norm arms.
            for p in (m.experts_w1, m.experts_w2, m.experts_w3):
                nn.init.normal_(p, mean=0.0, std=0.02)

    @torch.no_grad()
    def reset_parameters(self, seed=None):
        """Initialize every non-norm parameter from one controlled seed.

        The norm parameters are the treatment and keep the values their
        constructor gave them. Everything else must be identical across norm
        arms that share a seed, so a comparison differs only in the norm. The
        constructor's own pass is skipped by `init=False`; this is the single
        initialization pass, mirroring `GPT.reset_parameters` for the dense
        model. Expert banks are included through `_init_weights`.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.apply(self._init_weights)
        return self

    def forward(self, x, y=None, return_aux=False, return_logits=True):
        B, T = x.shape
        z = self.tok_emb(x)
        pe = self.pos_emb.weight[:T]
        z.add_(pe if pe.dtype == z.dtype else pe.to(z.dtype))
        z = self.drop(z)
        # A tensor even when `blocks` is empty (`--n-layers 0`): the caller
        # reads `.detach()` off the aux term every step.
        aux_total = z.new_zeros(())
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
_MOE_SEED_METRIC_KEYS = ('val_ce', 'val_loss', 'val_aux', 'final_loss',
                         'final_expert_cv', 'tokens_per_second')

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
                                 '--batch-size', str(args.batch_size),
                                 '--use-layerscale', '--no-bf16'])
        sanity_report_path = output_dir / 'sanity_overfit.json'
    cfg = GPTConfig(vocab_size=50304, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads, max_seq_len=args.seq_len, mlp_ratio=4.0, norm_name=args.norm)
    # Build without the constructor's initialization pass and initialize every
    # non-norm parameter once, after all norm modules (the treatment) exist.
    # This is what makes the same seed a true block: a norm whose constructor
    # draws RNG (DyT-C's alpha jitter) must not perturb the transformer/expert
    # initialization of the arm it is compared against.
    with torch.device(device):
        model = MoEGPT(cfg, n_experts=args.n_experts, init=False)
    model.reset_parameters(seed)
    model.to(device)
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
        'exclude_hash_file': file_fingerprint(args.exclude_hash_file) if getattr(args, 'exclude_hash_file', None) else None,
        'exclude_test_hash_file': file_fingerprint(args.exclude_test_hash_file) if getattr(args, 'exclude_test_hash_file', None) else None,
        'group_key': getattr(args, 'group_key', None),
        'exclude_group_file': file_fingerprint(args.exclude_group_file) if getattr(args, 'exclude_group_file', None) else None,
        'exclude_test_group_file': file_fingerprint(args.exclude_test_group_file) if getattr(args, 'exclude_test_group_file', None) else None,
    }
    if sanity_report_path is not None:
        run_manifest['overfit_sanity_check'] = {'enabled': True, **file_fingerprint(sanity_report_path)}
    else:
        run_manifest['overfit_sanity_check'] = {'enabled': False}
    run_manifest['status'] = 'running'
    save_run_manifest(manifest_path, run_manifest)
    print(f'[moe][track] run manifest -> {manifest_path}')
    rng = torch.Generator(device=device).manual_seed(seed)
    data_iter = None
    if args.dataset:
        try:
            # Both loaders go through the same split configuration: the training
            # boundary must reserve exactly the held-out validation range with the
            # same cleaning policy, otherwise validation documents could overlap
            # the documents the model trained on.
            loader = open_data_split('train', args)
            report = getattr(loader, 'split_report', None)
            if report:
                data_reports['train'] = report
                # Pin the run (and its manifest) to the resolved upstream commits,
                # not the mutable revision strings typed on the command line. This
                # mirrors train_gpt2.py: a later rerun of the recorded manifest
                # must select the same documents and tokenizer.
                resolved_revision = report.get('dataset_revision_resolved')
                current_revision = getattr(args, 'dataset_revision', None)
                if resolved_revision and resolved_revision != current_revision:
                    print(f'[moe][data] dataset revision pinned: {current_revision!r} -> {resolved_revision}')
                    args.dataset_revision = resolved_revision
                resolved_tokenizer = report.get('tokenizer_revision_resolved')
                current_tokenizer = getattr(args, 'tokenizer_revision', None)
                if resolved_tokenizer and resolved_tokenizer != current_tokenizer:
                    print(f'[moe][data] tokenizer revision pinned: {current_tokenizer!r} -> {resolved_tokenizer}')
                    args.tokenizer_revision = resolved_tokenizer
                run_manifest['hyperparameters'] = vars(args)
                run_manifest['data'].update({
                    'dataset_revision': getattr(args, 'dataset_revision', None),
                    'tokenizer_revision': getattr(args, 'tokenizer_revision', None),
                    'dataset_revision_resolved': resolved_revision,
                    'tokenizer_revision_resolved': resolved_tokenizer,
                })
                save_run_manifest(manifest_path, run_manifest)
                print(f'[moe][data] train split: {report}')
            data_iter = iter(loader)
        except BaseException:
            # The training try/finally starts only after the helpers below, so
            # an unreadable split here would otherwise leave a permanently
            # 'running' manifest behind. Record the setup failure and re-raise.
            run_manifest['status'] = 'failed'
            run_manifest['failed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            run_manifest['results_partial'] = {'steps_completed': 0,
                                               'last_step': None, 'last_loss': None,
                                               'n_loss_spikes': 0}
            save_run_manifest(manifest_path, run_manifest)
            raise

    def random_probe_batch(generator):
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len),
                          generator=generator, device=device)
        y = torch.full_like(x, cfg.ignore_index)
        y[:, :-1] = x[:, 1:]
        return (x, y)

    def next_batch():
        nonlocal data_iter
        if data_iter is None:
            if args.probe == 'random':
                return random_probe_batch(rng)
            return markov_batch(rng, cfg.ignore_index, args.batch_size, args.seq_len, device)
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        return (x.to(device, non_blocking=True), y.to(device, non_blocking=True))
    history = []
    history_rows: list = []      # serialized mirror of `history`, same indices
    stats_windowed = False

    def _serialize_row(row: dict) -> dict:
        """One history row with its device tensors materialized to Python floats."""
        item = {k: v for k, v in row.items() if k not in ('aux_t', 'expert_cv_t')}
        item['aux'] = float(row['aux_t'])
        cv = row['expert_cv_t'].tolist()
        item['expert_cv_per_block'] = cv
        item['expert_cv'] = cv[-1] if cv else None
        return item

    def flush_history() -> None:
        """Atomically persist the curve so far.

        Rewriting the (small) list every `--log-every` steps keeps the file
        current; tmp+rename keeps readers from ever seeing a partial write.
        Each row is serialized exactly once.
        """
        while len(history_rows) < len(history):
            history_rows.append(_serialize_row(history[len(history_rows)]))
        tmp = output_dir / 'history.json.tmp'
        tmp.write_text(json.dumps(history_rows, indent=2), encoding='utf-8')
        os.replace(tmp, output_dir / 'history.json')

    # Periodic held-out curve monitoring. Synthetic mode uses its own
    # generator (seed + 2); dataset mode skips the prefix the final validation
    # pass (seed + 1) will consume, so the two held-out samples are disjoint
    # and the final number stays comparable across rows. The curve tracks
    # held-out *cross-entropy* (the selection metric) against the training CE
    # window, not the aux-penalized total: routing regularization moves the
    # total for reasons unrelated to predictive quality.
    curve_every = int(getattr(args, 'curve_every', 0) or 0)
    if curve_every == 0:
        curve_every = max(1, int(args.max_steps) // 5)
    elif curve_every < 0:
        curve_every = 0
    curve_batches = max(1, int(getattr(args, 'curve_batches', 5) or 5))
    curve_rng = torch.Generator(device=device).manual_seed(seed + 2)
    curve_loader = None
    curve_iter = None
    curve_train: list = []
    curve_val: list = []
    curve_val_stderr: list = []
    curve_steps: list = []
    curve_diagnoses: list = []
    train_ce_window: list = []

    def skip_curve_prefix():
        # The final validation pass reads the first `--val-batches` windows of
        # the same split; skip them on every pass through the loader so curve
        # points and the final estimate stay disjoint, even after a wrap.
        for _ in range(max(0, int(args.val_batches))):
            try:
                next(curve_iter)
            except StopIteration:
                raise SystemExit(
                    '[moe][val] the validation split has no batches after the '
                    'final-validation prefix; refusing to draw a curve sample '
                    'that overlaps the final estimate. Reduce --val-batches or '
                    'provide more validation data.')

    def next_curve_batch():
        nonlocal curve_loader, curve_iter
        if not args.dataset:
            if args.probe == 'random':
                return random_probe_batch(curve_rng)
            return markov_batch(curve_rng, cfg.ignore_index, args.batch_size,
                                args.seq_len, device)
        if curve_loader is None:
            curve_loader = open_data_split('validation', args)
            rcurve = getattr(curve_loader, 'split_report', None)
            if rcurve:
                data_reports['validation_curve'] = rcurve
                print(f'[moe][data] validation curve split: {rcurve}')
            curve_iter = iter(curve_loader)
            skip_curve_prefix()
        try:
            vx, vy = next(curve_iter)
        except StopIteration:
            curve_iter = iter(curve_loader)
            skip_curve_prefix()
            try:
                vx, vy = next(curve_iter)
            except StopIteration:
                raise SystemExit(
                    '[moe][val] the validation split produced no curve batches; '
                    'refusing to report a run without a held-out curve. Check '
                    '--num-documents / the dataset revision, then re-run.')
        return vx.to(device, non_blocking=True), vy.to(device, non_blocking=True)

    finished = False
    t0 = time.time()
    tokens_processed = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    peak_device_memory_mb = None
    try:
        for step in range(args.max_steps):
            x, y = next_batch()
            tokens_processed += int(x.numel())
            if step % args.stats_every == 0:
                act_stats.attach(reset=not stats_windowed)
                stats_windowed = True
            logits, loss, aux, tpe_list = model(x, y, return_aux=True, return_logits=False)
            del logits
            # `loss` is not touched by backward, so the pre-backward value is the
            # value to record.
            loss_val = float(loss.detach())
            aux_val = float(aux.detach())
            if not (math.isfinite(loss_val) and math.isfinite(aux_val)):
                triage = triage_nonfinite(model, x, y, device_type=device,
                                          lr=opt.param_groups[0]['lr'], grad_clip=1.0)
                raise SystemExit(
                    f"[moe][nan] non-finite loss/aux at step {step}: "
                    f"triage stage='{triage['stage']}' ({triage['diagnosis']}); fixes: "
                    + ' | '.join(triage['recommendations'])
                )
            # Predictive quality without the router's aux penalty. This is the
            # quantity the held-out curve and the final val_ce are compared on.
            ce_val = loss_val - model.AUX_LOSS_COEF * aux_val
            loss.backward()
            if step % args.stats_every == 0:
                act_stats.detach()
                clear_logit_scratch_cache()
            expert_cv = _expert_cv_per_block(tpe_list)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not math.isfinite(float(grad_norm)):
                # Finite loss but non-finite gradients: stepping would write
                # NaN/Inf into the weights and only surface one step later as
                # a NaN loss. Run the ordered triage (stage=gradients) now.
                triage = triage_nonfinite(model, x, y, device_type=device,
                                          lr=opt.param_groups[0]['lr'], grad_clip=1.0)
                raise SystemExit(
                    f"[moe][nan] non-finite gradient at step {step} "
                    f"(pre-clip norm={float(grad_norm)}): triage stage="
                    f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
                    + ' | '.join(triage['recommendations'])
                )
            opt.step()
            opt.zero_grad(set_to_none=True)
            clear_cast_weight_cache()
            loss_spike.update(loss_val)
            history.append({'step': step, 'loss': loss_val, 'loss_ce': ce_val,
                            'aux_t': aux.detach(), 'expert_cv_t': expert_cv,
                            'n_loss_spikes': loss_spike.n_spikes})
            if curve_every > 0:
                train_ce_window.append(ce_val)
                # Always land a point on the final step, otherwise a max_steps
                # not divisible by curve_every leaves the diagnosis one window
                # behind the weights that are actually reported.
                if (step + 1) % curve_every == 0 or step + 1 == args.max_steps:
                    train_point = sum(train_ce_window) / len(train_ce_window)
                    train_ce_window = []
                    held_total: list = []
                    held_ce: list = []
                    held_aux: list = []
                    model.eval()
                    with torch.inference_mode():
                        for _ in range(curve_batches):
                            vx, vy = next_curve_batch()
                            _, total, vaux, _ = model(vx, vy, return_aux=True,
                                                      return_logits=False)
                            total_val = float(total)
                            vaux_val = float(vaux)
                            if not (math.isfinite(total_val) and math.isfinite(vaux_val)):
                                triage = triage_nonfinite(model, vx, vy, device_type=device,
                                                          lr=args.lr, grad_clip=1.0)
                                raise SystemExit(
                                    f"[moe][nan] held-out curve batch is non-finite "
                                    f"(total={total_val}, aux={vaux_val}): triage stage="
                                    f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
                                    + ' | '.join(triage['recommendations'])
                                )
                            held_total.append(total_val)
                            held_aux.append(vaux_val)
                            held_ce.append(total_val - model.AUX_LOSS_COEF * vaux_val)
                    model.train()
                    held_stats = mean_std_stderr(held_ce)
                    held_stderr = (held_stats['stderr']
                                   if held_stats and held_stats['n'] >= 2 else None)
                    curve_train.append(train_point)
                    curve_val.append(held_stats['mean'])
                    curve_val_stderr.append(held_stderr)
                    curve_steps.append(step + 1)
                    diagnosis = diagnose_train_val_curve(
                        curve_train, curve_val, train_steps=curve_steps,
                        val_steps=curve_steps,
                        log_vocab_size=math.log(cfg.vocab_size),
                        val_stderr=curve_val_stderr,
                    )
                    curve_diagnoses.append({
                        'step': step + 1,
                        'classification': diagnosis['classification'],
                        'train_trend': diagnosis['train_trend'],
                        'val_trend': diagnosis['val_trend'],
                        'train_window_ce': train_point,
                        'val_ce': curve_val[-1],
                        'val_ce_stderr': held_stderr,
                        'val_total': sum(held_total) / len(held_total),
                        'val_aux': sum(held_aux) / len(held_aux),
                        'val_noise_threshold': diagnosis['evidence'].get('val_noise_threshold'),
                        'val_change_within_noise': diagnosis['evidence'].get('val_change_within_noise'),
                        'gap_change_within_noise': diagnosis['evidence'].get('gap_change_within_noise'),
                        'generalization_gap': diagnosis['generalization_gap'],
                        'recommendations': diagnosis['recommendations'],
                    })
                    noise_str = '' if held_stderr is None else f' +/- {held_stderr:.4f}'
                    print(f"[moe][curve] step={step + 1} train_ce={train_point:.4f} "
                          f"val_ce={curve_val[-1]:.4f}{noise_str} -> "
                          f"{diagnosis['classification']} (train {diagnosis['train_trend']}, "
                          f"val {diagnosis['val_trend']})", flush=True)
                    if diagnosis['classification'] not in ('healthy', 'insufficient_data'):
                        for line in diagnosis['recommendations']:
                            print(f'[moe][curve]   action: {line}')
            if step % args.log_every == 0:
                exp_cv = float(expert_cv[-1]) if expert_cv.numel() else 0.0
                print(f'  [moe] step={step:4d} loss={loss_val:.4f} aux={float(aux.detach()):.4f} expert_cv={exp_cv:.3f}', flush=True)
                flush_history()
        # Held-out validation loss: real data comes from the chronological
        # validation split; the synthetic corpus uses a generator (seed + 1) the
        # optimizer never saw. No gradients flow from either.
        val_loss = val_ce = val_aux = val_ce_stderr = None
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
                            try:
                                vx, vy = next(val_iter)
                            except StopIteration:
                                raise SystemExit(
                                    '[moe][val] the validation split produced no batches; '
                                    'refusing to report a run without a held-out signal. '
                                    'Check --num-documents / the dataset revision, then re-run.')
                        vx = vx.to(device, non_blocking=True)
                        vy = vy.to(device, non_blocking=True)
                    elif args.probe == 'random':
                        vx, vy = random_probe_batch(val_rng)
                    else:
                        vx, vy = markov_batch(val_rng, cfg.ignore_index, args.batch_size, args.seq_len, device)
                    _, total, aux, _ = model(vx, vy, return_aux=True, return_logits=False)
                    total_val = float(total)
                    vaux = float(aux)
                    if not (math.isfinite(total_val) and math.isfinite(vaux)):
                        # The validation pass is also a forward path; a NaN
                        # here must be triaged on the exact batch instead of
                        # being written into val_ce as a selection metric.
                        triage = triage_nonfinite(model, vx, vy, device_type=device,
                                                  lr=args.lr, grad_clip=1.0)
                        raise SystemExit(
                            f"[moe][nan] held-out batch is non-finite "
                            f"(total={total_val}, aux={vaux}): triage stage="
                            f"'{triage['stage']}' ({triage['diagnosis']}); fixes: "
                            + ' | '.join(triage['recommendations'])
                        )
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
            val_ce_stats = mean_std_stderr(val_ce_losses)
            val_ce_stderr = (val_ce_stats['stderr']
                             if val_ce_stats and val_ce_stats['n'] >= 2 else None)
            noise_str = '' if val_ce_stderr is None else f' +/- {val_ce_stderr:.4f}'
            print(f'[moe][val] held-out cross-entropy={val_ce:.4f}{noise_str} '
                  f'(combined loss={val_loss:.4f}, aux={val_aux:.4f} - the combined '
                  f'number is confounded with routing and is not the comparison metric)')
        s = act_stats.summary()
        if val_loss is not None:
            s['val_loss'] = val_loss
            s['val_ce'] = val_ce
            s['val_aux'] = val_aux
        with open(output_dir / 'stats.json', 'w') as f:
            json.dump(s, f, indent=2, default=str)
        # Persist the final curve rows before hashing the artifact, so the
        # manifest fingerprint covers the complete history, not a partial flush.
        flush_history()
        run_manifest['data']['split_reports'] = data_reports
        run_manifest['artifacts'] = {
            'history_json': file_fingerprint(output_dir / 'history.json'),
            'stats_json': file_fingerprint(output_dir / 'stats.json'),
        }
        run_manifest['status'] = 'finished'
        run_manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        elapsed_s = time.time() - t0
        if device == 'cuda':
            peak_device_memory_mb = float(
                torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
        run_manifest['results'] = {
            'steps': args.max_steps,
            # Measured cost, not only the analytic parameter/FLOP estimate.
            'tokens_processed': tokens_processed,
            'tokens_per_second': (tokens_processed / elapsed_s) if elapsed_s > 0 else None,
            'elapsed_s': elapsed_s,
            'peak_device_memory_mb': peak_device_memory_mb,
            'final_loss': history[-1]['loss'] if history else None,
            'final_expert_cv': (float(history[-1]['expert_cv_t'][-1])
                                if history and history[-1]['expert_cv_t'].numel() else None),
            'n_params': count_parameters(model),
            'init_control': {
                'controlled_reset_after_build': True,
                'seed': seed,
                'non_norm_parameters': ('re-initialized once from the run seed after '
                                        'all norm modules were constructed'),
                'norm_parameters': 'treatment values from the norm constructor',
            },
            'val_loss': val_loss,
            'val_ce': val_ce,
            'val_ce_stderr': val_ce_stderr,
            'val_aux': val_aux,
            'n_loss_spikes': loss_spike.n_spikes,
            # The training curve is not the result: keep the held-out curve
            # points and the regime verdicts next to the final number.
            'curve_train_window_ce': curve_train,
            'curve_val_ce': curve_val,
            'curve_val_ce_stderr': curve_val_stderr,
            'curve_steps': curve_steps,
            'curve_diagnosis': curve_diagnoses[-1] if curve_diagnoses else None,
            'curve_diagnosis_history': curve_diagnoses,
            'curve_classification': (curve_diagnoses[-1]['classification']
                                     if curve_diagnoses else None),
        }
        save_run_manifest(manifest_path, run_manifest)
        print(f'[done] saved to {output_dir} | n_loss_spikes={loss_spike.n_spikes}')
        finished = True
        train_report = data_reports.get('train') or {}
        val_report = data_reports.get('validation') or {}
        result = dict(run_manifest['results'], seed=seed, output_dir=str(output_dir))
        # Keep a compact copy of the resolved data/tokenizer identity in the
        # returned per-seed record. The aggregate manifest otherwise only had
        # losses and cost, so `seeds_summary.json` could not prove that all
        # seeds in the mean were trained on the same upstream snapshot.
        result['data_provenance'] = {
            'dataset': getattr(args, 'dataset', None),
            'dataset_config': getattr(args, 'dataset_config', None),
            'dataset_revision': getattr(args, 'dataset_revision', None),
            'dataset_revision_resolved': (
                train_report.get('dataset_revision_resolved')
                or val_report.get('dataset_revision_resolved')),
            'tokenizer': getattr(args, 'tokenizer', None),
            'tokenizer_revision': getattr(args, 'tokenizer_revision', None),
            'tokenizer_revision_resolved': (
                train_report.get('tokenizer_revision_resolved')
                or val_report.get('tokenizer_revision_resolved')),
        }
        return result
    finally:
        # The curve and the partial status must survive NaN aborts and
        # external interrupts: the failure case is exactly when they are read.
        flush_history()
        if not finished:
            run_manifest['status'] = 'failed'
            run_manifest['failed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            # The flushed curve is what a failed run is diagnosed from.
            run_manifest.setdefault('artifacts', {})['history_json'] = file_fingerprint(
                output_dir / 'history.json')
            run_manifest['results_partial'] = {
                'steps_completed': len(history),
                'last_step': history[-1]['step'] if history else None,
                'last_loss': history[-1]['loss'] if history else None,
                'n_loss_spikes': loss_spike.n_spikes,
            }
            save_run_manifest(manifest_path, run_manifest)
            print(f"[moe][track] run did not finish: partial curve in "
                  f"{output_dir / 'history.json'}, manifest marked failed", flush=True)
        # Training is over (or aborted): drop the per-run scratch caches.
        clear_ce_scratch_cache()

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
    provenance_by_seed = {}
    for result in results:
        seed = _moe_seed_id(result.get('seed'))
        if seed is not None:
            provenance_by_seed[seed] = result.get('data_provenance')
    versions_by_seed = {
        seed: (None if not isinstance(prov, dict) else {
            'dataset_revision_resolved': prov.get('dataset_revision_resolved'),
            'tokenizer_revision_resolved': prov.get('tokenizer_revision_resolved'),
        })
        for seed, prov in provenance_by_seed.items()}
    # `_moe_data_version_guard` treats an explicit ``dataset: None`` as the
    # runner's synthetic corpus (generated deterministically from the seed), so
    # it has no external snapshot to pin. The aggregate must use the same
    # definition: marking every clean synthetic seed "unknown" contradicted the
    # comparison guard and made a complete synthetic replication look failed.
    real_version_seeds = []
    synthetic_data_seeds = []
    unknown_version_seeds = []
    for seed, prov in provenance_by_seed.items():
        if not isinstance(prov, dict) or 'dataset' not in prov:
            unknown_version_seeds.append(seed)
            continue
        dataset = prov.get('dataset')
        if dataset is None:
            synthetic_data_seeds.append(seed)
            continue
        if not isinstance(dataset, str) or not dataset.strip():
            unknown_version_seeds.append(seed)
            continue
        version = versions_by_seed.get(seed) or {}
        if all(_is_pinned_revision(value) for value in version.values()):
            real_version_seeds.append(seed)
        else:
            unknown_version_seeds.append(seed)
    known_versions = {
        json.dumps(versions_by_seed[seed], sort_keys=True)
        for seed in real_version_seeds}
    agg['data_provenance_by_seed'] = provenance_by_seed
    agg['data_versions_by_seed'] = versions_by_seed
    agg['real_data_seeds'] = sorted(real_version_seeds)
    agg['synthetic_data_seeds'] = sorted(synthetic_data_seeds)
    agg['unknown_data_version_seeds'] = sorted(unknown_version_seeds)
    agg['data_versions_complete'] = (bool(versions_by_seed)
                                     and not unknown_version_seeds)
    # Synthetic and real-data losses do not share a data-generating process,
    # and real runs must resolve one concrete snapshot. An all-synthetic cell
    # is consistent by construction.
    agg['data_versions_consistent'] = bool(
        agg['data_versions_complete']
        and not (real_version_seeds and synthetic_data_seeds)
        and len(known_versions) <= 1)
    for key in _MOE_SEED_METRIC_KEYS:
        vals = [r[key] for r in results if _usable(r.get(key))]
        stats = mean_std_stderr(vals)
        if stats is None:
            agg[key] = None
            agg[key + '_std'] = None
            continue
        agg[key] = stats['mean']
        agg[key + '_std'] = stats['std'] if stats['n'] >= 2 else None
    mem_vals = [r['peak_device_memory_mb'] for r in results
                if _usable(r.get('peak_device_memory_mb'))]
    mem_stats = mean_std_stderr(mem_vals)
    agg['peak_device_memory_mb'] = None if mem_stats is None else mem_stats['mean']
    agg['peak_device_memory_mb_std'] = (mem_stats['std']
                                        if mem_stats and mem_stats['n'] >= 2 else None)
    agg['cost_basis'] = ('tokens_per_second / peak_device_memory_mb are measured '
                         'this run; memory is null on CPU')
    param_counts = [int(r['n_params']) for r in results
                    if isinstance(r.get('n_params'), int)
                    and not isinstance(r.get('n_params'), bool)]
    # Every seed must report the same count: a missing count is not evidence of
    # consistency, and a moved count means the recipe did not hold.
    agg['n_params_consistent'] = bool(
        param_counts and len(param_counts) == len(results)
        and len(set(param_counts)) == 1)
    agg['n_params'] = param_counts[0] if agg['n_params_consistent'] else None
    # A readable manifest can still report NaN/Inf (a diverged validation
    # pass); it is not a usable replication and is listed rather than silently
    # pooled into the mean.
    agg['invalid_seeds'] = sorted({
        _moe_seed_id(r.get('seed')) for r in results
        if _moe_seed_id(r.get('seed')) is not None
        and any(isinstance(r.get(k), float) and not math.isfinite(r[k])
                for k in _MOE_SEED_METRIC_KEYS)})
    agg['n_seeds_requested'] = len(agg['seeds'])
    requested_seeds = list(agg['seeds'])
    result_seeds = [seed for seed in
                    (_moe_seed_id(r.get('seed')) for r in results)
                    if seed is not None]
    missing_seeds = [s for s in requested_seeds if s not in result_seeds]
    unexpected_seeds = [s for s in result_seeds if s not in requested_seeds]
    duplicate_seeds = sorted({s for s in result_seeds if result_seeds.count(s) > 1})
    seeds_without_val_ce = sorted({
        _moe_seed_id(r.get('seed')) for r in results
        if _moe_seed_id(r.get('seed')) is not None
        and not _usable(r.get('val_ce'))})
    agg['missing_seeds'] = missing_seeds
    agg['unexpected_seeds'] = unexpected_seeds
    agg['duplicate_seeds'] = duplicate_seeds
    agg['seeds_without_val_ce'] = seeds_without_val_ce
    # A pooled mean over fewer or divergent seeds is not a clean replication.
    # Record that verdict next to the mean and in the aggregate manifest, so a
    # reader cannot mistake an invalid seed or a moved data snapshot for a
    # complete study. Coverage is checked by seed identity, not merely by
    # count: two records of one seed must not stand in for a missing seed.
    agg['complete'] = bool(
        agg['n_seeds'] == len(requested_seeds)
        and len(result_seeds) == len(results)
        and not missing_seeds
        and not unexpected_seeds
        and not duplicate_seeds
        and not seeds_without_val_ce
        and not agg['invalid_seeds']
        and agg['n_params_consistent']
        and agg['data_versions_complete']
        and agg['data_versions_consistent'])
    agg['n_loss_spikes_total'] = sum(r['n_loss_spikes'] or 0 for r in results
                                     if isinstance(r.get('n_loss_spikes'), (int, float)))
    # The pooled val_ce can hide that a seed was still over/underfitting at the
    # fixed step count. Surface the per-seed curve regimes next to the mean.
    agg['curve_classification_counts'] = dict(Counter(
        r['curve_classification'] for r in results if r.get('curve_classification')))
    agg['curve_classifications_by_seed'] = {
        str(r.get('seed')): r.get('curve_classification') for r in results}
    agg['aggregation'] = ('Mean +/- Std across independent seeds; the unit of '
                          'replication is the seed, not the step. NaN/Inf seed '
                          'values are excluded and listed in invalid_seeds.')
    return agg

_NORM_CHOICES = ['ln', 'rmsnorm', 'dyt', 'dytc', 'dytd', 'dytcd']
# Everything that changes the trained weights, the held-out sample or the
# evaluation count. A cross-norm comparison must hold all of these fixed, or
# the norm contrast is confounded with a recipe change.
_MOE_PROTOCOL_KEYS = (
    'probe', 'dataset', 'dataset_config', 'dataset_revision', 'tokenizer',
    'tokenizer_revision', 'n_experts', 'dim', 'n_layers', 'n_heads',
    'seq_len', 'batch_size', 'max_steps', 'lr', 'val_frac', 'test_frac',
    'num_documents', 'min_doc_chars', 'max_doc_chars', 'dedup',
    'exclude_hash_file', 'exclude_test_hash_file', 'group_key',
    'exclude_group_file', 'exclude_test_group_file', 'train_shuffle_buffer',
    'val_batches', 'sanity_check',
)
_MOE_COMPARISON_METRICS = ('val_ce', 'final_expert_cv')
# Shape fields: a mismatch here means a different model scale, not a recipe drift.
_MOE_SHAPE_KEYS = frozenset({'n_experts', 'dim', 'n_layers', 'n_heads'})

def moe_protocol_signature(args) -> dict:
    """The comparison-relevant slice of the resolved MoE recipe."""
    return {key: getattr(args, key, None) for key in _MOE_PROTOCOL_KEYS}

def _moe_finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _moe_seed_id(value):
    """Canonical int seed id, or None for a bool/fractional/foreign value."""
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return None


def _moe_seed_metric(cell: dict, metric: str) -> dict:
    values = {}
    for record in cell.get('per_seed', []):
        seed = record.get('seed')
        value = record.get(metric)
        if (isinstance(seed, int) and not isinstance(seed, bool)
                and _moe_finite(value)):
            values[int(seed)] = float(value)
    return values

def _moe_cell_replicated(cell: dict) -> bool:
    """True when every requested seed produced a clean record for this cell.

    `aggregate_moe_seed_results` additionally folds in pinned-data provenance;
    that verdict stays visible in the cell summary but a synthetic run has no
    external snapshot to pin, so the cross-norm comparison applies this
    seed-level check plus the explicit data guard below.
    """
    raw_seeds = cell.get('seeds') or []
    records = cell.get('per_seed') or []
    if not isinstance(raw_seeds, list) or not raw_seeds:
        return False
    if any(isinstance(seed, bool) or not isinstance(seed, int)
           for seed in raw_seeds):
        return False
    seeds = [int(seed) for seed in raw_seeds]
    if len(set(seeds)) != len(seeds):
        return False
    record_seeds = []
    for record in records:
        if not isinstance(record, dict):
            return False
        seed = record.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int):
            return False
        record_seeds.append(int(seed))
    # Equal counts are not equal coverage: records for seeds 42/43/45 must not
    # stand in for a requested 42/43/44 set. Requiring the seed identities also
    # makes a missing per-seed completeness field fail closed here rather than
    # depending on the aggregate writer to have filled it.
    return bool(
        len(record_seeds) == len(seeds)
        and len(set(record_seeds)) == len(record_seeds)
        and set(record_seeds) == set(seeds)
        and not cell.get('missing_seeds') and not cell.get('unexpected_seeds')
        and not cell.get('duplicate_seeds') and not cell.get('invalid_seeds')
        and not cell.get('seeds_without_val_ce')
        and cell.get('n_params_consistent') is True)

def _moe_data_version_guard(cells: dict) -> tuple[bool, str]:
    """Cross-cell data-snapshot guard for a MoE norm comparison."""
    has_real = False
    has_synthetic = False
    keys = set()
    unknown = []
    for norm, cell in cells.items():
        provenance_by_seed = cell.get('data_provenance_by_seed') or {}
        expected = [int(seed) for seed in (cell.get('seeds') or [])]
        if not expected:
            expected = [int(record['seed']) for record in (cell.get('per_seed') or [])
                        if isinstance(record.get('seed'), int)
                        and not isinstance(record.get('seed'), bool)]
        if not expected:
            unknown.append((norm, '<no seed records>'))
            continue
        for seed in expected:
            provenance = provenance_by_seed.get(seed)
            if provenance is None and str(seed) in provenance_by_seed:
                provenance = provenance_by_seed[str(seed)]
            # Missing provenance is not a synthetic run: it is an unauditable
            # one, so fail closed instead of treating every absent record as
            # "no external snapshot to pin".
            if not isinstance(provenance, dict) or 'dataset' not in provenance:
                unknown.append((norm, seed))
                continue
            dataset = provenance.get('dataset')
            if dataset is None:
                # The runner's synthetic path records an explicit
                # ``dataset: None`` entry. An empty/partial provenance dict is
                # not that: it never said which corpus it used.
                has_synthetic = True
                continue
            if not isinstance(dataset, str) or not dataset.strip():
                unknown.append((norm, seed))
                continue
            has_real = True
            versions = {
                'dataset_revision_resolved': provenance.get('dataset_revision_resolved'),
                'tokenizer_revision_resolved': provenance.get('tokenizer_revision_resolved'),
            }
            if not all(_is_pinned_revision(value) for value in versions.values()):
                unknown.append((norm, seed))
                continue
            keys.add(json.dumps(versions, sort_keys=True))
    if unknown:
        return False, (f'cells {unknown} have missing or unpinned data provenance, '
                       'so the norm contrast is not tied to one snapshot')
    if has_real and has_synthetic:
        return False, ('the compared cells mix real-data and synthetic-corpus runs; '
                       'their losses do not share a data-generating process, so no '
                       'norm contrast can be certified')
    if has_synthetic:
        return True, ('synthetic corpus generated deterministically from seed '
                      '(+1 for validation); no external snapshot to pin')
    if len(keys) > 1:
        return False, 'the compared cells resolved different dataset/tokenizer revisions'
    return True, 'all compared cells resolved the same pinned dataset/tokenizer revisions'

def compare_moe_norms(cells: dict, baselines, protocols: dict, metrics=_MOE_COMPARISON_METRICS,
                      alpha: float = 0.05, margin: float = 0.0, power: float = 0.8) -> dict:
    """Paired cross-norm comparison for the MoE study.

    `cells` maps a norm name to its `aggregate_moe_seed_results` dict. Each
    non-reference norm is paired with the best replicated baseline on the
    common seeds, tested one-sided with the paired Student-t (the same test
    the dense ablation uses), Holm-corrected across the whole metric family,
    and annotated with the minimum detectable effect. A cell is only counted
    in the family when it is seed-complete, carries the same resolved recipe
    as the reference, and rests on the same data snapshot; everything else is
    reported descriptively so it cannot be read as a certified win.
    """
    metrics = tuple(metrics)
    out = {
        'available': False,
        'reference': None,
        'reference_baselines': [b for b in baselines if b in cells],
        'norms': list(cells),
        'metrics': list(metrics),
        'alpha': alpha,
        'margin': margin,
        'power_target': power,
        'data_versions_consistent': None,
        'data_version_basis': None,
        'reference_selection': None,
        'protocols': protocols,
        'rows': [],
        'multiple_comparison': None,
        'scale_replication': None,
    }
    replicated_baselines = [
        name for name in out['reference_baselines']
        if _moe_cell_replicated(cells[name])
        and any(_moe_finite(cells[name].get(metric)) for metric in metrics)]
    if not replicated_baselines:
        out['reason'] = ('no replicated baseline cell among '
                         f"{out['reference_baselines'] or 'the requested baselines'}")
        return out
    reference = min((name for name in replicated_baselines),
                    key=lambda name: (cells[name].get('val_ce')
                                      if _moe_finite(cells[name].get('val_ce'))
                                      else float('inf')))
    out['available'] = True
    out['reference'] = reference
    out['reference_selection'] = (
        f'lowest mean val_ce among replicated baselines {replicated_baselines}; '
        'the strongest baseline is the reference so a win is not against a straw man')
    data_ok, data_basis = _moe_data_version_guard(cells)
    out['data_versions_consistent'] = data_ok
    out['data_version_basis'] = data_basis
    reference_cell = cells[reference]

    for metric in metrics:
        reference_by_seed = _moe_seed_metric(reference_cell, metric)
        for norm, cell in cells.items():
            if norm == reference:
                continue
            candidate_by_seed = _moe_seed_metric(cell, metric)
            common = sorted(set(candidate_by_seed) & set(reference_by_seed))
            cand_values = [candidate_by_seed[seed] for seed in common]
            ref_values = [reference_by_seed[seed] for seed in common]
            ref_protocol = protocols.get(reference)
            cand_protocol = protocols.get(norm)
            protocol_complete = (
                isinstance(ref_protocol, dict) and isinstance(cand_protocol, dict)
                and all(key in ref_protocol for key in _MOE_PROTOCOL_KEYS)
                and all(key in cand_protocol for key in _MOE_PROTOCOL_KEYS))
            if not isinstance(ref_protocol, dict) or not isinstance(cand_protocol, dict):
                # A missing protocol is unknown, not equal: without both
                # recorded recipes the pair cannot be certified as a clean
                # same-scale norm contrast.
                mismatches = {'protocol': {
                    'reference': ref_protocol if isinstance(ref_protocol, dict) else None,
                    'candidate': cand_protocol if isinstance(cand_protocol, dict) else None,
                }}
            elif not protocol_complete:
                # A partial protocol dict is unknown too: `.get()` would make
                # every absent key compare equal to the other side's absent
                # key, and an all-None shape would look like a shared scale.
                mismatches = {'protocol_incomplete': {
                    'missing_reference': [key for key in _MOE_PROTOCOL_KEYS
                                          if key not in ref_protocol],
                    'missing_candidate': [key for key in _MOE_PROTOCOL_KEYS
                                          if key not in cand_protocol],
                }}
            else:
                mismatches = {
                    key: {'reference': ref_protocol.get(key),
                          'candidate': cand_protocol.get(key)}
                    for key in _MOE_PROTOCOL_KEYS
                    if ref_protocol.get(key) != cand_protocol.get(key)}
            shape_only_mismatch = (bool(mismatches) and protocol_complete
                                   and set(mismatches) <= _MOE_SHAPE_KEYS)
            a_shape = {key: (cand_protocol or {}).get(key) for key in _MOE_SHAPE_KEYS}
            b_shape = {key: (ref_protocol or {}).get(key) for key in _MOE_SHAPE_KEYS}
            # Scale identity is decided by the shape fields alone. A pair can
            # differ in both shape and a non-shape recipe field; that must not
            # be labelled same-scale merely because the mismatch set is not
            # shape-only. Missing/incomplete protocols are not evidence of a
            # shared scale.
            same_scale = protocol_complete and a_shape == b_shape
            entry = {
                'name': f'{norm} vs {reference}',
                'norm': norm,
                'reference': reference,
                'metric': metric,
                'scale': json.dumps(a_shape, sort_keys=True),
                'reference_scale': json.dumps(b_shape, sort_keys=True),
                'same_scale': same_scale,
                'direction': 'lower_is_better',
                'val_ce': cell.get('val_ce'),
                'val_ce_std': cell.get('val_ce_std'),
                'metric_mean': (sum(cand_values) / len(cand_values)) if cand_values else None,
                'reference_metric_mean': (sum(ref_values) / len(ref_values)) if ref_values else None,
                'final_expert_cv': cell.get('final_expert_cv'),
                'reference_final_expert_cv': reference_cell.get('final_expert_cv'),
                'n_params': cell.get('n_params'),
                'n_params_ratio_vs_reference': None,
                'paired_seeds': common,
                'protocol_mismatches': mismatches or None,
                'data_versions_consistent': data_ok,
                'hypothesis': cell.get('hypothesis'),
                'curve_classification_counts': cell.get('curve_classification_counts'),
                'beats_reference': False,
                'beats_reference_significantly': False,
                'test': None,
                'test_basis': None,
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
                'design_power': None,
            }
            cand_params = cell.get('n_params')
            ref_params = reference_cell.get('n_params')
            if _moe_finite(cand_params) and _moe_finite(ref_params) and ref_params:
                entry['n_params_ratio_vs_reference'] = cand_params / ref_params
            if len(common) >= 2:
                test = paired_t_one_sided(cand_values, ref_values, margin=margin, alpha=alpha)
                entry['test'] = test['test']
                entry['p_one_sided'] = test['p_one_sided']
                entry['t'] = test['t']
                entry['df'] = test['df']
                entry['delta_ci_95'] = [test['ci_low'], test['ci_high']]
                deltas = [cand - ref for cand, ref in zip(cand_values, ref_values)]
                stats = mean_std_stderr(deltas)
                entry.update({
                    'delta_vs_reference': (sum(deltas) / len(deltas)),
                    'n_seeds_favoring': sum(v < 0.0 for v in deltas),
                    'n_seeds_tied': sum(v == 0.0 for v in deltas),
                    'n_seeds_against': sum(v > 0.0 for v in deltas),
                    'per_seed_delta': {str(seed): value for seed, value in zip(common, deltas)},
                    'paired_delta_std': None if stats is None else stats['std'],
                })
                if test.get('df') and stats is not None:
                    # Use the paired-delta stderr directly instead of inverting
                    # the CI, and include the practical margin: the test is
                    # H1 mean(delta) < -margin, so the smallest raw effect that
                    # clears both the margin and the noise is margin + t*se.
                    if stats['stderr'] is not None:
                        entry['minimum_detectable_delta_95_power80'] = (
                            float(margin)
                            + (student_t_ppf(1.0 - alpha, float(test['df']))
                               + student_t_ppf(power, float(test['df'])))
                            * stats['stderr'])
                        entry['design_power'] = paired_design_summary(
                            stats['std'], len(common), margin=margin,
                            alpha=alpha, power=power)
                entry['beats_reference'] = bool(entry['delta_vs_reference'] < -margin)
            entry['test_basis'] = ('per-seed paired t' if len(common) >= 2
                                   else f'described only: {len(common)} shared seed(s)')
            reasons = []
            if mismatches:
                if shape_only_mismatch:
                    entry['test_basis'] = 'cross_scale'
                    reasons.append('different model scale, not a same-scale norm effect: '
                                   + ', '.join(sorted(mismatches)))
                else:
                    reasons.append('recipe differs from the reference: '
                                   + ', '.join(sorted(mismatches)))
            if not data_ok:
                reasons.append('data snapshot differs/unknown')
            if not _moe_cell_replicated(cell):
                reasons.append('cell is not seed-complete and clean')
            if not _moe_cell_replicated(reference_cell):
                reasons.append('reference cell is not seed-complete and clean')
            if len(common) < MIN_SEEDS_FOR_SIGNIFICANCE:
                reasons.append(f'{len(common)} shared seed(s) < {MIN_SEEDS_FOR_SIGNIFICANCE}')
            entry['excluded_from_family'] = '; '.join(reasons) or None
            out['rows'].append(entry)

    family = [entry for entry in out['rows']
              if entry['test'] is not None and entry['excluded_from_family'] is None]
    adjusted = holm_adjust([entry['p_one_sided'] for entry in family])
    for entry, p_holm in zip(family, adjusted):
        entry['p_one_sided_holm'] = p_holm
        entry['beats_reference_significantly'] = bool(
            entry['beats_reference'] and p_holm is not None and p_holm < alpha)
    scale_obs = []
    for entry in out['rows']:
        if (entry.get('same_scale') and entry.get('delta_vs_reference') is not None
                and entry.get('test') is not None
                and entry.get('excluded_from_family') is None):
            scale_obs.append({'scale': entry.get('scale'), 'metric': entry.get('metric'),
                              'contrast': entry.get('name'),
                              'delta': entry.get('delta_vs_reference'),
                              'certified': bool(entry.get('beats_reference_significantly')),
                              'p_one_sided': entry.get('p_one_sided_holm',
                                                       entry.get('p_one_sided')),
                              'n_seeds': len(entry.get('paired_seeds') or [])})
    out['scale_replication'] = scale_replication_summary(scale_obs)
    out['certified_norms'] = sorted({entry['norm'] for entry in family
                                     if entry['beats_reference_significantly']})
    described = [entry['name'] for entry in out['rows']
                 if entry['test'] is not None and entry['excluded_from_family'] is not None]
    out['rows_tested_descriptively_only'] = described
    out['multiple_comparison'] = (
        f'Holm-Bonferroni over {len(family)} seed-replicated cell(s) x metric(s) against '
        f'the same baseline ({reference}); a row is certified only when the adjusted p-value '
        f'is < {alpha}. Tests on mismatched recipes, divergent data snapshots, incomplete cells '
        'or fewer than 3 shared seeds are descriptive only and cannot be certified.')
    return out

def print_moe_norm_comparison(comparison: dict) -> None:
    if not comparison.get('available'):
        print(f"[moe][compare] not available: {comparison.get('reason')}")
        return
    print(f"[moe][compare] reference={comparison['reference']} "
          f"({comparison['reference_selection']}); data: {comparison['data_version_basis']}")
    for entry in comparison['rows']:
        delta = entry.get('delta_vs_reference')
        delta_str = 'n/a' if delta is None else f'{delta:+.4f}'
        ci = entry.get('delta_ci_95')
        ci_str = '' if ci is None else f' ci95=[{ci[0]:+.4f}, {ci[1]:+.4f}]'
        p = entry.get('p_one_sided')
        p_holm = entry.get('p_one_sided_holm')
        p_str = 'n/a' if p is None else f'{p:.3g}'
        p_holm_str = 'n/a' if p_holm is None else f'{p_holm:.3g}'
        mde = entry.get('minimum_detectable_delta_95_power80')
        mde_str = '' if mde is None else f' mde80={mde:.4f}'
        votes = ''
        if entry.get('n_seeds_favoring') is not None:
            n = len(entry.get('paired_seeds') or [])
            votes = (f" seeds_favoring={entry['n_seeds_favoring']}/{n}"
                     + (f" tied={entry['n_seeds_tied']}" if entry.get('n_seeds_tied') else ''))
        note = ''
        if entry.get('excluded_from_family'):
            note = f" [{entry['excluded_from_family']}]"
        ratio = entry.get('n_params_ratio_vs_reference')
        ratio_str = '' if ratio is None else f' params_ratio={ratio:.4f}'
        print(f"[moe][compare] {entry['norm']:>8s} {entry['metric']}: "
              f"mean={entry.get('metric_mean')} vs ref={entry.get('reference_metric_mean')} "
              f"delta={delta_str}{ci_str} p={p_str} p(holm)={p_holm_str}{mde_str}{votes}"
              f"{ratio_str} beats={entry['beats_reference']} "
              f"significant={entry['beats_reference_significantly']}{note}")
    if comparison.get('rows_tested_descriptively_only'):
        print('[moe][compare] descriptive only (not certified): '
              + ', '.join(comparison['rows_tested_descriptively_only']))
    if comparison.get('certified_norms'):
        print(f"[moe][compare] certified norms: {comparison['certified_norms']}")
    else:
        print('[moe][compare] no norm certified as beating the reference at '
              f"alpha={comparison['alpha']}")
    scale = comparison.get('scale_replication') or {}
    if scale.get('single_scale_only'):
        print(f"[moe][scale] WARNING: this study covers {scale.get('n_scales', 0)} model "
              f"scale(s) ({scale.get('scales')}); a MoE norm result at one width/depth/"
              'expert count is not a multi-scale claim. Rerun at another scale and combine '
              'the per-scale comparison files before generalizing.')
    elif scale.get('multi_scale_certified_metrics'):
        print(f"[moe][scale] replicated across scales: "
              f"{scale.get('multi_scale_certified_metrics')} over {scale.get('scales')}")
    else:
        print(f"[moe][scale] scales={scale.get('scales')} "
              f"replicated_metrics={scale.get('replicated_metrics')} "
              f"(no metric certified at >= 2 scales yet)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--norm', default='dytcd', choices=_NORM_CHOICES)
    p.add_argument('--norms', nargs='+', default=None, choices=_NORM_CHOICES, help='run several norms under one identical recipe and compare them against the best replicated baseline with a paired t-test, Holm correction and an MDE; requires --hypothesis')
    p.add_argument('--baselines', nargs='+', default=['ln', 'rmsnorm'], choices=_NORM_CHOICES, help='norm variants treated as references in a --norms study; the strongest replicated one is used as the comparison baseline')
    p.add_argument('--alpha', type=float, default=0.05, help='one-sided family-wise significance level for --norms comparisons')
    p.add_argument('--margin', type=float, default=0.0, help='practical-equivalence margin: a norm must beat the baseline by more than this for beats_* to be true')
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
    p.add_argument('--no-dedup', dest='dedup', action='store_false', help='disable duplicate document removal (exact and canonical text)')
    p.add_argument('--exclude-hash-file', default=None, help='training-document hash ledger (scripts/audit_data.py --write-train-hash-file); validation documents duplicated in train are dropped, never training documents')
    p.add_argument('--exclude-test-hash-file', default=None, help='validation-document hash ledger (scripts/audit_data.py --write-val-hash-file); applied to the sealed test split only, so test documents duplicated in validation are dropped (ignored by the train/validation loaders)')
    p.add_argument('--group-key', default=None, help='grouping-unit field (top-level name or dotted path) for group-holdout ledgers (see train_gpt2.py --group-key)')
    p.add_argument('--exclude-group-file', default=None, help='training-group hash ledger (scripts/audit_data.py --write-train-group-file --group-key K); validation drops every document whose group occurs in train')
    p.add_argument('--exclude-test-group-file', default=None, help='validation-group hash ledger (scripts/audit_data.py --write-val-group-file --group-key K); the sealed test split drops every document whose group occurs in validation')
    p.add_argument('--train-shuffle-buffer', type=int, default=0, help='shuffle window over packed TRAINING batches (0 = off); never applied to validation')
    p.add_argument('--val-batches', type=int, default=10, help='held-out validation batches at the end of training (0 disables)')
    p.add_argument('--curve-every', type=int, default=0, help='periodic held-out curve-monitoring period in steps (0 = auto: max_steps//5; negative = disabled). Curve batches use a separate generator (seed+2) / validation loader, so the final validation pass (seed+1) stays a disjoint sample')
    p.add_argument('--curve-batches', type=int, default=5, help='held-out batches per curve-monitoring point')
    p.add_argument('--tokenizer', default='gpt2')
    p.add_argument('--dataset-revision', default=None, help='dataset revision (commit hash) to pin; recorded in the run manifest so the trained-on data version is auditable (None = upstream default branch, which can move)')
    p.add_argument('--tokenizer-revision', default=None, help='tokenizer revision (commit hash) to pin; recorded in the run manifest (None = upstream default branch)')
    p.add_argument('--sanity-check', dest='sanity_check', action='store_true', default=True, help='per-seed small-batch overfit sanity check on the MoE model (router + aux loss) before training; aborts the run if it fails (default: on)')
    p.add_argument('--no-sanity-check', dest='sanity_check', action='store_false', help='skip the pre-flight overfit sanity check (use only when the check itself is the problem, e.g. offline runs where the preflight cannot be spawned)')
    p.add_argument('--hypothesis', default=None, help='the scientific hypothesis this study tests (e.g. "DyT-CD keeps expert load balanced and matches ln/RMSNorm val_ce at this scale"); recorded in the per-seed and aggregate run manifests so the result can be audited against the claim it was meant to support')
    args = p.parse_args()
    if args.n_experts < 1:
        p.error(f'--n-experts {args.n_experts} must be >= 1')
    for name, value, minimum in (
        ('--dim', args.dim, 1),
        ('--n-layers', args.n_layers, 1),
        ('--n-heads', args.n_heads, 1),
        ('--seq-len', args.seq_len, 1),
        ('--batch-size', args.batch_size, 1),
        ('--curve-batches', args.curve_batches, 1),
        ('--train-shuffle-buffer', args.train_shuffle_buffer, 0),
        ('--min-doc-chars', args.min_doc_chars, 0),
        ('--max-doc-chars', args.max_doc_chars, 0),
    ):
        if value < minimum:
            p.error(f'{name} {value} must be >= {minimum}')
    if args.dim % args.n_heads:
        p.error(f'--dim {args.dim} is not divisible by --n-heads {args.n_heads}')
    if args.num_documents is not None and args.num_documents < 1:
        p.error(f'--num-documents {args.num_documents} must be >= 1')
    if args.max_doc_chars and args.min_doc_chars > args.max_doc_chars:
        p.error(f'--min-doc-chars {args.min_doc_chars} exceeds --max-doc-chars '
                f'{args.max_doc_chars}; no document could pass the cleaner')
    for name, value in (('--val-frac', args.val_frac), ('--test-frac', args.test_frac)):
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            p.error(f'{name} {value} must be finite and in [0, 1)')
    if args.val_frac + args.test_frac >= 1.0:
        p.error(f'--val-frac {args.val_frac} + --test-frac {args.test_frac} leaves no '
                'training stream; the split fractions must sum to less than 1')
    if args.log_every < 1:
        p.error(f'--log-every {args.log_every} must be >= 1')
    if args.stats_every < 1:
        p.error(f'--stats-every {args.stats_every} must be >= 1')
    if args.val_batches < 0:
        p.error(f'--val-batches {args.val_batches} must be >= 0')
    if args.max_steps < 1:
        p.error(f'--max-steps {args.max_steps} must be >= 1')
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha < 1.0:
        p.error(f'--alpha {args.alpha} must be finite and strictly between 0 and 1')
    if not math.isfinite(args.margin) or args.margin < 0.0:
        p.error(f'--margin {args.margin} must be finite and >= 0')
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        p.error(f'--lr {args.lr} must be finite and positive')
    if len(set(args.baselines)) != len(args.baselines):
        p.error(f'--baselines contains duplicates: {args.baselines}; a baseline is one '
                'reference arm, not repeated evidence')
    raw_seeds = [args.seed] if args.seed is not None else list(args.seeds)
    if not raw_seeds:
        p.error('--seeds must not be empty')
    seeds = list(dict.fromkeys(int(s) for s in raw_seeds))
    if len(seeds) != len(raw_seeds):
        print(f'[moe][warn] duplicate seeds removed: {raw_seeds} -> {seeds} '
              f'(a repeated seed reruns the identical run, it is not an independent replication)')
    if len(seeds) < 3:
        print(f'[moe][warn] {len(seeds)} seed(s): expert load and held-out loss vary run to run; use >= 3 independent seeds before interpreting a difference')
    norms = list(dict.fromkeys(args.norms or [args.norm]))
    if args.norms:
        if args.norm not in norms:
            print(f'[moe][warn] --norm {args.norm} is ignored by --norms {norms}')
        if len(norms) < 2:
            p.error('--norms needs at least two norm variants')
        if not args.hypothesis:
            p.error('--norms requires --hypothesis: a cross-norm claim must be '
                    'registered before the runs so the tested family and direction are '
                    'not chosen after seeing the results')
    multi_norm = len(norms) > 1
    base = Path(args.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    cells: dict = {}
    for norm in norms:
        cell_args = copy.copy(args)
        cell_args.norm = norm
        cell_base = base / norm if multi_norm else base
        cell_base.mkdir(parents=True, exist_ok=True)
        cell_multi = len(seeds) > 1
        results = []
        for seed in seeds:
            out_dir = cell_base / f'seed{seed}' if cell_multi else cell_base
            print(f'[moe] norm={norm} seed={seed} -> {out_dir}')
            results.append(run_one(cell_args, seed, out_dir))
        if not (cell_multi or multi_norm):
            continue
        agg = aggregate_moe_seed_results(results, seeds)
        agg['hypothesis'] = args.hypothesis
        agg['norm'] = norm
        summary_path = cell_base / 'seeds_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(agg, f, indent=2, default=str)
        manifest = collect_run_metadata(seed=seeds[0])
        manifest['hyperparameters'] = vars(cell_args)
        manifest['seeds'] = seeds
        if args.hypothesis:
            manifest['hypothesis'] = args.hypothesis
        manifest['aggregate'] = agg
        manifest['data'] = {
            'per_seed': agg['data_provenance_by_seed'],
            'data_versions_by_seed': agg['data_versions_by_seed'],
            'unknown_data_version_seeds': agg['unknown_data_version_seeds'],
            'data_versions_complete': agg['data_versions_complete'],
            'data_versions_consistent': agg['data_versions_consistent'],
        }
        manifest = finish_metadata(
            manifest,
            results={
                'n_seeds': agg['n_seeds'],
                'n_seeds_requested': agg.get('n_seeds_requested'),
                'complete': agg.get('complete'),
                'val_ce': agg.get('val_ce'),
                'val_ce_std': agg.get('val_ce_std'),
                'invalid_seeds': agg.get('invalid_seeds'),
                'tokens_per_second': agg.get('tokens_per_second'),
                'peak_device_memory_mb': agg.get('peak_device_memory_mb'),
            },
            artifacts={'seeds_summary': file_fingerprint(summary_path)},
        )
        save_run_manifest(cell_base / 'run_manifest.json', manifest)
        val = agg.get('val_ce')
        val_std = agg.get('val_ce_std')
        if val is not None:
            std_str = '' if val_std is None else f' +/- {val_std:.4f}'
            loss = agg.get('val_loss')
            loss_str = '' if loss is None else f' (combined incl. aux={loss:.4f})'
            print(f"[moe][done] {norm}: val_ce={val:.4f}{std_str}{loss_str} "
                  f"over {agg['n_seeds']} seeds -> {summary_path}")
        else:
            print(f'[moe][done] no usable val_ce across seeds -> {summary_path}')
        if agg.get('tokens_per_second') is not None:
            tps_std = agg.get('tokens_per_second_std')
            tps_str = '' if tps_std is None else f' +/- {tps_std:.1f}'
            peak = agg.get('peak_device_memory_mb')
            peak_str = '' if peak is None else f', peak={peak:.1f} MB'
            print(f'[moe][done] measured cost={agg["tokens_per_second"]:.1f}'
                  f'{tps_str} tok/s{peak_str}')
        regimes = agg.get('curve_classification_counts')
        if regimes:
            print(f"[moe][curve] per-seed regimes: {regimes}; read the held-out curve "
                  f"before attributing a val_ce delta to the norm")
        if agg.get('invalid_seeds'):
            print(f"[moe][warn] seeds {agg['invalid_seeds']} produced NaN/Inf metrics; "
                  f"they are excluded from the pooled mean, not treated as clean replications")
        cells[norm] = agg
    if multi_norm:
        protocols = {norm: moe_protocol_signature(args) for norm in norms}
        comparison = compare_moe_norms(cells, args.baselines, protocols,
                                       alpha=args.alpha, margin=args.margin)
        comparison['hypothesis'] = args.hypothesis
        comparison_path = base / 'norm_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        study = collect_run_metadata(seed=seeds[0])
        study['hyperparameters'] = vars(args)
        study['seeds'] = seeds
        study['norms'] = norms
        study['hypothesis'] = args.hypothesis
        study['protocol'] = protocols
        study['cells'] = {
            norm: {
                'complete': cells[norm].get('complete'),
                'val_ce': cells[norm].get('val_ce'),
                'val_ce_std': cells[norm].get('val_ce_std'),
                'final_expert_cv': cells[norm].get('final_expert_cv'),
                'n_params': cells[norm].get('n_params'),
                'seeds_summary': str(base / norm / 'seeds_summary.json'),
            }
            for norm in norms}
        study = finish_metadata(
            study,
            results={
                'n_norms': len(norms),
                'n_seeds': len(seeds),
                'reference': comparison.get('reference'),
                'comparison_available': comparison.get('available'),
                'certified_norms': comparison.get('certified_norms'),
            },
            artifacts={
                'norm_comparison': file_fingerprint(comparison_path),
                'cell_summaries': {
                    norm: file_fingerprint(base / norm / 'seeds_summary.json')
                    for norm in norms},
            },
        )
        save_run_manifest(base / 'run_manifest.json', study)
        print_moe_norm_comparison(comparison)

def _expert_cv(tpe) -> float:
    if not isinstance(tpe, torch.Tensor) or tpe.numel() < 2:
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
    std = (torch.zeros(stacked.size(0), dtype=stacked.dtype, device=stacked.device)
           if stacked.size(-1) < 2 else stacked.std(dim=-1))
    vals = std / (stacked.mean(dim=-1) + 1e-06)
    if n == vals.numel():
        return vals
    out = torch.zeros(n, dtype=vals.dtype, device=vals.device)
    idx = torch.tensor([i for i, t in enumerate(present) if t is not None], device=vals.device)
    return out.index_copy_(0, idx, vals)
if __name__ == '__main__':
    main()
