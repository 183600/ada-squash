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
from adasquash.metrics import mean_std_stderr
from adasquash.model import GPTConfig, build_norm_kwargs, fused_cross_entropy_mean, clear_ce_scratch_cache, clear_cast_weight_cache, _LayerScaleResidual
from adasquash.repro import collect_run_metadata, file_fingerprint, finish_metadata, save_run_manifest, set_all_seeds
from adasquash.utils import ActivationStats, LossSpikeCounter, count_parameters
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
    run_manifest['status'] = 'running'
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
        return dict(run_manifest['results'], seed=seed, output_dir=str(output_dir))
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
    if args.log_every < 1:
        p.error(f'--log-every {args.log_every} must be >= 1')
    if args.stats_every < 1:
        p.error(f'--stats-every {args.stats_every} must be >= 1')
    if args.val_batches < 0:
        p.error(f'--val-batches {args.val_batches} must be >= 0')
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
        manifest = finish_metadata(
            manifest,
            results={
                'n_seeds': agg['n_seeds'],
                'val_ce': agg.get('val_ce'),
                'val_ce_std': agg.get('val_ce_std'),
                'invalid_seeds': agg.get('invalid_seeds'),
                'tokens_per_second': agg.get('tokens_per_second'),
                'peak_device_memory_mb': agg.get('peak_device_memory_mb'),
            },
            artifacts={'seeds_summary': file_fingerprint(summary_path)},
        )
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
