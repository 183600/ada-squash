from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from adasquash import build_norm, markov_batch
from adasquash.attention import CausalSelfAttention
from adasquash.model import GPTConfig, build_norm_kwargs, fused_cross_entropy_mean, _LayerScaleResidual
from adasquash.repro import collect_run_metadata, save_run_manifest, set_all_seeds
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
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-09)
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
        ends = flat_expert.bincount(minlength=n_experts).cumsum(0).tolist()
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
        tokens_per_expert = torch.zeros_like(logits[0]).scatter_add_(0, indices.reshape(-1), torch.ones_like(flat_gate)).detach()
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
            loss = fused_cross_entropy_mean(z.reshape(-1, z.size(-1)), self.lm_head.weight, y.reshape(-1), ignore_index=self.cfg.ignore_index) + 0.001 * aux_total
            logits = None
        else:
            logits = self.lm_head(z)
            if y is not None:
                ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=self.cfg.ignore_index)
                loss = ce + 0.001 * aux_total
        if return_aux:
            return (logits, loss, aux_total, tpe_list)
        return (logits, loss)

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
    p.add_argument('--seed', type=int, default=42)
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
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_all_seeds(args.seed)
    cfg = GPTConfig(vocab_size=50304, dim=args.dim, n_layers=args.n_layers, n_heads=args.n_heads, max_seq_len=args.seq_len, mlp_ratio=4.0, norm_name=args.norm)
    model = MoEGPT(cfg, n_experts=args.n_experts).to(device)
    print(f'[moe] total params = {count_parameters(model) / 1000000.0:.2f}M')
    from scripts.train_gpt2 import build_optimizer, open_data_split
    opt = build_optimizer(model, args.lr, 0.1, 0.9, 0.95)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.output_dir) / 'run_manifest.json'
    run_manifest = collect_run_metadata(seed=args.seed)
    run_manifest['hyperparameters'] = vars(args)
    data_reports: dict = {}
    run_manifest['data'] = {'dataset': args.dataset, 'dataset_config': args.dataset_config, 'split_reports': data_reports}
    save_run_manifest(manifest_path, run_manifest)
    print(f'[moe][track] run manifest -> {manifest_path}')
    rng = torch.Generator(device=device).manual_seed(args.seed)
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
        if y[:, -1].ne(cfg.ignore_index).any():
            y = y.clone()
            y[:, -1] = cfg.ignore_index
        if step % args.stats_every == 0:
            act_stats.attach(reset=not stats_windowed)
            stats_windowed = True
        logits, loss, aux, tpe_list = model(x, y, return_aux=True, return_logits=False)
        del logits
        loss.backward()
        if step % args.stats_every == 0:
            act_stats.detach()
        expert_cv = _expert_cv_per_block(tpe_list)
        exp_cv = expert_cv[-1]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val = loss.detach().item()
        aux_val = aux.detach().item()
        loss_spike.update(loss_val)
        history.append({'step': step, 'loss': loss_val, 'aux': aux_val, 'expert_cv': exp_cv, 'expert_cv_per_block': expert_cv, 'n_loss_spikes': loss_spike.n_spikes})
        if step % args.log_every == 0:
            print(f'  [moe] step={step:4d} loss={loss_val:.4f} aux={aux_val:.4f} expert_cv={exp_cv:.3f}', flush=True)
    with open(Path(args.output_dir) / 'history.json', 'w') as f:
        json.dump(history, f)
    # Held-out validation loss: real data comes from the chronological
    # validation split; the synthetic corpus uses a generator (seed + 1) the
    # optimizer never saw. No gradients flow from either.
    val_loss = None
    if args.val_batches > 0:
        model.eval()
        val_losses = []
        with torch.no_grad():
            if args.dataset:
                val_loader = open_data_split('validation', args)
                rval = getattr(val_loader, 'split_report', None)
                if rval:
                    data_reports['validation'] = rval
                    print(f'[moe][data] validation split: {rval}')
                val_iter = iter(val_loader)
            else:
                val_iter = None
            val_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
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
                val_losses.append(model(vx, vy, return_logits=False)[1].item())
        model.train()
        val_loss = sum(val_losses) / len(val_losses)
        print(f'[moe][val] held-out loss={val_loss:.4f}')
    s = act_stats.summary()
    if val_loss is not None:
        s['val_loss'] = val_loss
    with open(Path(args.output_dir) / 'stats.json', 'w') as f:
        json.dump(s, f, indent=2, default=str)
    run_manifest['data']['split_reports'] = data_reports
    run_manifest['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    run_manifest['results'] = {
        'steps': args.max_steps,
        'final_loss': history[-1]['loss'] if history else None,
        'final_expert_cv': history[-1]['expert_cv'] if history else None,
        'val_loss': val_loss,
        'n_loss_spikes': loss_spike.n_spikes,
    }
    save_run_manifest(manifest_path, run_manifest)
    print(f'[done] saved to {args.output_dir} | n_loss_spikes={loss_spike.n_spikes}')

def _expert_cv(tpe) -> float:
    if not isinstance(tpe, torch.Tensor):
        return 0.0
    return (tpe.std() / (tpe.mean() + 1e-06)).item()

def _expert_cv_per_block(tpe_list) -> list:
    cvs = [tpe.std() / (tpe.mean() + 1e-06) if isinstance(tpe, torch.Tensor) else None for tpe in tpe_list]
    tensors = [c.detach().reshape(()) for c in cvs if c is not None]
    if not tensors:
        return [0.0] * len(tpe_list)
    vals = iter(torch.stack(tensors).tolist())
    return [next(vals) if c is not None else 0.0 for c in cvs]
if __name__ == '__main__':
    main()
