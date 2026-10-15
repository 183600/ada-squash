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
from adasquash import GPT, markov_batch
from adasquash.model import MLP
from scripts.eval_downstream import load_ckpt

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

def make_dataset_probe(dataset, dataset_config, tokenizer, batch_size, seq_len):
    from scripts.train_gpt2 import build_dataloader
    loader = build_dataloader(dataset, dataset_config, seq_len, batch_size, tokenizer=tokenizer)
    it = iter(loader)

    def _next():
        nonlocal it
        try:
            return next(it)[0]
        except StopIteration:
            it = iter(loader)
            return next(it)[0]
    return _next

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--probe', default='markov', choices=['markov', 'dataset', 'random'], help="probe corpus. 'random' is uniform tokens: it has no next-token signal, so ppl_fp ~= vocab_size and the PPL drop is unmeasurable (kept only for parity)")
    p.add_argument('--dataset', default='HuggingFaceFW/fineweb-edu')
    p.add_argument('--dataset-config', default='sample-10BT')
    p.add_argument('--tokenizer', default='gpt2', help='tokenizer for --probe dataset; a checkpoint that recorded the one it was trained with wins')
    p.add_argument('--n-states', type=int, default=64, help="state count of the synthetic 'markov' probe")
    p.add_argument('--max-probe-coverage', type=float, default=0.5, help='reject a probe whose tokens are more than this fraction distinct (uniform noise is ~1.0)')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--seq-len', type=int, default=512)
    p.add_argument('--n-batches', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='runs/w8a8/results.json')
    args = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)

    def make_probe(rng, cfg, dataset_probe):
        if args.probe == 'random':
            return _probe_random(rng, cfg.vocab_size, args.batch_size, args.seq_len, device)
        if args.probe == 'markov':
            return _probe_markov(rng, args.batch_size, args.seq_len, device, n_states=args.n_states)
        return dataset_probe()
    results = []
    for ckpt_path in args.ckpts:
        rng = torch.Generator(device=device).manual_seed(args.seed)
        model, ckpt_args = load_ckpt(ckpt_path, device=device)
        dataset_probe = make_dataset_probe(args.dataset, args.dataset_config, ckpt_args.get('tokenizer') or args.tokenizer, args.batch_size, args.seq_len) if args.probe == 'dataset' else None
        batches = []
        for _ in range(max(1, args.n_batches)):
            x = make_probe(rng, model.cfg, dataset_probe)
            batches.append(x)
        # Reject a signal-free probe *before* measuring it. The coverage only
        # depends on the corpus, so running the fp32 and the W8A8 pass first
        # and rejecting afterwards threw away the whole measurement (and every
        # other checkpoint's, since the aggregate is written at the end) after
        # paying for all of it.
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
        r = {'ckpt': ckpt_path, 'probe': args.probe, 'ppl_fp': ppl_fp, 'ppl_w8a8': ppl_q, 'ppl_drop': ppl_q - ppl_fp, 'ppl_drop_pct': 100.0 * (ppl_q - ppl_fp) / ppl_fp, 'probe_token_coverage': coverage}
        if ppl_fp >= 0.5 * model.cfg.vocab_size:
            print(f'[w8a8] WARNING: {ckpt_path} ppl_fp={ppl_fp:.1f} is at the vocab_size={model.cfg.vocab_size} ceiling - the probe is fine (coverage={coverage:.1%}) but this checkpoint appears untrained, so ppl_drop_pct is not meaningful yet', file=sys.stderr)
        print(f'[w8a8] {ckpt_path}: {r}')
        results.append(r)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f'[done] saved to {out_path}')
if __name__ == '__main__':
    main()
