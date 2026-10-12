"""
MoE-tiny: a tiny 8-expert top-2 MoE block layered on top of the GPT
backbone. Used to study the "expert variance drift" hypothesis from the
DyT++ proposal: when each expert sees only ~1/N of the tokens, the
activation statistics per expert can drift much more than in dense
training, and a *per-token* normalization like RMSNorm/LN runs on a
distribution that's a noisy mixture of expert distributions.

DyT-CD, being purely element-wise, sidesteps this entirely.

Usage:
    python scripts/train_moe_tiny.py --norm dytcd --n-experts 8 \
        --batch-size 4 --seq-len 1024 --max-steps 1000
"""

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
from torch.optim import AdamW

from adasquash import build_norm
from adasquash.attention import CausalSelfAttention
from adasquash.model import GPTConfig, build_norm_kwargs
from adasquash.utils import (
    ActivationStats, LossSpikeCounter, count_parameters,
)


# ---------------------------------------------------------------------------
# MoE block
# ---------------------------------------------------------------------------
class MoELayer(nn.Module):
    """
    Top-2 MoE with auxiliary-load-balancing loss. For simplicity the
    router is a single linear; the experts are SwiGLU MLPs of the same
    hidden dim as the dense baseline.
    """

    def __init__(self, dim: int, hidden: int, n_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.0):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, n_experts, bias=False)
        # Each expert is a SwiGLU MLP
        hidden = int(hidden * 2 / 3)
        hidden = (hidden + 7) & ~7
        self.experts_w1 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.experts_w2 = nn.Parameter(torch.empty(n_experts, hidden, dim))
        self.experts_w3 = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.drop = nn.Dropout(dropout)
        # init
        for p in (self.experts_w1, self.experts_w2, self.experts_w3):
            nn.init.normal_(p, std=0.02)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        B, T, C = x.shape
        x_flat = x.view(B * T, C)  # [N, C]
        logits = self.router(x_flat)  # [N, E]
        # Top-k routing
        if self.top_k == 1:
            scores, indices = logits.softmax(-1).max(dim=-1, keepdim=True)
        else:
            scores, indices = logits.softmax(-1).topk(self.top_k, dim=-1)
        # Renormalise the selected gate values so they sum to 1 (Switch /
        # Mixtral convention).
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Build a [N, E] sparse one-hot routing tensor
        route = torch.zeros_like(logits).scatter_(-1, indices, scores)
        # Expert computation (vectorized: all experts evaluated, gated by
        # routing matrix).
        # `einsum` (not `F.linear`): the stacked expert weights are 3-D, so
        # each token must be applied to every expert explicitly.
        h1 = torch.einsum("nc,ech->neh", x_flat, self.experts_w1)  # [N, E, H]
        h3 = torch.einsum("nc,ech->neh", x_flat, self.experts_w3)  # [N, E, H]
        h = F.silu(h1)
        h.mul_(h3)  # [N, E, H] - in place, silu only needs its own input
        y_e = torch.einsum("neh,ehc->nec", h, self.experts_w2)  # [N, E, C]
        # Apply routing weights
        y = (y_e * route.unsqueeze(-1)).sum(dim=1)  # [N, C]
        y = self.drop(y)
        y = y.view(B, T, C)

        # Load balancing auxiliary loss (Switch Transformer style):
        # aux = n_experts * sum_i(f_i * P_i) where f_i is the fraction of
        # tokens *dispatched* to expert i and P_i is the mean router
        # probability. `route` rows are renormalised to sum to `top_k`, not
        # 1, so f_i has to be divided by `top_k` for the usual formulation
        # to hold - otherwise the loss is `top_k` times too large and its
        # 1e-3 weight silently behaves like a different, unrecorded one.
        tokens_per_expert = route.sum(0)  # [E]
        n_tokens = x_flat.shape[0]
        f_i = tokens_per_expert / (n_tokens * self.top_k)
        p_i = logits.softmax(-1).mean(dim=0)
        aux = self.n_experts * (f_i * p_i).sum()
        return y, aux, tokens_per_expert.detach()


class MoEBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, n_experts: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        norm_kwargs = build_norm_kwargs(cfg, layer_idx)

        self.norm1 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.attn = CausalSelfAttention(cfg.dim, cfg.n_heads,
                                        qk_norm=cfg.qk_norm,
                                        dropout=cfg.dropout)
        self.norm2 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.moe = MoELayer(cfg.dim, hidden, n_experts=n_experts, top_k=2,
                            dropout=cfg.dropout)
        self.use_layerscale = cfg.use_layerscale
        if self.use_layerscale:
            self.ls1 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale_init))
            self.ls2 = nn.Parameter(torch.full((cfg.dim,), cfg.layerscale_init))

    def forward(self, x, freqs_cis=None):
        h = self.norm1(x)
        a, _ = self.attn(h, freqs_cis=freqs_cis)
        x = x + (self.ls1 * a if self.use_layerscale else a)
        h = self.norm2(x)
        m, aux, tpe = self.moe(h)
        x = x + (self.ls2 * m if self.use_layerscale else m)
        return x, aux, tpe


class MoEGPT(nn.Module):
    """A GPT-style decoder with MoE MLPs in every block."""

    def __init__(self, cfg: GPTConfig, n_experts: int = 8):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [MoEBlock(cfg, n_experts, i) for i in range(cfg.n_layers)]
        )
        self.norm_f = build_norm(
            cfg.norm_name, cfg.dim,
            **build_norm_kwargs(cfg, cfg.n_layers - 1),
        )
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None, return_aux=False):
        B, T = x.shape
        z = self.tok_emb(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        z = z + self.pos_emb(pos)
        z = self.drop(z)

        aux_total = 0.0
        tpe_list = []
        for blk in self.blocks:
            z, aux, tpe = blk(z)
            aux_total = aux_total + aux
            tpe_list.append(tpe)
        z = self.norm_f(z)
        logits = self.lm_head(z)
        loss = None
        if y is not None:
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                 y.view(-1), ignore_index=self.cfg.ignore_index)
            loss = ce + 1e-3 * aux_total
        if return_aux:
            return logits, loss, aux_total, tpe_list
        return logits, loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--norm", default="dytcd", choices=["ln", "rmsnorm", "dyt", "dytc", "dytd", "dytcd"])
    p.add_argument("--n-experts", type=int, default=8)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--stats-every", type=int, default=200)
    p.add_argument("--output-dir", default="runs/moe-tiny")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    cfg = GPTConfig(
        vocab_size=50304, dim=args.dim, n_layers=args.n_layers,
        n_heads=args.n_heads, max_seq_len=args.seq_len,
        mlp_ratio=4.0, norm_name=args.norm,
    )
    model = MoEGPT(cfg, n_experts=args.n_experts).to(device)
    print(f"[moe] total params = {count_parameters(model)/1e6:.2f}M")
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    history = []
    for step in range(args.max_steps):
        x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), generator=rng, device=device)
        # Shift-by-one targets. The final position has no successor, so it is
        # padded with `cfg.ignore_index` (out-of-vocab) rather than token 0,
        # which is a real GPT-2 token ("!") that would otherwise be trained on.
        y = torch.full(
            (args.batch_size, args.seq_len), cfg.ignore_index,
            dtype=torch.long, device=device,
        )
        y[:, :-1] = x[:, 1:]
        if step % args.stats_every == 0:
            act_stats.attach()
        logits, loss, aux, tpe_list = model(x, y, return_aux=True)
        del logits
        loss.backward()
        if step % args.stats_every == 0:
            act_stats.detach()
        # Log expert variance drift
        tpe = tpe_list[-1]
        exp_cv = (tpe.std() / (tpe.mean() + 1e-6)).item() if isinstance(tpe, torch.Tensor) else 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_spike.update(loss.item())
        history.append({"step": step, "loss": loss.item(), "aux": aux.detach().item(),
                        "expert_cv": exp_cv})
        if step % args.log_every == 0:
            print(f"  [moe] step={step:4d} loss={loss.item():.4f} aux={aux.detach().item():.4f} "
                  f"expert_cv={exp_cv:.3f}", flush=True)
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f)
    s = act_stats.summary()
    with open(Path(args.output_dir) / "stats.json", "w") as f:
        json.dump(s, f, indent=2, default=str)
    print(f"[done] saved to {args.output_dir}")


if __name__ == "__main__":
    main()
