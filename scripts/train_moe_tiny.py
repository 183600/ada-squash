"""
MoE-tiny: a tiny 8-expert top-2 MoE block layered on top of the GPT
backbone. Used to study the expert-load-imbalance hypothesis from the
DyT++ proposal: when each expert sees only ~1/N of the tokens, the
activation statistics per expert can drift much more than in dense
training, and a *per-token* normalization like RMSNorm/LN runs on a
distribution that's a noisy mixture of expert distributions.

DyT-CD, being purely element-wise, sidesteps this entirely.

Note on the metric: what is actually measured is the coefficient of
variation of tokens-per-expert (`expert_cv`), i.e. *routing load
imbalance*, recorded per block. A per-expert activation statistic is not
computed here - the second norm runs on the residual stream *before*
routing, so there is no per-expert normalized tensor to measure.

Usage:
    python scripts/train_moe_tiny.py --norm dytcd --n-experts 8 \
        --batch-size 4 --seq-len 1024 --max-steps 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from adasquash import build_norm, markov_batch
from adasquash.attention import CausalSelfAttention
from adasquash.model import (
    GPTConfig, build_norm_kwargs, fused_cross_entropy_mean, _LayerScaleResidual,
)
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
        # The router distribution is needed twice - once for the top-k gate
        # values and once for the mean probability in the aux loss - so it is
        # computed once here and reused.
        probs = logits.softmax(-1)
        # Top-k routing
        if self.top_k == 1:
            scores, indices = probs.max(dim=-1, keepdim=True)
        else:
            scores, indices = probs.topk(self.top_k, dim=-1)
        # Renormalise the selected gate values so they sum to 1 (Switch /
        # Mixtral convention).
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        # Expert computation. Tokens are grouped by the expert(s) they route
        # to, so each expert runs its GEMMs on the [n_e, C] slice that
        # actually selects it rather than on all N tokens. The three stacked
        # expert weights stay as [E, ...]; `w1[e]` / `w3[e]` / `w2[e]` are
        # the [C, H] / [C, H] / [H, C] matrices of expert `e`, matching the
        # `...ech` / `...ehc` index order the einsum form used.
        n_tokens = x_flat.shape[0]
        n_experts = self.n_experts
        k = self.top_k
        flat_expert = indices.reshape(-1)              # [N * k]
        flat_token = torch.arange(n_tokens, device=x.device).repeat_interleave(k)
        flat_gate = scores.reshape(-1)                 # [N * k]
        # Sort by expert id so each expert's (token, gate) rows are contiguous.
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
                gate = gate_of[start:end].unsqueeze(-1)  # [n_e, 1]
                xe = x_flat.index_select(0, tok)         # [n_e, C]
                # `h` is the silu activation's own buffer: the gate product
                # folds into it with `mul_`, and the expert output is scaled
                # on its own buffer too.
                h = F.silu(xe @ self.experts_w1[e])
                h.mul_(xe @ self.experts_w3[e])
                y.index_add_(0, tok, (h @ self.experts_w2[e]).mul_(gate))
            start = end
        y = self.drop(y)
        y = y.view(B, T, C)

        # Load balancing auxiliary loss (Switch Transformer style):
        # aux = n_experts * sum_i(f_i * P_i) where f_i is the fraction of
        # *tokens* routed to expert i and P_i is the mean router
        # probability. f_i counts tokens, so it is built from the top-k
        # selection and detached: its value is a routing decision, not a
        # differentiable quantity, and letting gradients flow through it
        # makes the loss push the gates as well as the probabilities.
        # Summing the renormalised gate weights instead would silently
        # report routing *mass* under the name `tokens_per_expert` (and
        # `expert_cv` is defined on this quantity), so the two quantities
        # are kept distinct here.
        tokens_per_expert = torch.zeros_like(logits[0]).scatter_add_(
            0, indices.reshape(-1), torch.ones_like(flat_gate)
        ).detach()
        f_i = tokens_per_expert / n_tokens
        p_i = probs.mean(dim=0)
        aux = self.n_experts * (f_i * p_i).sum()
        return y, aux, tokens_per_expert


class MoEBlock(nn.Module):
    def __init__(self, cfg: GPTConfig, n_experts: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        norm_kwargs = build_norm_kwargs(cfg, layer_idx)

        self.norm1 = build_norm(cfg.norm_name, cfg.dim, **norm_kwargs)
        self.attn = CausalSelfAttention(cfg.dim, cfg.n_heads,
                                        qk_norm=cfg.qk_norm,
                                        alpha_init=cfg.qk_alpha_init,
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
        if self.use_layerscale:
            # One fused pass, as in the dense `Block`.
            x = _LayerScaleResidual.apply(x, a, self.ls1)
        else:
            x = x + a
        h = self.norm2(x)
        m, aux, tpe = self.moe(h)
        if self.use_layerscale:
            x = _LayerScaleResidual.apply(x, m, self.ls2)
        else:
            x = x + m
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
            **build_norm_kwargs(cfg, cfg.n_layers, is_final_norm=True),
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

    def forward(self, x, y=None, return_aux=False, return_logits=True):
        B, T = x.shape
        z = self.tok_emb(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        # In-place, like `GPT.forward`: the embedding's own backward needs only
        # the indices, so the row-sum never has to be preserved.
        z.add_(self.pos_emb(pos))
        z = self.drop(z)

        aux_total = 0.0
        tpe_list = []
        for blk in self.blocks:
            z, aux, tpe = blk(z)
            aux_total = aux_total + aux
            tpe_list.append(tpe)
        z = self.norm_f(z)
        loss = None
        if y is not None and not return_logits:
            # Fused with the output projection.
            loss = fused_cross_entropy_mean(
                z.reshape(-1, z.size(-1)), self.lm_head.weight, y.reshape(-1),
                ignore_index=self.cfg.ignore_index,
            ) + 1e-3 * aux_total
            logits = None
        else:
            logits = self.lm_head(z)
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
    p.add_argument("--probe", default="markov",
                   choices=["markov", "random"],
                   help="corpus. 'random' is uniform token ids: it has no "
                        "next-token signal, so the LM loss is pinned at "
                        "log(vocab_size) and the run learns nothing")
    p.add_argument("--dataset", default=None,
                   help="optional HF dataset name; when given, replaces the "
                        "synthetic corpus with packed real text")
    p.add_argument("--dataset-config", default=None)
    p.add_argument("--tokenizer", default="gpt2")
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
    # Same parameter grouping as the Phase-1 driver and the ablation runner:
    # weight decay on the matmul weights only. A single group would decay
    # the norm gains (RMSNorm/LN `weight`, DyT `gamma`) and the LayerScale
    # gates as well, so the norm comparison this study reports would not be
    # comparable with the training runs it is meant to explain - and it would
    # shrink the very residual gates the depth-conditioned axis depends on.
    from scripts.train_gpt2 import build_optimizer
    opt = build_optimizer(model, args.lr, 0.1, 0.9, 0.95)
    act_stats = ActivationStats(model)
    loss_spike = LossSpikeCounter()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rng = torch.Generator(device=device).manual_seed(args.seed)

    data_iter = None
    if args.dataset:
        from scripts.train_gpt2 import build_dataloader
        loader = build_dataloader(
            args.dataset, args.dataset_config, args.seq_len,
            args.batch_size, tokenizer=args.tokenizer,
        )
        data_iter = iter(loader)

    def next_batch():
        if data_iter is None:
            if args.probe == "random":
                # Uniform token ids: no next-token signal, so the LM loss is
                # pinned at log(vocab_size) and the run learns nothing. Kept
                # only for parity with the fp8 / w8a8 probes.
                x = torch.randint(
                    0, cfg.vocab_size, (args.batch_size, args.seq_len),
                    generator=rng, device=device,
                )
                y = torch.full_like(x, cfg.ignore_index)
                y[:, :-1] = x[:, 1:]
                return x, y
            # Learnable corpus: uniform token ids would pin the loss at
            # log(vocab_size), which carries no information about the norm
            # under test and gives the router nothing to learn from either.
            return markov_batch(
                rng, cfg.ignore_index, args.batch_size, args.seq_len, device
            )
        try:
            x, y = next(data_iter)
        except StopIteration:
            x, y = next(iter(loader))
        # The HF loader yields CPU tensors; the model, the synthetic corpus
        # and `rng` all live on `device`, so leaving the batch on the host
        # dies on the first step with an embedding device mismatch.
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    history = []
    # The reported activation statistics have to describe the same run the
    # loss / aux / expert_cv curves describe. Re-attaching with a reset per
    # window (the `ActivationStats.attach` default) would leave `stats.json`
    # reporting only the last window - step 800 of a 1000-step run - so the
    # first window resets the meters and the rest pool into it, the same
    # contract `scripts/ablation.py`'s runner relies on.
    stats_windowed = False
    for step in range(args.max_steps):
        x, y = next_batch()
        # The shift is already done by both corpus builders; the final
        # position has no successor and carries `cfg.ignore_index` (out of
        # vocabulary) rather than token 0, which is a real GPT-2 token ("!")
        # that would otherwise be trained on.
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
        # Log expert load imbalance. Reported per block: the motivating
        # hypothesis is a drift *with depth*, which a single last-block
        # number cannot show.
        expert_cv = _expert_cv_per_block(tpe_list)
        exp_cv = expert_cv[-1]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss_val = loss.detach().item()
        aux_val = aux.detach().item()
        loss_spike.update(loss_val)
        history.append({"step": step, "loss": loss_val,
                        "aux": aux_val,
                        "expert_cv": exp_cv,
                        "expert_cv_per_block": expert_cv,
                        "n_loss_spikes": loss_spike.n_spikes})
        if step % args.log_every == 0:
            print(f"  [moe] step={step:4d} loss={loss_val:.4f} aux={aux_val:.4f} "
                  f"expert_cv={exp_cv:.3f}", flush=True)
    with open(Path(args.output_dir) / "history.json", "w") as f:
        json.dump(history, f)
    s = act_stats.summary()
    with open(Path(args.output_dir) / "stats.json", "w") as f:
        json.dump(s, f, indent=2, default=str)
    print(f"[done] saved to {args.output_dir} | n_loss_spikes={loss_spike.n_spikes}")


def _expert_cv(tpe) -> float:
    """Coefficient of variation of tokens-per-expert (load imbalance)."""
    if not isinstance(tpe, torch.Tensor):
        return 0.0
    return (tpe.std() / (tpe.mean() + 1e-6)).item()


def _expert_cv_per_block(tpe_list) -> list:
    """`_expert_cv` for every block, with a single device-to-host transfer.

    Each block's CoV is a 0-dim tensor. They are stacked and converted
    together instead, which is one transfer for the whole model.
    """
    cvs = [
        tpe.std() / (tpe.mean() + 1e-6)
        for tpe in tpe_list if isinstance(tpe, torch.Tensor)
    ]
    if not cvs:
        return [0.0] * len(tpe_list)
    if len(cvs) == 1:
        return [float(cvs[0].item())]
    return torch.stack([c.detach() for c in cvs]).tolist()


if __name__ == "__main__":
    main()
