"""Top-k mixture-of-experts FFN + model.

Routing follows Switch Transformer / GShard (top-k, load-balancing auxiliary
loss) and Mixtral / DeepSeekMoE (top-2 experts per token, no capacity drop).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLM, ModelConfig
from .layers import SwiGLU, TransformerBlock


class MoEFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.router = nn.Linear(cfg.d_model, cfg.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.ffn_hidden) for _ in range(cfg.n_experts)]
        )
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        xf = x.reshape(-1, D)
        router_logits = self.router(xf)  # [N, E]
        probs = F.softmax(router_logits.float(), dim=-1).to(xf.dtype)
        topv, topi = probs.topk(self.top_k, dim=-1)  # [N, k]

        # Switch-Transformer style load balancing: E * sum(f_i * P_i).
        frac = F.one_hot(topi, self.n_experts).float().mean(0)  # [E]
        self.aux_loss = (frac * probs.mean(0)).sum() * self.n_experts

        out = torch.zeros_like(xf)
        for e in range(self.n_experts):
            token_idx, slot_idx = (topi == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            w = topv[token_idx, slot_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, self.experts[e](xf[token_idx]) * w)
        return out.view(B, T, D)


class MoELM(BaseLM):
    supports_cache = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, MoEFFN(cfg))
                for _ in range(cfg.n_layers)
            ]
        )

    def _run_blocks(self, x, cos, sin, past_kvs=None, use_cache=False):
        kvs = []
        aux = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, cos, sin, past_kv=past_kvs[i] if past_kvs is not None else None, use_cache=use_cache)
            kvs.append(kv)
            aux.append(blk.ffn.aux_loss)
        self.aux_loss = self.cfg.aux_loss_weight * torch.stack(aux).mean()
        return x, kvs
