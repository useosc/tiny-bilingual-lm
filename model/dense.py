"""Dense decoder-only model (Llama-style block stack)."""
from __future__ import annotations

import torch.nn as nn

from .base import BaseLM, ModelConfig
from .layers import SwiGLU, TransformerBlock


class DenseLM(BaseLM):
    supports_cache = True

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, SwiGLU(cfg.d_model, cfg.ffn_hidden))
                for _ in range(cfg.n_layers)
            ]
        )

    def _run_blocks(self, x, cos, sin, past_kvs=None, use_cache=False):
        kvs = []
        for i, blk in enumerate(self.blocks):
            x, kv = blk(x, cos, sin, past_kv=past_kvs[i] if past_kvs is not None else None, use_cache=use_cache)
            kvs.append(kv)
        return x, kvs
