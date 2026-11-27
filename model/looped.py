"""Looped (weight-shared / recurrent-depth) transformer.

The same block is applied ``n_loops`` times, trading unique parameters for
effective depth.  This mirrors "Looped Transformers are Better at Learning
Learning Algorithms" (Giannou et al., 2023) and the recurrent-depth line of
work (Huginn, 2025; Ouro, 2025).
"""
from __future__ import annotations

import torch.nn as nn

from .base import BaseLM, ModelConfig
from .layers import SwiGLU, TransformerBlock


class LoopedLM(BaseLM):
    supports_cache = False  # each loop needs its own KV, so we recompute

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.d_model, cfg.n_heads, cfg.n_kv_heads, SwiGLU(cfg.d_model, cfg.ffn_hidden))
                for _ in range(cfg.n_unique_layers)
            ]
        )

    def _run_blocks(self, x, cos, sin, past_kvs=None, use_cache=False):
        for _ in range(self.cfg.n_loops):
            for blk in self.blocks:
                x, _ = blk(x, cos, sin)
        return x, [None] * self.cfg.n_unique_layers
