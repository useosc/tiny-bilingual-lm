"""Model factory."""
from __future__ import annotations

import torch.nn as nn

from .base import BaseLM, ModelConfig
from .dense import DenseLM
from .looped import LoopedLM
from .moe import MoELM

ARCHS = {"dense": DenseLM, "moe": MoELM, "looped": LoopedLM}


def build_model(cfg: ModelConfig) -> BaseLM:
    if cfg.arch not in ARCHS:
        raise ValueError(f"unknown arch {cfg.arch!r}; expected one of {list(ARCHS)}")
    model = ARCHS[cfg.arch](cfg)
    model.init_weights()
    return model


def count_params(model: nn.Module, non_embedding: bool = False) -> int:
    n = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n -= sum(p.numel() for p in model.embed_tokens.parameters())
        if not model.cfg.tie_embeddings:
            n -= sum(p.numel() for p in model.lm_head.parameters())
    return n
