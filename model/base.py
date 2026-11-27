"""Shared language-model scaffold: embeddings, blocks, head, sampling."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm, TransformerBlock, SwiGLU, build_rope_cache, init_weights


@dataclass
class ModelConfig:
    arch: str = "dense"
    vocab_size: int = 32768
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden: int = 1024
    max_seq_len: int = 512
    tie_embeddings: bool = True
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    # MoE
    n_experts: int = 4
    top_k: int = 2
    aux_loss_weight: float = 0.01
    # Looped
    n_unique_layers: int = 2
    n_loops: int = 4

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def num_layers_effective(self) -> int:
        if self.arch == "looped":
            return self.n_unique_layers * self.n_loops
        return self.n_layers


class BaseLM(nn.Module):
    """Pre-norm decoder-only transformer; subclasses fill ``self.blocks``."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.norm = RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.register_buffer(
            "rope_cos",
            build_rope_cache(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)[0],
            persistent=False,
        )
        self.register_buffer(
            "rope_sin",
            build_rope_cache(cfg.d_model // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)[1],
            persistent=False,
        )
        self.aux_loss: torch.Tensor | None = None

    # ------------------------------------------------------------------ init
    def init_weights(self) -> None:
        self.apply(lambda m: init_weights(m, n_layers=self.cfg.n_layers))

    # --------------------------------------------------------------- forward
    def _run_blocks(self, x, cos, sin, past_kvs=None, use_cache=False):
        raise NotImplementedError

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_kvs: list | None = None,
        use_cache: bool = False,
        pos: torch.Tensor | None = None,
    ):
        x = self.embed_tokens(idx)
        B, T = idx.shape
        offset = 0
        if past_kvs is not None and past_kvs[0] is not None:
            offset = past_kvs[0][0].shape[2]
        if pos is None:
            pos = torch.arange(offset, offset + T, device=idx.device)
        cos = self.rope_cos.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        sin = self.rope_sin.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        x, new_kvs = self._run_blocks(x, cos, sin, past_kvs=past_kvs, use_cache=use_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss, new_kvs

    # -------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int | None = 50,
        eos_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        self.eval()
        cache_ok = use_cache and getattr(self, "supports_cache", True)
        past = None
        for _ in range(max_new_tokens):
            if past is None or not cache_ok:
                cur = idx[:, -self.cfg.max_seq_len :]
            else:
                cur = idx[:, -1:]
            logits, _, past = self.forward(cur, past_kvs=past, use_cache=cache_ok)
            logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
