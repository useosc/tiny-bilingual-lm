"""Core transformer building blocks (Llama-style, no external model deps).

Everything here is deliberately small and readable so the same file can be
reused by the dense, MoE and looped variants.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich, 2019; used by Llama)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape [max_seq_len, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)  # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D], cos/sin: [T, D]."""
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    return x * cos + rotate_half(x) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention with grouped KV heads (GQA)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or d_model // n_heads
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        offset = past_kv[0].shape[2] if past_kv is not None else 0
        cos, sin = cos[offset : offset + T], sin[offset : offset + T]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # Expand KV heads to query heads (GQA).
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # is_causal only valid when query length == key length (prefill); for
        # single-token decoding with a cache the full attention is correct.
        is_causal = T > 1 and past_kv is None
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y), new_kv


class SwiGLU(nn.Module):
    """Gated feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: attention + FFN (FFN is pluggable)."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, ffn: nn.Module):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, n_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = ffn

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        a, kv = self.attn(self.attn_norm(x), cos, sin, past_kv=past_kv, use_cache=use_cache)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv


def init_weights(module: nn.Module, n_layers: int = 1, std: float = 0.02) -> None:
    """GPT-2 style init with residual-scaled output projections."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
    # Scale the two projections that write into the residual stream.
    for name, p in module.named_parameters():
        if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
            with torch.no_grad():
                p.mul_(1.0 / math.sqrt(2 * n_layers))
