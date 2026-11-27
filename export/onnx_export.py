"""Export a trained dense checkpoint to ONNX (single decode step with KV cache).

The graph takes one token + its position + past K/V and returns next-token
logits plus updated K/V. Attention is written out explicitly (no SDPA) so the
graph stays portable to onnxruntime-web.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import build_model  # noqa: E402
from model.base import ModelConfig  # noqa: E402
from model.layers import apply_rope  # noqa: E402


class DecodeGraph(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model
        cfg = model.cfg
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.dh = cfg.d_model // cfg.n_heads
        self.rep = self.n_heads // self.n_kv
        self.scale = self.dh ** -0.5

    def forward(self, input_ids, pos, *past):
        m = self.m
        x = m.embed_tokens(input_ids)  # [1, 1, d]
        cos = m.rope_cos.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        sin = m.rope_sin.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        new = []
        for i, blk in enumerate(m.blocks):
            h = blk.attn_norm(x)
            q = blk.attn.q_proj(h).view(1, 1, self.n_heads, self.dh).transpose(1, 2)
            k = blk.attn.k_proj(h).view(1, 1, self.n_kv, self.dh).transpose(1, 2)
            v = blk.attn.v_proj(h).view(1, 1, self.n_kv, self.dh).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            k = torch.cat([past[2 * i], k], dim=2)
            v = torch.cat([past[2 * i + 1], v], dim=2)
            new += [k, v]
            kk = k.repeat_interleave(self.rep, dim=1)
            vv = v.repeat_interleave(self.rep, dim=1)
            att = torch.softmax(torch.matmul(q, kk.transpose(-1, -2)) * self.scale, dim=-1)
            y = torch.matmul(att, vv).transpose(1, 2).reshape(1, 1, -1)
            x = x + blk.attn.o_proj(y)
            x = x + blk.ffn(blk.ffn_norm(x))
        x = m.norm(x)
        return (m.lm_head(x), *new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="web/model/model.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    graph = DecodeGraph(model).eval()

    L = cfg.n_layers
    kvh, dh = cfg.n_kv_heads, cfg.d_model // cfg.n_heads
    past = [torch.zeros(1, kvh, 0, dh) for _ in range(2 * L)]
    ids = torch.zeros(1, 1, dtype=torch.long)
    pos = torch.zeros(1, dtype=torch.long)

    k_names = [f"past_k_{i}" for i in range(L)]
    v_names = [f"past_v_{i}" for i in range(L)]
    new_k = [f"new_k_{i}" for i in range(L)]
    new_v = [f"new_v_{i}" for i in range(L)]

    dynamic_axes = {"input_ids": {1: "seq"}, "pos": {0: "seq"}, "logits": {0: "seq"}}
    for i in range(L):
        dynamic_axes[f"past_k_{i}"] = {2: "past_len"}
        dynamic_axes[f"past_v_{i}"] = {2: "past_len"}
        dynamic_axes[f"new_k_{i}"] = {2: "total_len"}
        dynamic_axes[f"new_v_{i}"] = {2: "total_len"}

    torch.onnx.export(
        graph,
        (ids, pos, *past),
        args.out,
        input_names=["input_ids", "pos"] + k_names + v_names,
        output_names=["logits"] + new_k + new_v,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print("exported", args.out, round(os.path.getsize(args.out) / 1e6, 1), "MB")

    import json
    import shutil

    meta = {
        "arch": cfg.arch,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "n_kv_heads": cfg.n_kv_heads,
        "head_dim": cfg.d_model // cfg.n_heads,
        "d_model": cfg.d_model,
        "vocab_size": cfg.vocab_size,
        "max_seq_len": cfg.max_seq_len,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "step": ck.get("step"),
    }
    out_dir = os.path.dirname(args.out)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if os.path.exists("tokenizer/artifacts/tokenizer.json"):
        shutil.copy("tokenizer/artifacts/tokenizer.json", os.path.join(out_dir, "tokenizer.json"))
    print("wrote", os.path.join(out_dir, "config.json"))


if __name__ == "__main__":
    main()
