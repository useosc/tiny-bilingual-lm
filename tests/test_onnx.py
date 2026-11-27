"""Sanity-check the exported ONNX decoder: greedy/sampled generation with KV cache."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizers import Tokenizer  # noqa: E402

WEB = "web/model"


def generate(prompt: str, max_new: int = 40, temperature: float = 0.8, top_k: int = 40) -> tuple[str, float]:
    cfg = json.load(open(f"{WEB}/config.json", encoding="utf-8"))
    tok = Tokenizer.from_file(f"{WEB}/tokenizer.json")
    sess = ort.InferenceSession(f"{WEB}/model_int8.onnx", providers=["CPUExecutionProvider"])
    L, kvh, dh = cfg["n_layers"], cfg["n_kv_heads"], cfg["head_dim"]

    past = {f"past_{k}_{i}": np.zeros((1, kvh, 0, dh), dtype=np.float32) for i in range(L) for k in "kv"}
    rng = np.random.default_rng(0)
    ids = tok.encode(prompt).ids
    logits = None
    pos = 0
    for tid in ids:
        feeds = {"input_ids": np.array([[tid]], np.int64), "pos": np.array([pos], np.int64), **past}
        out = sess.run(None, feeds)
        logits = out[0][0, -1]
        past = {f"past_{k}_{i}": out[1 + i] if k == "k" else out[1 + L + i] for i in range(L) for k in "kv"}
        pos += 1

    t0 = time.time()
    out_ids = []
    for _ in range(max_new):
        v = logits.astype(np.float64) / temperature
        top = np.argpartition(-v, top_k)[:top_k]
        p = np.exp(v[top] - v[top].max())
        p /= p.sum()
        tid = int(rng.choice(top, p=p))
        out_ids.append(tid)
        feeds = {"input_ids": np.array([[tid]], np.int64), "pos": np.array([pos], np.int64), **past}
        out = sess.run(None, feeds)
        logits = out[0][0, -1]
        past = {f"past_{k}_{i}": out[1 + i] if k == "k" else out[1 + L + i] for i in range(L) for k in "kv"}
        pos += 1
    dt = time.time() - t0
    return tok.decode(out_ids), len(out_ids) / dt


if __name__ == "__main__":
    for prompt in ["人工智能是", "The quick brown fox"]:
        text, speed = generate(prompt)
        print(f"PROMPT: {prompt}\n{text}\n({speed:.1f} tok/s)\n")
