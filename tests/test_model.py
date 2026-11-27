"""Shape / cache / generation tests that do not need training data."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import build_model, count_params
from model.base import ModelConfig


def test_dense_forward_backward():
    cfg = ModelConfig()
    m = build_model(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, loss, kvs = m(x, y)
    assert logits.shape == (2, 32, cfg.vocab_size)
    assert loss.item() > 0
    loss.backward()
    assert m.embed_tokens.weight.grad is not None


def test_cache_matches_full_forward():
    cfg = ModelConfig()
    m = build_model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 16))
    with torch.no_grad():
        full, _, _ = m(x)
        past = None
        outs = []
        for i in range(16):
            out, _, past = m(x[:, i : i + 1], past_kvs=past, use_cache=True)
            outs.append(out)
        stepwise = torch.cat(outs, dim=1)
    assert torch.allclose(full, stepwise, atol=2e-4), "KV cache diverges from full forward"


def test_param_counts():
    assert count_params(build_model(ModelConfig())) == 25_172_352
    assert count_params(build_model(ModelConfig(arch="looped", n_unique_layers=2, n_loops=4))) == 15_730_560
    assert count_params(build_model(ModelConfig(arch="moe", n_experts=4, top_k=2, ffn_hidden=768))) == 44_059_008


def test_generate_shape():
    cfg = ModelConfig()
    m = build_model(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = m.generate(x, max_new_tokens=5)
    assert out.shape == (1, 9)


if __name__ == "__main__":
    test_dense_forward_backward()
    test_cache_matches_full_forward()
    test_param_counts()
    test_generate_shape()
    print("all model tests passed")
