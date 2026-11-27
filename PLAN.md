# PLAN — autonomous build plan

Goal: a small bilingual LM stack that trains on a 6 GB RTX 4050 and can be
lifted to a rented cluster later. This file is the source of truth across
agent sessions; update it as phases complete.

## Phase 0 — environment (done)
- [x] GPU: RTX 4050 Laptop 6 GB, CC 8.9; 24 CPU threads; 40 GB RAM; D: 219 GB free
- [x] torch 2.13.0+cu126, transformers 5.16, datasets 5.0, tokenizers 0.23
- [x] GitHub auth: user `useosc` (PAT scopes: repo, workflow)

## Phase 1 — data (target <= 200 GB, actual ~4 GB)
- [x] download 1 SkyPile shard (zh, 1.3 GB jsonl) + 1 FineWeb-Edu shard (en, 1.4 GB parquet)
- [ ] clean + dedup -> data/processed/{zh,en}.jsonl.gz + stats.json
- [ ] train 32k BPE on 120 M chars (60 zh + 60 en)
- [ ] tokenize -> data/tokenized/{zh,en}.bin
- [ ] mix 600 M tokens @ zh:en = 50:50 -> train.bin / val.bin

## Phase 2 — dense 25M (baseline)
- [ ] smoke test (50 steps) to validate trainer + throughput
- [ ] full run: 24k steps x 24 576 tokens = ~590 M tokens
- [ ] eval: val loss / ppl / bits-per-byte; sample generations

## Phase 3 — MoE variant
- [ ] 4 experts, top-2, aux loss 0.01, shorter run (~120 M tokens)
- [ ] compare val loss vs dense at equal compute (report both)

## Phase 4 — looped variant
- [ ] 2 unique blocks x 4 loops, ~120 M tokens
- [ ] compare params vs effective depth

## Phase 5 — packaging
- [ ] ONNX export with KV cache, int8 dynamic quant
- [ ] JS BPE + chat page, GitHub Pages
- [ ] docs (data/tokenizer/architecture/training/eval/deployment) + model cards

## Phase 6 — push
- [ ] create github.com/useosc/tiny-bilingual-lm, push commits, enable Pages
- [ ] final verification: fresh clone -> train 100 steps

## Decisions already made (do not re-litigate)
- Model family: Llama-style pre-norm, RMSNorm, SwiGLU, RoPE, GQA, tied embeddings.
- Dense 25M: d_model 384, 8 layers, 6 heads (head_dim 64), 2 KV heads, ffn 1024.
- Tokenizer: byte-level BPE 32768, NFC, specials `<|endoftext|> <|im_start|> <|im_end|>`.
- Data: SkyPile-150B (zh) + FineWeb-Edu (en), 50/50 by tokens, seq len 512.
- Optimizer: AdamW fused, lr 1e-3 cosine to 1e-4, warmup 300, wd 0.1, clip 1.0, bf16.
- Everything runs from repo root; artifacts are gitignored.
