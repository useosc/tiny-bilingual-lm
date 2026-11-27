# tiny-bilingual-lm

A small, fully reproducible **Chinese–English** language-model stack that runs
end-to-end on a single **6 GB laptop GPU** (RTX 4050 Laptop):

```
open corpora -> clean/dedup -> 32k BPE -> packed tokens -> dense -> MoE -> looped -> ONNX -> browser demo
```

The point is not to beat Qwen. The point is to own every line of the pipeline,
so the same code can be moved to a rented A100/H100 cluster later without a
rewrite.

## Status

| stage | artifact | status |
|---|---|---|
| data pipeline | SkyPile-150B (zh, 421 M chars) + FineWeb-Edu (en, 2.12 B chars), cleaned & deduped | done: 424 k zh + 431 k en docs |
| tokenizer | 32k byte-level BPE, zh 1.62 chars/token, en 4.33 chars/token | done |
| packed data | 496 M train + 4 M val tokens (uint16), zh:en = 50:50 | done |
| dense 25M | ~25.2 M params, 512 ctx, 20k steps ≈ 492 M tokens | training |
| MoE 4-expert | 44.1 M total params, top-2 routing + aux loss | queued |
| looped | 2 unique layers x 4 loops (15.7 M params, 8 effective layers) | queued |
| demo | ONNX int8 (~76 MB) + static chat page | code ready, export pending |
| docs | 7 design docs with citations | done, results pending |

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt

python data/scripts/download.py --zh-shards 1 --en-shards 1
python data/scripts/clean.py --only zh
python data/scripts/clean_en_parallel.py
python tokenizer/train_tokenizer.py
python data/scripts/pack_tokens.py
python data/scripts/mix.py --total-tokens 500000000 --zh-ratio 0.5

python train/train.py --config configs/dense_25m.toml
python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m
```

## Why these choices

Every architectural decision (width, depth, heads, data volume, mixture,
optimizer) is written down with its source paper in [`docs/`](docs/):

- [01-data.md](docs/01-data.md) - corpora, licences, cleaning, zh/en ratio
- [02-tokenizer.md](docs/02-tokenizer.md) - vocab size, BPE, measured compression
- [03-model-architecture.md](docs/03-model-architecture.md) - dims, heads, GQA, MoE, looped
- [04-training.md](docs/04-training.md) - budget, LR, batch size, memory math
- [05-evaluation.md](docs/05-evaluation.md) - val loss, bits-per-byte, samples
- [06-deployment.md](docs/06-deployment.md) - ONNX export and browser inference
- [07-decisions-log.md](docs/07-decisions-log.md) - ADR: every choice + evidence

## Repository layout

```
model/       dense / MoE / looped implementations (pure PyTorch)
train/       trainer with AMP, grad-accum, cosine LR, resume, eval
data/        download -> clean -> tokenize -> mix
tokenizer/   BPE training
eval/        checkpoint evaluation and sample generation
export/      ONNX export for the web demo
web/         static chat page (GitHub Pages)
docs/        design decisions with citations
```

## Licence

MIT (code). Datasets keep their own licences - see `docs/01-data.md`.
