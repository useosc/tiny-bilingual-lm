# tiny-bilingual-lm

A small, fully reproducible **Chineseâ€“English** language-model stack that runs
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
| data pipeline | SkyPile-150B (zh) + FineWeb-Edu (en), cleaned & deduped | see `data/processed/stats.json` |
| tokenizer | 32k byte-level BPE, zh/en balanced | `tokenizer/artifacts/` |
| dense 25M | ~25M params, 512 ctx | `checkpoints/dense_25m/` |
| MoE 4-expert | top-2 routing + aux loss | `checkpoints/moe_4e/` |
| looped | 2 unique layers Ã— 4 loops | `checkpoints/looped_16m/` |
| demo | ONNX int8 + static chat page | `web/` (GitHub Pages) |

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate   # Windows
pip install -r requirements.txt

python data/scripts/download.py --zh-shards 1 --en-shards 1
python data/scripts/clean.py
python tokenizer/train_tokenizer.py
python data/scripts/pack_tokens.py
python data/scripts/mix.py --total-tokens 500000000

python train/train.py --config configs/dense_25m.toml
python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt
```

## Why these choices

Every architectural decision (width, depth, heads, data volume, mixture,
optimizer) is written down with its source paper in [`docs/`](docs/):

- [01-data.md](docs/01-data.md) â€?corpora, licences, cleaning, zh/en ratio
- [02-tokenizer.md](docs/02-tokenizer.md) â€?vocab size, BPE vs unigram
- [03-model-architecture.md](docs/03-model-architecture.md) â€?dims, heads, GQA, MoE, looped
- [04-training.md](docs/04-training.md) â€?budget, LR, batch size, memory math
- [05-evaluation.md](docs/05-evaluation.md) â€?val loss, bits-per-byte, samples
- [06-deployment.md](docs/06-deployment.md) â€?ONNX export and browser inference

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

MIT (code). Datasets keep their own licences â€?see `docs/01-data.md`.
