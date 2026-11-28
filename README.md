# tiny-bilingual-lm

A small, fully reproducible **Chinese–English** language-model stack that runs
end-to-end on a single **6 GB laptop GPU** (RTX 4050 Laptop):

```
open corpora -> clean/dedup -> 32k BPE -> packed tokens -> dense -> MoE -> looped -> ONNX -> browser demo
```

The point is not to beat Qwen. The point is to own every line of the pipeline,
so the same code can be moved to a rented A100/H100 cluster later without a
rewrite.

**Live demo (runs fully in your browser, no backend):**
https://useosc.github.io/tiny-bilingual-lm/

## Status

| stage | artifact | status |
|---|---|---|
| data pipeline | SkyPile-150B (zh, 421 M chars) + FineWeb-Edu (en, 2.12 B chars), cleaned & deduped | done: 424 k zh + 431 k en docs |
| tokenizer | 32k byte-level BPE, zh 1.62 chars/token, en 4.33 chars/token | done |
| packed data | 496 M train + 4 M val tokens (uint16), zh:en = 50:50 | done |
| dense 25M | 25.2 M params, 512 ctx, 492 M tokens | **done**: val 4.098 / ppl 60.2 / BPB 1.349 |
| MoE 4-expert | 44.1 M total params, top-2 routing + aux loss | done: val 5.069 at 49 M tokens (dense at same tokens: 5.056) |
| looped | 2 unique layers x 4 loops (15.7 M params, 8 effective layers) | done: val 4.763 at 131 M tokens |
| demo | ONNX int8 (72.5 MB) + static chat page, final dense model | **live** |
| docs | 8 design docs with citations | done |

## Results (one RTX 4050 Laptop, 6 GB)

| run | params | tokens | val loss | ppl | bits/byte |
|---|---|---|---|---|---|
| dense | 25.2 M | 492 M | **4.098** | 60.2 | 1.349 |
| MoE (4 experts, top-2) | 44.1 M (29.9 M active) | 49 M | 5.069 | 159.0 | 1.669 |
| looped (2 x 4) | 15.7 M | 131 M | 4.763 | 117.1 | 1.568 |

At equal tokens (~49 M) dense 5.056 ≈ MoE 5.069 > looped 5.216: at this scale
neither MoE nor recurrence beats a plain dense model per token; MoE costs ~2x
compute and looped saves ~38% parameters. Details in [docs/08-results.md](docs/08-results.md).

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
- [08-results.md](docs/08-results.md) - measured results for the three variants

## Repository layout

```
model/       dense / MoE / looped implementations (pure PyTorch)
train/       trainer with AMP, grad-accum, cosine LR, resume, eval
data/        download -> clean -> tokenize -> mix
tokenizer/   BPE training
eval/        checkpoint evaluation and sample generation
export/      ONNX export for the web demo
web/         static chat page (GitHub Pages)
scripts/     one-shot pipeline, overnight chain, Pages deploy, plots
tests/       model/KV-cache/ONNX/BPE/browser tests
docs/        design decisions with citations
```

## Licence

MIT (code). Datasets keep their own licences - see `docs/01-data.md`.
