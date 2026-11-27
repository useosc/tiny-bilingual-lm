# Model card — tiny-bilingual-lm 25M

## Summary

A 25.2 M parameter decoder-only language model trained from scratch on
**~492 M tokens** of Chinese and English web text (50:50 by tokens), on a single
**6 GB RTX 4050 Laptop GPU**.

* Architecture: Llama-style pre-norm transformer — RMSNorm, SwiGLU, RoPE,
  grouped-query attention (6 query / 2 KV heads), tied embeddings.
* Context: 512 tokens.
* Tokenizer: 32 768-token byte-level BPE (this repo).
* Training: AdamW, lr 1e-3 cosine to 1e-4, 300-step warmup, bf16 autocast,
  batch 16 × grad-accum 3, 20 000 steps.

## Intended use

* A **pipeline reference**: proof that data → tokenizer → training → export →
  browser demo works end-to-end on commodity hardware.
* Research baseline for the MoE and looped variants in this repo.
* Chinese/English text continuation.

## Out-of-scope use

* Instruction following, factual QA, safety-critical decisions, production
  chat. The base model has no alignment or instruction tuning.
* Anything requiring >512 tokens of context.
* Languages other than Chinese/English.

## Training data

| source | language | docs | tokens |
|---|---|---|---|
| SkyPile-150B shard 2020-40 | zh | 424 224 | 259.6 M |
| FineWeb-Edu CC-MAIN-2025-26 shard 0 | en | 430 948 | 487.9 M |

Licences: Skywork Community License (commercial use allowed with conditions) and
ODC-BY. See `docs/01-data.md` for filters and known limitations (no MinHash
near-dedup in v1).

## Evaluation

(filled after training — see `docs/05-evaluation.md` and `eval/results/`)

| metric | value |
|---|---|
| validation loss (nats) | **4.098** |
| perplexity | 60.2 |
| bits per byte | 1.349 |
| tokens seen | 492 M (19.5 per parameter) |
| training wall-clock | ~6.6 h on one RTX 4050 Laptop |
| peak VRAM | ~5.8 GB |
| throughput | 17–29 k tokens/s |

## Limitations

* 25 M parameters with ~20 tokens/parameter: it can produce fluent phrases and
  short coherent passages, not reasoning.
* 512-token context; long documents are truncated.
* Chinese web data is noisy (ads, duplicated news); filters are heuristic.
* No safety tuning: outputs may reproduce biases present in web text.

## How to run

```bash
python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m
python export/onnx_export.py --ckpt checkpoints/dense_25m/best.pt --out web/model/model.onnx
```

## Licence

Code MIT; model weights MIT; data under the upstream dataset licences.
