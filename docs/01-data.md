# 01 — Data: sources, licences, pipeline, mixture

## 1. Sources (why these two)

| source | language | size used | licence | why |
|---|---|---|---|---|
| [SkyPile-150B](https://huggingface.co/datasets/Skywork/SkyPile-150B) `2020-40_zh_head_0000.jsonl` | Chinese web | 1.33 GB raw | Skywork Community License (commercial use permitted under conditions) | largest clean, permissively licensed Chinese web corpus; JSONL means no parquet streaming overhead |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) `CC-MAIN-2025-26/000_00000.parquet` | English web | 1.39 GB raw | ODC-BY | education-value filtered, the de-facto English baseline since 2024 (Penedo et al., 2024, arXiv:2406.17557) |

Considered and rejected for this first version:

* **FineWeb-2** (ODC-BY, 1000+ languages): the Chinese split ships as 4.8 GB
  parquet shards and its zh quality is below SkyPile; kept as a future
  diversity source.
* **CCI4.0** (BAAI): excellent Chinese coverage but the repo is gated and the
  licence needs manual confirmation — a blocker for a reproducible public repo.
* **MNBVC / WuDaoCorpora**: unclear licences, mostly research-only.
* **Nemotron-CC / CulturaX**: custom or undeclared licences.

Licensing note: dataset licence ≠ copyright of the underlying pages. For a
commercial product, keep provenance metadata and be ready to honour takedowns;
this repo stores only a filtered sample and documents every source.

## 2. Pipeline

```
download.py   shard-level download (curl / huggingface_hub), ~12 MB/s on the mirror
clean.py      NFKC -> whitespace/HTML cleanup -> language check -> heuristics -> exact dedup
train_tokenizer.py   32k byte-level BPE on a 120 M-char sample
pack_tokens.py       encode to uint16 shards + record bytes/token
mix.py               zh:en = 50:50 by tokens -> train.bin + val.bin
```

### Cleaning rules (implemented in `data/scripts/common.py`)

| rule | threshold | rationale |
|---|---|---|
| length | 200–30 000 chars | removes nav junk and giant dumps |
| Chinese ratio | ≥ 20 % CJK for zh docs | drops translated/English pages in the zh shard |
| English letters | ≥ 55 % and < 2 % CJK for en | drops non-English leakage |
| stop-word presence | ≥ 5 % of words in a stop-word list | catches gibberish and keyword spam |
| printable ratio | ≥ 80 % | drops mojibake and symbol soup |
| duplicate lines | top line ≤ 30 % of lines | drops menus, legal boilerplate |
| spam markers | `lorem ipsum`, 扫码关注, 版权所有 … | cheap precision filter |
| exact dedup | blake2b of normalized text | removes mirrors/reposts (a major issue in Chinese news) |

MinHash near-dedup is deliberately **not** in v1 (cost); it is the first
upgrade on the list for the rented-GPU phase (`docs/07-decisions-log.md`).

## 3. Mixture: why 50:50

* Target use is bilingual chat; equal token budget keeps both languages
  competent. Literature on mixture ratios (DoReMi, Xie et al., 2023,
  arXiv:2305.10429; RegMix, Liu et al., 2024, arXiv:2407.01492) shows that the
  optimum depends on the downstream mix, not on a universal constant.
* A 50:50 token split means roughly 50 % Chinese characters vs 50 % English
  words, i.e. Chinese gets *more documents* per token because Chinese is
  denser per character. This is intentional for a Chinese-first product.
* Ablation plan: 30:70 and 70:30 at 100 M tokens each, same compute, to measure
  the trade-off (Phase 2 stretch).

## 4. Disk budget (≤ 200 GB requirement)

| item | size |
|---|---|
| raw shards | 2.7 GB |
| cleaned jsonl.gz | ~1.0–1.5 GB |
| tokenized bins (500 M tokens × 2 B) | 1.0 GB |
| checkpoints (3 runs × last+best ≈ 300 MB each) | ~1.8 GB |
| ONNX export | < 0.1 GB |
| **total** | **< 7 GB** |

The remaining ~190 GB is free for a larger corpus when the pipeline moves to a
rented machine.

## 5. Statistics

Filled automatically by `clean.py` into `data/processed/stats.json`; the final
numbers for this build are recorded in the table below and in `docs/05-evaluation.md`.

| language | docs in | kept | kept % | chars kept | tokens |
|---|---|---|---|---|---|
| zh | 424 224 (all lines of shard 0000) | 424 224 | 100 % (shard is pre-filtered by Skywork) | 421 M | 259.6 M |
| en | 435 357 | 430 948 | 99.0 % | 2 115 M | 487.9 M |

Token counts come from `pack_tokens.py`; compression is 1.62 chars/token for
Chinese and 4.33 chars/token for English (`bytes_per_token` ≈ 4.38 for the mix).
