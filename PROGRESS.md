# PROGRESS — append-only build log

Format: `YYYY-MM-DD HH:MM | phase | what changed | next`

## 2026-11-27 21:40 | P0 | environment recon done
- GPU RTX 4050 Laptop 6 GB (CC 8.9), 24 threads, 40 GB RAM, D: 219 GB free.
- torch 2.13.0+cu126; datasets/tokenizers installed; `gh` missing but a PAT for `useosc` is in the Windows credential manager (`repo`, `workflow` scopes).
- Repo scaffolded.

## 2026-11-27 21:55 | P1 | corpora downloaded
- SkyPile-150B `2020-40_zh_head_0000.jsonl` 1.33 GB, FineWeb-Edu `CC-MAIN-2025-26/000_00000.parquet` 1.39 GB. Mirror ~10-35 MB/s; `hf_hub_download` stalled, curl worked; `-C -` broke the 302→xet redirect, plain download + a small Range-resume script worked.

## 2026-11-27 22:10 | P1 | cleaning done
- zh: 424,224 docs kept (500 MB gzip). en: 430,948 / 435,357 docs kept, 2.11 B chars (905 MB gzip).
- Single-process English cleaning was I/O-bound (9 MB/min); replaced with 6-process row-group cleaning (`clean_en_parallel.py`), ~6 min.
- Filter thresholds are permissive (~98 % keep). MinHash near-dedup is the first v2 upgrade.

## 2026-11-27 22:12 | P1 | tokenizer + packing running
- 32k byte-level BPE on 60 M zh + 60 M en chars.
- pack -> uint16 shards; mix -> 500 M tokens at zh:en = 50:50.

## Hardware measurements (for docs)
- dense bs=16: 0.327 s/step, 25k tok/s, 4.5 GB peak.
- dense bs=24: 0.99 s/step (driver spill, 6.5 GB > 6.4 GB) — rejected.
- MoE bs=12: ~0.55 s/step; Looped bs=16: 0.25 s/step.
