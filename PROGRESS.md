# PROGRESS — append-only build log

Format: `YYYY-MM-DD HH:MM | phase | what changed | next`

## 2026-11-27 21:40 | P0 | environment recon done
- GPU RTX 4050 Laptop 6 GB (CC 8.9), 24 threads, 40 GB RAM, D: 219 GB free.
- torch 2.13.0+cu126 works; datasets/tokenizers installed; `gh` missing but PAT for `useosc` works via git credential helper.
- Repo scaffolded, model/trainer/data scripts written.

## 2026-11-27 21:55 | P1 | corpora downloading
- SkyPile-150B `2020-40_zh_head_0000.jsonl` (1.33 GB) + FineWeb-Edu `CC-MAIN-2025-26/000_00000.parquet` (1.39 GB).
- Mirror throughput ~12 MB/s on warm cache.
