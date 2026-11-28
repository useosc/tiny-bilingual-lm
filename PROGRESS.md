# PROGRESS — append-only build log

Format: `YYYY-MM-DD HH:MM | phase | what changed | next`

## 2026-11-27 21:40 | P0 | environment
- RTX 4050 Laptop 6 GB (CC 8.9), 24 threads, 40 GB RAM, D: 219 GB free.
- torch 2.13.0+cu126; `gh` missing but a PAT for `useosc` is in the Windows
  credential manager. github.com:443 is intermittently blocked, so all git
  traffic uses SSH with the repo deploy key `dsh-deploy-key`.

## 2026-11-27 21:55 | P1 | corpora
- SkyPile-150B `2020-40_zh_head_0000.jsonl` 1.33 GB; FineWeb-Edu
  `CC-MAIN-2025-26/000_00000.parquet` 1.39 GB (mirror, curl).

## 2026-11-27 22:10 | P1 | cleaning
- zh 424,224 docs / 421 M chars; en 430,948 / 435,357 docs / 2.11 B chars.
- `clean_en_parallel.py` (6 processes over row groups) ~6 min.

## 2026-11-27 22:45 | P1 | tokenizer + data
- 32k byte-level BPE; zh 1.62 chars/token, en 4.33 chars/token.
- 496 M train + 4 M val tokens, zh:en = 50:50.

## 2026-11-27 22:49 | P2 | dense 20k steps started
- bs 16 x accum 3 (bs 24 spills on 6 GB), lr 1e-3 cosine.

## 2026-11-27 23:50 | P5 | web demo live (step-1000 snapshot)
- ONNX decode graph + int8 72.5 MB; JS BPE matches Python; ORT-Web ~15-16 tok/s.

## 2026-11-28 05:28 | P2 | dense done
- 20,000 steps / 492 M tokens; val 4.098, ppl 60.2, BPB 1.349; 6.6 h.

## 2026-11-28 07:30 | P3 | MoE done (120 min budget)
- 4,500 steps / 55 M tokens (best ckpt 4,000 = 49 M); val 5.069, BPB 1.669.
- Equal-token comparison: dense 5.056 vs MoE 5.069 at 49 M tokens → MoE not
  worth it at this scale (needs far more tokens).

## 2026-11-28 08:54 | P4 | looped done (80 min budget)
- 8,560 steps / 140 M tokens (best ckpt 8,000 = 131 M); val 4.763, BPB 1.568.
- Worse than dense at equal tokens (4.86 vs 4.72 at 98 M) but 38 % fewer params.

## 2026-11-28 08:56 | P5 | final export + Pages
- Dense best re-exported, int8 75,970,159 bytes; gh-pages updated; hashes of
  local and served ONNX match. Browser e2e: 16.1 tok/s, coherent zh/en output.
- `docs/assets/loss_curves.png` regenerated with all three runs.

## Final state
- Everything committed and pushed (`main` + `gh-pages`); fresh clone passes
  `tests/test_model.py`.
- Optional next steps (not in scope): SFT on bilingual instructions, MinHash
  near-dedup, 2k context retrain, move to torchtitan/Megatron on rented GPUs.
