# PROGRESS — append-only build log

Format: `YYYY-MM-DD HH:MM | phase | what changed | next`

## 2026-11-27 21:40 | P0 | environment
- RTX 4050 Laptop 6 GB (CC 8.9), 24 threads, 40 GB RAM, D: 219 GB free.
- torch 2.13.0+cu126; datasets/tokenizers present; `gh` missing, but a PAT for
  `useosc` sits in the Windows credential manager (scopes repo, workflow).
- github.com:443 is intermittently unreachable from this network; **use SSH**:
  a repo deploy key `dsh-deploy-key` is installed and `git@github.com:...` works
  for both `main` and `gh-pages`. Do not switch back to HTTPS.

## 2026-11-27 21:55 | P1 | corpora
- SkyPile-150B `2020-40_zh_head_0000.jsonl` 1.33 GB; FineWeb-Edu
  `CC-MAIN-2025-26/000_00000.parquet` 1.39 GB.
- `hf_hub_download` stalled; curl works; `-C -` breaks the 302→xet redirect.

## 2026-11-27 22:10 | P1 | cleaning
- zh 424,224 docs / 421 M chars; en 430,948 / 435,357 docs / 2.11 B chars.
- Single-process English cleaning was I/O bound; `clean_en_parallel.py`
  (6 processes over row groups) takes ~6 min.

## 2026-11-27 22:45 | P1 | tokenizer + data
- 32k byte-level BPE; zh 1.62 chars/token, en 4.33 chars/token.
- 496 M train tokens + 4 M val, zh:en = 50:50 (`data/tokenized/`).

## 2026-11-27 22:49 | P2 | dense training started
- configs/dense_25m.toml: bs 16 × accum 3, lr 1e-3 cosine, 20k steps.
- Step 1000: val 5.70 (bpb 1.83); step ~1400 loss 4.93.
- ETA ~04:40 at 24k tok/s. GPU ~5.8 GB / 6.1 GB, watch for OOM at eval.

## 2026-11-27 23:50 | P5 | web demo live (early snapshot)
- ONNX decode graph + int8 (72.5 MB); Python ORT 531–683 tok/s; ORT-Web WASM
  ~200 tok/s; JS BPE matches Python on all test strings.
- gh-pages pushed; Pages built: https://useosc.github.io/tiny-bilingual-lm/
  (currently serves the step-1000 checkpoint; re-export after training).

## NEXT (in order)
1. When dense reaches 20k steps (or time budget): run
   `python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m`.
2. Re-export final ONNX + int8, copy into `D:\deepseek\tiny-bilingual-lm-pages`,
   commit/push `gh-pages`, verify the Pages URL serves the new file size.
3. Train MoE (`--max-minutes 150`) then looped (`--max-minutes 100`), eval both.
4. Fill `docs/05-evaluation.md` table, model card numbers, README status.
5. Final commit + push main; verify fresh clone + 100-step smoke train.
6. Optional stretch: 1 h SFT on bilingual instructions to make the demo chat-like.
