"""Download a small bilingual pretraining sample from HuggingFace.

Sources (all open, see docs/01-data.md):
  * Chinese web:  SkyPile-150B  (Skywork Community License, commercial use OK)
  * English web:  FineWeb-Edu   (ODC-BY)

We download whole shards instead of streaming: the mirror serves shards at
~10 MB/s, which is far faster than per-document parquet streaming.
"""
from __future__ import annotations

import argparse
import os
import time

from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

SKY_FILES = [f"data/2020-40_zh_head_{i:04d}.jsonl" for i in range(5)]
FWEDU_FILES = [f"data/CC-MAIN-2025-26/000_{i:05d}.parquet" for i in range(5)]


def dl(repo: str, filename: str, out_dir: str) -> str:
    t0 = time.time()
    path = hf_hub_download(
        repo_id=repo, repo_type="dataset", filename=filename,
        local_dir=out_dir, endpoint=os.environ["HF_ENDPOINT"],
    )
    size = os.path.getsize(path) / 1e9
    print(f"[ok] {filename}  {size:.2f} GB  {time.time() - t0:.1f}s", flush=True)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--zh-shards", type=int, default=1)
    ap.add_argument("--en-shards", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for f in SKY_FILES[: args.zh_shards]:
        dl("Skywork/SkyPile-150B", f, os.path.join(args.out, "skypile"))
    for f in FWEDU_FILES[: args.en_shards]:
        dl("HuggingFaceFW/fineweb-edu", f, os.path.join(args.out, "fineweb_edu"))


if __name__ == "__main__":
    main()
