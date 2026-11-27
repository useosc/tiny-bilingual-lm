"""Clean and de-duplicate downloaded corpora into gzip JSONL.

Usage:
  python data/scripts/clean.py --zh data/raw/skypile --en data/raw/fineweb_edu --out data/processed
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import doc_hash, normalize, quality_ok  # noqa: E402


def iter_zh(paths: list[str]):
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)["text"]
                except Exception:
                    continue


def iter_en(paths: list[str]):
    import pyarrow.parquet as pq

    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
            for t in batch.column("text").to_pylist():
                if t:
                    yield t


def run(lang: str, paths: list[str], out_path: str, seen: set[str]) -> dict:
    stats = Counter()
    t0 = time.time()
    with gzip.open(out_path, "wt", encoding="utf-8", compresslevel=6) as out:
        it = iter_zh(paths) if lang == "zh" else iter_en(paths)
        for raw in it:
            stats["docs_in"] += 1
            text = normalize(raw)
            ok, reason = quality_ok(text, lang)
            if not ok:
                stats[f"drop_{reason}"] += 1
                continue
            h = doc_hash(text)
            if h in seen:
                stats["drop_duplicate"] += 1
                continue
            seen.add(h)
            stats["docs_out"] += 1
            stats["chars_out"] += len(text)
            out.write(json.dumps({"text": text, "lang": lang}, ensure_ascii=False) + "\n")
    stats["seconds"] = round(time.time() - t0, 1)
    stats["est_tokens"] = int(stats["chars_out"] / (1.3 if lang == "zh" else 4.0))
    return dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", default="data/raw/skypile")
    ap.add_argument("--en", default="data/raw/fineweb_edu")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--only", choices=["both", "zh", "en"], default="both")
    ap.add_argument("--limit-docs", type=int, default=0, help="0 = no limit (debugging aid)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    zh_paths = sorted(glob.glob(os.path.join(args.zh, "**", "*.jsonl"), recursive=True))
    en_paths = sorted(glob.glob(os.path.join(args.en, "**", "*.parquet"), recursive=True))
    print("zh shards:", zh_paths)
    print("en shards:", en_paths)

    seen: set[str] = set()
    report = {}
    if args.only in ("both", "zh"):
        report["zh"] = run("zh", zh_paths, os.path.join(args.out, "zh.jsonl.gz"), seen)
    if args.only in ("both", "en"):
        report["en"] = run("en", en_paths, os.path.join(args.out, "en.jsonl.gz"), seen)
    stats_path = os.path.join(args.out, "stats.json")
    merged = {}
    if os.path.exists(stats_path):
        merged = json.load(open(stats_path, encoding="utf-8"))
    merged.update(report)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
