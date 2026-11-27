"""Parallel English cleaning over parquet row groups (multiprocessing).

The single-process cleaner is I/O-bound on this machine (~9 MB/min gzip).
Splitting row groups across workers and concatenating gzip members keeps the
output a valid .jsonl.gz while using all cores.
"""
from __future__ import annotations

import gzip
import json
import multiprocessing as mp
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import doc_hash, normalize, quality_ok  # noqa: E402

SRC = "data/raw/fineweb_edu/en_00000.parquet"
TMP = "data/processed/_en_part"
DST = "data/processed/en.jsonl.gz"


def work(args: tuple[int, int, int]) -> tuple[str, int, int, int]:
    idx, rg_start, rg_end = args
    pf = pq.ParquetFile(SRC)
    path = f"{TMP}{idx}.jsonl.gz"
    kept = chars = total = 0
    seen: set[str] = set()
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as out:
        for rg in range(rg_start, rg_end):
            for t in pf.read_row_group(rg, columns=["text"]).column("text").to_pylist():
                total += 1
                if not t:
                    continue
                txt = normalize(t)
                ok, _ = quality_ok(txt, "en")
                if not ok:
                    continue
                h = doc_hash(txt)
                if h in seen:
                    continue
                seen.add(h)
                out.write(json.dumps({"text": txt, "lang": "en"}, ensure_ascii=False) + "\n")
                kept += 1
                chars += len(txt)
    return path, total, kept, chars


def main() -> None:
    pf = pq.ParquetFile(SRC)
    n_rg = pf.metadata.num_row_groups
    workers = 6
    bounds = [(i * n_rg) // workers for i in range(workers + 1)]
    jobs = [(i, bounds[i], bounds[i + 1]) for i in range(workers) if bounds[i] < bounds[i + 1]]
    print(f"{n_rg} row groups -> {len(jobs)} workers", flush=True)

    with mp.Pool(len(jobs)) as pool:
        results = pool.map(work, jobs)

    with open(DST, "wb") as out:
        for path, *_ in results:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            os.remove(path)

    total = sum(r[1] for r in results)
    kept = sum(r[2] for r in results)
    chars = sum(r[3] for r in results)
    stats = {"docs_in": total, "docs_out": kept, "chars_out": chars,
             "est_tokens": int(chars / 4.0), "size_mb": round(os.path.getsize(DST) / 1e6, 1)}
    print(json.dumps(stats, indent=2))

    stats_path = "data/processed/stats.json"
    merged = json.load(open(stats_path, encoding="utf-8")) if os.path.exists(stats_path) else {}
    merged["en"] = stats
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
