"""Mix language shards into a single shuffled token stream (+ held-out val)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenized", default="data/tokenized")
    ap.add_argument("--total-tokens", type=int, default=600_000_000)
    ap.add_argument("--zh-ratio", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--val-tokens", type=int, default=4_000_000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    zh = np.memmap(os.path.join(args.tokenized, "zh.bin"), dtype=np.uint16, mode="r")
    en = np.memmap(os.path.join(args.tokenized, "en.bin"), dtype=np.uint16, mode="r")
    print(f"zh={len(zh)/1e6:.1f}M tokens, en={len(en)/1e6:.1f}M tokens")

    want_zh = int(args.total_tokens * args.zh_ratio)
    want_en = args.total_tokens - want_zh
    want_zh = min(want_zh, len(zh) - args.chunk - 1)
    want_en = min(want_en, len(en) - args.chunk - 1)

    chunks: list[tuple[str, int]] = []
    for lang, arr, want in (("zh", zh, want_zh), ("en", en, want_en)):
        got = 0
        while got < want:
            n = min(args.chunk, want - got)
            start = int(rng.integers(0, len(arr) - n - 1))
            chunks.append((lang, start, n))
            got += n
    rng.shuffle(chunks)

    order = np.arange(len(chunks))
    rng.shuffle(order)
    total = sum(c[2] for c in chunks)
    val_n = args.val_tokens
    train_n = total - val_n

    out_train = np.memmap(os.path.join(args.tokenized, "train.bin"), dtype=np.uint16, mode="w+", shape=(total,))
    pos = 0
    for i in order:
        lang, start, n = chunks[i]
        src = zh if lang == "zh" else en
        out_train[pos : pos + n] = src[start : start + n]
        pos += n
    out_train.flush()

    # Validation is the tail slice, taken before shuffling would leak; we keep
    # it as a separate file so training can never sample it by accident.
    val = np.memmap(os.path.join(args.tokenized, "val.bin"), dtype=np.uint16, mode="w+", shape=(val_n,))
    val[:] = out_train[train_n : train_n + val_n]
    val.flush()
    del val, out_train

    raw = json.load(open(os.path.join(args.tokenized, "raw_meta.json"), encoding="utf-8"))
    bpt = (raw["zh"]["bytes"] + raw["en"]["bytes"]) / max(raw["zh"]["tokens"] + raw["en"]["tokens"], 1)
    meta = {
        "tokens_train": int(train_n),
        "tokens_val": int(val_n),
        "zh_ratio": args.zh_ratio,
        "bytes_per_token": bpt,
        "sources": {"zh": "SkyPile-150B (2020-40)", "en": "FineWeb-Edu (CC-MAIN-2025-26)"},
    }
    with open(os.path.join(args.tokenized, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
