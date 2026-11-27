"""Pack cleaned JSONL into uint16 token shards and record compression stats."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np
from tokenizers import Tokenizer


def encode_file(tok: Tokenizer, path: str, out_bin: str, batch_docs: int = 512) -> tuple[int, int]:
    eos = tok.token_to_id("<|endoftext|>")
    total_tokens = 0
    total_bytes = 0
    t0 = time.time()
    with open(out_bin, "wb") as out, gzip.open(path, "rt", encoding="utf-8") as f:
        buf, docs = [], 0
        for line in f:
            text = json.loads(line)["text"]
            buf.append(text)
            total_bytes += len(text.encode("utf-8"))
            if len(buf) >= batch_docs:
                encs = tok.encode_batch(buf, add_special_tokens=False)
                ids = np.concatenate(
                    [np.asarray(e.ids, dtype=np.uint16) for e in encs]
                    + [np.array([eos], dtype=np.uint16)] * len(encs)
                )
                out.write(ids.tobytes())
                total_tokens += len(ids)
                docs += len(buf)
                buf = []
                if docs % (batch_docs * 40) == 0:
                    print(f"  {docs} docs, {total_tokens/1e6:.1f}M tokens, "
                          f"{total_tokens/max(time.time()-t0,1)/1e3:.0f}k tok/s", flush=True)
        if buf:
            encs = tok.encode_batch(buf, add_special_tokens=False)
            ids = np.concatenate(
                [np.asarray(e.ids, dtype=np.uint16) for e in encs]
                + [np.array([eos], dtype=np.uint16)] * len(encs)
            )
            out.write(ids.tobytes())
            total_tokens += len(ids)
    print(f"[done] {path}: {total_tokens/1e6:.1f}M tokens -> {out_bin}", flush=True)
    return total_tokens, total_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/artifacts/tokenizer.json")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="data/tokenized")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(args.tokenizer)
    meta = {}
    for lang in ("zh", "en"):
        src = os.path.join(args.processed, f"{lang}.jsonl.gz")
        dst = os.path.join(args.out, f"{lang}.bin")
        n, b = encode_file(tok, src, dst)
        meta[lang] = {"tokens": n, "bytes": b, "bytes_per_token": b / max(n, 1)}
    meta["vocab_size"] = tok.get_vocab_size()
    meta["eos_id"] = tok.token_to_id("<|endoftext|>")
    with open(os.path.join(args.out, "raw_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
