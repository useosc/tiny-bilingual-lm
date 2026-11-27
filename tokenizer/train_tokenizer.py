"""Train a byte-level BPE tokenizer on the bilingual cleaned corpus.

Design notes (see docs/02-tokenizer.md):
  * byte-level BPE, 32k merges -> fits uint16 storage, keeps zh/en balanced;
  * NFC normalisation (same as Qwen) keeps full-width/half-width consistent;
  * special tokens cover EOS and a minimal chat format.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

from tokenizers import ByteLevelBPETokenizer, Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.normalizers import NFC, Sequence

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def sample_texts(path: str, budget_chars: int) -> list[str]:
    texts, total = [], 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)["text"]
            texts.append(t)
            total += len(t)
            if total >= budget_chars:
                break
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh", default="data/processed/zh.jsonl.gz")
    ap.add_argument("--en", default="data/processed/en.jsonl.gz")
    ap.add_argument("--out", default="tokenizer/artifacts")
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--zh-mb", type=float, default=60)
    ap.add_argument("--en-mb", type=float, default=60)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    texts = sample_texts(args.zh, int(args.zh_mb * 1_000_000))
    texts += sample_texts(args.en, int(args.en_mb * 1_000_000))
    print(f"tokenizer training docs={len(texts)} chars={sum(len(t) for t in texts)/1e6:.0f}M")

    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tok.normalizer = Sequence([NFC()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    path = os.path.join(args.out, "tokenizer.json")
    tok.save(path)

    # HuggingFace-compatible wrapper for downstream code (AutoTokenizer).
    with open(os.path.join(args.out, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump({"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"}, f)
    with open(os.path.join(args.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
                "model_max_length": 512,
                "clean_up_tokenization_spaces": False,
            },
            f,
        )
    print("saved", path, "vocab", tok.get_vocab_size())


if __name__ == "__main__":
    main()
