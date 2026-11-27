# 02 — Tokenizer

## Choice: byte-level BPE, 32 768 tokens

* **BPE** rather than unigram: better for morphologically rich and mixed-script
  text, and the reference implementation (`tokenizers`) is fast enough to train
  in minutes (Sennrich et al., 2016, arXiv:1508.07909).
* **Byte-level** (GPT-2 style) so every UTF-8 string is representable without an
  UNK token — essential when Chinese, English, emoji and code share a vocab.
* **32 768** because the embedding is 50 % of a 25 M model at this size:
  `V × d = 32768 × 384 = 12.6 M`. Qwen's 151 936 vocabulary (Qwen2 report,
  arXiv:2407.10671) is right for a 7 B+ model amortising 30+ languages; here it
  would make the embedding 58 M parameters — bigger than the rest of the model.
* **uint16 storage**: `V ≤ 65 535`, so two bytes per token, halving disk and
  host-to-device traffic versus int32.

## Normalisation and special tokens

* NFC normalisation (same convention as Qwen) so full-width/half-width and
  composed/decomposed forms map to the same bytes.
* Special tokens: `<|endoftext|>` (EOS/pad), `<|im_start|>`, `<|im_end|>` —
  enough to express a chat template and to allow a later SFT stage without
  re-tokenising the corpus.

## Expected compression

| language | rule of thumb | measured (this build) |
|---|---|---|
| Chinese | ~1 token / 1.2–1.5 chars | **1.62 chars/token** (421 M chars → 259.6 M tokens) |
| English | ~1 token / 4 chars | **4.33 chars/token** (2 115 M chars → 487.9 M tokens) |
| mixed | — | 4.38 UTF-8 bytes/token (used for BPB) |

The corpus statistics (`bytes_per_token`) are written by `pack_tokens.py` and
used to convert validation loss into bits-per-byte in `eval/evaluate.py`.

## Comparison with the Qwen tokenizer

| | this repo | Qwen2/3 |
|---|---|---|
| base vocab | 32 768 | 151 643 |
| embedding @ d=384 | 12.6 M | 58.2 M |
| zh compression | slightly worse | better (dedicated merges) |
| uint16 storage | yes | no (needs int32) |
| training cost | cheap | expensive at small scale |

Rule: tokenizer size should be ~1/8 to 1/4 of non-embedding parameters for
small models. At 12.6 M non-embedding, 32k is already at the upper end.

## Known limitations

* No dedicated FIM/whitespace-control tokens in v1 (add before code SFT).
* Byte fallback is implicit through the byte-level alphabet, but rare characters
  still cost 2–3 tokens; a bigger vocab would fix this at the cost of embedding
  parameters.
* The JS port in `web/bpe.js` follows the same GPT-2 regex; on unusual
  whitespace it can differ by a token or two from Python — acceptable for a demo,
  documented in the page.
