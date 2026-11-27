# 05 — Evaluation

## Protocol

1. **Validation loss** on the held-out 4 M-token slice (`data/tokenized/val.bin`),
   computed over 64 evenly spaced crops of length 512 → deterministic across runs.
2. **Perplexity** = exp(val loss).
3. **Bits per byte** = val loss / ln(2) / bytes_per_token. BPB is tokenizer
   independent, so the 32k tokenizer can be compared with Qwen's 151k later
   (Pile/OLMo convention, Gao et al., 2020, arXiv:2101.08303; Groeneveld et al.,
   2024, arXiv:2402.00838).
4. **Samples**: 8 fixed bilingual prompts, temperature 0.8, top-k 50, 80 new
   tokens. Stored under `eval/samples/<run>.txt`.
5. **Cost**: parameters, tokens seen, wall-clock, tokens/s, peak VRAM.

## How to run

```bash
python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m
python eval/evaluate.py --ckpt checkpoints/moe_4e/best.pt    --tag moe_4e
python eval/evaluate.py --ckpt checkpoints/looped_16m/best.pt --tag looped_16m
```

Results land in `eval/results/*.json`.

## Results (filled after each run)

| run | params | tokens seen | val loss | ppl | bpb | tok/s (train) |
|---|---|---|---|---|---|---|
| dense_25m | 25.2 M | 492 M | 4.098 | 60.2 | 1.349 | ~20 000 (peak) |
| moe_4e | 44.1 M (29.9 M active) | 49 M (4 000 steps) | 5.069 | 159.0 | 1.669 | ~6–10 k |
| looped_16m | 15.7 M | (filling) | | | | |

## What "good" looks like here

A 25 M model with 500 M tokens should reach a validation loss around **4.3–5.0**
nats on mixed web text (≈ 3.1–3.6 BPB), produce locally fluent phrases and
coherent short spans, and fall apart beyond a sentence or two. Anything below
4.0 nats on this data budget would suggest leakage; anything above 5.5 suggests
a training bug.
