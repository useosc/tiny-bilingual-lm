# 08 — Results

All numbers below come from `eval/results/*.json` and `checkpoints/*/metrics.jsonl`
on the RTX 4050 Laptop (6 GB). Validation is 4 M held-out tokens, 64 deterministic
crops; BPB uses 4.382 bytes/token.

## Dense 25.2 M — 20 k steps / 492 M tokens

| metric | value |
|---|---|
| params | 25,172,352 (12.59 M non-embedding) |
| tokens seen | 492 M (19.5 tokens/param) |
| final train loss | 3.81 |
| val loss (nats) | 4.098 |
| perplexity | 60.2 |
| bits per byte | 1.349 |
| wall-clock | 6.6 h |
| peak VRAM | ~5.8 GB |

![loss curves](assets/loss_curves.png)

## MoE 44.1 M (4 experts, top-2)

| metric | value |
|---|---|
| params | 44,059,008 (≈29.9 M active) |
| tokens seen | (from budget) |
| val loss | (filled) |
| bits per byte | (filled) |
| note | shorter run, compare at equal tokens |

## Looped 15.7 M (2 unique blocks × 4 loops)

| metric | value |
|---|---|
| params | 15,730,560 |
| effective depth | 8 |
| tokens seen | (from budget) |
| val loss | (filled) |
| bits per byte | (filled) |

## Qualitative samples

See `eval/samples/dense_25m.txt`, `moe_4e.txt`, `looped_16m.txt`. The page at
https://useosc.github.io/tiny-bilingual-lm/ runs the exported int8 dense model
entirely in the browser.

## Honest interpretation

* 25 M parameters cannot reason. The dense model produces locally fluent
  Chinese/English and drifts after a sentence; MoE and looped are research
  comparisons at a much smaller token budget, not better products.
* The value of this repo is the **pipeline**: every stage is reproducible,
  documented, and runs on hardware a student owns.
* Next step on rented GPUs: 1–2 B parameters, 30–60 B tokens, sequence length
  2 048, the same code moved to torchtitan/Megatron.
