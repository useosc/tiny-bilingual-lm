# 07 — Decisions log (ADR)

Every non-obvious choice, the alternative considered, and the evidence.
New decisions are appended at the bottom.

## D1 — 50:50 zh:en token ratio
**Context**: bilingual demo, unknown downstream mix.
**Choice**: equal tokens per language.
**Alternative**: 70:30 zh-heavy (better for a Chinese product) or 30:70 (cheaper,
English data is higher quality).
**Evidence**: mixture optima are downstream-dependent (DoReMi, arXiv:2305.10429;
RegMix, arXiv:2407.01492); no universal ratio exists. 50:50 is the least
surprising default and is trivial to ablate later.

## D2 — SkyPile + FineWeb-Edu as the v1 corpora
**Context**: need permissively licensed, easily downloadable zh + en web text.
**Choice**: SkyPile-150B (zh, Skywork Community License, commercial use allowed)
and FineWeb-Edu (en, ODC-BY).
**Alternatives rejected**: CCI4.0 (gated, licence needs confirmation), FineWeb-2
zh (4.8 GB shards, lower zh quality), MNBVC/WuDao (unclear licences),
Nemotron-CC (custom terms).

## D3 — 32 768-token byte-level BPE
See `docs/02-tokenizer.md`. Key constraint: embedding must not dominate a 25 M
model, and uint16 storage halves I/O.

## D4 — Dense 25 M: d=384, L=8, H=6, KV=2, FFN=1024
Rationale and derivations in `docs/03-model-architecture.md`. Parameters:
25.17 M total / 12.59 M non-embedding.

## D5 — Sequence length 512
**Context**: laptop memory and step time.
**Choice**: 512. RoPE makes later extension cheap; the rented-GPU phase will
retrain at 2 048 with θ=500 000 (Llama-3 recipe).

## D6 — AdamW, lr 1e-3, cosine, warmup 300
Small models tolerate higher LR; batch is 20× smaller than GPT-3-style runs, so
1e-3 (vs 6e-4 at 125 M) is justified. See `docs/04-training.md`.

## D7 — batch 16 × grad-accum 3 (not 24)
**Measured**: bs=24 peaks at 6 537 MB on a 6 439 MB card → Windows driver spills
to shared memory, step time 0.99 s vs 0.33 s at bs=16 (3× slower).
bs=16/4481 MB/0.33 s is the sweet spot; accum 3 restores 24 576 tokens/step.

## D8 — MoE: 4 experts, top-2, ffn 768
Top-k routing + Switch-style auxiliary loss (Switch, arXiv:2101.03961; GShard,
arXiv:2006.16668); top-2 is the Mixtral convention (arXiv:2401.04088). Four
experts keep the Python routing loop cheap on a laptop; production code would
use grouped GEMM and 8–64 experts.

## D9 — Looped: 2 unique blocks × 4 loops
Weight-shared depth (Giannou et al., arXiv:2301.13196; Yang et al.,
arXiv:2311.12424; Huginn, arXiv:2502.05171). Question: quality per parameter at
25 M scale.

## D10 — ONNX int8 for the demo, WASM execution
76 MB quantised graph, single decode step with KV cache, no server. Alternative
(transformers.js) cannot run a custom architecture; WebGPU would be faster but
adds a second code path.

## D11 — No SFT in v1
The deliverable is a pretraining pipeline; the chat page uses the base model
with a `用户：/助手：` prompt and says so. SFT is the first stretch goal once the
pretraining loop is proven.

## D12 — One repo, artifacts gitignored
Corpora, tokens and checkpoints are reproducible from scripts; only source,
docs and the final 76 MB ONNX (needed by Pages) are committed.
