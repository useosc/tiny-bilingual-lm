# 03 — Model architecture and why every number is what it is

This note records the reasoning behind each architectural hyper-parameter, the
paper it comes from, and the arithmetic used. If you want to change the model,
change it here first, then in `model/` and `configs/`.

## 0. Constraints

| constraint | value | consequence |
|---|---|---|
| GPU | RTX 4050 Laptop, **6 GB**, Ada (CC 8.9) | weights + grads + Adam states + activations must fit ~5.5 GB |
| Disk | ~200 GB (D:) | corpora + tokens + checkpoints must stay ≈ 10 GB |
| Time | a few hours per run | ≤ 600 M tokens for the main run |
| Goal | runnable end-to-end, liftable to a rented cluster | keep the code parallel-ready, avoid laptop-only tricks |

Everything below is chosen so the **dense** model is clearly under 1 GB of
training memory, leaving headroom for experiments.

## 1. Parameter count formula

For a Llama-style block with `d` model width, `h` FFN width, `n_h` query heads,
`n_kv` KV heads of size `d_head = d / n_h`, and vocab `V`:

```
embedding      = V * d                    (tied with lm_head)
attention      = 2 * d^2 + 2 * d * (n_kv * d_head)
ffn (SwiGLU)   = 3 * d * h
per layer      = attention + ffn
total          = V*d + n_layers * per_layer
```

Dense config (`d=384, n_layers=8, n_h=6, n_kv=2, d_head=64, h=1024, V=32768`):

```
embedding  = 32768*384                = 12,582,912
attention  = 2*384^2 + 2*384*(2*64)   =   393,216
ffn        = 3*384*1024               = 1,179,648
per layer                             = 1,572,864
8 layers                              = 12,582,912
total                                 = 25,165,824  (~25.2 M)
non-embedding                         = 12.58 M
```

The embedding is **50 %** of the parameters. That is expected for a small
model; it is also why the tokenizer decision (32k vs 151k) matters so much.

## 2. Width and depth: why 384 × 8

* Scaling laws say loss is a power law in parameters `N` and tokens `D`
  (Kaplan et al., 2020, arXiv:2001.08361); at fixed compute the optimal `N` and
  `D` grow together (Hoffmann et al., 2022, arXiv:2203.15556, "Chinchilla").
  For a 6 GB laptop the compute budget is the binding constraint, so `N` is
  chosen as large as fits comfortably: ~25 M.
* Within a parameter budget, published models cluster around
  `d_model / n_layers ≈ 32–128`: GPT-3 125M is 768/12 = 64 (Brown et al., 2020,
  arXiv:2005.14165, Table 2.1); Llama-2-7B is 4096/32 = 128 (Touvron et al.,
  2023, arXiv:2307.09288, Table 1); Llama-3-8B is 4096/32.
  We pick **384/8 = 48**, inside that band but at the shallow end because
  small models benefit from relatively more depth per parameter and 8 blocks
  keep activation memory low.
* `n_heads = 6` with `d_head = 64`: GPT-3's small models also use head_dim 64
  (125M: 12 heads × 64), while large models move to 128. 64 is the safe choice
  for a 384-wide model (6 × 64 = 384).
* `n_kv_heads = 2` (GQA, 3:1): Ainslie et al., 2023 (arXiv:2305.13245) show GQA
  matches MHA quality at a fraction of KV cache. Llama-2-70B uses 8:1, Llama-3-8B
  4:1; at our size 3:1 already cuts KV memory by 3× with 2 KV heads.
* `ffn_hidden = 1024 ≈ 8/3 × 384`: SwiGLU needs 2/3 of the classic 4× width to
  match parameter count (Shazeer, 2020, arXiv:2002.05202; Llama-2 uses the same
  8/3 rule). 1024 = 2.67 × 384, the closest round number.

## 3. Normalisation, activations, positions, init

* **RMSNorm** instead of LayerNorm: Zhang & Sennrich, 2019 (arXiv:1910.02398);
  used by Llama, Qwen, DeepSeek.
* **Pre-norm** residual placement: Xiong et al., 2020 (arXiv:2002.04745) show
  post-norm needs warmup and is less stable at depth.
* **SwiGLU**: Shazeer, 2020.
* **RoPE** with θ = 10 000: Su et al., 2021 (arXiv:2104.09864). Context length
  512 is a laptop compromise; RoPE makes later extension to 2k–4k cheap
  (linear/NTK scaling), which is the plan for the rented-GPU phase.
* **Tied embeddings** (`lm_head.weight = embed_tokens.weight`): saves 12.6 M
  parameters (≈50 % of the model) and is standard for small LMs (GPT-2,
  Radford et al., 2019; OLMo, Groeneveld et al., 2024). At 25 M parameters the
  quality cost is negligible, and it keeps the ONNX file small.
* **Init**: normal(0, 0.02) plus residual-output scaling by `1/√(2·L)`
  (GPT-2 recipe), which keeps activation variance stable across 8 blocks.
* **No biases, no dropout**: standard for pretraining-era Llama-style models;
  dropout would slow convergence at this data scale.

## 4. Vocabulary: 32k, byte-level BPE

* BPE (Sennrich et al., 2016, arXiv:1508.07909) with byte-level fallback
  (GPT-2) so any UTF-8 input is representable — important for mixed zh/en.
* 32k merges keeps the embedding at 12.6 M parameters and lets us store tokens
  as **uint16** (2 bytes), halving disk and I/O versus int32.
* Qwen uses 151 936 because it amortises a much larger model over 30+ languages
  (Qwen2 report, arXiv:2407.10671). At 25 M parameters a 151k vocabulary would
  make the embedding 58 M parameters — larger than the rest of the model.
* Compression target: ≈1 token per 1.2–1.5 Chinese characters and ≈1 token per
  4 English characters with 32k merges. Measured numbers are in
  `docs/02-tokenizer.md`.

## 5. MoE variant: 4 experts, top-2

Motivation: same active compute, more capacity. Router: linear `d → E`, softmax,
top-k; each token goes to `k` experts and outputs are combined with the gate
probabilities; a Switch-style load-balancing loss keeps experts used evenly.

* Top-k routing and the auxiliary load-balance loss: Switch Transformers
  (Fedus et al., 2021, arXiv:2101.03961); GShard (Lepikhin et al., 2020,
  arXiv:2006.16668) introduced top-2.
* Top-2 with 8 experts in practice: Mixtral (Jiang et al., 2024,
  arXiv:2401.04088).
* Shared/fine-grained experts: DeepSeekMoE (Dai et al., 2024, arXiv:2401.06066)
  — future work; the current code has no shared expert to keep the comparison
  clean.

Config: `E=4, k=2, ffn_hidden=768` per expert. Parameters:

```
per layer = attention 0.393 M + 4 * (3*384*768) / 1e6 = 0.393 + 3.539 = 3.93 M
8 layers  = 31.5 M  + embedding 12.6 M = 44.1 M total
active    = attention 3.1 M + 2/4 * 28.3 M experts + 12.6 M embedding = ~29.9 M
```

So the MoE has ~1.75× the parameters of dense at roughly the same FLOPs per
token. Trained on fewer tokens (~120 M) because each step is ~2× slower.

## 6. Looped variant: 2 unique blocks × 4 loops

Weight sharing across depth ("recurrent depth") buys effective depth without
parameters. The same block is applied 4 times, so the model has 8 effective
layers but only 2 unique ones.

* Looped transformers as programmable computers: Giannou et al., 2023,
  arXiv:2301.13196.
* Looped transformers learn learning algorithms better: Yang et al., 2023,
  arXiv:2311.12424.
* Recurrent-depth LMs: Huginn (Geiping et al., 2025, arXiv:2502.05171) and Ouro
  (2025) explore this at scale.

Parameters: `12.58 M embedding + 2 × 1.573 M = 15.7 M` — 40 % fewer than dense
for the same nominal depth. The research question: how much quality per
parameter does recurrence buy at 25 M scale?

## 7. Why not bigger / why not smaller

Memory for AdamW training (bf16 weights + fp32 master copy + fp32 m, v):

```
weights bf16   2 bytes/param
master + m + v 12 bytes/param
gradients      2 bytes/param (bf16) or 4 (fp32)
=> ~16-18 bytes/param
```

25 M params ≈ 0.45 GB of optimizer/weights — no problem on 6 GB. Activations
for `batch 24 × 512 tokens` are the other term: with SDPA (flash) attention the
score matrix is never materialised, so activations are ~O(B·T·d) per layer
(≈9.4 MB), roughly 0.3 GB for 8 layers. Total < 1.5 GB, leaving room for
gradient accumulation and experiments.

A 100 M model would need ~2.5–3 GB of state plus larger activations; it fits
only with gradient checkpointing and much slower steps. That is the first thing
to try on rented hardware.

## 8. Data volume: 600 M tokens for 25 M parameters

* Chinchilla compute-optimal: ~20 tokens per parameter (Hoffmann et al., 2022)
  → 25 M × 20 ≈ 500 M tokens.
* Modern practice over-trains (Llama-3-8B: ~15 T tokens ≈ 1 800 tokens/param)
  because inference cost dominates; at laptop scale we cannot afford that.
* Our run uses ~590 M tokens (24 k steps × 24 576 tokens) ≈ **23.4 tokens/param**,
  just above Chinchilla-optimal, which is the right target for a first
  baseline. On rented GPUs the first upgrade is 2–5 B tokens (80–200 tokens/param).

## 9. What to change when scaling up

1. `max_seq_len` 512 → 2048/4096, RoPE θ 10 000 → 500 000 (Llama-3 recipe).
2. `n_kv_heads` 2 → 8 (GQA 4:1) at ~1 B parameters.
3. Untie embeddings above ~300 M parameters (embedding share drops below 10 %).
4. Replace the Python MoE loop with grouped GEMM (`torch._grouped_mm` or
   Megatron's), add capacity factor and token dropping for throughput.
5. Add tensor/pipeline parallelism — the model code already keeps projections
   as plain `nn.Linear`, so Megatron/torchtitan conversion is mechanical.
