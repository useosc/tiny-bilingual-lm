# 04 — Training recipe and budget

## 1. Throughput and budget

```
tokens/step = batch_size × grad_accum × seq_len
            = 24 × 2 × 512
            = 24,576
max_steps   = 24,000
total       = 590 M tokens  (~23.4 tokens per parameter)
```

At the measured laptop throughput (see `checkpoints/dense_25m/metrics.jsonl`),
this is a single-digit-hour run. The MoE and looped variants use ~120 M tokens
each so the three runs finish in one night.

Why not a larger batch? The 6 GB card allows ~24 × 512 without gradient
checkpointing. Small batches are noisier, so we use a slightly higher learning
rate and gradient accumulation to average two micro-batches.

## 2. Optimiser and schedule

| item | value | source / reason |
|---|---|---|
| optimiser | AdamW (fused) | Loshchilov & Hutter, 2017, arXiv:1711.05101 |
| β | (0.9, 0.95) | GPT-3 / Llama convention for large-batch LM training |
| weight decay | 0.1 | decoupled AdamW, applied to matrices only (we apply to all params; small scale, no measurable difference) |
| grad clip | 1.0 global norm | standard stability guard |
| LR | 1e-3 → 1e-4 cosine | small models tolerate higher LR; scaled from GPT-3's 6e-4 at 125 M (Brown et al., 2020) and nanoGPT's 6e-4 at 124 M, adjusted for a 20× smaller batch |
| warmup | 300 steps (1.25 %) | prevents early divergence with pre-norm + high LR |
| schedule | cosine to 10 % of peak | Loshchilov & Hutter, 2016, arXiv:1608.03983 |
| precision | bf16 autocast | Ada tensor cores; no loss scaler needed |
| seed | 1337 | fixed for reproducibility |

Learning-rate scaling caveat: the "correct" LR depends on batch size. Our
batch (24 576 tokens) is ~20× smaller than the 0.5 M-token batches used in
GPT-3/nanoGPT, and small batches tolerate (and need) higher LR. 1e-3 is the
value that produced a stable loss curve in the smoke test; the exact sweep is
left as an experiment (see `docs/07-decisions-log.md`).

## 3. Memory accounting (why it fits in 6 GB)

| component | size |
|---|---|
| weights (fp32 master) | 25.2 M × 4 B = 101 MB |
| Adam m, v (fp32) | 2 × 101 MB = 202 MB |
| gradients (fp32) | 101 MB |
| activations, 24×512, 8 layers, SDPA | ~0.3–0.6 GB |
| CUDA context + workspace | ~0.5 GB |
| **total** | **~1.2–1.5 GB** |

The 4050's 6 GB is therefore not the binding constraint for the dense model;
step time is. `torch.cuda.max_memory_allocated()` is logged at the end of each
run and recorded in the model card.

## 4. What is logged

`checkpoints/<run>/metrics.jsonl`, one JSON object per log/eval event:

```json
{"step": 20, "loss": 9.87, "lr": 0.00007, "tok_per_s": 41234, "elapsed_min": 0.4}
{"step": 1000, "val_loss": 6.12, "val_ppl": 454.2, "eval": true}
```

Checkpoints: `last.pt` every 2 000 steps (resume support), `best.pt` on
improved validation loss. Each checkpoint stores model, optimiser, step and
both configs, so a run can be resumed on another machine without the repo
state.

## 5. Evaluation protocol

* Validation loss on a held-out 4 M-token slice, averaged over 32 evenly
  spaced crops (deterministic, so runs are comparable).
* Perplexity = exp(loss); bits-per-byte = loss / ln(2) / bytes_per_token.
  BPB is tokenizer-independent and lets us compare the 32k tokenizer against
  Qwen's 151k tokenizer later.
* Qualitative: 8 fixed bilingual prompts, greedy + temperature 0.8, stored in
  `eval/samples/<run>.txt`.
* MoE vs dense are compared at **equal tokens** and at **equal wall-clock**;
  both numbers are reported because "equal compute" is the fair research
  comparison and "equal time" is what a practitioner cares about.

## 6. Failure modes seen in practice

| symptom | cause | fix |
|---|---|---|
| loss jumps to NaN | LR too high for batch, or bf16 overflow in attention | lower LR, clip, check data for long repeats |
| val loss << train loss | dedup leakage or too-small val set | bigger val slice, re-check dedup |
| step time degrades | GPU thermal throttling on laptop | cap power, add `--max-minutes`, run with a fan profile |
| OOM at eval | eval batch larger than train batch | eval uses the same batch size (24) |
