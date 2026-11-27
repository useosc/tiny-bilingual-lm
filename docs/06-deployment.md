# 06 — Deployment: ONNX + in-browser inference

## Export

`export/onnx_export.py` builds a **single decode-step graph**:

* inputs: `input_ids[1,1]`, `pos[1]`, and 16 KV tensors `past_k_i / past_v_i`
  of shape `[1, 2, past_len, 64]`;
* outputs: `logits[1,1,32768]` and the updated KV tensors;
* attention is written with explicit `MatMul → Softmax → MatMul` (no SDPA) so
  onnxruntime-web runs it everywhere;
* `past_len` is dynamic, so the same graph handles the first token
  (`past_len = 0`) and token 512.

We export **only the dense model**. MoE/looped stay as training references.

```bash
python export/onnx_export.py --ckpt checkpoints/dense_25m/best.pt --out web/model/model.onnx
python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; \
quantize_dynamic('web/model/model.onnx','web/model/model_int8.onnx', \
weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul','Gemm'])"
```

| artifact | size |
|---|---|
| fp32 ONNX (tied weight duplicated by the exporter) | ~151 MB |
| int8 dynamic-quantised | **~76 MB** (embedding stays fp32) |
| tokenizer.json | ~1.5 MB |

76 MB is under GitHub's 100 MB per-file limit and is what the page loads.

## Browser runtime

`web/` is a static site: `index.html`, `app.js`, `bpe.js`, plus
`web/model/{model_int8.onnx,tokenizer.json,config.json}`.

* onnxruntime-web 1.20 (WASM, 4 threads), model fetched with a progress bar;
* `bpe.js` re-implements byte-level BPE in ~150 lines (NFC → GPT-2 regex split →
  byte map → merge ranks), matching the Python tokenizer for everyday text;
* generation keeps the KV cache in JS typed arrays, so each new token costs one
  decode step (3–45 ms on CPU in the Python ORT test; expect 5–20 tok/s in the
  browser depending on machine);
* the page is honest about the model: 25 M base model, continuation-style chat,
  no instruction tuning.

## Why not WebGPU / transformers.js

* transformers.js needs the architecture implemented in JS; our model is
  custom, so a hand-written ONNX graph is less work and exactly reproducible.
* WebGPU would be faster but adds a second code path; WASM works on every
  laptop including the one that trained the model.
