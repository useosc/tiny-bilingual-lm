// Node smoke test for the exact browser inference path (onnxruntime-web WASM).
import fs from "fs";
import * as ort from "onnxruntime-web";
import { BPE } from "../web/bpe.js";

ort.env.wasm.numThreads = 4;
const WEB = "../web/model";

const cfg = JSON.parse(fs.readFileSync(`${WEB}/config.json`, "utf8"));
const tokJson = JSON.parse(fs.readFileSync(`${WEB}/tokenizer.json`, "utf8"));
const bpe = new BPE(tokJson);
const modelBuf = fs.readFileSync(`${WEB}/model_int8.onnx`);
const session = await ort.InferenceSession.create(modelBuf, { executionProviders: ["wasm"] });

const L = cfg.n_layers, KV = cfg.n_kv_heads, DH = cfg.head_dim;
let past;

function initPast() {
  past = [];
  for (let i = 0; i < L; i++)
    for (let j = 0; j < 2; j++)
      past.push(new ort.Tensor("float32", new Float32Array(0), [1, KV, 0, DH]));
}

async function step(id, pos) {
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(id)]), [1, 1]),
    pos: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos)]), [1]),
  };
  for (let i = 0; i < L; i++) {
    feeds[`past_k_${i}`] = past[2 * i];
    feeds[`past_v_${i}`] = past[2 * i + 1];
  }
  const out = await session.run(feeds);
  past = [];
  for (let i = 0; i < L; i++) past.push(out[`new_k_${i}`], out[`new_v_${i}`]);
  return out.logits.data;
}

const prompt = process.argv[2] || "人工智能是";
initPast();
let logits = null, pos = 0;
for (const id of bpe.encode(prompt)) logits = await step(id, pos++);
const ids = [];
const t0 = performance.now();
for (let i = 0; i < 24; i++) {
  let best = 0;
  for (let j = 1; j < logits.length; j++) if (logits[j] > logits[best]) best = j;
  ids.push(best);
  logits = await step(best, pos++);
}
const dt = (performance.now() - t0) / 1000;
console.log("PROMPT:", prompt);
console.log(bpe.decode(ids));
console.log(`${(ids.length / dt).toFixed(1)} tok/s (WASM, node)`);
