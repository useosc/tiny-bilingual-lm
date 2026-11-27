import { BPE } from "./bpe.js";

const logEl = document.getElementById("log");
const inputEl = document.getElementById("input");
const sendEl = document.getElementById("send");
const statusEl = document.getElementById("status");
const loadMetaEl = document.getElementById("loadmeta");

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);

let tok = null;
let session = null;
let cfg = null;
let past = null;

const EOS = () => tok.special.get("<|endoftext|>");

async function load() {
  const t0 = performance.now();
  const [tokJson, modelCfg] = await Promise.all([
    fetch("model/tokenizer.json").then((r) => r.json()),
    fetch("model/config.json").then((r) => r.json()),
  ]);
  tok = new BPE(tokJson);
  cfg = modelCfg;
  statusEl.textContent = "加载 ONNX 模型…";

  const resp = await fetch("model/model_int8.onnx");
  const total = Number(resp.headers.get("content-length") || 0);
  const reader = resp.body.getReader();
  const chunks = [];
  let got = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) statusEl.textContent = `下载模型 ${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`;
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }

  session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
  initPast();
  sendEl.disabled = false;
  const secs = ((performance.now() - t0) / 1000).toFixed(1);
  statusEl.textContent = `就绪（${secs}s）· ${cfg.params_m}M params · 点发送试试`;
  loadMetaEl.textContent = "";
}

function initPast() {
  past = [];
  for (let i = 0; i < cfg.n_layers; i++) {
    for (const kind of ["k", "v"]) {
      past.push(new ort.Tensor("float32", new Float32Array(0), [1, cfg.n_kv_heads, 0, cfg.head_dim]));
    }
  }
}

async function step(tokenId, pos) {
  const feeds = {
    input_ids: new ort.Tensor("int64", BigInt64Array.from([BigInt(tokenId)]), [1, 1]),
    pos: new ort.Tensor("int64", BigInt64Array.from([BigInt(pos)]), [1]),
  };
  for (let i = 0; i < cfg.n_layers; i++) {
    feeds[`past_k_${i}`] = past[2 * i];
    feeds[`past_v_${i}`] = past[2 * i + 1];
  }
  const out = await session.run(feeds);
  past = [];
  for (let i = 0; i < cfg.n_layers; i++) {
    past.push(out[`new_k_${i}`], out[`new_v_${i}`]);
  }
  return out.logits.data;
}

function sample(logits, temperature, topk) {
  const v = Float32Array.from(logits);
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
    return best;
  }
  // softmax over top-k
  const idx = Array.from(v.keys()).sort((a, b) => v[b] - v[a]).slice(0, topk);
  let max = -Infinity;
  for (const i of idx) max = Math.max(max, v[i] / temperature);
  const probs = idx.map((i) => Math.exp(v[i] / temperature - max));
  const sum = probs.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let j = 0; j < idx.length; j++) {
    r -= probs[j];
    if (r <= 0) return idx[j];
  }
  return idx[idx.length - 1];
}

async function generate(prompt, maxTokens = 80, temperature = 0.9, topk = 40) {
  initPast();
  const ids = tok.encode(prompt).slice(-(cfg.max_seq_len - maxTokens - 8));
  let pos = 0;
  let logits = null;
  for (const id of ids) {
    logits = await step(id, pos++);
  }
  const outIds = [];
  const t0 = performance.now();
  for (let i = 0; i < maxTokens; i++) {
    const id = sample(logits, temperature, topk);
    if (id === EOS()) break;
    outIds.push(id);
    // stream text
    logEl.lastElementChild.textContent = tok.decode(outIds);
    logits = await step(id, pos++);
    if (pos >= cfg.max_seq_len) break;
  }
  const dt = (performance.now() - t0) / 1000;
  return { text: tok.decode(outIds), speed: outIds.length / dt };
}

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  logEl.appendChild(div);
  return div;
}

async function onSend() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = "";
  addMsg("user", text);
  sendEl.disabled = true;
  const bot = addMsg("bot", "");
  statusEl.textContent = "生成中…";
  try {
    const prompt = `用户：${text}\n助手：`;
    const { text: out, speed } = await generate(prompt, 80, 0.9, 40);
    bot.textContent = out || "（空）";
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${speed.toFixed(1)} tok/s · greedy-ish sampling`;
    bot.appendChild(meta);
    statusEl.textContent = "就绪";
  } catch (e) {
    bot.textContent = "出错：" + e.message;
    statusEl.textContent = "出错：" + e.message;
  } finally {
    sendEl.disabled = false;
  }
}

sendEl.addEventListener("click", onSend);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(); }
});

load().catch((e) => { statusEl.textContent = "加载失败：" + e.message; });
