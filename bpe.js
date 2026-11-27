// Minimal byte-level BPE encoder/decoder for the tokenizer produced by
// tokenizer/train_tokenizer.py. Mirrors HuggingFace tokenizers semantics:
// NFC normalise -> GPT-2 regex split -> byte map -> rank-based BPE merges.

export class BPE {
  constructor(tokJson) {
    this.vocab = tokJson.model.vocab;            // token -> id
    this.merges = tokJson.model.merges || [];    // ["a b", ...] or [["a","b"], ...]
    this.idToToken = new Map();
    for (const [t, id] of Object.entries(this.vocab)) this.idToToken.set(id, t);

    this.ranks = new Map();
    this.merges.forEach((m, i) => {
      const key = Array.isArray(m) ? m.join(" ") : m;
      this.ranks.set(key, i);
    });

    // special tokens
    this.special = new Map();
    (tokJson.added_tokens || []).forEach((t) => {
      this.special.set(t.content, t.id);
      this.idToToken.set(t.id, t.content);
    });

    const { enc, dec } = bytesToUnicode();
    this.byteEnc = enc;
    this.byteDec = dec;

    this.pattern =
      /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
    this.cache = new Map();
  }

  _bpe(chars) {
    let word = chars;
    while (word.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(word[i] + " " + word[i + 1]);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      word = word
        .slice(0, bestIdx)
        .concat([word[bestIdx] + word[bestIdx + 1]], word.slice(bestIdx + 2));
    }
    return word;
  }

  tokenizePiece(piece) {
    let cached = this.cache.get(piece);
    if (cached) return cached;
    const bytes = new TextEncoder().encode(piece);
    const chars = Array.from(bytes).map((b) => this.byteEnc[b]);
    const merged = this._bpe(chars);
    this.cache.set(piece, merged);
    return merged;
  }

  encode(text) {
    text = text.normalize("NFC");
    const ids = [];
    // Protect special tokens (they are matched literally).
    const parts = text.split(/(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)/g);
    for (const part of parts) {
      if (!part) continue;
      if (this.special.has(part)) {
        ids.push(this.special.get(part));
        continue;
      }
      for (const m of part.matchAll(this.pattern)) {
        const toks = this.tokenizePiece(m[0]);
        for (const t of toks) {
          const id = this.vocab[t];
          if (id !== undefined) ids.push(id);
        }
      }
    }
    return ids;
  }

  decode(ids) {
    const bytes = [];
    for (const id of ids) {
      const tok = this.idToToken.get(id);
      if (tok === undefined) continue;
      if (this.special.has(tok)) {
        bytes.push(...new TextEncoder().encode(tok));
        continue;
      }
      for (const ch of tok) {
        const b = this.byteDec[ch];
        if (b === undefined) bytes.push(...new TextEncoder().encode(ch));
        else bytes.push(b);
      }
    }
    return new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes));
  }
}

// GPT-2 byte<->unicode table.
function bytesToUnicode() {
  const bs = [];
  for (let i = "!".charCodeAt(0); i <= "~".charCodeAt(0); i++) bs.push(i);
  for (let i = 0xa1; i <= 0xac; i++) bs.push(i);
  for (let i = 0xae; i <= 0xff; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  const used = new Set(bs);
  for (let b = 0; b < 256; b++) {
    if (!used.has(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const enc = {};
  const dec = {};
  for (let i = 0; i < bs.length; i++) {
    enc[bs[i]] = String.fromCharCode(cs[i]);
    dec[String.fromCharCode(cs[i])] = bs[i];
  }
  return { enc, dec };
}
