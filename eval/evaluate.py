"""Evaluate a checkpoint: val loss, bits-per-byte, and bilingual samples.

Usage:
  python eval/evaluate.py --ckpt checkpoints/dense_25m/best.pt --tag dense_25m
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import build_model  # noqa: E402
from model.base import ModelConfig  # noqa: E402
from train.data import TokenDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--prompts", default="eval/prompts.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["model_cfg"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    meta = json.load(open("data/tokenized/meta.json", encoding="utf-8"))
    ds = TokenDataset("data/tokenized", cfg.max_seq_len)

    losses = []
    with torch.no_grad():
        for x, y in ds.val_loss_batches(args.batch, args.eval_batches):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            losses.append(loss.item())
    val = sum(losses) / len(losses)
    bpt = meta.get("bytes_per_token", 4.0)
    bpb = val / math.log(2) / bpt
    result = {
        "tag": args.tag,
        "step": ck.get("step"),
        "params": sum(p.numel() for p in model.parameters()),
        "val_loss": round(val, 4),
        "val_ppl": round(math.exp(val), 2),
        "bits_per_byte": round(bpb, 4),
    }
    print(json.dumps(result, indent=2))

    os.makedirs("eval/results", exist_ok=True)
    with open(f"eval/results/{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ---------------------------------------------------------- generations
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("tokenizer/artifacts/tokenizer.json")
    prompts = []
    with open(args.prompts, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))

    out_lines = [f"# {args.tag}  step={ck.get('step')}  val={val:.4f}  bpb={bpb:.4f}", ""]
    for item in prompts:
        ids = tok.encode(item["prompt"], add_special_tokens=False).ids
        x = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=0.8, top_k=50,
                               eos_id=tok.token_to_id("<|endoftext|>"))
        text = tok.decode(y[0].tolist(), skip_special_tokens=False)
        out_lines.append(f"[{item['lang']}] PROMPT: {item['prompt']}")
        out_lines.append(text.replace("<|endoftext|>", "").strip())
        out_lines.append("")

    os.makedirs("eval/samples", exist_ok=True)
    with open(f"eval/samples/{args.tag}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
