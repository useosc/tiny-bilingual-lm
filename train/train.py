"""Config-driven trainer for the tiny bilingual LM family.

Usage:
  python train/train.py --config configs/dense_25m.toml
  python train/train.py --config configs/dense_25m.toml --max-minutes 240
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import tomllib

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import TokenDataset  # noqa: E402
from model import build_model, count_params  # noqa: E402
from model.base import ModelConfig  # noqa: E402


def get_lr(step: int, cfg: dict) -> float:
    warmup = cfg["warmup_steps"]
    if step < warmup:
        return cfg["lr"] * (step + 1) / warmup
    if step >= cfg["max_steps"]:
        return cfg["min_lr"]
    ratio = (step - warmup) / max(1, cfg["max_steps"] - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return cfg["min_lr"] + coeff * (cfg["lr"] - cfg["min_lr"])


def evaluate(model, ds: TokenDataset, cfg: dict, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in ds.val_loss_batches(cfg["batch_size"], cfg["eval_batches"]):
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_ckpt(path: str, model, opt, step: int, cfg: dict, model_cfg: ModelConfig) -> None:
    tmp = path + ".tmp"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "train_cfg": cfg,
            "model_cfg": model_cfg.__dict__,
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--smoke", action="store_true", help="50 steps, tiny eval")
    args = ap.parse_args()

    with open(args.config, "rb") as f:
        full = tomllib.load(f)
    mcfg = ModelConfig.from_dict(full["model"])
    cfg = full["train"]
    cfg.setdefault("grad_accum", 1)
    cfg.setdefault("eval_interval", 500)
    cfg.setdefault("eval_batches", 20)
    cfg.setdefault("log_interval", 10)
    cfg.setdefault("save_interval", 1000)
    cfg.setdefault("weight_decay", 0.1)
    cfg.setdefault("beta1", 0.9)
    cfg.setdefault("beta2", 0.95)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("seed", 1337)
    cfg.setdefault("num_workers", 0)
    if args.smoke:
        cfg["max_steps"] = 50
        cfg["eval_interval"] = 25
        cfg["eval_batches"] = 4
        cfg["save_interval"] = 25

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(cfg["out_dir"], exist_ok=True)
    run_log = os.path.join(cfg["out_dir"], "metrics.jsonl")
    log_f = open(run_log, "a", encoding="utf-8")

    ds = TokenDataset(cfg["data_dir"], mcfg.max_seq_len)
    model = build_model(mcfg).to(device)
    n_all = count_params(model)
    n_nonemb = count_params(model, non_embedding=True)
    print(f"model={mcfg.arch} params={n_all/1e6:.2f}M (non-embedding {n_nonemb/1e6:.2f}M)")

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"], fused=True,
    )
    start_step = 0
    ckpt_path = os.path.join(cfg["out_dir"], "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        print(f"resumed from step {start_step}")

    tokens_per_step = cfg["batch_size"] * cfg["grad_accum"] * mcfg.max_seq_len
    print(f"tokens/step={tokens_per_step:,} max_steps={cfg['max_steps']}")
    model.train()
    t0 = time.time()
    t_log = t0
    loss_acc = 0.0
    best_val = float("inf")

    for step in range(start_step, cfg["max_steps"]):
        lr = get_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(cfg["grad_accum"]):
            x, y = ds.get_batch(cfg["batch_size"], "train", device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss, _ = model(x, y)
                if mcfg.arch == "moe" and model.aux_loss is not None:
                    loss = loss + model.aux_loss
            (loss / cfg["grad_accum"]).backward()
            total_loss += loss.item() / cfg["grad_accum"]

        if cfg["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        loss_acc += total_loss
        if (step + 1) % cfg["log_interval"] == 0:
            now = time.time()
            tok_s = tokens_per_step * cfg["log_interval"] / max(now - t_log, 1e-6)
            t_log = now
            rec = {
                "step": step + 1,
                "loss": round(loss_acc / cfg["log_interval"], 4),
                "lr": round(lr, 6),
                "tok_per_s": round(tok_s),
                "elapsed_min": round((now - t0) / 60, 2),
            }
            print(json.dumps(rec), flush=True)
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            loss_acc = 0.0

        if (step + 1) % cfg["eval_interval"] == 0 or (step + 1) == cfg["max_steps"]:
            val = evaluate(model, ds, cfg, device)
            rec = {"step": step + 1, "val_loss": round(val, 4), "val_ppl": round(math.exp(val), 2)}
            print("EVAL", json.dumps(rec), flush=True)
            log_f.write(json.dumps({**rec, "eval": True}) + "\n")
            log_f.flush()
            if val < best_val:
                best_val = val
                save_ckpt(os.path.join(cfg["out_dir"], "best.pt"), model, opt, step + 1, cfg, mcfg)

        if (step + 1) % cfg["save_interval"] == 0:
            save_ckpt(ckpt_path, model, opt, step + 1, cfg, mcfg)

        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            print(f"time budget reached at step {step + 1}", flush=True)
            save_ckpt(ckpt_path, model, opt, step + 1, cfg, mcfg)
            break

    save_ckpt(ckpt_path, model, opt, cfg["max_steps"], cfg, mcfg)
    log_f.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min, best val={best_val:.4f}")


if __name__ == "__main__":
    main()
