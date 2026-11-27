"""Plot training curves for all runs -> docs/assets/loss_curves.png."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = {
    "dense_25m": ("checkpoints/dense_25m/metrics.jsonl", "dense 25.2M"),
    "moe_4e": ("checkpoints/moe_4e/metrics.jsonl", "MoE 44.1M (top-2)"),
    "looped_16m": ("checkpoints/looped_16m/metrics.jsonl", "looped 15.7M (2x4)"),
}


def load(path: str):
    train, val = [], []
    if not os.path.exists(path):
        return train, val
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("eval"):
            val.append((r["step"], r["val_loss"]))
        elif "loss" in r:
            train.append((r["step"], r["loss"]))
    # drop a 50-step smoke prefix if present (duplicate low steps)
    seen = set()
    train = [x for x in train if not (x[0] in seen or seen.add(x[0]))]
    seen = set()
    val = [x for x in val if not (x[0] in seen or seen.add(x[0]))]
    return train, val


def main() -> None:
    os.makedirs("docs/assets", exist_ok=True)
    plt.figure(figsize=(7.5, 4.2), dpi=150)
    for name, (path, label) in RUNS.items():
        tr, va = load(path)
        if not tr and not va:
            continue
        if tr:
            xs, ys = zip(*tr)
            plt.plot(xs, ys, lw=1.0, alpha=0.55, label=f"{label} train")
        if va:
            xs, ys = zip(*va)
            plt.plot(xs, ys, "o-", ms=3, lw=1.6, label=f"{label} val")
    plt.xlabel("step")
    plt.ylabel("cross-entropy (nats)")
    plt.title("tiny-bilingual-lm training curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("docs/assets/loss_curves.png")
    print("saved docs/assets/loss_curves.png")


if __name__ == "__main__":
    main()
