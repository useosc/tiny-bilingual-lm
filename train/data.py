"""Token dataset: memory-mapped uint16 stream with random crops."""
from __future__ import annotations

import numpy as np
import torch


class TokenDataset:
    def __init__(self, root: str, seq_len: int):
        self.seq_len = seq_len
        self.train = np.memmap(f"{root}/train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(f"{root}/val.bin", dtype=np.uint16, mode="r")

    def get_batch(self, batch_size: int, split: str = "train", device: str = "cuda"):
        data = self.train if split == "train" else self.val
        max_start = len(data) - self.seq_len - 1
        if max_start <= 0:
            raise ValueError(f"{split} split too small: {len(data)} tokens")
        ix = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([data[i : i + self.seq_len] for i in ix])
        y = np.stack([data[i + 1 : i + 1 + self.seq_len] for i in ix])
        x = torch.from_numpy(x.astype(np.int64))
        y = torch.from_numpy(y.astype(np.int64))
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    def val_loss_batches(self, batch_size: int, n_batches: int):
        """Deterministic, evenly spaced validation crops."""
        data = self.val
        max_start = len(data) - self.seq_len - 1
        step = max(1, max_start // max(n_batches, 1))
        for i in range(n_batches):
            start = min(i * step, max_start)
            x = np.stack([data[start : start + self.seq_len]])
            y = np.stack([data[start + 1 : start + 1 + self.seq_len]])
            yield torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))
