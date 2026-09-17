"""Token stream loader over uint16 .bin shards with per-tier sampling weights.

Each tier directory holds files of concatenated documents separated by EOT. A batch is
assembled by drawing, for each sequence, a tier according to the mixture weights, then a
random contiguous window of seq_len+1 tokens from a random file of that tier (weighted by
file size). Random windows over a multi billion token stream approximate one epoch well
and need no shuffling pass. The mixture can be changed at any step (used for the anneal)."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch


class TierStream:
    def __init__(self, root: str, tiers: dict[str, float], seq_len: int, seed: int = 0, dtype=np.uint16):
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.tiers: dict[str, list[np.memmap]] = {}
        self.file_weights: dict[str, np.ndarray] = {}
        for t in tiers:
            files = sorted(glob.glob(os.path.join(root, t, "*.bin")))
            assert files, f"no shards for tier {t} under {root}"
            mm = [np.memmap(f, dtype=dtype, mode="r") for f in files]
            sizes = np.array([len(m) for m in mm], dtype=np.float64)
            self.tiers[t] = mm
            self.file_weights[t] = sizes / sizes.sum()
        self.set_mixture(tiers)

    def set_mixture(self, tiers: dict[str, float]):
        names = list(tiers)
        w = np.array([tiers[n] for n in names], dtype=np.float64)
        self.names, self.weights = names, w / w.sum()

    def tokens_available(self) -> dict[str, int]:
        return {t: int(sum(len(m) for m in mm)) for t, mm in self.tiers.items()}

    def batch(self, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        L = self.seq_len + 1
        out = np.empty((batch_size, L), dtype=np.int64)
        tier_idx = self.rng.choice(len(self.names), size=batch_size, p=self.weights)
        for i, ti in enumerate(tier_idx):
            t = self.names[ti]
            files = self.tiers[t]
            fi = self.rng.choice(len(files), p=self.file_weights[t])
            m = files[fi]
            start = self.rng.integers(0, len(m) - L)
            out[i] = m[start:start + L]
        x = torch.from_numpy(out[:, :-1]).to(device, non_blocking=True)
        y = torch.from_numpy(out[:, 1:]).to(device, non_blocking=True)
        return x, y


def heldout_batches(path: str, seq_len: int, max_tokens: int = 2_000_000, batch_size: int = 32):
    m = np.memmap(path, dtype=np.uint16, mode="r")
    n = min(len(m), max_tokens) // (seq_len + 1) * (seq_len + 1)
    arr = np.array(m[:n], dtype=np.int64).reshape(-1, seq_len + 1)
    for i in range(0, len(arr), batch_size):
        chunk = torch.from_numpy(arr[i:i + batch_size])
        yield chunk[:, :-1], chunk[:, 1:]
