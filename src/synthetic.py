"""Synthetic motion-segmentation benchmark for CPU smoke tests.

Each sample: two consecutive frames of a scene with one MOVING disc (target)
and one STATIC distractor disc. Appearance alone cannot separate the two;
the motion cue (frame difference) is required -> exercises the two-stream design.
Input  x: 4 x H x W  (RGB of frame t  +  1-channel abs frame difference)
Target y: 1 x H x W  (mask of the moving disc at frame t)
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class MovingShapes(Dataset):
    def __init__(self, n=4096, size=48, seed=0):
        self.n, self.size, self.seed = n, size, seed

    def __len__(self):
        return self.n

    @staticmethod
    def _disc(H, W, cx, cy, r):
        yy, xx = np.mgrid[0:H, 0:W]
        return ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 100003 + idx)
        H = W = self.size
        base = rng.uniform(0.25, 0.75, 3).astype(np.float32)[:, None, None] * np.ones(
            (3, H, W), dtype=np.float32)

        def noisy(img):
            return np.clip(img + rng.normal(0, 0.04, img.shape), 0, 1).astype(np.float32)

        r_m = int(rng.integers(5, 9))
        cx = int(rng.integers(r_m + 3, W - r_m - 3))
        cy = int(rng.integers(r_m + 3, H - r_m - 3))
        dx = int(rng.integers(-4, 5)); dy = int(rng.integers(-4, 5))
        if dx == 0 and dy == 0:
            dx = 3
        col_m = rng.uniform(0, 1, 3).astype(np.float32)
        r_s = int(rng.integers(4, 8))
        sx = int(rng.integers(r_s + 1, W - r_s - 1))
        sy = int(rng.integers(r_s + 1, H - r_s - 1))
        col_s = rng.uniform(0, 1, 3).astype(np.float32)
        static = self._disc(H, W, sx, sy, r_s)

        def frame(mcx, mcy):
            img = base.copy()
            img[:, static] = col_s[:, None]
            m = self._disc(H, W, mcx, mcy, r_m)
            img[:, m] = col_m[:, None]
            return noisy(img), m

        f0, _ = frame(cx, cy)
        cx1 = int(np.clip(cx + dx, r_m + 1, W - r_m - 2))
        cy1 = int(np.clip(cy + dy, r_m + 1, H - r_m - 2))
        f1, m1 = frame(cx1, cy1)
        diff = np.abs(f1 - f0).mean(0, keepdims=True).astype(np.float32)
        x = np.concatenate([f1, diff], axis=0)
        y = m1[None].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
