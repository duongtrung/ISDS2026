"""Precompute RAFT optical flow for DAVIS/FBMS sequences. Run on the Spark (GPU).

Usage:
  python src/precompute_flow.py --frames-root DAVIS/JPEGImages/480p \
      --out DAVIS/Flow --height 256 --width 448
Writes <out>/<seq>/<frame_t>.npy with flow from frame t-1 to t (H x W x 2).
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def load(p, hw, dev):
    im = Image.open(p).convert('RGB').resize((hw[1], hw[0]), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2, 0, 1))
    return (t / 127.5 - 1.0).unsqueeze(0).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=448)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).to(dev).eval()
    hw = (a.height, a.width)
    for seq_dir in sorted(Path(a.frames_root).iterdir()):
        if not seq_dir.is_dir():
            continue
        frames = sorted(list(seq_dir.glob('*.jpg')) + list(seq_dir.glob('*.png'))
                        + list(seq_dir.glob('*.bmp')))
        out_dir = Path(a.out) / seq_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for prev, cur in zip(frames, frames[1:]):
                flow = model(load(prev, hw, dev), load(cur, hw, dev))[-1][0]
                np.save(out_dir / f'{cur.stem}.npy',
                        flow.permute(1, 2, 0).cpu().numpy().astype(np.float32))
        print('done', seq_dir.name, flush=True)


if __name__ == '__main__':
    main()
