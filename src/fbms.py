"""FBMS-59 Testset loader (cross-dataset transfer target for RQ3).

Real FBMS-59 layout (as shipped in FBMS_Testset.zip):
  root/Testset/<seq>/<seq>_NNNN.jpg               dense frames
  root/Testset/<seq>/GroundTruth/<gtname>.ppm|pgm SPARSE annotations
  root/Flow/<seq>/<frame_stem>.npy                precomputed RAFT flow

Two GT conventions coexist in the Testset:
  * Sequences with '<seq>_NNNN_gt.ppm' (+ a same-named _gt.pgm and a PROB ppm):
    the COLOR .ppm is authoritative (background = white, each object a distinct
    color). The grayscale _gt.pgm is a lossy rendering that collapses some objects
    into the background, so we IGNORE it when a .ppm exists.
  * Sequences with only '<seq>_NN.pgm' (cars/people/marple/tennis...): the .pgm is
    the authoritative grayscale label image.
We therefore prefer .ppm over .pgm per frame. GT filename schemes vary; we derive
the frame stem by stripping a trailing '_gt' and any 'PROB'.

Only annotated frames that also have precomputed flow are used (a sequence's first
frame has no flow and is skipped, consistent with training).

Binarization: FBMS GT is a per-region LABEL image whose background value is not
consistent across sequences. We take background = the most frequent label along the
image border (white for the .ppm sequences, 0 for most .pgm ones) and treat every
other label/colour as moving foreground (union of all objects). Documented approx.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from davis import FLOW_SCALE

_FRAME_EXTS = ('.jpg', '.png', '.bmp', '.jpeg')


def _gt_to_stem(name):
    stem = Path(name).stem.replace('PROB', '')
    if stem.endswith('_gt'):
        stem = stem[:-3]
    return stem


def _foreground(arr):
    """Boolean foreground mask; background = most common label on the 1px border.
    Handles grayscale (H,W) and colour (H,W,3) label images."""
    if arr.ndim == 3:
        h, w, c = arr.shape
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
        cols, cnts = np.unique(border.reshape(-1, c), axis=0, return_counts=True)
        bg = cols[cnts.argmax()]
        return np.any(arr != bg, axis=2)
    b = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg = np.bincount(b.ravel()).argmax()
    return arr != bg


class FBMS59(Dataset):
    def __init__(self, root, split='Testset', size=(256, 448)):
        self.root = Path(root)
        self.size = size
        self.items = []
        for seq_dir in sorted((self.root / split).iterdir()):
            if not seq_dir.is_dir():
                continue
            gt_dir = seq_dir / 'GroundTruth'
            if not gt_dir.exists():
                continue
            frames = {p.stem: p for p in seq_dir.iterdir()
                      if p.suffix.lower() in _FRAME_EXTS}
            gt_by_stem = {}
            for p in sorted(gt_dir.glob('*.pgm')):          # lowest priority
                if 'PROB' not in p.name:
                    gt_by_stem.setdefault(_gt_to_stem(p.name), p)
            for p in sorted(gt_dir.glob('*.ppm')):          # prefer colour ppm
                if 'PROB' not in p.name:
                    gt_by_stem[_gt_to_stem(p.name)] = p
            for stem, gt in sorted(gt_by_stem.items()):
                frame = frames.get(stem)
                flow = self.root / 'Flow' / seq_dir.name / f'{stem}.npy'
                if frame is not None and flow.exists():
                    self.items.append((frame, gt, flow))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        frame, gt, flow_p = self.items[i]
        im = Image.open(frame).convert('RGB').resize(self.size[::-1], Image.BILINEAR)
        rgb = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        g = Image.open(gt)
        g = g.convert('RGB') if gt.suffix.lower() == '.ppm' else g.convert('L')
        fg = Image.fromarray((_foreground(np.asarray(g)) * 255).astype(np.uint8))
        fg = fg.resize(self.size[::-1], Image.NEAREST)
        y = (np.asarray(fg) > 127).astype(np.float32)[None]
        flow = np.load(flow_p).astype(np.float32).transpose(2, 0, 1) / FLOW_SCALE
        x = np.concatenate([rgb, flow], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)
