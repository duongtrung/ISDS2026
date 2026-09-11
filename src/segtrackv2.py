"""SegTrack-v2 loader (second cross-dataset transfer target for RQ3).

SegTrack-v2 (Li et al., ICCV 2013) has 14 sequences, some with multiple moving
objects. Standard release layout:
  root/JPEGImages/<seq>/*.png|*.bmp        # frames
  root/GroundTruth/<seq>/*.png             # single-object: masks directly, OR
  root/GroundTruth/<seq>/<obj>/*.png       # multi-object: one subdir per object
  root/Flow/<seq>/<frame_stem>.npy         # precomputed RAFT flow (t-1 -> t)

For binary motion segmentation we union all object masks per frame into a single
foreground. GT is matched to frames by sorted index within a sequence (robust to
heterogeneous filename schemes across sequences). Frame 0 is skipped (no flow).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from davis import FLOW_SCALE

_EXTS = ('*.png', '*.bmp', '*.jpg', '*.jpeg')


def _list_images(d):
    fs = []
    for e in _EXTS:
        fs += list(d.glob(e))
    return sorted(fs)


class SegTrackV2(Dataset):
    def __init__(self, root, split=None, size=(256, 448)):
        self.root = Path(root)
        self.size = size
        img_root = self.root / 'JPEGImages'
        gt_root = self.root / 'GroundTruth'
        flow_root = self.root / 'Flow'
        self.items = []
        for seq_dir in sorted(img_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            frames = _list_images(seq_dir)
            gseq = gt_root / seq_dir.name
            if not gseq.exists() or len(frames) < 2:
                continue
            subdirs = [d for d in sorted(gseq.iterdir()) if d.is_dir()]
            obj_dirs = subdirs if subdirs else [gseq]
            # stem -> gt path per object dir; pair GT to frames by FILENAME stem
            # (robust to sequences like 'worm' whose GT is off-by-one vs frames).
            gt_by_obj = [{p.stem: p for p in _list_images(d)} for d in obj_dirs]
            for fr in frames[1:]:                       # skip first frame (no flow)
                stem = fr.stem
                flow = flow_root / seq_dir.name / f'{stem}.npy'
                if not flow.exists():
                    continue
                gts = [g[stem] for g in gt_by_obj if stem in g]
                if not gts:
                    continue
                self.items.append((fr, gts, flow))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        frame, gts, flow_p = self.items[i]
        im = Image.open(frame).convert('RGB').resize(self.size[::-1], Image.BILINEAR)
        rgb = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        y = np.zeros(self.size, dtype=np.float32)
        for g in gts:
            ann = Image.open(g).convert('L').resize(self.size[::-1], Image.NEAREST)
            y = np.maximum(y, (np.asarray(ann) > 0).astype(np.float32))
        y = y[None]
        flow = np.load(flow_p).astype(np.float32).transpose(2, 0, 1) / FLOW_SCALE
        x = np.concatenate([rgb, flow], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)
