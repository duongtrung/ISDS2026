"""DAVIS-16 loader. UNTESTED ON REAL DATA in the CPU sandbox - validate on the Spark.

Expected layout (DAVIS 2016/2017 zip):
  root/JPEGImages/480p/<seq>/00000.jpg ...
  root/Annotations/480p/<seq>/00000.png ...
  root/ImageSets/2016/{train,val}.txt      (one sequence name per line)
Precomputed RAFT flow (see precompute_flow.py):
  root/Flow/<seq>/00001.npy   (flow from frame t-1 -> t, H x W x 2, float32)

Sample t (t >= 1): x = [RGB_t (3) + flow (2)] -> 5 channels, y = binary mask_t.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

FLOW_SCALE = 20.0  # divide raw flow by this for roughly [-1,1] inputs


class Davis16(Dataset):
    def __init__(self, root, split='train', size=(256, 448)):
        self.root = Path(root)
        self.size = size  # (H, W), both divisible by 8
        names = (self.root / 'ImageSets' / '2016' / f'{split}.txt'
                 ).read_text().split()
        self.items = []
        for seq in names:
            frames = sorted((self.root / 'JPEGImages' / '480p' / seq).glob('*.jpg'))
            for f in frames[1:]:
                self.items.append((seq, f.stem))

    def __len__(self):
        return len(self.items)

    def _img(self, p):
        im = Image.open(p).convert('RGB').resize(self.size[::-1], Image.BILINEAR)
        return np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0

    def __getitem__(self, i):
        seq, stem = self.items[i]
        rgb = self._img(self.root / 'JPEGImages' / '480p' / seq / f'{stem}.jpg')
        ann = Image.open(self.root / 'Annotations' / '480p' / seq / f'{stem}.png').convert('L')
        ann = ann.resize(self.size[::-1], Image.NEAREST)
        y = (np.asarray(ann) > 0).astype(np.float32)[None]
        flow = np.load(self.root / 'Flow' / seq / f'{stem}.npy').astype(np.float32)
        flow = flow.transpose(2, 0, 1) / FLOW_SCALE
        x = np.concatenate([rgb, flow], axis=0)
        return torch.from_numpy(x), torch.from_numpy(y)
