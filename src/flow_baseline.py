"""Parameter-free reference: threshold optical-flow magnitude (per-frame Otsu) into a
moving-foreground mask. No training, no learned energy -- it anchors the learned methods
against trivial motion cues. CPU only (reads precomputed flow via the loaders)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from davis import Davis16, FLOW_SCALE
from fbms import FBMS59
from segtrackv2 import SegTrackV2
from metrics import iou, boundary_f


def otsu(v):
    hist, edges = np.histogram(v, bins=64, range=(0, 1))
    p = hist.astype(float) / max(hist.sum(), 1)
    mids = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(p); mu = np.cumsum(p * mids); muT = mu[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        sb = (muT * w - mu) ** 2 / (w * (1 - w))
    k = int(np.nanargmax(sb))
    return mids[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/ext_results.json')
    a = ap.parse_args()
    dsets = {'davis': Davis16('data/DAVIS', 'val', (256, 448)),
             'fbms': FBMS59('data/FBMS', 'Testset', (256, 448)),
             'segtrack': SegTrackV2('data/SegTrackv2', size=(256, 448))}
    results = []
    for name, ds in dsets.items():
        dl = DataLoader(ds, batch_size=8, num_workers=4)
        Jv, Fv = [], []
        for x, y in dl:
            flow = x[:, 3:5] * FLOW_SCALE
            mag = torch.sqrt((flow ** 2).sum(1, keepdim=True))
            pred = torch.zeros_like(mag)
            for i in range(mag.size(0)):
                m = mag[i, 0].numpy()
                mn, mx = float(m.min()), float(m.max())
                mnorm = (m - mn) / (mx - mn + 1e-6)
                pred[i, 0] = torch.from_numpy((mnorm > otsu(mnorm.ravel())).astype('float32'))
            valid = y.flatten(1).sum(1) > 0
            Jv.append(iou(pred, y)[valid]); Fv.append(boundary_f(pred, y)[valid])
        Jv, Fv = torch.cat(Jv), torch.cat(Fv)
        rec = dict(dataset=name, variant='flowthresh', steps=0, modality='flow', ch=0,
                   J=round(float(Jv.mean()), 4), F=round(float(Fv.mean()), 4),
                   JF=round(float((Jv.mean() + Fv.mean()) / 2), 4),
                   ms_per_frame=0.0, n=int(Jv.numel()))
        results.append(rec); print(rec, flush=True)
    p = Path(a.out)
    prev = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(prev + results, indent=1))


if __name__ == '__main__':
    main()
