"""Across-seed aggregation for the (cheap) amortized/feed-forward path.

Evaluates seeds 0,1,2 (dirs runs_video, runs_video_s1, runs_video_s2) at the paper's
operating points on DAVIS-16 val, FBMS-59 and SegTrack-v2, and reports mean +/- std
ACROSS SEEDS (the honest error bar the reviewers asked for), plus the K=0 vs K=20
refinement direction per seed. The baseline is single-seed (runs_video/baseline.pt).
Writes results/seed_agg.json."""
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from davis import Davis16
from fbms import FBMS59
from segtrackv2 import SegTrackV2
from models import Encoder, EnergyNet, InitNet
from sampler import langevin_refine
from metrics import iou

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED_DIRS = {0: 'runs_video', 1: 'runs_video_s1', 2: 'runs_video_s2'}


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEV, weights_only=False)
    enc = Encoder(5).to(DEV); enc.load_state_dict(ck['enc']); enc.eval()
    init = InitNet().to(DEV); init.load_state_dict(ck['init']); init.eval()
    ene = None
    if 'ene' in ck:
        ene = EnergyNet().to(DEV); ene.load_state_dict(ck['ene']); ene.eval()
    return enc, ene, init


def ds_for(name):
    if name == 'davis':
        return Davis16('data/DAVIS', 'val', (256, 448))
    if name == 'fbms':
        return FBMS59('data/FBMS', 'Testset', (256, 448))
    return SegTrackV2('data/SegTrackv2', size=(256, 448))


def eval_J(enc, ene, init, ds, K, seed=0):
    torch.manual_seed(seed)
    dl = DataLoader(ds, batch_size=16, num_workers=4)
    tot, n = 0.0, 0
    for x, y in dl:
        x, y = x.to(DEV), y.to(DEV)
        feat = enc(x)
        pred = torch.sigmoid(init(feat)) if K == 0 else \
            torch.sigmoid(langevin_refine(ene, feat, init(feat), K, 5.0, 0.02))
        valid = y.flatten(1).sum(1) > 0
        tot += float(iou(pred, y)[valid].sum()); n += int(valid.sum())
    return tot / n


def agg(vals):
    t = torch.tensor(vals)
    return dict(mean=round(float(t.mean()), 4), std=round(float(t.std()), 4),
                per_seed=[round(v, 4) for v in vals])


def main():
    seeds = [s for s, d in SEED_DIRS.items() if Path(d, 'amortized.pt').exists()]
    print('seeds found:', seeds)
    out = {'seeds': seeds, 'davis': {}, 'fbms': {}, 'segtrack': {}}
    for name in ['davis', 'fbms', 'segtrack']:
        ds = ds_for(name)
        # feed-forward K=0, amortized K=5 on every dataset; also K=0/K=20 on DAVIS
        Ks = {'feedforward': [0], 'amortized': [5]}
        if name == 'davis':
            Ks['amortized'] = [0, 5, 20]
        for var, klist in Ks.items():
            for K in klist:
                vals = []
                for s in seeds:
                    enc, ene, init = load(Path(SEED_DIRS[s], f'{var}.pt'))
                    vals.append(eval_J(enc, ene, init, ds, K, seed=0))
                out[name][f'{var}_K{K}'] = agg(vals)
                print(name, var, f'K={K}', out[name][f'{var}_K{K}'], flush=True)
    Path('results/seed_agg.json').write_text(json.dumps(out, indent=1))
    print('wrote results/seed_agg.json')


if __name__ == '__main__':
    main()
