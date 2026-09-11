"""Fair tuning pass for the ORIGINAL long-run Langevin sampler (CLAUDE.md rule 3).

The long-run baseline is an inference-time object characterised by its sampler
hyperparameters. We grid over Langevin step-size x noise at a fixed number of
noise-initialised steps N on a DAVIS-16 val subset and report the configuration
that gives the baseline its best region J. That configuration is then used for the
baseline in the RQ2 frontier / RQ1 operating point, so no comparison is reported
against an under-tuned sampler.
"""
import argparse
import itertools
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from davis import Davis16
from models import Encoder, EnergyNet
from sampler import langevin_refine
from metrics import iou, boundary_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='runs_video/baseline.pt')
    ap.add_argument('--root', default='data/DAVIS')
    ap.add_argument('--n', type=int, default=100, help='steps N used during tuning')
    ap.add_argument('--val-n', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps-grid', type=float, nargs='+', default=[2.0, 5.0, 10.0])
    ap.add_argument('--noise-grid', type=float, nargs='+', default=[0.005, 0.02, 0.05])
    ap.add_argument('--out', default='results/sampler_tuning.json')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    enc = Encoder(5).to(dev); enc.load_state_dict(ck['enc']); enc.eval()
    ene = EnergyNet().to(dev); ene.load_state_dict(ck['ene']); ene.eval()

    ds = Davis16(a.root, 'val', (256, 448))
    g = torch.Generator().manual_seed(a.seed)
    idx = torch.randperm(len(ds), generator=g)[:a.val_n].tolist()
    dl = DataLoader(Subset(ds, idx), batch_size=8, num_workers=4)

    results = []
    for s, nz in itertools.product(a.steps_grid, a.noise_grid):
        torch.manual_seed(a.seed)  # same Langevin noise draw for every config
        Js = Fs = 0.0; n = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            feat = enc(x)
            l0 = torch.randn(feat.size(0), 1, feat.size(2), feat.size(3), device=dev) * 2.0
            pred = torch.sigmoid(langevin_refine(ene, feat, l0, a.n, s, nz))
            Js += float(iou(pred, y).sum()); Fs += float(boundary_f(pred, y).sum()); n += x.size(0)
        rec = dict(step_size=s, noise=nz, N=a.n, J=round(Js / n, 4), F=round(Fs / n, 4))
        results.append(rec); print(rec, flush=True)

    best = max(results, key=lambda r: r['J'])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(grid=results, best=best), indent=1))
    print('BEST', best, flush=True)


if __name__ == '__main__':
    main()
