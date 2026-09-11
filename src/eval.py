"""Evaluate both trained variants on a held-out synthetic test set.

Sweeps: baseline long-run steps N in {10,30,50,100} (noise init),
        amortized K in {0,1,2,5,10} (InitNet init).
Reports J, boundary F, J&F, and per-frame wall-clock (batch=1, full pipeline).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from synthetic import MovingShapes
from models import Encoder, EnergyNet, InitNet
from sampler import langevin_refine
from metrics import iou, boundary_f


def load(ck, dev):
    enc, ene = Encoder().to(dev), EnergyNet().to(dev)
    enc.load_state_dict(ck['enc'])
    ene.load_state_dict(ck['ene'])
    init = None
    if 'init' in ck:
        init = InitNet().to(dev)
        init.load_state_dict(ck['init'])
        init.eval()
    enc.eval()
    ene.eval()
    return enc, ene, init


def run(enc, ene, init_net, x, steps, step_size):
    feat = enc(x)
    if init_net is not None:
        l0 = init_net(feat)
    else:
        l0 = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device) * 2.0
    logits = langevin_refine(ene, feat, l0, steps, step_size)
    return torch.sigmoid(logits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--n-test', type=int, default=64)
    ap.add_argument('--size', type=int, default=48)
    ap.add_argument('--step-size', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--time-frames', type=int, default=24)
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = MovingShapes(n=a.n_test, size=a.size, seed=a.seed)
    xs = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev)
    ys = torch.stack([ds[i][1] for i in range(len(ds))]).to(dev)

    results = []
    for name, fname, sweeps, use_init in [
        ('baseline-longrun', 'baseline.pt', [10, 30, 50, 100], False),
        ('amortized-Kstep', 'amortized.pt', [0, 1, 2, 5, 10], True),
    ]:
        ck = torch.load(Path(a.runs) / fname, map_location=dev, weights_only=False)
        enc, ene, init_net = load(ck, dev)
        if use_init and init_net is None:
            raise RuntimeError('amortized checkpoint missing init net')
        for s in sweeps:
            preds = run(enc, ene, init_net if use_init else None, xs, s, a.step_size)
            J = float(iou(preds, ys).mean())
            F = float(boundary_f(preds, ys).mean())
            t0 = time.perf_counter()
            for i in range(a.time_frames):
                run(enc, ene, init_net if use_init else None,
                    xs[i:i + 1], s, a.step_size)
            ms = (time.perf_counter() - t0) / a.time_frames * 1000
            rec = dict(model=name, steps=s, J=round(J, 4), F=round(F, 4),
                       JF=round((J + F) / 2, 4), ms_per_frame=round(ms, 2))
            results.append(rec)
            print(rec, flush=True)

    Path('results').mkdir(exist_ok=True)
    Path('results/synthetic_results.json').write_text(json.dumps(results, indent=1))
    print('wrote results/synthetic_results.json')


if __name__ == '__main__':
    main()
