"""Train either variant on the synthetic benchmark.

baseline : negatives from long-run Langevin initialized at noise (original EBM).
amortized: negatives from InitNet one-shot proposal + K short Langevin steps;
           InitNet trained by cooperative teaching (regress the refined sample).

Shared loss: contrastive divergence  E(pos) - E(neg)
           + hinge vs structured hard negatives (spatially shifted GT masks)
           + L2 energy-magnitude regularizer.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from synthetic import MovingShapes
from models import Encoder, EnergyNet, InitNet
from sampler import langevin_refine
from metrics import iou


def shift_neg(y):
    B = y.size(0)
    sx = torch.randint(-12, 13, (B,))
    sy = torch.randint(-12, 13, (B,))
    return torch.stack([
        torch.roll(y[i], shifts=(int(sy[i]), int(sx[i])), dims=(1, 2))
        for i in range(B)
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['baseline', 'amortized'], required=True)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--size', type=int, default=48)
    ap.add_argument('--neg-steps', type=int, default=None,
                    help='Langevin steps for negatives (default: 30 baseline, 5 amortized)')
    ap.add_argument('--step-size', type=float, default=5.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = a.neg_steps if a.neg_steps is not None else (30 if a.variant == 'baseline' else 5)

    ds = MovingShapes(n=4096, size=a.size, seed=a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0)

    enc, ene = Encoder().to(dev), EnergyNet().to(dev)
    params = list(enc.parameters()) + list(ene.parameters())
    init = None
    if a.variant == 'amortized':
        init = InitNet().to(dev)
        params += list(init.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    bce = nn.BCEWithLogitsLoss()

    it, t0, log = 0, time.time(), []
    while it < a.iters:
        for x, y in dl:
            if it >= a.iters:
                break
            x, y = x.to(dev), y.to(dev)
            feat = enc(x)
            pos_e = ene(feat, y)

            if a.variant == 'baseline':
                l0 = torch.randn_like(y) * 2.0
            else:
                init_logits = init(feat.detach())
                l0 = init_logits
            neg_logits = langevin_refine(ene, feat.detach(), l0, steps, a.step_size)
            neg_mask = torch.sigmoid(neg_logits)
            neg_e = ene(feat, neg_mask)

            hard = shift_neg(y).to(dev)
            hard_e = ene(feat, hard)

            loss = (pos_e.mean() - neg_e.mean()
                    + torch.relu(1.0 + pos_e - hard_e).mean()
                    + 0.05 * (pos_e ** 2 + neg_e ** 2).mean())
            if a.variant == 'amortized':
                loss = loss + bce(init_logits, (neg_mask > 0.5).float().detach())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if it % 50 == 0 or it == a.iters - 1:
                rec = dict(it=it, loss=round(float(loss), 3),
                           pos_e=round(float(pos_e.mean()), 3),
                           neg_e=round(float(neg_e.mean()), 3),
                           neg_iou=round(float(iou(neg_mask.detach(), y).mean()), 3),
                           t=round(time.time() - t0, 1))
                log.append(rec)
                print(rec, flush=True)
            it += 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = dict(enc=enc.state_dict(), ene=ene.state_dict(),
              variant=a.variant, args=vars(a))
    if init is not None:
        ck['init'] = init.state_dict()
    torch.save(ck, out / f'{a.variant}.pt')
    (out / f'{a.variant}_trainlog.json').write_text(json.dumps(log, indent=1))
    print('saved', out / f'{a.variant}.pt', 'total_s', round(time.time() - t0, 1))


if __name__ == '__main__':
    main()
