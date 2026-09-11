"""Train one variant on real video data (DAVIS-16). 5-channel input (RGB+flow).

Variants
  baseline    : EBM; negatives from long-run Langevin initialized at noise
                (the "original" model). Fair-baseline knobs below.
  amortized   : EBM; negatives from InitNet one-shot proposal + K short Langevin
                steps; InitNet trained by cooperative teaching (regress refined).
  feedforward : purely supervised encoder+head, NO energy and NO Langevin.
                Ablation isolating what the learned energy + refinement add over
                a same-capacity feed-forward segmenter.

Fair-baseline tuning (CLAUDE.md rule 3): --step-size, --noise, --neg-steps
(N up to 400), and --persistent (persistent-CD replay buffer) are exposed so the
original long-run sampler gets a genuine tuning pass before any comparison.

Shared EBM loss: contrastive divergence E(pos)-E(neg) + hinge vs shifted-GT hard
negatives + L2 energy-magnitude regularizer.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from davis import Davis16
from models import Encoder, EnergyNet, InitNet, slice_modality, MODALITY_CH
from sampler import langevin_refine
from metrics import iou
from train import shift_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['baseline', 'amortized', 'feedforward'],
                    required=True)
    ap.add_argument('--davis-root', required=True)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=448)
    ap.add_argument('--neg-steps', type=int, default=None,
                    help='Langevin steps for negatives (default: 100 baseline, 5 amortized)')
    ap.add_argument('--step-size', type=float, default=5.0)
    ap.add_argument('--noise', type=float, default=0.02,
                    help='Langevin noise std (baseline tuning knob)')
    ap.add_argument('--ebm-weight', type=float, default=0.1,
                    help='amortized: weight of the EBM/CD terms relative to the init BCE')
    ap.add_argument('--ch', type=int, default=32, help='backbone width (capacity scenario)')
    ap.add_argument('--modality', choices=['both', 'rgb', 'flow'], default='both')
    ap.add_argument('--persistent', action='store_true',
                    help='persistent-CD replay buffer for the baseline negatives')
    ap.add_argument('--buf-size', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='runs_video')
    ap.add_argument('--tag', default=None,
                    help='checkpoint basename override (default: variant)')
    ap.add_argument('--eval-every', type=int, default=0,
                    help='periodic val-J eval interval in iters (0=off) for RQ1 curves')
    ap.add_argument('--val-n', type=int, default=128,
                    help='DAVIS-val subset size for periodic eval')
    ap.add_argument('--eval-steps', type=int, default=None,
                    help='refine steps used in periodic val (default: training steps; 0 feedforward)')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    steps = a.neg_steps if a.neg_steps is not None else (100 if a.variant == 'baseline' else 5)

    ds = Davis16(a.davis_root, 'train', (a.height, a.width))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4,
                    pin_memory=(dev == 'cuda'), drop_last=True)

    val_dl, eval_steps = None, 0
    if a.eval_every > 0:
        from torch.utils.data import Subset
        vds = Davis16(a.davis_root, 'val', (a.height, a.width))
        g = torch.Generator().manual_seed(a.seed)
        vidx = torch.randperm(len(vds), generator=g)[:a.val_n].tolist()
        val_dl = DataLoader(Subset(vds, vidx), batch_size=8, num_workers=2)
        eval_steps = a.eval_steps if a.eval_steps is not None else (
            0 if a.variant == 'feedforward' else steps)

    enc = Encoder(in_ch=MODALITY_CH[a.modality], ch=a.ch).to(dev)
    params = list(enc.parameters())
    ene = init = None
    if a.variant in ('baseline', 'amortized'):
        ene = EnergyNet(feat_ch=a.ch, ch=a.ch).to(dev)
        params += list(ene.parameters())
    if a.variant in ('amortized', 'feedforward'):
        init = InitNet(feat_ch=a.ch, ch=a.ch).to(dev)
        params += list(init.parameters())
    opt = torch.optim.Adam(params, lr=a.lr)
    bce = nn.BCEWithLogitsLoss()

    buf = None
    if a.persistent and a.variant == 'baseline':
        buf = torch.randn(a.buf_size, 1, a.height, a.width, device=dev) * 2.0

    def eval_val():
        Js, n = 0.0, 0
        with torch.no_grad():
            for xv, yv in val_dl:
                xv, yv = xv.to(dev), yv.to(dev)
                feat = enc(slice_modality(xv, a.modality))
                if a.variant == 'feedforward':
                    pred = torch.sigmoid(init(feat))
                elif a.variant == 'amortized':
                    pred = torch.sigmoid(langevin_refine(
                        ene, feat, init(feat), eval_steps, a.step_size, a.noise))
                else:
                    l0 = torch.randn(feat.size(0), 1, feat.size(2), feat.size(3),
                                     device=dev) * 2.0
                    pred = torch.sigmoid(langevin_refine(
                        ene, feat, l0, eval_steps, a.step_size, a.noise))
                Js += float(iou(pred, yv).sum()); n += xv.size(0)
        return Js / max(n, 1)

    it, t0, eval_t, log = 0, time.time(), 0.0, []
    while it < a.iters:
        for x, y in dl:
            if it >= a.iters:
                break
            if val_dl is not None and it > 0 and it % a.eval_every == 0:
                te = time.time(); vj = eval_val(); eval_t += time.time() - te
                rec = dict(it=it, val_J=round(vj, 4),
                           t=round(time.time() - t0 - eval_t, 1))
                log.append(rec); print(rec, flush=True)
            x, y = x.to(dev), y.to(dev)
            feat = enc(slice_modality(x, a.modality))

            if a.variant == 'feedforward':
                logits = init(feat)
                loss = bce(logits, y)
                opt.zero_grad(); loss.backward(); opt.step()
                if it % 200 == 0:
                    rec = dict(it=it, loss=round(loss.item(), 3),
                               pred_iou=round(float(iou(torch.sigmoid(logits).detach(), y).mean()), 3),
                               t=round(time.time() - t0 - eval_t, 1))
                    log.append(rec); print(rec, flush=True)
                it += 1
                continue

            # --- EBM variants (baseline / amortized) ---
            pos_e = ene(feat, y)
            idx = None
            if a.variant == 'baseline':
                if buf is not None:
                    idx = torch.randint(0, a.buf_size, (x.size(0),), device=dev)
                    l0 = buf[idx].clone()
                    reinit = torch.rand(x.size(0), device=dev) < 0.05
                    l0[reinit] = torch.randn_like(l0[reinit]) * 2.0
                else:
                    l0 = torch.randn_like(y) * 2.0
            else:
                init_logits = init(feat)   # amortized initializer co-trains the encoder
                l0 = init_logits
            neg_logits = langevin_refine(ene, feat.detach(), l0, steps, a.step_size, a.noise)
            if buf is not None and idx is not None:
                buf[idx] = neg_logits.detach()
            neg_mask = torch.sigmoid(neg_logits)
            neg_e = ene(feat, neg_mask)
            hard = shift_neg(y).to(dev)
            hard_e = ene(feat, hard)
            ebm = (pos_e.mean() - neg_e.mean()
                   + torch.relu(1.0 + pos_e - hard_e).mean()
                   + 0.05 * (pos_e ** 2 + neg_e ** 2).mean())
            if a.variant == 'amortized':
                # Data-ground the amortized initializer: BCE(init, GT) is the primary
                # driver so the initializer localizes like the feed-forward model,
                # while the energy is a down-weighted secondary term that shapes a
                # landscape for K-step refinement at inference. The original
                # cooperative target (regress the model's own refined samples) has no
                # data anchor and bootstrap-collapses to background on real video.
                loss = bce(init_logits, y) + a.ebm_weight * ebm
            else:
                loss = ebm
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 200 == 0:
                rec = dict(it=it, loss=round(loss.item(), 3),
                           neg_iou=round(float(iou(neg_mask.detach(), y).mean()), 3),
                           t=round(time.time() - t0 - eval_t, 1))
                log.append(rec); print(rec, flush=True)
            it += 1

    if val_dl is not None:
        vj = eval_val()
        log.append(dict(it=a.iters, val_J=round(vj, 4),
                        t=round(time.time() - t0 - eval_t, 1), final=True))
        print('final val_J', round(vj, 4), flush=True)
    train_s = round(time.time() - t0 - eval_t, 1)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.tag or a.variant
    ck = dict(enc=enc.state_dict(), variant=a.variant, args=vars(a),
              train_s=train_s, iters=a.iters)
    if ene is not None:
        ck['ene'] = ene.state_dict()
    if init is not None:
        ck['init'] = init.state_dict()
    torch.save(ck, out / f'{name}.pt')
    (out / f'{name}_trainlog.json').write_text(json.dumps(log, indent=1))
    print('saved', out / f'{name}.pt', 'train_s', train_s, flush=True)


if __name__ == '__main__':
    main()
