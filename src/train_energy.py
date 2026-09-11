"""Train the energy critic on top of the FROZEN feed-forward initializer, yielding
the amortized model.

Rationale: jointly training the energy CD objective through the shared encoder pulls
the encoder away from localization and degrades the initializer below the pure
feed-forward segmenter. We instead decompose the amortized EBM cleanly: the
data-grounded initializer (the feed-forward encoder+InitNet) amortizes the posterior,
and a separate energy critic is fit on its (frozen) features. Consequences:
  * amortized at K=0 is EXACTLY the feed-forward model (no regression from the energy);
  * K Langevin steps under the learned energy refine that proposal at inference;
  * the feed-forward ablation is precisely amortized(K=0).
Only the small EnergyNet is optimized (encoder+init frozen), so this is cheap. The
reported amortized training wall-clock adds this to the feed-forward's.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from davis import Davis16
from models import Encoder, EnergyNet, InitNet, slice_modality, MODALITY_CH
from sampler import langevin_refine
from metrics import iou
from train import shift_neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--davis-root', default='data/DAVIS')
    ap.add_argument('--init-ckpt', default='runs_video/feedforward.pt')
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=448)
    ap.add_argument('--neg-steps', type=int, default=5)
    ap.add_argument('--step-size', type=float, default=5.0)
    ap.add_argument('--noise', type=float, default=0.02)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=500)
    ap.add_argument('--val-n', type=int, default=128)
    ap.add_argument('--out', default='runs_video')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
    ch = ck['args'].get('ch', 32); modality = ck['args'].get('modality', 'both')
    enc = Encoder(MODALITY_CH[modality], ch).to(dev); enc.load_state_dict(ck['enc']); enc.eval()
    init = InitNet(feat_ch=ch, ch=ch).to(dev); init.load_state_dict(ck['init']); init.eval()
    for p in list(enc.parameters()) + list(init.parameters()):
        p.requires_grad_(False)
    ene = EnergyNet(feat_ch=ch, ch=ch).to(dev)
    opt = torch.optim.Adam(ene.parameters(), lr=a.lr)

    ds = Davis16(a.davis_root, 'train', (a.height, a.width))
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=4,
                    pin_memory=(dev == 'cuda'), drop_last=True)
    vds = Davis16(a.davis_root, 'val', (a.height, a.width))
    g = torch.Generator().manual_seed(a.seed)
    vidx = torch.randperm(len(vds), generator=g)[:a.val_n].tolist()
    vdl = DataLoader(Subset(vds, vidx), batch_size=8, num_workers=2)

    def eval_val(K):
        Js, n = 0.0, 0
        with torch.no_grad():
            for xv, yv in vdl:
                xv, yv = xv.to(dev), yv.to(dev)
                feat = enc(slice_modality(xv, modality)); l0 = init(feat)
                p = torch.sigmoid(langevin_refine(ene, feat, l0, K, a.step_size, a.noise))
                Js += float(iou(p, yv).sum()); n += xv.size(0)
        return Js / max(n, 1)

    it, t0, eval_t, log = 0, time.time(), 0.0, []
    while it < a.iters:
        for x, y in dl:
            if it >= a.iters:
                break
            if a.eval_every > 0 and it > 0 and it % a.eval_every == 0:
                te = time.time()
                rec = dict(it=it, val_J_K0=round(eval_val(0), 4),
                           val_J_K5=round(eval_val(a.neg_steps), 4))
                eval_t += time.time() - te
                rec['t'] = round(time.time() - t0 - eval_t, 1)
                log.append(rec); print(rec, flush=True)
            x, y = x.to(dev), y.to(dev)
            with torch.no_grad():
                feat = enc(slice_modality(x, modality)); l0 = init(feat)
            neg = langevin_refine(ene, feat, l0, a.neg_steps, a.step_size, a.noise)
            neg_mask = torch.sigmoid(neg)
            pos_e = ene(feat, y); neg_e = ene(feat, neg_mask)
            hard_e = ene(feat, shift_neg(y).to(dev))
            loss = (pos_e.mean() - neg_e.mean()
                    + torch.relu(1.0 + pos_e - hard_e).mean()
                    + 0.05 * (pos_e ** 2 + neg_e ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
            it += 1

    rec = dict(it=a.iters, val_J_K0=round(eval_val(0), 4),
               val_J_K5=round(eval_val(a.neg_steps), 4), final=True)
    energy_s = round(time.time() - t0 - eval_t, 1)
    rec['t'] = energy_s
    log.append(rec)
    out = Path(a.out)
    ck2 = dict(enc=enc.state_dict(), init=init.state_dict(), ene=ene.state_dict(),
               variant='amortized', args=vars(a),
               train_s=round(energy_s + ck.get('train_s', 0.0), 1),
               energy_train_s=energy_s, init_from=a.init_ckpt, iters=a.iters)
    ck2['args'].update(ch=ch, modality=modality)
    torch.save(ck2, out / 'amortized.pt')
    (out / 'amortized_trainlog.json').write_text(json.dumps(log, indent=1))
    print('saved amortized.pt (energy critic on frozen feed-forward) energy_s', energy_s,
          'total train_s', ck2['train_s'], flush=True)


if __name__ == '__main__':
    main()
