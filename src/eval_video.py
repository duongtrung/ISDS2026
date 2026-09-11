"""Evaluate a video checkpoint on DAVIS-16 val / FBMS-59 test / SegTrack-v2
(cross-dataset). Reports mean J (IoU), boundary F, J&F, and per-frame latency
for a step sweep. Results accumulate into a JSON file.

Handles three checkpoint kinds (read from ck['variant']):
  baseline    -> noise-init + Langevin (sweep over N)
  amortized   -> InitNet + K Langevin refinement steps (sweep over K)
  feedforward -> single forward pass, no energy/Langevin (steps ignored)

Reproducibility: --seed fixes the Langevin noise so runs are deterministic
(CLAUDE.md rule 4). --noise/--step-size mirror the training sampler settings.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from davis import Davis16
from fbms import FBMS59
from segtrackv2 import SegTrackV2
from models import Encoder, EnergyNet, InitNet, slice_modality, MODALITY_CH
from sampler import langevin_refine
from metrics import iou, boundary_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', choices=['davis', 'fbms', 'segtrack'], required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--steps', type=int, nargs='+', default=[0, 1, 2, 5, 10])
    ap.add_argument('--step-size', type=float, default=5.0)
    ap.add_argument('--noise', type=float, default=0.02)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=448)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--time-iters', type=int, default=50)
    ap.add_argument('--out', default='results/video_results.json')
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    variant = ck['variant']
    ch = ck['args'].get('ch', 32); modality = ck['args'].get('modality', 'both')

    enc = Encoder(MODALITY_CH[modality], ch).to(dev)
    enc.load_state_dict(ck['enc']); enc.eval()
    ene = None
    if 'ene' in ck:
        ene = EnergyNet(feat_ch=ch, ch=ch).to(dev); ene.load_state_dict(ck['ene']); ene.eval()
    init = None
    if 'init' in ck:
        init = InitNet(feat_ch=ch, ch=ch).to(dev); init.load_state_dict(ck['init']); init.eval()

    if a.dataset == 'davis':
        ds = Davis16(a.root, 'val', (a.height, a.width))
    elif a.dataset == 'fbms':
        ds = FBMS59(a.root, 'Testset', (a.height, a.width))
    else:
        ds = SegTrackV2(a.root, size=(a.height, a.width))
    dl = DataLoader(ds, batch_size=16, num_workers=4)   # GB10 128GB unified mem; better GPU util

    def predict(feat, s):
        """Return a probability mask (B,1,H,W) for the given refinement depth s."""
        if variant == 'feedforward':
            return torch.sigmoid(init(feat))
        if init is not None:  # amortized
            l0 = init(feat)
        else:                 # baseline: noise init
            l0 = torch.randn(feat.size(0), 1, feat.size(2), feat.size(3),
                             device=feat.device) * 2.0
        return torch.sigmoid(langevin_refine(ene, feat, l0, s, a.step_size, a.noise))

    steps_list = [0] if variant == 'feedforward' else a.steps
    results = []
    for s in steps_list:
        torch.manual_seed(a.seed)   # deterministic baseline noise per step (same RNG state for every s)
        Jv, Fv = [], []
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            feat = enc(slice_modality(x, modality))
            pred = predict(feat, s)
            valid = y.flatten(1).sum(1) > 0   # exclude empty-GT frames (uniform across datasets)
            Jv.append(iou(pred, y)[valid].cpu()); Fv.append(boundary_f(pred, y)[valid].cpu())
        Jv, Fv = torch.cat(Jv), torch.cat(Fv); n = Jv.numel()
        Jm, Fm = float(Jv.mean()), float(Fv.mean())
        # latency: single-frame forward, GPU-synced
        torch.manual_seed(a.seed)
        x1 = ds[0][0].unsqueeze(0).to(dev)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.time_iters):
            feat = enc(slice_modality(x1, modality))
            predict(feat, s)
        if dev == 'cuda':
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / a.time_iters * 1000
        rec = dict(dataset=a.dataset, variant=variant, steps=s, modality=modality, ch=ch,
                   J=round(Jm, 4), J_std=round(float(Jv.std()), 4), F=round(Fm, 4),
                   JF=round((Jm + Fm) / 2, 4), ms_per_frame=round(ms, 2), n=n)
        results.append(rec)
        print(rec, flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else []
    Path(a.out).write_text(json.dumps(prev + results, indent=1))


if __name__ == '__main__':
    main()
