"""Qualitative mask panels for the paper:
input | GT | feed-forward | baseline (N=100) | amortized (K=5)
for a few DAVIS-16 val frames -> paper/figures/qualitative.png.
Uses the fair-tuned baseline sampler config if results/best_sampler.txt exists."""
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from davis import Davis16
from models import Encoder, EnergyNet, InitNet
from sampler import langevin_refine

STEPS = {'feedforward': 0, 'baseline': 100, 'amortized': 5}
TITLE = {'feedforward': 'feed-forward', 'baseline': 'baseline N=100',
         'amortized': 'amortized K=5'}


def load(p, dev):
    ck = torch.load(p, map_location=dev, weights_only=False)
    enc = Encoder(5).to(dev); enc.load_state_dict(ck['enc']); enc.eval()
    ene = None
    if 'ene' in ck:
        ene = EnergyNet().to(dev); ene.load_state_dict(ck['ene']); ene.eval()
    init = None
    if 'init' in ck:
        init = InitNet().to(dev); init.load_state_dict(ck['init']); init.eval()
    return ck['variant'], enc, ene, init


def predict(variant, enc, ene, init, x, steps, step_size, noise):
    feat = enc(x)
    if variant == 'feedforward':
        return torch.sigmoid(init(feat))
    l0 = init(feat) if init is not None else torch.randn(
        x.size(0), 1, x.size(2), x.size(3), device=x.device) * 2.0
    return torch.sigmoid(langevin_refine(ene, feat, l0, steps, step_size, noise))


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    ds = Davis16('data/DAVIS', 'val', (256, 448))
    idx = [int(f * len(ds)) for f in (0.08, 0.3, 0.55, 0.82)]
    step_size, noise = 5.0, 0.02
    bs = Path('results/best_sampler.txt')
    if bs.exists():
        s = bs.read_text().split(); step_size, noise = float(s[0]), float(s[1])
    models = {}
    for name in ['feedforward', 'baseline', 'amortized']:
        fn = Path(f'runs_video/{name}.pt')
        if fn.exists():
            models[name] = load(fn, dev)
    have = [n for n in ['feedforward', 'baseline', 'amortized'] if n in models]
    cols = ['input', 'GT'] + have
    fig, axes = plt.subplots(len(idx), len(cols),
                             figsize=(2.1 * len(cols), 2.1 * len(idx)), dpi=140)
    for r, i in enumerate(idx):
        x, y = ds[i]; xin = x.unsqueeze(0).to(dev)
        rgb = np.clip(x[:3].permute(1, 2, 0).numpy(), 0, 1)
        axes[r, 0].imshow(rgb)
        axes[r, 1].imshow(rgb); axes[r, 1].imshow(y[0].numpy(), alpha=0.45, cmap='Reds')
        for c, name in enumerate(have, start=2):
            variant, enc, ene, init = models[name]
            with torch.no_grad():
                pr = predict(variant, enc, ene, init, xin, STEPS[name], step_size, noise)
            axes[r, c].imshow(rgb)
            axes[r, c].imshow((pr[0, 0].cpu().numpy() > 0.5), alpha=0.45, cmap='Reds')
        for c in range(len(cols)):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for c, name in enumerate(cols):
        axes[0, c].set_title(TITLE.get(name, name), fontsize=9)
    plt.tight_layout()
    out = Path('paper/figures/qualitative.png')
    plt.savefig(out, bbox_inches='tight'); print('wrote', out)


if __name__ == '__main__':
    main()
