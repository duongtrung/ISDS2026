"""Video-result figures for the paper, generated only from logged JSON:
  - paper/figures/frontier_davis.png : J&F-vs-latency frontier on DAVIS-16 val
    (RQ2), long-run baseline (N sweep) vs amortized (K sweep) vs feedforward.
  - paper/figures/train_efficiency.png : val J vs training wall-clock (RQ1),
    read from each variant's *_trainlog.json.
Never hand-edit numbers (CLAUDE.md rule 1)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = 'results/video_results.json'
RUNS = Path('runs_video')
FIG = Path('paper/figures'); FIG.mkdir(parents=True, exist_ok=True)

STYLE = {
    'baseline':    ('Original: long-run Langevin (N steps)', 'o', '#185FA5'),
    'amortized':   ('Proposed: amortized init + K-step refine', 's', '#534AB7'),
    'feedforward': ('Feed-forward (supervised, no energy)', '^', '#C6462F'),
}


def frontier():
    if not Path(RES).exists():
        print('no', RES, '- skip frontier'); return
    res = [r for r in json.load(open(RES)) if r['dataset'] == 'davis']
    if not res:
        print('no davis rows - skip frontier'); return
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=150)
    for var, (label, mk, col) in STYLE.items():
        pts = sorted([r for r in res if r['variant'] == var], key=lambda r: r['ms_per_frame'])
        if not pts:
            continue
        if len(pts) == 1:
            ax.scatter([pts[0]['ms_per_frame']], [pts[0]['JF']], marker=mk, s=70,
                       color=col, label=label, zorder=3)
        else:
            ax.plot([p['ms_per_frame'] for p in pts], [p['JF'] for p in pts],
                    marker=mk, color=col, label=label)
        for p in pts:
            ax.annotate(str(p['steps']), (p['ms_per_frame'], p['JF']),
                        textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax.set_xscale('log')
    ax.set_xlabel('Inference time per frame (ms, log scale)')
    ax.set_ylabel('J&F score on DAVIS-16 val')
    ax.set_title('Speed-accuracy frontier (DAVIS-16 val)')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=8)
    out = FIG / 'frontier_davis.png'
    plt.savefig(out, bbox_inches='tight'); plt.close()
    print('wrote', out)


def _curve(var):
    """(minutes, val_J) points for a variant. feed-forward/baseline log 'val_J';
    amortized is the energy critic on the frozen feed-forward, so its curve is the
    feed-forward curve followed by the energy phase (val_J at K=5), the energy log's
    time offset by the feed-forward's total training time."""
    p = RUNS / f'{var}_trainlog.json'
    if not p.exists():
        return []
    log = json.load(open(p))
    if var != 'amortized':
        return [(r['t'] / 60.0, r['val_J']) for r in log if 'val_J' in r]
    ff = _curve('feedforward')
    ff_total = ff[-1][0] if ff else 0.0
    energy = [((ff_total * 60 + r['t']) / 60.0, r.get('val_J_K5', r.get('val_J_K0')))
              for r in log if ('val_J_K5' in r or 'val_J_K0' in r)]
    return ff + energy


def train_efficiency():
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=150)
    any_curve = False
    for var, (label, mk, col) in STYLE.items():
        pts = _curve(var)
        if not pts:
            continue
        any_curve = True
        ax.plot([t for t, _ in pts], [j for _, j in pts],
                marker=mk, color=col, label=label, markersize=4)
    if not any_curve:
        print('no trainlogs - skip train_efficiency'); plt.close(); return
    ax.set_xlabel('Training wall-clock (minutes, GB10)')
    ax.set_ylabel('DAVIS-16 val J (IoU)')
    ax.set_title('Training efficiency: val J vs wall-clock (RQ1)')
    ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=8)
    out = FIG / 'train_efficiency.png'
    plt.savefig(out, bbox_inches='tight'); plt.close()
    print('wrote', out)


if __name__ == '__main__':
    frontier()
    train_efficiency()
