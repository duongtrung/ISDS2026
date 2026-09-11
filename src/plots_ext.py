"""Extension figures from logged JSON:
  paper/figures/transfer_bars.png    -- cross-dataset region J: long-run baseline vs ours
                                        (K=5) vs the trivial flow threshold.
  paper/figures/capacity_frontier.png -- K-refinement J&F-vs-latency frontier at ch=32 vs 64."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIG = Path('paper/figures'); FIG.mkdir(parents=True, exist_ok=True)
VID = json.load(open('results/video_results.json'))
EXT = json.load(open('results/ext_results.json'))
FLOW = {r['dataset']: r for r in json.load(open('results/flow_baseline.json'))}


def g(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


DS = [('davis', 'DAVIS-16'), ('fbms', 'FBMS-59'), ('segtrack', 'SegTrack-v2')]

# --- cross-dataset region-J bars ---
methods = [('Long-run Langevin (baseline)', 'baseline', 100, '#185FA5'),
           ('Ours ($K{=}5$)', 'amortized', 5, '#534AB7'),
           ('Flow threshold (trivial)', 'flow', 0, '#C6462F')]
fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=150)
x = np.arange(len(DS)); w = 0.27
for i, (lbl, var, k, col) in enumerate(methods):
    vals = [FLOW[d]['J'] if var == 'flow'
            else (g(VID, dataset=d, variant=var, steps=k) or {'J': 0})['J'] for d, _ in DS]
    ax.bar(x + (i - 1) * w, vals, w, label=lbl.replace('$K{=}5$', 'K=5'), color=col)
    for j, v in enumerate(vals):
        ax.text(x[j] + (i - 1) * w, v + 0.006, f"{v:.2f}", ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in DS])
ax.set_ylabel('Region J (IoU)')
ax.set_title('Cross-dataset region J (DAVIS-16 = train; FBMS/SegTrack zero-shot)')
ax.legend(fontsize=8, loc='upper left'); ax.grid(axis='y', alpha=0.3)
plt.savefig(FIG / 'transfer_bars.png', bbox_inches='tight'); plt.close()
print('wrote transfer_bars.png')

# --- capacity frontier (ch=32 vs 64 amortized K-sweep, DAVIS) ---
fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=150)
for ch, src, col, mk in [(32, VID, '#534AB7', 's'), (64, EXT, '#2E8B57', 'o')]:
    pts = sorted([r for r in src if r.get('ch', 32) == ch and r['variant'] == 'amortized'
                  and r.get('dataset', 'davis') == 'davis'], key=lambda r: r['ms_per_frame'])
    if not pts:
        continue
    ax.plot([p['ms_per_frame'] for p in pts], [p['JF'] for p in pts],
            marker=mk, color=col, label=f'ch={ch} ($K$ sweep)')
    for p in pts:
        ax.annotate(str(p['steps']), (p['ms_per_frame'], p['JF']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)
ax.set_xscale('log')
ax.set_xlabel('Inference time per frame (ms, log scale)')
ax.set_ylabel('J&F on DAVIS-16 val')
ax.set_title('Backbone capacity: refinement frontier (ch=32 vs 64)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.savefig(FIG / 'capacity_frontier.png', bbox_inches='tight'); plt.close()
print('wrote capacity_frontier.png')
