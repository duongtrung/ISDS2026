"""Speed-accuracy frontier plot from results/synthetic_results.json."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res = json.load(open('results/synthetic_results.json'))
fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
for name, label, marker, color in [
    ('baseline-longrun', 'Original: long-run Langevin (N steps)', 'o', '#185FA5'),
    ('amortized-Kstep', 'Proposed: amortized init + K-step refine', 's', '#534AB7'),
]:
    pts = sorted([r for r in res if r['model'] == name],
                 key=lambda r: r['ms_per_frame'])
    ax.plot([p['ms_per_frame'] for p in pts], [p['J'] for p in pts],
            marker=marker, label=label, color=color)
    for p in pts:
        ax.annotate(str(p['steps']), (p['ms_per_frame'], p['J']),
                    textcoords='offset points', xytext=(5, 4), fontsize=8)
ax.set_xscale('log')
ax.set_xlabel('Inference time per frame (ms, log scale)')
ax.set_ylabel('Region similarity J (IoU)')
ax.set_title('Speed-accuracy frontier - synthetic smoke test (CPU)')
ax.grid(alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
plt.savefig('results/frontier_synthetic.png', bbox_inches='tight')
print('wrote results/frontier_synthetic.png')
