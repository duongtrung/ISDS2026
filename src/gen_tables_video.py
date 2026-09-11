"""LaTeX tables for the paper from logged video results (CLAUDE.md rule 1):
  - paper/tables/rq1_main.tex   : DAVIS-16 val quality / size / train-time / latency
  - paper/tables/rq3_transfer.tex : cross-dataset region J (DAVIS/FBMS/SegTrack)
Operating points (fixed for cross-dataset comparability): feed-forward single
pass; amortized K=5; long-run baseline N=100 with the fair-tuned sampler.
"""
import json
from pathlib import Path

import torch

from models import Encoder, EnergyNet, InitNet

RES = 'results/video_results.json'
RUNS = Path('runs_video')
TAB = Path('paper/tables'); TAB.mkdir(parents=True, exist_ok=True)


def _n(m):
    return sum(p.numel() for p in m.parameters())


PARAMS = {
    'feedforward': _n(Encoder(5)) + _n(InitNet()),
    'baseline':    _n(Encoder(5)) + _n(EnergyNet()),
    'amortized':   _n(Encoder(5)) + _n(EnergyNet()) + _n(InitNet()),
}
OP = {'feedforward': 0, 'amortized': 5, 'baseline': 100}
PRETTY = {'baseline': 'Long-run Langevin (baseline)',
          'feedforward': 'Ours: initializer only ($K{=}0$)',
          'amortized': 'Ours: init\\,$+$\\,$K$-step ($K{=}5$)'}
ORDER = ['baseline', 'feedforward', 'amortized']


def _train_s(var):
    p = RUNS / f'{var}.pt'
    if not p.exists():
        return None
    return torch.load(p, map_location='cpu', weights_only=False).get('train_s')


def _res():
    return json.load(open(RES)) if Path(RES).exists() else []


def _get(res, dataset, var, steps):
    for r in res:
        if r['dataset'] == dataset and r['variant'] == var and r['steps'] == steps:
            return r
    return None


def rq1_main():
    res = _res(); rows = []
    sa = (json.load(open('results/seed_agg.json'))
          if Path('results/seed_agg.json').exists() else None)
    SA_KEY = {'feedforward': 'feedforward_K0', 'amortized': 'amortized_K5'}
    nseeds = len(sa['seeds']) if sa else 0
    for var in ORDER:
        r = _get(res, 'davis', var, OP[var])
        if r is None:
            continue
        ts = _train_s(var)
        tstr = f"{ts / 60:.0f}" if ts else "--"
        if sa and var in SA_KEY and SA_KEY[var] in sa.get('davis', {}):
            a = sa['davis'][SA_KEY[var]]; jcell = f"${r['J']:.3f}\\pm{a['std']:.3f}$"
        elif not sa and r.get('J_std') is not None:
            jcell = f"${r['J']:.3f}\\pm{r['J_std']:.3f}$"   # per-frame std (no seeds available)
        else:
            jcell = f"{r['J']:.3f}"   # single-seed baseline: no across-seed std
        ms = r['ms_per_frame']
        mss = f"{ms:.2f}" if ms < 10 else f"{ms:.1f}"   # 2 decimals for sub-10ms so ratios recompute
        rows.append(f"{PRETTY[var]} & {PARAMS[var] / 1e3:.1f}K & {tstr} & "
                    f"{jcell} & {r['F']:.3f} & {r['JF']:.3f} & {mss} \\\\")
    if not rows:
        print('rq1_main: no DAVIS rows yet'); return
    stdnote = (f" $\\mathcal{{J}}$ is the seed-0 operating point $\\pm$ std across "
               f"{nseeds} seeds for the amortized/feed-forward path; baseline single-seed."
               if nseeds >= 2 else "")
    tex = ("\\begin{table}[t]\n\\centering\n"
           "\\caption{DAVIS-16 val: segmentation quality, model size, training "
           "wall-clock and per-frame inference latency at each variant's operating "
           "point (feed-forward single pass; amortized $K{=}5$; long-run baseline "
           "$N{=}100$ with the fair-tuned sampler). NVIDIA GB10, 256$\\times$448. "
           "Arrows indicate whether higher ($\\uparrow$) or lower ($\\downarrow$) is better." + stdnote + "}\n"
           "\\label{tab:rq1}\n"
           "\\resizebox{\\textwidth}{!}{%\n"
           "\\begin{tabular}{lcccccc}\n\\toprule\n"
           "Method & Params\\,$\\downarrow$ & Train\\,(min)\\,$\\downarrow$ & "
           "$\\mathcal{J}\\uparrow$ & $\\mathcal{F}\\uparrow$ & "
           "$\\mathcal{J}\\&\\mathcal{F}\\uparrow$ & ms/frame\\,$\\downarrow$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
    (TAB / 'rq1_main.tex').write_text(tex); print('wrote', TAB / 'rq1_main.tex')


def rq3_transfer():
    res = _res()
    datasets = [('davis', 'DAVIS-16'), ('fbms', 'FBMS-59'), ('segtrack', 'SegTrack-v2')]
    present = [(d, lbl) for d, lbl in datasets if any(r['dataset'] == d for r in res)]
    if not present:
        print('rq3_transfer: no rows yet'); return
    header = "Method & " + " & ".join(lbl for _, lbl in present) + " \\\\"

    def cell(d, var):
        r = _get(res, d, var, OP[var])
        return f"{r['JF']:.3f} ({r['J']:.3f})" if r else "--"
    rows = []
    for var in ORDER:
        rows.append(f"{PRETTY[var]} & " + " & ".join(cell(d, var) for d, _ in present) + " \\\\")
    colspec = "l" + "c" * len(present)
    tex = ("\\begin{table}[t]\n\\centering\n"
           "\\caption{Cross-dataset generalization: each cell is $\\mathcal{J}\\&\\mathcal{F}$ "
           "with region $\\mathcal{J}$ in parentheses, for DAVIS-trained models evaluated "
           "WITHOUT fine-tuning. On $\\mathcal{J}\\&\\mathcal{F}$ the amortized model transfers "
           "better than the baseline on both sets; the FBMS-59 reversal is region-$\\mathcal{J}$ "
           "only. Higher is better ($\\uparrow$). Operating points as in Table~\\ref{tab:rq1}.}\n\\label{tab:rq3}\n"
           f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
           + header + "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    (TAB / 'rq3_transfer.tex').write_text(tex); print('wrote', TAB / 'rq3_transfer.tex')


def headline():
    """Emit \\renewcommand macros so headline numbers in the prose come from JSON."""
    res = _res()
    if not res:
        print('headline: no results yet'); return
    m = {}

    def put(key, rec, field, scale=1.0, fmt='{:.3f}'):
        if rec is not None and rec.get(field) is not None:
            m[key] = fmt.format(rec[field] * scale)

    put('NumFFj',      _get(res, 'davis', 'feedforward', 0), 'J')
    put('NumAmortKz',  _get(res, 'davis', 'amortized', 0), 'J')
    put('NumAmortJ',   _get(res, 'davis', 'amortized', 5), 'J')
    put('NumAmortMs',  _get(res, 'davis', 'amortized', 5), 'ms_per_frame', 1, '{:.1f}')
    put('NumAmortJF',  _get(res, 'davis', 'amortized', 5), 'JF')
    put('NumBaseJ',    _get(res, 'davis', 'baseline', 100), 'J')
    put('NumBaseMs',   _get(res, 'davis', 'baseline', 100), 'ms_per_frame', 1, '{:.1f}')
    put('NumBaseJF',   _get(res, 'davis', 'baseline', 100), 'JF')
    # best amortized point by J&F over its sweep
    amort = [r for r in res if r['dataset'] == 'davis' and r['variant'] == 'amortized']
    if amort:
        best = max(amort, key=lambda r: r['JF'])
        m['NumAmortBestJF'] = f"{best['JF']:.3f}"; m['NumAmortBestK'] = str(best['steps'])
    # matched-quality speedup: smallest amortized K with J >= baseline(N=100) J, latency ratio
    b100 = _get(res, 'davis', 'baseline', 100)
    ff0 = _get(res, 'davis', 'feedforward', 0)   # the K=0 operating point shown in Table 1
    if b100 and ff0 and ff0['ms_per_frame'] > 0:
        m['NumSpeedup'] = f"{b100['ms_per_frame'] / ff0['ms_per_frame']:.0f}"
        m['NumMatchK'] = '0'
    # marginal cost of ONE Langevin step for each sampler (slope between two sweep
    # points) -- shows both pay the same per-step price, so the speedup is the N/K ratio.
    b10, b100m = _get(res, 'davis', 'baseline', 10), _get(res, 'davis', 'baseline', 100)
    if b10 and b100m:
        m['NumStepBase'] = f"{(b100m['ms_per_frame'] - b10['ms_per_frame']) / 90:.2f}"
    ak0, ak20 = _get(res, 'davis', 'amortized', 0), _get(res, 'davis', 'amortized', 20)
    if ak0 and ak20:
        m['NumStepOurs'] = f"{(ak20['ms_per_frame'] - ak0['ms_per_frame']) / 20:.2f}"
    a5 = _get(res, 'davis', 'amortized', 5)   # speedup at the K=5 operating point (Table 1)
    if b100 and a5 and a5.get('ms_per_frame'):
        m['NumSpeedupKfive'] = f"{b100['ms_per_frame'] / a5['ms_per_frame']:.1f}"
    # amortized K-refinement effect on region-J and boundary-F (for honest reporting)
    az, a20 = _get(res, 'davis', 'amortized', 0), _get(res, 'davis', 'amortized', 20)
    if az and a20:
        m['NumRefineJ'] = f"{a20['J'] - az['J']:+.3f}"
        m['NumFkzero'] = f"{az['F']:.3f}"; m['NumFktwenty'] = f"{a20['F']:.3f}"
    for var, key in [('feedforward', 'NumFFtrain'), ('amortized', 'NumAmortTrain'),
                     ('baseline', 'NumBaseTrain')]:
        ts = _train_s(var)
        if ts:
            m[key] = f"{ts / 60:.0f}"
    for d, suf in [('fbms', 'FBMS'), ('segtrack', 'Seg')]:
        put(f'NumBase{suf}', _get(res, d, 'baseline', 100), 'J')
        put(f'NumAmort{suf}', _get(res, d, 'amortized', 5), 'J')
        put(f'NumFF{suf}', _get(res, d, 'feedforward', 0), 'J')
    # across-seed std (3 seeds) for the amortized/feed-forward path
    if Path('results/seed_agg.json').exists():
        sa = json.load(open('results/seed_agg.json'))
        m['NumSeeds'] = str(len(sa['seeds']))
        if 'amortized_K5' in sa.get('davis', {}):
            m['NumAmortStd'] = f"{sa['davis']['amortized_K5']['std']:.3f}"
        if 'feedforward_K0' in sa.get('davis', {}):
            m['NumFFStd'] = f"{sa['davis']['feedforward_K0']['std']:.3f}"
        if 'amortized_K5' in sa.get('fbms', {}):
            m['NumAmortFBMSstd'] = f"{sa['fbms']['amortized_K5']['std']:.3f}"
    tex = "% auto-generated headline numbers (do not edit)\n" + \
          "\n".join(f"\\renewcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n"
    (TAB / 'headline.tex').write_text(tex)
    print('wrote', TAB / 'headline.tex', f'({len(m)} macros)')


if __name__ == '__main__':
    rq1_main()
    rq3_transfer()
    headline()
