"""Generate LaTeX table for the paper from results/synthetic_results.json."""
import json
from pathlib import Path

res = json.load(open('results/synthetic_results.json'))
rows = []
for r in res:
    nm = 'Long-run Langevin (orig.)' if r['model'].startswith('baseline') \
        else 'Amortized + $K$-step (ours)'
    rows.append(f"{nm} & {r['steps']} & {r['J']:.3f} & {r['F']:.3f} & "
                f"{r['JF']:.3f} & {r['ms_per_frame']:.1f} \\\\")
tex = ("\\begin{table}[t]\n\\centering\n"
       "\\caption{Synthetic smoke-test results (CPU). Sanity check only; "
       "not reportable benchmark numbers.}\n\\label{tab:smoke}\n"
       "\\begin{tabular}{lccccc}\n\\hline\n"
       "Sampler & Steps & $\\mathcal{J}$ & $\\mathcal{F}$ & "
       "$\\mathcal{J}\\&\\mathcal{F}$ & ms/frame \\\\\n\\hline\n"
       + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n\\end{table}\n")
Path('paper/tables').mkdir(parents=True, exist_ok=True)
Path('paper/tables/synthetic_smoke.tex').write_text(tex)
print('wrote paper/tables/synthetic_smoke.tex')
