"""Generate paper/tables/datasets.tex: dataset statistics used in the paper.
Frame counts are computed from the loaders (each usable frame needs a preceding frame
for optical flow; FBMS-59 ground truth is sparse). Native resolution and download size
are measured constants recorded here for reproducibility."""
from pathlib import Path

from davis import Davis16
from fbms import FBMS59
from segtrackv2 import SegTrackV2

TAB = Path('paper/tables'); TAB.mkdir(parents=True, exist_ok=True)

ntr = len(Davis16('data/DAVIS', 'train', (256, 448)))
nva = len(Davis16('data/DAVIS', 'val', (256, 448)))
nfb = len(FBMS59('data/FBMS', 'Testset', (256, 448)))
nst = len(SegTrackV2('data/SegTrackv2', size=(256, 448)))

rows = [
    ("DAVIS-16", "train\\,/\\,val", "30\\,/\\,20", f"{ntr}\\,/\\,{nva}",
     "854$\\times$480", "binary", "1.9\\,GB"),
    ("FBMS-59", "zero-shot test", "30", f"{nfb}",
     "$\\le$640$\\times$480", "multi-obj.", "0.4\\,GB"),
    ("SegTrack-v2", "zero-shot", "14", f"{nst}",
     "$\\le$640$\\times$360", "multi-obj.", "0.2\\,GB"),
]
body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
tex = ("\\begin{table}[t]\n\\centering\n"
       "\\caption{Datasets. \\emph{Frames} are those actually used: each usable frame "
       "needs a preceding frame for optical flow, and FBMS-59 ground truth is sparse "
       "(annotated frames only). All frames are resized to $256\\times448$; DAVIS-16 is "
       "used for training and validation, FBMS-59 and SegTrack-v2 only for zero-shot "
       "transfer. Optical flow is precomputed with RAFT. Multi-object ground truth is "
       "unioned into a single binary foreground.}\n\\label{tab:data}\n"
       "\\begin{tabular}{lccccll}\n\\toprule\n"
       "Dataset & Role & Seq. & Frames & Native res. & GT & Download \\\\\n\\midrule\n"
       + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
(TAB / 'datasets.tex').write_text(tex)
print('wrote', TAB / 'datasets.tex', '| frames tr/va/fbms/seg =', ntr, nva, nfb, nst)
