#!/bin/bash
# Post-training: fair-tune the sampler, run the RQ2 frontier + RQ3 transfer evals,
# regenerate all figures/tables from JSON, render qualitative panels, compile paper.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
# tectonic may live outside the default PATH (e.g. a conda base env); add it if needed.
command -v tectonic >/dev/null || export PATH="$HOME/anaconda3/bin:$PATH"

bash scripts/run_rq2.sh          # sampler tuning + DAVIS frontier sweep + figures/tables
bash scripts/run_rq3.sh          # FBMS-59 + SegTrack-v2 transfer + table
$PY src/qualitative.py           # qualitative mask panels
$PY src/plots_video.py           # (idempotent) frontier + train-efficiency figures
$PY src/gen_tables_video.py      # (idempotent) rq1/rq3 tables + headline macros

( cd paper && tectonic main.tex --keep-logs && echo "paper/main.pdf built" ) \
    || echo "latex compile issue (non-fatal; assets are generated)"
echo "EVAL + ASSETS + PAPER DONE"
