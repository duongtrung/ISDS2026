#!/bin/bash
# RQ2 (speed-accuracy frontier): fair-tune the long-run sampler, then run a dense
# N/K sweep on DAVIS-16 val for all three variants -> results/video_results.json.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DAVIS=${DAVIS:-data/DAVIS}
export PYTHONUNBUFFERED=1

# Fresh DAVIS results (RQ3 appends FBMS/SegTrack later).
rm -f results/video_results.json

# 1. Rule-3 tuning pass of the ORIGINAL long-run sampler (step-size x noise at N=100).
$PY src/tune_sampler.py --ckpt runs_video/baseline.pt --root "$DAVIS" \
    --n 100 --val-n 128 --seed 0 --out results/sampler_tuning.json
$PY -c "import json;b=json.load(open('results/sampler_tuning.json'))['best'];open('results/best_sampler.txt','w').write(f\"{b['step_size']} {b['noise']}\n\")"
read STEP NOISE < results/best_sampler.txt || true
echo "tuned baseline sampler: step_size=$STEP noise=$NOISE"

# 2. Dense frontier sweep on DAVIS val.
$PY src/eval_video.py --ckpt runs_video/baseline.pt   --dataset davis --root "$DAVIS" \
    --steps 10 25 50 100 200 400 --step-size "$STEP" --noise "$NOISE" --seed 0
$PY src/eval_video.py --ckpt runs_video/amortized.pt  --dataset davis --root "$DAVIS" \
    --steps 0 1 2 3 5 8 10 20 --seed 0
$PY src/eval_video.py --ckpt runs_video/feedforward.pt --dataset davis --root "$DAVIS" \
    --steps 0 --seed 0

# 3. Figures/tables from JSON (never hand-edited).
$PY src/plots_video.py
$PY src/gen_tables_video.py
