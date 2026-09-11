#!/bin/bash
# Post-tuning eval: assumes results/sampler_tuning.json already exists (from
# src/tune_sampler.py). Runs the RQ2 DAVIS frontier + RQ3 transfer, regenerates all
# assets, and compiles the paper. Direct commands (no re-tuning).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1
# tectonic may live outside the default PATH (e.g. a conda base env); add it if needed.
command -v tectonic >/dev/null || export PATH="$HOME/anaconda3/bin:$PATH"
DAVIS=data/DAVIS; FBMS=data/FBMS; SEG=data/SegTrackv2

$PY -c "import json;b=json.load(open('results/sampler_tuning.json'))['best'];open('results/best_sampler.txt','w').write(f\"{b['step_size']} {b['noise']}\n\")"
read STEP NOISE < results/best_sampler.txt || true
echo "tuned baseline sampler: step=$STEP noise=$NOISE"

rm -f results/video_results.json

echo "### RQ2 DAVIS frontier"
$PY src/eval_video.py --ckpt runs_video/baseline.pt    --dataset davis --root "$DAVIS" --steps 10 25 50 100 200 400 --step-size "$STEP" --noise "$NOISE" --seed 0
$PY src/eval_video.py --ckpt runs_video/amortized.pt   --dataset davis --root "$DAVIS" --steps 0 1 2 3 5 8 10 20 --seed 0
$PY src/eval_video.py --ckpt runs_video/feedforward.pt --dataset davis --root "$DAVIS" --steps 0 --seed 0

echo "### RQ3 transfer (FBMS-59, SegTrack-v2)"
for pair in "fbms $FBMS" "segtrack $SEG"; do
  set -- $pair; DS=$1; ROOT=$2
  $PY src/eval_video.py --ckpt runs_video/baseline.pt    --dataset "$DS" --root "$ROOT" --steps 100 --step-size "$STEP" --noise "$NOISE" --seed 0
  $PY src/eval_video.py --ckpt runs_video/amortized.pt   --dataset "$DS" --root "$ROOT" --steps 5 --seed 0
  $PY src/eval_video.py --ckpt runs_video/feedforward.pt --dataset "$DS" --root "$ROOT" --steps 0 --seed 0
done

echo "### assets + paper"
$PY src/plots_video.py
$PY src/gen_tables_video.py
$PY src/qualitative.py
( cd paper && tectonic main.tex --keep-logs && echo "PAPER BUILT" ) || echo "latex compile issue (assets generated)"
echo "EVAL_DIRECT DONE"
