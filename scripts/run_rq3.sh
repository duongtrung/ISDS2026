#!/bin/bash
# RQ3 (cross-dataset generalization): DAVIS-trained checkpoints evaluated WITHOUT
# fine-tuning on FBMS-59 test and SegTrack-v2. Appends to results/video_results.json.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FBMS=${FBMS:-data/FBMS}
SEGTRACK=${SEGTRACK:-data/SegTrackv2}
export PYTHONUNBUFFERED=1

# Optical flow (once; skip if present)
[ -d "$FBMS/Flow" ]     || $PY src/precompute_flow.py --frames-root "$FBMS/Testset"    --out "$FBMS/Flow"     --height 256 --width 448
[ -d "$SEGTRACK/Flow" ] || $PY src/precompute_flow.py --frames-root "$SEGTRACK/JPEGImages" --out "$SEGTRACK/Flow" --height 256 --width 448

# Use the same fair-tuned long-run sampler config as RQ2.
read STEP NOISE < results/best_sampler.txt || true
echo "baseline sampler: step_size=$STEP noise=$NOISE"

for DS in fbms segtrack; do
  ROOT=$FBMS; [ "$DS" = segtrack ] && ROOT=$SEGTRACK
  $PY src/eval_video.py --ckpt runs_video/baseline.pt   --dataset "$DS" --root "$ROOT" \
      --steps 100 --step-size "$STEP" --noise "$NOISE" --seed 0
  $PY src/eval_video.py --ckpt runs_video/amortized.pt  --dataset "$DS" --root "$ROOT" \
      --steps 5 --seed 0
  $PY src/eval_video.py --ckpt runs_video/feedforward.pt --dataset "$DS" --root "$ROOT" \
      --steps 0 --seed 0
done

# Regenerate transfer table from JSON.
$PY src/gen_tables_video.py
