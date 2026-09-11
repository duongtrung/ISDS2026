#!/bin/bash
# RQ1 (speed at matched quality): train all three variants on DAVIS-16 and record
# training wall-clock + val J (each checkpoint's *_trainlog.json logs val_J-vs-t)
# and per-frame inference latency. The operating-point quality table is produced
# by run_rq2.sh (DAVIS-val eval) and src/gen_tables_video.py.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DAVIS=${DAVIS:-data/DAVIS}
export PYTHONUNBUFFERED=1

# 1. Optical flow (once; skip if already present)
[ -d "$DAVIS/Flow" ] || $PY src/precompute_flow.py \
    --frames-root "$DAVIS/JPEGImages/480p" --out "$DAVIS/Flow" --height 256 --width 448

# 2. Train feedforward (supervised), amortized (EBM + K-step), baseline (long-run
#    EBM, persistent-CD). One at a time so per-variant wall-clock is un-contended.
bash scripts/run_all_train.sh
