#!/bin/bash
# Sequential training of all three variants on DAVIS-16 (one at a time so the
# per-variant wall-clock used for RQ1 is measured on an un-contended GPU).
# LR 1e-3 for all (the video default 2e-4 undertrains the tiny 32-ch model).
# Baseline uses persistent-CD (Rule 3 "persistent chains"); the long-run sampler
# (step-size/noise/N) is tuned at INFERENCE in scripts/run_rq2.sh, which is what
# "give the long-run sampler a real tuning pass" means for an inference-time object.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DAVIS=data/DAVIS
export PYTHONUNBUFFERED=1
mkdir -p runs_video logs

echo "===== [1/3] feedforward (supervised, no energy) ====="
$PY src/train_video.py --variant feedforward --davis-root $DAVIS --iters 8000 --lr 1e-3 \
  --eval-every 500 --val-n 128 --out runs_video --seed 0 2>&1 | tee logs/train_feedforward.log

echo "===== [2/3] amortized = energy critic on the frozen feed-forward initializer ====="
# amortized(K=0) == feed-forward; K Langevin steps under the learned energy refine it.
$PY src/train_energy.py --davis-root $DAVIS --init-ckpt runs_video/feedforward.pt --iters 3000 \
  --neg-steps 5 --step-size 5 --noise 0.02 --lr 1e-3 --eval-every 500 --val-n 128 \
  --out runs_video --seed 0 2>&1 | tee logs/train_energy.log

echo "===== [3/3] baseline (persistent-CD EBM, N=15 train steps) ====="
$PY src/train_video.py --variant baseline --davis-root $DAVIS --iters 5000 --lr 1e-3 --persistent \
  --neg-steps 15 --step-size 5 --noise 0.02 --eval-every 500 --val-n 128 \
  --out runs_video --seed 0 2>&1 | tee logs/train_baseline.log

echo "ALL TRAINING DONE"
