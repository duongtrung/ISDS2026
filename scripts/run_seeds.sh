#!/bin/bash
# Extra seeds for the (cheap) amortized path = feed-forward + energy critic, to give
# across-seed error bars on the DAVIS operating points and transfer numbers (the
# reviewer's top statistical concern). The baseline (2 h/seed) is left single-seed:
# it is a degenerate non-localizing reference (J~0.074, flat across N=10..400), so its
# behaviour is seed-robust. Seed 0 already lives in runs_video/.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DAVIS=${DAVIS:-data/DAVIS}
export PYTHONUNBUFFERED=1
mkdir -p logs
for S in 1 2; do
  OUT=runs_video_s$S
  echo "===== seed $S: feed-forward ====="
  $PY src/train_video.py --variant feedforward --davis-root "$DAVIS" --iters 8000 --lr 1e-3 \
      --eval-every 0 --out "$OUT" --seed $S 2>&1 | tee logs/ff_s$S.log
  echo "===== seed $S: energy critic ====="
  $PY src/train_energy.py --davis-root "$DAVIS" --init-ckpt "$OUT/feedforward.pt" --iters 3000 \
      --neg-steps 5 --step-size 5 --noise 0.02 --lr 1e-3 --eval-every 0 --out "$OUT" --seed $S 2>&1 | tee logs/energy_s$S.log
done
echo "SEEDS DONE"
