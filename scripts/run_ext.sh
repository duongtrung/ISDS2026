#!/bin/bash
# Extension experiments -> results/ext_results.json (records tagged with modality + ch).
#  S4 modality ablation: feed-forward rgb / flow / both on DAVIS-16.
#  S5 backbone capacity: wider (ch=64) amortized path, K-sweep on DAVIS-16 val.
# Robust: no `tee` (which masks exit codes under set -e), skips already-trained
# checkpoints (resumable), and re-evaluates everything fresh at the end.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DAVIS=${DAVIS:-data/DAVIS}
export PYTHONUNBUFFERED=1
mkdir -p logs

train_ff() {  # $1 tag  $2 outdir  $3 extra-args
  if [ -f "$2/feedforward.pt" ]; then echo "skip $2 (exists)"; return; fi
  echo "train feed-forward $1"
  $PY src/train_video.py --variant feedforward --davis-root "$DAVIS" --iters 8000 --lr 1e-3 \
      --eval-every 0 --out "$2" --seed 0 $3 > "logs/ff_$1.log" 2>&1
}

train_ff rgb  runs_mod_rgb  "--modality rgb"
train_ff flow runs_mod_flow "--modality flow"
train_ff big  runs_big      "--ch 64"

if [ ! -f runs_big/amortized.pt ]; then
  echo "train energy critic (ch=64)"
  $PY src/train_energy.py --davis-root "$DAVIS" --init-ckpt runs_big/feedforward.pt --iters 3000 \
      --neg-steps 5 --step-size 5 --noise 0.02 --lr 1e-3 --eval-every 0 --out runs_big --seed 0 \
      > logs/energy_big.log 2>&1
fi

echo "eval all -> results/ext_results.json"
rm -f results/ext_results.json
E() { $PY src/eval_video.py --dataset davis --root "$DAVIS" --seed 0 --out results/ext_results.json "$@"; }
E --ckpt runs_mod_rgb/feedforward.pt  --steps 0
E --ckpt runs_mod_flow/feedforward.pt --steps 0
E --ckpt runs_video/feedforward.pt    --steps 0
E --ckpt runs_big/amortized.pt        --steps 0 1 2 5 10 20
E --ckpt runs_big/feedforward.pt      --steps 0
echo "EXT DONE"
