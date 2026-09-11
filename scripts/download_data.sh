#!/bin/bash
# Download + unpack the three datasets into data/, fix up the DAVIS split lists, and
# precompute RAFT optical flow for all of them (~16 GB on disk, ~1 h with a GPU).
# Idempotent: existing archives/extractions/flow are skipped.
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p data

fetch() {  # $1 url  $2 target
  if [ -f "$2" ]; then echo "have $(basename "$2")"; return; fi
  echo "downloading $(basename "$2") ..."
  curl -L -C - --retry 6 --retry-delay 5 --retry-all-errors -o "$2" "$1"
}

# ---------------------------------------------------------------- DAVIS-16
fetch "https://cgl.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip" data/DAVIS-data.zip
[ -d data/DAVIS/JPEGImages ] || unzip -q -o data/DAVIS-data.zip -d data/
# The archive ships the old 480p path-list format; src/davis.py expects one sequence
# name per line under ImageSets/2016/.
if [ ! -f data/DAVIS/ImageSets/2016/train.txt ]; then
  mkdir -p data/DAVIS/ImageSets/2016
  for s in train val; do
    awk '{print $1}' "data/DAVIS/ImageSets/480p/$s.txt" \
      | sed -E 's|.*/480p/([^/]+)/.*|\1|' | awk '!seen[$0]++' \
      > "data/DAVIS/ImageSets/2016/$s.txt"
  done
fi

# ---------------------------------------------------------------- FBMS-59 (test)
fetch "https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Testset.zip" \
      data/FBMS_Testset.zip
[ -d data/FBMS/Testset ] || { mkdir -p data/FBMS; unzip -q -o data/FBMS_Testset.zip -d data/FBMS/; }

# ---------------------------------------------------------------- SegTrack-v2
fetch "https://web.engr.oregonstate.edu/~lif/SegTrack2/SegTrackv2.zip" data/SegTrackv2.zip
[ -d data/SegTrackv2/JPEGImages ] || { unzip -q -o data/SegTrackv2.zip -d data/; \
  [ -d data/SegTrackv2 ] || mv data/SegTrackv2* data/SegTrackv2; }

# ---------------------------------------------------------------- optical flow (RAFT)
flow() {  # $1 frames-root  $2 out
  [ -d "$2" ] && [ -n "$(ls -A "$2" 2>/dev/null)" ] && { echo "have flow $2"; return; }
  echo "precomputing flow -> $2"
  $PY src/precompute_flow.py --frames-root "$1" --out "$2" --height 256 --width 448
}
flow data/DAVIS/JPEGImages/480p data/DAVIS/Flow
flow data/SegTrackv2/JPEGImages data/SegTrackv2/Flow
flow data/FBMS/Testset          data/FBMS/Flow

echo "DATA READY"
$PY src/dataset_stats.py || true
