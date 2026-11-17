#!/usr/bin/env bash
# Train one configuration on the 4 folds, then print the summary table.
#   bash scripts/run_cv.sh <encoder> <run_name> [extra train.py args ...]
#   bash scripts/run_cv.sh wavlm_large wavlm_asp
#   bash scripts/run_cv.sh xlsr300m xlsr_l2only --train_corpora l2arctic --shares non_vietnamese:0.5,vn_l2arctic:0.5
set -euo pipefail
PY=${PY:-.venv/bin/python}
ENC=$1; RUN=$2; shift 2
for k in 0 1 2 3; do
  if [ -f "runs/$RUN/fold$k/metrics.json" ]; then echo "skip fold $k (done)"; continue; fi
  $PY scripts/train.py --encoder "$ENC" --fold "$k" --run "$RUN" "$@"
done
$PY scripts/summarize.py
