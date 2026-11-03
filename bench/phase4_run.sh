#!/bin/sh
# One series phase 4 training run (bench/phase4_train.py) as its own process under the memory guard: the job stops cleanly
# past 96 GB of physical footprint, 1 GB of new swap or 3 h; the Metal allocator's watermarks set (macai.configure's values);
# the Hub offline. One GPU job at a time: run these one after another, never side by side.
#
#   HF_HOME=<the Hub cache holding the models> GUARD=<the guard script> PY=<the venv's python> \
#       bench/phase4_run.sh p41_ref_a --model 0.8b --arm reference --epochs 4 --batch 4 --accum 4
#
# The log (the guard's lines, the trainer's output, the guard's peak footprint at the end) goes to $LOGS/<name>.log, the run's
# record to $LOGS/<name>.json; LOGS defaults to bench/phase4_logs (the memory series: LOGS=bench/memory_logs).
set -eu
cd "$(dirname "$0")/.."
: "${GUARD:?set GUARD to the memory guard script}"
: "${HF_HOME:?set HF_HOME to the Hub cache holding the models}"
PY=${PY:-python}
LOGS=${LOGS:-bench/phase4_logs}
name=$1
shift
mkdir -p "$LOGS"
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.75 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.55 HF_HUB_OFFLINE=1 \
    "$PY" "$GUARD" --limit-gb 96 --swap-gb 1 --wall-s 10800 --log "$LOGS/$name.log" -- \
    "$PY" bench/phase4_train.py --name "$name" --logdir "$LOGS" "$@"
