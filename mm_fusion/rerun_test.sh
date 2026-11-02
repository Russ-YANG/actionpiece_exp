#!/usr/bin/env bash
# Re-run ONLY the test pass for one of the multimodal arms, from its best
# checkpoint.  Use this if a run dies during test evaluation (the legacy
# separate baseline hit `torch.AcceleratorError: CUDA error: unknown error` at
# 69% of its test pass and lost its test metrics).
#
# Usage:  rerun_test.sh <22|33|44> [eval_batch_size]
#
# Note that the effective test batch inside the pipeline is
# `eval_batch_size // n_inference_ensemble`, i.e. the default 16 below becomes
# a batch of 3 with the 5-ensemble setting, which is deliberately conservative.
set -uo pipefail

ARM=${1:?usage: rerun_test.sh <22|33|44> [eval_batch_size]}
EBS=${2:-16}

AP=/home/lzr/science/actionpiece_exp
cd "$AP" || exit 1
PY=/home/lzr/science/action_piece-pr-6/.venv/bin/python
CFG="experiments/mm_sep${ARM}_beauty.yaml"

if [ ! -f "$CFG" ]; then
  echo "config not found: $CFG" >&2
  exit 1
fi

CKPT=$(ls -t ckpt/beauty_mm_sep${ARM}_train-*.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "no checkpoint found for arm $ARM in ckpt/" >&2
  exit 1
fi

LOG="logs/mm_sep${ARM}_testonly.log"
echo "arm            : $ARM"
echo "config         : $CFG"
echo "checkpoint     : $CKPT ($(stat -c%s "$CKPT") bytes)"
echo "eval_batch_size: $EBS  -> effective test batch $((EBS / 5))"
echo "log            : $LOG"

PYTHONUNBUFFERED=1 "$PY" main.py \
  --config_file="$CFG" \
  --run_id="beauty_mm_sep${ARM}_test" \
  --eval_batch_size="$EBS" \
  --test_checkpoint="$AP/$CKPT" \
  > "$LOG" 2>&1

rc=$?
echo "exit code: $rc"
tail -5 "$LOG"
exit "$rc"
