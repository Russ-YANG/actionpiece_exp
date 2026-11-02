#!/usr/bin/env bash
# Driver v4.
#
# Replaces v3 part-way: arm 3+3 is ALREADY TRAINING under the old driver, so this
# one simply waits for that process to exit and then takes over the remaining
# arms.
#
# Why v4 exists: measured GPU memory during the 7-slot evaluation is 20,642 MiB
# of 24,463 MiB.  Beam-search memory scales with the token sequence length, which
# is n_categories * max_item_seq_len + 2 = 182 for 9 slots versus 142 for 7, so
# the 4+4 arm is projected at ~26 GiB and would OOM.  Evaluation batch size does
# not change the metric, only the batching, so 4+4 runs its beam-search
# evaluation with a smaller batch.
set -uo pipefail

AP=/home/lzr/science/actionpiece_exp
cd "$AP" || exit 1
mkdir -p logs
DRIVER_LOG=logs/mm_driver.log
PY=/home/lzr/science/action_piece-pr-6/.venv/bin/python

log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }

log "=== driver v4 started (takes over after the running 3+3 training) ==="

# ---- 1. wait for the 3+3 training that is already in flight ---------------
while pgrep -f 'run_id=beauty_mm_sep33_train' > /dev/null 2>&1; do
  sleep 120
done
log "arm 33: training finished"

# ---- 2. remaining arms -----------------------------------------------------
# arm:batch_size_for_beam_search_eval
for spec in 22:32 44:16; do
  a="${spec%%:*}"
  ebs="${spec##*:}"

  pidfile="logs/mm_sep${a}_tokenizer.pid"
  while [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; do
    log "arm $a: still waiting for its tokenizer build"
    sleep 120
  done

  if ! "$PY" mm_fusion/validate_vocab.py "$a" 2>&1 | tee -a "$DRIVER_LOG"; then
    log "arm $a: vocabulary validation FAILED, skipping"
    continue
  fi

  log "arm $a: starting training (eval_batch_size=$ebs)"
  free -h | head -2 | tee -a "$DRIVER_LOG"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader \
    | tee -a "$DRIVER_LOG"

  LOG="logs/mm_sep${a}_train.log"
  PYTHONUNBUFFERED=1 "$PY" main.py \
      --config_file="experiments/mm_sep${a}_beauty.yaml" \
      --tokenizer_only=False \
      --epochs=200 \
      --eval_batch_size="$ebs" \
      --run_id="beauty_mm_sep${a}_train" \
      > "$LOG" 2>&1
  rc=$?
  log "arm $a: training exited with code $rc (log: $LOG)"
  if [ "$rc" -ne 0 ]; then
    log "arm $a: last 20 log lines follow"
    tail -20 "$LOG" | tee -a "$DRIVER_LOG"
  fi
done

log "=== all three arms done ==="
log "ComfyUI restart: cd /home/lzr/ComfyUI && ./venv/bin/python -u main.py --listen 127.0.0.1 --port 8195 --vram-headroom 1"
