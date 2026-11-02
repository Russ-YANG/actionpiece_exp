#!/usr/bin/env bash
# Driver v5.
#
# Sequence:  seed-42 replica of 2+2  ->  the original 4+4 arm
#
# Two bugs from v4 are fixed here:
#
#  1. v4 waited on `kill -0 <pid>` for a tokenizer build. A process that has
#     exited but not been reaped is a zombie, and `kill -0` SUCCEEDS on zombies,
#     so the loop never ended and 4+4 sat idle for two hours. The wait now also
#     requires the process state not to be Z.
#
#  2. v4's finisher counterpart only watched for training processes, so it fired
#     in the gap between two arms. This driver writes a marker file when it is
#     completely done, which is a signal with no gap.
set -uo pipefail

AP=/home/lzr/science/actionpiece_exp
cd "$AP" || exit 1
mkdir -p logs
DRIVER_LOG=logs/mm_driver_v5.log
PY=/home/lzr/science/action_piece-pr-6/.venv/bin/python
DONE_MARKER=logs/mm_driver_v5.done

rm -f "$DONE_MARKER"
log () { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }

log "=== driver v5 started ==="

# ---- zombie-safe wait on a pidfile ----------------------------------------
wait_pidfile () {
  local pf="$1" label="$2" p state
  [ -f "$pf" ] || return 0
  p=$(cat "$pf" 2>/dev/null)
  [ -n "$p" ] || return 0
  while :; do
    kill -0 "$p" 2>/dev/null || break            # really gone
    state=$(ps -o stat= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$state" ] || break                     # gone between the two calls
    case "$state" in
      Z*) log "$label: pid $p is a zombie, treating as finished"; break ;;
    esac
    log "$label: still running (pid $p, state $state)"
    sleep 120
  done
}

# ---- 1. seed-42 2+2 tokenizer --------------------------------------------
wait_pidfile logs/mm_sep22_seed42_tokenizer.pid "arm 22 seed42 tokenizer"

if ! grep -q 'Tokenizer-only run completed' logs/mm_sep22_seed42_tokenizer.log 2>/dev/null; then
  log "arm 22 seed42: tokenizer did NOT report completion; check the log"
  tail -5 logs/mm_sep22_seed42_tokenizer.log | tee -a "$DRIVER_LOG"
fi

# ---- 2. arms ---------------------------------------------------------------
# spec: config-stem : eval_batch_size : label
for spec in \
    "mm_sep22_seed42:32:arm 22 seed42" \
    "mm_sep44:16:arm 44 seed2024" ; do
  stem="${spec%%:*}"
  rest="${spec#*:}"
  ebs="${rest%%:*}"
  label="${rest#*:}"

  if ! "$PY" mm_fusion/validate_vocab.py "$stem" 2>&1 | tee -a "$DRIVER_LOG"; then
    log "$label: vocabulary validation FAILED, skipping"
    continue
  fi

  log "$label: starting training (eval_batch_size=$ebs)"
  LOG="logs/${stem}_train.log"
  PYTHONUNBUFFERED=1 "$PY" main.py \
      --config_file="experiments/${stem}_beauty.yaml" \
      --tokenizer_only=False \
      --epochs=200 \
      --eval_batch_size="$ebs" \
      --run_id="beauty_${stem}_train" \
      > "$LOG" 2>&1
  rc=$?
  log "$label: training exited with code $rc (log: $LOG)"
  if [ "$rc" -ne 0 ]; then
    tail -20 "$LOG" | tee -a "$DRIVER_LOG"
  fi
done

log "=== driver v5: all arms done ==="
touch "$DONE_MARKER"
log "ComfyUI restart: cd /home/lzr/ComfyUI && ./venv/bin/python -u main.py --listen 127.0.0.1 --port 8195 --vram-headroom 1"
