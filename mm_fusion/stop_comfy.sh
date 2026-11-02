#!/usr/bin/env bash
# Record how ComfyUI was started so it can be restored verbatim, then stop it.
set -uo pipefail

PID=$(pgrep -f 'ComfyUI/.venv/bin/python' | head -1)
echo "ComfyUI pid: ${PID:-none}"
[ -z "${PID:-}" ] && exit 0

echo "=== full command line ==="
tr '\0' ' ' < "/proc/$PID/cmdline"; echo
echo "=== cwd ==="
readlink -f "/proc/$PID/cwd"
echo "=== environment essentials ==="
tr '\0' '\n' < "/proc/$PID/environ" | grep -E '^(PATH|CUDA_VISIBLE_DEVICES|PYTHONPATH|VIRTUAL_ENV)=' || true
echo "=== rss before ==="
ps -o rss= -p "$PID" | awk '{printf "%.2f GiB\n", $1/1048576}'

mkdir -p /home/lzr/science/actionpiece_exp/logs
{
  echo "# recorded $(date -Is) before stopping ComfyUI for the ActionPiece vocab run"
  echo "cmd: $(tr '\0' ' ' < /proc/$PID/cmdline)"
  echo "cwd: $(readlink -f /proc/$PID/cwd)"
} > /home/lzr/science/actionpiece_exp/logs/comfyui_restart_note.txt
cat /home/lzr/science/actionpiece_exp/logs/comfyui_restart_note.txt

echo
echo "=== sending SIGTERM ==="
kill -TERM "$PID"
for i in $(seq 1 20); do
  sleep 2
  if ! kill -0 "$PID" 2>/dev/null; then echo "exited after ${i}x2s"; break; fi
done
if kill -0 "$PID" 2>/dev/null; then
  echo "still alive, sending SIGKILL"
  kill -KILL "$PID"; sleep 3
fi
echo
echo "=== memory after ==="
free -h
