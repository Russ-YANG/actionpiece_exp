#!/usr/bin/env bash
cd /home/lzr/science/actionpiece_exp || exit 1
mkdir -p logs
ARM=${1:-33}
LOG="logs/mm_sep${ARM}_tokenizer.log"
echo "=== building tokenizer for arm ${ARM} -> ${LOG} ==="
nohup env PYTHONUNBUFFERED=1 \
  /home/lzr/science/action_piece-pr-6/.venv/bin/python main.py \
  --config_file="experiments/mm_sep${ARM}_beauty.yaml" \
  --run_id="beauty_mm_sep${ARM}_tokenizer" \
  > "$LOG" 2>&1 &
echo $! > "logs/mm_sep${ARM}_tokenizer.pid"
echo "pid $(cat logs/mm_sep${ARM}_tokenizer.pid)"
sleep 25
echo "--- first 60 lines of the log ---"
head -60 "$LOG"
