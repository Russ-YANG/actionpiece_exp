#!/usr/bin/env bash
AP=/home/lzr/science/actionpiece_exp
echo "=== driver log ==="
cat "$AP/logs/mm_driver.log" 2>/dev/null || echo "(no driver log yet)"
echo
echo "=== driver process ==="
pgrep -af 'mm_driver.sh' | head -3
echo
echo "=== build progress ==="
cd "$AP" || exit 1
printf '%-5s %-8s %-10s %-9s %s\n' ARM PID STATE ELAPSED VOCAB
for a in 22 33 44; do
  f="logs/mm_sep${a}_tokenizer.pid"
  [ -f "$f" ] || { echo "$a  (no pid)"; continue; }
  pid=$(cat "$f")
  kill -0 "$pid" 2>/dev/null && st=RUNNING || st=stopped
  el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  v=$(tr '\r' '\n' < "logs/mm_sep${a}_tokenizer.log" 2>/dev/null \
      | grep -oE "Vocab size: [0-9]+ / 40000" | tail -1)
  printf '%-5s %-8s %-10s %-9s %s\n' "$a" "$pid" "$st" "${el:-–}" "${v:-–}"
done
echo
free -h | head -2
