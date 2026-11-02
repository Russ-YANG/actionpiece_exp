#!/usr/bin/env bash
cd /home/lzr/science/actionpiece_exp || exit 1
echo "=== the three tokenizer builds ==="
printf '%-6s %-9s %-10s %-8s %s\n' ARM PID STATE ELAPSED VOCAB
for a in 22 33 44; do
  f="logs/mm_sep${a}_tokenizer.pid"
  [ -f "$f" ] || { printf '%-6s %s\n' "$a" "no pid file"; continue; }
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then st=RUNNING; else st=stopped; fi
  el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  v=$(tr '\r' '\n' < "logs/mm_sep${a}_tokenizer.log" | grep -oE "Vocab size: [0-9]+ / 40000" | tail -1)
  printf '%-6s %-9s %-10s %-8s %s\n' "$a" "$pid" "$st" "${el:-–}" "${v:-–}"
done

echo
echo "=== artefacts per arm ==="
for a in 22 33 44; do
  n=${a:0:1}
  echo "--- arm ${a} ---"
  ls -la cache/AmazonReviews2014/Beauty/processed/ 2>/dev/null \
    | grep -E "opq${n}x256|t${n}x256" | awk '{printf "   %10d  %s\n", $5, $9}'
done

echo
echo "=== resources ==="
free -h | head -2
uptime
echo "load / cpu count: $(nproc)"
