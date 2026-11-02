#!/usr/bin/env bash
# Post-finisher: once mm_fusion/RESULTS_TABLE.md exists, substitute it for the
# <!-- RESULTS_TABLE --> marker in RESULTS_DRAFT.md and write mm_fusion/RESULTS.md
set -uo pipefail

AP=/home/lzr/science/actionpiece_exp
cd "$AP" || exit 1
LOG=logs/mm_postfinish.log
PY=/home/lzr/science/action_piece-pr-6/.venv/bin/python
TABLE=mm_fusion/RESULTS_TABLE.md
DRAFT=mm_fusion/RESULTS_DRAFT.md
OUT=mm_fusion/RESULTS.md
MARKER='<!-- RESULTS_TABLE -->'

log () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "waiting for $TABLE"
while [ ! -s "$TABLE" ]; do
  sleep 120
done
log "found $TABLE ($(stat -c%s "$TABLE") bytes)"
sleep 5

"$PY" - "$DRAFT" "$TABLE" "$OUT" "$MARKER" <<'PY'
import sys

draft_path, table_path, out_path, marker = sys.argv[1:5]
draft = open(draft_path).read()
table = open(table_path).read().strip()

if marker not in draft:
    print('marker %r not found; writing draft unchanged' % marker)
    open(out_path, 'w').write(draft)
    raise SystemExit(1)

report = draft.replace(marker, table)
open(out_path, 'w').write(report)
print('wrote %s (%d bytes)' % (out_path, len(report)))
PY

rc=$?
log "splice exit code $rc"
if [ "$rc" -eq 0 ]; then
  log "=== final report ready: $OUT ==="
fi
