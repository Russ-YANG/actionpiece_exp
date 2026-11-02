#!/usr/bin/env bash
# Finisher v2.
#
# v1 waited only for `pgrep -f 'run_id=beauty_mm_sep'`.  That fired prematurely
# in the gap between arm 22 exiting and arm 44 starting, so the first results
# files were produced without 4+4.  v2 waits for the *driver* to exit first,
# because the driver only exits after every arm has been attempted.
set -uo pipefail

AP=/home/lzr/science/actionpiece_exp
cd "$AP" || exit 1
PY=/home/lzr/science/action_piece-pr-6/.venv/bin/python
LOG=logs/mm_finisher_v2.log

log () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== finisher v2 started ==="

# 1. the driver is the real completion signal
while pgrep -f 'bash mm_fusion/mm_driver.sh' > /dev/null 2>&1; do
  sleep 120
done
log "driver has exited"

# 2. belt and braces: no training process left either
while pgrep -f 'run_id=beauty_mm_sep' > /dev/null 2>&1; do
  sleep 60
done
log "no training processes left"

sleep 60   # let TensorBoard flush

log "regenerating the results table"
"$PY" mm_fusion/make_final_table.py --md mm_fusion/RESULTS_TABLE.md 2>&1 | tee -a "$LOG"

log "which arms produced test metrics"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TB = ('/home/lzr/science/actionpiece_exp/tensorboard/'
      'AmazonReviews2014/ActionPiece')
for arm in ('22', '33', '44'):
    sub = 'beauty_mm_sep%s_train' % arm
    found = None
    for run in sorted(os.listdir(TB)):
        if sub not in run:
            continue
        d = os.path.join(TB, run)
        for f in os.listdir(d):
            if f.startswith('events.out.tfevents'):
                a = EventAccumulator(os.path.join(d, f),
                                     size_guidance={'scalars': 0})
                a.Reload()
                tags = a.Tags().get('scalars', [])
                n = len(a.Scalars('Val_Metric/ndcg@10')) \
                    if 'Val_Metric/ndcg@10' in tags else 0
                found = (n, 'Test_Metric/ndcg@10' in tags)
    if found:
        n, t = found
        print('arm %s: val_epochs=%d test_metrics=%s%s'
              % (arm, n, t, '' if t else '   <-- TEST METRICS MISSING'))
    else:
        print('arm %s: NO RUN DIRECTORY' % arm)
PY

# 3. splice into the report
log "splicing the table into the report"
"$PY" - mm_fusion/RESULTS_DRAFT.md mm_fusion/RESULTS_TABLE.md \
        mm_fusion/RESULTS.md '<!-- RESULTS_TABLE -->' <<'PY' 2>&1 | tee -a "$LOG"
import sys
draft_path, table_path, out_path, marker = sys.argv[1:5]
draft = open(draft_path).read()
table = open(table_path).read().strip()
if marker not in draft:
    print('marker not found; writing draft unchanged')
    open(out_path, 'w').write(draft)
    raise SystemExit(1)
open(out_path, 'w').write(draft.replace(marker, table))
print('wrote %s' % out_path)
PY

log "=== done. results: mm_fusion/RESULTS.md ==="
log "ComfyUI restart: cd /home/lzr/ComfyUI && ./venv/bin/python -u main.py --listen 127.0.0.1 --port 8195 --vram-headroom 1"
