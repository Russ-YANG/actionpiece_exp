#!/usr/bin/env bash
# Finisher v3.
#
# v1 waited on training processes and fired in the gap between two arms.
# v2 waited on the driver process, which is better but still depends on the
# driver exiting rather than on it having finished its work.
# v3 waits on the marker file that driver v5 touches as its last action, so
# there is no gap left to misread.
set -uo pipefail

AP=/home/lzr/science/actionpiece_exp
cd "$AP" || exit 1
PY=/home/lzr/science/action_piece-pr-6/.venv/bin/python
MARKER=logs/mm_driver_v5.done
LOG=logs/mm_finisher_v3.log

log () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== finisher v3 started; waiting for $MARKER ==="
while [ ! -f "$MARKER" ]; do
  sleep 120
done
log "driver v5 reported completion"
sleep 60   # let TensorBoard flush

log "regenerating the results table"
"$PY" mm_fusion/make_final_table.py --md mm_fusion/RESULTS_TABLE.md 2>&1 | tee -a "$LOG"

log "test-metric presence per arm"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TB = ('/home/lzr/science/actionpiece_exp/tensorboard/'
      'AmazonReviews2014/ActionPiece')
for sub in ('beauty_mm_sep22_train', 'beauty_mm_sep33_train',
            'beauty_mm_sep44_train', 'beauty_mm_sep22_seed42_train'):
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
        print('%-26s val_epochs=%-4d test=%s%s'
              % (sub, n, t, '' if t else '   <-- TEST METRICS MISSING'))
    else:
        print('%-26s NO RUN DIRECTORY' % sub)
PY

log "splicing into the report"
"$PY" - mm_fusion/RESULTS_DRAFT.md mm_fusion/RESULTS_TABLE.md \
        mm_fusion/RESULTS.md '<!-- RESULTS_TABLE -->' <<'PY' 2>&1 | tee -a "$LOG"
import sys
draft_path, table_path, out_path, marker = sys.argv[1:5]
draft = open(draft_path).read()
table = open(table_path).read().strip()
if marker not in draft:
    print('marker not found; draft left unchanged')
    open(out_path, 'w').write(draft)
    raise SystemExit(1)
open(out_path, 'w').write(draft.replace(marker, table))
print('wrote %s' % out_path)
PY

log "=== done: mm_fusion/RESULTS.md ==="
