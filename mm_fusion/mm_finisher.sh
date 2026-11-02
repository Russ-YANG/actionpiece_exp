#!/usr/bin/env bash
# Autonomously finish the experiment: wait for all three trainings to end, then
# produce the final comparison table and flag any missing test metrics.
set -uo pipefail

AP=/home/lzr/science/actionpiece_exp
cd "$AP" || exit 1
PY=/home/lzr/science/action_piece-pr-6/.venv/bin/python
LOG=logs/mm_finisher.log

log () { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== finisher started; waiting for all three trainings to end ==="

while pgrep -f 'run_id=beauty_mm_sep' > /dev/null 2>&1; do
  sleep 300
done
log "no training process left"

sleep 60   # let TensorBoard flush

log "generating the final table"
"$PY" mm_fusion/make_final_table.py --md mm_fusion/RESULTS_TABLE.md 2>&1 | tee -a "$LOG"

log "checking which arms produced test metrics"
"$PY" - <<'PY' 2>&1 | tee -a "$LOG"
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TB = ('/home/lzr/science/actionpiece_exp/tensorboard/'
      'AmazonReviews2014/ActionPiece')
for arm in ('22', '33', '44'):
    sub = f'beauty_mm_sep{arm}_train'
    found = None
    for run in sorted(os.listdir(TB)):
        if sub not in run:
            continue
        d = os.path.join(TB, run)
        for f in os.listdir(d):
            if f.startswith('events.out.tfevents'):
                acc = EventAccumulator(os.path.join(d, f),
                                       size_guidance={'scalars': 0})
                acc.Reload()
                tags = acc.Tags().get('scalars', [])
                v = 'Val_Metric/ndcg@10' in tags
                t = 'Test_Metric/ndcg@10' in tags
                n = len(acc.Scalars('Val_Metric/ndcg@10')) if v else 0
                found = (n, v, t)
    if found:
        n, v, t = found
        print('arm %s: val_epochs=%d val=%s test=%s%s'
              % (arm, n, v, t,
                 '' if t else '   <-- TEST METRICS MISSING'))
    else:
        print('arm %s: NO RUN DIRECTORY' % arm)
PY

log "work left running:"
pgrep -af 'main.py' | tee -a "$LOG" || echo "  none"

log "=== finisher done. results: mm_fusion/RESULTS_TABLE.md ==="
log "ComfyUI restart: cd /home/lzr/ComfyUI && ./venv/bin/python -u main.py --listen 127.0.0.1 --port 8195 --vram-headroom 1"
