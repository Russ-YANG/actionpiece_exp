"""Compare arms on a smoothed validation curve.

`mm 3+3` has an epoch-to-epoch std around 0.0019 while the converged legacy arms
sit near 0.0004, so single-epoch comparisons are dominated by the new arm's own
fluctuation.  A moving average is the honest comparison; this prints both so the
difference is visible.
"""
import os
import statistics
from collections import deque

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = '/home/lzr/science/actionpiece_exp/tensorboard/AmazonReviews2014/ActionPiece'

RUNS = [
    ('mm 3+3 (new)', 'beauty_mm_sep33'),
    ('text only 4x256', 'beauty_e1_qwen3_vl_8b_train'),
    ('image only 4x256', 'beauty_e2_qwen3_vl_8b_image_train'),
    ('joint 4096 4x256', 'beauty_e3_qwen3_vl_8b_joint_d4096_train'),
    ('separate 2x65536 (old)', 'beauty_e4_freqmax_pack2'),
]

W = 5          # smoothing window


def series(sub, tag='Val_Metric/ndcg@10'):
    for run in sorted(os.listdir(TB)):
        if sub not in run:
            continue
        d = os.path.join(TB, run)
        evs = [f for f in os.listdir(d) if f.startswith('events.out.tfevents')]
        if not evs:
            continue
        acc = EventAccumulator(os.path.join(d, evs[0]),
                               size_guidance={'scalars': 0})
        acc.Reload()
        if tag in acc.Tags().get('scalars', []):
            return [s.value for s in acc.Scalars(tag)]
    return []


def smoothed(v, w=W):
    out = []
    dq = deque(maxlen=w)
    for x in v:
        dq.append(x)
        out.append(sum(dq) / len(dq))
    return out


data = {label: series(sub) for label, sub in RUNS}
new = data['mm 3+3 (new)']
n = len(new)
new_s = smoothed(new)

print('%d-epoch moving average of val nDCG@10' % W)
print('3+3 has %d epochs logged; best single epoch %.4f' % (n, max(new)))
print()
print('%-24s %-10s %-10s %-10s' % ('run', 'smoothed', 'best-single', 'best epoch'))
print('-' * 58)
for label, _ in RUNS:
    v = data[label]
    if not v:
        continue
    s = smoothed(v)
    window = s[n - 1] if len(s) >= n else s[-1]
    print('%-24s %-10.4f %-10.4f %-10d'
          % (label, window, max(v), v.index(max(v)) + 1))

print()
print('smoothed lead/lag of 3+3 versus each baseline at every epoch it covers:')
for label, _ in RUNS:
    if label.startswith('mm '):
        continue
    v = data[label]
    if not v:
        continue
    vs = smoothed(v)
    m = min(n, len(vs))
    diffs = [new_s[i] - vs[i] for i in range(m)]
    if not diffs:
        continue
    print('  %-24s mean %+.4f  latest %+.4f  min %+.4f  max %+.4f'
          % (label, sum(diffs) / len(diffs), diffs[-1], min(diffs), max(diffs)))
