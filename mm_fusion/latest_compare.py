"""Compare all runs at the epoch the new 3+3 arm has just reached."""
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = '/home/lzr/science/actionpiece_exp/tensorboard/AmazonReviews2014/ActionPiece'

RUNS = [
    ('mm 3+3 (new)', 'beauty_mm_sep33'),
    ('text only 4x256', 'beauty_e1_qwen3_vl_8b_train'),
    ('sep 2x65536 old', 'beauty_e4_freqmax_pack2'),
    ('joint 4096', 'beauty_e3_qwen3_vl_8b_joint_d4096_train'),
    ('image only', 'beauty_e2_qwen3_vl_8b_image_train'),
]


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


data = {label: series(sub) for label, sub in RUNS}
new = data['mm 3+3 (new)']
e = len(new)
print('3+3 has reached epoch %d; best so far %.4f (epoch %d)'
      % (e, max(new), new.index(max(new)) + 1))
print()
print('val nDCG@10 at epoch %d (= latest common point):' % e)
print('%-18s %-9s %-9s %-9s' % ('run', 'e%d' % e, 'best', 'best epoch'))
print('-' * 50)
for label, _ in RUNS:
    v = data[label]
    if len(v) >= e and e > 0:
        seg = v[:e]
    else:
        seg = v
    if not seg:
        continue
    cur = v[e - 1] if len(v) >= e else None
    print('%-18s %-9s %-9.4f %-9d'
          % (label, '%.4f' % cur if cur is not None else '-',
             max(v), v.index(max(v)) + 1))
