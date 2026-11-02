"""Check the falsifiable prediction against the finished arms.

The vocabulary diagnostic predicted 4+4 > 3+3 > 2+2 on the grounds that fewer
semantic slots means more of the random hash digit gets merged into semantic
tokens.  This prints what the finished arms actually show, so the report can
state the outcome rather than the hope.
"""
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = '/home/lzr/science/actionpiece_exp/tensorboard/AmazonReviews2014/ActionPiece'

ARMS = [
    ('2+2  (fewest slots)', 'beauty_mm_sep22_train'),
    ('3+3', 'beauty_mm_sep33_train'),
    ('4+4  (most slots)', 'beauty_mm_sep44_train'),
]


def load(sub, tag):
    for run in sorted(os.listdir(TB)):
        if sub not in run:
            continue
        d = os.path.join(TB, run)
        for f in os.listdir(d):
            if f.startswith('events.out.tfevents'):
                a = EventAccumulator(os.path.join(d, f),
                                     size_guidance={'scalars': 0})
                a.Reload()
                if tag in a.Tags().get('scalars', []):
                    return a
    return None


print('%-22s %-10s %-10s %-8s' % ('arm', 'val@10', 'test@10', 'epochs'))
print('-' * 54)
rows = []
for label, sub in ARMS:
    a = load(sub, 'Val_Metric/ndcg@10')
    if a is None:
        print('%-22s %-10s %-10s %-8s' % (label, '-', '-', 'not started'))
        continue
    v = [x.value for x in a.Scalars('Val_Metric/ndcg@10')]
    t = None
    if 'Test_Metric/ndcg@10' in a.Tags().get('scalars', []):
        t = max(x.value for x in a.Scalars('Test_Metric/ndcg@10'))
    rows.append((label, max(v), t))
    print('%-22s %-10.4f %-10s %-8d'
          % (label, max(v), '%.4f' % t if t is not None else 'n/a', len(v)))

print()
print('prediction was: 4+4 > 3+3 > 2+2  (more slots generalise better, because')
print('fewer slots means more of the random hash digit pollutes the vocabulary)')
print()
vals = [(lab, v) for lab, v, _ in rows]
if len(vals) >= 2:
    order = ' > '.join(lab.split()[0] for lab, _ in sorted(vals, key=lambda x: -x[1]))
    print('observed order by val nDCG@10: %s' % order)
    if order.startswith('4+4') and order.endswith('2+2'):
        print('-> consistent with the prediction')
    else:
        print('-> CONTRADICTS the prediction')
