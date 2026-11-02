"""Final comparison table: multimodal late-fusion arms vs the existing baselines.

Reads the TensorBoard event files that ActionPiece writes, so it works both while
a run is in progress and after it finishes.
"""
import os
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = '/home/lzr/science/actionpiece_exp/tensorboard/AmazonReviews2014/ActionPiece'

# label -> substring that identifies the run directory
ARMS = [
    ('mm 2+2 (new)', 'beauty_mm_sep22'),
    ('mm 3+3 (new)', 'beauty_mm_sep33'),
    ('mm 4+4 (new)', 'beauty_mm_sep44'),
    ('text only 4x256', 'beauty_e1_qwen3_vl_8b_train'),
    ('image only 4x256', 'beauty_e2_qwen3_vl_8b_image_train'),
    ('joint 4096 4x256', 'beauty_e3_qwen3_vl_8b_joint_d4096_train'),
    ('separate 2x65536 (old)', 'beauty_e4_freqmax_pack2'),
]

WANT = ['ndcg@10', 'recall@10']


def read(run_substr):
    for run in sorted(os.listdir(TB)):
        if run_substr not in run:
            continue
        d = os.path.join(TB, run)
        evs = [f for f in os.listdir(d) if f.startswith('events.out.tfevents')]
        if not evs:
            continue
        acc = EventAccumulator(os.path.join(d, evs[0]),
                               size_guidance={'scalars': 0})
        acc.Reload()
        tags = set(acc.Tags().get('scalars', []))
        out = {}
        for split, pfx in (('val', 'Val_Metric/'), ('test', 'Test_Metric/')):
            for w in WANT:
                t = pfx + w
                if t in tags:
                    vals = [s.value for s in acc.Scalars(t)]
                    out[f'{split}_{w}'] = max(vals)
                    out[f'{split}_{w}_epochs'] = len(vals)
        if out:
            return run, out
    return None, {}


print('%-26s %-11s %-11s %-11s %-11s' % ('arm', 'val nDCG@10', 'val R@10',
                                         'test nDCG@10', 'test R@10'))
print('-' * 74)
rows = []
for label, sub in ARMS:
    run, m = read(sub)
    if run is None:
        print('%-26s %s' % (label, '(no run found yet)'))
        continue
    def f(k):
        v = m.get(k)
        return '%.4f' % v if v is not None else '-'
    print('%-26s %-11s %-11s %-11s %-11s' % (label, f('val_ndcg@10'),
                                             f('val_recall@10'),
                                             f('test_ndcg@10'),
                                             f('test_recall@10')))
    rows.append((label, m))

print()
print('"-" means that split was never evaluated (the existing 2x65536 run')
print('crashed during its test pass, so it has validation numbers only).')
print()
for label, m in rows:
    n = m.get('val_ndcg@10_epochs')
    if n:
        print('  %-26s validation points recorded: %d' % (label, n))
