"""Estimate a noise floor for the validation metric.

Only one seed was run per arm, so the between-arm differences cannot be tested
directly.  As a proxy, measure how much a single run's own validation metric
fluctuates epoch-to-epoch once it has largely converged.  If the between-arm
differences are of the same order, the arms are not separable without replicates.
"""
import os
import statistics

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB = '/home/lzr/science/actionpiece_exp/tensorboard/AmazonReviews2014/ActionPiece'

RUNS = [
    ('mm 3+3 (new)', 'beauty_mm_sep33'),
    ('text only 4x256', 'beauty_e1_qwen3_vl_8b_train'),
    ('image only 4x256', 'beauty_e2_qwen3_vl_8b_image_train'),
    ('joint 4096 4x256', 'beauty_e3_qwen3_vl_8b_joint_d4096_train'),
    ('separate 2x65536 (old)', 'beauty_e4_freqmax_pack2'),
]

WINDOW = 30          # epochs treated as "converged"


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


print('epoch-to-epoch fluctuation of val nDCG@10 over the last %d logged epochs'
      % WINDOW)
print()
print('%-24s %-8s %-10s %-10s %-10s'
      % ('run', 'epochs', 'mean', 'std', 'best-minus-mean'))
print('-' * 68)

stds = []
for label, sub in RUNS:
    v = series(sub)
    if len(v) < 5:
        print('%-24s (only %d points)' % (label, len(v)))
        continue
    seg = v[-WINDOW:] if len(v) >= WINDOW else v
    m = statistics.mean(seg)
    sd = statistics.pstdev(seg)
    stds.append(sd)
    print('%-24s %-8d %-10.4f %-10.4f %-10.4f'
          % (label, len(seg), m, sd, max(v) - m))

print()
if stds:
    print('median epoch-to-epoch std: %.4f' % statistics.median(stds))
    print('between-arm differences seen so far are of order 0.003-0.005;')
    print('compare that with this noise floor before calling any arm better.')
print()
print('NOTE: this measures epoch-to-epoch instability of one run, which is a')
print('lower bound on seed-to-seed variance, not a substitute for replicates.')
