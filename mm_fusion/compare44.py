#!/usr/bin/env bash
cd /home/lzr/science/actionpiece_exp || exit 1
/home/lzr/science/action_piece-pr-6/.venv/bin/python - <<'PY'
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
TB='tensorboard/AmazonReviews2014/ActionPiece'

def series(sub):
    best=None
    for r in os.listdir(TB):
        if sub not in r: continue
        d=os.path.join(TB,r)
        if not os.path.isdir(d): continue
        evs=[f for f in os.listdir(d) if f.startswith('events.out.tfevents')]
        if not evs: continue
        m=max(os.path.getmtime(os.path.join(d,f)) for f in evs)
        if best is None or m>best[0]: best=(m,os.path.join(d,evs[0]))
    if best is None: return []
    a=EventAccumulator(best[1],size_guidance={'scalars':0}); a.Reload()
    t='Val_Metric/ndcg@10'
    return [x.value for x in a.Scalars(t)] if t in a.Tags().get('scalars',[]) else []

A={'2+2 seed2024':series('beauty_mm_sep22_train'),
   '2+2 seed42':series('beauty_mm_sep22_seed42_train'),
   '3+3':series('beauty_mm_sep33_train'),
   '4+4':series('beauty_mm_sep44_train')}

n44=len(A['4+4'])
print('4+4 has reached epoch %d; comparing all three arms at matched epochs' % n44)
print()
print('%-14s %-9s %-9s %-9s %-9s' % ('epoch','2+2 s2024','3+3','4+4','4+4 vs 3+3'))
print('-'*56)
for e in (30,50,70,90,100,105,n44):
    if e>n44: continue
    a=A['2+2 seed2024'][e-1] if len(A['2+2 seed2024'])>=e else None
    b=A['3+3'][e-1] if len(A['3+3'])>=e else None
    c=A['4+4'][e-1] if len(A['4+4'])>=e else None
    d=('%+.4f'%(c-b)) if (b is not None and c is not None) else '-'
    print('%-14d %-9s %-9s %-9s %-9s' % (e,
        '%.4f'%a if a else '-', '%.4f'%b if b else '-', '%.4f'%c if c else '-', d))

print()
print('best so far at the same epoch count:')
for lab in ('2+2 seed2024','2+2 seed42','3+3','4+4'):
    v=A[lab][:n44]
    print('  %-14s best %-8.4f (e%d)   latest %.4f   epochs %d'
          % (lab, max(v), v.index(max(v))+1, v[-1], len(A[lab])))

print()
print('paired lead/lag of 4+4 against 3+3 over every shared epoch:')
b=A['3+3'][:n44]; c=A['4+4'][:n44]
diffs=[cc-bb for bb,cc in zip(b,c)]
print('  mean %+.4f   latest %+.4f   min %+.4f   max %+.4f'
      % (sum(diffs)/len(diffs), diffs[-1], min(diffs), max(diffs)))
