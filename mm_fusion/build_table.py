"""Build the requested 2+2 / 3+3 (and 4+4 when ready) table.

Columns: OPQ token count, Recall@5, NDCG@5, Recall@10, NDCG@10.

Validation numbers are taken at the epoch where the model-selection metric
(val nDCG@10) peaked, so all four come from a single checkpoint.  Test numbers
come from that same saved checkpoint, and the test series has exactly one point
(the test pass runs once, at the end of training).
"""
import os
import re

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

AP = '/home/lzr/science/actionpiece_exp'
TB = f'{AP}/tensorboard/AmazonReviews2014/ActionPiece'
ARMS = [('2+2', '22'), ('3+3', '33'), ('4+4', '44')]
TOTAL_EPOCHS = 200
VAL_METRICS = ('recall@5', 'ndcg@5', 'recall@10', 'ndcg@10')


def cfg_tokens(arm):
    txt = open(f'{AP}/experiments/mm_sep{arm}_beauty.yaml').read()

    def get(k):
        m = re.search(r'^%s:\s*(\S+)' % k, txt, re.M)
        return m.group(1) if m else '?'
    return get('pq_n_codebooks'), get('image_pq_n_codebooks')


def load(sub):
    for run in sorted(os.listdir(TB)):
        if sub not in run:
            continue
        d = os.path.join(TB, run)
        for f in os.listdir(d):
            if f.startswith('events.out.tfevents'):
                a = EventAccumulator(os.path.join(d, f),
                                     size_guidance={'scalars': 0})
                a.Reload()
                return a
    return None


def series(a, split, metric):
    t = f'{split}_Metric/{metric}'
    if a is None or t not in a.Tags().get('scalars', []):
        return []
    return [x.value for x in a.Scalars(t)]


data = {}
for label, arm in ARMS:
    a = load(f'beauty_mm_sep{arm}_train')
    tc, ic = cfg_tokens(arm)
    val10 = series(a, 'Val', 'ndcg@10')
    entry = dict(tc=tc, ic=ic, epochs=len(val10), total=TOTAL_EPOCHS,
                 finished=len(val10) >= TOTAL_EPOCHS)
    if val10:
        e = val10.index(max(val10))
        entry['best_epoch'] = e + 1
        entry['val'] = {
            m: (series(a, 'Val', m)[e] if len(series(a, 'Val', m)) > e else None)
            for m in VAL_METRICS}
        # the test pass logs a single point, so it is not indexed by epoch
        entry['test'] = {
            m: (lambda s: s[-1] if s else None)(series(a, 'Test', m))
            for m in VAL_METRICS}
    data[label] = entry


def table(split, title):
    print(title)
    print()
    print('| OPQ 分词数 | Recall@5 | NDCG@5 | Recall@10 | NDCG@10 |')
    print('|---|---|---|---|---|')
    for label, _ in ARMS:
        d = data[label]
        tok = '%s (%s+%s)' % (label, d['tc'], d['ic'])
        if split not in d:
            print('| %s | %s | | | |'
                  % (tok, '训练中 %d/%d' % (d['epochs'], d['total'])
                     if d['epochs'] else '未开始'))
            continue
        v = d[split]
        if v.get('ndcg@10') is None:
            print('| %s | 尚未产生 | | | |' % tok)
            continue
        # do not present partial-training numbers as results
        if split == 'val' and not d['finished']:
            print('| %s | *训练中 %d/%d* | | | |'
                  % (tok, d['epochs'], d['total']))
            continue
        print('| %s | %.4f | %.4f | %.4f | %.4f |'
              % (tok, v['recall@5'], v['ndcg@5'], v['recall@10'], v['ndcg@10']))
    print()


table('val', '### 验证集 —— 取 val NDCG@10 最高的那个 epoch（四项同源）')
table('test', '### 测试集 —— 同一个最优 checkpoint')

print('### 训练进度')
print()
print('| arm | epochs | best epoch | 状态 |')
print('|---|---|---|---|')
for label, _ in ARMS:
    d = data[label]
    st = '完成' if d['finished'] else ('训练中' if d['epochs'] else '未开始')
    print('| %s | %d/%d | %s | %s |'
          % (label, d['epochs'], d['total'], d.get('best_epoch', '-'), st))
