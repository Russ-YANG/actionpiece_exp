"""Usage-weighted version of the hash-pollution diagnostic.

Token-type shares treat every vocabulary entry equally, but the merge `priority`
records how often each merged pair occurred, so weighting by it approximates how
much a token is actually used.  This checks whether the hash-pollution pattern
survives that weighting.
"""
import json
import os

B = ('/home/lzr/science/actionpiece_exp/cache/AmazonReviews2014/Beauty/'
     'processed')

ARMS = [
    ('mm 2+2', 2, 'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.'
                  't2x256.i2x256.seed2024.h128.v40000.json'),
    ('mm 3+3', 3, 'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.'
                  't3x256.i3x256.seed2024.h128.v40000.json'),
    ('mm 4+4', 4, 'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.'
                  't4x256.i4x256.seed2024.h128.v40000.json'),
]


def slot_sets(t2f):
    out = [None] * len(t2f)
    for start in range(len(t2f)):
        if out[start] is not None:
            continue
        stack = [start]
        while stack:
            i = stack[-1]
            if out[i] is not None:
                stack.pop()
                continue
            e = t2f[i]
            if len(e) == 2:
                out[i] = frozenset((e[0],))
                stack.pop()
                continue
            a, b = e[1], e[2]
            if out[a] is None:
                stack.append(a)
                continue
            if out[b] is None:
                stack.append(b)
                continue
            out[i] = out[a] | out[b]
            stack.pop()
    return out


print('%-9s %-14s %-14s %-14s %s'
      % ('arm', 'type: hash', 'weighted: hash', 'type: text+img',
         'weighted: text+img'))
print('-' * 74)

for label, n, fn in ARMS:
    p = os.path.join(B, fn)
    if not os.path.exists(p):
        print('%-9s MISSING' % label)
        continue
    meta = json.load(open(p))
    t2f = meta['token2feat']
    prio = meta['priority']
    slots = slot_sets(t2f)

    text_slots = set(range(n))
    image_slots = set(range(n, 2 * n))
    hash_slot = 2 * n

    # merged tokens only: the initial ones are never "polluted"
    n_hash_t = n_both_t = n_merged_t = 0
    w_hash = w_both = w_all = 0.0
    for i in range(len(t2f)):
        if len(t2f[i]) != 3:          # initial token
            continue
        n_merged_t += 1
        w = float(prio[i])
        w_all += w
        s = slots[i]
        if hash_slot in s:
            n_hash_t += 1
            w_hash += w
        if (s & text_slots) and (s & image_slots):
            n_both_t += 1
            w_both += w

    print('%-9s %-14s %-14s %-14s %s'
          % (label,
             '%.1f%%' % (100 * n_hash_t / n_merged_t),
             '%.1f%%' % (100 * w_hash / w_all),
             '%.1f%%' % (100 * n_both_t / n_merged_t),
             '%.1f%%' % (100 * w_both / w_all)))

print()
print('"type" counts merged vocabulary entries; "weighted" weights each entry by')
print('its merge priority (pair frequency), i.e. roughly by usage.')
