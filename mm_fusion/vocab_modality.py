"""Is the ActionPiece vocabulary genuinely multimodal?

In a saved vocabulary, `token2feat` holds
  * the first `n_init_feats` entries as [slot, value] (initial tokens), and
  * every later entry as the merge rule [-1, child_a, child_b].

So the slots a merged token covers must be obtained by expanding that binary
merge tree.  Slot layout for a late-fused arm:

    slots 0 .. n-1     text semantic codes
    slots n .. 2n-1    image semantic codes
    slot  2n           random hash bucket

If merges mostly stay inside one modality, the "fusion" is really two separate
token streams glued together.
"""
import json
import os
import sys
from collections import Counter

from functools import lru_cache

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

sys.setrecursionlimit(200000)


def build_slot_sets(t2f):
    """Iteratively expand every merge rule into the set of slots it covers."""
    out = [None] * len(t2f)
    memo = {}
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
            # merge rule [-1, a, b]
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


print('%-9s %-7s %-10s %-11s %-11s %-12s %s'
      % ('arm', 'tokens', 'text only', 'image only', 'text+image',
         'incl. hash', 'mean slots'))
print('-' * 80)

for label, n, fn in ARMS:
    p = os.path.join(B, fn)
    if not os.path.exists(p):
        print('%-9s MISSING' % label)
        continue
    meta = json.load(open(p))
    t2f = meta['token2feat']
    slots = build_slot_sets(t2f)

    text_slots = set(range(n))
    image_slots = set(range(n, 2 * n))
    hash_slot = 2 * n

    c = Counter()
    tot_slots = 0
    with_hash = 0
    for s in slots:
        tot_slots += len(s)
        with_hash += int(hash_slot in s)
        has_t, has_i = bool(s & text_slots), bool(s & image_slots)
        if has_t and has_i:
            c['both'] += 1
        elif has_t:
            c['text'] += 1
        elif has_i:
            c['image'] += 1
        elif hash_slot in s:
            c['hash'] += 1
        else:
            c['other'] += 1

    tot = len(t2f)
    print('%-9s %-7d %-10s %-11s %-11s %-12s %.2f'
          % (label, tot,
             '%.1f%%' % (100 * c['text'] / tot),
             '%.1f%%' % (100 * c['image'] / tot),
             '%.1f%%' % (100 * c['both'] / tot),
             '%.1f%%' % (100 * with_hash / tot),
             tot_slots / tot))

print()
print('The first %d entries are the initial single-slot tokens, so a large')
print('"text+image" share among the merged ones means the vocabulary really')
print('fuses the modalities.')
