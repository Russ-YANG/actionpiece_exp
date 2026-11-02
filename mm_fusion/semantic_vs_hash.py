"""Split each item identity into its semantic part and its random hash digit.

Every arm ends up with 12,101 unique identities because `_get_hashed_feat`
appends a collision-indexed random bucket.  What differs between arms is how
much of that identity is carried by the *semantic* codes and how many items fall
back on the random digit.
"""
import json
import os
from collections import Counter

B = ('/home/lzr/science/actionpiece_exp/cache/AmazonReviews2014/Beauty/'
     'processed')

CASES = [
    ('mm 2+2 (new)', 'item.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.seed2024.h128.feat'),
    ('mm 3+3 (new)', 'item.qwen_separate.Qwen3-VL-Embedding-8B.t3x256.i3x256.seed2024.h128.feat'),
    ('mm 4+4 (new)', 'item.qwen_separate.Qwen3-VL-Embedding-8B.t4x256.i4x256.seed2024.h128.feat'),
    ('text only 4x256', 'item.qwen_text.qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.opq4x256.seed2024.h128.feat'),
    ('image only 4x256', 'item.qwen_image.qwen_local.Qwen3-VL-Embedding-8B.d768.ib6b0d724b414.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.image.im2f0852d8f8d9.opq4x256.seed2024.h128.feat'),
    ('joint 4096 4x256', 'item.qwen_multimodal.qwen_local.Qwen3-VL-Embedding-8B.d4096.i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.joint.im2f0852d8f8d9.opq4x256.seed2024.h128.feat'),
    ('sep 2x65536 old', 'item.qwen_separate.Qwen3-VL-Embedding-8B.t768.th235523e6f242.i768.ihce07c0028c77.topq2x65536.iopq2x65536.seed2024.h128.feat'),
]

print('%-20s %-7s %-11s %-11s %-13s %s'
      % ('arm', 'width', 'distinct', 'distinct', 'share of items', 'largest'))
print('%-20s %-7s %-11s %-11s %-13s %s'
      % ('', '', 'full id', 'semantic', 'needing hash', 'semantic group'))
print('-' * 86)
for label, fn in CASES:
    p = os.path.join(B, fn)
    if not os.path.exists(p):
        print('%-20s MISSING' % label)
        continue
    d = json.load(open(p))
    feats = [tuple(v) for v in d.values()]
    w = len(feats[0])
    full = len(set(feats))
    sem = [f[:w - 1] for f in feats]          # drop the hash digit
    sem_set = set(sem)
    cnt = Counter(sem)
    share = 1.0 - (len(sem_set) / len(feats))  # items sharing a semantic tuple
    print('%-20s %-7d %-11d %-11d %-13.3f %d'
          % (label, w, full, len(sem_set), share, max(cnt.values())))
print()
print('"needing hash" = fraction of items whose semantic tuple is shared with')
print('another item, i.e. only distinguishable by the random hash digit.')
