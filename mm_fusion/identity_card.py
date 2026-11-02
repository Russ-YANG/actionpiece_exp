"""How many distinct item identities does each representation actually provide?

The semantic-ID cardinalities of each modality are not the whole story: what
matters for memorisation is the number of distinct *full* item feature tuples
(semantic codes + hash bucket) that the model sees.
"""
import json
import os

B = ('/home/lzr/science/actionpiece_exp/cache/AmazonReviews2014/Beauty/'
     'processed')

CASES = [
    ('mm 2+2 (new, 4 slots)',
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.seed2024.h128.feat'),
    ('mm 3+3 (new, 6 slots)',
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t3x256.i3x256.seed2024.h128.feat'),
    ('mm 4+4 (new, 8 slots)',
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t4x256.i4x256.seed2024.h128.feat'),
    ('text only 4x256',
     'item.qwen_text.qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.'
     'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.opq4x256.seed2024.'
     'h128.feat'),
    ('image only 4x256',
     'item.qwen_image.qwen_local.Qwen3-VL-Embedding-8B.d768.ib6b0d724b414.'
     'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.image.'
     'im2f0852d8f8d9.opq4x256.seed2024.h128.feat'),
    ('joint 4096 4x256',
     'item.qwen_multimodal.qwen_local.Qwen3-VL-Embedding-8B.d4096.'
     'i711daf44f948.r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.'
     'joint.im2f0852d8f8d9.opq4x256.seed2024.h128.feat'),
    ('sep 2x65536 old',
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t768.th235523e6f242.i768.'
     'ihce07c0028c77.topq2x65536.iopq2x65536.seed2024.h128.feat'),
]

print('%-26s %-8s %-12s %-12s %s'
      % ('arm', 'width', 'items', 'distinct', 'mean items per identity'))
print('-' * 84)
for label, fn in CASES:
    p = os.path.join(B, fn)
    if not os.path.exists(p):
        print('%-26s MISSING' % label)
        continue
    d = json.load(open(p))
    widths = {len(v) for v in d.values()}
    ident = {tuple(v) for v in d.values()}
    print('%-26s %-8s %-12d %-12d %.2f'
          % (label, '/'.join(map(str, sorted(widths))), len(d), len(ident),
             len(d) / len(ident)))
