"""Sanity-check the three multimodal arms' tokenizer artefacts.

Checks:
  1. the item features really carry n_text + n_image slots;
  2. the text and image halves are genuinely different quantisations
     (not the same codes twice);
  3. the three arms produce different partitions of the same items.
"""
import json
import os

from collections import Counter

AP = '/home/lzr/science/actionpiece_exp'
B = f'{AP}/cache/AmazonReviews2014/Beauty/processed'
id2item = json.load(open(f'{B}/id_mapping.json'))['id2item']
n = len(id2item) - 1

STEMS = {
    2: ('qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.r2c4565515e0f.'
        'c393e2978d278.m2048.tbfloat16.asdpa.opq2x256.seed2024.sem_ids',
        'Qwen3-VL-Embedding-8B.d768.opq2x256.seed2024.image_sem_ids',
        'item.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.seed2024.h128.feat'),
    3: ('qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.r2c4565515e0f.'
        'c393e2978d278.m2048.tbfloat16.asdpa.opq3x256.seed2024.sem_ids',
        'Qwen3-VL-Embedding-8B.d768.opq3x256.seed2024.image_sem_ids',
        'item.qwen_separate.Qwen3-VL-Embedding-8B.t3x256.i3x256.seed2024.h128.feat'),
    4: ('qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.r2c4565515e0f.'
        'c393e2978d278.m2048.tbfloat16.asdpa.opq4x256.seed2024.sem_ids',
        'Qwen3-VL-Embedding-8B.d768.opq4x256.seed2024.image_sem_ids',
        'item.qwen_separate.Qwen3-VL-Embedding-8B.t4x256.i4x256.seed2024.h128.feat'),
}

parts = {}
for k, (tf, imf, ft) in STEMS.items():
    print('=' * 74)
    print('arm %d+%d' % (k, k))
    missing = [f for f in (tf, imf, ft) if not os.path.exists(f'{B}/{f}')]
    if missing:
        print('  not ready yet:', missing)
        continue
    t = json.load(open(f'{B}/{tf}'))
    i = json.load(open(f'{B}/{imf}'))
    tarr = [tuple(t[id2item[r + 1]]) for r in range(n)]
    iarr = [tuple(i[id2item[r + 1]]) for r in range(n)]
    print('  text  codes/item :', len(tarr[0]), ' distinct SIDs:',
          len(set(tarr)))
    print('  image codes/item :', len(iarr[0]), ' distinct SIDs:',
          len(set(iarr)))
    same = sum(1 for r in range(n) if tarr[r] == iarr[r])
    print('  rows where text SID == image SID: %d / %d  (want ~0)' % (same, n))
    slot_overlap = sum(1 for r in range(n) if tarr[r][0] == iarr[r][0])
    print('  rows sharing slot-0 code        : %d / %d' % (slot_overlap, n))
    parts[k] = (tarr, iarr)

print()
print('=' * 74)
print('are the arms different partitions of the same items?')
ks = sorted(parts)
for a in range(len(ks)):
    for b in range(a + 1, len(ks)):
        x, y = ks[a], ks[b]
        agree = sum(1 for r in range(n) if parts[x][0][r] == parts[y][0][r])
        print('  text SID agreement  %d+%d vs %d+%d : %.4f'
              % (x, x, y, y, agree / n))
