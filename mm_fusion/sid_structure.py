"""Understand the structure of the cached 2-slot semantic IDs before building 3+3."""
import json
from collections import Counter

import numpy as np

AP = '/home/lzr/science/actionpiece_exp'
B = 'cache/AmazonReviews2014/Beauty/processed'
id2item = json.load(open(f'{AP}/{B}/id_mapping.json'))['id2item']
n = len(id2item) - 1

sets = {
    'TEXT  opq2x65536.freqmax_pack2': (
        'qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.r2c4565515e0f.'
        'c393e2978d278.m2048.tbfloat16.asdpa.opq2x65536.seed2024.freqmax_pack2.sem_ids'),
    'IMAGE opq2x65536.freqmax_pack2': (
        'qwen_local.Qwen3-VL-Embedding-8B.d768.ib6b0d724b414.r2c4565515e0f.'
        'c393e2978d278.m2048.tbfloat16.asdpa.image.im2f0852d8f8d9.'
        'opq2x65536.seed2024.freqmax_pack2.sem_ids'),
    'E1    text4096 opq4x256': (
        'beauty_e1_text4096_raw.sem_ids'),
    'E3    joint4096 opq4x256': (
        'qwen_local.Qwen3-VL-Embedding-8B.d4096.i711daf44f948.r2c4565515e0f.'
        'c393e2978d278.m2048.tbfloat16.asdpa.joint.im2f0852d8f8d9.'
        'opq4x256.seed2024.sem_ids'),
}

for label, fn in sets.items():
    d = json.load(open(f'{AP}/{B}/{fn}'))
    arr = np.array([d[id2item[i + 1]] for i in range(n)], dtype=np.int64)
    print('=' * 76)
    print('%s   codes/item=%d' % (label, arr.shape[1]))
    for slot in range(arr.shape[1]):
        col = arr[:, slot]
        c = Counter(col.tolist())
        top = c.most_common(5)
        print('  slot %d: distinct=%-6d  max=%-5d  min=%-5d  top5=%s'
              % (slot, len(c), col.max(), col.min(), top))
    # how concentrated is the joint distribution
    joint = Counter(map(tuple, arr.tolist()))
    print('  full SID: distinct=%d / %d   largest cell=%d'
          % (len(joint), n, joint.most_common(1)[0][1]))

print()
print('=' * 76)
print('interpretation aids')
print('  a 65536-entry PQ codebook over 12101 items gives at most 12101 codes;')
print('  if a slot shows far fewer distinct values it has been remapped')
print('  (that is what "freqmax" plausibly does); if a slot is nearly constant')
print('  it may be a packed/aggregated slot.')
