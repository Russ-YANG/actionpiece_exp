"""Sharp provenance test for the Beauty IMAGE vectors.

If RPG's Image-emb-768 really is the matrix the cached image semantic IDs were
built from, the 24 items whose image vector is exactly zero must all land in the
same quantisation cell (a zero vector is degenerate), and they must be mutual
nearest neighbours.  Genuine images in the same set will not collapse like that.
"""
import json
from collections import Counter

import numpy as np

AP = '/home/lzr/science/actionpiece_exp'
B = 'cache/AmazonReviews2014/Beauty/processed'
RAW = '/home/lzr/science/RPG_KDD2025/input/Qwen3VL-Beauty-Raw-Embeddings'

id2item = json.load(open(f'{AP}/{B}/id_mapping.json'))['id2item']
n = len(id2item) - 1

img = np.fromfile(f'{RAW}/Image-emb-768.image_emb', dtype=np.float32).reshape(n, 768)
txt = np.fromfile(f'{RAW}/Text-emb-768.sent_emb', dtype=np.float32).reshape(n, 768)

i_ids = json.load(open(
    f'{AP}/{B}/qwen_local.Qwen3-VL-Embedding-8B.d768.ib6b0d724b414.'
    'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.image.'
    'im2f0852d8f8d9.opq2x65536.seed2024.freqmax_pack2.sem_ids'))
t_ids = json.load(open(
    f'{AP}/{B}/qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.'
    'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.'
    'opq2x65536.seed2024.freqmax_pack2.sem_ids'))

img_sid = np.array([i_ids[id2item[i + 1]] for i in range(n)], dtype=np.int64)
txt_sid = np.array([t_ids[id2item[i + 1]] for i in range(n)], dtype=np.int64)

norms = np.linalg.norm(img, axis=1)
zero = np.flatnonzero(norms < 1e-8)
print('items with an exactly-zero image vector: %d' % len(zero))

print()
print('--- image semantic IDs of those zero-vector items ---')
for i in zero:
    print('   row %-6d %-12s image_sid=%s   text_sid=%s'
          % (i, id2item[i + 1], tuple(img_sid[i]), tuple(txt_sid[i])))

u = {tuple(img_sid[i]) for i in zero}
print()
print('distinct image SIDs among the zero-vector items : %d / %d' % (len(u), len(zero)))
print('distinct text  SIDs among the same items        : %d / %d'
      % (len({tuple(txt_sid[i]) for i in zero}), len(zero)))

print()
print('--- how often does a whole image SID repeat across the dataset? ---')
c = Counter(map(tuple, img_sid.tolist()))
sizes = sorted(c.values(), reverse=True)
print('  distinct image SIDs: %d for %d items' % (len(c), n))
print('  largest cell sizes : %s' % sizes[:8])
print('  items in the single largest cell: %d' % sizes[0])

print()
print('--- are the zero-vector rows mutual nearest neighbours in RPG image space? ---')
z = img[zero]
sim = z @ z.T
np.fill_diagonal(sim, -2)
nn1 = sim.argmax(axis=1)
print('  each zero row\'s nearest neighbour is another zero row: %s'
      % bool(np.isin(nn1, np.arange(len(zero))).all()))
print('  pairwise cosine of zero rows (min/max): %.4f / %.4f'
      % (sim[sim > -2].min(), sim[sim > -2].max()))

print()
print('--- control: do 24 random NON-zero image rows collapse to one SID? ---')
rng = np.random.default_rng(0)
for trial in range(5):
    pick = rng.choice(np.setdiff1d(np.arange(n), zero), size=24, replace=False)
    print('   trial %d: distinct image SIDs = %d / 24'
          % (trial, len({tuple(img_sid[i]) for i in pick})))
