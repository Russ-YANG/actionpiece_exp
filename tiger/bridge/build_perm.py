#!/usr/bin/env python3
"""Verify the Beauty item sets match and build the RPG->GRID permutation."""
import json

import numpy as np

GRID_MAP = '/home/lzr/science/GRID/data/amazon_data/beauty/mappings/item_id_map.tsv'
RPG_MAP = ('/home/lzr/science/RPG_KDD2025/input/Qwen3VL-Beauty-Raw-Embeddings/'
           'Item-id-mapping.json')

grid = {}
with open(GRID_MAP) as fh:
    next(fh)
    for line in fh:
        a, b = line.rstrip('\n').split('\t')[:2]
        grid[int(a)] = b
print(f'GRID: {len(grid)} items, index range {min(grid)}..{max(grid)}')

d = json.load(open(RPG_MAP))
id2item = d['id2item']
print(f'RPG id2item type = {type(id2item).__name__}, len = {len(id2item)}')
print(f'  [0] = {id2item[0]!r}   [1] = {id2item[1]!r}   [-1] = {id2item[-1]!r}')

if isinstance(id2item, list):
    # 1-based: index 0 is the padding slot
    rpg = {i: id2item[i + 1] for i in range(len(id2item) - 1)}
elif isinstance(id2item, dict):
    rpg = {int(k) - 1: v for k, v in id2item.items() if int(k) != 0}
else:
    raise TypeError(type(id2item))
print(f'RPG : {len(rpg)} items, index range {min(rpg)}..{max(rpg)}')

gs, rs = set(grid.values()), set(rpg.values())
print(f'\nsets equal? {gs == rs}')
print(f'  only in GRID: {len(gs - rs)}   only in RPG: {len(rs - gs)}')
if gs - rs:
    print('   e.g.', sorted(gs - rs)[:5])
if rs - gs:
    print('   e.g.', sorted(rs - gs)[:5])

if gs == rs:
    asin2rpg = {v: k for k, v in rpg.items()}
    perm = np.array([asin2rpg[grid[i]] for i in range(len(grid))], dtype=np.int64)
    print(f'\npermutation: len={len(perm)} '
          f'bijection={len(set(perm.tolist())) == len(perm)}')
    print(f'  GRID row0 asin={grid[0]} -> RPG row{perm[0]}')
    print(f'  GRID row1 asin={grid[1]} -> RPG row{perm[1]}')
    print(f'  identity? {np.array_equal(perm, np.arange(len(perm)))}')
    np.save('/mnt/m/science/tmp/rpg2grid_perm.npy', perm)
    print('  saved /mnt/m/science/tmp/rpg2grid_perm.npy')
