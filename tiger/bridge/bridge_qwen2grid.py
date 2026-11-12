#!/usr/bin/env python3
"""Bridge: Qwen Beauty .sent_emb (RPG item order) -> GRID pickle (GRID item order)."""
import os

import numpy as np
import torch

PERM = np.load('/mnt/m/science/tmp/rpg2grid_perm.npy')
SRC_DIR = '/home/lzr/science/RPG_KDD2025/cache/AmazonReviews2014/Beauty/processed'
OUT_ROOT = '/home/lzr/science/GRID/qwen_emb'

N = 12101
assert PERM.shape == (N,)
assert len(set(PERM.tolist())) == N

for dim in (128, 768, 1536):
    src = os.path.join(SRC_DIR, f'Text-trunc-{dim}.sent_emb')
    raw = np.fromfile(src, dtype=np.float32).reshape(-1, dim)
    assert raw.shape == (N, dim), raw.shape
    reordered = raw[PERM]                      # RPG order -> GRID order
    out_dir = os.path.join(OUT_ROOT, f'd{dim}', 'pickle')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'merged_predictions_tensor.pt')
    torch.save(torch.from_numpy(np.ascontiguousarray(reordered)), out)
    t = torch.load(out, map_location='cpu', weights_only=False)
    print(f'  d{dim:<5} {src.split("/")[-1]}')
    print(f'         -> {out}')
    print(f'         tensor {tuple(t.shape)} {t.dtype}  '
          f'|row0| = {t[0].norm().item():.6f}  '
          f'finite = {bool(torch.isfinite(t).all())}')
