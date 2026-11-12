#!/usr/bin/env python3
"""Compact check of the generated SID tensor."""
import numpy as np
import torch

SID = ('/home/lzr/science/GRID/logs/inference/runs/2026-11-12/10-59-32/'
       'pickle/merged_predictions_tensor.pt')
t = torch.load(SID, map_location='cpu', weights_only=False)

print('type :', type(t).__name__)
if isinstance(t, torch.Tensor):
    print('shape:', tuple(t.shape), t.dtype)
    a = t.numpy()
elif isinstance(t, (list, tuple)):
    print('len  :', len(t), '| element types:', {type(x).__name__ for x in t})
    a = np.stack([np.asarray(x) for x in t], axis=-1)
    print('stacked:', a.shape)
else:
    a = np.asarray(t)
    print('asarray:', a.shape)

print('dtype:', a.dtype, '| min', int(a.min()), '| max', int(a.max()))
if a.ndim == 2:
    n_rows, n_cols = a.shape
    uniq = len({tuple(r) for r in a.tolist()})
    print(f'rows={n_rows}  hierarchies={n_cols}  distinct SIDs={uniq} '
          f'({uniq/n_rows:.3%})')
    for j in range(n_cols):
        print(f'  layer {j}: {len(set(a[:, j].tolist()))} distinct codes '
              f'(max {int(a[:, j].max())})')
    print('  sample SIDs:', a[:3].tolist())
else:
    print('unexpected ndim', a.ndim)
