"""Measure what ActionPiece actually tokenises our arms into.

Two different counts share the word "token" in this pipeline:

  1. OPQ semantic codes -- the input features (text codes + image codes) plus
     one hashed bucket.  This is `n_categories`, i.e. what the table's
     "OPQ 分词数" column describes.
  2. ActionPiece vocabulary tokens -- what the model consumes after the
     BPE-like merge inside `ActionPieceCore.encode`.

Both are reported here, measured on the real Beauty item sequences.

`genrec.models.__init__` pulls in the model stack and trips a circular import,
so `core.py` and `utils.py` are loaded directly by file path with stub parent
packages that are never executed.
"""
import importlib.util
import json
import os
import sys
import types

import numpy as np

AP_DIR = '/home/lzr/science/actionpiece_exp/genrec/models/ActionPiece'
for _name in ('genrec', 'genrec.models', 'genrec.models.ActionPiece'):
    sys.modules.setdefault(_name, types.ModuleType(_name))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load('genrec.models.ActionPiece.utils', f'{AP_DIR}/utils.py')
_core = _load('genrec.models.ActionPiece.core', f'{AP_DIR}/core.py')
ActionPieceCore = _core.ActionPieceCore

B = ('/home/lzr/science/actionpiece_exp/cache/AmazonReviews2014/Beauty/'
     'processed')
MAX_ITEM_SEQ_LEN = 20
N_USERS = 400

ARMS = [
    ('2+2', 'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.'
            't2x256.i2x256.seed2024.h128.v40000.json',
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t2x256.i2x256.seed2024.h128.feat'),
    ('3+3', 'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.'
            't3x256.i3x256.seed2024.h128.v40000.json',
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t3x256.i3x256.seed2024.h128.feat'),
    ('4+4', 'actionpiece.qwen_separate.Qwen3-VL-Embedding-8B.'
            't4x256.i4x256.seed2024.h128.v40000.json',
     'item.qwen_separate.Qwen3-VL-Embedding-8B.t4x256.i4x256.seed2024.h128.feat'),
]

seqs = json.load(open(os.path.join(B, 'all_item_seqs.json')))
users = list(seqs)[:N_USERS]

print('%-6s %-11s %-13s %-14s %-13s %-8s %s'
      % ('arm', 'categories', 'codes/item', 'tok/seq before', 'tok/seq after',
         'ratio', 'max_len'))
print('-' * 88)

for label, vocab_fn, feat_fn in ARMS:
    vocab = ActionPieceCore.from_pretrained(
        os.path.join(B, vocab_fn), vocab_size=40000)
    raw = json.load(open(os.path.join(B, feat_fn)))
    item2feat = {str(k): tuple(v) for k, v in raw.items()}

    before, after, n_items, n_seq = 0, 0, 0, 0
    for u in users:
        items = [str(i) for i in seqs[u]][-(MAX_ITEM_SEQ_LEN + 1):]
        items = [i for i in items if i in item2feat]
        if len(items) < 2:
            continue
        state_seq = np.array([item2feat[i] for i in items])
        b = sum(len(f) for f in state_seq)
        a = len(vocab.encode(state_seq, shuffle='none'))
        before += b
        after += a
        n_items += len(items)
        n_seq += 1

    max_len = vocab.n_categories * MAX_ITEM_SEQ_LEN + 2
    print('%-6s %-11d %-13.2f %-14.2f %-13.2f %-8.2f %d'
          % (label, vocab.n_categories, before / n_items, before / n_seq,
             after / n_seq, after / before, max_len))

print()
print('codes/item      = OPQ semantic codes + 1 hashed bucket = n_categories')
print('tok/seq before  = those codes on one user sequence, before merging')
print('tok/seq after   = ActionPiece merged tokens for that sequence')
print('max_len         = n_categories * max_item_seq_len + 2 (BOS/EOS)')
print('sampled %d users, at most %d items each' % (N_USERS, MAX_ITEM_SEQ_LEN))
