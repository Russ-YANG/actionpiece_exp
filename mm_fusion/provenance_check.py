"""Provenance check: are the RPG Beauty vectors the encoder behind the
ActionPiece Beauty semantic IDs?

Idea: the cached 2-code semantic IDs are k-means cluster assignments.  If a
candidate embedding matrix really is the one those IDs were built from, then
items whose vectors are mutual nearest neighbours must fall in the same cluster
far more often than chance.  A different encoder (control) will not show that.

Positive control: RPG Text-emb-4096 (same text encoder family, higher dim).
Negative control: sentence-t5-base 768 (a genuinely different encoder).
"""
import json

import numpy as np
import torch

AP = '/home/lzr/science/actionpiece_exp'
RPG = '/home/lzr/science/RPG_KDD2025'
PR6 = '/home/lzr/science/action_piece-pr-6'
B = 'cache/AmazonReviews2014/Beauty/processed'
RAW = RPG + '/input/Qwen3VL-Beauty-Raw-Embeddings'

TEXT_SIDS = (f'{AP}/{B}/qwen_local.Qwen3-VL-Embedding-8B.d768.i711daf44f948.'
             'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.'
             'opq2x65536.seed2024.freqmax_pack2.sem_ids')
IMAGE_SIDS = (f'{AP}/{B}/qwen_local.Qwen3-VL-Embedding-8B.d768.ib6b0d724b414.'
              'r2c4565515e0f.c393e2978d278.m2048.tbfloat16.asdpa.image.'
              'im2f0852d8f8d9.opq2x65536.seed2024.freqmax_pack2.sem_ids')

# ---- item order -----------------------------------------------------------
ap_map = json.load(open(f'{AP}/{B}/id_mapping.json'))
rpg_map = json.load(open(f'{RAW}/Item-id-mapping.json'))
assert ap_map['id2item'] == rpg_map['id2item'], 'item orders differ'
id2item = ap_map['id2item']
n_items = len(id2item) - 1
print('items: %d (PAD excluded)   id2item identical between projects: OK'
      % n_items)


def load_ids(path):
    d = json.load(open(path))
    arr = np.array([d[id2item[i + 1]] for i in range(n_items)], dtype=np.int64)
    print('  %-58s codes/item=%d  distinct code0=%d'
          % (path.split('/')[-1][:58], arr.shape[1], len(np.unique(arr[:, 0]))))
    return arr


print()
print('semantic ID sets:')
t_ids = load_ids(TEXT_SIDS)
i_ids = load_ids(IMAGE_SIDS)

# ---- candidate embeddings -------------------------------------------------
cands = {
    'RPG Qwen text-768': (f'{RAW}/Text-emb-768.sent_emb', 768),
    'RPG Qwen image-768': (f'{RAW}/Image-emb-768.image_emb', 768),
    'RPG text-4096 (pos ctrl)': (f'{RAW}/Text-emb-4096.sent_emb', 4096),
    'sentence-t5-768 (neg ctrl)': (f'{PR6}/{B}/sentence-t5-base-modelscope.sent_emb', 768),
}

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print('\ndevice:', dev)

RNG = np.random.default_rng(0)
probe = RNG.choice(n_items, size=3000, replace=False)

print()
print('%-28s %-10s %-12s %-12s' % ('embedding', 'code0 match', 'full match', 'random base'))
print('-' * 70)

results = {}
for name, (path, dim) in cands.items():
    a = np.fromfile(path, dtype=np.float32).reshape(-1, dim)
    assert a.shape[0] == n_items, (name, a.shape)
    t = torch.from_numpy(a).to(dev)
    t = torch.nn.functional.normalize(t, dim=-1)
    # top-5 nearest neighbours excluding self, for the probe rows
    sub = t[torch.from_numpy(probe).to(dev)]
    sim = sub @ t.T                       # 3000 x n_items
    sim[torch.arange(len(probe)), torch.from_numpy(probe).to(dev)] = -2.0
    nn = sim.topk(5, dim=1).indices.cpu().numpy()
    nb = nn.reshape(-1)
    src = np.repeat(probe, 5)

    code0 = (t_ids[src, 0] == t_ids[nb, 0]).mean()
    full = ((t_ids[src] == t_ids[nb]).all(axis=1)).mean()
    # chance level: probability two random items share code0
    c0 = t_ids[:, 0]
    _, counts = np.unique(c0, return_counts=True)
    base = float((counts / len(c0)).__pow__(2).sum())
    results[name] = (code0, full, base)
    print('%-28s %-10.4f %-12.4f %-12.4f' % (name, code0, full, base))

print()
print('interpretation: a candidate that really is the source encoder should show')
print('a code0 match rate far above the random baseline; controls should not.')
