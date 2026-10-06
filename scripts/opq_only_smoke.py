#!/usr/bin/env python3
"""Synthetic CPU pipeline smoke; no Qwen execution or research-quality claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from opq_only import run


def fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    n, d = 10242, 16  # 10240 fitting points >= Faiss recommendation 39*256.
    rng = np.random.RandomState(73)
    x = rng.normal(size=(n, d)).astype('<f4')
    x[-1] = x[0]  # A deliberate catalog collision; not dropped.
    x.tofile(root / 'embedding.f32')
    items = [f'synthetic-{i:05d}' for i in rng.permutation(n)]
    ids = ['[PAD]'] + items
    meta = {'backend': 'qwen_local', 'model_id': 'Qwen/Qwen3-VL-Embedding-8B',
            'model_revision': 'synthetic-test-only', 'code_revision': 'synthetic-test-only',
            'instruction_sha256': 'synthetic-test-only', 'normalization': 'none_synthetic_fixture',
            'dimension': d, 'native_dimension': d, 'item_count': n,
            'item_order_sha256': hashlib.sha256('\n'.join(items).encode()).hexdigest(),
            'test_fixture': True, 'warning': 'Random vectors, NOT Qwen embeddings or real catalog evidence'}
    config = {'embedding_path': str(root / 'embedding.f32'),
              'embedding_manifest_path': str(root / 'embedding.manifest.json'),
              'id_mapping_path': str(root / 'id_mapping.json'), 'item_seqs_path': str(root / 'all_item_seqs.json'),
              'dimension': d, 'n_codebooks': 4, 'codebook_size': 256, 'seed': 2024, 'threads': 1,
              'split_rule': 'leave_one_out_sequence_minus_last_two', 'encoder': meta['model_id'],
              'preprocessing': meta['normalization'], 'output_dir': str(root / 'bundle')}
    for name, data in [('id_mapping.json', {'id2item': ids, 'item2id': {v: i for i, v in enumerate(ids)}}),
                       ('all_item_seqs.json', {'synthetic_user': items}),
                       ('embedding.manifest.json', meta), ('config.json', config)]:
        (root / name).write_text(json.dumps(data))
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', required=True, help='New directory; no overwrite')
    args = parser.parse_args()
    c = fixture(args.output_root)
    run(c, 'fit')
    run(c, 'validate')
