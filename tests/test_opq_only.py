import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import faiss
import numpy as np
import opq_only as opq
from opq_only_smoke import fixture


class OPQWorkflowTest(unittest.TestCase):
    def test_feature_identity_and_collisions(self):
        self.assertEqual(opq.restore_tokens([861, 524, 12, 263], 4), [12, 7, 12, 93])
        for tokens in ([12, 7, 12, 93], [12, 263, 524], [12, 263, 524, 1024]):
            with self.assertRaises(ValueError):
                opq.restore_tokens(tokens, 4)
        reverse, stats = opq.sid_maps({'a': [12, 7], 'b': [12, 7], 'c': [7, 12]})
        self.assertEqual(reverse['12,7'], ['a', 'b'])
        self.assertEqual(stats['duplicate_item_count'], 1)

    def test_input_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = fixture(Path(tmp) / 'fixture')
            for key, value in [('codebook_size', 16), ('dimension', 15), ('threads', 0),
                               ('split_rule', 'random')]:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    opq.load_inputs(c | {key: value}, fitting=True)
            sequences_path = Path(c['item_seqs_path'])
            original = sequences_path.read_text()
            sequences_path.write_text(json.dumps({'u': ['unknown-heldout-item']}))
            with self.assertRaisesRegex(ValueError, 'unknown item'):
                opq.load_inputs(c, fitting=True)
            ids = json.loads(Path(c['id_mapping_path']).read_text())['id2item'][1:]
            sequences_path.write_text(json.dumps({'u': ids[:10]}))
            with self.assertRaisesRegex(ValueError, 'at least 256'):
                opq.load_inputs(c, fitting=True)
            sequences_path.write_text(original)
            data_x = np.memmap(c['embedding_path'], mode='r+', dtype='<f4')
            data_x[0] = np.nan
            data_x.flush()
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
                opq.load_inputs(c)
            data_x[0] = 0.0
            data_x.flush()
            del data_x
            p = Path(c['id_mapping_path'])
            data = json.loads(p.read_text())
            data['id2item'][1], data['id2item'][2] = data['id2item'][2], data['id2item'][1]
            data['item2id'] = {v: i for i, v in enumerate(data['id2item'])}
            p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'order'):
                opq.load_inputs(c)

    def test_end_to_end_adequate_cluster_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(os.environ.get('OPQ_TEST_OUTPUT', str(Path(tmp) / 'fixture')))
            c = fixture(root)
            opq.run(c, 'fit')
            opq.run(c, 'validate')
            # Complete same-config bundles are validated and reused without another fit.
            from unittest.mock import patch
            with patch.object(opq, 'create_quantizer', side_effect=AssertionError('must reuse')):
                opq.run(c, 'fit')
            items, x, meta, rows = opq.load_inputs(c, fitting=True)
            self.assertEqual(len(rows), 10240)
            self.assertNotIn(items[-1], json.loads((root / 'bundle/fit_items.json').read_text()))
            stats = json.loads((root / 'bundle/collisions.json').read_text())
            self.assertTrue(any(items[0] in group and items[-1] in group for group in stats['groups'].values()))
            index = faiss.read_index(str(root / 'bundle/quantizer.faiss'))
            expected = opq.raw_codes(index, x, 4)
            # Check sa_encode equals historical inverted-list codes, using shuffled explicit IDs.
            order = np.random.RandomState(81).permutation(len(x))
            index.add_with_ids(np.ascontiguousarray(x[order]), order.astype(np.int64))
            ivf = faiss.downcast_index(index.index)
            lists = ivf.invlists
            n = lists.list_size(0)
            codes = faiss.rev_swig_ptr(lists.get_codes(0), n * lists.code_size).copy().reshape(n, 4)
            ids = faiss.rev_swig_ptr(lists.get_ids(0), n).copy()
            ordered = np.empty_like(codes)
            ordered[ids] = codes
            self.assertTrue(np.array_equal(expected, ordered))
            # Encode a new three-item catalog without fitting (below PQ training minimum).
            new_items = ['new-a', 'new-b', 'new-c']
            new_rows = [0, 123, len(x)-1]
            new_x = np.array(x[new_rows], dtype='<f4')
            new_x.tofile(root / 'new-embedding.f32')
            new_meta = meta | {'item_count': 3,
                'item_order_sha256': opq.hashlib.sha256('\n'.join(new_items).encode()).hexdigest()}
            (root / 'new-manifest.json').write_text(json.dumps(new_meta))
            new_ids = ['[PAD]'] + new_items
            (root / 'new-mapping.json').write_text(json.dumps({
                'id2item': new_ids, 'item2id': {v: i for i, v in enumerate(new_ids)}}))
            c2 = {k: v for k, v in c.items() if k not in ('item_seqs_path', 'split_rule')}
            c2.update(quantizer_dir=str(root / 'bundle'), output_dir=str(root / 'encoded'),
                      embedding_path=str(root / 'new-embedding.f32'),
                      embedding_manifest_path=str(root / 'new-manifest.json'),
                      id_mapping_path=str(root / 'new-mapping.json'))
            (root / 'new-config.json').write_text(json.dumps(c2, indent=2))
            with patch.object(opq, 'create_quantizer', side_effect=AssertionError('must not fit')):
                opq.run(c2, 'encode')
            opq.run(c2, 'validate')
            new_sids = json.loads((root / 'encoded/item.sem_ids').read_text())
            self.assertTrue(np.array_equal(np.array([new_sids[i] for i in new_items]), expected[new_rows]))
            bad = c2 | {'output_dir': str(root / 'bad-encoded')}
            metadata_path = root / 'bad-manifest.json'
            metadata = new_meta | {'instruction_sha256': 'different-instruction'}
            metadata_path.write_text(json.dumps(metadata))
            bad['embedding_manifest_path'] = str(metadata_path)
            with self.assertRaisesRegex(ValueError, 'identical encoder'):
                opq.run(bad, 'encode')
            self.assertFalse(any(name.startswith('genrec') for name in sys.modules))


if __name__ == '__main__':
    unittest.main()
