#!/usr/bin/env python3
"""Standalone raw OPQ SID workflow; imports no genrec/merge/model modules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import faiss
import numpy as np
from build_opq_sem_ids import atomic_json_dump, create_quantizer

SCHEMA = 1
FILES = ('item.sem_ids', 'item.tokens.json', 'sid_to_items.json',
         'collisions.json', 'fit_items.json', 'quantizer.faiss', 'manifest.json')
IDENTITY = ('backend', 'model_id', 'model_revision', 'code_revision',
            'instruction_sha256', 'dimension', 'native_dimension',
            'normalization', 'max_length', 'torch_dtype', 'attn_implementation')


def read_json(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def provenance(c):
    names = ['embedding_path', 'id_mapping_path', 'embedding_manifest_path']
    if 'item_seqs_path' in c:
        names.append('item_seqs_path')
    # Once per completed run/reuse, to detect stale reuse and transferred-input drift.
    return {k: {'path': str(Path(c[k]).resolve()), 'bytes': Path(c[k]).stat().st_size,
                'sha256': file_hash(c[k])} for k in names}


def signature(c):
    keys = ('dimension', 'n_codebooks', 'codebook_size', 'seed', 'threads',
            'split_rule', 'encoder', 'preprocessing')
    return {k: c[k] for k in keys if k in c}


def load_inputs(c, *, fitting=False):
    for key in ('dimension', 'n_codebooks', 'threads', 'seed', 'codebook_size'):
        if type(c[key]) is not int:
            raise ValueError(f'{key} must be an integer')
    if c['codebook_size'] != 256:
        raise ValueError('Only codebook_size=256: packed codes at other bit widths are unsupported')
    if c['dimension'] <= 0 or c['n_codebooks'] <= 0 or c['dimension'] % c['n_codebooks']:
        raise ValueError('dimension must be positive and divisible by n_codebooks')
    if c['threads'] <= 0:
        raise ValueError('threads must be positive')
    cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if cpus and c['threads'] > int(cpus):
        raise ValueError('threads exceeds SLURM_CPUS_PER_TASK')
    faiss.omp_set_num_threads(c['threads'])
    mapping = read_json(c['id_mapping_path'])
    ids = mapping['id2item']
    if not isinstance(ids, list) or len(ids) < 2 or ids[0] != '[PAD]':
        raise ValueError('id2item must be a list with [PAD] at index 0; row r maps to id2item[r+1]')
    items = ids[1:]
    if any(not isinstance(i, str) for i in items) or len(set(ids)) != len(ids):
        raise ValueError('Item IDs must be unique strings distinct from [PAD]')
    if any(type(mapping['item2id'].get(i)) is not int or mapping['item2id'][i] != r
           for r, i in enumerate(ids)) or set(mapping['item2id']) != set(ids):
        raise ValueError('item2id and id2item are not exact inverses')
    meta = read_json(c['embedding_manifest_path'])
    if meta.get('backend') != 'qwen_local' or meta.get('model_id') != 'Qwen/Qwen3-VL-Embedding-8B':
        raise ValueError('Requires verified local Qwen3-VL-Embedding-8B provenance; no API fallback')
    for key in ('model_revision', 'code_revision', 'instruction_sha256', 'normalization'):
        if not meta.get(key):
            raise ValueError(f'Embedding provenance missing {key}')
    if meta.get('dimension') != c['dimension'] or meta.get('item_count') != len(items):
        raise ValueError('Embedding manifest dimension/item count mismatch')
    order_hash = hashlib.sha256('\n'.join(items).encode()).hexdigest()
    if meta.get('item_order_sha256') != order_hash:
        raise ValueError('Embedding manifest item order does not match id_mapping')
    if c.get('encoder') != meta['model_id'] or c.get('preprocessing') != meta['normalization']:
        raise ValueError('Config encoder/preprocessing must match source manifest; no preprocessing is applied')
    path = Path(c['embedding_path'])
    expected = len(items) * c['dimension'] * 4
    if path.stat().st_size != expected:
        raise ValueError(f'Expected raw little-endian float32 bytes={expected}')
    x = np.memmap(path, mode='r', dtype='<f4', shape=(len(items), c['dimension']))
    if not np.isfinite(x).all():
        raise ValueError('Embedding contains NaN/Inf')
    fit_rows = None
    if fitting:
        if c.get('split_rule') != 'leave_one_out_sequence_minus_last_two':
            raise ValueError('Explicit supported split_rule required; no implicit split inference')
        seqs = read_json(c['item_seqs_path'])
        train = set()
        known = set(items)
        for seq in seqs.values():
            if not isinstance(seq, list) or any(i not in known for i in seq):
                raise ValueError('Invalid sequence or unknown item in sequences (including held-out positions)')
            train.update(seq[:-2])
        fit_rows = np.array([r for r, i in enumerate(items) if i in train], dtype=np.int64)
        if len(fit_rows) < 256:
            raise ValueError('PQ256 requires at least 256 distinct training items; micro-data is not an OPQ test')
        if len(fit_rows) < 39 * 256:
            print('WARNING: fewer than Faiss default recommended 39*256 training points; diagnostic only', flush=True)
    return items, x, meta, fit_rows


def raw_codes(index, x, m, batch_size=4096):
    ivf = faiss.downcast_index(index.index)
    if not index.is_trained or index.d != x.shape[1] or index.chain.size() != 1:
        raise ValueError('Untrained/incompatible OPQ index')
    if ivf.nlist != 1 or ivf.pq.M != m or ivf.pq.nbits != 8 or ivf.code_size != m:
        raise ValueError('Quantizer must be OPQ/IVF1/PQ with 8 bits per codebook')
    result = np.empty((len(x), m), dtype=np.uint8)
    for start in range(0, len(x), batch_size):
        # IVF1 uses zero bytes for the list ID. Reject any unexpected prefix.
        code = index.sa_encode(np.ascontiguousarray(x[start:start+batch_size], dtype=np.float32))
        if code.shape != (min(batch_size, len(x)-start), m):
            raise ValueError('Unexpected packed-code width/list prefix')
        result[start:start+len(code)] = code
    return result


def restore_tokens(tokens, m):
    result = [None] * m
    for token in tokens:
        if type(token) is not int or not 0 <= token < m * 256:
            raise ValueError('Invalid semantic token or special token')
        book, value = divmod(token, 256)
        if result[book] is not None:
            raise ValueError('Duplicate codebook')
        result[book] = value
    if any(value is None for value in result):
        raise ValueError('Missing codebook')
    return result


def check_sid(sids, items, m):
    if set(sids) != set(items):
        raise ValueError('SID item set mismatch')
    for item in items:
        code = sids[item]
        if not isinstance(code, list) or len(code) != m or any(type(v) is not int or not 0 <= v <= 255 for v in code):
            raise ValueError(f'Invalid SID for {item}')
    return np.array([sids[i] for i in items], dtype=np.uint8)


def sid_maps(sids):
    reverse = {}
    for item, code in sids.items():
        reverse.setdefault(','.join(map(str, code)), []).append(item)
    groups = {k: v for k, v in reverse.items() if len(v) > 1}
    sizes = [len(v) for v in groups.values()]
    stats = {'item_count': len(sids), 'unique_sid_count': len(reverse),
             'collision_group_count': len(groups), 'duplicate_item_count': sum(n-1 for n in sizes),
             'affected_item_count': sum(sizes), 'largest_group': max(map(len, reverse.values()), default=0),
             'groups': groups}
    return reverse, stats


def git_state():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        p = subprocess.run(['git', '-C', str(root), *args], text=True, capture_output=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain'),
            'entrypoint_sha256': file_hash(__file__),
            'builder_sha256': file_hash(Path(__file__).with_name('build_opq_sem_ids.py'))}


def verify_bundle(c, items, x, meta, out, current=None):
    manifest = read_json(out / 'manifest.json')
    if manifest.get('schema') != SCHEMA or manifest['signature'] != signature(c):
        raise ValueError('Existing bundle configuration mismatch')
    current = provenance(c) if current is None else current
    if {k: v['sha256'] for k, v in current.items()} != {k: v['sha256'] for k, v in manifest['inputs'].items()}:
        raise ValueError('Existing bundle input contents mismatch; do not overwrite')
    if file_hash(out / 'quantizer.faiss') != manifest['quantizer_sha256']:
        raise ValueError('Saved quantizer integrity mismatch')
    index = faiss.read_index(str(out / 'quantizer.faiss'))
    if index.ntotal != 0:
        raise ValueError('Expected trained, empty quantizer (not a catalog index)')
    sids = read_json(out / 'item.sem_ids')
    codes = check_sid(sids, items, c['n_codebooks'])
    if not np.array_equal(codes, raw_codes(index, x, c['n_codebooks'])):
        raise ValueError('Reloaded quantizer codes differ from saved SID')
    tokens = read_json(out / 'item.tokens.json')
    if set(tokens) != set(items) or any(restore_tokens(tokens[i], c['n_codebooks']) != sids[i] for i in items):
        raise ValueError('Token round trip failed')
    reverse, stats = sid_maps(sids)
    if reverse != read_json(out / 'sid_to_items.json') or stats != read_json(out / 'collisions.json'):
        raise ValueError('Collision reverse mapping mismatch')
    fit = read_json(out / 'fit_items.json')
    if manifest['mode'] == 'fit':
        _, _, _, rows = load_inputs(c, fitting=True)
        if fit != [items[r] for r in rows]:
            raise ValueError('Fitting item set mismatch')
    elif fit:
        raise ValueError('Encode-only bundle must not claim fitted items')
    print(json.dumps({'validated': str(out), 'item_count': len(items), 'fit_item_count': len(fit),
                      'collisions': {k: v for k, v in stats.items() if k != 'groups'}}, indent=2), flush=True)
    return manifest


def run(c, mode):
    items, x, meta, rows = load_inputs(c, fitting=mode in ('fit', 'preflight'))
    inputs = provenance(c)
    if meta.get('embedding_sha256') and inputs['embedding_path']['sha256'] != meta['embedding_sha256']:
        raise ValueError('Embedding content differs from source manifest hash')
    if mode == 'preflight':
        print(json.dumps({'items': len(items), 'fit_items': len(rows), 'shape': list(x.shape),
                          'source': meta, 'signature': signature(c), 'faiss': faiss.__version__}, indent=2))
        return
    out = Path(c['output_dir'])
    if mode == 'audit-sid':
        sids = read_json(c['existing_sid_path'])
        check_sid(sids, items, c['n_codebooks'])
        reverse, stats = sid_maps(sids)
        out.mkdir(parents=True, exist_ok=False)
        atomic_json_dump(sids, out / 'item.sem_ids')
        tokens = {i: [book*256+v for book, v in enumerate(code)] for i, code in sids.items()}
        if any(restore_tokens(list(reversed(tokens[i])), c['n_codebooks']) != sids[i] for i in items):
            raise RuntimeError('Historical SID token round trip failed')
        atomic_json_dump(tokens, out / 'item.tokens.json')
        atomic_json_dump(reverse, out / 'sid_to_items.json')
        atomic_json_dump(stats, out / 'collisions.json')
        atomic_json_dump({'status': 'historical_sid_structure_only', 'quantizer_reload_verified': False,
                          'existing_sid_path': c['existing_sid_path'], 'source_manifest': meta,
                          'stats': {k: v for k, v in stats.items() if k != 'groups'}}, out / 'audit.json')
        print(json.dumps(stats | {'groups': '(see collisions.json)'}, indent=2))
        return
    if mode == 'validate' or out.exists():
        if not all((out / f).is_file() for f in FILES):
            raise FileExistsError('Incomplete bundle: choose a new output directory; nothing overwritten')
        old = verify_bundle(c, items, x, meta, out, inputs)
        if mode != 'validate' and old['mode'] != mode:
            raise ValueError('Existing bundle mode mismatch')
        if mode == 'encode' and old['quantizer_source_sha256'] != file_hash(Path(c['quantizer_dir']) / 'quantizer.faiss'):
            raise ValueError('Existing encode bundle quantizer source mismatch')
        return
    started = time.monotonic()
    quantizer_source = None
    if mode == 'fit':
        index, opq, ivf, opq_pq = create_quantizer(c['dimension'], c['n_codebooks'], 256, c['seed'])
        opq.verbose = bool(c.get('verbose', False))
        ivf.verbose = bool(c.get('verbose', False))
        print(f'Fitting on {len(rows)} training items; encoding {len(items)} catalog items', flush=True)
        index.train(np.ascontiguousarray(x[rows], dtype=np.float32))
        fit_items = [items[r] for r in rows]
    elif mode == 'encode':
        source = Path(c['quantizer_dir'])
        old = read_json(source / 'manifest.json')
        if old['mode'] != 'fit':
            raise ValueError('New encoding must reference the original fit bundle')
        if any(meta.get(k) != old['embedding_source'].get(k) for k in IDENTITY):
            raise ValueError('New embeddings must have identical encoder revision, instruction and preprocessing')
        if any(c[k] != old['signature'][k] for k in ('dimension', 'n_codebooks', 'codebook_size')):
            raise ValueError('Quantizer/config shape mismatch')
        quantizer_source = file_hash(source / 'quantizer.faiss')
        if quantizer_source != old['quantizer_sha256']:
            raise ValueError('Quantizer source integrity mismatch')
        index = faiss.read_index(str(source / 'quantizer.faiss'))
        fit_items = []
    else:
        raise ValueError(f'Unknown mode {mode}')
    if index.ntotal:
        raise ValueError('Quantizer must be empty, with no indexed catalog')
    codes = raw_codes(index, x, c['n_codebooks'])
    out.mkdir(parents=True, exist_ok=False)
    # Manifest is written last; partial directories are never reused as complete.
    faiss.write_index(index, str(out / 'quantizer.faiss'))
    reloaded = faiss.read_index(str(out / 'quantizer.faiss'))
    if not np.array_equal(codes, raw_codes(reloaded, x, c['n_codebooks'])):
        raise RuntimeError('Save/reload encoding mismatch')
    sids = {i: codes[r].astype(int).tolist() for r, i in enumerate(items)}
    check_sid(sids, items, c['n_codebooks'])
    tokens = {i: [book * 256 + v for book, v in enumerate(code)] for i, code in sids.items()}
    if any(restore_tokens(list(reversed(tokens[i])), c['n_codebooks']) != sids[i] for i in items):
        raise RuntimeError('Codebook-identity token round trip failed')
    reverse, stats = sid_maps(sids)
    for name, value in [('item.sem_ids', sids), ('item.tokens.json', tokens),
                        ('sid_to_items.json', reverse), ('collisions.json', stats), ('fit_items.json', fit_items)]:
        atomic_json_dump(value, out / name)
    manifest = {'schema': SCHEMA, 'mode': mode, 'signature': signature(c), 'inputs': inputs,
                'embedding_source': meta, 'synthetic_test_fixture': bool(meta.get('test_fixture', False)), 'preprocessing_applied': 'none; read prepared float32 unchanged',
                'row_order': 'embedding row r = id2item[r+1]; id2item[0]=[PAD], no embedding row for PAD',
                'fit_item_count': len(fit_items), 'encoded_item_count': len(items),
                'split': c.get('split_rule') if mode == 'fit' else 'no fitting; original quantizer reused',
                'factory': f"OPQ{c['n_codebooks']},IVF1,PQ{c['n_codebooks']}x8",
                'faiss_seeds': {'OPQ_training_PQ': c['seed'], 'IVF_coarse': c['seed'], 'IVF_PQ': c['seed']} if mode == 'fit' else None,
                'by_residual': bool(faiss.downcast_index(index.index).by_residual),
                'training_defaults': 'Faiss defaults; no iteration/subsampling overrides',
                'training_parameters': ({'opq_niter': int(opq.niter),
                    'opq_niter_pq': int(opq.niter_pq), 'opq_niter_pq_0': int(opq.niter_pq_0),
                    'opq_max_train_points': int(opq.max_train_points),
                    'final_pq_niter': int(ivf.pq.cp.niter),
                    'pq_max_points_per_centroid': int(ivf.pq.cp.max_points_per_centroid)}
                    if mode == 'fit' else old['training_parameters']),
                'token_mapping': {'formula': 'codebook_id*256+code_value', 'semantic_range': [0, c['n_codebooks']*256-1],
                                  'special_tokens': {}, 'reserved_special_start': c['n_codebooks']*256},
                'reload_all_items_equal': True, 'collision_stats': {k: v for k, v in stats.items() if k != 'groups'},
                'quantizer_sha256': file_hash(out / 'quantizer.faiss'),
                'quantizer_source_sha256': quantizer_source,
                'quantizer_origin': str(Path(c['quantizer_dir']).resolve()) if mode == 'encode' else None,
                'git': git_state(), 'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                   'faiss': faiss.__version__, 'platform': platform.platform(), 'machine': platform.machine(),
                   'faiss_compile_options': faiss.get_compile_options(), 'slurm_job_id': os.getenv('SLURM_JOB_ID'),
                   'slurm_cpus_per_task': os.getenv('SLURM_CPUS_PER_TASK'),
                   'thread_env': {k: os.getenv(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}},
                'elapsed_seconds': time.monotonic() - started}
    atomic_json_dump(manifest, out / 'manifest.json')
    print(f'Completed {mode}: {out}; fit={len(fit_items)}, encoded={len(items)}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'fit', 'encode', 'validate', 'audit-sid'))
    parser.add_argument('--config', required=True, help='JSON; paths relative to working directory')
    args = parser.parse_args()
    run(read_json(args.config), args.mode)


if __name__ == '__main__':
    main()
