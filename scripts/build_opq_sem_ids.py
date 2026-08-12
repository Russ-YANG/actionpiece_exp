#!/usr/bin/env python3
"""Build ActionPiece OPQ semantic IDs without importing the full project."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding-path', required=True)
  parser.add_argument('--id-mapping-path', required=True)
  parser.add_argument('--item-seqs-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--dimension', type=int, required=True)
  parser.add_argument('--n-codebooks', type=int, default=4)
  parser.add_argument('--codebook-size', type=int, default=256)
  parser.add_argument('--threads', type=int, default=8)
  parser.add_argument('--seed', type=int, default=2024)
  parser.add_argument('--verbose', action='store_true')
  return parser.parse_args()


def atomic_json_dump(value: object, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temp_path = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
  try:
    with os.fdopen(fd, 'w') as handle:
      json.dump(value, handle)
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temp_path, path)
  except BaseException:
    try:
      os.unlink(temp_path)
    except FileNotFoundError:
      pass
    raise


def main() -> None:
  args = parse_args()
  np.random.seed(args.seed)
  faiss.omp_set_num_threads(args.threads)

  embedding_path = Path(args.embedding_path)
  output_path = Path(args.output_path)
  manifest_path = Path(f'{output_path}.manifest.json')
  if output_path.exists() or manifest_path.exists():
    raise FileExistsError(
        f'Refusing to overwrite an existing OPQ artifact: {output_path}'
    )
  with open(args.id_mapping_path) as handle:
    id_mapping = json.load(handle)
  with open(args.item_seqs_path) as handle:
    item_seqs = json.load(handle)

  id2item = id_mapping['id2item']
  item2id = id_mapping['item2id']
  n_items = len(id2item) - 1
  embeddings = np.fromfile(embedding_path, dtype=np.float32)
  if embeddings.size != n_items * args.dimension:
    raise ValueError(
        f'Embedding size mismatch: got {embeddings.size} float32 values, '
        f'expected {n_items * args.dimension}'
    )
  embeddings = np.ascontiguousarray(
      embeddings.reshape(n_items, args.dimension), dtype=np.float32
  )

  training_items = {
      item
      for sequence in item_seqs.values()
      if len(sequence) > 2
      for item in sequence[:-2]
  }
  training_mask = np.zeros(n_items, dtype=bool)
  for item in training_items:
    training_mask[item2id[item] - 1] = True
  training_embeddings = np.ascontiguousarray(embeddings[training_mask])

  print(f'embeddings: {embeddings.shape}', flush=True)
  print(f'training items: {training_embeddings.shape[0]} of {n_items}', flush=True)
  factory = (
      f'OPQ{args.n_codebooks},IVF1,'
      f'PQ{args.n_codebooks}x{int(np.log2(args.codebook_size))}'
  )
  print(f'training Faiss index: {factory}', flush=True)
  started = time.time()
  index = faiss.index_factory(
      args.dimension, factory, faiss.METRIC_INNER_PRODUCT
  )
  # NumPy's RNG does not control Faiss clustering. Explicitly seed the
  # coarse quantizer, the final IVF-PQ, and the PQ used while learning the
  # OPQ rotation so the filename's seed describes the actual artifact.
  opq = faiss.downcast_VectorTransform(index.chain.at(0))
  opq_training_pq = faiss.ProductQuantizer(
      args.dimension,
      args.n_codebooks,
      int(np.log2(args.codebook_size)),
  )
  opq_training_pq.cp.seed = args.seed
  opq.pq = opq_training_pq
  ivf_index = faiss.downcast_index(index.index)
  ivf_index.cp.seed = args.seed
  ivf_index.pq.cp.seed = args.seed
  if args.verbose:
    # Expose otherwise silent OPQ/PQ clustering progress when diagnosing a
    # slow training run.
    try:
      opq.verbose = True
      ivf_index.verbose = True
      ivf_index.pq.verbose = True
    except (AttributeError, RuntimeError):
      pass
  index.train(training_embeddings)
  print(f'index trained in {(time.time() - started) / 60:.1f} min', flush=True)
  index.add(embeddings)

  invlists = faiss.extract_index_ivf(ivf_index).invlists
  list_size = invlists.list_size(0)
  codes = faiss.rev_swig_ptr(
      invlists.get_codes(0), list_size * invlists.code_size
  ).reshape(-1, invlists.code_size)
  ids = faiss.rev_swig_ptr(invlists.get_ids(0), list_size).copy()
  if list_size != n_items:
    raise RuntimeError(f'Expected {n_items} indexed items, got {list_size}')

  ordered_codes = np.empty_like(codes)
  ordered_codes[ids] = codes
  item2sem_ids = {
      id2item[row + 1]: ordered_codes[row].astype(int).tolist()
      for row in range(n_items)
  }
  if set(item2sem_ids) != set(id2item[1:]):
    raise RuntimeError('Semantic-ID keys do not align with id_mapping.json')
  semantic_ids = np.asarray(list(item2sem_ids.values()), dtype=np.int64)
  if semantic_ids.shape != (n_items, args.n_codebooks):
    raise RuntimeError(
        f'Invalid semantic-ID shape {semantic_ids.shape}; expected '
        f'{(n_items, args.n_codebooks)}'
    )
  if semantic_ids.min() < 0 or semantic_ids.max() >= args.codebook_size:
    raise RuntimeError(
        f'Raw codes outside [0, {args.codebook_size - 1}]: '
        f'min={semantic_ids.min()}, max={semantic_ids.max()}'
    )
  sid_counts = Counter(map(tuple, semantic_ids.tolist()))
  collision_sizes = [count for count in sid_counts.values() if count > 1]
  collision_group_count = len(collision_sizes)
  collision_count = sum(count - 1 for count in collision_sizes)
  affected_item_count = sum(collision_sizes)
  largest_collision_group = max(collision_sizes, default=1)
  atomic_json_dump(item2sem_ids, output_path)
  sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
  manifest = {
      'artifact': str(output_path),
      'sha256': sha256,
      'embedding_path': str(embedding_path),
      'embedding_sha256': hashlib.sha256(embedding_path.read_bytes()).hexdigest(),
      'id_mapping_sha256': hashlib.sha256(
          Path(args.id_mapping_path).read_bytes()
      ).hexdigest(),
      'item_seqs_sha256': hashlib.sha256(
          Path(args.item_seqs_path).read_bytes()
      ).hexdigest(),
      'item_count': n_items,
      'training_item_count': int(training_mask.sum()),
      'dimension': args.dimension,
      'n_codebooks': args.n_codebooks,
      'codebook_size': args.codebook_size,
      'factory': factory,
      'seed': args.seed,
      'threads': args.threads,
      'faiss_version': getattr(faiss, '__version__', 'unknown'),
      'python_version': platform.python_version(),
      'raw_code_min': int(semantic_ids.min()),
      'raw_code_max': int(semantic_ids.max()),
      'exact_sid_collision_count': collision_count,
      'collision_group_count': collision_group_count,
      'affected_item_count': affected_item_count,
      'largest_collision_group': largest_collision_group,
  }
  atomic_json_dump(manifest, manifest_path)
  print(f'saved: {output_path}', flush=True)
  print(
      f'items: {len(item2sem_ids)}, code length: {semantic_ids.shape[1]}',
      flush=True,
  )
  print(
      'exact-SID collisions: '
      f'{collision_count} duplicate items in {collision_group_count} groups; '
      f'affected items: {affected_item_count}; '
      f'largest group: {largest_collision_group}',
      flush=True,
  )
  print(f'sha256: {sha256}', flush=True)
  print(f'manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
  main()
