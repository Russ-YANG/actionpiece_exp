#!/usr/bin/env python3
"""Derive normalized MRL prefix embeddings from a raw float32 matrix."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--source-path', required=True)
  parser.add_argument('--source-dimension', type=int, required=True)
  parser.add_argument('--item-count', type=int, required=True)
  parser.add_argument('--output-prefix', required=True)
  parser.add_argument('--dimensions', type=int, nargs='+', required=True)
  return parser.parse_args()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def atomic_json_dump(value: object, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
  try:
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
      json.dump(value, stream, ensure_ascii=False, indent=2)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  except BaseException:
    try:
      os.unlink(temporary)
    except FileNotFoundError:
      pass
    raise


def main() -> None:
  args = parse_args()
  if args.source_dimension <= 0 or args.item_count <= 0:
    raise ValueError('Dimensions and item count must be positive.')
  dimensions = list(dict.fromkeys(args.dimensions))
  if any(d <= 0 or d > args.source_dimension for d in dimensions):
    raise ValueError('Every target dimension must be in (0, source_dimension].')

  source_path = Path(args.source_path)
  expected_bytes = args.item_count * args.source_dimension * 4
  if source_path.stat().st_size != expected_bytes:
    raise RuntimeError(
        f'Source has {source_path.stat().st_size} bytes; expected '
        f'{expected_bytes}.'
    )
  outputs = [Path(f'{args.output_prefix}.d{d}.sent_emb') for d in dimensions]
  manifests = [Path(f'{path}.manifest.json') for path in outputs]
  existing = [path for path in outputs + manifests if path.exists()]
  if existing:
    raise FileExistsError(f'Refusing to overwrite: {existing[0]}')

  source = np.memmap(
      source_path,
      dtype=np.float32,
      mode='r',
      shape=(args.item_count, args.source_dimension),
  )
  source_sha256 = sha256_file(source_path)
  for dimension, output_path, manifest_path in zip(
      dimensions, outputs, manifests
  ):
    partial_path = output_path.with_suffix(output_path.suffix + '.partial')
    output = np.memmap(
        partial_path,
        dtype=np.float32,
        mode='w+',
        shape=(args.item_count, dimension),
    )
    for start in range(0, args.item_count, 512):
      end = min(start + 512, args.item_count)
      block = np.asarray(source[start:end, :dimension], dtype=np.float32)
      norms = np.linalg.norm(block, axis=1, keepdims=True)
      if np.any(norms == 0) or not np.isfinite(norms).all():
        raise RuntimeError(f'Invalid prefix norm at dimension {dimension}.')
      output[start:end] = block / norms
    output.flush()
    del output
    partial_path.replace(output_path)

    vectors = np.memmap(
        output_path,
        dtype=np.float32,
        mode='r',
        shape=(args.item_count, dimension),
    )
    if not np.isfinite(vectors).all():
      raise RuntimeError(f'{output_path} contains NaN or Inf.')
    norms = np.linalg.norm(vectors, axis=1)
    manifest = {
        'artifact': str(output_path),
        'sha256': sha256_file(output_path),
        'bytes': output_path.stat().st_size,
        'item_count': args.item_count,
        'dimension': dimension,
        'source_path': str(source_path),
        'source_dimension': args.source_dimension,
        'source_sha256': source_sha256,
        'operation': 'prefix_truncate_then_l2_normalize',
        'norm_min': float(norms.min()),
        'norm_max': float(norms.max()),
    }
    del vectors
    atomic_json_dump(manifest, manifest_path)
    print(
        f'd{dimension}: {output_path} sha256={manifest["sha256"]}',
        flush=True,
    )
  del source


if __name__ == '__main__':
  main()
