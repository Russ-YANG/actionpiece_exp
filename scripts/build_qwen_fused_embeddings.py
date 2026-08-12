#!/usr/bin/env python3
"""Concatenate aligned Qwen text/image embeddings for E5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--text-embedding-path', required=True)
  parser.add_argument('--image-embedding-path', required=True)
  parser.add_argument('--output-path', required=True)
  parser.add_argument('--item-count', type=int, required=True)
  parser.add_argument('--text-dimension', type=int, required=True)
  parser.add_argument('--image-dimension', type=int, required=True)
  parser.add_argument('--image-weight', type=float, default=1.0)
  parser.add_argument('--final-normalize', action='store_true')
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
      stream.write('\n')
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  except BaseException:
    try:
      os.unlink(temporary)
    except FileNotFoundError:
      pass
    raise


def load_embeddings(
    path: Path, item_count: int, dimension: int
) -> np.ndarray:
  expected_bytes = item_count * dimension * np.dtype(np.float32).itemsize
  if not path.is_file() or path.stat().st_size != expected_bytes:
    actual = path.stat().st_size if path.exists() else None
    raise ValueError(
        f'Embedding size mismatch for {path}: got {actual} bytes, '
        f'expected {expected_bytes}.'
    )
  return np.memmap(
      path, dtype=np.float32, mode='r', shape=(item_count, dimension)
  )


def main() -> None:
  args = parse_args()
  if args.item_count <= 0:
    raise ValueError('--item-count must be positive.')
  if args.text_dimension <= 0 or args.image_dimension <= 0:
    raise ValueError('Embedding dimensions must be positive.')
  if args.image_weight < 0:
    raise ValueError('--image-weight must be non-negative.')

  text_path = Path(args.text_embedding_path)
  image_path = Path(args.image_embedding_path)
  output_path = Path(args.output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  temporary_path = Path(f'{output_path}.partial')

  text = load_embeddings(
      text_path, args.item_count, args.text_dimension
  )
  image = load_embeddings(
      image_path, args.item_count, args.image_dimension
  )
  if not np.isfinite(text).all() or not np.isfinite(image).all():
    raise ValueError('Input embeddings contain NaN or Inf.')

  fused_dimension = args.text_dimension + args.image_dimension
  fused = np.memmap(
      temporary_path,
      dtype=np.float32,
      mode='w+',
      shape=(args.item_count, fused_dimension),
  )
  block_size = 1024
  for start in range(0, args.item_count, block_size):
    end = min(start + block_size, args.item_count)
    block = np.concatenate(
        [text[start:end], args.image_weight * image[start:end]], axis=1
    ).astype(np.float32, copy=False)
    if args.final_normalize:
      norms = np.linalg.norm(block, axis=1, keepdims=True)
      if np.any(norms == 0):
        raise ValueError('Fused embeddings contain a zero vector.')
      block /= norms
    fused[start:end] = block
  fused.flush()
  del fused
  temporary_path.replace(output_path)

  result = np.memmap(
      output_path,
      dtype=np.float32,
      mode='r',
      shape=(args.item_count, fused_dimension),
  )
  norms = np.linalg.norm(result, axis=1)
  manifest = {
      'method': 'concatenate',
      'item_count': args.item_count,
      'text_dimension': args.text_dimension,
      'image_dimension': args.image_dimension,
      'fused_dimension': fused_dimension,
      'image_weight': args.image_weight,
      'final_normalize': args.final_normalize,
      'missing_image_policy': 'zero_vector',
      'text_embedding_path': str(text_path),
      'text_embedding_sha256': sha256_file(text_path),
      'image_embedding_path': str(image_path),
      'image_embedding_sha256': sha256_file(image_path),
      'output_path': str(output_path),
      'output_sha256': sha256_file(output_path),
      'finite': bool(np.isfinite(result).all()),
      'norm_min': float(norms.min()),
      'norm_max': float(norms.max()),
  }
  atomic_json_dump(manifest, Path(f'{output_path}.manifest.json'))
  print(f'output: {output_path}')
  print(f'shape: {result.shape}')
  print(f'finite: {manifest["finite"]}')
  print(f'norm range: {manifest["norm_min"]:.6f} .. {manifest["norm_max"]:.6f}')
  print(f'sha256: {manifest["output_sha256"]}')


if __name__ == '__main__':
  main()
