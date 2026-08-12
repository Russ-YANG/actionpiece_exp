#!/usr/bin/env python3
"""Generate resumable E3 joint text-image embeddings with local Qwen3-VL."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))
from genrec.image_manifest import available_image_path
from genrec.image_manifest import load_image_manifest


MODULE_PATH = REPO_ROOT / 'genrec/models/ActionPiece/qwen_local.py'
SPEC = importlib.util.spec_from_file_location('actionpiece_qwen_local', MODULE_PATH)
QWEN_LOCAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QWEN_LOCAL)


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config-file',
      default='experiments/e3_qwen3_vl_8b_native_multimodal_beauty.yaml',
  )
  parser.add_argument(
      '--cache-dir'
  )
  parser.add_argument('--output')
  parser.add_argument('--limit', type=int)
  parser.add_argument('--batch-size', type=int)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config_path = Path(args.config_file)
  config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
  if config.get('metadata') != 'qwen_multimodal':
    raise ValueError('E3 embedding generation requires metadata=qwen_multimodal.')
  if config.get('sent_emb_backend') != 'qwen_local':
    raise ValueError('E3 embedding generation requires sent_emb_backend=qwen_local.')

  cache_dir = Path(
      args.cache_dir
      or config.get('dataset_cache_dir')
      or 'cache/AmazonReviews2014/Beauty'
  )
  processed_dir = cache_dir / 'processed'
  manifest_path = processed_dir / 'image_download_manifest.jsonl'
  actual_manifest_sha256 = sha256_file(manifest_path)
  expected_manifest_sha256 = config.get('qwen_local_image_manifest_sha256')
  if not expected_manifest_sha256:
    raise ValueError(
        'Set qwen_local_image_manifest_sha256 in the experiment config to '
        f'{actual_manifest_sha256}.'
    )
  if actual_manifest_sha256 != expected_manifest_sha256:
    raise RuntimeError(
        'Image manifest SHA-256 mismatch: '
        f'got {actual_manifest_sha256}, expected {expected_manifest_sha256}.'
    )

  id_mapping = json.loads(
      (processed_dir / 'id_mapping.json').read_text(encoding='utf-8')
  )
  metadata = json.loads(
      (processed_dir / 'metadata.qwen_text.json').read_text(encoding='utf-8')
  )
  image_status = load_image_manifest(manifest_path)
  expected_items = set(id_mapping['id2item'][1:])
  if set(image_status) != expected_items:
    raise RuntimeError(
        'Image manifest keys do not exactly match the item mapping: '
        f'{len(image_status)} manifest rows for {len(expected_items)} items.'
    )

  item_ids = id_mapping['id2item'][1:]
  if args.limit is not None:
    if args.limit <= 0:
      raise ValueError('--limit must be positive.')
    if not args.output:
      raise ValueError('--limit requires --output to protect the full cache.')
    item_ids = item_ids[: args.limit]
  texts = []
  image_paths = []
  for item in item_ids:
    item_metadata = metadata[item]
    texts.append(
        item_metadata['sentence']
        if isinstance(item_metadata, dict)
        else item_metadata
    )
    record = image_status[item]
    image_paths.append(available_image_path(record))

  stem = QWEN_LOCAL.build_qwen_local_artifact_stem(
      backend=config['sent_emb_backend'],
      model_id=config['sent_emb_model'],
      dimension=config['sent_emb_dim'],
      instruction=config['qwen_api_instruction'],
      model_revision=config['qwen_local_model_revision'],
      code_revision=config['qwen_local_code_revision'],
      max_length=config['qwen_local_max_length'],
      torch_dtype=config['qwen_local_torch_dtype'],
      attn_implementation=config['qwen_local_attn_implementation'],
      modality='text_image_joint',
      image_manifest_sha256=expected_manifest_sha256,
  )
  output_path = Path(args.output) if args.output else processed_dir / f'{stem}.sent_emb'
  encoder = QWEN_LOCAL.QwenLocalMultimodalEncoder(
      model_path=config['qwen_local_model_path'],
      model_id=config['sent_emb_model'],
      model_revision=config['qwen_local_model_revision'],
      repo_path=config['qwen_local_repo_path'],
      code_revision=config['qwen_local_code_revision'],
      instruction=config['qwen_api_instruction'],
      dimension=config['sent_emb_dim'],
      batch_size=args.batch_size or config['qwen_local_batch_size'],
      max_length=config['qwen_local_max_length'],
      torch_dtype=config['qwen_local_torch_dtype'],
      attn_implementation=config['qwen_local_attn_implementation'],
      require_cuda=config['qwen_local_require_cuda'],
  )
  embeddings = encoder.encode(
      texts,
      image_paths,
      item_ids,
      output_path,
      image_manifest_sha256=expected_manifest_sha256,
      progress_callback=lambda completed, total: print(
          f'E3 joint embeddings: {completed}/{total}', flush=True
      ),
  )
  print(f'output: {output_path}')
  print(f'shape: {embeddings.shape}')
  print(f'finite: {bool(np.isfinite(embeddings).all())}')
  print(f'norm range: {np.linalg.norm(embeddings, axis=1).min():.6f} '
        f'.. {np.linalg.norm(embeddings, axis=1).max():.6f}')
  print(f'sha256: {sha256_file(output_path)}')


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print('Interrupted; committed batches remain resumable.', file=sys.stderr)
    raise
