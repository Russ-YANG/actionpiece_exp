#!/usr/bin/env python3
"""Generate resumable text-only embeddings with local Qwen3-VL."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
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


def build_artifact_stem(config: dict) -> str:
  """Build the text cache name across old and new qwen_local modules."""
  if hasattr(QWEN_LOCAL, 'build_qwen_local_artifact_stem'):
    return QWEN_LOCAL.build_qwen_local_artifact_stem(
        backend=config['sent_emb_backend'],
        model_id=config['sent_emb_model'],
        dimension=config['sent_emb_dim'],
        instruction=config['qwen_api_instruction'],
        model_revision=config['qwen_local_model_revision'],
        code_revision=config['qwen_local_code_revision'],
        max_length=config['qwen_local_max_length'],
        torch_dtype=config['qwen_local_torch_dtype'],
        attn_implementation=config['qwen_local_attn_implementation'],
    )
  model_name = Path(config['sent_emb_model']).name
  model_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model_name)
  instruction_hash = hashlib.sha256(
      config['qwen_api_instruction'].encode('utf-8')
  ).hexdigest()[:12]
  revision = re.sub(
      r'[^A-Za-z0-9._-]+', '-', config['qwen_local_model_revision']
  )[:12]
  code_revision = re.sub(
      r'[^A-Za-z0-9._-]+', '-', config['qwen_local_code_revision']
  )[:12]
  dtype = re.sub(
      r'[^A-Za-z0-9._-]+', '-', config['qwen_local_torch_dtype']
  )
  attention = re.sub(
      r'[^A-Za-z0-9._-]+', '-', config['qwen_local_attn_implementation']
  )
  return (
      f'{config["sent_emb_backend"]}.{model_name}.'
      f'd{config["sent_emb_dim"]}.i{instruction_hash}.'
      f'r{revision}.c{code_revision}.m{config["qwen_local_max_length"]}.'
      f't{dtype}.a{attention}'
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--config-file', required=True)
  parser.add_argument('--cache-dir')
  parser.add_argument('--output')
  parser.add_argument('--limit', type=int)
  parser.add_argument('--batch-size', type=int)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config = yaml.safe_load(Path(args.config_file).read_text(encoding='utf-8'))
  if config.get('metadata') not in {
      'qwen_text', 'qwen_separate', 'qwen_fused'
  }:
    raise ValueError(
        'Text generation requires metadata=qwen_text, qwen_separate, '
        'or qwen_fused.'
    )
  if config.get('sent_emb_backend') != 'qwen_local':
    raise ValueError('Text generation requires sent_emb_backend=qwen_local.')

  category = config.get('category')
  configured_cache_dir = config.get('dataset_cache_dir')
  if not args.cache_dir and not configured_cache_dir and not category:
    raise ValueError(
        'Set dataset_cache_dir in the config or pass --cache-dir.'
    )
  cache_dir = Path(
      args.cache_dir
      or configured_cache_dir
      or f'cache/AmazonReviews2014/{category}'
  )
  processed_dir = cache_dir / 'processed'
  id_mapping = json.loads(
      (processed_dir / 'id_mapping.json').read_text(encoding='utf-8')
  )
  metadata = json.loads(
      (processed_dir / 'metadata.qwen_text.json').read_text(encoding='utf-8')
  )
  item_ids = id_mapping['id2item'][1:]
  if set(item_ids) != set(metadata):
    raise RuntimeError(
        'Metadata keys do not exactly match the item mapping: '
        f'{len(metadata)} metadata rows for {len(item_ids)} items.'
    )
  if args.limit is not None:
    if args.limit <= 0:
      raise ValueError('--limit must be positive.')
    if not args.output:
      raise ValueError('--limit requires --output to protect the full cache.')
    item_ids = item_ids[: args.limit]
  texts = [metadata[item] for item in item_ids]

  stem = build_artifact_stem(config)
  output_path = (
      Path(args.output)
      if args.output
      else processed_dir / f'{stem}.sent_emb'
  )
  encoder = QWEN_LOCAL.QwenLocalTextEncoder(
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
      item_ids,
      output_path,
      progress_callback=lambda completed, total: print(
          f'Text embeddings: {completed}/{total}', flush=True
      ),
  )
  norms = np.linalg.norm(embeddings, axis=1)
  print(f'output: {output_path}')
  print(f'shape: {embeddings.shape}')
  print(f'finite: {bool(np.isfinite(embeddings).all())}')
  print(f'norm range: {norms.min():.6f} .. {norms.max():.6f}')
  print(f'sha256: {sha256_file(output_path)}')


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print('Interrupted; committed batches remain resumable.', file=sys.stderr)
    raise
