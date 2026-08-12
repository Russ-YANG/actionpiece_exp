#!/usr/bin/env python3
"""Generate resumable E5 image-only embeddings with local Qwen3-VL."""

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


def build_image_artifact_stem(
    config: dict, image_manifest_sha256: str
) -> str:
  """Build the image cache name across old and new qwen_local modules."""
  if hasattr(QWEN_LOCAL, 'build_qwen_local_artifact_stem'):
    return QWEN_LOCAL.build_qwen_local_artifact_stem(
        backend=config['sent_emb_backend'],
        model_id=config['sent_emb_model'],
        dimension=config['image_emb_dim'],
        instruction=config['qwen_local_image_instruction'],
        model_revision=config['qwen_local_model_revision'],
        code_revision=config['qwen_local_code_revision'],
        max_length=config['qwen_local_max_length'],
        torch_dtype=config['qwen_local_torch_dtype'],
        attn_implementation=config['qwen_local_attn_implementation'],
        modality='image',
        image_manifest_sha256=image_manifest_sha256,
    )
  model_name = Path(config['sent_emb_model']).name
  model_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model_name)
  instruction_hash = hashlib.sha256(
      config['qwen_local_image_instruction'].encode('utf-8')
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
      f'd{config["image_emb_dim"]}.i{instruction_hash}.'
      f'r{revision}.c{code_revision}.m{config["qwen_local_max_length"]}.'
      f't{dtype}.a{attention}.image.im{image_manifest_sha256[:12]}'
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--config-file',
      default='experiments/e5_qwen3_vl_8b_separate_fused_beauty.yaml',
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
  config = yaml.safe_load(Path(args.config_file).read_text(encoding='utf-8'))
  if config.get('metadata') not in {'qwen_fused', 'qwen_separate'}:
    raise ValueError(
        'Image generation requires metadata=qwen_fused or qwen_separate.'
    )
  if config.get('sent_emb_backend') != 'qwen_local':
    raise ValueError('E5 image generation requires sent_emb_backend=qwen_local.')

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
  item_ids = id_mapping['id2item'][1:]
  if args.limit is not None:
    if args.limit <= 0:
      raise ValueError('--limit must be positive.')
    if not args.output:
      raise ValueError('--limit requires --output to protect the full cache.')
    item_ids = item_ids[: args.limit]

  image_status = load_image_manifest(manifest_path)
  if set(image_status) != set(id_mapping['id2item'][1:]):
    raise RuntimeError(
        'Image manifest keys do not exactly match the item mapping: '
        f'{len(image_status)} manifest rows for '
        f'{len(id_mapping["id2item"]) - 1} items.'
    )

  image_paths = []
  for item in item_ids:
    record = image_status[item]
    image_paths.append(available_image_path(record))

  stem = build_image_artifact_stem(config, expected_manifest_sha256)
  output_path = (
      Path(args.output)
      if args.output
      else processed_dir / f'{stem}.image_emb'
  )
  encoder = QWEN_LOCAL.QwenLocalImageEncoder(
      model_path=config['qwen_local_model_path'],
      model_id=config['sent_emb_model'],
      model_revision=config['qwen_local_model_revision'],
      repo_path=config['qwen_local_repo_path'],
      code_revision=config['qwen_local_code_revision'],
      instruction=config['qwen_local_image_instruction'],
      dimension=config['image_emb_dim'],
      batch_size=args.batch_size or config['qwen_local_batch_size'],
      max_length=config['qwen_local_max_length'],
      torch_dtype=config['qwen_local_torch_dtype'],
      attn_implementation=config['qwen_local_attn_implementation'],
      require_cuda=config['qwen_local_require_cuda'],
  )
  embeddings = encoder.encode(
      image_paths,
      item_ids,
      output_path,
      image_manifest_sha256=expected_manifest_sha256,
      progress_callback=lambda completed, total: print(
          f'E5 image embeddings: {completed}/{total}', flush=True
      ),
  )
  norms = np.linalg.norm(embeddings, axis=1)
  print(f'output: {output_path}')
  print(f'shape: {embeddings.shape}')
  print(f'finite: {bool(np.isfinite(embeddings).all())}')
  print(f'zero vectors: {int(np.count_nonzero(norms == 0))}')
  nonzero = norms[norms > 0]
  if nonzero.size:
    print(f'nonzero norm range: {nonzero.min():.6f} .. {nonzero.max():.6f}')
  print(f'sha256: {sha256_file(output_path)}')


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    print('Interrupted; committed batches remain resumable.', file=sys.stderr)
    raise
