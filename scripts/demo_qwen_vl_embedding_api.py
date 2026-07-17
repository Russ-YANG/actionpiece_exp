#!/usr/bin/env python3
"""Run a two-request Qwen3-VL-Embedding API smoke test.

The script reads the API key from an ignored .env.local file, selects one local
Beauty product, requests independent text/image embeddings and a native fused
embedding, then stores only non-secret summary statistics (not full vectors).
"""

import argparse
import base64
import json
import math
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests


DEFAULT_ENDPOINT = (
    'https://dashscope.aliyuncs.com/api/v1/services/embeddings/'
    'multimodal-embedding/multimodal-embedding'
)
DEFAULT_INSTRUCTION = (
    'Represent this e-commerce product for semantic similarity and sequential '
    'recommendation. Capture its category, function, attributes, style, and '
    'likely user intent.'
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--env-file', type=Path, default=Path('.env.local'))
  parser.add_argument(
      '--catalog',
      type=Path,
      default=Path(
          'cache/AmazonReviews2014/Beauty/processed/'
          'multimodal_catalog.jsonl'
      ),
  )
  parser.add_argument(
      '--image-manifest',
      type=Path,
      default=Path(
          'cache/AmazonReviews2014/Beauty/processed/'
          'image_download_manifest.jsonl'
      ),
  )
  parser.add_argument(
      '--output',
      type=Path,
      default=Path(
          'cache/AmazonReviews2014/Beauty/processed/'
          'qwen_vl_embedding_api_demo.json'
      ),
  )
  parser.add_argument('--model', default='qwen3-vl-embedding')
  parser.add_argument('--dimension', type=int, default=768)
  parser.add_argument('--instruction', default=DEFAULT_INSTRUCTION)
  parser.add_argument('--timeout', type=float, default=120.0)
  return parser.parse_args()


def read_env_file(path: Path) -> dict[str, str]:
  values = {}
  for raw_line in path.read_text(encoding='utf-8').splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#') or '=' not in line:
      continue
    key, value = line.split('=', 1)
    values[key.strip()] = value.strip().strip('"').strip("'")
  return values


def load_jsonl_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
  records = {}
  with path.open(encoding='utf-8') as stream:
    for line in stream:
      record = json.loads(line)
      records[record[key]] = record
  return records


def select_product(
    catalog_path: Path, image_manifest_path: Path
) -> tuple[dict[str, Any], Path]:
  images = load_jsonl_by_key(image_manifest_path, 'asin')
  with catalog_path.open(encoding='utf-8') as stream:
    for line in stream:
      product = json.loads(line)
      image = images.get(product['asin'])
      if not product.get('text') or not image:
        continue
      if image.get('status') != 'downloaded':
        continue
      image_path = Path(image['path'])
      if image_path.is_file():
        return product, image_path
  raise RuntimeError('No product with both text and a local image was found.')


def image_data_uri(path: Path) -> str:
  mime_type = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
  encoded = base64.b64encode(path.read_bytes()).decode('ascii')
  return f'data:{mime_type};base64,{encoded}'


def call_api(
    endpoint: str,
    api_key: str,
    model: str,
    contents: list[dict[str, str]],
    instruction: str,
    dimension: int,
    enable_fusion: bool,
    timeout: float,
) -> dict[str, Any]:
  response = requests.post(
      endpoint,
      headers={
          'Authorization': f'Bearer {api_key}',
          'Content-Type': 'application/json',
      },
      json={
          'model': model,
          'input': {'contents': contents},
          'parameters': {
              'dimension': dimension,
              'instruct': instruction,
              'enable_fusion': enable_fusion,
          },
      },
      timeout=timeout,
  )
  try:
    body = response.json()
  except requests.JSONDecodeError as exc:
    raise RuntimeError(
        f'API returned HTTP {response.status_code} with non-JSON content.'
    ) from exc
  if response.status_code != 200 or 'output' not in body:
    code = body.get('code', 'unknown')
    message = body.get('message', 'No error message returned.')
    raise RuntimeError(
        f'API request failed: HTTP {response.status_code}, {code}: {message}'
    )
  return body


def norm(vector: list[float]) -> float:
  return math.sqrt(sum(value * value for value in vector))


def cosine(left: list[float], right: list[float]) -> float:
  denominator = norm(left) * norm(right)
  if denominator == 0:
    return 0.0
  return sum(a * b for a, b in zip(left, right)) / denominator


def main() -> None:
  args = parse_args()
  env = read_env_file(args.env_file)
  api_key = env.get('DASHSCOPE_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
  if not api_key:
    raise RuntimeError('DASHSCOPE_API_KEY was not found.')
  endpoint = env.get('DASHSCOPE_BASE_URL', DEFAULT_ENDPOINT)

  product, image_path = select_product(args.catalog, args.image_manifest)
  contents = [
      {'text': product['text']},
      {'image': image_data_uri(image_path)},
  ]
  independent = call_api(
      endpoint,
      api_key,
      args.model,
      contents,
      args.instruction,
      args.dimension,
      False,
      args.timeout,
  )
  fused = call_api(
      endpoint,
      api_key,
      args.model,
      contents,
      args.instruction,
      args.dimension,
      True,
      args.timeout,
  )

  independent_embeddings = independent['output']['embeddings']
  fused_embeddings = fused['output']['embeddings']
  if len(independent_embeddings) != 2 or len(fused_embeddings) != 1:
    raise RuntimeError(
        'Unexpected embedding counts: '
        f'independent={len(independent_embeddings)}, fused={len(fused_embeddings)}'
    )
  text_vector = independent_embeddings[0]['embedding']
  image_vector = independent_embeddings[1]['embedding']
  fused_vector = fused_embeddings[0]['embedding']

  summary = {
      'model': args.model,
      'dimension_requested': args.dimension,
      'instruction': args.instruction,
      'product': {
          'asin': product['asin'],
          'text_characters': len(product['text']),
          'image_path': str(image_path),
          'image_bytes': image_path.stat().st_size,
      },
      'dimensions_returned': {
          'text': len(text_vector),
          'image': len(image_vector),
          'fused': len(fused_vector),
      },
      'norms': {
          'text': norm(text_vector),
          'image': norm(image_vector),
          'fused': norm(fused_vector),
      },
      'cosine_similarity': {
          'text_image': cosine(text_vector, image_vector),
          'text_fused': cosine(text_vector, fused_vector),
          'image_fused': cosine(image_vector, fused_vector),
      },
      'independent_usage': independent.get('usage'),
      'fused_usage': fused.get('usage'),
      'request_ids': {
          'independent': independent.get('request_id'),
          'fused': fused.get('request_id'),
      },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
      encoding='utf-8',
  )
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
