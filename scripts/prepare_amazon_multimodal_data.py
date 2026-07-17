#!/usr/bin/env python3
"""Prepare local text and image data for an Amazon Reviews category.

The script is deliberately independent of the training stack so data can be
prepared on machines without PyTorch, Accelerate, or Faiss. It reads the raw
Amazon Reviews 2014 files used by this repository, retains products occurring
in the review data, writes a deterministic JSONL catalog, and optionally
downloads product images with resumable status tracking.
"""

import argparse
import ast
import concurrent.futures
import gzip
import hashlib
import html
import io
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterable

from PIL import Image
import requests


TEXT_FIELDS = (
    'title',
    'price',
    'brand',
    'feature',
    'categories',
    'description',
)
IMAGE_FIELDS = ('imUrl', 'imageURLHighRes', 'imageURL')
FORMAT_EXTENSIONS = {
    'BMP': '.bmp',
    'GIF': '.gif',
    'JPEG': '.jpg',
    'PNG': '.png',
    'TIFF': '.tiff',
    'WEBP': '.webp',
}
_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Prepare text and local images from Amazon Reviews 2014.'
  )
  parser.add_argument('--category', default='Beauty')
  parser.add_argument('--cache-dir', type=Path, default=Path('cache'))
  parser.add_argument('--download-images', action='store_true')
  parser.add_argument('--workers', type=int, default=8)
  parser.add_argument('--timeout', type=float, default=15.0)
  parser.add_argument(
      '--retry-failures',
      action='store_true',
      help='Retry items already recorded as failed in the image manifest.',
  )
  return parser.parse_args()


def parse_gzip_records(path: Path) -> Iterable[dict[str, Any]]:
  with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as stream:
    for line in stream:
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        try:
          yield ast.literal_eval(
              line.replace('true', 'True')
              .replace('false', 'False')
              .replace('null', 'None')
          )
        except (SyntaxError, ValueError):
          continue


def collect_review_items(path: Path) -> tuple[set[str], int, int]:
  items = set()
  users = set()
  interactions = 0
  for record in parse_gzip_records(path):
    asin = record.get('asin')
    user = record.get('reviewerID')
    if isinstance(asin, str) and asin:
      items.add(asin)
    if isinstance(user, str) and user:
      users.add(user)
    interactions += 1
  return items, len(users), interactions


def flatten_text(value: Any) -> str:
  if value is None:
    return ''
  if isinstance(value, (str, int, float)):
    return str(value).strip()
  if isinstance(value, dict):
    return ' '.join(
        part for part in (flatten_text(v) for v in value.values()) if part
    )
  if isinstance(value, (list, tuple)):
    return ' '.join(part for part in map(flatten_text, value) if part)
  return str(value).strip()


def clean_text(value: Any) -> str:
  """Mirror the repository's current ASCII metadata cleaning."""
  if isinstance(value, list):
    text = ', '.join(map(str, value))
  else:
    text = str(value)
  text = html.unescape(text).strip()
  text = re.sub(r'<[^>]+>', '', text)
  text = re.sub(r'[\n\t]', ' ', text)
  text = re.sub(r' +', ' ', text)
  return re.sub(r'[^\x00-\x7F]', ' ', text)


def upstream_field_text(value: Any) -> str:
  """Mirror AmazonReviews2014._sent_process for strict comparisons."""
  sentence = ''
  if isinstance(value, float):
    sentence = f'{value}.'
  elif isinstance(value, list) and value and isinstance(value[0], list):
    # Preserve the upstream implementation exactly, including its final-character
    # slicing for nested category values.
    flattened = [clean_text(item)[:-1] for group in value for item in group]
    sentence = f'{", ".join(flattened)}.'
  elif isinstance(value, list):
    sentence = ''.join(clean_text(item) for item in value)
  else:
    sentence = clean_text(value)
  return sentence + ' '


def upstream_product_text(metadata: dict[str, Any]) -> str:
  return ''.join(
      upstream_field_text(metadata[field])
      for field in TEXT_FIELDS
      if field in metadata
  )


def structured_product_text(metadata: dict[str, Any]) -> str:
  """Create a field-labelled variant for a separate prompt ablation."""
  parts = []
  for field in TEXT_FIELDS:
    value = flatten_text(metadata.get(field))
    if value:
      parts.append(f'{field.capitalize()}: {value}')
  return '\n'.join(parts)


def first_image_url(metadata: dict[str, Any]) -> str | None:
  for field in IMAGE_FIELDS:
    value = metadata.get(field)
    if isinstance(value, str) and value:
      return value
    if isinstance(value, list):
      for candidate in value:
        if isinstance(candidate, str) and candidate:
          return candidate
  return None


def write_catalog(
    metadata_path: Path, review_items: set[str], catalog_path: Path
) -> tuple[list[dict[str, Any]], int]:
  records = []
  found_items = set()
  for metadata in parse_gzip_records(metadata_path):
    asin = metadata.get('asin')
    if asin not in review_items:
      continue
    found_items.add(asin)
    records.append({
        'asin': asin,
        'text': upstream_product_text(metadata),
        'structured_text': structured_product_text(metadata),
        'image_url': first_image_url(metadata),
    })
  records.sort(key=lambda record: record['asin'])
  with catalog_path.open('w', encoding='utf-8') as output:
    for record in records:
      output.write(json.dumps(record, ensure_ascii=False) + '\n')
  return records, len(review_items - found_items)


def load_status(path: Path) -> dict[str, dict[str, Any]]:
  if not path.exists():
    return {}
  status = {}
  with path.open(encoding='utf-8') as stream:
    for line in stream:
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        continue
      if isinstance(record.get('asin'), str):
        status[record['asin']] = record
  return status


def session() -> requests.Session:
  if not hasattr(_thread_local, 'session'):
    current = requests.Session()
    current.headers['User-Agent'] = 'ActionPiece research data preparation/1.0'
    _thread_local.session = current
  return _thread_local.session


def download_image(
    record: dict[str, Any], image_dir: Path, timeout: float
) -> dict[str, Any]:
  asin = record['asin']
  url = record.get('image_url')
  result = {'asin': asin, 'image_url': url}
  if not url:
    return {**result, 'status': 'missing_url'}
  try:
    response = session().get(url, timeout=timeout)
    response.raise_for_status()
    content = response.content
    with Image.open(io.BytesIO(content)) as image:
      image.verify()
      image_format = image.format
      width, height = image.size
    extension = FORMAT_EXTENSIONS.get(image_format, '.img')
    digest = hashlib.sha256(content).hexdigest()
    destination = image_dir / f'{asin}{extension}'
    destination.write_bytes(content)
    return {
        **result,
        'status': 'downloaded',
        'path': str(destination),
        'format': image_format,
        'width': width,
        'height': height,
        'bytes': len(content),
        'sha256': digest,
    }
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return {**result, 'status': 'failed', 'reason': repr(exc)}


def save_status(path: Path, status: dict[str, dict[str, Any]]) -> None:
  temporary = path.with_suffix(path.suffix + '.tmp')
  with temporary.open('w', encoding='utf-8') as output:
    for asin in sorted(status):
      output.write(json.dumps(status[asin], ensure_ascii=False) + '\n')
  temporary.replace(path)


def download_images(
    records: list[dict[str, Any]],
    image_dir: Path,
    manifest_path: Path,
    workers: int,
    timeout: float,
    retry_failures: bool,
) -> dict[str, dict[str, Any]]:
  status = load_status(manifest_path)
  pending = []
  for record in records:
    previous = status.get(record['asin'])
    if previous and previous.get('status') == 'downloaded':
      previous_path = previous.get('path')
      if previous_path and Path(previous_path).exists():
        continue
    if previous and previous.get('status') in {'failed', 'missing_url'}:
      if not retry_failures:
        continue
    pending.append(record)

  completed_since_save = 0
  with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    futures = {
        executor.submit(download_image, record, image_dir, timeout): record
        for record in pending
    }
    for index, future in enumerate(
        concurrent.futures.as_completed(futures), start=1
    ):
      result = future.result()
      status[result['asin']] = result
      completed_since_save += 1
      if completed_since_save >= 100:
        save_status(manifest_path, status)
        completed_since_save = 0
      if index % 250 == 0 or index == len(pending):
        print(f'Images processed: {index}/{len(pending)}', flush=True)
  save_status(manifest_path, status)
  return status


def main() -> None:
  args = parse_args()
  root = args.cache_dir / 'AmazonReviews2014' / args.category
  raw_dir = root / 'raw'
  processed_dir = root / 'processed'
  image_dir = root / 'images'
  processed_dir.mkdir(parents=True, exist_ok=True)
  image_dir.mkdir(parents=True, exist_ok=True)

  review_path = raw_dir / f'reviews_{args.category}_5.json.gz'
  metadata_path = raw_dir / f'meta_{args.category}.json.gz'
  for path in (review_path, metadata_path):
    if not path.exists():
      raise FileNotFoundError(f'Missing raw dataset file: {path}')

  review_items, users, interactions = collect_review_items(review_path)
  catalog_path = processed_dir / 'multimodal_catalog.jsonl'
  records, metadata_missing = write_catalog(
      metadata_path, review_items, catalog_path
  )
  with_image_url = sum(bool(record['image_url']) for record in records)
  with_text = sum(bool(record['text']) for record in records)

  summary = {
      'category': args.category,
      'users': users,
      'interactions': interactions,
      'review_items': len(review_items),
      'catalog_items': len(records),
      'items_missing_metadata': metadata_missing,
      'items_with_text': with_text,
      'items_with_image_url': with_image_url,
      'catalog_path': str(catalog_path),
  }

  if args.download_images:
    manifest_path = processed_dir / 'image_download_manifest.jsonl'
    status = download_images(
        records,
        image_dir,
        manifest_path,
        max(args.workers, 1),
        args.timeout,
        args.retry_failures,
    )
    counts: dict[str, int] = {}
    for result in status.values():
      name = result.get('status', 'unknown')
      counts[name] = counts.get(name, 0) + 1
    summary['image_status'] = counts
    summary['image_manifest_path'] = str(manifest_path)

  summary_path = processed_dir / 'multimodal_summary.json'
  summary_path.write_text(
      json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
      encoding='utf-8',
  )
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
