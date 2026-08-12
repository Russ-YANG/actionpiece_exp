#!/usr/bin/env python3
"""Build ActionPiece mappings and Qwen metadata from downloaded Amazon data."""

from __future__ import annotations

import argparse
import ast
import collections
import gzip
import html
import json
from pathlib import Path
import re
from typing import Any, Iterable


TEXT_FIELDS = (
    'title',
    'price',
    'brand',
    'feature',
    'categories',
    'description',
)
IMAGE_FIELDS = ('imUrl', 'imageURLHighRes', 'imageURL')


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--category', required=True)
  parser.add_argument('--cache-dir', type=Path, default=Path('cache'))
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


def clean_text(raw: Any) -> str:
  if isinstance(raw, list):
    text = ', '.join(map(str, raw))
  else:
    text = str(raw)
  text = html.unescape(text.strip())
  text = re.sub(r'<[^>]+>', '', text)
  text = re.sub(r'[\n\t]', ' ', text)
  text = re.sub(r' +', ' ', text)
  return re.sub(r'[^\x00-\x7F]', ' ', text)


def sent_process(raw: Any) -> str:
  sentence = ''
  if isinstance(raw, float):
    sentence = f'{raw}.'
  elif isinstance(raw, list) and raw and isinstance(raw[0], list):
    flattened = [clean_text(value)[:-1] for group in raw for value in group]
    sentence = f'{", ".join(flattened)}.'
  elif isinstance(raw, list):
    sentence = ''.join(clean_text(value) for value in raw)
  else:
    sentence = clean_text(raw)
  return sentence + ' '


def product_text(metadata: dict[str, Any]) -> str:
  return ''.join(
      sent_process(metadata[field])
      for field in TEXT_FIELDS
      if field in metadata
  )


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


def main() -> None:
  args = parse_args()
  root = args.cache_dir / 'AmazonReviews2014' / args.category
  raw_dir = root / 'raw'
  processed_dir = root / 'processed'
  processed_dir.mkdir(parents=True, exist_ok=True)
  reviews_path = raw_dir / f'reviews_{args.category}_5.json.gz'
  metadata_path = raw_dir / f'meta_{args.category}.json.gz'

  user_events: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
  interactions = 0
  for record in parse_gzip_records(reviews_path):
    user = record['reviewerID']
    item = record['asin']
    timestamp = int(record['unixReviewTime'])
    user_events[user].append((item, timestamp))
    interactions += 1

  id_mapping = {
      'user2id': {'[PAD]': 0},
      'item2id': {'[PAD]': 0},
      'id2user': ['[PAD]'],
      'id2item': ['[PAD]'],
  }
  sequences: dict[str, list[str]] = {}
  for user, events in user_events.items():
    events.sort(key=lambda value: value[1])
    items = [item for item, _ in events]
    id_mapping['user2id'][user] = len(id_mapping['id2user'])
    id_mapping['id2user'].append(user)
    for item in items:
      if item not in id_mapping['item2id']:
        id_mapping['item2id'][item] = len(id_mapping['id2item'])
        id_mapping['id2item'].append(item)
    sequences[user] = items

  expected_items = set(id_mapping['id2item'][1:])
  text_metadata: dict[str, str] = {}
  fused_metadata: dict[str, dict[str, str | None]] = {}
  for record in parse_gzip_records(metadata_path):
    item = record.get('asin')
    if item not in expected_items:
      continue
    text = product_text(record)
    image_url = first_image_url(record)
    text_metadata[item] = text
    fused_metadata[item] = {'sentence': text, 'image_url': image_url}

  missing = expected_items - set(text_metadata)
  if missing:
    raise RuntimeError(
        f'Metadata is missing for {len(missing)} mapped items; refusing to '
        'create misaligned Qwen inputs.'
    )

  outputs = {
      'all_item_seqs.json': sequences,
      'id_mapping.json': id_mapping,
      'item_sources.json': {
          item: [args.category] for item in id_mapping['id2item'][1:]
      },
      'metadata.qwen_text.json': text_metadata,
      'metadata.qwen_fused.json': fused_metadata,
  }
  for filename, value in outputs.items():
    path = processed_dir / filename
    path.write_text(
        json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8'
    )

  summary = {
      'category': args.category,
      'users': len(user_events),
      'items': len(expected_items),
      'interactions': interactions,
      'metadata_items': len(text_metadata),
  }
  print(json.dumps(summary, indent=2))


if __name__ == '__main__':
  main()
