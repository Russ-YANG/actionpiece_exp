#!/usr/bin/env python3
"""Summarize modality composition in an ActionPiece merge log."""

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Analyze text/image/hash merges in an ActionPiece JSONL log.'
  )
  parser.add_argument('log_path', type=Path, help='Path to merge_log.jsonl')
  parser.add_argument(
      '--text-slots',
      type=int,
      default=4,
      help='Number of text feature slots starting at 0 (default: 4).',
  )
  parser.add_argument(
      '--image-slots',
      type=int,
      default=4,
      help='Number of image feature slots after text slots (default: 4).',
  )
  parser.add_argument(
      '--hash-slot',
      type=int,
      default=8,
      help='Hash feature slot, or -1 to disable (default: 8).',
  )
  parser.add_argument(
      '--examples',
      type=int,
      default=3,
      help='Maximum example merge records shown per type (default: 3).',
  )
  return parser.parse_args()


def classify(
    features: list[list[int]],
    text_slots: set[int],
    image_slots: set[int],
    hash_slot: int,
) -> tuple[str, bool]:
  slots = {feature[0] for feature in features if feature}
  has_text = bool(slots & text_slots)
  has_image = bool(slots & image_slots)
  has_hash = hash_slot >= 0 and hash_slot in slots
  known_slots = text_slots | image_slots
  if hash_slot >= 0:
    known_slots.add(hash_slot)
  has_other = bool(slots - known_slots)

  if has_text and has_image:
    merge_type = 'text_image'
  elif has_text:
    merge_type = 'text_only'
  elif has_image:
    merge_type = 'image_only'
  elif has_hash:
    merge_type = 'hash_only'
  else:
    merge_type = 'unknown'

  if has_other:
    merge_type += '_with_unknown'
  return merge_type, has_hash


def format_pct(count: int, total: int) -> str:
  return f'{100 * count / total:.2f}%' if total else '0.00%'


def main() -> None:
  args = parse_args()
  text_slots = set(range(args.text_slots))
  image_start = args.text_slots
  image_slots = set(range(image_start, image_start + args.image_slots))

  counts: collections.Counter[str] = collections.Counter()
  first_steps: dict[str, int] = {}
  examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  hash_count = 0
  malformed_count = 0
  merge_count = 0

  with args.log_path.open(encoding='utf-8') as log_file:
    for line_number, line in enumerate(log_file, start=1):
      try:
        record = json.loads(line)
      except json.JSONDecodeError:
        malformed_count += 1
        continue
      if record.get('event') != 'merge':
        continue

      features = record.get('new_token_basic_features')
      if not isinstance(features, list):
        malformed_count += 1
        continue

      merge_count += 1
      merge_type, has_hash = classify(
          features, text_slots, image_slots, args.hash_slot
      )
      counts[merge_type] += 1
      hash_count += int(has_hash)
      first_steps.setdefault(merge_type, record.get('step', line_number))
      if len(examples[merge_type]) < args.examples:
        examples[merge_type].append({
            'step': record.get('step'),
            'new_token': record.get('new_token'),
            'features': features,
        })

  print(f'Log: {args.log_path}')
  print(f'Total merge events: {merge_count}')
  print(
      f'Slots: text={sorted(text_slots)}, image={sorted(image_slots)}, '
      f'hash={args.hash_slot if args.hash_slot >= 0 else "disabled"}'
  )
  print()
  print(f'{"Type":<24} {"Count":>10} {"Percent":>10} {"First step":>12}')
  print('-' * 60)
  for merge_type, count in counts.most_common():
    print(
        f'{merge_type:<24} {count:>10} {format_pct(count, merge_count):>10} '
        f'{first_steps[merge_type]:>12}'
    )
  print('-' * 60)
  print(
      f'Includes hash token: {hash_count} '
      f'({format_pct(hash_count, merge_count)})'
  )
  if malformed_count:
    print(f'Skipped malformed records: {malformed_count}')

  if args.examples:
    print('\nExamples:')
    for merge_type in counts:
      print(f'  {merge_type}:')
      for example in examples[merge_type]:
        print(
            f'    step={example["step"]}, token={example["new_token"]}, '
            f'features={example["features"]}'
        )


if __name__ == '__main__':
  main()
