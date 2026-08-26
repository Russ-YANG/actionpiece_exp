#!/usr/bin/env python3
"""Build a lossless Pack2 layout chosen by training-set pair frequency."""

import argparse
import collections
import json
from functools import lru_cache
from pathlib import Path


SLOT_NAMES = ('t1', 't2', 't3', 't4', 'i1', 'i2', 'i3', 'i4')


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--text-sem-ids', type=Path, required=True)
  parser.add_argument('--image-sem-ids', type=Path, required=True)
  parser.add_argument('--sequences', type=Path, required=True)
  parser.add_argument('--output-stream-a', type=Path, required=True)
  parser.add_argument('--output-stream-b', type=Path, required=True)
  parser.add_argument('--output-metadata', type=Path, required=True)
  parser.add_argument('--base', type=int, default=256)
  return parser.parse_args()


def best_matching(edge_scores: dict[tuple[int, int], int]):
  @lru_cache(None)
  def solve(nodes: tuple[int, ...]):
    if not nodes:
      return 0, ()
    first = nodes[0]
    best_score = -1
    best_pairs = ()
    for index in range(1, len(nodes)):
      second = nodes[index]
      remaining = nodes[1:index] + nodes[index + 1 :]
      score, pairs = solve(remaining)
      score += edge_scores[(first, second)]
      candidate_pairs = ((first, second),) + pairs
      if score > best_score or (
          score == best_score and candidate_pairs < best_pairs
      ):
        best_score, best_pairs = score, candidate_pairs
    return best_score, best_pairs

  return solve(tuple(range(8)))


def main() -> None:
  args = parse_args()
  text_ids = json.loads(args.text_sem_ids.read_text(encoding='utf-8'))
  image_ids = json.loads(args.image_sem_ids.read_text(encoding='utf-8'))
  sequences = json.loads(args.sequences.read_text(encoding='utf-8'))
  if text_ids.keys() != image_ids.keys():
    raise ValueError('Text and image item keys do not align.')

  item_codes = {
      item: list(text_ids[item]) + list(image_ids[item]) for item in text_ids
  }
  if any(len(codes) != 8 for codes in item_codes.values()):
    raise ValueError('Expected four text and four image codes per item.')

  # Match the tokenizer training split: leave the last two interactions out.
  item_frequency = collections.Counter(
      item for sequence in sequences.values() for item in sequence[:-2]
  )
  pair_counters = {}
  edge_scores = {}
  for left in range(8):
    for right in range(left + 1, 8):
      counts = collections.Counter()
      for item, frequency in item_frequency.items():
        codes = item_codes[item]
        counts[(codes[left], codes[right])] += frequency
      pair_counters[(left, right)] = counts
      edge_scores[(left, right)] = max(counts.values())

  total_score, matching = best_matching(edge_scores)
  matching = tuple(sorted(matching))

  packed = {}
  for item, codes in item_codes.items():
    packed[item] = [
        codes[left] * args.base + codes[right]
        for left, right in matching
    ]
  stream_a = {item: values[:2] for item, values in packed.items()}
  stream_b = {item: values[2:] for item, values in packed.items()}

  for path, values in (
      (args.output_stream_a, stream_a),
      (args.output_stream_b, stream_b),
  ):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values), encoding='utf-8')

  metadata = {
      'method': 'maximum_weight_perfect_matching',
      'edge_weight': 'maximum training-interaction frequency of a value pair',
      'training_split': 'all_item_seqs[:-2]',
      'base': args.base,
      'total_matching_score': total_score,
      'matching': [
          {
              'slots': [SLOT_NAMES[left], SLOT_NAMES[right]],
              'slot_indices': [left, right],
              'max_value_pair_frequency': edge_scores[(left, right)],
              'unique_value_pairs': len(pair_counters[(left, right)]),
          }
          for left, right in matching
      ],
      'stream_a_pairs': [list(pair) for pair in matching[:2]],
      'stream_b_pairs': [list(pair) for pair in matching[2:]],
  }
  args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
  args.output_metadata.write_text(
      json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8'
  )

  print(f'Matching score: {total_score}')
  for entry in metadata['matching']:
    print(
        f'{entry["slots"][0]}-{entry["slots"][1]}: '
        f'max frequency={entry["max_value_pair_frequency"]}, '
        f'unique pairs={entry["unique_value_pairs"]}'
    )
  print(f'Items: {len(packed)}')


if __name__ == '__main__':
  main()
