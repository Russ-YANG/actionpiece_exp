#!/usr/bin/env python3
"""Losslessly pack adjacent semantic-ID digits into larger integers."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('input_path', type=Path)
  parser.add_argument('output_path', type=Path)
  parser.add_argument('--group-size', type=int, default=2)
  parser.add_argument('--base', type=int, default=256)
  return parser.parse_args()


def pack_digits(digits: list[int], group_size: int, base: int) -> list[int]:
  if len(digits) % group_size:
    raise ValueError(
        f'Code length {len(digits)} is not divisible by {group_size}.'
    )
  packed = []
  for start in range(0, len(digits), group_size):
    value = 0
    for digit in digits[start : start + group_size]:
      if not isinstance(digit, int) or not 0 <= digit < base:
        raise ValueError(f'Invalid base-{base} digit: {digit!r}')
      value = value * base + digit
    packed.append(value)
  return packed


def main() -> None:
  args = parse_args()
  with args.input_path.open(encoding='utf-8') as input_file:
    semantic_ids = json.load(input_file)

  packed_ids = {
      item: pack_digits(codes, args.group_size, args.base)
      for item, codes in semantic_ids.items()
  }
  args.output_path.parent.mkdir(parents=True, exist_ok=True)
  with args.output_path.open('w', encoding='utf-8') as output_file:
    json.dump(packed_ids, output_file)

  lengths = {len(codes) for codes in packed_ids.values()}
  values = [value for codes in packed_ids.values() for value in codes]
  print(f'Items: {len(packed_ids)}')
  print(f'Packed lengths: {sorted(lengths)}')
  print(f'Packed value range: [{min(values)}, {max(values)}]')
  print(f'Output: {args.output_path}')


if __name__ == '__main__':
  main()
