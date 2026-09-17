#!/usr/bin/env python3
"""Measure the catalog-level token-length gap between two ActionPiece models."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

if repo_root := os.environ.get('ACTIONPIECE_REPO'):
  sys.path.insert(0, repo_root)

from genrec import utils as _genrec_utils  # Avoid the package import cycle.
from genrec.models.ActionPiece.core import ActionPieceCore


def _single_item_length(tokenizer: ActionPieceCore, features) -> int:
  tokens = [
      tokenizer.rank[(slot, feature)]
      for slot, feature in enumerate(features)
  ]
  while True:
    best = None
    for left in range(len(tokens)):
      for right in range(left + 1, len(tokens)):
        pair = (-1, min(tokens[left], tokens[right]), max(tokens[left], tokens[right]))
        merged = tokenizer.rank.get(pair)
        if merged is None:
          continue
        score = tokenizer.priority[merged]
        if best is None or score > best[0]:
          best = (score, left, right, merged)
    if best is None:
      return len(tokens)
    _, left, right, merged = best
    tokens = [merged] + [
        token for index, token in enumerate(tokens)
        if index not in (left, right)
    ]


def _lengths(tokenizer_path: Path, features_path: Path) -> np.ndarray:
  tokenizer = ActionPieceCore.from_pretrained(str(tokenizer_path))
  with features_path.open() as f:
    item2feat = json.load(f)
  return np.asarray([
      _single_item_length(tokenizer, features)
      for features in item2feat.values()
  ])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--short-tokenizer', type=Path, required=True)
  parser.add_argument('--short-features', type=Path, required=True)
  parser.add_argument('--long-tokenizer', type=Path, required=True)
  parser.add_argument('--long-features', type=Path, required=True)
  args = parser.parse_args()

  short = _lengths(args.short_tokenizer, args.short_features)
  long = _lengths(args.long_tokenizer, args.long_features)
  gap = float(long.mean() - short.mean())
  print(f'short: n={len(short)} mean={short.mean():.6f} median={np.median(short):.1f} range=[{short.min()}, {short.max()}]')
  print(f'long:  n={len(long)} mean={long.mean():.6f} median={np.median(long):.1f} range=[{long.min()}, {long.max()}]')
  print(f'mean_gap={gap:.6f} rounded_fixed_padding={round(gap)}')


if __name__ == '__main__':
  main()
