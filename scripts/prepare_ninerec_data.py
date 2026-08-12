#!/usr/bin/env python3
"""Convert an official NineRec subset into ActionPiece cache artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))
from genrec.datasets.NineRec.dataset import build_ninerec_processed


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument('--subset', required=True, help='For example DY or Bili_Music.')
  parser.add_argument(
      '--source-dir', type=Path, required=True,
      help='Directory containing the official NineRec subset files.',
  )
  parser.add_argument('--cache-dir', type=Path, default=Path('cache'))
  parser.add_argument(
      '--text-language', choices=('zh', 'en', 'bilingual'), default='bilingual'
  )
  parser.add_argument(
      '--pair-order',
      choices=('item-user-time', 'user-item-time'),
      default='item-user-time',
      help='Column order when only *_pair.csv is available.',
  )
  parser.add_argument('--behaviour-path', type=Path)
  parser.add_argument('--pair-path', type=Path)
  parser.add_argument('--item-path', type=Path)
  parser.add_argument('--cover-dir', type=Path)
  parser.add_argument('--require-all-images', action='store_true')
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  output_dir = args.cache_dir / 'NineRec' / args.subset / 'processed'
  summary = build_ninerec_processed(
      source_dir=args.source_dir,
      output_dir=output_dir,
      subset=args.subset,
      text_language=args.text_language,
      pair_order=args.pair_order,
      require_all_images=args.require_all_images,
      behaviour_path=args.behaviour_path,
      pair_path=args.pair_path,
      item_path=args.item_path,
      cover_dir=args.cover_dir,
  )
  print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
  main()
