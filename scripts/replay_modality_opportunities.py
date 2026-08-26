#!/usr/bin/env python3
"""Replay ActionPiece training with opportunity-adjusted modality logging."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types

from analyze_modality_merges import analyze_records


ROOT = Path(__file__).parents[1]


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--item-feat-path', type=Path, required=True)
  parser.add_argument('--item-seqs-path', type=Path, required=True)
  parser.add_argument('--output-log-path', type=Path, required=True)
  parser.add_argument('--output-summary-path', type=Path, required=True)
  parser.add_argument('--target-vocab-size', type=int, default=40000)
  parser.add_argument('--text-slots', type=int, default=4)
  parser.add_argument('--image-slots', type=int, default=4)
  parser.add_argument('--hash-slot', type=int, default=8)
  parser.add_argument('--bins', type=int, default=10)
  parser.add_argument('--overwrite', action='store_true')
  return parser.parse_args()


def _load_actionpiece_core():
  """Load the lightweight core without importing optional training packages."""
  names = ('genrec', 'genrec.models', 'genrec.models.ActionPiece')
  saved = {name: sys.modules.get(name) for name in names}
  utils_name = 'genrec.models.ActionPiece.utils'
  saved_utils = sys.modules.get(utils_name)
  try:
    for name in names:
      package = types.ModuleType(name)
      package.__path__ = []
      sys.modules[name] = package
    utils_spec = importlib.util.spec_from_file_location(
        utils_name, ROOT / 'genrec/models/ActionPiece/utils.py'
    )
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules[utils_name] = utils_module
    utils_spec.loader.exec_module(utils_module)
    core_spec = importlib.util.spec_from_file_location(
        'actionpiece_core_for_replay',
        ROOT / 'genrec/models/ActionPiece/core.py',
    )
    core_module = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(core_module)
    return core_module.ActionPieceCore
  finally:
    for name, module in saved.items():
      if module is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = module
    if saved_utils is None:
      sys.modules.pop(utils_name, None)
    else:
      sys.modules[utils_name] = saved_utils


def _load_json(path: Path):
  with path.open(encoding='utf-8') as handle:
    return json.load(handle)


def main() -> None:
  args = parse_args()
  for output_path in (args.output_log_path, args.output_summary_path):
    if output_path.exists() and not args.overwrite:
      raise FileExistsError(
          f'{output_path} exists; pass --overwrite to replace it.'
      )
    output_path.parent.mkdir(parents=True, exist_ok=True)

  item2feat = _load_json(args.item_feat_path)
  all_item_seqs = _load_json(args.item_seqs_path)
  train_sequences = [
      sequence[:-2]
      for sequence in all_item_seqs.values()
      if len(sequence) > 2
  ]
  missing = sorted(
      {
          item
          for sequence in train_sequences
          for item in sequence
          if item not in item2feat
      }
  )
  if missing:
    raise ValueError(
        f'{len(missing)} training items are missing from item features; '
        f'first={missing[:5]}'
    )

  actionpiece_core = _load_actionpiece_core()
  actionpiece = actionpiece_core(state2feat=item2feat)
  actionpiece.train(
      state_corpus=train_sequences,
      target_vocab_size=args.target_vocab_size,
      merge_log_path=str(args.output_log_path),
      merge_log_interval=1,
      modality_slots={
          'text_slots': list(range(args.text_slots)),
          'image_slots': list(
              range(args.text_slots, args.text_slots + args.image_slots)
          ),
          'hash_slot': args.hash_slot,
      },
  )

  with args.output_log_path.open(encoding='utf-8') as handle:
    records = [json.loads(line) for line in handle]
  summary = analyze_records(
      records,
      text_slots=set(range(args.text_slots)),
      image_slots=set(
          range(args.text_slots, args.text_slots + args.image_slots)
      ),
      hash_slot=args.hash_slot,
      bins=args.bins,
      example_limit=3,
  )
  summary.update({
      'log_path': str(args.output_log_path),
      'item_feat_path': str(args.item_feat_path),
      'item_seqs_path': str(args.item_seqs_path),
      'target_vocab_size': args.target_vocab_size,
      'train_sequence_count': len(train_sequences),
      'item_count': len(item2feat),
      'malformed_records': 0,
  })
  with args.output_summary_path.open('w', encoding='utf-8') as handle:
    json.dump(summary, handle, ensure_ascii=False, indent=2)
    handle.write('\n')


if __name__ == '__main__':
  main()
