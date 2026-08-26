"""Tests for opportunity-adjusted modality tracking during tokenizer training."""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).parents[1]
MODULE_NAMES = ('genrec', 'genrec.models', 'genrec.models.ActionPiece')
saved_modules = {name: sys.modules.get(name) for name in MODULE_NAMES}
saved_utils = sys.modules.get('genrec.models.ActionPiece.utils')
try:
  for name in MODULE_NAMES:
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
  utils_spec = importlib.util.spec_from_file_location(
      'genrec.models.ActionPiece.utils',
      ROOT / 'genrec/models/ActionPiece/utils.py',
  )
  actionpiece_utils = importlib.util.module_from_spec(utils_spec)
  sys.modules['genrec.models.ActionPiece.utils'] = actionpiece_utils
  utils_spec.loader.exec_module(actionpiece_utils)
  core_spec = importlib.util.spec_from_file_location(
      'actionpiece_core_for_test',
      ROOT / 'genrec/models/ActionPiece/core.py',
  )
  actionpiece_core = importlib.util.module_from_spec(core_spec)
  core_spec.loader.exec_module(actionpiece_core)
  ActionPieceCore = actionpiece_core.ActionPieceCore
  diff_cnt = actionpiece_core.diff_cnt
finally:
  for name, module in saved_modules.items():
    if module is None:
      sys.modules.pop(name, None)
    else:
      sys.modules[name] = module
  if saved_utils is None:
    sys.modules.pop('genrec.models.ActionPiece.utils', None)
  else:
    sys.modules['genrec.models.ActionPiece.utils'] = saved_utils


class ActionPieceModalityOpportunitiesTest(unittest.TestCase):

  def test_pair_count_diff_reports_pairs_that_disappear(self):
    old_counts = {(1, 2): 0.5, (2, 3): 1.0}
    new_counts = {(2, 3): 0.25, (3, 4): 2.0}

    self.assertEqual(
        diff_cnt(new_counts, old_counts),
        {(1, 2): -0.5, (2, 3): -0.75, (3, 4): 2.0},
    )

  def test_logs_candidate_exposure_before_selection(self):
    actionpiece = ActionPieceCore(state2feat={'item': [10, 11, 20, 21, 30]})
    with tempfile.TemporaryDirectory() as directory:
      log_path = Path(directory) / 'merge.jsonl'
      actionpiece.train(
          state_corpus=[['item']],
          target_vocab_size=actionpiece.n_init_feats + 1,
          merge_log_path=str(log_path),
          modality_slots={
              'text_slots': [0, 1],
              'image_slots': [2, 3],
              'hash_slot': 4,
          },
      )
      records = [json.loads(line) for line in log_path.read_text().splitlines()]

    start = records[0]
    merge = records[1]
    self.assertEqual(
        start['modality_slots'],
        {'text_slots': [0, 1], 'image_slots': [2, 3], 'hash_slot': 4},
    )
    opportunities = merge['modality_opportunities']
    self.assertEqual(
        opportunities['candidate_pair_counts'],
        {
            'hash_related': 4,
            'image_image': 1,
            'text_image': 4,
            'text_text': 1,
        },
    )
    self.assertEqual(opportunities['selected_category'], 'text_text')
    self.assertAlmostEqual(
        sum(opportunities['candidate_priority_mass'].values()), 4.0
    )


if __name__ == '__main__':
  unittest.main()
