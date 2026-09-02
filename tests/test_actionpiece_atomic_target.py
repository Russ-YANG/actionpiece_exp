"""Tests for the fixed-slot atomic target grammar."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


ROOT = Path(__file__).parents[1]
MODULE_NAMES = ('genrec', 'genrec.models', 'genrec.models.ActionPiece')
saved_modules = {name: sys.modules.get(name) for name in MODULE_NAMES}
saved_model = sys.modules.get('genrec.model')
try:
  for name in MODULE_NAMES:
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
  abstract_model_module = types.ModuleType('genrec.model')
  abstract_model_module.AbstractModel = object
  sys.modules['genrec.model'] = abstract_model_module
  model_spec = importlib.util.spec_from_file_location(
      'actionpiece_model_for_test',
      ROOT / 'genrec/models/ActionPiece/model.py',
  )
  actionpiece_model = importlib.util.module_from_spec(model_spec)
  model_spec.loader.exec_module(actionpiece_model)
  ActionPiece = actionpiece_model.ActionPiece
finally:
  for name, module in saved_modules.items():
    if module is None:
      sys.modules.pop(name, None)
    else:
      sys.modules[name] = module
  if saved_model is None:
    sys.modules.pop('genrec.model', None)
  else:
    sys.modules['genrec.model'] = saved_model


class _AtomicTargetTokenizer:

  def target_allowed_tokens(self, step):
    del step
    return (3,)


class ActionPieceAtomicTargetTest(unittest.TestCase):

  def test_beam_step_masks_tokens_outside_fixed_slot(self):
    model = object.__new__(ActionPiece)
    model.tokenizer = _AtomicTargetTokenizer()
    logits = torch.zeros(2, 1, 6)
    logits[:, :, 5] = 100.0
    decoder_input_ids = torch.zeros(2, 1, dtype=torch.long)
    decoder_output, _ = model.beam_search_step(
        logits=logits,
        decoder_input_ids=decoder_input_ids,
        beam_scores=torch.tensor([0.0, -1e9]),
        beam_idx_offset=torch.tensor([0, 0]),
        batch_size=1,
        num_beams=2,
    )

    self.assertEqual(decoder_output[:, -1].tolist(), [3, 3])


if __name__ == '__main__':
  unittest.main()
