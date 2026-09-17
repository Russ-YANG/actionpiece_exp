"""Focused checks for post-tokenization E1 history/target length controls."""

from contextlib import nullcontext
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from genrec import utils  # Initializes the project's model/tokenizer imports.
from genrec.models.ActionPiece.core import ActionPieceCore
from genrec.models.ActionPiece.model import ActionPiece
from genrec.models.ActionPiece.tokenizer import ActionPieceTokenizer

del utils


FEATURES = {
    'a': [0, 10, 0],
    'b': [1, 11, 0],
    'c': [2, 12, 0],
}


class LengthControlTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.core = ActionPieceCore(state2feat=FEATURES)
    self.config = {
        'metadata': 'qwen_text', 'separate_image_semantic_ids': 'learned',
        'length_control_history_tokens_per_item': 2,
        'length_control_target_tokens': 2,
        'history_tokenization_scope': 'sequence',
        'target_tokenization': 'actionpiece',
        'history_granularity': 'legacy', 'history_eval_granularity': None,
        'train_shuffle': 'none', 'n_inference_ensemble': -1,
        'rand_seed': 7, 'max_item_seq_len': 20,
        'accelerator': SimpleNamespace(
            process_index=0, is_main_process=True,
            main_process_first=lambda: nullcontext(),
        ),
        'num_layers': 1, 'num_decoder_layers': 1, 'd_model': 16, 'd_ff': 32,
        'num_heads': 2, 'd_kv': 8, 'dropout_rate': 0.0,
        'activation_function': 'relu', 'feed_forward_proj': 'relu',
        'num_beams': 2,
    }
    path = str(Path(self.tmp.name) / 'tokenizer.json')

    def initialize(tokenizer, dataset):
      del dataset
      tokenizer.item2feat = FEATURES
      tokenizer.actionpiece_path = path
      return self.core

    with mock.patch.object(ActionPieceTokenizer, '_init_tokenizer', initialize):
      self.tokenizer = ActionPieceTokenizer(self.config, None)

  def example(self):
    states = {
        item: [self.core.rank[(slot, value)] for slot, value in enumerate(feats)]
        for item, feats in FEATURES.items()
    }
    return {'state_seq': np.asarray([states['a'], states['b'], states['c']])}

  def test_controls_are_post_tokenization_attention_visible_and_targeted(self):
    batch = self.tokenizer.collate_fn_train([self.example()])
    baseline_history = self.core.encode(
        self.example()['state_seq'][:-1], shuffle='none'
    )
    expected_input = (
        [self.tokenizer.bos_token]
        + baseline_history
        + list(self.tokenizer.length_control_history_tokens) * 2
        + [self.tokenizer.eos_token]
    )
    self.assertEqual(batch['input_ids'][0].tolist(), expected_input)
    self.assertTrue(batch['attention_mask'][0].bool().all())
    expected_target = (
        self.core.encode(self.example()['state_seq'][-1:], shuffle='none')
        + list(self.tokenizer.length_control_target_tokens)
        + [self.tokenizer.eos_token]
    )
    labels = batch['labels'][0].tolist()
    self.assertEqual(labels[:len(expected_target)], expected_target)
    self.assertTrue(all(value == -100 for value in labels[len(expected_target):]))

  def test_generation_requires_suffix_then_strips_it_for_item_evaluation(self):
    model = ActionPiece(self.config, None, self.tokenizer)
    batch = self.tokenizer.collate_fn_test([self.example()])
    target = self.core.encode(
        self.example()['state_seq'][-1:], shuffle='none'
    )
    generated = torch.tensor([[
        0,
        *target,
        *self.tokenizer.length_control_target_tokens,
        self.tokenizer.eos_token,
    ]])
    with mock.patch.object(model, 'beam_search', return_value=generated):
      output = model.generate(batch)
    self.assertEqual(output.shape, (1, 1, 3))
    self.assertEqual(
        output[0, 0].tolist(), self.example()['state_seq'][-1].tolist()
    )

  def test_decoder_uses_target_suffix_as_stop_sequence(self):
    model = ActionPiece(self.config, None, self.tokenizer)
    logits = torch.zeros(1, self.tokenizer.vocab_size)
    masked = model._mask_target_logits(
        logits, step=0, decoder_input_ids=torch.tensor([[0]])
    )
    self.assertTrue(torch.isneginf(
        masked[:, list(self.tokenizer.length_control_history_tokens)]
    ).all())
    self.assertTrue(torch.isneginf(masked[:, self.tokenizer.eos_token]).all())
    self.assertTrue(torch.isneginf(
        masked[:, self.tokenizer.length_control_target_tokens[1]]
    ).all())
    after_first = model._mask_target_logits(
        logits, step=1,
        decoder_input_ids=torch.tensor([
            [0, self.tokenizer.length_control_target_tokens[0]]
        ]),
    )
    allowed = self.tokenizer.length_control_target_tokens[1]
    self.assertTrue(torch.isfinite(after_first[:, allowed]).all())
    self.assertEqual(torch.isfinite(after_first).sum().item(), 1)


if __name__ == '__main__':
  unittest.main()
