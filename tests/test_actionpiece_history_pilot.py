"""Behavioral checks for candidate paths, reproducible collation and atomic CE."""

from contextlib import nullcontext
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

# Initialize the existing project's utils/model/tokenizer import cycle in the
# same order used by Pipeline, without loading any data or embedding model.
from genrec import utils
from genrec.models.ActionPiece.core import ActionPieceCore
from genrec.models.ActionPiece.history import ItemHistoryRepresentations
from genrec.models.ActionPiece.model import ActionPiece
from genrec.models.ActionPiece.tokenizer import ActionPieceTokenizer


FEATURES = {
    'a': [0, 10, 20, 0],
    'b': [0, 10, 21, 1],
    'c': [1, 11, 20, 0],
    'd': [1, 11, 21, 1],
}


class HistoryPilotTest(unittest.TestCase):

  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.core = ActionPieceCore(state2feat=FEATURES)
    self.core.train(
        state_corpus=[['a', 'b', 'a', 'c', 'd']] * 3,
        target_vocab_size=self.core.n_init_feats + 3,
        allow_cross_action_merges=False, allowed_merge_slots=[0, 1, 2],
    )
    self.states = {
        item: [self.core.rank[(slot, value)] for slot, value in enumerate(feats)]
        for item, feats in FEATURES.items()
    }
    self.config = {
        'history_tokenization_scope': 'item', 'target_tokenization': 'atomic',
        'actionpiece_merge_hash': False, 'history_granularity': 'random',
        'history_eval_granularity': 'random', 'history_eval_seed': 91,
        'train_shuffle': 'none', 'n_inference_ensemble': -1, 'rand_seed': 7,
        'max_item_seq_len': 20,
        'accelerator': SimpleNamespace(
            process_index=0, is_main_process=True,
            main_process_first=lambda: nullcontext(),
        ),
        'num_layers': 1, 'num_decoder_layers': 1, 'd_model': 16, 'd_ff': 32,
        'num_heads': 2, 'd_kv': 8, 'dropout_rate': 0.0,
        'activation_function': 'relu', 'feed_forward_proj': 'relu', 'num_beams': 2,
    }

  def tokenizer(self, **overrides):
    config = {**self.config, **overrides}
    core = self.core
    path = str(Path(self.tmp.name) / 'tokenizer.json')

    def initialize(tokenizer, dataset):
      del dataset
      tokenizer.item2feat = FEATURES
      tokenizer.actionpiece_path = path
      return core

    with mock.patch.object(ActionPieceTokenizer, '_init_tokenizer', initialize):
      return ActionPieceTokenizer(config, None)

  def examples(self):
    return [
        {'state_seq': np.asarray([self.states[i] for i in sequence])}
        for sequence in [('a', 'b', 'c'), ('d', 'c', 'b', 'a')]
    ]

  def test_paths_recover_features_and_keep_hash_atomic(self):
    paths = ItemHistoryRepresentations(self.core, self.states.values())
    for item, state in self.states.items():
      path = paths.paths[tuple(state)]
      self.assertEqual(path[0], tuple(state))
      self.assertEqual(list(path[-1]), self.core.encode(np.asarray([state]), shuffle='none'))
      self.assertEqual([len(p) for p in path], list(range(len(state), len(path[-1]) - 1, -1)))
      for candidate in path:
        self.assertEqual(self.core.decode_single_state(candidate), list(enumerate(FEATURES[item])))
        self.assertIn(state[-1], candidate)
    self.assertGreater(paths.summary()['n_with_multiple_lengths'], 0)
    with self.assertRaises(ValueError):
      self.core.encode(np.asarray(list(self.states.values())), shuffle='none', return_merge_path=True)

  def test_cache_reuse_and_source_change(self):
    path = Path(self.tmp.name) / 'candidates.json'
    original = ItemHistoryRepresentations(self.core, self.states.values(), path)
    with mock.patch.object(self.core, 'encode', side_effect=AssertionError('cache miss')):
      loaded = ItemHistoryRepresentations(self.core, self.states.values(), path)
      self.assertEqual(original.paths, loaded.paths)
    self.core.priority[-1] += 1
    with mock.patch.object(self.core, 'encode', wraps=self.core.encode) as encode:
      ItemHistoryRepresentations(self.core, self.states.values(), path)
      self.assertEqual(encode.call_count, len(self.states))

  def test_policies_preserve_target_and_test_batch_alignment(self):
    batches = []
    for policy in ('raw', 'full', 'middle', 'random'):
      tokenizer = self.tokenizer(history_granularity=policy, history_eval_granularity=policy)
      training = tokenizer.collate_fn_train(self.examples())
      batches.append(training)
      for collate in (tokenizer.collate_fn_val, tokenizer.collate_fn_test):
        batch = collate(self.examples())
        self.assertEqual(batch['input_ids'].shape[0], 2)
        self.assertEqual(batch['labels'].shape, (2, 4))
        self.assertEqual(batch['labels'].tolist(), [self.states['c'], self.states['a']])
    for batch in batches:
      self.assertTrue(torch.equal(batch['labels'], batches[0]['labels']))
      self.assertEqual(batch['labels'].shape, (2, 5))
    self.assertTrue(torch.all(batches[1]['attention_mask'].sum(1) <= batches[0]['attention_mask'].sum(1)))

  def test_random_evaluation_is_history_only_and_does_not_consume_training_rng(self):
    tokenizer = self.tokenizer()
    examples = self.examples()
    first = tokenizer.collate_fn_val(examples)
    rng_before = copy.deepcopy(tokenizer.history_train_rng.bit_generator.state)
    second = tokenizer.collate_fn_val(examples[::-1])
    self.assertEqual(rng_before, tokenizer.history_train_rng.bit_generator.state)
    for i in range(2):
      first_tokens = first['input_ids'][i][first['attention_mask'][i].bool()]
      second_tokens = second['input_ids'][1-i][second['attention_mask'][1-i].bool()]
      self.assertTrue(torch.equal(first_tokens, second_tokens))
    examples[0]['state_seq'][-1] = self.states['d']
    changed_target = tokenizer.collate_fn_val(examples)
    self.assertTrue(torch.equal(first['input_ids'], changed_target['input_ids']))
    alone = tokenizer.collate_fn_test([examples[0]])
    first_tokens = first['input_ids'][0][first['attention_mask'][0].bool()]
    self.assertEqual(first_tokens.tolist(), alone['input_ids'][0].tolist())

  def test_random_training_varies_and_repeats_with_same_seed(self):
    left, right = self.tokenizer(), self.tokenizer()
    samples = []
    for _ in range(12):
      a = left.collate_fn_train(self.examples())['input_ids']
      b = right.collate_fn_train(self.examples())['input_ids']
      self.assertTrue(torch.equal(a, b))
      samples.append(str(a.tolist()))
    self.assertGreater(len(set(samples)), 1)

  def test_atomic_ce_backward_and_generation(self):
    tokenizer = self.tokenizer()
    model = ActionPiece(self.config, None, tokenizer)
    batch = tokenizer.collate_fn_train(self.examples())
    original_logits = []

    def capture(module, args, outputs):
      outputs.logits.retain_grad()
      original_logits.append(outputs.logits)

    hook = model.t5.register_forward_hook(capture)
    output = model(batch)
    hook.remove()
    losses = []
    for step in range(batch['labels'].shape[1]):
      allowed = tokenizer.target_allowed_tokens(step)
      local_labels = torch.tensor([allowed.index(int(label)) for label in batch['labels'][:, step]])
      losses.append(torch.nn.functional.cross_entropy(
          original_logits[0][:, step, list(allowed)], local_labels,
      ))
    self.assertTrue(torch.allclose(output.loss, torch.stack(losses).mean()))
    output.loss.backward()
    for step in range(batch['labels'].shape[1]):
      legal = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
      legal[list(tokenizer.target_allowed_tokens(step))] = True
      self.assertTrue(torch.isneginf(output.logits[:, step, ~legal]).all())
      self.assertEqual(original_logits[0].grad[:, step, ~legal].abs().sum().item(), 0)
    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    optimizer.step()
    model.eval()
    evaluation = tokenizer.collate_fn_test(self.examples())
    generated = model.generate(evaluation, n_return_sequences=2)
    self.assertEqual(generated.shape, (2, 2, 4))
    self.assertTrue((generated >= 0).all())

  def test_legacy_forward_keeps_original_loss(self):
    tokenizer = self.tokenizer()
    batch = tokenizer.collate_fn_train(self.examples())
    tokenizer.target_tokenization = 'actionpiece'
    model = ActionPiece(self.config, None, tokenizer)
    self.assertTrue(torch.equal(model(batch).loss, model.t5(**batch).loss))

  def test_pilot_rejects_confounding_configuration(self):
    for override in (
        {'train_shuffle': 'feature'}, {'n_inference_ensemble': 5},
        {'history_tokenization_scope': 'sequence'}, {'actionpiece_merge_hash': True},
        {'target_tokenization': 'actionpiece'},
    ):
      with self.assertRaises(ValueError):
        self.tokenizer(**override)

  def test_model_seed_does_not_change_cached_representation_identity(self):
    configs = [utils.get_config('ActionPiece', 'AmazonReviews2014', [
        'experiments/ahsb_beauty_common.yaml', 'experiments/ahsb_p0_raw.yaml'
    ], {'rand_seed': seed}) for seed in (2024, 2025)]
    stems = []
    for config in configs:
      tokenizer = object.__new__(ActionPieceTokenizer)
      tokenizer.config = config
      stems.append(tokenizer._feature_artifact_stem())
    self.assertEqual(stems[0], stems[1])


if __name__ == '__main__':
  unittest.main()
