"""EOS search against an exhaustive oracle and checkpoint-compatible T5."""

import math
from types import SimpleNamespace
import unittest

import torch

from genrec import utils  # Initialize the existing project import cycle.
from genrec.models.ActionPiece.model import ActionPiece


class ToyTokenizer:
  eos_token = 1
  padding_token = 0
  vocab_size = 7
  max_token_seq_len = 16
  length_control_history_tokens = ()
  length_control_target_tokens = ()

  def target_allowed_tokens(self, step):
    return None


class ToyT5(torch.nn.Module):
  """Deterministic prefix probabilities, independent of the search algorithm."""

  def __init__(self, distribution):
    super().__init__()
    self.distribution = distribution
    self.device = torch.device('cpu')
    self.config = SimpleNamespace(decoder_start_token_id=0)

  def get_encoder(self):
    return lambda input_ids, **kwargs: input_ids[:, 0]

  def forward(self, encoder_outputs, attention_mask, decoder_input_ids):
    rows = []
    for source, sequence in zip(encoder_outputs.tolist(), decoder_input_ids.tolist()):
      probs = self.distribution(tuple(sequence[1:]), source)
      rows.append([math.log(probs.get(i, 0)) if probs.get(i, 0) else -math.inf
                   for i in range(ToyTokenizer.vocab_size)])
    logits = torch.tensor(rows)[:, None, :].expand(-1, decoder_input_ids.shape[1], -1)
    return SimpleNamespace(logits=logits)


def toy_model(distribution, tokenizer=None, mode='eos_aware'):
  model = ActionPiece.__new__(ActionPiece)
  torch.nn.Module.__init__(model)
  model.tokenizer = tokenizer or ToyTokenizer()
  model.beam_search_mode = mode
  model.t5 = ToyT5(distribution)
  return model


def search(model, horizon=5, beams=8, returns=5, sources=(0,)):
  inputs = torch.tensor(sources)[:, None]
  sequences, scores = model.beam_search(
      inputs, torch.ones_like(inputs), max_length=horizon,
      num_beams=beams, num_return_sequences=returns, return_score=True,
  )
  # Public scores retain the legacy common-horizon divisor.
  return sequences, scores * (horizon - 1)


def content(sequence):
  values = sequence.tolist()[1:]
  return tuple(values[:values.index(1) + 1] if 1 in values else values)


class EOSBeamTest(unittest.TestCase):

  def test_batched_search_matches_exhaustive_variable_length_oracle(self):
    def distribution(prefix, source):
      if len(prefix) == 3:
        return {1: 1.0}
      p = 0.19 + 0.03 * source + 0.013 * sum(prefix)
      return {1: p, 2: (1 - p) * 0.63, 3: (1 - p) * 0.37}

    sequences, scores = search(toy_model(distribution), sources=(0, 1))
    for source in range(2):
      exhaustive = []

      def visit(prefix, probability):
        for token, p in distribution(prefix, source).items():
          sequence = prefix + (token,)
          if token == 1:
            exhaustive.append((sequence, math.log(probability * p)))
          else:
            visit(sequence, probability * p)

      visit((), 1.0)
      expected = sorted(exhaustive, key=lambda item: item[1], reverse=True)[:5]
      for rank, (sequence, score) in enumerate(expected):
        index = source * 5 + rank
        self.assertEqual(content(sequences[index]), sequence)
        self.assertAlmostEqual(scores[index].item(), score, places=5)

  def test_first_eos_scored_once_and_completion_survives_later_expansion(self):
    def distribution(prefix, source):
      if not prefix:
        return {1: 0.4, 2: 0.6}
      if 1 in prefix:
        return {i: 1 / 7 for i in range(7)}
      if prefix == (2,):
        return {1: 0.5, 3: 0.5}
      return {1: 1.0}

    model = toy_model(distribution)
    for horizon in (4, 8):
      sequences, scores = search(model, horizon=horizon, beams=3, returns=3)
      self.assertEqual(content(sequences[0]), (1,))
      self.assertAlmostEqual(scores[0].item(), math.log(0.4), places=6)
      self.assertEqual(len({content(row) for row in sequences}), 3)
      self.assertTrue((sequences[0, 2:] == 0).all())
    model.beam_search_mode = 'legacy'
    sequences, _ = search(model, horizon=8, beams=3, returns=3)
    self.assertNotEqual(content(sequences[0]), (1,))

  def test_unfinished_horizon_fallback_matches_legacy_when_eos_impossible(self):
    def distribution(prefix, source):
      p = 0.65 if not prefix else (0.55 if prefix[-1] == 2 else 0.9)
      return {2: p, 3: 1 - p}

    model = toy_model(distribution)
    corrected = search(model, horizon=3, beams=2, returns=2)
    model.beam_search_mode = 'legacy'
    legacy = search(model, horizon=3, beams=2, returns=2)
    self.assertTrue(torch.equal(corrected[0], legacy[0]))
    self.assertTrue(torch.allclose(corrected[1], legacy[1]))

  def test_atomic_single_path_does_not_duplicate_unreachable_beams(self):
    class AtomicTokenizer(ToyTokenizer):
      def target_allowed_tokens(self, step):
        return (2,) if step == 0 else (1,)

    model = toy_model(lambda *_: {i: 1 / 7 for i in range(7)}, AtomicTokenizer())
    sequences, scores = search(model, horizon=6, beams=4, returns=4)
    self.assertEqual(content(sequences[0]), (2, 1))
    self.assertEqual(scores[0].item(), 0)
    self.assertTrue(torch.isneginf(scores[1:]).all())
    self.assertTrue(all(content(row) == (1,) for row in sequences[1:]))

  def test_htpad_forced_suffix_keeps_grammar_and_zero_increment(self):
    tokenizer = ToyTokenizer()
    tokenizer.length_control_target_tokens = (4, 5, 6)

    def distribution(prefix, source):
      return {1: 0.1, 2: 0.6 if not prefix else 0.01,
              4: 0.2 if not prefix else 0.79, 5: 0.05, 6: 0.05}

    sequences, scores = search(toy_model(distribution, tokenizer),
                               horizon=6, beams=2, returns=2)
    self.assertEqual(content(sequences[0]), (2, 4, 5, 6, 1))
    expected = math.log(0.6 / 0.8) + math.log(0.79 / 0.8)
    self.assertAlmostEqual(scores[0].item(), expected, places=5)

  def test_real_t5_checkpoint_keys_and_atomic_search_compatibility(self):
    class AtomicTokenizer(ToyTokenizer):
      def target_allowed_tokens(self, step):
        return (2, 3) if step == 0 else (1,)

    config = {
        'num_layers': 1, 'num_decoder_layers': 1, 'd_model': 8, 'd_ff': 16,
        'num_heads': 2, 'd_kv': 4, 'dropout_rate': 0.0,
        'activation_function': 'relu', 'feed_forward_proj': 'relu',
        'n_inference_ensemble': -1,
    }
    baseline = ActionPiece(config, None, AtomicTokenizer()).eval()
    corrected = ActionPiece({**config, 'beam_search_mode': 'eos_aware'},
                            None, AtomicTokenizer()).eval()
    corrected.load_state_dict(baseline.state_dict(), strict=True)
    old_ids, old_scores = search(baseline, horizon=5, beams=2, returns=2, sources=(2, 3))
    new_ids, new_scores = search(corrected, horizon=5, beams=2, returns=2, sources=(2, 3))
    self.assertEqual([content(row) for row in old_ids], [content(row) for row in new_ids])
    self.assertTrue(torch.allclose(old_scores, new_scores, atol=1e-6))
    with self.assertRaisesRegex(ValueError, 'beam_search_mode'):
      ActionPiece({**config, 'beam_search_mode': 'typo'}, None, AtomicTokenizer())


if __name__ == '__main__':
  unittest.main()
