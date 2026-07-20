"""Tests for the resumable local Qwen text encoder."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
  import torch
except ImportError:
  torch = None


MODULE_PATH = (
    Path(__file__).parents[1]
    / 'genrec/models/ActionPiece/qwen_local.py'
)
SPEC = importlib.util.spec_from_file_location('actionpiece_qwen_local', MODULE_PATH)
QWEN_LOCAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QWEN_LOCAL)
QwenLocalTextEncoder = QWEN_LOCAL.QwenLocalTextEncoder


@unittest.skipIf(torch is None, 'torch is not installed')
class QwenLocalTextEncoderTest(unittest.TestCase):

  def make_encoder(self, model_factory, batch_size=2):
    return QwenLocalTextEncoder(
        model_path='/models/Qwen3-VL-Embedding-8B',
        model_id='Qwen/Qwen3-VL-Embedding-8B',
        model_revision='test-revision',
        repo_path='/repos/Qwen3-VL-Embedding',
        code_revision='test-code-revision',
        instruction='Represent the product.',
        dimension=3,
        batch_size=batch_size,
        max_length=32,
        torch_dtype='float32',
        attn_implementation='sdpa',
        require_cuda=False,
        model_factory=model_factory,
    )

  @staticmethod
  def fake_factory(call_log, fail_on_call=None):
    class FakeEmbedder:

      def __init__(self, **kwargs):
        call_log.append(('init', kwargs))
        self.process_calls = 0

      def process(self, inputs, normalize=True):
        del normalize
        self.process_calls += 1
        call_log.append(('process', inputs))
        if self.process_calls == fail_on_call:
          raise RuntimeError('simulated interruption')
        rows = []
        for value in inputs:
          base = float(len(value['text']))
          rows.append([base, base + 1, base + 2, base + 3])
        return torch.tensor(rows, dtype=torch.float32)

    return FakeEmbedder

  def test_truncates_normalizes_and_writes_manifest(self):
    with tempfile.TemporaryDirectory() as temporary:
      calls = []
      encoder = self.make_encoder(self.fake_factory(calls))
      output_path = Path(temporary) / 'embeddings.sent_emb'
      result = encoder.encode(
          ['a', 'bbbb', 'cc'], ['item-a', 'item-b', 'item-c'], output_path
      )

      expected = np.asarray([[1, 2, 3], [4, 5, 6], [2, 3, 4]], np.float32)
      expected /= np.linalg.norm(expected, axis=1, keepdims=True)
      np.testing.assert_allclose(result, expected, rtol=1e-6)
      self.assertTrue(output_path.exists())
      self.assertFalse(Path(f'{output_path}.partial').exists())
      manifest = json.loads(
          Path(f'{output_path}.manifest.json').read_text()
      )
      self.assertEqual(manifest['backend'], 'qwen_local')
      self.assertEqual(manifest['native_dimension'], 4)
      self.assertEqual(manifest['normalization'], 'truncate_then_l2')
      self.assertEqual(len(manifest['embedding_sha256']), 64)

  def test_resumes_after_committed_batch(self):
    with tempfile.TemporaryDirectory() as temporary:
      output_path = Path(temporary) / 'embeddings.sent_emb'
      interrupted_calls = []
      interrupted = self.make_encoder(
          self.fake_factory(interrupted_calls, fail_on_call=2)
      )
      with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
        interrupted.encode(
            ['a', 'bbbb', 'cc'],
            ['item-a', 'item-b', 'item-c'],
            output_path,
        )
      progress = json.loads(
          Path(f'{output_path}.progress.json').read_text()
      )
      self.assertEqual(progress['next_index'], 2)

      resumed_calls = []
      resumed = self.make_encoder(self.fake_factory(resumed_calls))
      result = resumed.encode(
          ['a', 'bbbb', 'cc'],
          ['item-a', 'item-b', 'item-c'],
          output_path,
      )
      process_calls = [call for call in resumed_calls if call[0] == 'process']
      self.assertEqual(len(process_calls), 1)
      self.assertEqual(process_calls[0][1][0]['text'], 'cc')
      self.assertEqual(result.shape, (3, 3))


if __name__ == '__main__':
  unittest.main()
