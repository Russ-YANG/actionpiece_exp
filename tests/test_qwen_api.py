"""Tests for the resumable Qwen API text encoder."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import requests

MODULE_PATH = (
    Path(__file__).parents[1]
    / 'genrec/models/ActionPiece/qwen_api.py'
)
SPEC = importlib.util.spec_from_file_location('actionpiece_qwen_api', MODULE_PATH)
QWEN_API = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QWEN_API)
QwenApiTextEncoder = QWEN_API.QwenApiTextEncoder


class FakeResponse:

  def __init__(self, body, status_code=200):
    self._body = body
    self.status_code = status_code

  def json(self):
    return self._body


def response_for_payload(payload, dimension):
  embeddings = []
  for index, content in enumerate(payload['input']['contents']):
    base = float(len(content['text']))
    embeddings.append({
        'index': index,
        'embedding': [base + offset for offset in range(dimension)],
    })
  return {
      'output': {'embeddings': embeddings},
      'usage': {'input_tokens': len(embeddings)},
      'request_id': f'request-{len(embeddings)}',
  }


class QwenApiTextEncoderTest(unittest.TestCase):

  def make_encoder(self, env_file, request_post, batch_size=2):
    return QwenApiTextEncoder(
        model='qwen3-vl-embedding',
        instruction='Represent the product.',
        dimension=3,
        batch_size=batch_size,
        max_retries=1,
        env_file=str(env_file),
        request_post=request_post,
    )

  def test_batches_and_preserves_item_order(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      env_file = root / '.env.local'
      env_file.write_text('DASHSCOPE_API_KEY=test-key\n')
      calls = []

      def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json, timeout))
        return FakeResponse(response_for_payload(json, 3))

      encoder = self.make_encoder(env_file, fake_post)
      output_path = root / 'embeddings.sent_emb'
      result = encoder.encode(
          ['a', 'bbbb', 'cc'], ['item-a', 'item-b', 'item-c'], output_path
      )

      self.assertEqual(result.shape, (3, 3))
      np.testing.assert_allclose(
          result,
          np.asarray([[1, 2, 3], [4, 5, 6], [2, 3, 4]], dtype=np.float32),
      )
      self.assertEqual(len(calls), 2)
      self.assertEqual(
          calls[0][2]['parameters']['instruct'], 'Represent the product.'
      )
      self.assertEqual(calls[0][2]['parameters']['dimension'], 3)
      self.assertFalse(calls[0][2]['parameters']['enable_fusion'])
      self.assertTrue(output_path.exists())
      self.assertFalse(Path(f'{output_path}.partial').exists())
      self.assertFalse(Path(f'{output_path}.progress.json').exists())
      request_log = Path(f'{output_path}.requests.jsonl')
      self.assertEqual(len(request_log.read_text().splitlines()), 2)
      manifest = json.loads(
          Path(f'{output_path}.manifest.json').read_text()
      )
      self.assertEqual(manifest['api_calls'], 2)
      self.assertEqual(manifest['usage']['input_tokens'], 3)
      self.assertEqual(manifest['item_count'], 3)
      self.assertEqual(manifest['dimension'], 3)
      self.assertEqual(len(manifest['embedding_sha256']), 64)

  def test_resumes_after_committed_batch(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      env_file = root / '.env.local'
      env_file.write_text('DASHSCOPE_API_KEY=test-key\n')
      output_path = root / 'embeddings.sent_emb'
      call_count = 0

      def interrupted_post(url, headers, json, timeout):
        del url, headers, timeout
        nonlocal call_count
        call_count += 1
        if call_count == 2:
          raise requests.ConnectionError('simulated interruption')
        return FakeResponse(response_for_payload(json, 3))

      encoder = self.make_encoder(env_file, interrupted_post)
      with self.assertRaisesRegex(RuntimeError, 'after 1 attempts'):
        encoder.encode(
            ['a', 'bbbb', 'cc'],
            ['item-a', 'item-b', 'item-c'],
            output_path,
        )

      progress = json.loads(
          Path(f'{output_path}.progress.json').read_text()
      )
      self.assertEqual(progress['next_index'], 2)
      resumed_payloads = []

      def resumed_post(url, headers, json, timeout):
        del url, headers, timeout
        resumed_payloads.append(json)
        return FakeResponse(response_for_payload(json, 3))

      resumed_encoder = self.make_encoder(env_file, resumed_post)
      result = resumed_encoder.encode(
          ['a', 'bbbb', 'cc'],
          ['item-a', 'item-b', 'item-c'],
          output_path,
      )
      self.assertEqual(len(resumed_payloads), 1)
      self.assertEqual(
          resumed_payloads[0]['input']['contents'], [{'text': 'cc'}]
      )
      np.testing.assert_allclose(
          result,
          np.asarray([[1, 2, 3], [4, 5, 6], [2, 3, 4]], dtype=np.float32),
      )

  def test_rejects_resume_with_changed_item_order(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      env_file = root / '.env.local'
      env_file.write_text('DASHSCOPE_API_KEY=test-key\n')
      output_path = root / 'embeddings.sent_emb'
      call_count = 0

      def interrupted_post(url, headers, json, timeout):
        del url, headers, timeout
        nonlocal call_count
        call_count += 1
        if call_count == 2:
          raise requests.ConnectionError('simulated interruption')
        return FakeResponse(response_for_payload(json, 3))

      encoder = self.make_encoder(env_file, interrupted_post)
      with self.assertRaises(RuntimeError):
        encoder.encode(
            ['a', 'bbbb', 'cc'],
            ['item-a', 'item-b', 'item-c'],
            output_path,
        )

      with self.assertRaisesRegex(RuntimeError, 'item_order_sha256'):
        encoder.encode(
            ['bbbb', 'a', 'cc'],
            ['item-b', 'item-a', 'item-c'],
            output_path,
        )


if __name__ == '__main__':
  unittest.main()
