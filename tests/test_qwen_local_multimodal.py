"""Tests for resumable local Qwen joint text-image embeddings."""

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
SPEC = importlib.util.spec_from_file_location('actionpiece_qwen_local_mm', MODULE_PATH)
QWEN_LOCAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QWEN_LOCAL)
QwenLocalImageEncoder = QWEN_LOCAL.QwenLocalImageEncoder
QwenLocalMultimodalEncoder = QWEN_LOCAL.QwenLocalMultimodalEncoder


@unittest.skipIf(torch is None, 'torch is not installed')
class QwenLocalMultimodalEncoderTest(unittest.TestCase):

  @staticmethod
  def fake_factory(call_log):
    class FakeEmbedder:

      def __init__(self, **kwargs):
        call_log.append(('init', kwargs))

      def process(self, inputs, normalize=True):
        del normalize
        call_log.append(('process', inputs))
        rows = []
        for value in inputs:
          base = float(len(value['text']) + (10 if 'image' in value else 0))
          rows.append([base, base + 1, base + 2, base + 3])
        return torch.tensor(rows, dtype=torch.float32)

    return FakeEmbedder

  def test_joint_inputs_fallback_truncation_and_manifest(self):
    with tempfile.TemporaryDirectory() as temporary:
      temporary = Path(temporary)
      image_path = temporary / 'item-a.jpg'
      image_path.write_bytes(b'fake-image')
      calls = []
      encoder = QwenLocalMultimodalEncoder(
          model_path='/models/Qwen3-VL-Embedding-8B',
          model_id='Qwen/Qwen3-VL-Embedding-8B',
          model_revision='model-revision',
          repo_path='/repos/Qwen3-VL-Embedding',
          code_revision='code-revision',
          instruction='Represent the product.',
          dimension=3,
          batch_size=2,
          max_length=32,
          torch_dtype='float32',
          attn_implementation='sdpa',
          require_cuda=False,
          model_factory=self.fake_factory(calls),
      )
      output_path = temporary / 'joint.sent_emb'
      result = encoder.encode(
          ['a', 'bbbb'],
          [image_path, None],
          ['item-a', 'item-b'],
          output_path,
          image_manifest_sha256='a' * 64,
      )

      expected = np.asarray([[11, 12, 13], [4, 5, 6]], np.float32)
      expected /= np.linalg.norm(expected, axis=1, keepdims=True)
      np.testing.assert_allclose(result, expected, rtol=1e-6)
      process_inputs = [call for call in calls if call[0] == 'process'][0][1]
      self.assertEqual(process_inputs[0]['image'], str(image_path.resolve()))
      self.assertNotIn('image', process_inputs[1])
      manifest = json.loads(
          Path(f'{output_path}.manifest.json').read_text(encoding='utf-8')
      )
      self.assertEqual(manifest['backend'], 'qwen_local_multimodal')
      self.assertEqual(manifest['modality'], 'text_image_joint')
      self.assertEqual(manifest['native_dimension'], 4)
      self.assertEqual(manifest['joint_item_count'], 1)
      self.assertEqual(manifest['missing_image_items'], ['item-b'])

  def test_joint_stem_is_separate_from_text(self):
    common = dict(
        backend='qwen_local',
        model_id='Qwen/Qwen3-VL-Embedding-8B',
        dimension=768,
        instruction='Represent the product.',
        model_revision='model-revision',
        code_revision='code-revision',
        max_length=2048,
        torch_dtype='bfloat16',
        attn_implementation='sdpa',
    )
    text_stem = QWEN_LOCAL.build_qwen_local_artifact_stem(**common)
    joint_stem = QWEN_LOCAL.build_qwen_local_artifact_stem(
        **common,
        modality='text_image_joint',
        image_manifest_sha256='b' * 64,
    )
    self.assertNotEqual(text_stem, joint_stem)
    self.assertIn('.joint.imbbbbbbbbbbbb', joint_stem)


@unittest.skipIf(torch is None, 'torch is not installed')
class QwenLocalImageEncoderTest(unittest.TestCase):

  @staticmethod
  def fake_factory(call_log):
    class FakeEmbedder:

      def __init__(self, **kwargs):
        call_log.append(('init', kwargs))

      def process(self, inputs, normalize=True):
        del normalize
        call_log.append(('process', inputs))
        return torch.tensor(
            [[3.0 + index, 4.0, 5.0, 6.0] for index, _ in enumerate(inputs)],
            dtype=torch.float32,
        )

    return FakeEmbedder

  def test_image_only_inputs_zero_missing_truncation_and_manifest(self):
    with tempfile.TemporaryDirectory() as temporary:
      temporary = Path(temporary)
      image_path = temporary / 'item-a.jpg'
      image_path.write_bytes(b'fake-image')
      calls = []
      encoder = QwenLocalImageEncoder(
          model_path='/models/Qwen3-VL-Embedding-8B',
          model_id='Qwen/Qwen3-VL-Embedding-8B',
          model_revision='model-revision',
          repo_path='/repos/Qwen3-VL-Embedding',
          code_revision='code-revision',
          instruction='Represent the product image.',
          dimension=3,
          batch_size=2,
          max_length=32,
          torch_dtype='float32',
          attn_implementation='sdpa',
          require_cuda=False,
          model_factory=self.fake_factory(calls),
      )
      output_path = temporary / 'image.image_emb'
      result = encoder.encode(
          [image_path, None],
          ['item-a', 'item-b'],
          output_path,
          image_manifest_sha256='c' * 64,
      )

      expected_first = np.asarray([3, 4, 5], np.float32)
      expected_first /= np.linalg.norm(expected_first)
      np.testing.assert_allclose(result[0], expected_first, rtol=1e-6)
      np.testing.assert_array_equal(result[1], np.zeros(3, np.float32))
      process_inputs = [call for call in calls if call[0] == 'process'][0][1]
      self.assertEqual(
          process_inputs,
          [{
              'image': str(image_path.resolve()),
              'instruction': 'Represent the product image.',
          }],
      )
      self.assertNotIn('text', process_inputs[0])
      manifest = json.loads(
          Path(f'{output_path}.manifest.json').read_text(encoding='utf-8')
      )
      self.assertEqual(manifest['backend'], 'qwen_local_image')
      self.assertEqual(manifest['modality'], 'image')
      self.assertEqual(manifest['missing_image_policy'], 'zero_vector')
      self.assertEqual(manifest['encoded_image_count'], 1)
      self.assertEqual(manifest['missing_image_items'], ['item-b'])

  def test_image_stem_is_separate_from_text_and_joint(self):
    common = dict(
        backend='qwen_local',
        model_id='Qwen/Qwen3-VL-Embedding-8B',
        dimension=768,
        instruction='Represent the product.',
        model_revision='model-revision',
        code_revision='code-revision',
        max_length=2048,
        torch_dtype='bfloat16',
        attn_implementation='sdpa',
    )
    text_stem = QWEN_LOCAL.build_qwen_local_artifact_stem(**common)
    image_stem = QWEN_LOCAL.build_qwen_local_artifact_stem(
        **common,
        modality='image',
        image_manifest_sha256='d' * 64,
    )
    joint_stem = QWEN_LOCAL.build_qwen_local_artifact_stem(
        **common,
        modality='text_image_joint',
        image_manifest_sha256='d' * 64,
    )
    self.assertNotEqual(text_stem, image_stem)
    self.assertNotEqual(joint_stem, image_stem)
    self.assertIn('.image.imdddddddddddd', image_stem)


if __name__ == '__main__':
  unittest.main()
