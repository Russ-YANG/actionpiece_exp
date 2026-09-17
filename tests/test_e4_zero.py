"""Focused checks for the E4-Zero text-OPQ4 plus NULL4 control."""

import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from genrec import utils
from genrec.models.ActionPiece.tokenizer import ActionPieceTokenizer


class E4ZeroTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    cls.config = utils.get_config(
        'ActionPiece',
        'AmazonReviews2014',
        ['experiments/e4_qwen3_vl_8b_text_null4_beauty.yaml'],
        {},
    )
    cls.train_config = utils.get_config(
        'ActionPiece',
        'AmazonReviews2014',
        ['experiments/e4_qwen3_vl_8b_text_null4_beauty_train.yaml'],
        {},
    )
    cls.original_train_config = utils.get_config(
        'ActionPiece',
        'AmazonReviews2014',
        ['experiments/e4_qwen3_vl_8b_separate_opq_beauty_train.yaml'],
        {},
    )

  def bare_tokenizer(self):
    tokenizer = object.__new__(ActionPieceTokenizer)
    tokenizer.config = dict(self.config)
    tokenizer.logger = logging.getLogger('e4-zero-test')
    return tokenizer

  def test_config_preserves_raw_e4_shape_with_reserved_null_code(self):
    config = self.config
    self.assertEqual(config['metadata'], 'qwen_separate')
    self.assertEqual(config['pq_n_codebooks'], 4)
    self.assertEqual(config['image_pq_n_codebooks'], 4)
    self.assertEqual(config['image_pq_codebook_size'], 256)
    self.assertEqual(config['separate_image_semantic_ids'], 'fixed_null')
    self.assertTrue(config['tokenizer_only'])

  def test_train_config_reuses_zero_tokenizer_and_original_e4_budget(self):
    train = self.train_config
    original = self.original_train_config
    self.assertFalse(train['tokenizer_only'])
    self.assertEqual(train['separate_image_semantic_ids'], 'fixed_null')
    for key in (
        'category', 'sent_emb_model', 'sent_emb_dim', 'pq_n_codebooks',
        'pq_codebook_size', 'image_pq_n_codebooks',
        'image_pq_codebook_size', 'actionpiece_vocab_size', 'train_batch_size',
        'eval_batch_size', 'epochs',
    ):
      self.assertEqual(train[key], original[key])
    tokenizer_only = self.bare_tokenizer()
    training = self.bare_tokenizer()
    training.config = dict(train)
    self.assertEqual(
        tokenizer_only._feature_artifact_stem(),
        training._feature_artifact_stem(),
    )

  def test_feature_build_reuses_original_text4_then_null4_and_hash(self):
    tokenizer = self.bare_tokenizer()
    with tempfile.TemporaryDirectory() as temporary:
      processed = Path(temporary) / 'processed'
      processed.mkdir()
      dataset = SimpleNamespace(cache_dir=temporary)
      source = processed / (
          f'item.{tokenizer._learned_separate_feature_artifact_stem()}.feat'
      )
      source.write_text(
          '{"item-a": [1,2,3,4,11,12,13,14,7],'
          ' "item-b": [5,6,7,8,21,22,23,24,9]}'
      )
      with mock.patch.object(
               tokenizer,
               '_get_image_sem_ids',
               side_effect=AssertionError('E4-Zero must not read image SIDs'),
           ), mock.patch.object(
               tokenizer,
               '_get_sem_ids',
               side_effect=AssertionError('E4-Zero must reuse E4 text SIDs'),
           ), mock.patch.object(
               tokenizer,
               '_get_hashed_feat',
               side_effect=lambda _, feats: {
                   item: (*values, index)
                   for index, (item, values) in enumerate(feats.items())
               },
           ):
        features = tokenizer._get_item2feat(dataset)
      self.assertEqual(features['item-a'], (1, 2, 3, 4, 256, 256, 256, 256, 0))
      self.assertEqual(features['item-b'], (5, 6, 7, 8, 256, 256, 256, 256, 1))
      feature_files = list(
          (Path(temporary) / 'processed').glob('item.*inull4x256*.feat')
      )
      self.assertEqual(len(feature_files), 1)

  def test_null_cache_identity_is_separate_from_original_e4(self):
    zero = self.bare_tokenizer()
    original = self.bare_tokenizer()
    original.config['separate_image_semantic_ids'] = 'learned'
    original.config['qwen_local_image_manifest_sha256'] = 'a' * 64
    self.assertIn('inull4x256', zero._feature_artifact_stem())
    self.assertNotEqual(
        zero._feature_artifact_stem(), original._feature_artifact_stem()
    )


if __name__ == '__main__':
  unittest.main()
