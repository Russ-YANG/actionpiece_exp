"""Offline tiny-data training, checkpoint reload and evaluation integration."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from datasets import Dataset
import torch

from genrec.pipeline import Pipeline
from genrec.dataset import AbstractDataset
from genrec.models.ActionPiece.core import ActionPieceCore
from genrec.models.ActionPiece.tokenizer import ActionPieceTokenizer
from genrec.trainer import Trainer


class TinyDataset(AbstractDataset):

  def __init__(self, config):
    super().__init__(config)
    self.cache_dir = config['cache_dir']
    self.all_item_seqs = {'u': ['a', 'b', 'c', 'd']}
    self.id_mapping = {
        'user2id': {'u': 0}, 'id2user': ['u'],
        'item2id': {'[PAD]': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4},
        'id2item': ['[PAD]', 'a', 'b', 'c', 'd'],
    }

  def split(self):
    self.split_data = {
        split: Dataset.from_dict({'item_seq': sequences})
        for split, sequences in {
            'train': [['a', 'b', 'c'], ['d', 'c', 'b']],
            'val': [['a', 'c', 'd'], ['d', 'a', 'b']],
            'test': [['a', 'b', 'c', 'd'], ['d', 'c', 'b', 'a']],
        }.items()
    }
    return self.split_data


class TinyTokenizer(ActionPieceTokenizer):

  def _feature_artifact_stem(self):
    return 'tiny'


class PipelinePilotTest(unittest.TestCase):

  def test_train_save_reload_switch_eval_policy_and_profile(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      processed = root / 'processed'
      processed.mkdir()
      features = {'a': [0, 10, 0], 'b': [0, 11, 1], 'c': [1, 10, 0], 'd': [1, 11, 1]}
      (processed / 'item.tiny.feat').write_text(json.dumps(features))
      core = ActionPieceCore(state2feat=features)
      core.train(
          state_corpus=[['a', 'b', 'c', 'd']] * 3,
          target_vocab_size=core.n_init_feats + 2,
          allow_cross_action_merges=False, allowed_merge_slots=[0, 1],
      )
      core.save(processed / f'actionpiece.tiny.history_item.no_hash.v{core.vocab_size}.json')
      config = {
          'cache_dir': directory, 'log_dir': str(root / 'logs'),
          'ckpt_dir': str(root / 'ckpt'), 'tensorboard_log_dir': str(root / 'tb'),
          'result_path': str(root / 'result.json'),
          'history_candidate_report': str(root / 'candidates.json'),
          'metadata': 'qwen_separate', 'actionpiece_vocab_size': core.vocab_size,
          'history_tokenization_scope': 'item', 'target_tokenization': 'atomic',
          'actionpiece_merge_hash': False, 'require_cached_item_features': True,
          'history_granularity': 'random', 'history_eval_granularity': 'full',
          'n_inference_ensemble': -1, 'train_shuffle': 'none', 'profile_history': True,
          'num_layers': 1, 'num_decoder_layers': 1, 'd_model': 16, 'd_ff': 32,
          'num_heads': 2, 'd_kv': 8, 'dropout_rate': 0.0,
          'epochs': 1, 'warmup_steps': 0, 'train_batch_size': 2, 'eval_batch_size': 2,
          'num_beams': 2, 'topk': [1, 2], 'val_metric': 'ndcg@2',
      }
      with mock.patch('genrec.utils.init_device', return_value=(torch.device('cpu'), False)), mock.patch('genrec.utils.get_dataset', return_value=TinyDataset):
        pipeline = Pipeline('ActionPiece', 'AmazonReviews2014', tokenizer=TinyTokenizer, config_dict=config)
        pipeline.run()
        result = json.loads((root / 'result.json').read_text())
        self.assertIn('ndcg@2', result['metrics'])
        self.assertTrue(Path(result['checkpoint']).is_file())
        self.assertEqual(result['profile']['encoder_sequences'], 2)
        self.assertEqual(len(result['train_epoch_profiles']), 1)
        self.assertFalse(result['evaluation_only'])
        self.assertTrue((root / 'candidates.json').is_file())

        config.update({
            'eval_checkpoint': result['checkpoint'], 'history_eval_granularity': 'raw',
            'beam_search_mode': 'eos_aware',
            'result_path': str(root / 'raw_eval.json'),
        })
        with mock.patch.object(Trainer, 'fit', side_effect=AssertionError('must not retrain')):
          evaluation = Pipeline('ActionPiece', 'AmazonReviews2014', tokenizer=TinyTokenizer, config_dict=config)
          evaluation.run()
        raw = json.loads((root / 'raw_eval.json').read_text())
        self.assertTrue(raw['evaluation_only'])
        self.assertEqual(raw['config']['beam_search_mode'], 'eos_aware')
        self.assertEqual(raw['train_epoch_profiles'], [])
        self.assertGreaterEqual(
            raw['profile']['mean_tokens_including_bos_eos'],
            result['profile']['mean_tokens_including_bos_eos'],
        )


if __name__ == '__main__':
  unittest.main()
