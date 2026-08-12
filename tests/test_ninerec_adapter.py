"""Tests for the NineRec-to-ActionPiece data adapter."""

import contextlib
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from genrec.datasets.NineRec.dataset import build_ninerec_processed
from genrec.datasets.NineRec.dataset import load_pairs
from genrec.datasets.NineRec.dataset import NineRec
from genrec.image_manifest import load_image_manifest


class FakeAccelerator:

  @contextlib.contextmanager
  def main_process_first(self):
    yield

  @property
  def is_main_process(self):
    return True


class NineRecAdapterTest(unittest.TestCase):

  @staticmethod
  def write_item_csv(path: Path) -> None:
    with path.open('w', encoding='utf-8', newline='') as stream:
      writer = csv.writer(stream)
      writer.writerow(['v2', '中文二，带逗号', 'English two, with comma'])
      writer.writerow(['v1', '中文一', 'English one'])
      writer.writerow(['v3', '中文三', 'English three'])

  def make_source(self, root: Path, include_v3_cover: bool = True) -> Path:
    source = root / 'official'
    source.mkdir()
    (source / 'DY_behaviour.tsv').write_text(
        'u2\tv2 v1\n'
        'u1\tv1 v3\n',
        encoding='utf-8',
    )
    self.write_item_csv(source / 'DY_item.csv')
    cover_dir = source / 'DY_cover'
    cover_dir.mkdir()
    (cover_dir / 'v1.jpg').write_bytes(b'v1-image')
    (cover_dir / 'v2.png').write_bytes(b'v2-image')
    if include_v3_cover:
      (cover_dir / 'v3.webp').write_bytes(b'v3-image')
    return source

  def test_builds_aligned_bilingual_artifacts_and_dataset(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = self.make_source(root)
      output = root / 'cache' / 'NineRec' / 'DY' / 'processed'
      summary = build_ninerec_processed(
          source_dir=source,
          output_dir=output,
          subset='DY',
          require_all_images=True,
      )

      self.assertEqual(summary['users'], 2)
      self.assertEqual(summary['items'], 3)
      self.assertEqual(summary['interactions'], 4)
      self.assertEqual(summary['missing_cover_items'], 0)
      self.assertEqual(len(summary['image_manifest_sha256']), 64)

      mapping = json.loads(
          (output / 'id_mapping.json').read_text(encoding='utf-8')
      )
      self.assertEqual(mapping['id2item'], ['[PAD]', 'v2', 'v1', 'v3'])
      metadata = json.loads(
          (output / 'metadata.qwen_text.json').read_text(encoding='utf-8')
      )
      self.assertEqual(
          metadata['v2'],
          'Chinese: 中文二，带逗号\nEnglish: English two, with comma',
      )
      fused = json.loads(
          (output / 'metadata.qwen_fused.json').read_text(encoding='utf-8')
      )
      self.assertEqual(fused['v1']['sentence'], metadata['v1'])
      self.assertTrue(Path(fused['v1']['image_url']).is_file())

      manifest = load_image_manifest(
          output / 'image_download_manifest.jsonl'
      )
      self.assertEqual(set(manifest), {'v1', 'v2', 'v3'})
      self.assertNotIn('asin', manifest['v1'])
      self.assertEqual(manifest['v1']['status'], 'downloaded')

      with mock.patch.object(NineRec, 'log'):
        dataset = NineRec({
            'subset': 'DY',
            'cache_dir': str(root / 'cache'),
            'metadata': 'qwen_text',
            'accelerator': FakeAccelerator(),
            'split': 'leave_one_out',
        })
      self.assertEqual(dataset.n_users, 3)  # Includes [PAD].
      self.assertEqual(dataset.n_items, 4)  # Includes [PAD].
      self.assertEqual(dataset.n_interactions, 4)
      self.assertEqual(dataset.item2meta, metadata)

  def test_pair_csv_uses_official_item_user_time_order(self):
    with tempfile.TemporaryDirectory() as temporary:
      pair_path = Path(temporary) / 'DY_pair.csv'
      pair_path.write_text(
          'v2,u1,20\n'
          'v1,u1,10\n'
          'v3,u2,5\n',
          encoding='utf-8',
      )
      self.assertEqual(
          load_pairs(pair_path),
          {'u1': ['v1', 'v2'], 'u2': ['v3']},
      )

  def test_image_manifest_accepts_legacy_amazon_asin(self):
    with tempfile.TemporaryDirectory() as temporary:
      path = Path(temporary) / 'manifest.jsonl'
      path.write_text(
          json.dumps({
              'asin': 'legacy-item',
              'status': 'missing',
              'path': None,
          }) + '\n',
          encoding='utf-8',
      )
      self.assertEqual(set(load_image_manifest(path)), {'legacy-item'})

  def test_require_all_images_rejects_missing_cover_before_writing(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      source = self.make_source(root, include_v3_cover=False)
      output = root / 'processed'
      with self.assertRaisesRegex(ValueError, 'missing for 1'):
        build_ninerec_processed(
            source_dir=source,
            output_dir=output,
            subset='DY',
            require_all_images=True,
        )
      self.assertFalse((output / 'id_mapping.json').exists())


if __name__ == '__main__':
  unittest.main()
