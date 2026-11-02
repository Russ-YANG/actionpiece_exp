"""Local NineRec dataset reader for ActionPiece experiments."""

import csv
import json
import os
from pathlib import Path
from typing import Any

from genrec.dataset import AbstractDataset


class NineRec(AbstractDataset):
  """Read prepared NineRec ``item.csv`` and ``behaviour.tsv`` files."""

  def __init__(self, config: dict[str, Any]):
    super().__init__(config)
    self.category = config['category']
    self.data_root = Path(config['data_root']) / self.category
    self.cache_dir = os.path.join(
        config['cache_dir'], 'NineRec', self.category
    )
    self._load_and_process()

  def _load_items(self, path: Path) -> None:
    text_column = self.config.get('text_column', 'text_zh')
    with path.open('r', encoding='utf-8-sig', newline='') as handle:
      for row in csv.reader(handle):
        if not row:
          continue
        item = row[0].strip()
        text_zh = row[1].strip() if len(row) > 1 else ''
        text_en = row[2].strip() if len(row) > 2 else ''
        if item in self.item2id:
          continue
        self.item2id[item] = len(self.id_mapping['id2item'])
        self.id_mapping['id2item'].append(item)
        if text_column == 'text_en':
          text = text_en
        elif text_column == 'text_bilingual':
          text = f'{text_zh} {text_en}'.strip()
        else:
          text = text_zh
        self.item2meta[item] = text

  def _load_behaviour(self, path: Path) -> None:
    with path.open('r', encoding='utf-8-sig') as handle:
      for line in handle:
        line = line.rstrip('\n')
        if not line or '\t' not in line:
          continue
        user, item_text = line.split('\t', 1)
        items = [item for item in item_text.split() if item in self.item2id]
        if len(items) < 3 or user in self.user2id:
          continue
        self.user2id[user] = len(self.id_mapping['id2user'])
        self.id_mapping['id2user'].append(user)
        self.all_item_seqs[user] = items

  def _load_and_process(self) -> None:
    processed = Path(self.cache_dir) / 'processed'
    seq_path = processed / 'all_item_seqs.json'
    mapping_path = processed / 'id_mapping.json'
    metadata_path = processed / 'metadata.sentence.json'
    processed.mkdir(parents=True, exist_ok=True)

    if seq_path.exists() and mapping_path.exists() and metadata_path.exists():
      self.all_item_seqs = json.loads(seq_path.read_text())
      self.id_mapping = json.loads(mapping_path.read_text())
      self.item2meta = json.loads(metadata_path.read_text())
      return

    self.item2meta = {}
    self._load_items(self.data_root / f'{self.category}_item.csv')
    self._load_behaviour(self.data_root / f'{self.category}_behaviour.tsv')
    seq_path.write_text(json.dumps(self.all_item_seqs, ensure_ascii=False))
    mapping_path.write_text(json.dumps(self.id_mapping, ensure_ascii=False))
    metadata_path.write_text(json.dumps(self.item2meta, ensure_ascii=False))
