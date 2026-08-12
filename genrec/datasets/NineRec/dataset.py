"""Dataset adapter for the official NineRec release."""

from __future__ import annotations

import collections
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from genrec.dataset import AbstractDataset


TEXT_LANGUAGES = {'zh', 'en', 'bilingual'}
PAIR_ORDERS = {'item-user-time', 'user-item-time'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
IMAGE_METADATA_MODES = {
    'sentence_image',
    'sentence_image_fused',
    'qwen_multimodal',
    'qwen_separate',
    'qwen_fused',
}


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_write_text(path: Path, value: str) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  handle, temporary = tempfile.mkstemp(
      prefix=f'.{path.name}.', dir=path.parent
  )
  try:
    with os.fdopen(handle, 'w', encoding='utf-8') as stream:
      stream.write(value)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  except BaseException:
    with contextlib.suppress(FileNotFoundError):
      os.unlink(temporary)
    raise


def _write_json(path: Path, value: object) -> None:
  _atomic_write_text(
      path,
      json.dumps(value, ensure_ascii=False, indent=2) + '\n',
  )


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
  _atomic_write_text(
      path,
      ''.join(
          json.dumps(record, ensure_ascii=False) + '\n'
          for record in records
      ),
  )


def load_behaviour(path: Path) -> dict[str, list[str]]:
  """Load the official tab-separated user and ordered-item sequences."""
  sequences: dict[str, list[str]] = {}
  with path.open('r', encoding='utf-8-sig') as stream:
    for line_number, raw_line in enumerate(stream, start=1):
      line = raw_line.rstrip('\r\n')
      if not line:
        continue
      fields = line.split('\t')
      if len(fields) != 2:
        raise ValueError(
            f'{path}:{line_number}: expected two tab-separated fields.'
        )
      user = fields[0].strip()
      items = fields[1].split()
      if not user or not items:
        raise ValueError(
            f'{path}:{line_number}: user and item sequence must be non-empty.'
        )
      if user in sequences:
        raise ValueError(f'{path}:{line_number}: duplicate user {user!r}.')
      sequences[user] = items
  if not sequences:
    raise ValueError(f'No behavior sequences found in {path}.')
  return sequences


def _timestamp(value: str, path: Path, line_number: int) -> float:
  try:
    return float(value)
  except ValueError as exc:
    raise ValueError(
        f'{path}:{line_number}: invalid timestamp {value!r}.'
    ) from exc


def load_pairs(
    path: Path, pair_order: str = 'item-user-time'
) -> dict[str, list[str]]:
  """Convert a NineRec pair CSV into timestamp-ordered user sequences."""
  if pair_order not in PAIR_ORDERS:
    raise ValueError(
        f'pair_order must be one of {sorted(PAIR_ORDERS)}, got {pair_order!r}.'
    )
  events: dict[str, list[tuple[float, int, str]]] = collections.defaultdict(list)
  with path.open('r', encoding='utf-8-sig', newline='') as stream:
    for line_number, row in enumerate(csv.reader(stream), start=1):
      if not row or all(not field.strip() for field in row):
        continue
      if len(row) < 3:
        raise ValueError(f'{path}:{line_number}: expected at least 3 columns.')
      first, second, raw_timestamp = (field.strip() for field in row[:3])
      if pair_order == 'item-user-time':
        item, user = first, second
      else:
        user, item = first, second
      if not user or not item:
        raise ValueError(
            f'{path}:{line_number}: user and item IDs must be non-empty.'
        )
      timestamp = _timestamp(raw_timestamp, path, line_number)
      events[user].append((timestamp, line_number, item))
  if not events:
    raise ValueError(f'No interaction pairs found in {path}.')
  return {
      user: [item for _, _, item in sorted(user_events)]
      for user, user_events in events.items()
  }


def load_item_texts(path: Path) -> dict[str, tuple[str, str]]:
  """Load item ID, Chinese text, and English text from *_item.csv."""
  items: dict[str, tuple[str, str]] = {}
  with path.open('r', encoding='utf-8-sig', newline='') as stream:
    for line_number, row in enumerate(csv.reader(stream), start=1):
      if not row or all(not field.strip() for field in row):
        continue
      item = row[0].strip()
      if line_number == 1 and item.lower() in {
          'item_id', 'video_id', 'item id', 'video id'
      }:
        continue
      if not item:
        raise ValueError(f'{path}:{line_number}: empty item ID.')
      if item in items:
        raise ValueError(f'{path}:{line_number}: duplicate item {item!r}.')
      chinese = row[1].strip() if len(row) > 1 else ''
      english = row[2].strip() if len(row) > 2 else ''
      if not chinese and not english:
        raise ValueError(f'{path}:{line_number}: item {item!r} has no text.')
      items[item] = (chinese, english)
  if not items:
    raise ValueError(f'No item metadata found in {path}.')
  return items


def format_item_text(chinese: str, english: str, language: str) -> str:
  if language not in TEXT_LANGUAGES:
    raise ValueError(
        f'text_language must be one of {sorted(TEXT_LANGUAGES)}, '
        f'got {language!r}.'
    )
  if language == 'zh':
    return chinese or english
  if language == 'en':
    return english or chinese
  if chinese and english:
    return f'Chinese: {chinese}\nEnglish: {english}'
  return chinese or english


def _find_existing(source_dir: Path, candidates: Iterable[str]) -> Path | None:
  for candidate in candidates:
    path = source_dir / candidate
    if path.is_file():
      return path
  return None


def discover_source_paths(
    source_dir: Path, subset: str
) -> tuple[Path | None, Path | None, Path, Path | None]:
  """Discover official files while allowing common archive layouts."""
  roots = [source_dir]
  nested = source_dir / subset
  if nested.is_dir():
    roots.insert(0, nested)
  for root in roots:
    behaviour = _find_existing(
        root, (f'{subset}_behaviour.tsv', 'behaviour.tsv')
    )
    pairs = _find_existing(root, (f'{subset}_pair.csv', 'pair.csv'))
    item_path = _find_existing(root, (f'{subset}_item.csv', 'item.csv'))
    cover_candidates = (root / f'{subset}_cover', root / 'cover')
    cover_dir = next((path for path in cover_candidates if path.is_dir()), None)
    if item_path is not None and (behaviour is not None or pairs is not None):
      return behaviour, pairs, item_path, cover_dir
  raise FileNotFoundError(
      f'Could not find {subset}_item.csv plus {subset}_behaviour.tsv or '
      f'{subset}_pair.csv below {source_dir}.'
  )


def _image_index(cover_dir: Path | None) -> dict[str, Path]:
  if cover_dir is None:
    return {}
  images: dict[str, Path] = {}
  for path in sorted(cover_dir.iterdir()):
    if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
      continue
    item = path.stem
    if item in images:
      raise ValueError(
          f'Multiple cover images found for item {item!r}: '
          f'{images[item]} and {path}.'
      )
    # Preserve relative source paths so a self-contained cache can be moved to
    # the training machine without embedding this computer's absolute path.
    images[item] = path
  return images


def _build_mapping(
    sequences: dict[str, list[str]]
) -> tuple[dict[str, Any], list[str]]:
  id_mapping: dict[str, Any] = {
      'user2id': {'[PAD]': 0},
      'item2id': {'[PAD]': 0},
      'id2user': ['[PAD]'],
      'id2item': ['[PAD]'],
  }
  for user, sequence in sequences.items():
    id_mapping['user2id'][user] = len(id_mapping['id2user'])
    id_mapping['id2user'].append(user)
    for item in sequence:
      if item not in id_mapping['item2id']:
        id_mapping['item2id'][item] = len(id_mapping['id2item'])
        id_mapping['id2item'].append(item)
  return id_mapping, id_mapping['id2item'][1:]


def build_ninerec_processed(
    source_dir: Path,
    output_dir: Path,
    subset: str,
    text_language: str = 'bilingual',
    pair_order: str = 'item-user-time',
    require_all_images: bool = False,
    behaviour_path: Path | None = None,
    pair_path: Path | None = None,
    item_path: Path | None = None,
    cover_dir: Path | None = None,
) -> dict[str, Any]:
  """Build the ActionPiece cache contract from an official NineRec subset."""
  if not subset or '/' in subset or '\\' in subset:
    raise ValueError(f'Invalid NineRec subset name: {subset!r}.')
  if text_language not in TEXT_LANGUAGES:
    raise ValueError(
        f'text_language must be one of {sorted(TEXT_LANGUAGES)}.'
    )
  if pair_order not in PAIR_ORDERS:
    raise ValueError(f'pair_order must be one of {sorted(PAIR_ORDERS)}.')

  if item_path is None or (behaviour_path is None and pair_path is None):
    discovered = discover_source_paths(source_dir, subset)
    behaviour_path = behaviour_path or discovered[0]
    pair_path = pair_path or discovered[1]
    item_path = item_path or discovered[2]
    cover_dir = cover_dir or discovered[3]

  if behaviour_path is not None:
    sequences = load_behaviour(behaviour_path)
    interaction_source = behaviour_path
    interaction_format = 'behaviour'
  elif pair_path is not None:
    sequences = load_pairs(pair_path, pair_order=pair_order)
    interaction_source = pair_path
    interaction_format = 'pair'
  else:
    raise ValueError('Either behaviour_path or pair_path is required.')
  if item_path is None:
    raise ValueError('item_path is required.')

  id_mapping, item_ids = _build_mapping(sequences)
  raw_texts = load_item_texts(item_path)
  missing_text = sorted(set(item_ids) - set(raw_texts))
  if missing_text:
    raise ValueError(
        f'Item metadata is missing for {len(missing_text)} interacted items; '
        f'examples: {missing_text[:5]}.'
    )
  text_metadata = {
      item: format_item_text(*raw_texts[item], text_language)
      for item in item_ids
  }

  images = _image_index(cover_dir)
  image_records = []
  missing_images = []
  image_metadata = {}
  for item in item_ids:
    image_path = images.get(item)
    if image_path is None:
      missing_images.append(item)
      record = {'item_id': item, 'status': 'missing', 'path': None}
      image_value = None
    else:
      record = {
          'item_id': item,
          'status': 'downloaded',
          'path': str(image_path),
      }
      image_value = str(image_path)
    image_records.append(record)
    image_metadata[item] = {
        'sentence': text_metadata[item],
        'image_url': image_value,
    }
  if require_all_images and missing_images:
    raise ValueError(
        f'Cover images are missing for {len(missing_images)} interacted items; '
        f'examples: {missing_images[:5]}.'
    )

  interactions = sum(len(sequence) for sequence in sequences.values())
  item_sources = {item: [subset] for item in item_ids}
  output_dir.mkdir(parents=True, exist_ok=True)
  outputs: dict[str, object] = {
      'all_item_seqs.json': sequences,
      'id_mapping.json': id_mapping,
      'item_sources.json': item_sources,
      'metadata.sentence.json': text_metadata,
      'metadata.qwen_text.json': text_metadata,
  }
  for mode in IMAGE_METADATA_MODES:
    outputs[f'metadata.{mode}.json'] = image_metadata
  for filename, value in outputs.items():
    _write_json(output_dir / filename, value)
  manifest_path = output_dir / 'image_download_manifest.jsonl'
  _write_jsonl(manifest_path, image_records)

  summary = {
      'dataset': 'NineRec',
      'subset': subset,
      'text_language': text_language,
      'interaction_format': interaction_format,
      'pair_order': pair_order if interaction_format == 'pair' else None,
      'users': len(sequences),
      'items': len(item_ids),
      'interactions': interactions,
      'metadata_items': len(text_metadata),
      'cover_items': len(item_ids) - len(missing_images),
      'missing_cover_items': len(missing_images),
      'missing_cover_examples': missing_images[:10],
      'interaction_source': str(interaction_source),
      'item_source': str(item_path),
      'cover_dir': str(cover_dir) if cover_dir is not None else None,
      'image_manifest_sha256': sha256_file(manifest_path),
  }
  _write_json(output_dir / 'ninerec_summary.json', summary)
  return summary


class NineRec(AbstractDataset):
  """Load a prepared NineRec source or downstream subset."""

  def __init__(self, config: dict[str, Any]):
    super().__init__(config)
    self.subset = str(config['subset'])
    self.category = self.subset
    self.cache_dir = os.path.join(config['cache_dir'], 'NineRec', self.subset)
    self.log(f'[DATASET] NineRec subset: {self.subset}')
    self._download_and_process_raw()

  def _download_and_process_raw(self) -> None:
    root = Path(self.cache_dir)
    processed_dir = root / 'processed'
    required = (
        processed_dir / 'all_item_seqs.json',
        processed_dir / 'id_mapping.json',
        processed_dir / f'metadata.{self.config["metadata"]}.json',
    )
    if not all(path.is_file() for path in required):
      raw_dir = root / 'raw'
      if not raw_dir.is_dir():
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(
            'NineRec is not prepared. Put the official subset under '
            f'{raw_dir} or run scripts/prepare_ninerec_data.py. Missing: '
            f'{missing}'
        )
      with self.accelerator.main_process_first():
        build_ninerec_processed(
            source_dir=raw_dir,
            output_dir=processed_dir,
            subset=self.subset,
            text_language=self.config.get(
                'ninerec_text_language', 'bilingual'
            ),
            pair_order=self.config.get(
                'ninerec_pair_order', 'item-user-time'
            ),
            require_all_images=bool(
                self.config.get('ninerec_require_all_images', False)
            ),
        )

    self.all_item_seqs = json.loads(
        (processed_dir / 'all_item_seqs.json').read_text(encoding='utf-8')
    )
    self.id_mapping = json.loads(
        (processed_dir / 'id_mapping.json').read_text(encoding='utf-8')
    )
    item_sources_path = processed_dir / 'item_sources.json'
    self.item_sources = (
        json.loads(item_sources_path.read_text(encoding='utf-8'))
        if item_sources_path.is_file()
        else {item: [self.subset] for item in self.id_mapping['id2item'][1:]}
    )
    metadata_path = processed_dir / f'metadata.{self.config["metadata"]}.json'
    self.item2meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    expected_items = set(self.id_mapping['id2item'][1:])
    if set(self.item2meta) != expected_items:
      raise RuntimeError(
          f'{metadata_path} keys do not align with id_mapping.json.'
      )
    sequence_items = {
        item for sequence in self.all_item_seqs.values() for item in sequence
    }
    if sequence_items != expected_items:
      raise RuntimeError(
          'all_item_seqs.json items do not align with id_mapping.json.'
      )
