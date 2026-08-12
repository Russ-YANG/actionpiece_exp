"""Helpers for item-aligned local image manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def manifest_item_id(record: dict[str, Any]) -> str:
  """Return a generic item_id while accepting legacy Amazon manifests."""
  item_id = record.get('item_id')
  asin = record.get('asin')
  if item_id is not None and asin is not None and item_id != asin:
    raise ValueError(
        f'Image manifest record has conflicting item_id and asin: {record}'
    )
  value = item_id if item_id is not None else asin
  if not isinstance(value, str) or not value:
    raise ValueError(
        'Image manifest record requires a non-empty string item_id '
        f'(or legacy asin): {record}'
    )
  return value


def load_image_manifest(path: Path) -> dict[str, dict[str, Any]]:
  """Load a JSONL image manifest and reject duplicate item records."""
  records: dict[str, dict[str, Any]] = {}
  with path.open('r', encoding='utf-8') as stream:
    for line_number, line in enumerate(stream, start=1):
      if not line.strip():
        continue
      try:
        record = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f'{path}:{line_number}: invalid JSON.') from exc
      item = manifest_item_id(record)
      if item in records:
        raise ValueError(f'{path}:{line_number}: duplicate item {item!r}.')
      records[item] = record
  if not records:
    raise ValueError(f'Image manifest is empty: {path}.')
  return records


def available_image_path(record: dict[str, Any]) -> Path | None:
  """Return an existing downloaded image path, otherwise None."""
  raw_path = record.get('path')
  path = Path(raw_path) if isinstance(raw_path, str) and raw_path else None
  if (
      record.get('status') == 'downloaded'
      and path is not None
      and path.is_file()
  ):
    return path
  return None
