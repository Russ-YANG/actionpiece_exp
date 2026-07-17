"""Resumable Qwen multimodal embedding API client for ActionPiece."""

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np
import requests


DEFAULT_ENDPOINT = (
    'https://dashscope.aliyuncs.com/api/v1/services/embeddings/'
    'multimodal-embedding/multimodal-embedding'
)


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
  """Read simple KEY=VALUE entries without modifying process environment."""
  env_path = Path(path)
  if not env_path.exists():
    return {}
  values = {}
  for raw_line in env_path.read_text(encoding='utf-8').splitlines():
    line = raw_line.strip()
    if not line or line.startswith('#') or '=' not in line:
      continue
    key, value = line.split('=', 1)
    values[key.strip()] = value.strip().strip('"').strip("'")
  return values


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(
      json.dumps(value, ensure_ascii=False, indent=2) + '\n',
      encoding='utf-8',
  )
  temporary.replace(path)


def _load_progress(path: Path) -> dict[str, Any]:
  if not path.exists():
    return {}
  return json.loads(path.read_text(encoding='utf-8'))


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def _summarize_request_log(path: Path) -> tuple[int, dict[str, int]]:
  calls = 0
  usage: dict[str, int] = {}
  if not path.exists():
    return calls, usage
  with path.open(encoding='utf-8') as stream:
    for line in stream:
      record = json.loads(line)
      calls += 1
      for key, value in (record.get('usage') or {}).items():
        if isinstance(value, int):
          usage[key] = usage.get(key, 0) + value
  return calls, usage


class QwenApiTextEncoder:
  """Encode text batches with resumable, item-aligned binary output."""

  def __init__(
      self,
      model: str,
      instruction: str,
      dimension: int,
      batch_size: int = 20,
      timeout: float = 120.0,
      max_retries: int = 5,
      env_file: str = '.env.local',
      endpoint: str | None = None,
      request_post: Callable[..., Any] | None = None,
  ):
    if dimension <= 0:
      raise ValueError('Embedding dimension must be positive.')
    if not 1 <= batch_size <= 20:
      raise ValueError('Qwen API batch size must be between 1 and 20.')
    self.model = model
    self.instruction = instruction
    self.dimension = dimension
    self.batch_size = batch_size
    self.timeout = timeout
    self.max_retries = max(max_retries, 1)
    self.env = read_env_file(env_file)
    self.endpoint = (
        endpoint
        or self.env.get('DASHSCOPE_BASE_URL')
        or DEFAULT_ENDPOINT
    )
    self.api_key = (
        os.getenv('DASHSCOPE_API_KEY')
        or self.env.get('DASHSCOPE_API_KEY')
    )
    if not self.api_key:
      raise RuntimeError(
          'DASHSCOPE_API_KEY was not found in the environment or env file.'
      )
    self.request_post = request_post or requests.post

  def _request(self, texts: Sequence[str]) -> dict[str, Any]:
    payload = {
        'model': self.model,
        'input': {'contents': [{'text': text} for text in texts]},
        'parameters': {
            'dimension': self.dimension,
            'instruct': self.instruction,
            'enable_fusion': False,
        },
    }
    last_error = None
    for attempt in range(self.max_retries):
      try:
        response = self.request_post(
            self.endpoint,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json=payload,
            timeout=self.timeout,
        )
        try:
          body = response.json()
        except (requests.JSONDecodeError, ValueError) as exc:
          raise RuntimeError(
              f'Qwen API returned HTTP {response.status_code} with non-JSON '
              'content.'
          ) from exc
        if response.status_code == 200 and 'output' in body:
          return body
        code = body.get('code', 'unknown')
        message = body.get('message', 'No error message returned.')
        error = RuntimeError(
            f'Qwen API failed: HTTP {response.status_code}, {code}: {message}'
        )
        # Authentication and invalid requests will not succeed after a retry.
        if response.status_code in {400, 401, 403}:
          raise error
        last_error = error
      except requests.RequestException as exc:
        last_error = exc
      if attempt + 1 < self.max_retries:
        time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f'Qwen API request failed after {self.max_retries} attempts.'
    ) from last_error

  def encode(
      self,
      texts: Sequence[str],
      item_ids: Sequence[str],
      output_path: str | os.PathLike[str],
      progress_callback: Callable[[int, int], None] | None = None,
  ) -> np.ndarray:
    """Encode all texts, resuming only from a committed batch boundary."""
    if len(texts) != len(item_ids):
      raise ValueError('texts and item_ids must have the same length.')
    if not texts:
      raise ValueError('At least one text is required.')

    final_path = Path(output_path)
    partial_path = Path(f'{final_path}.partial')
    progress_path = Path(f'{final_path}.progress.json')
    request_log_path = Path(f'{final_path}.requests.jsonl')
    final_path.parent.mkdir(parents=True, exist_ok=True)

    expected_bytes = len(texts) * self.dimension * np.dtype(np.float32).itemsize
    instruction_sha256 = hashlib.sha256(
        self.instruction.encode('utf-8')
    ).hexdigest()
    item_order_sha256 = hashlib.sha256(
        '\n'.join(item_ids).encode('utf-8')
    ).hexdigest()
    progress = _load_progress(progress_path)
    next_index = int(progress.get('next_index', 0))
    expected_identity = {
        'model': self.model,
        'instruction_sha256': instruction_sha256,
        'dimension': self.dimension,
        'item_count': len(texts),
        'item_order_sha256': item_order_sha256,
    }
    if next_index:
      mismatched = [
          key
          for key, expected in expected_identity.items()
          if progress.get(key) != expected
      ]
      if mismatched:
        raise RuntimeError(
            'Partial Qwen cache does not match this run: '
            + ', '.join(mismatched)
        )

    if next_index == 0:
      partial_path.unlink(missing_ok=True)
      request_log_path.unlink(missing_ok=True)
      embeddings = np.memmap(
          partial_path,
          dtype=np.float32,
          mode='w+',
          shape=(len(texts), self.dimension),
      )
    else:
      if (
          not partial_path.exists()
          or partial_path.stat().st_size != expected_bytes
      ):
        raise RuntimeError('Partial Qwen embedding file is missing or invalid.')
      embeddings = np.memmap(
          partial_path,
          dtype=np.float32,
          mode='r+',
          shape=(len(texts), self.dimension),
      )

    for start in range(next_index, len(texts), self.batch_size):
      end = min(start + self.batch_size, len(texts))
      body = self._request(texts[start:end])
      results = body['output'].get('embeddings', [])
      if len(results) != end - start:
        raise RuntimeError(
            'Qwen API returned an unexpected number of embeddings: '
            f'expected {end - start}, got {len(results)}.'
        )
      results = sorted(results, key=lambda result: result.get('index', 0))
      vectors = np.asarray(
          [result['embedding'] for result in results], dtype=np.float32
      )
      if vectors.shape != (end - start, self.dimension):
        raise RuntimeError(
            f'Qwen API returned embedding shape {vectors.shape}; expected '
            f'{(end - start, self.dimension)}.'
        )
      embeddings[start:end] = vectors
      embeddings.flush()
      with request_log_path.open('a', encoding='utf-8') as request_log:
        request_log.write(
            json.dumps(
                {
                    'start': start,
                    'end': end,
                    'first_item': item_ids[start],
                    'last_item': item_ids[end - 1],
                    'request_id': body.get('request_id'),
                    'usage': body.get('usage'),
                },
                ensure_ascii=False,
            )
            + '\n'
        )
      _write_json_atomic(
          progress_path,
          {
              **expected_identity,
              'next_index': end,
          },
      )
      if progress_callback:
        progress_callback(end, len(texts))

    embeddings.flush()
    del embeddings
    partial_path.replace(final_path)
    progress_path.unlink(missing_ok=True)
    calls, usage = _summarize_request_log(request_log_path)
    _write_json_atomic(
        Path(f'{final_path}.manifest.json'),
        {
            'backend': 'qwen_api',
            'model': self.model,
            'instruction': self.instruction,
            'instruction_sha256': instruction_sha256,
            'dimension': self.dimension,
            'item_count': len(texts),
            'first_item': item_ids[0],
            'last_item': item_ids[-1],
            'item_order_sha256': item_order_sha256,
            'api_calls': calls,
            'usage': usage,
            'embedding_file': str(final_path),
            'embedding_sha256': _sha256_file(final_path),
            'request_log': str(request_log_path),
        },
    )
    return np.fromfile(final_path, dtype=np.float32).reshape(
        len(texts), self.dimension
    )
