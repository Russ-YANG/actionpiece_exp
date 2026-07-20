"""Resumable local Qwen3-VL text embedding for ActionPiece."""

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np


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


class QwenLocalTextEncoder:
  """Encode text with a local Qwen3-VL model and resumable output."""

  def __init__(
      self,
      model_path: str,
      model_id: str,
      model_revision: str,
      repo_path: str,
      code_revision: str,
      instruction: str,
      dimension: int,
      batch_size: int = 8,
      max_length: int = 2048,
      torch_dtype: str = 'bfloat16',
      attn_implementation: str = 'sdpa',
      require_cuda: bool = True,
      model_factory: Callable[..., Any] | None = None,
  ):
    if dimension <= 0:
      raise ValueError('Embedding dimension must be positive.')
    if batch_size <= 0:
      raise ValueError('Local Qwen batch size must be positive.')
    if max_length <= 0:
      raise ValueError('Local Qwen max length must be positive.')
    self.model_path = model_path
    self.model_id = model_id
    self.model_revision = model_revision
    self.repo_path = repo_path
    self.code_revision = code_revision
    self.instruction = instruction
    self.dimension = dimension
    self.batch_size = batch_size
    self.max_length = max_length
    self.torch_dtype = torch_dtype
    self.attn_implementation = attn_implementation
    self.require_cuda = require_cuda
    self.model_factory = model_factory

  def _load_model(self):
    import torch

    if self.require_cuda and not torch.cuda.is_available():
      raise RuntimeError(
          'Local Qwen encoding requires CUDA, but no CUDA device is available.'
      )
    dtype = getattr(torch, self.torch_dtype, None)
    if dtype is None:
      raise ValueError(f'Unsupported torch dtype: {self.torch_dtype}')
    model_factory = self.model_factory
    if model_factory is None:
      repo_path = Path(self.repo_path).expanduser().resolve()
      module_path = repo_path / 'src/models/qwen3_vl_embedding.py'
      if not module_path.exists():
        raise RuntimeError(
            f'Qwen3-VL-Embedding source was not found at {module_path}.'
        )
      if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
      try:
        from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
      except ImportError as exc:
        raise RuntimeError(
            'Qwen3-VL-Embedding is not installed. Install the official '
            'QwenLM/Qwen3-VL-Embedding repository in this environment.'
        ) from exc
      model_factory = Qwen3VLEmbedder
    return model_factory(
        model_name_or_path=self.model_path,
        max_length=self.max_length,
        dtype=dtype,
        attn_implementation=self.attn_implementation,
    )

  def encode(
      self,
      texts: Sequence[str],
      item_ids: Sequence[str],
      output_path: str | Path,
      progress_callback: Callable[[int, int], None] | None = None,
  ) -> np.ndarray:
    """Encode all texts, resuming from the last committed batch boundary."""
    import torch

    if len(texts) != len(item_ids):
      raise ValueError('texts and item_ids must have the same length.')
    if not texts:
      raise ValueError('At least one text is required.')

    final_path = Path(output_path)
    partial_path = Path(f'{final_path}.partial')
    progress_path = Path(f'{final_path}.progress.json')
    final_path.parent.mkdir(parents=True, exist_ok=True)

    instruction_sha256 = hashlib.sha256(
        self.instruction.encode('utf-8')
    ).hexdigest()
    item_order_sha256 = hashlib.sha256(
        '\n'.join(item_ids).encode('utf-8')
    ).hexdigest()
    expected_identity = {
        'backend': 'qwen_local',
        'model_id': self.model_id,
        'model_revision': self.model_revision,
        'code_revision': self.code_revision,
        'instruction_sha256': instruction_sha256,
        'dimension': self.dimension,
        'max_length': self.max_length,
        'torch_dtype': self.torch_dtype,
        'attn_implementation': self.attn_implementation,
        'normalization': 'truncate_then_l2',
        'item_count': len(texts),
        'item_order_sha256': item_order_sha256,
    }
    progress = _load_progress(progress_path)
    next_index = int(progress.get('next_index', 0))
    if next_index:
      mismatched = [
          key
          for key, expected in expected_identity.items()
          if progress.get(key) != expected
      ]
      if mismatched:
        raise RuntimeError(
            'Partial local Qwen cache does not match this run: '
            + ', '.join(mismatched)
        )

    expected_bytes = len(texts) * self.dimension * np.dtype(np.float32).itemsize
    if next_index == 0:
      partial_path.unlink(missing_ok=True)
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
        raise RuntimeError(
            'Partial local Qwen embedding file is missing or invalid.'
        )
      embeddings = np.memmap(
          partial_path,
          dtype=np.float32,
          mode='r+',
          shape=(len(texts), self.dimension),
      )

    model = None
    native_dimension = progress.get('native_dimension')
    if next_index < len(texts):
      model = self._load_model()
    try:
      for start in range(next_index, len(texts), self.batch_size):
        end = min(start + self.batch_size, len(texts))
        model_inputs = [
            {'text': text, 'instruction': self.instruction}
            for text in texts[start:end]
        ]
        batch_embeddings = model.process(model_inputs, normalize=True)
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != end - start:
          raise RuntimeError(
              'Local Qwen returned an unexpected embedding shape: '
              f'{tuple(batch_embeddings.shape)}.'
          )
        native_dimension = int(batch_embeddings.shape[1])
        if self.dimension > native_dimension:
          raise RuntimeError(
              f'Requested dimension {self.dimension} exceeds the local Qwen '
              f'native dimension {native_dimension}.'
          )
        # Qwen3-VL uses Matryoshka representations. Truncate to the requested
        # prefix and normalize again so all configured dimensions have unit norm.
        batch_embeddings = batch_embeddings[:, : self.dimension].float()
        batch_embeddings = torch.nn.functional.normalize(
            batch_embeddings, p=2, dim=-1
        )
        vectors = batch_embeddings.cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(vectors).all():
          raise RuntimeError('Local Qwen returned NaN or Inf embeddings.')
        embeddings[start:end] = vectors
        embeddings.flush()
        _write_json_atomic(
            progress_path,
            {
                **expected_identity,
                'native_dimension': native_dimension,
                'next_index': end,
            },
        )
        if progress_callback:
          progress_callback(end, len(texts))
    finally:
      if model is not None:
        del model
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embeddings.flush()
    del embeddings
    partial_path.replace(final_path)
    progress_path.unlink(missing_ok=True)
    _write_json_atomic(
        Path(f'{final_path}.manifest.json'),
        {
            **expected_identity,
            'model_path': self.model_path,
            'repo_path': self.repo_path,
            'native_dimension': native_dimension,
            'batch_size': self.batch_size,
            'instruction': self.instruction,
            'first_item': item_ids[0],
            'last_item': item_ids[-1],
            'embedding_file': str(final_path),
            'embedding_sha256': _sha256_file(final_path),
            'torch_version': torch.__version__,
        },
    )
    return np.fromfile(final_path, dtype=np.float32).reshape(
        len(texts), self.dimension
    )
