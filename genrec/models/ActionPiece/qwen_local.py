"""Resumable local Qwen3-VL text embedding for ActionPiece."""

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np


def build_qwen_local_artifact_stem(
    *,
    backend: str,
    model_id: str,
    dimension: int,
    instruction: str,
    model_revision: str,
    code_revision: str,
    max_length: int,
    torch_dtype: str,
    attn_implementation: str,
    modality: str = 'text',
    image_manifest_sha256: str | None = None,
) -> str:
  """Build a portable cache identity for local Qwen embeddings."""
  import re

  model_name = Path(model_id).name
  safe_model_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model_name)
  instruction_hash = hashlib.sha256(instruction.encode('utf-8')).hexdigest()[:12]
  revision = re.sub(r'[^A-Za-z0-9._-]+', '-', model_revision)[:12]
  source_revision = re.sub(r'[^A-Za-z0-9._-]+', '-', code_revision)[:12]
  dtype = re.sub(r'[^A-Za-z0-9._-]+', '-', torch_dtype)
  attention = re.sub(r'[^A-Za-z0-9._-]+', '-', attn_implementation)
  stem = (
      f'{backend}.{safe_model_name}.d{dimension}.i{instruction_hash}.'
      f'r{revision}.c{source_revision}.m{max_length}.t{dtype}.a{attention}'
  )
  if modality == 'text_image_joint':
    if not image_manifest_sha256 or len(image_manifest_sha256) != 64:
      raise ValueError(
          'Joint multimodal caches require a full image manifest SHA-256.'
      )
    stem += f'.joint.im{image_manifest_sha256[:12]}'
  elif modality == 'image':
    if not image_manifest_sha256 or len(image_manifest_sha256) != 64:
      raise ValueError(
          'Image-only caches require a full image manifest SHA-256.'
      )
    stem += f'.image.im{image_manifest_sha256[:12]}'
  elif modality != 'text':
    raise ValueError(f'Unsupported Qwen modality: {modality}')
  return stem


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


class QwenLocalMultimodalEncoder(QwenLocalTextEncoder):
  """Encode aligned product text and images into one joint Qwen vector."""

  def encode(
      self,
      texts: Sequence[str],
      image_paths: Sequence[str | Path | None],
      item_ids: Sequence[str],
      output_path: str | Path,
      image_manifest_sha256: str,
      progress_callback: Callable[[int, int], None] | None = None,
  ) -> np.ndarray:
    """Encode joint inputs, using text-only fallback for missing images."""
    import torch

    if not (len(texts) == len(image_paths) == len(item_ids)):
      raise ValueError('texts, image_paths, and item_ids must have the same length.')
    if not texts:
      raise ValueError('At least one multimodal input is required.')
    if len(image_manifest_sha256) != 64:
      raise ValueError('image_manifest_sha256 must be a full SHA-256 digest.')

    resolved_images: list[str | None] = []
    missing_image_items = []
    image_input_digest = hashlib.sha256()
    for item_id, image_path in zip(item_ids, image_paths):
      path = Path(image_path).expanduser().resolve() if image_path else None
      image_input_digest.update(item_id.encode('utf-8'))
      image_input_digest.update(b'\0')
      if path is None or not path.is_file():
        resolved_images.append(None)
        missing_image_items.append(item_id)
        image_input_digest.update(b'MISSING\n')
      else:
        resolved_images.append(str(path))
        image_input_digest.update(_sha256_file(path).encode('ascii'))
        image_input_digest.update(b'\n')

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
        'backend': 'qwen_local_multimodal',
        'modality': 'text_image_joint',
        'model_id': self.model_id,
        'model_revision': self.model_revision,
        'code_revision': self.code_revision,
        'instruction_sha256': instruction_sha256,
        'dimension': self.dimension,
        'max_length': self.max_length,
        'torch_dtype': self.torch_dtype,
        'attn_implementation': self.attn_implementation,
        'normalization': 'truncate_then_l2',
        'missing_image_policy': 'text_only',
        'image_manifest_sha256': image_manifest_sha256,
        'image_input_sha256': image_input_digest.hexdigest(),
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
            'Partial local Qwen multimodal cache does not match this run: '
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
            'Partial local Qwen multimodal embedding file is missing or invalid.'
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
        model_inputs = []
        for text, image_path in zip(
            texts[start:end], resolved_images[start:end]
        ):
          model_input = {
              'text': text,
              'instruction': self.instruction,
          }
          if image_path is not None:
            model_input['image'] = image_path
          model_inputs.append(model_input)
        batch_embeddings = model.process(model_inputs, normalize=True)
        if batch_embeddings.ndim != 2 or batch_embeddings.shape[0] != end - start:
          raise RuntimeError(
              'Local Qwen returned an unexpected multimodal embedding shape: '
              f'{tuple(batch_embeddings.shape)}.'
          )
        native_dimension = int(batch_embeddings.shape[1])
        if self.dimension > native_dimension:
          raise RuntimeError(
              f'Requested dimension {self.dimension} exceeds the local Qwen '
              f'native dimension {native_dimension}.'
          )
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
            'joint_item_count': len(texts) - len(missing_image_items),
            'missing_image_count': len(missing_image_items),
            'missing_image_items': missing_image_items,
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


class QwenLocalImageEncoder(QwenLocalTextEncoder):
  """Encode product images alone, using zero vectors for missing images."""

  def encode(
      self,
      image_paths: Sequence[str | Path | None],
      item_ids: Sequence[str],
      output_path: str | Path,
      image_manifest_sha256: str,
      progress_callback: Callable[[int, int], None] | None = None,
  ) -> np.ndarray:
    """Encode image-only inputs in item order with resumable writes."""
    import torch

    if len(image_paths) != len(item_ids):
      raise ValueError('image_paths and item_ids must have the same length.')
    if not item_ids:
      raise ValueError('At least one image input is required.')
    if len(image_manifest_sha256) != 64:
      raise ValueError('image_manifest_sha256 must be a full SHA-256 digest.')

    resolved_images: list[str | None] = []
    missing_image_items = []
    image_input_digest = hashlib.sha256()
    for item_id, image_path in zip(item_ids, image_paths):
      path = Path(image_path).expanduser().resolve() if image_path else None
      image_input_digest.update(item_id.encode('utf-8'))
      image_input_digest.update(b'\0')
      if path is None or not path.is_file():
        resolved_images.append(None)
        missing_image_items.append(item_id)
        image_input_digest.update(b'MISSING\n')
      else:
        resolved_images.append(str(path))
        image_input_digest.update(_sha256_file(path).encode('ascii'))
        image_input_digest.update(b'\n')

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
        'backend': 'qwen_local_image',
        'modality': 'image',
        'model_id': self.model_id,
        'model_revision': self.model_revision,
        'code_revision': self.code_revision,
        'instruction_sha256': instruction_sha256,
        'dimension': self.dimension,
        'max_length': self.max_length,
        'torch_dtype': self.torch_dtype,
        'attn_implementation': self.attn_implementation,
        'normalization': 'truncate_then_l2_valid_images',
        'missing_image_policy': 'zero_vector',
        'image_manifest_sha256': image_manifest_sha256,
        'image_input_sha256': image_input_digest.hexdigest(),
        'item_count': len(item_ids),
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
            'Partial local Qwen image cache does not match this run: '
            + ', '.join(mismatched)
        )

    expected_bytes = len(item_ids) * self.dimension * np.dtype(np.float32).itemsize
    if next_index == 0:
      partial_path.unlink(missing_ok=True)
      embeddings = np.memmap(
          partial_path,
          dtype=np.float32,
          mode='w+',
          shape=(len(item_ids), self.dimension),
      )
      embeddings[:] = 0
      embeddings.flush()
    else:
      if (
          not partial_path.exists()
          or partial_path.stat().st_size != expected_bytes
      ):
        raise RuntimeError(
            'Partial local Qwen image embedding file is missing or invalid.'
        )
      embeddings = np.memmap(
          partial_path,
          dtype=np.float32,
          mode='r+',
          shape=(len(item_ids), self.dimension),
      )

    model = None
    native_dimension = progress.get('native_dimension')
    if any(path is not None for path in resolved_images[next_index:]):
      model = self._load_model()
    try:
      for start in range(next_index, len(item_ids), self.batch_size):
        end = min(start + self.batch_size, len(item_ids))
        valid_positions = [
            idx
            for idx in range(start, end)
            if resolved_images[idx] is not None
        ]
        if valid_positions:
          model_inputs = [
              {
                  'image': resolved_images[idx],
                  'instruction': self.instruction,
              }
              for idx in valid_positions
          ]
          batch_embeddings = model.process(model_inputs, normalize=True)
          if (
              batch_embeddings.ndim != 2
              or batch_embeddings.shape[0] != len(valid_positions)
          ):
            raise RuntimeError(
                'Local Qwen returned an unexpected image embedding shape: '
                f'{tuple(batch_embeddings.shape)}.'
            )
          native_dimension = int(batch_embeddings.shape[1])
          if self.dimension > native_dimension:
            raise RuntimeError(
                f'Requested dimension {self.dimension} exceeds the local Qwen '
                f'native dimension {native_dimension}.'
            )
          batch_embeddings = batch_embeddings[:, : self.dimension].float()
          batch_embeddings = torch.nn.functional.normalize(
              batch_embeddings, p=2, dim=-1
          )
          vectors = batch_embeddings.cpu().numpy().astype(
              np.float32, copy=False
          )
          if not np.isfinite(vectors).all():
            raise RuntimeError('Local Qwen returned NaN or Inf embeddings.')
          for idx, vector in zip(valid_positions, vectors):
            embeddings[idx] = vector
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
          progress_callback(end, len(item_ids))
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
            'encoded_image_count': len(item_ids) - len(missing_image_items),
            'missing_image_count': len(missing_image_items),
            'missing_image_items': missing_image_items,
            'first_item': item_ids[0],
            'last_item': item_ids[-1],
            'embedding_file': str(final_path),
            'embedding_sha256': _sha256_file(final_path),
            'torch_version': torch.__version__,
        },
    )
    return np.fromfile(final_path, dtype=np.float32).reshape(
        len(item_ids), self.dimension
    )
