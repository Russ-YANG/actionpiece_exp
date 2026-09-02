# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ActionPiece tokenizer for GenRec."""

import collections
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import faiss
from genrec.dataset import AbstractDataset
from genrec.image_manifest import available_image_path
from genrec.image_manifest import load_image_manifest
from genrec.image_manifest import sha256_file
from genrec.models.ActionPiece.core import ActionPieceCore
from genrec.models.ActionPiece.qwen_api import QwenApiTextEncoder
from genrec.models.ActionPiece.qwen_local import build_qwen_local_artifact_stem
from genrec.models.ActionPiece.qwen_local import QwenLocalMultimodalEncoder
from genrec.models.ActionPiece.qwen_local import QwenLocalTextEncoder
from genrec.tokenizer import AbstractTokenizer
import numpy as np
from PIL import Image
import requests
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import tqdm
from transformers import CLIPModel
from transformers import CLIPProcessor


class ActionPieceTokenizer(AbstractTokenizer):
  """The ActionPiece tokenizer is a tokenizer that encodes the attribute.

  features.

  Attributes:
      item2feat (dict): A dictionary mapping item IDs to their features.
      ignored_label (int): The label to be used for padding tokens.
      actionpiece (ActionPieceCore): ActionPiece core tokenizer.
      bos_token (int): The beginning token.
      eos_token (int): The end token.
      n_inference_ensemble (int): The number of inference ensemble.
      train_shuffle (str): The shuffle strategy for training.
      encoded_labels (dict): A dictionary mapping label sequences to their
        encoded representations.
      collate_fn (dict): A dictionary mapping split names to their corresponding
        collate functions.
  """

  def __init__(self, config: dict[Any, Any], dataset: AbstractDataset):
    super().__init__(config)

    self.item2feat = None
    self.ignored_label = -100
    self.history_tokenization_scope = config.get(
        'history_tokenization_scope', 'sequence'
    )
    self.target_tokenization = config.get(
        'target_tokenization', 'actionpiece'
    )
    if self.history_tokenization_scope not in {'sequence', 'item'}:
      raise ValueError(
          'history_tokenization_scope must be "sequence" or "item".'
      )
    if self.target_tokenization not in {'actionpiece', 'atomic'}:
      raise ValueError(
          'target_tokenization must be "actionpiece" or "atomic".'
      )
    self.actionpiece = self._init_tokenizer(dataset)
    self.bos_token = self.actionpiece.vocab_size
    self.eos_token = self.actionpiece.vocab_size + 1
    self.n_inference_ensemble = config['n_inference_ensemble']
    self.train_shuffle = config['train_shuffle']
    self.encoded_labels = {}
    self.atomic_target_tokens = self._get_atomic_target_tokens()
    self.collate_fn = {
        'train': self.collate_fn_train,
        'val': self.collate_fn_val,
        'test': self.collate_fn_test,
    }

  def _get_item_sentence(self, dataset: AbstractDataset, item: str) -> str:
    item_meta = dataset.item2meta[item]
    if isinstance(item_meta, dict):
      return item_meta['sentence']
    return item_meta

  def _sent_artifact_stem(self) -> str:
    """Return a cache-safe sentence encoder identity."""
    model_name = os.path.basename(self.config['sent_emb_model'])
    qwen_backends = {'qwen_api', 'qwen_local'}
    if self.config['sent_emb_backend'] not in qwen_backends:
      return model_name
    safe_model_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model_name)
    instruction_hash = hashlib.sha256(
        self.config['qwen_api_instruction'].encode('utf-8')
    ).hexdigest()[:12]
    stem = (
        f'{self.config["sent_emb_backend"]}.{safe_model_name}.'
        f'd{self.config["sent_emb_dim"]}.i{instruction_hash}'
    )
    if self.config['sent_emb_backend'] == 'qwen_local':
      modality = (
          'text_image_joint'
          if self.config['metadata'] == 'qwen_multimodal'
          else 'text'
      )
      return build_qwen_local_artifact_stem(
          backend=self.config['sent_emb_backend'],
          model_id=self.config['sent_emb_model'],
          dimension=self.config['sent_emb_dim'],
          instruction=self.config['qwen_api_instruction'],
          model_revision=self.config['qwen_local_model_revision'],
          code_revision=self.config['qwen_local_code_revision'],
          max_length=self.config['qwen_local_max_length'],
          torch_dtype=self.config['qwen_local_torch_dtype'],
          attn_implementation=self.config['qwen_local_attn_implementation'],
          modality=modality,
          image_manifest_sha256=(
              self.config['qwen_local_image_manifest_sha256']
              if modality == 'text_image_joint'
              else None
          ),
      )
    return stem

  def _semantic_artifact_stem(self) -> str:
    if self.config['sent_emb_backend'] not in {'qwen_api', 'qwen_local'}:
      return self._sent_artifact_stem()
    stem = (
        f'{self._sent_artifact_stem()}.'
        f'opq{self.config["pq_n_codebooks"]}x'
        f'{self.config["pq_codebook_size"]}.'
        f'seed{self.config["rand_seed"]}'
    )
    variant = self.config.get('semantic_id_variant')
    if variant:
      safe_variant = re.sub(r'[^A-Za-z0-9._-]+', '-', str(variant))
      stem = f'{stem}.{safe_variant}'
    return stem

  def _qwen_image_artifact_stem(self) -> str:
    return build_qwen_local_artifact_stem(
        backend=self.config['sent_emb_backend'],
        model_id=self.config['sent_emb_model'],
        dimension=self.config['image_emb_dim'],
        instruction=self.config['qwen_local_image_instruction'],
        model_revision=self.config['qwen_local_model_revision'],
        code_revision=self.config['qwen_local_code_revision'],
        max_length=self.config['qwen_local_max_length'],
        torch_dtype=self.config['qwen_local_torch_dtype'],
        attn_implementation=self.config['qwen_local_attn_implementation'],
        modality='image',
        image_manifest_sha256=self.config[
            'qwen_local_image_manifest_sha256'
        ],
    )

  def _fused_semantic_artifact_stem(self) -> str:
    image_weight = self.config['fused_image_weight']
    text_identity = hashlib.sha256(
        self._sent_artifact_stem().encode('utf-8')
    ).hexdigest()[:12]
    image_identity = hashlib.sha256(
        self._qwen_image_artifact_stem().encode('utf-8')
    ).hexdigest()[:12]
    return (
        f'qwen_fused.{os.path.basename(self.config["sent_emb_model"])}.'
        f't{self.config["sent_emb_dim"]}.th{text_identity}.'
        f'i{self.config["image_emb_dim"]}.ih{image_identity}.'
        f'w{image_weight}.'
        f'opq{self.config["pq_n_codebooks"]}x'
        f'{self.config["pq_codebook_size"]}.'
        f'seed{self.config["rand_seed"]}'
    )

  def _image_semantic_artifact_stem(self) -> str:
    """Return a cache-safe identity for independently quantized images."""
    stem = (
        f'{self._qwen_image_artifact_stem()}.'
        f'opq{self.config["image_pq_n_codebooks"]}x'
        f'{self.config["image_pq_codebook_size"]}.'
        f'seed{self.config["rand_seed"]}'
    )
    variant = self.config.get('semantic_id_variant')
    if variant:
      safe_variant = re.sub(r'[^A-Za-z0-9._-]+', '-', str(variant))
      stem = f'{stem}.{safe_variant}'
    return stem

  def _separate_feature_artifact_stem(self) -> str:
    """Return the E4 identity without creating overlong cache filenames."""
    text_identity = hashlib.sha256(
        self._semantic_artifact_stem().encode('utf-8')
    ).hexdigest()[:12]
    image_identity = hashlib.sha256(
        self._image_semantic_artifact_stem().encode('utf-8')
    ).hexdigest()[:12]
    return (
        f'qwen_separate.{os.path.basename(self.config["sent_emb_model"])}.'
        f't{self.config["sent_emb_dim"]}.th{text_identity}.'
        f'i{self.config["image_emb_dim"]}.ih{image_identity}.'
        f'topq{self.config["pq_n_codebooks"]}x'
        f'{self.config["pq_codebook_size"]}.'
        f'iopq{self.config["image_pq_n_codebooks"]}x'
        f'{self.config["image_pq_codebook_size"]}.'
        f'seed{self.config["rand_seed"]}.'
        f'h{self.config["n_hash_buckets"]}'
    )

  def _feature_artifact_stem(self) -> str:
    if self.config['metadata'] == 'qwen_image':
      return (
          f'qwen_image.{self._image_semantic_artifact_stem()}.'
          f'h{self.config["n_hash_buckets"]}'
      )
    if self.config['metadata'] == 'qwen_separate':
      return self._separate_feature_artifact_stem()
    if self.config['metadata'] == 'qwen_fused':
      return (
          f'{self._fused_semantic_artifact_stem()}.'
          f'h{self.config["n_hash_buckets"]}'
      )
    if self.config['metadata'] not in {'qwen_text', 'qwen_multimodal'}:
      return self.config['metadata']
    return (
        f'{self.config["metadata"]}.{self._semantic_artifact_stem()}.'
        f'h{self.config["n_hash_buckets"]}'
    )

  def _tokenizer_scope_suffix(self) -> str:
    """Return a cache suffix for tokenizer-training constraints."""
    if self.history_tokenization_scope == 'sequence':
      return ''
    hash_suffix = (
        '' if self.config.get('actionpiece_merge_hash', True) else '.no_hash'
    )
    return f'history_{self.history_tokenization_scope}{hash_suffix}.'

  def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
    """Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

    Args:
        dataset (AbstractDataset): The dataset containing the sentences to
          encode.
        output_path (str): The path to save the encoded sentence embeddings.

    Returns:
        numpy.ndarray: The encoded sentence embeddings.
    """
    assert self.config['metadata'] in [
        'sentence',
        'qwen_text',
        'qwen_multimodal',
        'qwen_separate',
        'qwen_fused',
        'all',
        'sentence_image',
        'sentence_image_fused',
    ]

    meta_sentences = []  # 1-base, meta_sentences[0] -> item_id = 1
    item_ids = []
    for i in range(1, dataset.n_items):
      item = dataset.id_mapping['id2item'][i]
      item_ids.append(item)
      meta_sentences.append(
          self._get_item_sentence(dataset, item)
      )

    if self.config['sent_emb_backend'] == 'qwen_api':
      if self.config['metadata'] != 'qwen_text':
        raise ValueError('Qwen API E1 requires metadata=qwen_text.')
      if self.config['sent_emb_pca'] > 0:
        raise ValueError('Qwen API E1 does not support sent_emb_pca.')
      encoder = QwenApiTextEncoder(
          model=self.config['sent_emb_model'],
          instruction=self.config['qwen_api_instruction'],
          dimension=self.config['sent_emb_dim'],
          batch_size=self.config['qwen_api_batch_size'],
          timeout=self.config['qwen_api_timeout'],
          max_retries=self.config['qwen_api_max_retries'],
          env_file=self.config['qwen_api_env_file'],
          endpoint=self.config['qwen_api_endpoint'],
      )
      sent_embs = encoder.encode(
          meta_sentences,
          item_ids,
          output_path,
          progress_callback=lambda completed, total: self.logger.info(
              '[TOKENIZER] Qwen text embeddings: %d/%d', completed, total
          ),
      )
    elif self.config['sent_emb_backend'] == 'qwen_local':
      if self.config['metadata'] not in {
          'qwen_text',
          'qwen_multimodal',
          'qwen_separate',
          'qwen_fused',
      }:
        raise ValueError(
            'Local Qwen requires metadata=qwen_text, qwen_multimodal, '
            'qwen_separate, or qwen_fused.'
        )
      if self.config['sent_emb_pca'] > 0:
        raise ValueError('Local Qwen does not support sent_emb_pca.')
      encoder_class = (
          QwenLocalMultimodalEncoder
          if self.config['metadata'] == 'qwen_multimodal'
          else QwenLocalTextEncoder
      )
      encoder = encoder_class(
          model_path=self.config['qwen_local_model_path'],
          model_id=self.config['sent_emb_model'],
          model_revision=self.config['qwen_local_model_revision'],
          repo_path=self.config['qwen_local_repo_path'],
          code_revision=self.config['qwen_local_code_revision'],
          instruction=self.config['qwen_api_instruction'],
          dimension=self.config['sent_emb_dim'],
          batch_size=self.config['qwen_local_batch_size'],
          max_length=self.config['qwen_local_max_length'],
          torch_dtype=self.config['qwen_local_torch_dtype'],
          attn_implementation=self.config['qwen_local_attn_implementation'],
          require_cuda=self.config['qwen_local_require_cuda'],
      )
      if self.config['metadata'] == 'qwen_multimodal':
        manifest_path = os.path.join(
            dataset.cache_dir, 'processed', 'image_download_manifest.jsonl'
        )
        manifest_path = Path(manifest_path)
        expected_manifest_sha256 = self.config.get(
            'qwen_local_image_manifest_sha256'
        )
        actual_manifest_sha256 = sha256_file(manifest_path)
        if actual_manifest_sha256 != expected_manifest_sha256:
          raise RuntimeError(
              'Image manifest SHA-256 mismatch: '
              f'got {actual_manifest_sha256}, '
              f'expected {expected_manifest_sha256}.'
          )
        image_status = load_image_manifest(manifest_path)
        if set(image_status) != set(item_ids):
          raise RuntimeError(
              'Image manifest keys do not exactly match the dataset item IDs.'
          )
        image_paths = [
            available_image_path(image_status[item]) for item in item_ids
        ]
        sent_embs = encoder.encode(
            meta_sentences,
            image_paths,
            item_ids,
            output_path,
            image_manifest_sha256=self.config[
                'qwen_local_image_manifest_sha256'
            ],
            progress_callback=lambda completed, total: self.logger.info(
                '[TOKENIZER] Local Qwen joint embeddings: %d/%d',
                completed,
                total,
            ),
        )
      else:
        sent_embs = encoder.encode(
            meta_sentences,
            item_ids,
            output_path,
            progress_callback=lambda completed, total: self.logger.info(
                '[TOKENIZER] Local Qwen text embeddings: %d/%d', completed, total
            ),
        )
    elif self.config['sent_emb_backend'] == 'sentence_transformers':
      sent_emb_model = SentenceTransformer(self.config['sent_emb_model']).to(
          self.config['device']
      )
      sent_embs = sent_emb_model.encode(
          meta_sentences,
          convert_to_numpy=True,
          batch_size=self.config['sent_emb_batch_size'],
          show_progress_bar=True,
          device=self.config['device'],
      )
    else:
      raise ValueError(
          f'Unknown sent_emb_backend: {self.config["sent_emb_backend"]}'
      )

    # PCA
    if self.config['sent_emb_pca'] > 0:
      self.logger.info('[TOKENIZER] Applying PCA to sentence embeddings...')

      pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
      sent_embs = pca.fit_transform(sent_embs)

    if self.config['sent_emb_backend'] not in {'qwen_api', 'qwen_local'}:
      sent_embs.tofile(output_path)
    return sent_embs

  def _get_sent_embs(self, dataset: AbstractDataset) -> np.ndarray:
    # Load or encode sentence embeddings
    sent_emb_path = os.path.join(
        dataset.cache_dir,
        'processed',
        f'{self._sent_artifact_stem()}.sent_emb',
    )
    sent_emb_dim = (
        self.config['sent_emb_dim']
        if self.config['sent_emb_pca'] <= 0
        else self.config['sent_emb_pca']
    )
    if os.path.exists(sent_emb_path):
      self.logger.info(
          f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...'
      )
      sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
          -1, sent_emb_dim
      )
    else:
      self.logger.info('[TOKENIZER] Encoding sentence embeddings...')
      sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
    expected_shape = (dataset.n_items - 1, sent_emb_dim)
    if sent_embs.shape != expected_shape:
      raise ValueError(
          f'[TOKENIZER] Sentence embedding shape {sent_embs.shape} does not '
          f'match expected shape {expected_shape}.'
      )
    if not np.isfinite(sent_embs).all():
      raise ValueError('[TOKENIZER] Sentence embeddings contain NaN or Inf.')
    self.logger.info(
        f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}'
    )
    return sent_embs

  def _get_image_url(self, dataset: AbstractDataset, item: str) -> str | None:
    item_meta = dataset.item2meta[item]
    if isinstance(item_meta, dict):
      return item_meta.get('image_url')
    return None

  def _log_image_failure(
      self, failure_log_path: str, item: str, image_url: str | None, reason: str
  ):
    with open(failure_log_path, 'a') as f:
      f.write(
          json.dumps(
              {
                  'item': item,
                  'image_url': image_url,
                  'reason': reason,
              }
          )
          + '\n'
      )

  def _download_image(
      self, item: str, image_url: str | None, failure_log_path: str
  ) -> Image.Image | None:
    if not image_url:
      self._log_image_failure(failure_log_path, item, image_url, 'missing_url')
      return None
    try:
      response = requests.get(
          image_url, timeout=self.config['image_download_timeout']
      )
      response.raise_for_status()
      return Image.open(io.BytesIO(response.content)).convert('RGB')
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._log_image_failure(failure_log_path, item, image_url, repr(exc))
      return None

  def _encode_image_emb(self, dataset: AbstractDataset, output_path: str):
    """Encode product images with CLIP and save item-aligned embeddings."""
    if self.config['metadata'] not in [
        'sentence_image',
        'sentence_image_fused',
    ]:
      raise ValueError(
          'Image embeddings require metadata=sentence_image or '
          'metadata=sentence_image_fused.'
      )

    image_emb_model = CLIPModel.from_pretrained(
        self.config['image_emb_model']
    ).to(self.config['device'])
    image_processor = CLIPProcessor.from_pretrained(
        self.config['image_emb_model']
    )
    image_emb_model.eval()

    failure_log_path = os.path.join(
        dataset.cache_dir, 'processed', 'image_download_failures.jsonl'
    )
    if os.path.exists(failure_log_path):
      os.remove(failure_log_path)

    image_emb_dim = self.config['image_emb_dim']
    image_embs = np.zeros((dataset.n_items - 1, image_emb_dim), dtype=np.float32)
    pending_images = []
    pending_indices = []

    def flush_batch():
      if not pending_images:
        return
      inputs = image_processor(
          images=pending_images, return_tensors='pt', padding=True
      )
      inputs = {k: v.to(self.config['device']) for k, v in inputs.items()}
      with torch.no_grad():
        batch_embs = image_emb_model.get_image_features(**inputs)
        batch_embs = torch.nn.functional.normalize(batch_embs, dim=-1)
      batch_embs = batch_embs.detach().cpu().numpy().astype(np.float32)
      if batch_embs.shape[-1] != image_emb_dim:
        raise ValueError(
            '[TOKENIZER] CLIP image embedding dimension mismatch: '
            f'got {batch_embs.shape[-1]}, expected {image_emb_dim}.'
        )
      for idx, emb in zip(pending_indices, batch_embs):
        image_embs[idx] = emb
      pending_images.clear()
      pending_indices.clear()

    self.logger.info('[TOKENIZER] Encoding image embeddings...')
    for i in tqdm.tqdm(range(1, dataset.n_items)):
      item = dataset.id_mapping['id2item'][i]
      image = self._download_image(
          item, self._get_image_url(dataset, item), failure_log_path
      )
      if image is None:
        continue
      pending_images.append(image)
      pending_indices.append(i - 1)
      if len(pending_images) >= self.config['image_emb_batch_size']:
        flush_batch()
    flush_batch()

    image_embs.tofile(output_path)
    self.logger.info(
        f'[TOKENIZER] Image download failures saved to {failure_log_path}'
    )
    return image_embs

  def _get_image_embs(self, dataset: AbstractDataset) -> np.ndarray:
    if self.config['metadata'] in {
        'qwen_image',
        'qwen_separate',
        'qwen_fused',
    }:
      image_emb_path = os.path.join(
          dataset.cache_dir,
          'processed',
          f'{self._qwen_image_artifact_stem()}.image_emb',
      )
      if not os.path.exists(image_emb_path):
        raise FileNotFoundError(
            'Qwen image-only embeddings are missing. Run '
            'scripts/generate_qwen_local_image_embeddings.py first: '
            f'{image_emb_path}'
        )
    else:
      image_emb_path = os.path.join(
          dataset.cache_dir,
          'processed',
          f'{os.path.basename(self.config["image_emb_model"])}.image_emb',
      )
    image_emb_dim = self.config['image_emb_dim']
    if os.path.exists(image_emb_path):
      self.logger.info(
          f'[TOKENIZER] Loading image embeddings from {image_emb_path}...'
      )
      image_embs = np.fromfile(image_emb_path, dtype=np.float32).reshape(
          -1, image_emb_dim
      )
    else:
      image_embs = self._encode_image_emb(dataset, image_emb_path)
    self.logger.info(
        f'[TOKENIZER] Image embeddings shape: {image_embs.shape}'
    )
    return image_embs

  def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
    items_for_training = set()
    for item_seq in dataset.split_data['train']['item_seq']:
      for item in item_seq:
        items_for_training.add(item)
    self.logger.info(
        f'[TOKENIZER] Items for training: {len(items_for_training)}'
        f' of {dataset.n_items - 1}'
    )
    mask = np.zeros(dataset.n_items - 1, dtype=bool)
    for item in items_for_training:
      mask[dataset.item2id[item] - 1] = True
    return mask

  def _emb_to_sem_id(
      self,
      dataset: AbstractDataset,
      embs: np.ndarray,
      n_codebooks: int,
      codebook_size: int,
  ) -> dict[Any, Any]:
    # Get the sentence embeddings for training
    training_item_mask = self._get_items_for_training(dataset)
    embs_for_training = embs[training_item_mask]

    # Train the index
    # Take the vector quantized codes as item features

    faiss.omp_set_num_threads(self.config['n_threads'])
    index = faiss.index_factory(
        embs.shape[-1],
        f'OPQ{n_codebooks},IVF1,PQ{n_codebooks}x{int(np.log2(codebook_size))}',
        faiss.METRIC_INNER_PRODUCT,
    )
    self.logger.info('[TOKENIZER] Training index...')
    index.train(embs_for_training)
    index.add(embs)

    ivf_index = faiss.downcast_index(index.index)
    invlists = faiss.extract_index_ivf(ivf_index).invlists
    ls = invlists.list_size(0)
    sem_ids = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
    sem_ids = sem_ids.reshape(-1, invlists.code_size)

    # Convert semantic IDs to a dictionary
    item2sem_ids = {}
    for i in range(sem_ids.shape[0]):
      item = dataset.id_mapping['id2item'][i + 1]
      item2sem_ids[item] = tuple(sem_ids[i].tolist())
    return item2sem_ids

  def _sent_emb_to_sem_id(
      self, dataset: AbstractDataset, sent_embs: np.ndarray
  ) -> dict[Any, Any]:
    return self._emb_to_sem_id(
        dataset,
        sent_embs,
        n_codebooks=self.config['pq_n_codebooks'],
        codebook_size=self.config['pq_codebook_size'],
    )

  def _get_sem_ids(self, dataset: AbstractDataset) -> dict[Any, Any]:
    """Get the semantic IDs from the dataset.

    If the semantic IDs are already cached, load them from the cache. Otherwise,
    generate the semantic IDs and save them to the cache.

    Args:
        dataset (AbstractDataset): The dataset containing the items to get
          semantic IDs for.

    Returns:
        item2sem_ids (dict): A dictionary mapping item IDs to their semantic
        IDs.
    """
    sem_ids_path = os.path.join(
        dataset.cache_dir,
        'processed',
        f'{self._semantic_artifact_stem()}.sem_ids',
    )
    if not os.path.exists(sem_ids_path):
      self.logger.info(
          '[TOKENIZER] Semantic IDs not found. Training index using Faiss...'
      )
      sent_embs = self._get_sent_embs(dataset)
      item2sem_ids = self._sent_emb_to_sem_id(dataset, sent_embs)
      # Save semantic IDs
      self.logger.info(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
      with open(sem_ids_path, 'w') as f:
        json.dump(item2sem_ids, f)
      return item2sem_ids
    else:
      self.logger.info(
          f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...'
      )
      with open(sem_ids_path, 'r') as f:
        item2sem_ids = json.load(f)
      return {
          k: v[: self.config['pq_n_codebooks']] for k, v in item2sem_ids.items()
      }

  def _image_emb_to_sem_id(
      self, dataset: AbstractDataset, image_embs: np.ndarray
  ) -> dict[Any, Any]:
    return self._emb_to_sem_id(
        dataset,
        image_embs,
        n_codebooks=self.config['image_pq_n_codebooks'],
        codebook_size=self.config['image_pq_codebook_size'],
    )

  def _get_image_sem_ids(self, dataset: AbstractDataset) -> dict[Any, Any]:
    if self.config['metadata'] in {'qwen_image', 'qwen_separate'}:
      image_sem_ids_filename = f'{self._image_semantic_artifact_stem()}.sem_ids'
    else:
      image_sem_ids_filename = (
          f'{os.path.basename(self.config["image_emb_model"])}.image_sem_ids'
      )
    image_sem_ids_path = os.path.join(
        dataset.cache_dir, 'processed', image_sem_ids_filename
    )
    if not os.path.exists(image_sem_ids_path):
      self.logger.info(
          '[TOKENIZER] Image semantic IDs not found. Training image index...'
      )
      image_embs = self._get_image_embs(dataset)
      item2sem_ids = self._image_emb_to_sem_id(dataset, image_embs)
      self.logger.info(
          f'[TOKENIZER] Saving image semantic IDs to {image_sem_ids_path}...'
      )
      with open(image_sem_ids_path, 'w') as f:
        json.dump(item2sem_ids, f)
      return item2sem_ids
    self.logger.info(
        f'[TOKENIZER] Loading image semantic IDs from {image_sem_ids_path}...'
    )
    with open(image_sem_ids_path, 'r') as f:
      item2sem_ids = json.load(f)
    return {
        k: v[: self.config['image_pq_n_codebooks']]
        for k, v in item2sem_ids.items()
    }

  def _normalize_embs(self, embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return embs / norms

  def _get_fused_sem_ids(self, dataset: AbstractDataset) -> dict[Any, Any]:
    image_weight = self.config['fused_image_weight']
    if self.config['metadata'] == 'qwen_fused':
      fused_filename = f'{self._fused_semantic_artifact_stem()}.sem_ids'
    else:
      sent_model_name = os.path.basename(self.config['sent_emb_model'])
      image_model_name = os.path.basename(self.config['image_emb_model'])
      fused_filename = (
          f'{sent_model_name}.{image_model_name}.'
          f'w{image_weight}.fused_sem_ids'
      )
    fused_sem_ids_path = os.path.join(
        dataset.cache_dir, 'processed', fused_filename
    )
    if not os.path.exists(fused_sem_ids_path):
      self.logger.info(
          '[TOKENIZER] Fused semantic IDs not found. '
          'Training fused text-image index...'
      )
      sent_embs = self._normalize_embs(self._get_sent_embs(dataset))
      image_embs = self._normalize_embs(self._get_image_embs(dataset))
      fused_embs = np.concatenate(
          [sent_embs, image_weight * image_embs], axis=-1
      ).astype(np.float32)
      if self.config.get('fused_final_normalize', False):
        fused_embs = self._normalize_embs(fused_embs).astype(np.float32)
      self.logger.info(
          f'[TOKENIZER] Fused embeddings shape: {fused_embs.shape}'
      )
      item2sem_ids = self._emb_to_sem_id(
          dataset,
          fused_embs,
          n_codebooks=self.config['pq_n_codebooks'],
          codebook_size=self.config['pq_codebook_size'],
      )
      self.logger.info(
          f'[TOKENIZER] Saving fused semantic IDs to {fused_sem_ids_path}...'
      )
      with open(fused_sem_ids_path, 'w') as f:
        json.dump(item2sem_ids, f)
      return item2sem_ids
    self.logger.info(
        f'[TOKENIZER] Loading fused semantic IDs from {fused_sem_ids_path}...'
    )
    with open(fused_sem_ids_path, 'r') as f:
      item2sem_ids = json.load(f)
    return {
        k: v[: self.config['pq_n_codebooks']] for k, v in item2sem_ids.items()
    }

  def _get_attr_ids(self, dataset: AbstractDataset):
    """Get the attribute IDs from the dataset."""
    if 'attr_id' in dataset.item2meta:
      return dataset.item2meta['attr_id']
    else:
      return None

  def _combine_features(
      self, item2sem_ids: dict[Any, Any], item2attr_ids: dict[Any, Any]
  ) -> dict[Any, Any]:
    """Combine the semantic IDs and attribute IDs to get the item features."""
    if item2attr_ids is None:
      return item2sem_ids
    item2feat = {}
    for item in item2sem_ids:
      feat = item2sem_ids[item] + item2attr_ids[item]
      item2feat[item] = feat
    return item2feat

  def _get_hashed_feat(
      self, dataset: AbstractDataset, item2feat: dict[Any, Any]
  ) -> dict[Any, Any]:
    """Add one digit of the original features as the hash buckets.

    The purpose is to avoid conflicts.

    Args:
        dataset (AbstractDataset): The dataset containing the items (not used).
        item2feat (dict): A dictionary mapping item IDs to their features.

    Returns:
        dict: A dictionary mapping item IDs to their hashed features.
    """
    feat2cnt = collections.defaultdict(int)
    feat2hash_ids = {}
    item2hashed_feat = {}

    for item in item2feat:
      feat = tuple(item2feat[item])
      if feat not in feat2hash_ids:
        feat2hash_ids[feat] = np.random.permutation(
            self.config['n_hash_buckets']
        )
      idx = feat2cnt[feat]

      if idx >= self.config['n_hash_buckets']:
        raise ValueError(
            '[TOKENIZER] Too many conflicts of semantic IDs found.'
            ' Please increase the number of hash buckets.'
        )

      item2hashed_feat[item] = (
          *item2feat[item],
          feat2hash_ids[feat][idx].item(),
      )
      feat2cnt[feat] += 1

    return item2hashed_feat

  def _get_item2feat(self, dataset: AbstractDataset) -> dict[Any, Any]:
    """Get the item features.

    If the features are already cached, load them from the cache. Otherwise,
    generate the features and save them to the cache.

    Args:
        dataset (AbstractDataset): The dataset containing the items to get
          features for.

    Returns:
        item2feat (dict): A dictionary mapping item IDs to their features.
    """
    feat_path = os.path.join(
        dataset.cache_dir,
        f'processed/item.{self._feature_artifact_stem()}.feat',
    )
    if os.path.exists(feat_path):
      self.logger.info(f'[TOKENIZER] Loading item features from {feat_path}...')
      with open(feat_path, 'r') as f:
        item2feat = json.load(f)
      return item2feat
    self.logger.info('[TOKENIZER] Generating item features...')
    if self.config['metadata'] == 'qwen_image':
      item2sem_ids = self._get_image_sem_ids(dataset)
    elif self.config['metadata'] in {'sentence_image_fused', 'qwen_fused'}:
      item2sem_ids = self._get_fused_sem_ids(dataset)
    else:
      item2sem_ids = self._get_sem_ids(dataset)
    if self.config['metadata'] in {'sentence_image', 'qwen_separate'}:
      item2image_sem_ids = self._get_image_sem_ids(dataset)
      item2sem_ids = {
          item: tuple(item2sem_ids[item]) + tuple(item2image_sem_ids[item])
          for item in item2sem_ids
      }
    item2attr_ids = self._get_attr_ids(dataset)
    item2feat = self._combine_features(item2sem_ids, item2attr_ids)
    item2hashed_feat = self._get_hashed_feat(dataset, item2feat)
    self.logger.info(f'[TOKENIZER] Saving item features to {feat_path}...')
    with open(feat_path, 'w') as f:
      json.dump(item2hashed_feat, f)
    return item2hashed_feat

  def _check_conflicts(self, item2feat: dict[Any, Any]):
    """Check if there are any conflicts in the item features.

    If there are conflicts, raise a ValueError.

    Args:
        item2feat (dict): A dictionary mapping item IDs to their features.
    """
    feat2cnt = collections.Counter()
    for feat in item2feat.values():
      feat = tuple(feat)
      feat2cnt[feat] += 1
    for feat in feat2cnt:
      if feat2cnt[feat] > 1:
        raise ValueError(f'[TOKENIZER] Conflicts found in features: {feat}')

  def _get_initial_token_sources(
      self, dataset: AbstractDataset, actionpiece: ActionPieceCore
  ):
    feat2sources = collections.defaultdict(set)
    item_sources = getattr(dataset, 'item_sources', None)
    for item, feats in self.item2feat.items():
      if item_sources:
        sources = item_sources.get(item, [dataset.category])
      else:
        sources = [dataset.category]
      for i, feat in enumerate(feats):
        feat2sources[(i, feat)].update(sources)

    return {
        actionpiece.rank[feat]: sorted(sources)
        for feat, sources in feat2sources.items()
        if feat in actionpiece.rank
    }

  def _tokenize_once(self, item_seq):
    state_seq = []
    for item in item_seq:
      feats = self.item2feat[item]
      tokenized_feats = []
      for i, feat in enumerate(feats):
        tokenized_feats.append(self.actionpiece.rank[(i, feat)])
      state_seq.append(tokenized_feats)
    return np.array(state_seq)

  def _encode_history(self, state_seq, shuffle):
    """Encode history jointly or independently within each item."""
    if self.history_tokenization_scope == 'sequence':
      return self.actionpiece.encode(state_seq, shuffle=shuffle)
    encoded = []
    for state in state_seq:
      encoded.extend(
          self.actionpiece.encode(np.asarray([state]), shuffle=shuffle)
      )
    return encoded

  def _get_atomic_target_tokens(self):
    """Group primitive vocabulary IDs by their fixed feature slot."""
    by_slot = [[] for _ in range(self.actionpiece.n_categories)]
    for token, feature in enumerate(
        self.actionpiece.vocab[: self.actionpiece.n_init_feats]
    ):
      slot = int(feature[0])
      if slot >= 0:
        by_slot[slot].append(token)
    return tuple(tuple(tokens) for tokens in by_slot)

  def target_allowed_tokens(self, step):
    """Return allowed decoder tokens at one target step, or None."""
    if self.target_tokenization != 'atomic':
      return None
    if step < self.actionpiece.n_categories:
      return self.atomic_target_tokens[step]
    if step == self.actionpiece.n_categories:
      return (self.eos_token,)
    return ()

  @property
  def generation_max_length(self):
    """Maximum decoder length including its start token."""
    if self.target_tokenization == 'atomic':
      return self.actionpiece.n_categories + 2
    return self.actionpiece.n_categories + 1

  def tokenize_function(
      self, example: dict[Any, Any], split: str
  ) -> dict[Any, Any]:
    max_item_seq_len = self.config['max_item_seq_len']
    item_seq = example['item_seq'][0]
    if split == 'train':
      # Tokenize each subsequence with maximum `max_item_seq_len` items
      n_return_examples = len(item_seq) - 1
      all_state_seq = []
      for i in range(n_return_examples):
        cur_item_seq = self._tokenize_once(
            item_seq[max(i + 1 - max_item_seq_len, 0) : i + 2]
        )
        all_state_seq.append(cur_item_seq)
    else:
      all_state_seq = [self._tokenize_once(item_seq[-(max_item_seq_len + 1) :])]
    return {
        'state_seq': all_state_seq,
    }

  def tokenize(self, datasets: dict[Any, Any]) -> dict[Any, Any]:
    tokenized_datasets = {}
    for split in datasets:
      tokenized_datasets[split] = datasets[split].map(
          lambda t: self.tokenize_function(t, split),  # pylint: disable=cell-var-from-loop
          batched=True,
          batch_size=1,
          remove_columns=datasets[split].column_names,
          num_proc=self.config['num_proc'],
          desc=f'Tokenizing {split} set: ',
      )
    for split in datasets:
      tokenized_datasets[split].set_format(type='numpy')
    return tokenized_datasets

  @property
  def vocab_size(self):
    return self.eos_token + 1

  @property
  def max_token_seq_len(self):
    # +2 for EOS and BOS
    return self.actionpiece.n_categories * self.config['max_item_seq_len'] + 2

  def _init_tokenizer(self, dataset: AbstractDataset):
    self.item2feat = self._get_item2feat(dataset)
    self._check_conflicts(self.item2feat)

    tokenizer_filename = 'actionpiece.json'
    if self.config['metadata'] == 'sentence_image':
      tokenizer_filename = 'actionpiece.sentence_image.json'
    elif self.config['metadata'] == 'sentence_image_fused':
      tokenizer_filename = 'actionpiece.sentence_image_fused.json'
    elif self.config['metadata'] in {
        'qwen_text',
        'qwen_multimodal',
        'qwen_image',
        'qwen_separate',
        'qwen_fused',
    }:
      tokenizer_filename = (
          f'actionpiece.{self._feature_artifact_stem()}.'
          f'{self._tokenizer_scope_suffix()}'
          f'v{self.config["actionpiece_vocab_size"]}.json'
      )
    tokenizer_path = os.path.join(dataset.cache_dir, 'processed', tokenizer_filename)
    if os.path.exists(tokenizer_path):
      # If trained tokenizer exists, load it
      self.logger.info(
          f'[TOKENIZER] Loading ActionPiece from {tokenizer_path}...'
      )
      actionpiece = ActionPieceCore.from_pretrained(
          tokenizer_path, vocab_size=self.config['actionpiece_vocab_size']
      )
    else:
      # Initialize ActionPiece from initial features
      self.logger.info('[TOKENIZER] Constructing ActionPiece vocabulary...')
      merge_log_path = None
      if self.config['actionpiece_merge_log']:
        merge_log_path = self.config['actionpiece_merge_log_path']
        if merge_log_path is None:
          merge_log_filename = 'actionpiece.merge_log.jsonl'
          if self.config['metadata'] in {
              'qwen_text',
              'qwen_multimodal',
              'qwen_image',
              'qwen_separate',
              'qwen_fused',
          }:
            merge_log_filename = (
                f'actionpiece.{self._feature_artifact_stem()}.'
                f'{self._tokenizer_scope_suffix()}'
                f'v{self.config["actionpiece_vocab_size"]}.merge_log.jsonl'
            )
          merge_log_path = os.path.join(
              dataset.cache_dir, 'processed', merge_log_filename
          )
        merge_log_dir = os.path.dirname(merge_log_path)
        if merge_log_dir:
          os.makedirs(merge_log_dir, exist_ok=True)
        self.logger.info(
            f'[TOKENIZER] Saving ActionPiece merge log to {merge_log_path}...'
        )
      actionpiece = ActionPieceCore(
          state2feat=self.item2feat,
      )
      actionpiece.token_sources = self._get_initial_token_sources(
          dataset, actionpiece
      )
      # Construct ActionPiece vocabulary
      modality_slots = None
      if (
          merge_log_path is not None
          and self.config['metadata'] in {'sentence_image', 'qwen_separate'}
      ):
        text_slot_count = int(self.config['pq_n_codebooks'])
        image_slot_count = int(self.config['image_pq_n_codebooks'])
        modality_slots = {
            'text_slots': list(range(text_slot_count)),
            'image_slots': list(
                range(text_slot_count, text_slot_count + image_slot_count)
            ),
            'hash_slot': text_slot_count + image_slot_count,
        }
      actionpiece.train(
          state_corpus=dataset.split_data['train']['item_seq'],
          target_vocab_size=self.config['actionpiece_vocab_size'],
          merge_log_path=merge_log_path,
          merge_log_interval=self.config['actionpiece_merge_log_interval'],
          modality_slots=modality_slots,
          allow_cross_action_merges=(
              self.history_tokenization_scope == 'sequence'
          ),
          allowed_merge_slots=(
              list(range(actionpiece.n_categories - 1))
              if not self.config.get('actionpiece_merge_hash', True)
              else None
          ),
      )
      actionpiece.save(tokenizer_path)
    return actionpiece

  def encode_labels(self, labels):
    """Cache the encoded labels for faster inference.

    Args:
        labels (np.ndarray): The labels to be encoded.

    Returns:
        encoded_labels (list[int]): The encoded labels.
    """
    if self.target_tokenization == 'atomic':
      return np.asarray(labels).reshape(-1).tolist()
    key = labels.tobytes()
    if key in self.encoded_labels:
      return self.encoded_labels[key]
    encoded_labels = self.actionpiece.encode(labels, shuffle='none')
    self.encoded_labels[key] = encoded_labels
    return encoded_labels

  def collate_fn_train(self, batch):
    """Tokenizing a batch of examples on-the-fly while training.

    Args:
        batch (list): A list of examples. batch['state_seq'] is a state
          sequence. batch['state_seq'][i] is a list of features.

    Returns:
        dict: A dictionary of tensors.
    """
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
      seq = data['state_seq'][:-1]
      lb = data['state_seq'][-1:]
      input_ids.append(
          [self.bos_token]
          + self._encode_history(seq, shuffle=self.train_shuffle)
          + [self.eos_token]
      )
      labels.append(self.encode_labels(lb) + [self.eos_token])
    seq_lens = [len(ids) for ids in input_ids]
    max_seq_len = max(seq_lens)
    for i in range(len(batch)):
      input_ids[i] = input_ids[i] + [self.padding_token] * (
          max_seq_len - seq_lens[i]
      )
      attention_mask.append(
          [1] * seq_lens[i] + [0] * (max_seq_len - seq_lens[i])
      )
      labels[i] = labels[i] + [self.ignored_label] * (
          self.actionpiece.n_categories + 1 - len(labels[i])
      )
    return {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'labels': torch.LongTensor(labels),
    }

  def collate_fn_val(self, batch):
    """Tokenizing a batch of examples on-the-fly while evaluating.

    Args:
        batch (list): A list of examples. batch['state_seq'] is a state
          sequence. batch['state_seq'][i] is a list of features.

    Returns:
        dict: A dictionary of tensors.
    """
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
      seq = data['state_seq'][:-1]
      lb = data['state_seq'][-1]
      # The labels should always be encoded by encode_plus
      input_ids.append(
          [self.bos_token]
          + self._encode_history(
              seq,
              shuffle='none' if self.n_inference_ensemble == -1 else 'feature',
          )
          + [self.eos_token]
      )
      labels.append(lb)
    seq_lens = [len(ids) for ids in input_ids]
    max_seq_len = max(seq_lens)
    for i, sequence_length in enumerate(seq_lens):
      input_ids[i] = input_ids[i] + [self.padding_token] * (
          max_seq_len - sequence_length
      )
      attention_mask.append(
          [1] * sequence_length + [0] * (max_seq_len - sequence_length)
      )
    return {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'labels': torch.LongTensor(np.array(labels)),
    }

  def collate_fn_test(self, batch):
    """Tokenizing a batch of examples on-the-fly while evaluating.

    Args:
        batch (list): A list of examples. batch['state_seq'] is a state
          sequence. batch['state_seq'][i] is a list of features.

    Returns:
        dict: A dictionary of tensors.
    """
    input_ids = []
    attention_mask = []
    labels = []
    for data in batch:
      seq = data['state_seq'][:-1]
      lb = data['state_seq'][-1]
      # The labels should always be encoded by encode_plus
      if self.n_inference_ensemble == -1:
        input_ids.append(
            [self.bos_token]
            + self._encode_history(seq, shuffle='none')
            + [self.eos_token]
        )
        labels.append(lb)
      for _ in range(self.n_inference_ensemble):
        input_ids.append(
            [self.bos_token]
            + self._encode_history(seq, shuffle='feature')
            + [self.eos_token]
        )
      labels.append(lb)
    seq_lens = [len(ids) for ids in input_ids]
    max_seq_len = max(seq_lens)
    for i, sequence_length in enumerate(seq_lens):
      input_ids[i] = input_ids[i] + [self.padding_token] * (
          max_seq_len - sequence_length
      )
      attention_mask.append(
          [1] * sequence_length + [0] * (max_seq_len - sequence_length)
      )
    return {
        'input_ids': torch.LongTensor(input_ids),
        'attention_mask': torch.LongTensor(attention_mask),
        'labels': torch.LongTensor(np.array(labels)),
    }
