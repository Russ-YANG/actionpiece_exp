"""Training-only semantic alignment for the raw OPQ4 autoregressive baseline."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# Cap on occurrences used for in-batch negatives; keeps the similarity matrix small.
DIAGNOSTIC_LIMIT = 1024


def pair_diagnostics(z, target, rows, limit=DIAGNOSTIC_LIMIT):
    """In-batch negatives, to separate item-specific alignment from collapse.

    Negative cosine has no negatives, so mapping every item to one shared
    direction also drives mean cosine to 1. Report positives against other
    items' text in the same batch. Uses a deterministic strided subsample (no
    RNG). Occurrences of the same item row are not negatives; items sharing four
    codes but different rows remain negatives, since their texts differ.
    Returns (positive, negative, gap, top1); negative/gap are NaN without any
    negative pair.
    """
    with torch.no_grad():
        step = max(1, -(-rows.numel() // limit))
        z = F.normalize(z[::step].detach().float(), dim=-1)
        target = F.normalize(target[::step].detach().float(), dim=-1)
        rows = rows[::step]
        similarity = z @ target.T
        positive = similarity.diagonal().mean()
        mask = rows[:, None] != rows[None, :]
        count = mask.sum()
        negative = (similarity * mask).sum() / count.clamp(min=1)
        negative = torch.where(count > 0, negative, torch.full_like(negative, float('nan')))
        top1 = (rows[similarity.argmax(dim=1)] == rows).float().mean()
    return positive, negative, positive - negative, top1


def _json(path):
    return json.loads(Path(path).read_text())


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_alignment_inputs(config, mapping):
    """Bind the frozen vectors to the OPQ bundle, source identity and item order.

    Does not generate embeddings, fit OPQ, or infer provenance from filenames.
    Hashes here bind final/cross-system artifacts, not intermediate pipeline steps.
    """
    weight = config['alignment_weight']
    if not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 0:
        raise ValueError('alignment_weight must be finite and nonnegative')
    bundle = _json(Path(config['sid_path']).with_name('manifest.json'))
    meta_path = Path(config['alignment_embedding_manifest_path'])
    meta = _json(meta_path)
    if bundle.get('synthetic_test_fixture') or meta.get('test_fixture'):
        raise ValueError('Synthetic fixtures are not experiment embeddings')
    if meta.get('backend') != 'qwen_local' or meta.get('model_id') != 'Qwen/Qwen3-VL-Embedding-8B':
        raise ValueError('Alignment requires local Qwen3-VL-Embedding-8B; no API fallback')
    for key in ('model_revision', 'code_revision', 'instruction_sha256', 'normalization'):
        if not meta.get(key):
            raise ValueError(f'Embedding provenance missing {key}')
    if bundle.get('embedding_source') != meta:
        raise ValueError('Alignment source differs from OPQ embedding_source')
    signature = bundle['signature']
    if (signature['n_codebooks'], signature['codebook_size']) != (4, 256):
        raise ValueError('Alignment entry requires OPQ4x256')
    if signature['dimension'] != meta['dimension'] or signature['preprocessing'] != meta['normalization']:
        raise ValueError('OPQ signature and embedding metadata differ')
    items = mapping['id2item'][1:]
    expected_order = hashlib.sha256('\n'.join(items).encode()).hexdigest()
    if meta.get('item_count') != len(items) or meta.get('item_order_sha256') != expected_order:
        raise ValueError('Alignment embedding item count/order mismatch')
    paths = {
        'embedding_path': Path(config['alignment_embedding_path']),
        'embedding_manifest_path': meta_path,
        'id_mapping_path': Path(config['processed_dir']) / 'id_mapping.json',
        'item_seqs_path': Path(config['processed_dir']) / 'all_item_seqs.json',
    }
    hashes = {}
    for key, path in paths.items():
        hashes[key] = _sha256(path)
        if hashes[key] != bundle['inputs'][key]['sha256']:
            raise ValueError(f'Alignment inputs differ from OPQ bundle: {key}')
    if meta.get('embedding_sha256') and hashes['embedding_path'] != meta['embedding_sha256']:
        raise ValueError('Embedding content differs from source manifest')
    count, dimension = len(items), meta['dimension']
    if paths['embedding_path'].stat().st_size != count * dimension * 4:
        raise ValueError('Alignment embedding byte count mismatch')
    text = np.fromfile(paths['embedding_path'], dtype='<f4').reshape(count, dimension)
    if not np.isfinite(text).all() or np.any(np.linalg.norm(text, axis=1) == 0):
        raise ValueError('Alignment embeddings contain nonfinite or zero rows')
    return text, {
        'source': meta, 'verified_input_sha256': hashes,
        'item_count': count, 'dimension': dimension,
        'scope': 'training history occurrences; four semantic slots; no disambiguation',
        'loss': 'negative_cosine(concat_projection(sid_embeddings), text_projection(frozen_text))',
    }


def add_alignment(model_base, tokenizer_base, config, mapping, features, text):
    """Return subclasses retaining the baseline's tokenization and generation."""
    items = mapping['id2item'][1:]

    class AlignmentTokenizer(tokenizer_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Full five-feature keys disambiguate items sharing their four codes.
            self.alignment_row_by_state = {
                tuple(self._tokenize_once([item])[0]): row
                for row, item in enumerate(items)
            }

        def collate_fn_train(self, batch):
            result = super().collate_fn_train(batch)
            result['alignment_rows'] = torch.tensor([
                self.alignment_row_by_state[tuple(state)]
                for example in batch for state in example['state_seq'][:-1]
            ], dtype=torch.long)
            return result

    class AlignmentModel(model_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.alignment_weight = float(config['alignment_weight'])
            dimension = self.t5.config.d_model
            # Aux initialization must not advance CPU/CUDA dropout or loader RNG.
            # Layers are created on CPU. All weights move with the enclosing model.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(int(config.get('alignment_init_seed', 91021)))
                self.alignment_sid_projection = nn.Linear(4 * dimension, dimension)
                self.alignment_text_projection = nn.Linear(text.shape[1], dimension)
            self.register_buffer('alignment_text', torch.from_numpy(text.copy()), persistent=False)
            codes = torch.tensor([
                [self.tokenizer.actionpiece.rank[(slot, features[item][slot])]
                 for slot in range(4)] for item in items
            ], dtype=torch.long)
            self.register_buffer('alignment_codes', codes, persistent=False)

        def forward(self, batch):
            rows = batch.get('alignment_rows')
            inputs = {key: value for key, value in batch.items() if key != 'alignment_rows'}
            outputs = super().forward(inputs)
            if rows is None or not rows.numel():
                if self.training and self.alignment_weight > 0:
                    raise ValueError('Training alignment requires valid history rows')
                return outputs
            rec_loss = outputs.loss
            # lambda=0 retains the original CE graph exactly, including gradients.
            with torch.set_grad_enabled(torch.is_grad_enabled() and self.alignment_weight > 0):
                embeddings = self.t5.get_input_embeddings()(self.alignment_codes[rows])
                z = self.alignment_sid_projection(embeddings.flatten(start_dim=1))
                target = self.alignment_text_projection(self.alignment_text[rows])
                align_loss = -F.cosine_similarity(z, target, dim=-1).mean()
            if self.alignment_weight > 0:
                outputs.loss = rec_loss + self.alignment_weight * align_loss
            outputs['alignment_rec_loss'] = rec_loss.detach()
            outputs['alignment_loss'] = align_loss.detach()
            outputs['alignment_cosine'] = -align_loss.detach()
            outputs['alignment_weighted_loss'] = self.alignment_weight * align_loss.detach()
            _, negative, gap, top1 = pair_diagnostics(z, target, rows)
            outputs['alignment_negative_cosine'] = negative
            outputs['alignment_cosine_gap'] = gap
            outputs['alignment_top1'] = top1
            outputs['alignment_count'] = rows.numel()
            return outputs

    return AlignmentModel, AlignmentTokenizer
