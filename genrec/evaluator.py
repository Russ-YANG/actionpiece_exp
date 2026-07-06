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

"""Evaluator for GenRec."""

import torch


class Evaluator:
  """Evaluator for GenRec."""

  def __init__(self, config, tokenizer):
    self.config = config
    self.tokenizer = tokenizer
    self.maxk = max(config['topk'])
    self.candidate_multiplier = config.get(
        'domain_filter_candidate_multiplier', 4
    )

  @property
  def max_candidates(self):
    return self.maxk * self.candidate_multiplier

  @property
  def eos_token(self):
    """Returns the end of sequence token."""
    return self.tokenizer.eos_token

  def calculate_pos_index(self, preds, labels):
    """Calculate the position index of the ground truth items.

    Args:
      preds: The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels: The ground truth token sequences, of shape (batch_size, seq_len).

    Returns:
      A boolean tensor of shape (batch_size, maxk) indicating whether the
      prediction at each position is correct.
    """
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == self.maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {self.maxk}'

    pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      if self.eos_token in cur_label:
        eos_pos = cur_label.index(self.eos_token)
        cur_label = cur_label[:eos_pos]
      for j in range(self.maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

  def recall_at_k(self, pos_index, k):
    return pos_index[:, :k].sum(dim=1).cpu().float()

  def ndcg_at_k(self, pos_index, k):
    # Assume only one ground truth item per example
    ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
    dcg = 1.0 / torch.log2(ranks + 1)
    dcg = torch.where(pos_index, dcg, 0)
    return dcg[:, :k].sum(dim=1).cpu().float()

  def calculate_err_index(self, preds):
    preds = preds.detach().cpu()

    return preds[:, :self.maxk, 0] == -1

  def filter_preds_by_source(self, preds, source_id):
    preds = preds.detach().cpu()
    source_id = source_id.detach().cpu()
    filtered = torch.full(
        (preds.shape[0], self.maxk, preds.shape[-1]),
        -1,
        dtype=preds.dtype,
    )
    for i in range(preds.shape[0]):
      allowed_labels = self.tokenizer.source_allowed_labels[source_id[i].item()]
      seen = set()
      out_idx = 0
      for pred in preds[i].tolist():
        pred_key = tuple(pred)
        if pred_key in allowed_labels and pred_key not in seen:
          filtered[i, out_idx] = torch.LongTensor(pred)
          seen.add(pred_key)
          out_idx += 1
          if out_idx == self.maxk:
            break
    return filtered

  def err_at_k(self, err_index, k):
    """Calculate the percentage of illegal predictions among the top k generated token sequences.

    Args:
        err_index (torch.Tensor): A boolean tensor indicating if the prediction
          is illegal, shape (batch_size, maxk).
        k (int): The top k predictions to consider.

    Returns:
        torch.Tensor: The percentage of illegal predictions, shape (batch_size).
    """
    return err_index[:, :k].float().mean(dim=1).cpu()

  def calculate_metrics(self, preds, labels, source_id=None):
    """Calculate the evaluation metrics.

    Args:
      preds (torch.Tensor): The predicted token sequences, of shape
        (batch_size, maxk, seq_len).
      labels (torch.Tensor): The ground truth token sequences, of shape
        (batch_size, seq_len).

    Returns:
        dict: A dictionary containing the evaluation metrics.
    """
    metric2func = {
        'recall': self.recall_at_k,
        'ndcg': self.ndcg_at_k,
        'err': self.err_at_k,
    }
    results = {}
    if source_id is not None:
      preds = self.filter_preds_by_source(preds, source_id)
    pos_index = self.calculate_pos_index(preds, labels)
    err_index = self.calculate_err_index(preds)
    for metric in self.config['metrics']:
      index = err_index if metric == 'err' else pos_index
      for k in self.config['topk']:
        key = f'{metric}@{k}'
        values = metric2func[metric](index, k)
        results[key] = values
        if source_id is not None:
          source_id_cpu = source_id.detach().cpu()
          for sid, source in enumerate(self.tokenizer.id2source):
            mask = source_id_cpu == sid
            if mask.any():
              results[f'{source}/{key}'] = values[mask]
    return results
