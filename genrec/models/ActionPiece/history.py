"""Cached, losslessly expandable item-local representations for P0/P1/P2.

These policies choose merge depth, not SID information content. There is no
learned allocator or hard history budget in this initial pilot.
"""

import collections
import json
from pathlib import Path

import numpy as np


POLICIES = {'raw', 'full', 'middle', 'random'}


class ItemHistoryRepresentations:
  """One deterministic merge trajectory per canonical item state."""

  def __init__(self, core, states, cache_path=None):
    states = sorted({tuple(int(t) for t in state) for state in states})
    # Store exact source identity once at the artifact boundary. This avoids
    # accidentally reusing token IDs from a different vocabulary or feature map.
    source = {
        'version': 1,
        'n_categories': core.n_categories,
        'vocab': [list(feature) for feature in core.vocab],
        'priority': list(core.priority),
        'states': [list(state) for state in states],
    }
    cache = Path(cache_path) if cache_path is not None else None
    saved = None
    if cache is not None and cache.exists():
      with cache.open() as stream:
        saved = json.load(stream)
    if saved is not None and saved['source'] == source:
      paths = saved['paths']
    else:
      paths = [
          core.encode(np.asarray([state]), shuffle='none', return_merge_path=True)
          for state in states
      ]
      if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # The caller serializes distributed initialization via main_process_first.
        temporary = cache.with_suffix(cache.suffix + '.tmp')
        with temporary.open('w') as stream:
          json.dump({'source': source, 'paths': paths}, stream)
        temporary.replace(cache)
    self.paths = {
        state: tuple(tuple(int(t) for t in encoding) for encoding in path)
        for state, path in zip(states, paths)
    }

  def encode(self, states, policy, rng=None):
    if policy not in POLICIES:
      raise ValueError(f'Unknown history policy: {policy}')
    output = []
    for state in states:
      path = self.paths[tuple(int(t) for t in state)]
      if policy == 'raw':
        index = 0
      elif policy == 'full':
        index = len(path) - 1
      elif policy == 'middle':
        index = (len(path) - 1) // 2
      else:
        if rng is None:
          raise ValueError('Random granularity requires an explicit RNG.')
        index = int(rng.integers(len(path)))
      output.extend(path[index])
    return output

  def summary(self):
    paths = list(self.paths.values())
    shortest = [len(path[-1]) for path in paths]
    return {
        'n_item_states': len(paths),
        'n_with_multiple_lengths': sum(len(path) > 1 for path in paths),
        'candidate_count_histogram': dict(sorted(collections.Counter(
            len(path) for path in paths
        ).items())),
        'shortest_length_histogram': dict(sorted(collections.Counter(shortest).items())),
        'mean_raw_length': float(np.mean([len(path[0]) for path in paths])),
        'mean_shortest_length': float(np.mean(shortest)),
        'lengths_include_hash_atom': True,
        'lengths_include_bos_eos': False,
        'scope': 'Unweighted catalog states, not interaction-weighted histories.',
    }
