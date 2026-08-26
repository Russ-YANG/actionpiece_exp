#!/usr/bin/env python3
"""Analyze modality composition and transitions in an ActionPiece merge log."""

import argparse
import collections
import itertools
import json
from pathlib import Path
import statistics
from typing import Any, Iterable


Counter = collections.Counter


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          'Analyze text/image/hash composition, operand transitions, phase '
          'trends, and frequency-weighted proportions in an ActionPiece log.'
      )
  )
  parser.add_argument('log_path', type=Path, help='Path to merge_log.jsonl')
  parser.add_argument(
      '--text-slots',
      type=int,
      default=4,
      help='Number of text feature slots starting at 0 (default: 4).',
  )
  parser.add_argument(
      '--image-slots',
      type=int,
      default=4,
      help='Number of image feature slots after text slots (default: 4).',
  )
  parser.add_argument(
      '--hash-slot',
      type=int,
      default=8,
      help='Hash feature slot, or -1 to disable (default: 8).',
  )
  parser.add_argument(
      '--bins',
      type=int,
      default=10,
      help='Number of chronological merge-step bins (default: 10).',
  )
  parser.add_argument(
      '--examples',
      type=int,
      default=3,
      help='Maximum example records shown per operand transition (default: 3).',
  )
  parser.add_argument(
      '--json-output',
      type=Path,
      help='Optional path for a machine-readable summary.',
  )
  parser.add_argument(
      '--exact-slot-permutation',
      action='store_true',
      help=(
          'Enumerate every equal-size semantic-slot partition and compare the '
          'real text/image boundary with the exact null distribution.'
      ),
  )
  return parser.parse_args()


def _feature_slots(features: Iterable[Iterable[int]]) -> set[int]:
  return {
      int(feature[0])
      for feature in features
      if isinstance(feature, (list, tuple)) and feature
  }


def classify_composition(
    features: list[list[int]],
    text_slots: set[int],
    image_slots: set[int],
    hash_slot: int,
) -> tuple[str, bool]:
  """Classify the recursively decoded basic features of one token."""
  slots = _feature_slots(features)
  has_text = bool(slots & text_slots)
  has_image = bool(slots & image_slots)
  has_hash = hash_slot >= 0 and hash_slot in slots
  known_slots = text_slots | image_slots
  if hash_slot >= 0:
    known_slots = known_slots | {hash_slot}
  has_other = bool(slots - known_slots)

  if has_text and has_image:
    kind = 'mixed'
  elif has_text:
    kind = 'text'
  elif has_image:
    kind = 'image'
  elif has_hash:
    kind = 'hash'
  else:
    kind = 'unknown'
  if has_other:
    kind += '_with_unknown'
  return kind, has_hash


def classify(
    features: list[list[int]],
    text_slots: set[int],
    image_slots: set[int],
    hash_slot: int,
) -> tuple[str, bool]:
  """Backward-compatible output-token labels used by the original script."""
  kind, has_hash = classify_composition(
      features, text_slots, image_slots, hash_slot
  )
  labels = {
      'text': 'text_only',
      'image': 'image_only',
      'mixed': 'text_image',
      'hash': 'hash_only',
  }
  return labels.get(kind, kind), has_hash


def classify_transition(
    left: tuple[str, bool], right: tuple[str, bool]
) -> str:
  """Classify an unordered pair of merge operands."""
  left_kind, left_hash = left
  right_kind, right_hash = right
  if left_hash or right_hash or left_kind == 'hash' or right_kind == 'hash':
    return 'hash_related'
  if 'unknown' in left_kind or 'unknown' in right_kind:
    return 'unknown'

  pair = frozenset((left_kind, right_kind))
  if left_kind == right_kind:
    return f'{left_kind}_{right_kind}'
  names = {
      frozenset(('text', 'image')): 'text_image',
      frozenset(('mixed', 'text')): 'mixed_text',
      frozenset(('mixed', 'image')): 'mixed_image',
  }
  return names.get(pair, '_'.join(sorted(pair)))


def _basic_operand_features(
    token: int,
    rule: Any,
    token_features: dict[int, list[list[int]]],
) -> list[list[int]] | None:
  if token in token_features:
    return token_features[token]
  if (
      isinstance(rule, list)
      and len(rule) == 2
      and all(isinstance(value, int) for value in rule)
  ):
    return [rule]
  return None


def _number(record: dict[str, Any], key: str) -> float:
  value = record.get(key, 0)
  return float(value) if isinstance(value, (int, float)) else 0.0


def _phase_group(transition: str) -> str:
  if transition in {'text_text', 'image_image', 'text_image'}:
    return transition
  if transition.startswith('mixed'):
    return 'mixed_related'
  if transition == 'hash_related':
    return transition
  return 'unknown'


def _safe_ratio(numerator: float, denominator: float) -> float:
  return numerator / denominator if denominator else 0.0


def analyze_opportunities(
    merge_records: list[dict[str, Any]],
) -> dict[str, Any]:
  """Compare selected merge classes with their candidate exposure."""
  observed: Counter[str] = Counter()
  expected_count: Counter[str] = Counter()
  expected_mass: Counter[str] = Counter()
  candidate_count_sums: Counter[str] = Counter()
  candidate_mass_sums: Counter[str] = Counter()
  covered = 0
  pure_selected_steps = 0
  pure_observed_cross = 0
  pure_expected_count = 0.0
  pure_expected_mass = 0.0
  pure_kinds = {'text_text', 'image_image', 'text_image'}

  for record in merge_records:
    opportunity = record.get('modality_opportunities')
    if not isinstance(opportunity, dict):
      continue
    counts = opportunity.get('candidate_pair_counts')
    masses = opportunity.get('candidate_priority_mass')
    selected = opportunity.get('selected_category')
    if not isinstance(counts, dict) or not isinstance(masses, dict):
      continue
    if not isinstance(selected, str):
      continue
    counts = {
        str(kind): max(float(value), 0.0)
        for kind, value in counts.items()
        if isinstance(value, (int, float))
    }
    masses = {
        str(kind): max(float(value), 0.0)
        for kind, value in masses.items()
        if isinstance(value, (int, float))
    }
    total_count = sum(counts.values())
    total_mass = sum(masses.values())
    if not total_count or not total_mass:
      continue
    covered += 1
    observed[selected] += 1
    for kind, value in counts.items():
      candidate_count_sums[kind] += value
      expected_count[kind] += value / total_count
    for kind, value in masses.items():
      candidate_mass_sums[kind] += value
      expected_mass[kind] += value / total_mass

    if selected in pure_kinds:
      pure_selected_steps += 1
      pure_observed_cross += int(selected == 'text_image')
      pure_count = sum(counts.get(kind, 0.0) for kind in pure_kinds)
      pure_mass = sum(masses.get(kind, 0.0) for kind in pure_kinds)
      pure_expected_count += _safe_ratio(
          counts.get('text_image', 0.0), pure_count
      )
      pure_expected_mass += _safe_ratio(
          masses.get('text_image', 0.0), pure_mass
      )

  categories = sorted(
      set(observed) | set(expected_count) | set(expected_mass)
  )
  category_summary = {}
  for kind in categories:
    actual = observed[kind]
    expected_by_count = expected_count[kind]
    expected_by_mass = expected_mass[kind]
    category_summary[kind] = {
        'selected_count': actual,
        'selected_rate': _safe_ratio(actual, covered),
        'mean_candidate_count': _safe_ratio(
            candidate_count_sums[kind], covered
        ),
        'mean_candidate_priority_mass': _safe_ratio(
            candidate_mass_sums[kind], covered
        ),
        'expected_selections_by_candidate_count': expected_by_count,
        'candidate_count_enrichment': _safe_ratio(actual, expected_by_count),
        'expected_selections_by_priority_mass': expected_by_mass,
        'priority_mass_enrichment': _safe_ratio(actual, expected_by_mass),
    }

  return {
      'available': bool(covered),
      'covered_merge_events': covered,
      'coverage': _safe_ratio(covered, len(merge_records)),
      'categories': category_summary,
      'first_cross_among_pure_selections': {
          'selected_pure_merge_events': pure_selected_steps,
          'observed_text_image_count': pure_observed_cross,
          'observed_text_image_rate': _safe_ratio(
              pure_observed_cross, pure_selected_steps
          ),
          'expected_text_image_by_candidate_count': pure_expected_count,
          'candidate_count_enrichment': _safe_ratio(
              pure_observed_cross, pure_expected_count
          ),
          'expected_text_image_by_priority_mass': pure_expected_mass,
          'priority_mass_enrichment': _safe_ratio(
              pure_observed_cross, pure_expected_mass
          ),
      },
  }


def _permutation_metrics(summary: dict[str, Any]) -> dict[str, float]:
  outputs = summary['output_composition']['rule_counts']
  transitions = summary['operand_transitions']['rule_counts']
  pure_kinds = ('text_text', 'image_image', 'text_image')
  fusion_kinds = ('text_image', 'mixed_text', 'mixed_image', 'mixed_mixed')
  nonhash_kinds = ('text_text', 'image_image', *fusion_kinds)
  return {
      'direct_cross_rate_among_pure': _safe_ratio(
          transitions.get('text_image', 0),
          sum(transitions.get(kind, 0) for kind in pure_kinds),
      ),
      'fusion_rule_rate_no_hash': _safe_ratio(
          sum(transitions.get(kind, 0) for kind in fusion_kinds),
          sum(transitions.get(kind, 0) for kind in nonhash_kinds),
      ),
      'output_mixed_rate': _safe_ratio(
          outputs.get('text_image', 0), sum(outputs.values())
      ),
  }


def _exact_null_summary(observed: float, values: list[float]) -> dict[str, Any]:
  mean = statistics.fmean(values)
  std = statistics.pstdev(values)
  tolerance = 1e-12
  deviation = abs(observed - mean)
  return {
      'observed': observed,
      'null_mean': mean,
      'null_std': std,
      'null_min': min(values),
      'null_max': max(values),
      'enrichment': _safe_ratio(observed, mean),
      'z_score': _safe_ratio(observed - mean, std),
      'exact_p_greater_equal': _safe_ratio(
          sum(value >= observed - tolerance for value in values), len(values)
      ),
      'exact_p_less_equal': _safe_ratio(
          sum(value <= observed + tolerance for value in values), len(values)
      ),
      'exact_p_two_sided': _safe_ratio(
          sum(
              abs(value - mean) >= deviation - tolerance for value in values
          ),
          len(values),
      ),
  }


def analyze_exact_slot_permutation(
    records: list[dict[str, Any]],
    text_slots: set[int],
    image_slots: set[int],
    hash_slot: int,
) -> dict[str, Any]:
  """Test the real modality boundary against all equal-size slot partitions."""
  if len(text_slots) != len(image_slots):
    raise ValueError('Exact slot permutation requires equal modality sizes.')
  semantic_slots = sorted(text_slots | image_slots)
  partitions = list(itertools.combinations(semantic_slots, len(text_slots)))
  real_partition = tuple(sorted(text_slots))
  distributions: dict[str, list[float]] = collections.defaultdict(list)
  observed = None
  for partition in partitions:
    candidate_text = set(partition)
    candidate_image = set(semantic_slots) - candidate_text
    summary = analyze_records(
        records,
        candidate_text,
        candidate_image,
        hash_slot,
        bins=1,
        example_limit=0,
    )
    metrics = _permutation_metrics(summary)
    if partition == real_partition:
      observed = metrics
    for name, value in metrics.items():
      distributions[name].append(value)
  if observed is None:
    raise ValueError('The real text slot partition was not enumerated.')
  return {
      'method': 'exhaustive_equal_size_semantic_slot_label_permutation',
      'real_text_slots': sorted(text_slots),
      'semantic_slots': semantic_slots,
      'labeled_partition_count': len(partitions),
      'unlabeled_partition_count': len(partitions) // 2,
      'metrics': {
          name: _exact_null_summary(observed[name], values)
          for name, values in distributions.items()
      },
  }


def analyze_records(
    records: list[dict[str, Any]],
    text_slots: set[int],
    image_slots: set[int],
    hash_slot: int,
    bins: int,
    example_limit: int = 3,
) -> dict[str, Any]:
  """Return rule-, frequency-, and phase-level modality statistics."""
  merge_records = [record for record in records if record.get('event') == 'merge']
  max_step = max(
      (
          record.get('step', 0)
          for record in merge_records
          if isinstance(record.get('step'), int)
      ),
      default=0,
  )
  bins = max(1, bins)
  token_features: dict[int, list[list[int]]] = {}
  output_counts: Counter[str] = Counter()
  output_priority: Counter[str] = Counter()
  output_affected: Counter[str] = Counter()
  transition_counts: Counter[str] = Counter()
  transition_priority: Counter[str] = Counter()
  transition_affected: Counter[str] = Counter()
  first_steps: dict[str, int] = {}
  examples: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  phase_counts: list[Counter[str]] = [Counter() for _ in range(bins)]
  phase_priority: list[Counter[str]] = [Counter() for _ in range(bins)]
  hash_output_count = 0
  unresolved_operands = 0

  for record in merge_records:
    features = record.get('new_token_basic_features')
    new_token = record.get('new_token')
    if not isinstance(features, list) or not isinstance(new_token, int):
      continue
    token_features[new_token] = features
    output_type, has_hash = classify(
        features, text_slots, image_slots, hash_slot
    )
    priority = _number(record, 'priority')
    affected = _number(record, 'affected_sequences')
    output_counts[output_type] += 1
    output_priority[output_type] += priority
    output_affected[output_type] += affected
    hash_output_count += int(has_hash)

    merged_tokens = record.get('merged_tokens')
    if not (
        isinstance(merged_tokens, list)
        and len(merged_tokens) == 2
        and all(isinstance(token, int) for token in merged_tokens)
    ):
      transition = 'unknown'
      unresolved_operands += 1
    else:
      left_features = _basic_operand_features(
          merged_tokens[0], record.get('left_token_rule'), token_features
      )
      right_features = _basic_operand_features(
          merged_tokens[1], record.get('right_token_rule'), token_features
      )
      if left_features is None or right_features is None:
        transition = 'unknown'
        unresolved_operands += 1
      else:
        transition = classify_transition(
            classify_composition(
                left_features, text_slots, image_slots, hash_slot
            ),
            classify_composition(
                right_features, text_slots, image_slots, hash_slot
            ),
        )

    transition_counts[transition] += 1
    transition_priority[transition] += priority
    transition_affected[transition] += affected
    step = record.get('step')
    if isinstance(step, int):
      first_steps.setdefault(transition, step)
      phase_index = min(bins - 1, max(0, (step - 1) * bins // max(max_step, 1)))
    else:
      phase_index = bins - 1
    phase_group = _phase_group(transition)
    phase_counts[phase_index][phase_group] += 1
    phase_priority[phase_index][phase_group] += priority
    if len(examples[transition]) < example_limit:
      examples[transition].append({
          'step': step,
          'new_token': new_token,
          'merged_tokens': merged_tokens,
          'features': features,
      })

  phases = []
  for index in range(bins):
    start = (max_step * index) // bins + 1 if max_step else 0
    end = (max_step * (index + 1)) // bins if max_step else 0
    phases.append({
        'bin': index + 1,
        'step_start': start,
        'step_end': end,
        'rule_counts': dict(phase_counts[index]),
        'priority_weights': dict(phase_priority[index]),
    })

  return {
      'total_merge_events': len(merge_records),
      'max_step': max_step,
      'slots': {
          'text': sorted(text_slots),
          'image': sorted(image_slots),
          'hash': hash_slot,
      },
      'output_composition': {
          'rule_counts': dict(output_counts),
          'priority_weights': dict(output_priority),
          'affected_sequence_weights': dict(output_affected),
          'includes_hash_count': hash_output_count,
      },
      'operand_transitions': {
          'rule_counts': dict(transition_counts),
          'priority_weights': dict(transition_priority),
          'affected_sequence_weights': dict(transition_affected),
          'first_steps': first_steps,
          'unresolved_operands': unresolved_operands,
          'examples': dict(examples),
      },
      'candidate_opportunity_adjustment': analyze_opportunities(merge_records),
      'phases': phases,
  }


def format_pct(value: float, total: float) -> str:
  return f'{100 * value / total:.2f}%' if total else '0.00%'


def _print_table(
    title: str,
    counts: dict[str, int],
    priorities: dict[str, float],
    affected: dict[str, float],
    first_steps: dict[str, int] | None = None,
) -> None:
  total_count = sum(counts.values())
  total_priority = sum(priorities.values())
  total_affected = sum(affected.values())
  print(f'\n{title}')
  print(
      f'{"Type":<22} {"Rules":>8} {"Rule %":>9} '
      f'{"Priority %":>11} {"Affected %":>12} {"First":>8}'
  )
  print('-' * 76)
  for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
    first = '' if first_steps is None else str(first_steps.get(name, ''))
    print(
        f'{name:<22} {count:>8} {format_pct(count, total_count):>9} '
        f'{format_pct(priorities.get(name, 0), total_priority):>11} '
        f'{format_pct(affected.get(name, 0), total_affected):>12} '
        f'{first:>8}'
    )


def _print_phases(phases: list[dict[str, Any]]) -> None:
  groups = [
      'text_text',
      'image_image',
      'text_image',
      'mixed_related',
      'hash_related',
      'unknown',
  ]
  print('\nChronological transition proportions (rule count)')
  print(
      f'{"Steps":<15}' + ''.join(f'{group:>15}' for group in groups)
  )
  print('-' * (15 + 15 * len(groups)))
  for phase in phases:
    counts = phase['rule_counts']
    total = sum(counts.values())
    step_range = f'{phase["step_start"]}-{phase["step_end"]}'
    print(
        f'{step_range:<15}'
        + ''.join(f'{format_pct(counts.get(group, 0), total):>15}' for group in groups)
    )


def _print_opportunities(opportunities: dict[str, Any]) -> None:
  if not opportunities['available']:
    print('\nCandidate opportunity adjustment: unavailable in this log.')
    return
  print('\nCandidate opportunity-adjusted selections')
  print(
      f'{"Type":<22} {"Selected":>9} {"Count exp.":>12} '
      f'{"Count enrich.":>14} {"Mass exp.":>12} {"Mass enrich.":>13}'
  )
  print('-' * 86)
  for kind, values in sorted(
      opportunities['categories'].items(),
      key=lambda item: (-item[1]['selected_count'], item[0]),
  ):
    print(
        f'{kind:<22} {values["selected_count"]:>9} '
        f'{values["expected_selections_by_candidate_count"]:>12.1f} '
        f'{values["candidate_count_enrichment"]:>14.3f} '
        f'{values["expected_selections_by_priority_mass"]:>12.1f} '
        f'{values["priority_mass_enrichment"]:>13.3f}'
    )
  first_cross = opportunities['first_cross_among_pure_selections']
  print(
      'First text-image merges among pure-modality selections: '
      f'{first_cross["observed_text_image_count"]}/'
      f'{first_cross["selected_pure_merge_events"]} '
      f'({100 * first_cross["observed_text_image_rate"]:.2f}%), '
      f'count enrichment={first_cross["candidate_count_enrichment"]:.3f}, '
      f'mass enrichment={first_cross["priority_mass_enrichment"]:.3f}'
  )


def _print_exact_permutation(permutation: dict[str, Any]) -> None:
  print('\nExact semantic-slot partition permutation')
  print(
      f'Compared the real boundary with '
      f'{permutation["labeled_partition_count"]} labeled partitions.'
  )
  print(
      f'{"Metric":<34} {"Observed":>10} {"Null mean":>11} '
      f'{"Enrich.":>9} {"p (greater)":>12} {"p (two)":>9}'
  )
  print('-' * 91)
  for name, values in permutation['metrics'].items():
    print(
        f'{name:<34} {values["observed"]:>10.4f} '
        f'{values["null_mean"]:>11.4f} {values["enrichment"]:>9.3f} '
        f'{values["exact_p_greater_equal"]:>12.4f} '
        f'{values["exact_p_two_sided"]:>9.4f}'
    )


def main() -> None:
  args = parse_args()
  text_slots = set(range(args.text_slots))
  image_start = args.text_slots
  image_slots = set(range(image_start, image_start + args.image_slots))
  records = []
  malformed_count = 0
  with args.log_path.open(encoding='utf-8') as log_file:
    for line in log_file:
      try:
        records.append(json.loads(line))
      except json.JSONDecodeError:
        malformed_count += 1

  summary = analyze_records(
      records,
      text_slots,
      image_slots,
      args.hash_slot,
      args.bins,
      args.examples,
  )
  summary['log_path'] = str(args.log_path)
  summary['malformed_records'] = malformed_count
  if args.exact_slot_permutation:
    summary['exact_slot_permutation'] = analyze_exact_slot_permutation(
        records, text_slots, image_slots, args.hash_slot
    )

  print(f'Log: {args.log_path}')
  print(f'Total merge events: {summary["total_merge_events"]}')
  print(
      f'Slots: text={summary["slots"]["text"]}, '
      f'image={summary["slots"]["image"]}, '
      f'hash={summary["slots"]["hash"]}'
  )
  output = summary['output_composition']
  _print_table(
      'Output-token composition',
      output['rule_counts'],
      output['priority_weights'],
      output['affected_sequence_weights'],
  )
  transitions = summary['operand_transitions']
  _print_table(
      'Exact operand transitions',
      transitions['rule_counts'],
      transitions['priority_weights'],
      transitions['affected_sequence_weights'],
      transitions['first_steps'],
  )
  print(
      '\nOutputs including hash: '
      f'{output["includes_hash_count"]} '
      f'({format_pct(output["includes_hash_count"], summary["total_merge_events"])})'
  )
  print(f'Unresolved operand transitions: {transitions["unresolved_operands"]}')
  if malformed_count:
    print(f'Skipped malformed records: {malformed_count}')
  _print_phases(summary['phases'])
  _print_opportunities(summary['candidate_opportunity_adjustment'])
  if args.exact_slot_permutation:
    _print_exact_permutation(summary['exact_slot_permutation'])

  if args.examples:
    print('\nTransition examples:')
    for transition, examples in transitions['examples'].items():
      print(f'  {transition}:')
      for example in examples:
        print(
            f'    step={example["step"]}, token={example["new_token"]}, '
            f'operands={example["merged_tokens"]}, '
            f'features={example["features"]}'
        )

  if args.json_output:
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    with args.json_output.open('w', encoding='utf-8') as output_file:
      json.dump(summary, output_file, ensure_ascii=False, indent=2)
    print(f'\nJSON summary saved: {args.json_output}')


if __name__ == '__main__':
  main()
