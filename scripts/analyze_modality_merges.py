#!/usr/bin/env python3
"""Analyze modality composition and transitions in an ActionPiece merge log."""

import argparse
import collections
import json
from pathlib import Path
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
