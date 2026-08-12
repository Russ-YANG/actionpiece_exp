"""Tests for E4 ActionPiece merge-modality analysis."""

import unittest

from scripts.analyze_modality_merges import analyze_records


def merge_record(
    step,
    new_token,
    operands,
    left_rule,
    right_rule,
    features,
    priority=10,
    affected=5,
):
  return {
      'event': 'merge',
      'step': step,
      'new_token': new_token,
      'merged_tokens': operands,
      'left_token_rule': left_rule,
      'right_token_rule': right_rule,
      'new_token_basic_features': features,
      'priority': priority,
      'affected_sequences': affected,
  }


class AnalyzeModalityMergesTest(unittest.TestCase):

  def test_tracks_recursive_operand_transitions_and_hash(self):
    records = [
        merge_record(
            1, 100, [10, 11], [0, 1], [1, 2], [[0, 1], [1, 2]]
        ),
        merge_record(
            2, 101, [20, 21], [4, 3], [5, 4], [[4, 3], [5, 4]]
        ),
        merge_record(
            3,
            102,
            [100, 101],
            [-1, 10, 11],
            [-1, 20, 21],
            [[0, 1], [1, 2], [4, 3], [5, 4]],
        ),
        merge_record(
            4,
            103,
            [102, 12],
            [-1, 100, 101],
            [2, 5],
            [[0, 1], [1, 2], [2, 5], [4, 3], [5, 4]],
        ),
        merge_record(
            5, 104, [30, 13], [8, 6], [3, 7], [[8, 6], [3, 7]]
        ),
    ]

    summary = analyze_records(
        records,
        text_slots={0, 1, 2, 3},
        image_slots={4, 5, 6, 7},
        hash_slot=8,
        bins=2,
        example_limit=1,
    )

    self.assertEqual(summary['total_merge_events'], 5)
    self.assertEqual(
        summary['operand_transitions']['rule_counts'],
        {
            'text_text': 1,
            'image_image': 1,
            'text_image': 1,
            'mixed_text': 1,
            'hash_related': 1,
        },
    )
    self.assertEqual(
        summary['output_composition']['rule_counts'],
        {'text_only': 2, 'image_only': 1, 'text_image': 2},
    )
    self.assertEqual(
        summary['output_composition']['includes_hash_count'], 1
    )
    self.assertEqual(
        summary['operand_transitions']['unresolved_operands'], 0
    )
    self.assertEqual(
        summary['operand_transitions']['first_steps']['text_image'], 3
    )


if __name__ == '__main__':
  unittest.main()
