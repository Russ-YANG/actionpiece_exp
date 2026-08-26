"""Tests for E4 ActionPiece merge-modality analysis."""

import unittest

from scripts.analyze_modality_merges import (
    analyze_exact_slot_permutation,
    analyze_records,
)


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
    self.assertFalse(summary['candidate_opportunity_adjustment']['available'])

  def test_adjusts_selected_merges_for_candidate_opportunities(self):
    records = [
        merge_record(
            1, 100, [10, 20], [0, 1], [4, 2], [[0, 1], [4, 2]]
        ),
        merge_record(
            2, 101, [11, 12], [1, 1], [2, 2], [[1, 1], [2, 2]]
        ),
    ]
    records[0]['modality_opportunities'] = {
        'selected_category': 'text_image',
        'candidate_pair_counts': {
            'text_text': 2,
            'image_image': 2,
            'text_image': 4,
        },
        'candidate_priority_mass': {
            'text_text': 1,
            'image_image': 1,
            'text_image': 2,
        },
        'candidate_max_priority': {
            'text_text': 0.5,
            'image_image': 0.5,
            'text_image': 1,
        },
    }
    records[1]['modality_opportunities'] = {
        'selected_category': 'text_text',
        'candidate_pair_counts': {
            'text_text': 2,
            'image_image': 2,
            'text_image': 4,
        },
        'candidate_priority_mass': {
            'text_text': 1,
            'image_image': 1,
            'text_image': 2,
        },
        'candidate_max_priority': {
            'text_text': 1,
            'image_image': 0.5,
            'text_image': 0.5,
        },
    }

    summary = analyze_records(
        records,
        text_slots={0, 1, 2, 3},
        image_slots={4, 5, 6, 7},
        hash_slot=8,
        bins=2,
        example_limit=0,
    )
    adjusted = summary['candidate_opportunity_adjustment']
    self.assertTrue(adjusted['available'])
    self.assertEqual(adjusted['covered_merge_events'], 2)
    first_cross = adjusted['first_cross_among_pure_selections']
    self.assertEqual(first_cross['observed_text_image_count'], 1)
    self.assertAlmostEqual(
        first_cross['expected_text_image_by_candidate_count'], 1.0
    )
    self.assertAlmostEqual(first_cross['candidate_count_enrichment'], 1.0)

  def test_exactly_enumerates_equal_size_slot_partitions(self):
    records = [
        merge_record(1, 100, [10, 20], [0, 1], [2, 2], [[0, 1], [2, 2]]),
        merge_record(2, 101, [11, 21], [1, 1], [3, 2], [[1, 1], [3, 2]]),
        merge_record(3, 102, [12, 13], [0, 2], [1, 3], [[0, 2], [1, 3]]),
    ]
    result = analyze_exact_slot_permutation(
        records,
        text_slots={0, 1},
        image_slots={2, 3},
        hash_slot=4,
    )
    self.assertEqual(result['labeled_partition_count'], 6)
    self.assertEqual(result['unlabeled_partition_count'], 3)
    metric = result['metrics']['direct_cross_rate_among_pure']
    self.assertAlmostEqual(metric['observed'], 2 / 3)
    self.assertGreaterEqual(metric['exact_p_greater_equal'], 1 / 3)


if __name__ == '__main__':
  unittest.main()
