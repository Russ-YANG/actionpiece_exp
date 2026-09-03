"""Tests for bounded run artifact filenames."""

from unittest import mock
import unittest

from genrec import utils


class UtilsFilenameTest(unittest.TestCase):

  @mock.patch('genrec.utils.get_command_line_args_str')
  def test_long_command_is_replaced_with_stable_hash(self, command_line_args):
    command_line_args.return_value = 'x' * 300
    config = {
        'run_id': 'beauty_e4_item_local_history_atomic9_target_d768_bs128',
        'run_local_time': 'Sep-03-2026_08-00-00',
    }

    filename = utils.get_file_name(config, suffix='.log')

    self.assertLessEqual(len(filename.encode()), 240)
    self.assertIn('-args', filename)
    self.assertTrue(filename.endswith('-.log'))

  @mock.patch('genrec.utils.get_command_line_args_str')
  def test_short_command_keeps_readable_name(self, command_line_args):
    command_line_args.return_value = 'main.py_--eval_batch_size=32'
    config = {
        'run_id': 'beauty',
        'run_local_time': 'Sep-03-2026_08-00-00',
    }

    filename = utils.get_file_name(config, suffix='.log')

    self.assertIn('main.py_--eval_batch_size=32', filename)


if __name__ == '__main__':
  unittest.main()
