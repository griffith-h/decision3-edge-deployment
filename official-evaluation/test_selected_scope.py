"""Selected final-only evaluation must not claim all historical stages were tested."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from build_comparison import build
from evaluation_verdict import require_final_complete
from stages import PIN


class SelectedScopeTests(unittest.TestCase):
    def verdict(self):
        return {'model_revision': PIN, 'final_stage': 'native-bf16',
                'final_stage_evaluation_complete': True, 'requests': 1,
                'sample_rows_sha256': 'frozen', 'all_stages_complete': False,
                'stages': [{'stage': 'native-bf16', 'sample_complete': True,
                            'completed_requests': 1, 'expected_requests': 1,
                            'board_method': {'count': 1}, 'counts': {'ok': 1}}]}

    def test_complete_final_only_does_not_require_cancelled_history(self):
        self.assertEqual(require_final_complete(self.verdict())['stage'], 'native-bf16')

    def test_incomplete_final_cannot_be_deployed(self):
        for change in ({'sample_complete': False}, {'completed_requests': 0}, {'counts': {}}):
            value = self.verdict()
            value['stages'][0].update(change)
            with self.assertRaisesRegex(ValueError, 'entire frozen sample'):
                require_final_complete(value)

    def test_report_respects_scope_and_preserves_full_history_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'protocol.json').write_text(json.dumps({'requested_stages': ['native-bf16'], 'model_revision': PIN}))
            stage = root / 'native-bf16'
            stage.mkdir()
            (stage / 'results.jsonl').write_text(json.dumps({'run_id': 'one', 'status': 'ok'}) + '\n')
            summary = {'board_method': {'count': 1, 'p50_ms': 10, 'p95_ms': 10},
                       'labelled_field_accuracy': {'accuracy': 1, 'correct': 1, 'denominator': 1, 'groups': {}},
                       'sampled_public_index': None, 'method_by_input_tokens': {},
                       'native_benchmark_metrics': {}, 'acceleration_counter_hits': {}}
            (stage / 'sample-summary.json').write_text(json.dumps(summary))
            manifest = {'run_ids_in_order': ['one'], 'kit_commit': 'pinned', 'sample_rows_sha256': 'frozen',
                        'seed': 20260919, 'benchmark_counts': {'test': 1}, 'group_count': 1}
            report = build(root, manifest, root / 'report.html')
            self.assertEqual(report['stages_total'], 1)
            self.assertFalse(report['all_stages_complete'])
            self.assertFalse(report['historical_comparison_performed'])
            self.assertTrue(report['all_requested_stages_complete'])
            self.assertTrue(report['final_stage_evaluation_complete'])
            require_final_complete(report)
            self.assertIn('本轮只测试最终', (root / 'report.html').read_text())


if __name__ == '__main__':
    unittest.main()
