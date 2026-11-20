"""Focused checks for framing, payload isolation and sample scoring scope."""
import io
import json
import os
from pathlib import Path
import selectors
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import board
from board_engine import BoardEngine
from score_sample import SampleSuite, labelled_fields, score


class HarnessTests(unittest.TestCase):
    def test_every_stage_warms_twice_before_ready(self):
        request = {'state': 'short', 'questions': {'q': {'type': 'noul'}}}
        model = types.SimpleNamespace(max_length=None, batch_size=64,
                                      system_one=Mock(return_value={'answers': {'q': {'noul': .5}}}))
        close, sent = Mock(), []
        with patch.object(board, 'setup', return_value=(model, close, lambda: {})), \
                patch.object(board, 'identity', return_value={}), \
                patch.object(Path, 'read_text', return_value=json.dumps(request)), \
                patch.object(board.sys, 'argv', ['board.py', '--stage', 'official-bf16']), \
                patch.object(board.sys, 'stdin', io.StringIO('')), \
                patch.object(board, 'send', side_effect=lambda value: sent.append((value, model.system_one.call_count))):
            board.main()
        self.assertEqual(model.system_one.call_count, 2)
        self.assertEqual(model.system_one.call_args_list[0].kwargs['state'], ' '.join(['alpha'] * 512))
        self.assertEqual(model.system_one.call_args_list[1].kwargs, request)
        self.assertEqual(sent[0][0]['event'], 'ready')
        self.assertEqual(sent[0][1], 2)
        close.assert_called_once()

    def test_native_stdout_and_multiple_buffered_frames(self):
        read, write = os.pipe()
        stream = os.fdopen(read, 'rb', buffering=0)
        engine = BoardEngine.__new__(BoardEngine)
        engine.buffer = b''
        engine.diag = io.BytesIO()
        engine.process = types.SimpleNamespace(stdout=stream)
        engine.selector = selectors.DefaultSelector()
        engine.selector.register(stream, selectors.EVENT_READ)
        try:
            os.write(write, b'native log\nnoise@@D3EVAL@@{"seq":1}\n@@D3EVAL@@{"seq":2}\n')
            self.assertEqual(engine.receive(1), {'seq': 1})
            self.assertEqual(engine.receive(1), {'seq': 2})
            self.assertIn(b'native log', engine.diag.getvalue())
        finally:
            os.close(write)
            stream.close()
            engine.selector.close()

    def test_inference_receives_only_input_and_questions(self):
        engine = BoardEngine.__new__(BoardEngine)
        engine.closed, engine.seq, engine.timeout = False, 0, 1
        wire = io.BytesIO()
        engine.process = types.SimpleNamespace(stdin=wire)
        engine.receive = lambda _: {'seq': 1, 'status': 'ok', 'response': {'answers': {}}}
        engine('input', {'q': {'type': 'noul'}})
        request = json.loads(wire.getvalue())
        self.assertEqual(set(request['payload']), {'state', 'questions'})
        self.assertNotIn('expected', request['payload'])

    def test_accuracy_keeps_failed_cases_in_denominator(self):
        rows = [{'questions': {'q': {'type': 'choice', 'criteria': {'A': 'a', 'B': 'b'}}},
                 'expected': {'q': 'A'}, '_evaluation': {'run_id': str(i), 'catalog_id': 28, 'group_id': str(i)}}
                for i in range(2)]
        results = {'0': {'status': 'ok', 'response': {'answers': {'q': {'choice': 'A'}}}},
                   '1': {'status': 'unsupported'}}
        actual = labelled_fields(rows, results)
        self.assertEqual((actual['correct'], actual['denominator'], actual['accuracy']), (1, 2, .5))

    def test_sample_rejects_duplicate_ids(self):
        row = {'_evaluation': {'run_id': 'same', 'catalog_id': 28}}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sample.jsonl'
            path.write_text((json.dumps(row) + '\n') * 2)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                SampleSuite(path)

    def test_native_scoring_uses_only_sample_and_keeps_errors(self):
        rows = [{'id': str(i), 'state': '',
                 'questions': {'q1': {'type': 'choice', 'instructions': 'q', 'criteria': {'A': 'a', 'B': 'b'}}},
                 'expected': {'q1': 'A'},
                 '_evaluation': {'run_id': str(i), 'catalog_id': 28, 'dataset': 'WinoGrande',
                                 'group_id': str(i), 'track': 'WinoGrande', 'payload_sha256': 'test'}}
                for i in range(3)]
        records = [{'run_id': str(i), 'catalog_id': 28, 'status': 'ok',
                    'response': {'answers': {'q1': {'type': 'choice', 'choice': 'A' if i == 0 else 'B',
                                                  'probabilities': {'A': .5, 'B': .5}}}},
                    'total_wall_ms': 2, 'model_request_wall_ms': 2,
                    'raw_output': {'board_method_ms': 1}}
                   for i in range(2)]
        records.append({'run_id': '2', 'catalog_id': 28, 'status': 'unsupported'})
        with tempfile.TemporaryDirectory() as folder:
            sample, results = Path(folder) / 'rows.jsonl', Path(folder) / 'results.jsonl'
            sample.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            results.write_text(''.join(json.dumps(r) + '\n' for r in records))
            result = score(sample, results, folder, 'probe')
            self.assertEqual(result['completed_requests'], 3)
            self.assertTrue(result['sample_complete'])
            self.assertEqual(result['labelled_field_accuracy']['accuracy'], 1 / 3)
            self.assertEqual(result['board_method']['count'], 2)
            self.assertIsNone(result['sampled_public_index'])


if __name__ == '__main__':
    unittest.main()
