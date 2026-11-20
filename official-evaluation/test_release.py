"""Focused deployment boundaries; no Docker or remote service is changed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch_release as release


class ReleaseTests(unittest.TestCase):
    def test_package_refuses_changed_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'asset.py').write_text('original')
            sha = hashlib.sha256(b'original').hexdigest()
            (root / 'manifest.json').write_text(json.dumps({'files': {'asset.py': sha}}))
            release.package(root)
            (root / 'asset.py').write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                release.package(root)

    def test_never_removes_unowned_container(self):
        with patch.object(release, 'inspect', return_value={'Config': {'Labels': {}}}), \
                patch.object(release, 'docker') as docker:
            with self.assertRaisesRegex(RuntimeError, 'another container'):
                release.remove_owned(release.NEW, 'expected-marker')
            docker.assert_not_called()

    def test_restores_old_even_when_failed_candidate_cleanup_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'service').mkdir()
            verdict = root / 'service/evaluation-result.json'
            verdict.write_text('{}')
            proof, acceptance = root / 'proof.json', root / 'acceptance.json'
            proof.write_text(json.dumps({'passed': True, 'manifest_sha256': 'marker',
                                        'port': 8901, 'regression_cases': 34}))
            acceptance.write_text(json.dumps({'accepted': True, 'reason': 'Reviewed metrics and failures',
                                             'verdict_sha256': release.digest(verdict),
                                             'sample_rows_sha256': 'sample'}))
            old = {'Id': 'old-id', 'State': {'Running': True}, 'Config': {'Env': []},
                   'HostConfig': {'RestartPolicy': {'Name': 'unless-stopped'}}}
            def cleanup(name, marker):
                if name == release.NEW:
                    raise RuntimeError('cleanup failed')
            with patch.object(release, 'package', return_value=({'sample_rows_sha256': 'sample'}, 'marker')), \
                    patch.object(release, 'inspect', side_effect=[old, None]), \
                    patch.object(release, 'health', return_value={'deployment': {'sample_rows_sha256': 'sample'}}), \
                    patch.object(release, 'docker'), patch.object(release, 'remove_owned', side_effect=cleanup), \
                    patch.object(release, 'launch', side_effect=RuntimeError('failed launch')), \
                    patch.object(release, 'restore') as restore:
                with self.assertRaises(RuntimeError):
                    release.activate(root, proof, acceptance)
                restore.assert_called_once_with(old)
            self.assertFalse(json.loads((root / 'activation.json').read_text())['passed'])

    def test_auth_is_passed_by_environment(self):
        env = release.api_environment({'Config': {'Env': ['DECISION_API_KEY=private-example']}})
        self.assertEqual(env['DECISION_API_KEY'], 'private-example')


if __name__ == '__main__':
    unittest.main()
