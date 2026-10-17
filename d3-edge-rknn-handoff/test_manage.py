"""Focused validation of configuration and Docker ownership boundaries."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('manage', Path(__file__).with_name('manage.py'))
manage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage)


class ManageTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((manage.ROOT / 'config.example.json').read_text())

    def load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(self.config))
            return manage.load_config(path)

    def test_relative_data_directory_and_portable_configuration(self):
        c = self.load()
        self.assertEqual(Path(c['data_dir']), manage.ROOT / 'data')
        self.assertEqual(c['bind_host'], '0.0.0.0')

    def test_reject_invalid_container_ip_port_cpu_and_restart(self):
        for key, invalid in [('container_name', '../other'), ('bind_host', 'host;cmd'),
                             ('port', 70000), ('cpu_set', '4-7;cmd'), ('restart', 'always')]:
            with self.subTest(key=key):
                original = self.config[key]
                self.config[key] = invalid
                with self.assertRaises(ValueError):
                    self.load()
                self.config[key] = original

    def test_refuse_unrelated_container(self):
        response = subprocess.CompletedProcess([], 0, json.dumps([{'Config': {'Labels': {}}}]))
        with patch.object(manage.subprocess, 'run', return_value=response):
            with self.assertRaisesRegex(RuntimeError, 'unrelated'):
                manage.inspect(self.config)

    def test_accept_owned_container(self):
        response = subprocess.CompletedProcess([], 0, json.dumps([{'Config': {'Labels': {manage.OWNER: '1'}}}]))
        with patch.object(manage.subprocess, 'run', return_value=response):
            self.assertEqual(manage.inspect(self.config)['Config']['Labels'][manage.OWNER], '1')

    def test_config_change_invalidates_container_fingerprint(self):
        previous = manage.config_hash(self.config)
        self.config['port'] += 1
        self.assertNotEqual(previous, manage.config_hash(self.config))


if __name__ == '__main__':
    unittest.main()
