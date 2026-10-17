"""Focused shell control-flow test; fake Docker/curl never touch real services."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

class RollbackTest(unittest.TestCase):
    def test_existing_absent_and_missing_original(self):
        for mode in ('existing', 'absent-new', 'absent-original'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                docker=root/'docker'
                docker.write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$TEST_LOG"
if [[ "$1" == inspect && "$2" == decision3-edge-optimized && "$TEST_MODE" == absent-new ]]; then exit 1; fi
if [[ "$1" == inspect && "$2" == decision3-edge-matmul && "$TEST_MODE" == absent-original ]]; then exit 1; fi
''')
                curl=root/'curl';curl.write_text('#!/bin/bash\nexit 0\n')
                docker.chmod(0o755);curl.chmod(0o755)
                env=dict(os.environ,PATH=tmp+':'+os.environ['PATH'],TEST_LOG=str(root/'calls'),TEST_MODE=mode)
                result=subprocess.run(['bash',str(Path(__file__).with_name('rollback.sh'))],env=env,capture_output=True,text=True)
                calls=(root/'calls').read_text().splitlines()
                if mode=='absent-original':
                    self.assertNotEqual(result.returncode,0)
                    self.assertEqual(calls,['inspect decision3-edge-matmul'])
                else:
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertIn('start decision3-edge-matmul',calls)
                    self.assertEqual('stop -t 60 decision3-edge-optimized' in calls,mode=='existing')

if __name__=='__main__':unittest.main()
