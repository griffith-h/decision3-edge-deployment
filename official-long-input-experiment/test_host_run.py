import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'official-evaluation'))
import host_run

class LaunchTests(unittest.TestCase):
    def test_lab_is_frozen_source_snapshot_not_last_historical_mount(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);release=root/'release';release.mkdir();(release/'image-id.txt').write_text('sha256:pinned')
            (root/'probe.py').write_text('# source')
            (root/'kernel.c').write_text('/* source */');(root/'libkernel.so').write_bytes(b'pinned native binary')
            with patch.object(host_run,'RELEASE',release),patch.object(host_run,'mounts',return_value={'models':'/external/models'}):
                cmd=host_run.build_command(root,'run1','short314',True)
            lab=next(x for x in cmd if 'target=/lab,' in x)
            self.assertIn('source='+str(root/'sources/run1'),lab)
            self.assertTrue(lab.endswith(',readonly'))
            self.assertEqual((root/'sources/run1/probe.py').read_text(),'# source')
            self.assertEqual((root/'sources/run1/libkernel.so').read_bytes(),b'pinned native binary')
            (root/'libkernel.so').write_bytes(b'new binary')
            self.assertEqual((root/'sources/run1/libkernel.so').read_bytes(),b'pinned native binary')
            self.assertIn('type=bind,source=/external/models,target=/models,readonly',cmd)
    def test_controlled_run_retains_exit_state_until_controller_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);release=root/'release';release.mkdir();(release/'image-id.txt').write_text('sha256:pinned')
            with patch.object(host_run,'RELEASE',release),patch.object(host_run,'mounts',return_value={}):
                controlled=host_run.build_command(root,'controlled','case',retain_exit_state=True)
                standalone=host_run.build_command(root,'standalone','case')
            self.assertNotIn('--rm',controlled);self.assertIn('--rm',standalone)
            self.assertIn('decision3.opt.run=controlled',controlled)
    def test_run_cannot_escape_evidence_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ('../escape','..','/absolute'):
                with self.assertRaises(ValueError):host_run.build_command(Path(folder),name,'short314')
            self.assertFalse((Path(folder)/'results').exists())
    def test_sdk_trace_preload_is_limited_to_cost_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);release=root/'release';release.mkdir();(release/'image-id.txt').write_text('sha256:pinned')
            with patch.object(host_run,'RELEASE',release),patch.object(host_run,'mounts',return_value={}):
                ordinary=host_run.build_command(root,'plain','diagnostic-home-pair')
                diagnostic=host_run.build_command(root,'ledger','request-cost-ledger')
            self.assertFalse(any('LD_PRELOAD' in x for x in ordinary))
            self.assertIn('LD_PRELOAD=/lab/librequest_sdk_trace.so',diagnostic)
if __name__=='__main__':unittest.main()
