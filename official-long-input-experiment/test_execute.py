"""Cancellation plus cleanup failure must still restore the retained endpoint."""
import datetime,json,os,signal,tempfile,types,unittest
from pathlib import Path
from unittest.mock import Mock,patch
import execute

class RecoveryTests(unittest.TestCase):
    def test_phase_budget_can_narrow_but_never_widen_authorized_limit(self):
        self.assertEqual(execute.request_budget({},120),120)
        self.assertEqual(execute.request_budget({'budget_seconds':120},600),120)
        self.assertEqual(execute.request_budget({'budget_seconds':600},120),120)
        for value in (0,-1,float('nan'),float('inf'),True,'120',None):
            with self.assertRaisesRegex(ValueError,'per-phase'):execute.request_budget({'budget_seconds':value},600)

    def test_controller_lock_covers_service_restoration_until_context_exits(self):
        with tempfile.TemporaryDirectory() as folder:
            previous=os.getcwd();os.chdir(folder)
            try:
                with execute.exclusive_controller(),patch.object(execute,'ssh') as ssh,patch.object(execute.subprocess,'run') as run:
                    with self.assertRaisesRegex(RuntimeError,'restoring the retained service'):
                        execute.main()
                    ssh.assert_not_called();run.assert_not_called()
                # A completed/failed outer controller releases the OS lock.
                with execute.exclusive_controller():pass
            finally:os.chdir(previous)

    def test_cancel_restores_service_even_if_another_container_owns_worker_name(self):
        events=[]
        old={'Id':'retained-id','State':{'Running':True}}
        def ssh(command,**kwargs):
            events.append(command)
            if command=='docker inspect '+execute.OLD:return types.SimpleNamespace(stdout=json.dumps([old]))
            if 'State.Health.Status' in command:return types.SimpleNamespace(stdout='healthy\n')
            return types.SimpleNamespace(stdout='')
        inspections=iter([types.SimpleNamespace(returncode=1),types.SimpleNamespace(returncode=0,stdout=json.dumps([{'Config':{'Labels':{'decision3.opt.run':'unrelated'}}}]))])
        def run(*args,**kwargs):return next(inspections)
        selector=Mock()
        selector.select.side_effect=lambda timeout:execute.interrupt_safely(signal.SIGTERM,None)
        process=Mock();process.stdout=object();process.returncode=-15
        frozen=Mock();frozen.now.return_value=datetime.datetime(2026,10,19,8,tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as folder:
            previous=os.getcwd();os.chdir(folder)
            try:
                Path('output/decision-index-audit/iterations/r001').mkdir(parents=True)
                with patch.object(execute,'ssh',side_effect=ssh),patch.object(execute.subprocess,'run',side_effect=run),patch.object(execute.subprocess,'Popen',return_value=process),patch.object(execute.selectors,'DefaultSelector',return_value=selector),patch.object(execute.datetime,'datetime',frozen),patch.object(execute.signal,'signal'),patch.object(execute.sys if hasattr(execute,'sys') else __import__('sys'),'argv',['execute.py','--case','short314']):
                    with self.assertRaisesRegex(RuntimeError,'Cleanup or retained'):
                        execute.main()
                status=json.loads(next(Path('output').rglob('execution.json')).read_text())
                self.assertTrue(status['retained_healthy'])
                self.assertFalse(status['request_completed'])
                self.assertTrue(status['cleanup_errors'])
                self.assertIn('docker start '+execute.OLD,events)
                self.assertFalse(any('docker rm' in command for command in events))
            finally:os.chdir(previous)

    def test_owned_oom_state_is_saved_before_removal_and_service_restored(self):
        events=[];old={'Id':'retained-id','State':{'Running':True}}
        stamp='20261019T080000-short314-baseline-plain'
        container={'Config':{'Labels':{'decision3.opt.run':stamp}},'State':{'Status':'exited','Running':False,'ExitCode':137,'OOMKilled':True}}
        def ssh(command,**kwargs):
            events.append(command)
            if command=='docker inspect '+execute.OLD:return types.SimpleNamespace(stdout=json.dumps([old]))
            if 'State.Health.Status' in command:return types.SimpleNamespace(stdout='healthy\n')
            return types.SimpleNamespace(stdout='')
        inspections=iter([types.SimpleNamespace(returncode=1),types.SimpleNamespace(returncode=0,stdout=json.dumps([container])),types.SimpleNamespace(returncode=0)])
        selector=Mock();selector.select.side_effect=lambda timeout:execute.interrupt_safely(signal.SIGTERM,None)
        process=Mock();process.stdout=object();process.returncode=137
        frozen=Mock();frozen.now.return_value=datetime.datetime(2026,10,19,8,tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as folder:
            previous=os.getcwd();os.chdir(folder)
            try:
                Path('output/decision-index-audit/iterations/r001').mkdir(parents=True)
                with patch.object(execute,'ssh',side_effect=ssh),patch.object(execute.subprocess,'run',side_effect=lambda *a,**k:next(inspections)),patch.object(execute.subprocess,'Popen',return_value=process),patch.object(execute.selectors,'DefaultSelector',return_value=selector),patch.object(execute.datetime,'datetime',frozen),patch.object(execute.signal,'signal'),patch.object(__import__('sys'),'argv',['execute.py','--case','short314']):
                    with self.assertRaises(KeyboardInterrupt):execute.main()
                status=json.loads(next(Path('output').rglob('execution.json')).read_text())
                self.assertTrue(status['worker_final_state']['OOMKilled']);self.assertEqual(status['worker_final_state']['ExitCode'],137)
                self.assertTrue(status['retained_healthy']);self.assertEqual(status['cleanup_errors'],[])
                self.assertIn('docker rm -f '+execute.NAME,events)
            finally:os.chdir(previous)

if __name__=='__main__':unittest.main()
