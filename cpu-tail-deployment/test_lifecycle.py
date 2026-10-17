"""Focused release-only failure and recovery paths; no RK hardware or real Docker."""
import importlib.util,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parent

def module(name):
 spec=importlib.util.spec_from_file_location('test_'+name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class Cutover(unittest.TestCase):
 def run_case(self,mode):
  m=module('cutover');calls=[]
  def docker(*args):calls.append(('docker',*args));return 'id'
  def run(args,**kwargs):
   calls.append(tuple(args))
   if args[:3]==['docker','inspect',m.NEW]:return types.SimpleNamespace(returncode=1)
   return types.SimpleNamespace(returncode=0)
  def wait(*args):
   if mode=='activation-fails':raise RuntimeError('startup failed')
   return {'status':'ready'}
  with tempfile.TemporaryDirectory() as tmp,patch.object(m,'docker',docker),patch.object(m,'inspect',return_value={'State':{'Running':True},'Id':'previous'}),patch.object(m,'health',return_value={'deployment':{'release':m.RELEASE if mode!='bad-stage' else 'wrong'}}),patch.object(m,'wait',side_effect=wait),patch.object(m.subprocess,'run',side_effect=run):
   root=Path(tmp)
   if mode in ('activation-fails','bad-stage'):
    with self.assertRaises(RuntimeError):m.activate(root)
   else:self.assertTrue(m.activate(root)['passed'])
   if mode=='activation-fails':self.assertIn(('bash',str(root/'rollback.sh')),calls)
   if mode=='bad-stage':self.assertFalse(calls)
   if mode=='success':self.assertFalse(any('rollback.sh' in str(x) for x in calls))
  return calls
 def test_success(self):self.run_case('success')
 def test_failed_activation_restores_old(self):self.run_case('activation-fails')
 def test_bad_stage_never_stops_old(self):self.run_case('bad-stage')

class RuntimeOwner(unittest.TestCase):
 def test_install_failure_restores_candidate_and_model(self):
  events=[]
  baseline=types.ModuleType('baseline');baseline.load_baseline=lambda **k:(object(),object(),{},lambda:events.append('model_close'))
  candidate=types.ModuleType('candidate')
  class Candidate:
   def __init__(self,*args):pass
   def use(self,cfg):events.append('use')
   def reset(self):events.append('candidate_reset')
  candidate.Candidate=Candidate
  lut=types.ModuleType('lut_ops');lut.Tables=lambda:None
  graph=types.ModuleType('mlp_graph');graph.HalfGraph=object;graph.lib=object()
  sub=types.ModuleType('mlp_subgraph');sub.install=Mock(side_effect=RuntimeError('partial graph install failed'))
  deps={'baseline':baseline,'candidate':candidate,'lut_ops':lut,'mlp_graph':graph,'mlp_subgraph':sub}
  with patch.dict(sys.modules,deps),patch.object(Path,'read_text',return_value=json.dumps({'mlp_graph':True})):
   m=module('runtime')
   with self.assertRaises(RuntimeError):m.Runtime()
  self.assertEqual(events,['use','candidate_reset','model_close'])

if __name__=='__main__':unittest.main()
