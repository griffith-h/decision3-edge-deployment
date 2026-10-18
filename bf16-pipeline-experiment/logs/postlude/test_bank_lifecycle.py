"""Untimed resource test after all formal/profile runs, using frozen libraries."""
import os,json,gc
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import torch
from split_mlp import SplitMlp
from accepted_owner import Owner
p=Path('/experiment');torch.set_num_threads(3);torch.manual_seed(29)
def rss():
 return next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
def fds():
 result={}
 for name in os.listdir('/proc/self/fd'):
  try:result[name]=os.readlink('/proc/self/fd/'+name)
  except FileNotFoundError:pass
 return result
owner_fds={'before_owner':fds()};app=Owner()
try:
 reference=json.loads(Path('/fusion/logs/references.json').read_text())['cases']['words-512']
 for mode in ('baseline','candidate'):app.use(mode);output=app.invoke(reference['request'])
 owner_fds['warm']=fds();owner_fds['owned_handles_warm']=[len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)]
 app.close();app.close();owner_fds['after_close']=fds();owner_fds['owned_handles_after_close']=[len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)];assert owner_fds['owned_handles_after_close']==[0,0,0]
finally:app.close()
del app;gc.collect();owner_fds['after_owner_gc']=fds()
with torch.inference_mode():
 members=[torch.nn.Linear(1024,3072,bias=False,dtype=torch.bfloat16).eval(),torch.nn.Linear(1024,3072,bias=False,dtype=torch.bfloat16).eval(),torch.nn.Linear(3072,1024,bias=False,dtype=torch.bfloat16).eval()];x=torch.randn(1,615,1024).bfloat16();raw=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16);table=SimpleNamespace(silu=torch.nn.functional.silu(raw))
engine=SimpleNamespace(pool=None,core={id(members[0]):1,id(members[1]):2},bucket={});r={'owner_fds':owner_fds,'before_first_bank':fds(),'cycles':[],'completed':False,'timed':False};bank=None
try:
 with torch.inference_mode():
  for i in range(3):
   engine.pool=ThreadPoolExecutor(2);bank=SplitMlp(engine,table);out=bank.run(members,x);before=fds();handles=len(bank.handles);assert handles==4 and torch.isfinite(out).all()
   bank.close();bank.close();assert not bank.handles;engine.pool.shutdown(wait=True);gc.collect();after=fds();r['cycles'].append({'cycle':i,'handles_warm':handles,'handles_after_close':len(bank.handles),'fds_warm':before,'fds_after_close':after,'rss_after_close':rss()})
  assert len({len(v['fds_warm']) for v in r['cycles']})==1
  assert len({len(v['fds_after_close']) for v in r['cycles']})==1
  assert r['cycles'][0]['fds_after_close']==r['cycles'][-1]['fds_after_close'];r['no_per_bank_fd_growth']=True;r['completed']=True
finally:
 if bank:bank.close()
 if engine.pool:engine.pool.shutdown(wait=True)
 gc.collect();r['after_pool_shutdown']=fds();r['source']=Path(__file__).read_text();(p/'logs/bank-lifecycle.json').write_text(json.dumps(r,indent=2))
print('BANK LIFECYCLE',r['completed'],[len(v['fds_after_close']) for v in r['cycles']],r['after_pool_shutdown'],flush=True)
