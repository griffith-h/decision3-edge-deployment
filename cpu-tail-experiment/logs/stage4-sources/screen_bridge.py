from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
from bridge_toggle import library
from loader import PackedEngine,install
import json,time
from pathlib import Path
prior=json.loads(Path('/work/logs/fusion-screen.json').read_text())['variants'];qualified={n:v for n,v in prior.items() if v['passed']};selected=min(qualified,key=lambda n:qualified[n]['timing']['median']);cfg=qualified[selected]['config']
model,engine,basecounts,close=load_baseline();table=Tables();ev=Evaluation(model);candidate=Candidate(model,engine,table)
r={'settings':state(),'config':cfg,'selected_from_fusion':selected,'variants':{},'sources':sources(),'scope':'one model, one 72-handle engine at a time; recreates contexts between bridge modes; not paired formal comparison'}
path=Path('/work/logs/bridge-screen.json');units=json.loads(Path('/work/logs/bridge-modes-unit.json').read_text())['variants']
try:
 for name in ('baseline','lut','native','m115','baseline-after'):
  if name in units and not units[name]['passed']:continue
  candidate.reset();engine.restore_forwards();engine.restore_forwards=None;engine.close()
  engine=PackedEngine('balanced')
  engine.lib=library(engine,'/previous/libconversion_bridge.so' if name.startswith('baseline') else '/work/lib'+('lut_bridge' if name=='lut' else 'native_bridge' if name=='native' else 'm115_bridge')+'.so')
  engine.groups=[];engine.restore_forwards=install(model,engine);candidate=Candidate(model,engine,table);candidate.use(cfg)
  ev.invoke();ev.invoke();row={'records':[]};r['variants'][name]=row
  for i in range(16):
   t=time.perf_counter();out=ev.invoke();row['records'].append({'seconds':time.perf_counter()-t,**ev.check(out)})
  row['timing']=stats([v['seconds'] for v in row['records']]);row.update(ev.validate());row['passed'] &= all(v['passed'] for v in row['records']);row['handles']=len(engine.handles)
  print('BRIDGE MODEL',name,row['timing']['median'],row['passed'],flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:
 candidate.reset()
 if engine.restore_forwards:engine.restore_forwards();engine.restore_forwards=None
 engine.close();close()
