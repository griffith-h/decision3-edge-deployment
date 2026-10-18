"""Independent instrumented profile, never mixed into formal latency samples."""
from selected_owner import *
from pathlib import Path
import threading,resource
import long_kernels as lk,tail_native as native,prepared_scan
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());ref=refs['cases']['words-512'];case=ref['request'];current=ref['output'];app=None;finished=False;r={'settings':state(),'records':[],'scope':'Five independent instrumented pairs; main-thread exclusive decomposition. Worker spans overlap and are never summed into model time. SDK wrapper spans include conversion and waiting, not pure device time.'};restores=[];stack=[];totals={};worker=[];main=threading.get_ident()
def wrap(obj,name,tag):
 old=getattr(obj,name)
 def call(*args,**kwargs):
  if threading.get_ident()!=main:
   start=time.perf_counter()
   try:return old(*args,**kwargs)
   finally:worker.append({'tag':tag,'start':start,'end':time.perf_counter()})
  start=time.perf_counter();frame={'child':0.};stack.append(frame)
  try:return old(*args,**kwargs)
  finally:
   elapsed=time.perf_counter()-start;stack.pop();row=totals.setdefault(tag,{'inclusive':0.,'exclusive':0.,'calls':0});row['inclusive']+=elapsed;row['exclusive']+=elapsed-frame['child'];row['calls']+=1
   if stack:stack[-1]['child']+=elapsed
 setattr(obj,name,call);restores.append(lambda:setattr(obj,name,old))
try:
 app=Owner();r['execution']=app.execution
 for mode in ('baseline','candidate'):
  for _ in range(3):app.use(mode);app.invoke(case)
 app.use('baseline');base=app.invoke(case);r['baseline_reference']=base
 class Holder:pass
 holder=Holder();holder.att=qwen.ALL_ATTENTION_FUNCTIONS['sdpa'];original=holder.att;wrap(holder,'att','attention_entry');qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',holder.att);restores.append(lambda:qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',original))
 wrap(qwen,'torch_chunk_gated_delta_rule','scan_entry');wrap(qwen,'causal_conv1d_fn','conv');wrap(app.projections,'run','projection_rows');wrap(prepared_scan,'prepare','scan_prepare');wrap(native,'validate_raw','raw_validate');wrap(native,'finite','finite_check')
 for lib in (app.scan_original,*app.scan_options.values()):wrap(lib,'d3_chunk_scan','scan_c')
 ids={id(m) for m in app.projections.modules.values()}
 for name,m in app.model.backbone.named_modules():
  typ=type(m).__name__
  if typ=='Qwen3_5MLP':wrap(m,'forward','mlp')
  elif 'Norm' in typ:wrap(m,'forward','norm_modules')
  elif isinstance(m,torch.nn.Linear) and id(m) not in ids and '.mlp.' not in name:wrap(m,'forward','small_linear')
 for name in ('mm_prepare_bf16','mm_copy_prepared','mm_run_bf16','mm_run_prepared_bf16','mm_run_prepared_scatter_bf16'):wrap(app.engine.lib,name,'bridge.'+name)
 handles=len(app.engine.handles)
 for i in range(5):
  for mode in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
   app.use(mode);totals={};worker.clear();t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
   rec={'mode':mode,'seconds':dt,'components':copy.deepcopy(totals),'worker_intervals':list(worker),'unwrapped':dt-sum(x['exclusive'] for x in totals.values()),'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};assert rec['unwrapped']>=0 and len(app.engine.handles)==handles;r['records'].append(rec)
   print('FINAL PROFILE',mode,dt,{k:round(v['inclusive']*1000,2) for k,v in totals.items() if k in ('attention_entry','scan_entry','mlp','projection_rows','conv')},flush=True)
 r['handles']=handles;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;finished=True
except BaseException as exc:r['error']=repr(exc);raise
finally:
 for restore in reversed(restores):restore()
 try:
  if app:app.close()
  r['owner_closed']=True
 finally:r['completed']=finished and r.get('owner_closed',False);r['final_settings']=state();(p/'logs/profile-selected.json').write_text(json.dumps(r,indent=2)+'\n')
