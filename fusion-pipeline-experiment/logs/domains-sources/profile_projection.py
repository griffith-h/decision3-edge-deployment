import sys
sys.path.insert(0,'/tail')
from runner import *
import experiment,ctypes,threading
original_load=experiment.load_baseline
experiment.load_baseline=lambda **kw:original_load(profile=True,library='/work/libpipeline_profile.so')
app=Owner();case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];p=Path('/experiment');r={'settings':state(),'records':[],'scope':'Instrumented native sums and interval unions; runtime_sync includes submit/driver/wait, not pure device time'};events=[];restores=[]
keys=('calls','shape','input_convert','copy_pad','runtime_sync','output_convert','run_start','run_end')
def native_read(handle,reset):
 a=(ctypes.c_double*8)();app.engine.lib.mm_profile_read(handle,a,reset);return dict(zip(keys,a))
def union(intervals):
 total=0.;end=-float('inf')
 for start,stop in sorted(intervals):total+=max(0.,stop-max(end,start));end=max(end,stop)
 return total
try:
 app.use('validate_mlp');ref=app.invoke(case);app.invoke(case)
 for name in ('mm_prepare_bf16','mm_copy_prepared','mm_run_bf16','mm_run_prepared_bf16','mm_run_prepared_scatter_bf16'):
  old=getattr(app.engine.lib,name)
  def fn(*args,_old=old,_name=name):
   start=time.perf_counter()
   try:return _old(*args)
   finally:events.append({'name':_name,'thread':threading.get_ident(),'start':start,'end':time.perf_counter()})
  setattr(app.engine.lib,name,fn);restores.append((name,old))
 for i in range(3):
  for h in app.engine.handles:native_read(h,1)
  events.clear();app.use('validate_mlp');t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
  values=[native_read(h,1) for h in app.engine.handles];active=[v for v in values if v['calls']]
  sums={k:sum(v[k] for v in values) for k in keys[:6]}
  r['records'].append({'seconds':dt,'output':out,'vs_reference':detailed(out,ref),'native_sums':sums,'native_handles':active,'events':list(events),'bridge_interval_union':union((x['start'],x['end']) for x in events),**app.counts()});assert detailed(out,ref)['all_numeric_max_delta']==0
  print('PROJECTION',dt,sums,'UNION',r['records'][-1]['bridge_interval_union'],flush=True)
 r['completed']=True;r['final_settings']=state();(p/'logs/profile-projection.json').write_text(json.dumps(r,indent=2)+'\n')
finally:
 for name,old in restores:setattr(app.engine.lib,name,old)
 app.close()
