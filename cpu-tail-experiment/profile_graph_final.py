"""Closed main-thread partition; SDK worker intervals counted by union."""
from application import Application,state,stats
from validate_candidate import sources
import torch,json,time,ctypes,threading,statistics
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
app=Application(profile=True);paired=json.loads(Path('/work/logs/final-graph-comparison.json').read_text());cfg=paired['selected_config'];app.use(cfg)
r={'config':cfg,'settings':state(),'sources':sources(),'records':[],'scope':'instrumented wall-time attribution, not pure NPU device compute; main exclusive spans close, worker overlaps not summed'};path=Path('/work/logs/final-graph-profile.json')
class Attribution:
 def __init__(self):self.stack=[];self.rows=defaultdict(lambda:{'inclusive':0.,'exclusive':0.,'calls':0})
 @contextmanager
 def span(self,key):
  self.stack.append([key,time.perf_counter(),0.])
  try:yield
  finally:
   key,start,child=self.stack.pop();dt=time.perf_counter()-start;row=self.rows[key];row['inclusive']+=dt;row['exclusive']+=dt-child;row['calls']+=1
   if self.stack:self.stack[-1][2]+=dt
prof=Attribution();main=threading.get_ident();wrapped=[];worker=[]
def wrap(owner,name,key):
 old=getattr(owner,name)
 def fn(*args,**kwargs):
  if threading.get_ident()!=main:
   t=time.perf_counter()
   try:return old(*args,**kwargs)
   finally:worker.append({'key':key,'start':t,'end':time.perf_counter()})
  with prof.span(key):return old(*args,**kwargs)
 setattr(owner,name,fn);wrapped.append((owner,name,old))
def read_native():
 rows=[]
 for h in app.engine.handles:
  values=(ctypes.c_double*8)();app.engine.lib.mm_profile_read(h,values,1);rows.append(dict(zip(('calls','shape','input_convert','copy_pad','runtime_sync','output_convert','run_start','run_end'),values)))
 return rows
def union(intervals):
 total=0.;end=-float('inf')
 for start,stop in sorted(intervals):total+=max(0.,stop-max(start,end));end=max(end,stop)
 return total
def plain(n):
 vals=[]
 for _ in range(n):t=time.perf_counter();out=app.invoke();vals.append(time.perf_counter()-t);assert app.check(out)['passed']
 return stats(vals)
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 app.invoke();app.invoke();r['before']=plain(15);save()
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 import cpu_attention,lut_ops,attention_prepare,residual,pipeline,mlp_graph
 for owner,name,key in [(app.model,'prepare','request.prepare'),(app.model,'run','request.run'),(app.model,'respond','request.respond'),
  (qwen,'torch_chunk_gated_delta_rule','scan'),(qwen,'causal_conv1d_fn','causal_conv'),(qwen,'apply_rotary_pos_emb','rotary'),
  (torch.nn.functional,'scaled_dot_product_attention','sdpa.stock'),(cpu_attention,'attention','sdpa.custom'),
  (lut_ops,'activation','activation.lookup'),(lut_ops.lib,'d3_norm_lut','norm.native'),(residual,'add_norm','residual_norm.fused'),
  (attention_prepare,'prepare','attention.norm_rope'),(attention_prepare,'gated','attention.gate'),(pipeline,'run_pair','bridge.raw_fork'),
  (app.engine,'group','bridge.group'),(app.engine,'output_buffer','bridge.allocate'),
  (app.engine.lib,'mm_prepare_bf16','bridge.prepare'),(app.engine.lib,'mm_copy_prepared','bridge.copy'),
  (app.engine.lib,'mm_run_bf16','bridge.serial'),(app.engine.lib,'mm_run_prepared_bf16','bridge.prepared'),
  (app.engine.lib,'mm_run_prepared_scatter_bf16','bridge.scatter'),(app.engine.lib,'mm_run_raw','bridge.raw'),
  (app.engine.lib,'mm_prepare_mlp','mlp.pipeline_prepare')]:wrap(owner,name,key)
 if cfg.get('mlp_graph'):wrap(mlp_graph.lib,'mlp_graph_run_bf16','mlp.graph_api')
 # Method wrappers retain the original empty hook state, so fusion is unchanged.
 for name,m in app.model.backbone.named_modules():
  kind=type(m).__name__
  if isinstance(m,torch.nn.Linear):key='linear.rknn.other' if hasattr(m,'_bp_original') else 'linear.cpu'
  elif 'RMSNorm' in kind:key='norm.other'
  elif 'GatedDeltaNet' in kind:key='delta.other'
  elif kind=='Qwen3_5Attention':key='attention.other'
  elif 'MLP' in kind:key='mlp.other'
  elif 'DecoderLayer' in kind:key='block.other'
  elif isinstance(m,torch.nn.Embedding):key='embedding'
  else:key='module.other.'+kind
  wrap(m,'forward',key)
 for i in range(10):
  read_native();prof=Attribution();worker.clear();before=app.counts()
  with prof.span('request.other'):out=app.invoke()
  assert app.check(out)['passed'];total=prof.rows['request.other']['inclusive'];assert abs(total-sum(v['exclusive'] for v in prof.rows.values()))<1e-7
  after=app.counts()
  if cfg.get('pipeline') and not cfg.get('mlp_graph'):assert after['cpu']['pipeline']['optimized']-before['cpu']['pipeline']['optimized']==12
  if cfg.get('attention_prepare'):assert after['cpu']['attention_prepare']['optimized']-before['cpu']['attention_prepare']['optimized']==6
  if cfg.get('mlp_graph'):assert after['graphs']['optimized']-before['graphs']['optimized']==12
  rows=read_native();active=[v for v in rows if v['calls']];assert all(v['calls']==1 for v in active)
  sdk=union([(v['run_start'],v['run_end']) for v in active]);spans=dict(prof.rows)
  bridge=sum(v['exclusive'] for k,v in spans.items() if k.startswith('bridge.'))
  assert bridge>=sdk-1e-5,(bridge,sdk)
  r['records'].append({'total':total,'spans':spans,'sdk_union':sdk,'bridge_including_sdk':bridge,'bridge_remaining':bridge-sdk,'native_handles':active,'worker_intervals':list(worker),'state':state()});print('FINAL PROFILE',i,total,sdk,flush=True);save()
 for owner,name,old in reversed(wrapped):setattr(owner,name,old)
 wrapped.clear();r['after']=plain(15);r['passed']=True;r['final_counts']=app.counts();save()
finally:
 for owner,name,old in reversed(wrapped):setattr(owner,name,old)
 app.close()
