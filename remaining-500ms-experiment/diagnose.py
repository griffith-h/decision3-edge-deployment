"""Main-thread exclusive partition; worker intervals reported separately."""
import argparse,ctypes,json,statistics,threading,time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import torch
from loader import open_model,compare,stats,state
from common import sources
model,engine,native,close=open_model(profile=True)
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
body=json.loads(Path('/base/request.json').read_text())
def invoke():return model.system_one(state=body['state'],questions=body['questions'])
result={'scope':'Frozen balanced direct-scatter native_t16_f5 candidate. CPU3, original weights/masks/12 layers. Main-thread exclusive spans close exactly; worker wall times overlap and are not added to the partition.',
        'settings':state(),'sources':sources()}
result['backend']='balanced-safe-scatter'
path=Path('/work/logs/profile.json')
def save():path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
class Attribution:
    def __init__(self):self.stack=[];self.rows=defaultdict(lambda:{'inclusive':0.,'exclusive':0.,'calls':0})
    @contextmanager
    def span(self,key):
        self.enter(key)
        try:yield
        finally:self.leave()
    def enter(self,key):self.stack.append([key,time.perf_counter(),0.])
    def leave(self):
        key,start,child=self.stack.pop();elapsed=time.perf_counter()-start
        r=self.rows[key];r['inclusive']+=elapsed;r['exclusive']+=elapsed-child;r['calls']+=1
        if self.stack:self.stack[-1][2]+=elapsed
prof=Attribution();main_thread=threading.get_ident();wrapped=[];hooks=[];worker=[]
def wrap(owner,name,key):
    original=getattr(owner,name)
    def forward(*args,**kwargs):
        if threading.get_ident()!=main_thread:
            t=time.perf_counter();out=original(*args,**kwargs);worker.append({'key':key,'start':t,'end':time.perf_counter()});return out
        with prof.span(key):return original(*args,**kwargs)
    setattr(owner,name,forward);wrapped.append((owner,name,original))
def timing(count):
    values=[]
    for _ in range(count):
        t=time.perf_counter();r=invoke();values.append(time.perf_counter()-t)
        assert compare(r,reference)['passed']
    return stats(values)
def native_read(reset=1):
    rows=[]
    for h in engine.handles:
        values=(ctypes.c_double*8)();engine.lib.mm_profile_read(h,values,reset)
        rows.append(dict(zip(('calls','shape','input_convert','copy_pad','runtime_sync','output_convert','run_start','run_end'),values)))
    return rows
def union(intervals):
    total=0.;end=-float('inf')
    for start,stop in sorted(intervals):
        total+=max(0.,stop-max(end,start));end=max(end,stop)
    return total
try:
    reference=invoke();result['response']=reference;result['before']=timing(15);save()
    for name in ('prepare','run','respond'):wrap(model,name,'request.'+name)
    for owner,name,key in [(qwen,'torch_chunk_gated_delta_rule','scan'),(qwen,'causal_conv1d_fn','causal_conv'),
        (qwen,'apply_rotary_pos_emb','rotary'),(torch.nn.functional,'scaled_dot_product_attention','sdpa'),
        (torch.nn.functional,'silu','silu'),(engine,'group','bridge.fork_submit_wait'),
        (engine,'output_buffer','bridge.allocate_bf16'),(engine.lib,'mm_prepare_bf16','bridge.prepare'),
        (engine.lib,'mm_copy_prepared','bridge.shared_copy'),(engine.lib,'mm_run_bf16','bridge.serial_native'),
        (engine.lib,'mm_run_prepared_bf16','bridge.worker_native'),(engine.lib,'mm_run_prepared_scatter_bf16','bridge.worker_scatter')]:wrap(owner,name,key)
    for name,module in model.backbone.named_modules():
        kind=type(module).__name__
        if isinstance(module,torch.nn.Linear):key='linear.rknn' if hasattr(module,'_bp_original') else 'linear.cpu'
        elif 'RMSNorm' in kind:key='norm'
        elif 'GatedDeltaNet' in kind:key='delta.other'
        elif kind=='Qwen3_5Attention':key='attention.other'
        elif isinstance(module,torch.nn.Embedding):key='embedding'
        elif 'MLP' in kind:key='mlp.other'
        elif 'DecoderLayer' in kind:key='block.other'
        else:key='module.other.'+kind
        def before(m,args,_key=key):prof.enter(_key)
        def after(m,args,out):prof.leave()
        hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    records=[]
    for i in range(10):
        native_read();prof=Attribution();worker.clear()
        with prof.span('request.other'):response=invoke()
        assert compare(response,reference)['passed']
        total=prof.rows['request.other']['inclusive'];assert abs(total-sum(r['exclusive'] for r in prof.rows.values()))<1e-7
        native_rows=native_read();active=[r for r in native_rows if r['calls']]
        assert len(active)==72 and all(r['calls']==1 for r in active)
        sdk_union=union([(r['run_start'],r['run_end']) for r in active])
        records.append({'total':total,'spans':dict(prof.rows),'native_handles':active,'worker_intervals':list(worker),'sdk_union':sdk_union,
                        'native_sums':{k:sum(r[k] for r in active) for k in ('shape','input_convert','copy_pad','runtime_sync','output_convert')}})
        print('PROFILE',i,total,sdk_union,flush=True)
        result['records']=records;save()
    for hook in hooks:hook.remove()
    hooks.clear()
    for owner,name,original in reversed(wrapped):setattr(owner,name,original)
    wrapped.clear()
    result['after']=timing(15)
    keys=set().union(*(r['spans'] for r in records))
    result['median_partition']={k:statistics.median(r['spans'].get(k,{}).get('exclusive',0.) for r in records) for k in keys}
    result['passed']=True;save()
    from torch.profiler import profile,ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU],record_shapes=True) as trace:
        r=invoke()
    assert compare(r,reference)['passed']
    result['aten']=[{'key':e.key,'self_cpu_ms':e.self_cpu_time_total/1000,'total_cpu_ms':e.cpu_time_total/1000,'calls':e.count,'shapes':e.input_shapes} for e in trace.key_averages(group_by_input_shape=True)]
    save()
    captures=[];original=qwen.torch_chunk_gated_delta_rule
    def capture(q,k,v,g,beta,*args,**kwargs):
        captures.append({'tensors':[x.detach().clone() for x in (q,k,v,g,beta)],'args':args,'kwargs':kwargs})
        return original(q,k,v,g,beta,*args,**kwargs)
    qwen.torch_chunk_gated_delta_rule=capture
    try:invoke()
    finally:qwen.torch_chunk_gated_delta_rule=original
    torch.save(captures,'/work/scan-inputs.pt')
finally:
    for hook in hooks:hook.remove()
    for owner,name,original in reversed(wrapped):setattr(owner,name,original)
    close()
