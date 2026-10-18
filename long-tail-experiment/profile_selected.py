from runner import *
from pathlib import Path
import long_kernels as lk
import tail_native as n
import prepared_scan
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
formal=json.loads((P/'logs/formal.json').read_text());assert formal['completed'];selected=formal['candidate'];case=formal['cases']['words-512']['request'];current=formal['cases']['words-512']['current_reference']
app=Owner();r={'candidate':selected,'records':[],'settings':state(),'scope':'Separate instrumented five alternating pairs. Graph worker calls are not individually timed: their elapsed times overlap.'};restores=[];stack=[];totals={}
def wrap(obj,name,tag):
    old=getattr(obj,name)
    def call(*args,**kwargs):
        frame={'child':0.};stack.append(frame);start=time.perf_counter()
        try:return old(*args,**kwargs)
        finally:
            elapsed=time.perf_counter()-start;stack.pop();v=totals.setdefault(tag,{'inclusive':0.,'exclusive':0.,'calls':0});v['inclusive']+=elapsed;v['exclusive']+=elapsed-frame['child'];v['calls']+=1
            if stack:stack[-1]['child']+=elapsed
    setattr(obj,name,call);restores.append(lambda:setattr(obj,name,old))
try:
    app.use('baseline');base=app.invoke(case)
    for mode in ('baseline',selected):
        app.use(mode);app.invoke(case);app.invoke(case)
    original=qwen.ALL_ATTENTION_FUNCTIONS['sdpa']
    class Holder:pass
    holder=Holder();holder.att=original;wrap(holder,'att','attention_entry');qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',holder.att);restores.append(lambda:qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',original))
    wrap(qwen,'torch_chunk_gated_delta_rule','scan_entry');wrap(qwen,'causal_conv1d_fn','conv');wrap(app.projections,'run','projection_rows')
    wrap(lk.att,'d3_attention','attention_c');wrap(n.att,'d3_attention_tiled','attention_c')
    wrap(prepared_scan,'prepare','scan_prepare');wrap(lk,'scan_prepared','scan_native_wrapper');wrap(n,'scan','scan_native_wrapper')
    wrap(lk.scan,'d3_chunk_scan','scan_c')
    for lib in n.scans.values():wrap(lib,'d3_chunk_scan','scan_c')
    wrap(n.rec,'d3_scan_fast','scan_c');wrap(n,'validate_raw','raw_validate_c');wrap(n,'finite','finite_c')
    ids={id(m) for m in app.projections.modules.values()}
    for name,m in app.model.backbone.named_modules():
        typ=type(m).__name__
        if typ=='Qwen3_5MLP':wrap(m,'forward','mlp')
        elif 'Norm' in typ:wrap(m,'forward','norm_modules')
        elif isinstance(m,torch.nn.Linear) and id(m) not in ids and '.mlp.' not in name:wrap(m,'forward','small_linear')
    before=len(app.engine.handles)
    for i in range(5):
        for mode in (('baseline',selected) if i%2==0 else (selected,'baseline')):
            app.use(mode);totals={};t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
            rec={'mode':mode,'seconds':dt,'components':copy.deepcopy(totals),'unwrapped':dt-sum(x['exclusive'] for x in totals.values()),'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};assert rec['unwrapped']>=0 and len(app.engine.handles)==before;r['records'].append(rec)
    r['handles']=before;r['completed']=True;r['final_settings']=state();(P/'logs/profile-selected.json').write_text(json.dumps(r,indent=2)+'\n');print('FINAL PROFILE PASS',selected,flush=True)
finally:
    for restore in reversed(restores):restore()
    app.close()
